"""Drafter-free speculative decoding for OpenVLA: the draft for step t is the action tokens OpenVLA itself produced at
step t-1. Exact (greedy) verification, so actions are identical to stock HF generate up to bf16 numerics.
Each verify pass feeds [last token] + remaining draft tokens to the LLM, accepts the longest matching prefix, adds the
verifier's own next token (bonus), crops the KV cache, and repeats until 7 action tokens exist.
Modes: hf = stock generate (fair AR reference), prev = previous-action draft, mine = our loop without draft. --check True also runs stock generate
every step and counts token mismatches. Closed loop on one LIBERO task, CUDA-synced timing.
Run with PYTHONPATH=$PROJ/third_party/openvla. Writes <out> JSON: per-episode success, step_ms, llm_calls, tokens."""
import json
import os
import time
from dataclasses import dataclass

import draccus
import numpy as np
import torch
from libero.libero import benchmark

from experiments.robot.libero.libero_utils import get_libero_dummy_action, get_libero_env, get_libero_image, quat2axisangle
from experiments.robot.openvla_utils import get_processor
from experiments.robot.robot_utils import (get_action, get_image_resize_size, get_model, invert_gripper_action,
                                           normalize_gripper_action, set_seed_everywhere)

MAX_STEPS = {"libero_spatial": 220, "libero_object": 280, "libero_goal": 300, "libero_10": 520}


@dataclass
class Cfg:
    model_family: str = "openvla"
    pretrained_checkpoint: str = ""
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    center_crop: bool = True
    task_suite_name: str = "libero_goal"
    task_id: int = 0
    num_trials_per_task: int = 3
    mode: str = "prev"
    check: bool = False
    relax_r: int = 0          # >0: also accept a draft token within relax_r bins of the verifier's choice ...
    relax_tau: float = 0.0    # ... if p(draft) >= relax_tau * p(verifier's choice). r=0 -> exact.
    out: str = "results/prevact/out.json"
    seed: int = 7


def crop(pkv, n):
    if isinstance(pkv, tuple):
        return tuple((k[:, :, :n], v[:, :, :n]) for k, v in pkv)
    pkv.crop(n) if hasattr(pkv, "crop") else [setattr(pkv, a, [x[:, :, :n] for x in getattr(pkv, a)])
                                             for a in ("key_cache", "value_cache")]
    if hasattr(pkv, "_seen_tokens"):
        pkv._seen_tokens = n
    return pkv


def make_generate(model, state, calls, r=0, tau=0.0):
    """Replacement for model.generate used by predict_action (greedy, batch 1)."""
    stock = model.generate

    @torch.no_grad()
    def gen(input_ids, max_new_tokens=7, pixel_values=None, attention_mask=None, **kw):
        out = model(input_ids=input_ids, pixel_values=pixel_values, use_cache=True)
        calls["n"] += 1
        pkv = out.past_key_values
        cache_len = out.logits.shape[1]
        g = [int(out.logits[0, -1].argmax())]
        draft = state.get("prev")
        while len(g) < max_new_tokens:
            k = len(g)
            d = draft[k:max_new_tokens] if draft is not None else []
            x = torch.tensor([[g[-1]] + list(d)], device=input_ids.device)
            o = model.language_model(input_ids=x, past_key_values=pkv, use_cache=True)
            calls["n"] += 1
            lg = o.logits[0]
            am = lg.argmax(-1).tolist()
            if r > 0 and len(d):
                pr = lg[:len(d)].float().softmax(-1)
                dt = torch.tensor(list(d), device=lg.device)
                ar = torch.arange(len(d), device=lg.device)
                ok = ((dt - pr.argmax(-1)).abs() <= r) & (pr[ar, dt] >= tau * pr.max(-1).values)
                ok = ok.tolist()
            else:
                ok = [False] * len(d)
            L = 0
            while L < len(d) and (am[L] == d[L] or ok[L]):
                L += 1
            g += list(d[:L])
            if len(g) < max_new_tokens:
                g.append(am[L])
            # cache now valid for: old cache + g[k-1] + accepted drafts
            cache_len += 1 + L
            pkv = crop(o.past_key_values, cache_len)
        g = g[:max_new_tokens]
        state["prev"] = g if state.get("use_draft", True) else None
        if state.get("check"):
            res = stock(input_ids, max_new_tokens=max_new_tokens, pixel_values=pixel_values, attention_mask=attention_mask,
                      do_sample=False, output_scores=True, return_dict_in_generate=True)
            ref = res.sequences[0, -max_new_tokens:].tolist()
            if ref != g:  # stock top-1 minus top-2 logit gap and stock prob at the first differing position
                j = next(i for i in range(max_new_tokens) if ref[i] != g[i])
                sc = res.scores[j][0].float()
                top2 = sc.topk(2).values
                state.setdefault("gaps", []).append([j, float(top2[0] - top2[1]), float(sc.softmax(-1).max()),
                                                     float(sc[ref[j]] - sc[g[j]])])
            state["mism"] += int(ref != g); state["n"] += 1
            if ref != g:
                state.setdefault("pairs", []).append([ref, g])
        state["last"] = g
        return torch.cat([input_ids, torch.tensor([g], device=input_ids.device)], dim=1)
    return gen


@draccus.wrap()
def main(cfg: Cfg):
    set_seed_everywhere(cfg.seed)
    cfg.unnorm_key = cfg.task_suite_name
    model = get_model(cfg)
    if cfg.unnorm_key not in model.norm_stats:
        cfg.unnorm_key += "_no_noops"
    processor = get_processor(cfg)
    calls = {"n": 0}
    state = {"check": cfg.check, "mism": 0, "n": 0}
    state["use_draft"] = cfg.mode == "prev"
    if cfg.mode in ("prev", "mine"):  # mine = our decode loop with no draft (plain AR), to separate numerics from drafting
        model.generate = make_generate(model, state, calls, cfg.relax_r, cfg.relax_tau)
    else:
        model.language_model.register_forward_hook(lambda *_: calls.__setitem__("n", calls["n"] + 1))
    suite = benchmark.get_benchmark_dict()[cfg.task_suite_name]()
    task = suite.get_task(cfg.task_id)
    init = suite.get_task_init_states(cfg.task_id)
    env, desc = get_libero_env(task, cfg.model_family, resolution=256)
    rec = {"gpu": torch.cuda.get_device_name(0), "mode": cfg.mode, "episodes": []}
    for ep in range(cfg.num_trials_per_task):
        env.reset(); obs = env.set_init_state(init[ep]); state["prev"] = None
        t, done, ms, nc, toks = 0, False, [], [], []
        while t < MAX_STEPS[cfg.task_suite_name] + 10:
            if t < 10:
                obs, _, done, _ = env.step(get_libero_dummy_action(cfg.model_family)); t += 1; continue
            img = get_libero_image(obs, get_image_resize_size(cfg))
            o = {"full_image": img, "state": np.concatenate((obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]),
                                                             obs["robot0_gripper_qpos"]))}
            calls["n"] = 0
            torch.cuda.synchronize(); s = time.perf_counter()
            a = get_action(cfg, model, o, desc, processor=processor)
            torch.cuda.synchronize(); ms.append(1000 * (time.perf_counter() - s)); nc.append(calls["n"])
            toks.append(state.get("last"))
            obs, _, done, _ = env.step(invert_gripper_action(normalize_gripper_action(a, binarize=True)).tolist())
            if done:
                break
            t += 1
        rec["episodes"].append({"success": bool(done), "step_ms": ms, "llm_calls": nc, "tokens": toks})
        print(f"ep {ep + 1}: success={done} mean_ms={np.mean(ms[1:]):.1f} llm_calls={np.mean(nc):.2f}"
              + (f" mismatches={state['mism']}/{state['n']}" if cfg.check else ""), flush=True)
    rec["check"] = {"mismatch": state["mism"], "n": state["n"], "pairs": state.get("pairs", []),
                    "gaps": state.get("gaps", [])}
    os.makedirs(os.path.dirname(os.path.abspath(cfg.out)), exist_ok=True)
    json.dump(rec, open(cfg.out, "w"))


if __name__ == "__main__":
    main()
