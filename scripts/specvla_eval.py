"""Spec-VLA LIBERO eval, generalized from third_party/SpecVLA/.../run_libero_goal_{AR,Spec,Spec_Relaxed}.py.

Their scripts are hard-coded to libero_goal with placeholder paths. This one takes any suite / task and
one of three modes, all through the same SpecVLA wrapper so the timing harness is identical:
  ar       plain autoregressive decoding (the speedup reference)
  spec     speculative decoding, exact acceptance (accept_threshold=0)
  relaxed  speculative decoding, accept if |draft_bin - verifier_bin| <= accept_threshold

Writes <out_dir>/<suite>_task<id>.json: per-episode success, per-step model latency (ms), and per-step
list of accept lengths (one entry per verify pass; only with --log_accept True).
--profile True: CUDA-synced per-step phase times (ms): vision, prefill (LLM part, vision excluded), verify (LLM tree
  passes / AR decode passes), draft (drafter topK_genrate), n_llm_calls. Syncs add overhead: use for the split, not totals.
--log_detail True: per verify pass, for each decided action token: action dim (0..6), source (prefill / draft accepted /
  draft rejected / bonus), draft token, verifier argmax token, verifier p(argmax), p(draft), mass within +-relax_threshold
  bins; plus per step gripper qpos, eef pos, gripper-contact flag, executed action. For error analysis.
Run with PYTHONPATH=$PROJ/third_party/SpecVLA:$PROJ/third_party/SpecVLA/openvla (see specvla_eval.sbatch).
"""
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import draccus
import numpy as np
import torch
from libero.libero import benchmark

from experiments.robot.libero.libero_utils import (
    get_libero_dummy_action,
    get_libero_env,
    get_libero_image,
    quat2axisangle,
    save_rollout_video,
)
from experiments.robot.openvla_utils import get_processor
from experiments.robot.robot_utils import (
    get_action,
    get_image_resize_size,
    get_model,
    invert_gripper_action,
    normalize_gripper_action,
    set_seed_everywhere,
)

MAX_STEPS = {"libero_spatial": 220, "libero_object": 280, "libero_goal": 300, "libero_10": 520, "libero_90": 400}


def _sync_timer(store, key):
    """forward pre/post hooks that add CUDA-synced elapsed ms to store[key]."""
    t0 = {}
    def pre(*_):
        torch.cuda.synchronize(); t0["t"] = time.perf_counter()
    def post(*_):
        torch.cuda.synchronize(); store[key] = store.get(key, 0.0) + 1000 * (time.perf_counter() - t0["t"])
    return pre, post


def install_profiler(model, prof):
    """Time vision encoder, every LLM call (1st per step = prefill, rest = verify/decode), and drafter."""
    base = model.base_model
    vpre, vpost = _sync_timer(prof, "vision")
    base.vision_backbone.register_forward_pre_hook(vpre)
    base.vision_backbone.register_forward_hook(vpost)
    t0 = {}
    def lpre(*_):
        torch.cuda.synchronize(); t0["t"] = time.perf_counter(); t0["v"] = prof.get("vision", 0.0)
    def lpost(*_):
        torch.cuda.synchronize()
        dt = 1000 * (time.perf_counter() - t0["t"]) - (prof.get("vision", 0.0) - t0["v"])  # LLM part only
        key = "prefill" if prof.get("n_llm_calls", 0) == 0 else "verify"
        prof[key] = prof.get(key, 0.0) + dt
        prof["n_llm_calls"] = prof.get("n_llm_calls", 0) + 1
    base.register_forward_pre_hook(lpre)
    base.register_forward_hook(lpost)
    if hasattr(model, "ea_layer"):
        orig = model.ea_layer.topK_genrate
        def timed(*a, **k):
            torch.cuda.synchronize(); s = time.perf_counter()
            out = orig(*a, **k)
            torch.cuda.synchronize(); prof["draft"] = prof.get("draft", 0.0) + 1000 * (time.perf_counter() - s)
            return out
        model.ea_layer.topK_genrate = timed


def install_detail_logger(detail, r):
    """Wrap SpecVLA's initialize_tree / evaluate_posterior (module globals used by eagenerate) to log each
    decided token. Token ids map 1:1 to action bins, so |draft - argmax| is the bin distance."""
    import openvla.prismatic.extern.hf.modeling_speculation as ms
    pos = {"n": 0}
    def tok_info(logit_row, draft_tok):
        p = torch.softmax(logit_row.float(), -1)
        a = int(torch.argmax(p))
        lo, hi = max(a - r, 0), min(a + r, p.shape[0] - 1)
        return a, float(p[a]), (None if draft_tok is None else float(p[draft_tok])), float(p[lo:hi + 1].sum())
    _init = ms.initialize_tree
    def init(*a, **k):
        out = _init(*a, **k)
        am, pa, _, mr = tok_info(out[4][0, -1], None)  # out[4] = orig logits of prefill
        detail.append({"dim": 0, "src": "prefill", "draft": None, "verif": am, "p_top": pa, "p_draft": None, "mass_r": mr})
        pos["n"] = 1
        return out
    ms.initialize_tree = init
    _ev = ms.evaluate_posterior
    def ev(logits, candidates, *a, **k):
        best, L, sp = _ev(logits, candidates, *a, **k)
        b, L = int(best), int(L)
        for j in range(min(L + 1, candidates.shape[1] - 1)):
            d = int(candidates[b, j + 1])
            if d < 0:
                break
            am, pa, pd, mr = tok_info(logits[b, j], d)
            detail.append({"dim": pos["n"] + j, "src": "accept" if j < L else "reject", "draft": d, "verif": am,
                           "p_top": pa, "p_draft": pd, "mass_r": mr})
        am, pa, _, mr = tok_info(logits[b, L], None)
        detail.append({"dim": pos["n"] + L, "src": "bonus", "draft": None, "verif": am, "p_top": pa, "p_draft": None,
                       "mass_r": mr})
        pos["n"] += L + 1
        return best, L, sp
    ms.evaluate_posterior = ev


def gripper_contact(env):
    """True if any gripper geom touches anything (robosuite check_contact); None if unavailable."""
    try:
        e = env.env
        return bool(e.check_contact(e.robots[0].gripper))
    except Exception:
        return None


@dataclass
class GenerateConfig:
    # Fields read by SpecVLA's get_model / get_action
    model_family: str = "openvla"
    pretrained_checkpoint: str = ""          # LOCAL path to the fine-tuned OpenVLA (drafter loads embeddings from it)
    spec_checkpoint: str = ""                # dir with the drafter's config.json + pytorch_model.bin
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    center_crop: bool = True
    use_spec: bool = True                    # always wrap with SpecVLA, even for AR, so the harness matches
    parallel_draft: bool = False
    accept_threshold: Optional[int] = None

    mode: str = "relaxed"                    # ar | spec | relaxed
    relax_threshold: int = 9                 # used in relaxed mode (paper's default r=9)
    task_suite_name: str = "libero_goal"
    task_id: int = 0
    num_trials_per_task: int = 50
    num_steps_wait: int = 10
    out_dir: str = "results/specvla"
    save_videos: bool = False
    log_accept: bool = False                 # record per-verify-pass accept lengths (adds a GPU sync per pass)
    profile: bool = False                    # CUDA-synced phase timing (see docstring)
    log_detail: bool = False                 # per-token draft/verifier logging + contact state (see docstring)
    seed: int = 7


@draccus.wrap()
def main(cfg: GenerateConfig) -> None:
    assert cfg.mode in ("ar", "spec", "relaxed")
    cfg.accept_threshold = {"ar": None, "spec": 0, "relaxed": cfg.relax_threshold}[cfg.mode]
    generate_mode = "AR" if cfg.mode == "ar" else "speculative"
    set_seed_everywhere(cfg.seed)
    accept_log = []
    if cfg.log_accept:  # wrap the module-level function eagenerate calls, without editing SpecVLA's code
        import openvla.prismatic.extern.hf.modeling_speculation as ms
        _orig = ms.evaluate_posterior
        def _logged(*a, **k):
            out = _orig(*a, **k)
            accept_log.append(int(out[1]))
            return out
        ms.evaluate_posterior = _logged

    cfg.unnorm_key = cfg.task_suite_name
    model = get_model(cfg)
    prof, detail = {}, []
    if cfg.profile:
        install_profiler(model, prof)
    if cfg.log_detail:
        install_detail_logger(detail, cfg.relax_threshold)
    if cfg.unnorm_key not in model.norm_stats and f"{cfg.unnorm_key}_no_noops" in model.norm_stats:
        cfg.unnorm_key = f"{cfg.unnorm_key}_no_noops"
    processor = get_processor(cfg)
    resize_size = get_image_resize_size(cfg)

    task_suite = benchmark.get_benchmark_dict()[cfg.task_suite_name]()
    task = task_suite.get_task(cfg.task_id)
    initial_states = task_suite.get_task_init_states(cfg.task_id)
    env, task_description = get_libero_env(task, cfg.model_family, resolution=256)
    max_steps = MAX_STEPS[cfg.task_suite_name]

    os.makedirs(cfg.out_dir, exist_ok=True)
    out_path = Path(cfg.out_dir) / f"{cfg.task_suite_name}_task{cfg.task_id}.json"
    record = {"config": {k: str(v) for k, v in vars(cfg).items()}, "task": task_description,
              "gpu": torch.cuda.get_device_name(0), "episodes": []}

    for ep in range(cfg.num_trials_per_task):
        env.reset()
        obs = env.set_init_state(initial_states[ep])
        t, done, step_ms, replay, cycles, profs, details, states = 0, False, [], [], [], [], [], []
        while t < max_steps + cfg.num_steps_wait:
            if t < cfg.num_steps_wait:
                obs, _, done, _ = env.step(get_libero_dummy_action(cfg.model_family))
                t += 1
                continue
            img = get_libero_image(obs, resize_size)
            if cfg.save_videos:
                replay.append(img)
            observation = {
                "full_image": img,
                "state": np.concatenate(
                    (obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])
                ),
            }
            action, (end, start) = get_action(cfg, model, observation, task_description, processor=processor,
                                              return_time=True, generate_mode=generate_mode)
            step_ms.append(1000 * (end - start))
            cycles.append(accept_log[:]); accept_log.clear()  # one entry per verify pass (empty unless log_accept)
            if cfg.profile:
                profs.append({k: round(v, 3) for k, v in prof.items()}); prof.clear()
            action = invert_gripper_action(normalize_gripper_action(action, binarize=True))
            if cfg.log_detail:
                details.append(detail[:]); detail.clear()
                states.append({"gripper_qpos": np.round(obs["robot0_gripper_qpos"], 4).tolist(),
                               "eef_pos": np.round(obs["robot0_eef_pos"], 4).tolist(),
                               "contact": gripper_contact(env), "action": np.round(action, 4).tolist()})
            obs, _, done, _ = env.step(action.tolist())
            if done:
                break
            t += 1
        ep_rec = {"success": bool(done), "step_ms": step_ms, "accept_lengths": cycles}
        if cfg.profile:
            ep_rec["profile"] = profs
        if cfg.log_detail:
            ep_rec["detail"], ep_rec["state"] = details, states
        record["episodes"].append(ep_rec)
        if cfg.save_videos:
            save_rollout_video(replay, ep + 1, success=done, task_description=task_description)
        n_succ = sum(e["success"] for e in record["episodes"])
        print(f"ep {ep + 1}: success={done} steps={len(step_ms)} median_ms={np.median(step_ms[1:] or step_ms):.1f} "
              f"| task success {n_succ}/{ep + 1}", flush=True)
        with open(out_path, "w") as f:  # rewrite each episode so a wall-kill keeps partial results
            json.dump(record, f)


if __name__ == "__main__":
    main()
