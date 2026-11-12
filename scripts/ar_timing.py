"""Time stock OpenVLA (HF generate via vla.predict_action, as in the OpenVLA baseline eval) per step, CUDA-synced,
and count LLM forward calls per action. Closed loop on one LIBERO task. Used by scripts/ar_timing.sbatch to check
whether SpecVLA's AR mode (ea_forward: 8 LLM calls for 7 tokens) is slower than the fair AR reference.
Run with PYTHONPATH=$PROJ/third_party/openvla. Writes <out> JSON: per-episode step_ms, llm_calls, success."""
import json
import sys
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
    out: str = "results/ar_timing/hf.json"
    seed: int = 7


@draccus.wrap()
def main(cfg: Cfg):
    set_seed_everywhere(cfg.seed)
    cfg.unnorm_key = cfg.task_suite_name
    model = get_model(cfg)
    if cfg.unnorm_key not in model.norm_stats:
        cfg.unnorm_key += "_no_noops"
    processor = get_processor(cfg)
    calls = {"n": 0}
    model.language_model.register_forward_hook(lambda *_: calls.__setitem__("n", calls["n"] + 1))
    suite = benchmark.get_benchmark_dict()[cfg.task_suite_name]()
    task = suite.get_task(cfg.task_id)
    init = suite.get_task_init_states(cfg.task_id)
    env, desc = get_libero_env(task, cfg.model_family, resolution=256)
    rec = {"gpu": torch.cuda.get_device_name(0), "episodes": []}
    for ep in range(cfg.num_trials_per_task):
        env.reset(); obs = env.set_init_state(init[ep])
        t, done, ms, nc = 0, False, [], []
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
            obs, _, done, _ = env.step(invert_gripper_action(normalize_gripper_action(a, binarize=True)).tolist())
            if done:
                break
            t += 1
        rec["episodes"].append({"success": bool(done), "step_ms": ms, "llm_calls": nc})
        print(f"ep {ep + 1}: success={done} mean_ms={np.mean(ms[1:]):.1f} llm_calls={np.median(nc)}", flush=True)
    import os; os.makedirs(os.path.dirname(cfg.out), exist_ok=True)
    json.dump(rec, open(cfg.out, "w"))


if __name__ == "__main__":
    main()
