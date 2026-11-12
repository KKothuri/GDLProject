## Gotchas
- Headless rendering on the cluster: set `MUJOCO_GL=egl`. Get one LIBERO episode rendering in week 1.
- Benchmark baseline and our method on the **same A40**; report relative speedups and state the hardware.
  Published numbers are on RTX 4090s, which have more memory bandwidth.
- Don't report latency from the laptop (int4 quantization changes timing).
- Spec-VLA code was not found publicly as of Oct 2026; we may need to reimplement the draft model.
  Spec-VLA trained its draft on 4xA100-80GB for ~6 hours; expect longer on 1–2 A40s.
## Timeline
| Week | Plan |
|---|---|
| Oct 26 – Nov 1 | Set up LIBERO + OpenVLA on A40s; headless rendering; reproduce baseline success |
| Nov 2 – 8 | Profile latency (prefill vs decode); exact speculative decoding; confidence-vs-contact check |
| Nov 9 – 15 | Train draft model; reproduce Spec-VLA baseline |
| Nov 16 – 22 | Build our acceptance rule; midpoint check |
| Nov 23 – 29 | Full runs on 4 LIBERO suites; ablations |
| Nov 30 – Dec 6 | Analysis; poster |
| Dec 7 / Dec 8 | Poster session / final report |
 
Fallback: if our rule doesn't win, the latency study and error analysis are still a complete project.
 
## Related work
- Spec-VLA (EMNLP 2025): https://arxiv.org/abs/2507.22424 — 1.08–1.15x exact, 1.22–1.42x relaxed.
- KERV (DAC 2026): https://arxiv.org/abs/2603.01581 — kinematic thresholds + Kalman filter.
- WA-SpecDec (Aug 2026): https://arxiv.org/abs/2608.08725 — world-model scene info; 1.5x at matched success.
- FLASH (speculative inference for pi0): https://arxiv.org/abs/2605.13778
- SV-VLA (speculative verification): https://arxiv.org/abs/2604.02965
- A3 / Dynamic Execution Commitment: https://arxiv.org/abs/2605.11567
- Fuzzy Speculative Decoding (LLM, divergence-based acceptance): https://arxiv.org/abs/2502.20704
- Practical investigation of training-free relaxed speculative decoding: https://arxiv.org/abs/2607.08690
- OpenVLA: https://arxiv.org/abs/2406.09246
- TurboVLA (no LLM, 31 ms/step, 97.7% LIBERO): https://arxiv.org/abs/2607.27205
- Survey of efficient VLAs: https://arxiv.org/abs/2510.24795
- Also cite: Leviathan et al. 2023 (speculative decoding), EAGLE-2, LIBERO, OpenVLA-OFT, pi0, FAST.
## How to work with me
- Keep explanations and any written text plain and simple, not "AI-sounding".
- Explain things step by step when I ask.
- Be honest about novelty and feasibility.
