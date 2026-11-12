#!/bin/bash
# Phase-time profile (vision / prefill / verify / draft, CUDA-synced) for ar, spec, relaxed r9 on task 0 of
# each suite, 3 episodes, all in ONE job on one A40. Runs scripts/specvla_eval.sbatch's body in a loop.
# Output: results/profile/<mode>/<suite>_task0.json ("profile" per step). Submit:
#   sbatch -p kira-lab -A kira-lab -q short --ntasks=1 --gpus-per-task=a40:1 -c 6 --mem=48G -t 03:55:00 \
#     --exclude=kitt,ig-88,cyborg,sonny,spd-13,deebot,synapse,voltron,megazord,gundam \
#     -J profile -o logs/profile_%j.out scripts/profile_phases.sh
source ${SLURM_SUBMIT_DIR:-$PWD}/env.sh
for IDX in 0 10 20 30; do for M in ar spec relaxed; do
  SLURM_ARRAY_TASK_ID=$IDX MODE=$M TRIALS=3 OUT_ROOT=results/profile EXTRA_ARGS="--profile True" \
    bash $PROJ/scripts/specvla_eval.sbatch
done; done
