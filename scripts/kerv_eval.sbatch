#!/bin/bash
# KERV on LIBERO: runs third_party/KERV/openvla/experiments/robot/libero/run_kerv_libero.py directly
# (no FlagScale launcher, runtime/kernel optimizations OFF = algorithm only: Kalman compensation + threshold schedule).
# Drafter = Spec-VLA's released drafter (same 1-block LLaMA architecture; KERV ships no weights).
# Array index = suite_idx*10 + task_id. TRIALS (default 50). Threshold 15 -> 5 as in the KERV paper.
# use_kalman_tree OFF: that runtime extension (Kalman paths added to the verify tree) doubles per-pass cost in plain
# PyTorch (474 vs 238 ms/step, goal task0) without cutting passes; only pays off with their static-tree/CUDA-graph runtime.
# OUT_TAG=_noKF writes to results/kerv_noKF/. KERV_FORCE_KF_EVERY=0 disables the forced Kalman completion (patched, default 4).
# EXTRA_ARGS="..." appends flags to run_kerv_libero.py. Smoke: TRIALS=1 sbatch --array=20 --qos=debug --time=00:30:00 scripts/kerv_eval.sbatch
#SBATCH --job-name=kerv
#SBATCH --partition=kira-lab
#SBATCH --account=kira-lab
#SBATCH --qos=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=a40:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=03:55:00
#SBATCH --array=0-39%8
#SBATCH --exclude=kitt,ig-88,cyborg,sonny,spd-13,deebot,synapse
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.out
source ${SLURM_SUBMIT_DIR:-$PWD}/env.sh  # submit from the project root
echo "NODE=$(hostname) GPU=$(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader|head -1) $(date)"
python -c "import torch; torch.zeros(8, device='cuda:0'); f=torch.cuda.mem_get_info()[0]/2**30; print(f'free GPU mem {f:.1f} GiB'); assert f > 40, 'GPU already in use by another process'" || { echo "!!! bad/busy GPU on $(hostname -s)"; exit 76; }
SUITES=(libero_spatial libero_object libero_goal libero_10)
CKPTS=(spatial object goal 10)
MAXSTEPS=(220 280 300 520)
IDX=${SLURM_ARRAY_TASK_ID:-0}; S=$((IDX / 10)); T=$((IDX % 10))
BASE=$(python -c "from huggingface_hub import snapshot_download as s; print(s('openvla/openvla-7b-finetuned-libero-${CKPTS[$S]}', local_files_only=True))")
OUT=$PROJ/results/kerv${OUT_TAG}/${SUITES[$S]}; mkdir -p $OUT; cd $OUT   # rollout videos go to ./rollouts
K=$PROJ/third_party/KERV
export PYTHONPATH=$K/runtime_opt:$K:$K/openvla
python $K/openvla/experiments/robot/libero/run_kerv_libero.py \
  --pretrained_checkpoint $BASE \
  --spec_checkpoint $PROJ/checkpoints/specvla_drafters/${SUITES[$S]} \
  --task_suite_name ${SUITES[$S]} --task_id $T \
  --num_trials_per_task ${TRIALS:-50} \
  --max_planning_steps ${MAXSTEPS[$S]} \
  --accept_threshold_start 15 --accept_threshold_min 5 --use_kalman_tree False \
  --center_crop True --local_log_dir $OUT/logs ${EXTRA_ARGS}
echo "EXIT=$? $(date)"
