#!/bin/bash
# OpenVLA baseline on LIBERO. Array index = suite_idx*10 + task_id (0-39), one task (50 trials) per job.
# Usage: sbatch scripts/libero_baseline.sbatch          (full: 4 suites x 10 tasks)
#        TRIALS=2 sbatch --array=0 --qos=debug --time=00:30:00 scripts/libero_baseline.sbatch   (smoke test)
#SBATCH --job-name=vla_base
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
IDX=${SLURM_ARRAY_TASK_ID:-0}; S=$((IDX / 10)); T=$((IDX % 10))
OUT=$PROJ/results/baseline_openvla/${SUITES[$S]}; mkdir -p $OUT; cd $OUT   # rollout videos go to ./rollouts
python $PROJ/third_party/openvla/experiments/robot/libero/run_libero_eval.py \
  --model_family openvla \
  --pretrained_checkpoint openvla/openvla-7b-finetuned-libero-${CKPTS[$S]} \
  --task_suite_name ${SUITES[$S]} --task_id $T \
  --num_trials_per_task ${TRIALS:-50} \
  --center_crop True --local_log_dir $OUT/logs
echo "EXIT=$? $(date)"
