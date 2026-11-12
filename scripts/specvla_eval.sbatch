#!/bin/bash
# Spec-VLA on LIBERO via scripts/specvla_eval.py. Array index = suite_idx*10 + task_id (0-39).
# MODE=ar|spec|relaxed (default relaxed), R=relax threshold (default 9), TRIALS (default 50).
# Usage: MODE=relaxed R=9 sbatch scripts/specvla_eval.sbatch
# OUT_ROOT=results/analysis (default results/specvla) + EXTRA_ARGS="--profile True" / "--log_detail True".
#        MODE=ar TRIALS=1 sbatch --array=20 --qos=debug --time=00:30:00 scripts/specvla_eval.sbatch   (smoke)
#SBATCH --job-name=specvla
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
#SBATCH --exclude=kitt,ig-88,cyborg,sonny,spd-13,deebot,synapse,voltron,megazord,gundam
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.out
source ${SLURM_SUBMIT_DIR:-$PWD}/env.sh  # submit from the project root
echo "NODE=$(hostname) GPU=$(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader|head -1) $(date)"
python -c "import torch; torch.zeros(8, device='cuda:0'); f=torch.cuda.mem_get_info()[0]/2**30; print(f'free GPU mem {f:.1f} GiB'); assert f > 40, 'GPU already in use by another process'" || { echo "!!! bad/busy GPU on $(hostname -s)"; exit 76; }
SUITES=(libero_spatial libero_object libero_goal libero_10)
CKPTS=(spatial object goal 10)
IDX=${SLURM_ARRAY_TASK_ID:-0}; S=$((IDX / 10)); T=$((IDX % 10))
MODE=${MODE:-relaxed}; R=${R:-9}
BASE=$(python -c "from huggingface_hub import snapshot_download as s; print(s('openvla/openvla-7b-finetuned-libero-${CKPTS[$S]}', local_files_only=True))")
TAG=$MODE; [ "$MODE" = relaxed ] && TAG=relaxed_r$R
OUT=$PROJ/${OUT_ROOT:-results/specvla}/$TAG; mkdir -p $OUT; cd $OUT
export PYTHONPATH=$PROJ/third_party/SpecVLA:$PROJ/third_party/SpecVLA/openvla
python $PROJ/scripts/specvla_eval.py \
  --pretrained_checkpoint $BASE \
  --spec_checkpoint $PROJ/checkpoints/specvla_drafters/${SUITES[$S]} \
  --mode $MODE --relax_threshold $R \
  --task_suite_name ${SUITES[$S]} --task_id $T \
  --num_trials_per_task ${TRIALS:-50} --out_dir $OUT ${EXTRA_ARGS}
echo "EXIT=$? $(date)"
