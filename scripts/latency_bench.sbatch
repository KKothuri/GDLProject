#!/bin/bash
# Same-GPU latency comparison: all 5 methods run back to back in ONE job (one A40), closed loop,
# task 0 of each suite, EPS episodes (default 3). Per-step model latency comes from each method's own logs:
#   results/latency/<method>/...  (specvla JSONs: step_ms; KERV: logs/*/*stats.json [start,end] pairs)
# Methods: ar, spec (exact), relaxed_r9 (scripts/specvla_eval.py); kerv (released, forced Kalman every 4th),
# kerv_noKF (KERV_FORCE_KF_EVERY=0). Summarize with scripts/latency_summary.py.
#SBATCH --job-name=latbench
#SBATCH --partition=kira-lab
#SBATCH --account=kira-lab
#SBATCH --qos=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=a40:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=03:55:00
#SBATCH --exclude=kitt,ig-88,cyborg,sonny,spd-13,deebot,synapse,voltron,megazord,gundam
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.out
source ${SLURM_SUBMIT_DIR:-$PWD}/env.sh  # submit from the project root
echo "NODE=$(hostname) GPU=$(nvidia-smi --query-gpu=name,driver_version,clocks.max.sm --format=csv,noheader|head -1) $(date)"
python -c "import torch; torch.zeros(8, device='cuda:0'); f=torch.cuda.mem_get_info()[0]/2**30; print(f'free GPU mem {f:.1f} GiB'); assert f > 40" || { echo "!!! busy GPU"; exit 76; }
EPS=${EPS:-3}
SUITES=(libero_spatial libero_object libero_goal libero_10)
CKPTS=(spatial object goal 10)
MAXSTEPS=(220 280 300 520)
R=$PROJ/results/latency; mkdir -p $R
K=$PROJ/third_party/KERV
for S in 0 1 2 3; do
  SUITE=${SUITES[$S]}
  BASE=$(python -c "from huggingface_hub import snapshot_download as s; print(s('openvla/openvla-7b-finetuned-libero-${CKPTS[$S]}', local_files_only=True))")
  DRAFT=$PROJ/checkpoints/specvla_drafters/$SUITE
  for MODE in ar spec relaxed; do
    TAG=$MODE; [ $MODE = relaxed ] && TAG=relaxed_r9
    echo "=== $SUITE $TAG $(date)"
    (cd $R && PYTHONPATH=$PROJ/third_party/SpecVLA:$PROJ/third_party/SpecVLA/openvla python $PROJ/scripts/specvla_eval.py \
      --mode $MODE --relax_threshold 9 --pretrained_checkpoint $BASE --spec_checkpoint $DRAFT \
      --task_suite_name $SUITE --task_id 0 --num_trials_per_task $EPS --out_dir $R/$TAG)
  done
  for KF in 4 0; do
    TAG=kerv; [ $KF = 0 ] && TAG=kerv_noKF
    echo "=== $SUITE $TAG $(date)"
    mkdir -p $R/$TAG/$SUITE
    (cd $R/$TAG/$SUITE && KERV_FORCE_KF_EVERY=$KF PYTHONPATH=$K/runtime_opt:$K:$K/openvla python $K/openvla/experiments/robot/libero/run_kerv_libero.py \
      --pretrained_checkpoint $BASE --spec_checkpoint $DRAFT \
      --task_suite_name $SUITE --task_id 0 --num_trials_per_task $EPS \
      --max_planning_steps ${MAXSTEPS[$S]} \
      --accept_threshold_start 15 --accept_threshold_min 5 --use_kalman_tree False \
      --center_crop True --local_log_dir $R/$TAG/$SUITE/logs)
  done
done
echo "DONE $(date)"
