#!/bin/bash
# Fair-AR check, one job on one A40: stock OpenVLA HF generate (scripts/ar_timing.py) vs SpecVLA AR mode (ea_forward)
# vs SpecVLA relaxed r9, goal task 0, 3 episodes each. Output: results/ar_timing/{hf.json, ar/, relaxed_r9/}.
#SBATCH --job-name=artiming
#SBATCH --partition=kira-lab
#SBATCH --account=kira-lab
#SBATCH --qos=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=a40:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=01:30:00
#SBATCH --exclude=kitt,ig-88,cyborg,sonny,spd-13,deebot,synapse,voltron,megazord,gundam
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.out
source ${SLURM_SUBMIT_DIR:-$PWD}/env.sh  # submit from the project root
echo "NODE=$(hostname) $(date)"
python -c "import torch; torch.zeros(8, device='cuda:0'); f=torch.cuda.mem_get_info()[0]/2**30; print(f'free GPU mem {f:.1f} GiB'); assert f > 40" || exit 76
BASE=$(python -c "from huggingface_hub import snapshot_download as s; print(s('openvla/openvla-7b-finetuned-libero-goal', local_files_only=True))")
R=$PROJ/results/ar_timing; mkdir -p $R; cd $R
PYTHONPATH=$PROJ/third_party/openvla python $PROJ/scripts/ar_timing.py --pretrained_checkpoint $BASE --out $R/hf.json
for M in ar relaxed; do
  TAG=$M; [ $M = relaxed ] && TAG=relaxed_r9
  PYTHONPATH=$PROJ/third_party/SpecVLA:$PROJ/third_party/SpecVLA/openvla python $PROJ/scripts/specvla_eval.py \
    --mode $M --pretrained_checkpoint $BASE --spec_checkpoint $PROJ/checkpoints/specvla_drafters/libero_goal \
    --task_suite_name libero_goal --task_id 0 --num_trials_per_task 3 --out_dir $R/$TAG --profile True
done
echo "DONE $(date)"
