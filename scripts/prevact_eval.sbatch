#!/bin/bash
# Full success run of the previous-action draft (scripts/prevact_eval.py, MODE=prev by default) on all 40 LIBERO tasks.
# Array index = suite*10 + task; 50 trials each. Output: results/prevact_full/<mode>/<suite>_t<task>.json.
# Submit from project root:  sbatch --array=0-39%4 scripts/prevact_eval.sbatch
#SBATCH --job-name=prevfull
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
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.out
source ${SLURM_SUBMIT_DIR:-$PWD}/env.sh
echo "NODE=$(hostname) $(date)"
python -c "import torch; torch.zeros(8, device='cuda:0'); f=torch.cuda.mem_get_info()[0]/2**30; print(f'free GPU mem {f:.1f} GiB'); assert f > 40" || exit 76
SUITES=(libero_spatial libero_object libero_goal libero_10); CK=(spatial object goal 10)
S=$((SLURM_ARRAY_TASK_ID / 10)); T=$((SLURM_ARRAY_TASK_ID % 10)); MODE=${MODE:-prev}
BASE=$(python -c "from huggingface_hub import snapshot_download as s; print(s('openvla/openvla-7b-finetuned-libero-${CK[$S]}', local_files_only=True))")
R=$PROJ/results/prevact_full/$MODE; mkdir -p $R; cd $R
PYTHONPATH=$PROJ/third_party/openvla python $PROJ/scripts/prevact_eval.py --pretrained_checkpoint $BASE \
  --task_suite_name ${SUITES[$S]} --task_id $T --mode $MODE --num_trials_per_task 50 --out $R/${SUITES[$S]}_t$T.json
echo "DONE $(date)"
