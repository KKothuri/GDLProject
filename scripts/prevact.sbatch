#!/bin/bash
# Previous-action draft (scripts/prevact_eval.py). One job, one A40, goal task0 by default:
#   1) prev mode with --check (token match vs stock generate, 1 episode)  2) hf vs prev timing, EPS episodes each.
# Override SUITE, TASK, EPS. Output: results/prevact/<suite>_t<task>_{check,hf,prev}.json. Submit from project root.
#SBATCH --job-name=prevact
#SBATCH --partition=kira-lab
#SBATCH --account=kira-lab
#SBATCH --qos=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=a40:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --exclude=kitt,ig-88,cyborg,sonny,spd-13,deebot,synapse,voltron,megazord,gundam
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.out
source ${SLURM_SUBMIT_DIR:-$PWD}/env.sh
echo "NODE=$(hostname) $(date)"
python -c "import torch; torch.zeros(8, device='cuda:0'); f=torch.cuda.mem_get_info()[0]/2**30; print(f'free GPU mem {f:.1f} GiB'); assert f > 40" || exit 76
SUITE=${SUITE:-libero_goal}; TASK=${TASK:-0}; EPS=${EPS:-3}
CK=${SUITE#libero_}
BASE=$(python -c "from huggingface_hub import snapshot_download as s; print(s('openvla/openvla-7b-finetuned-libero-$CK', local_files_only=True))")
R=$PROJ/results/prevact; mkdir -p $R; cd $R
export PYTHONPATH=$PROJ/third_party/openvla
A="--pretrained_checkpoint $BASE --task_suite_name $SUITE --task_id $TASK"
for M in ${CHECK_MODES:-prev}; do python $PROJ/scripts/prevact_eval.py $A --mode $M --check True --num_trials_per_task 1 --out $R/${SUITE}_t${TASK}_check_$M.json; done
[ -n "$CHECK_ONLY" ] && exit 0
for M in hf prev; do
  python $PROJ/scripts/prevact_eval.py $A --mode $M --num_trials_per_task $EPS --out $R/${SUITE}_t${TASK}_$M.json
done
echo "DONE $(date)"
