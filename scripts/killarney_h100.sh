#!/bin/bash
#SBATCH -A aip-aspuru
#SBATCH -D /project/aip-aspuru/aburger/hip
#SBATCH --time=71:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=128GB
#SBATCH --cpus-per-task=4
#SBATCH --job-name=hiph100
# Jobs must write their output to your scratch or project directory (home is read-only on compute nodes).
#SBATCH --output=/project/aip-aspuru/aburger/hip/outslurm/slurm-%j.txt 
#SBATCH --error=/project/aip-aspuru/aburger/hip/outslurm/slurm-%j.txt

# get environment variables
source .env
export WANDB_ENTITY=andreas-burger

# activate venv
source ${PYTHONBIN}/activate

#module load cuda/12.6
#module load gcc/12.3

# append command to slurmlog.txt
echo "sbatch scripts/killarney_h100.sh $@ # $SLURM_JOB_ID" >> slurmlog.txt

echo `date`: Job $SLURM_JOB_ID is allocated resources.
echo "Inside slurm_launcher.slrm ($0). received arguments: $@"

# hand over all arguments to the script
echo "Submitting ${HOMEROOT}/$@"

srun ${PYTHONBIN}/python ${HOMEROOT}/"$@"
