#!/bin/bash
#SBATCH  --output=logs/sbatch_log/%j.out
#SBATCH  --gres=gpu:1
#SBATCH  --mem=50G
source /scratch_net/biwidl203/lschlyter/anaconda3/etc/profile.d/conda.sh
conda activate hpc
python -u train.py "$@"