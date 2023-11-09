#!/bin/bash
#SBATCH  --output=logs/sbatch_log/%j.out
#SBATCH  --gres=gpu:1
#SBATCH  --mem=40G
#SBATCH  --account=staff
source /scratch_net/biwidl203/lschlyter/anaconda3/etc/profile.d/conda.sh
conda activate seg_net
python -u train.py "$@"