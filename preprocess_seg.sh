#!/bin/bash
#SBATCH  --output=logs/sbatch_log_bern_preprocess/%j.out
#SBATCH  --gres=gpu:1
#SBATCH  --mem=50G
source /scratch_net/biwidl203/lschlyter/anaconda3/etc/profile.d/conda.sh
conda activate seg
python -u bern_numpy_to_hdf5_for_training.py "$@"