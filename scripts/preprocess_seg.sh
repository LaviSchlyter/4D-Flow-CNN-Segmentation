#!/bin/bash
#SBATCH  --output=../logs/preprocess_logs/%j.out
#SBATCH  --gres=gpu:1
#SBATCH  --mem=50G
source /scratch_net/biwidl203/lschlyter/anaconda3/etc/profile.d/conda.sh
conda activate segmentation_pipeline
python -u ../src/helpers/bern_numpy_to_hdf5_for_training.py "$@"

echo "Preprocessing script finished"