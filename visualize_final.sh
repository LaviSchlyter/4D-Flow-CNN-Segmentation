#!/bin/bash
#SBATCH  --output=logs/sbatch_log_bern_centerline_viz/%j.out
#SBATCH  --gres=gpu:1
#SBATCH  --mem=50G
source /scratch_net/biwidl203/lschlyter/anaconda3/etc/profile.d/conda.sh
conda activate vtk_wrap
python -u visualize_final_segmentations.py "$@"