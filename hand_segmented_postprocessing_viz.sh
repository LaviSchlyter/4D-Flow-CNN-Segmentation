#!/bin/bash
#SBATCH  --output=logs/hand_segmented_postprocessing_logs/%j.out
#SBATCH  --gres=gpu:1
#SBATCH  --mem=50G
source /scratch_net/biwidl203/lschlyter/anaconda3/etc/profile.d/conda.sh
conda activate seg
python -u hand_segmented_centerline_postprocessing_viz.py "$@"

echo "Hand segmented postprocessing visualization script finished"