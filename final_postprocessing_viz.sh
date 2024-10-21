#!/bin/bash
#SBATCH  --output=logs/final_visualization_logs/%j.out
#SBATCH  --gres=gpu:1
#SBATCH  --mem=50G
source /scratch_net/biwidl203/lschlyter/anaconda3/etc/profile.d/conda.sh
conda activate seg
python -u final_segmentations_centerline_postprocessing_viz.py "$@"

echo "Final visualization script finished"