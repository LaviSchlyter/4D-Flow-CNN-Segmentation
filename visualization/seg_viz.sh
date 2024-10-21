#!/bin/bash
#SBATCH  --output=../logs/visualization_logs/%j.out
#SBATCH  --gres=gpu:1
#SBATCH  --mem=50G

CONFIG_PATH="/usr/bmicnas02/data-biwi-01/jeremy_students/lschlyter/CNN-segmentation/config/config_inference.yaml"
USE_CONFIG="False"
source /scratch_net/biwidl203/lschlyter/anaconda3/etc/profile.d/conda.sh
conda activate seg

if [ "$USE_CONFIG" = "True" ]; then
    python -u segmentation_visualization.py --config_path "$CONFIG_PATH"
else
    python -u segmentation_visualization.py "$@"
fi

