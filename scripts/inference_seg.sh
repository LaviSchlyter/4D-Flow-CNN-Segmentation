#!/bin/bash
#SBATCH  --output=../logs/inference_logs/%j.out
#SBATCH  --gres=gpu:1
#SBATCH  --mem=50G

CONFIG_PATH="/usr/bmicnas02/data-biwi-01/jeremy_students/lschlyter/CNN-segmentation/config/config_inference.yaml"
VISUALIZE_3D_SEG="False" # True does not work on the cluster yet
VISUALIZE_CENTERLINE_POSTPROCESSING="False" 
VISUALIZE_CENTERLINE_CROSS_SECTION="True"
source /scratch_net/biwidl203/lschlyter/anaconda3/etc/profile.d/conda.sh
conda activate seg

# Main inference script to run
#python -u ../src/inference/bern_inference.py --config_path "$CONFIG_PATH"

# If visualization is enabled, run the visualization script
if [ "$VISUALIZE_3D_SEG" = "True" ]; then
    
    echo "VISUALIZE_3D_SEG is enabled"
    python -u ../visualization/segmentation_visualization.py --config_path "$CONFIG_PATH"
fi

# If Visualization of centerline postprocessing is enabled, run the visualization script
if [ "$VISUALIZE_CENTERLINE_POSTPROCESSING" = "True" ]; then
    
    echo "VISUALIZE_CENTERLINE_POSTPROCESSING is enabled"
    python -u ../src/inference/cnn_seg_centerline_extraction.py --config_path "$CONFIG_PATH"
fi
# If Visualization of centerline cross section is enabled, run the visualization script
if [ "$VISUALIZE_CENTERLINE_CROSS_SECTION" = "True" ]; then
    
    echo "VISUALIZE_CENTERLINE_CROSS_SECTION is enabled"
    python -u ../src/inference/cnn_seg_cross_sectional_slices.py --config_path "$CONFIG_PATH"
fi

# finished
echo "Inference script finished"

