#!/bin/bash
#SBATCH  --gres=gpu:1
#SBATCH  --mem=40G

# --> Configure before running
CONFIG_PATH="/usr/bmicnas02/data-biwi-01/jeremy_students/lschlyter/CNN-segmentation/config/config_train.yaml"

# Define a timestamp for the log filename
TIMESTAMP=$(date +'%Y%m%d_%H%M')
LOG_DIR="../logs/train_logs"
LOG_FILE="${LOG_DIR}/${TIMESTAMP}_${SLURM_JOB_ID}.out"

# Ensure the log directory exists
mkdir -p $LOG_DIR

source /scratch_net/biwidl203/lschlyter/anaconda3/etc/profile.d/conda.sh
conda activate seg_net

# Redirect stdout and stderr to your dynamically generated log file
exec > >(tee -i $LOG_FILE)
exec 2>&1

# Cleanup: Remove all Slurm log files in the current directory
find . -name "slurm-*.out" -type f -exec rm {} \;

# Execute the Python script with the corrected config path
exec python -u ../src/training/train.py --config_path "$CONFIG_PATH"
echo "Training script finished"
