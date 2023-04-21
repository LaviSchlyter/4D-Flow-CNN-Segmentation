#!/bin/bash

#SBATCH --output=/scratch_net/biwidl203/lschlyter/jupyter_debug_gpu/logs/TRAIN-%x.%j.out
#SBATCH --error=/scratch_net/biwidl203/lschlyter/jupyter_debug_gpu/logs/TRAIN-%x.%j.err
#SBATCH --gres=gpu:1
#SBATCH --constraint='titan_xp|geforce_rtx_2080_ti|geforce_gtx_1080_ti'
#SBATCH --mail-type=BEGIN,FAIL
#SBATCH --account=student

# Launch the vs-server
remote.sh