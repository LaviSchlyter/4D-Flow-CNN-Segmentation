#!/bin/bash

#SBATCH --output=/scratch_net/biwidl203/lschlyter/jupyter_debug_gpu/logs/TRAIN-%x.%j.out
#SBATCH --error=/scratch_net/biwidl203/lschlyter/jupyter_debug_gpu/logs/TRAIN-%x.%j.err
#SBATCH --gres=gpu:1


# Launch the vs-server
remote.sh