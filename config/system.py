import os
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

# ==================================================================
# SET THESE PATHS MANUALLY
# ==================================================================

# ==================================================================
# name of the host - used to check if running on cluster or not
# ==================================================================
# Updated with my machine: 13.04
local_hostnames = ['biwidl203']

# ==================================================================
# project dirs
# ==================================================================
project_code_root = '/usr/bmicnas02/data-biwi-01/jeremy_students/lschlyter/CNN-segmentation'
project_data_freiburg_root = '/usr/bmicnas02/data-biwi-01/jeremy_students/previous_work/nicolas/data/freiburg'
project_data_freiburg_trial_root = '/usr/bmicnas02/data-biwi-01/jeremy_students/previous_work/nicolas/data/freiburg_trial'
project_data_bern_root = "/usr/bmicnas02/data-biwi-01/jeremy_students/lschlyter/CNN-segmentation/data"
# Note that this is the base direectory where the freiburg images have been saved a numpy arrays.
# The original dicom files are not here and they are not required for any further processing.
# Just for completeness, the bast path for the original dicom files of the freiburg dataset are here:
orig_data_root = '/usr/bmicnas02/data-biwi-01/nkarani_data/hpc_predict/data/freiburg/126_subjects'
project_data_root = '/usr/bmicnas02/data-biwi-01/jeremy_students/data/inselspital/kady'

# ==================================================================
# log root
# ==================================================================
log_root = os.path.join(project_code_root, 'logs/')
log_experiments_root = os.path.join(project_code_root, 'Saved_models/')