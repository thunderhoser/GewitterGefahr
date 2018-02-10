"""Writes Slurm file to run best-track algorithm.

--- DEFINITIONS ---

Slurm = workload-manager on supercomputer
"""

import os.path
import argparse
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.scripts import run_best_track

# TODO(thunderhoser): Need common module to deal with "email_address",
# "partition_name", "out_slurm_file_name", and other input args for Slurm files.

PYTHON_EXE_NAME = '/home/ralager/anaconda2/bin/python2.7'
PYTHON_SCRIPT_NAME = (
    '/condo/swatwork/ralager/gewittergefahr_master/gewittergefahr/scripts/'
    'run_best_track.py')

NUM_CORES = 1
NUM_NODES = 1
NUM_MEGABYTES = 8000
TIME_LIMIT_STRING = '48:00:00'

EMAIL_ADDRESS_INPUT_ARG = 'email_address'
PARTITION_NAME_INPUT_ARG = 'partition_name'
SLURM_FILE_INPUT_ARG = 'out_slurm_file_name'

EMAIL_ADDRESS_HELP_STRING = 'Slurm notifications will be sent here.'
PARTITION_NAME_HELP_STRING = (
    'Jobs will be run on this partition of the supercomputer.')
SLURM_FILE_HELP_STRING = (
    '[output] Path to Slurm file.  We suggest the extension ".qsub".')

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + EMAIL_ADDRESS_INPUT_ARG, type=str, required=True,
    help=EMAIL_ADDRESS_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + PARTITION_NAME_INPUT_ARG, type=str, required=False,
    help=PARTITION_NAME_HELP_STRING, default='swat_plus')

INPUT_ARG_PARSER.add_argument(
    '--' + SLURM_FILE_INPUT_ARG, type=str, required=True,
    help=SLURM_FILE_HELP_STRING)

INPUT_ARG_PARSER = run_best_track.add_input_arguments(INPUT_ARG_PARSER)


def _write_slurm_file(
        first_spc_date_string, last_spc_date_string, top_segmotion_dir_name,
        top_best_track_dir_name, tracking_scale_metres2, email_address,
        partition_name, slurm_file_name):
    """Writes Slurm file to run best-track.

    :param first_spc_date_string: See documentation for `run_best_track.py`.
    :param last_spc_date_string: See doc for `run_best_track.py`.
    :param top_segmotion_dir_name: See doc for `run_best_track.py`.
    :param top_best_track_dir_name: See doc for `run_best_track.py`.
    :param tracking_scale_metres2: See doc for `run_best_track.py`.
    :param email_address: Slurm notifications will be sent here.
    :param partition_name: Jobs will be run on this partition of the
        supercomputer.
    :param slurm_file_name: [output] Path to Slurm file.
    """

    file_system_utils.mkdir_recursive_if_necessary(file_name=slurm_file_name)

    # Create job name.
    _, pathless_slurm_file_name = os.path.split(slurm_file_name)
    slurm_job_name, _ = os.path.splitext(pathless_slurm_file_name)

    # Write file header.
    slurm_file_handle = open(slurm_file_name, 'w')
    slurm_file_handle.write('#!/usr/bin/bash\n\n')
    slurm_file_handle.write('#SBATCH --job-name="{0:s}"\n'.format(
        slurm_job_name))
    slurm_file_handle.write('#SBATCH --ntasks={0:d}\n'.format(NUM_CORES))
    slurm_file_handle.write('#SBATCH --nodes={0:d}\n'.format(NUM_NODES))
    slurm_file_handle.write('#SBATCH --mem={0:d}\n'.format(NUM_MEGABYTES))
    slurm_file_handle.write('#SBATCH --mail-user="{0:s}"\n'.format(
        email_address))
    slurm_file_handle.write('#SBATCH --mail-type=ALL\n')
    slurm_file_handle.write('#SBATCH -p "{0:s}"\n'.format(partition_name))
    slurm_file_handle.write('#SBATCH -t {0:s}\n\n'.format(TIME_LIMIT_STRING))

    slurm_file_handle.write(
        ('"{0:s}" -u "{1:s}" --{2:s}="{3:s}" --{4:s}="{5:s}" --{6:s}="{7:s}" '
         '--{8:s}="{9:s}" --{10:s}={11:d}').format(
             PYTHON_EXE_NAME, PYTHON_SCRIPT_NAME,
             run_best_track.FIRST_SPC_DATE_INPUT_ARG, first_spc_date_string,
             run_best_track.LAST_SPC_DATE_INPUT_ARG, last_spc_date_string,
             run_best_track.SEGMOTION_DIR_INPUT_ARG, top_segmotion_dir_name,
             run_best_track.BEST_TRACK_DIR_INPUT_ARG, top_best_track_dir_name,
             run_best_track.TRACKING_SCALE_INPUT_ARG, tracking_scale_metres2))
    slurm_file_handle.close()


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    FIRST_SPC_DATE_STRING = getattr(
        INPUT_ARG_OBJECT, run_best_track.FIRST_SPC_DATE_INPUT_ARG)
    LAST_SPC_DATE_STRING = getattr(
        INPUT_ARG_OBJECT, run_best_track.LAST_SPC_DATE_INPUT_ARG)
    TOP_SEGMOTION_DIR_NAME = getattr(
        INPUT_ARG_OBJECT, run_best_track.SEGMOTION_DIR_INPUT_ARG)
    TOP_BEST_TRACK_DIR_NAME = getattr(
        INPUT_ARG_OBJECT, run_best_track.BEST_TRACK_DIR_INPUT_ARG)
    TRACKING_SCALE_METRES2 = getattr(
        INPUT_ARG_OBJECT, run_best_track.TRACKING_SCALE_INPUT_ARG)

    EMAIL_ADDRESS = getattr(INPUT_ARG_OBJECT, EMAIL_ADDRESS_INPUT_ARG)
    PARTITION_NAME = getattr(INPUT_ARG_OBJECT, PARTITION_NAME_INPUT_ARG)
    SLURM_FILE_NAME = getattr(INPUT_ARG_OBJECT, SLURM_FILE_INPUT_ARG)

    _write_slurm_file(
        first_spc_date_string=FIRST_SPC_DATE_STRING,
        last_spc_date_string=LAST_SPC_DATE_STRING,
        top_segmotion_dir_name=TOP_SEGMOTION_DIR_NAME,
        top_best_track_dir_name=TOP_BEST_TRACK_DIR_NAME,
        tracking_scale_metres2=TRACKING_SCALE_METRES2,
        email_address=EMAIL_ADDRESS, partition_name=PARTITION_NAME,
        slurm_file_name=SLURM_FILE_NAME)
