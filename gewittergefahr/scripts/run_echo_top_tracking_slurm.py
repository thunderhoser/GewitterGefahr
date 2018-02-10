"""Writes Slurm file to execute run_echo_top_tracking.py on supercomputer."""

import os.path
import argparse
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import time_periods
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.scripts import run_echo_top_tracking

SPC_DATE_FORMAT = '%Y%m%d'
DAYS_TO_SECONDS = 86400

PYTHON_EXE_NAME = '/home/ralager/anaconda2/bin/python2.7'
PYTHON_SCRIPT_NAME = (
    '/condo/swatwork/ralager/gewittergefahr_master/gewittergefahr/scripts/'
    'run_echo_top_tracking.py')

NUM_CORES_PER_SPC_DATE = 1
NUM_NODES_PER_SPC_DATE = 1
NUM_MEGABYTES_PER_SPC_DATE = 8000
TIME_LIMIT_STRING = '48:00:00'

FIRST_SPC_DATE_INPUT_ARG = 'first_spc_date_string'
LAST_SPC_DATE_INPUT_ARG = 'last_spc_date_string'
MAX_NUM_SIMULTANEOUS_JOBS_INPUT_ARG = 'max_num_simultaneous_jobs'
EMAIL_ADDRESS_INPUT_ARG = 'email_address'
PARTITION_NAME_INPUT_ARG = 'partition_name'
SLURM_FILE_INPUT_ARG = 'out_slurm_file_name'

SPC_DATE_HELP_STRING = (
    'SPC (Storm Prediction Center) date in format "yyyymmdd".  Tracking will be'
    ' run independently for each date from `{0:s}`...`{1:s}`.  In other words, '
    'each date will be one job.').format(
        FIRST_SPC_DATE_INPUT_ARG, LAST_SPC_DATE_INPUT_ARG)
MAX_NUM_SIMULTANEOUS_JOBS_HELP_STRING = (
    'Maximum number of jobs (SPC dates) running at once.')
EMAIL_ADDRESS_HELP_STRING = 'Slurm notifications will be sent here.'
PARTITION_NAME_HELP_STRING = (
    'Jobs will be run on this partition of the supercomputer.')
SLURM_FILE_HELP_STRING = (
    '[output] Path to Slurm file.  We suggest the extension ".qsub".')

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + FIRST_SPC_DATE_INPUT_ARG, type=str, required=True,
    help=SPC_DATE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + LAST_SPC_DATE_INPUT_ARG, type=str, required=True,
    help=SPC_DATE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + MAX_NUM_SIMULTANEOUS_JOBS_INPUT_ARG, type=int, required=False,
    default=50, help=MAX_NUM_SIMULTANEOUS_JOBS_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + EMAIL_ADDRESS_INPUT_ARG, type=str, required=True,
    help=EMAIL_ADDRESS_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + PARTITION_NAME_INPUT_ARG, type=str, required=False,
    help=PARTITION_NAME_HELP_STRING, default='swat_plus')

INPUT_ARG_PARSER.add_argument(
    '--' + SLURM_FILE_INPUT_ARG, type=str, required=True,
    help=SLURM_FILE_HELP_STRING)

INPUT_ARG_PARSER = run_echo_top_tracking.add_input_arguments(INPUT_ARG_PARSER)


def _write_slurm_file(
        first_spc_date_string, last_spc_date_string, top_radar_dir_name,
        top_tracking_dir_name, max_num_simultaneous_jobs, email_address,
        partition_name, slurm_file_name):
    """Writes Slurm file to execute run_echo_top_tracking.py on supercomputer.

    :param first_spc_date_string: SPC (Storm Prediction Center) date in format
        "yyyymmdd".  Tracking will be run independently for each date from
        `first_spc_date_string`...`last_spc_date_string`.  In other words, each
        date will be one job.
    :param last_spc_date_string: See above.
    :param top_radar_dir_name: [input] Name of top-level directory with radar
        data.
    :param top_tracking_dir_name: [output] Name of top-level directory for
        storm-tracking data.
    :param max_num_simultaneous_jobs: Maximum number of jobs (SPC dates) running
        at once.
    :param email_address: Slurm notifications will be sent here.
    :param partition_name: Jobs will be run on this partition of the
        supercomputer.
    :param slurm_file_name: [output] Path to Slurm file.  We suggest the
        extension ".qsub".
    """

    file_system_utils.mkdir_recursive_if_necessary(file_name=slurm_file_name)

    # Create list of SPC dates.
    first_spc_date_unix_sec = time_conversion.string_to_unix_sec(
        first_spc_date_string, SPC_DATE_FORMAT)
    last_spc_date_unix_sec = time_conversion.string_to_unix_sec(
        last_spc_date_string, SPC_DATE_FORMAT)

    spc_dates_unix_sec = time_periods.range_and_interval_to_list(
        start_time_unix_sec=first_spc_date_unix_sec,
        end_time_unix_sec=last_spc_date_unix_sec,
        time_interval_sec=DAYS_TO_SECONDS, include_endpoint=True)
    num_spc_dates = len(spc_dates_unix_sec)

    # Create job name.
    _, pathless_slurm_file_name = os.path.split(slurm_file_name)
    slurm_job_name, _ = os.path.splitext(pathless_slurm_file_name)

    # Write file header.
    slurm_file_handle = open(slurm_file_name, 'w')
    slurm_file_handle.write('#!/usr/bin/bash\n\n')
    slurm_file_handle.write('#SBATCH --job-name="{0:s}"\n'.format(
        slurm_job_name))
    slurm_file_handle.write('#SBATCH --ntasks={0:d}\n'.format(
        NUM_CORES_PER_SPC_DATE))
    slurm_file_handle.write('#SBATCH --nodes={0:d}\n'.format(
        NUM_NODES_PER_SPC_DATE))
    slurm_file_handle.write('#SBATCH --mem={0:d}\n'.format(
        NUM_MEGABYTES_PER_SPC_DATE))
    slurm_file_handle.write('#SBATCH --mail-user="{0:s}"\n'.format(
        email_address))
    slurm_file_handle.write('#SBATCH --mail-type=ALL\n')
    slurm_file_handle.write('#SBATCH -p "{0:s}"\n'.format(partition_name))
    slurm_file_handle.write('#SBATCH -t {0:s}\n'.format(TIME_LIMIT_STRING))
    slurm_file_handle.write('#SBATCH --array=0-{0:d}%{1:d}\n\n'.format(
        num_spc_dates - 1, max_num_simultaneous_jobs))

    # Write array of SPC dates.
    spc_date_strings = [''] * num_spc_dates
    for i in range(num_spc_dates):
        spc_date_strings[i] = time_conversion.unix_sec_to_string(
            spc_dates_unix_sec[i], SPC_DATE_FORMAT)

    slurm_file_handle.write('SPC_DATE_STRINGS=(')
    for i in range(num_spc_dates):
        if i == 0:
            slurm_file_handle.write('"{0:s}"'.format(spc_date_strings[i]))
        else:
            slurm_file_handle.write(' "{0:s}"'.format(spc_date_strings[i]))
    slurm_file_handle.write(')\n\n')

    # Print given SPC date to log file.
    slurm_file_handle.write(
        'this_spc_date_string=${SPC_DATE_STRINGS[$SLURM_ARRAY_TASK_ID]}\n')
    slurm_file_handle.write(
        'echo "Slurm array task ID = ${SLURM_ARRAY_TASK_ID}; '
        'SPC date = ${this_spc_date_string}"\n\n')

    # Run storm-tracking for the given SPC date.
    slurm_file_handle.write(
        '"{0:s}" -u "{1:s}" --{2:s}='.format(
            PYTHON_EXE_NAME, PYTHON_SCRIPT_NAME,
            run_echo_top_tracking.FIRST_SPC_DATE_INPUT_ARG))
    slurm_file_handle.write('"${this_spc_date_string}"')

    slurm_file_handle.write(' --{0:s}='.format(
        run_echo_top_tracking.LAST_SPC_DATE_INPUT_ARG))
    slurm_file_handle.write('"${this_spc_date_string}"')

    slurm_file_handle.write(' --{0:s}="{1:s}" --{2:s}="{3:s}"'.format(
        run_echo_top_tracking.RADAR_DIR_INPUT_ARG, top_radar_dir_name,
        run_echo_top_tracking.TRACKING_DIR_INPUT_ARG, top_tracking_dir_name))
    slurm_file_handle.close()


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    FIRST_SPC_DATE_STRING = getattr(INPUT_ARG_OBJECT, FIRST_SPC_DATE_INPUT_ARG)
    LAST_SPC_DATE_STRING = getattr(INPUT_ARG_OBJECT, LAST_SPC_DATE_INPUT_ARG)
    TOP_RADAR_DIR_NAME = getattr(
        INPUT_ARG_OBJECT, run_echo_top_tracking.RADAR_DIR_INPUT_ARG)
    TOP_TRACKING_DIR_NAME = getattr(
        INPUT_ARG_OBJECT, run_echo_top_tracking.TRACKING_DIR_INPUT_ARG)

    MAX_NUM_SIMULTANEOUS_JOBS = getattr(
        INPUT_ARG_OBJECT, MAX_NUM_SIMULTANEOUS_JOBS_INPUT_ARG)
    EMAIL_ADDRESS = getattr(INPUT_ARG_OBJECT, EMAIL_ADDRESS_INPUT_ARG)
    PARTITION_NAME = getattr(INPUT_ARG_OBJECT, PARTITION_NAME_INPUT_ARG)
    SLURM_FILE_NAME = getattr(INPUT_ARG_OBJECT, SLURM_FILE_INPUT_ARG)

    _write_slurm_file(
        first_spc_date_string=FIRST_SPC_DATE_STRING,
        last_spc_date_string=LAST_SPC_DATE_STRING,
        top_radar_dir_name=TOP_RADAR_DIR_NAME,
        top_tracking_dir_name=TOP_TRACKING_DIR_NAME,
        max_num_simultaneous_jobs=MAX_NUM_SIMULTANEOUS_JOBS,
        email_address=EMAIL_ADDRESS, partition_name=PARTITION_NAME,
        slurm_file_name=SLURM_FILE_NAME)
