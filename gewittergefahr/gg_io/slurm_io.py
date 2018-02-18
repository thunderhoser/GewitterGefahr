"""Methods for writing Slurm files.

Slurm = workload-manager on Schooner, which is the University of Oklahoma
supercomputer.

A "Slurm file" specifies one job, which can be run from the terminal with the
following command:

sbatch ${slurm_file_name}
"""

import os.path
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.gg_utils import error_checking

NUM_CORES_PER_JOB_OR_ARRAY_TASK = 1
NUM_NODES_PER_JOB_OR_ARRAY_TASK = 1
NUM_MEGABYTES_PER_JOB_OR_ARRAY_TASK = 8000
TIME_LIMIT_STRING = '48:00:00'

FIRST_SPC_DATE_INPUT_ARG = 'first_spc_date_string'
LAST_SPC_DATE_INPUT_ARG = 'last_spc_date_string'
MAX_SIMULTANEOUS_TASKS_INPUT_ARG = 'max_num_simultaneous_tasks'
EMAIL_ADDRESS_INPUT_ARG = 'email_address'
PARTITION_NAME_INPUT_ARG = 'partition_name'
SLURM_FILE_INPUT_ARG = 'out_slurm_file_name'

MAX_SIMULTANEOUS_TASKS_DEFAULT = 50
DEFAULT_PARTITION_NAME = 'swat_plus'

SPC_DATE_HELP_STRING_FOR_ARRAY_JOB = (
    'SPC (Storm Prediction Center) date in format "yyyymmdd".  The job will be '
    'run independently for each date from `{0:s}`...`{1:s}`.  In other words, '
    'each date from `{0:s}`...`{1:s}` will be one task in the array.').format(
        FIRST_SPC_DATE_INPUT_ARG, LAST_SPC_DATE_INPUT_ARG)
SPC_DATE_HELP_STRING_NON_ARRAY_JOB = (
    'SPC (Storm Prediction Center) date in format "yyyymmdd".  The job will be '
    'run sequentially for all dates from `{0:s}`...`{1:s}`.').format(
        FIRST_SPC_DATE_INPUT_ARG, LAST_SPC_DATE_INPUT_ARG)

MAX_SIMULTANEOUS_TASKS_HELP_STRING = (
    'Maximum number of simultaneous tasks (in array) that can be running.')
EMAIL_ADDRESS_HELP_STRING = (
    'Your e-mail address.  Slurm notifications will be sent here.')
PARTITION_NAME_HELP_STRING = (
    'Job will be run on this partition of the supercomputer.  Examples: '
    '"swat", "swat_plus".')
SLURM_FILE_HELP_STRING = (
    'Path to output file.  We suggest (but do not enforce) the file extension '
    '".qsub".')


def add_input_arguments(argument_parser_object, use_array, use_spc_dates):
    """Adds input args for Slurm-file-writing script to ArgumentParser object.

    :param argument_parser_object: `argparse.ArgumentParser` object, which may
        or may not already contain input args.
    :param use_array: Boolean flag.  If True, will use task array, where each
        task corresponds to one SPC date.
    :param use_spc_dates: Boolean flag.  If True, will include arguments for
        first and last SPC dates.
    :return: argument_parser_object: Same as input object, but with new input
        args added.
    """

    error_checking.assert_is_boolean(use_array)
    error_checking.assert_is_boolean(use_spc_dates)
    if use_array:
        use_spc_dates = True

    argument_parser_object.add_argument(
        '--' + EMAIL_ADDRESS_INPUT_ARG, type=str, required=True,
        help=EMAIL_ADDRESS_HELP_STRING)

    argument_parser_object.add_argument(
        '--' + PARTITION_NAME_INPUT_ARG, type=str, required=False,
        help=PARTITION_NAME_HELP_STRING, default=DEFAULT_PARTITION_NAME)

    argument_parser_object.add_argument(
        '--' + SLURM_FILE_INPUT_ARG, type=str, required=True,
        help=SLURM_FILE_HELP_STRING)

    if use_array:
        argument_parser_object.add_argument(
            '--' + MAX_SIMULTANEOUS_TASKS_INPUT_ARG, type=int, required=False,
            default=MAX_SIMULTANEOUS_TASKS_DEFAULT,
            help=MAX_SIMULTANEOUS_TASKS_HELP_STRING)

    if use_spc_dates:
        if use_array:
            this_help_string = SPC_DATE_HELP_STRING_FOR_ARRAY_JOB
        else:
            this_help_string = SPC_DATE_HELP_STRING_NON_ARRAY_JOB

        argument_parser_object.add_argument(
            '--' + FIRST_SPC_DATE_INPUT_ARG, type=str, required=True,
            help=this_help_string)

        argument_parser_object.add_argument(
            '--' + LAST_SPC_DATE_INPUT_ARG, type=str, required=True,
            help=this_help_string)

    return argument_parser_object


def write_slurm_file_header(
        slurm_file_name, email_address, partition_name, use_array,
        num_array_tasks=None, max_num_simultaneous_tasks=None):
    """Writes header for Slurm file.

    :param slurm_file_name: Path to output file.
    :param email_address: Slurm notifications will be sent to this e-mail
        address.
    :param partition_name: Job will be run on this partition of the
        supercomputer.
    :param use_array: Boolean flag.  If True, will use task array.
    :param num_array_tasks: Number of tasks in array.  If use_array = False,
        leave this as None.
    :param max_num_simultaneous_tasks: Maximum number of simultaneous tasks (in
        array) that can be running.
    :return: slurm_file_handle: File handle, which can be used to continue
        writing to file.
    """

    file_system_utils.mkdir_recursive_if_necessary(file_name=slurm_file_name)
    error_checking.assert_is_string(email_address)
    error_checking.assert_is_string(partition_name)
    error_checking.assert_is_boolean(use_array)

    if use_array:
        error_checking.assert_is_integer(num_array_tasks)
        error_checking.assert_is_greater(num_array_tasks, 0)
        error_checking.assert_is_integer(max_num_simultaneous_tasks)
        error_checking.assert_is_greater(max_num_simultaneous_tasks, 0)

    # Create job name.
    _, pathless_slurm_file_name = os.path.split(slurm_file_name)
    slurm_job_name, _ = os.path.splitext(pathless_slurm_file_name)

    # Write file header.
    slurm_file_handle = open(slurm_file_name, 'w')
    slurm_file_handle.write('#!/usr/bin/bash\n\n')
    slurm_file_handle.write('#SBATCH --job-name="{0:s}"\n'.format(
        slurm_job_name))
    slurm_file_handle.write('#SBATCH --ntasks={0:d}\n'.format(
        NUM_CORES_PER_JOB_OR_ARRAY_TASK))
    slurm_file_handle.write('#SBATCH --nodes={0:d}\n'.format(
        NUM_NODES_PER_JOB_OR_ARRAY_TASK))
    slurm_file_handle.write('#SBATCH --mem={0:d}\n'.format(
        NUM_MEGABYTES_PER_JOB_OR_ARRAY_TASK))
    slurm_file_handle.write('#SBATCH --mail-user="{0:s}"\n'.format(
        email_address))
    slurm_file_handle.write('#SBATCH --mail-type=ALL\n')
    slurm_file_handle.write('#SBATCH -p "{0:s}"\n'.format(partition_name))
    slurm_file_handle.write('#SBATCH -t {0:s}\n'.format(TIME_LIMIT_STRING))

    if use_array:
        slurm_file_handle.write('#SBATCH --array=0-{0:d}%{1:d}\n\n'.format(
            num_array_tasks - 1, max_num_simultaneous_tasks))
    else:
        slurm_file_handle.write('\n')

    return slurm_file_handle


def write_spc_date_list_to_slurm_file(
        slurm_file_handle, first_spc_date_string, last_spc_date_string):
    """Writes list of SPC dates to Slurm file.

    :param slurm_file_handle: File handle.
    :param first_spc_date_string: First SPC date in range (format "yyyymmdd").
    :param last_spc_date_string: Last SPC date in range (format "yyyymmdd").
    """

    spc_date_strings = time_conversion.get_spc_dates_in_range(
        first_spc_date_string, last_spc_date_string)
    num_spc_dates = len(spc_date_strings)

    # Write list of SPC dates to file.
    slurm_file_handle.write('SPC_DATE_STRINGS=(')
    for i in range(num_spc_dates):
        if i == 0:
            slurm_file_handle.write('"{0:s}"'.format(spc_date_strings[i]))
        else:
            slurm_file_handle.write(' "{0:s}"'.format(spc_date_strings[i]))
    slurm_file_handle.write(')\n\n')

    # When each task is run, the following statement will write the task ID and
    # corresponding SPC date to the Slurm log file.
    slurm_file_handle.write(
        'this_spc_date_string=${SPC_DATE_STRINGS[$SLURM_ARRAY_TASK_ID]}\n')
    slurm_file_handle.write(
        'echo "Array-task ID = ${SLURM_ARRAY_TASK_ID} ... '
        'SPC date = ${this_spc_date_string}"\n\n')
