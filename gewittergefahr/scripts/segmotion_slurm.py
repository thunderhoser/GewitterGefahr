"""Writes Slurm file to run segmotion on radar data.

Radar data must be in MYRORSS format.  To convert GridRad data to MYRORSS
format, see `gridrad_to_myrorss_format.py`.

The resulting Slurm file will create one job for each SPC (Storm Prediction
Center) date.  SPC dates run from 1200-1200 UTC, not 0000-0000 UTC.  For
example, the SPC date "20180123" runs from 1200 UTC 23 Jan - 1200 UTC 24 Jan
2018.

--- DEFINITIONS ---

Slurm = workload-manager on supercomputer

segmotion = a storm-tracking algorithm in WDSS-II (Lakshmanan and Smith 2010)

WDSS-II = the Warning Decision Support System with Integrated Information
(Lakshmanan et. al 2007)

MYRORSS = Multi-year Reanalysis of Remotely Sensed Storms = archive of
composited radar data (Ortega et al. 2012)

--- REFERENCES ---

Lakshmanan, V., T. Smith, G. Stumpf, and K. Hondl, 2007: "The Warning Decision
    Support System -- Integrated Information". Weather and Forecasting, 22 (3),
    596-612.

Lakshmanan, V., and T. Smith, 2010: "Evaluating a storm tracking algorithm".
    26th Conference on Interactive Information Processing Systems, Atlanta, GA,
    American Meteorological Society.

Ortega, K., T. Smith, J. Zhang, C. Langston, Y. Qi, S. Stevens, and J. Tate,
    2012: "The Multi-year Reanalysis of Remotely Sensed Storms (MYRORSS)
    project". 26th Conference on Severe Local Storms, Nashville, TN, American
    Meteorological Society.

--- EXAMPLE USAGE ---

To run segmotion on GridRad data for 2011:

segmotion_for_gridrad_slurm.py --radar_data_source="gridrad"
--top_radar_dir_name="/condo/swatcommon/common/gridrad/myrorss_format"
--first_spc_date_string="20110101" --last_spc_date_string="20111231"
--email_address="ryan.lagerquist@ou.edu"
--out_slurm_file_name="segmotion_gridrad.qsub"

To run segmotion on MYRORSS data for 2011:

segmotion_for_gridrad_slurm.py --radar_data_source="myrorss"
--top_radar_dir_name="/condo/swatcommon/common/myrorss"
--first_spc_date_string="20110101" --last_spc_date_string="20111231"
--email_address="ryan.lagerquist@ou.edu"
--out_slurm_file_name="segmotion_myrorss.qsub"
"""

import os.path
import argparse
from gewittergefahr.gg_utils import radar_utils
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import time_periods
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.gg_utils import error_checking

SPC_DATE_FORMAT = '%Y%m%d'
DAYS_TO_SECONDS = 86400
VALID_RADAR_DATA_SOURCES = [
    radar_utils.MYRORSS_SOURCE_ID, radar_utils.GRIDRAD_SOURCE_ID]

NUM_TASKS_PER_SPC_DATE = 1
NUM_NODES_PER_SPC_DATE = 1
MEGABYTES_PER_SPC_DATE = 8000
TIME_LIMIT_STRING = '48:00:00'
TRACKING_FIELD_NAME = radar_utils.ECHO_TOP_40DBZ_NAME
TRACKING_FIELD_NAME_MYRORSS = radar_utils.ECHO_TOP_40DBZ_NAME_MYRORSS

SEGMOTION_ARG_STRING_FOR_GRIDRAD = (
    '-f "EchoTop_40" -d "4 20 1 -1" -t 0 -p "10,20,40:0:0,0,0" '
    '-k "percent:75:2:0:2" -A NoSuchProduct')
SEGMOTION_ARG_STRING_FOR_MYRORSS = (
    '-f "EchoTop_40" -d "4 20 1 -1" -t 0 -p "25,50,100:0:0,0,0" '
    '-k "percent:75:2:0:2" -A NoSuchProduct -X')
K_MEANS_FILE_NAME_ON_SCHOONER = (
    '/condo/swatwork/ralager/slurm/classification_k_means.xml')

RADAR_SOURCE_INPUT_ARG = 'radar_data_source'
RADAR_DIRECTORY_INPUT_ARG = 'top_radar_dir_name'
FIRST_SPC_DATE_INPUT_ARG = 'first_spc_date_string'
LAST_SPC_DATE_INPUT_ARG = 'last_spc_date_string'
EMAIL_ADDRESS_INPUT_ARG = 'email_address'
PARTITION_NAME_INPUT_ARG = 'partition_name'
SLURM_FILE_INPUT_ARG = 'out_slurm_file_name'

RADAR_SOURCE_HELP_STRING = (
    'Source of radar data.  Regardless of source, must be formatted like '
    'MYRORSS data.  Valid options are listed below:\n{0:s}').format(
        VALID_RADAR_DATA_SOURCES)
RADAR_DIRECTORY_HELP_STRING = (
    'Name of top-level directory with radar data (must include "{0:s}" in '
    'MYRORSS format).').format(TRACKING_FIELD_NAME_MYRORSS)
SPC_DATE_HELP_STRING = (
    'SPC (Storm Prediction Center) date in format "yyyymmdd".  segmotion will '
    'be run for all dates from `{0:s}`...`{1:s}`.').format(
        FIRST_SPC_DATE_INPUT_ARG, LAST_SPC_DATE_INPUT_ARG)
EMAIL_ADDRESS_HELP_STRING = 'Slurm notifications will be sent here.'
PARTITION_NAME_HELP_STRING = (
    'Jobs will be run on this partition of the supercomputer.')
SLURM_FILE_HELP_STRING = (
    '[output] Path to Slurm file.  We suggest the extension ".qsub".')

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + RADAR_SOURCE_INPUT_ARG, type=str, required=True,
    help=RADAR_SOURCE_HELP_STRING)
INPUT_ARG_PARSER.add_argument(
    '--' + RADAR_DIRECTORY_INPUT_ARG, type=str, required=True,
    help=RADAR_DIRECTORY_HELP_STRING)
INPUT_ARG_PARSER.add_argument(
    '--' + FIRST_SPC_DATE_INPUT_ARG, type=str, required=True,
    help=SPC_DATE_HELP_STRING)
INPUT_ARG_PARSER.add_argument(
    '--' + LAST_SPC_DATE_INPUT_ARG, type=str, required=True,
    help=SPC_DATE_HELP_STRING)
INPUT_ARG_PARSER.add_argument(
    '--' + EMAIL_ADDRESS_INPUT_ARG, type=str, required=True,
    help=EMAIL_ADDRESS_HELP_STRING)
INPUT_ARG_PARSER.add_argument(
    '--' + PARTITION_NAME_INPUT_ARG, type=str, required=False,
    help=PARTITION_NAME_HELP_STRING, default='swat_plus')
INPUT_ARG_PARSER.add_argument(
    '--' + SLURM_FILE_INPUT_ARG, type=str, required=True,
    help=SLURM_FILE_HELP_STRING)


def _check_radar_data_source(radar_data_source):
    """Ensures that data source is valid for segmotion.

    :param radar_data_source: Data source (string).
    :raises: ValueError: if `radar_data_source not in VALID_RADAR_DATA_SOURCES`.
    """

    if radar_data_source not in VALID_RADAR_DATA_SOURCES:
        error_string = (
            '\n\n{0:s}\n\nValid data sources (listed above) do not include '
            '"{1:s}".').format(VALID_RADAR_DATA_SOURCES, radar_data_source)
        raise ValueError(error_string)


def _write_slurm_file(
        radar_data_source, top_radar_dir_name, first_spc_date_string,
        last_spc_date_string, email_address, partition_name, slurm_file_name):
    """Writes Slurm file to run segmotion.

    :param radar_data_source: Data source (string).
    :param top_radar_dir_name: Name of top-level directory with radar data (must
        include "EchoTop_40" in MYRORSS format).
    :param first_spc_date_string: SPC (Storm Prediction Center) date in format
        "yyyymmdd".  segmotion will be run for all dates from
        `first_spc_date_string`...`last_spc_date_string`.
    :param last_spc_date_string: See above.
    :param email_address: Slurm notifications will be sent here.
    :param partition_name: Jobs will be run on this partition of the
        supercomputer.
    :param slurm_file_name: [output] Path to Slurm file.
    """

    _check_radar_data_source(radar_data_source)
    if radar_data_source == radar_utils.GRIDRAD_SOURCE_ID:
        segmotion_arg_string = SEGMOTION_ARG_STRING_FOR_GRIDRAD
    else:
        segmotion_arg_string = SEGMOTION_ARG_STRING_FOR_MYRORSS

    error_checking.assert_is_string(top_radar_dir_name)
    error_checking.assert_is_string(email_address)
    error_checking.assert_is_string(partition_name)
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
        NUM_TASKS_PER_SPC_DATE))
    slurm_file_handle.write('#SBATCH --nodes={0:d}\n'.format(
        NUM_NODES_PER_SPC_DATE))
    slurm_file_handle.write('#SBATCH --mem={0:d}\n'.format(
        MEGABYTES_PER_SPC_DATE))
    slurm_file_handle.write('#SBATCH --mail-user="{0:s}"\n'.format(
        email_address))
    slurm_file_handle.write('#SBATCH --mail-type=ALL\n')
    slurm_file_handle.write('#SBATCH -p "{0:s}"\n'.format(partition_name))
    slurm_file_handle.write('#SBATCH -t {0:s}\n'.format(TIME_LIMIT_STRING))
    slurm_file_handle.write('#SBATCH --array=0-{0:d}%50\n\n'.format(
        num_spc_dates - 1))

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

    # Find input files/directories for the given SPC date.
    slurm_file_handle.write(
        'this_spc_date_string=${SPC_DATE_STRINGS[$SLURM_ARRAY_TASK_ID]}\n')
    slurm_file_handle.write(
        'echo "Slurm array task ID = ${SLURM_ARRAY_TASK_ID}; '
        'SPC date = ${this_spc_date_string}"\n\n')

    slurm_file_handle.write('this_1day_radar_dir_name="{0:s}'.format(
        top_radar_dir_name))
    slurm_file_handle.write('/${this_spc_date_string}"\n')

    slurm_file_handle.write(
        'this_code_index_file_name="${this_1day_radar_dir_name}/'
        'code_index.xml"\n')
    slurm_file_handle.write(
        'this_1day_segmotion_dir_name="${this_1day_radar_dir_name}/'
        'segmotion"\n')
    slurm_file_handle.write('mkdir -p "${this_1day_segmotion_dir_name}"\n\n')

    # Run segmotion for the given SPC date.
    slurm_file_handle.write('cd "${this_1day_radar_dir_name}"\n')
    slurm_file_handle.write('makeIndex.pl $PWD code_index.xml\n\n')

    segmotion_command_string = (
        'w2segmotionll -i "${this_code_index_file_name}" -o ' +
        '"${this_1day_segmotion_dir_name}"')
    segmotion_command_string += ' -T "{0:s}" {1:s} -X "{2:s}"'.format(
        TRACKING_FIELD_NAME_MYRORSS, segmotion_arg_string,
        K_MEANS_FILE_NAME_ON_SCHOONER)
    slurm_file_handle.write('{0:s}\n\n'.format(segmotion_command_string))

    # Delete segmotion poop.
    slurm_file_handle.write(
        'rm -rf "${this_1day_segmotion_dir_name}/KMeans"\n')
    slurm_file_handle.write(
        'rm -rf "${this_1day_segmotion_dir_name}/WindField"\n')
    slurm_file_handle.write(
        'rm -rf "${this_1day_segmotion_dir_name}/GrowthRate"\n')
    slurm_file_handle.write(
        'rm -rf "${this_1day_segmotion_dir_name}/Motion_East"\n')
    slurm_file_handle.write(
        'rm -rf "${this_1day_segmotion_dir_name}/Motion_South"\n')
    slurm_file_handle.write(
        'rm -rf "${this_1day_segmotion_dir_name}/code_index.fam"')
    slurm_file_handle.close()


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    RADAR_DATA_SOURCE = getattr(INPUT_ARG_OBJECT, RADAR_SOURCE_INPUT_ARG)
    TOP_RADAR_DIR_NAME = getattr(INPUT_ARG_OBJECT, RADAR_DIRECTORY_INPUT_ARG)
    FIRST_SPC_DATE_STRING = getattr(
        INPUT_ARG_OBJECT, FIRST_SPC_DATE_INPUT_ARG)
    LAST_SPC_DATE_STRING = getattr(INPUT_ARG_OBJECT, LAST_SPC_DATE_INPUT_ARG)
    EMAIL_ADDRESS = getattr(INPUT_ARG_OBJECT, EMAIL_ADDRESS_INPUT_ARG)
    PARTITION_NAME = getattr(INPUT_ARG_OBJECT, PARTITION_NAME_INPUT_ARG)
    SLURM_FILE_NAME = getattr(INPUT_ARG_OBJECT, SLURM_FILE_INPUT_ARG)

    _write_slurm_file(
        radar_data_source=RADAR_DATA_SOURCE,
        top_radar_dir_name=TOP_RADAR_DIR_NAME,
        first_spc_date_string=FIRST_SPC_DATE_STRING,
        last_spc_date_string=LAST_SPC_DATE_STRING, email_address=EMAIL_ADDRESS,
        partition_name=PARTITION_NAME, slurm_file_name=SLURM_FILE_NAME)
