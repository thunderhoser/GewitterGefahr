"""Writes Slurm file to run segmotion storm-tracking algorithm on supercomputer.

Input radar data must be in MYRORSS format.  To convert GridRad data to MYRORSS
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

import argparse
from gewittergefahr.gg_io import slurm_io
from gewittergefahr.gg_utils import radar_utils
from gewittergefahr.gg_utils import time_conversion

TRACKING_FIELD_NAME = radar_utils.ECHO_TOP_40DBZ_NAME
TRACKING_FIELD_NAME_MYRORSS = radar_utils.ECHO_TOP_40DBZ_NAME_MYRORSS
VALID_RADAR_DATA_SOURCES = [
    radar_utils.MYRORSS_SOURCE_ID, radar_utils.GRIDRAD_SOURCE_ID]

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

RADAR_SOURCE_HELP_STRING = (
    'Source of radar data.  Regardless of source, must be formatted like '
    'MYRORSS data.  Valid options are listed below:\n{0:s}').format(
        VALID_RADAR_DATA_SOURCES)
RADAR_DIRECTORY_HELP_STRING = (
    'Name of top-level directory with radar data (must include "{0:s}" in '
    'MYRORSS format).').format(TRACKING_FIELD_NAME_MYRORSS)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER = slurm_io.add_input_arguments(
    argument_parser_object=INPUT_ARG_PARSER, use_array=True, use_spc_dates=True)

INPUT_ARG_PARSER.add_argument(
    '--' + RADAR_SOURCE_INPUT_ARG, type=str, required=True,
    help=RADAR_SOURCE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + RADAR_DIRECTORY_INPUT_ARG, type=str, required=True,
    help=RADAR_DIRECTORY_HELP_STRING)


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
        last_spc_date_string, max_num_simultaneous_tasks, email_address,
        partition_name, slurm_file_name):
    """Writes Slurm file to run segmotion storm-tracking algorithm on sprcmptr.

    :param radar_data_source: Data source (string).
    :param top_radar_dir_name: Name of top-level directory with radar data (must
        include "EchoTop_40" in MYRORSS format).
    :param first_spc_date_string: SPC (Storm Prediction Center) date in format
        "yyyymmdd".  Tracking will be run independently for each date from
        `first_spc_date_string`...`last_spc_date_string`.  In other words, each
        date will be one task.
    :param last_spc_date_string: See above.
    :param max_num_simultaneous_tasks: Max number of tasks (SPC dates) running
        at once.
    :param email_address: Slurm notifications will be sent to this e-mail
        address.
    :param partition_name: Job will be run on this partition of the
        supercomputer.
    :param slurm_file_name: Path to output file.
    """

    _check_radar_data_source(radar_data_source)
    if radar_data_source == radar_utils.GRIDRAD_SOURCE_ID:
        segmotion_arg_string = SEGMOTION_ARG_STRING_FOR_GRIDRAD
    else:
        segmotion_arg_string = SEGMOTION_ARG_STRING_FOR_MYRORSS

    spc_date_strings = time_conversion.get_spc_dates_in_range(
        first_spc_date_string, last_spc_date_string)
    num_spc_dates = len(spc_date_strings)

    slurm_file_handle = slurm_io.write_slurm_file_header(
        slurm_file_name=slurm_file_name, email_address=email_address,
        partition_name=partition_name, use_array=True,
        num_array_tasks=num_spc_dates,
        max_num_simultaneous_tasks=max_num_simultaneous_tasks)

    slurm_io.write_spc_date_list_to_slurm_file(
        slurm_file_handle=slurm_file_handle,
        first_spc_date_string=first_spc_date_string,
        last_spc_date_string=last_spc_date_string)

    # Write list of SPC years to log file.
    slurm_file_handle.write('SPC_YEAR_STRINGS=(')
    for i in range(num_spc_dates):
        if i == 0:
            slurm_file_handle.write('"{0:s}"'.format(spc_date_strings[i][:4]))
        else:
            slurm_file_handle.write(' "{0:s}"'.format(spc_date_strings[i][:4]))
    slurm_file_handle.write(')\n\n')

    # The following statement finds input/output directories for the given task
    # (SPC date).
    slurm_file_handle.write(
        'this_spc_year_string=${SPC_YEAR_STRINGS[$SLURM_ARRAY_TASK_ID]}\n')

    slurm_file_handle.write('this_1day_radar_dir_name="{0:s}'.format(
        top_radar_dir_name))
    slurm_file_handle.write(
        '/${this_spc_year_string}/${this_spc_date_string}"\n')

    slurm_file_handle.write(
        'this_code_index_file_name="${this_1day_radar_dir_name}/'
        'code_index.xml"\n')
    slurm_file_handle.write(
        'this_1day_segmotion_dir_name="${this_1day_radar_dir_name}/'
        'segmotion"\n')
    slurm_file_handle.write('mkdir -p "${this_1day_segmotion_dir_name}"\n\n')

    slurm_file_handle.write('cd "${this_1day_radar_dir_name}"\n')
    slurm_file_handle.write('makeIndex.pl $PWD code_index.xml\n\n')

    # The following statement runs segmotion for the given task (SPC date).
    segmotion_command_string = (
        'w2segmotionll -i "${this_code_index_file_name}" -o ' +
        '"${this_1day_segmotion_dir_name}"')
    segmotion_command_string += ' -T "{0:s}" {1:s} -X "{2:s}"'.format(
        TRACKING_FIELD_NAME_MYRORSS, segmotion_arg_string,
        K_MEANS_FILE_NAME_ON_SCHOONER)
    slurm_file_handle.write('{0:s}\n\n'.format(segmotion_command_string))

    # The following statement deletes unnecessary output from segmotion (the
    # vast majority of it).
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

    _write_slurm_file(
        radar_data_source=getattr(INPUT_ARG_OBJECT, RADAR_SOURCE_INPUT_ARG),
        top_radar_dir_name=getattr(INPUT_ARG_OBJECT, RADAR_DIRECTORY_INPUT_ARG),
        first_spc_date_string=getattr(
            INPUT_ARG_OBJECT, slurm_io.FIRST_SPC_DATE_INPUT_ARG),
        last_spc_date_string=getattr(
            INPUT_ARG_OBJECT, slurm_io.LAST_SPC_DATE_INPUT_ARG),
        max_num_simultaneous_tasks=getattr(
            INPUT_ARG_OBJECT, slurm_io.MAX_SIMULTANEOUS_TASKS_INPUT_ARG),
        email_address=getattr(
            INPUT_ARG_OBJECT, slurm_io.EMAIL_ADDRESS_INPUT_ARG),
        partition_name=getattr(
            INPUT_ARG_OBJECT, slurm_io.PARTITION_NAME_INPUT_ARG),
        slurm_file_name=getattr(
            INPUT_ARG_OBJECT, slurm_io.SLURM_FILE_INPUT_ARG))
