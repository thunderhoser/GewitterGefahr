"""Untars MYRORSS data."""

import argparse
import numpy
from gewittergefahr.gg_io import myrorss_io
from gewittergefahr.gg_utils import radar_utils
from gewittergefahr.gg_utils import time_conversion

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

INPUT_DIR_ARG_NAME = 'top_input_dir_name'
RADAR_FIELDS_ARG_NAME = 'radar_field_names'
REFL_HEIGHTS_ARG_NAME = 'refl_heights_m_asl'
FIRST_DATE_ARG_NAME = 'first_spc_date_string'
LAST_DATE_ARG_NAME = 'last_spc_date_string'
OUTPUT_DIR_ARG_NAME = 'top_output_dir_name'

INPUT_DIR_HELP_STRING = (
    'Name of top-level input directory, containing daily tar files.')

RADAR_FIELDS_HELP_STRING = (
    'List of radar fields to untar.  Each must be accepted by '
    '`radar_utils.check_field_name`.')

REFL_HEIGHTS_HELP_STRING = (
    'List of heights to untar for field "{0:s}" (in metres above sea level).  '
    'If `{1:s}` does not contain "{0:s}", you can leave this argument alone.'
).format(radar_utils.REFL_NAME, RADAR_FIELDS_ARG_NAME)

DATE_HELP_STRING = (
    'SPC date (format "yyyymmdd").  Data will be untarred for the period '
    '`{0:s}`...`{1:s}`.'
).format(FIRST_DATE_ARG_NAME, LAST_DATE_ARG_NAME)

OUTPUT_DIR_HELP_STRING = (
    'Name of top-level output directory.  Data will be untarred to here.')

DEFAULT_INPUT_DIR_NAME = '/condo/swatcommon/common/myrorss'
DEFAULT_OUTPUT_DIR_NAME = '/condo/swatwork/ralager/myrorss_temp'

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + INPUT_DIR_ARG_NAME, type=str, required=False,
    default=DEFAULT_INPUT_DIR_NAME, help=INPUT_DIR_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + RADAR_FIELDS_ARG_NAME, type=str, nargs='+', required=True,
    help=RADAR_FIELDS_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + REFL_HEIGHTS_ARG_NAME, type=int, nargs='+', required=False,
    default=[-1], help=REFL_HEIGHTS_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + FIRST_DATE_ARG_NAME, type=str, required=True, help=DATE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + LAST_DATE_ARG_NAME, type=str, required=True, help=DATE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=False,
    default=DEFAULT_OUTPUT_DIR_NAME, help=OUTPUT_DIR_HELP_STRING)


def _run(top_input_dir_name, radar_field_names, refl_heights_m_asl,
         first_spc_date_string, last_spc_date_string, top_output_dir_name):
    """Untars MYRORSS data.

    This is effectively the main method.

    :param top_input_dir_name: See documentation at top of file.
    :param radar_field_names: Same.
    :param refl_heights_m_asl: Same.
    :param first_spc_date_string: Same.
    :param last_spc_date_string: Same.
    :param top_output_dir_name: Same.
    """

    spc_date_strings = time_conversion.get_spc_dates_in_range(
        first_spc_date_string=first_spc_date_string,
        last_spc_date_string=last_spc_date_string)

    az_shear_field_names = list(
        set(radar_field_names) & set(radar_utils.SHEAR_NAMES)
    )

    non_shear_field_names = list(
        set(radar_field_names) - set(radar_utils.SHEAR_NAMES)
    )

    for this_spc_date_string in spc_date_strings:
        if len(az_shear_field_names) > 0:
            this_tar_file_name = (
                '{0:s}/{1:s}/azimuthal_shear_only/{2:s}.tar'
            ).format(
                top_input_dir_name, this_spc_date_string[:4],
                this_spc_date_string
            )

            myrorss_io.unzip_1day_tar_file(
                tar_file_name=this_tar_file_name,
                field_names=az_shear_field_names,
                spc_date_string=this_spc_date_string,
                top_target_directory_name=top_output_dir_name)
            print(SEPARATOR_STRING)

        if len(non_shear_field_names) > 0:
            this_tar_file_name = '{0:s}/{1:s}/{2:s}.tar'.format(
                top_input_dir_name, this_spc_date_string[:4],
                this_spc_date_string
            )

            myrorss_io.unzip_1day_tar_file(
                tar_file_name=this_tar_file_name,
                field_names=non_shear_field_names,
                spc_date_string=this_spc_date_string,
                top_target_directory_name=top_output_dir_name,
                refl_heights_m_asl=refl_heights_m_asl)
            print(SEPARATOR_STRING)


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        top_input_dir_name=getattr(INPUT_ARG_OBJECT, INPUT_DIR_ARG_NAME),
        radar_field_names=getattr(INPUT_ARG_OBJECT, RADAR_FIELDS_ARG_NAME),
        refl_heights_m_asl=numpy.array(
            getattr(INPUT_ARG_OBJECT, REFL_HEIGHTS_ARG_NAME), dtype=int
        ),
        first_spc_date_string=getattr(INPUT_ARG_OBJECT, FIRST_DATE_ARG_NAME),
        last_spc_date_string=getattr(INPUT_ARG_OBJECT, LAST_DATE_ARG_NAME),
        top_output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
