"""Tracks storms based on echo top."""

import argparse
from gewittergefahr.gg_io import myrorss_io
from gewittergefahr.gg_utils import radar_utils
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import echo_top_tracking

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

NATIVE_ECHO_TOP_FIELD_NAMES = [
    radar_utils.ECHO_TOP_18DBZ_NAME, radar_utils.ECHO_TOP_50DBZ_NAME
]

RADAR_DIR_ARG_NAME = 'input_radar_dir_name'
TARRED_RADAR_DIR_ARG_NAME = 'input_radar_dir_name_tarred'
ECHO_CLASSIFN_DIR_ARG_NAME = 'input_echo_classifn_dir_name'
FIRST_SPC_DATE_ARG_NAME = 'first_spc_date_string'
LAST_SPC_DATE_ARG_NAME = 'last_spc_date_string'
ECHO_TOP_FIELD_ARG_NAME = 'echo_top_field_name'
MIN_ECHO_TOP_ARG_NAME = 'min_echo_top_km'
MIN_INTERMAX_DISTANCE_ARG_NAME = 'min_intermax_distance_metres'
MAX_VELOCITY_DIFF_ARG_NAME = 'max_velocity_diff_m_s01'
MAX_LINK_DISTANCE_ARG_NAME = 'max_link_distance_m_s01'
OUTPUT_DIR_ARG_NAME = 'output_tracking_dir_name'

RADAR_DIR_HELP_STRING = (
    'Name of top-level radar directory.  Files therein will be found by '
    '`echo_top_tracking._find_input_radar_files`.')

TARRED_RADAR_DIR_HELP_STRING = (
    '[used only if {0:s} = "{1:s}" or "{2:s}"] Name of top-level directory with'
    ' tarred MYRORSS files.  These files will be untarred before tracking (into'
    ' `{3:s}`) and the untarred files will be deleted after tracking.'
).format(
    ECHO_TOP_FIELD_ARG_NAME, NATIVE_ECHO_TOP_FIELD_NAMES[0],
    NATIVE_ECHO_TOP_FIELD_NAMES[1], RADAR_DIR_ARG_NAME
)

ECHO_CLASSIFN_DIR_HELP_STRING = (
    'Name of top-level directory with echo-classification files.  Files therein'
    ' will be found by `echo_classification.find_classification_file` and read '
    'by `echo_classification.read_classifications`.  Tracking will be performed'
    ' only on convective pixels.  If you do not want to use a convective mask, '
    'leave this argument alone.')

SPC_DATE_HELP_STRING = (
    'SPC date (format "yyyymmdd").  Tracking will be performed for all SPC '
    'dates in the period `{0:s}`...`{1:s}`.'
).format(FIRST_SPC_DATE_ARG_NAME, LAST_SPC_DATE_ARG_NAME)

ECHO_TOP_FIELD_HELP_STRING = (
    'Name of echo-top field to use for tracking.  Must be accepted by '
    '`echo_top_tracking._check_radar_field`.')

MIN_ECHO_TOP_HELP_STRING = (
    'Minimum echo top.  Smaller values are not considered storms.')

MIN_INTERMAX_DISTANCE_HELP_STRING = (
    'Minimum distance between any pair of local maxima at the same time.  See '
    '`echo_top_tracking._remove_redundant_local_maxima` for details.')

MAX_VELOCITY_DIFF_HELP_STRING = (
    'Used to connect local maxima (storm objects) between times.  See '
    '`echo_top_tracking._link_local_maxima_in_time` for details.')

MAX_LINK_DISTANCE_HELP_STRING = (
    'Used to connect local maxima (storm objects) between times.  See '
    '`echo_top_tracking._link_local_maxima_in_time` for details.')

OUTPUT_DIR_HELP_STRING = (
    'Name of top-level output directory.  Output files will be written by '
    '`storm_tracking_io.write_processed_file`, to locations therein determined '
    'by `storm_tracking_io.find_processed_file`.')

TARRED_RADAR_DIR_NAME_DEFAULT = '/condo/swatcommon/common/myrorss'

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + RADAR_DIR_ARG_NAME, type=str, required=True,
    help=RADAR_DIR_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + TARRED_RADAR_DIR_ARG_NAME, type=str, required=False,
    default=TARRED_RADAR_DIR_NAME_DEFAULT, help=TARRED_RADAR_DIR_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + ECHO_CLASSIFN_DIR_ARG_NAME, type=str, required=False, default='',
    help=ECHO_CLASSIFN_DIR_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + FIRST_SPC_DATE_ARG_NAME, type=str, required=True,
    help=SPC_DATE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + LAST_SPC_DATE_ARG_NAME, type=str, required=True,
    help=SPC_DATE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + ECHO_TOP_FIELD_ARG_NAME, type=str, required=False,
    default=radar_utils.ECHO_TOP_40DBZ_NAME, help=ECHO_TOP_FIELD_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + MIN_ECHO_TOP_ARG_NAME, type=float, required=False, default=4.,
    help=MIN_ECHO_TOP_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + MIN_INTERMAX_DISTANCE_ARG_NAME, type=float, required=False,
    default=echo_top_tracking.DEFAULT_MIN_INTERMAX_DISTANCE_METRES,
    help=MIN_INTERMAX_DISTANCE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + MAX_VELOCITY_DIFF_ARG_NAME, type=float, required=False,
    default=echo_top_tracking.DEFAULT_MAX_VELOCITY_DIFF_M_S01,
    help=MAX_VELOCITY_DIFF_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + MAX_LINK_DISTANCE_ARG_NAME, type=float, required=False,
    default=echo_top_tracking.DEFAULT_MAX_LINK_DISTANCE_M_S01,
    help=MAX_LINK_DISTANCE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING)


def _run(top_radar_dir_name, top_radar_dir_name_tarred,
         top_echo_classifn_dir_name, first_spc_date_string,
         last_spc_date_string, echo_top_field_name, min_echo_top_km,
         min_intermax_distance_metres, max_velocity_diff_m_s01,
         max_link_distance_m_s01, top_output_dir_name):
    """Tracks storms based on echo top.

    This is effectively the main method.

    :param top_radar_dir_name: See documentation at top of file.
    :param top_radar_dir_name_tarred: Same.
    :param top_echo_classifn_dir_name: Same.
    :param first_spc_date_string: Same.
    :param last_spc_date_string: Same.
    :param echo_top_field_name: Same.
    :param min_echo_top_km: Same.
    :param min_intermax_distance_metres: Same.
    :param max_velocity_diff_m_s01: Same.
    :param max_link_distance_m_s01: Same.
    :param top_output_dir_name: Same.
    """

    if echo_top_field_name in NATIVE_ECHO_TOP_FIELD_NAMES:
        spc_date_strings = time_conversion.get_spc_dates_in_range(
            first_spc_date_string=first_spc_date_string,
            last_spc_date_string=last_spc_date_string)

        for this_spc_date_string in spc_date_strings:
            this_tar_file_name = '{0:s}/{1:s}/{2:s}.tar'.format(
                top_radar_dir_name_tarred, this_spc_date_string[:4],
                this_spc_date_string)

            myrorss_io.unzip_1day_tar_file(
                tar_file_name=this_tar_file_name,
                field_names=[echo_top_field_name],
                spc_date_string=this_spc_date_string,
                top_target_directory_name=top_radar_dir_name)

            print SEPARATOR_STRING

    if top_echo_classifn_dir_name in ['', 'None']:
        top_echo_classifn_dir_name = None

    echo_top_tracking.run_tracking(
        top_radar_dir_name=top_radar_dir_name,
        top_output_dir_name=top_output_dir_name,
        first_spc_date_string=first_spc_date_string,
        last_spc_date_string=last_spc_date_string,
        echo_top_field_name=echo_top_field_name,
        top_echo_classifn_dir_name=top_echo_classifn_dir_name,
        min_echo_top_km=min_echo_top_km,
        min_intermax_distance_metres=min_intermax_distance_metres,
        max_velocity_diff_m_s01=max_velocity_diff_m_s01,
        max_link_distance_m_s01=max_link_distance_m_s01,
        min_track_duration_seconds=0)

    print SEPARATOR_STRING

    if echo_top_field_name in NATIVE_ECHO_TOP_FIELD_NAMES:
        for this_spc_date_string in spc_date_strings:
            myrorss_io.remove_unzipped_data_1day(
                spc_date_string=this_spc_date_string,
                top_directory_name=top_radar_dir_name,
                field_names=[echo_top_field_name]
            )


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        top_radar_dir_name=getattr(INPUT_ARG_OBJECT, RADAR_DIR_ARG_NAME),
        top_radar_dir_name_tarred=getattr(
            INPUT_ARG_OBJECT, TARRED_RADAR_DIR_ARG_NAME),
        top_echo_classifn_dir_name=getattr(
            INPUT_ARG_OBJECT, ECHO_CLASSIFN_DIR_ARG_NAME),
        first_spc_date_string=getattr(
            INPUT_ARG_OBJECT, FIRST_SPC_DATE_ARG_NAME),
        last_spc_date_string=getattr(INPUT_ARG_OBJECT, LAST_SPC_DATE_ARG_NAME),
        echo_top_field_name=getattr(INPUT_ARG_OBJECT, ECHO_TOP_FIELD_ARG_NAME),
        min_echo_top_km=getattr(INPUT_ARG_OBJECT, MIN_ECHO_TOP_ARG_NAME),
        min_intermax_distance_metres=getattr(
            INPUT_ARG_OBJECT, MIN_INTERMAX_DISTANCE_ARG_NAME),
        max_velocity_diff_m_s01=getattr(
            INPUT_ARG_OBJECT, MAX_VELOCITY_DIFF_ARG_NAME),
        max_link_distance_m_s01=getattr(
            INPUT_ARG_OBJECT, MAX_LINK_DISTANCE_ARG_NAME),
        top_output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
