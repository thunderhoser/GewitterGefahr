"""Interpolates NWP sounding to each storm object at each lead time."""

import argparse
import numpy
from gewittergefahr.gg_io import storm_tracking_io as tracking_io
from gewittergefahr.gg_utils import soundings_only
from gewittergefahr.gg_utils import echo_top_tracking
from gewittergefahr.gg_utils import storm_tracking_utils as tracking_utils

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

WGRIB_EXE_NAME = '/condo/swatwork/ralager/wgrib/wgrib'
WGRIB2_EXE_NAME = '/condo/swatwork/ralager/grib2/wgrib2/wgrib2'

SPC_DATE_ARG_NAME = 'spc_date_string'
LEAD_TIMES_ARG_NAME = 'lead_times_seconds'
RUC_DIRECTORY_ARG_NAME = 'input_ruc_directory_name'
TRACKING_DIR_ARG_NAME = 'input_tracking_dir_name'
TRACKING_SCALE_ARG_NAME = 'tracking_scale_metres2'
OUTPUT_DIR_ARG_NAME = 'output_sounding_dir_name'

SPC_DATE_HELP_STRING = (
    'SPC (Storm Prediction Center) date in format "yyyymmdd".  The RUC (Rapid '
    'Update Cycle) sounding will be interpolated to each storm object on this '
    'date, at each lead time in `{0:s}`.'
).format(LEAD_TIMES_ARG_NAME)
LEAD_TIMES_HELP_STRING = 'See help string for `{0:s}`.'.format(
    SPC_DATE_ARG_NAME)
RUC_DIRECTORY_HELP_STRING = (
    'Name of top-level directory with grib files containing RUC (Rapid Update '
    'Cycle) data.')
TRACKING_DIR_HELP_STRING = (
    'Name of top-level directory with storm tracks (one file per time step, '
    'readable by `storm_tracking_io.read_processed_file`).')
TRACKING_SCALE_HELP_STRING = (
    'Tracking scale (minimum storm area).  Used to find input data.')
OUTPUT_DIR_HELP_STRING = (
    'Name of top-level directory for soundings (one file per SPC date, written '
    'by `soundings_only.write_soundings`).')

DEFAULT_LEAD_TIMES_SECONDS = [0]
DEFAULT_TRACKING_SCALE_METRES2 = int(numpy.round(
    echo_top_tracking.DUMMY_TRACKING_SCALE_METRES2))

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + SPC_DATE_ARG_NAME, type=str, required=True,
    help=SPC_DATE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + LEAD_TIMES_ARG_NAME, type=int, nargs='+', required=False,
    default=[0], help=LEAD_TIMES_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + RUC_DIRECTORY_ARG_NAME, type=str, required=True,
    help=RUC_DIRECTORY_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + TRACKING_DIR_ARG_NAME, type=str, required=True,
    help=TRACKING_DIR_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + TRACKING_SCALE_ARG_NAME, type=int, required=False,
    default=DEFAULT_TRACKING_SCALE_METRES2, help=TRACKING_SCALE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING)


def _interp_soundings(
        spc_date_string, lead_times_seconds, top_ruc_directory_name,
        top_tracking_dir_name, tracking_scale_metres2, top_output_dir_name):
    """Interpolates NWP sounding to each storm object at each lead time.

    :param spc_date_string: See documentation at top of file.
    :param lead_times_seconds: Same.
    :param top_ruc_directory_name: Same.
    :param top_tracking_dir_name: Same.
    :param tracking_scale_metres2: Same.
    :param top_output_dir_name: Same.
    """

    lead_times_seconds = numpy.array(lead_times_seconds, dtype=int)

    tracking_file_names = tracking_io.find_processed_files_one_spc_date(
        spc_date_string=spc_date_string,
        data_source=tracking_utils.SEGMOTION_SOURCE_ID,
        top_processed_dir_name=top_tracking_dir_name,
        tracking_scale_metres2=tracking_scale_metres2)

    storm_object_table = tracking_io.read_many_processed_files(
        tracking_file_names)
    print SEPARATOR_STRING

    sounding_dict_by_lead_time = (
        soundings_only.interp_soundings_to_storm_objects(
            storm_object_table=storm_object_table,
            top_grib_directory_name=top_ruc_directory_name,
            lead_times_seconds=lead_times_seconds, include_surface=False,
            all_ruc_grids=True, wgrib_exe_name=WGRIB_EXE_NAME,
            wgrib2_exe_name=WGRIB2_EXE_NAME, raise_error_if_missing=False))
    print SEPARATOR_STRING

    num_lead_times = len(sounding_dict_by_lead_time)
    for k in range(num_lead_times):
        this_lead_time_sec = sounding_dict_by_lead_time[
            k][soundings_only.LEAD_TIMES_KEY][0]
        this_sounding_file_name = soundings_only.find_sounding_file(
            top_directory_name=top_output_dir_name,
            spc_date_string=spc_date_string,
            lead_time_seconds=this_lead_time_sec, raise_error_if_missing=False)

        print 'Writing soundings to: "{0:s}"...'.format(this_sounding_file_name)
        soundings_only.write_soundings(
            sounding_dict=sounding_dict_by_lead_time[k],
            netcdf_file_name=this_sounding_file_name)


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _interp_soundings(
        spc_date_string=getattr(INPUT_ARG_OBJECT, SPC_DATE_ARG_NAME),
        lead_times_seconds=getattr(INPUT_ARG_OBJECT, LEAD_TIMES_ARG_NAME),
        top_ruc_directory_name=getattr(
            INPUT_ARG_OBJECT, RUC_DIRECTORY_ARG_NAME),
        top_tracking_dir_name=getattr(INPUT_ARG_OBJECT, TRACKING_DIR_ARG_NAME),
        tracking_scale_metres2=getattr(
            INPUT_ARG_OBJECT, TRACKING_SCALE_ARG_NAME),
        top_output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME))
