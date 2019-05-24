"""Interpolates NWP sounding to each storm object at each lead time."""

import argparse
import numpy
from gewittergefahr.gg_io import storm_tracking_io as tracking_io
from gewittergefahr.gg_utils import nwp_model_utils
from gewittergefahr.gg_utils import soundings
from gewittergefahr.gg_utils import echo_top_tracking
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import number_rounding
from gewittergefahr.gg_utils import storm_tracking_utils as tracking_utils

HOURS_TO_SECONDS = 3600
STORM_TIME_FORMAT = '%Y-%m-%d-%H%M%S'
MODEL_INIT_TIME_FORMAT = '%Y-%m-%d-%H'
SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

WGRIB_EXE_NAME = '/condo/swatwork/ralager/wgrib/wgrib'
WGRIB2_EXE_NAME = '/condo/swatwork/ralager/grib2/wgrib2/wgrib2'

FIRST_RAP_TIME_STRING = '2012-05-01-00'
FIRST_RAP_TIME_UNIX_SEC = time_conversion.string_to_unix_sec(
    FIRST_RAP_TIME_STRING, MODEL_INIT_TIME_FORMAT)

SPC_DATE_ARG_NAME = 'spc_date_string'
LEAD_TIMES_ARG_NAME = 'lead_times_seconds'
LAG_TIME_ARG_NAME = 'lag_time_for_convective_contamination_sec'
RUC_DIRECTORY_ARG_NAME = 'input_ruc_directory_name'
RAP_DIRECTORY_ARG_NAME = 'input_rap_directory_name'
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

LAG_TIME_HELP_STRING = (
    'Lag time (used to avoid convective contamination of soundings, where the '
    'sounding for storm S is heavily influenced by storm S).  This will be '
    'subtracted from each lead time in `{0:s}`.'
).format(LEAD_TIMES_ARG_NAME)

RUC_DIRECTORY_HELP_STRING = (
    'Name of top-level directory with grib files containing RUC (Rapid Update '
    'Cycle) data, which will be used for all model-initialization times < '
    '{0:s}.'
).format(FIRST_RAP_TIME_STRING)

RAP_DIRECTORY_HELP_STRING = (
    'Name of top-level directory with grib files containing RAP (Rapid Refresh)'
    ' data, which will be used for all model-initialization times >= {0:s}.'
).format(FIRST_RAP_TIME_STRING)

TRACKING_DIR_HELP_STRING = (
    'Name of top-level directory with storm tracks (one file per time step, '
    'readable by `storm_tracking_io.read_processed_file`).')

TRACKING_SCALE_HELP_STRING = (
    'Tracking scale (minimum storm area).  Used to find input data.')

OUTPUT_DIR_HELP_STRING = (
    'Name of top-level directory for soundings (one file per SPC date, written '
    'by `soundings.write_soundings`).')

DEFAULT_LEAD_TIMES_SECONDS = [0]

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + SPC_DATE_ARG_NAME, type=str, required=True,
    help=SPC_DATE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + LEAD_TIMES_ARG_NAME, type=int, nargs='+', required=False,
    default=[0], help=LEAD_TIMES_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + LAG_TIME_ARG_NAME, type=int, required=False,
    default=soundings.DEFAULT_LAG_TIME_FOR_CONVECTIVE_CONTAMINATION_SEC,
    help=LAG_TIME_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + RUC_DIRECTORY_ARG_NAME, type=str, required=True,
    help=RUC_DIRECTORY_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + RAP_DIRECTORY_ARG_NAME, type=str, required=True,
    help=RAP_DIRECTORY_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + TRACKING_DIR_ARG_NAME, type=str, required=True,
    help=TRACKING_DIR_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + TRACKING_SCALE_ARG_NAME, type=int, required=False,
    default=echo_top_tracking.DUMMY_TRACKING_SCALE_METRES2,
    help=TRACKING_SCALE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING)


def _interp_soundings(
        spc_date_string, lead_times_seconds,
        lag_time_for_convective_contamination_sec, top_ruc_directory_name,
        top_rap_directory_name, top_tracking_dir_name, tracking_scale_metres2,
        top_output_dir_name):
    """Interpolates NWP sounding to each storm object at each lead time.

    :param spc_date_string: See documentation at top of file.
    :param lead_times_seconds: Same.
    :param lag_time_for_convective_contamination_sec: Same.
    :param top_ruc_directory_name: Same.
    :param top_rap_directory_name: Same.
    :param top_tracking_dir_name: Same.
    :param tracking_scale_metres2: Same.
    :param top_output_dir_name: Same.
    :raises: ValueError: if model-initialization times needed are on opposite
        sides of 0000 UTC 1 May 2012 (the cutoff between RUC and RAP models).
    """

    lead_times_seconds = numpy.array(lead_times_seconds, dtype=int)

    tracking_file_names = tracking_io.find_files_one_spc_date(
        spc_date_string=spc_date_string,
        source_name=tracking_utils.SEGMOTION_NAME,
        top_tracking_dir_name=top_tracking_dir_name,
        tracking_scale_metres2=tracking_scale_metres2
    )[0]

    storm_object_table = tracking_io.read_many_files(tracking_file_names)
    print(SEPARATOR_STRING)

    first_storm_time_unix_sec = numpy.min(
        storm_object_table[tracking_utils.VALID_TIME_COLUMN].values
    )
    last_storm_time_unix_sec = numpy.max(
        storm_object_table[tracking_utils.VALID_TIME_COLUMN].values
    )

    first_init_time_unix_sec = number_rounding.floor_to_nearest(
        (first_storm_time_unix_sec + numpy.min(lead_times_seconds) -
         lag_time_for_convective_contamination_sec),
        HOURS_TO_SECONDS
    )

    last_init_time_unix_sec = number_rounding.floor_to_nearest(
        (last_storm_time_unix_sec + numpy.max(lead_times_seconds) -
         lag_time_for_convective_contamination_sec),
        HOURS_TO_SECONDS
    )

    extreme_init_times_unix_sec = numpy.array(
        [first_init_time_unix_sec, last_init_time_unix_sec], dtype=int
    )

    if numpy.all(extreme_init_times_unix_sec < FIRST_RAP_TIME_UNIX_SEC):
        top_grib_directory_name = top_ruc_directory_name
        model_name = nwp_model_utils.RUC_MODEL_NAME

    elif numpy.all(extreme_init_times_unix_sec >= FIRST_RAP_TIME_UNIX_SEC):
        top_grib_directory_name = top_rap_directory_name
        model_name = nwp_model_utils.RAP_MODEL_NAME

    else:
        first_storm_time_string = time_conversion.unix_sec_to_string(
            first_storm_time_unix_sec, STORM_TIME_FORMAT
        )
        last_storm_time_string = time_conversion.unix_sec_to_string(
            last_storm_time_unix_sec, STORM_TIME_FORMAT
        )
        first_init_time_string = time_conversion.unix_sec_to_string(
            first_init_time_unix_sec, MODEL_INIT_TIME_FORMAT
        )
        last_init_time_string = time_conversion.unix_sec_to_string(
            last_init_time_unix_sec, MODEL_INIT_TIME_FORMAT
        )

        error_string = (
            'First and last storm times are {0:s} and {1:s}.  Thus, first and '
            'last model-initialization times needed are {2:s} and {3:s}, which '
            'are on opposite sides of {4:s} (the cutoff between RUC and RAP '
            'models).  The code is not generalized enough to interp data from '
            'two different models.  Sorry, eh?'
        ).format(first_storm_time_string, last_storm_time_string,
                 first_init_time_string, last_init_time_string,
                 FIRST_RAP_TIME_STRING)

        raise ValueError(error_string)

    sounding_dict_by_lead_time = soundings.interp_soundings_to_storm_objects(
        storm_object_table=storm_object_table,
        top_grib_directory_name=top_grib_directory_name,
        model_name=model_name, use_all_grids=True,
        height_levels_m_agl=soundings.DEFAULT_HEIGHT_LEVELS_M_AGL,
        lead_times_seconds=lead_times_seconds,
        lag_time_for_convective_contamination_sec=
        lag_time_for_convective_contamination_sec,
        wgrib_exe_name=WGRIB_EXE_NAME, wgrib2_exe_name=WGRIB2_EXE_NAME,
        raise_error_if_missing=False)

    print(SEPARATOR_STRING)
    num_lead_times = len(lead_times_seconds)

    for k in range(num_lead_times):
        this_sounding_file_name = soundings.find_sounding_file(
            top_directory_name=top_output_dir_name,
            spc_date_string=spc_date_string,
            lead_time_seconds=lead_times_seconds[k],
            lag_time_for_convective_contamination_sec=
            lag_time_for_convective_contamination_sec,
            raise_error_if_missing=False)

        print('Writing soundings to: "{0:s}"...'.format(
            this_sounding_file_name))

        soundings.write_soundings(
            netcdf_file_name=this_sounding_file_name,
            sounding_dict_height_coords=sounding_dict_by_lead_time[k],
            lead_time_seconds=lead_times_seconds[k],
            lag_time_for_convective_contamination_sec=
            lag_time_for_convective_contamination_sec)


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _interp_soundings(
        spc_date_string=getattr(INPUT_ARG_OBJECT, SPC_DATE_ARG_NAME),
        lead_times_seconds=getattr(INPUT_ARG_OBJECT, LEAD_TIMES_ARG_NAME),
        lag_time_for_convective_contamination_sec=getattr(
            INPUT_ARG_OBJECT, LAG_TIME_ARG_NAME),
        top_ruc_directory_name=getattr(
            INPUT_ARG_OBJECT, RUC_DIRECTORY_ARG_NAME),
        top_rap_directory_name=getattr(
            INPUT_ARG_OBJECT, RAP_DIRECTORY_ARG_NAME),
        top_tracking_dir_name=getattr(INPUT_ARG_OBJECT, TRACKING_DIR_ARG_NAME),
        tracking_scale_metres2=getattr(
            INPUT_ARG_OBJECT, TRACKING_SCALE_ARG_NAME),
        top_output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
