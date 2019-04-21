"""Downloads zero-hour analyses from the RAP (Rapid Refresh) model."""

import time
import argparse
import numpy
from gewittergefahr.gg_io import nwp_model_io
from gewittergefahr.gg_utils import nwp_model_utils
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import time_periods

INPUT_TIME_FORMAT = '%Y%m%d%H'
DEFAULT_TIME_FORMAT = '%Y-%m-%d-%H'
HOURS_TO_SECONDS = 3600
SECONDS_TO_PAUSE_BETWEEN_FILES = 5

FIRST_INIT_TIME_INPUT_ARG = 'first_init_time_string'
LAST_INIT_TIME_INPUT_ARG = 'last_init_time_string'
LOCAL_DIR_INPUT_ARG = 'top_local_directory_name'

INIT_TIME_HELP_STRING = (
    'Model-initialization time (format "yyyymmddHH").  This script will '
    'download zero-hour forecasts for all hours from `{0:s}`...`{1:s}`.'
).format(FIRST_INIT_TIME_INPUT_ARG, LAST_INIT_TIME_INPUT_ARG)
LOCAL_DIR_HELP_STRING = 'Name of top-level local directory for grib files.'

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + FIRST_INIT_TIME_INPUT_ARG, type=str, required=True,
    help=INIT_TIME_HELP_STRING)
INPUT_ARG_PARSER.add_argument(
    '--' + LAST_INIT_TIME_INPUT_ARG, type=str, required=True,
    help=INIT_TIME_HELP_STRING)
INPUT_ARG_PARSER.add_argument(
    '--' + LOCAL_DIR_INPUT_ARG, type=str, required=True,
    help=LOCAL_DIR_HELP_STRING)


def _download_rap_analyses(first_init_time_string, last_init_time_string,
                           top_local_directory_name):
    """Downloads zero-hour analyses from the RAP (Rapid Refresh) model.

    :param first_init_time_string: See documentation at top of file.
    :param last_init_time_string: Same.
    :param top_local_directory_name: Same.
    """

    first_init_time_unix_sec = time_conversion.string_to_unix_sec(
        first_init_time_string, INPUT_TIME_FORMAT)
    last_init_time_unix_sec = time_conversion.string_to_unix_sec(
        last_init_time_string, INPUT_TIME_FORMAT)
    time_interval_sec = HOURS_TO_SECONDS * nwp_model_utils.get_time_steps(
        nwp_model_utils.RAP_MODEL_NAME
    )[1]

    init_times_unix_sec = time_periods.range_and_interval_to_list(
        start_time_unix_sec=first_init_time_unix_sec,
        end_time_unix_sec=last_init_time_unix_sec,
        time_interval_sec=time_interval_sec)
    init_time_strings = [
        time_conversion.unix_sec_to_string(t, DEFAULT_TIME_FORMAT)
        for t in init_times_unix_sec]

    num_init_times = len(init_times_unix_sec)
    local_file_names = [None] * num_init_times

    for i in range(num_init_times):
        local_file_names[i] = nwp_model_io.find_rap_file_any_grid(
            top_directory_name=top_local_directory_name,
            init_time_unix_sec=init_times_unix_sec[i], lead_time_hours=0,
            raise_error_if_missing=False)
        if local_file_names[i] is not None:
            continue

        local_file_names[i] = nwp_model_io.download_rap_file_any_grid(
            top_local_directory_name=top_local_directory_name,
            init_time_unix_sec=init_times_unix_sec[i], lead_time_hours=0,
            raise_error_if_fails=False)

        if local_file_names[i] is None:
            print '\nPROBLEM.  Download failed for {0:s}.\n\n'.format(
                init_time_strings[i])
        else:
            print '\nSUCCESS.  File was downloaded to "{0:s}".\n\n'.format(
                local_file_names[i])

        time.sleep(SECONDS_TO_PAUSE_BETWEEN_FILES)

    num_downloaded = numpy.sum(numpy.array(
        [f is not None for f in local_file_names]))
    print '{0:d} of {1:d} files were downloaded successfully!'.format(
        num_downloaded, num_init_times)


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _download_rap_analyses(
        first_init_time_string=getattr(
            INPUT_ARG_OBJECT, FIRST_INIT_TIME_INPUT_ARG),
        last_init_time_string=getattr(
            INPUT_ARG_OBJECT, LAST_INIT_TIME_INPUT_ARG),
        top_local_directory_name=getattr(INPUT_ARG_OBJECT, LOCAL_DIR_INPUT_ARG))
