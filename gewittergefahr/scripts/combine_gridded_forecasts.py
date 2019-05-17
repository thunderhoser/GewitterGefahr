"""Combines forecasts from different initial times."""

import argparse
import numpy
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import time_periods
from gewittergefahr.deep_learning import prediction_io

# TODO(thunderhoser): Either generalize or delete this code.

INPUT_TIME_FORMAT = '%Y-%m-%d-%H%M%S'
MIN_LEAD_TIME_SECONDS = 0
MAX_LEAD_TIME_SECONDS = 3600

INPUT_DIR_ARG_NAME = 'input_prediction_dir_name'
FIRST_INIT_TIME_ARG_NAME = 'first_init_time_string'
LAST_INIT_TIME_ARG_NAME = 'last_init_time_string'
OUTPUT_DIR_ARG_NAME = 'output_prediction_dir_name'

INPUT_DIR_HELP_STRING = (
    'Name of input directory.  Forecast grids will be found therein by '
    '`prediction_io.find_file` and read by '
    '`prediction_io.read_gridded_predictions`.')

INIT_TIME_HELP_STRING = (
    'Initial time (format "yyyy-mm-dd-HHMMSS").  This script will take the max,'
    ' at each grid cell, over all initial times from `{0:s}`...`{1:s}` with a '
    '{2:d}-second interval.'
).format(
    FIRST_INIT_TIME_ARG_NAME, LAST_INIT_TIME_ARG_NAME, MAX_LEAD_TIME_SECONDS
)

OUTPUT_DIR_HELP_STRING = (
    'Name of output directory.  The final grid will be written by'
    '`prediction_io.write_gridded_predictions`, to an exact location determined'
    ' by `prediction_io.find_file`.')

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + INPUT_DIR_ARG_NAME, type=str, required=True,
    help=INPUT_DIR_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + FIRST_INIT_TIME_ARG_NAME, type=str, required=True,
    help=INIT_TIME_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + LAST_INIT_TIME_ARG_NAME, type=str, required=True,
    help=INIT_TIME_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING)


def _run(top_input_dir_name, first_init_time_string, last_init_time_string,
         top_output_dir_name):
    """Combines forecasts from different initial times.

    This is effectively the main method.

    :param top_input_dir_name: See documentation at top of file.
    :param first_init_time_string: Same.
    :param last_init_time_string: Same.
    :param top_output_dir_name: Same.
    """

    first_init_time_unix_sec = time_conversion.string_to_unix_sec(
        first_init_time_string, INPUT_TIME_FORMAT)
    last_init_time_unix_sec = time_conversion.string_to_unix_sec(
        last_init_time_string, INPUT_TIME_FORMAT)

    init_times_unix_sec = time_periods.range_and_interval_to_list(
        start_time_unix_sec=first_init_time_unix_sec,
        end_time_unix_sec=last_init_time_unix_sec,
        time_interval_sec=MAX_LEAD_TIME_SECONDS, include_endpoint=True)

    probability_matrix = None
    gridded_forecast_dict = None

    for this_time_unix_sec in init_times_unix_sec:
        this_file_name = prediction_io.find_file(
            top_prediction_dir_name=top_input_dir_name,
            first_init_time_unix_sec=this_time_unix_sec,
            last_init_time_unix_sec=this_time_unix_sec, gridded=True,
            raise_error_if_missing=True)

        print 'Reading data from: "{0:s}"...'.format(this_file_name)
        gridded_forecast_dict = prediction_io.read_gridded_predictions(
            this_file_name)

        assert (
            gridded_forecast_dict[prediction_io.MIN_LEAD_TIME_KEY] ==
            MIN_LEAD_TIME_SECONDS
        )
        assert (
            gridded_forecast_dict[prediction_io.MAX_LEAD_TIME_KEY] ==
            MAX_LEAD_TIME_SECONDS
        )

        this_probability_matrix = gridded_forecast_dict[
            prediction_io.XY_PROBABILITIES_KEY
        ][0]

        if not isinstance(this_probability_matrix, numpy.ndarray):
            this_probability_matrix = this_probability_matrix.toarray()

        if probability_matrix is None:
            probability_matrix = this_probability_matrix + 0.
        else:
            probability_matrix = numpy.stack(
                (probability_matrix, this_probability_matrix), axis=-1
            )
            probability_matrix = numpy.nanmax(probability_matrix, axis=-1)

        print probability_matrix.shape

    print '\n'

    for this_key in prediction_io.LATLNG_KEYS:
        if this_key in gridded_forecast_dict:
            gridded_forecast_dict.pop(this_key)

    print init_times_unix_sec[0]
    print init_times_unix_sec[-1]
    print init_times_unix_sec[-1] + MAX_LEAD_TIME_SECONDS - init_times_unix_sec[0]

    gridded_forecast_dict[prediction_io.INIT_TIMES_KEY] = (
        init_times_unix_sec[[0]]
    )
    gridded_forecast_dict[prediction_io.MAX_LEAD_TIME_KEY] = (
        init_times_unix_sec[-1] + MAX_LEAD_TIME_SECONDS - init_times_unix_sec[0]
    )
    gridded_forecast_dict[prediction_io.XY_PROBABILITIES_KEY] = (
        [probability_matrix]
    )

    output_file_name = prediction_io.find_file(
        top_prediction_dir_name=top_output_dir_name,
        first_init_time_unix_sec=init_times_unix_sec[0],
        last_init_time_unix_sec=init_times_unix_sec[-1], gridded=True,
        raise_error_if_missing=False)

    print 'Writing final grid to: "{0:s}"...'.format(output_file_name)

    prediction_io.write_gridded_predictions(
        gridded_forecast_dict=gridded_forecast_dict,
        pickle_file_name=output_file_name)


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        top_input_dir_name=getattr(INPUT_ARG_OBJECT, INPUT_DIR_ARG_NAME),
        first_init_time_string=getattr(
            INPUT_ARG_OBJECT, FIRST_INIT_TIME_ARG_NAME),
        last_init_time_string=getattr(
            INPUT_ARG_OBJECT, LAST_INIT_TIME_ARG_NAME),
        top_output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
