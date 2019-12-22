"""Plots spatial distribution of false alarms."""

import argparse
import numpy
from gewittergefahr.gg_io import storm_tracking_io as tracking_io
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import storm_tracking_utils as tracking_utils
from gewittergefahr.gg_utils import echo_top_tracking
from gewittergefahr.gg_utils import error_checking
from gewittergefahr.deep_learning import prediction_io

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'
DUMMY_TRACKING_SCALE_METRES2 = echo_top_tracking.DUMMY_TRACKING_SCALE_METRES2

PREDICTION_FILE_ARG_NAME = 'input_prediction_file_name'
TRACKING_DIR_ARG_NAME = 'input_tracking_dir_name'
PROB_THRESHOLD_ARG_NAME = 'prob_threshold'
GRID_SPACING_ARG_NAME = 'grid_spacing_metres'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

PREDICTION_FILE_HELP_STRING = (
    'Path to prediction file.  Will be read by '
    '`prediction_io.read_ungridded_predictions`.'
)

TRACKING_DIR_HELP_STRING = (
    'Name of top-level directory with tracking data.  Files therein will be '
    'found by `storm_tracking_io.find_file` and read by'
    ' `storm_tracking_io.read_file`.'
)

PROB_THRESHOLD_HELP_STRING = (
    'Probability threshold.  Any negative example with forecast >= `{0:s}` will'
    ' be considered a false alarm.'
).format(PROB_THRESHOLD_ARG_NAME)

GRID_SPACING_HELP_STRING = 'Grid spacing.'
OUTPUT_DIR_HELP_STRING = 'Name of output directory (figure will be saved here).'

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + PREDICTION_FILE_ARG_NAME, type=str, required=True,
    help=PREDICTION_FILE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + TRACKING_DIR_ARG_NAME, type=str, required=True,
    help=TRACKING_DIR_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + PROB_THRESHOLD_ARG_NAME, type=float, required=True,
    help=PROB_THRESHOLD_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + GRID_SPACING_ARG_NAME, type=float, required=False, default=1e5,
    help=GRID_SPACING_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING
)


def _run(prediction_file_name, top_tracking_dir_name, prob_threshold,
         grid_spacing_metres, output_dir_name):
    """Plots spatial distribution of false alarms.

    This is effectively the main method.

    :param prediction_file_name: See documentation at top of file.
    :param top_tracking_dir_name: Same.
    :param prob_threshold: Same.
    :param grid_spacing_metres: Same.
    :param output_dir_name: Same.
    """

    error_checking.assert_is_greater(prob_threshold, 0.)
    error_checking.assert_is_less_than(prob_threshold, 1.)

    print('Reading predictions from: "{0:s}"...'.format(prediction_file_name))
    prediction_dict = prediction_io.read_ungridded_predictions(
        prediction_file_name
    )

    observed_labels = prediction_dict[prediction_io.OBSERVED_LABELS_KEY]
    forecast_labels = (
        prediction_dict[prediction_io.PROBABILITY_MATRIX_KEY][:, -1] >=
        prob_threshold
    ).astype(int)

    false_alarm_indices = numpy.where(
        observed_labels == 0, forecast_labels == 1
    )[0]

    num_examples = len(observed_labels)
    num_false_alarms = len(false_alarm_indices)
    print((
        'Number of false alarms (at probability threshold of {0:.3f}) = '
        '{1:d} of {2:d}'
    ).format(
        prob_threshold, num_false_alarms, num_examples
    ))

    full_storm_id_strings = prediction_dict[prediction_io.STORM_IDS_KEY]
    storm_times_unix_sec = prediction_dict[prediction_io.STORM_TIMES_KEY]

    file_times_unix_sec = numpy.unique(storm_times_unix_sec)
    num_files = len(file_times_unix_sec)
    tracking_file_names = [None] * num_files

    for i in range(num_files):
        tracking_file_names[i] = tracking_io.find_file(
            top_tracking_dir_name=top_tracking_dir_name,
            tracking_scale_metres2=DUMMY_TRACKING_SCALE_METRES2,
            source_name=tracking_utils.SEGMOTION_NAME,
            valid_time_unix_sec=file_times_unix_sec[i],
            spc_date_string=time_conversion.time_to_spc_date_string(
                file_times_unix_sec[i]
            ),
            raise_error_if_missing=True
        )

    print(SEPARATOR_STRING)
    storm_object_table = tracking_io.read_many_files(tracking_file_names)
    print(SEPARATOR_STRING)

    all_id_strings = (
        storm_object_table[tracking_utils.FULL_ID_COLUMN].values.tolist()
    )
    all_times_unix_sec = (
        storm_object_table[tracking_utils.VALID_TIME_COLUMN].values
    )

    good_indices = tracking_utils.find_storm_objects(
        all_id_strings=all_id_strings, all_times_unix_sec=all_times_unix_sec,
        id_strings_to_keep=full_storm_id_strings,
        times_to_keep_unix_sec=storm_times_unix_sec,
        allow_missing=False)

    storm_latitudes_deg = storm_object_table[
        tracking_utils.CENTROID_LATITUDE_COLUMN
    ].values[good_indices]

    storm_longitudes_deg = storm_object_table[
        tracking_utils.CENTROID_LONGITUDE_COLUMN
    ].values[good_indices]


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        prediction_file_name=getattr(
            INPUT_ARG_OBJECT, PREDICTION_FILE_ARG_NAME
        ),
        top_tracking_dir_name=getattr(INPUT_ARG_OBJECT, TRACKING_DIR_ARG_NAME),
        prob_threshold=getattr(INPUT_ARG_OBJECT, PROB_THRESHOLD_ARG_NAME),
        grid_spacing_metres=getattr(INPUT_ARG_OBJECT, GRID_SPACING_ARG_NAME),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
