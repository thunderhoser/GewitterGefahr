"""Uses azimuthal-shear thresholds to make probabilistic tornado predictions."""

import argparse
import numpy
from keras import backend as K
from scipy.stats import rankdata
from gewittergefahr.gg_utils import radar_utils
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.deep_learning import input_examples
from gewittergefahr.deep_learning import prediction_io
from gewittergefahr.deep_learning import testing_io
from gewittergefahr.deep_learning import training_validation_io as trainval_io

K.set_session(K.tf.Session(config=K.tf.ConfigProto(
    intra_op_parallelism_threads=1, inter_op_parallelism_threads=1,
    allow_soft_placement=False
)))

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

LARGE_INTEGER = int(1e10)
NUM_EXAMPLES_PER_BATCH = 1000
TARGET_NAME = 'tornado_lead-time=0000-3600sec_distance=00000-10000m'

EXAMPLE_DIR_ARG_NAME = 'input_example_dir_name'
FIRST_DATE_ARG_NAME = 'first_spc_date_string'
LAST_DATE_ARG_NAME = 'last_spc_date_string'
LOW_LEVEL_ARG_NAME = 'use_low_level'
MID_LEVEL_ARG_NAME = 'use_mid_level'
NUM_ROWS_ARG_NAME = 'num_radar_rows'
NUM_COLUMNS_ARG_NAME = 'num_radar_columns'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

EXAMPLE_DIR_HELP_STRING = (
    'Name of top-level directory with input examples.  Files therein will be '
    'found by `input_examples.find_example_file` and read by '
    '`input_examples.read_example_file`.')

SPC_DATE_HELP_STRING = (
    'SPC date (format "yyyymmdd").  Will create predictions for all examples in'
    ' the period `{0:s}`...`{1:s}`.'
).format(FIRST_DATE_ARG_NAME, LAST_DATE_ARG_NAME)

LEVEL_HELP_STRING = (
    'Boolean flag.  If `{0:s}` = 1, will try thresholds max low-level shear in '
    'storm.  If `{1:s}` = 1, will try thresholds on max mid-level shear in '
    'storm.  If both are 1, will try thresholds on max of both in storm.'
).format(LOW_LEVEL_ARG_NAME, MID_LEVEL_ARG_NAME)

NUM_ROWS_HELP_STRING = (
    'Number of rows in each storm-centered radar grid.  If you want to use the '
    'full grid, leave this argument empty.')

NUM_COLUMNS_HELP_STRING = (
    'Number of columns in each storm-centered radar grid.  If you want to use '
    'the full grid, leave this argument empty.')

OUTPUT_DIR_HELP_STRING = (
    'Name of output directory.  Results will be written by '
    '`prediction_io.write_ungridded_predictions`, to a location therein '
    'determined by `prediction_io.find_ungridded_file`.')

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + EXAMPLE_DIR_ARG_NAME, type=str, required=True,
    help=EXAMPLE_DIR_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + FIRST_DATE_ARG_NAME, type=str, required=True,
    help=SPC_DATE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + LAST_DATE_ARG_NAME, type=str, required=True,
    help=SPC_DATE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + LOW_LEVEL_ARG_NAME, type=int, required=True, help=LEVEL_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + MID_LEVEL_ARG_NAME, type=int, required=True, help=LEVEL_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + NUM_ROWS_ARG_NAME, type=int, required=False, default=-1,
    help=NUM_ROWS_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + NUM_COLUMNS_ARG_NAME, type=int, required=False, default=-1,
    help=NUM_COLUMNS_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING)


def _run(top_example_dir_name, first_spc_date_string, last_spc_date_string,
         use_low_level, use_mid_level, num_radar_rows, num_radar_columns,
         output_dir_name):
    """Uses az-shear thresholds to make probabilistic tornado predictions.

    This is effectively the main method.

    :param top_example_dir_name: See documentation at top of file.
    :param first_spc_date_string: Same.
    :param last_spc_date_string: Same.
    :param use_low_level: Same.
    :param use_mid_level: Same.
    :param num_radar_rows: Same.
    :param num_radar_columns: Same.
    :param output_dir_name: Same.
    :raises: ValueError: if `use_low_level == use_mid_level == False`.
    """

    if num_radar_rows <= 0 or num_radar_columns <= 0:
        num_reflectivity_rows = None
        num_reflectivity_columns = None
    else:
        num_reflectivity_rows = int(numpy.round(num_radar_rows / 2))
        num_reflectivity_columns = int(numpy.round(num_radar_columns / 2))

    if not (use_low_level or use_mid_level):
        error_string = (
            'At least one of `{0:s}` and `{1:s}` must be true.'
        ).format(LOW_LEVEL_ARG_NAME, MID_LEVEL_ARG_NAME)

        raise ValueError(error_string)

    radar_field_names = []
    if use_low_level:
        radar_field_names.append(radar_utils.LOW_LEVEL_SHEAR_NAME)
    if use_mid_level:
        radar_field_names.append(radar_utils.MID_LEVEL_SHEAR_NAME)

    example_file_names = input_examples.find_many_example_files(
        top_directory_name=top_example_dir_name, shuffled=False,
        first_spc_date_string=first_spc_date_string,
        last_spc_date_string=last_spc_date_string,
        raise_error_if_any_missing=False)

    option_dict = {
        trainval_io.EXAMPLE_FILES_KEY: example_file_names,
        trainval_io.NUM_EXAMPLES_PER_BATCH_KEY: NUM_EXAMPLES_PER_BATCH,
        trainval_io.FIRST_STORM_TIME_KEY:
            time_conversion.get_start_of_spc_date(first_spc_date_string),
        trainval_io.LAST_STORM_TIME_KEY:
            time_conversion.get_end_of_spc_date(last_spc_date_string),
        trainval_io.RADAR_FIELDS_KEY: radar_field_names,
        trainval_io.RADAR_HEIGHTS_KEY: numpy.array([1000], dtype=int),
        trainval_io.NUM_ROWS_KEY: num_reflectivity_rows,
        trainval_io.NUM_COLUMNS_KEY: num_reflectivity_columns,
        trainval_io.UPSAMPLE_REFLECTIVITY_KEY: False,
        trainval_io.SOUNDING_FIELDS_KEY: None,
        trainval_io.SOUNDING_HEIGHTS_KEY: None,
        trainval_io.NORMALIZATION_TYPE_KEY: None,
        trainval_io.TARGET_NAME_KEY: TARGET_NAME,
        trainval_io.BINARIZE_TARGET_KEY: False,
        trainval_io.SAMPLING_FRACTIONS_KEY: None
    }

    generator_object = testing_io.myrorss_generator_2d3d(
        option_dict=option_dict, desired_num_examples=LARGE_INTEGER)

    full_storm_id_strings = []
    storm_times_unix_sec = numpy.array([], dtype=int)
    predictor_values = numpy.array([], dtype=float)
    observed_labels = numpy.array([], dtype=int)

    while True:
        try:
            this_storm_object_dict = next(generator_object)
            print(SEPARATOR_STRING)
        except StopIteration:
            break

        full_storm_id_strings += this_storm_object_dict[testing_io.FULL_IDS_KEY]
        storm_times_unix_sec = numpy.concatenate((
            storm_times_unix_sec,
            this_storm_object_dict[testing_io.STORM_TIMES_KEY]
        ))

        this_shear_matrix_s01 = this_storm_object_dict[
            testing_io.INPUT_MATRICES_KEY][1]
        print(this_shear_matrix_s01.shape)

        these_predictor_values = numpy.max(
            this_shear_matrix_s01, axis=(1, 2, 3)
        )

        predictor_values = numpy.concatenate((
            predictor_values, these_predictor_values
        ))

        observed_labels = numpy.concatenate((
            observed_labels, this_storm_object_dict[testing_io.TARGET_ARRAY_KEY]
        ))

    forecast_probabilities = (
        rankdata(predictor_values, method='average') / len(predictor_values)
    )

    forecast_probabilities = numpy.reshape(
        forecast_probabilities, (len(forecast_probabilities), 1)
    )

    class_probability_matrix = numpy.hstack((
        1. - forecast_probabilities, forecast_probabilities
    ))

    output_file_name = prediction_io.find_ungridded_file(
        directory_name=output_dir_name, raise_error_if_missing=False)

    print('Writing results to: "{0:s}"...'.format(output_file_name))

    prediction_io.write_ungridded_predictions(
        netcdf_file_name=output_file_name,
        class_probability_matrix=class_probability_matrix,
        observed_labels=observed_labels, storm_ids=full_storm_id_strings,
        storm_times_unix_sec=storm_times_unix_sec,
        target_name=TARGET_NAME, model_file_name='None')


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        top_example_dir_name=getattr(INPUT_ARG_OBJECT, EXAMPLE_DIR_ARG_NAME),
        first_spc_date_string=getattr(INPUT_ARG_OBJECT, FIRST_DATE_ARG_NAME),
        last_spc_date_string=getattr(INPUT_ARG_OBJECT, LAST_DATE_ARG_NAME),
        use_low_level=bool(getattr(INPUT_ARG_OBJECT, LOW_LEVEL_ARG_NAME)),
        use_mid_level=bool(getattr(INPUT_ARG_OBJECT, MID_LEVEL_ARG_NAME)),
        num_radar_rows=getattr(INPUT_ARG_OBJECT, NUM_ROWS_ARG_NAME),
        num_radar_columns=getattr(INPUT_ARG_OBJECT, NUM_COLUMNS_ARG_NAME),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
