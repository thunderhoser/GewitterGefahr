"""Evaluates performance of different az-shear thresholds for tornado."""

import argparse
import numpy
from keras import backend as K
from scipy.stats import rankdata
from gewittergefahr.gg_utils import radar_utils
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.deep_learning import input_examples
from gewittergefahr.deep_learning import testing_io
from gewittergefahr.deep_learning import training_validation_io as trainval_io
from gewittergefahr.scripts import model_evaluation_helper as model_eval_helper

K.set_session(K.tf.Session(config=K.tf.ConfigProto(
    intra_op_parallelism_threads=1, inter_op_parallelism_threads=1,
    allow_soft_placement=False
)))

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

LARGE_INTEGER = int(1e10)
TARGET_NAME = 'tornado_lead-time=0000-3600sec_distance=00000-10000m'

EXAMPLE_DIR_ARG_NAME = 'input_example_dir_name'
FIRST_DATE_ARG_NAME = 'first_spc_date_string'
LAST_DATE_ARG_NAME = 'last_spc_date_string'
LOW_LEVEL_ARG_NAME = 'use_low_level'
MID_LEVEL_ARG_NAME = 'use_mid_level'
NUM_ROWS_ARG_NAME = 'num_radar_rows'
NUM_COLUMNS_ARG_NAME = 'num_radar_columns'
NUM_BOOTSTRAP_ARG_NAME = 'num_bootstrap_reps'
CONFIDENCE_LEVEL_ARG_NAME = 'confidence_level'
CLASS_FRACTION_KEYS_ARG_NAME = 'class_fraction_keys'
CLASS_FRACTION_VALUES_ARG_NAME = 'class_fraction_values'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

EXAMPLE_DIR_HELP_STRING = (
    'Name of top-level directory with input examples.  Files therein will be '
    'found by `input_examples.find_example_file` and read by '
    '`input_examples.read_example_file`.')

SPC_DATE_HELP_STRING = (
    'SPC date (format "yyyymmdd").  This script will evaluate predictions on '
    'examples from the period `{0:s}`...`{1:s}`.'
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

NUM_BOOTSTRAP_HELP_STRING = (
    'Number of bootstrap replicates.  If you do not want bootstrapping, leave '
    'this alone.')

CONFIDENCE_LEVEL_HELP_STRING = (
    '[used only if `{0:s}` > 1] Confidence level for bootstrapping, in range '
    '0...1.'
).format(NUM_BOOTSTRAP_ARG_NAME)

CLASS_FRACTION_KEYS_HELP_STRING = (
    'List of keys used to create input `class_to_sampling_fraction_dict` for '
    '`deep_learning_utils.sample_by_class`.  If you do not want class-'
    'conditional sampling, leave this alone.'
)

CLASS_FRACTION_VALUES_HELP_STRING = (
    'List of values used to create input `class_to_sampling_fraction_dict` for '
    '`deep_learning_utils.sample_by_class`.  If you do not want class-'
    'conditional sampling, leave this alone.'
)

OUTPUT_DIR_HELP_STRING = (
    'Name of output directory.  Results will be saved here.')

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
    '--' + NUM_BOOTSTRAP_ARG_NAME, type=int, required=False, default=1,
    help=NUM_BOOTSTRAP_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + CONFIDENCE_LEVEL_ARG_NAME, type=float, required=False, default=0.95,
    help=CONFIDENCE_LEVEL_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + CLASS_FRACTION_KEYS_ARG_NAME, type=int, nargs='+',
    required=False, default=[0], help=CLASS_FRACTION_KEYS_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + CLASS_FRACTION_VALUES_ARG_NAME, type=float, nargs='+',
    required=False, default=[0.], help=CLASS_FRACTION_VALUES_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING)


def _run(top_example_dir_name, first_spc_date_string, last_spc_date_string,
         use_low_level, use_mid_level, num_radar_rows, num_radar_columns,
         num_bootstrap_reps, confidence_level, class_fraction_keys,
         class_fraction_values, output_dir_name):
    """Evaluates performance of different az-shear thresholds for tornado.

    This is effectively the main method.

    :param top_example_dir_name: See documentation at top of file.
    :param first_spc_date_string: Same.
    :param last_spc_date_string: Same.
    :param use_low_level: Same.
    :param use_mid_level: Same.
    :param num_radar_rows: Same.
    :param num_radar_columns: Same.
    :param num_bootstrap_reps: Same.
    :param confidence_level: Same.
    :param class_fraction_keys: Same.
    :param class_fraction_values: Same.
    :param output_dir_name: Same.
    :raises: ValueError: if `use_low_level == use_mid_level == False`.
    """

    if num_radar_rows <= 0 or num_radar_columns <= 0:
        num_radar_rows = None
        num_radar_columns = None
    else:
        num_radar_rows = int(numpy.round(num_radar_rows / 2))
        num_radar_columns = int(numpy.round(num_radar_columns / 2))

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

    if len(class_fraction_keys) > 1:
        downsampling_dict = dict(list(zip(
            class_fraction_keys, class_fraction_values
        )))
    else:
        downsampling_dict = None

    example_file_names = input_examples.find_many_example_files(
        top_directory_name=top_example_dir_name, shuffled=False,
        first_spc_date_string=first_spc_date_string,
        last_spc_date_string=last_spc_date_string,
        raise_error_if_any_missing=False)

    option_dict = {
        trainval_io.EXAMPLE_FILES_KEY: example_file_names,
        trainval_io.FIRST_STORM_TIME_KEY:
            time_conversion.get_start_of_spc_date(first_spc_date_string),
        trainval_io.LAST_STORM_TIME_KEY:
            time_conversion.get_end_of_spc_date(last_spc_date_string),
        trainval_io.RADAR_FIELDS_KEY: radar_field_names,
        trainval_io.RADAR_HEIGHTS_KEY: numpy.array([1000], dtype=int),
        trainval_io.NUM_ROWS_KEY: num_radar_rows,
        trainval_io.NUM_COLUMNS_KEY: num_radar_columns,
        trainval_io.UPSAMPLE_REFLECTIVITY_KEY: False,
        trainval_io.SOUNDING_FIELDS_KEY: None,
        trainval_io.SOUNDING_HEIGHTS_KEY: None,
        trainval_io.NORMALIZATION_TYPE_KEY: None,
        trainval_io.TARGET_NAME_KEY: TARGET_NAME,
        trainval_io.BINARIZE_TARGET_KEY: False,
        trainval_io.SAMPLING_FRACTIONS_KEY: None
    }

    generator_object = testing_io.myrorss_generator_2d3d(
        option_dict=option_dict, num_examples_total=LARGE_INTEGER)

    predictor_values = numpy.array([], dtype=float)
    target_values = numpy.array([], dtype=int)

    for _ in range(len(example_file_names)):
        try:
            this_storm_object_dict = next(generator_object)
            print(SEPARATOR_STRING)
        except StopIteration:
            break

        this_shear_matrix_s01 = this_storm_object_dict[
            testing_io.INPUT_MATRICES_KEY][1]
        print(this_shear_matrix_s01.shape)

        these_predictor_values = numpy.squeeze(
            numpy.max(this_shear_matrix_s01, axis=(1, 2, 3))
        )

        predictor_values = numpy.concatenate((
            predictor_values, these_predictor_values
        ))

        target_values = numpy.concatenate((
            target_values, this_storm_object_dict[testing_io.TARGET_ARRAY_KEY]
        ))

    forecast_probabilities = (
        rankdata(predictor_values, method='average') / len(predictor_values)
    )

    model_eval_helper.run_evaluation(
        forecast_probabilities=forecast_probabilities,
        observed_labels=target_values, downsampling_dict=downsampling_dict,
        num_bootstrap_reps=num_bootstrap_reps,
        confidence_level=confidence_level, output_dir_name=output_dir_name)


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
        num_bootstrap_reps=getattr(INPUT_ARG_OBJECT, NUM_BOOTSTRAP_ARG_NAME),
        confidence_level=getattr(INPUT_ARG_OBJECT, CONFIDENCE_LEVEL_ARG_NAME),
        class_fraction_keys=numpy.array(
            getattr(INPUT_ARG_OBJECT, CLASS_FRACTION_KEYS_ARG_NAME), dtype=int
        ),
        class_fraction_values=numpy.array(
            getattr(INPUT_ARG_OBJECT, CLASS_FRACTION_VALUES_ARG_NAME),
            dtype=float
        ),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
