"""Evaluates CNN predictions."""

import argparse
import numpy
from gewittergefahr.gg_utils import model_evaluation as model_eval
from gewittergefahr.deep_learning import prediction_io
from gewittergefahr.scripts import model_evaluation_helper as model_eval_helper

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

PREDICTION_FILE_ARG_NAME = 'input_prediction_file_name'
THRESHOLD_ARG_NAME = 'binarization_threshold'
NUM_BOOTSTRAP_ARG_NAME = 'num_bootstrap_reps'
CONFIDENCE_LEVEL_ARG_NAME = 'confidence_level'
CLASS_FRACTION_KEYS_ARG_NAME = 'class_fraction_keys'
CLASS_FRACTION_VALUES_ARG_NAME = 'class_fraction_values'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

PREDICTION_FILE_HELP_STRING = (
    'Path to input file, containing CNN predictions.  Will be read by '
    '`prediction_io.read_ungridded_predictions`.')

THRESHOLD_HELP_STRING = (
    'Binarization threshold (used to turn probabilities into deterministic '
    'predictions).  If you make this negative, will use threshold that yields '
    'the best CSI.')

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
    '--' + PREDICTION_FILE_ARG_NAME, type=str, required=True,
    help=PREDICTION_FILE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + THRESHOLD_ARG_NAME, type=float, required=False, default=-1.,
    help=THRESHOLD_HELP_STRING)

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


def _run(prediction_file_name, binarization_threshold, num_bootstrap_reps,
         confidence_level, class_fraction_keys, class_fraction_values,
         output_dir_name):
    """Evaluates CNN predictions.

    This is effectively the main method.

    :param prediction_file_name: Same.
    :param binarization_threshold: Same.
    :param num_bootstrap_reps: Same.
    :param confidence_level: Same.
    :param class_fraction_keys: Same.
    :param class_fraction_values: Same.
    :param output_dir_name: Same.
    :raises: ValueError: if file contains multi-class predictions.
    """

    if binarization_threshold < 0:
        binarization_threshold = None

    print('Reading data from: "{0:s}"...'.format(prediction_file_name))
    prediction_dict = prediction_io.read_ungridded_predictions(
        prediction_file_name)

    observed_labels = prediction_dict[prediction_io.OBSERVED_LABELS_KEY]
    class_probability_matrix = prediction_dict[
        prediction_io.PROBABILITY_MATRIX_KEY]

    num_classes = class_probability_matrix.shape[1]
    if num_classes > 2:
        error_string = (
            'This script handles only binary, not {0:d}-class, classification.'
        ).format(num_classes)

        raise ValueError(error_string)

    forecast_probabilities = class_probability_matrix[:, -1]

    if len(class_fraction_keys) > 1:
        downsampling_dict = dict(list(zip(
            class_fraction_keys, class_fraction_values
        )))
    else:
        downsampling_dict = None

    main_output_file_name = model_eval.find_file(
        input_prediction_file_name=prediction_file_name,
        output_dir_name=output_dir_name, raise_error_if_missing=False)

    model_eval_helper.run_evaluation(
        forecast_probabilities=forecast_probabilities,
        observed_labels=observed_labels, downsampling_dict=downsampling_dict,
        best_prob_threshold=binarization_threshold,
        num_bootstrap_reps=num_bootstrap_reps,
        confidence_level=confidence_level,
        main_output_file_name=main_output_file_name)


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        prediction_file_name=getattr(
            INPUT_ARG_OBJECT, PREDICTION_FILE_ARG_NAME),
        binarization_threshold=getattr(INPUT_ARG_OBJECT, THRESHOLD_ARG_NAME),
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
