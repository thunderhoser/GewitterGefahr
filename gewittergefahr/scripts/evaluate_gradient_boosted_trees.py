"""Evaluates ensemble of gradient-boosted trees (GBT).

Input data consists of features (output of the last "Flatten" layer) from a
convolutional neural network (CNN).
"""

import argparse
from gewittergefahr.deep_learning import cnn
from gewittergefahr.deep_learning import gradient_boosting
from gewittergefahr.scripts import model_evaluation_helper as model_eval_helper

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

DATA_FILE_ARG_NAME = 'input_data_file_name'
MODEL_FILE_ARG_NAME = 'input_model_file_name'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

DATA_FILE_HELP_STRING = (
    'Path to file with input data (readable by `cnn.read_features`).  '
    'Preferably input data for evaluation should be independent of the training'
    ' data (the best way to ensure independence is separate training and '
    'evaluation data by year), but this is not enforced by the code.')

MODEL_FILE_HELP_STRING = (
    'Path to file with trained model (readable by '
    '`gradient_boosting.read_model`).')

OUTPUT_DIR_HELP_STRING = (
    'Name of output directory.  Figures and performance metrics (in general, '
    'results of model evaluation) will be saved here.')

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + DATA_FILE_ARG_NAME, type=str, required=True,
    help=DATA_FILE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + MODEL_FILE_ARG_NAME, type=str, required=True,
    help=MODEL_FILE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING)


def _evaluate_model(
        input_data_file_name, model_file_name, output_dir_name):
    """Evaluates ensemble of gradient-boosted trees (GBT).

    :param input_data_file_name: See documentation at top of file.
    :param model_file_name: Same.
    :param output_dir_name: Same.
    :raises: ValueError: if the target variable is non-binary.  This script is
        designed for binary classification only.
    """

    print('Reading input data from: "{0:s}"...'.format(input_data_file_name))
    feature_matrix, target_values, num_classes = cnn.read_features(
        input_data_file_name)

    if num_classes > 2:
        error_string = (
            'The target variable has {0:d} classes.  This script is designed '
            'for binary classification only.'
        ).format(num_classes)

        raise ValueError(error_string)

    print('Reading model from: "{0:s}"...'.format(model_file_name))
    model_object = gradient_boosting.read_model(model_file_name)

    print('Generating predictions for all {0:d} storm objects...\n'.format(
        len(target_values)
    ))

    forecast_probabilities = gradient_boosting.apply_model(
        model_object=model_object, feature_matrix=feature_matrix)[:, -1]

    model_eval_helper.run_evaluation(
        forecast_probabilities=forecast_probabilities,
        observed_labels=target_values, num_bootstrap_reps=1,
        output_dir_name=output_dir_name)


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _evaluate_model(
        input_data_file_name=getattr(INPUT_ARG_OBJECT, DATA_FILE_ARG_NAME),
        model_file_name=getattr(INPUT_ARG_OBJECT, MODEL_FILE_ARG_NAME),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME))
