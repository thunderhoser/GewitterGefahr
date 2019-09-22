"""Combines evaluation files.

Each file should contain model evaluation for different bootstrap replicates on
the same problem.
"""

import os.path
import argparse
from gewittergefahr.gg_utils import model_evaluation as model_eval

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

INPUT_FILES_ARG_NAME = 'input_file_names'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

INPUT_FILES_HELP_STRING = (
    'List of paths to input files (each readable by '
    '`model_evaluation.read_file`).')

OUTPUT_DIR_HELP_STRING = (
    'Name of output directory.  Combined file will be written here by '
    '`model_evaluation.write_file`, to an exact location determined by '
    '`model_evaluation.find_file`.')

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + INPUT_FILES_ARG_NAME, type=str, nargs='+', required=True,
    help=INPUT_FILES_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING)


def _run(input_file_names, output_dir_name):
    """Combines evaluation files.

    This is effectively the main method.

    :param input_file_names: See documentation at top of file.
    :param output_dir_name: Same.
    """

    evaluation_dict = model_eval.combine_evaluation_files(input_file_names)
    print(SEPARATOR_STRING)

    pathless_file_name = os.path.split(input_file_names[0])[-1]
    output_file_name = '{0:s}/{1:s}'.format(output_dir_name, pathless_file_name)

    print('Writing combined file to: "{0:s}"...'.format(output_file_name))
    model_eval.write_evaluation(
        pickle_file_name=output_file_name,
        forecast_probabilities=evaluation_dict[
            model_eval.FORECAST_PROBABILITIES_KEY],
        observed_labels=evaluation_dict[model_eval.OBSERVED_LABELS_KEY],
        best_prob_threshold=evaluation_dict[model_eval.BEST_THRESHOLD_KEY],
        all_prob_thresholds=evaluation_dict[model_eval.ALL_THRESHOLDS_KEY],
        num_examples_by_forecast_bin=evaluation_dict[
            model_eval.NUM_EXAMPLES_BY_BIN_KEY],
        downsampling_dict=evaluation_dict[model_eval.DOWNSAMPLING_DICT_KEY],
        evaluation_table=evaluation_dict[model_eval.EVALUATION_TABLE_KEY]
    )


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        input_file_names=getattr(INPUT_ARG_OBJECT, INPUT_FILES_ARG_NAME),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
