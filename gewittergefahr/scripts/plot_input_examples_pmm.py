"""Plots probability-matched mean (PMM) of many examples (storm objects)."""

import pickle
import os.path
import argparse
from gewittergefahr.deep_learning import cnn
from gewittergefahr.deep_learning import model_interpretation
from gewittergefahr.scripts import plot_input_examples

MODEL_FILE_KEY = model_interpretation.MODEL_FILE_KEY
MEAN_INPUT_MATRICES_KEY = model_interpretation.MEAN_INPUT_MATRICES_KEY

INPUT_FILE_ARG_NAME = 'input_file_name'
SAVE_PANELED_ARG_NAME = 'save_paneled_figs'
PLOT_SOUNDINGS_ARG_NAME = 'plot_soundings'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

INPUT_FILE_HELP_STRING = (
    'Path to input file, containing PMM over many examples (storm objects).  '
    'This should be a Pickle file with one dictionary, containing the keys '
    '"{0:s}" and "{1:s}".'
).format(MEAN_INPUT_MATRICES_KEY, MODEL_FILE_KEY)

SAVE_PANELED_HELP_STRING = (
    'Boolean flag.  If 1, will save paneled figures, leading to only a few '
    'figures.  If 0, will save each 2-D or 1-D field as one figure.')

PLOT_SOUNDINGS_HELP_STRING = 'Boolean flag.  If 1, will plot sounding.'

OUTPUT_DIR_HELP_STRING = (
    'Name of output directory (figures will be saved here).')

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + INPUT_FILE_ARG_NAME, type=str, required=True,
    help=INPUT_FILE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + SAVE_PANELED_ARG_NAME, type=int, required=False, default=0,
    help=SAVE_PANELED_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + PLOT_SOUNDINGS_ARG_NAME, type=int, required=False, default=1,
    help=PLOT_SOUNDINGS_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING)


def _run(input_file_name, save_paneled_figs, plot_soundings, output_dir_name):
    """Plots probability-matched mean (PMM) of many examples (storm objects).

    This is effectively the main method.

    :param input_file_name: See documentation at top of file.
    :param save_paneled_figs: Same.
    :param plot_soundings: Same.
    :param output_dir_name: Same.
    """

    print('Reading data from: "{0:s}"...'.format(input_file_name))

    pickle_file_handle = open(input_file_name, 'rb')
    input_dict = pickle.load(pickle_file_handle)
    pickle_file_handle.close()

    list_of_mean_input_matrices = input_dict[MEAN_INPUT_MATRICES_KEY]
    model_file_name = input_dict[MODEL_FILE_KEY]

    model_metafile_name = '{0:s}/model_metadata.p'.format(
        os.path.split(model_file_name)[0]
    )

    print('Reading metadata from: "{0:s}"...'.format(model_metafile_name))
    model_metadata_dict = cnn.read_model_metadata(model_metafile_name)

    plot_input_examples.plot_examples(
        list_of_predictor_matrices=list_of_mean_input_matrices,
        model_metadata_dict=model_metadata_dict,
        save_paneled_figs=save_paneled_figs,
        output_dir_name=output_dir_name, pmm_flag=True)


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        input_file_name=getattr(INPUT_ARG_OBJECT, INPUT_FILE_ARG_NAME),
        save_paneled_figs=bool(getattr(
            INPUT_ARG_OBJECT, SAVE_PANELED_ARG_NAME
        )),
        plot_soundings=bool(getattr(INPUT_ARG_OBJECT, PLOT_SOUNDINGS_ARG_NAME)),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
