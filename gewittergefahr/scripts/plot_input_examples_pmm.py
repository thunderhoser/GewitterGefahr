"""Plots PMM composite over many examples (storm objects).

PMM = probability-matched means
"""

import pickle
import os.path
import argparse
import numpy
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.deep_learning import cnn
from gewittergefahr.deep_learning import model_interpretation
from gewittergefahr.deep_learning import training_validation_io as trainval_io
from gewittergefahr.scripts import plot_input_examples

MODEL_FILE_KEY = model_interpretation.MODEL_FILE_KEY
MEAN_INPUT_MATRICES_KEY = model_interpretation.MEAN_INPUT_MATRICES_KEY

INPUT_FILE_ARG_NAME = 'input_file_name'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'
PLOT_SOUNDINGS_ARG_NAME = plot_input_examples.PLOT_SOUNDINGS_ARG_NAME
ALLOW_WHITESPACE_ARG_NAME = plot_input_examples.ALLOW_WHITESPACE_ARG_NAME
PLOT_PANEL_NAMES_ARG_NAME = plot_input_examples.PLOT_PANEL_NAMES_ARG_NAME
ADD_TITLES_ARG_NAME = plot_input_examples.ADD_TITLES_ARG_NAME
LABEL_CBARS_ARG_NAME = plot_input_examples.LABEL_CBARS_ARG_NAME

INPUT_FILE_HELP_STRING = (
    'Path to input file, containing PMM over many examples (storm objects).  '
    'This should be a Pickle file with one dictionary, containing the keys '
    '"{0:s}" and "{1:s}".'
).format(MEAN_INPUT_MATRICES_KEY, MODEL_FILE_KEY)

OUTPUT_DIR_HELP_STRING = (
    'Name of output directory (figures will be saved here).')

PLOT_SOUNDINGS_HELP_STRING = plot_input_examples.PLOT_SOUNDINGS_HELP_STRING
ALLOW_WHITESPACE_HELP_STRING = plot_input_examples.ALLOW_WHITESPACE_HELP_STRING
PLOT_PANEL_NAMES_HELP_STRING = plot_input_examples.PLOT_PANEL_NAMES_HELP_STRING
ADD_TITLES_HELP_STRING = plot_input_examples.ADD_TITLES_HELP_STRING
LABEL_CBARS_HELP_STRING = plot_input_examples.LABEL_CBARS_HELP_STRING

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + INPUT_FILE_ARG_NAME, type=str, required=True,
    help=INPUT_FILE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + PLOT_SOUNDINGS_ARG_NAME, type=int, required=False, default=1,
    help=PLOT_SOUNDINGS_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + ALLOW_WHITESPACE_ARG_NAME, type=int, required=False, default=1,
    help=ALLOW_WHITESPACE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + PLOT_PANEL_NAMES_ARG_NAME, type=int, required=False, default=1,
    help=PLOT_PANEL_NAMES_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + ADD_TITLES_ARG_NAME, type=int, required=False, default=1,
    help=ADD_TITLES_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + LABEL_CBARS_ARG_NAME, type=int, required=False, default=0,
    help=LABEL_CBARS_HELP_STRING)


def _run(input_file_name, plot_soundings, allow_whitespace, plot_panel_names,
         add_titles, label_colour_bars, output_dir_name):
    """Plots PMM composite over many examples (storm objects).

    This is effectively the main method.

    :param input_file_name: See documentation at top of file.
    :param plot_soundings: Same.
    :param allow_whitespace: Same.
    :param plot_panel_names: Same.
    :param add_titles: Same.
    :param label_colour_bars: Same.
    :param output_dir_name: Same.
    """

    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=output_dir_name)

    print('Reading data from: "{0:s}"...'.format(input_file_name))
    pickle_file_handle = open(input_file_name, 'rb')
    input_dict = pickle.load(pickle_file_handle)
    pickle_file_handle.close()

    list_of_mean_input_matrices = input_dict[MEAN_INPUT_MATRICES_KEY]
    for i in range(len(list_of_mean_input_matrices)):
        list_of_mean_input_matrices[i] = numpy.expand_dims(
            list_of_mean_input_matrices[i], axis=0
        )

    model_file_name = input_dict[MODEL_FILE_KEY]
    model_metafile_name = '{0:s}/model_metadata.p'.format(
        os.path.split(model_file_name)[0]
    )

    print('Reading metadata from: "{0:s}"...'.format(model_metafile_name))
    model_metadata_dict = cnn.read_model_metadata(model_metafile_name)
    model_metadata_dict[cnn.TRAINING_OPTION_DICT_KEY][
        trainval_io.UPSAMPLE_REFLECTIVITY_KEY
    ] = False

    plot_input_examples.plot_examples(
        list_of_predictor_matrices=list_of_mean_input_matrices,
        model_metadata_dict=model_metadata_dict, pmm_flag=True,
        output_dir_name=output_dir_name, plot_soundings=plot_soundings,
        allow_whitespace=allow_whitespace, plot_panel_names=plot_panel_names,
        add_titles=add_titles, label_colour_bars=label_colour_bars)


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        input_file_name=getattr(INPUT_ARG_OBJECT, INPUT_FILE_ARG_NAME),
        plot_soundings=bool(getattr(INPUT_ARG_OBJECT, PLOT_SOUNDINGS_ARG_NAME)),
        allow_whitespace=bool(getattr(
            INPUT_ARG_OBJECT, ALLOW_WHITESPACE_ARG_NAME
        )),
        plot_panel_names=bool(getattr(
            INPUT_ARG_OBJECT, PLOT_PANEL_NAMES_ARG_NAME
        )),
        add_titles=bool(getattr(INPUT_ARG_OBJECT, ADD_TITLES_ARG_NAME)),
        label_colour_bars=bool(getattr(
            INPUT_ARG_OBJECT, LABEL_CBARS_ARG_NAME
        )),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
