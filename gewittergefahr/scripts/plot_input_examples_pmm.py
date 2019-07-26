"""Plots probability-matched mean (PMM) of many examples (storm objects)."""

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
BAMS_FORMAT_ARG_NAME = 'bams_format'
ALLOW_WHITESPACE_ARG_NAME = 'allow_whitespace'
PLOT_SOUNDINGS_ARG_NAME = 'plot_soundings'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

INPUT_FILE_HELP_STRING = (
    'Path to input file, containing PMM over many examples (storm objects).  '
    'This should be a Pickle file with one dictionary, containing the keys '
    '"{0:s}" and "{1:s}".'
).format(MEAN_INPUT_MATRICES_KEY, MODEL_FILE_KEY)

BAMS_FORMAT_HELP_STRING = (
    'Boolean flag.  If 1, figures will be plotted in the format used for BAMS '
    '2019.  If you do not know what this means, just leave the argument alone.')

ALLOW_WHITESPACE_HELP_STRING = (
    'Boolean flag.  If 0, will plot with no whitespace between panels or around'
    ' outside of image.')

PLOT_SOUNDINGS_HELP_STRING = 'Boolean flag.  If 1, will plot sounding.'

OUTPUT_DIR_HELP_STRING = (
    'Name of output directory (figures will be saved here).')

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + INPUT_FILE_ARG_NAME, type=str, required=True,
    help=INPUT_FILE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + BAMS_FORMAT_ARG_NAME, type=int, required=False, default=0,
    help=BAMS_FORMAT_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + ALLOW_WHITESPACE_ARG_NAME, type=int, required=False, default=1,
    help=ALLOW_WHITESPACE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + PLOT_SOUNDINGS_ARG_NAME, type=int, required=False, default=1,
    help=PLOT_SOUNDINGS_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING)


def _run(input_file_name, bams_format, allow_whitespace, plot_soundings,
         output_dir_name):
    """Plots probability-matched mean (PMM) of many examples (storm objects).

    This is effectively the main method.

    :param input_file_name: See documentation at top of file.
    :param bams_format: Same.
    :param allow_whitespace: Same.
    :param plot_soundings: Same.
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
        model_metadata_dict=model_metadata_dict,
        output_dir_name=output_dir_name, plot_soundings=plot_soundings,
        bams_format=bams_format, allow_whitespace=allow_whitespace,
        pmm_flag=True)


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        input_file_name=getattr(INPUT_ARG_OBJECT, INPUT_FILE_ARG_NAME),
        bams_format=bool(getattr(INPUT_ARG_OBJECT, BAMS_FORMAT_ARG_NAME)),
        allow_whitespace=bool(getattr(
            INPUT_ARG_OBJECT, ALLOW_WHITESPACE_ARG_NAME
        )),
        plot_soundings=bool(getattr(INPUT_ARG_OBJECT, PLOT_SOUNDINGS_ARG_NAME)),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
