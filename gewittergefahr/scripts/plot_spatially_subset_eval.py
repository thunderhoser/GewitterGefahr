"""Plots spatially subset model evaluation."""

import os.path
import warnings
import argparse
import numpy
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot
from gewittergefahr.gg_utils import grids
from gewittergefahr.gg_utils import model_evaluation as model_eval
from gewittergefahr.gg_utils import file_system_utils

INPUT_DIR_ARG_NAME = 'input_dir_name'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

INPUT_DIR_HELP_STRING = (
    'Name of input directory.  Evaluation files therein will be found by '
    '`model_evaluation.find_file` and read by '
    '`model_evaluation.read_evaluation`.')

OUTPUT_DIR_HELP_STRING = (
    'Name of output directory.  Figures will be saved here.')

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + INPUT_DIR_ARG_NAME, type=str, required=True,
    help=INPUT_DIR_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING)


def _run(evaluation_dir_name, output_dir_name):
    """Plots spatially subset model evaluation.

    This is effectively the main method.

    :param evaluation_dir_name: See documentation at top of file.
    :param output_dir_name: Same.
    """

    grid_metafile_name = grids.find_equidistant_metafile(
        directory_name=evaluation_dir_name, raise_error_if_missing=True)

    print('Reading grid metadata from: "{0:s}"...'.format(grid_metafile_name))
    grid_metadata_dict = grids.read_equidistant_metafile(grid_metafile_name)

    projection_object = grid_metadata_dict[grids.PROJECTION_KEY]
    print(projection_object)


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        evaluation_dir_name=getattr(INPUT_ARG_OBJECT, INPUT_DIR_ARG_NAME),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
