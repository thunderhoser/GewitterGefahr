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

LAMBERT_CONFORMAL_STRING = 'lcc'

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


def _get_lcc_params(projection_object):
    """Finds parameters for LCC (Lambert conformal conic) projection.

    :param projection_object: Instance of `pyproj.Proj`.
    :return: standard_latitudes_deg: length-2 numpy array of standard latitudes
        (deg N).
    :return: central_longitude_deg: Central longitude (deg E).
    :raises: ValueError: if projection is not LCC.
    """

    projection_string = projection_object.srs
    words = projection_string.split()

    property_names = [w.split('=')[0][1:] for w in words]
    property_values = [w.split('=')[1] for w in words]
    projection_dict = dict(list(
        zip(property_names, property_values)
    ))

    if projection_dict['proj'] != LAMBERT_CONFORMAL_STRING:
        error_string = 'Grid projection should be "{0:s}", not "{1:s}".'.format(
            LAMBERT_CONFORMAL_STRING, projection_dict['proj']
        )

        raise ValueError(error_string)

    central_longitude_deg = float(projection_dict['lon_0'])
    standard_latitudes_deg = numpy.array([
        float(projection_dict['lat_1']), float(projection_dict['lat_2'])
    ])

    return standard_latitudes_deg, central_longitude_deg


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

    standard_latitudes_deg, central_longitude_deg = _get_lcc_params(
        grid_metadata_dict[grids.PROJECTION_KEY]
    )


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        evaluation_dir_name=getattr(INPUT_ARG_OBJECT, INPUT_DIR_ARG_NAME),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
