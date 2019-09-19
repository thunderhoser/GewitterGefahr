"""Plots spatially subset model evaluation."""

import os.path
import warnings
import argparse
import numpy
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot
from mpl_toolkits.basemap import Basemap
from gewittergefahr.gg_utils import grids
from gewittergefahr.gg_utils import projections
from gewittergefahr.gg_utils import model_evaluation as model_eval
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.gg_utils import error_checking
from gewittergefahr.plotting import plotting_utils

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

LAMBERT_CONFORMAL_STRING = 'lcc'

NUM_PARALLELS = 8
NUM_MERIDIANS = 6
RESOLUTION_STRING = 'l'
BORDER_COLOUR = numpy.full(3, 0.)

FIGURE_RESOLUTION_DPI = 300
FIGURE_WIDTH_INCHES = 15
FIGURE_HEIGHT_INCHES = 15

INPUT_DIR_ARG_NAME = 'input_dir_name'
COLOUR_MAP_ARG_NAME = 'colour_map_name'
MAX_PERCENTILE_ARG_NAME = 'max_colour_percentile'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

INPUT_DIR_HELP_STRING = (
    'Name of input directory.  Evaluation files therein will be found by '
    '`model_evaluation.find_file` and read by '
    '`model_evaluation.read_evaluation`.')

COLOUR_MAP_HELP_STRING = (
    'Name of colour map (must be accepted by `pyplot.get_cmap`.  Will be used '
    'to plot all scores.')

MAX_PERCENTILE_HELP_STRING = (
    'Used to determine min and max values in each colour map.  Max value will '
    'be [q]th percentile of all scores, and min value will be [100 - q]th '
    'percentile, where q = `{0:s}`.'
).format(MAX_PERCENTILE_ARG_NAME)

OUTPUT_DIR_HELP_STRING = (
    'Name of output directory.  Figures will be saved here.')

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + INPUT_DIR_ARG_NAME, type=str, required=True,
    help=INPUT_DIR_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + COLOUR_MAP_ARG_NAME, type=str, required=False, default='plasma',
    help=COLOUR_MAP_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + MAX_PERCENTILE_ARG_NAME, type=float, required=False, default=99.,
    help=MAX_PERCENTILE_HELP_STRING)

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


def _get_basemap(grid_metadata_dict):
    """Creates basemap.

    M = number of rows in grid
    M = number of columns in grid

    :param grid_metadata_dict: Dictionary returned by
        `grids.read_equidistant_metafile`.
    :return: basemap_object: Basemap handle (instance of
        `mpl_toolkits.basemap.Basemap`).
    :return: basemap_x_matrix_metres: M-by-N numpy array of x-coordinates under
        Basemap projection (different than pyproj projection).
    :return: basemap_y_matrix_metres: Same but for y-coordinates.
    """

    x_matrix_metres, y_matrix_metres = grids.xy_vectors_to_matrices(
        x_unique_metres=grid_metadata_dict[grids.X_COORDS_KEY],
        y_unique_metres=grid_metadata_dict[grids.Y_COORDS_KEY]
    )

    projection_object = grid_metadata_dict[grids.PROJECTION_KEY]

    latitude_matrix_deg, longitude_matrix_deg = (
        projections.project_xy_to_latlng(
            x_coords_metres=x_matrix_metres, y_coords_metres=y_matrix_metres,
            projection_object=projection_object)
    )

    standard_latitudes_deg, central_longitude_deg = _get_lcc_params(
        projection_object)

    basemap_object = Basemap(
        projection='lcc', lat_1=standard_latitudes_deg[0],
        lat_2=standard_latitudes_deg[1], lon_0=central_longitude_deg,
        rsphere=projections.DEFAULT_EARTH_RADIUS_METRES,
        ellps=projections.SPHERE_NAME, resolution=RESOLUTION_STRING,
        llcrnrx=x_matrix_metres[0, 0], llcrnry=y_matrix_metres[0, 0],
        urcrnrx=x_matrix_metres[-1, -1], urcrnry=y_matrix_metres[-1, -1]
    )

    basemap_x_matrix_metres, basemap_y_matrix_metres = basemap_object(
        longitude_matrix_deg, latitude_matrix_deg)

    return basemap_object, basemap_x_matrix_metres, basemap_y_matrix_metres


def _plot_one_score(
        score_matrix, grid_metadata_dict, colour_map_object, min_colour_value,
        max_colour_value):
    """Plots one score.

    M = number of rows in grid
    N = number of columns in grid

    :param score_matrix: M-by-N numpy array with values of a given score.
    :param grid_metadata_dict: Dictionary returned by
        `grids.read_equidistant_metafile`.
    :param colour_map_object: See documentation at top of file.
    :param min_colour_value: Minimum value in colour scheme.
    :param max_colour_value: Max value in colour scheme.
    :return: figure_object: Figure handle (instance of
        `matplotlib.figure.Figure`).
    :return: axes_object: Axes handle (instance of
        `matplotlib.axes._subplots.AxesSubplot`).
    """

    figure_object, axes_object = pyplot.subplots(
        1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
    )

    basemap_object, basemap_x_matrix_metres, basemap_y_matrix_metres = (
        _get_basemap(grid_metadata_dict)
    )

    num_grid_rows = score_matrix.shape[0]
    num_grid_columns = score_matrix.shape[1]
    x_spacing_metres = (
        (basemap_x_matrix_metres[0, -1] - basemap_x_matrix_metres[0, 0]) /
        (num_grid_columns - 1)
    )
    y_spacing_metres = (
        (basemap_y_matrix_metres[-1, 0] - basemap_y_matrix_metres[0, 0]) /
        (num_grid_rows - 1)
    )

    score_matrix_at_edges, edge_x_coords_metres, edge_y_coords_metres = (
        grids.xy_field_grid_points_to_edges(
            field_matrix=score_matrix,
            x_min_metres=basemap_x_matrix_metres[0, 0],
            y_min_metres=basemap_y_matrix_metres[0, 0],
            x_spacing_metres=x_spacing_metres,
            y_spacing_metres=y_spacing_metres)
    )

    # score_matrix_at_edges = numpy.ma.masked_where(
    #     numpy.isnan(score_matrix_at_edges), score_matrix_at_edges
    # )

    score_matrix_at_edges[numpy.isnan(score_matrix_at_edges)] = -1

    plotting_utils.plot_coastlines(
        basemap_object=basemap_object, axes_object=axes_object,
        line_colour=BORDER_COLOUR)

    plotting_utils.plot_countries(
        basemap_object=basemap_object, axes_object=axes_object,
        line_colour=BORDER_COLOUR)

    plotting_utils.plot_states_and_provinces(
        basemap_object=basemap_object, axes_object=axes_object,
        line_colour=BORDER_COLOUR)

    plotting_utils.plot_parallels(
        basemap_object=basemap_object, axes_object=axes_object,
        num_parallels=NUM_PARALLELS)

    plotting_utils.plot_meridians(
        basemap_object=basemap_object, axes_object=axes_object,
        num_meridians=NUM_MERIDIANS)

    basemap_object.pcolormesh(
        edge_x_coords_metres, edge_y_coords_metres,
        score_matrix_at_edges, cmap=colour_map_object,
        vmin=min_colour_value, vmax=max_colour_value, shading='flat',
        edgecolors='None', axes=axes_object, zorder=-1e12)

    plotting_utils.plot_linear_colour_bar(
        axes_object_or_matrix=axes_object, data_matrix=score_matrix,
        colour_map_object=colour_map_object, min_value=min_colour_value,
        max_value=max_colour_value, orientation_string='horizontal',
        extend_min=True, extend_max=True)

    return figure_object, axes_object


def _run(evaluation_dir_name, colour_map_name, max_colour_percentile,
         output_dir_name):
    """Plots spatially subset model evaluation.

    This is effectively the main method.

    :param evaluation_dir_name: See documentation at top of file.
    :param colour_map_name: Same.
    :param max_colour_percentile: Same.
    :param output_dir_name: Same.
    """

    colour_map_object = pyplot.get_cmap(colour_map_name)
    error_checking.assert_is_geq(max_colour_percentile, 90.)
    error_checking.assert_is_leq(max_colour_percentile, 100.)

    grid_metafile_name = grids.find_equidistant_metafile(
        directory_name=evaluation_dir_name, raise_error_if_missing=True)

    print('Reading grid metadata from: "{0:s}"...'.format(grid_metafile_name))
    grid_metadata_dict = grids.read_equidistant_metafile(grid_metafile_name)
    print(SEPARATOR_STRING)

    num_grid_rows = len(grid_metadata_dict[grids.Y_COORDS_KEY])
    num_grid_columns = len(grid_metadata_dict[grids.X_COORDS_KEY])

    auc_matrix = numpy.full((num_grid_rows, num_grid_columns), numpy.nan)
    csi_matrix = numpy.full((num_grid_rows, num_grid_columns), numpy.nan)
    pod_matrix = numpy.full((num_grid_rows, num_grid_columns), numpy.nan)
    far_matrix = numpy.full((num_grid_rows, num_grid_columns), numpy.nan)
    num_examples_matrix = numpy.full(
        (num_grid_rows, num_grid_columns), 0, dtype=int
    )
    num_positive_examples_matrix = numpy.full(
        (num_grid_rows, num_grid_columns), 0, dtype=int
    )

    for i in range(num_grid_rows):
        for j in range(num_grid_columns):
            this_eval_file_name = model_eval.find_file(
                directory_name=evaluation_dir_name, grid_row=i, grid_column=j,
                raise_error_if_missing=False)

            if not os.path.isfile(this_eval_file_name):
                warning_string = (
                    'Cannot find file (this may or may not be a problem).  '
                    'Expected at: "{0:s}"'
                ).format(this_eval_file_name)

                warnings.warn(warning_string)
                continue

            print('Reading data from: "{0:s}"...'.format(this_eval_file_name))
            this_evaluation_dict = model_eval.read_evaluation(
                this_eval_file_name)

            num_examples_matrix[i, j] = len(
                this_evaluation_dict[model_eval.OBSERVED_LABELS_KEY]
            )
            num_positive_examples_matrix[i, j] = numpy.sum(
                this_evaluation_dict[model_eval.OBSERVED_LABELS_KEY]
            )

            this_evaluation_table = this_evaluation_dict[
                model_eval.EVALUATION_TABLE_KEY]

            auc_matrix[i, j] = numpy.nanmean(
                this_evaluation_table[model_eval.AUC_KEY].values
            )
            csi_matrix[i, j] = numpy.nanmean(
                this_evaluation_table[model_eval.CSI_KEY].values
            )
            pod_matrix[i, j] = numpy.nanmean(
                this_evaluation_table[model_eval.POD_KEY].values
            )
            far_matrix[i, j] = 1. - numpy.nanmean(
                this_evaluation_table[model_eval.SUCCESS_RATIO_KEY].values
            )

    print(SEPARATOR_STRING)
    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=output_dir_name)

    # Plot AUC.
    max_colour_value = numpy.nanpercentile(auc_matrix, max_colour_percentile)
    min_colour_value = numpy.maximum(
        numpy.nanpercentile(auc_matrix, 100. - max_colour_percentile),
        0.5
    )

    figure_object, axes_object = _plot_one_score(
        score_matrix=auc_matrix, grid_metadata_dict=grid_metadata_dict,
        colour_map_object=colour_map_object,
        min_colour_value=min_colour_value, max_colour_value=max_colour_value)

    axes_object.set_title('AUC (area under ROC curve)')
    output_file_name = '{0:s}/spatial_auc_plot.jpg'.format(output_dir_name)
    print('Saving figure to: "{0:s}"...'.format(output_file_name))

    figure_object.savefig(
        output_file_name, dpi=FIGURE_RESOLUTION_DPI, pad_inches=0,
        bbox_inches='tight')
    pyplot.close(figure_object)

    # Plot CSI.
    max_colour_value = numpy.nanpercentile(csi_matrix, max_colour_percentile)
    min_colour_value = numpy.nanpercentile(
        csi_matrix, 100. - max_colour_percentile)

    figure_object, axes_object = _plot_one_score(
        score_matrix=csi_matrix, grid_metadata_dict=grid_metadata_dict,
        colour_map_object=colour_map_object,
        min_colour_value=min_colour_value, max_colour_value=max_colour_value)

    axes_object.set_title('CSI (critical success index)')
    output_file_name = '{0:s}/spatial_csi_plot.jpg'.format(output_dir_name)
    print('Saving figure to: "{0:s}"...'.format(output_file_name))

    figure_object.savefig(
        output_file_name, dpi=FIGURE_RESOLUTION_DPI, pad_inches=0,
        bbox_inches='tight')
    pyplot.close(figure_object)

    # Plot POD.
    max_colour_value = numpy.nanpercentile(pod_matrix, max_colour_percentile)
    min_colour_value = numpy.nanpercentile(
        pod_matrix, 100. - max_colour_percentile)

    figure_object, axes_object = _plot_one_score(
        score_matrix=pod_matrix, grid_metadata_dict=grid_metadata_dict,
        colour_map_object=colour_map_object,
        min_colour_value=min_colour_value, max_colour_value=max_colour_value)

    axes_object.set_title('POD (probability of detection)')
    output_file_name = '{0:s}/spatial_pod_plot.jpg'.format(output_dir_name)
    print('Saving figure to: "{0:s}"...'.format(output_file_name))

    figure_object.savefig(
        output_file_name, dpi=FIGURE_RESOLUTION_DPI, pad_inches=0,
        bbox_inches='tight')
    pyplot.close(figure_object)

    # Plot FAR.
    max_colour_value = numpy.nanpercentile(far_matrix, max_colour_percentile)
    min_colour_value = numpy.nanpercentile(
        far_matrix, 100. - max_colour_percentile)

    figure_object, axes_object = _plot_one_score(
        score_matrix=far_matrix, grid_metadata_dict=grid_metadata_dict,
        colour_map_object=colour_map_object,
        min_colour_value=min_colour_value, max_colour_value=max_colour_value)

    axes_object.set_title('FAR (false-alarm rate)')
    output_file_name = '{0:s}/spatial_far_plot.jpg'.format(output_dir_name)
    print('Saving figure to: "{0:s}"...'.format(output_file_name))

    figure_object.savefig(
        output_file_name, dpi=FIGURE_RESOLUTION_DPI, pad_inches=0,
        bbox_inches='tight')
    pyplot.close(figure_object)


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        evaluation_dir_name=getattr(INPUT_ARG_OBJECT, INPUT_DIR_ARG_NAME),
        colour_map_name=getattr(INPUT_ARG_OBJECT, COLOUR_MAP_ARG_NAME),
        max_colour_percentile=getattr(
            INPUT_ARG_OBJECT, MAX_PERCENTILE_ARG_NAME),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
