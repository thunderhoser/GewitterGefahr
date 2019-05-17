"""Plots CNN forecasts on the RAP grid."""

import argparse
import numpy
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as pyplot
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import projections
from gewittergefahr.gg_utils import nwp_model_utils
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.deep_learning import prediction_io
from gewittergefahr.plotting import plotting_utils
from gewittergefahr.plotting import probability_plotting
from gewittergefahr.plotting import imagemagick_utils

FILE_NAME_TIME_FORMAT = '%Y-%m-%d-%H%M%S'

TEST_LATITUDES_DEG = numpy.array([25.])
TEST_LONGITUDES_DEG = numpy.array([265.])
PYPROJ_OBJECT = nwp_model_utils.init_projection(nwp_model_utils.RAP_MODEL_NAME)

TITLE_FONT_SIZE = 20
BORDER_COLOUR = numpy.full(3, 0.)
GRID_LINE_COLOUR = numpy.full(3, 1.)
PARALLEL_SPACING_DEG = 5.
MERIDIAN_SPACING_DEG = 15.
FIGURE_RESOLUTION_DPI = 300

INPUT_FILE_ARG_NAME = 'input_prediction_file_name'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

INPUT_FILE_HELP_STRING = (
    'Path to input file (will be read by '
    '`prediction_io.read_gridded_predictions`).')

OUTPUT_DIR_HELP_STRING = (
    'Name of output directory.  Figures will be saved here.')

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + INPUT_FILE_ARG_NAME, type=str, required=True,
    help=INPUT_FILE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING)


def _get_projection_offsets(
        basemap_object, pyproj_object, test_latitudes_deg, test_longitudes_deg):
    """Finds offsets between basemap and pyproj projections.

    P = number of points used to find offsets

    :param basemap_object: Instance of `mpl_toolkits.basemap.Basemap`.
    :param pyproj_object: Instance of `pyproj.Proj`.  The two objects should
        encode the same projection, just with different false easting/northing.
    :param test_latitudes_deg: length-P numpy array of latitudes (deg N).
    :param test_longitudes_deg: length-P numpy array of longitudes (deg E).
    :return: x_offset_metres: x-offset (basemap minus pyproj).
    :return: y_offset_metres: y-offset (basemap minus pyproj).
    """

    pyproj_x_coords_metres, pyproj_y_coords_metres = (
        projections.project_latlng_to_xy(
            latitudes_deg=test_latitudes_deg,
            longitudes_deg=test_longitudes_deg, projection_object=pyproj_object)
    )

    basemap_x_coords_metres, basemap_y_coords_metres = basemap_object(
        test_longitudes_deg, test_latitudes_deg)

    x_offset_metres = numpy.mean(
        basemap_x_coords_metres - pyproj_x_coords_metres
    )
    y_offset_metres = numpy.mean(
        basemap_y_coords_metres - pyproj_y_coords_metres
    )

    return x_offset_metres, y_offset_metres


def _plot_forecast_one_time(gridded_forecast_dict, time_index, output_dir_name):
    """Plots gridded forecast at one time.

    :param gridded_forecast_dict: Dictionary returned by
        `prediction_io.read_gridded_predictions`.
    :param time_index: Will plot the [i]th gridded forecast, where
        i = `time_index`.
    :param output_dir_name: Name of output directory.  Figure will be saved
        here.
    """

    latlng_limit_dict = {
        plotting_utils.MIN_LATITUDE_KEY: 27.27,
        plotting_utils.MAX_LATITUDE_KEY: 43.34,
        plotting_utils.MIN_LONGITUDE_KEY: 259.48,
        plotting_utils.MAX_LONGITUDE_KEY: 282.30
    }

    axes_object, basemap_object = plotting_utils.init_map_with_nwp_projection(
        model_name=nwp_model_utils.RAP_MODEL_NAME,
        grid_name=nwp_model_utils.NAME_OF_130GRID, xy_limit_dict=None,
        latlng_limit_dict=latlng_limit_dict
    )[1:]

    x_offset_metres, y_offset_metres = _get_projection_offsets(
        basemap_object=basemap_object, pyproj_object=PYPROJ_OBJECT,
        test_latitudes_deg=TEST_LATITUDES_DEG,
        test_longitudes_deg=TEST_LONGITUDES_DEG)

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
        bottom_left_lat_deg=-90., upper_right_lat_deg=90.,
        parallel_spacing_deg=PARALLEL_SPACING_DEG, line_colour=GRID_LINE_COLOUR)

    plotting_utils.plot_meridians(
        basemap_object=basemap_object, axes_object=axes_object,
        bottom_left_lng_deg=0., upper_right_lng_deg=360.,
        meridian_spacing_deg=MERIDIAN_SPACING_DEG, line_colour=GRID_LINE_COLOUR)

    probability_matrix = gridded_forecast_dict[
        prediction_io.XY_PROBABILITIES_KEY
    ][time_index]

    # If necessary, convert from sparse to dense matrix.
    if not isinstance(probability_matrix, numpy.ndarray):
        probability_matrix = probability_matrix.toarray()

    x_coords_metres = (
        gridded_forecast_dict[prediction_io.GRID_X_COORDS_KEY] + x_offset_metres
    )
    y_coords_metres = (
        gridded_forecast_dict[prediction_io.GRID_Y_COORDS_KEY] + y_offset_metres
    )

    probability_plotting.plot_xy_grid(
        probability_matrix=probability_matrix,
        x_min_metres=numpy.min(x_coords_metres),
        y_min_metres=numpy.min(y_coords_metres),
        x_spacing_metres=numpy.diff(x_coords_metres[:2])[0],
        y_spacing_metres=numpy.diff(y_coords_metres[:2])[0],
        axes_object=axes_object, basemap_object=basemap_object)

    colour_map_object, colour_norm_object = (
        probability_plotting.get_default_colour_map()
    )

    # probability_matrix[
    #     probability_matrix < colour_norm_object.boundaries[0]
    # ] = numpy.nan
    #
    # nwp_plotting.plot_subgrid(
    #     field_matrix=probability_matrix,
    #     model_name=nwp_model_utils.RAP_MODEL_NAME,
    #     axes_object=axes_object, basemap_object=basemap_object,
    #     colour_map_object=colour_map_object,
    #     colour_norm_object=colour_norm_object,
    #     grid_id=nwp_model_utils.NAME_OF_130GRID,
    #     first_row_in_full_grid=40, first_column_in_full_grid=50, opacity=1.)

    plotting_utils.add_colour_bar(
        axes_object_or_list=axes_object, values_to_colour=probability_matrix,
        colour_map=colour_map_object, colour_norm_object=colour_norm_object,
        orientation='horizontal', extend_min=True, extend_max=True,
        fraction_of_axis_length=0.8)

    init_time_unix_sec = gridded_forecast_dict[prediction_io.INIT_TIMES_KEY][
        time_index]
    init_time_string = time_conversion.unix_sec_to_string(
        init_time_unix_sec, FILE_NAME_TIME_FORMAT)

    min_lead_time_seconds = gridded_forecast_dict[
        prediction_io.MIN_LEAD_TIME_KEY]
    max_lead_time_seconds = gridded_forecast_dict[
        prediction_io.MAX_LEAD_TIME_KEY]

    first_valid_time_unix_sec = init_time_unix_sec + min_lead_time_seconds
    first_valid_time_string = time_conversion.unix_sec_to_string(
        first_valid_time_unix_sec, FILE_NAME_TIME_FORMAT)

    last_valid_time_unix_sec = init_time_unix_sec + max_lead_time_seconds
    last_valid_time_string = time_conversion.unix_sec_to_string(
        last_valid_time_unix_sec, FILE_NAME_TIME_FORMAT)

    title_string = 'Forecast init {0:s}, valid {1:s} to {2:s}'.format(
        init_time_string, first_valid_time_string, last_valid_time_string
    )
    pyplot.title(title_string, fontsize=TITLE_FONT_SIZE)

    output_file_name = (
        '{0:s}/gridded_forecast_init-{1:s}_lead-{2:04d}-{3:04d}sec.jpg'
    ).format(
        output_dir_name, init_time_string, min_lead_time_seconds,
        max_lead_time_seconds
    )

    print 'Saving figure to: "{0:s}"...'.format(output_file_name)
    pyplot.savefig(output_file_name, dpi=FIGURE_RESOLUTION_DPI)
    pyplot.close()

    imagemagick_utils.trim_whitespace(input_file_name=output_file_name,
                                      output_file_name=output_file_name)


def _run(input_prediction_file_name, output_dir_name):
    """Plots CNN forecasts on the RAP grid.

    This is effectively the main method.

    :param input_prediction_file_name: See documentation at top of file.
    :param output_dir_name: Same.
    """

    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=output_dir_name)

    print 'Reading data from: "{0:s}"...'.format(input_prediction_file_name)
    gridded_forecast_dict = prediction_io.read_gridded_predictions(
        input_prediction_file_name)

    num_times = len(gridded_forecast_dict[prediction_io.INIT_TIMES_KEY])

    for i in range(num_times):
        _plot_forecast_one_time(
            gridded_forecast_dict=gridded_forecast_dict, time_index=i,
            output_dir_name=output_dir_name)


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        input_prediction_file_name=getattr(
            INPUT_ARG_OBJECT, INPUT_FILE_ARG_NAME),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
