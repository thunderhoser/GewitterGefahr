"""Plots CNN forecasts on the RAP grid."""

import argparse
import numpy
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as pyplot
from gewittergefahr.gg_io import tornado_io
from gewittergefahr.gg_utils import number_rounding
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import projections
from gewittergefahr.gg_utils import nwp_model_utils
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.deep_learning import prediction_io
from gewittergefahr.plotting import plotting_utils
from gewittergefahr.plotting import probability_plotting
from gewittergefahr.plotting import imagemagick_utils

SENTINEL_VALUE = -9999
FILE_NAME_TIME_FORMAT = '%Y-%m-%d-%H%M%S'

TEST_LATITUDES_DEG = numpy.array([25.])
TEST_LONGITUDES_DEG = numpy.array([265.])
PYPROJ_OBJECT = nwp_model_utils.init_projection(nwp_model_utils.RAP_MODEL_NAME)

TORNADO_MARKER_TYPE = '^'
TORNADO_MARKER_SIZE = 8
TORNADO_MARKER_EDGE_WIDTH = 1
TORNADO_MARKER_COLOUR = numpy.array([217, 95, 2], dtype=float) / 255
# TORNADO_MARKER_COLOUR = numpy.array([228, 26, 28], dtype=float) / 255

NUM_PARALLELS = 8
NUM_MERIDIANS = 6
BORDER_COLOUR = numpy.full(3, 0.)
GRID_LINE_COLOUR = numpy.full(3, 1.)

TITLE_FONT_SIZE = 20
FIGURE_RESOLUTION_DPI = 300

INPUT_FILE_ARG_NAME = 'input_prediction_file_name'
TORNADO_DIR_ARG_NAME = 'input_tornado_dir_name'
MIN_LATITUDE_ARG_NAME = 'min_plot_latitude_deg'
MAX_LATITUDE_ARG_NAME = 'max_plot_latitude_deg'
MIN_LONGITUDE_ARG_NAME = 'min_plot_longitude_deg'
MAX_LONGITUDE_ARG_NAME = 'max_plot_longitude_deg'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

INPUT_FILE_HELP_STRING = (
    'Path to input file (will be read by '
    '`prediction_io.read_gridded_predictions`).')

TORNADO_DIR_HELP_STRING = (
    'Name of directory with tornado reports.  Files therein will be found by '
    '`tornado_io.find_processed_file` and read by '
    '`tornado_io.read_processed_file`.  If you do not want to plot tornado '
    'reports, make this an empty string.')

LATITUDE_HELP_STRING = (
    'Latitude (deg N, in range -90...90).  Plotting area will be '
    '`{0:s}`...`{1:s}`.  To let plotting area be determined by data, make this '
    '{2:d}.'
).format(MIN_LATITUDE_ARG_NAME, MAX_LATITUDE_ARG_NAME, SENTINEL_VALUE)

LONGITUDE_HELP_STRING = (
    'Longitude (deg E, in range 0...360).  Plotting area will be '
    '`{0:s}`...`{1:s}`.  To let plotting area be determined by data, make this '
    '{2:d}.'
).format(MIN_LONGITUDE_ARG_NAME, MAX_LONGITUDE_ARG_NAME, SENTINEL_VALUE)

OUTPUT_DIR_HELP_STRING = (
    'Name of output directory.  Figures will be saved here.')

DEFAULT_TORNADO_DIR_NAME = (
    '/condo/swatwork/ralager/tornado_observations/processed')

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + INPUT_FILE_ARG_NAME, type=str, required=True,
    help=INPUT_FILE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + TORNADO_DIR_ARG_NAME, type=str, required=False,
    default=DEFAULT_TORNADO_DIR_NAME, help=TORNADO_DIR_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + MIN_LATITUDE_ARG_NAME, type=float, required=False,
    default=SENTINEL_VALUE, help=LATITUDE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + MAX_LATITUDE_ARG_NAME, type=float, required=False,
    default=SENTINEL_VALUE, help=LATITUDE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + MIN_LONGITUDE_ARG_NAME, type=float, required=False,
    default=SENTINEL_VALUE, help=LONGITUDE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + MAX_LONGITUDE_ARG_NAME, type=float, required=False,
    default=SENTINEL_VALUE, help=LONGITUDE_HELP_STRING)

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


def _plot_forecast_one_time(
        gridded_forecast_dict, time_index, min_plot_latitude_deg,
        max_plot_latitude_deg, min_plot_longitude_deg, max_plot_longitude_deg,
        output_dir_name, tornado_dir_name=None):
    """Plots gridded forecast at one time.

    :param gridded_forecast_dict: Dictionary returned by
        `prediction_io.read_gridded_predictions`.
    :param time_index: Will plot the [i]th gridded forecast, where
        i = `time_index`.
    :param min_plot_latitude_deg: See documentation at top of file.
    :param max_plot_latitude_deg: Same.
    :param min_plot_longitude_deg: Same.
    :param max_plot_longitude_deg: Same.
    :param output_dir_name: Name of output directory.  Figure will be saved
        here.
    :param tornado_dir_name: See documentation at top of file.
    """

    init_time_unix_sec = gridded_forecast_dict[prediction_io.INIT_TIMES_KEY][
        time_index
    ]
    min_lead_time_seconds = gridded_forecast_dict[
        prediction_io.MIN_LEAD_TIME_KEY
    ]
    max_lead_time_seconds = gridded_forecast_dict[
        prediction_io.MAX_LEAD_TIME_KEY
    ]

    first_valid_time_unix_sec = init_time_unix_sec + min_lead_time_seconds
    last_valid_time_unix_sec = init_time_unix_sec + max_lead_time_seconds

    tornado_latitudes_deg = numpy.array([])
    tornado_longitudes_deg = numpy.array([])

    if tornado_dir_name is not None:
        first_year = int(
            time_conversion.unix_sec_to_string(first_valid_time_unix_sec, '%Y')
        )
        last_year = int(
            time_conversion.unix_sec_to_string(last_valid_time_unix_sec, '%Y')
        )

        for this_year in range(first_year, last_year + 1):
            this_file_name = tornado_io.find_processed_file(
                directory_name=tornado_dir_name, year=this_year)

            print 'Reading tornado reports from: "{0:s}"...'.format(
                this_file_name)
            this_tornado_table = tornado_io.read_processed_file(this_file_name)

            this_tornado_table = this_tornado_table.loc[
                (this_tornado_table[tornado_io.START_TIME_COLUMN]
                 >= first_valid_time_unix_sec)
                & (this_tornado_table[tornado_io.START_TIME_COLUMN]
                   <= last_valid_time_unix_sec)
                ]

            tornado_latitudes_deg = numpy.concatenate((
                tornado_latitudes_deg,
                this_tornado_table[tornado_io.START_LAT_COLUMN].values
            ))

            tornado_longitudes_deg = numpy.concatenate((
                tornado_longitudes_deg,
                this_tornado_table[tornado_io.START_LNG_COLUMN].values
            ))

        print '\n'

    custom_area = all([
        x is not None for x in
        [min_plot_latitude_deg, max_plot_latitude_deg, min_plot_longitude_deg,
         max_plot_longitude_deg]
    ])

    if custom_area:
        latlng_limit_dict = {
            plotting_utils.MIN_LATITUDE_KEY: min_plot_latitude_deg,
            plotting_utils.MAX_LATITUDE_KEY: max_plot_latitude_deg,
            plotting_utils.MIN_LONGITUDE_KEY: min_plot_longitude_deg,
            plotting_utils.MAX_LONGITUDE_KEY: max_plot_longitude_deg
        }
    else:
        latlng_limit_dict = None

    axes_object, basemap_object = plotting_utils.init_map_with_nwp_projection(
        model_name=nwp_model_utils.RAP_MODEL_NAME,
        grid_name=nwp_model_utils.NAME_OF_130GRID, xy_limit_dict=None,
        latlng_limit_dict=latlng_limit_dict, resolution_string='i'
    )[1:]

    x_offset_metres, y_offset_metres = _get_projection_offsets(
        basemap_object=basemap_object, pyproj_object=PYPROJ_OBJECT,
        test_latitudes_deg=TEST_LATITUDES_DEG,
        test_longitudes_deg=TEST_LONGITUDES_DEG)

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

    # TODO(thunderhoser): Put this business into a method.
    parallel_spacing_deg = (
        (max_plot_latitude_deg - min_plot_latitude_deg) / (NUM_PARALLELS - 1)
    )
    meridian_spacing_deg = (
        (max_plot_longitude_deg - min_plot_longitude_deg) / (NUM_MERIDIANS - 1)
    )

    if parallel_spacing_deg < 1.:
        parallel_spacing_deg = number_rounding.round_to_nearest(
            parallel_spacing_deg, 0.1)
    else:
        parallel_spacing_deg = numpy.round(parallel_spacing_deg)

    if meridian_spacing_deg < 1.:
        meridian_spacing_deg = number_rounding.round_to_nearest(
            meridian_spacing_deg, 0.1)
    else:
        meridian_spacing_deg = numpy.round(meridian_spacing_deg)

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
        parallel_spacing_deg=parallel_spacing_deg, line_colour=GRID_LINE_COLOUR)

    plotting_utils.plot_meridians(
        basemap_object=basemap_object, axes_object=axes_object,
        bottom_left_lng_deg=0., upper_right_lng_deg=360.,
        meridian_spacing_deg=meridian_spacing_deg, line_colour=GRID_LINE_COLOUR)

    colour_map_object, colour_norm_object = (
        probability_plotting.get_default_colour_map()
    )

    plotting_utils.add_colour_bar(
        axes_object_or_list=axes_object, values_to_colour=probability_matrix,
        colour_map=colour_map_object, colour_norm_object=colour_norm_object,
        orientation='horizontal', extend_min=True, extend_max=True,
        fraction_of_axis_length=0.8)

    if len(tornado_latitudes_deg) > 0:
        tornado_x_coords_metres, tornado_y_coords_metres = basemap_object(
            tornado_longitudes_deg, tornado_latitudes_deg)

        axes_object.plot(
            tornado_x_coords_metres, tornado_y_coords_metres, linestyle='None',
            marker=TORNADO_MARKER_TYPE, markersize=TORNADO_MARKER_SIZE,
            markeredgewidth=TORNADO_MARKER_EDGE_WIDTH,
            markerfacecolor=TORNADO_MARKER_COLOUR,
            markeredgecolor=TORNADO_MARKER_COLOUR)

    init_time_string = time_conversion.unix_sec_to_string(
        init_time_unix_sec, FILE_NAME_TIME_FORMAT
    )

    # first_valid_time_string = time_conversion.unix_sec_to_string(
    #     first_valid_time_unix_sec, FILE_NAME_TIME_FORMAT
    # )
    # last_valid_time_string = time_conversion.unix_sec_to_string(
    #     last_valid_time_unix_sec, FILE_NAME_TIME_FORMAT
    # )
    # title_string = 'Forecast init {0:s}, valid {1:s} to {2:s}'.format(
    #     init_time_string, first_valid_time_string, last_valid_time_string
    # )
    # pyplot.title(title_string, fontsize=TITLE_FONT_SIZE)

    output_file_name = (
        '{0:s}/gridded_forecast_init-{1:s}_lead-{2:06d}-{3:06d}sec.jpg'
    ).format(
        output_dir_name, init_time_string, min_lead_time_seconds,
        max_lead_time_seconds
    )

    print 'Saving figure to: "{0:s}"...'.format(output_file_name)
    pyplot.savefig(output_file_name, dpi=FIGURE_RESOLUTION_DPI)
    pyplot.close()

    imagemagick_utils.trim_whitespace(input_file_name=output_file_name,
                                      output_file_name=output_file_name)


def _run(input_prediction_file_name, tornado_dir_name, min_plot_latitude_deg,
         max_plot_latitude_deg, min_plot_longitude_deg,
         max_plot_longitude_deg, output_dir_name):
    """Plots CNN forecasts on the RAP grid.

    This is effectively the main method.

    :param input_prediction_file_name: See documentation at top of file.
    :param tornado_dir_name: Same.
    :param min_plot_latitude_deg: Same.
    :param max_plot_latitude_deg: Same.
    :param min_plot_longitude_deg: Same.
    :param max_plot_longitude_deg: Same.
    :param output_dir_name: Same.
    """

    if tornado_dir_name in ['', 'None']:
        tornado_dir_name = None

    if min_plot_latitude_deg <= SENTINEL_VALUE:
        min_plot_latitude_deg = None
    if max_plot_latitude_deg <= SENTINEL_VALUE:
        max_plot_latitude_deg = None
    if min_plot_longitude_deg <= SENTINEL_VALUE:
        min_plot_longitude_deg = None
    if max_plot_longitude_deg <= SENTINEL_VALUE:
        max_plot_longitude_deg = None

    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=output_dir_name)

    print 'Reading data from: "{0:s}"...'.format(input_prediction_file_name)
    gridded_forecast_dict = prediction_io.read_gridded_predictions(
        input_prediction_file_name)

    num_times = len(gridded_forecast_dict[prediction_io.INIT_TIMES_KEY])

    for i in range(num_times):
        _plot_forecast_one_time(
            gridded_forecast_dict=gridded_forecast_dict, time_index=i,
            min_plot_latitude_deg=min_plot_latitude_deg,
            max_plot_latitude_deg=max_plot_latitude_deg,
            min_plot_longitude_deg=min_plot_longitude_deg,
            max_plot_longitude_deg=max_plot_longitude_deg,
            output_dir_name=output_dir_name, tornado_dir_name=tornado_dir_name)


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        input_prediction_file_name=getattr(
            INPUT_ARG_OBJECT, INPUT_FILE_ARG_NAME),
        tornado_dir_name=getattr(INPUT_ARG_OBJECT, TORNADO_DIR_ARG_NAME),
        min_plot_latitude_deg=getattr(INPUT_ARG_OBJECT, MIN_LATITUDE_ARG_NAME),
        max_plot_latitude_deg=getattr(INPUT_ARG_OBJECT, MAX_LATITUDE_ARG_NAME),
        min_plot_longitude_deg=getattr(
            INPUT_ARG_OBJECT, MIN_LONGITUDE_ARG_NAME),
        max_plot_longitude_deg=getattr(
            INPUT_ARG_OBJECT, MAX_LONGITUDE_ARG_NAME),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
