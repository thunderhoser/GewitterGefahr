"""Plots tornado-warning polygons."""

import pickle
import argparse
import numpy
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as pyplot
from gewittergefahr.gg_utils import polygons
from gewittergefahr.gg_utils import number_rounding
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.plotting import plotting_utils
from gewittergefahr.plotting import imagemagick_utils

SENTINEL_VALUE = -9999

LINE_WIDTH = 2.
NUM_PARALLELS = 8
NUM_MERIDIANS = 6
LATLNG_BUFFER_DEG = 0.5
FIGURE_RESOLUTION_DPI = 300

POLYGON_COLUMN = 'polygon_object_latlng'

INPUT_FILE_ARG_NAME = 'input_pickle_file_name'
BORDER_COLOUR_ARG_NAME = 'border_colour'
POLYGON_COLOUR_ARG_NAME = 'polygon_colour'
MIN_LATITUDE_ARG_NAME = 'min_plot_latitude_deg'
MAX_LATITUDE_ARG_NAME = 'max_plot_latitude_deg'
MIN_LONGITUDE_ARG_NAME = 'min_plot_longitude_deg'
MAX_LONGITUDE_ARG_NAME = 'max_plot_longitude_deg'
OUTPUT_FILE_ARG_NAME = 'output_file_name'

INPUT_FILE_HELP_STRING = (
    'Path to input file (created by convert_warning_polygons.py).')

BORDER_COLOUR_HELP_STRING = (
    'Colour of political borders (length-3 array with [R, G, B], each in range '
    '0...255).')

POLYGON_COLOUR_HELP_STRING = (
    'Colour of warning polygons (length-3 array with [R, G, B], each in range '
    '0...255).')

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

OUTPUT_FILE_HELP_STRING = 'Path to output file.  Figure will be saved here.'

# DEFAULT_BORDER_COLOUR = numpy.array([139, 69, 19], dtype=int)
DEFAULT_BORDER_COLOUR = numpy.full(3, 152. / 255)
DEFAULT_POLYGON_COLOUR = numpy.full(3, 0, dtype=int)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + INPUT_FILE_ARG_NAME, type=str, required=True,
    help=INPUT_FILE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + BORDER_COLOUR_ARG_NAME, type=int, nargs=3, required=False,
    default=DEFAULT_BORDER_COLOUR, help=BORDER_COLOUR_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + POLYGON_COLOUR_ARG_NAME, type=int, nargs=3, required=False,
    default=DEFAULT_POLYGON_COLOUR, help=POLYGON_COLOUR_HELP_STRING)

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
    '--' + OUTPUT_FILE_ARG_NAME, type=str, required=True,
    help=OUTPUT_FILE_HELP_STRING)


def _run(input_file_name, border_colour, polygon_colour, min_plot_latitude_deg,
         max_plot_latitude_deg, min_plot_longitude_deg, max_plot_longitude_deg,
         output_file_name):
    """Plots tornado-warning polygons.

    This is effectively the main method.

    :param input_file_name: See documentation at top of file.
    :param border_colour: Same.
    :param polygon_colour: Same.
    :param min_plot_latitude_deg: Same.
    :param max_plot_latitude_deg: Same.
    :param min_plot_longitude_deg: Same.
    :param max_plot_longitude_deg: Same.
    :param output_file_name: Same.
    """

    print('Reading data from: "{0:s}"...'.format(input_file_name))
    pickle_file_handle = open(input_file_name, 'rb')
    warning_table = pickle.load(pickle_file_handle)
    pickle_file_handle.close()

    num_warnings = len(warning_table.index)
    warning_min_latitudes_deg = numpy.full(num_warnings, numpy.nan)
    warning_max_latitudes_deg = numpy.full(num_warnings, numpy.nan)
    warning_min_longitudes_deg = numpy.full(num_warnings, numpy.nan)
    warning_max_longitudes_deg = numpy.full(num_warnings, numpy.nan)

    for i in range(num_warnings):
        this_vertex_dict_latlng = polygons.polygon_object_to_vertex_arrays(
            warning_table[POLYGON_COLUMN].values[i]
        )

        warning_min_latitudes_deg[i] = numpy.min(
            this_vertex_dict_latlng[polygons.EXTERIOR_Y_COLUMN]
        )
        warning_max_latitudes_deg[i] = numpy.max(
            this_vertex_dict_latlng[polygons.EXTERIOR_Y_COLUMN]
        )
        warning_min_longitudes_deg[i] = numpy.min(
            this_vertex_dict_latlng[polygons.EXTERIOR_X_COLUMN]
        )
        warning_max_longitudes_deg[i] = numpy.max(
            this_vertex_dict_latlng[polygons.EXTERIOR_X_COLUMN]
        )

    if min_plot_latitude_deg <= SENTINEL_VALUE:
        min_plot_latitude_deg = (
            numpy.min(warning_min_latitudes_deg) - LATLNG_BUFFER_DEG
        )

    if max_plot_latitude_deg <= SENTINEL_VALUE:
        max_plot_latitude_deg = (
            numpy.max(warning_min_latitudes_deg) + LATLNG_BUFFER_DEG
        )

    if min_plot_longitude_deg <= SENTINEL_VALUE:
        min_plot_longitude_deg = (
            numpy.min(warning_min_longitudes_deg) - LATLNG_BUFFER_DEG
        )

    if max_plot_longitude_deg <= SENTINEL_VALUE:
        max_plot_longitude_deg = (
            numpy.max(warning_min_longitudes_deg) + LATLNG_BUFFER_DEG
        )

    good_latitude_flags = numpy.logical_and(
        warning_max_latitudes_deg >= min_plot_latitude_deg,
        warning_min_latitudes_deg <= max_plot_latitude_deg
    )

    good_longitude_flags = numpy.logical_and(
        warning_max_longitudes_deg >= min_plot_longitude_deg,
        warning_min_longitudes_deg <= max_plot_longitude_deg
    )

    good_indices = numpy.where(numpy.logical_and(
        good_latitude_flags, good_longitude_flags
    ))[0]

    warning_table = warning_table.iloc[good_indices]

    _, axes_object, basemap_object = (
        plotting_utils.init_equidistant_cylindrical_map(
            min_latitude_deg=min_plot_latitude_deg,
            max_latitude_deg=max_plot_latitude_deg,
            min_longitude_deg=min_plot_longitude_deg,
            max_longitude_deg=max_plot_longitude_deg, resolution_string='i')
    )

    parallel_spacing_deg = (
        (max_plot_latitude_deg - min_plot_latitude_deg) / (NUM_PARALLELS - 1)
    )
    meridian_spacing_deg = (
        (max_plot_longitude_deg - min_plot_longitude_deg) / (NUM_MERIDIANS - 1)
    )

    parallel_spacing_deg = number_rounding.round_to_nearest(
        parallel_spacing_deg, 0.1)
    meridian_spacing_deg = number_rounding.round_to_nearest(
        meridian_spacing_deg, 0.1)

    plotting_utils.plot_coastlines(
        basemap_object=basemap_object, axes_object=axes_object,
        line_colour=border_colour)
    plotting_utils.plot_countries(
        basemap_object=basemap_object, axes_object=axes_object,
        line_colour=border_colour)
    plotting_utils.plot_states_and_provinces(
        basemap_object=basemap_object, axes_object=axes_object,
        line_colour=border_colour)
    plotting_utils.plot_parallels(
        basemap_object=basemap_object, axes_object=axes_object,
        bottom_left_lat_deg=-90., upper_right_lat_deg=90.,
        parallel_spacing_deg=parallel_spacing_deg)
    plotting_utils.plot_meridians(
        basemap_object=basemap_object, axes_object=axes_object,
        bottom_left_lng_deg=0., upper_right_lng_deg=360.,
        meridian_spacing_deg=meridian_spacing_deg)

    num_warnings = len(warning_table.index)

    for i in range(num_warnings):
        this_vertex_dict_latlng = polygons.polygon_object_to_vertex_arrays(
            warning_table[POLYGON_COLUMN].values[i]
        )

        these_x_coords_metres, these_y_coords_metres = basemap_object(
            this_vertex_dict_latlng[polygons.EXTERIOR_X_COLUMN],
            this_vertex_dict_latlng[polygons.EXTERIOR_Y_COLUMN]
        )

        axes_object.plot(
            these_x_coords_metres, these_y_coords_metres, color=polygon_colour,
            linestyle='solid', linewidth=LINE_WIDTH)

    print('Saving figure to: "{0:s}"...'.format(output_file_name))

    file_system_utils.mkdir_recursive_if_necessary(file_name=output_file_name)
    pyplot.savefig(output_file_name, dpi=FIGURE_RESOLUTION_DPI)
    pyplot.close()

    imagemagick_utils.trim_whitespace(input_file_name=output_file_name,
                                      output_file_name=output_file_name)


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        input_file_name=getattr(INPUT_ARG_OBJECT, INPUT_FILE_ARG_NAME),
        border_colour=numpy.array(
            getattr(INPUT_ARG_OBJECT, BORDER_COLOUR_ARG_NAME), dtype=float
        ) / 255,
        polygon_colour=numpy.array(
            getattr(INPUT_ARG_OBJECT, POLYGON_COLOUR_ARG_NAME), dtype=float
        ) / 255,
        min_plot_latitude_deg=getattr(INPUT_ARG_OBJECT, MIN_LATITUDE_ARG_NAME),
        max_plot_latitude_deg=getattr(INPUT_ARG_OBJECT, MAX_LATITUDE_ARG_NAME),
        min_plot_longitude_deg=getattr(
            INPUT_ARG_OBJECT, MIN_LONGITUDE_ARG_NAME),
        max_plot_longitude_deg=getattr(
            INPUT_ARG_OBJECT, MAX_LONGITUDE_ARG_NAME),
        output_file_name=getattr(INPUT_ARG_OBJECT, OUTPUT_FILE_ARG_NAME)
    )
