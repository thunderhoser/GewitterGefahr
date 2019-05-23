"""Plots SPC convective outlook."""

import pickle
import argparse
import numpy
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as pyplot
from descartes import PolygonPatch
from gewittergefahr.gg_utils import polygons
from gewittergefahr.gg_utils import number_rounding
from gewittergefahr.gg_utils import nwp_model_utils
from gewittergefahr.plotting import plotting_utils
from gewittergefahr.plotting import imagemagick_utils

SENTINEL_VALUE = -9999

OUTLOOK_OPACITY = 0.5
WARNING_LINE_WIDTH = 2.
RISK_TYPE_FONT_SIZE = 30

NUM_PARALLELS = 8
NUM_MERIDIANS = 6
LATLNG_BUFFER_DEG = 0.5
FIGURE_RESOLUTION_DPI = 300

MARGINAL_RISK_STRING = 'marginal'
SLIGHT_RISK_STRING = 'slight'
ENHANCED_RISK_STRING = 'enhanced'
MODERATE_RISK_STRING = 'moderate'
HIGH_RISK_STRING = 'high'

RISK_TYPE_STRING_TO_ENUM = {
    MARGINAL_RISK_STRING: 0,
    SLIGHT_RISK_STRING: 1,
    ENHANCED_RISK_STRING: 2,
    MODERATE_RISK_STRING: 3,
    HIGH_RISK_STRING: 4
}

RISK_TYPE_STRING_TO_COLOUR = {
    MARGINAL_RISK_STRING: numpy.array([128, 197, 128]),
    SLIGHT_RISK_STRING: numpy.array([246, 246, 128]),
    ENHANCED_RISK_STRING: numpy.array([230, 194, 128]),
    MODERATE_RISK_STRING: numpy.array([230, 139, 123]),
    HIGH_RISK_STRING: numpy.array([254, 125, 254])
}

# RISK_TYPE_STRING_TO_COLOUR = {
#     MARGINAL_RISK_STRING: numpy.array([71, 133, 71]),
#     SLIGHT_RISK_STRING: numpy.array([255, 150, 0]),
#     ENHANCED_RISK_STRING: numpy.array([255, 127, 0]),
#     MODERATE_RISK_STRING: numpy.array([205, 0, 0]),
#     HIGH_RISK_STRING: numpy.array([255, 0, 255])
# }

for THIS_KEY in RISK_TYPE_STRING_TO_COLOUR:
    RISK_TYPE_STRING_TO_COLOUR[THIS_KEY] = (
        RISK_TYPE_STRING_TO_COLOUR[THIS_KEY].astype(float) / 255
    )

RISK_TYPE_COLUMN = 'risk_type_string'
POLYGON_COLUMN = 'polygon_object_latlng'

MIN_LATITUDE_COLUMN = 'min_latitude_deg'
MAX_LATITUDE_COLUMN = 'max_latitude_deg'
MIN_LONGITUDE_COLUMN = 'min_longitude_deg'
MAX_LONGITUDE_COLUMN = 'max_longitude_deg'

OUTLOOK_FILE_ARG_NAME = 'input_outlook_file_name'
WARNING_FILE_ARG_NAME = 'input_warning_file_name'
BORDER_COLOUR_ARG_NAME = 'border_colour'
WARNING_COLOUR_ARG_NAME = 'warning_colour'
MIN_LATITUDE_ARG_NAME = 'min_plot_latitude_deg'
MAX_LATITUDE_ARG_NAME = 'max_plot_latitude_deg'
MIN_LONGITUDE_ARG_NAME = 'min_plot_longitude_deg'
MAX_LONGITUDE_ARG_NAME = 'max_plot_longitude_deg'
OUTPUT_FILE_ARG_NAME = 'output_file_name'

OUTLOOK_FILE_HELP_STRING = (
    'Path to file with SPC convective outlook (created by '
    'convert_spc_outlook.py).')

WARNING_FILE_HELP_STRING = (
    'Path to file with tornado-warning polygons for the same day (created by '
    'convert_warning_polygons.py).  If you do not want to plot warnings, leave '
    'this argument alone.')

BORDER_COLOUR_HELP_STRING = (
    'Colour of political borders (length-3 array with [R, G, B], each in range '
    '0...255).')

WARNING_COLOUR_HELP_STRING = (
    '[used only if `{0:s}` is not empty] Colour of warning polygons (length-3 '
    'array with [R, G, B], each in range 0...255).'
).format(WARNING_FILE_ARG_NAME)

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
# DEFAULT_BORDER_COLOUR = numpy.full(3, 152, dtype=int)

DEFAULT_BORDER_COLOUR = numpy.full(3, 0, dtype=int)
DEFAULT_WARNING_COLOUR = numpy.full(3, 0, dtype=int)
LATLNG_COLOUR = numpy.full(3, 1.)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + OUTLOOK_FILE_ARG_NAME, type=str, required=True,
    help=OUTLOOK_FILE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + WARNING_FILE_ARG_NAME, type=str, required=False, default='',
    help=WARNING_FILE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + BORDER_COLOUR_ARG_NAME, type=int, nargs=3, required=False,
    default=DEFAULT_BORDER_COLOUR, help=BORDER_COLOUR_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + WARNING_COLOUR_ARG_NAME, type=int, nargs=3, required=False,
    default=DEFAULT_WARNING_COLOUR, help=WARNING_COLOUR_HELP_STRING)

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


def _get_bounding_boxes(outlook_or_warning_table):
    """Returns lat-long bounding box for each polygon (outlook or warning).

    :param outlook_or_warning_table: pandas DataFrame from file created by
        convert_spc_outlook.py or convert_warning_polygons.py.
    :return: outlook_or_warning_table: Same as input but with extra columns
        listed below.
    outlook_or_warning_table['min_latitude_deg']: Minimum latitude (deg N) in
        polygon.
    outlook_or_warning_table['max_latitude_deg']: Max latitude (deg N) in
        polygon.
    outlook_or_warning_table['min_longitude_deg']: Minimum longitude (deg E) in
        polygon.
    outlook_or_warning_table['max_longitude_deg']: Max longitude (deg E) in
        polygon.
    """

    num_polygons = len(outlook_or_warning_table.index)

    min_latitudes_deg = numpy.full(num_polygons, numpy.nan)
    max_latitudes_deg = numpy.full(num_polygons, numpy.nan)
    min_longitudes_deg = numpy.full(num_polygons, numpy.nan)
    max_longitudes_deg = numpy.full(num_polygons, numpy.nan)

    for i in range(num_polygons):
        this_vertex_dict_latlng = polygons.polygon_object_to_vertex_arrays(
            outlook_or_warning_table[POLYGON_COLUMN].values[i]
        )

        min_latitudes_deg[i] = numpy.min(
            this_vertex_dict_latlng[polygons.EXTERIOR_Y_COLUMN]
        )
        max_latitudes_deg[i] = numpy.max(
            this_vertex_dict_latlng[polygons.EXTERIOR_Y_COLUMN]
        )
        min_longitudes_deg[i] = numpy.min(
            this_vertex_dict_latlng[polygons.EXTERIOR_X_COLUMN]
        )
        max_longitudes_deg[i] = numpy.max(
            this_vertex_dict_latlng[polygons.EXTERIOR_X_COLUMN]
        )

    return outlook_or_warning_table.assign(**{
        MIN_LATITUDE_COLUMN: min_latitudes_deg,
        MAX_LATITUDE_COLUMN: max_latitudes_deg,
        MIN_LONGITUDE_COLUMN: min_longitudes_deg,
        MAX_LONGITUDE_COLUMN: max_longitudes_deg
    })


def _get_plotting_limits(
        min_plot_latitude_deg, max_plot_latitude_deg, min_plot_longitude_deg,
        max_plot_longitude_deg, outlook_table, warning_table=None):
    """Returns lat-long limits for plotting.

    :param min_plot_latitude_deg: See documentation at top of file.  If
        `min_plot_latitude_deg == SENTINEL_VALUE`, it will be replaced.
        Otherwise, it will be unaltered.
    :param max_plot_latitude_deg: Same.
    :param min_plot_longitude_deg: Same.
    :param max_plot_longitude_deg: Same.
    :param outlook_table: pandas DataFrame with convective-outlook polygons,
        created by `_get_bounding_boxes`.
    :param warning_table: Same but for tornado warnings.  If tornado warnings
        are not to be plotted, leave this as None.
    :return: latitude_limits_deg: length-2 numpy array with [min, max] latitudes
        in deg N.
    :return: longitude_limits_deg: length-2 numpy array with [min, max]
        longitudes in deg E.
    """

    if min_plot_latitude_deg <= SENTINEL_VALUE:
        if warning_table is None:
            all_min_latitudes_deg = outlook_table[MIN_LATITUDE_COLUMN].values
        else:
            all_min_latitudes_deg = numpy.concatenate((
                outlook_table[MIN_LATITUDE_COLUMN].values,
                warning_table[MIN_LATITUDE_COLUMN].values
            ))

        min_plot_latitude_deg = -LATLNG_BUFFER_DEG + numpy.min(
            all_min_latitudes_deg)

    if max_plot_latitude_deg <= SENTINEL_VALUE:
        if warning_table is None:
            all_max_latitudes_deg = outlook_table[MAX_LATITUDE_COLUMN].values
        else:
            all_max_latitudes_deg = numpy.concatenate((
                outlook_table[MAX_LATITUDE_COLUMN].values,
                warning_table[MAX_LATITUDE_COLUMN].values
            ))

        max_plot_latitude_deg = LATLNG_BUFFER_DEG + numpy.max(
            all_max_latitudes_deg)

    if min_plot_longitude_deg <= SENTINEL_VALUE:
        if warning_table is None:
            all_min_longitudes_deg = outlook_table[MIN_LONGITUDE_COLUMN].values
        else:
            all_min_longitudes_deg = numpy.concatenate((
                outlook_table[MIN_LONGITUDE_COLUMN].values,
                warning_table[MIN_LONGITUDE_COLUMN].values
            ))

        min_plot_longitude_deg = -LATLNG_BUFFER_DEG + numpy.min(
            all_min_longitudes_deg)

    if max_plot_longitude_deg <= SENTINEL_VALUE:
        if warning_table is None:
            all_max_longitudes_deg = outlook_table[MAX_LONGITUDE_COLUMN].values
        else:
            all_max_longitudes_deg = numpy.concatenate((
                outlook_table[MAX_LONGITUDE_COLUMN].values,
                warning_table[MAX_LONGITUDE_COLUMN].values
            ))

        max_plot_longitude_deg = LATLNG_BUFFER_DEG + numpy.max(
            all_max_longitudes_deg)

    latitude_limits_deg = numpy.array([
        min_plot_latitude_deg, max_plot_latitude_deg
    ])
    longitude_limits_deg = numpy.array([
        min_plot_longitude_deg, max_plot_longitude_deg
    ])

    return latitude_limits_deg, longitude_limits_deg


def _run(input_outlook_file_name, input_warning_file_name, border_colour,
         warning_colour, min_plot_latitude_deg, max_plot_latitude_deg,
         min_plot_longitude_deg, max_plot_longitude_deg, output_file_name):
    """Plots SPC convective outlook.

    This is effectively the main method.

    :param input_outlook_file_name: See documentation at top of file.
    :param input_warning_file_name: Same.
    :param border_colour: Same.
    :param warning_colour: Same.
    :param min_plot_latitude_deg: Same.
    :param max_plot_latitude_deg: Same.
    :param min_plot_longitude_deg: Same.
    :param max_plot_longitude_deg: Same.
    :param output_file_name: Same.
    """

    print 'Reading SPC outlook from: "{0:s}"...'.format(input_outlook_file_name)
    pickle_file_handle = open(input_outlook_file_name, 'rb')
    outlook_table = pickle.load(pickle_file_handle)
    pickle_file_handle.close()

    risk_type_enums = numpy.array([
        RISK_TYPE_STRING_TO_ENUM[s]
        for s in outlook_table[RISK_TYPE_COLUMN].values
    ], dtype=int)

    sort_indices = numpy.argsort(risk_type_enums)
    outlook_table = outlook_table.iloc[sort_indices]

    outlook_table = _get_bounding_boxes(outlook_table)

    if input_warning_file_name in ['', 'None']:
        warning_table = None
    else:
        print 'Reading tornado warnings from: "{0:s}"...'.format(
            input_warning_file_name)

        pickle_file_handle = open(input_warning_file_name, 'rb')
        warning_table = pickle.load(pickle_file_handle)
        pickle_file_handle.close()

        warning_table = _get_bounding_boxes(warning_table)

    latitude_limits_deg, longitude_limits_deg = _get_plotting_limits(
        min_plot_latitude_deg=min_plot_latitude_deg,
        max_plot_latitude_deg=max_plot_latitude_deg,
        min_plot_longitude_deg=min_plot_longitude_deg,
        max_plot_longitude_deg=max_plot_longitude_deg,
        outlook_table=outlook_table, warning_table=warning_table)

    min_plot_latitude_deg = latitude_limits_deg[0]
    max_plot_latitude_deg = latitude_limits_deg[1]
    min_plot_longitude_deg = longitude_limits_deg[0]
    max_plot_longitude_deg = longitude_limits_deg[1]

    print (
        'Plotting limits = [{0:.2f}, {1:.2f}] deg N and [{2:.2f}, {3:.2f}] '
        'deg E'
    ).format(min_plot_latitude_deg, max_plot_latitude_deg,
             min_plot_longitude_deg, max_plot_longitude_deg)

    latlng_limit_dict = {
        plotting_utils.MIN_LATITUDE_KEY: min_plot_latitude_deg,
        plotting_utils.MAX_LATITUDE_KEY: max_plot_latitude_deg,
        plotting_utils.MIN_LONGITUDE_KEY: min_plot_longitude_deg,
        plotting_utils.MAX_LONGITUDE_KEY: max_plot_longitude_deg
    }

    axes_object, basemap_object = plotting_utils.init_map_with_nwp_projection(
        model_name=nwp_model_utils.RAP_MODEL_NAME,
        grid_name=nwp_model_utils.NAME_OF_130GRID, xy_limit_dict=None,
        latlng_limit_dict=latlng_limit_dict, resolution_string='i'
    )[1:]

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
        line_colour=border_colour)

    plotting_utils.plot_countries(
        basemap_object=basemap_object, axes_object=axes_object,
        line_colour=border_colour)

    plotting_utils.plot_states_and_provinces(
        basemap_object=basemap_object, axes_object=axes_object,
        line_colour=border_colour)

    # plotting_utils.plot_parallels(
    #     basemap_object=basemap_object, axes_object=axes_object,
    #     bottom_left_lat_deg=-90., upper_right_lat_deg=90.,
    #     parallel_spacing_deg=parallel_spacing_deg, line_colour=LATLNG_COLOUR)
    #
    # plotting_utils.plot_meridians(
    #     basemap_object=basemap_object, axes_object=axes_object,
    #     bottom_left_lng_deg=0., upper_right_lng_deg=360.,
    #     meridian_spacing_deg=meridian_spacing_deg, line_colour=LATLNG_COLOUR)

    _, unique_risk_type_indices = numpy.unique(
        outlook_table[RISK_TYPE_COLUMN].values, return_index=True)

    num_outlooks = len(outlook_table.index)
    legend_handles = []
    legend_strings = []

    for i in range(num_outlooks):
        this_risk_type_string = outlook_table[RISK_TYPE_COLUMN].values[i]
        this_colour = RISK_TYPE_STRING_TO_COLOUR[this_risk_type_string]

        this_vertex_dict_latlng = polygons.polygon_object_to_vertex_arrays(
            outlook_table[POLYGON_COLUMN].values[i]
        )

        these_x_coords_metres, these_y_coords_metres = basemap_object(
            this_vertex_dict_latlng[polygons.EXTERIOR_X_COLUMN],
            this_vertex_dict_latlng[polygons.EXTERIOR_Y_COLUMN]
        )

        this_polygon_object_xy = polygons.vertex_arrays_to_polygon_object(
            exterior_x_coords=these_x_coords_metres,
            exterior_y_coords=these_y_coords_metres)

        this_patch_object = PolygonPatch(
            this_polygon_object_xy, lw=0., ec=this_colour,
            fc=this_colour, alpha=OUTLOOK_OPACITY)
        this_handle = axes_object.add_patch(this_patch_object)

        if i in unique_risk_type_indices:
            this_string = '{0:s}{1:s}'.format(
                this_risk_type_string[0].upper(), this_risk_type_string[1:]
            )

            legend_strings.append(this_string)
            legend_handles.append(this_handle)

    if warning_table is None:
        num_warnings = 0
    else:
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
            these_x_coords_metres, these_y_coords_metres, color=warning_colour,
            linestyle='solid', linewidth=WARNING_LINE_WIDTH)

    axes_object.legend(legend_handles, legend_strings, loc='upper left')

    print 'Saving figure to: "{0:s}"...'.format(output_file_name)
    pyplot.savefig(output_file_name, dpi=FIGURE_RESOLUTION_DPI)
    pyplot.close()

    imagemagick_utils.trim_whitespace(input_file_name=output_file_name,
                                      output_file_name=output_file_name)


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        input_outlook_file_name=getattr(
            INPUT_ARG_OBJECT, OUTLOOK_FILE_ARG_NAME),
        input_warning_file_name=getattr(
            INPUT_ARG_OBJECT, WARNING_FILE_ARG_NAME),
        border_colour=numpy.array(
            getattr(INPUT_ARG_OBJECT, BORDER_COLOUR_ARG_NAME), dtype=float
        ) / 255,
        warning_colour=numpy.array(
            getattr(INPUT_ARG_OBJECT, WARNING_COLOUR_ARG_NAME), dtype=float
        ) / 255,
        min_plot_latitude_deg=getattr(INPUT_ARG_OBJECT, MIN_LATITUDE_ARG_NAME),
        max_plot_latitude_deg=getattr(INPUT_ARG_OBJECT, MAX_LATITUDE_ARG_NAME),
        min_plot_longitude_deg=getattr(
            INPUT_ARG_OBJECT, MIN_LONGITUDE_ARG_NAME),
        max_plot_longitude_deg=getattr(
            INPUT_ARG_OBJECT, MAX_LONGITUDE_ARG_NAME),
        output_file_name=getattr(INPUT_ARG_OBJECT, OUTPUT_FILE_ARG_NAME)
    )
