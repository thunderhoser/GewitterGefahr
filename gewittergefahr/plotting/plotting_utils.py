"""Helper methods for plotting (mostly 2-D georeferenced maps)."""

import numpy
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as pyplot
import matplotlib.colors
from mpl_toolkits.basemap import Basemap
from gewittergefahr.gg_utils import nwp_model_utils
from gewittergefahr.gg_utils import number_rounding
from gewittergefahr.gg_utils import longitude_conversion as lng_conversion
from gewittergefahr.gg_utils import error_checking

DEFAULT_FIGURE_WIDTH_INCHES = 15.
DEFAULT_FIGURE_HEIGHT_INCHES = 15.
DEFAULT_RESOLUTION_STRING = 'l'

DEFAULT_NUM_PARALLELS = 8
DEFAULT_NUM_MERIDIANS = 6

DEFAULT_COUNTRY_WIDTH = 2.
DEFAULT_PROVINCE_WIDTH = 1.
DEFAULT_COUNTY_WIDTH = 0.5
DEFAULT_COAST_WIDTH = 1.
DEFAULT_RIVER_WIDTH = 0.5
DEFAULT_GRID_LINE_WIDTH = 1.

DEFAULT_COUNTRY_COLOUR = numpy.array([139, 69, 19], dtype=float) / 255
DEFAULT_PROVINCE_COLOUR = numpy.array([139, 69, 19], dtype=float) / 255
DEFAULT_COUNTY_COLOUR = numpy.array([152, 152, 152], dtype=float) / 255
DEFAULT_COAST_COLOUR = numpy.array([31, 120, 180], dtype=float) / 255
DEFAULT_RIVER_COLOUR = numpy.array([166, 206, 227], dtype=float) / 255
DEFAULT_GRID_LINE_COLOUR = numpy.full(3, 0.)

DEFAULT_COUNTRY_Z_ORDER = -1000.
DEFAULT_PROVINCE_Z_ORDER = -100.
DEFAULT_COUNTY_Z_ORDER = -10.
DEFAULT_COAST_Z_ORDER = -1e5
DEFAULT_RIVER_Z_ORDER = -1e4
DEFAULT_GRID_LINE_Z_ORDER = -1e6

ELLIPSOID_NAME = 'sphere'
EARTH_RADIUS_METRES = 6370997.

VERTICAL_CBAR_PADDING = 0.05
HORIZONTAL_CBAR_PADDING = 0.06
DEFAULT_CBAR_ORIENTATION_STRING = 'horizontal'

DEFAULT_LABEL_FONT_SIZE = 50
DEFAULT_LABEL_FONT_COLOUR = numpy.full(3, 0.)
DEFAULT_LABEL_X_NORMALIZED = 0.
DEFAULT_LABEL_Y_NORMALIZED = 1.

FONT_SIZE = 30
pyplot.rc('font', size=FONT_SIZE)
pyplot.rc('axes', titlesize=FONT_SIZE)
pyplot.rc('axes', labelsize=FONT_SIZE)
pyplot.rc('xtick', labelsize=FONT_SIZE)
pyplot.rc('ytick', labelsize=FONT_SIZE)
pyplot.rc('legend', fontsize=FONT_SIZE)
pyplot.rc('figure', titlesize=FONT_SIZE)

X_MIN_KEY = 'x_min_metres'
X_MAX_KEY = 'x_max_metres'
Y_MIN_KEY = 'y_min_metres'
Y_MAX_KEY = 'y_max_metres'

MIN_LATITUDE_KEY = 'min_latitude_deg'
MAX_LATITUDE_KEY = 'max_latitude_deg'
MIN_LONGITUDE_KEY = 'min_longitude_deg'
MAX_LONGITUDE_KEY = 'max_longitude_deg'


def _check_basemap_args(
        min_latitude_deg, max_latitude_deg, min_longitude_deg,
        max_longitude_deg, resolution_string):
    """Error-checks input args for creating basemap.

    Latitudes must be in deg N, and longitudes must be in deg E.

    Both output values are in deg E, with positive values (180-360) in the
    western hemisphere.  The inputs may be positive or negative in WH.

    :param min_latitude_deg: Minimum latitude in map (bottom-left corner).
    :param max_latitude_deg: Max latitude in map (top-right corner).
    :param min_longitude_deg: Minimum longitude in map (bottom-left corner).
    :param max_longitude_deg: Max longitude in map (top-right corner).
    :param resolution_string: Resolution of boundaries (political borders,
        lakes, rivers, etc.) in basemap.  Options are "c" for crude, "l" for
        low, "i" for intermediate, "h" for high, and "f" for full.
    :return: min_longitude_deg: Minimum longitude (deg E, positive in western
        hemisphere).
    :return: max_longitude_deg: Max longitude (deg E, positive in western
        hemisphere).
    """

    error_checking.assert_is_valid_latitude(min_latitude_deg)
    error_checking.assert_is_valid_latitude(max_latitude_deg)
    error_checking.assert_is_greater(max_latitude_deg, min_latitude_deg)

    min_longitude_deg = lng_conversion.convert_lng_positive_in_west(
        min_longitude_deg)
    max_longitude_deg = lng_conversion.convert_lng_positive_in_west(
        max_longitude_deg)

    error_checking.assert_is_greater(max_longitude_deg, min_longitude_deg)

    error_checking.assert_is_string(resolution_string)

    return min_longitude_deg, max_longitude_deg


def colour_from_numpy_to_tuple(input_colour):
    """Converts colour from numpy array to tuple (if necessary).

    :param input_colour: Colour (possibly length-3 or length-4 numpy array).
    :return: output_colour: Colour (possibly length-3 or length-4 tuple).
    """

    if not isinstance(input_colour, numpy.ndarray):
        return input_colour

    error_checking.assert_is_numpy_array(input_colour, num_dimensions=1)

    num_entries = len(input_colour)
    error_checking.assert_is_geq(num_entries, 3)
    error_checking.assert_is_leq(num_entries, 4)

    return tuple(input_colour.tolist())


def label_axes(axes_object, label_string, font_size=DEFAULT_LABEL_FONT_SIZE,
               font_colour=DEFAULT_LABEL_FONT_COLOUR,
               x_coord_normalized=DEFAULT_LABEL_X_NORMALIZED,
               y_coord_normalized=DEFAULT_LABEL_Y_NORMALIZED):
    """Adds text label to axes.

    :param axes_object: Axes (instance of
        `matplotlib.axes._subplots.AxesSubplot`).
    :param label_string: Label.
    :param font_size: Font size.
    :param font_colour: Font colour.
    :param x_coord_normalized: Normalized x-coordinate (from 0...1, where 1 is
        the right side).
    :param y_coord_normalized: Normalized y-coordinate (from 0...1, where 1 is
        the top).
    """

    error_checking.assert_is_string(label_string)
    # error_checking.assert_is_geq(x_coord_normalized, 0.)
    # error_checking.assert_is_leq(x_coord_normalized, 1.)
    # error_checking.assert_is_geq(y_coord_normalized, 0.)
    # error_checking.assert_is_leq(y_coord_normalized, 1.)

    axes_object.text(
        x_coord_normalized, y_coord_normalized, label_string,
        fontsize=font_size, color=colour_from_numpy_to_tuple(font_colour),
        horizontalalignment='right', verticalalignment='bottom',
        transform=axes_object.transAxes)


def create_paneled_figure(
        num_rows, num_columns, figure_width_inches=DEFAULT_FIGURE_WIDTH_INCHES,
        figure_height_inches=DEFAULT_FIGURE_HEIGHT_INCHES,
        shared_x_axis=False, shared_y_axis=False, keep_aspect_ratio=True):
    """Creates paneled figure.

    This method only initializes the panels.  It does not plot anything.

    J = number of panel rows
    K = number of panel columns

    :param num_rows: J in the above discussion.
    :param num_columns: K in the above discussion.
    :param figure_width_inches: Width of the entire figure (including all
        panels).
    :param figure_height_inches: Height of the entire figure (including all
        panels).
    :param shared_x_axis: Boolean flag.  If True, all panels will share the same
        x-axis.
    :param shared_y_axis: Boolean flag.  If True, all panels will share the same
        y-axis.
    :param keep_aspect_ratio: Boolean flag.  If True, the aspect ratio of each
        panel will be preserved (reflect the aspect ratio of the data plotted
        therein).
    :return: figure_object: Figure handle (instance of
        `matplotlib.figure.Figure`).
    :return: axes_object_matrix: J-by-K numpy array of axes handles (instances
        of `matplotlib.axes._subplots.AxesSubplot`).
    """

    error_checking.assert_is_boolean(shared_x_axis)
    error_checking.assert_is_boolean(shared_y_axis)
    error_checking.assert_is_boolean(keep_aspect_ratio)

    figure_object, axes_object_matrix = pyplot.subplots(
        num_rows, num_columns, sharex=shared_x_axis, sharey=shared_y_axis,
        figsize=(figure_width_inches, figure_height_inches)
    )

    if num_rows == num_columns == 1:
        axes_object_matrix = numpy.full(
            (1, 1), axes_object_matrix, dtype=object
        )

    if num_rows == 1 or num_columns == 1:
        axes_object_matrix = numpy.reshape(
            axes_object_matrix, (num_rows, num_columns)
        )

    pyplot.subplots_adjust(
        left=0.02, bottom=0.02, right=0.98, top=0.95, hspace=0, wspace=0
    )

    if not keep_aspect_ratio:
        return figure_object, axes_object_matrix

    for i in range(num_rows):
        for j in range(num_columns):
            axes_object_matrix[i][j].set(
                adjustable='box-forced', aspect='equal'
            )

    return figure_object, axes_object_matrix


def create_lambert_conformal_map(
        min_latitude_deg, max_latitude_deg, min_longitude_deg,
        max_longitude_deg, standard_latitudes_deg=numpy.full(2, 25.),
        central_longitude_deg=265.,
        figure_width_inches=DEFAULT_FIGURE_WIDTH_INCHES,
        figure_height_inches=DEFAULT_FIGURE_HEIGHT_INCHES,
        resolution_string=DEFAULT_RESOLUTION_STRING):
    """Creates Lambert conformal map.

    This method only initializes a map with the Lambert conformal projection.
    It does not plot anything.

    Latitudes must be in deg N, and longitudes must be in deg E.

    :param min_latitude_deg: See doc for `_check_basemap_args`.
    :param max_latitude_deg: Same.
    :param min_longitude_deg: Same.
    :param max_longitude_deg: Same.
    :param standard_latitudes_deg: length-2 numpy array of standard latitudes
        for projection.
    :param central_longitude_deg: Central longitude for projection.
    :param figure_width_inches: Figure width.
    :param figure_height_inches: Figure height.
    :param resolution_string: See doc for `_check_basemap_args`.
    :return: figure_object: Figure handle (instance of
        `matplotlib.figure.Figure`).
    :return: axes_object: Axes handle (instance of
        `matplotlib.axes._subplots.AxesSubplot`).
    :return: basemap_object: Basemap handle (instance of
        `mpl_toolkits.basemap.Basemap`).
    """

    min_longitude_deg, max_longitude_deg = _check_basemap_args(
        min_latitude_deg=min_latitude_deg, max_latitude_deg=max_latitude_deg,
        min_longitude_deg=min_longitude_deg,
        max_longitude_deg=max_longitude_deg,
        resolution_string=resolution_string)

    error_checking.assert_is_valid_lat_numpy_array(standard_latitudes_deg)
    error_checking.assert_is_numpy_array(
        standard_latitudes_deg,
        exact_dimensions=numpy.array([2], dtype=int)
    )

    error_checking.assert_is_non_array(central_longitude_deg)
    central_longitude_deg = lng_conversion.convert_lng_positive_in_west(
        central_longitude_deg)

    figure_object, axes_object = pyplot.subplots(
        1, 1, figsize=(figure_width_inches, figure_height_inches)
    )

    basemap_object = Basemap(
        projection='lcc', lat_1=standard_latitudes_deg[0],
        lat_2=standard_latitudes_deg[1], lon_0=central_longitude_deg,
        rsphere=EARTH_RADIUS_METRES, ellps=ELLIPSOID_NAME,
        resolution=resolution_string, llcrnrlat=min_latitude_deg,
        llcrnrlon=min_longitude_deg, urcrnrlat=max_latitude_deg,
        urcrnrlon=max_longitude_deg)

    return figure_object, axes_object, basemap_object


def create_equidist_cylindrical_map(
        min_latitude_deg, max_latitude_deg, min_longitude_deg,
        max_longitude_deg, figure_width_inches=DEFAULT_FIGURE_WIDTH_INCHES,
        figure_height_inches=DEFAULT_FIGURE_HEIGHT_INCHES,
        resolution_string=DEFAULT_RESOLUTION_STRING):
    """Creates equidistant cylindrical map.

    This method only initializes a map with the equidistant cylindrical
    projection.  It does not plot anything.

    :param min_latitude_deg: See doc for `_check_basemap_args`.
    :param max_latitude_deg: Same.
    :param min_longitude_deg: Same.
    :param max_longitude_deg: Same.
    :param figure_width_inches: Figure width.
    :param figure_height_inches: Figure height.
    :param resolution_string: See doc for `_check_basemap_args`.
    :return: figure_object: See doc for `create_lambert_conformal_map`.
    :return: axes_object: Same.
    :return: basemap_object: Same.
    """

    min_longitude_deg, max_longitude_deg = _check_basemap_args(
        min_latitude_deg=min_latitude_deg, max_latitude_deg=max_latitude_deg,
        min_longitude_deg=min_longitude_deg,
        max_longitude_deg=max_longitude_deg,
        resolution_string=resolution_string)

    figure_object, axes_object = pyplot.subplots(
        1, 1, figsize=(figure_width_inches, figure_height_inches)
    )

    basemap_object = Basemap(
        projection='cyl', resolution=resolution_string,
        llcrnrlat=min_latitude_deg, llcrnrlon=min_longitude_deg,
        urcrnrlat=max_latitude_deg, urcrnrlon=max_longitude_deg)

    return figure_object, axes_object, basemap_object


def create_map_with_nwp_proj(
        model_name, grid_name=None, latlng_limit_dict=None, xy_limit_dict=None,
        figure_width_inches=DEFAULT_FIGURE_WIDTH_INCHES,
        figure_height_inches=DEFAULT_FIGURE_HEIGHT_INCHES,
        resolution_string=DEFAULT_RESOLUTION_STRING):
    """Initializes map with same projection as NWP model.

    However, this map will have false easting = false northing = 0 metres.

    If `latlng_limit_dict is not None`, corners of the map will be determined by
    lat-long coords.

    If `xy_limit_dict is not None`, corners of the map will be determined by
    x-y coords.

    If both are None, corners of the map will be x-y corners of model grid.

    :param model_name: See doc for `nwp_model_utils.check_grid_name`.
    :param grid_name: See doc for `nwp_model_utils.check_grid_name`.
    :param latlng_limit_dict: Dictionary with the following keys:
    latlng_limit_dict['min_latitude_deg']: Minimum latitude (deg N) in map.
    latlng_limit_dict['max_latitude_deg']: Max latitude (deg N) in map.
    latlng_limit_dict['min_longitude_deg']: Minimum longitude (deg E) in map.
    latlng_limit_dict['max_longitude_deg']: Max longitude (deg E) in map.

    :param xy_limit_dict: Dictionary with the following keys:
    xy_limit_dict['x_min_metres']: Minimum x-coord in map.
    xy_limit_dict['x_max_metres']: Max x-coord in map.
    xy_limit_dict['y_min_metres']: Minimum y-coord in map.
    xy_limit_dict['y_max_metres']: Max y-coord in map.

    :param figure_width_inches: Figure width.
    :param figure_height_inches: Figure height.
    :param resolution_string: See doc for `create_lambert_conformal_map`.
    :return: figure_object: Same.
    :return: axes_object: Same.
    :return: basemap_object: Same.
    """

    nwp_model_utils.check_grid_name(model_name=model_name, grid_name=grid_name)

    standard_latitudes_deg, central_longitude_deg = (
        nwp_model_utils.get_projection_params(model_name)
    )

    if latlng_limit_dict is None and xy_limit_dict is None:
        all_x_coords_metres, all_y_coords_metres = (
            nwp_model_utils.get_xy_grid_cell_edges(
                model_name=model_name, grid_name=grid_name)
        )

        false_easting_metres, false_northing_metres = (
            nwp_model_utils.get_false_easting_and_northing(
                model_name=model_name, grid_name=grid_name)
        )

        all_x_coords_metres -= false_easting_metres
        all_y_coords_metres -= false_northing_metres

        xy_limit_dict = {
            X_MIN_KEY: numpy.min(all_x_coords_metres),
            X_MAX_KEY: numpy.max(all_x_coords_metres),
            Y_MIN_KEY: numpy.min(all_y_coords_metres),
            Y_MAX_KEY: numpy.max(all_y_coords_metres)
        }

    figure_object, axes_object = pyplot.subplots(
        1, 1, figsize=(figure_width_inches, figure_height_inches)
    )

    if latlng_limit_dict is not None:
        min_latitude_deg = latlng_limit_dict[MIN_LATITUDE_KEY]
        max_latitude_deg = latlng_limit_dict[MAX_LATITUDE_KEY]

        error_checking.assert_is_valid_lat_numpy_array(
            numpy.array([min_latitude_deg, max_latitude_deg])
        )

        min_longitude_deg = lng_conversion.convert_lng_positive_in_west(
            latlng_limit_dict[MIN_LONGITUDE_KEY]
        )

        max_longitude_deg = lng_conversion.convert_lng_positive_in_west(
            latlng_limit_dict[MAX_LONGITUDE_KEY]
        )

        error_checking.assert_is_greater(max_latitude_deg, min_latitude_deg)
        error_checking.assert_is_greater(max_longitude_deg, min_longitude_deg)

        basemap_object = Basemap(
            projection='lcc', lat_1=standard_latitudes_deg[0],
            lat_2=standard_latitudes_deg[1], lon_0=central_longitude_deg,
            rsphere=EARTH_RADIUS_METRES, ellps=ELLIPSOID_NAME,
            resolution=resolution_string, llcrnrlat=min_latitude_deg,
            llcrnrlon=min_longitude_deg, urcrnrlat=max_latitude_deg,
            urcrnrlon=max_longitude_deg)
    else:
        x_min_metres = xy_limit_dict[X_MIN_KEY]
        x_max_metres = xy_limit_dict[X_MAX_KEY]
        y_min_metres = xy_limit_dict[Y_MIN_KEY]
        y_max_metres = xy_limit_dict[Y_MAX_KEY]

        error_checking.assert_is_greater(x_max_metres, x_min_metres)
        error_checking.assert_is_greater(y_max_metres, y_min_metres)

        basemap_object = Basemap(
            projection='lcc', lat_1=standard_latitudes_deg[0],
            lat_2=standard_latitudes_deg[1], lon_0=central_longitude_deg,
            rsphere=EARTH_RADIUS_METRES, ellps=ELLIPSOID_NAME,
            resolution=resolution_string,
            llcrnrx=x_min_metres, urcrnrx=x_max_metres,
            llcrnry=y_min_metres, urcrnry=y_max_metres)

    return figure_object, axes_object, basemap_object


def plot_countries(
        basemap_object, axes_object, line_width=DEFAULT_COUNTRY_WIDTH,
        line_colour=DEFAULT_COUNTRY_COLOUR, z_order=DEFAULT_COUNTRY_Z_ORDER):
    """Plots national borders.

    :param basemap_object: Basemap handle (instance of
        `mpl_toolkits.basemap.Basemap`).
    :param axes_object: Axes handle (instance of
        `matplotlib.axes._subplots.AxesSubplot`).
    :param line_width: Border width.
    :param line_colour: Border colour.
    :param z_order: z-order.  Higher numbers mean that national borders will be
        plotted closer to the "top" (on top of other features).
    """

    basemap_object.drawcountries(
        linewidth=line_width, color=colour_from_numpy_to_tuple(line_colour),
        ax=axes_object, zorder=z_order
    )


def plot_states_and_provinces(
        basemap_object, axes_object, line_width=DEFAULT_PROVINCE_WIDTH,
        line_colour=DEFAULT_PROVINCE_COLOUR, z_order=DEFAULT_PROVINCE_Z_ORDER):
    """Plots state and provincial borders.

    :param basemap_object: See doc for `plot_countries`.
    :param axes_object: Same.
    :param line_width: Same.
    :param line_colour: Same.
    :param z_order: Same.
    """

    basemap_object.drawstates(
        linewidth=line_width, color=colour_from_numpy_to_tuple(line_colour),
        ax=axes_object, zorder=z_order
    )


def plot_counties(
        basemap_object, axes_object, line_width=DEFAULT_COUNTY_WIDTH,
        line_colour=DEFAULT_COUNTY_COLOUR, z_order=DEFAULT_COUNTY_Z_ORDER):
    """Plots county borders.

    :param basemap_object: See doc for `plot_countries`.
    :param axes_object: Same.
    :param line_width: Same.
    :param line_colour: Same.
    :param z_order: Same.
    """

    basemap_object.drawcounties(
        linewidth=line_width, color=colour_from_numpy_to_tuple(line_colour),
        ax=axes_object, zorder=z_order
    )


def plot_coastlines(
        basemap_object, axes_object, line_width=DEFAULT_COAST_WIDTH,
        line_colour=DEFAULT_COAST_COLOUR, z_order=DEFAULT_COAST_Z_ORDER):
    """Plots coastlines (for some reason this includes lakes -- ugh).

    :param basemap_object: See doc for `plot_countries`.
    :param axes_object: Same.
    :param line_width: Same.
    :param line_colour: Same.
    :param z_order: Same.
    """

    basemap_object.drawcoastlines(
        linewidth=line_width, color=colour_from_numpy_to_tuple(line_colour),
        ax=axes_object, zorder=z_order
    )


def plot_rivers(
        basemap_object, axes_object, line_width=DEFAULT_RIVER_WIDTH,
        line_colour=DEFAULT_RIVER_COLOUR, z_order=DEFAULT_RIVER_Z_ORDER):
    """Plots rivers.

    :param basemap_object: See doc for `plot_countries`.
    :param axes_object: Same.
    :param line_width: Same.
    :param line_colour: Same.
    :param z_order: Same.
    """

    basemap_object.drawrivers(
        linewidth=line_width, color=colour_from_numpy_to_tuple(line_colour),
        ax=axes_object, zorder=z_order
    )


def plot_parallels(
        basemap_object, axes_object, min_latitude_deg, max_latitude_deg,
        num_parallels=DEFAULT_NUM_PARALLELS, line_width=DEFAULT_GRID_LINE_WIDTH,
        line_colour=DEFAULT_GRID_LINE_COLOUR,
        z_order=DEFAULT_GRID_LINE_Z_ORDER):
    """Plots parallels (grid lines for latitude).

    :param basemap_object: See doc for `plot_countries`.
    :param axes_object: Same.
    :param min_latitude_deg: Minimum latitude for grid lines.
    :param max_latitude_deg: Max latitude for grid lines.
    :param num_parallels: Number of parallels.
    :param line_width: See doc for `plot_countries`.
    :param line_colour: Same.
    :param z_order: Same.
    """

    error_checking.assert_is_valid_latitude(min_latitude_deg)
    error_checking.assert_is_valid_latitude(max_latitude_deg)
    error_checking.assert_is_greater(max_latitude_deg, min_latitude_deg)

    error_checking.assert_is_integer(num_parallels)
    error_checking.assert_is_geq(num_parallels, 2)

    parallel_spacing_deg = (
        (max_latitude_deg - min_latitude_deg) / (num_parallels - 1)
    )

    if parallel_spacing_deg < 1.:
        parallel_spacing_deg = number_rounding.round_to_nearest(
            parallel_spacing_deg, 0.1)
    else:
        parallel_spacing_deg = numpy.round(parallel_spacing_deg)

    min_latitude_deg = number_rounding.ceiling_to_nearest(
        min_latitude_deg, parallel_spacing_deg
    )
    max_latitude_deg = number_rounding.floor_to_nearest(
        max_latitude_deg, parallel_spacing_deg
    )
    num_parallels = 1 + int(numpy.round(
        (max_latitude_deg - min_latitude_deg) / parallel_spacing_deg
    ))
    latitudes_deg = numpy.linspace(
        min_latitude_deg, max_latitude_deg, num=num_parallels
    )

    basemap_object.drawparallels(
        latitudes_deg, color=colour_from_numpy_to_tuple(line_colour),
        linewidth=line_width, labels=[True, False, False, False],
        ax=axes_object, zorder=z_order
    )


def plot_meridians(
        basemap_object, axes_object, min_longitude_deg, max_longitude_deg,
        num_meridians=DEFAULT_NUM_MERIDIANS, line_width=DEFAULT_GRID_LINE_WIDTH,
        line_colour=DEFAULT_GRID_LINE_COLOUR,
        z_order=DEFAULT_GRID_LINE_Z_ORDER):
    """Plots meridians (grid lines for longitude).

    :param basemap_object: See doc for `plot_countries`.
    :param axes_object: Same.
    :param min_longitude_deg: Minimum longitude for grid lines.
    :param max_longitude_deg: Max longitude for grid lines.
    :param num_meridians: Number of meridians.
    :param line_width: See doc for `plot_countries`.
    :param line_colour: Same.
    :param z_order: Same.
    """

    min_longitude_deg = lng_conversion.convert_lng_positive_in_west(
        min_longitude_deg)
    max_longitude_deg = lng_conversion.convert_lng_positive_in_west(
        max_longitude_deg)

    error_checking.assert_is_greater(max_longitude_deg, min_longitude_deg)
    error_checking.assert_is_integer(num_meridians)
    error_checking.assert_is_geq(num_meridians, 2)

    meridian_spacing_deg = (
        (max_longitude_deg - min_longitude_deg) / (num_meridians - 1)
    )

    if meridian_spacing_deg < 1.:
        meridian_spacing_deg = number_rounding.round_to_nearest(
            meridian_spacing_deg, 0.1)
    else:
        meridian_spacing_deg = numpy.round(meridian_spacing_deg)

    min_longitude_deg = number_rounding.ceiling_to_nearest(
        min_longitude_deg, meridian_spacing_deg
    )
    max_longitude_deg = number_rounding.floor_to_nearest(
        max_longitude_deg, meridian_spacing_deg
    )
    num_meridians = 1 + int(numpy.round(
        (max_longitude_deg - min_longitude_deg) / meridian_spacing_deg
    ))
    longitudes_deg = numpy.linspace(
        min_longitude_deg, max_longitude_deg, num=num_meridians
    )

    basemap_object.drawmeridians(
        longitudes_deg, color=colour_from_numpy_to_tuple(line_colour),
        linewidth=line_width, labels=[False, False, False, True],
        ax=axes_object, zorder=z_order
    )


def plot_colour_bar(
        axes_object_or_matrix, data_matrix, colour_map_object,
        colour_norm_object, orientation_string=DEFAULT_CBAR_ORIENTATION_STRING,
        extend_min=True, extend_max=True, fraction_of_axis_length=1.,
        font_size=FONT_SIZE):
    """Plots colour bar.

    :param axes_object_or_matrix: Either one axis handle (instance of
        `matplotlib.axes._subplots.AxesSubplot`) or a numpy array thereof.
    :param data_matrix: numpy array of values to which the colour map applies.
    :param colour_map_object: Colour map (instance of `matplotlib.pyplot.cm` or
        similar).
    :param colour_norm_object: Colour normalization (maps from data space to
        colour-bar space, which goes from 0...1).  This should be an instance of
        `matplotlib.colors.Normalize`.
    :param orientation_string: Orientation ("vertical" or "horizontal").
    :param extend_min: Boolean flag.  If True, values below the minimum
        specified by `colour_norm_object` are possible, so the colour bar will
        be plotted with an arrow at the bottom.
    :param extend_max: Boolean flag.  If True, values above the max specified by
        `colour_norm_object` are possible, so the colour bar will be plotted
        with an arrow at the top.
    :param fraction_of_axis_length: The colour bar will take up this fraction of
        the axis length (x-axis if orientation_string = "horizontal", y-axis if
        orientation_string = "vertical").
    :param font_size: Font size for tick marks on colour bar.
    :return: colour_bar_object: Colour-bar handle (instance of
        `matplotlib.pyplot.colorbar`).
    """

    error_checking.assert_is_real_numpy_array(data_matrix)
    error_checking.assert_is_boolean(extend_min)
    error_checking.assert_is_boolean(extend_max)
    error_checking.assert_is_greater(fraction_of_axis_length, 0.)
    error_checking.assert_is_leq(fraction_of_axis_length, 1.)

    scalar_mappable_object = pyplot.cm.ScalarMappable(
        cmap=colour_map_object, norm=colour_norm_object
    )
    scalar_mappable_object.set_array(data_matrix)

    if extend_min and extend_max:
        extend_arg = 'both'
    elif extend_min:
        extend_arg = 'min'
    elif extend_max:
        extend_arg = 'max'
    else:
        extend_arg = 'neither'

    if orientation_string == 'horizontal':
        padding_arg = HORIZONTAL_CBAR_PADDING
    else:
        padding_arg = VERTICAL_CBAR_PADDING

    if isinstance(axes_object_or_matrix, numpy.ndarray):
        axes_arg = axes_object_or_matrix.ravel().tolist()
    else:
        axes_arg = axes_object_or_matrix

    colour_bar_object = pyplot.colorbar(
        ax=axes_arg, mappable=scalar_mappable_object,
        orientation=orientation_string, pad=padding_arg, extend=extend_arg,
        shrink=fraction_of_axis_length)

    colour_bar_object.ax.tick_params(labelsize=font_size)

    if orientation_string == 'horizontal':
        colour_bar_object.ax.set_xticklabels(
            colour_bar_object.ax.get_xticklabels(), rotation=90
        )

    return colour_bar_object


def plot_linear_colour_bar(
        axes_object_or_matrix, data_matrix, colour_map_object, min_value,
        max_value, orientation_string=DEFAULT_CBAR_ORIENTATION_STRING,
        extend_min=True, extend_max=True, fraction_of_axis_length=1.,
        font_size=FONT_SIZE):
    """Plots colour bar with linear scale.

    :param axes_object_or_matrix: See doc for `plot_colour_bar`.
    :param data_matrix: Same.
    :param colour_map_object: Same.
    :param min_value: Minimum value in colour bar.
    :param max_value: Max value in colour bar.
    :param orientation_string: See doc for `plot_colour_bar`.
    :param extend_min: Same.
    :param extend_max: Same.
    :param fraction_of_axis_length: Same.
    :param font_size: Same.
    :return: colour_bar_object: Same.
    """

    error_checking.assert_is_greater(max_value, min_value)
    colour_norm_object = matplotlib.colors.Normalize(
        vmin=min_value, vmax=max_value, clip=False)

    return plot_colour_bar(
        axes_object_or_matrix=axes_object_or_matrix, data_matrix=data_matrix,
        colour_map_object=colour_map_object,
        colour_norm_object=colour_norm_object,
        orientation_string=orientation_string,
        extend_min=extend_min, extend_max=extend_max,
        fraction_of_axis_length=fraction_of_axis_length, font_size=font_size)
