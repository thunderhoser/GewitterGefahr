"""Helper methods for geographic maps."""

import numpy
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as pyplot
import matplotlib.colors
from mpl_toolkits.basemap import Basemap
from gewittergefahr.gg_utils import nwp_model_utils
from gewittergefahr.gg_utils import number_rounding as rounder
from gewittergefahr.gg_utils import longitude_conversion as lng_conversion
from gewittergefahr.gg_utils import error_checking

DEFAULT_FIG_WIDTH_INCHES = 15.
DEFAULT_FIG_HEIGHT_INCHES = 15.
DEFAULT_BOUNDARY_RESOLUTION_STRING = 'l'

DEFAULT_PARALLEL_SPACING_DEG = 5.
DEFAULT_MERIDIAN_SPACING_DEG = 10.

DEFAULT_COUNTRY_WIDTH = 2.
DEFAULT_STATE_PROVINCE_WIDTH = 1.
DEFAULT_COUNTY_WIDTH = 0.5
DEFAULT_COAST_WIDTH = 1.
DEFAULT_RIVER_WIDTH = 0.5
DEFAULT_PARALLEL_MERIDIAN_WIDTH = 1.

DEFAULT_COUNTRY_COLOUR = numpy.array([139., 69., 19.]) / 255
DEFAULT_STATE_PROVINCE_COLOUR = numpy.array([139., 69., 19.]) / 255
DEFAULT_COUNTY_COLOUR = numpy.array([152., 152., 152.]) / 255
DEFAULT_COAST_COLOUR = numpy.array([31., 120., 180.]) / 255
DEFAULT_RIVER_COLOUR = numpy.array([166., 206., 227.]) / 255
DEFAULT_PARALLEL_MERIDIAN_COLOUR = numpy.array([0., 0., 0.]) / 255

Z_ORDER_MERIDIANS_AND_PARALLELS = -1e6
Z_ORDER_RIVERS = -1e5
Z_ORDER_COAST = -1e4
Z_ORDER_COUNTRIES = -1000
Z_ORDER_STATES_AND_PROVINCES = -100
Z_ORDER_COUNTIES = -10

ELLIPSOID = 'sphere'
EARTH_RADIUS_METRES = 6370997.

# Constants for LCC (Lambert conformal conic) projection.
DEFAULT_LCC_STANDARD_LATS_DEG = numpy.array([25., 25.])
DEFAULT_LCC_CENTRAL_LNG_DEG = 265.

PADDING_FOR_VERTICAL_COLOUR_BAR = 0.05
PADDING_FOR_HORIZ_COLOUR_BAR = 0.075
DEFAULT_COLOUR_BAR_ORIENTATION = 'horizontal'

DEFAULT_ANNOT_FONT_SIZE = 50
DEFAULT_ANNOT_FONT_COLOUR = numpy.full(3, 0.)
DEFAULT_ANNOT_X_IMAGE_RELATIVE = 0.
DEFAULT_ANNOT_Y_IMAGE_RELATIVE = 1.

FONT_SIZE = 30
pyplot.rc('font', size=FONT_SIZE)
pyplot.rc('axes', titlesize=FONT_SIZE)
pyplot.rc('axes', labelsize=FONT_SIZE)
pyplot.rc('xtick', labelsize=FONT_SIZE)
pyplot.rc('ytick', labelsize=FONT_SIZE)
pyplot.rc('legend', fontsize=FONT_SIZE)
pyplot.rc('figure', titlesize=FONT_SIZE)


def annotate_axes(
        axes_object, annotation_string, font_size=DEFAULT_ANNOT_FONT_SIZE,
        font_colour=DEFAULT_ANNOT_FONT_COLOUR,
        x_coord_image_relative=DEFAULT_ANNOT_X_IMAGE_RELATIVE,
        y_coord_image_relative=DEFAULT_ANNOT_Y_IMAGE_RELATIVE):
    """Adds text annotation to axes.

    :param axes_object: Instance of `matplotlib.axes._subplots.AxesSubplot`.
    :param annotation_string: Text annotation.
    :param font_size: Font size.
    :param font_colour: Font colour (in any format accepted by matplotlib).
    :param x_coord_image_relative: Image-relative x-coordinate (from 0...1,
        where 1 is the right side).
    :param y_coord_image_relative: Image-relative y-coordinate (from 0...1,
        where 1 is the top).
    """

    axes_object.text(
        x_coord_image_relative, y_coord_image_relative, annotation_string,
        fontsize=font_size, color=font_colour, horizontalalignment='right',
        verticalalignment='bottom', transform=axes_object.transAxes)


def init_panels(num_panel_rows, num_panel_columns, figure_width_inches,
                figure_height_inches):
    """Initializes paneled figure.

    :param num_panel_rows: Number of panel rows.
    :param num_panel_columns: Number of panel columns.
    :param figure_width_inches: Figure width.
    :param figure_height_inches: Figure height.
    :return: figure_object: Instance of `matplotlib.figure.Figure`.
    :return: axes_objects_2d_list: 2-D list, where axes_objects_2d_list[j][k] is
        a `matplotlib.axes._subplots.AxesSubplot` object for the [j]th row and
        [k]th column.
    """

    figure_object, axes_objects_2d_list = pyplot.subplots(
        num_panel_rows, num_panel_columns, sharex=True, sharey=True,
        figsize=(figure_width_inches, figure_height_inches))

    if num_panel_rows == num_panel_columns == 1:
        axes_objects_2d_list = [[axes_objects_2d_list]]
    elif num_panel_columns == 1:
        axes_objects_2d_list = [[a] for a in axes_objects_2d_list]
    elif num_panel_rows == 1:
        axes_objects_2d_list = [axes_objects_2d_list]

    pyplot.subplots_adjust(
        left=0.02, bottom=0.02, right=0.98, top=0.95, hspace=0, wspace=0)

    for i in range(num_panel_rows):
        for j in range(num_panel_columns):
            axes_objects_2d_list[i][j].set(
                adjustable='box-forced', aspect='equal')

    return figure_object, axes_objects_2d_list


def init_lambert_conformal_map(
        min_latitude_deg, max_latitude_deg, min_longitude_deg,
        max_longitude_deg, standard_latitudes_deg=DEFAULT_LCC_STANDARD_LATS_DEG,
        central_longitude_deg=DEFAULT_LCC_CENTRAL_LNG_DEG,
        fig_width_inches=DEFAULT_FIG_WIDTH_INCHES,
        fig_height_inches=DEFAULT_FIG_HEIGHT_INCHES,
        resolution_string=DEFAULT_BOUNDARY_RESOLUTION_STRING):
    """Initializes map with LCC (Lambert conformal conic) projection.

    :param min_latitude_deg: Latitude at bottom-left corner (deg N).
    :param max_latitude_deg: Latitude at upper-right corner (deg N).
    :param min_longitude_deg: Longitude at bottom-left corner (deg E).
    :param max_longitude_deg: Longitude at upper-right corner (deg E).
    :param standard_latitudes_deg: length-2 numpy array of standard parallels
        (deg N).  standard_latitudes_deg[i] is the (i + 1)th standard parallel.
    :param central_longitude_deg: Central meridian (deg E).
    :param fig_width_inches: Figure width.
    :param fig_height_inches: Figure height.
    :param resolution_string: Resolution for boundaries (e.g., coastlines and
        political borders).  Options are "c" for crude, "l" for low, "i" for
        intermediate, "h" for high, and "f" for full.  Keep in mind that higher-
        resolution boundaries take much longer to draw.
    :return: figure_object: Instance of `matplotlib.figure.Figure`.
    :return: axes_object: Instance of `matplotlib.axes._subplots.AxesSubplot`.
    :return: basemap_object: Instance of `mpl_toolkits.basemap.Basemap`.
    """

    error_checking.assert_is_valid_lat_numpy_array(standard_latitudes_deg)
    error_checking.assert_is_numpy_array(
        standard_latitudes_deg, exact_dimensions=numpy.array([2]))

    error_checking.assert_is_non_array(central_longitude_deg)
    central_longitude_deg = lng_conversion.convert_lng_positive_in_west(
        central_longitude_deg)

    error_checking.assert_is_greater(fig_width_inches, 0)
    error_checking.assert_is_greater(fig_height_inches, 0)
    error_checking.assert_is_string(resolution_string)
    error_checking.assert_is_valid_latitude(min_latitude_deg)
    error_checking.assert_is_valid_latitude(max_latitude_deg)

    error_checking.assert_is_non_array(max_longitude_deg)
    error_checking.assert_is_non_array(min_longitude_deg)
    min_longitude_deg = lng_conversion.convert_lng_positive_in_west(
        min_longitude_deg)
    max_longitude_deg = lng_conversion.convert_lng_positive_in_west(
        max_longitude_deg)

    figure_object, axes_object = pyplot.subplots(
        1, 1, figsize=(fig_width_inches, fig_height_inches))

    basemap_object = Basemap(
        projection='lcc', lat_1=standard_latitudes_deg[0],
        lat_2=standard_latitudes_deg[1], lon_0=central_longitude_deg,
        rsphere=EARTH_RADIUS_METRES, ellps=ELLIPSOID,
        resolution=resolution_string, llcrnrlat=min_latitude_deg,
        llcrnrlon=min_longitude_deg, urcrnrlat=max_latitude_deg,
        urcrnrlon=max_longitude_deg)

    return figure_object, axes_object, basemap_object


def init_map_with_nwp_projection(
        model_name, min_latitude_deg, max_latitude_deg, min_longitude_deg,
        max_longitude_deg, grid_id=None,
        fig_width_inches=DEFAULT_FIG_WIDTH_INCHES,
        fig_height_inches=DEFAULT_FIG_HEIGHT_INCHES,
        resolution_string=DEFAULT_BOUNDARY_RESOLUTION_STRING):
    """Initializes map with NWP (numerical weather prediction)-model projection.

    If min_latitude_deg = max_latitude_deg = min_longitude_deg =
    max_longitude_deg = None, corners of the map will be corners of the model
    grid.

    :param model_name: Name of NWP model.
    :param min_latitude_deg: Latitude at bottom-left corner (deg N) of map.
    :param max_latitude_deg: Latitude at upper-right corner (deg N) of map.
    :param min_longitude_deg: Longitude at bottom-left corner (deg E) of map.
    :param max_longitude_deg: Longitude at upper-right corner (deg E) of map.
    :param grid_id: String ID for model grid.
    :param fig_width_inches: Figure width.
    :param fig_height_inches: Figure height.
    :param resolution_string: See documentation for init_lambert_conformal_map.
    :return: figure_object: Instance of `matplotlib.figure.Figure`.
    :return: axes_object: Instance of `matplotlib.axes._subplots.AxesSubplot`.
    :return: basemap_object: Instance of `mpl_toolkits.basemap.Basemap`.
    """

    nwp_model_utils.check_grid_id(model_name, grid_id)
    map_corners_deg = [
        min_latitude_deg, max_latitude_deg, min_longitude_deg,
        max_longitude_deg]

    if any([c is None for c in map_corners_deg]):
        grid_cell_edge_x_metres, grid_cell_edge_y_metres = (
            nwp_model_utils.get_xy_grid_cell_edges(model_name, grid_id))
        min_latitude_deg_as_array, min_longitude_deg_as_array = (
            nwp_model_utils.project_xy_to_latlng(
                numpy.array([grid_cell_edge_x_metres[0]]),
                numpy.array([grid_cell_edge_y_metres[0]]),
                projection_object=None, model_name=model_name, grid_id=grid_id))
        max_latitude_deg_as_array, max_longitude_deg_as_array = (
            nwp_model_utils.project_xy_to_latlng(
                numpy.array([grid_cell_edge_x_metres[-1]]),
                numpy.array([grid_cell_edge_y_metres[-1]]),
                projection_object=None, model_name=model_name, grid_id=grid_id))

        min_latitude_deg = min_latitude_deg_as_array[0]
        max_latitude_deg = max_latitude_deg_as_array[0]
        min_longitude_deg = min_longitude_deg_as_array[0]
        max_longitude_deg = max_longitude_deg_as_array[0]

    standard_latitudes_deg, central_longitude_deg = (
        nwp_model_utils.get_projection_params(model_name))

    return init_lambert_conformal_map(
        standard_latitudes_deg=standard_latitudes_deg,
        central_longitude_deg=central_longitude_deg,
        fig_width_inches=fig_width_inches, fig_height_inches=fig_height_inches,
        resolution_string=resolution_string, min_latitude_deg=min_latitude_deg,
        max_latitude_deg=max_latitude_deg, min_longitude_deg=min_longitude_deg,
        max_longitude_deg=max_longitude_deg)


def init_equidistant_cylindrical_map(
        min_latitude_deg, max_latitude_deg, min_longitude_deg,
        max_longitude_deg, fig_width_inches=DEFAULT_FIG_WIDTH_INCHES,
        fig_height_inches=DEFAULT_FIG_HEIGHT_INCHES,
        resolution_string=DEFAULT_BOUNDARY_RESOLUTION_STRING):
    """Initializes map with equidistant cylindrical projection.

    :param min_latitude_deg: Latitude at bottom-left corner (deg N).
    :param max_latitude_deg: Latitude at upper-right corner (deg N).
    :param min_longitude_deg: Longitude at bottom-left corner (deg E).
    :param max_longitude_deg: Longitude at upper-right corner (deg E).
    :param fig_width_inches: Figure width.
    :param fig_height_inches: Figure height.
    :param resolution_string: See documentation for init_lambert_conformal_map.
    :return: figure_object: Instance of `matplotlib.figure.Figure`.
    :return: axes_object: Instance of `matplotlib.axes._subplots.AxesSubplot`.
    :return: basemap_object: Instance of `mpl_toolkits.basemap.Basemap`.
    """

    error_checking.assert_is_greater(fig_width_inches, 0)
    error_checking.assert_is_greater(fig_height_inches, 0)
    error_checking.assert_is_string(resolution_string)
    error_checking.assert_is_valid_latitude(min_latitude_deg)
    error_checking.assert_is_valid_latitude(max_latitude_deg)

    error_checking.assert_is_non_array(max_longitude_deg)
    error_checking.assert_is_non_array(min_longitude_deg)
    min_longitude_deg = lng_conversion.convert_lng_positive_in_west(
        min_longitude_deg)
    max_longitude_deg = lng_conversion.convert_lng_positive_in_west(
        max_longitude_deg)

    figure_object, axes_object = pyplot.subplots(
        1, 1, figsize=(fig_width_inches, fig_height_inches))

    basemap_object = Basemap(
        projection='cyl', resolution=resolution_string,
        llcrnrlat=min_latitude_deg, llcrnrlon=min_longitude_deg,
        urcrnrlat=max_latitude_deg, urcrnrlon=max_longitude_deg)

    return figure_object, axes_object, basemap_object


def plot_countries(
        basemap_object, axes_object, line_width=DEFAULT_COUNTRY_WIDTH,
        line_colour=DEFAULT_COUNTRY_COLOUR):
    """Plots national borders.

    :param basemap_object: Instance of `mpl_toolkits.basemap.Basemap`.
    :param axes_object: Instance of `matplotlib.axes._subplots.AxesSubplot`.
    :param line_width: Line width (real positive number).
    :param line_colour: Colour (in any format accepted by `matplotlib.colors`).
    """

    basemap_object.drawcountries(
        linewidth=line_width, color=line_colour, ax=axes_object,
        zorder=Z_ORDER_COUNTRIES)


def plot_states_and_provinces(
        basemap_object, axes_object, line_width=DEFAULT_STATE_PROVINCE_WIDTH,
        line_colour=DEFAULT_STATE_PROVINCE_COLOUR):
    """Plots state and provincial borders.

    :param basemap_object: Instance of `mpl_toolkits.basemap.Basemap`.
    :param axes_object: Instance of `matplotlib.axes._subplots.AxesSubplot`.
    :param line_width: Line width (real positive number).
    :param line_colour: Colour (in any format accepted by `matplotlib.colors`).
    """

    basemap_object.drawstates(
        linewidth=line_width, color=line_colour, ax=axes_object,
        zorder=Z_ORDER_STATES_AND_PROVINCES)


def plot_counties(
        basemap_object, axes_object, line_width=DEFAULT_COUNTY_WIDTH,
        line_colour=DEFAULT_COUNTY_COLOUR):
    """Plots county borders.

    :param basemap_object: Instance of `mpl_toolkits.basemap.Basemap`.
    :param axes_object: Instance of `matplotlib.axes._subplots.AxesSubplot`.
    :param line_width: Line width (real positive number).
    :param line_colour: Colour (in any format accepted by `matplotlib.colors`).
    """

    basemap_object.drawcounties(
        linewidth=line_width, color=line_colour, ax=axes_object,
        zorder=Z_ORDER_COUNTIES)


def plot_coastlines(
        basemap_object, axes_object, line_width=DEFAULT_COAST_WIDTH,
        line_colour=DEFAULT_COAST_COLOUR):
    """Plots coastlines (with oceans and lakes).

    :param basemap_object: Instance of `mpl_toolkits.basemap.Basemap`.
    :param axes_object: Instance of `matplotlib.axes._subplots.AxesSubplot`.
    :param line_width: Line width (real positive number).
    :param line_colour: Colour (in any format accepted by `matplotlib.colors`).
    """

    basemap_object.drawcoastlines(
        linewidth=line_width, color=line_colour, ax=axes_object,
        zorder=Z_ORDER_COAST)


def plot_rivers(
        basemap_object, axes_object, line_width=DEFAULT_RIVER_WIDTH,
        line_colour=DEFAULT_RIVER_COLOUR):
    """Plots rivers.

    :param basemap_object: Instance of `mpl_toolkits.basemap.Basemap`.
    :param axes_object: Instance of `matplotlib.axes._subplots.AxesSubplot`.
    :param line_width: Line width (real positive number).
    :param line_colour: Colour (in any format accepted by `matplotlib.colors`).
    """

    basemap_object.drawrivers(
        linewidth=line_width, color=line_colour, ax=axes_object,
        zorder=Z_ORDER_RIVERS)


def plot_parallels(
        basemap_object, axes_object, bottom_left_lat_deg, upper_right_lat_deg,
        parallel_spacing_deg=DEFAULT_PARALLEL_SPACING_DEG,
        line_width=DEFAULT_PARALLEL_MERIDIAN_WIDTH,
        line_colour=DEFAULT_PARALLEL_MERIDIAN_COLOUR):
    """Draws parallels (lines of equal latitude).

    :param basemap_object: Instance of `mpl_toolkits.basemap.Basemap`.
    :param axes_object: Instance of `matplotlib.axes._subplots.AxesSubplot`.
    :param bottom_left_lat_deg: Latitude at bottom-left corner (deg N).
    :param upper_right_lat_deg: Latitude at upper-right corner (deg N).
    :param parallel_spacing_deg: Spacing between successive parallels (deg N).
    :param line_width: Line width (real positive number).
    :param line_colour: Colour (in any format accepted by `matplotlib.colors`).
    """

    error_checking.assert_is_valid_latitude(bottom_left_lat_deg)
    error_checking.assert_is_valid_latitude(upper_right_lat_deg)
    error_checking.assert_is_greater(upper_right_lat_deg, bottom_left_lat_deg)
    error_checking.assert_is_greater(parallel_spacing_deg, 0)

    min_parallel_deg = rounder.ceiling_to_nearest(
        bottom_left_lat_deg, parallel_spacing_deg)
    max_parallel_deg = rounder.floor_to_nearest(
        upper_right_lat_deg, parallel_spacing_deg)
    num_parallels_deg = int(
        1 + (max_parallel_deg - min_parallel_deg) / parallel_spacing_deg)
    parallels_deg = numpy.linspace(
        min_parallel_deg, max_parallel_deg, num=num_parallels_deg)
    basemap_object.drawparallels(
        parallels_deg, color=line_colour, linewidth=line_width,
        labels=[True, False, False, False], ax=axes_object,
        zorder=Z_ORDER_MERIDIANS_AND_PARALLELS)


def plot_meridians(
        basemap_object, axes_object, bottom_left_lng_deg, upper_right_lng_deg,
        meridian_spacing_deg=DEFAULT_MERIDIAN_SPACING_DEG,
        line_width=DEFAULT_PARALLEL_MERIDIAN_WIDTH,
        line_colour=DEFAULT_PARALLEL_MERIDIAN_COLOUR):
    """Draws meridians (lines of equal longitude).

    :param basemap_object: Instance of `mpl_toolkits.basemap.Basemap`.
    :param axes_object: Instance of `matplotlib.axes._subplots.AxesSubplot`.
    :param bottom_left_lng_deg: Longitude at bottom-left corner (deg E).
    :param upper_right_lng_deg: Longitude at upper-right corner (deg E).
    :param meridian_spacing_deg: Spacing between successive meridians (deg N).
    :param line_width: Line width (real positive number).
    :param line_colour: Colour (in any format accepted by `matplotlib.colors`).
    """

    bottom_left_lng_deg = lng_conversion.convert_lng_positive_in_west(
        bottom_left_lng_deg)
    upper_right_lng_deg = lng_conversion.convert_lng_positive_in_west(
        upper_right_lng_deg)

    error_checking.assert_is_greater(upper_right_lng_deg, bottom_left_lng_deg)
    error_checking.assert_is_greater(meridian_spacing_deg, 0)

    min_meridian_deg = rounder.ceiling_to_nearest(
        bottom_left_lng_deg, meridian_spacing_deg)
    max_meridian_deg = rounder.floor_to_nearest(
        upper_right_lng_deg, meridian_spacing_deg)
    num_meridians_deg = int(
        1 + (max_meridian_deg - min_meridian_deg) / meridian_spacing_deg)
    meridians_deg = numpy.linspace(
        min_meridian_deg, max_meridian_deg, num=num_meridians_deg)
    basemap_object.drawmeridians(
        meridians_deg, color=line_colour, linewidth=line_width,
        labels=[False, False, False, True], ax=axes_object,
        zorder=Z_ORDER_MERIDIANS_AND_PARALLELS)


def add_colour_bar(
        axes_object_or_list, values_to_colour, colour_map, colour_norm_object,
        orientation=DEFAULT_COLOUR_BAR_ORIENTATION, extend_min=True,
        extend_max=True, fraction_of_axis_length=1., font_size=FONT_SIZE):
    """Adds colour bar to existing plot.

    :param axes_object_or_list: `matplotlib.axes._subplots.AxesSubplot` object
        or list thereof.
    :param values_to_colour: numpy array of values to which the colour map will
        be applied.
    :param colour_map: Instance of `matplotlib.pyplot.cm`.
    :param colour_norm_object: Instance of `matplotlib.colors.Normalize`.
    :param orientation: Orientation (either "horizontal" or "vertical").
    :param extend_min: Boolean flag.  If extend_min = True, will add arrow to
        bottom end of colour bar.  Otherwise, bottom of colour bar will be
        rectangular.
    :param extend_max: Same as extend_min, but for upper end of colour bar.
    :param fraction_of_axis_length: The colour bar will have this fraction of
        the length of the axis that it parallels.  For example, if the colour
        bar is vertical, it will have this fraction of the y-axis length.
    :param font_size: Font size.
    :return: colour_bar_object: Instance of `matplotlib.pyplot.colorbar` created
        by this method.
    """

    error_checking.assert_is_real_numpy_array(values_to_colour)
    error_checking.assert_is_boolean(extend_min)
    error_checking.assert_is_boolean(extend_max)
    error_checking.assert_is_greater(fraction_of_axis_length, 0.)
    error_checking.assert_is_leq(fraction_of_axis_length, 1.)

    scalar_mappable_object = pyplot.cm.ScalarMappable(
        cmap=colour_map, norm=colour_norm_object)
    scalar_mappable_object.set_array(values_to_colour)

    if extend_min and extend_max:
        extend_argument = 'both'
    elif extend_min:
        extend_argument = 'min'
    elif extend_max:
        extend_argument = 'max'
    else:
        extend_argument = 'neither'

    if orientation == 'horizontal':
        this_padding = PADDING_FOR_HORIZ_COLOUR_BAR
    else:
        this_padding = PADDING_FOR_VERTICAL_COLOUR_BAR

    colour_bar_object = pyplot.colorbar(
        ax=axes_object_or_list, mappable=scalar_mappable_object,
        orientation=orientation, pad=this_padding, extend=extend_argument,
        shrink=fraction_of_axis_length)

    colour_bar_object.ax.tick_params(labelsize=font_size)

    if orientation == 'horizontal':
        colour_bar_object.ax.set_xticklabels(
            colour_bar_object.ax.get_xticklabels(), rotation=90)

    return colour_bar_object


def add_linear_colour_bar(
        axes_object_or_list, values_to_colour, colour_map, colour_min,
        colour_max, orientation=DEFAULT_COLOUR_BAR_ORIENTATION, extend_min=True,
        extend_max=True, fraction_of_axis_length=1., font_size=FONT_SIZE):
    """Adds linear colour bar to existing plot.

    :param axes_object_or_list: `matplotlib.axes._subplots.AxesSubplot` object
        or list thereof.
    :param values_to_colour: numpy array of values to which the colour map will
        be applied.
    :param colour_map: Instance of `matplotlib.pyplot.cm`.
    :param colour_min: Minimum value for colour map.
    :param colour_max: Maximum value for colour map.
    :param orientation: Orientation (either "horizontal" or "vertical").
    :param extend_min: See doc for `add_colour_bar`.
    :param extend_max: Same.
    :param fraction_of_axis_length: Same.
    :param font_size: Font size.
    :return: colour_bar_object: Instance of `matplotlib.pyplot.colorbar` created
        by this method.
    """

    error_checking.assert_is_greater(colour_max, colour_min)
    colour_norm_object = matplotlib.colors.Normalize(
        vmin=colour_min, vmax=colour_max, clip=False)

    return add_colour_bar(
        axes_object_or_list, values_to_colour=values_to_colour,
        colour_map=colour_map, colour_norm_object=colour_norm_object,
        orientation=orientation, extend_min=extend_min, extend_max=extend_max,
        fraction_of_axis_length=fraction_of_axis_length, font_size=font_size)
