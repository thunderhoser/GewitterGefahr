"""Helper methods for geographic maps."""

import numpy
import matplotlib.pyplot as pyplot
from mpl_toolkits.basemap import Basemap
from gewittergefahr.gg_utils import number_rounding as rounder
from gewittergefahr.gg_utils import longitude_conversion as lng_conversion
from gewittergefahr.gg_utils import error_checking

# TODO(thunderhoser): When manipulating a figure, should pass around
# `matplotlib.axes.Axes` objects, not `mpl_toolkits.basemap.Basemap`.

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

Z_ORDER_MERIDIANS_AND_PARALLELS = -100
Z_ORDER_RIVERS = -90
Z_ORDER_COAST = -80
Z_ORDER_COUNTRIES = -70
Z_ORDER_STATES_AND_PROVINCES = -60
Z_ORDER_COUNTIES = -50

ELLIPSOID = 'sphere'
EARTH_RADIUS_METRES = 6370997.

# Constants for LCC (Lambert conformal conic) projection.
DEFAULT_LCC_STANDARD_LATS_DEG = numpy.array([25., 25.])
DEFAULT_LCC_CENTRAL_LNG_DEG = 265.

FONT_SIZE = 30
pyplot.rc('font', size=FONT_SIZE)
pyplot.rc('axes', titlesize=FONT_SIZE)
pyplot.rc('axes', labelsize=FONT_SIZE)
pyplot.rc('xtick', labelsize=FONT_SIZE)
pyplot.rc('ytick', labelsize=FONT_SIZE)
pyplot.rc('legend', fontsize=FONT_SIZE)
pyplot.rc('figure', titlesize=FONT_SIZE)


def init_lambert_conformal_map(
        standard_latitudes_deg=None, central_longitude_deg=None,
        fig_width_inches=DEFAULT_FIG_WIDTH_INCHES,
        fig_height_inches=DEFAULT_FIG_HEIGHT_INCHES,
        resolution_string=DEFAULT_BOUNDARY_RESOLUTION_STRING,
        min_latitude_deg=None, max_latitude_deg=None, min_longitude_deg=None,
        max_longitude_deg=None):
    """Initializes map with LCC (Lambert conformal conic) projection.

    :param standard_latitudes_deg: length-2 numpy array of standard parallels
        (deg N).  standard_latitudes_deg[i] is the (i + 1)th standard parallel.
    :param central_longitude_deg: Central meridian (deg E).
    :param fig_width_inches: Figure width.
    :param fig_height_inches: Figure height.
    :param resolution_string: Resolution for boundaries (e.g., coastlines and
        political borders).  Options are "c" for crude, "l" for low, "i" for
        intermediate, "h" for high, and "f" for full.  Keep in mind that higher-
        resolution boundaries take much longer to draw.
    :param min_latitude_deg: Latitude at bottom-left corner (deg N).
    :param max_latitude_deg: Latitude at upper-right corner (deg N).
    :param min_longitude_deg: Longitude at bottom-left corner (deg E).
    :param max_longitude_deg: Longitude at upper-right corner (deg E).
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

    pyplot.figure(figsize=(fig_width_inches, fig_height_inches))

    return Basemap(
        projection='lcc', lat_1=standard_latitudes_deg[0],
        lat_2=standard_latitudes_deg[1], lon_0=central_longitude_deg,
        rsphere=EARTH_RADIUS_METRES, ellps=ELLIPSOID,
        resolution=resolution_string, llcrnrlat=min_latitude_deg,
        llcrnrlon=min_longitude_deg, urcrnrlat=max_latitude_deg,
        urcrnrlon=max_longitude_deg)


def plot_countries(basemap_object, line_width=DEFAULT_COUNTRY_WIDTH,
                   line_colour=DEFAULT_COUNTRY_COLOUR):
    """Plots national borders.

    :param basemap_object: Instance of `mpl_toolkits.basemap.Basemap`.
    :param line_width: Line width (real positive number).
    :param line_colour: Colour (in any format accepted by `matplotlib.colors`).
    """

    basemap_object.drawcountries(
        linewidth=line_width, color=line_colour, zorder=Z_ORDER_COUNTRIES)


def plot_states_and_provinces(basemap_object,
                              line_width=DEFAULT_STATE_PROVINCE_WIDTH,
                              line_colour=DEFAULT_STATE_PROVINCE_COLOUR):
    """Plots state and provincial borders.

    :param basemap_object: Instance of `mpl_toolkits.basemap.Basemap`.
    :param line_width: Line width (real positive number).
    :param line_colour: Colour (in any format accepted by `matplotlib.colors`).
    """

    basemap_object.drawstates(
        linewidth=line_width, color=line_colour,
        zorder=Z_ORDER_STATES_AND_PROVINCES)


def plot_counties(basemap_object, line_width=DEFAULT_COUNTY_WIDTH,
                  line_colour=DEFAULT_COUNTY_COLOUR):
    """Plots county borders.

    :param basemap_object: Instance of `mpl_toolkits.basemap.Basemap`.
    :param line_width: Line width (real positive number).
    :param line_colour: Colour (in any format accepted by `matplotlib.colors`).
    """

    basemap_object.drawcounties(
        linewidth=line_width, color=line_colour, zorder=Z_ORDER_COUNTIES)


def plot_coastlines(basemap_object, line_width=DEFAULT_COAST_WIDTH,
                    line_colour=DEFAULT_COAST_COLOUR):
    """Plots coastlines (with oceans and lakes).

    :param basemap_object: Instance of `mpl_toolkits.basemap.Basemap`.
    :param line_width: Line width (real positive number).
    :param line_colour: Colour (in any format accepted by `matplotlib.colors`).
    """

    basemap_object.drawcoastlines(
        linewidth=line_width, color=line_colour, zorder=Z_ORDER_COAST)


def plot_rivers(basemap_object, line_width=DEFAULT_RIVER_WIDTH,
                line_colour=DEFAULT_RIVER_COLOUR):
    """Plots rivers.

    :param basemap_object: Instance of `mpl_toolkits.basemap.Basemap`.
    :param line_width: Line width (real positive number).
    :param line_colour: Colour (in any format accepted by `matplotlib.colors`).
    """

    basemap_object.drawrivers(
        linewidth=line_width, color=line_colour, zorder=Z_ORDER_RIVERS)


def plot_parallels(basemap_object, bottom_left_lat_deg=None,
                   upper_right_lat_deg=None,
                   parallel_spacing_deg=DEFAULT_PARALLEL_SPACING_DEG,
                   line_width=DEFAULT_PARALLEL_MERIDIAN_WIDTH,
                   line_colour=DEFAULT_PARALLEL_MERIDIAN_COLOUR):
    """Draws parallels (lines of equal latitude).

    :param basemap_object: Instance of `mpl_toolkits.basemap.Basemap`.
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
        parallels_deg, labels=[True, False, False, False], linewidth=line_width,
        color=line_colour, zorder=Z_ORDER_MERIDIANS_AND_PARALLELS)


def plot_meridians(basemap_object, bottom_left_lng_deg=None,
                   upper_right_lng_deg=None,
                   meridian_spacing_deg=DEFAULT_MERIDIAN_SPACING_DEG,
                   line_width=DEFAULT_PARALLEL_MERIDIAN_WIDTH,
                   line_colour=DEFAULT_PARALLEL_MERIDIAN_COLOUR):
    """Draws meridians (lines of equal longitude).

    :param basemap_object: Instance of `mpl_toolkits.basemap.Basemap`.
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
        meridians_deg, labels=[False, False, False, True], linewidth=line_width,
        color=line_colour, zorder=Z_ORDER_MERIDIANS_AND_PARALLELS)
