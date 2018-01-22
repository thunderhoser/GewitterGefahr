"""Plotting methods for wind."""

import numpy
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as pyplot
from gewittergefahr.gg_utils import longitude_conversion as lng_conversion
from gewittergefahr.gg_utils import error_checking

DEFAULT_BARB_LENGTH = 6
DEFAULT_EMPTY_BARB_RADIUS = 0.25
FILL_EMPTY_BARB_DEFAULT = False
DEFAULT_COLOUR_MAP = pyplot.cm.rainbow
DEFAULT_COLOUR_MINIMUM_KT = 0.
DEFAULT_COLOUR_MAXIMUM_KT = 50.

METRES_PER_SECOND_TO_KT = 3.6 / 1.852


def plot_wind_barbs(
        basemap_object=None, axes_object=None,
        latitudes_deg=None, longitudes_deg=None, u_winds_m_s01=None,
        v_winds_m_s01=None, barb_length=DEFAULT_BARB_LENGTH,
        empty_barb_radius=DEFAULT_EMPTY_BARB_RADIUS,
        fill_empty_barb=FILL_EMPTY_BARB_DEFAULT, colour_map=DEFAULT_COLOUR_MAP,
        colour_minimum_kt=DEFAULT_COLOUR_MINIMUM_KT,
        colour_maximum_kt=DEFAULT_COLOUR_MAXIMUM_KT):
    """Plots wind barbs.

    N = number of wind barbs

    :param basemap_object: Instance of `mpl_toolkits.basemap.Basemap`.
    :param axes_object: Instance of `matplotlib.axes._subplots.AxesSubplot`.
    :param latitudes_deg: length-N numpy array of latitudes (deg N).
    :param longitudes_deg: length-N numpy array of longitudes (deg E).
    :param u_winds_m_s01: length-N numpy array of eastward wind velocities
        (m/s).
    :param v_winds_m_s01: length-N numpy array of northward wind velocities
        (m/s).
    :param barb_length: Length of each wind barb.
    :param empty_barb_radius: Radius of circle for 0-metre-per-second wind barb.
    :param fill_empty_barb: Boolean flag.  If fill_empty_barb = True,
        0-metre-per-second wind barb will be a filled circle.  Otherwise, it
        will be an empty circle.
    :param colour_map: Instance of `matplotlib.pyplot.cm`.
    :param colour_minimum_kt: Minimum speed for colour map (kt or nautical miles
        per hour).
    :param colour_maximum_kt: Maximum speed for colour map (kt or nautical miles
        per hour).
    """

    error_checking.assert_is_valid_lat_numpy_array(latitudes_deg)
    error_checking.assert_is_numpy_array(latitudes_deg, num_dimensions=1)
    num_points = len(latitudes_deg)

    longitudes_deg = lng_conversion.convert_lng_positive_in_west(longitudes_deg)
    error_checking.assert_is_numpy_array(
        longitudes_deg, exact_dimensions=numpy.array([num_points]))

    error_checking.assert_is_numpy_array_without_nan(u_winds_m_s01)
    error_checking.assert_is_numpy_array(
        u_winds_m_s01, exact_dimensions=numpy.array([num_points]))

    error_checking.assert_is_numpy_array_without_nan(v_winds_m_s01)
    error_checking.assert_is_numpy_array(
        v_winds_m_s01, exact_dimensions=numpy.array([num_points]))

    error_checking.assert_is_geq(colour_minimum_kt, 0.)
    error_checking.assert_is_greater(colour_maximum_kt, colour_minimum_kt)

    x_coords_metres, y_coords_metres = basemap_object(
        longitudes_deg, latitudes_deg)

    size_dict = {'emptybarb': empty_barb_radius}
    colour_limits_kt = numpy.array([colour_minimum_kt, colour_maximum_kt])
    wind_speeds_m_s01 = numpy.sqrt(u_winds_m_s01 ** 2 + v_winds_m_s01 ** 2)

    axes_object.barbs(
        x_coords_metres, y_coords_metres,
        u_winds_m_s01 * METRES_PER_SECOND_TO_KT,
        v_winds_m_s01 * METRES_PER_SECOND_TO_KT,
        wind_speeds_m_s01 * METRES_PER_SECOND_TO_KT, length=barb_length,
        sizes=size_dict, fill_empty=fill_empty_barb, rounding=False,
        cmap=colour_map, clim=colour_limits_kt)
