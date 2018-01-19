"""Plotting methods for radar data.

--- DEFINITIONS ---

MYRORSS = Multi-year Reanalysis of Remotely Sensed Storms

MRMS = Multi-radar Multi-sensor
"""

import numpy
import matplotlib.pyplot as pyplot
import matplotlib.colors
from gewittergefahr.gg_utils import grids
from gewittergefahr.gg_utils import radar_utils
from gewittergefahr.gg_utils import error_checking

REFLECTIVITY_PLOTTING_UNITS = 'dBZ'
SHEAR_PLOTTING_UNITS = 's^-1'
ECHO_TOP_PLOTTING_UNITS = 'kft'
MESH_PLOTTING_UNITS = 'mm'
SHI_PLOTTING_UNITS = ''
VIL_PLOTTING_UNITS = 'mm'

KM_TO_KFT = 3.2808  # kilometres to kilofeet


def _get_default_refl_colour_map():
    """Returns default colour map for reflectivity.

    :return: colour_map_object: Instance of `matplotlib.colors.ListedColormap`.
    :return: colour_norm_object: Instance of `matplotlib.colors.BoundaryNorm`.
    :return: colour_bounds_dbz: See documentation for _get_default_colour_map.
        Units in this case are dBZ, or decibels of reflectivity.
    """

    main_colour_list = [
        numpy.array([4., 233., 231.]), numpy.array([1., 159., 244.]),
        numpy.array([3., 0., 244.]), numpy.array([2., 253., 2.]),
        numpy.array([1., 197., 1.]), numpy.array([0., 142., 0.]),
        numpy.array([253., 248., 2.]), numpy.array([229., 188., 0.]),
        numpy.array([253., 149., 0.]), numpy.array([253., 0., 0.]),
        numpy.array([212., 0., 0.]), numpy.array([188., 0., 0.]),
        numpy.array([248., 0., 253.]), numpy.array([152., 84., 198.])]

    for i in range(len(main_colour_list)):
        main_colour_list[i] /= 255

    colour_map_object = matplotlib.colors.ListedColormap(main_colour_list)
    colour_map_object.set_under(numpy.array([1., 1., 1.]))
    colour_map_object.set_over(numpy.array([1., 1., 1.]))

    main_colour_bounds_dbz = numpy.array(
        [0.1, 5., 10., 15., 20., 25., 30., 35., 40., 45., 50., 55., 60., 65.,
         70.])
    colour_norm_object = matplotlib.colors.BoundaryNorm(
        main_colour_bounds_dbz, colour_map_object.N)

    colour_bounds_dbz = numpy.concatenate((
        numpy.array([0.]), main_colour_bounds_dbz, numpy.array([100.])))
    return colour_map_object, colour_norm_object, colour_bounds_dbz


def _get_default_shear_colour_map():
    """Returns default colour map for azimuthal shear.

    :return: colour_map_object: Instance of `matplotlib.colors.ListedColormap`.
    :return: colour_norm_object: Instance of `matplotlib.colors.BoundaryNorm`.
    :return: colour_bounds_s01: See documentation for _get_default_colour_map.
        Units in this case are inverse seconds.
    """

    main_colour_list = [
        numpy.array([152., 50., 203.]), numpy.array([0., 45., 254.]),
        numpy.array([0., 152., 254.]), numpy.array([152., 203., 254.]),
        numpy.array([255., 255., 255.]), numpy.array([0., 101., 0.]),
        numpy.array([0., 152., 0.]), numpy.array([0., 203., 0.]),
        numpy.array([0., 254., 101.]), numpy.array([254., 254., 50.]),
        numpy.array([254., 203., 0.]), numpy.array([254., 152., 0.]),
        numpy.array([254., 101., 0.]), numpy.array([254., 0., 0.]),
        numpy.array([254., 0., 152.])]

    for i in range(len(main_colour_list)):
        main_colour_list[i] /= 255

    colour_map_object = matplotlib.colors.ListedColormap(main_colour_list)
    colour_map_object.set_under(numpy.array([101., 0., 152.]) / 255)
    colour_map_object.set_over(numpy.array([1., 1., 1.]))

    main_colour_bounds_s01 = numpy.array(
        [-10., -7.5, -5., -3., -1., 1., 2.5, 4., 5.5, 7., 8.5, 10., 12.5, 15.,
         17.5, 20.]) / 1000
    colour_norm_object = matplotlib.colors.BoundaryNorm(
        main_colour_bounds_s01, colour_map_object.N)

    colour_bounds_s01 = numpy.concatenate((
        numpy.array([-0.1]), main_colour_bounds_s01, numpy.array([0.1])))
    return colour_map_object, colour_norm_object, colour_bounds_s01


def _get_default_echo_top_colour_map():
    """Returns default colour map for echo tops.

    :return: colour_map_object: Instance of `matplotlib.colors.ListedColormap`.
    :return: colour_norm_object: Instance of `matplotlib.colors.BoundaryNorm`.
    :return: colour_bounds_kft: See documentation for _get_default_colour_map.
        Units in this case are kilofeet (why, America, why?).
    """

    main_colour_list = [
        numpy.array([120., 120., 120.]), numpy.array([16., 220., 244.]),
        numpy.array([11., 171., 247.]), numpy.array([9., 144., 202.]),
        numpy.array([48., 6., 134.]), numpy.array([4., 248., 137.]),
        numpy.array([10., 185., 6.]), numpy.array([1., 241., 8.]),
        numpy.array([255., 186., 1.]), numpy.array([255., 251., 0.]),
        numpy.array([132., 17., 22.]), numpy.array([233., 16., 1.])]

    for i in range(len(main_colour_list)):
        main_colour_list[i] /= 255

    colour_map_object = matplotlib.colors.ListedColormap(main_colour_list)
    colour_map_object.set_under(numpy.array([1., 1., 1.]))
    colour_map_object.set_over(numpy.array([1., 1., 1.]))

    main_colour_bounds_kft = numpy.array(
        [0.1, 10., 15., 20., 25., 30., 35., 40., 45., 50., 55., 60., 65.])
    colour_norm_object = matplotlib.colors.BoundaryNorm(
        main_colour_bounds_kft, colour_map_object.N)

    colour_bounds_kft = numpy.concatenate((
        numpy.array([0.]), main_colour_bounds_kft, numpy.array([100.])))
    return colour_map_object, colour_norm_object, colour_bounds_kft


def _get_default_mesh_colour_map():
    """Returns default colour map for MESH (maximum estimated hail size).

    :return: colour_map_object: Instance of `matplotlib.colors.ListedColormap`.
    :return: colour_norm_object: Instance of `matplotlib.colors.BoundaryNorm`.
    :return: colour_bounds_mm: See documentation for _get_default_colour_map.
        Units in this case are mm.
    """

    main_colour_list = [
        numpy.array([152., 152., 152.]), numpy.array([152., 203., 254.]),
        numpy.array([0., 152., 254.]), numpy.array([0., 45., 254.]),
        numpy.array([0., 101., 0.]), numpy.array([0., 152., 0.]),
        numpy.array([0., 203., 0.]), numpy.array([254., 254., 50.]),
        numpy.array([254., 203., 0.]), numpy.array([254., 152., 0.]),
        numpy.array([254., 0., 0.]), numpy.array([254., 0., 152.]),
        numpy.array([152., 50., 203.])]

    for i in range(len(main_colour_list)):
        main_colour_list[i] /= 255

    colour_map_object = matplotlib.colors.ListedColormap(main_colour_list)
    colour_map_object.set_under(numpy.array([1., 1., 1.]))
    colour_map_object.set_over(numpy.array([1., 1., 1.]))

    main_colour_bounds_mm = numpy.array(
        [0.1, 15.9, 22.2, 28.6, 34.9, 41.3, 47.6, 54., 60.3, 65., 70., 75., 80.,
         85.])
    colour_norm_object = matplotlib.colors.BoundaryNorm(
        main_colour_bounds_mm, colour_map_object.N)

    colour_bounds_mm = numpy.concatenate((
        numpy.array([0.]), main_colour_bounds_mm, numpy.array([200.])))
    return colour_map_object, colour_norm_object, colour_bounds_mm


def _get_default_shi_colour_map():
    """Returns default colour map for SHI (severe-hail index).

    :return: colour_map_object: Instance of `matplotlib.colors.ListedColormap`.
    :return: colour_norm_object: Instance of `matplotlib.colors.BoundaryNorm`.
    :return: colour_bounds: See documentation for _get_default_colour_map.
        Units in this case are dimensionless.
    """

    main_colour_list = [
        numpy.array([152., 152., 152.]), numpy.array([152., 203., 254.]),
        numpy.array([0., 152., 254.]), numpy.array([0., 45., 254.]),
        numpy.array([0., 101., 0.]), numpy.array([0., 152., 0.]),
        numpy.array([0., 203., 0.]), numpy.array([254., 254., 50.]),
        numpy.array([254., 203., 0.]), numpy.array([254., 152., 0.]),
        numpy.array([254., 0., 0.]), numpy.array([254., 0., 152.]),
        numpy.array([152., 50., 203.]), numpy.array([101., 0., 152.])]

    for i in range(len(main_colour_list)):
        main_colour_list[i] /= 255

    colour_map_object = matplotlib.colors.ListedColormap(main_colour_list)
    colour_map_object.set_under(numpy.array([1., 1., 1.]))
    colour_map_object.set_over(numpy.array([1., 1., 1.]))

    main_colour_bounds = numpy.array(
        [1., 30., 60., 90., 120., 150., 180., 210., 240., 270., 300., 330.,
         360., 390., 420.])
    colour_norm_object = matplotlib.colors.BoundaryNorm(
        main_colour_bounds, colour_map_object.N)

    colour_bounds = numpy.concatenate((
        numpy.array([0.]), main_colour_bounds, numpy.array([1000.])))
    return colour_map_object, colour_norm_object, colour_bounds


def _get_default_vil_colour_map():
    """Returns default colour map for VIL (vertically integrated liquid).

    :return: colour_map_object: Instance of `matplotlib.colors.ListedColormap`.
    :return: colour_norm_object: Instance of `matplotlib.colors.BoundaryNorm`.
    :return: colour_bounds_mm: See documentation for _get_default_colour_map.
        Units in this case are mm.
    """

    main_colour_list = [
        numpy.array([16., 71., 101.]), numpy.array([0., 99., 132.]),
        numpy.array([46., 132., 181.]), numpy.array([74., 166., 218.]),
        numpy.array([122., 207., 255.]), numpy.array([179., 0., 179.]),
        numpy.array([222., 83., 222.]), numpy.array([255., 136., 255.]),
        numpy.array([253., 191., 253.]), numpy.array([255., 96., 0.]),
        numpy.array([255., 128., 32.]), numpy.array([255., 208., 0.]),
        numpy.array([180., 0., 0.]), numpy.array([224., 0., 0.])]

    for i in range(len(main_colour_list)):
        main_colour_list[i] /= 255

    colour_map_object = matplotlib.colors.ListedColormap(main_colour_list)
    colour_map_object.set_under(numpy.array([1., 1., 1.]))
    colour_map_object.set_over(numpy.array([1., 1., 1.]))

    main_colour_bounds_mm = numpy.array(
        [0.1, 5., 10., 15., 20., 25., 30., 35., 40., 45., 50., 55., 60., 65.,
         70.])
    colour_norm_object = matplotlib.colors.BoundaryNorm(
        main_colour_bounds_mm, colour_map_object.N)

    colour_bounds_mm = numpy.concatenate((
        numpy.array([0.]), main_colour_bounds_mm, numpy.array([200.])))
    return colour_map_object, colour_norm_object, colour_bounds_mm


def _get_default_colour_map(field_name):
    """Returns default colour map for the given radar field.

    N = number of colours

    :param field_name: Name of radar field (string).
    :return: colour_map_object: Instance of `matplotlib.colors.ListedColormap`.
    :return: colour_norm_object: Instance of `matplotlib.colors.BoundaryNorm`.
    :return: colour_bounds: length-(N + 1) numpy array of colour boundaries.
        colour_bounds[0] and colour_bounds[1] are the boundaries for the 1st
        colour; colour_bounds[1] and colour_bounds[2] are the boundaries for the
        2nd colour; ...; colour_bounds[i] and colour_bounds[i + 1] are the
        boundaries for the (i + 1)th colour.
    """

    if field_name in radar_utils.REFLECTIVITY_NAMES:
        return _get_default_refl_colour_map()

    if field_name in radar_utils.SHEAR_NAMES:
        return _get_default_shear_colour_map()

    if field_name in radar_utils.ECHO_TOP_NAMES:
        return _get_default_echo_top_colour_map()

    if field_name == radar_utils.MESH_NAME:
        return _get_default_mesh_colour_map()

    if field_name == radar_utils.SHI_NAME:
        return _get_default_shi_colour_map()

    if field_name == radar_utils.VIL_NAME:
        return _get_default_vil_colour_map()


def _get_plotting_units(field_name):
    """Returns plotting units of radar field.

    :param field_name: Name of radar field (string).
    :return: plotting_units: String describing units.
    """

    if field_name in radar_utils.REFLECTIVITY_NAMES:
        return REFLECTIVITY_PLOTTING_UNITS

    if field_name in radar_utils.SHEAR_NAMES:
        return SHEAR_PLOTTING_UNITS

    if field_name in radar_utils.ECHO_TOP_NAMES:
        return ECHO_TOP_PLOTTING_UNITS

    if field_name == radar_utils.MESH_NAME:
        return MESH_PLOTTING_UNITS

    if field_name == radar_utils.SHI_NAME:
        return SHI_PLOTTING_UNITS

    if field_name == radar_utils.VIL_NAME:
        return VIL_PLOTTING_UNITS


def _convert_to_plotting_units(field_matrix_gg_units, field_name):
    """Converts field from standard GewitterGefahr units to plotting units.

    :param field_matrix_gg_units: numpy array with values of radar field (in
        GewitterGefahr units, which are included in `field_name`).
    :param field_name: Name of radar field (string).
    :return: field_matrix_plotting_units: Same as input, but in plotting units.
    """

    if field_name in radar_utils.ECHO_TOP_NAMES:
        return field_matrix_gg_units * KM_TO_KFT

    return field_matrix_gg_units


def plot_latlng_grid(axes_object=None, field_name=None, field_matrix=None,
                     min_latitude_deg=None, min_longitude_deg=None,
                     lat_spacing_deg=None, lng_spacing_deg=None,
                     colour_map=None, colour_minimum=None, colour_maximum=None):
    """Plots lat-long grid of a single radar field.

    Because this method plots a lat-long grid, rather than an x-y grid, the
    projection used for the basemap must be cylindrical equidistant (which is
    the same as a lat-long projection).

    This is the most convenient way to plot radar data, since both MYRORSS and
    MRMS data are on a lat-long grid.

    M = number of rows (unique grid-point latitudes)
    N = number of columns (unique grid-point longitudes)

    :param axes_object: Instance of `matplotlib.axes._subplots.AxesSubplot`.
    :param field_name: Name of radar field.
    :param field_matrix: M-by-N numpy array with values of radar field.  This
        should be in standard GewitterGefahr units (will be converted to
        plotting units by convert_to_plotting_units).  Latitude should increase
        while traveling down a column, and longitude should increase while
        traveling right across a row.
    :param min_latitude_deg: Minimum latitude over all grid points (deg N).
    :param min_longitude_deg: Minimum longitude over all grid points (deg E).
    :param lat_spacing_deg: Meridional spacing between adjacent grid points.
    :param lng_spacing_deg: Zonal spacing between adjacent grid points.
    :param colour_map: Instance of `matplotlib.pyplot.cm`.
    :param colour_minimum: Minimum value for colour map.
    :param colour_maximum: Maximum value for colour map.
    """

    radar_utils.check_field_name(field_name)
    field_matrix = _convert_to_plotting_units(field_matrix, field_name)
    field_matrix = numpy.ma.masked_where(
        numpy.isnan(field_matrix), field_matrix)

    grid_point_latitudes_deg, grid_point_longitudes_deg = (
        grids.get_latlng_grid_points(
            min_latitude_deg=min_latitude_deg,
            min_longitude_deg=min_longitude_deg,
            lat_spacing_deg=lat_spacing_deg, lng_spacing_deg=lng_spacing_deg,
            num_rows=field_matrix.shape[0], num_columns=field_matrix.shape[1]))

    if colour_map is None:
        colour_map, colour_norm_object, _ = _get_default_colour_map(field_name)
        colour_minimum = colour_norm_object.boundaries[0]
        colour_maximum = colour_norm_object.boundaries[-1]
    else:
        error_checking.assert_is_greater(colour_maximum, colour_minimum)
        colour_norm_object = None

    pyplot.pcolormesh(
        grid_point_longitudes_deg, grid_point_latitudes_deg, field_matrix,
        cmap=colour_map, norm=colour_norm_object, vmin=colour_minimum,
        vmax=colour_maximum, shading='flat', edgecolors='None',
        axes=axes_object)
