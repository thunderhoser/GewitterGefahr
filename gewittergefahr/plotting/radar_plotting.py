"""Plotting methods for radar data."""

import numpy
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as pyplot
import matplotlib.colors
from gewittergefahr.gg_utils import grids
from gewittergefahr.gg_utils import radar_utils
from gewittergefahr.gg_utils import error_checking
from gewittergefahr.plotting import plotting_utils

REFL_PLOTTING_UNIT_STRING = 'dBZ'
SHEAR_PLOTTING_UNIT_STRING = 's^-1'
ECHO_TOP_PLOTTING_UNIT_STRING = 'kft'
MESH_PLOTTING_UNIT_STRING = 'mm'
SHI_PLOTTING_UNIT_STRING = ''
VIL_PLOTTING_UNIT_STRING = 'mm'

KM_TO_KILOFEET = 3.2808
METRES_TO_KM = 1e-3

DEFAULT_FIGURE_WIDTH_INCHES = 15.
DEFAULT_FIGURE_HEIGHT_INCHES = 15.


def _get_friendly_colour_list():
    """Returns colours in "colourblind-friendly" scheme used by GridRad viewer.

    :return: colour_list: 1-D list, where each element is a length-3 numpy array
        with [R, G, B].
    """

    colour_list = [
        numpy.array([242., 247., 233.]), numpy.array([220., 240., 212.]),
        numpy.array([193., 233., 196.]), numpy.array([174., 225., 196.]),
        numpy.array([156., 218., 205.]), numpy.array([138., 200., 211.]),
        numpy.array([122., 163., 204.]), numpy.array([106., 119., 196.]),
        numpy.array([112., 92., 189.]), numpy.array([137., 78., 182.]),
        numpy.array([167., 64., 174.]), numpy.array([167., 52., 134.]),
        numpy.array([160., 41., 83.]), numpy.array([153., 30., 30.])]

    for i in range(len(colour_list)):
        colour_list[i] /= 255

    return colour_list


def _get_modern_colour_list():
    """Returns list of colours in "modern" scheme used by GridRad viewer.

    :return: colour_list: 1-D list, where each element is a length-3 numpy array
        with [R, G, B].
    """

    colour_list = [
        numpy.array([0., 0., 0.]), numpy.array([64., 64., 64.]),
        numpy.array([131., 131., 131.]), numpy.array([0., 24., 255.]),
        numpy.array([0., 132., 255.]), numpy.array([0., 255., 255.]),
        numpy.array([5., 192., 127.]), numpy.array([5., 125., 0.]),
        numpy.array([105., 192., 0.]), numpy.array([255., 255., 0.]),
        numpy.array([255., 147., 8.]), numpy.array([255., 36., 15.]),
        numpy.array([255., 0., 255.]), numpy.array([255., 171., 255.])]

    for i in range(len(colour_list)):
        colour_list[i] /= 255

    return colour_list


def _get_default_refl_colour_scheme():
    """Returns default colour scheme for reflectivity.

    :return: colour_map_object: Instance of `matplotlib.colors.ListedColormap`.
    :return: colour_norm_object: Instance of `matplotlib.colors.BoundaryNorm`.
    :return: colour_bounds_dbz: See doc for `get_default_colour_scheme`.  In
        this case, units are dBZ (decibels of reflectivity).
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

    main_colour_bounds_dbz = numpy.array(
        [0.1, 5., 10., 15., 20., 25., 30., 35., 40., 45., 50., 55., 60., 65.,
         70.])
    colour_norm_object = matplotlib.colors.BoundaryNorm(
        main_colour_bounds_dbz, colour_map_object.N)

    colour_bounds_dbz = numpy.concatenate((
        numpy.array([0.]), main_colour_bounds_dbz, numpy.array([100.])))
    return colour_map_object, colour_norm_object, colour_bounds_dbz


def _get_default_zdr_colour_scheme():
    """Returns default colour scheme for Z_DR (differential reflectivity).

    :return: colour_map_object: Instance of `matplotlib.colors.ListedColormap`.
    :return: colour_norm_object: Instance of `matplotlib.colors.BoundaryNorm`.
    :return: colour_bounds_db: See doc for `get_default_colour_scheme`.  In
        this case, units are dB (decibels).
    """

    main_colour_list = _get_modern_colour_list()
    colour_map_object = matplotlib.colors.ListedColormap(main_colour_list)

    main_colour_bounds_db = numpy.array(
        [-1., -0.5, 0., 0.25, 0.5, 0.75, 1., 1.25, 1.5, 1.75, 2., 2.5, 3., 3.5,
         4.])
    colour_norm_object = matplotlib.colors.BoundaryNorm(
        main_colour_bounds_db, colour_map_object.N)

    colour_bounds_db = numpy.concatenate((
        numpy.array([-100.]), main_colour_bounds_db, numpy.array([100.])))
    return colour_map_object, colour_norm_object, colour_bounds_db


def _get_default_kdp_colour_scheme():
    """Returns default colour scheme for K_DP (specific differential phase).

    :return: colour_map_object: Instance of `matplotlib.colors.ListedColormap`.
    :return: colour_norm_object: Instance of `matplotlib.colors.BoundaryNorm`.
    :return: colour_bounds_deg_km01: See doc for `get_default_colour_scheme`.
        In this case, units are degrees per km.
    """

    main_colour_list = _get_modern_colour_list()
    colour_map_object = matplotlib.colors.ListedColormap(main_colour_list)

    main_colour_bounds_deg_km01 = numpy.array(
        [-1., -0.5, 0., 0.25, 0.5, 0.75, 1., 1.25, 1.5, 1.75, 2., 2.5, 3., 3.5,
         4.])
    colour_norm_object = matplotlib.colors.BoundaryNorm(
        main_colour_bounds_deg_km01, colour_map_object.N)

    colour_bounds_deg_km01 = numpy.concatenate((
        numpy.array([-100.]), main_colour_bounds_deg_km01, numpy.array([100.])))
    return colour_map_object, colour_norm_object, colour_bounds_deg_km01


def _get_default_rho_hv_colour_scheme():
    """Returns default colour scheme for rho_hv (correlation coefficient).

    :return: colour_map_object: Instance of `matplotlib.colors.ListedColormap`.
    :return: colour_norm_object: Instance of `matplotlib.colors.BoundaryNorm`.
    :return: colour_bounds: See doc for `get_default_colour_scheme`.  In this
        case, units are dimensionless.
    """

    main_colour_list = _get_modern_colour_list()
    colour_map_object = matplotlib.colors.ListedColormap(main_colour_list)
    colour_map_object.set_under(numpy.array([1., 1., 1.]))

    main_colour_bounds = numpy.array(
        [0.7, 0.75, 0.8, 0.85, 0.9, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97,
         0.98, 0.99, 1.])
    colour_norm_object = matplotlib.colors.BoundaryNorm(
        main_colour_bounds, colour_map_object.N)

    colour_bounds = numpy.concatenate((
        numpy.array([-100.]), main_colour_bounds, numpy.array([100.])))
    return colour_map_object, colour_norm_object, colour_bounds


def _get_default_spectrum_width_colour_scheme():
    """Returns default colour scheme for sigma_v (velocity-spectrum width).

    :return: colour_map_object: Instance of `matplotlib.colors.ListedColormap`.
    :return: colour_norm_object: Instance of `matplotlib.colors.BoundaryNorm`.
    :return: colour_bounds_m_s01: See doc for `get_default_colour_scheme`.  In
        this case, units are metres per second.
    """

    main_colour_list = [
        numpy.array([0, 0, 128.]), numpy.array([0, 54.3, 105.4]),
        numpy.array([0, 169.4, 255]), numpy.array([0, 253.6, 215.6]),
        numpy.array([0, 253.8, 41.8]), numpy.array([107.8, 219.1, 0]),
        numpy.array([168.6, 255, 43.8]), numpy.array([255, 245.2, 0]),
        numpy.array([255, 198.3, 8.6]), numpy.array([255, 6.6, 0]),
        numpy.array([255, 0, 141.7]), numpy.array([165.5, 44.8, 255]),
        numpy.array([239.8, 155.3, 241.8]), numpy.array([254, 248, 254.])
    ]

    for i in range(len(main_colour_list)):
        main_colour_list[i] /= 255

    colour_map_object = matplotlib.colors.ListedColormap(main_colour_list)
    colour_map_object.set_under(numpy.array([1, 1, 1.]))
    colour_map_object.set_over(numpy.array([1, 1, 1.]))

    main_colour_bounds_m_s01 = numpy.array(
        [0.1, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 5, 6, 7, 8, 9, 10])
    colour_norm_object = matplotlib.colors.BoundaryNorm(
        main_colour_bounds_m_s01, colour_map_object.N)

    colour_bounds_m_s01 = numpy.concatenate((
        numpy.array([-100.]), main_colour_bounds_m_s01, numpy.array([100.])))
    return colour_map_object, colour_norm_object, colour_bounds_m_s01


def _get_default_vorticity_colour_scheme():
    """Returns default colour scheme for vorticity.

    :return: colour_map_object: Instance of `matplotlib.colors.ListedColormap`.
    :return: colour_norm_object: Instance of `matplotlib.colors.BoundaryNorm`.
    :return: colour_bounds_s01: See doc for `get_default_colour_scheme`.  In
        this case, units are seconds^-1.
    """

    main_colour_list = [
        numpy.array([0, 0, 76.5]), numpy.array([0, 0, 118.5]),
        numpy.array([0, 0, 163.3]), numpy.array([0, 0, 208.1]),
        numpy.array([0, 0, 252.9]), numpy.array([61, 61, 255.]),
        numpy.array([125, 125, 255.]), numpy.array([189, 189, 255.]),
        numpy.array([253, 253, 255.]), numpy.array([255, 193, 193.]),
        numpy.array([255, 129, 129.]), numpy.array([255, 65, 65.]),
        numpy.array([255, 1, 1.]), numpy.array([223.5, 0, 0]),
        numpy.array([191.5, 0, 0]), numpy.array([159.5, 0, 0]),
        numpy.array([127.5, 0, 0]), numpy.array([95.5, 0, 0])
    ]

    for i in range(len(main_colour_list)):
        main_colour_list[i] /= 255

    colour_map_object = matplotlib.colors.ListedColormap(main_colour_list)
    colour_map_object.set_under(numpy.array([1, 1, 1.]))
    colour_map_object.set_over(numpy.array([1, 1, 1.]))

    main_colour_bounds_s01 = numpy.array(
        [-7, -6, -5, -4, -3, -2, -1.5, -1, -0.5, 0.5, 1, 1.5, 2, 3, 4, 5, 6, 7]
    ) / 1000
    colour_norm_object = matplotlib.colors.BoundaryNorm(
        main_colour_bounds_s01, colour_map_object.N)

    colour_bounds_s01 = numpy.concatenate((
        numpy.array([-0.1]), main_colour_bounds_s01, numpy.array([0.1])))
    return colour_map_object, colour_norm_object, colour_bounds_s01


def _get_default_divergence_colour_scheme():
    """Returns default colour scheme for divergence.

    :return: colour_map_object: Instance of `matplotlib.colors.ListedColormap`.
    :return: colour_norm_object: Instance of `matplotlib.colors.BoundaryNorm`.
    :return: colour_bounds_s01: See doc for `get_default_colour_scheme`.  In
        this case, units are seconds^-1.
    """

    return _get_default_shear_colour_scheme()


def _get_default_shear_colour_scheme():
    """Returns default colour scheme for azimuthal shear.

    :return: colour_map_object: Instance of `matplotlib.colors.ListedColormap`.
    :return: colour_norm_object: Instance of `matplotlib.colors.BoundaryNorm`.
    :return: colour_bounds_s01: See doc for `get_default_colour_scheme`.  In
        this case, units are seconds^-1.
    """

    main_colour_list = [
        numpy.array([0, 0, 76.5]), numpy.array([0, 0, 118.5]),
        numpy.array([0, 0, 163.3]), numpy.array([0, 0, 208.1]),
        numpy.array([0, 0, 252.9]), numpy.array([61, 61, 255.]),
        numpy.array([125, 125, 255.]), numpy.array([189, 189, 255.]),
        numpy.array([253, 253, 255.]), numpy.array([255, 193, 193.]),
        numpy.array([255, 129, 129.]), numpy.array([255, 65, 65.]),
        numpy.array([255, 1, 1.]), numpy.array([223.5, 0, 0]),
        numpy.array([191.5, 0, 0]), numpy.array([159.5, 0, 0]),
        numpy.array([127.5, 0, 0]), numpy.array([95.5, 0, 0])
    ]

    for i in range(len(main_colour_list)):
        main_colour_list[i] /= 255

    colour_map_object = matplotlib.colors.ListedColormap(main_colour_list)
    colour_map_object.set_under(numpy.array([1, 1, 1.]))
    colour_map_object.set_over(numpy.array([1, 1, 1.]))

    main_colour_bounds_s01 = numpy.array(
        [-20, -17.5, -15, -12.5, -10, -7.5, -5, -3, -1, 1, 3, 5, 7.5, 10, 12.5,
         15, 17.5, 20]
    ) / 1000
    colour_norm_object = matplotlib.colors.BoundaryNorm(
        main_colour_bounds_s01, colour_map_object.N)

    colour_bounds_s01 = numpy.concatenate((
        numpy.array([-0.1]), main_colour_bounds_s01, numpy.array([0.1])))
    return colour_map_object, colour_norm_object, colour_bounds_s01


def _get_old_shear_colour_scheme():
    """Returns old colour scheme for azimuthal shear.

    :return: colour_map_object: Instance of `matplotlib.colors.ListedColormap`.
    :return: colour_norm_object: Instance of `matplotlib.colors.BoundaryNorm`.
    :return: colour_bounds_s01: See doc for `get_default_colour_scheme`.  In
        this case, units are seconds^-1.
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


def _get_default_echo_top_colour_scheme():
    """Returns default colur scheme for echo tops.

    :return: colour_map_object: Instance of `matplotlib.colors.ListedColormap`.
    :return: colour_norm_object: Instance of `matplotlib.colors.BoundaryNorm`.
    :return: colour_bounds_s01: See doc for `get_default_colour_scheme`.  In
        this case, units are kilofeet (why, America, why?).
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

    main_colour_bounds_kft = numpy.array(
        [0.1, 10., 15., 20., 25., 30., 35., 40., 45., 50., 55., 60., 65.])
    colour_norm_object = matplotlib.colors.BoundaryNorm(
        main_colour_bounds_kft, colour_map_object.N)

    colour_bounds_kft = numpy.concatenate((
        numpy.array([0.]), main_colour_bounds_kft, numpy.array([100.])))
    return colour_map_object, colour_norm_object, colour_bounds_kft


def _get_default_mesh_colour_scheme():
    """Returns default colour scheme for MESH (max estimated size of hail).

    :return: colour_map_object: Instance of `matplotlib.colors.ListedColormap`.
    :return: colour_norm_object: Instance of `matplotlib.colors.BoundaryNorm`.
    :return: colour_bounds_mm: See doc for `get_default_colour_scheme`.  In
        this case, units are millimetres.
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

    main_colour_bounds_mm = numpy.array(
        [0.1, 15.9, 22.2, 28.6, 34.9, 41.3, 47.6, 54., 60.3, 65., 70., 75., 80.,
         85.])
    colour_norm_object = matplotlib.colors.BoundaryNorm(
        main_colour_bounds_mm, colour_map_object.N)

    colour_bounds_mm = numpy.concatenate((
        numpy.array([0.]), main_colour_bounds_mm, numpy.array([200.])))
    return colour_map_object, colour_norm_object, colour_bounds_mm


def _get_default_shi_colour_scheme():
    """Returns default colour scheme for SHI (severe-hail index).

    :return: colour_map_object: Instance of `matplotlib.colors.ListedColormap`.
    :return: colour_norm_object: Instance of `matplotlib.colors.BoundaryNorm`.
    :return: colour_bounds: See doc for `get_default_colour_scheme`.  In this
        case, units are dimensionless.
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

    main_colour_bounds = numpy.array(
        [1., 30., 60., 90., 120., 150., 180., 210., 240., 270., 300., 330.,
         360., 390., 420.])
    colour_norm_object = matplotlib.colors.BoundaryNorm(
        main_colour_bounds, colour_map_object.N)

    colour_bounds = numpy.concatenate((
        numpy.array([0.]), main_colour_bounds, numpy.array([1000.])))
    return colour_map_object, colour_norm_object, colour_bounds


def _get_default_vil_colour_scheme():
    """Returns default colour scheme for VIL (vertically integrated liquid).

    :return: colour_map_object: Instance of `matplotlib.colors.ListedColormap`.
    :return: colour_norm_object: Instance of `matplotlib.colors.BoundaryNorm`.
    :return: colour_bounds_mm: See doc for `get_default_colour_scheme`.  In
        this case, units are millimetres.
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

    main_colour_bounds_mm = numpy.array(
        [0.1, 5., 10., 15., 20., 25., 30., 35., 40., 45., 50., 55., 60., 65.,
         70.])
    colour_norm_object = matplotlib.colors.BoundaryNorm(
        main_colour_bounds_mm, colour_map_object.N)

    colour_bounds_mm = numpy.concatenate((
        numpy.array([0.]), main_colour_bounds_mm, numpy.array([200.])))
    return colour_map_object, colour_norm_object, colour_bounds_mm


def _convert_to_plotting_units(field_matrix, field_name):
    """Converts field from default (GewitterGefahr) units to plotting units.

    :param field_matrix: numpy array in default units.
    :param field_name: Name of radar field (must be accepted by
        `radar_utils.check_field_name`).
    :return: field_matrix: Same as input, but now in plotting units.
    """

    radar_utils.check_field_name(field_name)
    if field_name in radar_utils.ECHO_TOP_NAMES:
        return field_matrix * KM_TO_KILOFEET

    return field_matrix


def get_default_colour_scheme(field_name):
    """Returns default colour scheme for the given radar field.

    N = number of colours

    :param field_name: Name of radar field (must be accepted by
        `radar_utils.check_field_name`).
    :return: colour_map_object: Instance of `matplotlib.colors.ListedColormap`.
    :return: colour_norm_object: Instance of `matplotlib.colors.BoundaryNorm`.
    :return: colour_bounds: length-(N + 1) numpy array of colour boundaries.
        colour_bounds[0] and colour_bounds[1] are the boundaries for the 1st
        colour; colour_bounds[1] and colour_bounds[2] are the boundaries for the
        2nd colour; ...; colour_bounds[i] and colour_bounds[i + 1] are the
        boundaries for the (i + 1)th colour.
    """

    radar_utils.check_field_name(field_name)

    if field_name in radar_utils.REFLECTIVITY_NAMES:
        return _get_default_refl_colour_scheme()

    if field_name in radar_utils.SHEAR_NAMES:
        return _get_default_shear_colour_scheme()

    if field_name in radar_utils.ECHO_TOP_NAMES:
        return _get_default_echo_top_colour_scheme()

    if field_name == radar_utils.MESH_NAME:
        return _get_default_mesh_colour_scheme()

    if field_name == radar_utils.SHI_NAME:
        return _get_default_shi_colour_scheme()

    if field_name == radar_utils.VIL_NAME:
        return _get_default_vil_colour_scheme()

    if field_name == radar_utils.DIFFERENTIAL_REFL_NAME:
        return _get_default_zdr_colour_scheme()

    if field_name == radar_utils.SPEC_DIFF_PHASE_NAME:
        return _get_default_kdp_colour_scheme()

    if field_name == radar_utils.CORRELATION_COEFF_NAME:
        return _get_default_rho_hv_colour_scheme()

    if field_name == radar_utils.SPECTRUM_WIDTH_NAME:
        return _get_default_spectrum_width_colour_scheme()

    if field_name == radar_utils.VORTICITY_NAME:
        return _get_default_vorticity_colour_scheme()

    if field_name == radar_utils.DIVERGENCE_NAME:
        return _get_default_divergence_colour_scheme()

    return None


def get_plotting_units(field_name):
    """Returns string with plotting units for the given radar field.

    :param field_name: Name of radar field (must be accepted by
        `radar_utils.check_field_name`).
    :return: unit_string: String with plotting units.
    """

    radar_utils.check_field_name(field_name)

    if field_name in radar_utils.REFLECTIVITY_NAMES:
        return REFL_PLOTTING_UNIT_STRING

    if field_name in radar_utils.SHEAR_NAMES:
        return SHEAR_PLOTTING_UNIT_STRING

    if field_name in radar_utils.ECHO_TOP_NAMES:
        return ECHO_TOP_PLOTTING_UNIT_STRING

    if field_name == radar_utils.MESH_NAME:
        return MESH_PLOTTING_UNIT_STRING

    if field_name == radar_utils.SHI_NAME:
        return SHI_PLOTTING_UNIT_STRING

    if field_name == radar_utils.VIL_NAME:
        return VIL_PLOTTING_UNIT_STRING

    return None


def plot_latlng_grid(
        field_matrix, field_name, axes_object, min_grid_point_latitude_deg,
        min_grid_point_longitude_deg, latitude_spacing_deg,
        longitude_spacing_deg, colour_map_object=None, colour_norm_object=None):
    """Plots lat-long grid as colour map.

    M = number of rows (unique grid-point latitudes)
    N = number of columns (unique grid-point longitudes)

    Because this method plots a lat-long grid (rather than an x-y grid), if you
    have used Basemap to plot borders or anything else, the only acceptable
    projection is cylindrical equidistant (in which x = longitude and
    y = latitude, so no coordinate conversion is necessary).

    To use the default colour scheme for the given radar field, leave
    `colour_map_object` and `colour_norm_object` empty.

    :param field_matrix: M-by-N numpy array with values of radar field.
    :param field_name: Name of radar field (must be accepted by
        `radar_utils.check_field_name`).
    :param axes_object: Instance of `matplotlib.axes._subplots.AxesSubplot`.
    :param min_grid_point_latitude_deg: Minimum latitude (deg N) over all grid
        points.  This should be the latitude in the first row of `field_matrix`
        -- i.e., at `field_matrix[0, :]`.
    :param min_grid_point_longitude_deg: Minimum longitude (deg E) over all grid
        points.  This should be the longitude in the first column of
        `field_matrix` -- i.e., at `field_matrix[:, 0]`.
    :param latitude_spacing_deg: Spacing (deg N) between grid points in adjacent
        rows.
    :param longitude_spacing_deg: Spacing (deg E) between grid points in
        adjacent columns.
    :param colour_map_object: Instance of `matplotlib.pyplot.cm`.
    :param colour_norm_object: Instance of `matplotlib.colors.BoundaryNorm`.
    """

    field_matrix = _convert_to_plotting_units(
        field_matrix=field_matrix, field_name=field_name)

    (field_matrix_at_edges, grid_cell_edge_latitudes_deg,
     grid_cell_edge_longitudes_deg
    ) = grids.latlng_field_grid_points_to_edges(
        field_matrix=field_matrix, min_latitude_deg=min_grid_point_latitude_deg,
        min_longitude_deg=min_grid_point_longitude_deg,
        lat_spacing_deg=latitude_spacing_deg,
        lng_spacing_deg=longitude_spacing_deg)

    field_matrix_at_edges = numpy.ma.masked_where(
        numpy.isnan(field_matrix_at_edges), field_matrix_at_edges)

    if colour_map_object is None or colour_norm_object is None:
        colour_map_object, colour_norm_object, _ = get_default_colour_scheme(
            field_name)

    pyplot.pcolormesh(
        grid_cell_edge_longitudes_deg, grid_cell_edge_latitudes_deg,
        field_matrix_at_edges, cmap=colour_map_object, norm=colour_norm_object,
        vmin=colour_norm_object.boundaries[0],
        vmax=colour_norm_object.boundaries[-1], shading='flat',
        edgecolors='None', axes=axes_object)


def plot_2d_grid_without_coords(
        field_matrix, field_name, axes_object, annotation_string=None,
        colour_map_object=None, colour_norm_object=None):
    """Plots 2-D grid as colour map.

    M = number of rows in grid
    N = number of columns in grid

    In this case the grid is not georeferenced (convenient for storm-centered
    radar images).

    To use the default colour scheme for the given radar field, leave
    `colour_map_object` and `colour_norm_object` empty.

    :param field_matrix: M-by-N numpy array of radar values.
    :param field_name: Same.
    :param axes_object: Same.
    :param annotation_string: Annotation (will be printed in the bottom-center).
        If you want no annotation, leave this alone.
    :param colour_map_object: See doc for `plot_latlng_grid`.
    :param colour_norm_object: Same.
    """

    error_checking.assert_is_numpy_array_without_nan(field_matrix)
    error_checking.assert_is_numpy_array(field_matrix, num_dimensions=2)

    field_matrix = _convert_to_plotting_units(
        field_matrix=field_matrix, field_name=field_name)
    field_matrix = numpy.ma.masked_where(
        numpy.isnan(field_matrix), field_matrix)

    if colour_map_object is None or colour_norm_object is None:
        colour_map_object, colour_norm_object, _ = get_default_colour_scheme(
            field_name)

    axes_object.pcolormesh(
        field_matrix, cmap=colour_map_object, norm=colour_norm_object,
        vmin=colour_norm_object.boundaries[0],
        vmax=colour_norm_object.boundaries[-1], shading='flat',
        edgecolors='None')

    if annotation_string is not None:
        error_checking.assert_is_string(annotation_string)
        axes_object.text(
            0.5, 0.01, annotation_string, fontsize=20, color='k',
            horizontalalignment='center', verticalalignment='bottom',
            transform=axes_object.transAxes)

    axes_object.set_xticks([])
    axes_object.set_yticks([])


def plot_many_2d_grids_without_coords(
        field_matrix, field_name_by_pair, height_by_pair_metres,
        ground_relative, num_panel_rows,
        figure_width_inches=DEFAULT_FIGURE_WIDTH_INCHES,
        figure_height_inches=DEFAULT_FIGURE_HEIGHT_INCHES):
    """Plots each 2-D grid as colour map (one per field/height pair).

    M = number of grid rows
    N = number of grid columns
    C = number of field/height pairs

    This method uses the default colour scheme for each radar field.

    :param field_matrix: M-by-N-by-C numpy array of radar values.
    :param field_name_by_pair: length-C list with names of radar fields, in the
        order that they appear in `field_matrix`.  Each field name must be
        accepted by `radar_utils.check_field_name`.
    :param height_by_pair_metres: length-C numpy array of radar heights.
    :param ground_relative: Boolean flag.  If True, heights in
        `height_by_pair_metres` are ground-relative.  If False,
        sea-level-relative.
    :param num_panel_rows: Number of rows in paneled figure (different than M,
        the number of grid rows).
    :param figure_width_inches: Figure width.
    :param figure_height_inches: Figure height.
    :return: figure_object: Instance of `matplotlib.figure.Figure`.
    :return: axes_objects_2d_list: 2-D list, where each item is an instance of
        `matplotlib.axes._subplots.AxesSubplot`.
    """

    error_checking.assert_is_numpy_array(field_matrix, num_dimensions=3)

    num_field_height_pairs = field_matrix.shape[2]
    error_checking.assert_is_numpy_array(
        numpy.array(field_name_by_pair),
        exact_dimensions=numpy.array([num_field_height_pairs]))

    error_checking.assert_is_integer_numpy_array(height_by_pair_metres)
    error_checking.assert_is_geq_numpy_array(height_by_pair_metres, 0)
    error_checking.assert_is_numpy_array(
        height_by_pair_metres,
        exact_dimensions=numpy.array([num_field_height_pairs]))

    error_checking.assert_is_boolean(ground_relative)
    error_checking.assert_is_integer(num_panel_rows)
    error_checking.assert_is_geq(num_panel_rows, 1)
    error_checking.assert_is_leq(num_panel_rows, num_field_height_pairs)

    num_panel_columns = int(numpy.ceil(
        float(num_field_height_pairs) / num_panel_rows))
    figure_object, axes_objects_2d_list = plotting_utils.init_panels(
        num_panel_rows=num_panel_rows, num_panel_columns=num_panel_columns,
        figure_width_inches=figure_width_inches,
        figure_height_inches=figure_height_inches)

    for i in range(num_panel_rows):
        for j in range(num_panel_columns):
            this_fh_pair_index = i * num_panel_columns + j
            if this_fh_pair_index >= num_field_height_pairs:
                break

            this_annotation_string = '{0:s}\nat {1:.1f} km'.format(
                field_name_by_pair[this_fh_pair_index],
                height_by_pair_metres[this_fh_pair_index] * METRES_TO_KM)
            if ground_relative:
                this_annotation_string += ' AGL'
            else:
                this_annotation_string += ' ASL'

            plot_2d_grid_without_coords(
                field_matrix=field_matrix[..., this_fh_pair_index],
                field_name=field_name_by_pair[this_fh_pair_index],
                axes_object=axes_objects_2d_list[i][j],
                annotation_string=this_annotation_string)

    return figure_object, axes_objects_2d_list


def plot_3d_grid_without_coords(
        field_matrix, field_name, grid_point_heights_metres, ground_relative,
        num_panel_rows, figure_width_inches=DEFAULT_FIGURE_WIDTH_INCHES,
        figure_height_inches=DEFAULT_FIGURE_HEIGHT_INCHES,
        colour_map_object=None, colour_norm_object=None):
    """Plots 3-D grid as many colour maps (one per height).

    M = number of grid rows
    N = number of grid columns
    H = number of grid heights

    To use the default colour scheme for the given radar field, leave
    `colour_map_object` and `colour_norm_object` empty.

    :param field_matrix: M-by-N-by-H numpy array with values of radar field.
    :param field_name: Name of radar field (must be accepted by
        `radar_utils.check_field_name`).
    :param grid_point_heights_metres: length-H integer numpy array of heights.
    :param ground_relative: Boolean flag.  If True, heights in
        `height_by_pair_metres` are ground-relative.  If False,
        sea-level-relative.
    :param num_panel_rows: Number of rows in paneled figure (different than M,
        the number of grid rows).
    :param figure_width_inches: Figure width.
    :param figure_height_inches: Figure height.
    :param colour_map_object: See doc for `plot_latlng_grid`.
    :param colour_norm_object: Same.
    :return: figure_object: Instance of `matplotlib.figure.Figure`.
    :return: axes_objects_2d_list: 2-D list, where each item is an instance of
        `matplotlib.axes._subplots.AxesSubplot`.
    """

    error_checking.assert_is_numpy_array(field_matrix, num_dimensions=3)

    num_heights = field_matrix.shape[2]
    error_checking.assert_is_integer_numpy_array(grid_point_heights_metres)
    error_checking.assert_is_geq_numpy_array(grid_point_heights_metres, 0)
    error_checking.assert_is_numpy_array(
        grid_point_heights_metres, exact_dimensions=numpy.array([num_heights]))

    error_checking.assert_is_boolean(ground_relative)
    error_checking.assert_is_integer(num_panel_rows)
    error_checking.assert_is_geq(num_panel_rows, 1)
    error_checking.assert_is_leq(num_panel_rows, num_heights)

    num_panel_columns = int(numpy.ceil(float(num_heights) / num_panel_rows))
    figure_object, axes_objects_2d_list = plotting_utils.init_panels(
        num_panel_rows=num_panel_rows, num_panel_columns=num_panel_columns,
        figure_width_inches=figure_width_inches,
        figure_height_inches=figure_height_inches)

    for i in range(num_panel_rows):
        for j in range(num_panel_columns):
            this_height_index = i * num_panel_columns + j
            if this_height_index >= num_heights:
                break

            this_annotation_string = '{0:.1f} km'.format(
                grid_point_heights_metres[this_height_index] * METRES_TO_KM)
            if ground_relative:
                this_annotation_string += ' AGL'
            else:
                this_annotation_string += ' ASL'

            plot_2d_grid_without_coords(
                field_matrix=field_matrix[..., this_height_index],
                field_name=field_name, axes_object=axes_objects_2d_list[i][j],
                annotation_string=this_annotation_string,
                colour_map_object=colour_map_object,
                colour_norm_object=colour_norm_object)

    return figure_object, axes_objects_2d_list
