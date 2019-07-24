"""Plotting methods for radar data."""

import numpy
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as pyplot
import matplotlib.colors
from gewittergefahr.gg_utils import grids
from gewittergefahr.gg_utils import radar_utils
from gewittergefahr.gg_utils import error_checking
from gewittergefahr.deep_learning import input_examples
from gewittergefahr.plotting import plotting_utils

SHEAR_VORT_DIV_NAMES = [
    radar_utils.VORTICITY_NAME, radar_utils.DIVERGENCE_NAME,
    radar_utils.LOW_LEVEL_SHEAR_NAME, radar_utils.MID_LEVEL_SHEAR_NAME
]

KM_TO_KILOFEET = 3.2808
METRES_TO_KM = 1e-3
PER_SECOND_TO_PER_KILOSECOND = 1e3

DEFAULT_FONT_SIZE = 20

FIELD_NAME_TO_VERBOSE_DICT = {
    radar_utils.ECHO_TOP_15DBZ_NAME: '15-dBZ echo top (kft ASL)',
    radar_utils.ECHO_TOP_18DBZ_NAME: '18-dBZ echo top (kft ASL)',
    radar_utils.ECHO_TOP_20DBZ_NAME: '20-dBZ echo top (kft ASL)',
    radar_utils.ECHO_TOP_25DBZ_NAME: '25-dBZ echo top (kft ASL)',
    radar_utils.ECHO_TOP_40DBZ_NAME: '40-dBZ echo top (kft ASL)',
    radar_utils.ECHO_TOP_50DBZ_NAME: '50-dBZ echo top (kft ASL)',
    radar_utils.LOW_LEVEL_SHEAR_NAME: r'Low-level shear (ks$^{-1}$)',
    radar_utils.MID_LEVEL_SHEAR_NAME: r'Mid-level shear (ks$^{-1}$)',
    radar_utils.MESH_NAME: 'Max estimated hail size (mm)',
    radar_utils.REFL_NAME: 'Reflectivity (dBZ)',
    radar_utils.REFL_COLUMN_MAX_NAME: 'Column-max reflectivity (dBZ)',
    radar_utils.REFL_0CELSIUS_NAME: r'0 $^{\circ}C$ reflectivity (dBZ)',
    radar_utils.REFL_M10CELSIUS_NAME: r'-10 $^{\circ}C$ reflectivity (dBZ)',
    radar_utils.REFL_M20CELSIUS_NAME: r'-20 $^{\circ}C$ reflectivity (dBZ)',
    radar_utils.REFL_LOWEST_ALTITUDE_NAME: 'Lowest-altitude refl (dBZ)',
    radar_utils.SHI_NAME: 'Severe-hail index',
    radar_utils.VIL_NAME: 'Vertically integ liquid (mm)',
    radar_utils.DIFFERENTIAL_REFL_NAME: 'Diff reflectivity (dB)',
    radar_utils.SPEC_DIFF_PHASE_NAME: r'Spec diff phase ($^{\circ}$ km$^{-1}$)',
    radar_utils.CORRELATION_COEFF_NAME: 'Correlation coefficient',
    radar_utils.SPECTRUM_WIDTH_NAME: r'Spectrum width (m s$^{-1}$)',
    radar_utils.VORTICITY_NAME: r'Vorticity (ks$^{-1}$)',
    radar_utils.DIVERGENCE_NAME: r'Divergence (ks$^{-1}$)'
}


def _get_friendly_colours():
    """Returns colours in colourblind-friendly scheme used by GridRad viewer.

    :return: colour_list: 1-D list, where each element is a numpy array with the
        [R, G, B] values in that order.
    """

    colour_list = [
        [242, 247, 233], [220, 240, 212], [193, 233, 196], [174, 225, 196],
        [156, 218, 205], [138, 200, 211], [122, 163, 204], [106, 119, 196],
        [112, 92, 189], [137, 78, 182], [167, 64, 174], [167, 52, 134],
        [160, 41, 83], [153, 30, 30]
    ]

    for i in range(len(colour_list)):
        colour_list[i] = numpy.array(colour_list[i], dtype=float) / 255

    return colour_list


def _get_modern_colours():
    """Returns colours in "modern" scheme used by GridRad viewer.

    :return: colour_list: See doc for `_get_friendly_colours`.
    """

    colour_list = [
        [0, 0, 0], [64, 64, 64], [131, 131, 131], [0, 24, 255],
        [0, 132, 255], [0, 255, 255], [5, 192, 127], [5, 125, 0],
        [105, 192, 0], [255, 255, 0], [255, 147, 8], [255, 36, 15],
        [255, 0, 255], [255, 171, 255]
    ]

    for i in range(len(colour_list)):
        colour_list[i] = numpy.array(colour_list[i], dtype=float) / 255

    return colour_list


def _get_reflectivity_colour_scheme():
    """Returns colour scheme for reflectivity.

    :return: colour_map_object: Instance of `matplotlib.colors.ListedColormap`.
    :return: colour_norm_object: Instance of `matplotlib.colors.BoundaryNorm`.
    """

    colour_list = [
        [4, 233, 231], [1, 159, 244], [3, 0, 244], [2, 253, 2],
        [1, 197, 1], [0, 142, 0], [253, 248, 2], [229, 188, 0],
        [253, 149, 0], [253, 0, 0], [212, 0, 0], [188, 0, 0],
        [248, 0, 253], [152, 84, 198]
    ]

    for i in range(len(colour_list)):
        colour_list[i] = numpy.array(colour_list[i], dtype=float) / 255

    colour_map_object = matplotlib.colors.ListedColormap(colour_list)
    colour_map_object.set_under(numpy.full(3, 1))

    colour_bounds_dbz = numpy.array(
        [0.1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70])
    colour_norm_object = matplotlib.colors.BoundaryNorm(
        colour_bounds_dbz, colour_map_object.N)

    return colour_map_object, colour_norm_object


def _get_zdr_colour_scheme():
    """Returns colour scheme for Z_DR (differential reflectivity).

    :return: colour_map_object: Instance of `matplotlib.colors.ListedColormap`.
    :return: colour_norm_object: Instance of `matplotlib.colors.BoundaryNorm`.
    """

    colour_list = _get_modern_colours()
    colour_map_object = matplotlib.colors.ListedColormap(colour_list)

    colour_bounds_db = numpy.array(
        [-1, -0.5, 0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 2.5, 3, 3.5, 4])
    colour_norm_object = matplotlib.colors.BoundaryNorm(
        colour_bounds_db, colour_map_object.N)

    return colour_map_object, colour_norm_object


def _get_kdp_colour_scheme():
    """Returns colour scheme for K_DP (specific differential phase).

    :return: colour_map_object: Instance of `matplotlib.colors.ListedColormap`.
    :return: colour_norm_object: Instance of `matplotlib.colors.BoundaryNorm`.
    """

    colour_list = _get_modern_colours()
    colour_map_object = matplotlib.colors.ListedColormap(colour_list)

    colour_bounds_deg_km01 = numpy.array(
        [-1, -0.5, 0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 2.5, 3, 3.5, 4])
    colour_norm_object = matplotlib.colors.BoundaryNorm(
        colour_bounds_deg_km01, colour_map_object.N)

    return colour_map_object, colour_norm_object


def _get_rho_hv_colour_scheme():
    """Returns colour scheme for rho_hv (cross-polar correlation coefficient).

    :return: colour_map_object: Instance of `matplotlib.colors.ListedColormap`.
    :return: colour_norm_object: Instance of `matplotlib.colors.BoundaryNorm`.
    """

    colour_list = _get_modern_colours()
    colour_map_object = matplotlib.colors.ListedColormap(colour_list)
    colour_map_object.set_under(numpy.full(3, 1))

    colour_bounds_unitless = numpy.array(
        [0.7, 0.75, 0.8, 0.85, 0.9, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97,
         0.98, 0.99, 1]
    )

    colour_norm_object = matplotlib.colors.BoundaryNorm(
        colour_bounds_unitless, colour_map_object.N)

    return colour_map_object, colour_norm_object


def _get_spectrum_width_colour_scheme():
    """Returns colour scheme for velocity-spectrum width.

    :return: colour_map_object: Instance of `matplotlib.colors.ListedColormap`.
    :return: colour_norm_object: Instance of `matplotlib.colors.BoundaryNorm`.
    """

    colour_list = _get_friendly_colours()
    colour_map_object = matplotlib.colors.ListedColormap(colour_list)
    colour_map_object.set_under(numpy.full(3, 1))

    colour_bounds_m_s01 = numpy.array(
        [0.1, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 5, 6, 7, 8, 9, 10])
    colour_norm_object = matplotlib.colors.BoundaryNorm(
        colour_bounds_m_s01, colour_map_object.N)

    return colour_map_object, colour_norm_object


def _get_vorticity_colour_scheme():
    """Returns colour scheme for vorticity.

    :return: colour_map_object: Instance of `matplotlib.colors.ListedColormap`.
    :return: colour_norm_object: Instance of `matplotlib.colors.BoundaryNorm`.
    """

    colour_list = [
        [0, 0, 76.5], [0, 0, 118.5], [0, 0, 163.3], [0, 0, 208.1],
        [0, 0, 252.9], [61, 61, 255], [125, 125, 255], [189, 189, 255],
        [253, 253, 255], [255, 193, 193], [255, 129, 129], [255, 65, 65],
        [255, 1, 1], [223.5, 0, 0], [191.5, 0, 0], [159.5, 0, 0],
        [127.5, 0, 0], [95.5, 0, 0]
    ]

    for i in range(len(colour_list)):
        colour_list[i] = numpy.array(colour_list[i], dtype=float) / 255

    colour_map_object = matplotlib.colors.ListedColormap(colour_list)
    colour_bounds_ks01 = numpy.array(
        [-7, -6, -5, -4, -3, -2, -1.5, -1, -0.5, 0.5, 1, 1.5, 2, 3, 4, 5, 6, 7])
    colour_norm_object = matplotlib.colors.BoundaryNorm(
        colour_bounds_ks01, colour_map_object.N)

    return colour_map_object, colour_norm_object


def _get_az_shear_colour_scheme():
    """Returns colour scheme for azimuthal shear.

    :return: colour_map_object: Instance of `matplotlib.colors.ListedColormap`.
    :return: colour_norm_object: Instance of `matplotlib.colors.BoundaryNorm`.
    """

    colour_list = [
        [0, 0, 76.5], [0, 0, 118.5], [0, 0, 163.3], [0, 0, 208.1],
        [0, 0, 252.9], [61, 61, 255], [125, 125, 255], [189, 189, 255],
        [253, 253, 255], [255, 193, 193], [255, 129, 129], [255, 65, 65],
        [255, 1, 1], [223.5, 0, 0], [191.5, 0, 0], [159.5, 0, 0],
        [127.5, 0, 0], [95.5, 0, 0]
    ]

    for i in range(len(colour_list)):
        colour_list[i] = numpy.array(colour_list[i], dtype=float) / 255

    colour_map_object = matplotlib.colors.ListedColormap(colour_list)
    colour_bounds_ks01 = 2 * numpy.array(
        [-7, -6, -5, -4, -3, -2, -1.5, -1, -0.5, 0.5, 1, 1.5, 2, 3, 4, 5, 6, 7])
    colour_norm_object = matplotlib.colors.BoundaryNorm(
        colour_bounds_ks01, colour_map_object.N)

    return colour_map_object, colour_norm_object


def _get_divergence_colour_scheme():
    """Returns colour scheme for divergence.

    :return: colour_map_object: Instance of `matplotlib.colors.ListedColormap`.
    :return: colour_norm_object: Instance of `matplotlib.colors.BoundaryNorm`.
    """

    return _get_az_shear_colour_scheme()


def _get_echo_top_colour_scheme():
    """Returns colour scheme for echo top.

    :return: colour_map_object: Instance of `matplotlib.colors.ListedColormap`.
    :return: colour_norm_object: Instance of `matplotlib.colors.BoundaryNorm`.
    """

    colour_list = [
        [120, 120, 120], [16, 220, 244], [11, 171, 247], [9, 144, 202],
        [48, 6, 134], [4, 248, 137], [10, 185, 6], [1, 241, 8],
        [255, 186, 1], [255, 251, 0], [132, 17, 22], [233, 16, 1]
    ]

    for i in range(len(colour_list)):
        colour_list[i] = numpy.array(colour_list[i], dtype=float) / 255

    colour_map_object = matplotlib.colors.ListedColormap(colour_list)
    colour_map_object.set_under(numpy.full(3, 1))

    colour_bounds_kft = numpy.array(
        [0.1, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65])
    colour_norm_object = matplotlib.colors.BoundaryNorm(
        colour_bounds_kft, colour_map_object.N)

    return colour_map_object, colour_norm_object


def _get_mesh_colour_scheme():
    """Returns colour scheme for MESH (maximum estimated size of hail).

    :return: colour_map_object: Instance of `matplotlib.colors.ListedColormap`.
    :return: colour_norm_object: Instance of `matplotlib.colors.BoundaryNorm`.
    """

    colour_list = [
        [152, 152, 152], [152, 203, 254], [0, 152, 254], [0, 45, 254],
        [0, 101, 0], [0, 152, 0], [0, 203, 0], [254, 254, 50],
        [254, 203, 0], [254, 152, 0], [254, 0, 0], [254, 0, 152],
        [152, 50, 203]
    ]

    for i in range(len(colour_list)):
        colour_list[i] = numpy.array(colour_list[i], dtype=float) / 255

    colour_map_object = matplotlib.colors.ListedColormap(colour_list)
    colour_map_object.set_under(numpy.full(3, 1))

    colour_bounds_mm = numpy.array(
        [0.1, 15.9, 22.2, 28.6, 34.9, 41.3, 47.6, 54, 60.3, 65, 70, 75, 80, 85])
    colour_norm_object = matplotlib.colors.BoundaryNorm(
        colour_bounds_mm, colour_map_object.N)

    return colour_map_object, colour_norm_object


def _get_shi_colour_scheme():
    """Returns colour scheme for SHI (severe-hail index).

    :return: colour_map_object: Instance of `matplotlib.colors.ListedColormap`.
    :return: colour_norm_object: Instance of `matplotlib.colors.BoundaryNorm`.
    """

    colour_list = [
        [152, 152, 152], [152, 203, 254], [0, 152, 254], [0, 45, 254],
        [0, 101, 0], [0, 152, 0], [0, 203, 0], [254, 254, 50],
        [254, 203, 0], [254, 152, 0], [254, 0, 0], [254, 0, 152],
        [152, 50, 203], [101, 0, 152]
    ]

    for i in range(len(colour_list)):
        colour_list[i] = numpy.array(colour_list[i], dtype=float) / 255

    colour_map_object = matplotlib.colors.ListedColormap(colour_list)
    colour_map_object.set_under(numpy.full(3, 1))

    colour_bounds_unitless = numpy.array(
        [1, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330, 360, 390, 420])
    colour_norm_object = matplotlib.colors.BoundaryNorm(
        colour_bounds_unitless, colour_map_object.N)

    return colour_map_object, colour_norm_object


def _get_vil_colour_scheme():
    """Returns colour scheme for VIL (vertically integrated liquid).

    :return: colour_map_object: Instance of `matplotlib.colors.ListedColormap`.
    :return: colour_norm_object: Instance of `matplotlib.colors.BoundaryNorm`.
    """

    colour_list = [
        [16, 71, 101], [0, 99, 132], [46, 132, 181], [74, 166, 218],
        [122, 207, 255], [179, 0, 179], [222, 83, 222], [255, 136, 255],
        [253, 191, 253], [255, 96, 0], [255, 128, 32], [255, 208, 0],
        [180, 0, 0], [224, 0, 0]
    ]

    for i in range(len(colour_list)):
        colour_list[i] = numpy.array(colour_list[i], dtype=float) / 255

    colour_map_object = matplotlib.colors.ListedColormap(colour_list)
    colour_map_object.set_under(numpy.full(3, 1))

    colour_bounds_mm = numpy.array(
        [0.1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70])
    colour_norm_object = matplotlib.colors.BoundaryNorm(
        colour_bounds_mm, colour_map_object.N)

    return colour_map_object, colour_norm_object


def _field_to_plotting_units(field_matrix, field_name):
    """Converts radar field from default units to plotting units.

    :param field_matrix: numpy array in default units.
    :param field_name: Name of radar field (must be accepted by
        `radar_utils.check_field_name`).
    :return: new_field_matrix: Same as input, except in plotting units.
    """

    radar_utils.check_field_name(field_name)

    if field_name in radar_utils.ECHO_TOP_NAMES:
        return field_matrix * KM_TO_KILOFEET

    if field_name in SHEAR_VORT_DIV_NAMES:
        return field_matrix * PER_SECOND_TO_PER_KILOSECOND

    return field_matrix


def _field_name_to_plotting_units(field_name):
    """Converts field *name* from default units to plotting units.

    :param field_name: Name of radar field (must be accepted by
        `radar_utils.check_field_name`).
    :return: new_field_name: Same as input, except in plotting units.
    """

    radar_utils.check_field_name(field_name)

    if field_name in radar_utils.ECHO_TOP_NAMES:
        return field_name.replace('_km', '_kft')

    if field_name in SHEAR_VORT_DIV_NAMES:
        return field_name.replace('_s01', '_ks01')

    return field_name


def layer_ops_to_field_and_panel_names(
        list_of_layer_operation_dicts, include_units=True):
    """Converts list of layer operations to list of field and panel names.

    P = number of layer operations = number of panels

    :param list_of_layer_operation_dicts: See doc for
        `input_examples.reduce_examples_3d_to_2d`.
    :param include_units: Boolean flag.  If True, panel names will include
        units.
    :return: field_name_by_panel: length-P list with names of radar fields.
    :return: panel_names: length-P list of panel names (to be printed at bottoms
        of panels).
    """

    error_checking.assert_is_boolean(include_units)

    num_panels = len(list_of_layer_operation_dicts)
    field_name_by_panel = [''] * num_panels
    panel_names = [''] * num_panels

    for i in range(num_panels):
        this_operation_dict = list_of_layer_operation_dicts[i]
        field_name_by_panel[i] = this_operation_dict[
            input_examples.RADAR_FIELD_KEY]

        this_min_height_m_agl = int(numpy.round(
            this_operation_dict[input_examples.MIN_HEIGHT_KEY] * METRES_TO_KM
        ))
        this_max_height_m_agl = int(numpy.round(
            this_operation_dict[input_examples.MAX_HEIGHT_KEY] * METRES_TO_KM
        ))

        this_field_name_verbose = FIELD_NAME_TO_VERBOSE_DICT[
            field_name_by_panel[i]
        ]

        if not include_units:
            this_field_name_verbose = this_field_name_verbose[
                :this_field_name_verbose.find(' (')
            ]

        panel_names[i] = '{0:s}\n{1:s} from {2:d}-{3:d} km AGL'.format(
            this_field_name_verbose,
            this_operation_dict[input_examples.OPERATION_NAME_KEY].upper(),
            this_min_height_m_agl, this_max_height_m_agl
        )

    return field_name_by_panel, panel_names


def radar_fields_and_heights_to_panel_names(
        field_names, heights_m_agl, include_units=True):
    """Converts list of radar field/height pairs to panel names.

    P = number of panels

    :param field_names: length-P list with names of radar fields.  Each must be
        accepted by `radar_utils.check_field_name`.
    :param heights_m_agl: length-P numpy array of heights (metres above ground
        level).
    :param include_units: Boolean flag.  If True, panel names will include
        units.
    :return: panel_names: length-P list of panel names (to be printed at bottoms
        of panels).
    """

    error_checking.assert_is_boolean(include_units)

    error_checking.assert_is_string_list(field_names)
    error_checking.assert_is_numpy_array(
        numpy.array(field_names), num_dimensions=1)
    num_panels = len(field_names)

    error_checking.assert_is_numpy_array(
        heights_m_agl, exact_dimensions=numpy.array([num_panels])
    )
    error_checking.assert_is_geq_numpy_array(heights_m_agl, 0.)
    heights_m_agl = numpy.round(heights_m_agl).astype(int)

    panel_names = [''] * num_panels

    for i in range(num_panels):
        this_field_name_verbose = FIELD_NAME_TO_VERBOSE_DICT[field_names[i]]

        if not include_units:
            this_field_name_verbose = this_field_name_verbose[
                :this_field_name_verbose.find(' (')
            ]

        panel_names[i] = '{0:s}\nat {1:.2f} km AGL'.format(
            this_field_name_verbose, heights_m_agl[i] * METRES_TO_KM)

    return panel_names


def get_default_colour_scheme(field_name):
    """Returns default colour scheme for radar field.

    :param field_name: Field name (must be accepted by
        `radar_utils.check_field_name`).
    :return: colour_map_object: Instance of `matplotlib.colors.ListedColormap`.
    :return: colour_norm_object: Instance of `matplotlib.colors.BoundaryNorm`.
    """

    radar_utils.check_field_name(field_name)

    if field_name in radar_utils.REFLECTIVITY_NAMES:
        return _get_reflectivity_colour_scheme()

    if field_name in radar_utils.SHEAR_NAMES:
        return _get_az_shear_colour_scheme()

    if field_name in radar_utils.ECHO_TOP_NAMES:
        return _get_echo_top_colour_scheme()

    if field_name == radar_utils.MESH_NAME:
        return _get_mesh_colour_scheme()

    if field_name == radar_utils.SHI_NAME:
        return _get_shi_colour_scheme()

    if field_name == radar_utils.VIL_NAME:
        return _get_vil_colour_scheme()

    if field_name == radar_utils.DIFFERENTIAL_REFL_NAME:
        return _get_zdr_colour_scheme()

    if field_name == radar_utils.SPEC_DIFF_PHASE_NAME:
        return _get_kdp_colour_scheme()

    if field_name == radar_utils.CORRELATION_COEFF_NAME:
        return _get_rho_hv_colour_scheme()

    if field_name == radar_utils.SPECTRUM_WIDTH_NAME:
        return _get_spectrum_width_colour_scheme()

    if field_name == radar_utils.VORTICITY_NAME:
        return _get_vorticity_colour_scheme()

    if field_name == radar_utils.DIVERGENCE_NAME:
        return _get_divergence_colour_scheme()

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
    :param colour_map_object: Instance of `matplotlib.pyplot.cm`.  If this is
        None, the default colour scheme for `field_name` will be used.
    :param colour_norm_object: Instance of `matplotlib.colors.BoundaryNorm`.  If
        this is None, the default colour scheme for `field_name` will be used.
    """

    field_matrix = _field_to_plotting_units(
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

    use_default_colour_scheme = (
        colour_map_object is None or colour_norm_object is None
    )

    if use_default_colour_scheme:
        colour_map_object, colour_norm_object = get_default_colour_scheme(
            field_name)
    else:
        if hasattr(colour_norm_object, 'boundaries'):
            colour_norm_object.boundaries = _field_to_plotting_units(
                field_matrix=colour_norm_object.boundaries,
                field_name=field_name)
        else:
            colour_norm_object.vmin = _field_to_plotting_units(
                field_matrix=colour_norm_object.vmin, field_name=field_name)
            colour_norm_object.vmax = _field_to_plotting_units(
                field_matrix=colour_norm_object.vmax, field_name=field_name)

    if hasattr(colour_norm_object, 'boundaries'):
        min_colour_value = colour_norm_object.boundaries[0]
        max_colour_value = colour_norm_object.boundaries[-1]
    else:
        min_colour_value = colour_norm_object.vmin
        max_colour_value = colour_norm_object.vmax

    if not use_default_colour_scheme:
        min_colour_value = _field_to_plotting_units(
            field_matrix=min_colour_value, field_name=field_name)
        max_colour_value = _field_to_plotting_units(
            field_matrix=max_colour_value, field_name=field_name)

    pyplot.pcolormesh(
        grid_cell_edge_longitudes_deg, grid_cell_edge_latitudes_deg,
        field_matrix_at_edges, cmap=colour_map_object, norm=colour_norm_object,
        vmin=min_colour_value, vmax=max_colour_value, shading='flat',
        edgecolors='None', axes=axes_object, zorder=-1e9)


def plot_2d_grid_without_coords(
        field_matrix, field_name, axes_object, font_size=DEFAULT_FONT_SIZE,
        annotation_string=None, colour_map_object=None,
        colour_norm_object=None):
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
    :param font_size: Font size for annotation.
    :param annotation_string: Annotation (will be printed in the bottom-center).
        If you want no annotation, leave this alone.
    :param colour_map_object: See doc for `plot_latlng_grid`.
    :param colour_norm_object: Same.
    :return: colour_map_object: Same as input, except default might have been
        set.
    :return: colour_norm_object: Same as input, except default might have been
        set.
    """

    error_checking.assert_is_numpy_array_without_nan(field_matrix)
    error_checking.assert_is_numpy_array(field_matrix, num_dimensions=2)

    field_matrix = _field_to_plotting_units(
        field_matrix=field_matrix, field_name=field_name)
    field_matrix = numpy.ma.masked_where(
        numpy.isnan(field_matrix), field_matrix)

    use_default_colour_scheme = (
        colour_map_object is None or colour_norm_object is None
    )

    if use_default_colour_scheme:
        colour_map_object, colour_norm_object = get_default_colour_scheme(
            field_name)
    else:
        if hasattr(colour_norm_object, 'boundaries'):
            colour_norm_object.boundaries = _field_to_plotting_units(
                field_matrix=colour_norm_object.boundaries,
                field_name=field_name)
        else:
            colour_norm_object.vmin = _field_to_plotting_units(
                field_matrix=colour_norm_object.vmin, field_name=field_name)
            colour_norm_object.vmax = _field_to_plotting_units(
                field_matrix=colour_norm_object.vmax, field_name=field_name)

    if hasattr(colour_norm_object, 'boundaries'):
        min_colour_value = colour_norm_object.boundaries[0]
        max_colour_value = colour_norm_object.boundaries[-1]
    else:
        min_colour_value = colour_norm_object.vmin
        max_colour_value = colour_norm_object.vmax

    axes_object.pcolormesh(
        field_matrix, cmap=colour_map_object, norm=colour_norm_object,
        vmin=min_colour_value, vmax=max_colour_value, shading='flat',
        edgecolors='None', zorder=-1e9)

    x_coord_limits = axes_object.get_xlim()
    x_grid_coords = numpy.linspace(
        x_coord_limits[0], x_coord_limits[1], num=5, dtype=float
    )[1:-1]

    y_coord_limits = axes_object.get_ylim()
    y_grid_coords = numpy.linspace(
        y_coord_limits[0], y_coord_limits[1], num=5, dtype=float
    )[1:-1]

    axes_object.set_xticks(x_grid_coords)
    axes_object.set_yticks(y_grid_coords)
    axes_object.grid(
        b=True, which='major', axis='both', linestyle='--', linewidth=2)

    axes_object.xaxis.set_ticklabels([])
    axes_object.yaxis.set_ticklabels([])
    axes_object.xaxis.set_ticks_position('none')
    axes_object.yaxis.set_ticks_position('none')

    if annotation_string is not None:
        error_checking.assert_is_string(annotation_string)

        bounding_box_dict = {
            'facecolor': 'white',
            'alpha': 0.5,
            'edgecolor': 'black',
            'linewidth': 2
        }

        axes_object.text(
            0.5, 0.01, annotation_string, fontsize=font_size, fontweight='bold',
            bbox=bounding_box_dict, color='k', horizontalalignment='center',
            verticalalignment='bottom', transform=axes_object.transAxes,
            zorder=1e10)

    return colour_map_object, colour_norm_object


def plot_many_2d_grids_without_coords(
        field_matrix, field_name_by_panel, num_panel_rows=None,
        figure_object=None, axes_object_matrix=None, panel_names=None,
        colour_map_object_by_panel=None, colour_norm_object_by_panel=None,
        plot_colour_bar_by_panel=None, font_size=DEFAULT_FONT_SIZE,
        row_major=True):
    """Plots 2-D colour map in each panel (one per field/height pair).

    M = number of rows in spatial grid
    N = number of columns in spatial grid
    P = number of panels (field/height pairs)

    This method uses the default colour scheme for each radar field.

    If `num_panel_rows is None`, this method needs arguments `figure_object` and
    `axes_object_matrix` -- and vice-versa.

    :param field_matrix: M-by-N-by-P numpy array of radar values.
    :param field_name_by_panel: length-P list of field names.
    :param num_panel_rows: Number of rows in paneled figure (different than M,
        which is number of rows in spatial grid).
    :param figure_object: See doc for `plotting_utils.create_paneled_figure`.
    :param axes_object_matrix: See above.
    :param panel_names: length-P list of panel names (will be printed at bottoms
        of panels).  If you do not want panel names, make this None.
    :param colour_map_object_by_panel: length-P list of `matplotlib.pyplot.cm`
        objects.  If this is None, the default will be used for each field.
    :param colour_norm_object_by_panel: length-P list of
        `matplotlib.colors.BoundaryNorm` objects.  If this is None, the default
        will be used for each field.
    :param plot_colour_bar_by_panel: length-P numpy array of Boolean flags.  If
        plot_colour_bar_by_panel[k] = True, horizontal colour bar will be
        plotted under [k]th panel.  If you want to plot colour bar for every
        panel, leave this as None.
    :param font_size: Font size.
    :param row_major: Boolean flag.  If True, panels will be filled along rows
        first, then down columns.  If False, down columns first, then along
        rows.
    :return: figure_object: See doc for `plotting_utils.create_paneled_figure`.
    :return: axes_object_matrix: Same.
    :raises: ValueError: if `colour_map_object_by_panel` or
        `colour_norm_object_by_panel` has different length than number of
        panels.
    """

    error_checking.assert_is_boolean(row_major)
    error_checking.assert_is_numpy_array(field_matrix, num_dimensions=3)
    num_panels = field_matrix.shape[2]

    if panel_names is None:
        panel_names = [None] * num_panels
    if plot_colour_bar_by_panel is None:
        plot_colour_bar_by_panel = numpy.full(num_panels, True, dtype=bool)

    these_expected_dim = numpy.array([num_panels], dtype=int)
    error_checking.assert_is_numpy_array(
        numpy.array(panel_names), exact_dimensions=these_expected_dim
    )
    error_checking.assert_is_numpy_array(
        numpy.array(field_name_by_panel), exact_dimensions=these_expected_dim
    )

    error_checking.assert_is_boolean_numpy_array(plot_colour_bar_by_panel)
    error_checking.assert_is_numpy_array(
        plot_colour_bar_by_panel, exact_dimensions=these_expected_dim)

    if (colour_map_object_by_panel is None
            or colour_norm_object_by_panel is None):
        colour_map_object_by_panel = [None] * num_panels
        colour_norm_object_by_panel = [None] * num_panels

    error_checking.assert_is_list(colour_map_object_by_panel)
    error_checking.assert_is_list(colour_norm_object_by_panel)

    if len(colour_map_object_by_panel) != num_panels:
        error_string = (
            'Number of colour maps ({0:d}) should equal number of panels '
            '({1:d}).'
        ).format(len(colour_map_object_by_panel), num_panels)

        raise ValueError(error_string)

    if len(colour_norm_object_by_panel) != num_panels:
        error_string = (
            'Number of colour-normalizers ({0:d}) should equal number of panels'
            ' ({1:d}).'
        ).format(len(colour_norm_object_by_panel), num_panels)

        raise ValueError(error_string)

    if figure_object is None:
        error_checking.assert_is_integer(num_panel_rows)
        error_checking.assert_is_geq(num_panel_rows, 1)
        error_checking.assert_is_leq(num_panel_rows, num_panels)

        num_panel_columns = int(numpy.ceil(
            float(num_panels) / num_panel_rows
        ))

        figure_object, axes_object_matrix = (
            plotting_utils.create_paneled_figure(
                num_rows=num_panel_rows, num_columns=num_panel_columns,
                shared_x_axis=False, shared_y_axis=False,
                keep_aspect_ratio=True)
        )
    else:
        error_checking.assert_is_numpy_array(
            axes_object_matrix, num_dimensions=2)

        num_panel_rows = axes_object_matrix.shape[0]
        num_panel_columns = axes_object_matrix.shape[1]

    if row_major:
        order_string = 'C'
    else:
        order_string = 'F'

    for k in range(num_panels):
        this_panel_row, this_panel_column = numpy.unravel_index(
            k, (num_panel_rows, num_panel_columns), order=order_string
        )

        # this_colour_map_object, this_colour_norm_object = (
        #     plot_2d_grid_without_coords(
        #         field_matrix=field_matrix[..., k],
        #         field_name=field_name_by_panel[k],
        #         axes_object=axes_object_matrix[
        #             this_panel_row, this_panel_column],
        #         annotation_string=panel_names[k], font_size=font_size,
        #         colour_map_object=colour_map_object_by_panel[k],
        #         colour_norm_object=colour_norm_object_by_panel[k]
        #     )
        # )

        this_colour_map_object, this_colour_norm_object = (
            plot_2d_grid_without_coords(
                field_matrix=field_matrix[..., k],
                field_name=field_name_by_panel[k],
                axes_object=axes_object_matrix[
                    this_panel_row, this_panel_column],
                annotation_string=None, font_size=font_size,
                colour_map_object=colour_map_object_by_panel[k],
                colour_norm_object=colour_norm_object_by_panel[k]
            )
        )

        if not plot_colour_bar_by_panel[k]:
            continue

        this_extend_min_flag = field_name_by_panel[k] in SHEAR_VORT_DIV_NAMES

        this_colour_bar_object = plotting_utils.plot_colour_bar(
            axes_object_or_matrix=axes_object_matrix[
                this_panel_row, this_panel_column],
            data_matrix=field_matrix[..., k],
            colour_map_object=this_colour_map_object,
            colour_norm_object=this_colour_norm_object,
            orientation_string='horizontal',
            extend_min=this_extend_min_flag, extend_max=True,
            fraction_of_axis_length=0.75, font_size=font_size)

        this_colour_bar_object.set_label(
            panel_names[k].replace('\n', '; '), fontsize=font_size,
            fontweight='bold')

    for k in range(num_panel_rows * num_panel_columns):
        if k < num_panels:
            continue

        this_panel_row, this_panel_column = numpy.unravel_index(
            k, (num_panel_rows, num_panel_columns), order=order_string
        )

        axes_object_matrix[this_panel_row, this_panel_column].axis('off')

    return figure_object, axes_object_matrix


def plot_3d_grid_without_coords(
        field_matrix, field_name, grid_point_heights_metres, ground_relative,
        num_panel_rows=None, figure_object=None, axes_object_matrix=None,
        font_size=DEFAULT_FONT_SIZE, colour_map_object=None,
        colour_norm_object=None):
    """Plots 3-D grid as many colour maps (one per height).

    M = number of grid rows
    N = number of grid columns
    H = number of grid heights

    To use the default colour scheme for the given radar field, leave
    `colour_map_object` and `colour_norm_object` empty.

    If `num_panel_rows is None`, this method needs arguments `figure_object` and
    `axes_object_matrix` -- and vice-versa.

    :param field_matrix: M-by-N-by-H numpy array with values of radar field.
    :param field_name: Name of radar field (must be accepted by
        `radar_utils.check_field_name`).
    :param grid_point_heights_metres: length-H integer numpy array of heights.
    :param ground_relative: Boolean flag.  If True, heights in
        `height_by_pair_metres` are ground-relative.  If False,
        sea-level-relative.
    :param num_panel_rows: Number of rows in paneled figure (different than M,
        the number of grid rows).
    :param figure_object: See doc for `plotting_utils.create_paneled_figure`.
    :param axes_object_matrix: See above.
    :param font_size: Font size for colour-bar ticks and panel labels.
    :param colour_map_object: See doc for `plot_latlng_grid`.
    :param colour_norm_object: Same.
    :return: figure_object: See doc for `plotting_utils.init_panels`.
    :return: axes_object_matrix: Same.
    """

    error_checking.assert_is_numpy_array(field_matrix, num_dimensions=3)
    error_checking.assert_is_geq_numpy_array(grid_point_heights_metres, 0)
    grid_point_heights_metres = numpy.round(
        grid_point_heights_metres
    ).astype(int)

    num_heights = field_matrix.shape[2]
    these_expected_dim = numpy.array([num_heights], dtype=int)
    error_checking.assert_is_numpy_array(
        grid_point_heights_metres, exact_dimensions=these_expected_dim)

    error_checking.assert_is_boolean(ground_relative)

    if figure_object is None:
        error_checking.assert_is_integer(num_panel_rows)
        error_checking.assert_is_geq(num_panel_rows, 1)
        error_checking.assert_is_leq(num_panel_rows, num_heights)

        num_panel_columns = int(numpy.ceil(
            float(num_heights) / num_panel_rows
        ))

        figure_object, axes_object_matrix = (
            plotting_utils.create_paneled_figure(
                num_rows=num_panel_rows, num_columns=num_panel_columns,
                shared_x_axis=False, shared_y_axis=False,
                keep_aspect_ratio=True)
        )
    else:
        error_checking.assert_is_numpy_array(
            axes_object_matrix, num_dimensions=2)

        num_panel_rows = axes_object_matrix.shape[0]
        num_panel_columns = axes_object_matrix.shape[1]

    for i in range(num_panel_rows):
        for j in range(num_panel_columns):
            this_height_index = i * num_panel_columns + j

            if this_height_index >= num_heights:
                axes_object_matrix[i, j].axis('off')
                continue

            this_annotation_string = '{0:.1f} km'.format(
                grid_point_heights_metres[this_height_index] * METRES_TO_KM
            )

            if ground_relative:
                this_annotation_string += ' AGL'
            else:
                this_annotation_string += ' ASL'

            plot_2d_grid_without_coords(
                field_matrix=field_matrix[..., this_height_index],
                field_name=field_name, axes_object=axes_object_matrix[i, j],
                annotation_string=this_annotation_string,
                colour_map_object=colour_map_object,
                colour_norm_object=colour_norm_object, font_size=font_size)

    return figure_object, axes_object_matrix
