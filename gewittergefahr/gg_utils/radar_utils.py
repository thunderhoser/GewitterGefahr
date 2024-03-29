"""Methods for handling radar data from all sources.

These sources are MYRORSS, MRMS, and GridRad.

--- DEFINITIONS ---

MYRORSS = Multi-year Reanalysis of Remotely Sensed Storms (Ortega et al. 2012)

MRMS = Multi-radar Multi-sensor network (Smith et al. 2016)

GridRad = radar-compositing software by Cameron Homeyer and Kenneth Bowman
(http://gridrad.org/pdf/GridRad-v3.1-Algorithm-Description.pdf)

--- REFERENCES ---

Ortega, K., and Coauthors, 2012: "The multi-year reanalysis of remotely sensed
    storms (MYRORSS) project". Conference on Severe Local Storms, Nashville, TN,
    American Meteorological Society.

Smith, T., and Coauthors, 2016: "Multi-radar Multi-sensor (MRMS) severe weather
    and aviation products: Initial operating capabilities". Bulletin of the
    American Meteorological Society, 97 (9), 1617-1630.
"""

import numpy
import scipy.interpolate
from gewittergefahr.gg_utils import number_rounding as rounder
from gewittergefahr.gg_utils import longitude_conversion as lng_conversion
from gewittergefahr.gg_utils import error_checking

NW_GRID_POINT_LAT_COLUMN = 'nw_grid_point_lat_deg'
NW_GRID_POINT_LNG_COLUMN = 'nw_grid_point_lng_deg'
LAT_SPACING_COLUMN = 'lat_spacing_deg'
LNG_SPACING_COLUMN = 'lng_spacing_deg'
NUM_LAT_COLUMN = 'num_lat_in_grid'
NUM_LNG_COLUMN = 'num_lng_in_grid'
HEIGHT_COLUMN = 'height_m_asl'
UNIX_TIME_COLUMN = 'unix_time_sec'
FIELD_NAME_COLUMN = 'field_name'
SENTINEL_VALUE_COLUMN = 'sentinel_values'

ECHO_TOP_15DBZ_NAME = 'echo_top_15dbz_km'
ECHO_TOP_18DBZ_NAME = 'echo_top_18dbz_km'
ECHO_TOP_20DBZ_NAME = 'echo_top_20dbz_km'
ECHO_TOP_25DBZ_NAME = 'echo_top_25dbz_km'
ECHO_TOP_40DBZ_NAME = 'echo_top_40dbz_km'
ECHO_TOP_50DBZ_NAME = 'echo_top_50dbz_km'
LOW_LEVEL_SHEAR_NAME = 'low_level_shear_s01'
MID_LEVEL_SHEAR_NAME = 'mid_level_shear_s01'
MESH_NAME = 'mesh_mm'
REFL_NAME = 'reflectivity_dbz'
REFL_COLUMN_MAX_NAME = 'reflectivity_column_max_dbz'
REFL_0CELSIUS_NAME = 'reflectivity_0celsius_dbz'
REFL_M10CELSIUS_NAME = 'reflectivity_m10celsius_dbz'
REFL_M20CELSIUS_NAME = 'reflectivity_m20celsius_dbz'
REFL_LOWEST_ALTITUDE_NAME = 'reflectivity_lowest_altitude_dbz'
SHI_NAME = 'shi'
VIL_NAME = 'vil_mm'
DIFFERENTIAL_REFL_NAME = 'differential_reflectivity_db'
SPEC_DIFF_PHASE_NAME = 'specific_differential_phase_deg_km01'
CORRELATION_COEFF_NAME = 'correlation_coefficient'
SPECTRUM_WIDTH_NAME = 'spectrum_width_m_s01'
VORTICITY_NAME = 'vorticity_s01'
DIVERGENCE_NAME = 'divergence_s01'
STORM_ID_NAME = 'storm_id_string'

RADAR_FIELD_NAMES = [
    ECHO_TOP_15DBZ_NAME, ECHO_TOP_18DBZ_NAME,
    ECHO_TOP_20DBZ_NAME, ECHO_TOP_25DBZ_NAME,
    ECHO_TOP_40DBZ_NAME, ECHO_TOP_50DBZ_NAME,
    LOW_LEVEL_SHEAR_NAME, MID_LEVEL_SHEAR_NAME,
    MESH_NAME, REFL_NAME,
    REFL_COLUMN_MAX_NAME, REFL_0CELSIUS_NAME,
    REFL_M10CELSIUS_NAME, REFL_M20CELSIUS_NAME,
    REFL_LOWEST_ALTITUDE_NAME, SHI_NAME, VIL_NAME,
    DIFFERENTIAL_REFL_NAME, SPEC_DIFF_PHASE_NAME,
    CORRELATION_COEFF_NAME, SPECTRUM_WIDTH_NAME,
    VORTICITY_NAME, DIVERGENCE_NAME,
    STORM_ID_NAME
]

FIELD_NAME_TO_VERBOSE_DICT = {
    ECHO_TOP_15DBZ_NAME: '15-dBZ echo top (m ASL)',
    ECHO_TOP_18DBZ_NAME: '18-dBZ echo top (m ASL)',
    ECHO_TOP_20DBZ_NAME: '20-dBZ echo top (m ASL)',
    ECHO_TOP_25DBZ_NAME: '25-dBZ echo top (m ASL)',
    ECHO_TOP_40DBZ_NAME: '40-dBZ echo top (m ASL)',
    ECHO_TOP_50DBZ_NAME: '50-dBZ echo top (m ASL)',
    LOW_LEVEL_SHEAR_NAME: r'Low-level shear (s$^{-1}$)',
    MID_LEVEL_SHEAR_NAME: r'Mid-level shear (s$^{-1}$)',
    MESH_NAME: 'Max estimated hail size (mm)',
    REFL_NAME: 'Reflectivity (dBZ)',
    REFL_COLUMN_MAX_NAME: 'Composite reflectivity (dBZ)',
    REFL_0CELSIUS_NAME: r'0 $^{\circ}C$ reflectivity (dBZ)',
    REFL_M10CELSIUS_NAME: r'-10 $^{\circ}C$ reflectivity (dBZ)',
    REFL_M20CELSIUS_NAME: r'-20 $^{\circ}C$ reflectivity (dBZ)',
    REFL_LOWEST_ALTITUDE_NAME: 'Lowest-altitude reflectivity (dBZ)',
    SHI_NAME: 'Severe-hail index',
    VIL_NAME: 'Vertically integrated liquid (mm)',
    DIFFERENTIAL_REFL_NAME: 'Differential reflectivity (dB)',
    SPEC_DIFF_PHASE_NAME: r'Specific differential phase ($^{\circ}$ km$^{-1}$)',
    CORRELATION_COEFF_NAME: 'Correlation coefficient',
    SPECTRUM_WIDTH_NAME: r'Spectrum width (m s$^{-1}$)',
    VORTICITY_NAME: r'Vorticity (s$^{-1}$)',
    DIVERGENCE_NAME: r'Divergence (s$^{-1}$)'
}

FIELD_NAME_TO_VERBOSE_UNITLESS_DICT = {
    ECHO_TOP_15DBZ_NAME: '15-dBZ echo top',
    ECHO_TOP_18DBZ_NAME: '18-dBZ echo top',
    ECHO_TOP_20DBZ_NAME: '20-dBZ echo top',
    ECHO_TOP_25DBZ_NAME: '25-dBZ echo top',
    ECHO_TOP_40DBZ_NAME: '40-dBZ echo top',
    ECHO_TOP_50DBZ_NAME: '50-dBZ echo top',
    LOW_LEVEL_SHEAR_NAME: 'Low-level shear',
    MID_LEVEL_SHEAR_NAME: 'Mid-level shear',
    MESH_NAME: 'Max estimated hail size',
    REFL_NAME: 'Reflectivity',
    REFL_COLUMN_MAX_NAME: 'Composite reflectivity',
    REFL_0CELSIUS_NAME: r'0 $^{\circ}C$ reflectivity',
    REFL_M10CELSIUS_NAME: r'-10 $^{\circ}C$ reflectivity',
    REFL_M20CELSIUS_NAME: r'-20 $^{\circ}C$ reflectivity',
    REFL_LOWEST_ALTITUDE_NAME: 'Lowest-altitude reflectivity',
    SHI_NAME: 'Severe-hail index',
    VIL_NAME: 'Vertically integrated liquid',
    DIFFERENTIAL_REFL_NAME: 'Differential reflectivity',
    SPEC_DIFF_PHASE_NAME: 'Specific differential phase',
    CORRELATION_COEFF_NAME: 'Correlation coefficient',
    SPECTRUM_WIDTH_NAME: 'Spectrum width',
    VORTICITY_NAME: 'Vorticity',
    DIVERGENCE_NAME: 'Divergence'
}

SHEAR_NAMES = [LOW_LEVEL_SHEAR_NAME, MID_LEVEL_SHEAR_NAME]
ECHO_TOP_NAMES = [
    ECHO_TOP_15DBZ_NAME, ECHO_TOP_18DBZ_NAME, ECHO_TOP_20DBZ_NAME,
    ECHO_TOP_25DBZ_NAME, ECHO_TOP_40DBZ_NAME, ECHO_TOP_50DBZ_NAME
]
REFLECTIVITY_NAMES = [
    REFL_NAME, REFL_COLUMN_MAX_NAME, REFL_0CELSIUS_NAME, REFL_M10CELSIUS_NAME,
    REFL_M20CELSIUS_NAME, REFL_LOWEST_ALTITUDE_NAME
]

FIELD_NAME_TO_MYRORSS_DICT = {
    ECHO_TOP_15DBZ_NAME: 'EchoTop_15',
    ECHO_TOP_18DBZ_NAME: 'EchoTop_18',
    ECHO_TOP_20DBZ_NAME: 'EchoTop_20',
    ECHO_TOP_25DBZ_NAME: 'EchoTop_25',
    ECHO_TOP_40DBZ_NAME: 'EchoTop_40',
    ECHO_TOP_50DBZ_NAME: 'EchoTop_50',
    LOW_LEVEL_SHEAR_NAME: 'MergedLLShear',
    MID_LEVEL_SHEAR_NAME: 'MergedMLShear',
    MESH_NAME: 'MESH',
    REFL_NAME: 'MergedReflectivityQC',
    REFL_COLUMN_MAX_NAME: 'MergedReflectivityQCComposite',
    REFL_0CELSIUS_NAME: 'Reflectivity_0C',
    REFL_M10CELSIUS_NAME: 'Reflectivity_-10C',
    REFL_M20CELSIUS_NAME: 'Reflectivity_-20C',
    REFL_LOWEST_ALTITUDE_NAME: 'ReflectivityAtLowestAltitude',
    SHI_NAME: 'SHI',
    VIL_NAME: 'VIL',
    STORM_ID_NAME: 'ClusterID'
}

FIELD_NAME_TO_MRMS_DICT = {
    ECHO_TOP_18DBZ_NAME: 'EchoTop_18',
    ECHO_TOP_50DBZ_NAME: 'EchoTop_50',
    LOW_LEVEL_SHEAR_NAME: 'MergedAzShear_0-2kmAGL',
    MID_LEVEL_SHEAR_NAME: 'MergedAzShear_3-6kmAGL',
    MESH_NAME: 'MESH',
    REFL_NAME: 'MergedReflectivityQC',
    REFL_COLUMN_MAX_NAME: 'MergedReflectivityQCComposite',
    REFL_0CELSIUS_NAME: 'Reflectivity_0C',
    REFL_M10CELSIUS_NAME: 'Reflectivity_-10C',
    REFL_M20CELSIUS_NAME: 'Reflectivity_-20C',
    REFL_LOWEST_ALTITUDE_NAME: 'ReflectivityAtLowestAltitude',
    SHI_NAME: 'SHI',
    VIL_NAME: 'VIL'
}

FIELD_NAME_TO_GRIDRAD_DICT = {
    REFL_NAME: 'ZH',
    SPECTRUM_WIDTH_NAME: 'SW',
    VORTICITY_NAME: 'VOR',
    DIVERGENCE_NAME: 'DIV',
    DIFFERENTIAL_REFL_NAME: 'ZDR',
    SPEC_DIFF_PHASE_NAME: 'KDP',
    CORRELATION_COEFF_NAME: 'RHV'
}

MRMS_SOURCE_ID = 'mrms'
MYRORSS_SOURCE_ID = 'myrorss'
GRIDRAD_SOURCE_ID = 'gridrad'
DATA_SOURCE_IDS = [MRMS_SOURCE_ID, MYRORSS_SOURCE_ID, GRIDRAD_SOURCE_ID]

SHEAR_HEIGHT_M_ASL = 250
DEFAULT_HEIGHT_MYRORSS_M_ASL = 250
DEFAULT_HEIGHT_MRMS_M_ASL = 500


def check_data_source(data_source):
    """Ensures that data source is recognized.

    :param data_source: Data source (string).
    :raises: ValueError: if `data_source not in DATA_SOURCE_IDS`.
    """

    error_checking.assert_is_string(data_source)

    if data_source not in DATA_SOURCE_IDS:
        error_string = (
            '\n{0:s}\nValid data sources (listed above) do not include "{1:s}".'
        ).format(str(DATA_SOURCE_IDS), data_source)

        raise ValueError(error_string)


def check_field_name(field_name):
    """Ensures that name of radar field is recognized.

    :param field_name: Name of radar field in GewitterGefahr format.
    :raises: ValueError: if name of radar field is not recognized.
    """

    error_checking.assert_is_string(field_name)

    if field_name not in RADAR_FIELD_NAMES:
        error_string = (
            '\n{0:s}\nValid radar fields (listed above) do not include "{1:s}".'
        ).format(str(RADAR_FIELD_NAMES), field_name)

        raise ValueError(error_string)


def field_name_to_verbose(field_name, include_units=True):
    """Converts field name from default format to verbose.

    :param field_name: Field name in default format (must be accepted by
        `check_field_name`).
    :param include_units: Boolean flag.  If True, verbose name will include
        units.
    :return: field_name_verbose: Verbose field name.
    """

    error_checking.assert_is_boolean(include_units)

    if include_units:
        return FIELD_NAME_TO_VERBOSE_DICT[field_name]

    return FIELD_NAME_TO_VERBOSE_UNITLESS_DICT[field_name]


def field_name_orig_to_new(field_name_orig, data_source_name):
    """Converts field name from original to new format.

    "Original format" means in original data source (MYRORSS, MRMS, or GridRad).

    "New format" means the default format in GewitterGefahr, which is Pythonic
    and includes units.

    :param field_name_orig: Field name in original format.
    :param data_source_name: Data source (must be accepted by
        `check_data_source`).
    :return: field_name: Field name in new format.
    """

    check_data_source(data_source_name)

    if data_source_name == MYRORSS_SOURCE_ID:
        conversion_dict = FIELD_NAME_TO_MYRORSS_DICT
    elif data_source_name == MRMS_SOURCE_ID:
        conversion_dict = FIELD_NAME_TO_MRMS_DICT
    elif data_source_name == GRIDRAD_SOURCE_ID:
        conversion_dict = FIELD_NAME_TO_GRIDRAD_DICT

    conversion_dict = dict([
        (value, key) for key, value in conversion_dict.items()
    ])

    return conversion_dict[field_name_orig]


def field_name_new_to_orig(field_name, data_source_name):
    """Converts field name from new to original format.

    "Original format" means in original data source (MYRORSS, MRMS, or GridRad).

    "New format" means the default format in GewitterGefahr, which is Pythonic
    and includes units.

    :param field_name: Field name in new format.
    :param data_source_name: Data source (must be accepted by
        `check_data_source`).
    :return: field_name_orig: Field name in original format.
    """

    check_data_source(data_source_name)

    if data_source_name == MYRORSS_SOURCE_ID:
        conversion_dict = FIELD_NAME_TO_MYRORSS_DICT
    elif data_source_name == MRMS_SOURCE_ID:
        conversion_dict = FIELD_NAME_TO_MRMS_DICT
    elif data_source_name == GRIDRAD_SOURCE_ID:
        conversion_dict = FIELD_NAME_TO_GRIDRAD_DICT

    return conversion_dict[field_name]


def field_name_to_echo_top_refl(field_name):
    """Parses critical echo-top reflectivity from name of radar field.

    :param field_name: Field name (must be in list `ECHO_TOP_NAMES`).
    :return: critical_reflectivity_dbz: Critical reflectivity.
    """

    if field_name not in ECHO_TOP_NAMES:
        error_string = (
            '\n{0:s}\nValid echo-top fields (listed above) do not include '
            '"{1:s}".'
        ).format(str(ECHO_TOP_NAMES), field_name)

        raise ValueError(error_string)

    critical_reflectivity_dbz = int(
        field_name.replace('echo_top_', '').replace('dbz_km', '')
    )
    return float(critical_reflectivity_dbz)


def get_valid_heights(data_source, field_name=None):
    """Finds valid heights for given data source and field.

    :param data_source: Data source (string).
    :param field_name: Field name in GewitterGefahr format (string).
    :return: valid_heights_m_asl: 1-D numpy array of valid heights (integer
        metres above sea level).
    :raises: ValueError: if field name is "full_id_string".
    """

    check_data_source(data_source)

    if data_source == GRIDRAD_SOURCE_ID:
        first_heights_m_asl = numpy.linspace(500, 7000, num=14, dtype=int)
        second_heights_m_asl = numpy.linspace(8000, 22000, num=15, dtype=int)

        return numpy.concatenate((first_heights_m_asl, second_heights_m_asl))

    check_field_name(field_name)
    if field_name == STORM_ID_NAME:
        raise ValueError('Field name cannot be "{0:s}".'.format(field_name))

    if data_source == MYRORSS_SOURCE_ID:
        default_height_m_asl = DEFAULT_HEIGHT_MYRORSS_M_ASL
    else:
        default_height_m_asl = DEFAULT_HEIGHT_MRMS_M_ASL

    if field_name in ECHO_TOP_NAMES:
        return numpy.array([default_height_m_asl])
    if field_name == LOW_LEVEL_SHEAR_NAME:
        return numpy.array([SHEAR_HEIGHT_M_ASL])
    if field_name == MID_LEVEL_SHEAR_NAME:
        return numpy.array([SHEAR_HEIGHT_M_ASL])
    if field_name == REFL_COLUMN_MAX_NAME:
        return numpy.array([default_height_m_asl])
    if field_name == MESH_NAME:
        return numpy.array([default_height_m_asl])
    if field_name == REFL_0CELSIUS_NAME:
        return numpy.array([default_height_m_asl])
    if field_name == REFL_M10CELSIUS_NAME:
        return numpy.array([default_height_m_asl])
    if field_name == REFL_M20CELSIUS_NAME:
        return numpy.array([default_height_m_asl])
    if field_name == REFL_LOWEST_ALTITUDE_NAME:
        return numpy.array([default_height_m_asl])
    if field_name == SHI_NAME:
        return numpy.array([default_height_m_asl])
    if field_name == VIL_NAME:
        return numpy.array([default_height_m_asl])

    if field_name == REFL_NAME:
        first_heights_m_asl = numpy.linspace(250, 3000, num=12, dtype=int)
        second_heights_m_asl = numpy.linspace(3500, 9000, num=12, dtype=int)
        third_heights_m_asl = numpy.linspace(10000, 20000, num=11, dtype=int)

        return numpy.concatenate((
            first_heights_m_asl, second_heights_m_asl, third_heights_m_asl
        ))


def check_heights(data_source, heights_m_asl, field_name=None):
    """Ensures validity of radar heights for the given source and field.

    :param data_source: Data source (string).
    :param heights_m_asl: 1-D numpy array of heights (metres above sea level).
    :param field_name: Field name in GewitterGefahr format (string).
    :raises: ValueError: if any element of `heights_m_asl` is invalid.
    """

    error_checking.assert_is_real_numpy_array(heights_m_asl)
    error_checking.assert_is_numpy_array(heights_m_asl, num_dimensions=1)

    integer_heights_m_asl = numpy.round(heights_m_asl).astype(int)
    valid_heights_m_asl = get_valid_heights(
        data_source=data_source, field_name=field_name)

    for this_height_m_asl in integer_heights_m_asl:
        if this_height_m_asl in valid_heights_m_asl:
            continue

        error_string = (
            '\n\n{0:s}\n\nValid heights for source "{1:s}" and field "{2:s}" '
            '(listed above in metres ASL) do not include the following: '
            '{3:d}'
        ).format(
            str(valid_heights_m_asl), data_source,
            'None' if field_name is None else field_name, this_height_m_asl
        )

        raise ValueError(error_string)


def rowcol_to_latlng(
        grid_rows, grid_columns, nw_grid_point_lat_deg, nw_grid_point_lng_deg,
        lat_spacing_deg, lng_spacing_deg):
    """Converts radar coordinates from row-column to lat-long.

    P = number of input grid points

    :param grid_rows: length-P numpy array with row indices of grid points
        (increasing from north to south).
    :param grid_columns: length-P numpy array with column indices of grid points
        (increasing from west to east).
    :param nw_grid_point_lat_deg: Latitude (deg N) of northwesternmost grid
        point.
    :param nw_grid_point_lng_deg: Longitude (deg E) of northwesternmost grid
        point.
    :param lat_spacing_deg: Spacing (deg N) between meridionally adjacent grid
        points.
    :param lng_spacing_deg: Spacing (deg E) between zonally adjacent grid
        points.
    :return: latitudes_deg: length-P numpy array with latitudes (deg N) of grid
        points.
    :return: longitudes_deg: length-P numpy array with longitudes (deg E) of
        grid points.
    """

    error_checking.assert_is_real_numpy_array(grid_rows)
    error_checking.assert_is_geq_numpy_array(grid_rows, -0.5, allow_nan=True)
    error_checking.assert_is_numpy_array(grid_rows, num_dimensions=1)
    num_points = len(grid_rows)

    error_checking.assert_is_real_numpy_array(grid_columns)
    error_checking.assert_is_geq_numpy_array(grid_columns, -0.5, allow_nan=True)
    error_checking.assert_is_numpy_array(
        grid_columns, exact_dimensions=numpy.array([num_points]))

    error_checking.assert_is_valid_latitude(nw_grid_point_lat_deg)
    nw_grid_point_lng_deg = lng_conversion.convert_lng_positive_in_west(
        nw_grid_point_lng_deg, allow_nan=False)

    error_checking.assert_is_greater(lat_spacing_deg, 0.)
    error_checking.assert_is_greater(lng_spacing_deg, 0.)

    latitudes_deg = rounder.round_to_nearest(
        nw_grid_point_lat_deg - lat_spacing_deg * grid_rows,
        lat_spacing_deg / 2)
    longitudes_deg = rounder.round_to_nearest(
        nw_grid_point_lng_deg + lng_spacing_deg * grid_columns,
        lng_spacing_deg / 2)
    return latitudes_deg, lng_conversion.convert_lng_positive_in_west(
        longitudes_deg, allow_nan=True)


def latlng_to_rowcol(
        latitudes_deg, longitudes_deg, nw_grid_point_lat_deg,
        nw_grid_point_lng_deg, lat_spacing_deg, lng_spacing_deg):
    """Converts radar coordinates from lat-long to row-column.

    P = number of input grid points

    :param latitudes_deg: length-P numpy array with latitudes (deg N) of grid
        points.
    :param longitudes_deg: length-P numpy array with longitudes (deg E) of
        grid points.
    :param nw_grid_point_lat_deg: Latitude (deg N) of northwesternmost grid
        point.
    :param nw_grid_point_lng_deg: Longitude (deg E) of northwesternmost grid
        point.
    :param lat_spacing_deg: Spacing (deg N) between meridionally adjacent grid
        points.
    :param lng_spacing_deg: Spacing (deg E) between zonally adjacent grid
        points.
    :return: grid_rows: length-P numpy array with row indices of grid points
        (increasing from north to south).
    :return: grid_columns: length-P numpy array with column indices of grid
        points (increasing from west to east).
    """

    error_checking.assert_is_valid_lat_numpy_array(
        latitudes_deg, allow_nan=True)
    error_checking.assert_is_numpy_array(latitudes_deg, num_dimensions=1)
    num_points = len(latitudes_deg)

    longitudes_deg = lng_conversion.convert_lng_positive_in_west(
        longitudes_deg, allow_nan=True)
    error_checking.assert_is_numpy_array(
        longitudes_deg, exact_dimensions=numpy.array([num_points]))

    error_checking.assert_is_valid_latitude(nw_grid_point_lat_deg)
    nw_grid_point_lng_deg = lng_conversion.convert_lng_positive_in_west(
        nw_grid_point_lng_deg, allow_nan=False)

    error_checking.assert_is_greater(lat_spacing_deg, 0.)
    error_checking.assert_is_greater(lng_spacing_deg, 0.)

    grid_columns = rounder.round_to_nearest(
        (longitudes_deg - nw_grid_point_lng_deg) / lng_spacing_deg, 0.5)
    grid_rows = rounder.round_to_nearest(
        (nw_grid_point_lat_deg - latitudes_deg) / lat_spacing_deg, 0.5)
    return grid_rows, grid_columns


def get_center_of_grid(
        nw_grid_point_lat_deg, nw_grid_point_lng_deg, lat_spacing_deg,
        lng_spacing_deg, num_grid_rows, num_grid_columns):
    """Finds center of radar grid.

    :param nw_grid_point_lat_deg: Latitude (deg N) of northwesternmost grid
        point.
    :param nw_grid_point_lng_deg: Longitude (deg E) of northwesternmost grid
        point.
    :param lat_spacing_deg: Spacing (deg N) between meridionally adjacent grid
        points.
    :param lng_spacing_deg: Spacing (deg E) between zonally adjacent grid
        points.
    :param num_grid_rows: Number of rows (unique grid-point latitudes).
    :param num_grid_columns: Number of columns (unique grid-point longitudes).
    :return: center_latitude_deg: Latitude (deg N) at center of grid.
    :return: center_longitude_deg: Longitude (deg E) at center of grid.
    """

    error_checking.assert_is_valid_latitude(nw_grid_point_lat_deg)
    nw_grid_point_lng_deg = lng_conversion.convert_lng_positive_in_west(
        nw_grid_point_lng_deg, allow_nan=False)

    error_checking.assert_is_greater(lat_spacing_deg, 0.)
    error_checking.assert_is_greater(lng_spacing_deg, 0.)
    error_checking.assert_is_integer(num_grid_rows)
    error_checking.assert_is_greater(num_grid_rows, 1)
    error_checking.assert_is_integer(num_grid_columns)
    error_checking.assert_is_greater(num_grid_columns, 1)

    min_latitude_deg = nw_grid_point_lat_deg - (
        (num_grid_rows - 1) * lat_spacing_deg)
    max_longitude_deg = nw_grid_point_lng_deg + (
        (num_grid_columns - 1) * lng_spacing_deg)

    return (numpy.mean(numpy.array([min_latitude_deg, nw_grid_point_lat_deg])),
            numpy.mean(numpy.array([nw_grid_point_lng_deg, max_longitude_deg])))


def get_echo_top_single_column(
        reflectivities_dbz, heights_m_asl, critical_reflectivity_dbz,
        check_args=False):
    """Finds echo top for a single column (horizontal location).

    "Echo top" = maximum height with reflectivity >= critical value.

    H = number of heights

    :param reflectivities_dbz: length-H numpy array of reflectivities.
    :param heights_m_asl: length-H numpy array of heights (metres above sea
        level).  This method assumes that heights are sorted in ascending order.
    :param critical_reflectivity_dbz: Critical reflectivity.
    :param check_args: Boolean flag.  If True, will check input arguments for
        errors.
    :return: echo_top_m_asl: Echo top.
    """

    error_checking.assert_is_boolean(check_args)
    if check_args:
        error_checking.assert_is_real_numpy_array(reflectivities_dbz)
        error_checking.assert_is_numpy_array(
            reflectivities_dbz, num_dimensions=1)

        num_heights = len(reflectivities_dbz)
        error_checking.assert_is_geq_numpy_array(heights_m_asl, 0.)
        error_checking.assert_is_numpy_array(
            heights_m_asl, exact_dimensions=numpy.array([num_heights]))

        error_checking.assert_is_greater(critical_reflectivity_dbz, 0.)

    critical_indices = numpy.where(
        reflectivities_dbz >= critical_reflectivity_dbz)[0]
    if len(critical_indices) == 0:
        return numpy.nan

    highest_critical_index = critical_indices[-1]
    subcritical_indices = numpy.where(
        reflectivities_dbz < critical_reflectivity_dbz)[0]
    subcritical_indices = subcritical_indices[
        subcritical_indices > highest_critical_index]

    if len(subcritical_indices) == 0:
        try:
            height_spacing_metres = (
                heights_m_asl[highest_critical_index + 1] -
                heights_m_asl[highest_critical_index])
        except IndexError:
            height_spacing_metres = (
                heights_m_asl[highest_critical_index] -
                heights_m_asl[highest_critical_index - 1])

        extrap_height_metres = height_spacing_metres * (
            1. - critical_reflectivity_dbz /
            reflectivities_dbz[highest_critical_index])
        return heights_m_asl[highest_critical_index] + extrap_height_metres

    adjacent_subcritical_index = subcritical_indices[0]
    indices_for_interp = numpy.array(
        [highest_critical_index, adjacent_subcritical_index], dtype=int)

    # if len(critical_indices) > 1:
    #     adjacent_critical_index = critical_indices[-2]
    #     indices_for_interp = numpy.array(
    #         [adjacent_critical_index, highest_critical_index,
    #          adjacent_subcritical_index], dtype=int)
    # else:
    #     indices_for_interp = numpy.array(
    #         [highest_critical_index, adjacent_subcritical_index], dtype=int)

    interp_object = scipy.interpolate.interp1d(
        reflectivities_dbz[indices_for_interp],
        heights_m_asl[indices_for_interp], kind='linear', bounds_error=False,
        fill_value='extrapolate', assume_sorted=False)
    return interp_object(critical_reflectivity_dbz)
