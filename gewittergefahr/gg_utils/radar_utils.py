"""Processing methods for radar data."""

import copy
import numpy
import scipy.interpolate
from gewittergefahr.gg_utils import number_rounding as rounder
from gewittergefahr.gg_utils import longitude_conversion as lng_conversion
from gewittergefahr.gg_utils import error_checking

METRES_TO_KM = 0.001

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

ECHO_TOP_18DBZ_NAME = 'echo_top_18dbz_km'
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
STORM_ID_NAME = 'storm_id'

ECHO_TOP_NAMES = [ECHO_TOP_18DBZ_NAME, ECHO_TOP_50DBZ_NAME, ECHO_TOP_40DBZ_NAME]
SHEAR_NAMES = [LOW_LEVEL_SHEAR_NAME, MID_LEVEL_SHEAR_NAME]
REFLECTIVITY_NAMES = [
    REFL_NAME, REFL_COLUMN_MAX_NAME, REFL_0CELSIUS_NAME, REFL_M10CELSIUS_NAME,
    REFL_M20CELSIUS_NAME, REFL_LOWEST_ALTITUDE_NAME]

ECHO_TOP_18DBZ_NAME_MYRORSS = 'EchoTop_18'
ECHO_TOP_40DBZ_NAME_MYRORSS = 'EchoTop_40'
ECHO_TOP_50DBZ_NAME_MYRORSS = 'EchoTop_50'
LOW_LEVEL_SHEAR_NAME_MYRORSS = 'MergedLLShear'
MID_LEVEL_SHEAR_NAME_MYRORSS = 'MergedMLShear'
MESH_NAME_MYRORSS = 'MESH'
REFL_NAME_MYRORSS = 'MergedReflectivityQC'
REFL_COLUMN_MAX_NAME_MYRORSS = 'MergedReflectivityQCComposite'
REFL_0CELSIUS_NAME_MYRORSS = 'Reflectivity_0C'
REFL_M10CELSIUS_NAME_MYRORSS = 'Reflectivity_-10C'
REFL_M20CELSIUS_NAME_MYRORSS = 'Reflectivity_-20C'
REFL_LOWEST_ALTITUDE_NAME_MYRORSS = 'ReflectivityAtLowestAltitude'
SHI_NAME_MYRORSS = 'SHI'
VIL_NAME_MYRORSS = 'VIL'
STORM_ID_NAME_MYRORSS = 'ClusterID'

ECHO_TOP_18DBZ_NAME_MRMS = 'EchoTop_18'
ECHO_TOP_50DBZ_NAME_MRMS = 'EchoTop_50'
LOW_LEVEL_SHEAR_NAME_MRMS = 'MergedAzShear_0-2kmAGL'
MID_LEVEL_SHEAR_NAME_MRMS = 'MergedAzShear_3-6kmAGL'
MESH_NAME_MRMS = 'MESH'
REFL_NAME_MRMS = 'MergedReflectivityQC'
REFL_COLUMN_MAX_NAME_MRMS = 'MergedReflectivityQCComposite'
REFL_0CELSIUS_NAME_MRMS = 'Reflectivity_0C'
REFL_M10CELSIUS_NAME_MRMS = 'Reflectivity_-10C'
REFL_M20CELSIUS_NAME_MRMS = 'Reflectivity_-20C'
REFL_LOWEST_ALTITUDE_NAME_MRMS = 'ReflectivityAtLowestAltitude'
SHI_NAME_MRMS = 'SHI'
VIL_NAME_MRMS = 'VIL'

REFL_NAME_GRIDRAD = 'ZH'
DIFFERENTIAL_REFL_NAME_GRIDRAD = 'ZDR'
SPEC_DIFF_PHASE_NAME_GRIDRAD = 'KDP'
CORRELATION_COEFF_NAME_GRIDRAD = 'RHV'
SPECTRUM_WIDTH_NAME_GRIDRAD = 'SW'
VORTICITY_NAME_GRIDRAD = 'VOR'
DIVERGENCE_NAME_GRIDRAD = 'DIV'

MRMS_SOURCE_ID = 'mrms'
MYRORSS_SOURCE_ID = 'myrorss'
GRIDRAD_SOURCE_ID = 'gridrad'
DATA_SOURCE_IDS = [MRMS_SOURCE_ID, MYRORSS_SOURCE_ID, GRIDRAD_SOURCE_ID]

RADAR_FIELD_NAMES = [
    ECHO_TOP_18DBZ_NAME, ECHO_TOP_40DBZ_NAME,
    ECHO_TOP_50DBZ_NAME, LOW_LEVEL_SHEAR_NAME,
    MID_LEVEL_SHEAR_NAME, MESH_NAME, REFL_NAME,
    REFL_COLUMN_MAX_NAME, REFL_0CELSIUS_NAME,
    REFL_M10CELSIUS_NAME, REFL_M20CELSIUS_NAME,
    REFL_LOWEST_ALTITUDE_NAME, SHI_NAME, VIL_NAME,
    DIFFERENTIAL_REFL_NAME, SPEC_DIFF_PHASE_NAME,
    CORRELATION_COEFF_NAME, SPECTRUM_WIDTH_NAME,
    VORTICITY_NAME, DIVERGENCE_NAME,
    STORM_ID_NAME]

RADAR_FIELD_NAMES_MYRORSS = [
    ECHO_TOP_18DBZ_NAME_MYRORSS, ECHO_TOP_40DBZ_NAME_MYRORSS,
    ECHO_TOP_50DBZ_NAME_MYRORSS, LOW_LEVEL_SHEAR_NAME_MYRORSS,
    MID_LEVEL_SHEAR_NAME_MYRORSS, MESH_NAME_MYRORSS, REFL_NAME_MYRORSS,
    REFL_COLUMN_MAX_NAME_MYRORSS, REFL_0CELSIUS_NAME_MYRORSS,
    REFL_M10CELSIUS_NAME_MYRORSS, REFL_M20CELSIUS_NAME_MYRORSS,
    REFL_LOWEST_ALTITUDE_NAME_MYRORSS, SHI_NAME_MYRORSS, VIL_NAME_MYRORSS,
    STORM_ID_NAME_MYRORSS]

RADAR_FIELD_NAMES_MRMS = [
    ECHO_TOP_18DBZ_NAME_MRMS, ECHO_TOP_50DBZ_NAME_MRMS,
    LOW_LEVEL_SHEAR_NAME_MRMS, MID_LEVEL_SHEAR_NAME_MRMS, MESH_NAME_MRMS,
    REFL_NAME_MRMS, REFL_COLUMN_MAX_NAME_MRMS, REFL_0CELSIUS_NAME_MRMS,
    REFL_M10CELSIUS_NAME_MRMS, REFL_M20CELSIUS_NAME_MRMS,
    REFL_LOWEST_ALTITUDE_NAME_MRMS, SHI_NAME_MRMS, VIL_NAME_MRMS]

RADAR_FIELD_NAMES_GRIDRAD = [
    REFL_NAME_GRIDRAD, DIFFERENTIAL_REFL_NAME_GRIDRAD,
    SPEC_DIFF_PHASE_NAME_GRIDRAD, CORRELATION_COEFF_NAME_GRIDRAD,
    SPECTRUM_WIDTH_NAME_GRIDRAD, VORTICITY_NAME_GRIDRAD,
    DIVERGENCE_NAME_GRIDRAD]

RADAR_FIELD_NAMES_MYRORSS_PADDED = [
    ECHO_TOP_18DBZ_NAME_MYRORSS, ECHO_TOP_40DBZ_NAME_MYRORSS,
    ECHO_TOP_50DBZ_NAME_MYRORSS, LOW_LEVEL_SHEAR_NAME_MYRORSS,
    MID_LEVEL_SHEAR_NAME_MYRORSS, MESH_NAME_MYRORSS, REFL_NAME_MYRORSS,
    REFL_COLUMN_MAX_NAME_MYRORSS, REFL_0CELSIUS_NAME_MYRORSS,
    REFL_M10CELSIUS_NAME_MYRORSS, REFL_M20CELSIUS_NAME_MYRORSS,
    REFL_LOWEST_ALTITUDE_NAME_MYRORSS, SHI_NAME_MYRORSS, VIL_NAME_MYRORSS,
    None, None,
    None, None,
    None, None,
    STORM_ID_NAME_MYRORSS]

RADAR_FIELD_NAMES_MRMS_PADDED = [
    ECHO_TOP_18DBZ_NAME_MRMS, None,
    ECHO_TOP_50DBZ_NAME_MRMS, LOW_LEVEL_SHEAR_NAME_MRMS,
    MID_LEVEL_SHEAR_NAME_MRMS, MESH_NAME_MRMS, REFL_NAME_MRMS,
    REFL_COLUMN_MAX_NAME_MRMS, REFL_0CELSIUS_NAME_MRMS,
    REFL_M10CELSIUS_NAME_MRMS, REFL_M20CELSIUS_NAME_MRMS,
    REFL_LOWEST_ALTITUDE_NAME_MRMS, SHI_NAME_MRMS, VIL_NAME_MRMS,
    None, None,
    None, None,
    None, None,
    None]

RADAR_FIELD_NAMES_GRIDRAD_PADDED = [
    None, None,
    None, None,
    None, None, REFL_NAME_GRIDRAD,
    None, None,
    None, None,
    None, None, None,
    DIFFERENTIAL_REFL_NAME_GRIDRAD, SPEC_DIFF_PHASE_NAME_GRIDRAD,
    CORRELATION_COEFF_NAME_GRIDRAD, SPECTRUM_WIDTH_NAME_GRIDRAD,
    VORTICITY_NAME_GRIDRAD, DIVERGENCE_NAME_GRIDRAD,
    None]

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
            '\n\n' + str(DATA_SOURCE_IDS) +
            '\n\nValid data sources (listed above) do not include "' +
            data_source + '".')
        raise ValueError(error_string)


def check_field_name(field_name):
    """Ensures that name of radar field is recognized.

    :param field_name: Name of radar field in GewitterGefahr format.
    :raises: ValueError: if name of radar field is not recognized.
    """

    if field_name not in RADAR_FIELD_NAMES:
        error_string = (
            '\n\n' + str(RADAR_FIELD_NAMES) +
            '\n\nValid field names (listed above) do not include "' +
            field_name + '".')
        raise ValueError(error_string)


def check_field_name_orig(field_name_orig, data_source):
    """Ensures that name of radar field is recognized.

    :param field_name_orig: Name of radar field in original (either MYRORSS or
        MRMS) format.
    :param data_source: Data source (string).
    :raises: ValueError: if name of radar field is not recognized.
    """

    check_data_source(data_source)

    if data_source == MYRORSS_SOURCE_ID:
        valid_field_names = RADAR_FIELD_NAMES_MYRORSS
    elif data_source == MRMS_SOURCE_ID:
        valid_field_names = RADAR_FIELD_NAMES_MRMS
    elif data_source == GRIDRAD_SOURCE_ID:
        valid_field_names = RADAR_FIELD_NAMES_GRIDRAD

    if field_name_orig not in valid_field_names:
        error_string = (
            '\n\n' + str(valid_field_names) +
            '\n\nValid field names (listed above) do not include "' +
            field_name_orig + '".')
        raise ValueError(error_string)


def field_name_orig_to_new(field_name_orig, data_source):
    """Converts field name from original to new format.

    "Original format" = in original data source (examples: MYRORSS, MRMS,
    GridRad).

    "New format" = GewitterGefahr format, which is Pythonic and includes units
    at the end.

    :param field_name_orig: Name of radar field in original format.
    :param data_source: Data source (string).
    :return: field_name: Name of radar field in new format.
    """

    error_checking.assert_is_string(field_name_orig)
    check_data_source(data_source)

    if data_source == MYRORSS_SOURCE_ID:
        all_orig_field_names = RADAR_FIELD_NAMES_MYRORSS_PADDED
    elif data_source == MRMS_SOURCE_ID:
        all_orig_field_names = RADAR_FIELD_NAMES_MRMS_PADDED
    elif data_source == GRIDRAD_SOURCE_ID:
        all_orig_field_names = RADAR_FIELD_NAMES_GRIDRAD_PADDED

    found_flags = [s == field_name_orig for s in all_orig_field_names]
    return RADAR_FIELD_NAMES[numpy.where(found_flags)[0][0]]


def field_name_new_to_orig(field_name, data_source):
    """Converts field name from new to original format.

    "Original format" = in original data source (examples: MYRORSS, MRMS,
    GridRad).

    "New format" = GewitterGefahr format, which is Pythonic and includes units
    at the end.

    :param field_name: Name of radar field in new format.
    :param data_source: Data source (string).
    :return: field_name_orig: Name of radar field in original format.
    :raises: ValueError: if field does not exist for given data source.
    """

    check_data_source(data_source)

    if data_source == MYRORSS_SOURCE_ID:
        all_orig_field_names = RADAR_FIELD_NAMES_MYRORSS_PADDED
    elif data_source == MRMS_SOURCE_ID:
        all_orig_field_names = RADAR_FIELD_NAMES_MRMS_PADDED
    elif data_source == GRIDRAD_SOURCE_ID:
        all_orig_field_names = RADAR_FIELD_NAMES_GRIDRAD_PADDED

    found_flags = [s == field_name for s in RADAR_FIELD_NAMES]
    field_name_orig = all_orig_field_names[numpy.where(found_flags)[0][0]]
    if field_name_orig is None:
        error_string = (
            'Field "{0:s}" does not exist for data source "{1:s}"').format(
                field_name, data_source)
        raise ValueError(error_string)

    return field_name_orig


def get_valid_heights_for_field(field_name, data_source):
    """Returns valid heights for radar field.

    :param field_name: Name of radar field in GewitterGefahr format.
    :param data_source: Data source (string).
    :return: valid_heights_m_asl: 1-D numpy array of valid heights (integer
        metres above sea level).
    :raises: ValueError: if data source is "gridrad".
    :raises: ValueError: if field name is "storm_id".
    """

    check_data_source(data_source)
    if data_source == GRIDRAD_SOURCE_ID:
        raise ValueError('Data source cannot be "{0:s}".'.format(data_source))

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
        return numpy.array(
            [250, 500, 750, 1000, 1250, 1500, 1750, 2000, 2250, 2500, 2750,
             3000, 3500, 4000, 4500, 5000, 5500, 6000, 6500, 7000, 7500, 8000,
             8500, 9000, 10000, 11000, 12000, 13000, 14000, 15000, 16000, 17000,
             18000, 19000, 20000])


def check_reflectivity_heights(heights_m_asl, data_source):
    """Ensures that reflectivity heights are valid.

    :param heights_m_asl: 1-D numpy array of reflectivity heights (metres above
        sea level).
    :param data_source: Data source (string).
    :raises: ValueError: if any element of heights_m_asl is invalid.
    """

    error_checking.assert_is_real_numpy_array(heights_m_asl)
    error_checking.assert_is_numpy_array(heights_m_asl, num_dimensions=1)

    integer_heights_m_asl = numpy.round(heights_m_asl).astype(int)
    valid_heights_m_asl = get_valid_heights_for_field(REFL_NAME, data_source)

    for this_height_m_asl in integer_heights_m_asl:
        if this_height_m_asl in valid_heights_m_asl:
            continue

        error_string = (
            '\n\n' + str(valid_heights_m_asl) +
            '\n\nValid reflectivity heights (metres ASL, listed above) do not '
            'include ' + str(this_height_m_asl) + ' m ASL.')
        raise ValueError(error_string)


def field_and_height_arrays_to_dict(
        field_names, data_source, refl_heights_m_asl=None):
    """Converts two arrays (field names and reflectivity heights) to dictionary.

    :param field_names: 1-D list with names of radar fields in GewitterGefahr
        format.
    :param data_source: Data source (string).
    :param refl_heights_m_asl: 1-D numpy array of reflectivity heights (metres
        above sea level).
    :return: field_to_heights_dict_m_asl: Dictionary, where each key comes from
        `field_names` and each value is a 1-D numpy array of heights (metres
        above sea level).
    """

    field_to_heights_dict_m_asl = {}

    for this_field_name in field_names:
        if this_field_name == REFL_NAME:
            check_reflectivity_heights(
                refl_heights_m_asl, data_source=data_source)
            field_to_heights_dict_m_asl.update(
                {this_field_name: refl_heights_m_asl})
        else:
            field_to_heights_dict_m_asl.update({
                this_field_name: get_valid_heights_for_field(
                    this_field_name, data_source=data_source)})

    return field_to_heights_dict_m_asl


def unique_fields_and_heights_to_pairs(
        unique_field_names, data_source, refl_heights_m_asl=None):
    """Converts unique arrays (field names and refl heights) to non-unique ones.

    F = number of unique field names
    N = number of field-height pairs

    :param unique_field_names: length-F list with names of radar fields in
        GewitterGefahr format.
    :param data_source: Data source (string).
    :param refl_heights_m_asl: 1-D numpy array of reflectivity heights (metres
        above sea level).
    :return: field_name_by_pair: length-N list of field names.
    :return: height_by_pair_m_asl: length-N numpy array of radar heights (metres
        above sea level).
    """

    field_name_by_pair = []
    height_by_pair_m_asl = numpy.array([])

    for this_field_name in unique_field_names:
        if this_field_name == REFL_NAME:
            check_reflectivity_heights(
                refl_heights_m_asl, data_source=data_source)
            these_heights_m_asl = copy.deepcopy(refl_heights_m_asl)
        else:
            these_heights_m_asl = get_valid_heights_for_field(
                this_field_name, data_source=data_source)

        field_name_by_pair += [this_field_name] * len(these_heights_m_asl)
        height_by_pair_m_asl = numpy.concatenate((
            height_by_pair_m_asl, these_heights_m_asl))

    return field_name_by_pair, height_by_pair_m_asl


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
