"""IO methods for radar data from MYRORSS or MRMS.

MYRORSS = Multi-year Reanalysis of Remotely Sensed Storms

MRMS = Multi-radar Multi-sensor
"""

import copy
import os
import glob
import numpy
import pandas
from gewittergefahr.gg_io import netcdf_io
from gewittergefahr.gg_utils import number_rounding as rounder
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import time_periods
from gewittergefahr.gg_utils import longitude_conversion as lng_conversion
from gewittergefahr.gg_utils import grids
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

NW_GRID_POINT_LAT_COLUMN_ORIG = 'Latitude'
NW_GRID_POINT_LNG_COLUMN_ORIG = 'Longitude'
LAT_SPACING_COLUMN_ORIG = 'LatGridSpacing'
LNG_SPACING_COLUMN_ORIG = 'LonGridSpacing'
NUM_LAT_COLUMN_ORIG = 'Lat'
NUM_LNG_COLUMN_ORIG = 'Lon'
NUM_PIXELS_COLUMN_ORIG = 'pixel'
HEIGHT_COLUMN_ORIG = 'Height'
UNIX_TIME_COLUMN_ORIG = 'Time'
FIELD_NAME_COLUMN_ORIG = 'TypeName'
SENTINEL_VALUE_COLUMNS_ORIG = ['MissingData', 'RangeFolded']

GRID_ROW_COLUMN = 'grid_row'
GRID_COLUMN_COLUMN = 'grid_column'
GRID_LAT_COLUMN = 'latitude_deg'
GRID_LNG_COLUMN = 'longitude_deg'
NUM_GRID_CELL_COLUMN = 'num_grid_cells'

GRID_ROW_COLUMN_ORIG = 'pixel_x'
GRID_COLUMN_COLUMN_ORIG = 'pixel_y'
NUM_GRID_CELL_COLUMN_ORIG = 'pixel_count'

ECHO_TOP_18DBZ_NAME = 'echo_top_18dbz_km'
ECHO_TOP_40DBZ_NAME = 'echo_top_40dbz_km'
ECHO_TOP_50DBZ_NAME = 'echo_top_50dbz_km'
LOW_LEVEL_SHEAR_NAME = 'low_level_shear_s01'
MID_LEVEL_SHEAR_NAME = 'mid_level_shear_s01'
REFL_NAME = 'reflectivity_dbz'
REFL_COLUMN_MAX_NAME = 'reflectivity_column_max_dbz'
MESH_NAME = 'mesh_mm'
REFL_0CELSIUS_NAME = 'reflectivity_0celsius_dbz'
REFL_M10CELSIUS_NAME = 'reflectivity_m10celsius_dbz'
REFL_M20CELSIUS_NAME = 'reflectivity_m20celsius_dbz'
REFL_LOWEST_ALTITUDE_NAME = 'reflectivity_lowest_altitude_dbz'
SHI_NAME = 'shi'
VIL_NAME = 'vil_mm'
STORM_ID_NAME = 'storm_id'

ECHO_TOP_NAMES = [ECHO_TOP_18DBZ_NAME, ECHO_TOP_50DBZ_NAME, ECHO_TOP_40DBZ_NAME]
SHEAR_NAMES = [LOW_LEVEL_SHEAR_NAME, MID_LEVEL_SHEAR_NAME]
REFLECTIVITY_NAMES = [
    REFL_NAME, REFL_COLUMN_MAX_NAME, REFL_0CELSIUS_NAME, REFL_M10CELSIUS_NAME,
    REFL_M20CELSIUS_NAME, REFL_LOWEST_ALTITUDE_NAME]

ECHO_TOP_18DBZ_NAME_ORIG = 'EchoTop_18'
ECHO_TOP_40DBZ_NAME_ORIG = 'EchoTop_40'
ECHO_TOP_50DBZ_NAME_ORIG = 'EchoTop_50'
REFL_NAME_ORIG = 'MergedReflectivityQC'
REFL_COLUMN_MAX_NAME_ORIG = 'MergedReflectivityQCComposite'
MESH_NAME_ORIG = 'MESH'
REFL_0CELSIUS_NAME_ORIG = 'Reflectivity_0C'
REFL_M10CELSIUS_NAME_ORIG = 'Reflectivity_-10C'
REFL_M20CELSIUS_NAME_ORIG = 'Reflectivity_-20C'
REFL_LOWEST_ALTITUDE_NAME_ORIG = 'ReflectivityAtLowestAltitude'
SHI_NAME_ORIG = 'SHI'
VIL_NAME_ORIG = 'VIL'
STORM_ID_NAME_ORIG = 'ClusterID'

MRMS_SOURCE_ID = 'mrms'
MYRORSS_SOURCE_ID = 'myrorss'
DATA_SOURCE_IDS = [MRMS_SOURCE_ID, MYRORSS_SOURCE_ID]

LOW_LEVEL_SHEAR_NAME_MYRORSS = 'MergedLLShear'
MID_LEVEL_SHEAR_NAME_MYRORSS = 'MergedMLShear'
LOW_LEVEL_SHEAR_NAME_MRMS = 'MergedAzShear_0-2kmAGL'
MID_LEVEL_SHEAR_NAME_MRMS = 'MergedAzShear_3-6kmAGL'

RADAR_FIELD_NAMES = [
    ECHO_TOP_18DBZ_NAME, ECHO_TOP_40DBZ_NAME,
    ECHO_TOP_50DBZ_NAME, LOW_LEVEL_SHEAR_NAME,
    MID_LEVEL_SHEAR_NAME, REFL_NAME, REFL_COLUMN_MAX_NAME,
    MESH_NAME, REFL_0CELSIUS_NAME, REFL_M10CELSIUS_NAME,
    REFL_M20CELSIUS_NAME, REFL_LOWEST_ALTITUDE_NAME, SHI_NAME,
    VIL_NAME, STORM_ID_NAME]

RADAR_FIELD_NAMES_MYRORSS = [
    ECHO_TOP_18DBZ_NAME_ORIG, ECHO_TOP_40DBZ_NAME_ORIG,
    ECHO_TOP_50DBZ_NAME_ORIG, LOW_LEVEL_SHEAR_NAME_MYRORSS,
    MID_LEVEL_SHEAR_NAME_MYRORSS, REFL_NAME_ORIG, REFL_COLUMN_MAX_NAME_ORIG,
    MESH_NAME_ORIG, REFL_0CELSIUS_NAME_ORIG, REFL_M10CELSIUS_NAME_ORIG,
    REFL_M20CELSIUS_NAME_ORIG, REFL_LOWEST_ALTITUDE_NAME_ORIG, SHI_NAME_ORIG,
    VIL_NAME_ORIG, STORM_ID_NAME_ORIG]

RADAR_FIELD_NAMES_MRMS = [
    ECHO_TOP_18DBZ_NAME_ORIG, ECHO_TOP_40DBZ_NAME_ORIG,
    ECHO_TOP_50DBZ_NAME_ORIG, LOW_LEVEL_SHEAR_NAME_MRMS,
    MID_LEVEL_SHEAR_NAME_MRMS, REFL_NAME_ORIG, REFL_COLUMN_MAX_NAME_ORIG,
    MESH_NAME_ORIG, REFL_0CELSIUS_NAME_ORIG, REFL_M10CELSIUS_NAME_ORIG,
    REFL_M20CELSIUS_NAME_ORIG, REFL_LOWEST_ALTITUDE_NAME_ORIG, SHI_NAME_ORIG,
    VIL_NAME_ORIG, STORM_ID_NAME_ORIG]

TIME_FORMAT_SECONDS = '%Y%m%d-%H%M%S'
TIME_FORMAT_MINUTES = '%Y%m%d-%H%M'
TIME_FORMAT_FOR_LOG_MESSAGES = '%Y-%m-%d-%H%M%S'
DAYS_TO_SECONDS = 86400
MINUTES_TO_SECONDS = 60

METRES_TO_KM = 1e-3
SENTINEL_TOLERANCE = 10.

SHEAR_HEIGHT_M_ASL = 250
DEFAULT_HEIGHT_MYRORSS_M_ASL = 250
DEFAULT_HEIGHT_MRMS_M_ASL = 500
DEFAULT_MAX_TIME_OFFSET_FOR_AZ_SHEAR_SEC = 180

ZIPPED_FILE_EXTENSION = '.gz'
UNZIPPED_FILE_EXTENSION = '.netcdf'


def _check_data_source(data_source):
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


def _check_field_name_orig(field_name_orig, data_source):
    """Ensures that name of radar field is recognized.

    :param field_name_orig: Name of radar field in original (either MYRORSS or
        MRMS) format.
    :param data_source: Data source (string).
    :raises: ValueError: if name of radar field is not recognized.
    """

    if data_source == MYRORSS_SOURCE_ID:
        valid_field_names = RADAR_FIELD_NAMES_MYRORSS
    else:
        valid_field_names = RADAR_FIELD_NAMES_MRMS

    if field_name_orig not in valid_field_names:
        error_string = (
            '\n\n' + str(valid_field_names) +
            '\n\nValid field names (listed above) do not include "' +
            field_name_orig + '".')
        raise ValueError(error_string)


def _field_name_orig_to_new(field_name_orig, data_source):
    """Converts field name from original to new format.

    "Original format" = MYRORSS or MRMS
    "New format" = GewitterGefahr format, which is Pythonic and includes units
                   at the end

    :param field_name_orig: Name of radar field in original (either MYRORSS or
        MRMS) format.
    :param data_source: Data source (string).
    :return: field_name: Name of radar field in new format.
    """

    if data_source == MYRORSS_SOURCE_ID:
        all_orig_field_names = RADAR_FIELD_NAMES_MYRORSS
    else:
        all_orig_field_names = RADAR_FIELD_NAMES_MRMS

    found_flags = [s == field_name_orig for s in all_orig_field_names]
    return RADAR_FIELD_NAMES[numpy.where(found_flags)[0][0]]


def _check_reflectivity_heights(heights_m_asl, data_source):
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


def _get_pathless_raw_file_pattern(unix_time_sec):
    """Generates glob pattern for pathless name of raw file.

    This method rounds the time step to the nearest minute and allows the file
    to be either zipped or unzipped.

    The pattern generated by this method is meant for input to `glob.glob`.
    This method is the "pattern" version of _get_pathless_raw_file_name.

    :param unix_time_sec: Valid time.
    :return: pathless_raw_file_pattern: Pathless glob pattern for raw file.
    """

    return '{0:s}*{1:s}*'.format(
        time_conversion.unix_sec_to_string(unix_time_sec, TIME_FORMAT_MINUTES),
        UNZIPPED_FILE_EXTENSION)


def _get_pathless_raw_file_name(unix_time_sec, zipped=True):
    """Generates pathless name for raw file.

    :param unix_time_sec: Valid time.
    :param zipped: Boolean flag.  If True, will generate name for zipped file.
        If False, will generate name for unzipped file.
    :return: pathless_raw_file_name: Pathless name for raw file.
    """

    if zipped:
        return '{0:s}{1:s}{2:s}'.format(
            time_conversion.unix_sec_to_string(
                unix_time_sec, TIME_FORMAT_SECONDS),
            UNZIPPED_FILE_EXTENSION, ZIPPED_FILE_EXTENSION)

    return '{0:s}{1:s}'.format(
        time_conversion.unix_sec_to_string(unix_time_sec, TIME_FORMAT_SECONDS),
        UNZIPPED_FILE_EXTENSION)


def _raw_file_name_to_time(raw_file_name):
    """Parses time from file name.

    :param raw_file_name: Path to raw file.
    :return: unix_time_sec: Valid time.
    """

    _, time_string = os.path.split(raw_file_name)
    time_string = time_string.replace(ZIPPED_FILE_EXTENSION, '').replace(
        UNZIPPED_FILE_EXTENSION, '')
    return time_conversion.string_to_unix_sec(time_string, TIME_FORMAT_SECONDS)


def _remove_sentinels_from_sparse_grid(
        sparse_grid_table, field_name, sentinel_values):
    """Removes sentinel values from sparse grid.

    :param sparse_grid_table: pandas DataFrame with columns produced by
        `read_data_from_sparse_grid_file`.
    :param field_name: Name of radar field in GewitterGefahr format.
    :param sentinel_values: 1-D numpy array of sentinel values.
    :return: sparse_grid_table: Same as input, except that rows with a sentinel
        value are removed.
    """

    num_rows = len(sparse_grid_table.index)
    sentinel_flags = numpy.full(num_rows, False, dtype=bool)

    for this_sentinel_value in sentinel_values:
        these_sentinel_flags = numpy.isclose(
            sparse_grid_table[field_name].values, this_sentinel_value,
            atol=SENTINEL_TOLERANCE)
        sentinel_flags = numpy.logical_or(sentinel_flags, these_sentinel_flags)

    sentinel_indices = numpy.where(sentinel_flags)[0]
    return sparse_grid_table.drop(
        sparse_grid_table.index[sentinel_indices], axis=0, inplace=False)


def _remove_sentinels_from_full_grid(field_matrix, sentinel_values):
    """Removes sentinel values from full grid.

    M = number of rows (unique grid-point latitudes)
    N = number of columns (unique grid-point longitudes)

    :param field_matrix: M-by-N numpy array with radar field.
    :param sentinel_values: 1-D numpy array of sentinel values.
    :return: field_matrix: Same as input, except that sentinel values are
        replaced with NaN.
    """

    num_grid_rows = field_matrix.shape[0]
    num_grid_columns = field_matrix.shape[1]
    num_grid_points = num_grid_rows * num_grid_columns

    field_matrix = numpy.reshape(field_matrix, num_grid_points)
    sentinel_flags = numpy.full(num_grid_points, False, dtype=bool)

    for this_sentinel_value in sentinel_values:
        these_sentinel_flags = numpy.isclose(
            field_matrix, this_sentinel_value, atol=SENTINEL_TOLERANCE)
        sentinel_flags = numpy.logical_or(sentinel_flags, these_sentinel_flags)

    sentinel_indices = numpy.where(sentinel_flags)[0]
    field_matrix[sentinel_indices] = numpy.nan
    return numpy.reshape(field_matrix, (num_grid_rows, num_grid_columns))


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


def field_name_new_to_orig(field_name, data_source):
    """Converts field name from new to original format.

    "Original format" = MYRORSS or MRMS
    "New format" = GewitterGefahr format, which is Pythonic and includes units
                   at the end

    :param field_name: Name of radar field in new format.
    :param data_source: Data source (string).
    :return: field_name_orig: Name of radar field in original (either MYRORSS or
        MRMS) format.
    """

    if data_source == MYRORSS_SOURCE_ID:
        all_orig_field_names = RADAR_FIELD_NAMES_MYRORSS
    else:
        all_orig_field_names = RADAR_FIELD_NAMES_MRMS

    found_flags = [s == field_name for s in RADAR_FIELD_NAMES]
    return all_orig_field_names[numpy.where(found_flags)[0][0]]


def get_valid_heights_for_field(field_name, data_source):
    """Returns valid heights for radar field.

    :param field_name: Name of radar field in GewitterGefahr format.
    :param data_source: Data source (string).
    :return: valid_heights_m_asl: 1-D numpy array of valid heights (integer
        metres above sea level).
    :raises: ValueError: if field_name = "storm_id".
    """

    check_field_name(field_name)
    _check_data_source(data_source)
    if field_name == STORM_ID_NAME:
        raise ValueError('Field name cannot be "{0:s}".'.format(STORM_ID_NAME))

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
            _check_reflectivity_heights(refl_heights_m_asl, data_source)
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
            _check_reflectivity_heights(refl_heights_m_asl, data_source)
            these_heights_m_asl = copy.deepcopy(refl_heights_m_asl)
        else:
            these_heights_m_asl = get_valid_heights_for_field(
                this_field_name, data_source=data_source)

        field_name_by_pair += [this_field_name] * len(these_heights_m_asl)
        height_by_pair_m_asl = numpy.concatenate((
            height_by_pair_m_asl, these_heights_m_asl))

    return field_name_by_pair, height_by_pair_m_asl


def get_relative_dir_for_raw_files(field_name, data_source, height_m_asl=None):
    """Generates relative path for raw files.

    :param field_name: Name of radar field in GewitterGefahr format.
    :param data_source: Data source (string).
    :param height_m_asl: Radar height (metres above sea level).
    :return: relative_directory_name: Relative path for raw files.
    """

    if field_name == REFL_NAME:
        _check_reflectivity_heights(numpy.array([height_m_asl]), data_source)
    else:
        height_m_asl = get_valid_heights_for_field(
            field_name, data_source=data_source)[0]

    return '{0:s}/{1:05.2f}'.format(
        field_name_new_to_orig(field_name, data_source=data_source),
        float(height_m_asl) * METRES_TO_KM)


def find_raw_azimuthal_shear_file(
        desired_time_unix_sec, spc_date_unix_sec, field_name, data_source,
        top_directory_name,
        max_time_offset_sec=DEFAULT_MAX_TIME_OFFSET_FOR_AZ_SHEAR_SEC,
        raise_error_if_missing=False):
    """Finds raw az-shear file.

    This file should contain one az-shear field (examples: low-level az-shear,
    mid-level az-shear) for one valid time.

    If you know the exact valid time, use `find_raw_file`.  However, az-shear is
    "special," and its valid times are usually offset from those of other radar
    fields.  This method accounts for said offset.

    :param desired_time_unix_sec: Desired valid time.
    :param spc_date_unix_sec: SPC date.
    :param field_name: Field name in GewitterGefahr format.
    :param data_source: Data source (string).
    :param top_directory_name: Name of top-level directory with raw files.
    :param max_time_offset_sec: Maximum offset between actual and desired valid
        time.  For example, if `desired_time_unix_sec` is 162933 UTC 5 Jan 2018
        and `max_time_offset_sec` = 60, this method will look for az-shear at
        valid times from 162833...163033 UTC 5 Jan 2018.
    :param raise_error_if_missing: Boolean flag.  If True and no az-shear file
        can be found, this method will raise an error.  If False and no az-shear
        file can be found, will return None.
    :return: raw_file_name: Path to raw az-shear file.
    :raises: ValueError: if raise_error_if_missing = True and file is missing.
    """

    error_checking.assert_is_integer(desired_time_unix_sec)
    error_checking.assert_is_integer(max_time_offset_sec)
    error_checking.assert_is_greater(max_time_offset_sec, 0)
    error_checking.assert_is_boolean(raise_error_if_missing)

    first_allowed_minute_unix_sec = numpy.round(int(rounder.floor_to_nearest(
        float(desired_time_unix_sec - max_time_offset_sec),
        MINUTES_TO_SECONDS)))
    last_allowed_minute_unix_sec = numpy.round(int(rounder.floor_to_nearest(
        float(desired_time_unix_sec + max_time_offset_sec),
        MINUTES_TO_SECONDS)))

    allowed_minutes_unix_sec = time_periods.range_and_interval_to_list(
        start_time_unix_sec=first_allowed_minute_unix_sec,
        end_time_unix_sec=last_allowed_minute_unix_sec,
        time_interval_sec=MINUTES_TO_SECONDS, include_endpoint=True).astype(int)

    spc_date_string = time_conversion.time_to_spc_date_string(spc_date_unix_sec)
    relative_directory_name = get_relative_dir_for_raw_files(
        field_name=field_name, data_source=data_source)

    raw_file_names = []
    for this_time_unix_sec in allowed_minutes_unix_sec:
        this_pathless_file_pattern = _get_pathless_raw_file_pattern(
            this_time_unix_sec)
        this_file_pattern = '{0:s}/{1:s}/{2:s}/{3:s}'.format(
            top_directory_name, spc_date_string, relative_directory_name,
            this_pathless_file_pattern)
        raw_file_names += glob.glob(this_file_pattern)

    file_times_unix_sec = []
    for this_raw_file_name in raw_file_names:
        file_times_unix_sec.append(_raw_file_name_to_time(this_raw_file_name))

    if len(file_times_unix_sec):
        file_times_unix_sec = numpy.array(file_times_unix_sec)
        time_differences_sec = numpy.absolute(
            file_times_unix_sec - desired_time_unix_sec)
        nearest_index = numpy.argmin(time_differences_sec)
        min_time_diff_sec = time_differences_sec[nearest_index]
    else:
        min_time_diff_sec = numpy.inf

    if min_time_diff_sec > max_time_offset_sec:
        if raise_error_if_missing:
            desired_time_string = time_conversion.unix_sec_to_string(
                desired_time_unix_sec, TIME_FORMAT_FOR_LOG_MESSAGES)
            log_string = ('Could not find "{0:s}" file within {1:d} seconds of '
                          '{2:s}').format(field_name, max_time_offset_sec,
                                          desired_time_string)
            raise ValueError(log_string)

        return None

    return raw_file_names[nearest_index]


def find_raw_file(
        unix_time_sec, spc_date_unix_sec, field_name, data_source,
        top_directory_name, height_m_asl=None, raise_error_if_missing=True):
    """Finds raw file.

    This file should contain one radar field at one height and valid time.

    :param unix_time_sec: Valid time.
    :param spc_date_unix_sec: SPC date.
    :param field_name: Name of radar field in GewitterGefahr format.
    :param data_source: Data source (string).
    :param top_directory_name: Name of top-level directory with raw files.
    :param height_m_asl: Radar height (metres above sea level).
    :param raise_error_if_missing: Boolean flag.  If True and file is missing,
        this method will raise an error.  If False and file is missing, will
        return *expected* path to raw file.
    :return: raw_file_name: Path to raw file.
    :raises: ValueError: if raise_error_if_missing = True and file is missing.
    """

    error_checking.assert_is_string(top_directory_name)
    error_checking.assert_is_boolean(raise_error_if_missing)

    relative_directory_name = get_relative_dir_for_raw_files(
        field_name=field_name, height_m_asl=height_m_asl,
        data_source=data_source)
    directory_name = '{0:s}/{1:s}/{2:s}'.format(
        top_directory_name,
        time_conversion.time_to_spc_date_string(spc_date_unix_sec),
        relative_directory_name)

    pathless_file_name = _get_pathless_raw_file_name(unix_time_sec, zipped=True)
    raw_file_name = '{0:s}/{1:s}'.format(directory_name, pathless_file_name)

    if raise_error_if_missing and not os.path.isfile(raw_file_name):
        pathless_file_name = _get_pathless_raw_file_name(
            unix_time_sec, zipped=False)
        raw_file_name = '{0:s}/{1:s}'.format(directory_name, pathless_file_name)

    if raise_error_if_missing and not os.path.isfile(raw_file_name):
        raise ValueError(
            'Cannot find raw file.  Expected at location: ' + raw_file_name)

    return raw_file_name


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

    # TODO(thunderhoser): move this method to radar_utils.py.

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


def read_metadata_from_raw_file(
        netcdf_file_name, data_source, raise_error_if_fails=True):
    """Reads metadata from raw (either MYRORSS or MRMS) file.

    This file should contain one radar field at one height and valid time.

    :param netcdf_file_name: Path to input file.
    :param data_source: Data source (string).
    :param raise_error_if_fails: Boolean flag.  If True and file cannot be read,
        this method will raise an error.  If False and file cannot be read, will
        return None.
    :return: metadata_dict: Dictionary with the following keys.
    metadata_dict['nw_grid_point_lat_deg']: Latitude (deg N) of northwesternmost
        grid point.
    metadata_dict['nw_grid_point_lng_deg']: Longitude (deg E) of
        northwesternmost grid point.
    metadata_dict['lat_spacing_deg']: Spacing (deg N) between meridionally
        adjacent grid points.
    metadata_dict['lng_spacing_deg']: Spacing (deg E) between zonally adjacent
        grid points.
    metadata_dict['num_lat_in_grid']: Number of rows (unique grid-point
        latitudes).
    metadata_dict['num_lng_in_grid']: Number of columns (unique grid-point
        longitudes).
    metadata_dict['height_m_asl']: Radar height (metres above ground level).
    metadata_dict['unix_time_sec']: Valid time.
    metadata_dict['field_name']: Name of radar field in GewitterGefahr format.
    metadata_dict['field_name_orig']: Name of radar field in original (either
        MYRORSS or MRMS) format.
    metadata_dict['sentinel_values']: 1-D numpy array of sentinel values.
    """

    error_checking.assert_file_exists(netcdf_file_name)
    netcdf_dataset = netcdf_io.open_netcdf(
        netcdf_file_name, raise_error_if_fails)
    if netcdf_dataset is None:
        return None

    field_name_orig = str(getattr(netcdf_dataset, FIELD_NAME_COLUMN_ORIG))

    metadata_dict = {
        NW_GRID_POINT_LAT_COLUMN:
            getattr(netcdf_dataset, NW_GRID_POINT_LAT_COLUMN_ORIG),
        NW_GRID_POINT_LNG_COLUMN: lng_conversion.convert_lng_positive_in_west(
            getattr(netcdf_dataset, NW_GRID_POINT_LNG_COLUMN_ORIG),
            allow_nan=False),
        LAT_SPACING_COLUMN: getattr(netcdf_dataset, LAT_SPACING_COLUMN_ORIG),
        LNG_SPACING_COLUMN: getattr(netcdf_dataset, LNG_SPACING_COLUMN_ORIG),
        NUM_LAT_COLUMN: netcdf_dataset.dimensions[NUM_LAT_COLUMN_ORIG].size + 1,
        NUM_LNG_COLUMN: netcdf_dataset.dimensions[NUM_LNG_COLUMN_ORIG].size + 1,
        HEIGHT_COLUMN: getattr(netcdf_dataset, HEIGHT_COLUMN_ORIG),
        UNIX_TIME_COLUMN: getattr(netcdf_dataset, UNIX_TIME_COLUMN_ORIG),
        FIELD_NAME_COLUMN_ORIG: field_name_orig,
        FIELD_NAME_COLUMN:
            _field_name_orig_to_new(field_name_orig, data_source)}

    metadata_dict[NW_GRID_POINT_LAT_COLUMN] = rounder.floor_to_nearest(
        metadata_dict[NW_GRID_POINT_LAT_COLUMN],
        metadata_dict[LAT_SPACING_COLUMN])
    metadata_dict[NW_GRID_POINT_LNG_COLUMN] = rounder.ceiling_to_nearest(
        metadata_dict[NW_GRID_POINT_LNG_COLUMN],
        metadata_dict[LNG_SPACING_COLUMN])

    sentinel_values = []
    for this_column in SENTINEL_VALUE_COLUMNS_ORIG:
        sentinel_values.append(getattr(netcdf_dataset, this_column))

    metadata_dict.update({SENTINEL_VALUE_COLUMN: numpy.array(sentinel_values)})
    netcdf_dataset.close()
    return metadata_dict


def read_data_from_sparse_grid_file(
        netcdf_file_name, field_name_orig, data_source, sentinel_values,
        raise_error_if_fails=True):
    """Reads sparse radar grid from raw (either MYRORSS or MRMS) file.

    This file should contain one radar field at one height and valid time.

    :param netcdf_file_name: Path to input file.
    :param field_name_orig: Name of radar field in original (either MYRORSS or
        MRMS) format.
    :param data_source: Data source (string).
    :param sentinel_values: 1-D numpy array of sentinel values.
    :param raise_error_if_fails: Boolean flag.  If True and file cannot be read,
        this method will raise an error.  If False and file cannot be read, will
        return None.
    :return: sparse_grid_table: pandas DataFrame with the following columns.
        Each row corresponds to one grid point.
    sparse_grid_table.grid_row: Row index.
    sparse_grid_table.grid_column: Column index.
    sparse_grid_table.<field_name>: Radar measurement (column name is produced
        by _field_name_orig_to_new).
    sparse_grid_table.num_grid_cells: Number of consecutive grid points with the
        same radar measurement.  Counting is row-major (to the right along the
        row, then down to the next column if necessary).
    """

    error_checking.assert_file_exists(netcdf_file_name)
    error_checking.assert_is_numpy_array_without_nan(sentinel_values)
    error_checking.assert_is_numpy_array(sentinel_values, num_dimensions=1)

    netcdf_dataset = netcdf_io.open_netcdf(
        netcdf_file_name, raise_error_if_fails)
    if netcdf_dataset is None:
        return None

    field_name = _field_name_orig_to_new(field_name_orig, data_source)
    num_values = len(netcdf_dataset.variables[GRID_ROW_COLUMN_ORIG])

    if num_values == 0:
        sparse_grid_dict = {
            GRID_ROW_COLUMN: numpy.array([], dtype=int),
            GRID_COLUMN_COLUMN: numpy.array([], dtype=int),
            NUM_GRID_CELL_COLUMN: numpy.array([], dtype=int),
            field_name: numpy.array([])}
    else:
        sparse_grid_dict = {
            GRID_ROW_COLUMN: netcdf_dataset.variables[GRID_ROW_COLUMN_ORIG][:],
            GRID_COLUMN_COLUMN:
                netcdf_dataset.variables[GRID_COLUMN_COLUMN_ORIG][:],
            NUM_GRID_CELL_COLUMN:
                netcdf_dataset.variables[NUM_GRID_CELL_COLUMN_ORIG][:],
            field_name: netcdf_dataset.variables[field_name_orig][:]}

    netcdf_dataset.close()
    sparse_grid_table = pandas.DataFrame.from_dict(sparse_grid_dict)
    return _remove_sentinels_from_sparse_grid(
        sparse_grid_table, field_name=field_name,
        sentinel_values=sentinel_values)


def read_data_from_full_grid_file(
        netcdf_file_name, metadata_dict, raise_error_if_fails=True):
    """Reads full radar grid from raw (either MYRORSS or MRMS) file.

    This file should contain one radar field at one height and valid time.

    :param netcdf_file_name: Path to input file.
    :param metadata_dict: Dictionary created by `read_metadata_from_raw_file`.
    :param raise_error_if_fails: Boolean flag.  If True and file cannot be read,
        this method will raise an error.  If False and file cannot be read, will
        return None for all output vars.
    :return: field_matrix: M-by-N numpy array with radar field.  Latitude
        increases while moving up each column, and longitude increases while
        moving right along each row.
    :return: grid_point_latitudes_deg: length-M numpy array of grid-point
        latitudes (deg N).  This array is monotonically decreasing.
    :return: grid_point_longitudes_deg: length-N numpy array of grid-point
        longitudes (deg E).  This array is monotonically increasing.
    """

    error_checking.assert_file_exists(netcdf_file_name)
    netcdf_dataset = netcdf_io.open_netcdf(
        netcdf_file_name, raise_error_if_fails)
    if netcdf_dataset is None:
        return None, None, None

    field_matrix = netcdf_dataset.variables[
        metadata_dict[FIELD_NAME_COLUMN_ORIG]]
    netcdf_dataset.close()

    min_latitude_deg = metadata_dict[NW_GRID_POINT_LAT_COLUMN] - (
        metadata_dict[LAT_SPACING_COLUMN] * (metadata_dict[NUM_LAT_COLUMN] - 1))
    grid_point_latitudes_deg, grid_point_longitudes_deg = (
        grids.get_latlng_grid_points(
            min_latitude_deg=min_latitude_deg,
            min_longitude_deg=metadata_dict[NW_GRID_POINT_LNG_COLUMN],
            lat_spacing_deg=metadata_dict[LAT_SPACING_COLUMN],
            lng_spacing_deg=metadata_dict[LNG_SPACING_COLUMN],
            num_rows=metadata_dict[NUM_LAT_COLUMN],
            num_columns=metadata_dict[NUM_LNG_COLUMN]))

    field_matrix = _remove_sentinels_from_full_grid(
        field_matrix, metadata_dict[SENTINEL_VALUE_COLUMN])
    return (numpy.flipud(field_matrix), grid_point_latitudes_deg[::-1],
            grid_point_longitudes_deg)
