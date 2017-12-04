"""IO methods for radar data from MYRORSS or MRMS

--- DEFINITIONS ---

MYRORSS = Multi-year Reanalysis of Remotely Sensed Storms

MRMS = Multi-radar Multi-sensor
"""

import copy
import os
import numpy
import pandas
from gewittergefahr.gg_io import netcdf_io
from gewittergefahr.gg_utils import number_rounding as rounder
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import longitude_conversion as lng_conversion
from gewittergefahr.gg_utils import grids
from gewittergefahr.gg_utils import error_checking

# TODO(thunderhoser): need to improve terminology.  In some cases I use "field"
# to mean one radar variable at any height, and in some cases I use it to mean
# one variable at one height.  I should probably use it consistently for the
# latter.

# TODO(thunderhoser): Radar heights are in metres above sea level (m ASL), not
# m AGL.  I don't know why I thought the latter.  Need to change terminology to
# reflect this.

NW_GRID_POINT_LAT_COLUMN = 'nw_grid_point_lat_deg'
NW_GRID_POINT_LNG_COLUMN = 'nw_grid_point_lng_deg'
LAT_SPACING_COLUMN = 'lat_spacing_deg'
LNG_SPACING_COLUMN = 'lng_spacing_deg'
NUM_LAT_COLUMN = 'num_lat_in_grid'
NUM_LNG_COLUMN = 'num_lng_in_grid'
HEIGHT_COLUMN = 'height_m_agl'
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

ECHO_TOP_NAMES = [ECHO_TOP_18DBZ_NAME, ECHO_TOP_50DBZ_NAME]
SHEAR_NAMES = [LOW_LEVEL_SHEAR_NAME, MID_LEVEL_SHEAR_NAME]
REFLECTIVITY_NAMES = [
    REFL_NAME, REFL_COLUMN_MAX_NAME, REFL_0CELSIUS_NAME, REFL_M10CELSIUS_NAME,
    REFL_M20CELSIUS_NAME, REFL_LOWEST_ALTITUDE_NAME]

ECHO_TOP_18DBZ_NAME_ORIG = 'EchoTop_18'
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
    ECHO_TOP_18DBZ_NAME, ECHO_TOP_50DBZ_NAME, LOW_LEVEL_SHEAR_NAME,
    MID_LEVEL_SHEAR_NAME, REFL_NAME, REFL_COLUMN_MAX_NAME, MESH_NAME,
    REFL_0CELSIUS_NAME, REFL_M10CELSIUS_NAME, REFL_M20CELSIUS_NAME,
    REFL_LOWEST_ALTITUDE_NAME, SHI_NAME, VIL_NAME, STORM_ID_NAME]

RADAR_FIELD_NAMES_MYRORSS = [
    ECHO_TOP_18DBZ_NAME_ORIG, ECHO_TOP_50DBZ_NAME_ORIG,
    LOW_LEVEL_SHEAR_NAME_MYRORSS, MID_LEVEL_SHEAR_NAME_MYRORSS, REFL_NAME_ORIG,
    REFL_COLUMN_MAX_NAME_ORIG, MESH_NAME_ORIG, REFL_0CELSIUS_NAME_ORIG,
    REFL_M10CELSIUS_NAME_ORIG, REFL_M20CELSIUS_NAME_ORIG,
    REFL_LOWEST_ALTITUDE_NAME_ORIG, SHI_NAME_ORIG, VIL_NAME_ORIG,
    STORM_ID_NAME_ORIG]

RADAR_FIELD_NAMES_MRMS = [
    ECHO_TOP_18DBZ_NAME_ORIG, ECHO_TOP_50DBZ_NAME_ORIG,
    LOW_LEVEL_SHEAR_NAME_MRMS, MID_LEVEL_SHEAR_NAME_MRMS, REFL_NAME_ORIG,
    REFL_COLUMN_MAX_NAME_ORIG, MESH_NAME_ORIG, REFL_0CELSIUS_NAME_ORIG,
    REFL_M10CELSIUS_NAME_ORIG, REFL_M20CELSIUS_NAME_ORIG,
    REFL_LOWEST_ALTITUDE_NAME_ORIG, SHI_NAME_ORIG, VIL_NAME_ORIG,
    STORM_ID_NAME_ORIG]

HEIGHT_ARRAY_COLUMN = 'heights_m_agl'
SENTINEL_TOLERANCE = 10.

TIME_FORMAT_SECONDS = '%Y%m%d-%H%M%S'
DAYS_TO_SECONDS = 86400
METRES_TO_KM = 1e-3

SHEAR_HEIGHT_M_AGL = 0
DEFAULT_HEIGHT_MYRORSS_M_AGL = 250
DEFAULT_HEIGHT_MRMS_M_AGL = 500

ZIPPED_FILE_EXTENSION = '.gz'
UNZIPPED_FILE_EXTENSION = '.netcdf'


def _check_data_source(data_source):
    """Ensures that data source is either "myrorss" or "mrms".

    :param data_source: Data source (string).
    :raises: ValueError: data source is neither "myrorss" nor "mrms".
    """

    error_checking.assert_is_string(data_source)
    if data_source not in DATA_SOURCE_IDS:
        error_string = (
            '\n\n' + str(DATA_SOURCE_IDS) +
            '\n\nValid data sources (listed above) do not include "' +
            data_source + '".')
        raise ValueError(error_string)


def _check_field_name_orig(field_name_orig, data_source=None):
    """Ensures that name of radar field is recognized.

    :param field_name_orig: Name of radar field in original (either MYRORSS or
        MRMS) format.
    :param data_source: Data source (either "myrorss" or "mrms").
    :raises: ValueError: if name of radar field is not recognized.
    """

    if data_source == MYRORSS_SOURCE_ID:
        valid_field_names = copy.deepcopy(RADAR_FIELD_NAMES_MYRORSS)
    else:
        valid_field_names = copy.deepcopy(RADAR_FIELD_NAMES_MRMS)

    if field_name_orig not in valid_field_names:
        error_string = (
            '\n\n' + str(valid_field_names) +
            '\n\nValid field names (listed above) do not include "' +
            field_name_orig + '".')
        raise ValueError(error_string)


def _field_name_orig_to_new(field_name_orig, data_source=None):
    """Converts field name from original (either MYRORSS or MRMS) to new format.

    :param field_name_orig: Name of radar field in original (either MYRORSS or
        MRMS) format.
    :param data_source: Data source (either "myrorss" or "mrms").
    :return: field_name: Field name in new format.
    """

    if data_source == MYRORSS_SOURCE_ID:
        all_orig_field_names = copy.deepcopy(RADAR_FIELD_NAMES_MYRORSS)
    else:
        all_orig_field_names = copy.deepcopy(RADAR_FIELD_NAMES_MRMS)

    found_flags = [s == field_name_orig for s in all_orig_field_names]
    return RADAR_FIELD_NAMES[numpy.where(found_flags)[0][0]]


def _get_valid_heights_for_field(field_name, data_source=None):
    """Finds valid heights for radar field.

    :param field_name: Name of radar field in new format (as opposed to MYRORSS
        or MRMS format).
    :param data_source: Data source (either "myrorss" or "mrms").
    :return: valid_heights_m_agl: 1-D numpy array of valid heights (integer
        metres above ground level).
    :raises: ValueError: if field_name = "storm_id".
    """

    if field_name == STORM_ID_NAME:
        raise ValueError('Field may be any radar field other than "' +
                         STORM_ID_NAME + '".')

    if data_source == MYRORSS_SOURCE_ID:
        default_height_m_agl = copy.deepcopy(DEFAULT_HEIGHT_MYRORSS_M_AGL)
    else:
        default_height_m_agl = copy.deepcopy(DEFAULT_HEIGHT_MRMS_M_AGL)

    if field_name == ECHO_TOP_18DBZ_NAME:
        return numpy.array([default_height_m_agl])
    elif field_name == ECHO_TOP_50DBZ_NAME:
        return numpy.array([default_height_m_agl])
    elif field_name == LOW_LEVEL_SHEAR_NAME:
        return numpy.array([SHEAR_HEIGHT_M_AGL])
    elif field_name == MID_LEVEL_SHEAR_NAME:
        return numpy.array([SHEAR_HEIGHT_M_AGL])
    elif field_name == REFL_NAME:
        return numpy.array(
            [250, 500, 750, 1000, 1250, 1500, 1750, 2000, 2250, 2500, 2750,
             3000, 3500, 4000, 4500, 5000, 5500, 6000, 6500, 7000, 7500, 8000,
             8500, 9000, 10000, 11000, 12000, 13000, 14000, 15000, 16000, 17000,
             18000, 19000, 20000])
    elif field_name == REFL_COLUMN_MAX_NAME:
        return numpy.array([default_height_m_agl])
    elif field_name == MESH_NAME:
        return numpy.array([default_height_m_agl])
    elif field_name == REFL_0CELSIUS_NAME:
        return numpy.array([default_height_m_agl])
    elif field_name == REFL_M10CELSIUS_NAME:
        return numpy.array([default_height_m_agl])
    elif field_name == REFL_M20CELSIUS_NAME:
        return numpy.array([default_height_m_agl])
    elif field_name == REFL_LOWEST_ALTITUDE_NAME:
        return numpy.array([default_height_m_agl])
    elif field_name == SHI_NAME:
        return numpy.array([default_height_m_agl])
    elif field_name == VIL_NAME:
        return numpy.array([default_height_m_agl])


def _check_reflectivity_heights(heights_m_agl):
    """Ensures that all reflectivity heights are valid.

    :param heights_m_agl: 1-D numpy array of heights (integer metres above
        ground level).
    :raises: ValueError: if any element of heights_m_agl is invalid.
    """

    error_checking.assert_is_integer_numpy_array(heights_m_agl)
    error_checking.assert_is_numpy_array(heights_m_agl, num_dimensions=1)

    # Data source doesn't matter in this method call (i.e., replacing
    # MYRORSS_SOURCE_ID with MRMS_SOURCE_ID would have no effect).
    valid_heights_m_agl = _get_valid_heights_for_field(REFL_NAME,
                                                       MYRORSS_SOURCE_ID)

    for this_height_m_agl in heights_m_agl:
        if this_height_m_agl in valid_heights_m_agl:
            continue

        error_string = (
            '\n\n' + str(valid_heights_m_agl) +
            '\n\nValid reflectivity heights (metres AGL, listed above) do not '
            'include ' + str(this_height_m_agl) + ' m AGL.')
        raise ValueError(error_string)


def _get_pathless_raw_file_name(unix_time_sec, zipped=True):
    """Generates pathless name for raw file.

    This file should contain one radar field at one height and one time step.

    :param unix_time_sec: Time in Unix format.
    :param zipped: Boolean flag.  If zipped = True, will look for zipped file.
        If zipped = False, will look for unzipped file.
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


def _remove_sentinels_from_sparse_grid(sparse_grid_table, field_name=None,
                                       sentinel_values=None):
    """Removes sentinel values from sparse radar grid.

    :param sparse_grid_table: pandas DataFrame with columns documented in
        read_data_from_sparse_grid_file.
    :param field_name: Name of radar field in new format (as opposed to
        MYRORSS/MRMS format).
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
    sparse_grid_table.drop(sparse_grid_table.index[sentinel_indices], axis=0,
                           inplace=True)
    return sparse_grid_table


def _remove_sentinels_from_full_grid(field_matrix, sentinel_values):
    """Removes sentinel values from full radar grid.

    M = number of rows (unique grid-point latitudes)
    N = number of columns (unique grid-point longitudes)

    :param field_matrix: M-by-N numpy array with values of radar field.
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

    :param field_name: Name of radar field in new format (as opposed to MYRORSS
        or MRMS format).
    :raises: ValueError: if name of radar field is not recognized.
    """

    if field_name not in RADAR_FIELD_NAMES:
        error_string = (
            '\n\n' + str(RADAR_FIELD_NAMES) +
            '\n\nValid field names (listed above) do not include "' +
            field_name + '".')
        raise ValueError(error_string)


def field_name_new_to_orig(field_name, data_source=None):
    """Converts field name from new to original format.

    New format = GewitterGefahr
    Original format = either MYRORSS or MRMS

    :param field_name: Name of radar field in new format.
    :param data_source: Data source (either "myrorss" or "mrms").
    :return: field_name_orig: Field name in original format.
    """

    _check_data_source(data_source)
    if data_source == MYRORSS_SOURCE_ID:
        all_orig_field_names = copy.deepcopy(RADAR_FIELD_NAMES_MYRORSS)
    else:
        all_orig_field_names = copy.deepcopy(RADAR_FIELD_NAMES_MRMS)

    found_flags = [s == field_name for s in RADAR_FIELD_NAMES]
    return all_orig_field_names[numpy.where(found_flags)[0][0]]


def field_and_height_arrays_to_dict(field_names, refl_heights_m_agl=None,
                                    data_source=None):
    """Converts two arrays (radar-field names and reflectivity heights) to dict.

    :param field_names: 1-D list with names of radar fields in new format (as
        opposed to MYRORSS or MRMS format).
    :param refl_heights_m_agl: 1-D numpy array of reflectivity heights (metres
        above ground level).
    :param data_source: Data source (either "myrorss" or "mrms").
    :return: field_to_heights_dict_m_agl: Dictionary, where each key is the name
        of a radar field and each value is 1-D numpy array of heights (metres
        above ground level).
    """

    field_to_heights_dict_m_agl = {}

    for this_field_name in field_names:
        if this_field_name == REFL_NAME:
            _check_reflectivity_heights(refl_heights_m_agl)
            field_to_heights_dict_m_agl.update(
                {this_field_name: refl_heights_m_agl})
        else:
            field_to_heights_dict_m_agl.update({
                this_field_name: _get_valid_heights_for_field(
                    this_field_name, data_source=data_source)})

    return field_to_heights_dict_m_agl


def unique_fields_and_heights_to_pairs(field_names, refl_heights_m_agl=None,
                                       data_source=None):
    """Converts unique arrays to non-unique arrays.

    F = number of unique field names
    N = number of field-height pairs

    :param field_names: length-F list with names of radar fields in new format
        (rather than MYRORSS or MRMS format).
    :param refl_heights_m_agl: 1-D numpy array of reflectivity heights (metres
        above ground level).
    :param data_source: Data source (either "myrorss" or "mrms").
    :return: field_name_by_pair: length-N list of field names.
    :return: height_by_pair_m_agl: length-N numpy array of radar heights (metres
        above ground level).
    """

    field_name_by_pair = []
    height_by_pair_m_agl = numpy.array([])

    for this_field_name in field_names:
        if this_field_name == REFL_NAME:
            _check_reflectivity_heights(refl_heights_m_agl)
            these_heights_m_agl = copy.deepcopy(refl_heights_m_agl)
        else:
            these_heights_m_agl = _get_valid_heights_for_field(
                this_field_name, data_source=data_source)

        field_name_by_pair += [this_field_name] * len(these_heights_m_agl)
        height_by_pair_m_agl = numpy.concatenate((
            height_by_pair_m_agl, these_heights_m_agl))

    return field_name_by_pair, height_by_pair_m_agl


def get_relative_dir_for_raw_files(field_name=None, height_m_agl=None,
                                   data_source=None):
    """Generates relative path for raw files.

    :param field_name: Name of radar field in new format (as opposed to MYRORSS
        or MRMS format).
    :param height_m_agl: Height (metres above ground level).
    :param data_source: Data source (either "myrorss" or "mrms").
    :return: relative_raw_directory_name: Relative path for raw files.
    """

    if field_name == REFL_NAME:
        _check_reflectivity_heights(numpy.array([height_m_agl]))
    else:
        height_m_agl = _get_valid_heights_for_field(
            field_name, data_source=data_source)[0]

    return '{0:s}/{1:05.2f}'.format(
        field_name_new_to_orig(field_name, data_source=data_source),
        float(height_m_agl) * METRES_TO_KM)


def find_raw_file(unix_time_sec=None, spc_date_unix_sec=None, field_name=None,
                  height_m_agl=None, data_source=None, top_directory_name=None,
                  raise_error_if_missing=True):
    """Finds raw file on local machine.

    This file should contain one radar field at one height and one time step.

    :param unix_time_sec: Time in Unix format.
    :param spc_date_unix_sec: SPC date in Unix format.
    :param field_name: Name of radar field in new format (as opposed to MYRORSS
        or MRMS format).
    :param height_m_agl: Height (metres above ground level).
    :param data_source: Data source (either "myrorss" or "mrms").
    :param top_directory_name: Top-level directory for raw files.
    :param raise_error_if_missing: Boolean flag.  If raise_error_if_missing =
        True and file is missing, will raise error.
    :return: raw_file_name: Path to raw file.  If raise_error_if_missing = False
        and file is missing, this will be the *expected* path.
    :raises: ValueError: if raise_error_if_missing = True and file is missing.
    """

    error_checking.assert_is_string(top_directory_name)
    error_checking.assert_is_boolean(raise_error_if_missing)

    relative_directory_name = get_relative_dir_for_raw_files(
        field_name=field_name, height_m_agl=height_m_agl,
        data_source=data_source)

    pathless_file_name = _get_pathless_raw_file_name(unix_time_sec, zipped=True)
    raw_file_name = '{0:s}/{1:s}/{2:s}/{3:s}'.format(
        top_directory_name,
        time_conversion.time_to_spc_date_string(spc_date_unix_sec),
        relative_directory_name, pathless_file_name)

    if raise_error_if_missing and not os.path.isfile(raw_file_name):
        pathless_file_name = _get_pathless_raw_file_name(
            unix_time_sec, zipped=False)
        raw_file_name = '{0:s}/{1:s}/{2:s}/{3:s}'.format(
            top_directory_name,
            time_conversion.time_to_spc_date_string(spc_date_unix_sec),
            relative_directory_name, pathless_file_name)

    if raise_error_if_missing and not os.path.isfile(raw_file_name):
        raise ValueError(
            'Cannot find raw file.  Expected at location: ' + raw_file_name)

    return raw_file_name


def rowcol_to_latlng(grid_rows, grid_columns, nw_grid_point_lat_deg=None,
                     nw_grid_point_lng_deg=None, lat_spacing_deg=None,
                     lng_spacing_deg=None):
    """Converts radar coordinates from row-column to lat-long.

    P = number of points

    :param grid_rows: length-P numpy array of row indices (increasing from north
        to south).
    :param grid_columns: length-P numpy array of column indices (increasing from
        west to east).
    :param nw_grid_point_lat_deg: Latitude (deg N) of northwesternmost grid
        point.
    :param nw_grid_point_lng_deg: Longitude (deg E) of northwesternmost grid
        point.
    :param lat_spacing_deg: Spacing (deg N) between adjacent rows.
    :param lng_spacing_deg: Spacing (deg E) between adjacent columns.
    :return: latitudes_deg: length-P numpy array of latitudes (deg N).
    :return: longitudes_deg: length-P numpy array of longitudes (deg E).
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

    error_checking.assert_is_greater(lat_spacing_deg, 0)
    error_checking.assert_is_greater(lng_spacing_deg, 0)

    latitudes_deg = rounder.round_to_nearest(
        nw_grid_point_lat_deg - lat_spacing_deg * grid_rows,
        lat_spacing_deg / 2)
    longitudes_deg = rounder.round_to_nearest(
        nw_grid_point_lng_deg + lng_spacing_deg * grid_columns,
        lng_spacing_deg / 2)
    return latitudes_deg, lng_conversion.convert_lng_positive_in_west(
        longitudes_deg, allow_nan=False)


def latlng_to_rowcol(latitudes_deg, longitudes_deg, nw_grid_point_lat_deg=None,
                     nw_grid_point_lng_deg=None, lat_spacing_deg=None,
                     lng_spacing_deg=None):
    """Converts radar coordinates from lat-long to row-column.

    P = number of points

    :param latitudes_deg: length-P numpy array of latitudes (deg N).
    :param longitudes_deg: length-P numpy array of longitudes (deg E).
    :param nw_grid_point_lat_deg: Latitude (deg N) of northwesternmost grid
        point.
    :param nw_grid_point_lng_deg: Longitude (deg E) of northwesternmost grid
        point.
    :param lat_spacing_deg: Spacing (deg N) between adjacent rows.
    :param lng_spacing_deg: Spacing (deg E) between adjacent columns.
    :return: grid_rows: length-P numpy array of row indices (increasing from
        north to south).
    :return: grid_columns: length-P numpy array of column indices (increasing
        from west to east).
    """

    error_checking.assert_is_valid_lat_numpy_array(latitudes_deg,
                                                   allow_nan=True)
    error_checking.assert_is_numpy_array(latitudes_deg, num_dimensions=1)
    num_points = len(latitudes_deg)

    longitudes_deg = lng_conversion.convert_lng_positive_in_west(longitudes_deg,
                                                                 allow_nan=True)
    error_checking.assert_is_numpy_array(
        longitudes_deg, exact_dimensions=numpy.array([num_points]))

    error_checking.assert_is_valid_latitude(nw_grid_point_lat_deg)
    nw_grid_point_lng_deg = lng_conversion.convert_lng_positive_in_west(
        nw_grid_point_lng_deg, allow_nan=False)

    error_checking.assert_is_greater(lat_spacing_deg, 0)
    error_checking.assert_is_greater(lng_spacing_deg, 0)

    grid_columns = rounder.round_to_nearest(
        (longitudes_deg - nw_grid_point_lng_deg) / lng_spacing_deg, 0.5)
    grid_rows = rounder.round_to_nearest(
        (nw_grid_point_lat_deg - latitudes_deg) / lat_spacing_deg, 0.5)
    return grid_rows, grid_columns


def get_center_of_grid(nw_grid_point_lat_deg=None, nw_grid_point_lng_deg=None,
                       lat_spacing_deg=None, lng_spacing_deg=None,
                       num_grid_rows=None, num_grid_columns=None):
    """Finds center of grid.

    :param nw_grid_point_lat_deg: Latitude (deg N) of northwesternmost grid
        point.
    :param nw_grid_point_lng_deg: Longitude (deg E) of northwesternmost grid
        point.
    :param lat_spacing_deg: Spacing (deg N) between adjacent rows.
    :param lng_spacing_deg: Spacing (deg E) between adjacent columns.
    :param num_grid_rows: Number of rows (unique grid-point latitudes).
    :param num_grid_columns: Number of columns (unique grid-point longitudes).
    :return: center_latitude_deg: Latitude (deg N) at center of grid.
    :return: center_longitude_deg: Longitude (deg E) at center of grid.
    """

    error_checking.assert_is_valid_latitude(nw_grid_point_lat_deg)
    nw_grid_point_lng_deg = lng_conversion.convert_lng_positive_in_west(
        nw_grid_point_lng_deg, allow_nan=False)

    error_checking.assert_is_greater(lat_spacing_deg, 0)
    error_checking.assert_is_greater(lng_spacing_deg, 0)
    error_checking.assert_is_integer(num_grid_rows)
    error_checking.assert_is_greater(num_grid_rows, 0)
    error_checking.assert_is_integer(num_grid_columns)
    error_checking.assert_is_greater(num_grid_columns, 0)

    min_latitude_deg = nw_grid_point_lat_deg - (
        (num_grid_rows - 1) * lat_spacing_deg)

    max_longitude_deg = nw_grid_point_lng_deg + (
        (num_grid_columns - 1) * lng_spacing_deg)

    return (numpy.mean(numpy.array([min_latitude_deg, nw_grid_point_lat_deg])),
            numpy.mean(numpy.array([nw_grid_point_lng_deg, max_longitude_deg])))


def read_metadata_from_raw_file(netcdf_file_name, data_source=None,
                                raise_error_if_fails=True):
    """Reads metadata raw (either MYRORSS or MRMS) file..

    This file should contain one radar field at one height and one time step.

    :param netcdf_file_name: Path to input file.
    :param data_source: Data source (either "myrorss" or "mrms").
    :param raise_error_if_fails: Boolean flag.  If True and file cannot be
        opened, this method will raise an error.  If False and file cannot be
        opened, this method will return None.
    :return: metadata_dict: Dictionary with the following keys.
    metadata_dict['nw_grid_point_lat_deg']: Latitude (deg N) of northwesternmost
        grid point.
    metadata_dict['nw_grid_point_lng_deg']: Longitude (deg E) of
        northwesternmost grid point.
    metadata_dict['lat_spacing_deg']: Spacing (deg N) between adjacent rows.
    metadata_dict['lng_spacing_deg']: Spacing (deg E) between adjacent columns.
    metadata_dict['num_lat_in_grid']: Number of rows (unique grid-point
        latitudes).
    metadata_dict['num_lng_in_grid']: Number of columns (unique grid-point
        longitudes).
    metadata_dict['height_m_agl']: Height (metres above ground level).
    metadata_dict['unix_time_sec']: Time in Unix format.
    metadata_dict['field_name']: Name of radar field in new format.
    metadata_dict['field_name_orig']: Name of radar field in original (MYRORSS
        or MRMS) format.
    metadata_dict['sentinel_values']: 1-D numpy array of sentinel values.
    """

    error_checking.assert_file_exists(netcdf_file_name)
    netcdf_dataset = netcdf_io.open_netcdf(netcdf_file_name,
                                           raise_error_if_fails)
    if netcdf_dataset is None:
        return None

    field_name_orig = str(getattr(netcdf_dataset, FIELD_NAME_COLUMN_ORIG))

    metadata_dict = {
        NW_GRID_POINT_LAT_COLUMN: getattr(netcdf_dataset,
                                          NW_GRID_POINT_LAT_COLUMN_ORIG),
        NW_GRID_POINT_LNG_COLUMN: lng_conversion.convert_lng_positive_in_west(
            getattr(netcdf_dataset, NW_GRID_POINT_LNG_COLUMN_ORIG),
            allow_nan=False),
        LAT_SPACING_COLUMN: getattr(netcdf_dataset, LAT_SPACING_COLUMN_ORIG),
        LNG_SPACING_COLUMN: getattr(netcdf_dataset, LNG_SPACING_COLUMN_ORIG),
        NUM_LAT_COLUMN: netcdf_dataset.dimensions[NUM_LAT_COLUMN_ORIG].size,
        NUM_LNG_COLUMN: netcdf_dataset.dimensions[NUM_LNG_COLUMN_ORIG].size,
        HEIGHT_COLUMN: getattr(netcdf_dataset, HEIGHT_COLUMN_ORIG),
        UNIX_TIME_COLUMN: getattr(netcdf_dataset, UNIX_TIME_COLUMN_ORIG),
        FIELD_NAME_COLUMN_ORIG: field_name_orig,
        FIELD_NAME_COLUMN: _field_name_orig_to_new(field_name_orig,
                                                   data_source=data_source)}

    metadata_dict[NW_GRID_POINT_LAT_COLUMN] = rounder.floor_to_nearest(
        metadata_dict[NW_GRID_POINT_LAT_COLUMN],
        metadata_dict[LAT_SPACING_COLUMN])
    metadata_dict[NW_GRID_POINT_LNG_COLUMN] = rounder.ceiling_to_nearest(
        metadata_dict[NW_GRID_POINT_LNG_COLUMN],
        metadata_dict[LNG_SPACING_COLUMN])

    metadata_dict[NUM_LAT_COLUMN] = int(rounder.round_to_nearest(
        metadata_dict[NUM_LAT_COLUMN], 100)) + 1
    metadata_dict[NUM_LNG_COLUMN] = int(rounder.round_to_nearest(
        metadata_dict[NUM_LNG_COLUMN], 100)) + 1

    sentinel_values = numpy.full(len(SENTINEL_VALUE_COLUMNS_ORIG), numpy.nan)
    for i in range(len(SENTINEL_VALUE_COLUMNS_ORIG)):
        sentinel_values[i] = getattr(netcdf_dataset,
                                     SENTINEL_VALUE_COLUMNS_ORIG[i])

    metadata_dict.update({SENTINEL_VALUE_COLUMN: sentinel_values})
    netcdf_dataset.close()
    return metadata_dict


def read_data_from_sparse_grid_file(netcdf_file_name, field_name_orig=None,
                                    data_source=None, sentinel_values=None,
                                    raise_error_if_fails=True):
    """Reads sparse radar grid from raw (either MYRORSS or MRMS) file.

    This file should contain one radar field at one height and one time step.

    :param netcdf_file_name: Path to input file.
    :param field_name_orig: Name of radar field in original (either MYRORSS or
        MRMS) format.
    :param data_source: Data source (either "myrorss" or "mrms").
    :param sentinel_values: 1-D numpy array of sentinel values.
    :param raise_error_if_fails: Boolean flag.  If True and file cannot be
        opened, this method will raise an error.  If False and file cannot be
        opened, this method will return None.
    :return: sparse_grid_table: pandas DataFrame with the following columns.
        Each row corresponds to one grid cell.
    sparse_grid_table.grid_row: Row index.
    sparse_grid_table.grid_column: Column index.
    sparse_grid_table.<field_name>: Radar measurement (field_name is determined
        by the method `field_name_orig_to_new`).
    sparse_grid_table.num_grid_cells: Number of consecutive grid cells --
        starting at the current one and counting along rows first, columns
        second -- with the same radar measurement.
    """

    error_checking.assert_file_exists(netcdf_file_name)
    error_checking.assert_is_numpy_array_without_nan(sentinel_values)
    error_checking.assert_is_numpy_array(sentinel_values, num_dimensions=1)

    netcdf_dataset = netcdf_io.open_netcdf(netcdf_file_name,
                                           raise_error_if_fails)
    if netcdf_dataset is None:
        return None

    field_name = _field_name_orig_to_new(field_name_orig,
                                         data_source=data_source)
    num_values = len(netcdf_dataset.variables[GRID_ROW_COLUMN_ORIG])

    if num_values == 0:
        sparse_grid_dict = {
            GRID_ROW_COLUMN: numpy.array([], dtype=int),
            GRID_COLUMN_COLUMN: numpy.array([], dtype=int),
            NUM_GRID_CELL_COLUMN: numpy.array([], dtype=int),
            field_name: numpy.array([], dtype=int)}
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
        sparse_grid_table, field_name, sentinel_values)


def read_data_from_full_grid_file(netcdf_file_name, metadata_dict,
                                  raise_error_if_fails=True):
    """Reads full radar grid from raw (either MYRORSS or MRMS) file.

    This file should contain one radar field at one height and one time step.

    M = number of rows (unique grid-point latitudes)
    N = number of columns (unique grid-point longitudes)

    :param netcdf_file_name: Path to input file.
    :param metadata_dict: Dictionary with metadata for NetCDF file, created by
        read_metadata_from_raw_file.
    :param raise_error_if_fails: Boolean flag.  If True and file cannot be
        opened, this method will raise an error.  If False and file cannot be
        opened, this method will return None for all output variables.
    :return: field_matrix: M-by-N numpy array with values of radar field.
    :return: unique_grid_point_lat_deg: length-M numpy array of grid-point
        latitudes (deg N).  If array is increasing (decreasing), latitude
        increases (decreases) while traveling down the columns of field_matrix.
    :return: unique_grid_point_lng_deg: length-N numpy array of grid-point
        longitudes (deg E).  If array is increasing (decreasing), longitude
        increases (decreases) while traveling right across the rows of
        field_matrix.
    """

    error_checking.assert_file_exists(netcdf_file_name)

    netcdf_dataset = netcdf_io.open_netcdf(netcdf_file_name,
                                           raise_error_if_fails)
    if netcdf_dataset is None:
        return None, None, None

    field_matrix = netcdf_dataset.variables[
        metadata_dict[FIELD_NAME_COLUMN_ORIG]]
    netcdf_dataset.close()

    min_latitude_deg = metadata_dict[NW_GRID_POINT_LAT_COLUMN] - (
        metadata_dict[LAT_SPACING_COLUMN] * (metadata_dict[NUM_LAT_COLUMN] - 1))
    unique_grid_point_lat_deg, unique_grid_point_lng_deg = (
        grids.get_latlng_grid_points(
            min_latitude_deg=min_latitude_deg,
            min_longitude_deg=metadata_dict[NW_GRID_POINT_LNG_COLUMN],
            lat_spacing_deg=metadata_dict[LAT_SPACING_COLUMN],
            lng_spacing_deg=metadata_dict[LNG_SPACING_COLUMN],
            num_rows=metadata_dict[NUM_LAT_COLUMN],
            num_columns=metadata_dict[NUM_LNG_COLUMN]))

    field_matrix = _remove_sentinels_from_full_grid(
        field_matrix, metadata_dict[SENTINEL_VALUE_COLUMN])
    return (numpy.flipud(field_matrix), unique_grid_point_lat_deg[::-1],
            unique_grid_point_lng_deg)
