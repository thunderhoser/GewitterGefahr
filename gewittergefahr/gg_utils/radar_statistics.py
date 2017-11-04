"""Methods for computing radar statistics.

These are usually spatial statistics based on values inside a storm object.
"""

import pickle
import numpy
import pandas
import scipy.stats
from gewittergefahr.gg_io import radar_io
from gewittergefahr.gg_io import storm_tracking_io as tracking_io
from gewittergefahr.gg_utils import radar_sparse_to_full as radar_s2f
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import number_rounding as rounder
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.gg_utils import error_checking

TOLERANCE = 1e-6
TIME_FORMAT_FOR_LOG_MESSAGES = '%Y-%m-%d-%H%M%S'
STORM_COLUMNS_TO_KEEP = [tracking_io.STORM_ID_COLUMN, tracking_io.TIME_COLUMN]

RADAR_FIELD_NAME_KEY = 'radar_field_name'
RADAR_HEIGHT_KEY = 'radar_height_m_agl'
STATISTIC_NAME_KEY = 'statistic_name'
PERCENTILE_LEVEL_KEY = 'percentile_level'

MIN_ROW_IN_SUBGRID_COLUMN = 'min_row_in_subgrid'
MAX_ROW_IN_SUBGRID_COLUMN = 'max_row_in_subgrid'
MIN_COLUMN_IN_SUBGRID_COLUMN = 'min_column_in_subgrid'
MAX_COLUMN_IN_SUBGRID_COLUMN = 'max_column_in_subgrid'
NUM_PADDED_ROWS_AT_START_COLUMN = 'num_padded_rows_at_start'
NUM_PADDED_ROWS_AT_END_COLUMN = 'num_padded_rows_at_end'
NUM_PADDED_COLUMNS_AT_START_COLUMN = 'num_padded_columns_at_start'
NUM_PADDED_COLUMNS_AT_END_COLUMN = 'num_padded_columns_at_end'

GRID_METADATA_KEYS_TO_COMPARE = [
    radar_io.NW_GRID_POINT_LAT_COLUMN, radar_io.NW_GRID_POINT_LNG_COLUMN,
    radar_io.LAT_SPACING_COLUMN, radar_io.LNG_SPACING_COLUMN,
    radar_io.NUM_LAT_COLUMN, radar_io.NUM_LNG_COLUMN]

STORM_OBJECT_TO_GRID_PTS_COLUMNS = [
    tracking_io.STORM_ID_COLUMN, tracking_io.GRID_POINT_ROW_COLUMN,
    tracking_io.GRID_POINT_COLUMN_COLUMN]
GRID_POINT_LATLNG_COLUMNS = [tracking_io.GRID_POINT_LAT_COLUMN,
                             tracking_io.GRID_POINT_LNG_COLUMN]

AVERAGE_NAME = 'mean'
STANDARD_DEVIATION_NAME = 'standard_deviation'
SKEWNESS_NAME = 'skewness'
KURTOSIS_NAME = 'kurtosis'
STATISTIC_NAMES = [
    AVERAGE_NAME, STANDARD_DEVIATION_NAME, SKEWNESS_NAME, KURTOSIS_NAME]
DEFAULT_STATISTIC_NAMES = [
    AVERAGE_NAME, STANDARD_DEVIATION_NAME, SKEWNESS_NAME, KURTOSIS_NAME]

DEFAULT_PERCENTILE_LEVELS = numpy.array([0., 5., 25., 50., 75., 95., 100.])
PERCENTILE_LEVEL_PRECISION = 0.1

DEFAULT_RADAR_FIELD_NAMES = set(radar_io.RADAR_FIELD_NAMES)
DEFAULT_RADAR_FIELD_NAMES.remove(radar_io.REFL_NAME)
DEFAULT_RADAR_FIELD_NAMES.remove(radar_io.STORM_ID_NAME)
DEFAULT_RADAR_FIELD_NAMES = list(DEFAULT_RADAR_FIELD_NAMES)


def _radar_field_and_statistic_to_column_name(
        radar_field_name=None, radar_height_m_agl=None, statistic_name=None):
    """Generates column name for radar field and statistic.

    :param radar_field_name: Name of radar field.
    :param radar_height_m_agl: Radar height (metres above ground level).
    :param statistic_name: Name of statistic.
    :return: column_name: Name of column.
    """

    if radar_field_name == radar_io.REFL_NAME:
        return '{0:s}_{1:d}m_{2:s}'.format(
            radar_field_name, int(radar_height_m_agl), statistic_name)

    return '{0:s}_{1:s}'.format(
        radar_field_name, statistic_name)


def _radar_field_and_percentile_to_column_name(
        radar_field_name=None, radar_height_m_agl=None, percentile_level=None):
    """Generates column name for radar field and percentile level.

    :param radar_field_name: Name of radar field.
    :param radar_height_m_agl: Radar height (metres above ground level).
    :param percentile_level: Percentile level.
    :return: column_name: Name of column.
    """

    if radar_field_name == radar_io.REFL_NAME:
        return '{0:s}_{1:d}m_percentile{2:05.1f}'.format(
            radar_field_name, int(radar_height_m_agl), percentile_level)

    return '{0:s}_percentile{1:05.1f}'.format(
        radar_field_name, percentile_level)


def _column_name_to_statistic_params(column_name):
    """Determines parameters of statistic from column name.

    If column name does not correspond to a statistic, this method will return
    None.

    :param column_name: Name of column.
    :return: parameter_dict: Dictionary with the following keys.
    parameter_dict['radar_field_name']: Name of radar field on which statistic
        is based.
    parameter_dict['radar_height_m_agl']: Radar height (metres above ground
        level).  If radar field is not single-elevation reflectivity, this will
        be None.
    parameter_dict['statistic_name']: Name of statistic.  If statistic is a
        percentile, this will be None.
    parameter_dict['percentile_level']: Percentile level.  If statistic is non-
        percentile, this will be None.
    """

    column_name_parts = column_name.split('_')
    if len(column_name_parts) < 2:
        return None

    # Determine statistic name or percentile level.
    if column_name_parts[-1] in STATISTIC_NAMES:
        statistic_name = column_name_parts[-1]
        percentile_level = None
    else:
        statistic_name = None
        if not column_name_parts[-1].startswith('percentile'):
            return None

        try:
            percentile_level = float(column_name_parts[-1][len('percentile'):])
        except ValueError:
            return None

    # Determine radar field.
    radar_field_name = '_'.join(column_name_parts[:-1])
    radar_height_part = None
    try:
        radar_io.check_field_name(radar_field_name)
    except ValueError:
        radar_field_name = '_'.join(column_name_parts[:-2])
        radar_height_part = column_name_parts[-2]

    try:
        radar_io.check_field_name(radar_field_name)
    except ValueError:
        return None

    # Determine radar height.
    if radar_height_part is None:
        radar_height_m_agl = None
    else:
        if not radar_height_part.endswith('m'):
            return None

        try:
            radar_height_m_agl = int(radar_height_part[:-1])
        except ValueError:
            return None

    return {RADAR_FIELD_NAME_KEY: radar_field_name,
            RADAR_HEIGHT_KEY: radar_height_m_agl,
            STATISTIC_NAME_KEY: statistic_name,
            PERCENTILE_LEVEL_KEY: percentile_level}


def _center_points_latlng_to_rowcol(center_latitudes_deg, center_longitudes_deg,
                                    nw_grid_point_lat_deg=None,
                                    nw_grid_point_lng_deg=None,
                                    lat_spacing_deg=None, lng_spacing_deg=None):
    """Converts center points from lat-long to row-column coordinates.

    Each "center point" is meant for input to extract_points_as_2d_array.

    P = number of center points

    :param center_latitudes_deg: length-P numpy array of latitudes (deg N).
    :param center_longitudes_deg: length-P numpy array of longitudes (deg E).
    :param nw_grid_point_lat_deg: Latitude (deg N) of northwesternmost grid
        point.
    :param nw_grid_point_lng_deg: Longitude (deg E) of northwesternmost grid
        point.
    :param lat_spacing_deg: Spacing (deg N) between adjacent rows.
    :param lng_spacing_deg: Spacing (deg E) between adjacent columns.
    :return: center_row_indices: Row indices (half-integers) of center points.
    :return: center_column_indices: Column indices (half-integers) of center
        points.
    """

    center_row_indices, center_column_indices = radar_io.latlng_to_rowcol(
        center_latitudes_deg, center_longitudes_deg,
        nw_grid_point_lat_deg=nw_grid_point_lat_deg,
        nw_grid_point_lng_deg=nw_grid_point_lng_deg,
        lat_spacing_deg=lat_spacing_deg, lng_spacing_deg=lng_spacing_deg)

    return (rounder.round_to_half_integer(center_row_indices),
            rounder.round_to_half_integer(center_column_indices))


def _get_rowcol_indices_for_subgrid(num_rows_in_full_grid=None,
                                    num_columns_in_full_grid=None,
                                    center_row_index=None,
                                    center_column_index=None,
                                    num_rows_in_subgrid=None,
                                    num_columns_in_subgrid=None):
    """Generates row-column indices for subgrid.

    These row-column indices are meant for input to extract_points_as_2d_array.

    :param num_rows_in_full_grid: Number of rows in full grid (integer).
    :param num_columns_in_full_grid: Number of columns in full grid (integer).
    :param center_row_index: Row index (half-integer) at center point of
        subgrid.
    :param center_column_index: Column index (half-integer) at center point of
        subgrid.
    :param num_rows_in_subgrid: Number of rows in subgrid (even integer).
    :param num_columns_in_subgrid: Number of columns in subgrid (even integer).

    :return: subgrid_dict: Dictionary with the following keys.
    subgrid_dict['min_row_in_subgrid']: Minimum row (integer) in subgrid.  If
        min_row_in_subgrid = i, this means the [i]th row of the full grid is the
        first row in the subgrid.
    subgrid_dict['max_row_in_subgrid']: Maximum row (integer) in subgrid.
    subgrid_dict['min_column_in_subgrid']: Minimum column (integer) in subgrid.
    subgrid_dict['max_column_in_subgrid']: Maximum column (integer) in subgrid.
    subgrid_dict['num_padded_rows_at_start']: Number of NaN rows at beginning
        (top) of subgrid.
    subgrid_dict['num_padded_rows_at_end']: Number of NaN rows at end (bottom)
        of subgrid.
    subgrid_dict['num_padded_columns_at_start']: Number of NaN columns at
        beginning (left) of subgrid.
    subgrid_dict['num_padded_columns_at_end']: Number of NaN columns at end
        (right) of subgrid.
    """

    min_row_in_subgrid = int(numpy.ceil(
        center_row_index - num_rows_in_subgrid / 2))
    if min_row_in_subgrid >= 0:
        num_padded_rows_at_start = 0
    else:
        num_padded_rows_at_start = -1 * min_row_in_subgrid
        min_row_in_subgrid = 0

    max_row_in_subgrid = int(numpy.floor(
        center_row_index + num_rows_in_subgrid / 2))
    if max_row_in_subgrid <= num_rows_in_full_grid - 1:
        num_padded_rows_at_end = 0
    else:
        num_padded_rows_at_end = max_row_in_subgrid - (
            num_rows_in_full_grid - 1)
        max_row_in_subgrid = num_rows_in_full_grid - 1

    min_column_in_subgrid = int(numpy.ceil(
        center_column_index - num_columns_in_subgrid / 2))
    if min_column_in_subgrid >= 0:
        num_padded_columns_at_start = 0
    else:
        num_padded_columns_at_start = -1 * min_column_in_subgrid
        min_column_in_subgrid = 0

    max_column_in_subgrid = int(numpy.floor(
        center_column_index + num_columns_in_subgrid / 2))
    if max_column_in_subgrid <= num_columns_in_full_grid - 1:
        num_padded_columns_at_end = 0
    else:
        num_padded_columns_at_end = max_column_in_subgrid - (
            num_columns_in_full_grid - 1)
        max_column_in_subgrid = num_columns_in_full_grid - 1

    return {
        MIN_ROW_IN_SUBGRID_COLUMN: min_row_in_subgrid,
        MAX_ROW_IN_SUBGRID_COLUMN: max_row_in_subgrid,
        MIN_COLUMN_IN_SUBGRID_COLUMN: min_column_in_subgrid,
        MAX_COLUMN_IN_SUBGRID_COLUMN: max_column_in_subgrid,
        NUM_PADDED_ROWS_AT_START_COLUMN: num_padded_rows_at_start,
        NUM_PADDED_ROWS_AT_END_COLUMN: num_padded_rows_at_end,
        NUM_PADDED_COLUMNS_AT_START_COLUMN: num_padded_columns_at_start,
        NUM_PADDED_COLUMNS_AT_END_COLUMN: num_padded_columns_at_end}


def _are_grids_equal(metadata_dict_orig, metadata_dict_new):
    """Indicates whether or not two grids are equal.

    :param metadata_dict_orig: Dictionary (with keys specified by
        `radar_io.read_metadata_from_raw_file`) describing original radar grid.
    :param metadata_dict_new: Dictionary (with keys specified by
        `radar_io.read_metadata_from_raw_file`) describing new radar grid.
    :return: are_grids_equal_flag: Boolean flag.
    """

    metadata_dict_orig = {
        k: metadata_dict_orig[k] for k in GRID_METADATA_KEYS_TO_COMPARE}
    metadata_dict_new = {
        k: metadata_dict_new[k] for k in GRID_METADATA_KEYS_TO_COMPARE}

    return metadata_dict_orig == metadata_dict_new


def _check_statistic_names(statistic_names, percentile_levels):
    """Ensures that statistic names are valid.

    :param statistic_names: 1-D list with names of non-percentile-based
        statistics.
    :param percentile_levels: 1-D numpy array of percentile levels.
    :return: percentile_levels: Same as input, but rounded to the nearest 0.1%.
    :raises: ValueError: if any element of `statistic_names` is not in
        `STATISTIC_NAMES`.
    """

    error_checking.assert_is_string_list(statistic_names)
    error_checking.assert_is_numpy_array(
        numpy.array(statistic_names), num_dimensions=1)

    error_checking.assert_is_numpy_array(percentile_levels, num_dimensions=1)
    error_checking.assert_is_geq_numpy_array(percentile_levels, 0.)
    error_checking.assert_is_leq_numpy_array(percentile_levels, 100.)

    for this_name in statistic_names:
        if this_name in STATISTIC_NAMES:
            continue

        error_string = (
            '\n\n' + str(STATISTIC_NAMES) + '\n\nValid statistic names ' +
            '(listed above) do not include the following: "' + this_name + '"')
        raise ValueError(error_string)

    return numpy.unique(
        rounder.round_to_nearest(percentile_levels, PERCENTILE_LEVEL_PRECISION))


def get_statistic_columns(statistic_table):
    """Returns names of columns with radar statistics.

    :param statistic_table: pandas DataFrame.
    :return: statistic_column_names: 1-D list containing names of columns with
        radar statistics.  If there are no columns with radar stats, this is
        None.
    """

    column_names = list(statistic_table)
    statistic_column_names = None

    for this_column_name in column_names:
        this_parameter_dict = _column_name_to_statistic_params(this_column_name)
        if this_parameter_dict is None:
            continue

        if statistic_column_names is None:
            statistic_column_names = [this_column_name]
        else:
            statistic_column_names.append(this_column_name)

    return statistic_column_names


def extract_points_as_1d_array(field_matrix, row_indices=None,
                               column_indices=None):
    """Extracts radar values from several grid points into 1-D array.

    M = number of rows (unique grid-point latitudes)
    N = number of columns (unique grid-point longitudes)
    P = number of points to extract

    :param field_matrix: M-by-N numpy array with values of a single radar field.
    :param row_indices: length-P numpy array with row indices of points to
        extract.
    :param column_indices: length-P numpy array with column indices of points to
        extract.
    :return: extracted_values: length-P numpy array of values extracted from
        field_matrix.
    """

    error_checking.assert_is_real_numpy_array(field_matrix)
    error_checking.assert_is_numpy_array(field_matrix, num_dimensions=2)
    num_grid_rows = field_matrix.shape[0]
    num_grid_columns = field_matrix.shape[1]

    error_checking.assert_is_integer_numpy_array(row_indices)
    error_checking.assert_is_geq_numpy_array(row_indices, 0)
    error_checking.assert_is_less_than_numpy_array(row_indices, num_grid_rows)

    error_checking.assert_is_integer_numpy_array(column_indices)
    error_checking.assert_is_geq_numpy_array(column_indices, 0)
    error_checking.assert_is_less_than_numpy_array(column_indices,
                                                   num_grid_columns)

    return field_matrix[row_indices, column_indices]


def extract_points_as_2d_array(field_matrix, center_row_index=None,
                               center_column_index=None,
                               num_rows_in_subgrid=None,
                               num_columns_in_subgrid=None):
    """Extracts radar values from contiguous subset of grid into 2-D array.

    M = number of rows (unique grid-point latitudes) in full grid
    N = number of columns (unique grid-point longitudes) in full grid
    m = number of rows in subgrid
    n = number of columns in subgrid

    :param field_matrix: M-by-N numpy array with values of a single radar field.
    :param center_row_index: Row index (half-integer) at center of subgrid.
    :param center_column_index: Column index (half-integer) at center of
        subgrid.
    :param num_rows_in_subgrid: Number of rows in subgrid.
    :param num_columns_in_subgrid: Number of columns in subgrid.
    :return: field_submatrix: m-by-n numpy array with values in subgrid.
    """

    error_checking.assert_is_real_numpy_array(field_matrix)
    error_checking.assert_is_numpy_array(field_matrix, num_dimensions=2)
    num_rows_in_full_grid = field_matrix.shape[0]
    num_columns_in_full_grid = field_matrix.shape[1]

    subgrid_dict = _get_rowcol_indices_for_subgrid(
        num_rows_in_full_grid=num_rows_in_full_grid,
        num_columns_in_full_grid=num_columns_in_full_grid,
        center_row_index=center_row_index,
        center_column_index=center_column_index,
        num_rows_in_subgrid=num_rows_in_subgrid,
        num_columns_in_subgrid=num_columns_in_subgrid)

    min_row_in_subgrid = subgrid_dict[MIN_ROW_IN_SUBGRID_COLUMN]
    max_row_in_subgrid = subgrid_dict[MAX_ROW_IN_SUBGRID_COLUMN]
    min_column_in_subgrid = subgrid_dict[MIN_COLUMN_IN_SUBGRID_COLUMN]
    max_column_in_subgrid = subgrid_dict[MAX_COLUMN_IN_SUBGRID_COLUMN]

    field_submatrix = field_matrix[
        min_row_in_subgrid:(max_row_in_subgrid + 1),
        min_column_in_subgrid:(max_column_in_subgrid + 1)]

    num_padded_rows_at_start = subgrid_dict[NUM_PADDED_ROWS_AT_START_COLUMN]
    num_padded_rows_at_end = subgrid_dict[NUM_PADDED_ROWS_AT_END_COLUMN]
    num_padded_columns_at_start = subgrid_dict[
        NUM_PADDED_COLUMNS_AT_START_COLUMN]
    num_padded_columns_at_end = subgrid_dict[NUM_PADDED_COLUMNS_AT_END_COLUMN]

    if num_padded_rows_at_start > 0:
        nan_matrix = numpy.full(
            (num_padded_rows_at_start, field_submatrix.shape[1]), numpy.nan)
        field_submatrix = numpy.vstack((nan_matrix, field_submatrix))

    if num_padded_rows_at_end > 0:
        nan_matrix = numpy.full(
            (num_padded_rows_at_end, field_submatrix.shape[1]), numpy.nan)
        field_submatrix = numpy.vstack((field_submatrix, nan_matrix))

    if num_padded_columns_at_start > 0:
        nan_matrix = numpy.full(
            (field_submatrix.shape[0], num_padded_columns_at_start), numpy.nan)
        field_submatrix = numpy.hstack((nan_matrix, field_submatrix))

    if num_padded_columns_at_end > 0:
        nan_matrix = numpy.full(
            (field_submatrix.shape[0], num_padded_columns_at_end), numpy.nan)
        field_submatrix = numpy.hstack((field_submatrix, nan_matrix))

    return field_submatrix


def get_grid_points_in_storm_objects(storm_object_table,
                                     metadata_dict_for_storm_objects,
                                     new_metadata_dict):
    """Finds grid points inside each storm object.

    :param storm_object_table: pandas DataFrame with columns specified by
        `storm_tracking_io.write_processed_file`.
    :param metadata_dict_for_storm_objects: Dictionary (with keys specified by
        `radar_io.read_metadata_from_raw_file`) describing grid used to create
        storm objects.
    :param new_metadata_dict: Dictionary (with keys specified by
        `radar_io.read_metadata_from_raw_file`) describing new grid, for which
        points in each storm object will be found.
    :return: storm_object_to_grid_points_table: pandas DataFrame with the
        following columns.  Each row is one storm object.
    storm_object_to_grid_points_table.storm_id: String ID for storm cell.
    storm_object_to_grid_points_table.grid_point_rows: 1-D numpy array with row
        indices (integers) of grid points in storm object.
    storm_object_to_grid_points_table.grid_point_columns: 1-D numpy array with
        column indices (integers) of grid points in storm object.
    """

    if _are_grids_equal(metadata_dict_for_storm_objects, new_metadata_dict):
        return storm_object_table[STORM_OBJECT_TO_GRID_PTS_COLUMNS]

    storm_object_to_grid_points_table = storm_object_table[
        STORM_OBJECT_TO_GRID_PTS_COLUMNS + GRID_POINT_LATLNG_COLUMNS]
    num_storm_objects = len(storm_object_to_grid_points_table.index)

    for i in range(num_storm_objects):
        (storm_object_to_grid_points_table[
            tracking_io.GRID_POINT_ROW_COLUMN].values[i],
         storm_object_to_grid_points_table[
             tracking_io.GRID_POINT_COLUMN_COLUMN].values[i]) = (
                 radar_io.latlng_to_rowcol(
                     storm_object_to_grid_points_table[
                         tracking_io.GRID_POINT_LAT_COLUMN].values[i],
                     storm_object_to_grid_points_table[
                         tracking_io.GRID_POINT_LNG_COLUMN].values[i],
                     nw_grid_point_lat_deg=
                     new_metadata_dict[radar_io.NW_GRID_POINT_LAT_COLUMN],
                     nw_grid_point_lng_deg=
                     new_metadata_dict[radar_io.NW_GRID_POINT_LNG_COLUMN],
                     lat_spacing_deg=
                     new_metadata_dict[radar_io.LAT_SPACING_COLUMN],
                     lng_spacing_deg=
                     new_metadata_dict[radar_io.LNG_SPACING_COLUMN]))

    return storm_object_to_grid_points_table[STORM_OBJECT_TO_GRID_PTS_COLUMNS]


def get_spatial_statistics(radar_field, statistic_names=DEFAULT_STATISTIC_NAMES,
                           percentile_levels=DEFAULT_PERCENTILE_LEVELS):
    """Computes spatial statistics for a single radar field.

    "Single field" = one variable at one elevation, one time step, many spatial
    locations.

    Radar field may have any number of dimensions (1-D, 2-D, etc.).

    N = number of non-percentile-based statistics
    P = number of percentile levels

    :param radar_field: numpy array.  Each position in the array should be a
        different spatial location.
    :param statistic_names: length-N list of non-percentile-based statistics.
    :param percentile_levels: length-P numpy array of percentile levels.
    :return: statistic_values: length-N numpy with values of non-percentile-
        based statistics.
    :return: percentile_values: length-P numpy array of percentiles.
    """

    error_checking.assert_is_real_numpy_array(radar_field)
    percentile_levels = _check_statistic_names(
        statistic_names, percentile_levels)

    num_statistics = len(statistic_names)
    statistic_values = numpy.full(num_statistics, numpy.nan)
    for i in range(num_statistics):
        if statistic_names[i] == AVERAGE_NAME:
            statistic_values[i] = numpy.nanmean(radar_field)
        elif statistic_names[i] == STANDARD_DEVIATION_NAME:
            statistic_values[i] = numpy.nanstd(radar_field, ddof=1)
        elif statistic_names[i] == SKEWNESS_NAME:
            statistic_values[i] = scipy.stats.skew(
                radar_field, bias=False, nan_policy='omit', axis=None)
        elif statistic_names[i] == KURTOSIS_NAME:
            statistic_values[i] = scipy.stats.kurtosis(
                radar_field, fisher=True, bias=False, nan_policy='omit',
                axis=None)

    percentile_values = numpy.nanpercentile(
        radar_field, percentile_levels, interpolation='linear')
    return statistic_values, percentile_values


def get_stats_for_storm_objects(
        storm_object_table, metadata_dict_for_storm_objects,
        statistic_names=DEFAULT_STATISTIC_NAMES,
        percentile_levels=DEFAULT_PERCENTILE_LEVELS,
        radar_field_names=DEFAULT_RADAR_FIELD_NAMES,
        reflectivity_heights_m_agl=None,
        radar_data_source=radar_io.MYRORSS_SOURCE_ID,
        top_radar_directory_name=None):
    """Computes radar statistics for one or more storm objects.

    F = number of radar field-height pairs
    K = number of statistics (percentile- and non-percentile-based)

    :param storm_object_table: pandas DataFrame with columns documented in
        `storm_tracking_io.write_processed_file`.  May contain additional
        columns.
    :param metadata_dict_for_storm_objects: Dictionary (with keys specified by
        `radar_io.read_metadata_from_raw_file`) describing grid used to create
        storm objects.
    :param statistic_names: 1-D list of non-percentile-based statistics.
    :param percentile_levels: 1-D numpy array of percentile levels.
    :param radar_field_names: 1-D list with names of radar fields.
    :param reflectivity_heights_m_agl: 1-D numpy array of heights (metres above
        ground level) for radar field "reflectivity_dbz".  If "reflectivity_dbz"
        is not in `radar_field_names`, this can be left as None.
    :param radar_data_source: Data source for radar field.
    :param top_radar_directory_name: Name of top-level directory with radar
        files from given source.
    :return: storm_radar_statistic_table: pandas DataFrame with 2 + K * F
        columns, where the last K * F columns are one for each statistic-field
        pair.  Names of these columns are determined by
        _radar_field_and_statistic_to_column_name and
        _radar_field_and_percentile_to_column_name.  The first 2 columns are
        listed below.
    storm_radar_statistic_table.storm_id: String ID for storm cell.  Same as
        input column `storm_object_table.storm_id`.
    storm_radar_statistic_table.unix_time_sec: Valid time.  Same as input column
        `storm_object_table.unix_time_sec`.
    """

    percentile_levels = _check_statistic_names(
        statistic_names, percentile_levels)

    radar_field_name_by_pair, radar_height_by_pair_m_agl = (
        radar_io.unique_fields_and_heights_to_pairs(
            radar_field_names, refl_heights_m_agl=reflectivity_heights_m_agl,
            data_source=radar_data_source))
    num_radar_fields = len(radar_field_name_by_pair)

    storm_object_time_matrix = storm_object_table.as_matrix(
        columns=[tracking_io.TIME_COLUMN, tracking_io.SPC_DATE_COLUMN])
    unique_time_matrix = numpy.vstack(
        {tuple(this_row) for this_row in storm_object_time_matrix}).astype(int)
    unique_storm_times_unix_sec = unique_time_matrix[:, 0]
    unique_spc_dates_unix_sec = unique_time_matrix[:, 1]

    num_unique_storm_times = len(unique_storm_times_unix_sec)
    radar_file_name_matrix = numpy.full(
        (num_unique_storm_times, num_radar_fields), '', dtype=object)

    for i in range(num_unique_storm_times):
        for j in range(num_radar_fields):
            radar_file_name_matrix[i, j] = radar_io.find_raw_file(
                unix_time_sec=unique_storm_times_unix_sec[i],
                spc_date_unix_sec=unique_spc_dates_unix_sec[i],
                field_name=radar_field_name_by_pair[j],
                height_m_agl=radar_height_by_pair_m_agl[j],
                data_source=radar_data_source,
                top_directory_name=top_radar_directory_name,
                raise_error_if_missing=True)

    num_statistics = len(statistic_names)
    num_percentiles = len(percentile_levels)
    num_storms = len(storm_object_table.index)
    statistic_matrix = numpy.full(
        (num_storms, num_radar_fields, num_statistics), numpy.nan)
    percentile_matrix = numpy.full(
        (num_storms, num_radar_fields, num_percentiles), numpy.nan)

    for j in range(num_radar_fields):
        for i in range(num_unique_storm_times):
            this_time_string = time_conversion.unix_sec_to_string(
                unique_storm_times_unix_sec[i], TIME_FORMAT_FOR_LOG_MESSAGES)
            print ('Computing stats for "' + str(radar_field_name_by_pair[j]) +
                   '" at ' + str(radar_height_by_pair_m_agl[j]) +
                   ' m AGL and ' + this_time_string + '...')

            if i == 0:
                metadata_dict_for_this_field = (
                    radar_io.read_metadata_from_raw_file(
                        radar_file_name_matrix[i, j],
                        data_source=radar_data_source))
                storm_object_to_grid_pts_table_this_field = (
                    get_grid_points_in_storm_objects(
                        storm_object_table, metadata_dict_for_storm_objects,
                        metadata_dict_for_this_field))

            sparse_grid_table_this_field = (
                radar_io.read_data_from_sparse_grid_file(
                    radar_file_name_matrix[i, j],
                    field_name_orig=metadata_dict_for_this_field[
                        radar_io.FIELD_NAME_COLUMN_ORIG],
                    data_source=radar_data_source,
                    sentinel_values=metadata_dict_for_this_field[
                        radar_io.SENTINEL_VALUE_COLUMN]))

            radar_matrix_this_field, _, _ = radar_s2f.sparse_to_full_grid(
                sparse_grid_table_this_field, metadata_dict_for_this_field)
            radar_matrix_this_field[numpy.isnan(radar_matrix_this_field)] = 0.

            these_storm_flags = numpy.logical_and(
                storm_object_table[tracking_io.TIME_COLUMN].values ==
                unique_storm_times_unix_sec[i],
                storm_object_table[tracking_io.SPC_DATE_COLUMN].values ==
                unique_spc_dates_unix_sec[i])
            these_storm_indices = numpy.where(these_storm_flags)[0]

            for this_storm_index in these_storm_indices:
                radar_values_this_storm = extract_points_as_1d_array(
                    radar_matrix_this_field,
                    row_indices=storm_object_to_grid_pts_table_this_field[
                        tracking_io.GRID_POINT_ROW_COLUMN].values[
                            this_storm_index].astype(int),
                    column_indices=storm_object_to_grid_pts_table_this_field[
                        tracking_io.GRID_POINT_COLUMN_COLUMN].values[
                            this_storm_index].astype(int))

                (statistic_matrix[this_storm_index, j, :],
                 percentile_matrix[this_storm_index, j, :]) = (
                     get_spatial_statistics(
                         radar_values_this_storm,
                         statistic_names=statistic_names,
                         percentile_levels=percentile_levels))

    storm_radar_statistic_dict = {}
    for j in range(num_radar_fields):
        for k in range(num_statistics):
            this_column_name = _radar_field_and_statistic_to_column_name(
                radar_field_name=radar_field_name_by_pair[j],
                radar_height_m_agl=radar_height_by_pair_m_agl[j],
                statistic_name=statistic_names[k])

            storm_radar_statistic_dict.update(
                {this_column_name: statistic_matrix[:, j, k]})

        for k in range(num_percentiles):
            this_column_name = _radar_field_and_percentile_to_column_name(
                radar_field_name=radar_field_name_by_pair[j],
                radar_height_m_agl=radar_height_by_pair_m_agl[j],
                percentile_level=percentile_levels[k])

            storm_radar_statistic_dict.update(
                {this_column_name: percentile_matrix[:, j, k]})

    storm_radar_statistic_table = pandas.DataFrame.from_dict(
        storm_radar_statistic_dict)
    return pandas.concat(
        [storm_object_table[STORM_COLUMNS_TO_KEEP],
         storm_radar_statistic_table], axis=1)


def write_stats_for_storm_objects(storm_radar_statistic_table,
                                  pickle_file_name):
    """Writes radar statistics for storm objects to a Pickle file.

    :param storm_radar_statistic_table: pandas DataFrame created by
        get_stats_for_storm_objects.
    :param pickle_file_name: Path to output file.
    :raises: ValueError: if storm_radar_statistic_table does not contain any
        column with radar statistics.
    """

    statistic_column_names = get_statistic_columns(storm_radar_statistic_table)
    if statistic_column_names is None:
        raise ValueError(
            'storm_radar_statistic_table does not contain any column with '
            'radar statistics.')

    file_system_utils.mkdir_recursive_if_necessary(file_name=pickle_file_name)
    columns_to_write = STORM_COLUMNS_TO_KEEP + statistic_column_names

    pickle_file_handle = open(pickle_file_name, 'wb')
    pickle.dump(storm_radar_statistic_table[columns_to_write],
                pickle_file_handle)
    pickle_file_handle.close()


def read_stats_for_storm_objects(pickle_file_name):
    """Reads radar statistics for storm objects from a Pickle file.

    :param pickle_file_name: Path to input file.
    :return: storm_radar_statistic_table: pandas DataFrame with columns
        documented in get_stats_for_storm_objects.
    """

    pickle_file_handle = open(pickle_file_name, 'rb')
    storm_radar_statistic_table = pickle.load(pickle_file_handle)
    pickle_file_handle.close()

    error_checking.assert_columns_in_dataframe(
        storm_radar_statistic_table, STORM_COLUMNS_TO_KEEP)

    statistic_column_names = get_statistic_columns(storm_radar_statistic_table)
    if statistic_column_names is None:
        raise ValueError(
            'storm_radar_statistic_table does not contain any column with '
            'radar statistics.')

    return storm_radar_statistic_table
