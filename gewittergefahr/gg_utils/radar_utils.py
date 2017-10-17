"""Methods for processing radar data."""

import numpy
import scipy.stats
from gewittergefahr.gg_io import radar_io
from gewittergefahr.gg_io import storm_tracking_io
from gewittergefahr.gg_utils import number_rounding as rounder
from gewittergefahr.gg_utils import error_checking

TOLERANCE = 1e-6

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
    storm_tracking_io.STORM_ID_COLUMN, storm_tracking_io.GRID_POINT_ROW_COLUMN,
    storm_tracking_io.GRID_POINT_COLUMN_COLUMN]
GRID_POINT_LATLNG_COLUMNS = [storm_tracking_io.GRID_POINT_LAT_COLUMN,
                             storm_tracking_io.GRID_POINT_LNG_COLUMN]

MEAN_STRING = 'mean'
STDEV_STRING = 'standard_deviation'
SKEWNESS_STRING = 'skewness'
KURTOSIS_STRING = 'kurtosis'
VALID_STRING_STATS = [
    MEAN_STRING, STDEV_STRING, SKEWNESS_STRING, KURTOSIS_STRING]
DEFAULT_STRING_STATS = [
    MEAN_STRING, STDEV_STRING, SKEWNESS_STRING, KURTOSIS_STRING]

DEFAULT_PERCENTILE_LEVELS = numpy.array([0., 5., 25., 50., 75., 95., 100.])


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


def _check_stats_to_compute(string_statistics, percentile_levels):
    """Ensures that statistics to compute are valid.

    :param string_statistics: 1-D list of non-percentile-based statistics.
    :param percentile_levels: 1-D numpy array of percentile levels.
    :raises: ValueError: if any element of string_statistics is not in
        VALID_STRING_STATS.
    """

    error_checking.assert_is_string_list(string_statistics)
    error_checking.assert_is_numpy_array(
        numpy.array(string_statistics), num_dimensions=1)

    error_checking.assert_is_numpy_array(percentile_levels, num_dimensions=1)
    error_checking.assert_is_geq_numpy_array(percentile_levels, 0.)
    error_checking.assert_is_leq_numpy_array(percentile_levels, 100.)

    for this_string in string_statistics:
        if this_string in VALID_STRING_STATS:
            continue

        error_string = (
            '\n\n' + str(VALID_STRING_STATS) + '\n\nValid non-percentile ' +
            'stats (listed above) do not include the following: "' +
            this_string + '"')
        raise ValueError(error_string)


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
            storm_tracking_io.GRID_POINT_ROW_COLUMN].values[i],
         storm_object_to_grid_points_table[
             storm_tracking_io.GRID_POINT_COLUMN_COLUMN].values[i]) = (
                 radar_io.latlng_to_rowcol(
                     storm_object_to_grid_points_table[
                         storm_tracking_io.GRID_POINT_LAT_COLUMN].values[i],
                     storm_object_to_grid_points_table[
                         storm_tracking_io.GRID_POINT_LNG_COLUMN].values[i],
                     nw_grid_point_lat_deg=
                     new_metadata_dict[radar_io.NW_GRID_POINT_LAT_COLUMN],
                     nw_grid_point_lng_deg=
                     new_metadata_dict[radar_io.NW_GRID_POINT_LNG_COLUMN],
                     lat_spacing_deg=
                     new_metadata_dict[radar_io.LAT_SPACING_COLUMN],
                     lng_spacing_deg=
                     new_metadata_dict[radar_io.LNG_SPACING_COLUMN]))

    return storm_object_to_grid_points_table[STORM_OBJECT_TO_GRID_PTS_COLUMNS]


def get_spatial_statistics(radar_field, string_statistics=DEFAULT_STRING_STATS,
                           percentile_levels=DEFAULT_PERCENTILE_LEVELS):
    """Computes spatial statistics for a single radar field.

    "Single field" = one variable at one elevation, one time step, several
    spatial locations.

    Radar field may have any number of dimensions (1-D, 2-D, etc.).

    N = number of non-percentile statistics
    P = number of percentiles

    :param radar_field: numpy array.  Each position in the array should be a
        different spatial location.
    :param string_statistics: length-N list of non-percentile statistics.
    :param percentile_levels: length-P numpy array of percentile levels.
    :return: string_stat_results: length-N numpy array of values for non-
        percentile statistics.
    :return: percentile_results: length-P numpy array of values for percentiles.
    """

    error_checking.assert_is_real_numpy_array(radar_field)
    _check_stats_to_compute(string_statistics, percentile_levels)

    num_string_stats = len(string_statistics)
    string_stat_results = numpy.full(num_string_stats, numpy.nan)
    for i in range(num_string_stats):
        if string_statistics[i] == MEAN_STRING:
            string_stat_results[i] = numpy.nanmean(radar_field)
        elif string_statistics[i] == STDEV_STRING:
            string_stat_results[i] = numpy.nanstd(radar_field, ddof=1)
        elif string_statistics[i] == SKEWNESS_STRING:
            string_stat_results[i] = scipy.stats.skew(
                radar_field, bias=False, nan_policy='omit', axis=None)
        elif string_statistics[i] == KURTOSIS_STRING:
            string_stat_results[i] = scipy.stats.kurtosis(
                radar_field, fisher=True, bias=False, nan_policy='omit',
                axis=None)

    percentile_results = numpy.nanpercentile(
        radar_field, percentile_levels, interpolation='linear')
    return string_stat_results, percentile_results
