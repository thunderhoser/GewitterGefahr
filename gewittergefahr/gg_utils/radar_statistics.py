"""Methods for computing radar statistics.

These are usually spatial statistics based on values inside a storm object.
"""

import os
import numpy
import scipy.stats
from gewittergefahr.gg_io import radar_io
from gewittergefahr.gg_io import storm_tracking_io as tracking_io
from gewittergefahr.gg_utils import radar_sparse_to_full as radar_s2f
from gewittergefahr.gg_utils import unzipping
from gewittergefahr.gg_utils import time_conversion
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
        percentile_levels=DEFAULT_PERCENTILE_LEVELS, radar_field_name=None,
        radar_height_m_agl=None, radar_data_source=radar_io.MYRORSS_SOURCE_ID,
        top_radar_directory_name=None):
    """Computes radar statistics for one or more storm objects.

    K = total number of statistics (percentile- and non-percentile-based)

    :param storm_object_table: pandas DataFrame with columns documented in
        `storm_tracking_io.write_processed_file`.  May contain additional
        columns.
    :param metadata_dict_for_storm_objects: Dictionary (with keys specified by
        `radar_io.read_metadata_from_raw_file`) describing grid used to create
        storm objects.
    :param statistic_names: 1-D list of non-percentile-based statistics.
    :param percentile_levels: 1-D numpy array of percentile levels.
    :param radar_field_name: Name of radar field.
    :param radar_height_m_agl: Height of radar field (metres above ground
        level).
    :param radar_data_source: Data source for radar field.
    :param top_radar_directory_name: Name of top-level directory with radar
        files from given source.
    :return: storm_object_table: Same as input, but with K additional columns.
        For non-percentile-based statistics, the column name will be
        "<radar_field_name>_<statistic_name>", where <statistic_name> comes from
        the list `STATISTIC_NAMES`.  For percentile-based statistics, the column
        name will be "<radar_field_name>_percentile<p>".
    """

    percentile_levels = _check_statistic_names(
        statistic_names, percentile_levels)

    storm_object_time_matrix = storm_object_table.as_matrix(
        columns=[tracking_io.TIME_COLUMN, tracking_io.SPC_DATE_COLUMN])
    unique_time_matrix = numpy.vstack(
        {tuple(this_row) for this_row in storm_object_time_matrix}).astype(int)
    unique_storm_times_unix_sec = unique_time_matrix[:, 0]
    unique_spc_dates_unix_sec = unique_time_matrix[:, 1]

    num_unique_storm_times = len(unique_storm_times_unix_sec)
    radar_file_names = [''] * num_unique_storm_times

    for i in range(num_unique_storm_times):
        radar_file_names[i] = radar_io.find_raw_file(
            unix_time_sec=unique_storm_times_unix_sec[i],
            spc_date_unix_sec=unique_spc_dates_unix_sec[i],
            field_name=radar_field_name, height_m_agl=radar_height_m_agl,
            data_source=radar_data_source,
            top_directory_name=top_radar_directory_name,
            raise_error_if_missing=True)

    num_statistics = len(statistic_names)
    num_percentiles = len(percentile_levels)
    num_storms = len(storm_object_table.index)
    statistic_matrix = numpy.full((num_storms, num_statistics), numpy.nan)
    percentile_matrix = numpy.full((num_storms, num_percentiles), numpy.nan)

    for i in range(num_unique_storm_times):
        this_time_string = time_conversion.unix_sec_to_string(
            unique_storm_times_unix_sec[i], '%Y-%m-%d-%H%M%S')
        print ('Computing stats for "' + str(radar_field_name) + '" at ' +
               str(radar_height_m_agl) + ' m AGL and ' + this_time_string +
               '...')

        if i == 0:
            metadata_dict_for_radar_field = (
                radar_io.read_metadata_from_raw_file(
                    radar_file_names[0], data_source=radar_data_source))
            storm_object_to_grid_points_table = (
                get_grid_points_in_storm_objects(
                    storm_object_table, metadata_dict_for_storm_objects,
                    metadata_dict_for_radar_field))

        this_sparse_grid_table = radar_io.read_sparse_grid_from_raw_file(
            radar_file_names[i],
            field_name_orig=metadata_dict_for_radar_field[
                radar_io.FIELD_NAME_COLUMN_ORIG],
            data_source=radar_data_source,
            sentinel_values=metadata_dict_for_radar_field[
                radar_io.SENTINEL_VALUE_COLUMN])

        this_field_matrix, _, _ = radar_s2f.sparse_to_full_grid(
            this_sparse_grid_table, metadata_dict_for_radar_field)

        these_storm_flags = numpy.logical_and(
            storm_object_table[tracking_io.TIME_COLUMN].values ==
            unique_storm_times_unix_sec[i],
            storm_object_table[tracking_io.SPC_DATE_COLUMN].values ==
            unique_spc_dates_unix_sec[i])
        these_storm_indices = numpy.where(these_storm_flags)[0]

        for this_storm_index in these_storm_indices:
            this_field_this_storm_values = extract_points_as_1d_array(
                this_field_matrix,
                row_indices=storm_object_to_grid_points_table[
                    tracking_io.GRID_POINT_ROW_COLUMN].values[
                    this_storm_index].astype(int),
                column_indices=storm_object_to_grid_points_table[
                    tracking_io.GRID_POINT_COLUMN_COLUMN].values[
                    this_storm_index].astype(int))

            (statistic_matrix[this_storm_index, :],
             percentile_matrix[this_storm_index, :]) = get_spatial_statistics(
                this_field_this_storm_values, statistic_names=statistic_names,
                percentile_levels=percentile_levels)

    argument_dict = {}
    for j in range(num_statistics):
        this_column_name = '{0:s}_{1:s}'.format(
            radar_field_name, statistic_names[j])
        argument_dict.update({this_column_name: statistic_matrix[:, j]})

    for k in range(num_percentiles):
        this_column_name = '{0:s}_percentile{1:f}'.format(
            radar_field_name, percentile_levels[k])
        argument_dict.update({this_column_name: percentile_matrix[:, k]})

    return storm_object_table.assign(**argument_dict)
