"""Methods for processing storm tracks and outlines."""

import copy
import numpy
import pandas
from gewittergefahr.gg_io import storm_tracking_io as tracking_io
from gewittergefahr.gg_utils import error_checking

FLATTENED_INDEX_COLUMN = 'flattened_index'
STORM_ID_COLUMN = 'storm_id'

COLUMNS_TO_CHANGE_WHEN_MERGING_SCALES = [
    tracking_io.CENTROID_LAT_COLUMN, tracking_io.CENTROID_LNG_COLUMN,
    tracking_io.GRID_POINT_LAT_COLUMN, tracking_io.GRID_POINT_LNG_COLUMN,
    tracking_io.GRID_POINT_ROW_COLUMN, tracking_io.GRID_POINT_COLUMN_COLUMN,
    tracking_io.POLYGON_OBJECT_LATLNG_COLUMN,
    tracking_io.POLYGON_OBJECT_ROWCOL_COLUMN]


def _get_grid_points_in_storms(storm_object_table, num_grid_rows=None,
                               num_grid_columns=None):
    """Finds grid points in all storm objects.

    N = number of storm objects
    P = number of grid points in a storm object

    :param storm_object_table: N-row pandas DataFrame in format specified by
        `storm_tracking_io.write_processed_file`.
    :param num_grid_rows: Number of rows (unique grid-point latitudes).
    :param num_grid_columns: Number of columns (unique grid-point longitudes).
    :return: grid_points_in_storms_table: P-row pandas DataFrame with the
        following columns.
    grid_points_in_storms_table.flattened_index: Flattened index (integer) of
        grid point.
    grid_points_in_storms_table.storm_id: String ID for storm cell.
    """

    error_checking.assert_is_integer(num_grid_rows)
    error_checking.assert_is_greater(num_grid_rows, 0)
    error_checking.assert_is_integer(num_grid_columns)
    error_checking.assert_is_greater(num_grid_columns, 0)

    grid_point_row_indices = numpy.array([])
    grid_point_column_indices = numpy.array([])
    grid_point_storm_ids = []

    num_storms = len(storm_object_table.index)
    for i in range(num_storms):
        grid_point_row_indices = numpy.concatenate((
            grid_point_row_indices,
            storm_object_table[tracking_io.GRID_POINT_ROW_COLUMN].values[i]))
        grid_point_column_indices = numpy.concatenate((
            grid_point_column_indices,
            storm_object_table[tracking_io.GRID_POINT_COLUMN_COLUMN].values[i]))

        this_num_grid_points = len(
            storm_object_table[tracking_io.GRID_POINT_ROW_COLUMN].values[i])
        this_storm_id_list = (
            [storm_object_table[tracking_io.STORM_ID_COLUMN].values[i]] *
            this_num_grid_points)
        grid_point_storm_ids += this_storm_id_list

    grid_point_flattened_indices = numpy.ravel_multi_index(
        (grid_point_row_indices.astype(int),
         grid_point_column_indices.astype(int)),
        (num_grid_rows, num_grid_columns))

    grid_points_in_storms_dict = {
        FLATTENED_INDEX_COLUMN: grid_point_flattened_indices,
        STORM_ID_COLUMN: grid_point_storm_ids
    }
    return pandas.DataFrame.from_dict(grid_points_in_storms_dict)


def merge_storms_at_two_scales(storm_object_table_small_scale=None,
                               storm_object_table_large_scale=None,
                               num_grid_rows=None, num_grid_columns=None):
    """Merges storm objects at two tracking scales, using the probSevere method.

    For each large-scale object L, if there is only one small-scale object S
    inside L, replace S with L.  In other words, if there is only one small-
    scale object inside a large-scale object, the small-scale object is grown to
    the exact size and dimensions of the large-scale object.

    N_s = number of small-scale storm objects
    N_l = number of large-scale storm objects

    :param storm_object_table_small_scale: pandas DataFrame with N_s rows, in
        format specified by `storm_tracking_io.write_processed_file`.
    :param storm_object_table_large_scale: pandas DataFrame with N_l rows, in
        format specified by `storm_tracking_io.write_processed_file`.
    :param num_grid_rows: Number of rows (unique grid-point latitudes) in grid
        used to detect storm objects.
    :param num_grid_columns: Number of columns (unique grid-point longitudes) in
        grid used to detect storm objects.
    :return: storm_object_table_merged: Same as input, except that some
        small-scale objects may have been replaced with a larger-scale object.
    """

    # TODO(thunderhoser): prevent a small-scale storm object from being grown
    # more than once.

    grid_point_table_small_scale = _get_grid_points_in_storms(
        storm_object_table_small_scale, num_grid_rows=num_grid_rows,
        num_grid_columns=num_grid_columns)

    storm_ids_large_scale = storm_object_table_large_scale[
        tracking_io.STORM_ID_COLUMN].values
    num_storms_large_scale = len(storm_ids_large_scale)
    storm_object_table_merged = copy.deepcopy(storm_object_table_small_scale)

    for i in range(num_storms_large_scale):
        these_row_indices = storm_object_table_large_scale[
            tracking_io.GRID_POINT_ROW_COLUMN].values[i]
        these_column_indices = storm_object_table_large_scale[
            tracking_io.GRID_POINT_COLUMN_COLUMN].values[i]
        these_flattened_indices = numpy.ravel_multi_index(
            (these_row_indices, these_column_indices),
            (num_grid_rows, num_grid_columns))

        these_point_in_storm_flags = [
            j in these_flattened_indices for j in grid_point_table_small_scale[
                FLATTENED_INDEX_COLUMN].values]
        if not numpy.any(these_point_in_storm_flags):
            continue

        these_point_in_storm_indices = numpy.where(
            these_point_in_storm_flags)[0]
        these_storm_ids_small_scale = numpy.unique(
            grid_point_table_small_scale[STORM_ID_COLUMN].values[
                these_point_in_storm_indices])
        if len(these_storm_ids_small_scale) != 1:
            continue

        this_storm_id_small_scale = these_storm_ids_small_scale[0]
        these_storm_flags_small_scale = [
            s == this_storm_id_small_scale for s in
            storm_object_table_small_scale[tracking_io.STORM_ID_COLUMN].values]
        this_storm_index_small_scale = numpy.where(
            these_storm_flags_small_scale)[0][0]

        for this_column in COLUMNS_TO_CHANGE_WHEN_MERGING_SCALES:
            storm_object_table_merged[this_column].values[
                this_storm_index_small_scale] = storm_object_table_large_scale[
                    this_column].values[i]

    return storm_object_table_merged
