"""Converts radar data from sparse to full grid.

Radar data may come from either MYRORSS (Multi-year Reanalysis of Remotely
Sensed Storms) or MRMS (Multi-radar Multi-sensor).
"""

import copy
import numpy
from gewittergefahr.gg_io import myrorss_and_mrms_io
from gewittergefahr.gg_utils import grids
from gewittergefahr.gg_utils import radar_utils

MIN_CENTER_LAT_COLUMN = 'min_center_lat_deg'
MAX_CENTER_LAT_COLUMN = 'max_center_lat_deg'
MIN_CENTER_LNG_COLUMN = 'min_center_lng_deg'
MAX_CENTER_LNG_COLUMN = 'max_center_lng_deg'


def _convert(sparse_grid_table, field_name, num_grid_rows, num_grid_columns,
             ignore_if_below=None):
    """Converts data from sparse to full grid.

    M = number of rows in grid
    N = number of columns in grid

    :param sparse_grid_table: pandas DataFrame created by
        `myrorss_and_mrms_io.read_data_from_sparse_grid_file`.
    :param field_name: Name of radar field.  Must be a column in
        `sparse_grid_table`.
    :param num_grid_rows: M in the above discussion.
    :param num_grid_columns: N in the above discussion.
    :param ignore_if_below: Minimum value to consider.  This method will ignore
        all values < `ignore_if_below` -- in other words, will not put them in
        the full grid.  If `ignore_if_below is None`, this method will put all
        values in the full grid.
    :return: full_radar_matrix: M-by-N numpy array with values of `field_name`.
    """

    if ignore_if_below is not None:
        these_indices = numpy.where(
            sparse_grid_table[field_name].values >= ignore_if_below
        )[0]
        sparse_grid_table = sparse_grid_table.iloc[these_indices]

    start_indices_tuple = (
        sparse_grid_table[myrorss_and_mrms_io.GRID_ROW_COLUMN].values,
        sparse_grid_table[myrorss_and_mrms_io.GRID_COLUMN_COLUMN].values
    )
    start_indices_flat = numpy.ravel_multi_index(
        start_indices_tuple, (num_grid_rows, num_grid_columns)
    )
    end_indices_flat = (
        start_indices_flat +
        sparse_grid_table[myrorss_and_mrms_io.NUM_GRID_CELL_COLUMN].values - 1
    )
    data_values_flat = sparse_grid_table[field_name].values

    full_matrix = numpy.full(num_grid_rows * num_grid_columns, numpy.nan)

    for i, j, k in zip(start_indices_flat, end_indices_flat, data_values_flat):
        full_matrix[i:(j + 1)] = k

    return numpy.reshape(full_matrix, (num_grid_rows, num_grid_columns))


def sparse_to_full_grid(sparse_grid_table, metadata_dict, ignore_if_below=None):
    """Converts data from sparse to full grid (public wrapper for _convert).

    M = number of rows (unique grid-point latitudes)
    N = number of columns (unique grid-point longitudes)

    :param sparse_grid_table: pandas DataFrame created by
        `myrorss_and_mrms_io.read_data_from_sparse_grid_file`.
    :param metadata_dict: Dictionary created by
        `myrorss_and_mrms_io.read_metadata_from_raw_file`.
    :param ignore_if_below: This method will ignore radar values <
        `ignore_if_below`.  If None, this method will consider all values.
    :return: full_matrix: M-by-N numpy array of radar values.  Latitude
        decreases down each column, and longitude increases to the right along
        each row.
    :return: grid_point_latitudes_deg: length-M numpy array of grid-point
        latitudes (deg N), sorted in descending order.
    :return: grid_point_longitudes_deg: length-N numpy array of grid-point
        longitudes (deg E), sorted in acending order.
    """

    min_latitude_deg = (
        metadata_dict[radar_utils.NW_GRID_POINT_LAT_COLUMN] - (
            metadata_dict[radar_utils.LAT_SPACING_COLUMN] *
            (metadata_dict[radar_utils.NUM_LAT_COLUMN] - 1)
        )
    )

    unique_grid_point_lat_deg, unique_grid_point_lng_deg = (
        grids.get_latlng_grid_points(
            min_latitude_deg=min_latitude_deg,
            min_longitude_deg=
            metadata_dict[radar_utils.NW_GRID_POINT_LNG_COLUMN],
            lat_spacing_deg=metadata_dict[radar_utils.LAT_SPACING_COLUMN],
            lng_spacing_deg=metadata_dict[radar_utils.LNG_SPACING_COLUMN],
            num_rows=metadata_dict[radar_utils.NUM_LAT_COLUMN],
            num_columns=metadata_dict[radar_utils.NUM_LNG_COLUMN])
    )

    full_matrix = _convert(
        sparse_grid_table=copy.deepcopy(sparse_grid_table),
        field_name=metadata_dict[radar_utils.FIELD_NAME_COLUMN],
        num_grid_rows=metadata_dict[radar_utils.NUM_LAT_COLUMN],
        num_grid_columns=metadata_dict[radar_utils.NUM_LNG_COLUMN],
        ignore_if_below=ignore_if_below)

    return (
        full_matrix, unique_grid_point_lat_deg[::-1], unique_grid_point_lng_deg
    )
