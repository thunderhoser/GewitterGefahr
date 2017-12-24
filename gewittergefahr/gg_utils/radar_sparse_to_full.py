"""Converts radar data from sparse to full grid.

Radar data may come from either MYRORSS (Multi-year Reanalysis of Remotely
Sensed Storms) or MRMS (Multi-radar Multi-sensor).
"""

import numpy
from gewittergefahr.gg_io import radar_io
from gewittergefahr.gg_utils import grids

MIN_CENTER_LAT_COLUMN = 'min_center_lat_deg'
MAX_CENTER_LAT_COLUMN = 'max_center_lat_deg'
MIN_CENTER_LNG_COLUMN = 'min_center_lng_deg'
MAX_CENTER_LNG_COLUMN = 'max_center_lng_deg'


def _convert(sparse_grid_table, field_name, num_grid_rows, num_grid_columns,
             ignore_if_below=None):
    """Converts data from sparse to full grid.

    M = number of rows (unique grid-point latitudes)
    N = number of columns (unique grid-point longitudes)

    :param sparse_grid_table: pandas DataFrame created by
        `radar_io.read_data_from_sparse_grid_file`.
    :param field_name: Name of radar field.  This should also be a column name
        in `sparse_grid_table`.
    :param num_grid_rows: Number of rows in grid.
    :param num_grid_columns: Number of columns in grid.
    :param ignore_if_below: This method will ignore values of `field_name` <
        `ignore_if_below`.  If None, this method will consider all values.
    :return: full_matrix: M-by-N numpy array of radar values.
    """

    if ignore_if_below is None:
        num_sparse_values = len(sparse_grid_table.index)
        sparse_indices_to_consider = numpy.linspace(
            0, num_sparse_values - 1, num=num_sparse_values, dtype=int)
    else:
        sparse_indices_to_consider = numpy.where(
            sparse_grid_table[field_name].values >= ignore_if_below)[0]

    new_sparse_grid_table = sparse_grid_table.iloc[sparse_indices_to_consider]
    data_start_indices = numpy.ravel_multi_index(
        (new_sparse_grid_table[radar_io.GRID_ROW_COLUMN].values,
         new_sparse_grid_table[radar_io.GRID_COLUMN_COLUMN].values),
        (num_grid_rows, num_grid_columns))
    data_end_indices = (data_start_indices + new_sparse_grid_table[
        radar_io.NUM_GRID_CELL_COLUMN].values - 1)

    num_data_runs = len(data_start_indices)
    num_data_values = numpy.sum(
        new_sparse_grid_table[radar_io.NUM_GRID_CELL_COLUMN].values).astype(int)

    data_indices = numpy.full(num_data_values, numpy.nan, dtype=int)
    data_values = numpy.full(num_data_values, numpy.nan)
    num_values_added = 0

    for i in range(num_data_runs):
        these_data_indices = range(
            data_start_indices[i], data_end_indices[i] + 1)
        this_num_values = len(these_data_indices)

        these_array_indices = range(
            num_values_added, num_values_added + this_num_values)
        num_values_added += this_num_values

        data_indices[these_array_indices] = these_data_indices
        data_values[these_array_indices] = new_sparse_grid_table[
            field_name].values[i]

    full_matrix = numpy.full(num_grid_rows * num_grid_columns, numpy.nan)
    full_matrix[data_indices] = data_values
    return numpy.reshape(full_matrix, (num_grid_rows, num_grid_columns))


def sparse_to_full_grid(sparse_grid_table, metadata_dict, ignore_if_below=None):
    """Converts data from sparse to full grid (public wrapper for _convert).

    M = number of rows (unique grid-point latitudes)
    N = number of columns (unique grid-point longitudes)

    :param sparse_grid_table: pandas DataFrame created by
        `radar_io.read_data_from_sparse_grid_file`.
    :param metadata_dict: Dictionary created by
        `radar_io.read_metadata_from_raw_file`.
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

    min_latitude_deg = metadata_dict[radar_io.NW_GRID_POINT_LAT_COLUMN] - (
        metadata_dict[radar_io.LAT_SPACING_COLUMN] * (
            metadata_dict[radar_io.NUM_LAT_COLUMN] - 1))

    (unique_grid_point_lat_deg, unique_grid_point_lng_deg) = (
        grids.get_latlng_grid_points(
            min_latitude_deg=min_latitude_deg,
            min_longitude_deg=metadata_dict[radar_io.NW_GRID_POINT_LNG_COLUMN],
            lat_spacing_deg=metadata_dict[radar_io.LAT_SPACING_COLUMN],
            lng_spacing_deg=metadata_dict[radar_io.LNG_SPACING_COLUMN],
            num_rows=metadata_dict[radar_io.NUM_LAT_COLUMN],
            num_columns=metadata_dict[radar_io.NUM_LNG_COLUMN]))

    full_matrix = _convert(
        sparse_grid_table, field_name=metadata_dict[radar_io.FIELD_NAME_COLUMN],
        num_grid_rows=metadata_dict[radar_io.NUM_LAT_COLUMN],
        num_grid_columns=metadata_dict[radar_io.NUM_LNG_COLUMN],
        ignore_if_below=ignore_if_below)

    return (
        full_matrix, unique_grid_point_lat_deg[::-1], unique_grid_point_lng_deg)
