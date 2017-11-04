"""Converts radar data from sparse to full grid.

Radar data may come from either MYRORSS (Multi-year Reanalysis of Remotely
Sensed Storms) or MRMS (Multi-radar Multi-sensor).
"""

import numpy
from gewittergefahr.gg_io import radar_io
from gewittergefahr.gg_utils import grids

# TODO(thunderhoser): allow sparse-to-full operation for subset of full grid.

MIN_CENTER_LAT_COLUMN = 'min_center_lat_deg'
MAX_CENTER_LAT_COLUMN = 'max_center_lat_deg'
MIN_CENTER_LNG_COLUMN = 'min_center_lng_deg'
MAX_CENTER_LNG_COLUMN = 'max_center_lng_deg'


def _convert(sparse_grid_table, field_name=None, num_grid_rows=None,
             num_grid_columns=None):
    """Converts data from sparse to full grid.

    M = number of rows (unique grid-point latitudes)
    N = number of columns (unique grid-point longitudes)

    :param sparse_grid_table: pandas DataFrame created by
        `radar_io.read_data_from_sparse_grid_file`.
    :param field_name: Name of radar field (also column name in
        sparse_grid_table).
    :param num_grid_rows: Number of unique grid-point latitudes.
    :param num_grid_columns: Number of unique grid-point longitudes.
    :return: full_matrix: M-by-N numpy array of radar values.
    """

    data_start_indices = numpy.ravel_multi_index(
        (sparse_grid_table[radar_io.GRID_ROW_COLUMN].values,
         sparse_grid_table[radar_io.GRID_COLUMN_COLUMN].values),
        (num_grid_rows, num_grid_columns))

    data_end_indices = (data_start_indices + sparse_grid_table[
        radar_io.NUM_GRID_CELL_COLUMN].values - 1)

    num_data_runs = len(data_start_indices)
    num_data_values = numpy.sum(
        sparse_grid_table[radar_io.NUM_GRID_CELL_COLUMN].values).astype(int)

    data_indices = numpy.full(num_data_values, numpy.nan, dtype=int)
    data_values = numpy.full(num_data_values, numpy.nan)
    num_values_added = 0

    for i in range(num_data_runs):
        these_data_indices = range(data_start_indices[i],
                                   data_end_indices[i] + 1)
        this_num_values = len(these_data_indices)

        these_array_indices = range(
            num_values_added, num_values_added + this_num_values)
        num_values_added += this_num_values

        data_indices[these_array_indices] = these_data_indices
        data_values[these_array_indices] = sparse_grid_table[
            field_name].values[i]

    full_matrix = numpy.full(num_grid_rows * num_grid_columns, numpy.nan)
    full_matrix[data_indices] = data_values
    return numpy.reshape(full_matrix, (num_grid_rows, num_grid_columns))


def sparse_to_full_grid(sparse_grid_table, metadata_dict):
    """Converts data from sparse to full grid (wrapper method for _convert).

    M = number of rows (unique grid-point latitudes)
    N = number of columns (unique grid-point longitudes)

    :param sparse_grid_table: pandas DataFrame created by
        `radar_io.read_data_from_sparse_grid_file`.
    :param metadata_dict: Dictionary created by
        `radar_io.read_metadata_from_raw_file`.
    :return: full_matrix: M-by-N numpy array of radar values.
    :return: unique_grid_point_lat_deg: length-M numpy array of grid-point
        latitudes (deg N).  If array is increasing (decreasing), latitude
        increases (decreases) while traveling down the columns of full_matrix.
    :return: unique_grid_point_lng_deg: length-N numpy array of grid-point
        longitudes (deg E).  If array is increasing (decreasing), longitude
        increases (decreases) while traveling right across the rows of
        full_matrix.
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
        num_grid_columns=metadata_dict[radar_io.NUM_LNG_COLUMN])

    return (numpy.flipud(full_matrix), unique_grid_point_lat_deg,
            unique_grid_point_lng_deg)
