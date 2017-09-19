"""Converts MYRORSS* data from sparse grid to full grid.

* MYRORSS = Multi-year Reanalysis of Remotely Sensed Storms
"""

import copy
import numpy
from gewittergefahr.gg_io import myrorss_io

# TODO(thunderhoser): add error-checking to all methods.
# TODO(thunderhoser): replace main method with named high-level method.

MIN_CENTER_LAT_COLUMN = 'min_center_lat_deg'
MAX_CENTER_LAT_COLUMN = 'max_center_lat_deg'
MIN_CENTER_LNG_COLUMN = 'min_center_lng_deg'
MAX_CENTER_LNG_COLUMN = 'max_center_lng_deg'


def _get_bounding_box_of_grid_points(nw_grid_point_lat_deg=None,
                                     nw_grid_point_lng_deg=None,
                                     lat_spacing_deg=None, lng_spacing_deg=None,
                                     num_lat_in_grid=None,
                                     num_lng_in_grid=None):
    """Determines bounding box of grid points (centers of grid cells).

    :param nw_grid_point_lat_deg: Latitude (deg N) at center of
        northwesternmost grid point.
    :param nw_grid_point_lng_deg: Longitude (deg E) at center of
        northwesternmost grid point.
    :param lat_spacing_deg: Spacing (deg N) between adjacent grid rows.
    :param lng_spacing_deg: Spacing (deg E) between adjacent grid columns.
    :param num_lat_in_grid: Number of grid rows (unique latitudes).
    :param num_lng_in_grid: Number of grid columns (unique longitudes).
    :return: bounding_box_dict: Dictionary with the following keys.
    bounding_box_dict.min_center_lat_deg: Minimum grid-point latitude (deg N).
    bounding_box_dict.max_center_lat_deg: Maximum grid-point latitude (deg N).
    bounding_box_dict.min_center_lng_deg: Minimum grid-point longitude (deg E).
    bounding_box_dict.max_center_lng_deg: Maximum grid-point longitude (deg E).
    """

    max_center_lat_deg = copy.deepcopy(nw_grid_point_lat_deg)
    min_center_lng_deg = myrorss_io.convert_lng_positive_in_west(
        nw_grid_point_lng_deg)

    min_center_lat_deg = max_center_lat_deg - (
        num_lat_in_grid - 1) * lat_spacing_deg
    max_center_lng_deg = myrorss_io.convert_lng_positive_in_west(
        min_center_lng_deg + (num_lng_in_grid - 1) * lng_spacing_deg)

    return {'min_center_lat_deg': min_center_lat_deg,
            'max_center_lat_deg': max_center_lat_deg,
            'min_center_lng_deg': min_center_lng_deg,
            'max_center_lng_deg': max_center_lng_deg}


def _generate_grid_points(bounding_box_dict, num_lat_in_grid=None,
                          num_lng_in_grid=None):
    """Determines grid points (centers of grid cells).

    :param bounding_box_dict: Dictionary created by
        get_bounding_box_of_grid_points.
    :param num_lat_in_grid: Number of grid rows (unique latitudes).
    :param num_lng_in_grid: Number of grid columns (unique longitudes).
    :return: unique_center_lat_deg: 1-D numpy array of unique grid-point
        latitudes (deg N).
    :return: unique_center_lng_deg: 1-D numpy array of unique grid-point
        longitudes (deg E).
    """

    unique_center_lat_deg = numpy.linspace(
        bounding_box_dict['min_center_lat_deg'],
        bounding_box_dict['max_center_lat_deg'], num=num_lat_in_grid)

    unique_center_lng_deg = numpy.linspace(
        bounding_box_dict['min_center_lng_deg'],
        bounding_box_dict['max_center_lng_deg'], num=num_lng_in_grid)

    return unique_center_lat_deg, myrorss_io.convert_lng_positive_in_west(
        unique_center_lng_deg)


def _generate_grid_cell_edges(bounding_box_dict, lat_spacing_deg=None,
                              lng_spacing_deg=None, num_lat_in_grid=None,
                              num_lng_in_grid=None):
    """Determines edges of grid cells.

    :param bounding_box_dict: Dictionary created by
        get_bounding_box_of_grid_points.
    :param lat_spacing_deg: Spacing (deg N) between adjacent grid rows.
    :param lng_spacing_deg: Spacing (deg E) between adjacent grid columns.
    :param num_lat_in_grid: Number of grid rows (unique grid-point latitudes).
    :param num_lng_in_grid: Number of grid columns (unique grid-point
        longitudes).
    :return: unique_edge_lat_deg: 1-D numpy array of unique edge latitudes
        (deg N).
    :return: unique_edge_lng_deg: 1-D numpy array of unique edge longitudes
        (deg E).
    """

    min_edge_lat_deg = bounding_box_dict[
        'min_center_lat_deg'] - lat_spacing_deg / 2
    max_edge_lat_deg = bounding_box_dict[
        'max_center_lat_deg'] + lat_spacing_deg / 2
    unique_edge_lat_deg = numpy.linspace(min_edge_lat_deg, max_edge_lat_deg,
                                         num=num_lat_in_grid + 1)

    min_edge_lng_deg = bounding_box_dict[
        'min_center_lng_deg'] - lng_spacing_deg / 2
    max_edge_lng_deg = bounding_box_dict[
        'max_center_lng_deg'] + lng_spacing_deg / 2
    unique_edge_lng_deg = numpy.linspace(min_edge_lng_deg, max_edge_lng_deg,
                                         num=num_lng_in_grid + 1)

    return unique_edge_lat_deg, myrorss_io.convert_lng_positive_in_west(
        unique_edge_lng_deg)


def grid_vectors_to_matrices(unique_latitudes_deg, unique_longitudes_deg):
    """Converts grid coordinates from unique vectors to matrices.

    Coordinates may be either grid points or grid-cell edges.

    :param unique_latitudes_deg: 1-D numpy array of unique latitudes (deg N).
    :param unique_longitudes_deg: 1-D numpy array of unique longitudes (deg E).
    :return: latitude_matrix_deg: 2-D numpy array of latitudes (deg N), where
        each column is the same.  If unique_latitudes_deg is increasing
        (decreasing), latitudes in latitude_matrix_deg increase (decrease)
        while going down a column.
    :return: longitude_matrix_deg: 2-D numpy array of longitudes (deg E), where
        each row is the same.  If unique_longitudes_deg is increasing
        (decreasing), longitudes in longitude_matrix_deg increase (decrease)
        while going right across a row.
    """

    num_unique_lat = len(unique_latitudes_deg)
    num_unique_lng = len(unique_longitudes_deg)

    latitude_matrix_deg = numpy.full((num_unique_lat, num_unique_lng),
                                     numpy.nan)
    longitude_matrix_deg = numpy.full((num_unique_lat, num_unique_lng),
                                      numpy.nan)

    for i in range(num_unique_lat):
        longitude_matrix_deg[i, :] = unique_longitudes_deg

    for j in range(num_unique_lng):
        latitude_matrix_deg[:, j] = unique_latitudes_deg

    return latitude_matrix_deg, longitude_matrix_deg


def sparse_to_full_grid(sparse_grid_table, var_name=None, num_lat_in_grid=None,
                        num_lng_in_grid=None):
    """Converts MYRORSS data from sparse to full grid.

    M = number of grid-point latitudes
    N = number of grid-point longitudes

    :param sparse_grid_table: pandas DataFrame in format produced by
        `myrorss_io.read_sparse_grid_from_netcdf`.
    :param var_name: Name of radar variable.
    :param num_lat_in_grid: Number of grid rows (unique latitudes).
    :param num_lng_in_grid: Number of grid columns (unique longitudes).
    :return: data_matrix: M-by-N numpy array with values of radar variable.
    """

    data_start_indices = (
        numpy.ravel_multi_index((sparse_grid_table[
                                     myrorss_io.GRID_ROW_COLUMN].values,
                                 sparse_grid_table[
                                     myrorss_io.GRID_COLUMN_COLUMN].values),
                                (num_lat_in_grid, num_lng_in_grid)))

    data_end_indices = (data_start_indices + sparse_grid_table[
        myrorss_io.NUM_GRID_CELL_COLUMN].values - 1)

    num_data_runs = len(data_start_indices)
    num_data_values = numpy.sum(
        sparse_grid_table[myrorss_io.NUM_GRID_CELL_COLUMN].values)

    data_indices = numpy.full(num_data_values, numpy.nan, dtype=int)
    data_values = numpy.full(num_data_values, numpy.nan)
    num_values_added = 0

    for i in range(num_data_runs):
        these_data_indices = range(data_start_indices[i],
                                   data_end_indices[i] + 1)
        this_num_values = len(these_data_indices)

        these_array_indices = range(num_values_added,
                                    num_values_added + this_num_values)
        num_values_added += this_num_values

        data_indices[these_array_indices] = these_data_indices
        data_values[these_array_indices] = sparse_grid_table[var_name].values[i]

    data_matrix = numpy.full(num_lat_in_grid * num_lng_in_grid, numpy.nan)
    data_matrix[data_indices] = data_values
    return numpy.reshape(data_matrix, (num_lat_in_grid, num_lng_in_grid))


def sparse_to_full_grid_wrapper(sparse_grid_table, metadata_dict):
    """Converts MYRORSS data from sparse to full grid.

    The grid should contain one variable at one elevation and one time step
    (e.g., 1-km reflectivity at 1235 UTC).

    M = number of grid-point latitudes
    N = number of grid-point longitudes

    :param sparse_grid_table: pandas DataFrame in format produced by
        `myrorss_io.read_sparse_grid_from_netcdf`.
    :param metadata_dict: Dictionary in format produced by
        `myrorss_io.read_metadata_from_netcdf`.
    :return: data_matrix: M-by-N numpy array with values of radar variable.
    :return: unique_center_lat_deg: length-M numpy array of unique latitudes
        (deg north).  If unique_center_lat_deg is increasing (decreasing),
        latitudes in data_matrix increase (decrease) while going down a column.
    :return: unique_center_lng_deg: length-N numpy array of unique longitudes
        (deg east).  If unique_center_lng_deg is increasing (decreasing),
        longitudes in data_matrix increase (decrease) while going right across a
        row.
    """

    bounding_box_dict = _get_bounding_box_of_grid_points(
        nw_grid_point_lat_deg=metadata_dict[
            myrorss_io.NW_GRID_POINT_LAT_COLUMN],
        nw_grid_point_lng_deg=metadata_dict[
            myrorss_io.NW_GRID_POINT_LNG_COLUMN],
        lat_spacing_deg=metadata_dict[myrorss_io.LAT_SPACING_COLUMN],
        lng_spacing_deg=metadata_dict[myrorss_io.LNG_SPACING_COLUMN],
        num_lat_in_grid=metadata_dict[myrorss_io.NUM_LAT_COLUMN],
        num_lng_in_grid=metadata_dict[myrorss_io.NUM_LNG_COLUMN])

    (unique_center_lat_deg, unique_center_lng_deg) = _generate_grid_points(
        bounding_box_dict,
        num_lat_in_grid=metadata_dict[myrorss_io.NUM_LAT_COLUMN],
        num_lng_in_grid=metadata_dict[myrorss_io.NUM_LNG_COLUMN])

    data_matrix = sparse_to_full_grid(sparse_grid_table, var_name=metadata_dict[
        myrorss_io.VAR_NAME_COLUMN], num_lat_in_grid=metadata_dict[
        myrorss_io.NUM_LAT_COLUMN], num_lng_in_grid=metadata_dict[
        myrorss_io.NUM_LNG_COLUMN])

    return data_matrix, unique_center_lat_deg, unique_center_lng_deg
