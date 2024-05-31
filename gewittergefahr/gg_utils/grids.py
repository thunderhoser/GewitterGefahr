"""Processing methods for gridded data.

DEFINITIONS

"Grid point" = center of grid cell (as opposed to edges of the grid cell).
"""

import pickle
import os.path
import numpy
from gewittergefahr.gg_utils import general_utils
from gewittergefahr.gg_utils import longitude_conversion as lng_conversion
from gewittergefahr.gg_utils import projections
from gewittergefahr.gg_utils import number_rounding
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.gg_utils import error_checking

TOLERANCE = 1e-6

DEGREES_LAT_TO_METRES = 60 * 1852
DEGREES_TO_RADIANS = numpy.pi / 180

DUMMY_LATITUDE_SPACING_DEG = 0.01
DUMMY_LONGITUDE_SPACING_DEG = 0.01

X_COORD_MATRIX_KEY = 'x_coord_matrix_metres'
Y_COORD_MATRIX_KEY = 'y_coord_matrix_metres'
X_COORDS_KEY = 'x_coords_metres'
Y_COORDS_KEY = 'y_coords_metres'
PROJECTION_KEY = 'projection_object'

EQUIDISTANT_METADATA_KEYS = [X_COORDS_KEY, Y_COORDS_KEY, PROJECTION_KEY]


def _check_input_events(
        event_x_coords_metres, event_y_coords_metres, integer_event_ids):
    """Checks inputs to `count_events_on_equidistant_grid`.

    :param event_x_coords_metres: See doc for
        `count_events_on_equidistant_grid`.
    :param event_y_coords_metres: Same.
    :param integer_event_ids: Same.
    """

    error_checking.assert_is_numpy_array_without_nan(event_x_coords_metres)
    error_checking.assert_is_numpy_array(
        event_x_coords_metres, num_dimensions=1)
    num_events = len(event_x_coords_metres)

    error_checking.assert_is_numpy_array_without_nan(event_y_coords_metres)
    error_checking.assert_is_numpy_array(
        event_y_coords_metres, exact_dimensions=numpy.array([num_events]))

    if integer_event_ids is not None:
        error_checking.assert_is_integer_numpy_array(integer_event_ids)
        error_checking.assert_is_numpy_array(
            integer_event_ids, exact_dimensions=numpy.array([num_events]))


def get_xy_grid_points(x_min_metres=None, y_min_metres=None,
                       x_spacing_metres=None, y_spacing_metres=None,
                       num_rows=None, num_columns=None):
    """Generates unique x- and y-coords of grid points in regular x-y grid.

    M = number of rows in grid
    N = number of columns in grid

    :param x_min_metres: Minimum x-coordinate over all grid points.
    :param y_min_metres: Minimum y-coordinate over all grid points.
    :param x_spacing_metres: Spacing between adjacent grid points in x-
        direction.  Alternate interpretation: length of each grid cell in x-
        direction.
    :param y_spacing_metres: Spacing between adjacent grid points in y-
        direction.  Alternate interpretation: length of each grid cell in y-
        direction.
    :param num_rows: Number of rows (unique grid-point y-values) in grid.
    :param num_columns: Number of columns (unique grid-point x-values) in grid.
    :return: grid_point_x_metres: length-N numpy array with x-coordinates of
        grid points.
    :return: grid_point_y_metres: length-M numpy array with y-coordinates of
        grid points.
    """

    error_checking.assert_is_not_nan(x_min_metres)
    error_checking.assert_is_not_nan(y_min_metres)
    error_checking.assert_is_greater(x_spacing_metres, 0.)
    error_checking.assert_is_greater(y_spacing_metres, 0.)
    error_checking.assert_is_integer(num_rows)
    error_checking.assert_is_greater(num_rows, 0)
    error_checking.assert_is_integer(num_columns)
    error_checking.assert_is_greater(num_columns, 0)

    x_max_metres = x_min_metres + (num_columns - 1) * x_spacing_metres
    y_max_metres = y_min_metres + (num_rows - 1) * y_spacing_metres

    grid_point_x_metres = numpy.linspace(x_min_metres, x_max_metres,
                                         num=num_columns)
    grid_point_y_metres = numpy.linspace(y_min_metres, y_max_metres,
                                         num=num_rows)

    return grid_point_x_metres, grid_point_y_metres


def get_latlng_grid_points(min_latitude_deg=None, min_longitude_deg=None,
                           lat_spacing_deg=None, lng_spacing_deg=None,
                           num_rows=None, num_columns=None):
    """Generates unique lat and long of grid points in regular lat-long grid.

    M = number of rows in grid
    N = number of columns in grid

    :param min_latitude_deg: Minimum latitude over all grid points (deg N).
    :param min_longitude_deg: Minimum longitude over all grid points (deg E).
    :param lat_spacing_deg: Meridional spacing between adjacent grid points.
        Alternate interpretation: length of each grid cell in N-S direction.
    :param lng_spacing_deg: Zonal spacing between adjacent grid points.
        Alternate interpretation: length of each grid cell in E-W direction.
    :param num_rows: Number of rows (unique grid-point latitudes) in grid.
    :param num_columns: Number of columns (unique grid-point longitudes) in
        grid.
    :return: grid_point_latitudes_deg: length-M numpy array with latitudes of
        grid points (deg N).
    :return: grid_point_longitudes_deg: length-N numpy array with longitudes of
        grid points (deg E).
    """

    error_checking.assert_is_valid_latitude(min_latitude_deg)
    min_longitude_deg = lng_conversion.convert_lng_positive_in_west(
        min_longitude_deg, allow_nan=False)
    error_checking.assert_is_greater(lat_spacing_deg, 0.)
    error_checking.assert_is_greater(lng_spacing_deg, 0.)
    error_checking.assert_is_integer(num_rows)
    error_checking.assert_is_greater(num_rows, 0)
    error_checking.assert_is_integer(num_columns)
    error_checking.assert_is_greater(num_columns, 0)

    max_latitude_deg = min_latitude_deg + (num_rows - 1) * lat_spacing_deg
    max_longitude_deg = min_longitude_deg + (num_columns - 1) * lng_spacing_deg

    grid_point_latitudes_deg = numpy.linspace(min_latitude_deg,
                                              max_latitude_deg, num=num_rows)
    grid_point_longitudes_deg = numpy.linspace(min_longitude_deg,
                                               max_longitude_deg,
                                               num=num_columns)

    return grid_point_latitudes_deg, grid_point_longitudes_deg


def get_xy_grid_cell_edges(x_min_metres=None, y_min_metres=None,
                           x_spacing_metres=None, y_spacing_metres=None,
                           num_rows=None, num_columns=None):
    """Generates unique x- and y-coords of grid-cell edges in regular x-y grid.

    M = number of rows in grid
    N = number of columns in grid

    :param x_min_metres: See documentation for get_xy_grid_points.
    :param y_min_metres: See documentation for get_xy_grid_points.
    :param x_spacing_metres: See documentation for get_xy_grid_points.
    :param y_spacing_metres: See documentation for get_xy_grid_points.
    :param num_rows: See documentation for get_xy_grid_points.
    :param num_columns: See documentation for get_xy_grid_points.
    :return: grid_cell_edge_x_metres: length-(N + 1) numpy array with x-
        coordinates of grid points.
    :return: grid_cell_edge_y_metres: length-(M + 1) numpy array with y-
        coordinates of grid points.
    """

    grid_point_x_metres, grid_point_y_metres = get_xy_grid_points(
        x_min_metres=x_min_metres, y_min_metres=y_min_metres,
        x_spacing_metres=x_spacing_metres, y_spacing_metres=y_spacing_metres,
        num_rows=num_rows, num_columns=num_columns)

    grid_cell_edge_x_metres = numpy.concatenate((
        grid_point_x_metres - x_spacing_metres / 2,
        grid_point_x_metres[[-1]] + x_spacing_metres / 2))
    grid_cell_edge_y_metres = numpy.concatenate((
        grid_point_y_metres - y_spacing_metres / 2,
        grid_point_y_metres[[-1]] + y_spacing_metres / 2))

    return grid_cell_edge_x_metres, grid_cell_edge_y_metres


def get_latlng_grid_cell_edges(min_latitude_deg=None, min_longitude_deg=None,
                               lat_spacing_deg=None, lng_spacing_deg=None,
                               num_rows=None, num_columns=None):
    """Generates unique lat and lng of grid-cell edges in regular lat-lng grid.

    M = number of rows in grid
    N = number of columns in grid

    :param min_latitude_deg: See documentation for get_latlng_grid_points.
    :param min_longitude_deg: See documentation for get_latlng_grid_points.
    :param lat_spacing_deg: See documentation for get_latlng_grid_points.
    :param lng_spacing_deg: See documentation for get_latlng_grid_points.
    :param num_rows: See documentation for get_latlng_grid_points.
    :param num_columns: See documentation for get_latlng_grid_points.
    :return: grid_cell_edge_latitudes_deg: length-(M + 1) numpy array with
        latitudes of grid-cell edges (deg N).
    :return: grid_cell_edge_longitudes_deg: length-(N + 1) numpy array with
        longitudes of grid-cell edges (deg E).
    """

    (grid_point_latitudes_deg,
     grid_point_longitudes_deg) = get_latlng_grid_points(
         min_latitude_deg=min_latitude_deg, min_longitude_deg=min_longitude_deg,
         lat_spacing_deg=lat_spacing_deg, lng_spacing_deg=lng_spacing_deg,
         num_rows=num_rows, num_columns=num_columns)

    grid_cell_edge_latitudes_deg = numpy.concatenate((
        grid_point_latitudes_deg - lat_spacing_deg / 2,
        grid_point_latitudes_deg[[-1]] + lat_spacing_deg / 2))
    grid_cell_edge_longitudes_deg = numpy.concatenate((
        grid_point_longitudes_deg - lng_spacing_deg / 2,
        grid_point_longitudes_deg[[-1]] + lng_spacing_deg / 2))

    return grid_cell_edge_latitudes_deg, grid_cell_edge_longitudes_deg


def xy_vectors_to_matrices(x_unique_metres, y_unique_metres):
    """For regular x-y grid, converts vectors of x- and y-coords to matrices.

    This method works for coordinates of either grid points or grid-cell edges.

    M = number of rows in grid
    N = number of columns in grid

    :param x_unique_metres: length-N numpy array with x-coordinates of either
        grid points or grid-cell edges.
    :param y_unique_metres: length-M numpy array with y-coordinates of either
        grid points or grid-cell edges.
    :return: x_matrix_metres: M-by-N numpy array, where x_matrix_metres[*, j] =
        x_unique_metres[j].  Each row in this matrix is the same.
    :return: y_matrix_metres: M-by-N numpy array, where y_matrix_metres[i, *] =
        y_unique_metres[i].  Each column in this matrix is the same.
    """

    error_checking.assert_is_numpy_array_without_nan(x_unique_metres)
    error_checking.assert_is_numpy_array(x_unique_metres, num_dimensions=1)
    error_checking.assert_is_numpy_array_without_nan(y_unique_metres)
    error_checking.assert_is_numpy_array(y_unique_metres, num_dimensions=1)

    return numpy.meshgrid(x_unique_metres, y_unique_metres)


def latlng_vectors_to_matrices(unique_latitudes_deg, unique_longitudes_deg):
    """Converts vectors of lat and long coordinates to matrices.

    This method works only for a regular lat-long grid.  Works for coordinates
    of either grid points or grid-cell edges.

    M = number of rows in grid
    N = number of columns in grid

    :param unique_latitudes_deg: length-M numpy array with latitudes (deg N) of
        either grid points or grid-cell edges.
    :param unique_longitudes_deg: length-N numpy array with longitudes (deg E)
        of either grid points or grid-cell edges.
    :return: latitude_matrix_deg: M-by-N numpy array, where
        latitude_matrix_deg[i, *] = unique_latitudes_deg[i].  Each column in
        this matrix is the same.
    :return: longitude_matrix_deg: M-by-N numpy array, where
        longitude_matrix_deg[*, j] = unique_longitudes_deg[j].  Each row in this
        matrix is the same.
    """

    error_checking.assert_is_valid_lat_numpy_array(unique_latitudes_deg)
    error_checking.assert_is_numpy_array(unique_latitudes_deg, num_dimensions=1)
    error_checking.assert_is_numpy_array(unique_longitudes_deg,
                                         num_dimensions=1)
    unique_longitudes_deg = lng_conversion.convert_lng_positive_in_west(
        unique_longitudes_deg, allow_nan=False)

    (longitude_matrix_deg, latitude_matrix_deg) = numpy.meshgrid(
        unique_longitudes_deg, unique_latitudes_deg)
    return latitude_matrix_deg, longitude_matrix_deg


def xy_field_grid_points_to_edges(field_matrix=None, x_min_metres=None,
                                  y_min_metres=None, x_spacing_metres=None,
                                  y_spacing_metres=None):
    """Re-references x-y field from grid points to edges.

    M = number of rows (unique grid-point x-coordinates)
    N = number of columns (unique grid-point y-coordinates)

    :param field_matrix: M-by-N numpy array with values of some variable
        (examples: temperature, radar reflectivity, etc.).  y should increase
        while traveling down a column, and x should increase while traveling
        right across a row.
    :param x_min_metres: Minimum x-coordinate over all grid points.
    :param y_min_metres: Minimum y-coordinate over all grid points.
    :param x_spacing_metres: Spacing between adjacent grid points in x-
        direction.
    :param y_spacing_metres: Spacing between adjacent grid points in y-
        direction.
    :return: field_matrix: Same as input, except that dimensions are now (M + 1)
        by (N + 1).  The last row and last column contain only NaN's.
    :return: grid_cell_edge_x_metres: length-(N + 1) numpy array with x-
        coordinates of grid-cell edges.
    :return: grid_cell_edge_y_metres: length-(M + 1) numpy array with y-
        coordinates of grid-cell edges.
    """

    error_checking.assert_is_real_numpy_array(field_matrix)
    error_checking.assert_is_numpy_array(field_matrix, num_dimensions=2)

    num_rows = field_matrix.shape[0]
    num_columns = field_matrix.shape[1]

    grid_cell_edge_x_metres, grid_cell_edge_y_metres = get_xy_grid_cell_edges(
        x_min_metres=x_min_metres, y_min_metres=y_min_metres,
        x_spacing_metres=x_spacing_metres, y_spacing_metres=y_spacing_metres,
        num_rows=num_rows, num_columns=num_columns)

    nan_row = numpy.full((1, num_columns), numpy.nan)
    field_matrix = numpy.vstack((field_matrix, nan_row))

    nan_column = numpy.full((num_rows + 1, 1), numpy.nan)
    field_matrix = numpy.hstack((field_matrix, nan_column))

    return field_matrix, grid_cell_edge_x_metres, grid_cell_edge_y_metres


def latlng_field_grid_points_to_edges(
        field_matrix=None, min_latitude_deg=None, min_longitude_deg=None,
        lat_spacing_deg=None, lng_spacing_deg=None):
    """Re-references lat-long field from grid points to edges.

    M = number of rows (unique grid-point latitudes)
    N = number of columns (unique grid-point longitudes)

    :param field_matrix: M-by-N numpy array with values of some variable
        (examples: temperature, radar reflectivity, etc.).  Latitude should
        increase while traveling down a column, and longitude should increase
        while traveling right across a row.
    :param min_latitude_deg: See documentation for get_latlng_grid_points.
    :param min_longitude_deg: See documentation for get_latlng_grid_points.
    :param lat_spacing_deg: See documentation for get_latlng_grid_points.
    :param lng_spacing_deg: See documentation for get_latlng_grid_points.
    :param num_rows: See documentation for get_latlng_grid_points.
    :param num_columns: See documentation for get_latlng_grid_points.
    :return: field_matrix: Same as input, except that dimensions are now (M + 1)
        by (N + 1).  The last row and last column contain only NaN's.
    :return: grid_cell_edge_latitudes_deg: length-(M + 1) numpy array with
        latitudes of grid-cell edges (deg N).
    :return: grid_cell_edge_longitudes_deg: length-(N + 1) numpy array with
        longitudes of grid-cell edges (deg E).
    """

    error_checking.assert_is_real_numpy_array(field_matrix)
    error_checking.assert_is_numpy_array(field_matrix, num_dimensions=2)

    num_rows = field_matrix.shape[0]
    num_columns = field_matrix.shape[1]

    (grid_cell_edge_latitudes_deg,
     grid_cell_edge_longitudes_deg) = get_latlng_grid_cell_edges(
         min_latitude_deg=min_latitude_deg, min_longitude_deg=min_longitude_deg,
         lat_spacing_deg=lat_spacing_deg, lng_spacing_deg=lng_spacing_deg,
         num_rows=num_rows, num_columns=num_columns)

    nan_row = numpy.full((1, num_columns), numpy.nan)
    field_matrix = numpy.vstack((field_matrix, nan_row))

    nan_column = numpy.full((num_rows + 1, 1), numpy.nan)
    field_matrix = numpy.hstack((field_matrix, nan_column))

    return (field_matrix, grid_cell_edge_latitudes_deg,
            grid_cell_edge_longitudes_deg)


def extract_latlng_subgrid(
        data_matrix, grid_point_latitudes_deg, grid_point_longitudes_deg,
        center_latitude_deg, center_longitude_deg,
        max_distance_from_center_metres):
    """Extracts subset of lat-long grid, centered at a given point.

    M = number of rows (grid-point latitudes) in full grid
    N = number of columns (grid-point longitudes) in full grid
    m = number of rows (grid-point latitudes) in sub-grid
    n = number of columns (grid-point longitudes) in sub-grid

    :param data_matrix: M-by-N numpy array with data values.
    :param grid_point_latitudes_deg: length-M numpy array of grid-point
        latitudes (deg N).  grid_point_latitudes_deg[i] is the latitude
        everywhere in the [i]th row of data_matrix.
    :param grid_point_longitudes_deg: length-M numpy array of grid-point
        latitudes (deg E).  grid_point_longitudes_deg[j] is the longitude
        everywhere in the [j]th column of data_matrix.
    :param center_latitude_deg: Latitude (deg N) of target point, around which
        the sub-grid will be centered.
    :param center_longitude_deg: Longitude (deg E) of target point, around which
        the sub-grid will be centered.
    :param max_distance_from_center_metres: Max distance of any grid point from
        target point.  Values at more distant grid points will be changed to
        NaN.
    :return: subgrid_matrix: m-by-n numpy array with data values.
    :return: rows_in_full_grid: length-m numpy array.  If rows_in_full_grid[i]
        = k, the [i]th row in the subgrid = [k]th row in the full grid.
    """

    max_latitude_diff_deg = (
        max_distance_from_center_metres / DEGREES_LAT_TO_METRES
    )

    degrees_lng_to_metres = DEGREES_LAT_TO_METRES * numpy.cos(
        center_latitude_deg * DEGREES_TO_RADIANS
    )
    max_longitude_diff_deg = (
        max_distance_from_center_metres / degrees_lng_to_metres
    )

    min_latitude_deg = center_latitude_deg - max_latitude_diff_deg
    max_latitude_deg = center_latitude_deg + max_latitude_diff_deg
    min_longitude_deg = center_longitude_deg - max_longitude_diff_deg
    max_longitude_deg = center_longitude_deg + max_longitude_diff_deg

    valid_row_flags = numpy.logical_and(
        grid_point_latitudes_deg >= min_latitude_deg,
        grid_point_latitudes_deg <= max_latitude_deg)

    valid_row_indices = numpy.where(valid_row_flags)[0]
    min_valid_row_index = numpy.min(valid_row_indices)
    max_valid_row_index = numpy.max(valid_row_indices)

    valid_column_flags = numpy.logical_and(
        grid_point_longitudes_deg >= min_longitude_deg,
        grid_point_longitudes_deg <= max_longitude_deg)

    valid_column_indices = numpy.where(valid_column_flags)[0]
    min_valid_column_index = numpy.min(valid_column_indices)
    max_valid_column_index = numpy.max(valid_column_indices)

    subgrid_data_matrix = data_matrix[
        min_valid_row_index:(max_valid_row_index + 1),
        min_valid_column_index:(max_valid_column_index + 1)
    ]

    subgrid_lat_matrix_deg, subgrid_lng_matrix_deg = (
        latlng_vectors_to_matrices(
            unique_latitudes_deg=grid_point_latitudes_deg[valid_row_indices],
            unique_longitudes_deg=grid_point_longitudes_deg[
                valid_column_indices]
        )
    )

    lat_distance_matrix_metres = DEGREES_LAT_TO_METRES * (
        subgrid_lat_matrix_deg - center_latitude_deg
    )
    lng_distance_matrix_metres = degrees_lng_to_metres * (
        subgrid_lng_matrix_deg - center_longitude_deg
    )
    distance_matrix_metres = numpy.sqrt(
        lat_distance_matrix_metres ** 2 + lng_distance_matrix_metres ** 2
    )

    subgrid_data_matrix[
        distance_matrix_metres > max_distance_from_center_metres
    ] = numpy.nan

    return subgrid_data_matrix, valid_row_indices, valid_column_indices


def count_events_on_equidistant_grid(
        event_x_coords_metres, event_y_coords_metres,
        grid_point_x_coords_metres, grid_point_y_coords_metres,
        integer_event_ids=None, effective_radius_metres=None,
        event_weights=None):
    """Counts number of events in, or near, each cell of an equidistant grid.

    M = number of rows (unique grid-point y-coordinates)
    N = number of columns (unique grid-point x-coordinates)
    K = number of events

    :param event_x_coords_metres: length-K numpy array with x-coordinates of
        events.
    :param event_y_coords_metres: length-K numpy array with y-coordinates of
        events.
    :param grid_point_x_coords_metres: length-N numpy array with unique
        x-coordinates at grid points.
    :param grid_point_y_coords_metres: length-M numpy array with unique
        y-coordinates at grid points.
    :param integer_event_ids: length-K numpy array of event IDs (integers).
        Each event ID will be counted only once per grid cell.  If
        `event_ids is None`, will assume that each location in
        `event_x_coords_metres` and `event_y_coords_metres` comes from a unique
        event.
    :param effective_radius_metres: If None, will count the number of events
        inside each grid cell.  If specified, will count the number of events
        within `effective_radius_metres` of each grid point.
    :param event_weights: length-K numpy array of positive weights.  If None,
        each event will be given a weight of 1.0.
    :return: num_events_matrix:
        [if `effective_radius_metres is None`]
        M-by-N numpy array, where element [i, j] is the number of events in grid
        cell [i, j].

        [if `effective_radius_metres is not None`]
        M-by-N numpy array, where element [i, j] is the number of events within
        `effective_radius_metres` of grid point [i, j].

    :return: event_ids_by_grid_cell_dict: Dictionary, where key [i, j] is a 1-D
        numpy array of event IDs assigned to grid cell [i, j].  If
        `event_ids is None`, this will also be `None`.
    """

    _check_input_events(
        event_x_coords_metres=event_x_coords_metres,
        event_y_coords_metres=event_y_coords_metres,
        integer_event_ids=integer_event_ids
    )

    error_checking.assert_is_numpy_array_without_nan(grid_point_x_coords_metres)
    error_checking.assert_is_numpy_array(
        grid_point_x_coords_metres, num_dimensions=1
    )

    error_checking.assert_is_numpy_array_without_nan(grid_point_y_coords_metres)
    error_checking.assert_is_numpy_array(
        grid_point_y_coords_metres, num_dimensions=1
    )

    num_events = len(event_x_coords_metres)
    if event_weights is None:
        event_weights = numpy.full(num_events, 1.)

    error_checking.assert_is_numpy_array(
        event_weights, exact_dimensions=numpy.array([num_events], dtype=int)
    )
    error_checking.assert_is_greater_numpy_array(event_weights, 0.)

    if effective_radius_metres is not None:
        error_checking.assert_is_greater(effective_radius_metres, 0.)
        effective_radius_metres2 = effective_radius_metres ** 2

        grid_point_x_matrix_metres, grid_point_y_matrix_metres = (
            xy_vectors_to_matrices(
                x_unique_metres=grid_point_x_coords_metres,
                y_unique_metres=grid_point_y_coords_metres
            )
        )

    num_grid_rows = len(grid_point_y_coords_metres)
    num_grid_columns = len(grid_point_x_coords_metres)
    num_events_matrix = numpy.full(
        (num_grid_rows, num_grid_columns), 0, dtype=float
    )

    if integer_event_ids is None:
        event_ids_by_grid_cell_dict = None
    else:
        event_ids_by_grid_cell_dict = {}

        for i in range(num_grid_rows):
            for j in range(num_grid_columns):
                event_ids_by_grid_cell_dict[i, j] = []

    num_events = len(event_x_coords_metres)

    for k in range(num_events):
        if numpy.mod(k, 1000) == 0:
            print('Have assigned {0:d} of {1:d} events to grid cells...'.format(
                k, num_events
            ))

        if effective_radius_metres is None:
            _, this_row = general_utils.find_nearest_value(
                grid_point_y_coords_metres, event_y_coords_metres[k]
            )
            _, this_column = general_utils.find_nearest_value(
                grid_point_x_coords_metres, event_x_coords_metres[k]
            )

            if integer_event_ids is None:
                num_events_matrix[this_row, this_column] += event_weights[k]
            else:
                event_ids_by_grid_cell_dict[this_row, this_column].append(
                    integer_event_ids[k]
                )

            continue

        these_rows_to_try = numpy.where(
            numpy.absolute(
                grid_point_y_coords_metres - event_y_coords_metres[k]
            )
            <= effective_radius_metres
        )[0]

        these_columns_to_try = numpy.where(
            numpy.absolute(
                grid_point_x_coords_metres - event_x_coords_metres[k]
            )
            <= effective_radius_metres
        )[0]

        this_x_submatrix_metres = grid_point_x_matrix_metres[
            these_rows_to_try[:, None], these_columns_to_try
        ]
        this_y_submatrix_metres = grid_point_y_matrix_metres[
            these_rows_to_try[:, None], these_columns_to_try
        ]
        this_distance_submatrix_metres2 = (
            (this_x_submatrix_metres - event_x_coords_metres[k]) ** 2 +
            (this_y_submatrix_metres - event_y_coords_metres[k]) ** 2
        )

        these_subrows, these_subcolumns = numpy.where(
            this_distance_submatrix_metres2 <= effective_radius_metres2
        )
        these_rows = these_rows_to_try[these_subrows]
        these_columns = these_columns_to_try[these_subcolumns]

        if integer_event_ids is None:
            num_events_matrix[these_rows, these_columns] += event_weights[k]
        else:
            for m in range(len(these_rows)):
                event_ids_by_grid_cell_dict[
                    these_rows[m], these_columns[m]
                ].append(integer_event_ids[k])

    print('Have assigned all {0:d} events to grid cells!'.format(num_events))

    if integer_event_ids is not None:
        for i in range(num_grid_rows):
            for j in range(num_grid_columns):
                event_ids_by_grid_cell_dict[i, j] = numpy.array(
                    list(set(event_ids_by_grid_cell_dict[i, j])), dtype=int
                )
                num_events_matrix[i, j] = len(event_ids_by_grid_cell_dict[i, j])

    return num_events_matrix, event_ids_by_grid_cell_dict


def get_latlng_grid_points_in_radius(
        test_latitude_deg, test_longitude_deg, effective_radius_metres,
        grid_point_latitudes_deg=None, grid_point_longitudes_deg=None,
        grid_point_dict=None):
    """Finds lat-long grid points within radius of test point.

    One of the following sets of input args must be specified:

    - grid_point_latitudes_deg and grid_point_longitudes_deg
    - grid_point_dict

    M = number of rows (unique grid-point latitudes)
    N = number of columns (unique grid-point longitudes)
    K = number of grid points within radius of test point

    :param test_latitude_deg: Latitude (deg N) of test point.
    :param test_longitude_deg: Longitude (deg E) of test point.
    :param effective_radius_metres: Effective radius (will find all grid points
        within this radius of test point).
    :param grid_point_latitudes_deg: length-M numpy array with latitudes (deg N)
        of grid points.
    :param grid_point_longitudes_deg: length-N numpy array with longitudes
        (deg E) of grid points.
    :param grid_point_dict: Dictionary created by a previous run of this method
        (see output documentation).
    :return: rows_in_radius: length-K numpy array with row indices of grid
        points near test point.
    :return: columns_in_radius: Same but for columns.
    :return: grid_point_dict: Dictionary with the following keys.
    grid_point_dict['grid_point_x_matrix_metres']: M-by-N numpy array with
        x-coordinates of grid points.
    grid_point_dict['grid_point_y_matrix_metres']: M-by-N numpy array with
        y-coordinates of grid points.
    grid_point_dict['projection_object']: Instance of `pyproj.Proj`, which can
        be used to convert future test points from lat-long to x-y coordinates.
    """

    if grid_point_dict is None:
        (grid_point_lat_matrix_deg, grid_point_lng_matrix_deg
        ) = latlng_vectors_to_matrices(
            unique_latitudes_deg=grid_point_latitudes_deg,
            unique_longitudes_deg=grid_point_longitudes_deg)

        projection_object = projections.init_azimuthal_equidistant_projection(
            central_latitude_deg=numpy.mean(grid_point_latitudes_deg),
            central_longitude_deg=numpy.mean(grid_point_longitudes_deg))

        (grid_point_x_matrix_metres, grid_point_y_matrix_metres
        ) = projections.project_latlng_to_xy(
            latitudes_deg=grid_point_lat_matrix_deg,
            longitudes_deg=grid_point_lng_matrix_deg,
            projection_object=projection_object)

        grid_point_dict = {
            X_COORD_MATRIX_KEY: grid_point_x_matrix_metres,
            Y_COORD_MATRIX_KEY: grid_point_y_matrix_metres,
            PROJECTION_KEY: projection_object
        }

    error_checking.assert_is_valid_latitude(test_latitude_deg)
    error_checking.assert_is_geq(effective_radius_metres, 0.)
    test_longitude_deg = lng_conversion.convert_lng_positive_in_west(
        longitudes_deg=numpy.array([test_longitude_deg]), allow_nan=False)[0]

    (test_x_coords_metres, test_y_coords_metres
    ) = projections.project_latlng_to_xy(
        latitudes_deg=numpy.array([test_latitude_deg]),
        longitudes_deg=numpy.array([test_longitude_deg]),
        projection_object=grid_point_dict[PROJECTION_KEY])
    test_x_coord_metres = test_x_coords_metres[0]
    test_y_coord_metres = test_y_coords_metres[0]

    valid_x_flags = numpy.absolute(
        grid_point_dict[X_COORD_MATRIX_KEY] - test_x_coord_metres
    ) <= effective_radius_metres
    valid_y_flags = numpy.absolute(
        grid_point_dict[Y_COORD_MATRIX_KEY] - test_y_coord_metres
    ) <= effective_radius_metres
    rows_to_try, columns_to_try = numpy.where(numpy.logical_and(
        valid_x_flags, valid_y_flags))

    distances_to_try_metres = numpy.sqrt(
        (grid_point_dict[X_COORD_MATRIX_KEY][rows_to_try, columns_to_try] -
         test_x_coord_metres) ** 2 +
        (grid_point_dict[Y_COORD_MATRIX_KEY][rows_to_try, columns_to_try] -
         test_y_coord_metres) ** 2)
    valid_indices = numpy.where(
        distances_to_try_metres <= effective_radius_metres)[0]

    return (rows_to_try[valid_indices], columns_to_try[valid_indices],
            grid_point_dict)


def create_equidistant_grid(
        min_latitude_deg, max_latitude_deg, min_longitude_deg,
        max_longitude_deg, x_spacing_metres, y_spacing_metres, azimuthal=True):
    """Creates equidistant grid.

    M = number of rows
    N = number of columns

    :param min_latitude_deg: Minimum latitude (deg N) in grid.
    :param max_latitude_deg: Max latitude (deg N) in grid.
    :param min_longitude_deg: Minimum longitude (deg E) in grid.
    :param max_longitude_deg: Max longitude (deg E) in grid.
    :param x_spacing_metres: Spacing between grid points in adjacent columns.
    :param y_spacing_metres: Spacing between grid points in adjacent rows.
    :param azimuthal: Boolean flag.  If True, will create azimuthal equidistant
        grid.  If False, will create Lambert conformal grid.
    :return: grid_dict: Dictionary with the following keys.
    grid_dict['grid_point_x_coords_metres']: length-N numpy array with unique
        x-coordinates at grid points.
    grid_dict['grid_point_y_coords_metres']: length-M numpy array with unique
        y-coordinates at grid points.
    grid_dict['projection_object']: Instance of `pyproj.Proj` (used to convert
        between lat-long coordinates and the x-y coordinates of the grid).
    """

    # Check input args.
    error_checking.assert_is_valid_latitude(min_latitude_deg)
    error_checking.assert_is_valid_latitude(max_latitude_deg)
    error_checking.assert_is_greater(max_latitude_deg, min_latitude_deg)
    error_checking.assert_is_greater(x_spacing_metres, 0.)
    error_checking.assert_is_greater(y_spacing_metres, 0.)
    error_checking.assert_is_boolean(azimuthal)

    min_longitude_deg = lng_conversion.convert_lng_negative_in_west(
        min_longitude_deg, allow_nan=False)
    max_longitude_deg = lng_conversion.convert_lng_negative_in_west(
        max_longitude_deg, allow_nan=False)
    error_checking.assert_is_greater(max_longitude_deg, min_longitude_deg)

    # Create lat-long grid.
    num_grid_rows = 1 + int(numpy.round(
        (max_latitude_deg - min_latitude_deg) / DUMMY_LATITUDE_SPACING_DEG
    ))
    num_grid_columns = 1 + int(numpy.round(
        (max_longitude_deg - min_longitude_deg) / DUMMY_LONGITUDE_SPACING_DEG
    ))

    unique_latitudes_deg, unique_longitudes_deg = get_latlng_grid_points(
        min_latitude_deg=min_latitude_deg, min_longitude_deg=min_longitude_deg,
        lat_spacing_deg=DUMMY_LATITUDE_SPACING_DEG,
        lng_spacing_deg=DUMMY_LONGITUDE_SPACING_DEG,
        num_rows=num_grid_rows, num_columns=num_grid_columns)

    latitude_matrix_deg, longitude_matrix_deg = latlng_vectors_to_matrices(
        unique_latitudes_deg=unique_latitudes_deg,
        unique_longitudes_deg=unique_longitudes_deg)

    # Create projection.
    central_latitude_deg = 0.5 * (min_latitude_deg + max_latitude_deg)
    central_longitude_deg = 0.5 * (min_longitude_deg + max_longitude_deg)

    if azimuthal:
        projection_object = projections.init_azimuthal_equidistant_projection(
            central_latitude_deg=central_latitude_deg,
            central_longitude_deg=central_longitude_deg)
    else:
        projection_object = projections.init_lcc_projection(
            standard_latitudes_deg=numpy.full(2, central_latitude_deg),
            central_longitude_deg=central_longitude_deg
        )

    # Convert lat-long grid to preliminary x-y grid.
    prelim_x_matrix_metres, prelim_y_matrix_metres = (
        projections.project_latlng_to_xy(
            latitudes_deg=latitude_matrix_deg,
            longitudes_deg=longitude_matrix_deg,
            projection_object=projection_object)
    )

    # Find corners of preliminary x-y grid.
    x_min_metres = numpy.min(prelim_x_matrix_metres)
    x_max_metres = numpy.max(prelim_x_matrix_metres)
    y_min_metres = numpy.min(prelim_y_matrix_metres)
    y_max_metres = numpy.max(prelim_y_matrix_metres)

    # Find corners of final x-y grid.
    x_min_metres = number_rounding.floor_to_nearest(
        x_min_metres, x_spacing_metres)
    x_max_metres = number_rounding.ceiling_to_nearest(
        x_max_metres, x_spacing_metres)
    y_min_metres = number_rounding.floor_to_nearest(
        y_min_metres, y_spacing_metres)
    y_max_metres = number_rounding.ceiling_to_nearest(
        y_max_metres, y_spacing_metres)

    # Create final x-y grid.
    num_grid_rows = 1 + int(numpy.round(
        (y_max_metres - y_min_metres) / y_spacing_metres
    ))
    num_grid_columns = 1 + int(numpy.round(
        (x_max_metres - x_min_metres) / x_spacing_metres
    ))

    unique_x_coords_metres, unique_y_coords_metres = get_xy_grid_points(
        x_min_metres=x_min_metres, y_min_metres=y_min_metres,
        x_spacing_metres=x_spacing_metres, y_spacing_metres=y_spacing_metres,
        num_rows=num_grid_rows, num_columns=num_grid_columns)

    return {
        X_COORDS_KEY: unique_x_coords_metres,
        Y_COORDS_KEY: unique_y_coords_metres,
        PROJECTION_KEY: projection_object
    }


def find_events_in_grid_cell(
        event_x_coords_metres, event_y_coords_metres, grid_edge_x_coords_metres,
        grid_edge_y_coords_metres, row_index, column_index, verbose):
    """Finds events in a certain grid cell.

    E = number of events
    M = number of rows in grid
    N = number of columns in grid

    :param event_x_coords_metres: length-E numpy array of x-coordinates.
    :param event_y_coords_metres: length-E numpy array of y-coordinates.
    :param grid_edge_x_coords_metres: length-(N + 1) numpy array with
        x-coordinates at edges of grid cells.
    :param grid_edge_y_coords_metres: length-(M + 1) numpy array with
        y-coordinates at edges of grid cells.
    :param row_index: Will find events in [i]th row of grid, where
        i = `row_index.`
    :param column_index: Will find events in [j]th column of grid, where
        j = `column_index.`
    :param verbose: Boolean flag.  If True, messages will be printed to command
        window.
    :return: desired_indices: 1-D numpy array with indices of events in desired
        grid cell.
    """

    error_checking.assert_is_numpy_array_without_nan(event_x_coords_metres)
    error_checking.assert_is_numpy_array(
        event_x_coords_metres, num_dimensions=1)

    num_events = len(event_x_coords_metres)
    these_expected_dim = numpy.array([num_events], dtype=int)

    error_checking.assert_is_numpy_array_without_nan(event_y_coords_metres)
    error_checking.assert_is_numpy_array(
        event_y_coords_metres, exact_dimensions=these_expected_dim)

    error_checking.assert_is_numpy_array(
        grid_edge_x_coords_metres, num_dimensions=1)
    error_checking.assert_is_greater_numpy_array(
        numpy.diff(grid_edge_x_coords_metres), 0
    )

    error_checking.assert_is_numpy_array(
        grid_edge_y_coords_metres, num_dimensions=1)
    error_checking.assert_is_greater_numpy_array(
        numpy.diff(grid_edge_y_coords_metres), 0
    )

    error_checking.assert_is_integer(row_index)
    error_checking.assert_is_geq(row_index, 0)
    error_checking.assert_is_integer(column_index)
    error_checking.assert_is_geq(column_index, 0)
    error_checking.assert_is_boolean(verbose)

    x_min_metres = grid_edge_x_coords_metres[column_index]
    x_max_metres = grid_edge_x_coords_metres[column_index + 1]
    y_min_metres = grid_edge_y_coords_metres[row_index]
    y_max_metres = grid_edge_y_coords_metres[row_index + 1]

    if row_index == len(grid_edge_y_coords_metres) - 2:
        y_max_metres += TOLERANCE
    if column_index == len(grid_edge_x_coords_metres) - 2:
        x_max_metres += TOLERANCE

    # TODO(thunderhoser): If need be, I could speed this up by computing
    # `row_flags` only once per row and `column_flags` only once per column.
    row_flags = numpy.logical_and(
        event_y_coords_metres >= y_min_metres,
        event_y_coords_metres < y_max_metres
    )

    if not numpy.any(row_flags):
        if verbose:
            print('0 of {0:d} events are in grid cell ({1:d}, {2:d})!'.format(
                num_events, row_index, column_index
            ))

        return numpy.array([], dtype=int)

    column_flags = numpy.logical_and(
        event_x_coords_metres >= x_min_metres,
        event_x_coords_metres < x_max_metres
    )

    if not numpy.any(column_flags):
        if verbose:
            print('0 of {0:d} events are in grid cell ({1:d}, {2:d})!'.format(
                num_events, row_index, column_index
            ))

        return numpy.array([], dtype=int)

    desired_indices = numpy.where(numpy.logical_and(
        row_flags, column_flags
    ))[0]

    if verbose:
        print('{0:d} of {1:d} events are in grid cell ({2:d}, {3:d})!'.format(
            len(desired_indices), num_events, row_index, column_index
        ))

    return desired_indices


def find_equidistant_metafile(directory_name, raise_error_if_missing=True):
    """Finds metafile for equidistant grid.

    :param directory_name: Directory name.
    :param raise_error_if_missing: Boolean flag.  If file is missing and
        `raise_error_if_missing = True`, this method will error out.
    :return: metafile_name: Path to metafile.  If file is missing and
        `raise_error_if_missing = False`, this will be the expected path.
    :raises: ValueError: if file is missing and `raise_error_if_missing = True`.
    """

    error_checking.assert_is_string(directory_name)
    error_checking.assert_is_boolean(raise_error_if_missing)

    metafile_name = '{0:s}/equidistant_grid.p'.format(directory_name)

    if raise_error_if_missing and not os.path.isfile(metafile_name):
        error_string = 'Cannot find file.  Expected at: "{0:s}"'.format(
            metafile_name)
        raise ValueError(error_string)

    return metafile_name


def write_equidistant_metafile(grid_dict, pickle_file_name):
    """Writes metadata for equidistant grid to Pickle file.

    :param grid_dict: See doc for `create_equidistant_grid`.
    :param pickle_file_name: Path to output file.
    :raises: ValueError: if any keys are missing from dictionary.
    """

    missing_keys = list(
        set(EQUIDISTANT_METADATA_KEYS) - set(grid_dict.keys())
    )

    if len(missing_keys) > 0:
        error_string = (
            '\n{0:s}\nKeys listed above were expected, but not found, in '
            'dictionary.'
        ).format(str(missing_keys))

        raise ValueError(error_string)

    file_system_utils.mkdir_recursive_if_necessary(file_name=pickle_file_name)

    pickle_file_handle = open(pickle_file_name, 'wb')
    pickle.dump(grid_dict, pickle_file_handle)
    pickle_file_handle.close()


def read_equidistant_metafile(pickle_file_name):
    """Reads metadata for equidistant grid from Pickle file.

    :param pickle_file_name: Path to output file.
    :return: grid_dict: See doc for `create_equidistant_grid`.
    :raises: ValueError: if any keys are missing from dictionary.
    """

    error_checking.assert_is_string(pickle_file_name)

    pickle_file_handle = open(pickle_file_name, 'rb')
    grid_dict = pickle.load(pickle_file_handle)
    pickle_file_handle.close()

    missing_keys = list(
        set(EQUIDISTANT_METADATA_KEYS) - set(grid_dict.keys())
    )

    if len(missing_keys) == 0:
        return grid_dict

    error_string = (
        '\n{0:s}\nKeys listed above were expected, but not found, in file '
        '"{1:s}".'
    ).format(str(missing_keys), pickle_file_name)

    raise ValueError(error_string)
