"""Processing methods for gridded data.

DEFINITIONS

"Grid point" = center of grid cell (as opposed to edges of the grid cell).
"""

import numpy
from gewittergefahr.gg_utils import general_utils
from gewittergefahr.gg_utils import longitude_conversion as lng_conversion
from gewittergefahr.gg_utils import projections
from gewittergefahr.gg_utils import error_checking

DEGREES_LAT_TO_METRES = 60 * 1852
DEGREES_TO_RADIANS = numpy.pi / 180

GRID_POINT_X_MATRIX_KEY = 'grid_point_x_matrix_metres'
GRID_POINT_Y_MATRIX_KEY = 'grid_point_y_matrix_metres'
PROJECTION_OBJECT_KEY = 'projection_object'


def _check_input_events(
        event_x_coords_metres, event_y_coords_metres, integer_event_ids):
    """Checks inputs to `count_events_on*equidistant_grid`.

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
    :return: row_offset: If this is dR, row 0 in the subgrid is row dR in the
        full grid.
    :return: column_offset: If this is dC, column 0 in the subgrid is column dC
        in the full grid.
    """

    max_latitude_diff_deg = (
        max_distance_from_center_metres / DEGREES_LAT_TO_METRES)

    degrees_lng_to_metres = DEGREES_LAT_TO_METRES * numpy.cos(
        center_latitude_deg * DEGREES_TO_RADIANS)
    max_longitude_diff_deg = (
        max_distance_from_center_metres / degrees_lng_to_metres)

    min_latitude_deg = center_latitude_deg - max_latitude_diff_deg
    max_latitude_deg = center_latitude_deg + max_latitude_diff_deg
    min_longitude_deg = center_longitude_deg - max_longitude_diff_deg
    max_longitude_deg = center_longitude_deg + max_longitude_diff_deg

    valid_row_flags = numpy.logical_and(
        grid_point_latitudes_deg >= min_latitude_deg,
        grid_point_latitudes_deg <= max_latitude_deg)
    valid_column_flags = numpy.logical_and(
        grid_point_longitudes_deg >= min_longitude_deg,
        grid_point_longitudes_deg <= max_longitude_deg)

    valid_row_indices = numpy.where(valid_row_flags)[0]
    min_valid_row_index = numpy.min(valid_row_indices)
    max_valid_row_index = numpy.max(valid_row_indices)

    valid_column_indices = numpy.where(valid_column_flags)[0]
    min_valid_column_index = numpy.min(valid_column_indices)
    max_valid_column_index = numpy.max(valid_column_indices)

    subgrid_data_matrix = data_matrix[
        min_valid_row_index:(max_valid_row_index + 1),
        min_valid_column_index:(max_valid_column_index + 1)]

    subgrid_point_lat_matrix, subgrid_point_lng_matrix = (
        latlng_vectors_to_matrices(
            grid_point_latitudes_deg[valid_row_indices],
            grid_point_longitudes_deg[valid_column_indices]))
    lat_distance_matrix_metres = (
        (subgrid_point_lat_matrix - center_latitude_deg) *
        DEGREES_LAT_TO_METRES)
    lng_distance_matrix_metres = (
        (subgrid_point_lng_matrix - center_longitude_deg) *
        degrees_lng_to_metres)
    distance_matrix_metres = numpy.sqrt(
        lat_distance_matrix_metres ** 2 + lng_distance_matrix_metres ** 2)

    bad_indices = numpy.where(
        distance_matrix_metres > max_distance_from_center_metres)
    subgrid_data_matrix[bad_indices] = numpy.nan
    return subgrid_data_matrix, min_valid_row_index, min_valid_column_index


def count_events_on_equidistant_grid(
        event_x_coords_metres, event_y_coords_metres,
        grid_point_x_coords_metres, grid_point_y_coords_metres,
        integer_event_ids=None):
    """Counts number of events in each cell of an equidistant grid.

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
    :return: num_events_matrix: M-by-N numpy array, where element [i, j] is the
        number of events in grid cell [i, j].
    :return: event_ids_by_grid_cell_dict: Dictionary, where key [i, j] is a 1-D
        numpy array of event IDs assigned to grid cell [i, j].  If
        `event_ids is None`, this will also be `None`.
    """

    _check_input_events(
        event_x_coords_metres=event_x_coords_metres,
        event_y_coords_metres=event_y_coords_metres,
        integer_event_ids=integer_event_ids)
    num_events = len(event_x_coords_metres)

    error_checking.assert_is_numpy_array_without_nan(grid_point_x_coords_metres)
    error_checking.assert_is_numpy_array(
        grid_point_x_coords_metres, num_dimensions=1)

    error_checking.assert_is_numpy_array_without_nan(grid_point_y_coords_metres)
    error_checking.assert_is_numpy_array(
        grid_point_y_coords_metres, num_dimensions=1)

    num_grid_rows = len(grid_point_y_coords_metres)
    num_grid_columns = len(grid_point_x_coords_metres)
    num_events_matrix = numpy.full(
        (num_grid_rows, num_grid_columns), 0, dtype=int)

    if integer_event_ids is None:
        event_ids_by_grid_cell_dict = None
    else:
        event_ids_by_grid_cell_dict = {}
        for i in range(num_grid_rows):
            for j in range(num_grid_columns):
                event_ids_by_grid_cell_dict[i, j] = []

    for k in range(num_events):
        if numpy.mod(k, 1000) == 0:
            print 'Have assigned {0:d} of {1:d} events to grid cells...'.format(
                k, num_events)

        _, this_row = general_utils.find_nearest_value(
            grid_point_y_coords_metres, event_y_coords_metres[k])
        _, this_column = general_utils.find_nearest_value(
            grid_point_x_coords_metres, event_x_coords_metres[k])

        if integer_event_ids is None:
            num_events_matrix[this_row, this_column] += 1
        else:
            event_ids_by_grid_cell_dict[this_row, this_column].append(
                integer_event_ids[k])

    print 'Have assigned all {0:d} events to grid cells!'.format(num_events)

    if integer_event_ids is not None:
        for i in range(num_grid_rows):
            for j in range(num_grid_columns):
                event_ids_by_grid_cell_dict[i, j] = numpy.array(
                    list(set(event_ids_by_grid_cell_dict[i, j])), dtype=int)
                num_events_matrix[i, j] = len(event_ids_by_grid_cell_dict[i, j])

    return num_events_matrix, event_ids_by_grid_cell_dict


def count_events_on_non_equidistant_grid(
        event_x_coords_metres, event_y_coords_metres,
        grid_point_x_matrix_metres, grid_point_y_matrix_metres,
        effective_radius_metres, integer_event_ids=None):
    """Counts number of events near each point of a non-equidistant grid.

    M = number of rows in grid
    N = number of columns in grid
    K = number of events

    :param event_x_coords_metres: length-K numpy array with x-coordinates of
        events.
    :param event_y_coords_metres: length-K numpy array with y-coordinates of
        events.
    :param grid_point_x_matrix_metres: M-by-N numpy array with x-coordinates of
        grid points.
    :param grid_point_y_matrix_metres: M-by-N numpy array with y-coordinates of
        grid points.
    :param effective_radius_metres: At each grid point [i, j], the value
        computed will be number of events within `effective_radius_metres` of
        point [i, j].
    :param integer_event_ids: See doc for `count_events_on_equidistant_grid`.
    :return: num_events_matrix: M-by-N numpy array, where element [i, j] is the
        number of events within `effective_radius_metres` of grid point [i, j].
    :return: event_ids_by_grid_cell_dict: See doc for
        `count_events_on_equidistant_grid`.
    """

    _check_input_events(
        event_x_coords_metres=event_x_coords_metres,
        event_y_coords_metres=event_y_coords_metres,
        integer_event_ids=integer_event_ids)
    num_events = len(event_x_coords_metres)

    error_checking.assert_is_numpy_array_without_nan(grid_point_x_matrix_metres)
    error_checking.assert_is_numpy_array(
        grid_point_x_matrix_metres, num_dimensions=2)

    error_checking.assert_is_numpy_array_without_nan(grid_point_y_matrix_metres)
    error_checking.assert_is_numpy_array(
        grid_point_y_matrix_metres,
        exact_dimensions=numpy.array(grid_point_x_matrix_metres.shape))

    error_checking.assert_is_greater(effective_radius_metres, 0.)
    effective_radius_metres2 = effective_radius_metres ** 2

    num_grid_rows = grid_point_x_matrix_metres.shape[0]
    num_grid_columns = grid_point_x_matrix_metres.shape[1]
    num_events_matrix = numpy.full(
        (num_grid_rows, num_grid_columns), 0, dtype=int)

    if integer_event_ids is None:
        event_ids_by_grid_cell_dict = None
    else:
        event_ids_by_grid_cell_dict = {}
        for i in range(num_grid_rows):
            for j in range(num_grid_columns):
                event_ids_by_grid_cell_dict[i, j] = []

    for k in range(num_events):
        if numpy.mod(k, 1000) == 0:
            print 'Have assigned {0:d} of {1:d} events to grid cells...'.format(
                k, num_events)

        these_check_x_flags = numpy.logical_and(
            grid_point_x_matrix_metres >=
            event_x_coords_metres[k] - effective_radius_metres,
            grid_point_x_matrix_metres <=
            event_x_coords_metres[k] + effective_radius_metres)
        these_check_y_flags = numpy.logical_and(
            grid_point_y_matrix_metres >=
            event_y_coords_metres[k] - effective_radius_metres,
            grid_point_y_matrix_metres <=
            event_y_coords_metres[k] + effective_radius_metres)
        these_indices_to_check_as_tuple = numpy.where(
            numpy.logical_and(these_check_x_flags, these_check_y_flags))

        these_distances_metres2 = (
            (grid_point_x_matrix_metres[these_indices_to_check_as_tuple] -
             event_x_coords_metres[k]) ** 2 +
            (grid_point_y_matrix_metres[these_indices_to_check_as_tuple] -
             event_y_coords_metres[k]) ** 2
        )
        these_nearby_indices = numpy.where(
            these_distances_metres2 <= effective_radius_metres2)[0]
        these_nearby_rows = these_indices_to_check_as_tuple[
            0][these_nearby_indices]
        these_nearby_columns = these_indices_to_check_as_tuple[
            1][these_nearby_indices]

        if integer_event_ids is None:
            num_events_matrix[these_nearby_rows, these_nearby_columns] += 1
        else:
            for m in range(len(these_nearby_rows)):
                event_ids_by_grid_cell_dict[
                    these_nearby_rows[m], these_nearby_columns[m]
                ].append(integer_event_ids[k])

    print 'Have assigned all {0:d} events to grid cells!'.format(num_events)

    if integer_event_ids is not None:
        for i in range(num_grid_rows):
            for j in range(num_grid_columns):
                event_ids_by_grid_cell_dict[i, j] = numpy.array(
                    list(set(event_ids_by_grid_cell_dict[i, j])), dtype=int)
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
            GRID_POINT_X_MATRIX_KEY: grid_point_x_matrix_metres,
            GRID_POINT_Y_MATRIX_KEY: grid_point_y_matrix_metres,
            PROJECTION_OBJECT_KEY: projection_object
        }

    error_checking.assert_is_valid_latitude(test_latitude_deg)
    error_checking.assert_is_geq(effective_radius_metres, 0.)
    test_longitude_deg = lng_conversion.convert_lng_positive_in_west(
        longitudes_deg=numpy.array([test_longitude_deg]), allow_nan=False)[0]

    (test_x_coords_metres, test_y_coords_metres
    ) = projections.project_latlng_to_xy(
        latitudes_deg=numpy.array([test_latitude_deg]),
        longitudes_deg=numpy.array([test_longitude_deg]),
        projection_object=grid_point_dict[PROJECTION_OBJECT_KEY])
    test_x_coord_metres = test_x_coords_metres[0]
    test_y_coord_metres = test_y_coords_metres[0]

    valid_x_flags = numpy.absolute(
        grid_point_dict[GRID_POINT_X_MATRIX_KEY] - test_x_coord_metres
    ) <= effective_radius_metres
    valid_y_flags = numpy.absolute(
        grid_point_dict[GRID_POINT_Y_MATRIX_KEY] - test_y_coord_metres
    ) <= effective_radius_metres
    rows_to_try, columns_to_try = numpy.where(numpy.logical_and(
        valid_x_flags, valid_y_flags))

    distances_to_try_metres = numpy.sqrt(
        (grid_point_dict[GRID_POINT_X_MATRIX_KEY][rows_to_try, columns_to_try] -
         test_x_coord_metres) ** 2 +
        (grid_point_dict[GRID_POINT_Y_MATRIX_KEY][rows_to_try, columns_to_try] -
         test_y_coord_metres) ** 2)
    valid_indices = numpy.where(
        distances_to_try_metres <= effective_radius_metres)[0]

    return (rows_to_try[valid_indices], columns_to_try[valid_indices],
            grid_point_dict)
