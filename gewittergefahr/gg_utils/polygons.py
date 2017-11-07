"""Processing methods for polygons.

Currently the only polygons in GewitterGefahr are storm-cell outlines.  However,
I may add other polygons.

When a method says that x- and y-coordinates may be in one of three formats, the
three formats are as follows:

[1] metres;
[2] degrees of longitude and latitude, respectively;
[3] columns and rows, respectively, in a grid (where y-coordinate increases with
    row number and x-coordinate increases with column number).
"""

import copy
import numpy
import cv2
import shapely.geometry
from gewittergefahr.gg_utils import grids
from gewittergefahr.gg_utils import projections
from gewittergefahr.gg_utils import error_checking

TOLERANCE = 1e-6

UP_DIRECTION_NAME = 'up'
DOWN_DIRECTION_NAME = 'down'
RIGHT_DIRECTION_NAME = 'right'
LEFT_DIRECTION_NAME = 'left'
UP_RIGHT_DIRECTION_NAME = 'up_right'
UP_LEFT_DIRECTION_NAME = 'up_left'
DOWN_RIGHT_DIRECTION_NAME = 'down_right'
DOWN_LEFT_DIRECTION_NAME = 'down_left'
COMPLEX_DIRECTIONS = [UP_RIGHT_DIRECTION_NAME, UP_LEFT_DIRECTION_NAME,
                      DOWN_RIGHT_DIRECTION_NAME, DOWN_LEFT_DIRECTION_NAME]

EXTERIOR_X_COLUMN = 'exterior_x_coords'
EXTERIOR_Y_COLUMN = 'exterior_y_coords'
HOLE_X_COLUMN = 'hole_x_coords_list'
HOLE_Y_COLUMN = 'hole_y_coords_list'


def _check_vertex_arrays(x_coordinates, y_coordinates, allow_nan=True):
    """Checks vertex arrays for errors.

    x- and y-coordinates may be in one of the formats listed at the top.

    V = number of vertices

    :param x_coordinates: length-V numpy array with x-coordinates of vertices.
        The first NaN separates the exterior from the first hole, and the [i]th
        NaN separates the [i - 1]th hole from the [i]th hole.
    :param y_coordinates: Same as above, except for y-coordinates.
    :param allow_nan: Boolean flag.  If allow_nan = False and NaN is found in
        either of the vertex arrays, this method will raise an error.
        Otherwise, the only restriction on NaN's is that, if
        x_coordinates[i] = NaN, y_coordinates[i] must be NaN -- and vice-versa.
    :raises: ValueError: if allow_nan = True and NaN is found in either of the
        vertex arrays.
    :raises: ValueError: if x_coordinates[i] = NaN and y_coordinates[i] != NaN,
        or vice-versa.
    """

    error_checking.assert_is_boolean(allow_nan)

    if allow_nan:
        error_checking.assert_is_real_numpy_array(x_coordinates)
        error_checking.assert_is_real_numpy_array(y_coordinates)
    else:
        error_checking.assert_is_numpy_array_without_nan(x_coordinates)
        error_checking.assert_is_numpy_array_without_nan(y_coordinates)

    error_checking.assert_is_numpy_array(x_coordinates, num_dimensions=1)
    num_vertices = len(x_coordinates)
    error_checking.assert_is_numpy_array(
        y_coordinates, exact_dimensions=numpy.array([num_vertices]))

    x_nan_indices = numpy.where(numpy.isnan(x_coordinates))[0]
    y_nan_indices = numpy.where(numpy.isnan(y_coordinates))[0]
    if not numpy.array_equal(x_nan_indices, y_nan_indices):
        error_string = (
            '\nThe following elements of `x_coordinates` are NaN:\n' +
            str(x_nan_indices) +
            '\nThe following elements of `y_coordinates` are NaN:\n' +
            str(y_nan_indices) +
            '\nAs shown above, NaN entries (polygon discontinuities) do not '
            'match.')
        raise ValueError(error_string)


def _get_longest_inner_list(list_of_lists):
    """Finds longest list in a list.

    :param list_of_lists: 1-D list of lists.
    :return: longest_list: Longest of inner lists.
    """

    num_lists = len(list_of_lists)
    list_lengths = numpy.full(num_lists, 0, dtype=int)
    for i in range(num_lists):
        list_lengths[i] = len(list_of_lists[i])

    return list_of_lists[numpy.argmax(list_lengths)]


def _get_longest_vertex_arrays_without_nan(vertex_x_coords, vertex_y_coords):
    """Finds longest sequence of vertices without NaN.

    This is equivalent to finding the simple polygon with the most vertices.

    x- and y-coordinates may be in one of the formats listed at the top.

    V = number of vertices

    :param vertex_x_coords: length-V numpy array with x-coordinates of vertices.
        The first NaN separates the exterior from the first hole, and the [i]th
        NaN separates the [i - 1]th hole from the [i]th hole.
    :param vertex_y_coords: Same as above, except for y-coordinates.
    :return: simple_vertex_x_coords: Longest subsequence of vertex_x_coords
        without NaN.
    :return: simple_vertex_y_coords: Longest subsequence of vertex_y_coords
        without NaN.
    """

    _check_vertex_arrays(vertex_x_coords, vertex_y_coords, allow_nan=True)

    nan_flags = numpy.isnan(vertex_x_coords)
    if not numpy.any(nan_flags):
        return vertex_x_coords, vertex_y_coords

    nan_indices = numpy.where(nan_flags)[0]
    num_simple_polygons = len(nan_indices) + 1
    start_indices_by_polygon = numpy.full(num_simple_polygons, -1, dtype=int)
    end_indices_by_polygon = numpy.full(num_simple_polygons, -1, dtype=int)

    for i in range(num_simple_polygons):
        if i == 0:
            start_indices_by_polygon[i] = 0
            end_indices_by_polygon[i] = nan_indices[i] - 1
        elif i == num_simple_polygons - 1:
            start_indices_by_polygon[i] = nan_indices[i - 1] + 1
            end_indices_by_polygon[i] = len(vertex_x_coords) - 1
        else:
            start_indices_by_polygon[i] = nan_indices[i - 1] + 1
            end_indices_by_polygon[i] = nan_indices[i] - 1

    num_vertices_by_polygon = (
        end_indices_by_polygon - start_indices_by_polygon + 1)
    max_index = numpy.argmax(num_vertices_by_polygon)

    simple_vertex_x_coords = (
        vertex_x_coords[start_indices_by_polygon[max_index]:
                        (end_indices_by_polygon[max_index] + 1)])
    simple_vertex_y_coords = (
        vertex_y_coords[start_indices_by_polygon[max_index]:
                        (end_indices_by_polygon[max_index] + 1)])

    return simple_vertex_x_coords, simple_vertex_y_coords


def _vertex_arrays_to_list(vertex_x_coords, vertex_y_coords):
    """Converts vertex coordinates from two arrays to one list.

    x- and y-coordinates may be in one of the formats listed at the top.  In
    this case, coordinates may not contain NaN's (simple polygons only).

    V = number of vertices

    :param vertex_x_coords: length-V numpy array with x-coordinates of vertices.
    :param vertex_y_coords: length-V numpy array with y-coordinates of vertices.
    :return: vertex_coords_list: length-V list, where the [i]th element is a
        tuple with (x-coordinate, y-coordinate).
    """

    _check_vertex_arrays(vertex_x_coords, vertex_y_coords, allow_nan=False)

    num_vertices = len(vertex_x_coords)
    vertex_coords_list = []
    for i in range(num_vertices):
        vertex_coords_list.append((vertex_x_coords[i], vertex_y_coords[i]))

    return vertex_coords_list


def _vertex_list_to_arrays(vertex_coords_list):
    """Converts vertex coordinates from one list to two arrays.

    x- and y-coordinates may be in one of the formats listed at the top.  In
    this case, coordinates may not contain NaN's (simple polygons only).

    V = number of vertices

    :param vertex_coords_list: length-V list, where the [i]th element is a
        tuple with (x-coordinate, y-coordinate).
    :return: vertex_x_coords: length-V numpy array with x-coordinates of
        vertices.
    :return: vertex_y_coords: length-V numpy array with y-coordinates of
        vertices.
    """

    num_vertices = len(vertex_coords_list)
    vertex_x_coords = numpy.full(num_vertices, numpy.nan)
    vertex_y_coords = numpy.full(num_vertices, numpy.nan)

    for i in range(num_vertices):
        vertex_x_coords[i] = vertex_coords_list[i][0]
        vertex_y_coords[i] = vertex_coords_list[i][1]

    return vertex_x_coords, vertex_y_coords


def _get_direction_of_vertex_pair(first_row, second_row, first_column,
                                  second_column):
    """Finds direction between two vertices (from the first to the second).

    This method assumes that row number increases downward and column number
    increases to the right.

    There are 8 possible directions:

    - up
    - down
    - right
    - left
    - up and right (45-degree angle)
    - up and left (45-degree angle)
    - down and right (45-degree angle)
    - down and left (45-degree angle)

    :param first_row: Row number of first vertex.
    :param second_row: Row number of second vertex.
    :param first_column: Column number of first vertex.
    :param second_column: Column number of second vertex.
    :return: direction_string: String indicating direction from first to second
        vertex.  Possible strings are "up", "down", "right", "left", "up_right",
        "up_left", "down_right", and "down_left".
    """

    if first_column == second_column:
        if second_row > first_row:
            return DOWN_DIRECTION_NAME
        if second_row < first_row:
            return UP_DIRECTION_NAME

    if first_row == second_row:
        if second_column > first_column:
            return RIGHT_DIRECTION_NAME
        if second_column < first_column:
            return LEFT_DIRECTION_NAME

    if second_row > first_row:
        if second_column > first_column:
            return DOWN_RIGHT_DIRECTION_NAME
        if second_column < first_column:
            return DOWN_LEFT_DIRECTION_NAME

    if second_row < first_row:
        if second_column > first_column:
            return UP_RIGHT_DIRECTION_NAME
        if second_column < first_column:
            return UP_LEFT_DIRECTION_NAME

    return None


def _remove_redundant_vertices(row_indices_orig, column_indices_orig):
    """Removes redundant vertices from a simple polygon.

    v = original number of vertices
    V = final number of vertices

    :param row_indices_orig: length-v numpy array with row numbers of original
        vertices.
    :param column_indices_orig: length-v numpy array with column numbers of
        original vertices.
    :return: row_indices: length-V numpy array with row numbers of non-redundant
        vertices.
    :return: column_indices: length-V numpy array with column numbers of non-
        redundant vertices.
    """

    _check_vertex_arrays(column_indices_orig, row_indices_orig, allow_nan=False)

    num_vertices_orig = len(row_indices_orig)
    row_indices = numpy.array([])
    column_indices = numpy.array([])

    for i in range(num_vertices_orig - 1):
        found_flags = numpy.logical_and(
            row_indices == row_indices_orig[i],
            column_indices == column_indices_orig[i])

        if not numpy.any(found_flags):
            row_indices = numpy.concatenate(
                (row_indices, row_indices_orig[[i]]))
            column_indices = numpy.concatenate(
                (column_indices, column_indices_orig[[i]]))
        else:
            found_index = numpy.where(found_flags)[0][0]
            row_indices = row_indices[0:(found_index + 1)]
            column_indices = column_indices[0:(found_index + 1)]

    row_indices = numpy.concatenate((row_indices, row_indices[[0]]))
    column_indices = numpy.concatenate((column_indices, column_indices[[0]]))
    return row_indices, column_indices


def _patch_diag_connections_in_binary_matrix(binary_matrix):
    """Patches diagonal connections in binary image matrix.

    When two pixels p and qare connected only diagonally, this method "patches"
    the connection by adding another pixel -- adjacent to both p and q -- to the
    image.  In other words, this method flips one bit in the image from False to
    True.

    If diagonal connections are not patched, grid_points_in_poly_to_vertices
    will create disjoint polygons.

    M = number of rows in binary image
    N = number of columns in binary image

    :param binary_matrix: M-by-N numpy array of Boolean flags.
        binary_matrix[i, j] indicates whether or not pixel [i, j] is inside the
        polygon.
    :return: binary_matrix: Same as input, except that diagonal connections have
        been patched.
    """

    num_rows = binary_matrix.shape[0]
    num_columns = binary_matrix.shape[1]
    found_diag_connection = True

    while found_diag_connection:
        found_diag_connection = False

        for i in range(num_rows - 1):
            for j in range(num_columns):
                if not binary_matrix[i, j]:
                    continue

                if j != 0 and binary_matrix[i + 1, j - 1]:
                    if not (binary_matrix[i + 1, j] or binary_matrix[i, j - 1]):
                        binary_matrix[i + 1, j] = True
                        found_diag_connection = True
                        break

                if j != num_columns - 1 and binary_matrix[i + 1, j + 1]:
                    if not (binary_matrix[i + 1, j] or binary_matrix[i, j + 1]):
                        binary_matrix[i + 1, j] = True
                        found_diag_connection = True
                        break

            if found_diag_connection:
                break

    return binary_matrix


def _binary_matrix_to_grid_points_in_poly(binary_matrix, first_row_index,
                                          first_column_index):
    """Converts binary image matrix to list of grid points in polygon.

    M = number of rows in subgrid
    N = number of columns in subgrid
    P = number of grid points in polygon

    :param binary_matrix: M-by-N numpy array of Boolean flags.
        binary_matrix[i, j] indicates whether or not pixel [i, j] -- in the
        subgrid, not necessarily the full grid -- is inside the polygon.
    :param first_row_index: Used to convert row numbers from the subgrid to the
        full grid.  Row 0 in the subgrid = row `first_row_index` in the full
        grid.
    :param first_column_index: Used to convert column numbers from the subgrid
        to the full grid.  Column 0 in the subgrid = column `first_column_index`
        in the full grid.
    :return: row_indices: length-P numpy array with row numbers of grid points
        in polygon.  These are rows in the full grid.
    :return: column_indices: Same as above, except for columns.
    """

    num_rows_in_subgrid = binary_matrix.shape[0]
    num_columns_in_subgrid = binary_matrix.shape[1]
    binary_vector = numpy.reshape(binary_matrix,
                                  num_rows_in_subgrid * num_columns_in_subgrid)

    linear_indices_in_subgrid = numpy.where(binary_vector)[0]
    (row_indices_in_subgrid, column_indices_in_subgrid) = numpy.unravel_index(
        linear_indices_in_subgrid,
        (num_rows_in_subgrid, num_columns_in_subgrid))

    return (row_indices_in_subgrid + first_row_index,
            column_indices_in_subgrid + first_column_index)


def _vertices_from_grid_points_to_edges(row_indices_orig, column_indices_orig):
    """Moves vertices from grid points to grid-cell edges.

    This ensures that vertices follow the outlines of grid cells, rather than
    cutting through grid cells.

    This method works only for simple polygons sorted in counterclockwise order.

    v = original number of vertices
    V = final number of vertices

    :param row_indices_orig: length-v numpy array with row numbers (integers) of
        original vertices.
    :param column_indices_orig: length-v numpy array with column numbers
        (integers) of original vertices.
    :return: row_indices: length-V numpy array with row numbers (half-integers)
        of final vertices (not cutting through grid cells).
    :return: column_indices: Same as above, except for columns.
    """

    error_checking.assert_is_integer_numpy_array(row_indices_orig)
    error_checking.assert_is_integer_numpy_array(column_indices_orig)

    num_orig_vertices = len(row_indices_orig)
    row_indices = numpy.array([])
    column_indices = numpy.array([])

    for i in range(num_orig_vertices - 1):
        this_direction = _get_direction_of_vertex_pair(
            row_indices_orig[i], row_indices_orig[i + 1],
            column_indices_orig[i], column_indices_orig[i + 1])

        # TODO(thunderhoser): put the following block in another method.
        if this_direction in COMPLEX_DIRECTIONS:
            this_absolute_row_diff = numpy.absolute(
                row_indices_orig[i + 1] - row_indices_orig[i])
            this_absolute_column_diff = numpy.absolute(
                column_indices_orig[i + 1] - column_indices_orig[i])
            this_num_steps = int(numpy.min(numpy.array(
                [this_absolute_row_diff, this_absolute_column_diff])))

            these_row_indices_orig = numpy.linspace(
                float(row_indices_orig[i]), float(row_indices_orig[i + 1]),
                num=this_num_steps + 1)
            these_column_indices_orig = numpy.linspace(
                float(column_indices_orig[i]),
                float(column_indices_orig[i + 1]), num=this_num_steps + 1)

            these_row_integer_flags = numpy.isclose(
                these_row_indices_orig, numpy.round(these_row_indices_orig),
                atol=TOLERANCE)
            these_column_integer_flags = numpy.isclose(
                these_column_indices_orig,
                numpy.round(these_column_indices_orig), atol=TOLERANCE)
            these_valid_flags = numpy.logical_and(
                these_row_integer_flags, these_column_integer_flags)

            these_valid_indices = numpy.where(these_valid_flags)[0]
            these_row_indices_orig = these_row_indices_orig[these_valid_indices]
            these_column_indices_orig = these_column_indices_orig[
                these_valid_indices]
            this_num_steps = len(these_row_indices_orig) - 1

        if this_direction == UP_DIRECTION_NAME:
            rows_to_append = numpy.array(
                [row_indices_orig[i] + 0.5, row_indices_orig[i + 1] - 0.5])
            columns_to_append = numpy.array([column_indices_orig[i] + 0.5,
                                             column_indices_orig[i + 1] + 0.5])
        elif this_direction == DOWN_DIRECTION_NAME:
            rows_to_append = numpy.array(
                [row_indices_orig[i] - 0.5, row_indices_orig[i + 1] + 0.5])
            columns_to_append = numpy.array([column_indices_orig[i] - 0.5,
                                             column_indices_orig[i + 1] - 0.5])
        elif this_direction == RIGHT_DIRECTION_NAME:
            rows_to_append = numpy.array(
                [row_indices_orig[i] + 0.5, row_indices_orig[i + 1] + 0.5])
            columns_to_append = numpy.array([column_indices_orig[i] - 0.5,
                                             column_indices_orig[i + 1] + 0.5])
        elif this_direction == LEFT_DIRECTION_NAME:
            rows_to_append = numpy.array(
                [row_indices_orig[i] - 0.5, row_indices_orig[i + 1] - 0.5])
            columns_to_append = numpy.array([column_indices_orig[i] + 0.5,
                                             column_indices_orig[i + 1] - 0.5])
        else:
            rows_to_append = numpy.array([])
            columns_to_append = numpy.array([])

            for j in range(this_num_steps):
                if this_direction == UP_RIGHT_DIRECTION_NAME:
                    these_rows_to_append = numpy.array(
                        [these_row_indices_orig[j] + 0.5,
                         these_row_indices_orig[j + 1] + 0.5,
                         these_row_indices_orig[j + 1] + 0.5])
                    these_columns_to_append = numpy.array(
                        [these_column_indices_orig[j] + 0.5,
                         these_column_indices_orig[j] + 0.5,
                         these_column_indices_orig[j + 1] + 0.5])
                elif this_direction == UP_LEFT_DIRECTION_NAME:
                    these_rows_to_append = numpy.array(
                        [these_row_indices_orig[j] - 0.5,
                         these_row_indices_orig[j] - 0.5,
                         these_row_indices_orig[j + 1] - 0.5])
                    these_columns_to_append = numpy.array(
                        [these_column_indices_orig[j] + 0.5,
                         these_column_indices_orig[j + 1] + 0.5,
                         these_column_indices_orig[j + 1] + 0.5])
                elif this_direction == DOWN_RIGHT_DIRECTION_NAME:
                    these_rows_to_append = numpy.array(
                        [these_row_indices_orig[j] + 0.5,
                         these_row_indices_orig[j] + 0.5,
                         these_row_indices_orig[j + 1] + 0.5])
                    these_columns_to_append = numpy.array(
                        [these_column_indices_orig[j] - 0.5,
                         these_column_indices_orig[j + 1] - 0.5,
                         these_column_indices_orig[j + 1] - 0.5])
                elif this_direction == DOWN_LEFT_DIRECTION_NAME:
                    these_rows_to_append = numpy.array(
                        [these_row_indices_orig[j] - 0.5,
                         these_row_indices_orig[j + 1] - 0.5,
                         these_row_indices_orig[j + 1] - 0.5])
                    these_columns_to_append = numpy.array(
                        [these_column_indices_orig[j] - 0.5,
                         these_column_indices_orig[j] - 0.5,
                         these_column_indices_orig[j + 1] - 0.5])

                rows_to_append = numpy.concatenate((
                    rows_to_append, these_rows_to_append))
                columns_to_append = numpy.concatenate((
                    columns_to_append, these_columns_to_append))

        row_indices = numpy.concatenate((row_indices, rows_to_append))
        column_indices = numpy.concatenate((column_indices, columns_to_append))

    return row_indices, column_indices


def separate_exterior_and_holes(vertex_x_coords, vertex_y_coords):
    """Separates exterior of polygon from holes in polygon.

    x- and y-coordinates may be in one of the formats listed at the top.

    V = number of vertices
    H = number of holes
    V_e = number of exterior vertices
    V_hi = number of vertices in [i]th hole

    :param vertex_x_coords: length-V numpy array with x-coordinates of vertices.
        The first NaN separates the exterior from the first hole, and the [i]th
        NaN separates the [i - 1]th hole from the [i]th hole.
    :param vertex_y_coords: Same as above, except for y-coordinates.
    :return: vertex_dict: Dictionary with the following keys.
    vertex_dict.exterior_x_coords: numpy array (length V_e) with x-coordinates
        of exterior vertices.
    vertex_dict.exterior_y_coords: numpy array (length V_e) with y-coordinates
        of exterior vertices.
    vertex_dict.hole_x_coords_list: length-H list, where the [i]th item is a
        numpy array (length V_hi) with x-coordinates of interior vertices.
    vertex_dict.hole_y_coords_list: Same as above, except for y-coordinates.
    """

    _check_vertex_arrays(vertex_x_coords, vertex_y_coords, allow_nan=True)

    nan_flags = numpy.isnan(vertex_x_coords)
    if not numpy.any(nan_flags):
        return {EXTERIOR_X_COLUMN: vertex_x_coords,
                EXTERIOR_Y_COLUMN: vertex_y_coords,
                HOLE_X_COLUMN: [], HOLE_Y_COLUMN: []}

    nan_indices = numpy.where(nan_flags)[0]
    num_holes = len(nan_indices)
    exterior_x_coords = vertex_x_coords[0:nan_indices[0]]
    exterior_y_coords = vertex_y_coords[0:nan_indices[0]]
    hole_x_coords_list = []
    hole_y_coords_list = []

    for i in range(num_holes):
        if i == num_holes - 1:
            this_hole_x_coords = vertex_x_coords[(nan_indices[i] + 1):]
            this_hole_y_coords = vertex_y_coords[(nan_indices[i] + 1):]
        else:
            this_hole_x_coords = (
                vertex_x_coords[(nan_indices[i] + 1):nan_indices[i + 1]])
            this_hole_y_coords = (
                vertex_y_coords[(nan_indices[i] + 1):nan_indices[i + 1]])

        hole_x_coords_list.append(this_hole_x_coords)
        hole_y_coords_list.append(this_hole_y_coords)

    return {EXTERIOR_X_COLUMN: exterior_x_coords,
            EXTERIOR_Y_COLUMN: exterior_y_coords,
            HOLE_X_COLUMN: hole_x_coords_list,
            HOLE_Y_COLUMN: hole_y_coords_list}


def merge_exterior_and_holes(exterior_x_coords, exterior_y_coords,
                             hole_x_coords_list=None, hole_y_coords_list=None):
    """Merges exterior of polygon with holes in polygon.

    x- and y-coordinates may be in one of the formats listed at the top.

    V = number of vertices
    H = number of holes
    V_e = number of exterior vertices
    V_hi = number of vertices in [i]th hole

    :param exterior_x_coords: numpy array (length V_e) with x-coordinates of
        exterior vertices.
    :param exterior_y_coords: numpy array (length V_e) with y-coordinates of
        exterior vertices.
    :param hole_x_coords_list: length-H list, where the [i]th item is a numpy
        array (length V_hi) with x-coordinates of interior vertices.
    :param hole_y_coords_list: Same as above, except for y-coordinates.
    :return: vertex_x_coords: length-V numpy array with x-coordinates of
        vertices.  The first NaN separates the exterior from the first hole, and
        the [i]th NaN separates the [i - 1]th hole from the [i]th hole.
    :return: vertex_y_coords: Same as above, except for y-coordinates.
    """

    _check_vertex_arrays(exterior_x_coords, exterior_y_coords, allow_nan=False)

    vertex_x_coords = copy.deepcopy(exterior_x_coords)
    vertex_y_coords = copy.deepcopy(exterior_y_coords)
    if hole_x_coords_list is None:
        return vertex_x_coords, vertex_y_coords

    num_holes = len(hole_x_coords_list)
    for i in range(num_holes):
        _check_vertex_arrays(
            hole_x_coords_list[i], hole_y_coords_list[i], allow_nan=False)

    single_nan_array = numpy.array([numpy.nan])
    for i in range(num_holes):
        vertex_x_coords = numpy.concatenate(
            (vertex_x_coords, single_nan_array, hole_x_coords_list[i]))
        vertex_y_coords = numpy.concatenate(
            (vertex_y_coords, single_nan_array, hole_y_coords_list[i]))

    return vertex_x_coords, vertex_y_coords


def project_latlng_to_xy(polygon_object_latlng, projection_object=None,
                         false_easting_metres=0, false_northing_metres=0.):
    """Converts polygon from lat-long to x-y coordinates.

    :param polygon_object_latlng: Instance of `shapely.geometry.Polygon`, with
        vertices in lat-long coordinates.
    :param projection_object: Projection object created by `pyproj.Proj`.  If
        projection_object = None, will use azimuthal equidistant projection
        centered at polygon centroid.
    :param false_easting_metres: False easting.  Will be added to all x-
        coordinates.
    :param false_northing_metres: False northing.  Will be added to all y-
        coordinates.
    :return: polygon_object_xy: Instance of `shapely.geometry.Polygon`, with
        vertices in x-y coordinates.
    :return: projection_object: Object (created by `pyproj.Proj`) used to
        convert coordinates.
    """

    if projection_object is None:
        centroid_object_latlng = polygon_object_latlng.centroid
        projection_object = projections.init_azimuthal_equidistant_projection(
            centroid_object_latlng.y, centroid_object_latlng.x)
        false_easting_metres = 0.
        false_northing_metres = 0.

    vertex_dict = polygon_object_to_vertex_arrays(polygon_object_latlng)
    vertex_dict[EXTERIOR_X_COLUMN], vertex_dict[EXTERIOR_Y_COLUMN] = (
        projections.project_latlng_to_xy(
            vertex_dict[EXTERIOR_Y_COLUMN], vertex_dict[EXTERIOR_X_COLUMN],
            projection_object=projection_object,
            false_easting_metres=false_easting_metres,
            false_northing_metres=false_northing_metres))

    num_holes = len(vertex_dict[HOLE_X_COLUMN])
    for i in range(num_holes):
        vertex_dict[HOLE_X_COLUMN][i], vertex_dict[HOLE_Y_COLUMN][i] = (
            projections.project_latlng_to_xy(
                vertex_dict[HOLE_Y_COLUMN][i], vertex_dict[HOLE_X_COLUMN][i],
                projection_object=projection_object,
                false_easting_metres=false_easting_metres,
                false_northing_metres=false_northing_metres))

    if num_holes == 0:
        polygon_object_xy = vertex_arrays_to_polygon_object(
            vertex_dict[EXTERIOR_X_COLUMN], vertex_dict[EXTERIOR_Y_COLUMN])
    else:
        polygon_object_xy = vertex_arrays_to_polygon_object(
            vertex_dict[EXTERIOR_X_COLUMN], vertex_dict[EXTERIOR_Y_COLUMN],
            hole_x_coords_list=vertex_dict[HOLE_X_COLUMN],
            hole_y_coords_list=vertex_dict[HOLE_Y_COLUMN])

    return polygon_object_xy, projection_object


def project_xy_to_latlng(polygon_object_xy, projection_object,
                         false_easting_metres=0, false_northing_metres=0.):
    """Converts polygon from x-y to lat-long coordinates.

    :param polygon_object_xy: Instance of `shapely.geometry.Polygon`, with
        vertices in x-y coordinates.
    :param projection_object: Projection object created by `pyproj.Proj`.
    :param false_easting_metres: False easting.  Will be subtracted from all x-
        coordinates before conversion.
    :param false_northing_metres: False northing.  Will be subtracted from all
        y-coordinates before conversion.
    :return: polygon_object_latlng: Instance of `shapely.geometry.Polygon`, with
        vertices in lat-long coordinates.
    """

    vertex_dict = polygon_object_to_vertex_arrays(polygon_object_xy)
    vertex_dict[EXTERIOR_Y_COLUMN], vertex_dict[EXTERIOR_X_COLUMN] = (
        projections.project_xy_to_latlng(
            vertex_dict[EXTERIOR_X_COLUMN], vertex_dict[EXTERIOR_Y_COLUMN],
            projection_object=projection_object,
            false_easting_metres=false_easting_metres,
            false_northing_metres=false_northing_metres))

    num_holes = len(vertex_dict[HOLE_X_COLUMN])
    for i in range(num_holes):
        vertex_dict[HOLE_Y_COLUMN][i], vertex_dict[HOLE_X_COLUMN][i] = (
            projections.project_xy_to_latlng(
                vertex_dict[HOLE_X_COLUMN][i], vertex_dict[HOLE_Y_COLUMN][i],
                projection_object=projection_object,
                false_easting_metres=false_easting_metres,
                false_northing_metres=false_northing_metres))

    if num_holes == 0:
        polygon_object_latlng = vertex_arrays_to_polygon_object(
            vertex_dict[EXTERIOR_X_COLUMN], vertex_dict[EXTERIOR_Y_COLUMN])
    else:
        polygon_object_latlng = vertex_arrays_to_polygon_object(
            vertex_dict[EXTERIOR_X_COLUMN], vertex_dict[EXTERIOR_Y_COLUMN],
            hole_x_coords_list=vertex_dict[HOLE_X_COLUMN],
            hole_y_coords_list=vertex_dict[HOLE_Y_COLUMN])

    return polygon_object_latlng


def grid_points_in_poly_to_binary_matrix(row_indices, column_indices):
    """Converts list of grid points in polygon to binary image matrix.

    P = number of grid points in polygon
    M = max(row_indices) - min(row_indices) + 3 = number of rows in subgrid
    N = max(column_indices) - min(column_indices) + 3 = number of columns in
        subgrid

    :param row_indices: length-P numpy array with row numbers of grid points in
        polygon.
    :param column_indices: length-P numpy array with column numbers of grid
        points in polygon.
    :return: binary_matrix: M-by-N numpy array of Boolean flags.
        binary_matrix[i, j] indicates whether or not pixel [i, j] -- in the
        subgrid, not necessarily the full grid -- is inside the polygon.
    :return: first_row_index: Same as min(row_indices) - 1.  This can be used
        later to convert row numbers from the subgrid to the full grid.
    :return: first_column_index: Same as min(column_indices) - 1.  This can be
        used later to convert column numbers from the subgrid to the full grid.
    """

    error_checking.assert_is_integer_numpy_array(row_indices)
    error_checking.assert_is_geq_numpy_array(row_indices, 0)
    error_checking.assert_is_numpy_array(row_indices, num_dimensions=1)

    error_checking.assert_is_integer_numpy_array(column_indices)
    error_checking.assert_is_geq_numpy_array(column_indices, 0)
    error_checking.assert_is_numpy_array(column_indices, num_dimensions=1)

    num_rows_in_subgrid = max(row_indices) - min(row_indices) + 3
    num_columns_in_subgrid = max(column_indices) - min(column_indices) + 3

    first_row_index = min(row_indices) - 1
    first_column_index = min(column_indices) - 1
    row_indices_in_subgrid = row_indices - first_row_index
    column_indices_in_subgrid = column_indices - first_column_index

    linear_indices_in_subgrid = numpy.ravel_multi_index(
        (row_indices_in_subgrid, column_indices_in_subgrid),
        (num_rows_in_subgrid, num_columns_in_subgrid))

    binary_vector = numpy.full(num_rows_in_subgrid * num_columns_in_subgrid,
                               False, dtype=bool)
    binary_vector[linear_indices_in_subgrid] = True
    binary_matrix = numpy.reshape(binary_vector,
                                  (num_rows_in_subgrid, num_columns_in_subgrid))

    return binary_matrix, first_row_index, first_column_index


def sort_vertices_counterclockwise(vertex_x_coords, vertex_y_coords):
    """Sorts vertices of a simple polygon in counterclockwise order.

    This method assumes that vertices are already sorted in one of two ways
    (clockwise or counterclockwise), so this method either reverses the order or
    leaves the original.

    V = number of vertices

    :param vertex_x_coords: length-V numpy array with x-coordinates of vertices.
    :param vertex_y_coords: length-V numpy array with y-coordinates of vertices.
    :return: vertex_x_coords: length-V numpy array of x-coordinates in CCW
        order.
    :return: vertex_y_coords: length-V numpy array of y-coordinates in CCW
        order.
    """

    _check_vertex_arrays(vertex_x_coords, vertex_y_coords, allow_nan=False)

    num_vertices = len(vertex_x_coords)
    signed_area = 0.

    for i in range(num_vertices - 1):
        this_x_difference = vertex_x_coords[i + 1] - vertex_x_coords[i]
        this_y_sum = vertex_y_coords[i + 1] + vertex_y_coords[i]
        if this_x_difference == this_y_sum == 0:
            continue

        signed_area = signed_area + (this_x_difference * this_y_sum)

    is_polygon_ccw = signed_area < 0
    if is_polygon_ccw:
        return vertex_x_coords, vertex_y_coords

    return vertex_x_coords[::-1], vertex_y_coords[::-1]


def get_latlng_centroid(latitudes_deg, longitudes_deg):
    """Finds centroid of a set of lat-long points.

    N = number of points

    :param latitudes_deg: length-N numpy array of latitudes (deg N).
    :param longitudes_deg: length-N numpy array of longitudes (deg E).
    :return: centroid_lat_deg: Latitude of centroid (deg N).
    :return: centroid_lng_deg: Longitude of centroid (deg E).
    """

    # TODO(thunderhoser): This method belongs somewhere else, since it can be
    # used for any collection of points (not just polygon vertices).

    return (numpy.mean(latitudes_deg[numpy.invert(numpy.isnan(latitudes_deg))]),
            numpy.mean(
                longitudes_deg[numpy.invert(numpy.isnan(longitudes_deg))]))


def vertex_arrays_to_polygon_object(exterior_x_coords, exterior_y_coords,
                                    hole_x_coords_list=None,
                                    hole_y_coords_list=None):
    """Converts arrays of vertex coords to `shapely.geometry.Polygon` object.

    H = number of holes
    V_e = number of exterior vertices
    V_hi = number of vertices in [i]th hole

    :param exterior_x_coords: numpy array (length V_e) with x-coordinates of
        exterior vertices.
    :param exterior_y_coords: numpy array (length V_e) with y-coordinates of
        exterior vertices.
    :param hole_x_coords_list: length-H list, where the [i]th item is a numpy
        array (length V_hi) with x-coordinates of interior vertices.
    :return: polygon_object: Instance of `shapely.geometry.Polygon`.
    :raises: ValueError: if the resulting polygon is invalid.
    """

    exterior_coords_list = _vertex_arrays_to_list(
        exterior_x_coords, exterior_y_coords)

    if hole_x_coords_list is None:
        return shapely.geometry.Polygon(shell=exterior_coords_list)

    num_holes = len(hole_x_coords_list)
    list_of_hole_coords_lists = []
    for i in range(num_holes):
        list_of_hole_coords_lists.append(_vertex_arrays_to_list(
            hole_x_coords_list[i], hole_y_coords_list[i]))

    polygon_object = shapely.geometry.Polygon(
        shell=exterior_coords_list, holes=tuple(list_of_hole_coords_lists))

    if not polygon_object.is_valid:
        raise ValueError('Resulting polygon is invalid.')

    return polygon_object


def polygon_object_to_vertex_arrays(polygon_object):
    """Converts `shapely.geometry.Polygon` object to arrays of vertex coords.

    H = number of holes
    V_e = number of exterior vertices
    V_hi = number of vertices in [i]th hole

    :param polygon_object: Instance of `shapely.geometry.Polygon`.
    :return: vertex_dict: Dictionary with the following keys.
    vertex_dict.exterior_x_coords: numpy array (length V_e) with x-coordinates
        of exterior vertices.
    vertex_dict.exterior_y_coords: numpy array (length V_e) with y-coordinates
        of exterior vertices.
    vertex_dict.hole_x_coords_list: length-H list, where the [i]th item is a
        numpy array (length V_hi) with x-coordinates of interior vertices.
    vertex_dict.hole_y_coords_list: Same as above, except for y-coordinates.
    """

    exterior_x_coords, exterior_y_coords = _vertex_list_to_arrays(
        list(polygon_object.exterior.coords))

    num_holes = len(polygon_object.interiors)
    if num_holes == 0:
        return {EXTERIOR_X_COLUMN: exterior_x_coords,
                EXTERIOR_Y_COLUMN: exterior_y_coords,
                HOLE_X_COLUMN: [], HOLE_Y_COLUMN: []}

    hole_x_coords_list = []
    hole_y_coords_list = []
    for i in range(num_holes):
        (this_hole_x_coords, this_hole_y_coords) = _vertex_list_to_arrays(
            list(polygon_object.interiors[i].coords))

        hole_x_coords_list.append(this_hole_x_coords)
        hole_y_coords_list.append(this_hole_y_coords)

    return {EXTERIOR_X_COLUMN: exterior_x_coords,
            EXTERIOR_Y_COLUMN: exterior_y_coords,
            HOLE_X_COLUMN: hole_x_coords_list,
            HOLE_Y_COLUMN: hole_y_coords_list}


def grid_points_in_poly_to_vertices(grid_point_row_indices,
                                    grid_point_column_indices):
    """Converts list of grid points in polygon to list of vertices.

    The resulting vertices follow grid-cell edges, rather than cutting through
    grid cells.  Vertices are sorted in counterclockwise order.

    If there are disjoint polygons, this method returns the longest polygon
    (that with the most vertices).  If there are holes inside the longest
    polygon, this method removes the holes.  In other words, this method always
    returns a simple polygon and tries to return the largest one.

    P = number of grid points in polygon
    V = number of vertices

    :param grid_point_row_indices: length-P numpy array with row numbers
        (integers) of grid points in polygon.
    :param grid_point_column_indices: length-P numpy array with column numbers
        (integers) of grid points in polygon.
    :return: vertex_row_indices: length-V numpy array with row numbers
        (half-integers) of vertices.
    :return: vertex_column_indices: length-V numpy array with column numbers
        (half-integers) of vertices.
    """

    error_checking.assert_is_integer_numpy_array(grid_point_row_indices)
    error_checking.assert_is_numpy_array_without_nan(grid_point_row_indices)
    error_checking.assert_is_numpy_array(
        grid_point_row_indices, num_dimensions=1)
    num_grid_points = len(grid_point_row_indices)

    error_checking.assert_is_integer_numpy_array(grid_point_column_indices)
    error_checking.assert_is_numpy_array_without_nan(grid_point_column_indices)
    error_checking.assert_is_numpy_array(
        grid_point_column_indices,
        exact_dimensions=numpy.array([num_grid_points]))

    (binary_matrix, first_row_index, first_column_index) = (
        grid_points_in_poly_to_binary_matrix(
            grid_point_row_indices, grid_point_column_indices))
    binary_matrix = _patch_diag_connections_in_binary_matrix(binary_matrix)

    _, contour_list, _ = cv2.findContours(
        binary_matrix.astype(numpy.uint8), cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE)

    contour_matrix = _get_longest_inner_list(contour_list)
    contour_matrix = numpy.array(contour_matrix)[:, 0, :]
    num_contour_points = contour_matrix.shape[0]

    num_vertices = num_contour_points + 1
    vertex_row_indices = numpy.full(num_vertices, numpy.nan)
    vertex_column_indices = numpy.full(num_vertices, numpy.nan)

    for i in range(num_vertices):
        if i == num_vertices - 1:
            vertex_row_indices[i] = contour_matrix[0, 1]
            vertex_column_indices[i] = contour_matrix[0, 0]
        else:
            vertex_row_indices[i] = contour_matrix[i, 1]
            vertex_column_indices[i] = contour_matrix[i, 0]

    vertex_row_indices += first_row_index
    vertex_column_indices += first_column_index
    vertex_row_indices, vertex_column_indices = (
        _vertices_from_grid_points_to_edges(
            vertex_row_indices.astype(int), vertex_column_indices.astype(int)))
    return _remove_redundant_vertices(vertex_row_indices, vertex_column_indices)


def simple_polygon_to_grid_points(vertex_row_indices, vertex_column_indices):
    """Finds grid points in simple polygon.

    V = number of vertices
    P = number of grid points in polygon

    :param vertex_row_indices: length-V numpy array with row numbers
        (half-integers) of vertices.
    :param vertex_column_indices: length-V numpy array with column numbers
        (half-integers) of vertices.
    :return: grid_point_row_indices: length-P numpy array with row numbers
        (integers) of grid points in polygon.
    :return: grid_point_column_indices: length-P numpy array with column numbers
        (integers) of grid points in polygon.
    """

    polygon_object = vertex_arrays_to_polygon_object(
        vertex_column_indices, vertex_row_indices)

    min_grid_point_row = numpy.floor(numpy.min(vertex_row_indices))
    max_grid_point_row = numpy.ceil(numpy.max(vertex_row_indices))
    num_grid_point_rows = max_grid_point_row - min_grid_point_row + 1
    min_grid_point_column = numpy.floor(numpy.min(vertex_column_indices))
    max_grid_point_column = numpy.ceil(numpy.max(vertex_column_indices))
    num_grid_point_columns = max_grid_point_column - min_grid_point_column + 1

    unique_grid_point_rows = numpy.linspace(
        min_grid_point_row, max_grid_point_row, num=num_grid_point_rows,
        dtype=int)
    unique_grid_point_columns = numpy.linspace(
        min_grid_point_column, max_grid_point_column,
        num=num_grid_point_columns, dtype=int)

    (grid_point_column_matrix,
     grid_point_row_matrix) = grids.xy_vectors_to_matrices(
         unique_grid_point_columns, unique_grid_point_rows)

    grid_point_row_vector = numpy.reshape(grid_point_row_matrix,
                                          grid_point_row_matrix.size)
    grid_point_column_vector = numpy.reshape(grid_point_column_matrix,
                                             grid_point_column_matrix.size)

    num_grid_points = len(grid_point_row_vector)
    in_polygon_flags = numpy.full(num_grid_points, False, dtype=bool)
    for i in range(num_grid_points):
        in_polygon_flags[i] = is_point_in_or_on_polygon(
            polygon_object, query_x_coordinate=grid_point_column_vector[i],
            query_y_coordinate=grid_point_row_vector[i])

    in_polygon_indices = numpy.where(in_polygon_flags)[0]
    return (grid_point_row_vector[in_polygon_indices],
            grid_point_column_vector[in_polygon_indices])


def fix_probsevere_vertices(row_indices_orig, column_indices_orig):
    """Fixes vertices of storm object generated by probSevere.

    Specifically, this method moves vertices from grid points to grid-cell
    edges.  This ensures that vertices follow the outlines of grid cells, rather
    than cutting through grid cells.

    v = original number of vertices
    V = final number of vertices

    :param row_indices_orig: length-v numpy array with row numbers (integers) of
        original vertices.
    :param column_indices_orig: length-v numpy array with column numbers
        (integers) of original vertices.
    :return: row_indices: length-V numpy array with row numbers (half-integers)
        of new vertices.
    :return: column_indices: length-V numpy array with column numbers (half-
        integers) of new vertices.
    """

    column_indices_orig, row_indices_orig = (
        _get_longest_vertex_arrays_without_nan(
            column_indices_orig, row_indices_orig))

    if (row_indices_orig[0] != row_indices_orig[-1] or
            column_indices_orig[0] != column_indices_orig[-1]):
        row_indices_orig = numpy.concatenate((
            row_indices_orig, numpy.array([row_indices_orig[0]])))
        column_indices_orig = numpy.concatenate((
            column_indices_orig, numpy.array([column_indices_orig[0]])))

    (column_indices_orig, row_indices_orig) = sort_vertices_counterclockwise(
        column_indices_orig, -1 * row_indices_orig)
    row_indices_orig *= -1

    row_indices, column_indices = _vertices_from_grid_points_to_edges(
        row_indices_orig.astype(int), column_indices_orig.astype(int))
    return _remove_redundant_vertices(row_indices, column_indices)


def is_point_in_or_on_polygon(polygon_object=None, query_x_coordinate=None,
                              query_y_coordinate=None):
    """Returns True if point is inside/touching the polygon, False otherwise.

    x- and y-coordinates may be in one of the formats listed at the top.
    However, coordinates in the first three input args must all be in the same
    format.

    :param polygon_object: Instance of `shapely.geometry.Polygon`.
    :param query_x_coordinate: x-coordinate of query point.
    :param query_y_coordinate: y-coordinate of query point.
    :return: point_in_or_on_poly_flag: Boolean flag.  True if point is
        inside/touching the polygon, False otherwise.
    """

    error_checking.assert_is_not_nan(query_x_coordinate)
    error_checking.assert_is_not_nan(query_y_coordinate)

    point_object = shapely.geometry.Point(
        query_x_coordinate, query_y_coordinate)
    if polygon_object.contains(point_object):
        return True

    return polygon_object.touches(point_object)


def buffer_simple_polygon(orig_vertex_x_metres, orig_vertex_y_metres,
                          min_buffer_dist_metres=numpy.nan,
                          max_buffer_dist_metres=None, preserve_angles=False):
    """Creates buffer around simple polygon.

    v = number of vertices in original polygon
    V = number of vertices in buffer

    :param orig_vertex_x_metres: length-v numpy array with x-coordinates of
        vertices.
    :param orig_vertex_y_metres: length-v numpy array with y-coordinates of
        vertices.
    :param min_buffer_dist_metres: Minimum buffer distance.  If defined, the
        original polygon will *not* be included in the buffer (example: if you
        want "0-5 km outside the storm," make min_buffer_dist_metres = 0; if you
        want "5-10 km outside the storm," make min_buffer_dist_metres = 5000).
        If NaN, the original polygon *will* be included in the buffer (example:
        if you want "within 5 km of the storm" or "within 10 km of the storm,"
        make min_buffer_dist_metres = NaN).
    :param max_buffer_dist_metres: Maximum buffer distance.
    :param preserve_angles: Boolean flag.  If True, will preserve the angles of
        all vertices.  However, this means that buffer distances will not be
        strictly respected.  It is highly recommended that you leave this as
        False.
    :return: buffer_polygon_object: Instance of `shapely.geometry.polygon`.
    """

    _check_vertex_arrays(
        orig_vertex_x_metres, orig_vertex_y_metres, allow_nan=False)
    error_checking.assert_is_real_number(min_buffer_dist_metres)
    error_checking.assert_is_geq(min_buffer_dist_metres, 0., allow_nan=True)

    error_checking.assert_is_not_nan(max_buffer_dist_metres)
    if not numpy.isnan(min_buffer_dist_metres):
        error_checking.assert_is_greater(
            max_buffer_dist_metres, min_buffer_dist_metres)

    error_checking.assert_is_boolean(preserve_angles)

    if preserve_angles:
        join_style = shapely.geometry.JOIN_STYLE.mitre
    else:
        join_style = shapely.geometry.JOIN_STYLE.round

    orig_polygon_object = vertex_arrays_to_polygon_object(
        orig_vertex_x_metres, orig_vertex_y_metres)
    max_buffer_polygon_object = orig_polygon_object.buffer(
        max_buffer_dist_metres, join_style=join_style)

    if numpy.isnan(min_buffer_dist_metres):
        return max_buffer_polygon_object

    min_buffer_polygon_object = orig_polygon_object.buffer(
        min_buffer_dist_metres, join_style=join_style)
    min_buffer_vertex_dict = polygon_object_to_vertex_arrays(
        min_buffer_polygon_object)
    max_buffer_vertex_dict = polygon_object_to_vertex_arrays(
        max_buffer_polygon_object)

    return vertex_arrays_to_polygon_object(
        max_buffer_vertex_dict[EXTERIOR_X_COLUMN],
        max_buffer_vertex_dict[EXTERIOR_Y_COLUMN],
        hole_x_coords_list=[min_buffer_vertex_dict[EXTERIOR_X_COLUMN]],
        hole_y_coords_list=[min_buffer_vertex_dict[EXTERIOR_Y_COLUMN]])
