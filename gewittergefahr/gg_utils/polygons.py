"""Methods for handling polygons.

In general, x- and y- coordinates may be in one of three formats:

[1] Metres.
[2] Longitude (deg E) and latitude (deg N), respectively.
[3] Columns and rows in a grid, respectively.
"""

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
COMPLEX_DIRECTION_NAMES = [UP_RIGHT_DIRECTION_NAME, UP_LEFT_DIRECTION_NAME,
                           DOWN_RIGHT_DIRECTION_NAME, DOWN_LEFT_DIRECTION_NAME]

EXTERIOR_X_COLUMN = 'exterior_x_coords'
EXTERIOR_Y_COLUMN = 'exterior_y_coords'
HOLE_X_COLUMN = 'hole_x_coords_list'
HOLE_Y_COLUMN = 'hole_y_coords_list'


def _check_vertex_arrays(x_coordinates, y_coordinates, allow_nan=True):
    """Checks vertex arrays for errors.

    x- and y-coordinates may be in one of three formats (see docstring at top of
    file).

    V = number of vertices

    :param x_coordinates: length-V numpy array with x-coordinates of vertices.
        The first NaN separates the exterior from the first hole, and the [i]th
        NaN separates the [i - 1]th hole from the [i]th hole.
    :param y_coordinates: Same as above, except for y-coordinates.
    :param allow_nan: Boolean flag.  If True, input arrays may contain NaN's
        (however, NaN's must occur at the exact same positions in the two
        arrays).
    :raises: ValueError: if allow_nan = True but NaN's do not occur at the same
        positions in the two arrays.
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
            '\nNaN''s occur at the following positions in `x_coordinates`:\n' +
            str(x_nan_indices) +
            '\nand the following positions in `y_coordinates`:\n' +
            str(y_nan_indices) +
            '\nNaN''s should occur at the same positions in the two arrays.')
        raise ValueError(error_string)


def _get_longest_inner_list(outer_list):
    """Finds longest inner list.

    :param outer_list: 1-D list of 1-D lists.
    :return: longest_inner_list: Longest inner list.
    """

    num_outer_lists = len(outer_list)
    lengths_of_inner_lists = numpy.full(num_outer_lists, 0, dtype=int)
    for i in range(num_outer_lists):
        lengths_of_inner_lists[i] = len(outer_list[i])

    return outer_list[numpy.argmax(lengths_of_inner_lists)]


def _get_longest_simple_polygon(vertex_x_coords, vertex_y_coords):
    """Finds longest simple polygon (that with the most vertices).

    x- and y-coordinates may be in one of three formats (see docstring at top of
    file).

    V = number of vertices

    :param vertex_x_coords: See documentation for _check_vertex_arrays.
    :param vertex_y_coords: See documentation for _check_vertex_arrays.
    :return: vertex_x_coords_simple: Longest subsequence of `vertex_x_coords`
        without NaN.
    :return: vertex_y_coords_simple: Longest subsequence of `vertex_y_coords`
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
    """Converts vertices of simple polygon from two arrays to one list.

    x- and y-coordinates may be in one of three formats (see docstring at top of
    file).

    V = number of vertices

    :param vertex_x_coords: See documentation for _check_vertex_arrays.
    :param vertex_y_coords: See documentation for _check_vertex_arrays.
    :return: vertex_coords_as_list: length-V list, where each element is an
        (x, y) tuple.
    """

    _check_vertex_arrays(vertex_x_coords, vertex_y_coords, allow_nan=False)

    num_vertices = len(vertex_x_coords)
    vertex_coords_as_list = []
    for i in range(num_vertices):
        vertex_coords_as_list.append((vertex_x_coords[i], vertex_y_coords[i]))

    return vertex_coords_as_list


def _get_edge_direction(first_row, second_row, first_column, second_column):
    """Finds direction of polygon edge.

    This method assumes that row number increases downward and column number
    increases rightward.

    There are 8 possible directions: up, down, left, right, up-left, up-right,
    down-left, down-right.

    :param first_row: Row number of first vertex (start point of edge).
    :param second_row: Row number of second vertex (end point of edge).
    :param first_column: Column number of first vertex.
    :param second_column: Column number of second vertex.
    :return: direction_string: String indicating direction of edge.  May be
        "up", "down", "left", "right", "up_left", "up_right", "down_left", or
        "down_right".
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


def _remove_redundant_vertices(row_indices_orig, column_indices_orig):
    """Removes redundant vertices from a simple polygon.

    V_0 = number of original vertices
    V = number of final vertices

    :param row_indices_orig: numpy array (length V_0) with row numbers of
        original vertices.
    :param column_indices_orig: numpy array (length V_0) with column numbers of
        original vertices.
    :return: row_indices: length-V numpy array with row numbers of final (non-
        redundant) vertices.
    :return: column_indices: length-V numpy array with column numbers of final
        (non-redundant) vertices.
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
    """Patches diagonal connections in binary image matrices.

    When two pixels (p and q) are connected only diagonally, this method
    "patches" the connection by adding another pixel -- adjacent to both p and q
    -- to the image.  In other words, this method flips one bit of the image
    from False to True.

    If diagonal connections are not patched, grid_points_in_poly_to_vertices
    will create disjoint polygons, rather than one simple polygon.

    M = number of rows in binary image
    N = number of columns in binary image

    :param binary_matrix: M-by-N numpy array of Boolean flags.  If
        binary_matrix[i, j] = True, pixel [i, j] is inside the polygon.
    :return: binary_matrix: Same as input, except that diagonal connections are
        patched.
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


def _binary_matrix_to_grid_points_in_poly(
        binary_matrix, first_row_index, first_column_index):
    """Converts binary image matrix to list of grid points in polygon.

    M = number of rows in binary image
    N = number of columns in binary image
    P = number of grid points in polygon.

    :param binary_matrix: M-by-N numpy array of Boolean flags.  If
        binary_matrix[i, j] = True, pixel [i, j] is inside the polygon.
    :param first_row_index: Used to convert row numbers from the binary image
        (which spans only a subgrid) to the full grid.  Row 0 in the subgrid =
        row `first_row_index` in the full grid.
    :param first_column_index: Same as above, but for column numbers.
    :return: row_indices: length-P numpy array with row numbers (relative to the
        full grid) of grid points in polygon.
    :return: column_indices: Same as above, but for column numbers.
    """

    num_rows_in_subgrid = binary_matrix.shape[0]
    num_columns_in_subgrid = binary_matrix.shape[1]
    binary_vector = numpy.reshape(
        binary_matrix, num_rows_in_subgrid * num_columns_in_subgrid)

    linear_indices_in_subgrid = numpy.where(binary_vector)[0]
    (row_indices_in_subgrid, column_indices_in_subgrid) = numpy.unravel_index(
        linear_indices_in_subgrid,
        (num_rows_in_subgrid, num_columns_in_subgrid))

    return (row_indices_in_subgrid + first_row_index,
            column_indices_in_subgrid + first_column_index)


def _vertices_from_grid_points_to_edges(row_indices_orig, column_indices_orig):
    """Moves vertices of simple polygon from grid points to grid-cell edges.

    This method ensures that vertices follow the outlines of grid cells, rather
    than cutting through grid cells.

    V_0 = number of original vertices
    V = number of final vertices

    :param row_indices_orig: numpy array (length V_0) with row numbers
        (integers) of original vertices.
    :param column_indices_orig: numpy array (length V_0) with column numbers
        (integers) of original vertices.
    :return: row_indices: length-V numpy array with row numbers (half-integers)
        of final vertices.
    :return: column_indices: length-V numpy array with column numbers (half-
        integers) of final vertices.
    """

    error_checking.assert_is_integer_numpy_array(row_indices_orig)
    error_checking.assert_is_integer_numpy_array(column_indices_orig)

    num_vertices_orig = len(row_indices_orig)
    row_indices = numpy.array([])
    column_indices = numpy.array([])

    # Handle case of only one unique vertex.
    rowcol_matrix_orig = numpy.hstack((
        numpy.reshape(row_indices_orig, (num_vertices_orig, 1)),
        numpy.reshape(column_indices_orig, (num_vertices_orig, 1))))
    unique_rowcol_matrix_orig = numpy.vstack(
        {tuple(this_row) for this_row in rowcol_matrix_orig}).astype(int)

    if unique_rowcol_matrix_orig.shape[0] == 1:
        row_indices = row_indices_orig[0] + numpy.array(
            [0.5, 0.5, -0.5, -0.5, 0.5])
        column_indices = column_indices_orig[0] + numpy.array(
            [-0.5, 0.5, 0.5, -0.5, -0.5])
        return row_indices, column_indices

    for i in range(num_vertices_orig - 1):
        this_direction_name = _get_edge_direction(
            first_row=row_indices_orig[i], second_row=row_indices_orig[i + 1],
            first_column=column_indices_orig[i],
            second_column=column_indices_orig[i + 1])

        if this_direction_name == UP_DIRECTION_NAME:
            rows_to_append = numpy.array(
                [row_indices_orig[i] + 0.5, row_indices_orig[i + 1] - 0.5])
            columns_to_append = numpy.array(
                [column_indices_orig[i] + 0.5,
                 column_indices_orig[i + 1] + 0.5])

        elif this_direction_name == DOWN_DIRECTION_NAME:
            rows_to_append = numpy.array(
                [row_indices_orig[i] - 0.5, row_indices_orig[i + 1] + 0.5])
            columns_to_append = numpy.array(
                [column_indices_orig[i] - 0.5,
                 column_indices_orig[i + 1] - 0.5])

        elif this_direction_name == RIGHT_DIRECTION_NAME:
            rows_to_append = numpy.array(
                [row_indices_orig[i] + 0.5, row_indices_orig[i + 1] + 0.5])
            columns_to_append = numpy.array(
                [column_indices_orig[i] - 0.5,
                 column_indices_orig[i + 1] + 0.5])

        elif this_direction_name == LEFT_DIRECTION_NAME:
            rows_to_append = numpy.array(
                [row_indices_orig[i] - 0.5, row_indices_orig[i + 1] - 0.5])
            columns_to_append = numpy.array(
                [column_indices_orig[i] + 0.5,
                 column_indices_orig[i + 1] - 0.5])

        else:
            rows_to_append, columns_to_append = (
                _vertices_from_grid_points_to_edges_complex_direction(
                    first_row=row_indices_orig[i],
                    second_row=row_indices_orig[i + 1],
                    first_column=column_indices_orig[i],
                    second_column=column_indices_orig[i + 1]))

        row_indices = numpy.concatenate((row_indices, rows_to_append))
        column_indices = numpy.concatenate((column_indices, columns_to_append))

    if not (row_indices[-1] == row_indices[0] and
            column_indices[-1] == column_indices[0]):
        row_indices = numpy.concatenate((row_indices, row_indices[[0]]))
        column_indices = numpy.concatenate(
            (column_indices, column_indices[[0]]))

    return row_indices, column_indices


def _vertices_from_grid_points_to_edges_complex_direction(
        first_row, second_row, first_column, second_column):
    """Moves vertices along one polygon edge from grid pts to grid-cell edges.

    This method should be used only when the polygon edge has a complex
    direction (up-left, up-right, down-left, or down-right).

    V = number of vertices after moving from grid points to grid-cell edges

    :param first_row: Row number of first vertex (integer).
    :param second_row: Row number of second vertex (integer).
    :param first_column: Column number of first vertex (integer).
    :param second_column: Column number of second vertex (integer).
    :return: new_rows: length-V numpy array of row numbers (half-integers).
    :return: new_columns: length-V numpy array of column numbers (half-
        integers).
    """

    direction_name = _get_edge_direction(
        first_row=first_row, second_row=second_row, first_column=first_column,
        second_column=second_column)

    absolute_row_diff = numpy.absolute(second_row - first_row)
    absolute_column_diff = numpy.absolute(second_column - first_column)
    num_steps = int(min([absolute_row_diff, absolute_column_diff]))

    row_indices_orig = numpy.linspace(
        float(first_row), float(second_row), num=num_steps + 1)
    column_indices_orig = numpy.linspace(
        float(first_column), float(second_column), num=num_steps + 1)

    integer_row_flags = numpy.isclose(
        row_indices_orig, numpy.round(row_indices_orig), atol=TOLERANCE)
    integer_column_flags = numpy.isclose(
        column_indices_orig, numpy.round(column_indices_orig), atol=TOLERANCE)
    valid_flags = numpy.logical_and(integer_row_flags, integer_column_flags)

    valid_indices = numpy.where(valid_flags)[0]
    row_indices_orig = row_indices_orig[valid_indices]
    column_indices_orig = column_indices_orig[valid_indices]
    num_steps = len(row_indices_orig) - 1

    new_rows = numpy.array([])
    new_columns = numpy.array([])

    for j in range(num_steps):
        if direction_name == UP_RIGHT_DIRECTION_NAME:
            these_new_rows = numpy.array(
                [row_indices_orig[j] + 0.5, row_indices_orig[j + 1] + 0.5,
                 row_indices_orig[j + 1] + 0.5])
            these_new_columns = numpy.array(
                [column_indices_orig[j] + 0.5, column_indices_orig[j] + 0.5,
                 column_indices_orig[j + 1] + 0.5])

        elif direction_name == UP_LEFT_DIRECTION_NAME:
            these_new_rows = numpy.array(
                [row_indices_orig[j] - 0.5, row_indices_orig[j] - 0.5,
                 row_indices_orig[j + 1] - 0.5])
            these_new_columns = numpy.array(
                [column_indices_orig[j] + 0.5, column_indices_orig[j + 1] + 0.5,
                 column_indices_orig[j + 1] + 0.5])

        elif direction_name == DOWN_RIGHT_DIRECTION_NAME:
            these_new_rows = numpy.array(
                [row_indices_orig[j] + 0.5, row_indices_orig[j] + 0.5,
                 row_indices_orig[j + 1] + 0.5])
            these_new_columns = numpy.array(
                [column_indices_orig[j] - 0.5, column_indices_orig[j + 1] - 0.5,
                 column_indices_orig[j + 1] - 0.5])

        elif direction_name == DOWN_LEFT_DIRECTION_NAME:
            these_new_rows = numpy.array(
                [row_indices_orig[j] - 0.5, row_indices_orig[j + 1] - 0.5,
                 row_indices_orig[j + 1] - 0.5])
            these_new_columns = numpy.array(
                [column_indices_orig[j] - 0.5, column_indices_orig[j] - 0.5,
                 column_indices_orig[j + 1] - 0.5])

        new_rows = numpy.concatenate((new_rows, these_new_rows))
        new_columns = numpy.concatenate((new_columns, these_new_columns))

    return new_rows, new_columns


def project_latlng_to_xy(
        polygon_object_latlng, projection_object=None, false_easting_metres=0,
        false_northing_metres=0.):
    """Converts polygon from lat-long to x-y coordinates.

    :param polygon_object_latlng: `shapely.geometry.Polygon` object with
        vertices in lat-long coordinates.
    :param projection_object: `pyproj.Proj` object.  If None, this method will
        create an azimuthal equidistant projection centered at the polygon
        centroid.
    :param false_easting_metres: False easting (will be added to all x-
        coordinates).
    :param false_northing_metres: False northing (will be added to all y-
        coordinates).
    :return: polygon_object_xy_metres: `shapely.geometry.Polygon` object with
        vertices in x-y coordinates.
    :return: projection_object: `pyproj.Proj` object.  If input was defined,
        this is simply the input object.  If input was None, this is the object
        created on the fly.
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


def project_xy_to_latlng(
        polygon_object_xy_metres, projection_object, false_easting_metres=0,
        false_northing_metres=0.):
    """Converts polygon from x-y to lat-long coordinates.

    :param polygon_object_xy_metres: `shapely.geometry.Polygon` object with
        vertices in x-y coordinates.
    :param projection_object: `pyproj.Proj` object.  Will be used to convert
        coordinates.
    :param false_easting_metres: False easting (will be subtracted from all x-
        coordinates before converting).
    :param false_northing_metres: False northing (will be subtracted from all y-
        coordinates before converting).
    :return: polygon_object_latlng: `shapely.geometry.Polygon` object with
        vertices in lat-long coordinates.
    """

    vertex_dict = polygon_object_to_vertex_arrays(polygon_object_xy_metres)
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
    """Converts list of grid points in polygon to binary image.

    P = number of grid points in polygon
    M = max(row_indices) - min(row_indices) + 3 = number of rows in binary image
    N = max(column_indices) - min(column_indices) + 3 = number of columns in
        binary image

    :param row_indices: length-P numpy array with row numbers (integers) of grid
        points in polygon.
    :param column_indices: length-P numpy array with column numbers (integers)
        of grid points in polygon.
    :return: binary_matrix: M-by-N numpy array of Boolean flags.  If
        binary_matrix[i, j] = True, pixel [i, j] is inside the polygon.
    :return: first_row_index: Used to convert row numbers from the binary image
        (which spans only a subgrid) to the full grid.  Row 0 in the subgrid =
        row `first_row_index` in the full grid.
    :return: first_column_index: Same as above, but for column numbers.
    """

    error_checking.assert_is_integer_numpy_array(row_indices)
    error_checking.assert_is_geq_numpy_array(row_indices, 0)
    error_checking.assert_is_numpy_array(row_indices, num_dimensions=1)
    num_points_in_polygon = len(row_indices)

    error_checking.assert_is_integer_numpy_array(column_indices)
    error_checking.assert_is_geq_numpy_array(column_indices, 0)
    error_checking.assert_is_numpy_array(
        column_indices, exact_dimensions=numpy.array([num_points_in_polygon]))

    num_rows_in_subgrid = max(row_indices) - min(row_indices) + 3
    num_columns_in_subgrid = max(column_indices) - min(column_indices) + 3

    first_row_index = min(row_indices) - 1
    first_column_index = min(column_indices) - 1
    row_indices_in_subgrid = row_indices - first_row_index
    column_indices_in_subgrid = column_indices - first_column_index

    linear_indices_in_subgrid = numpy.ravel_multi_index(
        (row_indices_in_subgrid, column_indices_in_subgrid),
        (num_rows_in_subgrid, num_columns_in_subgrid))

    binary_vector = numpy.full(
        num_rows_in_subgrid * num_columns_in_subgrid, False, dtype=bool)
    binary_vector[linear_indices_in_subgrid] = True
    binary_matrix = numpy.reshape(
        binary_vector, (num_rows_in_subgrid, num_columns_in_subgrid))

    return binary_matrix, first_row_index, first_column_index


def sort_counterclockwise(vertex_x_coords, vertex_y_coords):
    """Sorts vertices of a simple polygon in counterclockwise order.

    This method assumes that vertices are already sorted either clockwise or
    counterclockwise.  Thus, this method decides whether to leave the order as
    is or reverse it.

    V = number of vertices

    :param vertex_x_coords: See documentation for _check_vertex_arrays.
    :param vertex_y_coords: See documentation for _check_vertex_arrays.
    :return: vertex_x_coords: Same as input, except in counterclockwise order.
    :return: vertex_y_coords: Same as input, except in counterclockwise order.
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

    if signed_area < 0:
        return vertex_x_coords, vertex_y_coords
    return vertex_x_coords[::-1], vertex_y_coords[::-1]


def vertex_arrays_to_polygon_object(
        exterior_x_coords, exterior_y_coords, hole_x_coords_list=None,
        hole_y_coords_list=None):
    """Converts polygon from vertex arrays to `shapely.geometry.Polygon` object.

    V_e = number of exterior vertices
    H = number of holes
    V_hi = number of vertices in [i]th hole

    :param exterior_x_coords: numpy array (length V_e) with x-coordinates of
        exterior vertices.
    :param exterior_y_coords: numpy array (length V_e) with y-coordinates of
        exterior vertices.
    :param hole_x_coords_list: length-H list, where the [i]th item is a numpy
        array (length V_hi) with x-coordinates of interior vertices.
    :param hole_y_coords_list: Same as above, except for y-coordinates.
    :return: polygon_object: `shapely.geometry.Polygon` object.
    :raises: ValueError: if the polygon is invalid.
    """

    exterior_coords_as_list = _vertex_arrays_to_list(
        exterior_x_coords, exterior_y_coords)
    if hole_x_coords_list is None:
        return shapely.geometry.Polygon(shell=exterior_coords_as_list)

    num_holes = len(hole_x_coords_list)
    outer_list_of_hole_coords = []
    for i in range(num_holes):
        outer_list_of_hole_coords.append(_vertex_arrays_to_list(
            hole_x_coords_list[i], hole_y_coords_list[i]))

    polygon_object = shapely.geometry.Polygon(
        shell=exterior_coords_as_list, holes=tuple(outer_list_of_hole_coords))

    if not polygon_object.is_valid:
        raise ValueError('Resulting polygon is invalid.')

    return polygon_object


def polygon_object_to_vertex_arrays(polygon_object):
    """Converts polygon from `shapely.geometry.Polygon` object to vertex arrays.

    V_e = number of exterior vertices
    H = number of holes
    V_hi = number of vertices in [i]th hole

    :param polygon_object: `shapely.geometry.Polygon` object.
    :return: vertex_dict: Dictionary with the following keys.
    vertex_dict['exterior_x_coords']: numpy array (length V_e) with
        x-coordinates of exterior vertices.
    vertex_dict['exterior_y_coords']: numpy array (length V_e) with
        y-coordinates of exterior vertices.
    vertex_dict['hole_x_coords_list']: length-H list, where the [i]th item is a
        numpy array (length V_hi) with x-coordinates of interior vertices.
    vertex_dict['hole_y_coords_list']: Same as above, except for y-coordinates.
    """

    num_holes = len(polygon_object.interiors)
    hole_x_coords_list = []
    hole_y_coords_list = []

    for i in range(num_holes):
        hole_x_coords_list.append(
            numpy.array(polygon_object.interiors[i].xy[0]))
        hole_y_coords_list.append(
            numpy.array(polygon_object.interiors[i].xy[1]))

    return {EXTERIOR_X_COLUMN: numpy.array(polygon_object.exterior.xy[0]),
            EXTERIOR_Y_COLUMN: numpy.array(polygon_object.exterior.xy[1]),
            HOLE_X_COLUMN: hole_x_coords_list,
            HOLE_Y_COLUMN: hole_y_coords_list}


def grid_points_in_poly_to_vertices(
        grid_point_row_indices, grid_point_column_indices):
    """Converts list of grid points in polygon to vertices.

    The resulting vertices are sorted counterclockwise and follow grid-cell
    edges (rather than cutting through grid cells).

    This method returns a simple polygon.  If the list of grid points defines
    several disjoint polygons, this method returns the longest one (that with
    the most vertices).  Also, this method removes holes from the interior of
    said polygon.

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
    error_checking.assert_is_geq_numpy_array(grid_point_row_indices, 0)
    error_checking.assert_is_numpy_array(
        grid_point_row_indices, num_dimensions=1)
    num_grid_points = len(grid_point_row_indices)

    error_checking.assert_is_integer_numpy_array(grid_point_column_indices)
    error_checking.assert_is_geq_numpy_array(grid_point_column_indices, 0)
    error_checking.assert_is_numpy_array(
        grid_point_column_indices,
        exact_dimensions=numpy.array([num_grid_points]))

    binary_matrix, first_row_index, first_column_index = (
        grid_points_in_poly_to_binary_matrix(
            grid_point_row_indices, grid_point_column_indices))
    binary_matrix = _patch_diag_connections_in_binary_matrix(binary_matrix)

    print numpy.sum(binary_matrix)

    if numpy.sum(binary_matrix) == 1:
        vertex_row_indices, vertex_column_indices = numpy.where(binary_matrix)
    else:
        _, contour_list, _ = cv2.findContours(
            binary_matrix.astype(numpy.uint8), cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE)

        contour_matrix = _get_longest_inner_list(contour_list)
        contour_matrix = numpy.array(contour_matrix)[:, 0, :]

        num_vertices = contour_matrix.shape[0] + 1
        vertex_row_indices = numpy.full(num_vertices, -1, dtype=int)
        vertex_column_indices = numpy.full(num_vertices, -1, dtype=int)

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
            vertex_row_indices, vertex_column_indices))

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

    min_grid_point_row = int(numpy.floor(numpy.min(vertex_row_indices)))
    max_grid_point_row = int(numpy.ceil(numpy.max(vertex_row_indices)))
    num_grid_point_rows = max_grid_point_row - min_grid_point_row + 1
    grid_point_rows = numpy.linspace(
        min_grid_point_row, max_grid_point_row, num=num_grid_point_rows,
        dtype=int)

    min_grid_point_column = int(numpy.floor(numpy.min(vertex_column_indices)))
    max_grid_point_column = int(numpy.ceil(numpy.max(vertex_column_indices)))
    num_grid_point_columns = max_grid_point_column - min_grid_point_column + 1
    grid_point_columns = numpy.linspace(
        min_grid_point_column, max_grid_point_column,
        num=num_grid_point_columns, dtype=int)

    grid_point_column_matrix, grid_point_row_matrix = (
        grids.xy_vectors_to_matrices(grid_point_columns, grid_point_rows))
    grid_point_row_vector = numpy.reshape(
        grid_point_row_matrix, grid_point_row_matrix.size)
    grid_point_column_vector = numpy.reshape(
        grid_point_column_matrix, grid_point_column_matrix.size)

    num_grid_points = len(grid_point_row_vector)
    in_polygon_flags = numpy.full(num_grid_points, False, dtype=bool)
    for i in range(num_grid_points):
        in_polygon_flags[i] = point_in_or_on_polygon(
            polygon_object, query_x_coordinate=grid_point_column_vector[i],
            query_y_coordinate=grid_point_row_vector[i])

    in_polygon_indices = numpy.where(in_polygon_flags)[0]
    return (grid_point_row_vector[in_polygon_indices],
            grid_point_column_vector[in_polygon_indices])


def fix_probsevere_vertices(row_indices_orig, column_indices_orig):
    """Fixes vertices of storm object created by probSevere.

    The resulting vertices are sorted counterclockwise and follow grid-cell
    edges (rather than cutting through grid cells).

    V_0 = number of original vertices
    V = number of final vertices

    :param row_indices_orig: numpy array (length V_0) with row numbers of
        original vertices.
    :param column_indices_orig: numpy array (length V_0) with column numbers of
        original vertices.
    :return: row_indices: length-V numpy array with row numbers of final (non-
        redundant) vertices.
    :return: column_indices: length-V numpy array with column numbers of final
        (non-redundant) vertices.
    """

    column_indices_orig, row_indices_orig = _get_longest_simple_polygon(
        column_indices_orig, row_indices_orig)
    row_indices_orig = row_indices_orig.astype(int)
    column_indices_orig = column_indices_orig.astype(int)

    if (row_indices_orig[0] != row_indices_orig[-1] or
            column_indices_orig[0] != column_indices_orig[-1]):
        row_indices_orig = numpy.concatenate((
            row_indices_orig, row_indices_orig[[0]]))
        column_indices_orig = numpy.concatenate((
            column_indices_orig, column_indices_orig[[0]]))

    column_indices_orig, row_indices_orig = sort_counterclockwise(
        column_indices_orig, -1 * row_indices_orig)
    row_indices_orig *= -1

    row_indices, column_indices = _vertices_from_grid_points_to_edges(
        row_indices_orig, column_indices_orig)
    return _remove_redundant_vertices(row_indices, column_indices)


def point_in_or_on_polygon(
        polygon_object, query_x_coordinate, query_y_coordinate):
    """Returns True if point is inside/touching the polygon, False otherwise.

    x- and y-coordinates may be in one of three formats (see docstring at top of
    file).  However, the 3 input arguments must have coordinates in the same
    format.

    :param polygon_object: `shapely.geometry.Polygon` object.
    :param query_x_coordinate: x-coordinate of query point.
    :param query_y_coordinate: y-coordinate of query point.
    :return: result: Boolean flag.  True if point is inside/touching the
        polygon, False otherwise.
    """

    error_checking.assert_is_not_nan(query_x_coordinate)
    error_checking.assert_is_not_nan(query_y_coordinate)

    point_object = shapely.geometry.Point(
        query_x_coordinate, query_y_coordinate)
    if polygon_object.contains(point_object):
        return True

    return polygon_object.touches(point_object)


def buffer_simple_polygon(
        vertex_x_metres, vertex_y_metres, max_buffer_dist_metres,
        min_buffer_dist_metres=numpy.nan, preserve_angles=False):
    """Creates buffer around simple polygon.

    V_0 = number of original vertices
    V = number of final vertices

    :param vertex_x_metres: numpy array (length V_0) with x-coordinates of
        original vertices.
    :param vertex_y_metres: numpy array (length V_0) with y-coordinates of
        original vertices.
    :param max_buffer_dist_metres: Max buffer distance.
    :param min_buffer_dist_metres: Minimum buffer distance.  If NaN, the buffer
        will be inclusive (i.e., the original polygon will be included in the
        buffer).  Otherwise, the buffer will be exclusive (i.e., the buffer will
        not include the original polygon).  For example, if
        `min_buffer_dist_metres` = NaN and `max_buffer_dist_metres` = 5, the
        buffer will include the original polygon and an area of 5 metres outside
        the original polygon.  However, if `min_buffer_dist_metres` = 0 and
        `max_buffer_dist_metres` = 5, the buffer will include only an area of 5
        metres outside the original polygon.  If `min_buffer_dist_metres` = 1
        and `max_buffer_dist_metres` = 5, the buffer will include only an area
        of 1-5 metres outside the original polygon.
    :param preserve_angles: Boolean flag.  If True, will preserve the angles of
        all vertices in the original polygon, which means that distance will not
        be strictly respected.  If False, will preserve buffer distances, which
        means that vertex angles will not be strictly respected.  We highly
        recommend keeping this as False (True only for unit tests).
    :return: buffered_polygon_object: `shapely.geometry.Polygon` object.
    """

    _check_vertex_arrays(vertex_x_metres, vertex_y_metres, allow_nan=False)
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
        vertex_x_metres, vertex_y_metres)
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
