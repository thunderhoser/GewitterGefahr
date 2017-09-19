"""Processing methods for polygons.

Currently the only polygons in AASSWP are storm-cell outlines.  However, I may
end up adding other polygons.
"""

import numpy
import cv2

# TODO(thunderhoser): add error-checking to all methods.

UP_DIRECTION_NAME = 'up'
DOWN_DIRECTION_NAME = 'down'
RIGHT_DIRECTION_NAME = 'right'
LEFT_DIRECTION_NAME = 'left'
UP_RIGHT_DIRECTION_NAME = 'up_right'
UP_LEFT_DIRECTION_NAME = 'up_left'
DOWN_RIGHT_DIRECTION_NAME = 'down_right'
DOWN_LEFT_DIRECTION_NAME = 'down_left'


def _get_longest_inner_list(list_of_lists):
    """Finds longest list in a list.

    :param list_of_lists: 1-D list of lists.
    :return: longest_list: Longest list.
    """

    num_lists = len(list_of_lists)
    list_lengths = numpy.full(num_lists, 0, dtype=int)
    for i in range(num_lists):
        list_lengths[i] = len(list_of_lists[i])

    return list_of_lists[numpy.argmax(list_lengths)]


def _get_direction_of_vertex_pair(first_row, second_row, first_column,
                                  second_column):
    """Finds direction between a pair of vertices.  The 8 valid directions are:

    - up
    - down
    - right
    - left
    - up and right (45-degree angle)
    - up and left (45-degree angle)
    - down and right (45-degree angle)
    - down and left (45-degree angle)

    :param first_row: Row index of first vertex.
    :param second_row: Row index of second vertex.
    :param first_column: Column index of first vertex.
    :param second_column: Column index of second vertex.
    :return: direction_string: String indicating direction from first to second
        vertex (may be "up", "down", "right", "left", "up_right", "up_left",
        "down_right", or "down_left").
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


def _remove_redundant_vertices(vertex_rows_orig, vertex_columns_orig):
    """Removes redundant vertices from a polygon.

    v = original number of vertices
    V = final number of vertices

    :param vertex_rows_orig: length-v numpy array with row indices of original
        vertices.
    :param vertex_columns_orig: length-v numpy array with column indices of
        original vertices.
    :return: vertex_rows: length-V numpy array with row indices of final
        vertices.
    :return: vertex_columns: length-V numpy array with column indices of final
        vertices.
    """

    num_vertices_orig = len(vertex_rows_orig)
    vertex_rows = numpy.array([])
    vertex_columns = numpy.array([])

    for i in range(num_vertices_orig - 1):
        found_flags = numpy.logical_and(vertex_rows == vertex_rows_orig[i],
                                        vertex_columns == vertex_columns_orig[
                                            i])

        if not numpy.any(found_flags):
            vertex_rows = numpy.concatenate(
                (vertex_rows, vertex_rows_orig[[i]]))
            vertex_columns = numpy.concatenate(
                (vertex_columns, vertex_columns_orig[[i]]))
        else:
            found_index = numpy.where(found_flags)[0][0]
            vertex_rows = vertex_rows[0:(found_index + 1)]
            vertex_columns = vertex_columns[0:(found_index + 1)]

    vertex_rows = numpy.concatenate((vertex_rows, vertex_rows[[0]]))
    vertex_columns = numpy.concatenate((vertex_columns, vertex_columns[[0]]))
    return vertex_rows, vertex_columns


def _points_in_poly_to_binary_matrix(row_indices, column_indices):
    """Converts list of grid points in polygon to binary matrix.

    P = number of grid points in polygon
    M = max(row_indices) - min(row_indices) + 3 = number of rows in subgrid
    N = max(column_indices) - min(column_indices) + 3 = number of columns in
        subgrid

    :param row_indices: length-P numpy array with row indices of grid points in
        polygon.
    :param column_indices: length-P numpy array with column indices of grid
        points in polygon.
    :return: binary_matrix: M-by-N numpy array of Booleans, where the [i, j]
        entry indicates whether or not the [i]th row and [j]th column (in the
        subgrid, not the full grid) is inside the polygon.
    :return: first_row_index: Same as min(row_indices) - 1.  Can be used later
        to convert the subgrid back to the full grid.
    :return: first_column_index: Same as min(column_indices) - 1.  Can be used
        later to convert the subgrid back to the full grid.
    """

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


def _binary_matrix_to_points_in_poly(binary_matrix, first_row_index,
                                     first_column_index):
    """Converts binary matrix to list of grid points in polygon.

    M = number of rows in subgrid
    N = number of columns in subgrid
    P = number of grid points in polygon

    :param binary_matrix: M-by-N numpy array of Booleans.  If
        binary_matrix[i, j] = True, the [i]th row and [j]th column of the
        subgrid is inside the polygon.
    :param first_row_index: Row 0 of the subgrid = row `first_row_index` of the
        full grid.
    :param first_column_index: Column 0 of the subgrid = column
        `first_column_index` of the full grid.
    :return: row_indices: length-P numpy array with row indices of grid points
        in polygon.
    :return: column_indices: length-P numpy array with column indices of grid
        points in polygon.
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


def _adjust_vertices_to_grid_cell_edges(vertex_rows_orig, vertex_columns_orig):
    """Adjusts vertices so that they follow grid-cell edges*.

    * Rather than cutting through grid cells.

    This method assumes that the polygon is traversed counterclockwise.

    v = number of vertices in original polygon
    V = number of vertices in new polygon

    :param vertex_rows_orig: length-v numpy array with row coordinates of
        original vertices.
    :param vertex_columns_orig: length-v numpy array with column coordinates of
        original vertices.
    :return: vertex_rows: length-V numpy array with row coordinates of new
        vertices.
    :return: vertex_columns: length-V numpy array with column coordinates of new
        vertices.
    """

    num_orig_vertices = len(vertex_rows_orig)
    vertex_rows = numpy.array([])
    vertex_columns = numpy.array([])

    for i in range(num_orig_vertices - 1):
        this_direction = (
            _get_direction_of_vertex_pair(vertex_rows_orig[i],
                                          vertex_rows_orig[i + 1],
                                          vertex_columns_orig[i],
                                          vertex_columns_orig[i + 1]))

        if this_direction == UP_DIRECTION_NAME:
            rows_to_append = numpy.array(
                [vertex_rows_orig[i] + 0.5, vertex_rows_orig[i + 1] - 0.5])
            columns_to_append = numpy.array([vertex_columns_orig[i] + 0.5,
                                             vertex_columns_orig[i + 1] + 0.5])
        elif this_direction == DOWN_DIRECTION_NAME:
            rows_to_append = numpy.array(
                [vertex_rows_orig[i] - 0.5, vertex_rows_orig[i + 1] + 0.5])
            columns_to_append = numpy.array([vertex_columns_orig[i] - 0.5,
                                             vertex_columns_orig[i + 1] - 0.5])
        elif this_direction == RIGHT_DIRECTION_NAME:
            rows_to_append = numpy.array(
                [vertex_rows_orig[i] + 0.5, vertex_rows_orig[i + 1] + 0.5])
            columns_to_append = numpy.array([vertex_columns_orig[i] - 0.5,
                                             vertex_columns_orig[i + 1] + 0.5])
        elif this_direction == LEFT_DIRECTION_NAME:
            rows_to_append = numpy.array(
                [vertex_rows_orig[i] - 0.5, vertex_rows_orig[i + 1] - 0.5])
            columns_to_append = numpy.array([vertex_columns_orig[i] + 0.5,
                                             vertex_columns_orig[i + 1] - 0.5])
        elif this_direction == UP_RIGHT_DIRECTION_NAME:
            rows_to_append = numpy.array(
                [vertex_rows_orig[i] + 0.5, vertex_rows_orig[i] - 0.5,
                 vertex_rows_orig[i] - 0.5])
            columns_to_append = numpy.array(
                [vertex_columns_orig[i] + 0.5, vertex_columns_orig[i] + 0.5,
                 vertex_columns_orig[i] + 1.5])
        elif this_direction == UP_LEFT_DIRECTION_NAME:
            rows_to_append = numpy.array(
                [vertex_rows_orig[i] - 0.5, vertex_rows_orig[i] - 0.5,
                 vertex_rows_orig[i] - 1.5])
            columns_to_append = numpy.array(
                [vertex_columns_orig[i] + 0.5, vertex_columns_orig[i] - 0.5,
                 vertex_columns_orig[i] - 0.5])
        elif this_direction == DOWN_RIGHT_DIRECTION_NAME:
            rows_to_append = numpy.array(
                [vertex_rows_orig[i] + 0.5, vertex_rows_orig[i] + 0.5,
                 vertex_rows_orig[i] + 1.5])
            columns_to_append = numpy.array(
                [vertex_columns_orig[i] - 0.5, vertex_columns_orig[i] + 0.5,
                 vertex_columns_orig[i] + 0.5])
        elif this_direction == DOWN_LEFT_DIRECTION_NAME:
            rows_to_append = numpy.array(
                [vertex_rows_orig[i] + 0.5, vertex_rows_orig[i] - 0.5,
                 vertex_rows_orig[i] - 0.5])
            columns_to_append = numpy.array(
                [vertex_columns_orig[i] - 0.5, vertex_columns_orig[i] - 0.5,
                 vertex_columns_orig[i] - 1.5])

        vertex_rows = numpy.concatenate((vertex_rows, rows_to_append))
        vertex_columns = numpy.concatenate((vertex_columns, columns_to_append))

    return vertex_rows, vertex_columns


def points_in_poly_to_vertices(row_indices, column_indices):
    """Converts list of grid points in polygon to list of vertices.

    This method returns one simple polygon with vertices ordered
    counterclockwise.  If there are disjoint polygons, this method will return
    the one with the most vertices (probably the one with the greatest area).
    If there are holes inside the polygon, this method simply removes them.

    P = number of grid points in polygon
    V = number of vertices

    :param row_indices: length-P numpy array with row indices of grid points in
        polygon.  All integers.
    :param column_indices: length-P numpy array with column indices of grid
        points in polygon.  All integers.
    :return: vertex_rows: length-V numpy array with row indices of vertices.
        All half-integers.
    :return: vertex_columns: length-V numpy array with column indices of
        vertices.  All half-integers.
    """

    (binary_matrix, first_row_index,
     first_column_index) = _points_in_poly_to_binary_matrix(row_indices,
                                                            column_indices)

    _, contour_list, _ = cv2.findContours(binary_matrix.astype(numpy.uint8),
                                          cv2.RETR_EXTERNAL,
                                          cv2.CHAIN_APPROX_SIMPLE)

    # If there are disjoint polygons, keep the one with the most vertices
    # (probably the one with the greatest area).
    contour_matrix = _get_longest_inner_list(contour_list)
    contour_matrix = numpy.array(contour_matrix)[:, 0, :]
    num_contour_points = contour_matrix.shape[0]

    num_vertices = num_contour_points + 1
    vertex_rows = numpy.full(num_vertices, numpy.nan)
    vertex_columns = numpy.full(num_vertices, numpy.nan)

    for i in range(num_vertices):
        if i == num_vertices - 1:
            vertex_rows[i] = contour_matrix[0, 1]
            vertex_columns[i] = contour_matrix[0, 0]
        else:
            vertex_rows[i] = contour_matrix[i, 1]
            vertex_columns[i] = contour_matrix[i, 0]

    vertex_rows += first_row_index
    vertex_columns += first_column_index
    vertex_rows, vertex_columns = _adjust_vertices_to_grid_cell_edges(
        vertex_rows, vertex_columns)
    return _remove_redundant_vertices(vertex_rows, vertex_columns)
