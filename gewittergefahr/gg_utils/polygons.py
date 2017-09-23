"""Processing methods for polygons.

Currently the only polygons in AASSWP are storm-cell outlines.  However, I may
end up adding other polygons.
"""

import copy
import numpy
import cv2
import shapely.geometry

# TODO(thunderhoser): add error-checking to all methods.

UP_DIRECTION_NAME = 'up'
DOWN_DIRECTION_NAME = 'down'
RIGHT_DIRECTION_NAME = 'right'
LEFT_DIRECTION_NAME = 'left'
UP_RIGHT_DIRECTION_NAME = 'up_right'
UP_LEFT_DIRECTION_NAME = 'up_left'
DOWN_RIGHT_DIRECTION_NAME = 'down_right'
DOWN_LEFT_DIRECTION_NAME = 'down_left'

EXTERIOR_X_COLUMN = 'exterior_x_metres'
EXTERIOR_Y_COLUMN = 'exterior_y_metres'
HOLE_X_COLUMN = 'hole_x_metres_list'
HOLE_Y_COLUMN = 'hole_y_metres_list'


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


def _separate_exterior_and_holes(vertex_x_metres, vertex_y_metres):
    """Separates exterior of polygon from holes in polygon.

    V = number of vertices
    H = number of holes
    V_e = number of exterior vertices
    V_hi = number of vertices in [i]th hole

    :param vertex_x_metres: length-V numpy array with x-coordinates of vertices.
        The first NaN separates the exterior from the first hole; the [i]th NaN
        separates the [i - 1]th hole from the [i]th hole.
    :param vertex_y_metres: Same as above, except for y-coordinates.
    :return: vertex_dict: Dictionary with the following keys.
    vertex_dict.exterior_x_metres: numpy array (length V_e) with x-coordinates
        of exterior vertices.
    vertex_dict.exterior_y_metres: numpy array (length V_e) with y-coordinates
        of exterior vertices.
    vertex_dict.hole_x_metres_list: List of H elements, where the [i]th element
        is a numpy array (length V_hi) with x-coordinates of interior vertices.
    vertex_dict.hole_y_metres_list: Same as above, except for y-coordinates.
    """

    nan_indices = numpy.where(numpy.isnan(vertex_x_metres))[0]
    if len(nan_indices) == 0:
        return {EXTERIOR_X_COLUMN: vertex_x_metres,
                EXTERIOR_Y_COLUMN: vertex_y_metres, HOLE_X_COLUMN: [],
                HOLE_Y_COLUMN: []}

    num_holes = len(nan_indices)
    exterior_x_metres = vertex_x_metres[0:nan_indices[0]]
    exterior_y_metres = vertex_y_metres[0:nan_indices[0]]
    hole_x_metres_list = []
    hole_y_metres_list = []

    for i in range(num_holes):
        if i == num_holes - 1:
            this_hole_x_metres = vertex_x_metres[(nan_indices[i] + 1):]
            this_hole_y_metres = vertex_y_metres[(nan_indices[i] + 1):]
        else:
            this_hole_x_metres = vertex_x_metres[
                                 (nan_indices[i] + 1):nan_indices[i + 1]]
            this_hole_y_metres = vertex_y_metres[
                                 (nan_indices[i] + 1):nan_indices[i + 1]]

        hole_x_metres_list.append(this_hole_x_metres)
        hole_y_metres_list.append(this_hole_y_metres)

    return {EXTERIOR_X_COLUMN: exterior_x_metres,
            EXTERIOR_Y_COLUMN: exterior_y_metres,
            HOLE_X_COLUMN: hole_x_metres_list,
            HOLE_Y_COLUMN: hole_y_metres_list}


def _merge_exterior_and_holes(exterior_vertex_x_metres,
                              exterior_vertex_y_metres,
                              hole_x_vertex_metres_list=None,
                              hole_y_vertex_metres_list=None):
    """Merges exterior of polygon with holes in polygon.

    V = number of vertices
    H = number of holes
    V_e = number of exterior vertices
    V_hi = number of vertices in [i]th hole

    :param exterior_vertex_x_metres: numpy array (length V_e) with x-coordinates
        of exterior vertices.
    :param exterior_vertex_y_metres: numpy array (length V_e) with y-coordinates
        of exterior vertices.
    :param hole_x_vertex_metres_list: List of H elements, where the [i]th
        element is a numpy array (length V_hi) with x-coordinates of interior
        vertices.
    :param hole_y_vertex_metres_list: Same as above, except for y-coordinates.
    :return: vertex_x_metres: length-V numpy array with x-coordinates of
        vertices.  The first NaN separates the exterior from the first hole; the
        [i]th NaN separates the [i - 1]th hole from the [i]th hole.
    :return: vertex_y_metres: Same as above, except for y-coordinates.
    """

    vertex_x_metres = copy.deepcopy(exterior_vertex_x_metres)
    vertex_y_metres = copy.deepcopy(exterior_vertex_y_metres)
    if hole_x_vertex_metres_list is None:
        return vertex_x_metres, vertex_y_metres

    num_holes = len(hole_x_vertex_metres_list)
    single_nan_array = numpy.array([numpy.nan])

    for i in range(num_holes):
        vertex_x_metres = numpy.concatenate(
            (vertex_x_metres, single_nan_array, hole_x_vertex_metres_list[i]))
        vertex_y_metres = numpy.concatenate(
            (vertex_y_metres, single_nan_array, hole_y_vertex_metres_list[i]))

    return vertex_x_metres, vertex_y_metres


def _vertex_arrays_to_list(vertex_x_metres, vertex_y_metres):
    """Converts vertex coordinates from two arrays to one list.

    V = number of vertices

    :param vertex_x_metres: length-V numpy array with x-coordinates of vertices.
    :param vertex_y_metres: length-V numpy array with y-coordinates of vertices.
    :return: vertex_metres_list: List of V elements, where the [i]th element is
        a tuple with (x-coordinate, y-coordinate).
    """

    num_vertices = len(vertex_x_metres)
    vertex_metres_list = []
    for i in range(num_vertices):
        vertex_metres_list.append((vertex_x_metres[i], vertex_y_metres[i]))

    return vertex_metres_list


def _vertex_list_to_arrays(vertex_metres_list):
    """Converts vertex coordinates from one list to two arrays.

    V = number of vertices

    :param vertex_metres_list: List of V elements, where the [i]th element is
        a tuple with (x-coordinate, y-coordinate).
    :return: vertex_x_metres: length-V numpy array with x-coordinates of
        vertices.
    :return: vertex_y_metres: length-V numpy array with y-coordinates of
        vertices.
    """

    num_vertices = len(vertex_metres_list)
    vertex_x_metres = numpy.full(num_vertices, numpy.nan)
    vertex_y_metres = numpy.full(num_vertices, numpy.nan)

    for i in range(num_vertices):
        vertex_x_metres[i] = vertex_metres_list[i][0]
        vertex_y_metres[i] = vertex_metres_list[i][1]

    return vertex_x_metres, vertex_y_metres


def _vertices_to_polygon_object(exterior_vertex_x_metres,
                                exterior_vertex_y_metres,
                                hole_x_vertex_metres_list=None,
                                hole_y_vertex_metres_list=None):
    """Converts arrays of vertex coords to `shapely.geometry.Polygon` object.

    :param exterior_vertex_x_metres: numpy array (length V_e) with x-coordinates
        of exterior vertices.
    :param exterior_vertex_y_metres: numpy array (length V_e) with y-coordinates
        of exterior vertices.
    :param hole_x_vertex_metres_list: List of H elements, where the [i]th
        element is a numpy array (length V_hi) with x-coordinates of interior
        vertices.
    :param hole_y_vertex_metres_list: Same as above, except for y-coordinates.
    :return: polygon_object: Instance of `shapely.geometry.Polygon`.
    :raises: ValueError: if resulting polygon is invalid.
    """

    exterior_vertex_metres_list = _vertex_arrays_to_list(
        exterior_vertex_x_metres, exterior_vertex_y_metres)

    if hole_x_vertex_metres_list is None:
        return shapely.geometry.Polygon(shell=exterior_vertex_metres_list)

    num_holes = len(hole_x_vertex_metres_list)
    hole_vertex_metres_list_of_lists = []
    for i in range(num_holes):
        hole_vertex_metres_list_of_lists.append(
            _vertex_arrays_to_list(hole_x_vertex_metres_list[i],
                                   hole_y_vertex_metres_list[i]))

    polygon_object = shapely.geometry.Polygon(
        shell=exterior_vertex_metres_list,
        holes=tuple(hole_vertex_metres_list_of_lists))

    if not polygon_object.is_valid:
        raise ValueError('Resulting polygon is invalid.')

    return polygon_object


def _polygon_object_to_vertices(polygon_object):
    """Converts `shapely.geometry.Polygon` object to arrays of vertex coords.

    H = number of holes
    V_e = number of exterior vertices
    V_hi = number of vertices in [i]th hole

    :param polygon_object: Instance of `shapely.geometry.Polygon`.
    :return: vertex_dict: Dictionary with the following keys.
    vertex_dict.exterior_x_metres: numpy array (length V_e) with x-coordinates
        of exterior vertices.
    vertex_dict.exterior_y_metres: numpy array (length V_e) with y-coordinates
        of exterior vertices.
    vertex_dict.hole_x_metres_list: List of H elements, where the [i]th element
        is a numpy array (length V_hi) with x-coordinates of interior vertices.
    vertex_dict.hole_y_metres_list: Same as above, except for y-coordinates.
    """

    exterior_x_metres, exterior_y_metres = _vertex_list_to_arrays(
        list(polygon_object.exterior.coords))

    num_holes = len(polygon_object.interiors)
    if num_holes == 0:
        return {EXTERIOR_X_COLUMN: exterior_x_metres,
                EXTERIOR_Y_COLUMN: exterior_y_metres, HOLE_X_COLUMN: [],
                HOLE_Y_COLUMN: []}

    hole_x_metres_list = []
    hole_y_metres_list = []
    for i in range(num_holes):
        (this_hole_x_metres, this_hole_y_metres) = _vertex_list_to_arrays(
            list(polygon_object.interiors[i].coords))

        hole_x_metres_list.append(this_hole_x_metres)
        hole_y_metres_list.append(this_hole_y_metres)

    return {EXTERIOR_X_COLUMN: exterior_x_metres,
            EXTERIOR_Y_COLUMN: exterior_y_metres,
            HOLE_X_COLUMN: hole_x_metres_list,
            HOLE_Y_COLUMN: hole_y_metres_list}


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


def _patch_diag_connections_in_binary_matrix(binary_matrix):
    """Patches diagonal connections in binary image matrix.

    When two pixels P and Q are connected only diagonally, this method "patches"
    the connection by adding another pixel -- adjacent to both P and Q -- to the
    image.  In other words, this method flips one bit in the image from False to
    True.

    If diagonal connections are not patched, points_in_poly_to_vertices will
    create disjoint polygons.

    M = number of rows in binary image
    N = number of columns in binary image

    :param binary_matrix: M-by-N numpy array of Boolean flags.  The flag at each
        pixel [i, j] indicates whether or not the pixel is inside a polygon.
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
                [vertex_rows_orig[i] + 0.5, vertex_rows_orig[i + 1] + 0.5,
                 vertex_rows_orig[i + 1] + 0.5])
            columns_to_append = numpy.array(
                [vertex_columns_orig[i] + 0.5, vertex_columns_orig[i] + 0.5,
                 vertex_columns_orig[i + 1] + 0.5])
        elif this_direction == UP_LEFT_DIRECTION_NAME:
            rows_to_append = numpy.array(
                [vertex_rows_orig[i] - 0.5, vertex_rows_orig[i] - 0.5,
                 vertex_rows_orig[i + 1] - 0.5])
            columns_to_append = numpy.array(
                [vertex_columns_orig[i] + 0.5, vertex_columns_orig[i + 1] + 0.5,
                 vertex_columns_orig[i + 1] + 0.5])
        elif this_direction == DOWN_RIGHT_DIRECTION_NAME:
            rows_to_append = numpy.array(
                [vertex_rows_orig[i] + 0.5, vertex_rows_orig[i] + 0.5,
                 vertex_rows_orig[i + 1] + 0.5])
            columns_to_append = numpy.array(
                [vertex_columns_orig[i] - 0.5, vertex_columns_orig[i + 1] - 0.5,
                 vertex_columns_orig[i + 1] - 0.5])
        elif this_direction == DOWN_LEFT_DIRECTION_NAME:
            rows_to_append = numpy.array(
                [vertex_rows_orig[i] - 0.5, vertex_rows_orig[i + 1] - 0.5,
                 vertex_rows_orig[i + 1] - 0.5])
            columns_to_append = numpy.array(
                [vertex_columns_orig[i] - 0.5, vertex_columns_orig[i + 1] - 0.5,
                 vertex_columns_orig[i + 1] - 0.5])

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
    binary_matrix = _patch_diag_connections_in_binary_matrix(binary_matrix)

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


def make_buffer_around_simple_polygon(orig_vertex_x_metres,
                                      orig_vertex_y_metres,
                                      min_buffer_dist_metres=numpy.nan,
                                      max_buffer_dist_metres=None,
                                      preserve_angles=False):
    """Creates buffer around simple polygon.

    v = number of vertices in original polygon
    V = number of vertices in new polygon (after applying buffer)

    :param orig_vertex_x_metres: length-v numpy array with x-coordinates of
        original vertices.
    :param orig_vertex_y_metres: length-v numpy array with y-coordinates of
        original vertices.
    :param min_buffer_dist_metres: Minimum buffer distance.  If defined, the
        original polygon will *not* be included in the new polygon (e.g., "0-5
        km outside the storm").  If NaN, the original polygon *will* be
        included in the new polygon (e.g, "within 5 km of the storm").
    :param max_buffer_dist_metres: Maximum buffer distance.
    :param preserve_angles: Boolean flag.  If True, will preserve the angles of
        all vertices.  In other words, the buffered polygon will have the same
        vertex angles as the original.  However, this means that buffer
        distances (`min_buffer_dist_metres` and `max_buffer_dist_metres`) will
        not be strictly respected.  It is highly recommended that you leave this
        flag as False.
    :return: buffer_vertex_x_metres: length-V numpy array with x-coordinates of
        vertices.  The first NaN separates the exterior from the first hole; the
        [i]th NaN separates the [i - 1]th hole from the [i]th hole.
    :return: buffer_vertex_y_metres: Same as above, except for y-coordinates.
    """

    if preserve_angles:
        join_style = shapely.geometry.JOIN_STYLE.mitre
    else:
        join_style = shapely.geometry.JOIN_STYLE.round

    orig_polygon_object = _vertices_to_polygon_object(orig_vertex_x_metres,
                                                      orig_vertex_y_metres)

    max_buffer_polygon_object = orig_polygon_object.buffer(
        max_buffer_dist_metres, join_style=join_style)
    max_buffer_vertex_dict = _polygon_object_to_vertices(
        max_buffer_polygon_object)

    if numpy.isnan(min_buffer_dist_metres):
        return (max_buffer_vertex_dict[EXTERIOR_X_COLUMN],
                max_buffer_vertex_dict[EXTERIOR_Y_COLUMN])

    min_buffer_polygon_object = orig_polygon_object.buffer(
        min_buffer_dist_metres, join_style=join_style)
    min_buffer_vertex_dict = _polygon_object_to_vertices(
        min_buffer_polygon_object)

    return _merge_exterior_and_holes(
        max_buffer_vertex_dict[EXTERIOR_X_COLUMN],
        max_buffer_vertex_dict[EXTERIOR_Y_COLUMN],
        hole_x_vertex_metres_list=[min_buffer_vertex_dict[EXTERIOR_X_COLUMN]],
        hole_y_vertex_metres_list=[min_buffer_vertex_dict[EXTERIOR_Y_COLUMN]])
