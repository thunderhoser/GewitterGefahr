"""Processing methods for shapes (closed polygons and polylines)."""

import numpy
from gewittergefahr.gg_utils import error_checking

TOLERANCE = 1e-6
MIN_VERTICES_IN_POLYGON_OR_LINE = 4


def pad_closed_polygon(polygon_object, num_padding_vertices=0,
                       check_input_args=True):
    """Pads closed polygon (by adding duplicate vertices at either end).

    V_p = number of vertices after padding

    :param polygon_object: Instance of `shapely.geometry.Polygon`.
    :param num_padding_vertices: Number of duplicate vertices to add at either
        end.
    :param check_input_args: Boolean flag.  If True, will error-check input
        arguments.  If False, will not.
    :return: vertex_x_coords_padded: numpy array (length V_p) with x-coordinates
        of vertices.
    :return: vertex_y_coords_padded: numpy array (length V_p) with y-coordinates
        of vertices.
    """

    vertex_x_coords = numpy.asarray(polygon_object.exterior.xy[0])[:-1]
    vertex_y_coords = numpy.asarray(polygon_object.exterior.xy[1])[:-1]
    num_vertices = len(vertex_x_coords)

    if check_input_args:
        error_checking.assert_is_geq(
            num_vertices, MIN_VERTICES_IN_POLYGON_OR_LINE)
        error_checking.assert_is_integer(num_padding_vertices)
        error_checking.assert_is_greater(num_padding_vertices, 0)
        error_checking.assert_is_less_than(num_padding_vertices, num_vertices)

    vertex_x_coords_start = vertex_x_coords[-num_padding_vertices:]
    vertex_y_coords_start = vertex_y_coords[-num_padding_vertices:]
    vertex_x_coords_end = vertex_x_coords[:num_padding_vertices]
    vertex_y_coords_end = vertex_y_coords[:num_padding_vertices]

    vertex_x_coords_padded = numpy.concatenate((
        vertex_x_coords_start, vertex_x_coords, vertex_x_coords_end))
    vertex_y_coords_padded = numpy.concatenate((
        vertex_y_coords_start, vertex_y_coords, vertex_y_coords_end))

    return vertex_x_coords_padded, vertex_y_coords_padded


def pad_polyline(vertex_x_coords, vertex_y_coords, num_padding_vertices=0,
                 check_input_args=True):
    """Pads polyline* by adding extrapolated vertices at either end.

    * as opposed to closed polygon

    V_u = number of unique vertices
    V_p = number of vertices after padding

    :param vertex_x_coords: numpy array (length V_u) with x-coordinates of
        vertices.
    :param vertex_y_coords: numpy array (length V_u) with y-coordinates of
        vertices.
    :param num_padding_vertices: Number of extrapolated vertices to add at
        either end.
    :param check_input_args: Boolean flag.  If True, will error-check input
        arguments.  If False, will not.
    :return: vertex_x_coords_padded: numpy array (length V_p) with x-coordinates
        of vertices.
    :return: vertex_y_coords_padded: numpy array (length V_p) with y-coordinates
        of vertices.
    """

    num_vertices = vertex_x_coords.size

    if check_input_args:
        error_checking.assert_is_geq(
            num_vertices, MIN_VERTICES_IN_POLYGON_OR_LINE)

        error_checking.assert_is_numpy_array_without_nan(vertex_x_coords)
        error_checking.assert_is_numpy_array(vertex_x_coords, num_dimensions=1)
        error_checking.assert_is_numpy_array_without_nan(vertex_y_coords)
        error_checking.assert_is_numpy_array(
            vertex_y_coords, exact_dimensions=numpy.array([num_vertices]))

        error_checking.assert_is_integer(num_padding_vertices)
        error_checking.assert_is_greater(num_padding_vertices, 0)
        error_checking.assert_is_less_than(num_padding_vertices, num_vertices)

    x_difference = vertex_x_coords[1] - vertex_x_coords[0]
    vertex_x_coords_start = numpy.linspace(
        vertex_x_coords[0] - num_padding_vertices * x_difference,
        vertex_x_coords[0] - x_difference, num=num_padding_vertices)

    y_difference = vertex_y_coords[1] - vertex_y_coords[0]
    vertex_y_coords_start = numpy.linspace(
        vertex_y_coords[0] - num_padding_vertices * y_difference,
        vertex_y_coords[0] - y_difference, num=num_padding_vertices)

    x_difference = vertex_x_coords[-1] - vertex_x_coords[-2]
    vertex_x_coords_end = numpy.linspace(
        vertex_x_coords[-1] + x_difference,
        vertex_x_coords[-1] + num_padding_vertices * x_difference,
        num=num_padding_vertices)

    y_difference = vertex_y_coords[-1] - vertex_y_coords[-2]
    vertex_y_coords_end = numpy.linspace(
        vertex_y_coords[-1] + y_difference,
        vertex_y_coords[-1] + num_padding_vertices * y_difference,
        num=num_padding_vertices)

    vertex_x_coords_padded = numpy.concatenate((
        vertex_x_coords_start, vertex_x_coords, vertex_x_coords_end))
    vertex_y_coords_padded = numpy.concatenate((
        vertex_y_coords_start, vertex_y_coords, vertex_y_coords_end))

    return vertex_x_coords_padded, vertex_y_coords_padded
