"""Processing methods for shapes (closed polygons and polylines)."""

import numpy
from scipy.interpolate import UnivariateSpline
from gewittergefahr.gg_utils import error_checking

MIN_VERTICES_IN_POLYGON_OR_LINE = 4
SPLINE_DEGREE = 4


def _get_curvature(vertex_x_padded_metres, vertex_y_padded_metres):
    """Computes signed curvature at each vertex, using interpolating splines.

    Curvature = inverse of turning radius.

    This method is based on curvature.py, found here:
    https://gist.github.com/elyase/451cbc00152cb99feac6

    V_p = total number of vertices (including duplicates used for padding)
    V_u = number of unique vertices

    :param vertex_x_padded_metres: numpy array (length V_p) with x-coordinates
        of vertices.
    :param vertex_y_padded_metres: numpy array (length V_p) with y-coordinates
        of vertices.
    :return: vertex_curvatures_metres01: numpy array (length V_u) of curvatures
        (inverse metres).
    """

    num_padded_vertices = len(vertex_x_padded_metres)
    vertex_indices_padded = numpy.linspace(
        0, num_padded_vertices - 1, num=num_padded_vertices, dtype=int)

    interp_object_for_x = UnivariateSpline(
        vertex_indices_padded, vertex_x_padded_metres, k=SPLINE_DEGREE)
    interp_object_for_y = UnivariateSpline(
        vertex_indices_padded, vertex_y_padded_metres, k=SPLINE_DEGREE)

    vertex_indices_unique = vertex_indices_padded[SPLINE_DEGREE:-SPLINE_DEGREE]
    x_derivs_metres_per_vertex = interp_object_for_x.derivative(1)(
        vertex_indices_unique)
    x_derivs_metres2_per_vertex2 = interp_object_for_x.derivative(2)(
        vertex_indices_unique)

    y_derivs_metres_per_vertex = interp_object_for_y.derivative(1)(
        vertex_indices_unique)
    y_derivs_metres2_per_vertex2 = interp_object_for_y.derivative(2)(
        vertex_indices_unique)

    numerators = (x_derivs_metres_per_vertex * y_derivs_metres2_per_vertex2) - (
        y_derivs_metres_per_vertex * x_derivs_metres2_per_vertex2)
    denominators = numpy.power(
        x_derivs_metres_per_vertex ** 2 + x_derivs_metres_per_vertex ** 2, 1.5)
    return numerators / denominators


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


def get_curvature_for_closed_polygon(polygon_object_xy_metres):
    """Computes signed curvature at each vertex of closed polygon.

    V_u = number of unique vertices in polygon (not counting the duplicate of
          the first vertex, used to close the polygon).

    :param polygon_object_xy_metres: Instance of `shapely.geometry.Polygon`,
        with x- and y-coordinates in metres.
    :return: vertex_curvatures_metres01: numpy array (length V_u) of curvatures
        (inverse metres).
    """

    vertex_x_padded_metres, vertex_y_padded_metres = pad_closed_polygon(
        polygon_object_xy_metres, num_padding_vertices=SPLINE_DEGREE,
        check_input_args=False)

    return _get_curvature(vertex_x_padded_metres, vertex_y_padded_metres)


def get_curvature_for_polyline(vertex_x_metres, vertex_y_metres):
    """Computes signed curvature at each vertex of polyline*.

    * as opposed to closed polygon

    V = number of vertices

    :param vertex_x_metres: length-V numpy array with x-coordinates of vertices.
    :param vertex_y_metres: length-V numpy array with y-coordinates of vertices.
    :return: vertex_curvatures_metres01: length-V numpy array of curvatures
        (inverse metres).
    """

    vertex_x_padded_metres, vertex_y_padded_metres = pad_polyline(
        vertex_x_metres, vertex_y_metres, num_padding_vertices=SPLINE_DEGREE,
        check_input_args=False)

    return _get_curvature(vertex_x_padded_metres, vertex_y_padded_metres)
