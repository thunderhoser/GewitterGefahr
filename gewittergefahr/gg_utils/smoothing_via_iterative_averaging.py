"""Implements the SIA algorithm from Mansouryar and Hedayati (2012).

--- DEFINITIONS ---

SIA = smoothing via iterative averaging

--- REFERENCES ---

Mansouryar, Mohsen, and Amin Hedayati. "Smoothing via iterative averaging (SIA)
    a basic technique for line smoothing." International Journal of Computer and
    Electrical Engineering 4.3 (2012): 307.
"""

import copy
import numpy
from gewittergefahr.gg_utils import error_checking

MIN_VERTICES_IN_POLYGON_OR_LINE = 4
NUM_VERTICES_IN_HALF_WINDOW_DEFAULT = 1
NUM_ITERATIONS_DEFAULT = 3


def _pad_closed_polygon_for_sia(vertex_x_coords, vertex_y_coords,
                                num_vertices_in_half_window):
    """Pads closed polygon in preparation for SIA.

    V_u = number of unique vertices
    V_p = number of vertices after padding

    :param vertex_x_coords: numpy array (length V_u) with x-coordinates of
        vertices.
    :param vertex_y_coords: numpy array (length V_u) with y-coordinates of
        vertices.
    :param num_vertices_in_half_window: Number of vertices in smoothing half-
        window.
    :return: vertex_x_coords_padded: numpy array (length V_p) with x-coordinates
        of vertices.
    :return: vertex_y_coords_padded: numpy array (length V_p) with y-coordinates
        of vertices.
    """

    vertex_x_coords_start = vertex_x_coords[-num_vertices_in_half_window:]
    vertex_y_coords_start = vertex_y_coords[-num_vertices_in_half_window:]
    vertex_x_coords_end = vertex_x_coords[:num_vertices_in_half_window]
    vertex_y_coords_end = vertex_y_coords[:num_vertices_in_half_window]

    vertex_x_coords_padded = numpy.concatenate((
        vertex_x_coords_start, vertex_x_coords, vertex_x_coords_end))
    vertex_y_coords_padded = numpy.concatenate((
        vertex_y_coords_start, vertex_y_coords, vertex_y_coords_end))

    return vertex_x_coords_padded, vertex_y_coords_padded


def _pad_polyline_for_sia(vertex_x_coords, vertex_y_coords,
                          num_vertices_in_half_window):
    """Pads polyline (not closed polygon) in preparation for SIA.

    V_u = number of unique vertices
    V_p = number of vertices after padding

    :param vertex_x_coords: numpy array (length V_u) with x-coordinates of
        vertices.
    :param vertex_y_coords: numpy array (length V_u) with y-coordinates of
        vertices.
    :param num_vertices_in_half_window: Number of vertices in smoothing half-
        window.
    :return: vertex_x_coords_padded: numpy array (length V_p) with x-coordinates
        of vertices.
    :return: vertex_y_coords_padded: numpy array (length V_p) with y-coordinates
        of vertices.
    """

    x_difference = vertex_x_coords[1] - vertex_x_coords[0]
    vertex_x_coords_start = numpy.linspace(
        vertex_x_coords[0] - num_vertices_in_half_window * x_difference,
        vertex_x_coords[0] - x_difference, num=num_vertices_in_half_window)

    y_difference = vertex_y_coords[1] - vertex_y_coords[0]
    vertex_y_coords_start = numpy.linspace(
        vertex_y_coords[0] - num_vertices_in_half_window * y_difference,
        vertex_y_coords[0] - y_difference, num=num_vertices_in_half_window)

    x_difference = vertex_x_coords[-1] - vertex_x_coords[-2]
    vertex_x_coords_end = numpy.linspace(
        vertex_x_coords[-1] + x_difference,
        vertex_x_coords[-1] + num_vertices_in_half_window * x_difference,
        num=num_vertices_in_half_window)

    y_difference = vertex_y_coords[-1] - vertex_y_coords[-2]
    vertex_y_coords_end = numpy.linspace(
        vertex_y_coords[-1] + y_difference,
        vertex_y_coords[-1] + num_vertices_in_half_window * y_difference,
        num=num_vertices_in_half_window)

    vertex_x_coords_padded = numpy.concatenate((
        vertex_x_coords_start, vertex_x_coords, vertex_x_coords_end))
    vertex_y_coords_padded = numpy.concatenate((
        vertex_y_coords_start, vertex_y_coords, vertex_y_coords_end))

    return vertex_x_coords_padded, vertex_y_coords_padded


def _sia_one_iteration(vertex_x_coords_padded, vertex_y_coords_padded,
                       num_vertices_in_half_window):
    """Runs SIA for one iteration.

    V_u = number of unique vertices
    V_p = number of vertices after padding

    :param vertex_x_coords_padded: numpy array (length V_p) with x-coordinates
        of vertices.
    :param vertex_y_coords_padded: numpy array (length V_p) with y-coordinates
        of vertices.
    :param num_vertices_in_half_window: Number of vertices in smoothing half-
        window.
    :return: vertex_x_coords_smoothed: numpy array (length V_u) with smoothed
        x-coordinates of vertices.
    :return: vertex_y_coords_smoothed: numpy array (length V_u) with smoothed
        y-coordinates of vertices.
    """

    num_vertices_padded = len(vertex_x_coords_padded)
    vertex_x_coords_smoothed = numpy.full(num_vertices_padded, numpy.nan)
    vertex_y_coords_smoothed = numpy.full(num_vertices_padded, numpy.nan)

    start_index = num_vertices_in_half_window
    end_index = len(vertex_x_coords_padded) - num_vertices_in_half_window

    for i in range(start_index, end_index):
        vertex_x_coords_smoothed[i] = numpy.mean(
            vertex_x_coords_padded[
                (i - num_vertices_in_half_window):
                (i + num_vertices_in_half_window + 1)])
        vertex_y_coords_smoothed[i] = numpy.mean(
            vertex_y_coords_padded[
                (i - num_vertices_in_half_window):
                (i + num_vertices_in_half_window + 1)])

    vertex_x_coords_smoothed = (
        vertex_x_coords_smoothed[
            num_vertices_in_half_window:-num_vertices_in_half_window])
    vertex_y_coords_smoothed = (
        vertex_y_coords_smoothed[
            num_vertices_in_half_window:-num_vertices_in_half_window])
    return vertex_x_coords_smoothed, vertex_y_coords_smoothed


def sia_for_closed_polygon(
        polygon_object,
        num_vertices_in_half_window=NUM_VERTICES_IN_HALF_WINDOW_DEFAULT,
        num_iterations=NUM_ITERATIONS_DEFAULT, check_input_args=True):
    """Implements the SIA algorithm for a closed polygon.

    This method smooths only the exterior of the polygon, ignoring the interior
    (holes).

    V = number of exterior vertices

    :param polygon_object: Instance of `shapely.geometry.Polygon`.
    :param num_vertices_in_half_window: Number of vertices in smoothing half-
        window.  Number of vertices in full window =
        2 * num_vertices_in_half_window + 1.
    :param num_iterations: Number of iterations.
    :param check_input_args: Boolean flag.  If True, will error-check input
        arguments.  If False, will not.
    :return: vertex_x_coords_smoothed: length-V numpy array with smoothed
        x-coordinates of vertices.
    :return: vertex_y_coords_smoothed: length-V numpy array with smoothed
        y-coordinates of vertices.
    """

    vertex_x_coords_smoothed = numpy.asarray(polygon_object.exterior.xy[0])[:-1]
    vertex_y_coords_smoothed = numpy.asarray(polygon_object.exterior.xy[1])[:-1]
    num_vertices = len(vertex_x_coords_smoothed)

    if check_input_args:
        error_checking.assert_is_geq(
            num_vertices, MIN_VERTICES_IN_POLYGON_OR_LINE)
        error_checking.assert_is_integer(num_vertices_in_half_window)
        error_checking.assert_is_geq(num_vertices_in_half_window, 1)
        error_checking.assert_is_integer(num_iterations)
        error_checking.assert_is_geq(num_iterations, 1)

    num_vertices_in_half_window = numpy.min(
        numpy.array([num_vertices_in_half_window, num_vertices - 1]))

    for _ in range(num_iterations):
        vertex_x_coords_padded, vertex_y_coords_padded = (
            _pad_closed_polygon_for_sia(
                vertex_x_coords_smoothed, vertex_y_coords_smoothed,
                num_vertices_in_half_window))

        vertex_x_coords_smoothed, vertex_y_coords_smoothed = _sia_one_iteration(
            vertex_x_coords_padded, vertex_y_coords_padded,
            num_vertices_in_half_window)

    vertex_x_coords_smoothed = numpy.concatenate((
        vertex_x_coords_smoothed, numpy.array([vertex_x_coords_smoothed[0]])))
    vertex_y_coords_smoothed = numpy.concatenate((
        vertex_y_coords_smoothed, numpy.array([vertex_y_coords_smoothed[0]])))

    return vertex_x_coords_smoothed, vertex_y_coords_smoothed


def sia_for_polyline(
        vertex_x_coords, vertex_y_coords,
        num_vertices_in_half_window=NUM_VERTICES_IN_HALF_WINDOW_DEFAULT,
        num_iterations=NUM_ITERATIONS_DEFAULT, check_input_args=True):
    """Implements the SIA algorithm for a polyline (not a closed polygon).

    V = number of vertices

    :param vertex_x_coords: length-V numpy array with x-coordinates of vertices.
    :param vertex_y_coords: length-V numpy array with y-coordinates of vertices.
    :param num_vertices_in_half_window: Number of vertices in smoothing half-
        window.  Number of vertices in full window =
        2 * num_vertices_in_half_window + 1.
    :param num_iterations: Number of iterations.
    :param check_input_args: Boolean flag.  If True, will error-check input
        arguments.  If False, will not.
    :return: vertex_x_coords_smoothed: length-V numpy array with smoothed
        x-coordinates of vertices.
    :return: vertex_y_coords_smoothed: length-V numpy array with smoothed
        y-coordinates of vertices.
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

        error_checking.assert_is_integer(num_vertices_in_half_window)
        error_checking.assert_is_geq(num_vertices_in_half_window, 1)
        error_checking.assert_is_integer(num_iterations)
        error_checking.assert_is_geq(num_iterations, 1)

    num_vertices_in_half_window = numpy.min(
        numpy.array([num_vertices_in_half_window, num_vertices - 1]))

    vertex_x_coords_smoothed = copy.deepcopy(vertex_x_coords)
    vertex_y_coords_smoothed = copy.deepcopy(vertex_y_coords)

    for _ in range(num_iterations):
        vertex_x_coords_padded, vertex_y_coords_padded = (
            _pad_polyline_for_sia(
                vertex_x_coords_smoothed, vertex_y_coords_smoothed,
                num_vertices_in_half_window))

        vertex_x_coords_smoothed, vertex_y_coords_smoothed = _sia_one_iteration(
            vertex_x_coords_padded, vertex_y_coords_padded,
            num_vertices_in_half_window)

    return vertex_x_coords_smoothed, vertex_y_coords_smoothed
