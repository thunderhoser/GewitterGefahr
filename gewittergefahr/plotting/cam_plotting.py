"""Plots class-activation maps (CAM)."""

import numpy
import matplotlib
from gewittergefahr.gg_utils import error_checking

matplotlib.use('agg')
DEFAULT_CONTOUR_WIDTH = 2


def plot_2d_grid(class_activation_matrix_2d, axes_object, colour_map_object,
                 max_contour_level, contour_interval,
                 line_width=DEFAULT_CONTOUR_WIDTH):
    """Plots 2-D class-activation map with line contours.

    M = number of rows in spatial grid
    N = number of columns in spatial grid

    :param class_activation_matrix_2d: M-by-N numpy array of class activations.
    :param axes_object: Instance of `matplotlib.axes._subplots.AxesSubplot`.
        Will plot on these axes.
    :param colour_map_object: Colour scheme (instance of
        `matplotlib.pyplot.cm`).
    :param max_contour_level: Max value to plot.  Minimum value will be 0.
    :param contour_interval: Interval (in class-activation units) between
        successive contours.
    :param line_width: Width of contour lines.
    """

    error_checking.assert_is_numpy_array_without_nan(class_activation_matrix_2d)
    error_checking.assert_is_numpy_array(
        class_activation_matrix_2d, num_dimensions=2)

    error_checking.assert_is_greater(max_contour_level, 0.)
    error_checking.assert_is_greater(contour_interval, 0.)
    error_checking.assert_is_less_than(contour_interval, max_contour_level)

    num_grid_rows = class_activation_matrix_2d.shape[0]
    num_grid_columns = class_activation_matrix_2d.shape[1]

    x_coords_unique = numpy.linspace(
        0, num_grid_columns, num=num_grid_columns + 1, dtype=float)
    x_coords_unique = x_coords_unique[:-1]
    x_coords_unique = x_coords_unique + numpy.diff(x_coords_unique[:2]) / 2

    y_coords_unique = numpy.linspace(
        0, num_grid_rows, num=num_grid_rows + 1, dtype=float)
    y_coords_unique = y_coords_unique[:-1]
    y_coords_unique = y_coords_unique + numpy.diff(y_coords_unique[:2]) / 2

    x_coord_matrix, y_coord_matrix = numpy.meshgrid(x_coords_unique,
                                                    y_coords_unique)

    num_contours = int(numpy.round(
        1 + max_contour_level / contour_interval
    ))

    contour_levels = numpy.linspace(0., max_contour_level, num=num_contours)

    axes_object.contour(
        x_coord_matrix, y_coord_matrix, class_activation_matrix_2d,
        contour_levels[1:], cmap=colour_map_object,
        vmin=numpy.min(contour_levels), vmax=numpy.max(contour_levels),
        linewidths=line_width, linestyles='solid', zorder=1e6)


def plot_many_2d_grids(
        class_activation_matrix_3d, axes_object_matrix, colour_map_object,
        max_contour_level, contour_interval, line_width=DEFAULT_CONTOUR_WIDTH,
        row_major=True):
    """Plots the same 2-D class-activation map for each predictor.

    M = number of rows in spatial grid
    N = number of columns in spatial grid
    P = number of predictors

    :param class_activation_matrix_3d: M-by-N-by-P numpy array of class
        activations.
    :param axes_object_matrix: See doc for `plotting_utils.init_panels`.
    :param colour_map_object: See doc for `plot_2d_grid`.
    :param max_contour_level: Same.
    :param contour_interval: Same.
    :param line_width: Same.
    :param row_major: Boolean flag.  If True, panels will be filled along rows
        first, then down columns.  If False, down columns first, then along
        rows.
    """

    error_checking.assert_is_numpy_array_without_nan(class_activation_matrix_3d)
    error_checking.assert_is_numpy_array(
        class_activation_matrix_3d, num_dimensions=3)
    error_checking.assert_is_boolean(row_major)

    if row_major:
        order_string = 'C'
    else:
        order_string = 'F'

    num_predictors = class_activation_matrix_3d.shape[-1]
    num_panel_rows = axes_object_matrix.shape[0]
    num_panel_columns = axes_object_matrix.shape[1]

    for k in range(num_predictors):
        this_panel_row, this_panel_column = numpy.unravel_index(
            k, (num_panel_rows, num_panel_columns), order=order_string
        )

        plot_2d_grid(
            class_activation_matrix_2d=class_activation_matrix_3d[..., k],
            axes_object=axes_object_matrix[this_panel_row, this_panel_column],
            colour_map_object=colour_map_object,
            max_contour_level=max_contour_level,
            contour_interval=contour_interval, line_width=line_width)
