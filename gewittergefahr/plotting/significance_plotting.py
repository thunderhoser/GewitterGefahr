"""Plotting methods for statistical significance (mainly on grids)."""

import numpy
import matplotlib
matplotlib.use('agg')
from gewittergefahr.gg_utils import error_checking

DEFAULT_MARKER_TYPE = '.'
DEFAULT_MARKER_SIZE = 6
DEFAULT_MARKER_COLOUR = numpy.full(3, 0.)
DEFAULT_MARKER_EDGE_WIDTH = 1


def plot_2d_grid_without_coords(
        significance_matrix, axes_object, marker_type=DEFAULT_MARKER_TYPE,
        marker_size=DEFAULT_MARKER_SIZE, marker_colour=DEFAULT_MARKER_COLOUR,
        marker_edge_width=DEFAULT_MARKER_EDGE_WIDTH):
    """Plots 2-D significance grid with markers.

    M = number of rows in grid
    N = number of columns in grid

    :param significance_matrix: M-by-N numpy array of Boolean flags, indicating
        where some value is significant.
    :param axes_object: Will plot on these axes (instance of
        `matplotlib.axes._subplots.AxesSubplot`).
    :param marker_type: Marker type (in any format accepted by matplotlib).
    :param marker_size: Marker size.
    :param marker_colour: Marker colour as length-3 numpy array.
    :param marker_edge_width: Marker-edge width.
    """

    error_checking.assert_is_boolean_numpy_array(significance_matrix)
    error_checking.assert_is_numpy_array(significance_matrix, num_dimensions=2)
    error_checking.assert_is_numpy_array(marker_colour)

    sig_rows, sig_columns = numpy.where(significance_matrix)

    axes_object.plot(
        sig_columns + 0.5, sig_rows + 0.5, linestyle='None',
        marker=marker_type, markerfacecolor=marker_colour,
        markeredgecolor=marker_colour, markersize=marker_size,
        markeredgewidth=marker_edge_width)


def plot_many_2d_grids_without_coords(
        significance_matrix, axes_object_matrix,
        marker_type=DEFAULT_MARKER_TYPE, marker_size=DEFAULT_MARKER_SIZE,
        marker_colour=DEFAULT_MARKER_COLOUR,
        marker_edge_width=DEFAULT_MARKER_EDGE_WIDTH, row_major=True):
    """Plots many 2-D significance grid with markers.

    M = number of rows in grid
    N = number of columns in grid
    C = number of grids

    :param significance_matrix: M-by-N-by-C numpy array of Boolean flags,
        indicating where some value is significant.
    :param axes_object_matrix: See doc for
        `plotting_utils.create_paneled_figure`.
    :param marker_type: Marker type (in any format accepted by matplotlib).
    :param marker_size: Marker size.
    :param marker_colour: Marker colour as length-3 numpy array.
    :param marker_edge_width: Marker-edge width.
    :param row_major: Boolean flag.  If True, panels will be filled along rows
        first, then down columns.  If False, down columns first, then along
        rows.
    """

    error_checking.assert_is_boolean_numpy_array(significance_matrix)
    error_checking.assert_is_numpy_array(significance_matrix, num_dimensions=3)
    error_checking.assert_is_numpy_array(marker_colour)
    error_checking.assert_is_boolean(row_major)

    if row_major:
        order_string = 'C'
    else:
        order_string = 'F'

    num_grids = significance_matrix.shape[-1]
    num_panel_rows = axes_object_matrix.shape[0]
    num_panel_columns = axes_object_matrix.shape[1]

    for k in range(num_grids):
        this_panel_row, this_panel_column = numpy.unravel_index(
            k, (num_panel_rows, num_panel_columns), order=order_string
        )

        plot_2d_grid_without_coords(
            significance_matrix=significance_matrix[..., k],
            axes_object=axes_object_matrix[this_panel_row, this_panel_column],
            marker_type=marker_type, marker_size=marker_size,
            marker_colour=marker_colour, marker_edge_width=marker_edge_width)
