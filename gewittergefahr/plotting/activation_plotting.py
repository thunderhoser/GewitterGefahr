"""Plotting methods for activation of CNN components.

CNN = convolutional neural network
"""

import matplotlib
from gewittergefahr.gg_utils import error_checking

matplotlib.use('agg')

DEFAULT_FIG_WIDTH_INCHES = 15.
DEFAULT_FIG_HEIGHT_INCHES = 15.
DOTS_PER_INCH = 600

# TODO(thunderhoser): Add methods to plot for many examples, neurons, or
# channels.


def plot_activations_as_2d_grid(
        activation_matrix, axes_object, colour_map_object,
        colour_norm_object=None, min_colour_value=None, max_colour_value=None,
        label_tick_marks=False):
    """Plots activations as 2-D colour map.

    M = number of rows in activation grid
    N = number of columns in activation grid

    :param activation_matrix: M-by-N numpy array of activations.
    :param axes_object: Instance of `matplotlib.axes._subplots.AxesSubplot`.
    :param colour_map_object: Instance of `matplotlib.pyplot.cm`.
    :param colour_norm_object: Instance of `matplotlib.colors.BoundaryNorm`.
    :param min_colour_value: [used only if `colour_norm_object is None`]
        Minimum activation in colour scheme.
    :param max_colour_value: [used only if `colour_norm_object is None`]
        Max activation in colour scheme.
    :param label_tick_marks: Boolean flag.  If True, will label tick marks on
        the x- and y-axes.
    """

    error_checking.assert_is_numpy_array_without_nan(activation_matrix)
    error_checking.assert_is_numpy_array(activation_matrix, num_dimensions=2)
    error_checking.assert_is_boolean(label_tick_marks)

    if colour_norm_object is None:
        error_checking.assert_is_greater(min_colour_value, max_colour_value)
        colour_norm_object = None
    else:
        min_colour_value = colour_norm_object.boundaries[0]
        max_colour_value = colour_norm_object.boundaries[-1]

    axes_object.pcolormesh(
        activation_matrix, cmap=colour_map_object, norm=colour_norm_object,
        vmin=min_colour_value, vmax=max_colour_value, shading='flat',
        edgecolors='None')

    if not label_tick_marks:
        axes_object.set_xticks([])
        axes_object.set_yticks([])
