"""Plotting methods for saliency maps."""

import numpy
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as pyplot
from gewittergefahr.gg_utils import number_rounding
from gewittergefahr.gg_utils import error_checking

POSITIVE_LINE_STYLE = 'solid'
NEGATIVE_LINE_STYLE = 'dashed'
DEFAULT_COLOUR_MAP_OBJECT = pyplot.cm.gist_yarg
PIXEL_PADDING_FOR_CONTOUR_LABELS = 10
STRING_FORMAT_FOR_POSITIVE_LABELS = '%.3f'
STRING_FORMAT_FOR_NEGATIVE_LABELS = '-%.3f'
FONT_SIZE_FOR_CONTOUR_LABELS = 20

DEFAULT_LINE_WIDTH = 1.5
DEFAULT_NUM_CONTOUR_LEVELS = 12


def plot_saliency_field_2d(
        saliency_matrix, axes_object, max_contour_value,
        colour_map_object=DEFAULT_COLOUR_MAP_OBJECT, label_contours=False,
        line_width=DEFAULT_LINE_WIDTH,
        num_contour_levels=DEFAULT_NUM_CONTOUR_LEVELS):
    """Plots 2-D saliency field with unfilled, coloured contours.

    M = number of rows in spatial grid
    N = number of columns in spatial grid

    :param saliency_matrix: M-by-N numpy array of saliency values.
    :param axes_object: Instance of `matplotlib.axes._subplots.AxesSubplot`.
    :param max_contour_value: Max saliency value with a contour assigned to it.
        Minimum saliency value will be -1 * max_contour_value.  Positive values
        will be shown with solid contours, and negative values with dashed
        contours.
    :return: colour_map_object: Instance of `matplotlib.colors.ListedColormap`.
    :param label_contours: Boolean flag.  If True, each contour will be labeled
        with the corresponding value.
    :param line_width: Width of contour lines (scalar).
    :param num_contour_levels: Number of contour levels (i.e., number of
        saliency values corresponding to a contour).
    """

    error_checking.assert_is_numpy_array_without_nan(saliency_matrix)
    error_checking.assert_is_numpy_array(saliency_matrix, num_dimensions=2)
    error_checking.assert_is_greater(max_contour_value, 0.)
    error_checking.assert_is_boolean(label_contours)
    error_checking.assert_is_integer(num_contour_levels)
    error_checking.assert_is_greater(num_contour_levels, 0)
    num_contour_levels = int(
        number_rounding.ceiling_to_nearest(num_contour_levels, 2))

    positive_contour_levels = numpy.linspace(
        0., max_contour_value, num=num_contour_levels / 2 + 1)
    positive_contour_levels = positive_contour_levels[1:]
    positive_contour_object = axes_object.contour(
        saliency_matrix, levels=positive_contour_levels, cmap=colour_map_object,
        vmin=0., vmax=max_contour_value, linewidths=line_width,
        linestyles=POSITIVE_LINE_STYLE)

    if label_contours:
        pyplot.clabel(
            positive_contour_object, inline=True,
            inline_spacing=PIXEL_PADDING_FOR_CONTOUR_LABELS,
            fmt=STRING_FORMAT_FOR_POSITIVE_LABELS,
            fontsize=FONT_SIZE_FOR_CONTOUR_LABELS)

    negative_contour_object = axes_object.contour(
        -1 * saliency_matrix, levels=positive_contour_levels,
        cmap=colour_map_object, vmin=0., vmax=max_contour_value,
        linewidths=line_width, linestyles=NEGATIVE_LINE_STYLE)

    if label_contours:
        pyplot.clabel(
            negative_contour_object, inline=True,
            inline_spacing=PIXEL_PADDING_FOR_CONTOUR_LABELS,
            fmt=STRING_FORMAT_FOR_NEGATIVE_LABELS,
            fontsize=FONT_SIZE_FOR_CONTOUR_LABELS)
