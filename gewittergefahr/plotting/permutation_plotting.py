"""Plots results of permutation test for predictor importance."""

import numpy
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as pyplot
from gewittergefahr.gg_utils import error_checking
from gewittergefahr.deep_learning import permutation

DEFAULT_FACE_COLOUR = numpy.array([166, 206, 227], dtype=float) / 255
DEFAULT_EDGE_COLOUR = numpy.full(3, 0.)
DEFAULT_EDGE_WIDTH = 2.

TEXT_COLOUR = numpy.full(3, 0.)
FIGURE_WIDTH_INCHES = 15
FIGURE_HEIGHT_INCHES = 15

FONT_SIZE = 20
pyplot.rc('font', size=FONT_SIZE)
pyplot.rc('axes', titlesize=FONT_SIZE)
pyplot.rc('axes', labelsize=FONT_SIZE)
pyplot.rc('xtick', labelsize=FONT_SIZE)
pyplot.rc('ytick', labelsize=FONT_SIZE)
pyplot.rc('legend', fontsize=FONT_SIZE)
pyplot.rc('figure', titlesize=FONT_SIZE)


def _label_bars(axes_object, y_coords, y_strings):
    """Labels bars in graph.

    J = number of bars

    :param axes_object: Will plot on these axes (instance of
        `matplotlib.axes._subplots.AxesSubplot`).
    :param y_coords: length-J numpy array with y-coordinates of bars.
    :param y_strings: length-J list of labels.
    """

    x_min, x_max = pyplot.xlim()
    x_coord_for_text = x_min + 0.01 * (x_max - x_min)

    for j in range(len(y_coords)):
        axes_object.text(
            x_coord_for_text, y_coords[j], y_strings[j], color=TEXT_COLOUR,
            horizontalalignment='left', verticalalignment='center')


def plot_lakshmanan_results(
        result_dict, axes_object, plot_percent_increase=False,
        bar_face_colour=DEFAULT_FACE_COLOUR,
        bar_edge_colour=DEFAULT_EDGE_COLOUR,
        bar_edge_width=DEFAULT_EDGE_WIDTH):
    """Plots results of Lakshmanan (multi-pass) permutation test.

    :param result_dict: See doc for `plot_breiman_results`.
    :param axes_object: Same.
    :param plot_percent_increase: Same.
    :param bar_face_colour: Same.
    :param bar_edge_colour: Same.
    :param bar_edge_width: Same.
    """

    error_checking.assert_is_boolean(plot_percent_increase)

    x_coords = numpy.concatenate((
        numpy.array([result_dict[permutation.ORIGINAL_COST_KEY]]),
        result_dict[permutation.HIGHEST_COSTS_KEY]
    ))

    if plot_percent_increase:
        x_coords = 100 * x_coords / x_coords[0]

    y_strings = (
        ['No permutation'] + result_dict[permutation.SELECTED_PREDICTORS_KEY]
    )

    y_coords = numpy.linspace(
        0, len(y_strings) - 1, num=len(y_strings), dtype=float
    )[::-1]

    axes_object.barh(
        y_coords, x_coords, color=bar_face_colour, edgecolor=bar_edge_colour,
        linewidth=bar_edge_width)

    axes_object.set_yticks([], [])
    axes_object.set_ylabel('Predictor permuted')

    if plot_percent_increase:
        axes_object.set_xlabel('Cross-entropy (percentage of original)')
    else:
        axes_object.set_xlabel('Cross-entropy (absolute)')

    _label_bars(axes_object=axes_object, y_coords=y_coords, y_strings=y_strings)


def plot_breiman_results(result_dict, axes_object, plot_percent_increase=False,
                         bar_face_colour=DEFAULT_FACE_COLOUR,
                         bar_edge_colour=DEFAULT_EDGE_COLOUR,
                         bar_edge_width=DEFAULT_EDGE_WIDTH):
    """Plots results of Breiman (single-pass) permutation test.

    :param result_dict: Dictionary created by
        `permutation.run_permutation_test`.
    :param axes_object: Will plot on these axes (instance of
        `matplotlib.axes._subplots.AxesSubplot`).
    :param plot_percent_increase: Boolean flag.  If True, x-axis will be
        percentage of original cost (before permutation).  If False, will be
        actual cost.
    :param bar_face_colour: Interior colour (any format accepted by
        `matplotlib.colors`) of each bar in the graph.
    :param bar_edge_colour: Edge colour of each bar in the graph.
    :param bar_edge_width: Edge width of each bar in the graph.
    """

    error_checking.assert_is_boolean(plot_percent_increase)

    cost_values = result_dict[permutation.STEP1_COSTS_KEY]
    predictor_names = result_dict[permutation.STEP1_PREDICTORS_KEY]

    sort_indices = numpy.argsort(cost_values)
    cost_values = cost_values[sort_indices]
    predictor_names = [predictor_names[k] for k in sort_indices]

    x_coords = numpy.concatenate((
        numpy.array([result_dict[permutation.ORIGINAL_COST_KEY]]),
        cost_values
    ))

    if plot_percent_increase:
        x_coords = 100 * x_coords / x_coords[0]

    y_strings = ['No permutation'] + predictor_names
    y_coords = numpy.linspace(
        0, len(y_strings) - 1, num=len(y_strings), dtype=float)

    axes_object.barh(
        y_coords, x_coords, color=bar_face_colour, edgecolor=bar_edge_colour,
        linewidth=bar_edge_width)

    axes_object.set_yticks([], [])
    axes_object.set_ylabel('Predictor permuted')

    if plot_percent_increase:
        axes_object.set_xlabel('Cross-entropy (percentage of original)')
    else:
        axes_object.set_xlabel('Cross-entropy (absolute)')

    _label_bars(axes_object=axes_object, y_coords=y_coords, y_strings=y_strings)
