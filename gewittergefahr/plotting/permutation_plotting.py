"""Plots results of permutation test for predictor importance."""

import numpy
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as pyplot
from gewittergefahr.gg_utils import error_checking
from gewittergefahr.deep_learning import permutation

# DEFAULT_REFERENCE_LINE_COLOUR = numpy.array(
#     [102, 194, 165], dtype=float
# ) / 255

SOUNDING_PREDICTOR_NAMES = [
    r'$u$-wind',
    r'$v$-wind',
    'Relative humidity',
    'Specific humidity',
    'Virtual potential temperature'
]

# DEFAULT_FACE_COLOUR = numpy.array([217, 95, 2], dtype=float) / 255
# SOUNDING_COLOUR = numpy.array([117, 112, 179], dtype=float) / 255
# NO_PERMUTATION_COLOUR = numpy.array([27, 158, 119], dtype=float) / 255

DEFAULT_FACE_COLOUR = numpy.array([252, 141, 98], dtype=float) / 255
SOUNDING_COLOUR = numpy.array([141, 160, 203], dtype=float) / 255
NO_PERMUTATION_COLOUR = numpy.full(3, 1.)

DEFAULT_EDGE_COLOUR = numpy.full(3, 0.)
DEFAULT_REFERENCE_LINE_COLOUR = numpy.full(3, 152. / 255)

DEFAULT_EDGE_WIDTH = 2
DEFAULT_REFERENCE_LINE_WIDTH = 4

ERROR_BAR_COLOUR = numpy.full(3, 152. / 255)
# ERROR_BAR_COLOUR = numpy.full(3, 0.)
ERROR_BAR_CAP_SIZE = 6
ERROR_BAR_DICT = {'alpha': 0.75, 'linewidth': 4, 'capthick': 4}

TEXT_COLOUR = numpy.full(3, 0.)
FIGURE_WIDTH_INCHES = 15
FIGURE_HEIGHT_INCHES = 15

FONT_SIZE = 20
LABEL_FONT_SIZE = 20

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
    x_coord_for_text = x_min + 0.025 * (x_max - x_min)

    for j in range(len(y_coords)):
        axes_object.text(
            x_coord_for_text, y_coords[j], '   ' + y_strings[j],
            color=TEXT_COLOUR, horizontalalignment='left',
            verticalalignment='center', fontsize=LABEL_FONT_SIZE)


def _predictor_name_to_face_colour(predictor_name):
    """Converts predictor name to face colour for bar graph.

    :param predictor_name: Predictor name (string).
    :return: face_colour: Colour as length-3 numpy array.
    """

    if predictor_name in SOUNDING_PREDICTOR_NAMES:
        return SOUNDING_COLOUR

    return DEFAULT_FACE_COLOUR


def plot_breiman_results(
        permutation_dict, axes_object, num_predictors_to_plot=None,
        plot_percent_increase=False, bar_face_colour=None,
        bar_edge_colour=DEFAULT_EDGE_COLOUR,
        bar_edge_width=DEFAULT_EDGE_WIDTH,
        reference_line_colour=DEFAULT_REFERENCE_LINE_COLOUR,
        reference_line_width=DEFAULT_REFERENCE_LINE_WIDTH):
    """Plots results of Breiman (single-pass) permutation test.

    :param permutation_dict: Dictionary created by
        `permutation.run_permutation_test`.
    :param axes_object: Will plot on these axes (instance of
        `matplotlib.axes._subplots.AxesSubplot`).
    :param num_predictors_to_plot: Number of predictors to plot.  Will plot only
        the K most important, where K = `num_predictors_to_plot`.  If
        `num_predictors_to_plot is None`, will plot all predictors.
    :param plot_percent_increase: Boolean flag.  If True, x-axis will be
        percentage of original cost (before permutation).  If False, will be
        actual cost.
    :param bar_face_colour: Interior colour (any format accepted by
        `matplotlib.colors`) of each bar in the graph.  If this is None, will
        use the method `_predictor_name_to_face_colour` to make bar colours.
    :param bar_edge_colour: Edge colour of each bar in the graph.
    :param bar_edge_width: Edge width of each bar in the graph.
    :param reference_line_colour: Colour of reference line (dashed vertical
        line, showing cost with no permutation).
    :param reference_line_width: Width of reference line.
    """

    error_checking.assert_is_boolean(plot_percent_increase)
    predictor_names = permutation_dict[permutation.STEP1_PREDICTORS_KEY]

    if num_predictors_to_plot is None:
        num_predictors_to_plot = len(predictor_names)

    error_checking.assert_is_integer(num_predictors_to_plot)
    error_checking.assert_is_greater(num_predictors_to_plot, 0)
    num_predictors_to_plot = min([num_predictors_to_plot, len(predictor_names)])

    original_cost_bs_array = permutation_dict[permutation.ORIGINAL_COST_KEY]
    cost_by_predictor_bs_matrix = permutation_dict[permutation.STEP1_COSTS_KEY]

    sort_indices = numpy.argsort(
        cost_by_predictor_bs_matrix[:, 1]
    )[-num_predictors_to_plot:]

    cost_by_predictor_bs_matrix = cost_by_predictor_bs_matrix[sort_indices, :]
    predictor_names = [predictor_names[k] for k in sort_indices]

    original_cost_bs_matrix = numpy.reshape(
        original_cost_bs_array, (1, original_cost_bs_array.size)
    )

    x_coord_matrix = numpy.concatenate(
        (original_cost_bs_matrix, cost_by_predictor_bs_matrix), axis=0)

    if numpy.any(x_coord_matrix < 0):
        x_coord_matrix *= -1
        x_label_string = 'AUC'

        if plot_percent_increase:
            x_coord_matrix = 200 * (x_coord_matrix - 0.5)
            x_label_string += ' (percent improvement above 0.5)'
    else:
        x_label_string = 'Cost'

        if plot_percent_increase:
            x_coord_matrix = 100 * x_coord_matrix / x_coord_matrix[0, 1]
            x_label_string += ' (percentage of original)'

    y_strings = ['No permutation'] + predictor_names
    y_coords = numpy.linspace(
        0, len(y_strings) - 1, num=len(y_strings), dtype=float)

    if bar_face_colour is None:
        bar_face_colours = [
            _predictor_name_to_face_colour(n) for n in predictor_names
        ]

        face_colour_arg = [NO_PERMUTATION_COLOUR] + bar_face_colours
    else:
        face_colour_arg = bar_face_colour

    negative_errors = x_coord_matrix[:, 1] - x_coord_matrix[:, 0]
    positive_errors = x_coord_matrix[:, 2] - x_coord_matrix[:, 1]

    negative_errors = numpy.reshape(
        negative_errors, (1, negative_errors.size)
    )
    positive_errors = numpy.reshape(
        positive_errors, (1, positive_errors.size)
    )

    error_matrix = numpy.vstack((
        negative_errors, positive_errors))

    axes_object.barh(
        y_coords, x_coord_matrix[:, 1], color=face_colour_arg,
        edgecolor=bar_edge_colour, linewidth=bar_edge_width,
        xerr=error_matrix, ecolor=ERROR_BAR_COLOUR,
        capsize=ERROR_BAR_CAP_SIZE, error_kw=ERROR_BAR_DICT)

    reference_x_coords = numpy.full(2, x_coord_matrix[0, 1])
    reference_y_coords = numpy.array(
        [numpy.min(y_coords) - 0.75, numpy.max(y_coords) + 0.75]
    )

    axes_object.plot(
        reference_x_coords, reference_y_coords, color=reference_line_colour,
        linestyle='--', linewidth=reference_line_width)

    axes_object.set_yticks([], [])
    axes_object.set_xlabel(x_label_string)
    axes_object.set_ylabel('Predictor permuted')

    _label_bars(axes_object=axes_object, y_coords=y_coords, y_strings=y_strings)
    axes_object.set_ylim(numpy.min(y_coords) - 0.75, numpy.max(y_coords) + 0.75)


def plot_lakshmanan_results(
        permutation_dict, axes_object, num_steps_to_plot=None,
        plot_percent_increase=False, bar_face_colour=None,
        bar_edge_colour=DEFAULT_EDGE_COLOUR,
        bar_edge_width=DEFAULT_EDGE_WIDTH,
        reference_line_colour=DEFAULT_REFERENCE_LINE_COLOUR,
        reference_line_width=DEFAULT_REFERENCE_LINE_WIDTH):
    """Plots results of Lakshmanan (multi-pass) permutation test.

    :param permutation_dict: See doc for `plot_breiman_results`.
    :param axes_object: Same.
    :param num_steps_to_plot: See doc for `num_predictors_to_plot` in
        `plot_breiman_results`.
    :param plot_percent_increase: See doc for `plot_breiman_results`.
    :param bar_face_colour: Same.
    :param bar_edge_colour: Same.
    :param bar_edge_width: Same.
    :param reference_line_colour: Same.
    :param reference_line_width: Same.
    """

    error_checking.assert_is_boolean(plot_percent_increase)

    highest_cost_by_step_bs_matrix = permutation_dict[
        permutation.HIGHEST_COSTS_KEY]
    predictor_name_by_step = permutation_dict[
        permutation.SELECTED_PREDICTORS_KEY]

    if num_steps_to_plot is None:
        num_steps_to_plot = len(predictor_name_by_step)

    error_checking.assert_is_integer(num_steps_to_plot)
    error_checking.assert_is_greater(num_steps_to_plot, 0)
    num_steps_to_plot = min([
        num_steps_to_plot, len(predictor_name_by_step)
    ])

    highest_cost_by_step_bs_matrix = highest_cost_by_step_bs_matrix[
        :num_steps_to_plot, :]
    predictor_name_by_step = predictor_name_by_step[:num_steps_to_plot]

    original_cost_bs_array = permutation_dict[permutation.ORIGINAL_COST_KEY]
    original_cost_bs_matrix = numpy.reshape(
        original_cost_bs_array, (1, original_cost_bs_array.size)
    )

    x_coord_matrix = numpy.concatenate(
        (original_cost_bs_matrix, highest_cost_by_step_bs_matrix), axis=0)

    if numpy.any(x_coord_matrix < 0):
        x_coord_matrix *= -1
        x_label_string = 'AUC'

        if plot_percent_increase:
            x_coord_matrix = 200 * (x_coord_matrix - 0.5)
            x_label_string += ' (percent improvement above 0.5)'
    else:
        x_label_string = 'Cost'

        if plot_percent_increase:
            x_coord_matrix = 100 * x_coord_matrix / x_coord_matrix[0, 1]
            x_label_string += ' (percentage of original)'

    y_strings = ['No permutation'] + predictor_name_by_step
    y_coords = numpy.linspace(
        0, len(y_strings) - 1, num=len(y_strings), dtype=float
    )[::-1]

    if bar_face_colour is None:
        bar_face_colours = [
            _predictor_name_to_face_colour(n) for n in predictor_name_by_step
        ]

        face_colour_arg = [NO_PERMUTATION_COLOUR] + bar_face_colours
    else:
        face_colour_arg = bar_face_colour

    negative_errors = x_coord_matrix[:, 1] - x_coord_matrix[:, 0]
    positive_errors = x_coord_matrix[:, 2] - x_coord_matrix[:, 1]

    negative_errors = numpy.reshape(
        negative_errors, (1, negative_errors.size)
    )
    positive_errors = numpy.reshape(
        positive_errors, (1, positive_errors.size)
    )

    error_matrix = numpy.vstack((
        negative_errors, positive_errors))

    axes_object.barh(
        y_coords, x_coord_matrix[:, 1], color=face_colour_arg,
        edgecolor=bar_edge_colour, linewidth=bar_edge_width,
        xerr=error_matrix, ecolor=ERROR_BAR_COLOUR,
        capsize=ERROR_BAR_CAP_SIZE, error_kw=ERROR_BAR_DICT)

    reference_x_coords = numpy.full(2, x_coord_matrix[0, 1])
    reference_y_coords = numpy.array(
        [numpy.min(y_coords) - 0.75, numpy.max(y_coords) + 0.75]
    )

    axes_object.plot(
        reference_x_coords, reference_y_coords, color=reference_line_colour,
        linestyle='--', linewidth=reference_line_width)

    axes_object.set_yticks([], [])
    axes_object.set_xlabel(x_label_string)
    axes_object.set_ylabel('Predictor permuted')

    _label_bars(axes_object=axes_object, y_coords=y_coords, y_strings=y_strings)
    axes_object.set_ylim(numpy.min(y_coords) - 0.75, numpy.max(y_coords) + 0.75)
