"""Plots results of permutation test for predictor importance."""

import numpy
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot
from scipy.stats import percentileofscore
from gewittergefahr.gg_utils import error_checking
from gewittergefahr.deep_learning import permutation_utils
from gewittergefahr.plotting import plotting_utils

DEFAULT_CONFIDENCE_LEVEL = 0.95

SOUNDING_PREDICTOR_NAMES = [
    r'$u$-wind',
    r'$v$-wind',
    'Relative humidity',
    'Specific humidity',
    'Virtual potential temperature'
]

# DEFAULT_FACE_COLOUR = numpy.array([252, 141, 98], dtype=float) / 255
# SOUNDING_COLOUR = numpy.array([141, 160, 203], dtype=float) / 255

DEFAULT_FACE_COLOUR = numpy.array([27, 158, 119], dtype=float) / 255
SOUNDING_COLOUR = numpy.array([217, 95, 2], dtype=float) / 255
NO_PERMUTATION_COLOUR = numpy.full(3, 1.)

BAR_EDGE_WIDTH = 2
BAR_EDGE_COLOUR = numpy.full(3, 0.)

REFERENCE_LINE_WIDTH = 4
REFERENCE_LINE_COLOUR = numpy.full(3, 152. / 255)

# ERROR_BAR_COLOUR = numpy.full(3, 0.)
ERROR_BAR_COLOUR = numpy.array([117, 112, 179], dtype=float) / 255
ERROR_BAR_CAP_SIZE = 8
ERROR_BAR_DICT = {'alpha': 0.5, 'linewidth': 4, 'capthick': 4}

BAR_TEXT_COLOUR = numpy.full(3, 0.)
BAR_FONT_SIZE = 22
DEFAULT_FONT_SIZE = 30
FIGURE_WIDTH_INCHES = 15
FIGURE_HEIGHT_INCHES = 15

pyplot.rc('font', size=DEFAULT_FONT_SIZE)
pyplot.rc('axes', titlesize=DEFAULT_FONT_SIZE)
pyplot.rc('axes', labelsize=DEFAULT_FONT_SIZE)
pyplot.rc('xtick', labelsize=DEFAULT_FONT_SIZE)
pyplot.rc('ytick', labelsize=DEFAULT_FONT_SIZE)
pyplot.rc('legend', fontsize=DEFAULT_FONT_SIZE)
pyplot.rc('figure', titlesize=DEFAULT_FONT_SIZE)


def _label_bars(axes_object, y_tick_coords, y_tick_strings, significant_flags):
    """Labels bars in graph.

    J = number of bars

    :param axes_object: Will plot on these axes (instance of
        `matplotlib.axes._subplots.AxesSubplot`).
    :param y_tick_coords: length-J numpy array with y-coordinates of bars.
    :param y_tick_strings: length-J list of labels.
    :param significant_flags: length-J numpy array of Boolean flags.  If
        significant_flags[i] = True, the [i]th step has a significantly
        different cost than the [i + 1]th step.
    """

    this_colour = plotting_utils.colour_from_numpy_to_tuple(BAR_TEXT_COLOUR)

    for j in range(len(y_tick_coords)):
        y_tick_strings[j] = y_tick_strings[j].replace(
            'Surface geopotential height', 'Orographic height'
        )

        axes_object.text(
            0., y_tick_coords[j], '      ' + y_tick_strings[j],
            color=this_colour, horizontalalignment='left',
            verticalalignment='center',
            fontweight='bold' if significant_flags[j] else 'normal',
            fontsize=BAR_FONT_SIZE
        )


def _predictor_name_to_face_colour(predictor_name):
    """Converts predictor name to face colour for bar graph.

    :param predictor_name: Predictor name (string).
    :return: face_colour: Colour as length-3 tuple.
    """

    if predictor_name in SOUNDING_PREDICTOR_NAMES:
        return plotting_utils.colour_from_numpy_to_tuple(SOUNDING_COLOUR)

    return plotting_utils.colour_from_numpy_to_tuple(DEFAULT_FACE_COLOUR)


def _get_error_matrix(cost_matrix, is_cost_auc, confidence_level,
                      backwards_flag, multipass_flag):
    """Creates error matrix (used to plot error bars).

    S = number of steps in permutation test
    B = number of bootstrap replicates

    :param cost_matrix: S-by-B numpy array of costs.
    :param is_cost_auc: Boolean flag.  If True, cost function is AUC (area under
        receiver-operating-characteristic curve).
    :param confidence_level: Confidence level (in range 0...1).
    :param backwards_flag: Boolean flag, indicating whether the test is forward
        or backwards.
    :param multipass_flag: Boolean flag, indicating whether the test is
        single-pass or multi-pass.
    :return: error_matrix: 2-by-S numpy array, where the first row contains
        negative errors and second row contains positive errors.
    :return: significant_flags: length-S numpy array of Boolean flags.  If
        significant_flags[i] = True, the [i]th step has a significantly
        different cost than the [i + 1]th step.
    """

    num_steps = cost_matrix.shape[0]
    significant_flags = numpy.full(num_steps, False, dtype=bool)

    for i in range(num_steps - 1):
        if backwards_flag:
            these_diffs = cost_matrix[i + 1, :] - cost_matrix[i, :]
        else:
            these_diffs = cost_matrix[i, :] - cost_matrix[i + 1, :]

        if not is_cost_auc:
            these_diffs *= -1

        print(numpy.mean(these_diffs))

        this_percentile = percentileofscore(
            a=these_diffs, score=0., kind='mean'
        )

        if multipass_flag:
            significant_flags[i] = this_percentile <= 5.
        else:
            significant_flags[i + 1] = this_percentile <= 5.

        print((
            'Percentile of 0 in (cost at step {0:d}) - (cost at step {1:d}) = '
            '{2:.4f}'
        ).format(
            i + 1, i, this_percentile
        ))

    print(significant_flags)
    print('\n')

    error_checking.assert_is_geq(confidence_level, 0.9)
    error_checking.assert_is_less_than(confidence_level, 1.)

    mean_costs = numpy.mean(cost_matrix, axis=-1)
    min_costs = numpy.percentile(
        cost_matrix, 50 * (1. - confidence_level), axis=-1
    )
    max_costs = numpy.percentile(
        cost_matrix, 50 * (1. + confidence_level), axis=-1
    )

    negative_errors = mean_costs - min_costs
    positive_errors = max_costs - mean_costs

    negative_errors = numpy.reshape(negative_errors, (1, negative_errors.size))
    positive_errors = numpy.reshape(positive_errors, (1, positive_errors.size))
    error_matrix = numpy.vstack((negative_errors, positive_errors))

    return error_matrix, significant_flags


def _plot_bars(
        cost_matrix, clean_cost_array, predictor_names,
        plot_percent_increase, backwards_flag, multipass_flag, confidence_level,
        axes_object, bar_face_colour):
    """Plots bar graph for either single-pass or multi-pass test.

    P = number of predictors permuted or unpermuted
    B = number of bootstrap replicates

    :param cost_matrix: (P + 1)-by-B numpy array of costs.  The first row
        contains costs at the beginning of the test -- before (un)permuting any
        variables -- and the [i]th row contains costs after (un)permuting the
        variable represented by predictor_names[i - 1].
    :param clean_cost_array: length-B numpy array of costs with clean
        (unpermuted) predictors.
    :param predictor_names: length-P list of predictor names (used to label
        bars).
    :param plot_percent_increase: Boolean flag.  If True, the x-axis will show
        percentage of original cost.  If False, will show actual cost.
    :param backwards_flag: Boolean flag.  If True, will plot backwards version
        of permutation, where each step involves *un*permuting a variable.  If
        False, will plot forward version, where each step involves permuting a
        variable.
    :param multipass_flag: Boolean flag.  If True, plotting multi-pass version
        of test.  If False, plotting single-pass version.
    :param confidence_level: Confidence level for error bars (in range 0...1).
    :param axes_object: Will plot on these axes (instance of
        `matplotlib.axes._subplots.AxesSubplot`).  If None, will create new
        axes.
    :param bar_face_colour: Interior colour (in any format accepted by
        matplotlib), used for each bar in the graph.  If None, will use the
        method `_predictor_name_to_face_colour` to determine bar colours.
    """

    mean_clean_cost = numpy.mean(clean_cost_array)
    is_cost_auc = numpy.any(cost_matrix < 0)

    if is_cost_auc:
        cost_matrix *= -1
        mean_clean_cost *= -1

        if plot_percent_increase:
            cost_matrix = 2 * (cost_matrix - 0.5)
            mean_clean_cost = 2 * (mean_clean_cost - 0.5)
            x_axis_label_string = 'AUC (fractional improvement over random)'
        else:
            x_axis_label_string = 'Area under ROC curve'
    else:
        x_axis_label_string = 'Cross-entropy'

        if plot_percent_increase:
            cost_matrix = cost_matrix / mean_clean_cost
            mean_clean_cost = 1.
            x_axis_label_string += ' (fraction of original)'

    if backwards_flag:
        y_tick_strings = ['All permuted'] + predictor_names
    else:
        y_tick_strings = ['None permuted'] + predictor_names

    y_tick_coords = numpy.linspace(
        0, len(y_tick_strings) - 1, num=len(y_tick_strings), dtype=float
    )

    if multipass_flag:
        y_tick_coords = y_tick_coords[::-1]

    if bar_face_colour is None:
        face_colour_arg = [
            _predictor_name_to_face_colour(n) for n in predictor_names
        ]

        face_colour_arg.insert(
            0, plotting_utils.colour_from_numpy_to_tuple(NO_PERMUTATION_COLOUR)
        )
    else:
        face_colour_arg = plotting_utils.colour_from_numpy_to_tuple(
            bar_face_colour)

    if axes_object is None:
        _, axes_object = pyplot.subplots(
            1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
        )

    mean_costs = numpy.mean(cost_matrix, axis=-1)
    num_steps = cost_matrix.shape[0]
    num_bootstrap_reps = cost_matrix.shape[1]

    if num_bootstrap_reps > 1:
        error_matrix, significant_flags = _get_error_matrix(
            cost_matrix=cost_matrix, is_cost_auc=is_cost_auc,
            confidence_level=confidence_level,
            backwards_flag=backwards_flag, multipass_flag=multipass_flag
        )

        x_min = numpy.min(mean_costs - error_matrix[0, :])
        x_max = numpy.max(mean_costs + error_matrix[1, :])

        axes_object.barh(
            y_tick_coords, mean_costs, color=face_colour_arg,
            edgecolor=plotting_utils.colour_from_numpy_to_tuple(
                BAR_EDGE_COLOUR),
            linewidth=BAR_EDGE_WIDTH, xerr=error_matrix,
            ecolor=plotting_utils.colour_from_numpy_to_tuple(ERROR_BAR_COLOUR),
            capsize=ERROR_BAR_CAP_SIZE, error_kw=ERROR_BAR_DICT
        )
    else:
        significant_flags = numpy.full(num_steps, False, dtype=bool)
        x_min = numpy.min(mean_costs)
        x_max = numpy.max(mean_costs)

        axes_object.barh(
            y_tick_coords, mean_costs, color=face_colour_arg,
            edgecolor=plotting_utils.colour_from_numpy_to_tuple(
                BAR_EDGE_COLOUR),
            linewidth=BAR_EDGE_WIDTH
        )

    reference_x_coords = numpy.full(2, mean_clean_cost)
    reference_y_tick_coords = numpy.array([
        numpy.min(y_tick_coords) - 0.75, numpy.max(y_tick_coords) + 0.75
    ])

    axes_object.plot(
        reference_x_coords, reference_y_tick_coords,
        color=plotting_utils.colour_from_numpy_to_tuple(REFERENCE_LINE_COLOUR),
        linestyle='--', linewidth=REFERENCE_LINE_WIDTH
    )

    axes_object.set_yticks([], [])
    axes_object.set_xlabel(x_axis_label_string)

    if backwards_flag:
        axes_object.set_ylabel('Variable cleaned')
    else:
        axes_object.set_ylabel('Variable permuted')

    axes_object.set_xlim(
        numpy.min(y_tick_coords) - 0.75, numpy.max(y_tick_coords) + 0.75
    )

    x_max *= 1.01
    if x_min <= 0:
        x_min *= 1.01
    else:
        x_min = 0.

    axes_object.set_xlim(x_min, x_max)

    _label_bars(
        axes_object=axes_object, y_tick_coords=y_tick_coords,
        y_tick_strings=y_tick_strings, significant_flags=significant_flags
    )

    axes_object.set_ylim(
        numpy.min(y_tick_coords) - 0.75, numpy.max(y_tick_coords) + 0.75
    )


def plot_single_pass_test(
        permutation_dict, axes_object=None, num_predictors_to_plot=None,
        plot_percent_increase=False, confidence_level=DEFAULT_CONFIDENCE_LEVEL,
        bar_face_colour=None):
    """Plots results of single-pass (Breiman) permutation test.

    :param permutation_dict: Dictionary created by
        `permutation.run_forward_test` or `permutation.run_backwards_test`.
    :param axes_object: See doc for `_plot_bars`.
    :param num_predictors_to_plot: Number of predictors to plot.  Will plot only
        the K most important, where K = `num_predictors_to_plot`.  If None, will
        plot all predictors.
    :param plot_percent_increase: See doc for `_plot_bars`.
    :param confidence_level: Same.
    :param bar_face_colour: Same.
    """

    # Check input args.
    predictor_names = permutation_dict[permutation_utils.STEP1_PREDICTORS_KEY]
    if num_predictors_to_plot is None:
        num_predictors_to_plot = len(predictor_names)

    error_checking.assert_is_integer(num_predictors_to_plot)
    error_checking.assert_is_greater(num_predictors_to_plot, 0)
    num_predictors_to_plot = min([
        num_predictors_to_plot, len(predictor_names)
    ])

    error_checking.assert_is_boolean(plot_percent_increase)

    # Set up plotting args.
    backwards_flag = permutation_dict[permutation_utils.BACKWARDS_FLAG]
    perturbed_cost_matrix = permutation_dict[
        permutation_utils.STEP1_COST_MATRIX_KEY]
    mean_perturbed_costs = numpy.mean(perturbed_cost_matrix, axis=-1)

    if backwards_flag:
        sort_indices = numpy.argsort(
            mean_perturbed_costs
        )[:num_predictors_to_plot][::-1]
    else:
        sort_indices = numpy.argsort(
            mean_perturbed_costs
        )[-num_predictors_to_plot:]

    perturbed_cost_matrix = perturbed_cost_matrix[sort_indices, :]
    predictor_names = [predictor_names[k] for k in sort_indices]

    original_cost_array = permutation_dict[
        permutation_utils.ORIGINAL_COST_ARRAY_KEY
    ]
    original_cost_matrix = numpy.reshape(
        original_cost_array, (1, original_cost_array.size)
    )
    cost_matrix = numpy.concatenate(
        (original_cost_matrix, perturbed_cost_matrix), axis=0
    )

    # Do plotting.
    if backwards_flag:
        clean_cost_array = permutation_dict[
            permutation_utils.BEST_COST_MATRIX_KEY][-1, :]
    else:
        clean_cost_array = original_cost_array

    _plot_bars(
        cost_matrix=cost_matrix, clean_cost_array=clean_cost_array,
        predictor_names=predictor_names,
        plot_percent_increase=plot_percent_increase,
        backwards_flag=backwards_flag, multipass_flag=False,
        confidence_level=confidence_level, axes_object=axes_object,
        bar_face_colour=bar_face_colour)


def plot_multipass_test(
        permutation_dict, axes_object=None, num_predictors_to_plot=None,
        plot_percent_increase=False, confidence_level=DEFAULT_CONFIDENCE_LEVEL,
        bar_face_colour=None):
    """Plots results of multi-pass (Lakshmanan) permutation test.

    :param permutation_dict: See doc for `plot_single_pass_test`.
    :param axes_object: Same.
    :param num_predictors_to_plot: Same.
    :param plot_percent_increase: Same.
    :param confidence_level: Same.
    :param bar_face_colour: Same.
    """

    # Check input args.
    predictor_names = permutation_dict[permutation_utils.BEST_PREDICTORS_KEY]
    if num_predictors_to_plot is None:
        num_predictors_to_plot = len(predictor_names)

    error_checking.assert_is_integer(num_predictors_to_plot)
    error_checking.assert_is_greater(num_predictors_to_plot, 0)
    num_predictors_to_plot = min([
        num_predictors_to_plot, len(predictor_names)
    ])

    error_checking.assert_is_boolean(plot_percent_increase)

    # Set up plotting args.
    backwards_flag = permutation_dict[permutation_utils.BACKWARDS_FLAG]
    perturbed_cost_matrix = permutation_dict[
        permutation_utils.BEST_COST_MATRIX_KEY]

    perturbed_cost_matrix = perturbed_cost_matrix[:num_predictors_to_plot, :]
    predictor_names = predictor_names[:num_predictors_to_plot]

    original_cost_array = permutation_dict[
        permutation_utils.ORIGINAL_COST_ARRAY_KEY
    ]
    original_cost_matrix = numpy.reshape(
        original_cost_array, (1, original_cost_array.size)
    )
    cost_matrix = numpy.concatenate(
        (original_cost_matrix, perturbed_cost_matrix), axis=0
    )

    # Do plotting.
    if backwards_flag:
        clean_cost_array = permutation_dict[
            permutation_utils.BEST_COST_MATRIX_KEY][-1, :]
    else:
        clean_cost_array = original_cost_array

    _plot_bars(
        cost_matrix=cost_matrix, clean_cost_array=clean_cost_array,
        predictor_names=predictor_names,
        plot_percent_increase=plot_percent_increase,
        backwards_flag=backwards_flag, multipass_flag=True,
        confidence_level=confidence_level, axes_object=axes_object,
        bar_face_colour=bar_face_colour)
