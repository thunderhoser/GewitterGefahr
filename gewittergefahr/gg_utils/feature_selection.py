"""Implements various feature-selection algorithms.

--- REFERENCES ---

Lakshmanan, V., C. Karstens, J. Krause, K. Elmore, A. Ryzhkov, and S. Berkseth,
    2015: "Which polarimetric variables are important for weather/no-weather
    discrimination?" Journal of Atmospheric and Oceanic Technology, 32 (6),
    1209-1223.

Webb, A.R., 2003: "Statistical Pattern Recognition". John Wiley & Sons.
"""

import copy
from itertools import combinations
import numpy
import pandas
import keras.utils
import sklearn.base
import sklearn.metrics
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as pyplot
from gewittergefahr.gg_utils import model_evaluation as model_eval
from gewittergefahr.gg_utils import error_checking

# TODO(thunderhoser): A lot of this module is "dead code" right now.  Some of it
# assumes that the task is binary classification, and some of it assumes
# classification in general (binary or multi-class).

FONT_SIZE = 30
FEATURE_NAME_FONT_SIZE = 18
FIG_WIDTH_INCHES = 15
FIG_HEIGHT_INCHES = 15
FIG_PADDING_AT_BOTTOM_PERCENT = 25.
DOTS_PER_INCH = 600

DEFAULT_BAR_FACE_COLOUR = numpy.array([228., 26., 28.]) / 255
DEFAULT_BAR_EDGE_COLOUR = numpy.array([0., 0., 0.]) / 255
DEFAULT_BAR_EDGE_WIDTH = 2.

pyplot.rc('font', size=FONT_SIZE)
pyplot.rc('axes', titlesize=FONT_SIZE)
pyplot.rc('axes', labelsize=FONT_SIZE)
pyplot.rc('xtick', labelsize=FONT_SIZE)
pyplot.rc('ytick', labelsize=FONT_SIZE)
pyplot.rc('legend', fontsize=FONT_SIZE)
pyplot.rc('figure', titlesize=FONT_SIZE)

MIN_FRACTIONAL_COST_DECREASE_SFS_DEFAULT = 0.01
MIN_FRACTIONAL_COST_DECREASE_SBS_DEFAULT = -0.01

DEFAULT_NUM_FORWARD_STEPS_FOR_SFS = 2
DEFAULT_NUM_BACKWARD_STEPS_FOR_SFS = 1
DEFAULT_NUM_FORWARD_STEPS_FOR_SBS = 1
DEFAULT_NUM_BACKWARD_STEPS_FOR_SBS = 2

SELECTED_FEATURES_KEY = 'selected_feature_names'
REMOVED_FEATURES_KEY = 'removed_feature_names'
FEATURE_NAME_KEY = 'feature_name'
VALIDATION_COST_KEY = 'validation_cost'
VALIDATION_XENTROPY_KEY = 'validation_cross_entropy'
VALIDATION_AUC_KEY = 'validation_auc'
TESTING_XENTROPY_KEY = 'testing_cross_entropy'
TESTING_AUC_KEY = 'testing_auc'
VALIDATION_COST_BY_STEP_KEY = 'validation_cost_by_step'

PERMUTATION_TYPE = 'permutation'
FORWARD_SELECTION_TYPE = 'forward'
BACKWARD_SELECTION_TYPE = 'backward'

MIN_PROBABILITY = 1e-15
MAX_PROBABILITY = 1. - MIN_PROBABILITY


def _check_sequential_selection_inputs(
        training_table, validation_table, feature_names, target_name,
        num_features_to_add_per_step=1, num_features_to_remove_per_step=1,
        testing_table=None):
    """Checks inputs for sequential forward or backward selection.

    :param training_table: pandas DataFrame, where each row is one training
        example.
    :param validation_table: pandas DataFrame, where each row is one validation
        example.
    :param feature_names: length-F list with names of features (predictor
        variables).  Each feature must be a column in training_table,
        validation_table, and testing_table.
    :param target_name: Name of target variable (predictand).  Must be a column
        in training_table, validation_table, and testing_table.
    :param num_features_to_add_per_step: Number of features to add at each
        forward step.
    :param num_features_to_remove_per_step: Number of features to remove at each
        backward step.
    :param testing_table: pandas DataFrame, where each row is one testing
        example.
    """

    error_checking.assert_is_string_list(feature_names)
    error_checking.assert_is_numpy_array(
        numpy.asarray(feature_names), num_dimensions=1)

    error_checking.assert_is_string(target_name)
    variable_names = feature_names + [target_name]
    error_checking.assert_columns_in_dataframe(training_table, variable_names)
    error_checking.assert_columns_in_dataframe(validation_table, variable_names)
    if testing_table is not None:
        error_checking.assert_columns_in_dataframe(
            testing_table, variable_names)

    num_features = len(feature_names)
    error_checking.assert_is_integer(num_features_to_add_per_step)
    error_checking.assert_is_geq(num_features_to_add_per_step, 1)
    error_checking.assert_is_less_than(
        num_features_to_add_per_step, num_features)

    error_checking.assert_is_integer(num_features_to_remove_per_step)
    error_checking.assert_is_geq(num_features_to_remove_per_step, 1)
    error_checking.assert_is_less_than(
        num_features_to_remove_per_step, num_features)

    # Ensure that label is binary.
    error_checking.assert_is_integer_numpy_array(
        training_table[target_name].values)
    error_checking.assert_is_geq_numpy_array(
        training_table[target_name].values, 0)
    error_checking.assert_is_leq_numpy_array(
        training_table[target_name].values, 1)


def _forward_selection_step(
        training_table, validation_table, selected_feature_names,
        remaining_feature_names, target_name, estimator_object, cost_function,
        num_features_to_add=1):
    """Performs one forward selection step (i.e., adds features to the model).

    The best set of L features is added to the model, where L >= 1.

    Each member of `selected_feature_names` and `remaining_feature_names`, as
    well as `target_name`, must be a column in both `training_table` and
    `validation_table`.

    :param training_table: pandas DataFrame, where each row is one training
        example.
    :param validation_table: pandas DataFrame, where each row is one validation
        example.
    :param selected_feature_names: 1-D list with names of selected features
        (those already in the model).
    :param remaining_feature_names: 1-D list with names of remaining features
        (those which may be added to the model).
    :param target_name: Name of target variable (predictand).
    :param estimator_object: Instance of scikit-learn estimator.  Must implement
        the methods `fit` and `predict_proba`.
    :param cost_function: Cost function to be minimized.  Should have the
        following format.  E = number of examples, and K = number of classes.
    Input: class_probability_matrix: E-by-K numpy array of predicted
        probabilities, where class_probability_matrix[i, k] = probability that
        [i]th example belongs to [k]th class.
    Input: observed_values: length-E numpy array of observed values (integer
        class labels).
    Output: cost: Scalar value.

    :param num_features_to_add: Number of features to add (L in the above
        discussion).
    :return: min_cost: Minimum cost given by adding any set of L features from
        `remaining_feature_names` to the model.
    :return: best_feature_names: length-L list of features whose addition
        resulted in `min_cost`.
    """

    combination_object = combinations(
        remaining_feature_names, num_features_to_add)
    list_of_remaining_feature_combos = []
    for this_tuple in list(combination_object):
        list_of_remaining_feature_combos.append(list(this_tuple))

    num_remaining_feature_combos = len(list_of_remaining_feature_combos)
    cost_by_feature_combo = numpy.full(num_remaining_feature_combos, numpy.nan)

    for j in range(num_remaining_feature_combos):
        these_feature_names = (
            selected_feature_names + list_of_remaining_feature_combos[j])

        new_estimator_object = sklearn.base.clone(estimator_object)
        new_estimator_object.fit(
            training_table.as_matrix(columns=these_feature_names),
            training_table[target_name].values)

        this_probability_matrix = new_estimator_object.predict_proba(
            validation_table.as_matrix(columns=these_feature_names))
        cost_by_feature_combo[j] = cost_function(
            this_probability_matrix, validation_table[target_name].values)

    min_cost = numpy.min(cost_by_feature_combo)
    best_index = numpy.argmin(cost_by_feature_combo)
    return min_cost, list_of_remaining_feature_combos[best_index]


def _backward_selection_step(
        training_table, validation_table, selected_feature_names, target_name,
        estimator_object, cost_function, num_features_to_remove=1):
    """Performs one backward selection step (removes features from the model).

    The worst set of R features is removed from the model, where R >= 1.

    Each member of `selected_feature_names`, as well as `target_name`, must be a
    column in both `training_table` and `validation_table`.

    :param training_table: pandas DataFrame, where each row is one training
        example.
    :param validation_table: pandas DataFrame, where each row is one validation
        example.
    :param selected_feature_names: 1-D list with names of selected features
        (each of which may be removed from the model).
    :param target_name: Name of target variable (predictand).
    :param estimator_object: Instance of scikit-learn estimator.  Must implement
        the methods `fit` and `predict_proba`.
    :param cost_function: See doc for `_forward_selection_step`.
    :param num_features_to_remove: Number of features to remove (R in the above
        discussion).
    :return: min_cost: Minimum cost given by removing any set of R features in
        `selected_feature_names` from the model.
    :return: best_feature_names: length-R list of features whose removal
        resulted in `min_cost`.
    """

    combination_object = combinations(
        selected_feature_names, num_features_to_remove)
    list_of_selected_feature_combos = []
    for this_tuple in list(combination_object):
        list_of_selected_feature_combos.append(list(this_tuple))

    num_selected_feature_combos = len(list_of_selected_feature_combos)
    cost_by_feature_combo = numpy.full(num_selected_feature_combos, numpy.nan)

    for j in range(num_selected_feature_combos):
        these_feature_names = set(selected_feature_names)
        for this_name in list_of_selected_feature_combos[j]:
            these_feature_names.remove(this_name)
        these_feature_names = list(these_feature_names)

        new_estimator_object = sklearn.base.clone(estimator_object)
        new_estimator_object.fit(
            training_table.as_matrix(columns=these_feature_names),
            training_table[target_name].values)

        this_probability_matrix = new_estimator_object.predict_proba(
            validation_table.as_matrix(columns=these_feature_names))
        cost_by_feature_combo[j] = cost_function(
            this_probability_matrix, validation_table[target_name].values)

    min_cost = numpy.min(cost_by_feature_combo)
    worst_index = numpy.argmin(cost_by_feature_combo)
    return min_cost, list_of_selected_feature_combos[worst_index]


def _evaluate_feature_selection(
        training_table, validation_table, testing_table, estimator_object,
        selected_feature_names, target_name):
    """Evaluates feature selection.

    Specifically, this method computes 4 performance metrics:

    - validation cross-entropy
    - validation AUC (area under ROC curve)
    - testing cross-entropy
    - testing AUC

    :param training_table: pandas DataFrame, where each row is one training
        example.
    :param validation_table: pandas DataFrame, where each row is one validation
        example.
    :param testing_table: pandas DataFrame, where each row is one testing
        example.
    :param estimator_object: Instance of scikit-learn estimator.  Must implement
        the methods `fit` and `predict_proba`.
    :param selected_feature_names: 1-D list with names of selected features.
        Each one must be a column in training_table, validation_table, and
        testing_table.
    :param target_name: Name of target variable (predictand).  Must be a column
        in training_table, validation_table, and testing_table.
    :return: feature_selection_dict: Dictionary with the following keys.
    feature_selection_dict['selected_feature_names']: 1-D list with names of
        selected features.
    feature_selection_dict['validation_cross_entropy']: Cross-entropy on
        validation data.
    feature_selection_dict['validation_auc']: Area under ROC curve on validation
        data.
    feature_selection_dict['testing_cross_entropy']: Cross-entropy on testing
        data.
    feature_selection_dict['testing_auc']: Area under ROC curve on testing data.
    """

    new_estimator_object = sklearn.base.clone(estimator_object)
    new_estimator_object.fit(
        training_table.as_matrix(columns=selected_feature_names),
        training_table[target_name].values)

    forecast_probs_for_validation = new_estimator_object.predict_proba(
        validation_table.as_matrix(columns=selected_feature_names))[:, 1]
    validation_cross_entropy = model_eval.get_cross_entropy(
        forecast_probs_for_validation,
        validation_table[target_name].values)
    validation_auc = sklearn.metrics.roc_auc_score(
        validation_table[target_name].values, forecast_probs_for_validation)

    forecast_probs_for_testing = new_estimator_object.predict_proba(
        testing_table.as_matrix(columns=selected_feature_names))[:, 1]
    testing_cross_entropy = model_eval.get_cross_entropy(
        forecast_probs_for_testing, testing_table[target_name].values)
    testing_auc = sklearn.metrics.roc_auc_score(
        testing_table[target_name].values, forecast_probs_for_testing)

    return {SELECTED_FEATURES_KEY: selected_feature_names,
            VALIDATION_XENTROPY_KEY: validation_cross_entropy,
            VALIDATION_AUC_KEY: validation_auc,
            TESTING_XENTROPY_KEY: testing_cross_entropy,
            TESTING_AUC_KEY: testing_auc}


def _plot_selection_results(
        used_feature_names, validation_cost_by_feature, selection_type,
        validation_cost_before_permutn=None, plot_feature_names=False,
        bar_face_colour=DEFAULT_BAR_FACE_COLOUR,
        bar_edge_colour=DEFAULT_BAR_EDGE_COLOUR,
        bar_edge_width=DEFAULT_BAR_EDGE_WIDTH):
    """Plots bar graph with results of feature-selection algorithm.

    K = number of features added, removed, or permuted

    :param used_feature_names: length-K list of feature names in order of their
        addition to the model, removal from the model, or permutation in the
        model's training data.
    :param validation_cost_by_feature: length-K list of validation costs.
        validation_cost_by_feature[k] is cost after adding, removing, or
        permuting feature_names[k].
    :param selection_type: Type of selection algorithm used.  May be "forward",
        "backward", or "permutation".
    :param validation_cost_before_permutn: Validation cost before permutation.
        If selection_type != "permutation", you can leave this as None.
    :param plot_feature_names: Boolean flag.  If True, will plot feature names
        on x-axis.  If False, will plot ordinal numbers on x-axis.
    :param bar_face_colour: Colour (in any format accepted by
        `matplotlib.colors`) for interior of bars.
    :param bar_edge_colour: Colour for edge of bars.
    :param bar_edge_width: Width for edge of bars.
    """

    num_features_used = len(used_feature_names)

    if selection_type == PERMUTATION_TYPE:
        x_coords_at_bar_edges = numpy.linspace(
            -0.5, num_features_used + 0.5, num=num_features_used + 2)
        y_values = numpy.concatenate((
            numpy.array([validation_cost_before_permutn]),
            validation_cost_by_feature))
    else:
        x_coords_at_bar_edges = numpy.linspace(
            0.5, num_features_used + 0.5, num=num_features_used + 1)
        y_values = copy.deepcopy(validation_cost_by_feature)

    x_width_of_bar = x_coords_at_bar_edges[1] - x_coords_at_bar_edges[0]
    x_coords_at_bar_centers = x_coords_at_bar_edges[:-1] + x_width_of_bar / 2

    figure_object, axes_object = pyplot.subplots(
        1, 1, figsize=(FIG_WIDTH_INCHES, FIG_HEIGHT_INCHES))
    figure_object.subplots_adjust(bottom=FIG_PADDING_AT_BOTTOM_PERCENT / 100)
    axes_object.bar(
        x_coords_at_bar_centers, y_values, x_width_of_bar,
        color=bar_face_colour, edgecolor=bar_edge_colour,
        linewidth=bar_edge_width)

    pyplot.xticks(x_coords_at_bar_centers, axes=axes_object)
    axes_object.set_xlim(numpy.min(x_coords_at_bar_edges),
                         numpy.max(x_coords_at_bar_edges))
    axes_object.set_ylim(0., 1.05 * numpy.max(y_values))

    if selection_type == PERMUTATION_TYPE:
        if plot_feature_names:
            axes_object.set_xlabel('Feature permuted')
        else:
            axes_object.set_xlabel('Number of features permuted')

    elif selection_type == FORWARD_SELECTION_TYPE:
        if plot_feature_names:
            axes_object.set_xlabel('Feature selected')
        else:
            axes_object.set_xlabel('Number of features selected')

    elif selection_type == BACKWARD_SELECTION_TYPE:
        if plot_feature_names:
            axes_object.set_xlabel('Feature removed')
        else:
            axes_object.set_xlabel('Number of features removed')

    if plot_feature_names:
        if selection_type == PERMUTATION_TYPE:
            pyplot.xticks(
                x_coords_at_bar_centers, [' '] + used_feature_names.tolist(),
                rotation='vertical', fontsize=FEATURE_NAME_FONT_SIZE)
        else:
            pyplot.xticks(x_coords_at_bar_centers, used_feature_names,
                          rotation='vertical', fontsize=FEATURE_NAME_FONT_SIZE)

    axes_object.set_ylabel('Validation cost')


def _cross_entropy_function(class_probability_matrix, observed_values):
    """Cross-entropy cost function.

    This function works for binary or multi-class classification.

    E = number of examples
    K = number of classes

    :param class_probability_matrix: E-by-K numpy array of predicted
        probabilities, where class_probability_matrix[i, k] = probability that
        [i]th example belongs to [k]th class.
    :param observed_values: length-E numpy array of observed values (integer
        class labels).
    :return: cross_entropy: Scalar.
    """

    num_examples = class_probability_matrix.shape[0]
    num_classes = class_probability_matrix.shape[1]

    class_probability_matrix[
        class_probability_matrix < MIN_PROBABILITY
        ] = MIN_PROBABILITY
    class_probability_matrix[
        class_probability_matrix > MAX_PROBABILITY
        ] = MAX_PROBABILITY

    target_matrix = keras.utils.to_categorical(
        observed_values, num_classes
    ).astype(int)

    return -1 * numpy.sum(
        target_matrix * numpy.log2(class_probability_matrix)
    ) / num_examples


def sequential_forward_selection(
        training_table, validation_table, testing_table, feature_names,
        target_name, estimator_object, cost_function=_cross_entropy_function,
        num_features_to_add_per_step=1, min_fractional_cost_decrease=
        MIN_FRACTIONAL_COST_DECREASE_SFS_DEFAULT):
    """Runs the SFS (sequential forward selection) algorithm.

    SFS is defined in Chapter 9 of Webb (2003).

    f = number of features selected

    :param training_table: See documentation for
        _check_sequential_selection_inputs.
    :param validation_table: See doc for _check_sequential_selection_inputs.
    :param testing_table: See doc for _check_sequential_selection_inputs.
    :param feature_names: See doc for _check_sequential_selection_inputs.
    :param target_name: See doc for _check_sequential_selection_inputs.
    :param estimator_object: Instance of scikit-learn estimator.  Must implement
        the methods `fit` and `predict_proba`.
    :param cost_function: See doc for `_forward_selection_step`.
    :param num_features_to_add_per_step: Number of features to add at each step.
    :param min_fractional_cost_decrease: Stopping criterion.  Once the
        fractional cost decrease over one step is <
        `min_fractional_cost_decrease`, SFS will stop.  Must be in range (0, 1).
    :return: sfs_dictionary: Same as output from _evaluate_feature_selection,
        but with one additional key.
    sfs_dictionary['validation_cost_by_step']: length-f numpy array of
        validation costs.  The [i]th element is the cost with i features added.
        In other words, validation_cost_by_step[0] is the cost with 1 feature
        added; validation_cost_by_step[1] is the cost with 2 features added;
        ...; etc.
    """

    _check_sequential_selection_inputs(
        training_table=training_table, validation_table=validation_table,
        testing_table=testing_table, feature_names=feature_names,
        target_name=target_name,
        num_features_to_add_per_step=num_features_to_add_per_step)

    error_checking.assert_is_greater(min_fractional_cost_decrease, 0.)
    error_checking.assert_is_less_than(min_fractional_cost_decrease, 1.)

    # Initialize values.
    selected_feature_names = []
    remaining_feature_names = copy.deepcopy(feature_names)

    num_features = len(feature_names)
    min_cost_by_num_selected = numpy.full(num_features + 1, numpy.nan)
    min_cost_by_num_selected[0] = numpy.inf

    while len(remaining_feature_names) >= num_features_to_add_per_step:
        num_selected_features = len(selected_feature_names)
        num_remaining_features = len(remaining_feature_names)

        print((
            'Step {0:d} of sequential forward selection: {1:d} features '
            'selected, {2:d} remaining...'
        ).format(
            num_selected_features + 1, num_selected_features,
            num_remaining_features
        ))

        min_new_cost, these_best_feature_names = _forward_selection_step(
            training_table=training_table, validation_table=validation_table,
            selected_feature_names=selected_feature_names,
            remaining_feature_names=remaining_feature_names,
            target_name=target_name, estimator_object=estimator_object,
            cost_function=cost_function,
            num_features_to_add=num_features_to_add_per_step)

        print((
            'Minimum cost ({0:.4f}) given by adding features shown below '
            '(previous minimum = {1:.4f}).\n{2:s}\n'
        ).format(
            min_new_cost, min_cost_by_num_selected[num_selected_features],
            str(these_best_feature_names)
        ))

        stopping_criterion = min_cost_by_num_selected[num_selected_features] * (
            1. - min_fractional_cost_decrease
        )

        if min_new_cost > stopping_criterion:
            break

        selected_feature_names += these_best_feature_names
        remaining_feature_names = [
            s for s in remaining_feature_names
            if s not in these_best_feature_names
        ]

        min_cost_by_num_selected[
            (num_selected_features + 1):(len(selected_feature_names) + 1)
        ] = min_new_cost

    sfs_dictionary = _evaluate_feature_selection(
        training_table=training_table, validation_table=validation_table,
        testing_table=testing_table, estimator_object=estimator_object,
        selected_feature_names=selected_feature_names, target_name=target_name)

    num_selected_features = len(selected_feature_names)

    sfs_dictionary.update({
        VALIDATION_COST_BY_STEP_KEY:
            min_cost_by_num_selected[1:(num_selected_features + 1)]
    })

    return sfs_dictionary


def sfs_with_backward_steps(
        training_table, validation_table, testing_table, feature_names,
        target_name, estimator_object, cost_function=_cross_entropy_function,
        num_forward_steps=DEFAULT_NUM_FORWARD_STEPS_FOR_SFS,
        num_backward_steps=DEFAULT_NUM_BACKWARD_STEPS_FOR_SFS,
        num_features_to_add_per_forward_step=1,
        num_features_to_remove_per_backward_step=1,
        min_fractional_cost_decrease=
        MIN_FRACTIONAL_COST_DECREASE_SFS_DEFAULT):
    """Runs SFS (sequential forward selection) with backward steps.

    This method is called "plus-l-minus-r selection" in Chapter 9 of Webb
    (2003), where l > r.

    --- DEFINITIONS ---

    "Forward step" = addition of `num_features_to_add_per_step` features.
    "Backward step" = removal of `num_features_to_remove_per_step` features.
    "Major step" = l forward steps followed by r backward steps.

    :param training_table: See documentation for
        _check_sequential_selection_inputs.
    :param validation_table: See doc for _check_sequential_selection_inputs.
    :param testing_table: See doc for _check_sequential_selection_inputs.
    :param feature_names: See doc for _check_sequential_selection_inputs.
    :param target_name: See doc for _check_sequential_selection_inputs.
    :param estimator_object: Instance of scikit-learn estimator.  Must implement
        the methods `fit` and `predict_proba`.
    :param cost_function: See doc for `_forward_selection_step`.
    :param num_forward_steps: l-value (number of forward steps per major step).
    :param num_backward_steps: r-value (number of backward steps per major
        step).
    :param num_features_to_add_per_forward_step: Number of features to add at
        each forward step.
    :param num_features_to_remove_per_backward_step: Number of features to
        remove at each backward step.
    :param min_fractional_cost_decrease: Stopping criterion.  Once the
        fractional cost decrease over a major step is <
        `min_fractional_cost_decrease`, SFS will stop.  Must be in range (0, 1).
    :return: sfs_dictionary: See doc for sequential_forward_selection.
    """

    _check_sequential_selection_inputs(
        training_table=training_table, validation_table=validation_table,
        testing_table=testing_table, feature_names=feature_names,
        target_name=target_name,
        num_features_to_add_per_step=num_features_to_add_per_forward_step,
        num_features_to_remove_per_step=
        num_features_to_remove_per_backward_step)

    error_checking.assert_is_integer(num_forward_steps)
    error_checking.assert_is_geq(num_forward_steps, 1)
    error_checking.assert_is_integer(num_backward_steps)
    error_checking.assert_is_geq(num_backward_steps, 1)

    num_features_to_add_per_major_step = (
        num_forward_steps * num_features_to_add_per_forward_step
    )
    num_features_to_remove_per_major_step = (
        num_backward_steps * num_features_to_remove_per_backward_step
    )

    num_features = len(feature_names)
    error_checking.assert_is_less_than(
        num_features_to_add_per_major_step, num_features)
    error_checking.assert_is_less_than(
        num_features_to_remove_per_major_step,
        num_features_to_add_per_major_step)

    error_checking.assert_is_greater(min_fractional_cost_decrease, 0.)
    error_checking.assert_is_less_than(min_fractional_cost_decrease, 1.)

    # Initialize values.
    selected_feature_names = []
    remaining_feature_names = copy.deepcopy(feature_names)

    min_cost_by_num_selected = numpy.full(num_features + 1, numpy.nan)
    min_cost_by_num_selected[0] = numpy.inf
    major_step_num = 0

    while (len(selected_feature_names) + num_features_to_add_per_major_step <=
           num_features):

        major_step_num += 1
        min_cost_last_major_step = min_cost_by_num_selected[
            len(selected_feature_names)
        ]
        selected_feature_names_last_major_step = copy.deepcopy(
            selected_feature_names)

        for i in range(num_forward_steps):
            num_selected_features = len(selected_feature_names)
            num_remaining_features = len(remaining_feature_names)

            print((
                'Major step {0:d}, forward step {1:d}: {2:d} features selected,'
                ' {3:d} remaining...'
            ).format(
                major_step_num, i + 1, num_selected_features,
                num_remaining_features
            ))

            min_new_cost, these_best_feature_names = _forward_selection_step(
                training_table=training_table,
                validation_table=validation_table,
                selected_feature_names=selected_feature_names,
                remaining_feature_names=remaining_feature_names,
                target_name=target_name, estimator_object=estimator_object,
                cost_function=cost_function,
                num_features_to_add=num_features_to_add_per_forward_step)

            print((
                'Minimum cost ({0:.4f}) given by adding features shown below '
                '(previous minimum = {1:.4f}).\n{2:s}\n'
            ).format(
                min_new_cost, min_cost_by_num_selected[num_selected_features],
                str(these_best_feature_names)
            ))

            selected_feature_names += these_best_feature_names
            remaining_feature_names = [
                s for s in remaining_feature_names
                if s not in these_best_feature_names]

            min_cost_by_num_selected[
            (num_selected_features + 1):(len(selected_feature_names) + 1)
            ] = min_new_cost

        for i in range(num_backward_steps):
            num_selected_features = len(selected_feature_names)

            print((
                'Major step {0:d}, backward step {1:d}: {2:d}/{3:d} features '
                'selected...'
            ).format(
                major_step_num, i + 1, num_selected_features, num_features
            ))

            min_new_cost, these_worst_feature_names = _backward_selection_step(
                training_table=training_table,
                validation_table=validation_table,
                selected_feature_names=selected_feature_names,
                target_name=target_name, estimator_object=estimator_object,
                cost_function=cost_function,
                num_features_to_remove=num_features_to_remove_per_backward_step)

            print((
                'Minimum cost ({0:.4f}) given by removing features shown below '
                '(previous minimum = {1:.4f}).\n{2:s}\n'
            ).format(
                min_new_cost, min_cost_by_num_selected[num_selected_features],
                str(these_worst_feature_names)
            ))

            remaining_feature_names += these_worst_feature_names
            selected_feature_names = [
                s for s in selected_feature_names
                if s not in these_worst_feature_names
            ]

            min_cost_by_num_selected[
                (len(selected_feature_names) + 1):
            ] = numpy.nan
            min_cost_by_num_selected[
                len(selected_feature_names)
            ] = min_new_cost

        print('\n')
        stopping_criterion = min_cost_last_major_step * (
            1. - min_fractional_cost_decrease
        )

        if (min_cost_by_num_selected[len(selected_feature_names)] >
                stopping_criterion):
            selected_feature_names = copy.deepcopy(
                selected_feature_names_last_major_step)
            break

    sfs_dictionary = _evaluate_feature_selection(
        training_table=training_table, validation_table=validation_table,
        testing_table=testing_table, estimator_object=estimator_object,
        selected_feature_names=selected_feature_names, target_name=target_name)

    num_selected_features = len(selected_feature_names)

    sfs_dictionary.update({
        VALIDATION_COST_BY_STEP_KEY:
            min_cost_by_num_selected[1:(num_selected_features + 1)]
    })

    return sfs_dictionary


def floating_sfs(
        training_table, validation_table, testing_table, feature_names,
        target_name, estimator_object, cost_function=_cross_entropy_function,
        num_features_to_add_per_step=1, min_fractional_cost_decrease=
        MIN_FRACTIONAL_COST_DECREASE_SFS_DEFAULT):
    """Runs the SFFS (sequential forward floating selection) algorithm.

    SFFS is defined in Chapter 9 of Webb (2003).

    :param training_table: See documentation for
        _check_sequential_selection_inputs.
    :param validation_table: See doc for _check_sequential_selection_inputs.
    :param testing_table: See doc for _check_sequential_selection_inputs.
    :param feature_names: See doc for _check_sequential_selection_inputs.
    :param target_name: See doc for _check_sequential_selection_inputs.
    :param estimator_object: See doc for sequential_forward_selection.
    :param cost_function: See doc for `_forward_selection_step`.
    :param num_features_to_add_per_step: Number of features to add at each step.
    :param min_fractional_cost_decrease: See doc for
        sequential_forward_selection.
    :return: sfs_dictionary: See doc for sequential_forward_selection.
    """

    _check_sequential_selection_inputs(
        training_table=training_table, validation_table=validation_table,
        testing_table=testing_table, feature_names=feature_names,
        target_name=target_name,
        num_features_to_add_per_step=num_features_to_add_per_step)

    error_checking.assert_is_greater(min_fractional_cost_decrease, 0.)
    error_checking.assert_is_less_than(min_fractional_cost_decrease, 1.)

    # Initialize values.
    selected_feature_names = []
    remaining_feature_names = copy.deepcopy(feature_names)

    num_features = len(feature_names)
    min_cost_by_num_selected = numpy.full(num_features + 1, numpy.nan)
    min_cost_by_num_selected[0] = numpy.inf
    major_step_num = 0

    while len(remaining_feature_names) >= num_features_to_add_per_step:
        major_step_num += 1
        num_selected_features = len(selected_feature_names)
        num_remaining_features = len(remaining_feature_names)

        print((
            'Major step {0:d} of SFFS: {1:d} features selected, {2:d} '
            'remaining...'
        ).format(
            major_step_num, num_selected_features, num_remaining_features
        ))

        min_new_cost, these_best_feature_names = _forward_selection_step(
            training_table=training_table, validation_table=validation_table,
            selected_feature_names=selected_feature_names,
            remaining_feature_names=remaining_feature_names,
            target_name=target_name, estimator_object=estimator_object,
            cost_function=cost_function,
            num_features_to_add=num_features_to_add_per_step)

        print((
            'Minimum cost ({0:.4f}) given by adding features shown below '
            '(previous minimum = {1:.4f}).\n{2:s}\n'
        ).format(
            min_new_cost, min_cost_by_num_selected[num_selected_features],
            str(these_best_feature_names)
        ))

        stopping_criterion = min_cost_by_num_selected[num_selected_features] * (
            1. - min_fractional_cost_decrease
        )

        if min_new_cost > stopping_criterion:
            break

        selected_feature_names += these_best_feature_names
        remaining_feature_names = [
            s for s in remaining_feature_names
            if s not in these_best_feature_names
        ]

        min_cost_by_num_selected[
            (num_selected_features + 1):(len(selected_feature_names) + 1)
        ] = min_new_cost

        if len(selected_feature_names) < 2:
            continue

        backward_step_num = 0

        while len(selected_feature_names) >= 2:
            backward_step_num += 1
            num_selected_features = len(selected_feature_names)

            print((
                'Major step {0:d}, backward step {1:d}: {2:d}/{3:d} features '
                'selected...'
            ).format(
                major_step_num, backward_step_num, num_selected_features,
                num_features
            ))

            min_new_cost, this_worst_feature_name_as_list = (
                _backward_selection_step(
                    training_table=training_table,
                    validation_table=validation_table,
                    selected_feature_names=selected_feature_names,
                    target_name=target_name, estimator_object=estimator_object,
                    cost_function=cost_function, num_features_to_remove=1)
            )

            this_worst_feature_name = this_worst_feature_name_as_list[0]

            # Cannot remove feature that was just added in the forward step.
            if backward_step_num == 1:
                this_worst_feature_index = selected_feature_names.index(
                    this_worst_feature_name
                )

                if this_worst_feature_index == num_selected_features - 1:
                    break

            # Remove feature only if this improves performance.
            if (min_new_cost >= min_cost_by_num_selected[
                    num_selected_features - 1]):
                break

            print((
                'Minimum cost ({0:.4f}) given by removing feature "{1:s}" '
                '(previous minimum = {2:.4f}).'
            ).format(
                min_new_cost, this_worst_feature_name,
                min_cost_by_num_selected[num_selected_features - 1]
            ))

            remaining_feature_names.append(this_worst_feature_name)
            selected_feature_names.remove(this_worst_feature_name)
            min_cost_by_num_selected[num_selected_features] = numpy.nan
            min_cost_by_num_selected[num_selected_features - 1] = min_new_cost

        print('\n')

    sfs_dictionary = _evaluate_feature_selection(
        training_table=training_table, validation_table=validation_table,
        testing_table=testing_table, estimator_object=estimator_object,
        selected_feature_names=selected_feature_names, target_name=target_name)

    num_selected_features = len(selected_feature_names)

    sfs_dictionary.update({
        VALIDATION_COST_BY_STEP_KEY:
            min_cost_by_num_selected[1:(num_selected_features + 1)]
    })

    return sfs_dictionary


def sequential_backward_selection(
        training_table, validation_table, testing_table, feature_names,
        target_name, estimator_object, cost_function=_cross_entropy_function,
        num_features_to_remove_per_step=1, min_fractional_cost_decrease=
        MIN_FRACTIONAL_COST_DECREASE_SBS_DEFAULT):
    """Runs the SBS (sequential backward selection) algorithm.

    SBS is defined in Chapter 9 of Webb (2003).

    f = number of features selected

    :param training_table: See documentation for
        _check_sequential_selection_inputs.
    :param validation_table: See doc for _check_sequential_selection_inputs.
    :param testing_table: See doc for _check_sequential_selection_inputs.
    :param feature_names: See doc for _check_sequential_selection_inputs.
    :param target_name: See doc for _check_sequential_selection_inputs.
    :param estimator_object: See doc for sequential_forward_selection.
    :param cost_function: See doc for `_forward_selection_step`.
    :param num_features_to_remove_per_step: Number of features to remove at each
        step.
    :param min_fractional_cost_decrease: Stopping criterion.  Once the
        fractional cost decrease over one step is <
        `min_fractional_cost_decrease`, SBS will stop.  Must be in range
        (-1, 1).  If negative, cost may increase slightly without SBS stopping.
    :return: sbs_dictionary: Same as output from _evaluate_feature_selection,
        but with two additional keys.
    sbs_dictionary['removed_feature_names']: length-f list with names of
        features removed (in order of their removal).
    sbs_dictionary['validation_cost_by_step']: length-f numpy array of
        validation costs.  The [i]th element is the cost with i features
        removed.  In other words, validation_cost_by_step[0] is the cost with 1
        feature removed; validation_cost_by_step[1] is the cost with 2 features
        removed; ...; etc.
    """

    _check_sequential_selection_inputs(
        training_table=training_table, validation_table=validation_table,
        testing_table=testing_table, feature_names=feature_names,
        target_name=target_name,
        num_features_to_remove_per_step=num_features_to_remove_per_step)

    error_checking.assert_is_greater(min_fractional_cost_decrease, -1.)
    error_checking.assert_is_less_than(min_fractional_cost_decrease, 1.)

    # Initialize values.
    removed_feature_names = []
    selected_feature_names = copy.deepcopy(feature_names)

    num_features = len(feature_names)
    min_cost_by_num_removed = numpy.full(num_features + 1, numpy.nan)
    min_cost_by_num_removed[0] = numpy.inf

    while len(selected_feature_names) >= num_features_to_remove_per_step:
        num_removed_features = len(removed_feature_names)
        num_selected_features = len(selected_feature_names)

        print((
            'Step {0:d} of sequential backward selection: {1:d} features '
            'removed, {2:d} remaining...'
        ).format(
            num_removed_features + 1, num_removed_features,
            num_selected_features
        ))

        min_new_cost, these_worst_feature_names = _backward_selection_step(
            training_table=training_table, validation_table=validation_table,
            selected_feature_names=selected_feature_names,
            target_name=target_name, estimator_object=estimator_object,
            cost_function=cost_function,
            num_features_to_remove=num_features_to_remove_per_step)

        print((
            'Minimum cost ({0:.4f}) given by removing features shown below '
            '(previous minimum = {1:.4f}).\n{2:s}\n'
        ).format(
            min_new_cost, min_cost_by_num_removed[num_removed_features],
            str(these_worst_feature_names)
        ))

        stopping_criterion = min_cost_by_num_removed[num_removed_features] * (
            1. - min_fractional_cost_decrease
        )

        if min_new_cost > stopping_criterion:
            break

        removed_feature_names += these_worst_feature_names
        selected_feature_names = [
            s for s in selected_feature_names
            if s not in these_worst_feature_names
        ]

        min_cost_by_num_removed[
            (num_removed_features + 1):(len(removed_feature_names) + 1)
        ] = min_new_cost

    sbs_dictionary = _evaluate_feature_selection(
        training_table=training_table, validation_table=validation_table,
        testing_table=testing_table, estimator_object=estimator_object,
        selected_feature_names=selected_feature_names, target_name=target_name)

    num_removed_features = len(removed_feature_names)

    sbs_dictionary.update({
        REMOVED_FEATURES_KEY: removed_feature_names,
        VALIDATION_COST_BY_STEP_KEY:
            min_cost_by_num_removed[1:(num_removed_features + 1)]
    })

    return sbs_dictionary


def sbs_with_forward_steps(
        training_table, validation_table, testing_table, feature_names,
        target_name, estimator_object, cost_function=_cross_entropy_function,
        num_forward_steps=DEFAULT_NUM_FORWARD_STEPS_FOR_SBS,
        num_backward_steps=DEFAULT_NUM_BACKWARD_STEPS_FOR_SBS,
        num_features_to_add_per_forward_step=1,
        num_features_to_remove_per_backward_step=1,
        min_fractional_cost_decrease=
        MIN_FRACTIONAL_COST_DECREASE_SBS_DEFAULT):
    """Runs SBS (sequential backward selection) with forward steps.

    This method is called "plus-l-minus-r selection" in Chapter 9 of Webb
    (2003), where l < r.

    --- DEFINITIONS ---

    "Forward step" = addition of `num_features_to_add_per_step` features.
    "Backward step" = removal of `num_features_to_remove_per_step` features.
    "Major step" = r backward steps followed by l forward steps.

    :param training_table: See documentation for
        _check_sequential_selection_inputs.
    :param validation_table: See doc for _check_sequential_selection_inputs.
    :param testing_table: See doc for _check_sequential_selection_inputs.
    :param feature_names: See doc for _check_sequential_selection_inputs.
    :param target_name: See doc for _check_sequential_selection_inputs.
    :param estimator_object: Instance of scikit-learn estimator.  Must implement
        the methods `fit` and `predict_proba`.
    :param cost_function: See doc for `_forward_selection_step`.
    :param num_forward_steps: l-value (number of forward steps per major step).
    :param num_backward_steps: r-value (number of backward steps per major
        step).
    :param num_features_to_add_per_forward_step: Number of features to add at
        each forward step.
    :param num_features_to_remove_per_backward_step: Number of features to
        remove at each backward step.
    :param min_fractional_cost_decrease: Stopping criterion.  Once the
        fractional cost decrease over a major step is <
        `min_fractional_cost_decrease`, SBS will stop.  Must be in range
        (-1, 1).
    :return: sbs_dictionary: See documentation for
        sequential_backward_selection.
    """

    _check_sequential_selection_inputs(
        training_table=training_table, validation_table=validation_table,
        testing_table=testing_table, feature_names=feature_names,
        target_name=target_name,
        num_features_to_add_per_step=num_features_to_add_per_forward_step,
        num_features_to_remove_per_step=
        num_features_to_remove_per_backward_step)

    error_checking.assert_is_integer(num_forward_steps)
    error_checking.assert_is_geq(num_forward_steps, 1)
    error_checking.assert_is_integer(num_backward_steps)
    error_checking.assert_is_geq(num_backward_steps, 1)

    num_features_to_add_per_major_step = (
        num_forward_steps * num_features_to_add_per_forward_step
    )
    num_features_to_remove_per_major_step = (
        num_backward_steps * num_features_to_remove_per_backward_step
    )

    num_features = len(feature_names)
    error_checking.assert_is_less_than(
        num_features_to_remove_per_major_step, num_features)
    error_checking.assert_is_less_than(
        num_features_to_add_per_major_step,
        num_features_to_remove_per_major_step)

    error_checking.assert_is_greater(min_fractional_cost_decrease, -1.)
    error_checking.assert_is_less_than(min_fractional_cost_decrease, 1.)

    # Initialize values.
    removed_feature_names = []
    selected_feature_names = copy.deepcopy(feature_names)

    min_cost_by_num_removed = numpy.full(num_features + 1, numpy.nan)
    min_cost_by_num_removed[0] = numpy.inf
    major_step_num = 0

    while (len(selected_feature_names) - num_features_to_remove_per_major_step
           > 0):
        major_step_num += 1

        min_cost_last_major_step = min_cost_by_num_removed[
            len(removed_feature_names)
        ]
        removed_feature_names_last_major_step = copy.deepcopy(
            removed_feature_names)
        selected_feature_names_last_major_step = copy.deepcopy(
            selected_feature_names)

        for i in range(num_backward_steps):
            num_removed_features = len(removed_feature_names)
            num_selected_features = len(selected_feature_names)

            print((
                'Major step {0:d}, backward step {1:d}: {2:d} features removed,'
                ' {3:d} remaining...'
            ).format(
                major_step_num, i + 1, num_removed_features,
                num_selected_features
            ))

            min_new_cost, these_worst_feature_names = _backward_selection_step(
                training_table=training_table,
                validation_table=validation_table,
                selected_feature_names=selected_feature_names,
                target_name=target_name, estimator_object=estimator_object,
                cost_function=cost_function,
                num_features_to_remove=num_features_to_remove_per_backward_step)

            print((
                'Minimum cost ({0:.4f}) given by removing features shown below '
                '(previous minimum = {1:.4f}).\n{2:s}\n'
            ).format(
                min_new_cost, min_cost_by_num_removed[num_removed_features],
                str(these_worst_feature_names)
            ))

            removed_feature_names += these_worst_feature_names
            selected_feature_names = [
                s for s in selected_feature_names
                if s not in these_worst_feature_names
            ]

            min_cost_by_num_removed[
                (num_removed_features + 1):(len(removed_feature_names) + 1)
            ] = min_new_cost

        for i in range(num_forward_steps):
            num_removed_features = len(removed_feature_names)

            print((
                'Major step {0:d}, forward step {1:d}: {2:d}/{3:d} features '
                'removed...'
            ).format(
                major_step_num, i + 1, num_removed_features, num_features
            ))

            min_new_cost, these_best_feature_names = _forward_selection_step(
                training_table=training_table,
                validation_table=validation_table,
                selected_feature_names=selected_feature_names,
                remaining_feature_names=removed_feature_names,
                target_name=target_name, estimator_object=estimator_object,
                cost_function=cost_function,
                num_features_to_add=num_features_to_add_per_forward_step)

            print((
                'Minimum cost ({0:.4f}) given by adding features shown below '
                '(previous minimum = {1:.4f}).\n{2:s}\n'
            ).format(
                min_new_cost, min_cost_by_num_removed[num_removed_features],
                str(these_best_feature_names)
            ))

            selected_feature_names += these_best_feature_names
            removed_feature_names = [
                s for s in removed_feature_names
                if s not in these_best_feature_names
            ]

            min_cost_by_num_removed[
                (len(removed_feature_names) + 1):
            ] = numpy.nan
            min_cost_by_num_removed[
                len(removed_feature_names)
            ] = min_new_cost

        print('\n')
        stopping_criterion = min_cost_last_major_step * (
            1. - min_fractional_cost_decrease
        )

        if (min_cost_by_num_removed[len(removed_feature_names)] >
                stopping_criterion):
            removed_feature_names = copy.deepcopy(
                removed_feature_names_last_major_step)
            selected_feature_names = copy.deepcopy(
                selected_feature_names_last_major_step)
            break

    sbs_dictionary = _evaluate_feature_selection(
        training_table=training_table, validation_table=validation_table,
        testing_table=testing_table, estimator_object=estimator_object,
        selected_feature_names=selected_feature_names, target_name=target_name)

    num_removed_features = len(removed_feature_names)

    sbs_dictionary.update({
        REMOVED_FEATURES_KEY: removed_feature_names,
        VALIDATION_COST_BY_STEP_KEY:
            min_cost_by_num_removed[1:(num_removed_features + 1)]
    })

    return sbs_dictionary


def floating_sbs(
        training_table, validation_table, testing_table, feature_names,
        target_name, estimator_object, cost_function=_cross_entropy_function,
        num_features_to_remove_per_step=1, min_fractional_cost_decrease=
        MIN_FRACTIONAL_COST_DECREASE_SBS_DEFAULT):
    """Runs the SBFS (sequential backward floating selection) algorithm.

    SBFS is defined in Chapter 9 of Webb (2003).

    :param training_table: See documentation for
        _check_sequential_selection_inputs.
    :param validation_table: See doc for _check_sequential_selection_inputs.
    :param testing_table: See doc for _check_sequential_selection_inputs.
    :param feature_names: See doc for _check_sequential_selection_inputs.
    :param target_name: See doc for _check_sequential_selection_inputs.
    :param estimator_object: See doc for sequential_forward_selection.
    :param cost_function: See doc for `_forward_selection_step`.
    :param num_features_to_remove_per_step: Number of features to remove at each
        step.
    :param min_fractional_cost_decrease: See doc for
        sequential_backward_selection.
    :return: sbs_dictionary: See doc for sequential_backward_selection.
    """

    _check_sequential_selection_inputs(
        training_table=training_table, validation_table=validation_table,
        testing_table=testing_table, feature_names=feature_names,
        target_name=target_name,
        num_features_to_remove_per_step=num_features_to_remove_per_step)

    error_checking.assert_is_greater(min_fractional_cost_decrease, -1.)
    error_checking.assert_is_less_than(min_fractional_cost_decrease, 1.)

    # Initialize values.
    removed_feature_names = []
    selected_feature_names = copy.deepcopy(feature_names)

    num_features = len(feature_names)
    min_cost_by_num_removed = numpy.full(num_features + 1, numpy.nan)
    min_cost_by_num_removed[0] = numpy.inf
    major_step_num = 0

    while len(selected_feature_names) > num_features_to_remove_per_step:
        major_step_num += 1
        num_removed_features = len(removed_feature_names)
        num_selected_features = len(selected_feature_names)

        print((
            'Major step {0:d} of SBFS: {1:d} features removed, {2:d} '
            'remaining...'
        ).format(
            major_step_num, num_removed_features, num_selected_features
        ))

        min_new_cost, these_worst_feature_names = _backward_selection_step(
            training_table=training_table, validation_table=validation_table,
            selected_feature_names=selected_feature_names,
            target_name=target_name, estimator_object=estimator_object,
            cost_function=cost_function,
            num_features_to_remove=num_features_to_remove_per_step)

        print((
            'Minimum cost ({0:.4f}) given by removing features shown below '
            '(previous minimum = {1:.4f}).\n{2:s}\n'
        ).format(
            min_new_cost, min_cost_by_num_removed[num_removed_features],
            str(these_worst_feature_names)
        ))

        stopping_criterion = min_cost_by_num_removed[num_removed_features] * (
            1. - min_fractional_cost_decrease
        )

        if min_new_cost > stopping_criterion:
            break

        removed_feature_names += these_worst_feature_names
        selected_feature_names = [
            s for s in selected_feature_names
            if s not in these_worst_feature_names
        ]

        min_cost_by_num_removed[
            (num_removed_features + 1):(len(removed_feature_names) + 1)
        ] = min_new_cost

        if len(removed_feature_names) < 2:
            continue

        forward_step_num = 0
        while len(removed_feature_names) >= 2:
            forward_step_num += 1
            num_removed_features = len(removed_feature_names)

            print((
                'Major step {0:d}, forward step {1:d}: {2:d}/{3:d} features '
                'removed...'
            ).format(
                major_step_num, forward_step_num, num_removed_features,
                num_features
            ))

            min_new_cost, this_best_feature_name_as_list = (
                _forward_selection_step(
                    training_table=training_table,
                    validation_table=validation_table,
                    selected_feature_names=selected_feature_names,
                    remaining_feature_names=removed_feature_names,
                    target_name=target_name, estimator_object=estimator_object,
                    cost_function=cost_function, num_features_to_add=1)
            )

            this_best_feature_name = this_best_feature_name_as_list[0]

            # Cannot add feature that was just removed in the backward step.
            if forward_step_num == 1:
                this_best_feature_index = removed_feature_names.index(
                    this_best_feature_name
                )

                if this_best_feature_index == num_removed_features - 1:
                    break

            # Add feature only if it improves performance.
            if (min_new_cost >= min_cost_by_num_removed[
                    num_removed_features - 1]):
                break

            print((
                'Minimum cost ({0:.4f}) given by adding feature "{1:s}" '
                '(previous minimum = {2:.4f}).'
            ).format(
                min_new_cost, this_best_feature_name,
                min_cost_by_num_removed[num_removed_features - 1]
            ))

            selected_feature_names.append(this_best_feature_name)
            removed_feature_names.remove(this_best_feature_name)
            min_cost_by_num_removed[num_removed_features] = numpy.nan
            min_cost_by_num_removed[num_removed_features - 1] = min_new_cost

        print('\n')

    sbs_dictionary = _evaluate_feature_selection(
        training_table=training_table, validation_table=validation_table,
        testing_table=testing_table, estimator_object=estimator_object,
        selected_feature_names=selected_feature_names, target_name=target_name)

    num_removed_features = len(removed_feature_names)

    sbs_dictionary.update({
        REMOVED_FEATURES_KEY: removed_feature_names,
        VALIDATION_COST_BY_STEP_KEY:
            min_cost_by_num_removed[1:(num_removed_features + 1)]
    })

    return sbs_dictionary


def permutation_selection(
        training_table, validation_table, feature_names, target_name,
        estimator_object, cost_function=_cross_entropy_function):
    """Runs the permutation algorithm (Lakshmanan et al. 2015).

    :param training_table: See documentation for
        _check_sequential_selection_inputs.
    :param validation_table: See doc for _check_sequential_selection_inputs.
    :param feature_names: See doc for _check_sequential_selection_inputs.
    :param target_name: See doc for _check_sequential_selection_inputs.
    :param estimator_object: See doc for sequential_forward_selection.
    :param cost_function: See doc for `_forward_selection_step`.
    :return: permutation_table: pandas DataFrame with the following columns.
        Each row corresponds to one feature.  Order of rows = order in which
        features were permuted.  In other words, the feature in the [i]th row
        was the [i]th to be permuted.
    permutation_table.feature_name: Feature name.
    permutation_table.validation_cost: Validation cost after permuting feature
        (and keeping features in previous rows permuted).
    permutation_table.validation_cross_entropy: Same but for cross-entropy.
    permutation_table.validation_auc: Same but for area under ROC curve.

    orig_validation_cost: Validation cost with no permuted features.
    orig_validation_cross_entropy: Same but for cross-entropy.
    orig_validation_auc: Same but for area under ROC curve.
    """

    _check_sequential_selection_inputs(
        training_table=training_table, validation_table=validation_table,
        feature_names=feature_names, target_name=target_name)

    # Find cost before permutation.
    new_estimator_object = sklearn.base.clone(estimator_object)
    new_estimator_object.fit(
        training_table.as_matrix(columns=feature_names),
        training_table[target_name].values
    )

    forecast_probs_for_validation = new_estimator_object.predict_proba(
        validation_table.as_matrix(columns=feature_names)
    )[:, 1]

    orig_validation_cost = cost_function(
        forecast_probs_for_validation, validation_table[target_name].values
    )
    orig_validation_cross_entropy = model_eval.get_cross_entropy(
        forecast_probs_for_validation,
        validation_table[target_name].values
    )
    orig_validation_auc = sklearn.metrics.roc_auc_score(
        validation_table[target_name].values, forecast_probs_for_validation
    )

    # Initialize values.
    remaining_feature_names = copy.deepcopy(feature_names)
    permutation_dict = {
        FEATURE_NAME_KEY: [], VALIDATION_COST_KEY: [],
        VALIDATION_XENTROPY_KEY: [], VALIDATION_AUC_KEY: []
    }

    while remaining_feature_names:  # While there are still features to permute.
        num_permuted_features = len(permutation_dict[FEATURE_NAME_KEY])
        num_remaining_features = len(remaining_feature_names)

        print((
            'Step {0:d} of permutation selection: {1:d} features permuted, '
            '{2:d} remaining...'
        ).format(
            num_permuted_features + 1, num_permuted_features,
            num_remaining_features
        ))

        new_cost_by_feature = numpy.full(num_remaining_features, numpy.nan)
        new_xentropy_by_feature = numpy.full(num_remaining_features, numpy.nan)
        new_auc_by_feature = numpy.full(num_remaining_features, numpy.nan)
        permuted_values_for_best_feature = None

        for j in range(num_remaining_features):
            this_validation_matrix = validation_table.as_matrix(
                columns=feature_names)

            column_to_permute = feature_names.index(remaining_feature_names[j])
            this_validation_matrix[:, column_to_permute] = (
                numpy.random.permutation(
                    this_validation_matrix[:, column_to_permute])
            )

            these_forecast_probabilities = new_estimator_object.predict_proba(
                this_validation_matrix
            )[:, 1]

            new_cost_by_feature[j] = cost_function(
                these_forecast_probabilities,
                validation_table[target_name].values
            )
            new_xentropy_by_feature[j] = model_eval.get_cross_entropy(
                these_forecast_probabilities,
                validation_table[target_name].values
            )
            new_auc_by_feature[j] = sklearn.metrics.roc_auc_score(
                validation_table[target_name].values,
                these_forecast_probabilities
            )

            if numpy.nanargmax(new_cost_by_feature) == j:
                permuted_values_for_best_feature = (
                    this_validation_matrix[:, column_to_permute]
                )

        max_new_cost = numpy.max(new_cost_by_feature)
        this_best_feature_index = numpy.argmax(new_cost_by_feature)
        this_best_feature_name = remaining_feature_names[
            this_best_feature_index
        ]

        if len(permutation_dict[VALIDATION_COST_KEY]):
            max_previous_cost = permutation_dict[VALIDATION_COST_KEY][-1]
        else:
            max_previous_cost = copy.deepcopy(orig_validation_cost)

        print((
            'Maximum cost ({0:.4f}) given by permuting feature "{1:s}" '
            '(previous max = {2:.4f}).'
        ).format(
            max_new_cost, this_best_feature_name, max_previous_cost
        ))

        remaining_feature_names.remove(this_best_feature_name)
        validation_table = validation_table.assign(**{
            this_best_feature_name: permuted_values_for_best_feature
        })

        permutation_dict[FEATURE_NAME_KEY].append(this_best_feature_name)
        permutation_dict[VALIDATION_COST_KEY].append(
            new_cost_by_feature[this_best_feature_index]
        )
        permutation_dict[VALIDATION_XENTROPY_KEY].append(
            new_xentropy_by_feature[this_best_feature_index]
        )
        permutation_dict[VALIDATION_AUC_KEY].append(
            new_auc_by_feature[this_best_feature_index]
        )

    permutation_table = pandas.DataFrame.from_dict(permutation_dict)
    return (permutation_table, orig_validation_cost,
            orig_validation_cross_entropy, orig_validation_auc)


def plot_forward_selection_results(
        forward_selection_dict, plot_feature_names=False,
        bar_face_colour=DEFAULT_BAR_FACE_COLOUR,
        bar_edge_colour=DEFAULT_BAR_EDGE_COLOUR,
        bar_edge_width=DEFAULT_BAR_EDGE_WIDTH):
    """Plots bar graph with results of any forward-selection algorithm.

    :param forward_selection_dict: Dictionary returned by
        `sequential_forward_selection`, `sfs_with_backward_steps`, or
        `floating_sfs`.
    :param plot_feature_names: See documentation for _plot_selection_results.
    :param bar_face_colour: See doc for _plot_selection_results.
    :param bar_edge_colour: See doc for _plot_selection_results.
    :param bar_edge_width: See doc for _plot_selection_results.
    """

    _plot_selection_results(
        used_feature_names=forward_selection_dict[SELECTED_FEATURES_KEY],
        validation_cost_by_feature=
        forward_selection_dict[VALIDATION_COST_BY_STEP_KEY],
        selection_type=FORWARD_SELECTION_TYPE,
        plot_feature_names=plot_feature_names, bar_face_colour=bar_face_colour,
        bar_edge_colour=bar_edge_colour, bar_edge_width=bar_edge_width)


def plot_backward_selection_results(
        backward_selection_dict, plot_feature_names=False,
        bar_face_colour=DEFAULT_BAR_FACE_COLOUR,
        bar_edge_colour=DEFAULT_BAR_EDGE_COLOUR,
        bar_edge_width=DEFAULT_BAR_EDGE_WIDTH):
    """Plots bar graph with results of any backward-selection algorithm.

    :param backward_selection_dict: Dictionary returned by
        `sequential_backward_selection`, `sbs_with_forward_steps`, or
        `floating_sbs`.
    :param plot_feature_names: See documentation for _plot_selection_results.
    :param bar_face_colour: See doc for _plot_selection_results.
    :param bar_edge_colour: See doc for _plot_selection_results.
    :param bar_edge_width: See doc for _plot_selection_results.
    """

    _plot_selection_results(
        used_feature_names=backward_selection_dict[REMOVED_FEATURES_KEY],
        validation_cost_by_feature=
        backward_selection_dict[VALIDATION_COST_BY_STEP_KEY],
        selection_type=BACKWARD_SELECTION_TYPE,
        plot_feature_names=plot_feature_names, bar_face_colour=bar_face_colour,
        bar_edge_colour=bar_edge_colour, bar_edge_width=bar_edge_width)


def plot_permutation_results(
        permutation_table, orig_validation_cost, plot_feature_names=False,
        bar_face_colour=DEFAULT_BAR_FACE_COLOUR,
        bar_edge_colour=DEFAULT_BAR_EDGE_COLOUR,
        bar_edge_width=DEFAULT_BAR_EDGE_WIDTH):
    """Plots bar graph with results of permutation selection.

    :param permutation_table: pandas DataFrame returned by
        `permutation_selection`.
    :param orig_validation_cost: Validation cost before permutation.
    :param plot_feature_names: See documentation for _plot_selection_results.
    :param bar_face_colour: See doc for _plot_selection_results.
    :param bar_edge_colour: See doc for _plot_selection_results.
    :param bar_edge_width: See doc for _plot_selection_results.
    """

    _plot_selection_results(
        used_feature_names=permutation_table[FEATURE_NAME_KEY].values,
        validation_cost_by_feature=
        permutation_table[VALIDATION_XENTROPY_KEY].values,
        selection_type=PERMUTATION_TYPE,
        validation_cost_before_permutn=orig_validation_cost,
        plot_feature_names=plot_feature_names, bar_face_colour=bar_face_colour,
        bar_edge_colour=bar_edge_colour, bar_edge_width=bar_edge_width)
