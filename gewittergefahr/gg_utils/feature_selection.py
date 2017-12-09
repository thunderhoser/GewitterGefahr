"""Implements various feature-selection algorithms.

NOTE: Currently this module handles binary classification only.  At some point I
may add methods that deal with multi-class classification or regression.

--- REFERENCES ---

Lakshmanan, V., C. Karstens, J. Krause, K. Elmore, A. Ryzhkov, and S. Berkseth,
    2015: "Which polarimetric variables are important for weather/no-weather
    discrimination?" Journal of Atmospheric and Oceanic Technology, 32 (6),
    1209-1223.

Webb, A.R., 2003: "Statistical Pattern Recognition". John Wiley & Sons.
"""

import copy
import numpy
import pandas
import sklearn.base
import sklearn.metrics
import matplotlib.pyplot as pyplot
from gewittergefahr.gg_utils import model_evaluation as model_eval
from gewittergefahr.gg_utils import error_checking

# TODO(thunderhoser): This module has a lot of duplicated code.  Need to clean
# up.

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

DEFAULT_MIN_FRACTIONAL_XENTROPY_DECR_FOR_SFS = 0.01
DEFAULT_MIN_FRACTIONAL_XENTROPY_DECR_FOR_SBS = -0.01

DEFAULT_NUM_FORWARD_STEPS_FOR_SFS = 2
DEFAULT_NUM_BACKWARD_STEPS_FOR_SFS = 1
DEFAULT_NUM_FORWARD_STEPS_FOR_SBS = 1
DEFAULT_NUM_BACKWARD_STEPS_FOR_SBS = 2

SELECTED_FEATURES_KEY = 'selected_feature_names'
REMOVED_FEATURES_KEY = 'removed_feature_names'
FEATURE_NAME_KEY = 'feature_name'
VALIDATION_XENTROPY_KEY = 'validation_cross_entropy'
VALIDATION_AUC_KEY = 'validation_auc'
TESTING_XENTROPY_KEY = 'testing_cross_entropy'
TESTING_AUC_KEY = 'testing_auc'
VALIDATION_XENTROPY_BY_STEP_KEY = 'validation_cross_entropy_by_step'

PERMUTATION_TYPE = 'permutation'
FORWARD_SELECTION_TYPE = 'forward'
BACKWARD_SELECTION_TYPE = 'backward'


def _check_sequential_selection_inputs(
        training_table=None, validation_table=None, testing_table=None,
        feature_names=None, target_name=None):
    """Checks inputs for sequential forward or backward selection.

    :param training_table: pandas DataFrame, where each row is one training
        example.
    :param validation_table: pandas DataFrame, where each row is one validation
        example.
    :param testing_table: pandas DataFrame, where each row is one testing
        example.
    :param feature_names: length-F list with names of features (predictor
        variables).  Each feature must be a column in training_table,
        validation_table, and testing_table.
    :param target_name: Name of target variable (predictand).  Must be a column
        in training_table, validation_table, and testing_table.
    """

    error_checking.assert_is_string(target_name)
    error_checking.assert_is_string_list(feature_names)
    error_checking.assert_is_numpy_array(
        numpy.asarray(feature_names), num_dimensions=1)

    variable_names = feature_names + [target_name]
    error_checking.assert_columns_in_dataframe(training_table, variable_names)
    error_checking.assert_columns_in_dataframe(validation_table, variable_names)
    if testing_table is not None:
        error_checking.assert_columns_in_dataframe(
            testing_table, variable_names)

    # Ensure that label is binary.
    error_checking.assert_is_integer_numpy_array(
        training_table[target_name].values)
    error_checking.assert_is_geq_numpy_array(
        training_table[target_name].values, 0)
    error_checking.assert_is_leq_numpy_array(
        training_table[target_name].values, 1)


def _evaluate_feature_selection(
        training_table=None, validation_table=None, testing_table=None,
        estimator_object=None, selected_feature_names=None, target_name=None):
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
        feature_name_by_step=None, cross_entropy_by_step=None,
        selection_type=None, cross_entropy_before_permutn=None,
        plot_feature_names=False, bar_face_colour=DEFAULT_BAR_FACE_COLOUR,
        bar_edge_colour=DEFAULT_BAR_EDGE_COLOUR,
        bar_edge_width=DEFAULT_BAR_EDGE_WIDTH):
    """Plots bar graph with results of feature-selection algorithm.

    K = number of steps executed in feature-selection algorithm

    :param feature_name_by_step: length-K list of feature names.  If algorithm
        was permutation, these are the features permuted, in order of their
        permutation.  If algorithm was SFS, SFS with backward steps, or SFFS,
        these are features added in order of their addition.  If algorithm was
        SBS, SBS with forward steps, or SBFS, these are features removed in
        order of their removal.
    :param cross_entropy_by_step: length-K list of cross-entropies.
        cross_entropy_by_step[k] is cross-entropy after the [k]th step.
    :param selection_type: Type of selection algorithm used.  May be
        "permutation", "forward", or "backward".
    :param cross_entropy_before_permutn: Cross-entropy before permutation.  If
        selection_type != "permutation", you can leave this as None.
    :param plot_feature_names: Boolean flag.  If True, will plot feature names
        on x-axis.  If False, will plot ordinal numbers on x-axis.
    :param bar_face_colour: Colour (in any format accepted by
        `matplotlib.colors`) for interior of bars.
    :param bar_edge_colour: Colour for edge of bars.
    :param bar_edge_width: Width for edge of bars.
    """

    num_steps = len(feature_name_by_step)

    if selection_type == PERMUTATION_TYPE:
        x_coords_at_bar_edges = numpy.linspace(
            -0.5, num_steps + 0.5, num=num_steps + 2)
        y_values = numpy.concatenate((
            numpy.array([cross_entropy_before_permutn]), cross_entropy_by_step))
    else:
        x_coords_at_bar_edges = numpy.linspace(
            0.5, num_steps + 0.5, num=num_steps + 1)
        y_values = copy.deepcopy(cross_entropy_by_step)

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
                x_coords_at_bar_centers, [' '] + feature_name_by_step.tolist(),
                rotation='vertical', fontsize=FEATURE_NAME_FONT_SIZE)
        else:
            pyplot.xticks(x_coords_at_bar_centers, feature_name_by_step,
                          rotation='vertical', fontsize=FEATURE_NAME_FONT_SIZE)

    axes_object.set_ylabel('Validation cross-entropy')


def sequential_forward_selection(
        training_table=None, validation_table=None, testing_table=None,
        feature_names=None, target_name=None, estimator_object=None,
        min_fractional_xentropy_decrease=
        DEFAULT_MIN_FRACTIONAL_XENTROPY_DECR_FOR_SFS):
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
    :param min_fractional_xentropy_decrease: Stopping criterion.  Once the
        fractional decrease in cross-entropy from adding a feature is <
        `min_fractional_xentropy_decrease`, SFS will stop.  Must be in range
        (0, 1).
    :return: sfs_dictionary: Same as output from _evaluate_feature_selection,
        except with one additional key.
    sfs_dictionary['validation_xentropy_by_step']: length-f numpy array of
        validation cross-entropies.  The [i]th element is the cross-entropy
        after the [i]th feature addition.
    """

    _check_sequential_selection_inputs(
        training_table=training_table, validation_table=validation_table,
        testing_table=testing_table, feature_names=feature_names,
        target_name=target_name)

    error_checking.assert_is_greater(min_fractional_xentropy_decrease, 0.)
    error_checking.assert_is_less_than(min_fractional_xentropy_decrease, 1.)

    # Initialize feature sets; initialize min cross-entropy to ridiculously high
    # number.
    selected_feature_names = []
    remaining_feature_names = copy.deepcopy(feature_names)

    num_features = len(feature_names)
    min_cross_entropy_by_num_selected = numpy.full(num_features, numpy.nan)
    min_cross_entropy_by_num_selected[0] = 1e10

    while remaining_feature_names:  # While there are still features to select.
        num_selected_features = len(selected_feature_names)
        num_remaining_features = len(remaining_feature_names)
        new_xentropy_by_feature = numpy.full(num_remaining_features, numpy.nan)

        print ('Step {0:d} of sequential forward selection: {1:d} features '
               'selected, {2:d} remaining...').format(
                   num_selected_features + 1, num_selected_features,
                   num_remaining_features)

        for j in range(num_remaining_features):
            these_feature_names = (
                selected_feature_names + [remaining_feature_names[j]])

            new_estimator_object = sklearn.base.clone(estimator_object)
            new_estimator_object.fit(
                training_table.as_matrix(columns=these_feature_names),
                training_table[target_name].values)

            these_forecast_probabilities = new_estimator_object.predict_proba(
                validation_table.as_matrix(columns=these_feature_names))[:, 1]
            new_xentropy_by_feature[j] = model_eval.get_cross_entropy(
                these_forecast_probabilities,
                validation_table[target_name].values)

        min_new_cross_entropy = numpy.min(new_xentropy_by_feature)
        this_best_feature_index = numpy.argmin(new_xentropy_by_feature)
        this_best_feature_name = remaining_feature_names[
            this_best_feature_index]

        print (
            'Minimum cross-entropy ({0:.4f}) given by adding feature '
            '"{1:s}"; previous min cross-entropy = {2:.4f}').format(
                min_new_cross_entropy, this_best_feature_name,
                min_cross_entropy_by_num_selected[num_selected_features])

        stop_if_cross_entropy_above = (
            min_cross_entropy_by_num_selected[num_selected_features] * (
                1. - min_fractional_xentropy_decrease))
        if min_new_cross_entropy > stop_if_cross_entropy_above:
            break

        min_cross_entropy_by_num_selected[
            num_selected_features + 1] = min_new_cross_entropy

        selected_feature_names.append(this_best_feature_name)
        remaining_feature_names = set(remaining_feature_names)
        remaining_feature_names.remove(this_best_feature_name)
        remaining_feature_names = list(remaining_feature_names)

    sfs_dictionary = _evaluate_feature_selection(
        training_table=training_table, validation_table=validation_table,
        testing_table=testing_table, estimator_object=estimator_object,
        selected_feature_names=selected_feature_names, target_name=target_name)

    num_selected_features = len(selected_feature_names)
    sfs_dictionary.update(
        {VALIDATION_XENTROPY_BY_STEP_KEY:
             min_cross_entropy_by_num_selected[1:(num_selected_features + 1)]})
    return sfs_dictionary


def sfs_with_backward_steps(
        training_table=None, validation_table=None, testing_table=None,
        feature_names=None, target_name=None, estimator_object=None,
        num_forward_steps=DEFAULT_NUM_FORWARD_STEPS_FOR_SFS,
        num_backward_steps=DEFAULT_NUM_BACKWARD_STEPS_FOR_SFS,
        min_fractional_xentropy_decrease=
        DEFAULT_MIN_FRACTIONAL_XENTROPY_DECR_FOR_SFS):
    """Runs SFS (sequential forward selection) with backward steps.

    This method is called "plus l - take away r selection" in Chapter 9 of Webb
    (2003), where l > r.

    --- DEFINITIONS ---

    "Forward step" = addition of one feature.
    "Backward step" = removal of one feature.
    "Major step" = l forward steps followed by r backward steps.

    :param training_table: See documentation for
        _check_sequential_selection_inputs.
    :param validation_table: See doc for _check_sequential_selection_inputs.
    :param testing_table: See doc for _check_sequential_selection_inputs.
    :param feature_names: See doc for _check_sequential_selection_inputs.
    :param target_name: See doc for _check_sequential_selection_inputs.
    :param estimator_object: Instance of scikit-learn estimator.  Must implement
        the methods `fit` and `predict_proba`.
    :param num_forward_steps: l-value (number of forward steps per major step).
    :param num_backward_steps: r-value (number of backward steps per major
        step).
    :param min_fractional_xentropy_decrease: Stopping criterion.  Once the
        fractional decrease in cross-entropy from a major step is <
        `min_fractional_xentropy_decrease`, algorithm will stop.  Must be in
        range (0, 1).
    :return: sfs_dictionary: See doc for sequential_forward_selection.
    """

    _check_sequential_selection_inputs(
        training_table=training_table, validation_table=validation_table,
        testing_table=testing_table, feature_names=feature_names,
        target_name=target_name)

    num_features = len(feature_names)

    error_checking.assert_is_integer(num_forward_steps)
    error_checking.assert_is_geq(num_forward_steps, 2)
    error_checking.assert_is_leq(num_forward_steps, num_features)

    error_checking.assert_is_integer(num_backward_steps)
    error_checking.assert_is_geq(num_backward_steps, 1)
    error_checking.assert_is_less_than(num_backward_steps, num_forward_steps)

    error_checking.assert_is_greater(min_fractional_xentropy_decrease, 0.)
    error_checking.assert_is_less_than(min_fractional_xentropy_decrease, 1.)

    # Initialize feature sets; initialize min cross-entropy to ridiculously high
    # number.
    major_step_num = 0
    selected_feature_names = []
    remaining_feature_names = copy.deepcopy(feature_names)

    min_cross_entropy_by_num_selected = numpy.full(num_features, numpy.nan)
    min_cross_entropy_by_num_selected[0] = 1e10

    while len(selected_feature_names) + num_forward_steps <= num_features:
        major_step_num += 1
        min_cross_entropy_prev_major_step = min_cross_entropy_by_num_selected[
            len(selected_feature_names)]
        selected_feature_names_prev_major_step = copy.deepcopy(
            selected_feature_names)

        for i in range(num_forward_steps):
            num_selected_features = len(selected_feature_names)
            num_remaining_features = len(remaining_feature_names)
            new_xentropy_by_feature = numpy.full(
                num_remaining_features, numpy.nan)

            print ('Major step {0:d}, forward step {1:d}: {2:d} features '
                   'selected, {3:d} remaining...').format(
                       major_step_num, i + 1, num_selected_features,
                       num_remaining_features)

            for j in range(num_remaining_features):
                these_feature_names = (
                    selected_feature_names + [remaining_feature_names[j]])

                new_estimator_object = sklearn.base.clone(estimator_object)
                new_estimator_object.fit(
                    training_table.as_matrix(columns=these_feature_names),
                    training_table[target_name].values)

                these_forecast_probabilities = (
                    new_estimator_object.predict_proba(
                        validation_table.as_matrix(
                            columns=these_feature_names))[:, 1])
                new_xentropy_by_feature[j] = model_eval.get_cross_entropy(
                    these_forecast_probabilities,
                    validation_table[target_name].values)

            min_new_cross_entropy = numpy.min(new_xentropy_by_feature)
            this_best_feature_index = numpy.argmin(new_xentropy_by_feature)
            this_best_feature_name = remaining_feature_names[
                this_best_feature_index]

            print (
                'Minimum cross-entropy ({0:.4f}) given by adding feature '
                '"{1:s}"; previous min cross-entropy = {2:.4f}').format(
                    min_new_cross_entropy, this_best_feature_name,
                    min_cross_entropy_by_num_selected[num_selected_features])

            min_cross_entropy_by_num_selected[
                num_selected_features + 1] = min_new_cross_entropy

            selected_feature_names.append(this_best_feature_name)
            remaining_feature_names = set(remaining_feature_names)
            remaining_feature_names.remove(this_best_feature_name)
            remaining_feature_names = list(remaining_feature_names)

        for i in range(num_backward_steps):
            num_selected_features = len(selected_feature_names)
            new_xentropy_by_feature = numpy.full(
                num_selected_features, numpy.nan)

            print ('Major step {0:d}, backward step {1:d}: {2:d}/{3:d} '
                   'features selected...').format(
                       major_step_num, i + 1, num_selected_features,
                       num_features)

            for j in range(num_selected_features):
                these_feature_names = set(selected_feature_names)
                these_feature_names.remove(selected_feature_names[j])
                these_feature_names = list(these_feature_names)

                new_estimator_object = sklearn.base.clone(estimator_object)
                new_estimator_object.fit(
                    training_table.as_matrix(columns=these_feature_names),
                    training_table[target_name].values)

                these_forecast_probabilities = (
                    new_estimator_object.predict_proba(
                        validation_table.as_matrix(
                            columns=these_feature_names))[:, 1])
                new_xentropy_by_feature[j] = model_eval.get_cross_entropy(
                    these_forecast_probabilities,
                    validation_table[target_name].values)

            min_new_cross_entropy = numpy.min(new_xentropy_by_feature)
            this_worst_feature_index = numpy.argmin(new_xentropy_by_feature)
            this_worst_feature_name = selected_feature_names[
                this_worst_feature_index]

            print (
                'Minimum cross-entropy ({0:.4f}) given by removing feature '
                '"{1:s}"; previous min cross-entropy = {2:.4f}').format(
                    min_new_cross_entropy, this_worst_feature_name,
                    min_cross_entropy_by_num_selected[num_selected_features])

            min_cross_entropy_by_num_selected[num_selected_features] = numpy.nan
            min_cross_entropy_by_num_selected[
                num_selected_features - 1] = min_new_cross_entropy

            remaining_feature_names.append(this_worst_feature_name)
            selected_feature_names = set(selected_feature_names)
            selected_feature_names.remove(this_worst_feature_name)
            selected_feature_names = list(selected_feature_names)

        print '\n'
        stop_if_cross_entropy_above = min_cross_entropy_prev_major_step * (
            1. - min_fractional_xentropy_decrease)

        if (min_cross_entropy_by_num_selected[len(selected_feature_names)] >
                stop_if_cross_entropy_above):
            selected_feature_names = copy.deepcopy(
                selected_feature_names_prev_major_step)
            break

    sfs_dictionary = _evaluate_feature_selection(
        training_table=training_table, validation_table=validation_table,
        testing_table=testing_table, estimator_object=estimator_object,
        selected_feature_names=selected_feature_names, target_name=target_name)

    num_selected_features = len(selected_feature_names)
    sfs_dictionary.update(
        {VALIDATION_XENTROPY_BY_STEP_KEY:
             min_cross_entropy_by_num_selected[1:(num_selected_features + 1)]})
    return sfs_dictionary


def floating_sfs(
        training_table=None, validation_table=None, testing_table=None,
        feature_names=None, target_name=None, estimator_object=None,
        min_fractional_xentropy_decrease=
        DEFAULT_MIN_FRACTIONAL_XENTROPY_DECR_FOR_SFS):
    """Runs the SFFS (sequential forward floating selection) algorithm.

    SFFS is defined in Chapter 9 of Webb (2003).

    :param training_table: See documentation for
        _check_sequential_selection_inputs.
    :param validation_table: See doc for _check_sequential_selection_inputs.
    :param testing_table: See doc for _check_sequential_selection_inputs.
    :param feature_names: See doc for _check_sequential_selection_inputs.
    :param target_name: See doc for _check_sequential_selection_inputs.
    :param estimator_object: See doc for sequential_forward_selection.
    :param min_fractional_xentropy_decrease: See doc for
        sequential_forward_selection.
    :return: sfs_dictionary: See doc for sequential_forward_selection.
    """

    _check_sequential_selection_inputs(
        training_table=training_table, validation_table=validation_table,
        testing_table=testing_table, feature_names=feature_names,
        target_name=target_name)

    error_checking.assert_is_greater(min_fractional_xentropy_decrease, 0.)
    error_checking.assert_is_less_than(min_fractional_xentropy_decrease, 1.)

    # Initialize feature sets; initialize min cross-entropy to ridiculously high
    # number.
    major_step_num = 0
    selected_feature_names = []
    remaining_feature_names = copy.deepcopy(feature_names)

    num_features = len(feature_names)
    min_cross_entropy_by_num_selected = numpy.full(num_features, numpy.nan)
    min_cross_entropy_by_num_selected[0] = 1e10

    while remaining_feature_names:  # While there are still features to select.
        major_step_num += 1
        num_selected_features = len(selected_feature_names)
        num_remaining_features = len(remaining_feature_names)
        new_xentropy_by_feature = numpy.full(num_remaining_features, numpy.nan)

        print ('Major step {0:d} of SFFS: {1:d} features selected, {2:d} '
               'remaining...').format(major_step_num, num_selected_features,
                                      num_remaining_features)

        for j in range(num_remaining_features):
            these_feature_names = (
                selected_feature_names + [remaining_feature_names[j]])

            new_estimator_object = sklearn.base.clone(estimator_object)
            new_estimator_object.fit(
                training_table.as_matrix(columns=these_feature_names),
                training_table[target_name].values)

            these_forecast_probabilities = new_estimator_object.predict_proba(
                validation_table.as_matrix(columns=these_feature_names))[:, 1]
            new_xentropy_by_feature[j] = model_eval.get_cross_entropy(
                these_forecast_probabilities,
                validation_table[target_name].values)

        min_new_cross_entropy = numpy.min(new_xentropy_by_feature)
        this_best_feature_index = numpy.argmin(new_xentropy_by_feature)
        this_best_feature_name = remaining_feature_names[
            this_best_feature_index]

        print (
            'Minimum cross-entropy ({0:.4f}) given by adding feature '
            '"{1:s}"; previous min cross-entropy = {2:.4f}').format(
                min_new_cross_entropy, this_best_feature_name,
                min_cross_entropy_by_num_selected[num_selected_features])

        stop_if_cross_entropy_above = (
            min_cross_entropy_by_num_selected[num_selected_features] * (
                1. - min_fractional_xentropy_decrease))
        if min_new_cross_entropy > stop_if_cross_entropy_above:
            break

        min_cross_entropy_by_num_selected[
            num_selected_features + 1] = min_new_cross_entropy
        selected_feature_names.append(this_best_feature_name)
        remaining_feature_names = set(remaining_feature_names)
        remaining_feature_names.remove(this_best_feature_name)
        remaining_feature_names = list(remaining_feature_names)

        if len(selected_feature_names) < 2:
            continue

        backward_step_num = 0
        while len(selected_feature_names) >= 2:
            backward_step_num += 1
            num_selected_features = len(selected_feature_names)
            new_xentropy_by_feature = numpy.full(
                num_selected_features, numpy.nan)

            print ('Major step {0:d}, backward step {1:d}: {2:d}/{3:d} '
                   'features selected...').format(
                       major_step_num, backward_step_num, num_selected_features,
                       num_features)

            for j in range(num_selected_features):
                these_feature_names = set(selected_feature_names)
                these_feature_names.remove(selected_feature_names[j])
                these_feature_names = list(these_feature_names)

                new_estimator_object = sklearn.base.clone(estimator_object)
                new_estimator_object.fit(
                    training_table.as_matrix(columns=these_feature_names),
                    training_table[target_name].values)

                these_forecast_probabilities = (
                    new_estimator_object.predict_proba(
                        validation_table.as_matrix(
                            columns=these_feature_names))[:, 1])
                new_xentropy_by_feature[j] = model_eval.get_cross_entropy(
                    these_forecast_probabilities,
                    validation_table[target_name].values)

            min_new_cross_entropy = numpy.min(new_xentropy_by_feature)
            this_worst_feature_index = numpy.argmin(new_xentropy_by_feature)
            this_worst_feature_name = selected_feature_names[
                this_worst_feature_index]

            if backward_step_num == 1:
                if this_worst_feature_index == num_selected_features - 1:
                    break  # Cannot remove feature that was just added.

            else:
                if (min_new_cross_entropy >= min_cross_entropy_by_num_selected[
                        num_selected_features - 1]):
                    break  # Remove feature only if it improves performance.

            print (
                'Minimum cross-entropy ({0:.4f}) given by removing feature '
                '"{1:s}"; previous min cross-entropy = {2:.4f}').format(
                    min_new_cross_entropy, this_worst_feature_name,
                    min_cross_entropy_by_num_selected[
                        num_selected_features - 1])

            min_cross_entropy_by_num_selected[num_selected_features] = numpy.nan
            min_cross_entropy_by_num_selected[
                num_selected_features - 1] = min_new_cross_entropy

            remaining_feature_names.append(this_worst_feature_name)
            selected_feature_names = set(selected_feature_names)
            selected_feature_names.remove(this_worst_feature_name)
            selected_feature_names = list(selected_feature_names)

        print '\n'

    sfs_dictionary = _evaluate_feature_selection(
        training_table=training_table, validation_table=validation_table,
        testing_table=testing_table, estimator_object=estimator_object,
        selected_feature_names=selected_feature_names, target_name=target_name)

    num_selected_features = len(selected_feature_names)
    sfs_dictionary.update(
        {VALIDATION_XENTROPY_BY_STEP_KEY:
             min_cross_entropy_by_num_selected[1:(num_selected_features + 1)]})
    return sfs_dictionary


def sequential_backward_selection(
        training_table=None, validation_table=None, testing_table=None,
        feature_names=None, target_name=None, estimator_object=None,
        min_fractional_xentropy_decrease=
        DEFAULT_MIN_FRACTIONAL_XENTROPY_DECR_FOR_SBS):
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
    :param min_fractional_xentropy_decrease: Stopping criterion.  Once the
        fractional decrease in cross-entropy from removing a feature is <
        `min_fractional_xentropy_decrease`, SBS will stop.  Must be in range
        (-1, 1).  If negative, cross-entropy may increase slightly without SBS
        stopping.
    :return: sbs_dictionary: Same as output from _evaluate_feature_selection,
        but with the following additional keys.
    sbs_dictionary['removed_feature_names']: length-f with names of removed
        features (in order of their removal).
    sbs_dictionary['validation_xentropy_by_step']: length-f numpy array of
        validation cross-entropies.  The [i]th element is the cross-entropy
        after the [i]th feature removal.
    """

    _check_sequential_selection_inputs(
        training_table=training_table, validation_table=validation_table,
        testing_table=testing_table, feature_names=feature_names,
        target_name=target_name)

    error_checking.assert_is_greater(min_fractional_xentropy_decrease, -1.)
    error_checking.assert_is_less_than(min_fractional_xentropy_decrease, 1.)

    # Initialize feature sets; initialize min cross-entropy to ridiculously high
    # number.
    removed_feature_names = []
    selected_feature_names = copy.deepcopy(feature_names)

    num_features = len(feature_names)
    min_cross_entropy_by_num_removed = numpy.full(num_features, numpy.nan)
    min_cross_entropy_by_num_removed[0] = 1e10

    while selected_feature_names:  # While there are still features to remove.
        num_removed_features = len(removed_feature_names)
        num_selected_features = len(selected_feature_names)
        new_xentropy_by_feature = numpy.full(num_selected_features, numpy.nan)

        print ('Step {0:d} of sequential backward selection: {1:d} features '
               'removed, {2:d} remaining...').format(
                   num_removed_features + 1, num_removed_features,
                   num_selected_features)

        for j in range(num_selected_features):
            these_feature_names = set(selected_feature_names)
            these_feature_names.remove(selected_feature_names[j])
            these_feature_names = list(these_feature_names)

            new_estimator_object = sklearn.base.clone(estimator_object)
            new_estimator_object.fit(
                training_table.as_matrix(columns=these_feature_names),
                training_table[target_name].values)

            these_forecast_probabilities = new_estimator_object.predict_proba(
                validation_table.as_matrix(columns=these_feature_names))[:, 1]
            new_xentropy_by_feature[j] = model_eval.get_cross_entropy(
                these_forecast_probabilities,
                validation_table[target_name].values)

        min_new_cross_entropy = numpy.min(new_xentropy_by_feature)
        this_worst_feature_index = numpy.argmin(new_xentropy_by_feature)
        this_worst_feature_name = selected_feature_names[
            this_worst_feature_index]

        print (
            'Minimum cross-entropy ({0:.4f}) given by removing feature '
            '"{1:s}"; previous min cross-entropy = {2:.4f}').format(
                min_new_cross_entropy, this_worst_feature_name,
                min_cross_entropy_by_num_removed[num_removed_features])

        stop_if_cross_entropy_above = (
            min_cross_entropy_by_num_removed[num_removed_features] * (
                1. - min_fractional_xentropy_decrease))
        if min_new_cross_entropy > stop_if_cross_entropy_above:
            break

        min_cross_entropy_by_num_removed[
            num_removed_features + 1] = min_new_cross_entropy

        removed_feature_names.append(this_worst_feature_name)
        selected_feature_names = set(selected_feature_names)
        selected_feature_names.remove(this_worst_feature_name)
        selected_feature_names = list(selected_feature_names)

    sbs_dictionary = _evaluate_feature_selection(
        training_table=training_table, validation_table=validation_table,
        testing_table=testing_table, estimator_object=estimator_object,
        selected_feature_names=selected_feature_names, target_name=target_name)

    num_removed_features = len(removed_feature_names)
    sbs_dictionary.update({REMOVED_FEATURES_KEY: removed_feature_names})
    sbs_dictionary.update(
        {VALIDATION_XENTROPY_BY_STEP_KEY:
             min_cross_entropy_by_num_removed[1:(num_removed_features + 1)]})
    return sbs_dictionary


def sbs_with_forward_steps(
        training_table=None, validation_table=None, testing_table=None,
        feature_names=None, target_name=None, estimator_object=None,
        num_forward_steps=DEFAULT_NUM_FORWARD_STEPS_FOR_SBS,
        num_backward_steps=DEFAULT_NUM_BACKWARD_STEPS_FOR_SBS,
        min_fractional_xentropy_decrease=
        DEFAULT_MIN_FRACTIONAL_XENTROPY_DECR_FOR_SBS):
    """Runs SBS (sequential backward selection) with forward steps.

    This method is called "plus l - take away r selection" in Chapter 9 of Webb
    (2003), where l < r.

    --- DEFINITIONS ---

    "Forward step" = addition of one feature.
    "Backward step" = removal of one feature.
    "Major step" = r backward steps followed by l forward steps.

    :param training_table: See documentation for
        _check_sequential_selection_inputs.
    :param validation_table: See doc for _check_sequential_selection_inputs.
    :param testing_table: See doc for _check_sequential_selection_inputs.
    :param feature_names: See doc for _check_sequential_selection_inputs.
    :param target_name: See doc for _check_sequential_selection_inputs.
    :param estimator_object: Instance of scikit-learn estimator.  Must implement
        the methods `fit` and `predict_proba`.
    :param num_forward_steps: l-value (number of forward steps per major step).
    :param num_backward_steps: r-value (number of backward steps per major
        step).
    :param min_fractional_xentropy_decrease: Stopping criterion.  Once the
        fractional decrease in cross-entropy from a major step is <
        `min_fractional_xentropy_decrease`, algorithm will stop.  Must be in
        range (-1, 1).  If negative, cross-entropy may increase slightly without
        the algorithm stopping.
    :return: sbs_dictionary: See documentation for
        sequential_backward_selection.
    """

    _check_sequential_selection_inputs(
        training_table=training_table, validation_table=validation_table,
        testing_table=testing_table, feature_names=feature_names,
        target_name=target_name)

    num_features = len(feature_names)

    error_checking.assert_is_integer(num_backward_steps)
    error_checking.assert_is_geq(num_backward_steps, 2)
    error_checking.assert_is_leq(num_backward_steps, num_features)

    error_checking.assert_is_integer(num_forward_steps)
    error_checking.assert_is_geq(num_forward_steps, 1)
    error_checking.assert_is_less_than(num_forward_steps, num_backward_steps)

    error_checking.assert_is_greater(min_fractional_xentropy_decrease, -1.)
    error_checking.assert_is_less_than(min_fractional_xentropy_decrease, 1.)

    # Initialize feature sets; initialize min cross-entropy to ridiculously high
    # number.
    major_step_num = 0
    removed_feature_names = []
    selected_feature_names = copy.deepcopy(feature_names)

    min_cross_entropy_by_num_removed = numpy.full(num_features, numpy.nan)
    min_cross_entropy_by_num_removed[0] = 1e10

    while len(selected_feature_names) - num_backward_steps >= 0:
        major_step_num += 1

        min_cross_entropy_prev_major_step = min_cross_entropy_by_num_removed[
            len(removed_feature_names)]
        removed_feature_names_prev_major_step = copy.deepcopy(
            removed_feature_names)
        selected_feature_names_prev_major_step = copy.deepcopy(
            selected_feature_names)

        for i in range(num_backward_steps):
            num_removed_features = len(removed_feature_names)
            num_selected_features = len(selected_feature_names)
            new_xentropy_by_feature = numpy.full(
                num_selected_features, numpy.nan)

            print ('Major step {0:d}, backward step {1:d}: {2:d} features '
                   'removed, {3:d} remaining...').format(
                       major_step_num, i + 1, num_removed_features,
                       num_selected_features)

            for j in range(num_selected_features):
                these_feature_names = set(selected_feature_names)
                these_feature_names.remove(selected_feature_names[j])
                these_feature_names = list(these_feature_names)

                new_estimator_object = sklearn.base.clone(estimator_object)
                new_estimator_object.fit(
                    training_table.as_matrix(columns=these_feature_names),
                    training_table[target_name].values)

                these_forecast_probabilities = (
                    new_estimator_object.predict_proba(
                        validation_table.as_matrix(
                            columns=these_feature_names))[:, 1])
                new_xentropy_by_feature[j] = model_eval.get_cross_entropy(
                    these_forecast_probabilities,
                    validation_table[target_name].values)

            min_new_cross_entropy = numpy.min(new_xentropy_by_feature)
            this_worst_feature_index = numpy.argmin(new_xentropy_by_feature)
            this_worst_feature_name = selected_feature_names[
                this_worst_feature_index]

            print (
                'Minimum cross-entropy ({0:.4f}) given by removing feature '
                '"{1:s}"; previous min cross-entropy = {2:.4f}').format(
                    min_new_cross_entropy, this_worst_feature_name,
                    min_cross_entropy_by_num_removed[num_removed_features])

            min_cross_entropy_by_num_removed[
                num_removed_features + 1] = min_new_cross_entropy

            removed_feature_names.append(this_worst_feature_name)
            selected_feature_names = set(selected_feature_names)
            selected_feature_names.remove(this_worst_feature_name)
            selected_feature_names = list(selected_feature_names)

        for i in range(num_forward_steps):
            num_removed_features = len(removed_feature_names)
            new_xentropy_by_feature = numpy.full(
                num_removed_features, numpy.nan)

            print ('Major step {0:d}, forward step {1:d}: {2:d}/{3:d} '
                   'features removed...').format(
                       major_step_num, i + 1, num_removed_features,
                       num_features)

            for j in range(num_removed_features):
                these_feature_names = (
                    selected_feature_names + [removed_feature_names[j]])

                new_estimator_object = sklearn.base.clone(estimator_object)
                new_estimator_object.fit(
                    training_table.as_matrix(columns=these_feature_names),
                    training_table[target_name].values)

                these_forecast_probabilities = (
                    new_estimator_object.predict_proba(
                        validation_table.as_matrix(
                            columns=these_feature_names))[:, 1])
                new_xentropy_by_feature[j] = model_eval.get_cross_entropy(
                    these_forecast_probabilities,
                    validation_table[target_name].values)

            min_new_cross_entropy = numpy.min(new_xentropy_by_feature)
            this_best_feature_index = numpy.argmin(new_xentropy_by_feature)
            this_best_feature_name = removed_feature_names[
                this_best_feature_index]

            print (
                'Minimum cross-entropy ({0:.4f}) given by adding feature '
                '"{1:s}"; previous min cross-entropy = {2:.4f}').format(
                    min_new_cross_entropy, this_best_feature_name,
                    min_cross_entropy_by_num_removed[num_removed_features])

            min_cross_entropy_by_num_removed[num_removed_features] = numpy.nan
            min_cross_entropy_by_num_removed[
                num_removed_features + 1] = min_new_cross_entropy

            selected_feature_names.append(this_best_feature_name)
            removed_feature_names = set(removed_feature_names)
            removed_feature_names.remove(this_best_feature_name)
            removed_feature_names = list(removed_feature_names)

        print '\n'
        stop_if_cross_entropy_above = min_cross_entropy_prev_major_step * (
            1. - min_fractional_xentropy_decrease)

        if (min_cross_entropy_by_num_removed[len(removed_feature_names)] >
                stop_if_cross_entropy_above):
            removed_feature_names = copy.deepcopy(
                removed_feature_names_prev_major_step)
            selected_feature_names = copy.deepcopy(
                selected_feature_names_prev_major_step)
            break

    sbs_dictionary = _evaluate_feature_selection(
        training_table=training_table, validation_table=validation_table,
        testing_table=testing_table, estimator_object=estimator_object,
        selected_feature_names=selected_feature_names, target_name=target_name)

    num_removed_features = len(removed_feature_names)
    sbs_dictionary.update({REMOVED_FEATURES_KEY: removed_feature_names})
    sbs_dictionary.update(
        {VALIDATION_XENTROPY_BY_STEP_KEY:
             min_cross_entropy_by_num_removed[1:(num_removed_features + 1)]})
    return sbs_dictionary


def floating_sbs(
        training_table=None, validation_table=None, testing_table=None,
        feature_names=None, target_name=None, estimator_object=None,
        min_fractional_xentropy_decrease=
        DEFAULT_MIN_FRACTIONAL_XENTROPY_DECR_FOR_SBS):
    """Runs the SBFS (sequential backward floating selection) algorithm.

    SBFS is defined in Chapter 9 of Webb (2003).

    :param training_table: See documentation for
        _check_sequential_selection_inputs.
    :param validation_table: See doc for _check_sequential_selection_inputs.
    :param testing_table: See doc for _check_sequential_selection_inputs.
    :param feature_names: See doc for _check_sequential_selection_inputs.
    :param target_name: See doc for _check_sequential_selection_inputs.
    :param estimator_object: See doc for sequential_forward_selection.
    :param min_fractional_xentropy_decrease: See doc for
        sequential_backward_selection.
    :return: sbs_dictionary: See doc for sequential_backward_selection.
    """

    _check_sequential_selection_inputs(
        training_table=training_table, validation_table=validation_table,
        testing_table=testing_table, feature_names=feature_names,
        target_name=target_name)

    error_checking.assert_is_greater(min_fractional_xentropy_decrease, -1.)
    error_checking.assert_is_less_than(min_fractional_xentropy_decrease, 1.)

    # Initialize feature sets; initialize min cross-entropy to ridiculously high
    # number.
    major_step_num = 0
    removed_feature_names = []
    selected_feature_names = copy.deepcopy(feature_names)

    num_features = len(feature_names)
    min_cross_entropy_by_num_removed = numpy.full(num_features, numpy.nan)
    min_cross_entropy_by_num_removed[0] = 1e10

    while selected_feature_names:  # While there are still features to remove.
        major_step_num += 1
        num_removed_features = len(removed_feature_names)
        num_selected_features = len(selected_feature_names)
        new_xentropy_by_feature = numpy.full(num_selected_features, numpy.nan)

        print ('Major step {0:d} of SBFS: {1:d} features removed, {2:d} '
               'remaining...').format(
                   major_step_num, num_removed_features, num_selected_features)

        for j in range(num_selected_features):
            these_feature_names = set(selected_feature_names)
            these_feature_names.remove(selected_feature_names[j])
            these_feature_names = list(these_feature_names)

            new_estimator_object = sklearn.base.clone(estimator_object)
            new_estimator_object.fit(
                training_table.as_matrix(columns=these_feature_names),
                training_table[target_name].values)

            these_forecast_probabilities = new_estimator_object.predict_proba(
                validation_table.as_matrix(columns=these_feature_names))[:, 1]
            new_xentropy_by_feature[j] = model_eval.get_cross_entropy(
                these_forecast_probabilities,
                validation_table[target_name].values)

        min_new_cross_entropy = numpy.min(new_xentropy_by_feature)
        this_worst_feature_index = numpy.argmin(new_xentropy_by_feature)
        this_worst_feature_name = selected_feature_names[
            this_worst_feature_index]

        print (
            'Minimum cross-entropy ({0:.4f}) given by removing feature '
            '"{1:s}"; previous min cross-entropy = {2:.4f}').format(
                min_new_cross_entropy, this_worst_feature_name,
                min_cross_entropy_by_num_removed[num_removed_features])

        stop_if_cross_entropy_above = (
            min_cross_entropy_by_num_removed[num_removed_features] * (
                1. - min_fractional_xentropy_decrease))
        if min_new_cross_entropy > stop_if_cross_entropy_above:
            break

        min_cross_entropy_by_num_removed[
            num_removed_features + 1] = min_new_cross_entropy
        removed_feature_names.append(this_worst_feature_name)
        selected_feature_names = set(selected_feature_names)
        selected_feature_names.remove(this_worst_feature_name)
        selected_feature_names = list(selected_feature_names)

        if len(removed_feature_names) < 2:
            continue

        forward_step_num = 0
        while len(removed_feature_names) >= 2:
            forward_step_num += 1
            num_removed_features = len(removed_feature_names)
            new_xentropy_by_feature = numpy.full(
                num_removed_features, numpy.nan)

            print ('Major step {0:d}, forward step {1:d}: {2:d}/{3:d} '
                   'features removed...').format(
                       major_step_num, forward_step_num, num_removed_features,
                       num_features)

            for j in range(num_removed_features):
                these_feature_names = (
                    selected_feature_names + [removed_feature_names[j]])

                new_estimator_object = sklearn.base.clone(estimator_object)
                new_estimator_object.fit(
                    training_table.as_matrix(columns=these_feature_names),
                    training_table[target_name].values)

                these_forecast_probabilities = (
                    new_estimator_object.predict_proba(
                        validation_table.as_matrix(
                            columns=these_feature_names))[:, 1])
                new_xentropy_by_feature[j] = model_eval.get_cross_entropy(
                    these_forecast_probabilities,
                    validation_table[target_name].values)

            min_new_cross_entropy = numpy.min(new_xentropy_by_feature)
            this_best_feature_index = numpy.argmin(new_xentropy_by_feature)
            this_best_feature_name = removed_feature_names[
                this_best_feature_index]

            if forward_step_num == 1:
                if this_best_feature_index == num_removed_features - 1:
                    break  # Cannot add feature that was just removed.

            else:
                if (min_new_cross_entropy >= min_cross_entropy_by_num_removed[
                        num_removed_features - 1]):
                    break  # Add feature only if it improves performance.

            print (
                'Minimum cross-entropy ({0:.4f}) given by adding feature '
                '"{1:s}"; previous min cross-entropy = {2:.4f}').format(
                    min_new_cross_entropy, this_best_feature_name,
                    min_cross_entropy_by_num_removed[num_removed_features - 1])

            min_cross_entropy_by_num_removed[num_removed_features] = numpy.nan
            min_cross_entropy_by_num_removed[
                num_removed_features - 1] = min_new_cross_entropy

            selected_feature_names.append(this_best_feature_name)
            removed_feature_names = set(removed_feature_names)
            removed_feature_names.remove(this_best_feature_name)
            removed_feature_names = list(removed_feature_names)

        print '\n'

    sbs_dictionary = _evaluate_feature_selection(
        training_table=training_table, validation_table=validation_table,
        testing_table=testing_table, estimator_object=estimator_object,
        selected_feature_names=selected_feature_names, target_name=target_name)

    num_removed_features = len(removed_feature_names)
    sbs_dictionary.update({REMOVED_FEATURES_KEY: removed_feature_names})
    sbs_dictionary.update(
        {VALIDATION_XENTROPY_BY_STEP_KEY:
             min_cross_entropy_by_num_removed[1:(num_removed_features + 1)]})
    return sbs_dictionary


def permutation_selection(
        training_table=None, validation_table=None, feature_names=None,
        target_name=None, estimator_object=None):
    """Runs the permutation algorithm (Lakshmanan et al. 2015).

    :param training_table: See documentation for
        _check_sequential_selection_inputs.
    :param validation_table: See doc for _check_sequential_selection_inputs.
    :param feature_names: See doc for _check_sequential_selection_inputs.
    :param target_name: See doc for _check_sequential_selection_inputs.
    :param estimator_object: See doc for sequential_forward_selection.
    :return: permutation_table: pandas DataFrame with the following columns.
        Each row corresponds to one feature.  Order of rows = order in which
        features were permuted.  In other words, the feature in the [i]th row
        was the [i]th to be permuted.
    permutation_table.feature_name: Name of feature.
    permutation_table.validation_cross_entropy: Validation cross-entropy after
        permuting feature (and keeping features in previous rows permuted).
    permutation_table.validation_auc: Same but for area under ROC curve.

    orig_validation_cross_entropy: Validation cross-entropy with no permuted
        features.
    orig_validation_auc: Validation area under ROC curve with no permuted
        features.
    """

    _check_sequential_selection_inputs(
        training_table=training_table, validation_table=validation_table,
        testing_table=None, feature_names=feature_names,
        target_name=target_name)

    # Find validation cross-entropy and AUC before permutation.
    new_estimator_object = sklearn.base.clone(estimator_object)
    new_estimator_object.fit(
        training_table.as_matrix(columns=feature_names),
        training_table[target_name].values)

    forecast_probs_for_validation = new_estimator_object.predict_proba(
        validation_table.as_matrix(columns=feature_names))[:, 1]
    orig_validation_cross_entropy = model_eval.get_cross_entropy(
        forecast_probs_for_validation, validation_table[target_name].values)
    orig_validation_auc = sklearn.metrics.roc_auc_score(
        validation_table[target_name].values, forecast_probs_for_validation)

    # Initialize values for permutation algorithm.
    remaining_feature_names = copy.deepcopy(feature_names)
    permutation_dict = {
        FEATURE_NAME_KEY: [], VALIDATION_XENTROPY_KEY: [],
        VALIDATION_AUC_KEY: []
    }

    while remaining_feature_names:  # While there are still features to permute.
        num_permuted_features = len(permutation_dict[FEATURE_NAME_KEY])
        num_remaining_features = len(remaining_feature_names)

        print ('Step {0:d} of permutation selection: {1:d} features permuted, '
               '{2:d} remaining...').format(
                   num_permuted_features + 1, num_permuted_features,
                   num_remaining_features)

        new_xentropy_by_feature = numpy.full(num_remaining_features, numpy.nan)
        new_auc_by_feature = numpy.full(num_remaining_features, numpy.nan)
        permuted_values_for_best_feature = None

        for j in range(num_remaining_features):
            this_validation_matrix = validation_table.as_matrix(
                columns=feature_names)
            column_to_permute = feature_names.index(remaining_feature_names[j])
            this_validation_matrix[:, column_to_permute] = (
                numpy.random.permutation(
                    this_validation_matrix[:, column_to_permute]))

            these_forecast_probabilities = new_estimator_object.predict_proba(
                this_validation_matrix)[:, 1]
            new_xentropy_by_feature[j] = model_eval.get_cross_entropy(
                these_forecast_probabilities,
                validation_table[target_name].values)
            new_auc_by_feature[j] = sklearn.metrics.roc_auc_score(
                validation_table[target_name].values,
                these_forecast_probabilities)

            if numpy.nanargmax(new_xentropy_by_feature) == j:
                permuted_values_for_best_feature = (
                    this_validation_matrix[:, column_to_permute])

        max_new_cross_entropy = numpy.max(new_xentropy_by_feature)
        this_best_feature_index = numpy.argmax(new_xentropy_by_feature)
        this_best_feature_name = remaining_feature_names[
            this_best_feature_index]

        if len(permutation_dict[VALIDATION_XENTROPY_KEY]):
            max_prev_cross_entropy = permutation_dict[
                VALIDATION_XENTROPY_KEY][-1]
        else:
            max_prev_cross_entropy = copy.deepcopy(
                orig_validation_cross_entropy)

        print (
            'Max cross-entropy ({0:.4f}) given by permuting feature "{1:s}"; '
            'previous max cross-entropy = {2:.4f}').format(
                max_new_cross_entropy, this_best_feature_name,
                max_prev_cross_entropy)

        validation_table = validation_table.assign(
            **{this_best_feature_name: permuted_values_for_best_feature})

        permutation_dict[FEATURE_NAME_KEY].append(this_best_feature_name)
        permutation_dict[VALIDATION_XENTROPY_KEY].append(
            new_xentropy_by_feature[this_best_feature_index])
        permutation_dict[VALIDATION_AUC_KEY].append(
            new_auc_by_feature[this_best_feature_index])

        remaining_feature_names = set(remaining_feature_names)
        remaining_feature_names.remove(this_best_feature_name)
        remaining_feature_names = list(remaining_feature_names)

    permutation_table = pandas.DataFrame.from_dict(permutation_dict)
    return permutation_table, orig_validation_cross_entropy, orig_validation_auc


def plot_forward_selection_results(
        forward_selection_dict, plot_feature_names=False,
        bar_face_colour=DEFAULT_BAR_FACE_COLOUR,
        bar_edge_colour=DEFAULT_BAR_EDGE_COLOUR,
        bar_edge_width=DEFAULT_BAR_EDGE_WIDTH):
    """Plots bar graph with results of any forward-selection algorithm.

    :param forward_selection_dict: Dictionary returned by
        `sequential_forward_selection`, `sfs_with_backward_steps`, or
        `floating_sfs`.
    :param plot_feature_names: See documentation for
        _plot_selection_results.
    :param bar_face_colour: See doc for _plot_selection_results.
    :param bar_edge_colour: See doc for _plot_selection_results.
    :param bar_edge_width: See doc for _plot_selection_results.
    """

    _plot_selection_results(
        feature_name_by_step=forward_selection_dict[SELECTED_FEATURES_KEY],
        cross_entropy_by_step=forward_selection_dict[
            VALIDATION_XENTROPY_BY_STEP_KEY],
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
    :param plot_feature_names: See documentation for
        _plot_selection_results.
    :param bar_face_colour: See doc for _plot_selection_results.
    :param bar_edge_colour: See doc for _plot_selection_results.
    :param bar_edge_width: See doc for _plot_selection_results.
    """

    _plot_selection_results(
        feature_name_by_step=backward_selection_dict[REMOVED_FEATURES_KEY],
        cross_entropy_by_step=backward_selection_dict[
            VALIDATION_XENTROPY_BY_STEP_KEY],
        selection_type=BACKWARD_SELECTION_TYPE,
        plot_feature_names=plot_feature_names, bar_face_colour=bar_face_colour,
        bar_edge_colour=bar_edge_colour, bar_edge_width=bar_edge_width)


def plot_permutation_results(
        permutation_table, plot_feature_names=False,
        orig_validation_cross_entropy=None,
        bar_face_colour=DEFAULT_BAR_FACE_COLOUR,
        bar_edge_colour=DEFAULT_BAR_EDGE_COLOUR,
        bar_edge_width=DEFAULT_BAR_EDGE_WIDTH):
    """Plots bar graph with results of permutation selection.

    :param permutation_table: pandas DataFrame returned by
        `permutation_selection`.
    :param plot_feature_names: See documentation for
        _plot_selection_results.
    :param orig_validation_cross_entropy: Validation cross-entropy before
        permutation.
    :param bar_face_colour: See doc for _plot_selection_results.
    :param bar_edge_colour: See doc for _plot_selection_results.
    :param bar_edge_width: See doc for _plot_selection_results.
    """

    _plot_selection_results(
        feature_name_by_step=permutation_table[FEATURE_NAME_KEY].values,
        cross_entropy_by_step=permutation_table[VALIDATION_XENTROPY_KEY].values,
        selection_type=PERMUTATION_TYPE,
        cross_entropy_before_permutn=orig_validation_cross_entropy,
        plot_feature_names=plot_feature_names, bar_face_colour=bar_face_colour,
        bar_edge_colour=bar_edge_colour, bar_edge_width=bar_edge_width)
