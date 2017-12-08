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
from gewittergefahr.gg_utils import model_evaluation as model_eval
from gewittergefahr.gg_utils import error_checking

DEFAULT_MIN_FRACTIONAL_XENTROPY_DECR_FOR_SFS = 0.01
DEFAULT_MIN_FRACTIONAL_XENTROPY_DECR_FOR_SBS = -0.01

SELECTED_FEATURES_KEY = 'selected_feature_names'
REMOVED_FEATURES_KEY = 'removed_feature_names'
FEATURE_NAME_KEY = 'feature_name'
VALIDATION_XENTROPY_KEY = 'validation_cross_entropy'
VALIDATION_AUC_KEY = 'validation_auc'
TESTING_XENTROPY_KEY = 'testing_cross_entropy'
TESTING_AUC_KEY = 'testing_auc'


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


def sequential_forward_selection(
        training_table=None, validation_table=None, testing_table=None,
        feature_names=None, target_name=None, estimator_object=None,
        min_fractional_xentropy_decrease=
        DEFAULT_MIN_FRACTIONAL_XENTROPY_DECR_FOR_SFS):
    """Runs the SFS (sequential forward selection) algorithm.

    SFS is defined in Chapter 9 of Webb (2003).

    F = number of features

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
    :return: sfs_dictionary: Same as output from _evaluate_feature_selection.
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
    min_cross_entropy = 1e10

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
                min_cross_entropy)

        stop_if_cross_entropy_above = min_cross_entropy * (
            1. - min_fractional_xentropy_decrease)
        if min_new_cross_entropy > stop_if_cross_entropy_above:
            break

        min_cross_entropy = copy.deepcopy(min_new_cross_entropy)
        selected_feature_names.append(this_best_feature_name)
        remaining_feature_names = set(remaining_feature_names)
        remaining_feature_names.remove(this_best_feature_name)
        remaining_feature_names = list(remaining_feature_names)

    return _evaluate_feature_selection(
        training_table=training_table, validation_table=validation_table,
        testing_table=testing_table, estimator_object=estimator_object,
        selected_feature_names=selected_feature_names, target_name=target_name)


def sequential_backward_selection(
        training_table=None, validation_table=None, testing_table=None,
        feature_names=None, target_name=None, estimator_object=None,
        min_fractional_xentropy_decrease=
        DEFAULT_MIN_FRACTIONAL_XENTROPY_DECR_FOR_SBS):
    """Runs the SBS (sequential backward selection) algorithm.

    SBS is defined in Chapter 9 of Webb (2003).

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
        except with one additional key.
    sbs_dictionary['removed_feature_names']: 1-D list with names of removed
        features (in order of their removal).
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
    min_cross_entropy = 1e10

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
                min_cross_entropy)

        stop_if_cross_entropy_above = min_cross_entropy * (
            1. - min_fractional_xentropy_decrease)
        if min_new_cross_entropy > stop_if_cross_entropy_above:
            break

        min_cross_entropy = copy.deepcopy(min_new_cross_entropy)
        removed_feature_names.append(this_worst_feature_name)
        selected_feature_names = set(selected_feature_names)
        selected_feature_names.remove(this_worst_feature_name)
        selected_feature_names = list(selected_feature_names)

    sbs_dictionary = _evaluate_feature_selection(
        training_table=training_table, validation_table=validation_table,
        testing_table=testing_table, estimator_object=estimator_object,
        selected_feature_names=selected_feature_names, target_name=target_name)
    sbs_dictionary.update({REMOVED_FEATURES_KEY: removed_feature_names})
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
