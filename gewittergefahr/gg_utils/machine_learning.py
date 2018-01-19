"""Handles storm-based machine learning.

"Storm-based" means that each example is one storm object.  Thus, each forecast
is for one storm object, within some distance buffer and some range of lead
times.  Forecasts made by this module are non-spatial (i.e., they do not
consider storm motion and are not tied to a spatial location whatsoever -- they
are tied only to the storm object and the corresponding storm track).  To
convert forecasts from storm-based to spatial, see gridded_forecasts.py.

Currently this module does binary classification only.
"""

import numpy
import sklearn
from gewittergefahr.gg_utils import error_checking

DEFAULT_PENALTY_MULT_FOR_LOGISTIC = 1e-4
DEFAULT_L1_WEIGHT = 0.5
DEFAULT_MAX_NUM_EPOCHS_FOR_LOGISTIC = 10
DEFAULT_CONVERGENCE_TOL_FOR_LOGISTIC = 1e-4

DEFAULT_NUM_TREES_FOR_RANDOM_FOREST = 500
DEFAULT_LOSS_FOR_RANDOM_FOREST = 'entropy'
DEFAULT_MAX_DEPTH_FOR_RANDOM_FOREST = None
DEFAULT_MIN_EXAMPLES_PER_RF_SPLIT = None
DEFAULT_MIN_EXAMPLES_PER_RF_LEAF = 30

DEFAULT_NUM_TREES_FOR_GBT = 500
DEFAULT_LOSS_FOR_GBT = 'exponential'
DEFAULT_LEARNING_RATE_FOR_GBT = 0.1
DEFAULT_MAX_DEPTH_FOR_GBT = None
DEFAULT_MIN_EXAMPLES_PER_GBT_SPLIT = None
DEFAULT_MIN_EXAMPLES_PER_GBT_LEAF = 30
DEFAULT_SUBSAMPLING_FRACTION_FOR_GBT = 1.

DEFAULT_HIDDEN_LAYER_SIZES_FOR_NN = numpy.array([100, 50], dtype=int)
DEFAULT_ACTIVATION_FUNCTION_FOR_NN = 'relu'
DEFAULT_SOLVER_FOR_NN = 'adam'
VALID_SOLVERS_FOR_NN = ['sgd', 'adam']
DEFAULT_L2_WEIGHT_FOR_NN = 1e-4
DEFAULT_BATCH_SIZE_FOR_NN = 128
DEFAULT_LEARNING_RATE_FOR_NN = 1e-3
DEFAULT_MAX_NUM_EPOCHS_FOR_NN = 500
DEFAULT_CONVERGENCE_TOLERANCE_FOR_NN = 1e-4
DEFAULT_EARLY_STOPPING_FRACTION_FOR_NN = 0.1


def _check_training_data(training_table, feature_names, target_name):
    """Checks training data for errors.

    :param training_table: pandas DataFrame, where each row is one training
        example.
    :param feature_names: 1-D list with names of features (predictor variables).
        Each feature must be a column in training_table.
    :param target_name: Name of target variable (predictand).  Must be a column
        in training_table.
    """

    error_checking.assert_is_string(target_name)
    error_checking.assert_is_string_list(feature_names)
    error_checking.assert_is_numpy_array(
        numpy.array(feature_names), num_dimensions=1)
    error_checking.assert_columns_in_dataframe(
        training_table, feature_names + [target_name])


def _check_decision_tree_hyperparams(
        num_trees, loss_function_string, num_features_total,
        num_features_per_split, max_depth, min_examples_per_split,
        min_examples_per_leaf, learning_rate=None, subsampling_fraction=None):
    """Checks decision-tree hyperparameters (input parameters) for errors.

    :param num_trees: Number of trees in ensemble.
    :param loss_function_string: Loss function.  This method ensures only that
        the loss function is a string.  The specific learning method will
        determine whether or not the string is valid.
    :param num_features_total: Number of features in training data.
    :param num_features_per_split: Number of features to investigate at each
        split point (branch node).
    :param max_depth: Max depth of any tree in ensemble.
    :param min_examples_per_split: Minimum number of examples (storm objects) at
        a split point (branch node).
    :param min_examples_per_leaf: Minimum number of examples (storm objects) at
        a leaf node.
    :param learning_rate: [for gradient-boosting only] Learning rate (used to
        decrease the contribution of each successive tree).
    :param subsampling_fraction: [for gradient-boosting only] Fraction of
        examples to use in training each tree.
    """

    error_checking.assert_is_integer(num_trees)
    error_checking.assert_is_geq(num_trees, 2)
    error_checking.assert_is_string(loss_function_string)
    error_checking.assert_is_integer(num_features_per_split)
    error_checking.assert_is_greater(num_features_per_split, 0)
    error_checking.assert_is_leq(num_features_per_split, num_features_total)

    error_checking.assert_is_integer(max_depth)
    error_checking.assert_is_greater(max_depth, 0)
    error_checking.assert_is_integer(min_examples_per_split)
    error_checking.assert_is_greater(min_examples_per_split, 1)
    error_checking.assert_is_integer(min_examples_per_leaf)
    error_checking.assert_is_greater(min_examples_per_leaf, 0)

    if learning_rate is not None:
        error_checking.assert_is_greater(learning_rate, 0.)
        error_checking.assert_is_less_than(learning_rate, 1.)

    if subsampling_fraction is not None:
        error_checking.assert_is_greater(subsampling_fraction, 0.)
        error_checking.assert_is_leq(subsampling_fraction, 1.)


def train_logistic_regression(
        training_table, feature_names, target_name, fit_intercept,
        convergence_tolerance=DEFAULT_CONVERGENCE_TOL_FOR_LOGISTIC,
        penalty_multiplier=DEFAULT_PENALTY_MULT_FOR_LOGISTIC,
        l1_weight=DEFAULT_L1_WEIGHT,
        max_num_epochs=DEFAULT_MAX_NUM_EPOCHS_FOR_LOGISTIC):
    """Trains logistic-regression model for binary classification.

    :param training_table: See documentation for _check_training_data.
    :param feature_names: See doc for _check_training_data.
    :param target_name: See doc for _check_training_data.
    :param fit_intercept: Boolean flag.  If True, will fit the intercept (bias)
        coefficient.  If False, will assume intercept = 0.
    :param convergence_tolerance: Stopping criterion.  Training will stop when
        `loss > previous_loss - convergence_tolerance`.
    :param penalty_multiplier: Coefficient used to multiply L1 and L2 penalties
        in loss function.
    :param l1_weight: Weight for L1 regularization penalty.  L2 weight will be
        `1 - l1_weight`.
    :param max_num_epochs: Max number of training epochs (passes over training
        data).
    :return: model_object: Trained model (instance of
        `sklearn.linear_model.SGDClassifier`).
    """

    _check_training_data(
        training_table=training_table, feature_names=feature_names,
        target_name=target_name)

    error_checking.assert_is_boolean(fit_intercept)
    error_checking.assert_is_greater(convergence_tolerance, 0.)
    error_checking.assert_is_greater(penalty_multiplier, 0.)
    error_checking.assert_is_geq(l1_weight, 0.)
    error_checking.assert_is_leq(l1_weight, 1.)
    error_checking.assert_is_integer(max_num_epochs)
    error_checking.assert_is_greater(max_num_epochs, 0)

    model_object = sklearn.linear_model.SGDClassifier(
        loss='log', penalty='elasticnet', alpha=penalty_multiplier,
        l1_ratio=l1_weight, fit_intercept=fit_intercept,
        max_iter=max_num_epochs, tol=convergence_tolerance, verbose=1)

    model_object.fit(
        training_table.as_matrix(columns=feature_names),
        training_table[target_name].values)
    return model_object


def train_random_forest(
        training_table, feature_names, target_name,
        num_trees=DEFAULT_NUM_TREES_FOR_RANDOM_FOREST,
        loss_function=DEFAULT_LOSS_FOR_RANDOM_FOREST,
        num_features_per_split=None,
        max_depth=DEFAULT_MAX_DEPTH_FOR_RANDOM_FOREST,
        min_examples_per_split=DEFAULT_MIN_EXAMPLES_PER_RF_SPLIT,
        min_examples_per_leaf=DEFAULT_MIN_EXAMPLES_PER_RF_LEAF):
    """Trains random-forest model for binary classification.

    :param training_table: See documentation for _check_training_data.
    :param feature_names: See doc for _check_training_data.
    :param target_name: See doc for _check_training_data.
    :param num_trees: Number of trees in forest.
    :param loss_function: Loss function (either "entropy" or "gini").
    :param num_features_per_split: Number of features to investigate at each
        split point (branch node).  `None` defaults to sqrt(len(feature_names)).
    :param max_depth: Max depth of any tree in forest.
    :param min_examples_per_split: Minimum number of examples (storm objects) at
        a split point (branch node).  `None` defaults to 2.
    :param min_examples_per_leaf: Minimum number of examples (storm objects) at
        a leaf node.  `None` defaults to 1.
    :return: model_object: Trained model (instance of
        `sklearn.ensemble.RandomForestClassifier`).
    """

    _check_training_data(
        training_table=training_table, feature_names=feature_names,
        target_name=target_name)
    num_features_total = len(feature_names)

    if num_features_per_split is None:
        num_features_per_split = int(
            numpy.round(numpy.sqrt(num_features_total)))
    if min_examples_per_split is None:
        min_examples_per_split = 2
    if min_examples_per_leaf is None:
        min_examples_per_leaf = 1

    _check_decision_tree_hyperparams(
        num_trees=num_trees, loss_function_string=loss_function,
        num_features_total=num_features_total,
        num_features_per_split=num_features_per_split, max_depth=max_depth,
        min_examples_per_split=min_examples_per_split,
        min_examples_per_leaf=min_examples_per_leaf)

    error_checking.assert_is_string(loss_function)

    model_object = sklearn.ensemble.RandomForestClassifier(
        n_estimators=num_trees, criterion=loss_function,
        max_features=num_features_per_split, max_depth=max_depth,
        min_samples_split=min_examples_per_split,
        min_samples_leaf=min_examples_per_leaf, bootstrap=True, verbose=1)

    model_object.fit(
        training_table.as_matrix(columns=feature_names),
        training_table[target_name].values)
    return model_object


def train_gradient_boosted_trees(
        training_table, feature_names, target_name,
        num_trees=DEFAULT_NUM_TREES_FOR_GBT, loss_function=DEFAULT_LOSS_FOR_GBT,
        learning_rate=DEFAULT_LEARNING_RATE_FOR_GBT,
        num_features_per_split=None, max_depth=DEFAULT_MAX_DEPTH_FOR_GBT,
        min_examples_per_split=DEFAULT_MIN_EXAMPLES_PER_GBT_SPLIT,
        min_examples_per_leaf=DEFAULT_MIN_EXAMPLES_PER_GBT_LEAF,
        subsampling_fraction=DEFAULT_SUBSAMPLING_FRACTION_FOR_GBT):
    """Trains gradient-boosted trees for binary classification.

    :param training_table: See documentation for _check_training_data.
    :param feature_names: See doc for _check_training_data.
    :param target_name: See doc for _check_training_data.
    :param num_trees: Number of trees in ensemble.
    :param loss_function: Loss function (either "deviance" or "exponential").
    :param learning_rate: Learning rate (used to decrease the contribution of
        each successive tree).
    :param num_features_per_split: Number of features to investigate at each
        split point (branch node).  `None` defaults to len(feature_names).
    :param max_depth: Max depth of any tree in forest.
    :param min_examples_per_split: Minimum number of examples (storm objects) at
        a split point (branch node).  `None` defaults to 2.
    :param min_examples_per_leaf: Minimum number of examples (storm objects) at
        a leaf node.  `None` defaults to 1.
    :param subsampling_fraction: Fraction of examples to use in training each
        tree.
    :return: model_object: Trained model (instance of
        `sklearn.ensemble.GradientBoostingClassifier`).
    """

    _check_training_data(
        training_table=training_table, feature_names=feature_names,
        target_name=target_name)
    num_features_total = len(feature_names)

    if num_features_per_split is None:
        num_features_per_split = num_features_total
    if min_examples_per_split is None:
        min_examples_per_split = 2
    if min_examples_per_leaf is None:
        min_examples_per_leaf = 1

    _check_decision_tree_hyperparams(
        num_trees=num_trees, loss_function_string=loss_function,
        num_features_total=num_features_total,
        num_features_per_split=num_features_per_split, max_depth=max_depth,
        min_examples_per_split=min_examples_per_split,
        min_examples_per_leaf=min_examples_per_leaf)

    model_object = sklearn.ensemble.GradientBoostingClassifier(
        loss=loss_function, learning_rate=learning_rate, n_estimators=num_trees,
        max_depth=max_depth, min_samples_split=min_examples_per_split,
        min_samples_leaf=min_examples_per_leaf, subsample=subsampling_fraction,
        max_features=num_features_per_split, verbose=1)

    model_object.fit(
        training_table.as_matrix(columns=feature_names),
        training_table[target_name].values)
    return model_object


def train_neural_net(
        training_table, feature_names, target_name,
        hidden_layer_sizes=DEFAULT_HIDDEN_LAYER_SIZES_FOR_NN,
        hidden_layer_activation_function=DEFAULT_ACTIVATION_FUNCTION_FOR_NN,
        solver=DEFAULT_SOLVER_FOR_NN, l2_weight=DEFAULT_L2_WEIGHT_FOR_NN,
        num_examples_per_batch=DEFAULT_BATCH_SIZE_FOR_NN,
        learning_rate=DEFAULT_LEARNING_RATE_FOR_NN,
        max_num_epochs=DEFAULT_MAX_NUM_EPOCHS_FOR_NN,
        convergence_tolerance=DEFAULT_CONVERGENCE_TOLERANCE_FOR_NN,
        allow_early_stopping=True,
        early_stopping_fraction=DEFAULT_EARLY_STOPPING_FRACTION_FOR_NN):
    """Trains a neural net for binary classification.

    H = number of hidden layers

    :param training_table: See documentation for _check_training_data.
    :param feature_names: See doc for _check_training_data.
    :param target_name: See doc for _check_training_data.
    :param hidden_layer_sizes: length-H numpy array, where the [i]th element is
        the number of nodes in the [i]th hidden layer.
    :param hidden_layer_activation_function: Activation function for hidden
        layers.  See `sklearn.neural_network.MLPClassifier` documentation for
        valid options.
    :param solver:  Solver.  Valid options are "sgd" and "adam".
    :param l2_weight: Weight for L2 penalty.
    :param num_examples_per_batch: Number of examples per training batch.
    :param learning_rate: Learning rate.
    :param max_num_epochs: Max number of training epochs (passes over training
        data).
    :param convergence_tolerance: Stopping criterion.  Training will stop when
        loss has improved by < `convergence_tolerance` for each of two
        consecutive epochs.
    :param allow_early_stopping: Boolean flag.  If True, some training data will
        be set aside as "validation data" to check for early stopping.  In this
        case, training will stop when loss has improved by <
        `convergence_tolerance` for each of two consecutive epochs.
    :param early_stopping_fraction: Fraction of training examples to use when
        checking early-stopping criterion.
    :return: model_object: Trained model (instance of
        `sklearn.neural_network.MLPClassifier`).
    :raises: ValueError: if `solver not in VALID_SOLVERS_FOR_NN`.
    """

    _check_training_data(
        training_table=training_table, feature_names=feature_names,
        target_name=target_name)

    error_checking.assert_is_integer_numpy_array(hidden_layer_sizes)
    error_checking.assert_is_numpy_array(hidden_layer_sizes, num_dimensions=1)
    error_checking.assert_is_greater_numpy_array(hidden_layer_sizes, 0)

    error_checking.assert_is_string(solver)
    if solver not in VALID_SOLVERS_FOR_NN:
        error_string = (
            '\n\n{0:s}\n\nValid solvers (listed above) do not include "{1:s}".'
        ).format(VALID_SOLVERS_FOR_NN, solver)
        raise ValueError(error_string)

    error_checking.assert_is_integer(num_examples_per_batch)
    error_checking.assert_is_geq(num_examples_per_batch, 2)
    error_checking.assert_is_greater(learning_rate, 0.)
    error_checking.assert_is_leq(learning_rate, 1.)
    error_checking.assert_is_integer(max_num_epochs)
    error_checking.assert_is_greater(max_num_epochs, 0)
    error_checking.assert_is_greater(convergence_tolerance, 0.)
    error_checking.assert_is_boolean(allow_early_stopping)

    if allow_early_stopping:
        error_checking.assert_is_greater(early_stopping_fraction, 0.)
        error_checking.assert_is_less_than(early_stopping_fraction, 0.5)

    model_object = sklearn.neural_network.MLPClassifier(
        hidden_layer_sizes=hidden_layer_sizes,
        activation=hidden_layer_activation_function, solver=solver,
        alpha=l2_weight, batch_size=num_examples_per_batch,
        learning_rate_init=learning_rate, max_iter=max_num_epochs,
        tol=convergence_tolerance, verbose=1,
        early_stopping=allow_early_stopping,
        validation_fraction=early_stopping_fraction)

    model_object.fit(
        training_table.as_matrix(columns=feature_names),
        training_table[target_name].values)
    return model_object
