"""Methods for creating, training, and applying gradient-boosted trees (GBT).

The input data for GBT are features (outputs of the last "Flatten" layer)
created by a convolutional neural network (CNN).
"""

import numpy
import xgboost
import keras.utils
from gewittergefahr.gg_utils import error_checking

# TODO(thunderhoser): Still need some way to do oversampling/undersampling.

DEFAULT_NUM_TREES = 100
DEFAULT_LEARNING_RATE = 0.1
DEFAULT_MAX_TREE_DEPTH = 3
DEFAULT_FRACTION_OF_EXAMPLES_PER_TREE = 1. - numpy.exp(-1.)
DEFAULT_FRACTION_OF_FEATURES_PER_SPLIT = 1.
DEFAULT_L2_WEIGHT = 0.001


def create_model(
        num_classes, num_trees=DEFAULT_NUM_TREES,
        learning_rate=DEFAULT_LEARNING_RATE, max_depth=DEFAULT_MAX_TREE_DEPTH,
        fraction_of_examples_per_tree=DEFAULT_FRACTION_OF_EXAMPLES_PER_TREE,
        fraction_of_features_per_split=DEFAULT_FRACTION_OF_FEATURES_PER_SPLIT,
        l2_weight=DEFAULT_L2_WEIGHT):
    """Creates GBT model for classification.

    :param num_classes: Number of target classes.  If num_classes = 2, the model
        will do binary probabilistic classification.  If num_classes > 2, the
        model will do multiclass probabilistic classification.
    :param num_trees: Number of trees.
    :param learning_rate: Learning rate.
    :param max_depth: Maximum depth (applied to each tree).
    :param fraction_of_examples_per_tree: Fraction of examples (storm objects)
        to be used in training each tree.
    :param fraction_of_features_per_split: Fraction of features (predictor
        variables) to be used at each split point.
    :param l2_weight: L2-regularization weight.
    :return: model_object: Untrained instance of `xgboost.XGBClassifier`.
    """

    error_checking.assert_is_integer(num_classes)
    error_checking.assert_is_geq(num_classes, 2)
    error_checking.assert_is_integer(num_trees)
    error_checking.assert_is_geq(num_trees, 10)
    error_checking.assert_is_leq(num_trees, 1000)
    error_checking.assert_is_greater(learning_rate, 0.)
    error_checking.assert_is_less_than(learning_rate, 1.)
    error_checking.assert_is_integer(max_depth)
    error_checking.assert_is_geq(max_depth, 1)
    error_checking.assert_is_leq(max_depth, 10)
    error_checking.assert_is_greater(fraction_of_examples_per_tree, 0.)
    error_checking.assert_is_leq(fraction_of_examples_per_tree, 1.)
    error_checking.assert_is_greater(fraction_of_features_per_split, 0.)
    error_checking.assert_is_leq(fraction_of_features_per_split, 1.)
    error_checking.assert_is_geq(l2_weight, 0.)

    if num_classes == 2:
        objective_function_string = 'binary:logistic'
    else:
        objective_function_string = 'multi:softprob'

    return xgboost.XGBClassifier(
        max_depth=max_depth, learning_rate=learning_rate,
        n_estimators=num_trees, silent=False,
        objective=objective_function_string,
        subsample=fraction_of_examples_per_tree,
        colsample_bylevel=fraction_of_features_per_split, reg_lambda=l2_weight,
        num_class=num_classes)


def train_model(
        model_object, num_classes, training_feature_matrix,
        training_target_values, num_iters_for_early_stopping=None,
        validation_feature_matrix=None, validation_target_values=None):
    """Trains GBT model for classification.

    T = number of training examples
    V = number of validation examples
    Z = number of features (input variables)

    :param model_object: Instance of `xgboost.XGBClassifier`.  The easiest way
        to create one is to use `create_model`.
    :param num_classes: Number of target classes.
    :param training_feature_matrix: T-by-Z numpy array of features for training.
    :param training_target_values: length-T integer numpy array of target values
        for training.  If target_values[i] = k, the [i]th example (storm object)
        belongs to the [k]th class.
    :param num_iters_for_early_stopping: Number of iterations for early
        stopping.  If validation error has not improved with the last
        `num_iters_for_early_stopping` trees added, training will be stopped.
        If you don't want on-the-fly validation, leave this argument alone.
    :param validation_feature_matrix:
        [used only if `num_iters_for_early_stopping is not None`]
        V-by-Z numpy array of features for validation.
    :param validation_target_values:
        [used only if `num_iters_for_early_stopping is not None`]
        Same as `training_target_values`, but length is V rather than T.
    """

    error_checking.assert_is_numpy_array(
        training_feature_matrix, num_dimensions=2)
    num_training_examples = training_feature_matrix.shape[0]
    num_features = training_feature_matrix.shape[1]

    error_checking.assert_is_integer_numpy_array(training_target_values)
    error_checking.assert_is_numpy_array(
        training_target_values,
        exact_dimensions=numpy.array([num_training_examples]))
    error_checking.assert_is_geq_numpy_array(training_target_values, 0)

    training_target_matrix = numpy.full(
        (num_training_examples, num_classes), 0, dtype=int)
    for i in range(num_training_examples):
        training_target_matrix[i, training_target_values[i]] = 1

    if num_iters_for_early_stopping is None:
        print training_feature_matrix.shape
        print training_target_matrix.shape

        model_object.fit(
            training_feature_matrix, training_target_matrix,
            eval_metric='logloss', verbose=True)
    else:
        error_checking.assert_is_integer(num_iters_for_early_stopping)
        error_checking.assert_is_geq(num_iters_for_early_stopping, 1)

        error_checking.assert_is_numpy_array(
            validation_feature_matrix, num_dimensions=2)
        num_validation_examples = validation_feature_matrix.shape[0]
        error_checking.assert_is_numpy_array(
            validation_feature_matrix,
            exact_dimensions=numpy.array(
                [num_validation_examples, num_features]))

        error_checking.assert_is_integer_numpy_array(validation_target_values)
        error_checking.assert_is_numpy_array(
            validation_target_values,
            exact_dimensions=numpy.array([num_validation_examples]))
        error_checking.assert_is_geq_numpy_array(validation_target_values, 0)

        model_object.fit(
            training_feature_matrix, training_target_values,
            eval_metric='logloss', verbose=True,
            early_stopping_rounds=num_iters_for_early_stopping,
            eval_set=[(validation_feature_matrix, validation_target_values)])
