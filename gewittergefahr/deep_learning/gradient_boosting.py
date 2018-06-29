"""Methods for creating, training, and applying gradient-boosted trees (GBT).

The input data for GBT are features (outputs of the last "Flatten" layer)
created by a convolutional neural network (CNN).
"""

import pickle
import numpy
import xgboost
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.gg_utils import error_checking

DEFAULT_NUM_TREES = 100
DEFAULT_LEARNING_RATE = 0.1
DEFAULT_MAX_TREE_DEPTH = 3
DEFAULT_FRACTION_OF_EXAMPLES_PER_TREE = 1. - numpy.exp(-1.)
DEFAULT_FRACTION_OF_FEATURES_PER_SPLIT = 1.
DEFAULT_L2_WEIGHT = 0.001

NUM_CLASSES_KEY = 'num_classes'
NUM_TREES_KEY = 'num_trees'
LEARNING_RATE_KEY = 'learning_rate'
MAX_DEPTH_KEY = 'max_depth'
FRACTION_OF_EXAMPLES_PER_TREE_KEY = 'fraction_of_examples_per_tree'
FRACTION_OF_FEATURES_PER_SPLIT_KEY = 'fraction_of_features_per_split'
L2_WEIGHT_KEY = 'l2_weight'
NUM_ITERS_FOR_EARLY_STOPPING_KEY = 'num_iters_for_early_stopping'
TRAINING_FILE_KEY = 'training_file_name'
VALIDATION_FILE_KEY = 'validation_file_name'

MODEL_METADATA_KEYS = [
    NUM_CLASSES_KEY, NUM_TREES_KEY, LEARNING_RATE_KEY, MAX_DEPTH_KEY,
    FRACTION_OF_EXAMPLES_PER_TREE_KEY, FRACTION_OF_FEATURES_PER_SPLIT_KEY,
    L2_WEIGHT_KEY, NUM_ITERS_FOR_EARLY_STOPPING_KEY, TRAINING_FILE_KEY,
    VALIDATION_FILE_KEY
]


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
        return xgboost.XGBClassifier(
            max_depth=max_depth, learning_rate=learning_rate,
            n_estimators=num_trees, silent=False, objective='binary:logistic',
            subsample=fraction_of_examples_per_tree,
            colsample_bylevel=fraction_of_features_per_split,
            reg_lambda=l2_weight)

    return xgboost.XGBClassifier(
        max_depth=max_depth, learning_rate=learning_rate,
        n_estimators=num_trees, silent=False, objective='multi:softprob',
        subsample=fraction_of_examples_per_tree,
        colsample_bylevel=fraction_of_features_per_split, reg_lambda=l2_weight,
        num_class=num_classes)


def train_model(
        model_object, training_feature_matrix, training_target_values,
        num_iters_for_early_stopping=None, validation_feature_matrix=None,
        validation_target_values=None):
    """Trains GBT model for classification.

    T = number of training examples
    V = number of validation examples
    Z = number of features (input variables)

    :param model_object: Instance of `xgboost.XGBClassifier`.  The easiest way
        to create one is to use `create_model`.
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

    if num_iters_for_early_stopping is None:
        model_object.fit(
            training_feature_matrix, training_target_values,
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


def apply_model(model_object, feature_matrix):
    """Applies trained GBT model to new examples.

    E = number of examples (storm objects)
    Z = number of features (input variables)

    :param model_object: Trained instance of `xgboost.XGBClassifier`.
    :param feature_matrix: E-by-Z numpy array of features.
    :return: class_probability_matrix: E-by-K numpy array of class
        probabilities.  class_probability_matrix[i, k] is the forecast
        probability that the [i]th storm object belongs to the [k]th class.
        Classes are mutually exclusive and collectively exhaustive, so the sum
        across each row is 1.
    """

    return model_object.predict_proba(feature_matrix)


def write_model(model_object, pickle_file_name):
    """Writes model to Pickle file.

    :param model_object: Instance (preferably trained) of
        `xgboost.XGBClassifier`.
    :param pickle_file_name: Path to output file.
    """

    file_system_utils.mkdir_recursive_if_necessary(file_name=pickle_file_name)
    pickle_file_handle = open(pickle_file_name, 'wb')
    pickle.dump(model_object, pickle_file_handle)
    pickle_file_handle.close()


def read_model(pickle_file_name):
    """Reads model from Pickle file.

    :param pickle_file_name: Path to input file.
    :return: model_object: Instance of `xgboost.XGBClassifier`.
    """

    pickle_file_handle = open(pickle_file_name, 'rb')
    model_object = pickle.load(pickle_file_handle)
    pickle_file_handle.close()

    return model_object


def write_model_metadata(
        num_classes, num_trees, learning_rate, max_depth,
        fraction_of_examples_per_tree, fraction_of_features_per_split,
        l2_weight, num_iters_for_early_stopping, training_file_name,
        validation_file_name, pickle_file_name):
    """Writes metadata for GBT model to Pickle file.

    :param num_classes: See documentation for `create_model`.
    :param num_trees: Same.
    :param learning_rate: Same.
    :param max_depth: Same.
    :param fraction_of_examples_per_tree: Same.
    :param fraction_of_features_per_split: Same.
    :param l2_weight: Same.
    :param num_iters_for_early_stopping: [may be None]
        See documentation for `train_model`.
    :param training_file_name: Path to file with training data (readable by
        `cnn.read_features`).
    :param validation_file_name: [may be None]
        Path to file with validation data (readable by `cnn.read_features`).
    :param pickle_file_name: Path to output file.
    """

    model_metadata_dict = {
        NUM_CLASSES_KEY: num_classes,
        NUM_TREES_KEY: num_trees,
        LEARNING_RATE_KEY: learning_rate,
        MAX_DEPTH_KEY: max_depth,
        FRACTION_OF_EXAMPLES_PER_TREE_KEY: fraction_of_examples_per_tree,
        FRACTION_OF_FEATURES_PER_SPLIT_KEY: fraction_of_features_per_split,
        L2_WEIGHT_KEY: l2_weight,
        NUM_ITERS_FOR_EARLY_STOPPING_KEY: num_iters_for_early_stopping,
        TRAINING_FILE_KEY: training_file_name,
        VALIDATION_FILE_KEY: validation_file_name
    }

    file_system_utils.mkdir_recursive_if_necessary(file_name=pickle_file_name)
    pickle_file_handle = open(pickle_file_name, 'wb')
    pickle.dump(model_metadata_dict, pickle_file_handle)
    pickle_file_handle.close()


def read_model_metadata(pickle_file_name):
    """Reads model metadata from Pickle file.

    :param pickle_file_name: Path to input file.
    :return: model_metadata_dict: Dictionary with all keys in the list
        `MODEL_METADATA_KEYS`.
    :raises: ValueError: if dictionary does not contain all keys in the list
        `MODEL_METADATA_KEYS`.
    """

    pickle_file_handle = open(pickle_file_name, 'rb')
    model_metadata_dict = pickle.load(pickle_file_handle)
    pickle_file_handle.close()

    expected_keys_as_set = set(MODEL_METADATA_KEYS)
    actual_keys_as_set = set(model_metadata_dict.keys())
    if not set(expected_keys_as_set).issubset(actual_keys_as_set):
        error_string = (
            '\n\n{0:s}\nExpected keys are listed above.  Keys found in file '
            '("{1:s}") are listed below.  Some expected keys were not found.'
            '\n{2:s}\n'
        ).format(MODEL_METADATA_KEYS, pickle_file_name,
                 model_metadata_dict.keys())

        raise ValueError(error_string)

    return model_metadata_dict
