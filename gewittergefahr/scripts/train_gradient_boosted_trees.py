"""Trains ensemble of gradient-boosted trees (GBT).

Training data consists of features (output of the last "Flatten" layer) from a
convolutional neural network (CNN).
"""

import os.path
import argparse
import sklearn.metrics
from gewittergefahr.deep_learning import cnn
from gewittergefahr.deep_learning import gradient_boosting

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

TRAINING_FILE_ARG_NAME = 'input_training_file_name'
VALIDATION_FILE_ARG_NAME = 'input_validation_file_name'
MODEL_FILE_ARG_NAME = 'output_model_file_name'
NUM_TREES_ARG_NAME = 'num_trees'
LEARNING_RATE_ARG_NAME = 'learning_rate'
MAX_TREE_DEPTH_ARG_NAME = 'max_tree_depth'
FRACTION_OF_EXAMPLES_PER_TREE_ARG_NAME = 'fraction_of_examples_per_tree'
FRACTION_OF_FEATURES_PER_SPLIT_ARG_NAME = 'fraction_of_features_per_split'
L2_WEIGHT_ARG_NAME = 'l2_weight'
NUM_ITERS_FOR_EARLY_STOPPING_ARG_NAME = 'num_iters_for_early_stopping'

TRAINING_FILE_HELP_STRING = (
    'Path to file with training data (readable by `cnn.read_features`).')
VALIDATION_FILE_HELP_STRING = (
    'Path to file with validation data (readable by `cnn.read_features`).  In '
    'this context, "validation" means on-the-fly validation.  Specifically, '
    'validation data will be used to report validation loss after the addition '
    'of each tree, and potentially to stop training early (see documentation '
    'for `{0:s}`).  To train without validation data, leave `{1:s}` alone.'
).format(NUM_ITERS_FOR_EARLY_STOPPING_ARG_NAME, VALIDATION_FILE_ARG_NAME)
MODEL_FILE_HELP_STRING = (
    'Path to output file (will be written by `gradient_boosting.write_model`).')
NUM_TREES_HELP_STRING = (
    'Max number of trees in ensemble.  If training is stopped early (see '
    'documentation for `{0:s}`), the ensemble may end up with fewer trees.'
).format(NUM_ITERS_FOR_EARLY_STOPPING_ARG_NAME)
LEARNING_RATE_HELP_STRING = 'Learning rate.'
MAX_TREE_DEPTH_HELP_STRING = 'Max depth of any tree in the ensemble.'
FRACTION_OF_EXAMPLES_PER_TREE_HELP_STRING = (
    'Fraction of training examples (storm objects) to be used in training each '
    'tree.')
FRACTION_OF_FEATURES_PER_SPLIT_HELP_STRING = (
    'Fraction of features (predictor variables) to be used in deciding each '
    'split point.')
L2_WEIGHT_HELP_STRING = 'L2 regularization weight.'
NUM_ITERS_FOR_EARLY_STOPPING_HELP_STRING = (
    'Number of iterations for early stopping.  If validation loss has not '
    'improved with the addition of the last `{0:s}` trees, training will be '
    'stopped early (before reaching `{1:s}` trees).'
).format(NUM_ITERS_FOR_EARLY_STOPPING_ARG_NAME, NUM_TREES_ARG_NAME)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + TRAINING_FILE_ARG_NAME, type=str, required=True,
    help=TRAINING_FILE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + VALIDATION_FILE_ARG_NAME, type=str, required=False, default='None',
    help=VALIDATION_FILE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + MODEL_FILE_ARG_NAME, type=str, required=True,
    help=MODEL_FILE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + NUM_TREES_ARG_NAME, type=int, required=False,
    default=gradient_boosting.DEFAULT_NUM_TREES, help=NUM_TREES_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + LEARNING_RATE_ARG_NAME, type=float, required=False,
    default=gradient_boosting.DEFAULT_LEARNING_RATE,
    help=LEARNING_RATE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + MAX_TREE_DEPTH_ARG_NAME, type=int, required=False,
    default=gradient_boosting.DEFAULT_MAX_TREE_DEPTH,
    help=MAX_TREE_DEPTH_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + FRACTION_OF_EXAMPLES_PER_TREE_ARG_NAME, type=float, required=False,
    default=gradient_boosting.DEFAULT_FRACTION_OF_EXAMPLES_PER_TREE,
    help=FRACTION_OF_EXAMPLES_PER_TREE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + FRACTION_OF_FEATURES_PER_SPLIT_ARG_NAME, type=float, required=False,
    default=gradient_boosting.DEFAULT_FRACTION_OF_FEATURES_PER_SPLIT,
    help=FRACTION_OF_FEATURES_PER_SPLIT_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + L2_WEIGHT_ARG_NAME, type=float, required=False,
    default=gradient_boosting.DEFAULT_L2_WEIGHT, help=L2_WEIGHT_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + NUM_ITERS_FOR_EARLY_STOPPING_ARG_NAME, type=int, required=False,
    default=-1, help=NUM_ITERS_FOR_EARLY_STOPPING_HELP_STRING)


def _train_model(
        input_training_file_name, input_validation_file_name,
        output_model_file_name, num_trees, learning_rate, max_tree_depth,
        fraction_of_examples_per_tree, fraction_of_features_per_split,
        l2_weight, num_iters_for_early_stopping):
    """Trains ensemble of gradient-boosted trees.

    :param input_training_file_name: See documentation at top of file.
    :param input_validation_file_name: Same.
    :param output_model_file_name: Same.
    :param num_trees: Same.
    :param learning_rate: Same.
    :param max_tree_depth: Same.
    :param fraction_of_examples_per_tree: Same.
    :param fraction_of_features_per_split: Same.
    :param l2_weight: Same.
    :param num_iters_for_early_stopping: Same.
    """

    print 'Reading training data from: "{0:s}"...'.format(
        input_training_file_name)
    (training_feature_matrix, training_target_values, num_classes
    ) = cnn.read_features(input_training_file_name)

    if input_validation_file_name == 'None':
        input_validation_file_name = None
        num_iters_for_early_stopping = None

    if input_validation_file_name is not None:
        print 'Reading validation data from: "{0:s}"...'.format(
            input_validation_file_name)
        (validation_feature_matrix, validation_target_values, num_classes
        ) = cnn.read_features(input_validation_file_name)

    print 'Setting up model architecture...'
    model_object = gradient_boosting.create_model(
        num_classes=num_classes, num_trees=num_trees,
        learning_rate=learning_rate, max_depth=max_tree_depth,
        fraction_of_examples_per_tree=fraction_of_examples_per_tree,
        fraction_of_features_per_split=fraction_of_features_per_split,
        l2_weight=l2_weight)

    model_directory_name, _ = os.path.split(output_model_file_name)
    metadata_file_name = '{0:s}/model_metadata.p'.format(model_directory_name)

    print 'Writing metadata to: "{0:s}"...'.format(metadata_file_name)
    gradient_boosting.write_model_metadata(
        num_classes=num_classes, num_trees=num_trees,
        learning_rate=learning_rate, max_depth=max_tree_depth,
        fraction_of_examples_per_tree=fraction_of_examples_per_tree,
        fraction_of_features_per_split=fraction_of_features_per_split,
        l2_weight=l2_weight,
        num_iters_for_early_stopping=num_iters_for_early_stopping,
        training_file_name=input_training_file_name,
        validation_file_name=input_validation_file_name,
        pickle_file_name=metadata_file_name)
    print SEPARATOR_STRING

    if input_validation_file_name is None:
        gradient_boosting.train_model(
            model_object=model_object,
            training_feature_matrix=training_feature_matrix,
            training_target_values=training_target_values)
    else:
        gradient_boosting.train_model(
            model_object=model_object,
            training_feature_matrix=training_feature_matrix,
            training_target_values=training_target_values,
            num_iters_for_early_stopping=num_iters_for_early_stopping,
            validation_feature_matrix=validation_feature_matrix,
            validation_target_values=validation_target_values)

    print SEPARATOR_STRING

    if input_validation_file_name is not None and num_classes == 2:
        print 'Applying model to validation examples...'
        validation_forecast_probs = gradient_boosting.apply_model(
            model_object=model_object, feature_matrix=validation_feature_matrix
        )[:, 1]
        validation_auc = sklearn.metrics.roc_auc_score(
            y_true=validation_target_values, y_score=validation_forecast_probs)

        print 'Validation AUC: {0:.4f}'.format(validation_auc)

    print 'Writing trained model to: "{0:s}"...'.format(output_model_file_name)
    gradient_boosting.write_model(
        model_object=model_object, pickle_file_name=output_model_file_name)


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _train_model(
        input_training_file_name=getattr(
            INPUT_ARG_OBJECT, TRAINING_FILE_ARG_NAME),
        input_validation_file_name=getattr(
            INPUT_ARG_OBJECT, VALIDATION_FILE_ARG_NAME),
        output_model_file_name=getattr(INPUT_ARG_OBJECT, MODEL_FILE_ARG_NAME),
        num_trees=getattr(INPUT_ARG_OBJECT, NUM_TREES_ARG_NAME),
        learning_rate=getattr(INPUT_ARG_OBJECT, LEARNING_RATE_ARG_NAME),
        max_tree_depth=getattr(INPUT_ARG_OBJECT, MAX_TREE_DEPTH_ARG_NAME),
        fraction_of_examples_per_tree=getattr(
            INPUT_ARG_OBJECT, FRACTION_OF_EXAMPLES_PER_TREE_ARG_NAME),
        fraction_of_features_per_split=getattr(
            INPUT_ARG_OBJECT, FRACTION_OF_FEATURES_PER_SPLIT_ARG_NAME),
        l2_weight=getattr(INPUT_ARG_OBJECT, L2_WEIGHT_ARG_NAME),
        num_iters_for_early_stopping=getattr(
            INPUT_ARG_OBJECT, NUM_ITERS_FOR_EARLY_STOPPING_ARG_NAME))
