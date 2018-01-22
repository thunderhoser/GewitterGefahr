"""Examples of feature selection using GewitterGefahr.

This is very similar to the answer key for Assignment 6 of CS 4033/5033 (Machine
Learning), Fall 2017, University of Oklahoma.  The assignment questions are in a
PDF in the same directory as this module.
"""

import os.path
import numpy
import pandas
import matplotlib.pyplot as pyplot
import sklearn.neural_network
from gewittergefahr.gg_utils import feature_selection

DOTS_PER_INCH = 600
SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

DEFAULT_TRAINING_FRACTION = 0.5
DEFAULT_VALIDATION_FRACTION = 0.3
NUM_FEATURES_TO_ADD_PER_STEP = 2
NUM_FEATURES_TO_REMOVE_PER_STEP = 2

HADRON_STRING_LABEL = 'h'
GAMMA_PARTICLE_STRING_LABEL = 'g'
HADRON_LABEL = 0
GAMMA_PARTICLE_LABEL = 1

MAJOR_AXIS_COLUMN = 'fLength'
MINOR_AXIS_COLUMN = 'fWidth'
LOG10_SUM_PIXELS_COLUMN = 'fSize'
FRACTION_2HIGHEST_PIXELS_COLUMN = 'fConc'
FRACTION_HIGHEST_PIXEL_COLUMN = 'fConcl'
HIGHEST_PIXEL_DISTANCE_COLUMN = 'fAsym'
ROOT3_MOMENT3_MAJOR_AXIS_COLUMN = 'fM3Long'
ROOT3_MOMENT3_MINOR_AXIS_COLUMN = 'fM3Trans'
MAJOR_AXIS_ANGLE_COLUMN = 'fAlpha'
ORIGIN_TO_CENTER_DISTANCE_COLUMN = 'fDist'
LABEL_COLUMN = 'class'

FEATURE_COLUMNS = [
    MAJOR_AXIS_COLUMN, MINOR_AXIS_COLUMN, LOG10_SUM_PIXELS_COLUMN,
    FRACTION_2HIGHEST_PIXELS_COLUMN, FRACTION_HIGHEST_PIXEL_COLUMN,
    HIGHEST_PIXEL_DISTANCE_COLUMN, ROOT3_MOMENT3_MAJOR_AXIS_COLUMN,
    ROOT3_MOMENT3_MINOR_AXIS_COLUMN, MAJOR_AXIS_ANGLE_COLUMN,
    ORIGIN_TO_CENTER_DISTANCE_COLUMN]
INPUT_COLUMNS = FEATURE_COLUMNS + [LABEL_COLUMN]

INPUT_COLUMN_TYPE_DICT = {}
for this_column in FEATURE_COLUMNS:
    INPUT_COLUMN_TYPE_DICT.update({this_column: float})
INPUT_COLUMN_TYPE_DICT.update({LABEL_COLUMN: str})

WORKING_DIRECTORY_NAME = os.path.dirname(__file__)
DEFAULT_INPUT_FILE_NAME = os.path.join(
    WORKING_DIRECTORY_NAME, 'feature_selection_data.csv')


def _labels_from_string_to_integer(string_labels):
    """Converts labels from string to integer format.

    N = number of examples

    :param string_labels: length-N list of string labels.
    :return: integer_labels: length-N numpy array of integer labels.
    """

    string_labels_as_numpy_array = numpy.asarray(string_labels)
    gamma_particle_indices = numpy.where(
        string_labels_as_numpy_array == GAMMA_PARTICLE_STRING_LABEL)[0]

    integer_labels = numpy.full(len(string_labels), HADRON_LABEL, dtype=int)
    integer_labels[gamma_particle_indices] = GAMMA_PARTICLE_LABEL
    return integer_labels


def _read_input_data(csv_file_name=DEFAULT_INPUT_FILE_NAME):
    """Reads input data (features and labels).

    N = number of examples

    :param csv_file_name: Path to input file.
    :return: input_table: N-row pandas DataFrame, with all columns in the list
        `INPUT_COLUMNS`.
    """

    input_table = pandas.read_csv(
        csv_file_name, header=0, usecols=INPUT_COLUMNS,
        dtype=INPUT_COLUMN_TYPE_DICT)

    observed_labels = _labels_from_string_to_integer(
        input_table[LABEL_COLUMN].values)
    input_table.drop(LABEL_COLUMN, axis=1, inplace=True)
    return input_table.assign(**{LABEL_COLUMN: observed_labels})


def _split_training_validation_testing(
        input_table, training_fraction=DEFAULT_TRAINING_FRACTION,
        validation_fraction=DEFAULT_VALIDATION_FRACTION):
    """Splits examples into training, validation, and testing.

    This method assumes that all examples are mutually independent, which means
    that random splitting ensures independence among the 3 sets.

    :param input_table: pandas DataFrame created by _read_input_data.
    :param training_fraction: Fraction of examples to use for training.  Must be
        in (0, 1).
    :param validation_fraction: Fraction of examples to use for validation.
        Must be in (0, 1).  Also, training_fraction + validation_fraction must
        be in (0, 1).
    :return: training_table: Same as input_table, except with a subset of rows
        (training examples only).
    :return: validation_table: Same as above, but for validation examples.
    :return: testing_table: Same as above, but for testing examples.
    """

    num_examples = len(input_table.index)
    example_indices = numpy.linspace(
        0, num_examples - 1, num=num_examples, dtype=int)

    num_training_examples = int(numpy.round(training_fraction * num_examples))
    num_validation_examples = int(numpy.round(
        validation_fraction * num_examples))

    training_indices = numpy.random.choice(
        example_indices, size=num_training_examples, replace=False)
    training_flags = numpy.full(num_examples, False, dtype=bool)
    training_flags[training_indices] = True
    non_training_indices = numpy.where(numpy.invert(training_flags))[0]

    validation_indices = numpy.random.choice(
        non_training_indices, size=num_validation_examples, replace=False)
    testing_flags = numpy.full(num_examples, True, dtype=bool)
    testing_flags[training_indices] = False
    testing_flags[validation_indices] = False
    testing_indices = numpy.where(testing_flags)[0]

    return (input_table.iloc[training_indices],
            input_table.iloc[validation_indices],
            input_table.iloc[testing_indices])


def _print_forward_selection_results(forward_selection_dict):
    """Prints results of any forward-selection algorithm to command window.

    :param forward_selection_dict: Dictionary returned by
        `feature_selection.sequential_forward_selection`,
        `feature_selection.sfs_with_backward_steps`, or
        `feature_selection.floating_sfs`.
    """

    print ('\nFeatures were selected in the following order: ' + str(
        forward_selection_dict[feature_selection.SELECTED_FEATURES_KEY]))

    num_selected_features = len(
        forward_selection_dict[feature_selection.SELECTED_FEATURES_KEY])

    print ('Validation AUC and cross-entropy of {0:d}-feature model: {1:.3f}, '
           '{2:.4f}').format(
               num_selected_features,
               forward_selection_dict[feature_selection.VALIDATION_AUC_KEY],
               forward_selection_dict[
                   feature_selection.VALIDATION_XENTROPY_KEY])
    print ('Testing AUC and cross-entropy of {0:d}-feature model: {1:.3f}, '
           '{2:.4f}').format(
               num_selected_features,
               forward_selection_dict[feature_selection.TESTING_AUC_KEY],
               forward_selection_dict[feature_selection.TESTING_XENTROPY_KEY])


def _print_backward_selection_results(backward_selection_dict):
    """Prints results of any backward-selection algorithm to command window.

    :param backward_selection_dict: Dictionary returned by
        `feature_selection.sequential_backward_selection`,
        `feature_selection.sbs_with_forward_steps`, or
        `feature_selection.floating_sbs`.
    """

    print ('\nFeatures were removed in the following order: ' + str(
        backward_selection_dict[feature_selection.REMOVED_FEATURES_KEY]))
    print ('The following features remain (were selected): ' + str(
        backward_selection_dict[feature_selection.SELECTED_FEATURES_KEY]))

    num_selected_features = len(
        backward_selection_dict[feature_selection.SELECTED_FEATURES_KEY])

    print ('Validation AUC and cross-entropy of {0:d}-feature model: {1:.3f}, '
           '{2:.4f}').format(
               num_selected_features,
               backward_selection_dict[feature_selection.VALIDATION_AUC_KEY],
               backward_selection_dict[
                   feature_selection.VALIDATION_XENTROPY_KEY])
    print ('Testing AUC and cross-entropy of {0:d}-feature model: {1:.3f}, '
           '{2:.4f}').format(
               num_selected_features,
               backward_selection_dict[feature_selection.TESTING_AUC_KEY],
               backward_selection_dict[feature_selection.TESTING_XENTROPY_KEY])


if __name__ == '__main__':
    INPUT_TABLE = _read_input_data()
    TRAINING_TABLE, VALIDATION_TABLE, TESTING_TABLE = (
        _split_training_validation_testing(INPUT_TABLE))

    # Sequential forward selection.
    SFS_DICTIONARY = feature_selection.sequential_forward_selection(
        training_table=TRAINING_TABLE, validation_table=VALIDATION_TABLE,
        testing_table=TESTING_TABLE, feature_names=FEATURE_COLUMNS,
        target_name=LABEL_COLUMN,
        estimator_object=sklearn.neural_network.MLPClassifier(),
        num_features_to_add_per_step=NUM_FEATURES_TO_ADD_PER_STEP)
    _print_forward_selection_results(SFS_DICTIONARY)

    SFS_IMAGE_FILE_NAME = '{0:s}/sfs_results.jpg'.format(WORKING_DIRECTORY_NAME)
    print 'Saving bar graph to "{0:s}"...{1:s}'.format(
        SFS_IMAGE_FILE_NAME, SEPARATOR_STRING)

    feature_selection.plot_forward_selection_results(
        SFS_DICTIONARY, plot_feature_names=True)
    pyplot.savefig(SFS_IMAGE_FILE_NAME, dpi=DOTS_PER_INCH)
    pyplot.close()

    # Sequential forward selection with backward steps.
    SFS_WITH_BACKWARD_STEPS_DICT = feature_selection.sfs_with_backward_steps(
        training_table=TRAINING_TABLE, validation_table=VALIDATION_TABLE,
        testing_table=TESTING_TABLE, feature_names=FEATURE_COLUMNS,
        target_name=LABEL_COLUMN,
        estimator_object=sklearn.neural_network.MLPClassifier(),
        num_features_to_add_per_forward_step=NUM_FEATURES_TO_ADD_PER_STEP,
        num_features_to_remove_per_backward_step=
        NUM_FEATURES_TO_REMOVE_PER_STEP)
    _print_forward_selection_results(SFS_WITH_BACKWARD_STEPS_DICT)

    SFS_BACKWARD_IMAGE_FILE_NAME = (
        '{0:s}/sfs_with_backward_steps_results.jpg'.format(
            WORKING_DIRECTORY_NAME))
    print 'Saving bar graph to "{0:s}"...{1:s}'.format(
        SFS_BACKWARD_IMAGE_FILE_NAME, SEPARATOR_STRING)

    feature_selection.plot_forward_selection_results(
        SFS_WITH_BACKWARD_STEPS_DICT, plot_feature_names=True)
    pyplot.savefig(SFS_BACKWARD_IMAGE_FILE_NAME, dpi=DOTS_PER_INCH)
    pyplot.close()

    # Sequential forward floating selection.
    SFFS_DICTIONARY = feature_selection.floating_sfs(
        training_table=TRAINING_TABLE, validation_table=VALIDATION_TABLE,
        testing_table=TESTING_TABLE, feature_names=FEATURE_COLUMNS,
        target_name=LABEL_COLUMN,
        estimator_object=sklearn.neural_network.MLPClassifier(),
        num_features_to_add_per_step=NUM_FEATURES_TO_ADD_PER_STEP)
    _print_forward_selection_results(SFFS_DICTIONARY)

    SFFS_IMAGE_FILE_NAME = '{0:s}/sffs_results.jpg'.format(
        WORKING_DIRECTORY_NAME)
    print 'Saving bar graph to "{0:s}"...{1:s}'.format(
        SFFS_IMAGE_FILE_NAME, SEPARATOR_STRING)

    feature_selection.plot_forward_selection_results(
        SFFS_DICTIONARY, plot_feature_names=True)
    pyplot.savefig(SFFS_IMAGE_FILE_NAME, dpi=DOTS_PER_INCH)
    pyplot.close()

    # Sequential backward selection.
    SBS_DICTIONARY = feature_selection.sequential_backward_selection(
        training_table=TRAINING_TABLE, validation_table=VALIDATION_TABLE,
        testing_table=TESTING_TABLE, feature_names=FEATURE_COLUMNS,
        target_name=LABEL_COLUMN,
        estimator_object=sklearn.neural_network.MLPClassifier(),
        num_features_to_remove_per_step=NUM_FEATURES_TO_REMOVE_PER_STEP)
    _print_backward_selection_results(SBS_DICTIONARY)

    SBS_IMAGE_FILE_NAME = '{0:s}/sbs_results.jpg'.format(WORKING_DIRECTORY_NAME)
    print 'Saving bar graph to "{0:s}"...{1:s}'.format(
        SBS_IMAGE_FILE_NAME, SEPARATOR_STRING)

    feature_selection.plot_backward_selection_results(
        SBS_DICTIONARY, plot_feature_names=True)
    pyplot.savefig(SBS_IMAGE_FILE_NAME, dpi=DOTS_PER_INCH)
    pyplot.close()

    # Sequential backward selection with forward steps.
    SBS_WITH_FORWARD_STEPS_DICT = feature_selection.sbs_with_forward_steps(
        training_table=TRAINING_TABLE, validation_table=VALIDATION_TABLE,
        testing_table=TESTING_TABLE, feature_names=FEATURE_COLUMNS,
        target_name=LABEL_COLUMN,
        estimator_object=sklearn.neural_network.MLPClassifier(),
        num_features_to_add_per_forward_step=NUM_FEATURES_TO_ADD_PER_STEP,
        num_features_to_remove_per_backward_step=
        NUM_FEATURES_TO_REMOVE_PER_STEP)
    _print_backward_selection_results(SBS_WITH_FORWARD_STEPS_DICT)

    SBS_FORWARD_IMAGE_FILE_NAME = (
        '{0:s}/sbs_with_forward_steps_results.jpg'.format(
            WORKING_DIRECTORY_NAME))
    print 'Saving bar graph to "{0:s}"...{1:s}'.format(
        SBS_FORWARD_IMAGE_FILE_NAME, SEPARATOR_STRING)

    feature_selection.plot_backward_selection_results(
        SBS_WITH_FORWARD_STEPS_DICT, plot_feature_names=True)
    pyplot.savefig(SBS_FORWARD_IMAGE_FILE_NAME, dpi=DOTS_PER_INCH)
    pyplot.close()

    # Sequential backward floating selection.
    SBFS_DICTIONARY = feature_selection.floating_sbs(
        training_table=TRAINING_TABLE, validation_table=VALIDATION_TABLE,
        testing_table=TESTING_TABLE, feature_names=FEATURE_COLUMNS,
        target_name=LABEL_COLUMN,
        estimator_object=sklearn.neural_network.MLPClassifier(),
        num_features_to_remove_per_step=NUM_FEATURES_TO_REMOVE_PER_STEP)
    _print_backward_selection_results(SBFS_DICTIONARY)

    SBFS_IMAGE_FILE_NAME = '{0:s}/sbfs_results.jpg'.format(
        WORKING_DIRECTORY_NAME)
    print 'Saving bar graph to "{0:s}"...{1:s}'.format(
        SBFS_IMAGE_FILE_NAME, SEPARATOR_STRING)

    feature_selection.plot_backward_selection_results(
        SBFS_DICTIONARY, plot_feature_names=True)
    pyplot.savefig(SBFS_IMAGE_FILE_NAME, dpi=DOTS_PER_INCH)
    pyplot.close()

    # Permutation selection (actually importance-ranking, because permutation
    # does not explicitly select variables).
    (PERMUTATION_TABLE,
     ORIG_VALIDATION_COST,
     ORIG_VALIDATION_CROSS_ENTROPY,
     ORIG_VALIDATION_AUC) = feature_selection.permutation_selection(
         training_table=TRAINING_TABLE, validation_table=VALIDATION_TABLE,
         feature_names=FEATURE_COLUMNS, target_name=LABEL_COLUMN,
         estimator_object=sklearn.neural_network.MLPClassifier())

    print '\nFeatures were permuted in the following order:'
    print (
        'No permutation: validation AUC = {0:.3f}, '
        'cross-entropy = {1:.4f}').format(
            ORIG_VALIDATION_AUC, ORIG_VALIDATION_CROSS_ENTROPY)

    NUM_FEATURES = len(PERMUTATION_TABLE.index)
    for f in range(NUM_FEATURES):
        print (
            'Permutation #{0:d} ({1:s}): new validation AUC = {2:.3f}, '
            'cross-entropy = {3:.4f}').format(
                f + 1,
                PERMUTATION_TABLE[feature_selection.FEATURE_NAME_KEY].values[f],
                PERMUTATION_TABLE[
                    feature_selection.VALIDATION_AUC_KEY].values[f],
                PERMUTATION_TABLE[
                    feature_selection.VALIDATION_XENTROPY_KEY].values[f])

    PERMUTATION_IMAGE_FILE_NAME = '{0:s}/permutation_results.jpg'.format(
        WORKING_DIRECTORY_NAME)
    print '\nSaving bar graph to "{0:s}"...{1:s}'.format(
        PERMUTATION_IMAGE_FILE_NAME, SEPARATOR_STRING)

    feature_selection.plot_permutation_results(
        PERMUTATION_TABLE, plot_feature_names=True,
        orig_validation_cost=ORIG_VALIDATION_COST)
    pyplot.savefig(PERMUTATION_IMAGE_FILE_NAME, dpi=DOTS_PER_INCH)
    pyplot.close()
