"""Methods for running the permutation test.

The permutation test (or "permutation-based variable importance") is explained
in Lakshmanan et al. (2015).

--- REFERENCES ---

Lakshmanan, V., C. Karstens, J. Krause, K. Elmore, A. Ryzhkov, and S. Berkseth,
    2015: "Which polarimetric variables are important for weather/no-weather
    discrimination?" Journal of Atmospheric and Oceanic Technology, 32 (6),
    1209-1223.
"""

import copy
import pickle
import numpy
import keras.utils
from sklearn.metrics import roc_auc_score as sklearn_auc
from gewittergefahr.gg_io import storm_tracking_io as tracking_io
from gewittergefahr.gg_utils import bootstrapping
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.gg_utils import error_checking
from gewittergefahr.deep_learning import cnn

DEFAULT_NUM_BOOTSTRAP_ITERS = 1
DEFAULT_CONFIDENCE_LEVEL = 0.95

MIN_PROBABILITY = 1e-15
MAX_PROBABILITY = 1. - MIN_PROBABILITY

# Mandatory keys in result dictionary (see `write_results`).
NUM_BOOTSTRAP_ITERS_KEY = 'num_bootstrap_iters'
CONFIDENCE_LEVEL_KEY = 'bootstrap_confidence_level'
SELECTED_PREDICTORS_KEY = 'selected_predictor_name_by_step'
HIGHEST_COSTS_KEY = 'highest_cost_by_step_bs_matrix'
ORIGINAL_COST_KEY = 'original_cost_bs_array'
STEP1_PREDICTORS_KEY = 'step1_predictor_names'
STEP1_COSTS_KEY = 'step1_cost_bs_matrix'

REQUIRED_KEYS = [
    NUM_BOOTSTRAP_ITERS_KEY, CONFIDENCE_LEVEL_KEY, SELECTED_PREDICTORS_KEY,
    HIGHEST_COSTS_KEY, ORIGINAL_COST_KEY, STEP1_PREDICTORS_KEY, STEP1_COSTS_KEY
]

# Optional keys in result dictionary (see `write_results`).
STORM_IDS_KEY = tracking_io.STORM_IDS_KEY + ''
STORM_TIMES_KEY = tracking_io.STORM_TIMES_KEY + ''
MODEL_FILE_KEY = 'model_file_name'
TARGET_VALUES_KEY = 'target_values'


def prediction_function_2d_cnn(model_object, list_of_input_matrices):
    """Prediction function for 2-D GewitterGefahr CNN.

    E = number of examples
    K = number of target classes

    :param model_object: See doc for `run_permutation_test`.
    :param list_of_input_matrices: Same.
    :return: class_probability_matrix: E-by-K numpy array, where
        class_probability_matrix[i, k] is the probability that the [i]th example
        belongs to the [k]th class.
    """

    if len(list_of_input_matrices) == 2:
        sounding_matrix = list_of_input_matrices[1]
    else:
        sounding_matrix = None

    return cnn.apply_2d_or_3d_cnn(
        model_object=model_object, radar_image_matrix=list_of_input_matrices[0],
        sounding_matrix=sounding_matrix)


def prediction_function_3d_cnn(model_object, list_of_input_matrices):
    """Prediction function for 3-D GewitterGefahr CNN.

    E = number of examples
    K = number of target classes

    :param model_object: See doc for `run_permutation_test`.
    :param list_of_input_matrices: Same.
    :return: class_probability_matrix: E-by-K numpy array, where
        class_probability_matrix[i, k] is the probability that the [i]th example
        belongs to the [k]th class.
    """

    if len(list_of_input_matrices) == 2:
        sounding_matrix = list_of_input_matrices[1]
    else:
        sounding_matrix = None

    return cnn.apply_2d_or_3d_cnn(
        model_object=model_object, radar_image_matrix=list_of_input_matrices[0],
        sounding_matrix=sounding_matrix)


def prediction_function_2d3d_cnn(model_object, list_of_input_matrices):
    """Prediction function for hybrid 2D/3D GewitterGefahr CNN.

    E = number of examples
    K = number of target classes

    :param model_object: See doc for `run_permutation_test`.
    :param list_of_input_matrices: Same.
    :return: class_probability_matrix: E-by-K numpy array, where
        class_probability_matrix[i, k] is the probability that the [i]th example
        belongs to the [k]th class.
    """

    if len(list_of_input_matrices) == 3:
        sounding_matrix = list_of_input_matrices[2]
    else:
        sounding_matrix = None

    return cnn.apply_2d3d_cnn(
        model_object=model_object,
        reflectivity_matrix_dbz=list_of_input_matrices[0],
        azimuthal_shear_matrix_s01=list_of_input_matrices[1],
        sounding_matrix=sounding_matrix)


def cross_entropy_function(
        target_values, class_probability_matrix, test_mode=False):
    """Cross-entropy cost function.

    This function works for binary or multi-class classification.

    :param target_values: See doc for `run_permutation_test`.
    :param class_probability_matrix: Same.
    :param test_mode: Never mind.  Leave this alone.
    :return: cross_entropy: Scalar.
    """

    error_checking.assert_is_boolean(test_mode)

    num_examples = class_probability_matrix.shape[0]
    num_classes = class_probability_matrix.shape[1]

    class_probability_matrix[
        class_probability_matrix < MIN_PROBABILITY
    ] = MIN_PROBABILITY
    class_probability_matrix[
        class_probability_matrix > MAX_PROBABILITY
    ] = MAX_PROBABILITY

    target_matrix = keras.utils.to_categorical(
        target_values, num_classes
    ).astype(int)

    if test_mode:
        return -1 * numpy.sum(
            target_matrix * numpy.log(class_probability_matrix)
        ) / num_examples

    return -1 * numpy.sum(
        target_matrix * numpy.log2(class_probability_matrix)
    ) / num_examples


def negative_auc_function(target_values, class_probability_matrix):
    """Computes negative AUC (area under ROC curve).

    This function works only for binary classification!

    :param target_values: See doc for `run_permutation_test`.
    :param class_probability_matrix: Same.
    :return: negative_auc: Negative AUC.
    :raises: TypeError: if `class_probability_matrix` contains more than 2
        classes.
    """

    num_classes = class_probability_matrix.shape[-1]

    if num_classes != 2:
        error_string = (
            'This function works only for binary classification, not '
            '{0:d}-class classification.'
        ).format(num_classes)

        raise TypeError(error_string)

    return -1 * sklearn_auc(
        y_true=target_values, y_score=class_probability_matrix[:, -1]
    )


def run_permutation_test(
        model_object, list_of_input_matrices, predictor_names_by_matrix,
        target_values, prediction_function, cost_function,
        num_bootstrap_iters=DEFAULT_NUM_BOOTSTRAP_ITERS,
        bootstrap_confidence_level=DEFAULT_CONFIDENCE_LEVEL):
    """Runs the permutation test.

    N = number of input matrices
    E = number of examples
    C_j = number of channels (predictors) in the [j]th matrix
    K = number of target classes

    :param model_object: Trained instance of `keras.models.Model` or
        `keras.models.Sequential`.
    :param list_of_input_matrices: length-N list of matrices (numpy arrays), in
        the order that they were fed to the model for training.  In other words,
        if the order of training matrices was [radar images, soundings], the
        order of these matrices must be [radar images, soundings].  The first
        axis of each matrix should have length E, and the last axis of the [j]th
        matrix should have length C_j.
    :param predictor_names_by_matrix: length-N list of lists.  The [j]th list
        should be a list of predictor variables in the [j]th matrix, with length
        C_j.
    :param target_values: length-E numpy array of target values (integer class
        labels).
    :param prediction_function: Function used to generate predictions from the
        model.  Should have the following inputs and outputs.
    Input: model_object: Same as input to this method.
    Input: list_of_input_matrices: Same as input to this method, except maybe
        with permuted values.
    Output: class_probability_matrix: E-by-K numpy array, where
        class_probability_matrix[i, k] is the probability that the [i]th example
        belongs to the [k]th class.

    :param cost_function: Function used to evaluate predictions from the model.
        Should have the following inputs and outputs.  This method will assume
        that lower values are better.  In other words, the cost function must be
        negatively oriented.
    Input: target_values: Same as input to this method.
    Input: class_probability_matrix: Output from `prediction_function`.
    Output: cost: Scalar value.

    :param num_bootstrap_iters: Number of bootstrapping iterations (used to
        compute the cost function after each permutation).  If
        `num_bootstrap_iters <= 1`, bootstrapping will not be used.
    :param bootstrap_confidence_level: Confidence level for bootstrapping.  This
        method will return the q-percent confidence interval for each cost,
        where q = 100 * `bootstrap_confidence_level`.

    :return: result_dict: Dictionary with the following keys.
        S = number of steps (loops through predictors) taken by algorithm
        P = number of predictors

    result_dict['num_bootstrap_iters']: See input.
    result_dict['bootstrap_confidence_level']: See input.
    result_dict['selected_predictor_name_by_step']: length-S list with names of
        selected predictors.
    result_dict['highest_cost_by_step_bs_matrix']: S-by-3 numpy array, where
        highest_cost_by_step_bs_matrix[i, 0] = minimum of confidence interval
        for highest cost at [i]th step; highest_cost_by_step_bs_matrix[i, 1] =
        mean; and highest_cost_by_step_bs_matrix[i, 2] = max.
    result_dict['original_cost_bs_array']: length-3 numpy array, where
        original_cost_bs_array[0] = minimum of confidence interval for cost
        without permutation; original_cost_bs_array[1] = mean; and
        original_cost_bs_array[2] = max.
    result_dict['step1_predictor_names']: length-P list of predictor names.
    result_dict['step1_cost_bs_matrix']: S-by-3 numpy array, where
        step1_cost_bs_matrix[i, 0] = minimum of confidence interval for cost
        after permuting only step1_predictor_names[i];
        step1_cost_bs_matrix[i, 1] = mean; and step1_cost_bs_matrix[i, 2] = max.

    :raises: ValueError: if length of `list_of_input_matrices` != length of
        `predictor_names_by_matrix`.
    :raises: ValueError: if any input matrix has < 3 dimensions.
    """

    # Check input args.
    error_checking.assert_is_integer_numpy_array(target_values)
    error_checking.assert_is_geq_numpy_array(target_values, 0)

    if len(list_of_input_matrices) != len(predictor_names_by_matrix):
        error_string = (
            'Number of input matrices ({0:d}) should equal number of predictor-'
            'name lists ({1:d}).'
        ).format(len(list_of_input_matrices), len(predictor_names_by_matrix))

        raise ValueError(error_string)

    num_input_matrices = len(list_of_input_matrices)
    num_examples = len(target_values)

    for j in range(num_input_matrices):
        error_checking.assert_is_numpy_array_without_nan(
            list_of_input_matrices[j])

        this_num_dimensions = len(list_of_input_matrices[j].shape)
        if this_num_dimensions < 3:
            error_string = (
                '{0:d}th input matrix has {1:d} dimensions.  Should have at '
                'least 3.'
            ).format(j + 1, this_num_dimensions)

            raise ValueError(error_string)

        error_checking.assert_is_string_list(predictor_names_by_matrix[j])
        this_num_predictors = len(predictor_names_by_matrix[j])

        these_expected_dimensions = (
            (num_examples,) + list_of_input_matrices[j].shape[1:-1] +
            (this_num_predictors,)
        )
        these_expected_dimensions = numpy.array(
            these_expected_dimensions, dtype=int)

        error_checking.assert_is_numpy_array(
            list_of_input_matrices[j],
            exact_dimensions=these_expected_dimensions)

    error_checking.assert_is_integer(num_bootstrap_iters)
    num_bootstrap_iters = max([num_bootstrap_iters, 1])
    error_checking.assert_is_greater(bootstrap_confidence_level, 0.)
    error_checking.assert_is_less_than(bootstrap_confidence_level, 1.)

    # Get original cost (with no permutation).
    class_probability_matrix = prediction_function(
        model_object, list_of_input_matrices)

    all_original_costs = numpy.full(num_bootstrap_iters, numpy.nan)

    for k in range(num_bootstrap_iters):
        _, these_indices = bootstrapping.draw_sample(target_values)

        all_original_costs[k] = cost_function(
            target_values[these_indices],
            class_probability_matrix[these_indices, ...]
        )

    min_original_cost, max_original_cost = (
        bootstrapping.get_confidence_interval(
            stat_values=all_original_costs,
            confidence_level=bootstrap_confidence_level)
    )

    original_cost_bs_array = numpy.array([
        min_original_cost, numpy.mean(all_original_costs), max_original_cost
    ])

    print 'Original cost (no permutation): {0:s}'.format(
        str(original_cost_bs_array)
    )

    # Initialize output variables.
    remaining_predictor_names_by_matrix = copy.deepcopy(
        predictor_names_by_matrix)
    step_num = 0

    # Do dirty work.
    step1_predictor_names = []
    selected_predictor_name_by_step = []

    step1_cost_bs_matrix = None
    highest_cost_by_step_bs_matrix = None

    while True:
        print '\n'
        step_num += 1

        highest_cost_bs_matrix = numpy.full(3, -numpy.inf)
        best_matrix_index = None
        best_predictor_name = None
        best_predictor_permuted_values = None

        stopping_criterion = True

        for j in range(num_input_matrices):
            if len(remaining_predictor_names_by_matrix[j]) == 0:
                continue

            for this_predictor_name in remaining_predictor_names_by_matrix[j]:
                stopping_criterion = False

                print (
                    'Trying predictor "{0:s}" at step {1:d} of permutation '
                    'test...'
                ).format(this_predictor_name, step_num)

                these_input_matrices = copy.deepcopy(list_of_input_matrices)
                this_predictor_index = predictor_names_by_matrix[j].index(
                    this_predictor_name)

                these_input_matrices[j][..., this_predictor_index] = numpy.take(
                    these_input_matrices[j][..., this_predictor_index],
                    indices=numpy.random.permutation(
                        these_input_matrices[j].shape[0]),
                    axis=0
                )

                this_probability_matrix = prediction_function(
                    model_object, these_input_matrices)

                all_these_costs = numpy.full(num_bootstrap_iters, numpy.nan)

                for k in range(num_bootstrap_iters):
                    _, these_indices = bootstrapping.draw_sample(target_values)

                    all_these_costs[k] = cost_function(
                        target_values[these_indices],
                        this_probability_matrix[these_indices, ...]
                    )

                this_min_cost, this_max_cost = (
                    bootstrapping.get_confidence_interval(
                        stat_values=all_these_costs,
                        confidence_level=bootstrap_confidence_level)
                )

                this_cost_bs_array = numpy.array([
                    this_min_cost, numpy.mean(all_these_costs), this_max_cost
                ])

                this_cost_bs_matrix = numpy.reshape(
                    this_cost_bs_array, (1, this_cost_bs_array.size)
                )

                print 'Resulting cost = {0:s}\n'.format(
                    str(this_cost_bs_matrix)
                )

                if step_num == 1:
                    step1_predictor_names.append(this_predictor_name)

                    if step1_cost_bs_matrix is None:
                        step1_cost_bs_matrix = this_cost_bs_matrix + 0.
                    else:
                        step1_cost_bs_matrix = numpy.concatenate(
                            (step1_cost_bs_matrix, this_cost_bs_matrix), axis=0)

                if this_cost_bs_matrix[0, 1] < highest_cost_bs_matrix[0, 1]:
                    continue

                highest_cost_bs_matrix = this_cost_bs_matrix + 0.
                best_matrix_index = j + 0
                best_predictor_name = this_predictor_name + ''
                best_predictor_permuted_values = (
                    these_input_matrices[j][..., this_predictor_index] + 0.
                )

        if stopping_criterion:  # No more predictors to permute.
            break

        selected_predictor_name_by_step.append(best_predictor_name)

        if highest_cost_by_step_bs_matrix is None:
            highest_cost_by_step_bs_matrix = highest_cost_bs_matrix + 0.
        else:
            highest_cost_by_step_bs_matrix = numpy.concatenate(
                (highest_cost_by_step_bs_matrix, highest_cost_bs_matrix),
                axis=0)

        # Remove best predictor from list.
        remaining_predictor_names_by_matrix[best_matrix_index].remove(
            best_predictor_name)

        # Leave values of best predictor permuted.
        this_best_predictor_index = predictor_names_by_matrix[
            best_matrix_index].index(best_predictor_name)

        list_of_input_matrices[best_matrix_index][
            ..., this_best_predictor_index
        ] = best_predictor_permuted_values + 0.

        print 'Best predictor = "{0:s}" ... new cost = {1:.4e}'.format(
            best_predictor_name, highest_cost_bs_matrix)

    return {
        NUM_BOOTSTRAP_ITERS_KEY: num_bootstrap_iters,
        CONFIDENCE_LEVEL_KEY: bootstrap_confidence_level,
        SELECTED_PREDICTORS_KEY: selected_predictor_name_by_step,
        HIGHEST_COSTS_KEY: highest_cost_by_step_bs_matrix,
        ORIGINAL_COST_KEY: original_cost_bs_array,
        STEP1_PREDICTORS_KEY: step1_predictor_names,
        STEP1_COSTS_KEY: step1_cost_bs_matrix
    }


def write_results(result_dict, pickle_file_name):
    """Writes results to Pickle file.

    :param result_dict: Dictionary created by `run_permutation_test`, maybe with
        additional keys.
    :param pickle_file_name: Path to output file.
    :raises: ValueError: if any required keys are not found in the dictionary.
    """

    missing_keys = list(set(REQUIRED_KEYS) - set(result_dict.keys()))

    if len(missing_keys) == 0:
        file_system_utils.mkdir_recursive_if_necessary(
            file_name=pickle_file_name)

        pickle_file_handle = open(pickle_file_name, 'wb')
        pickle.dump(result_dict, pickle_file_handle)
        pickle_file_handle.close()

        return

    error_string = (
        '\n{0:s}\nKeys listed above were expected, but not found, in '
        'dictionary.'
    ).format(str(missing_keys))

    raise ValueError(error_string)


def read_results(pickle_file_name):
    """Reads results from Pickle file.

    :param pickle_file_name: Path to input file.
    :return: result_dict: Dictionary created by `run_permutation_test`, maybe
        with additional keys.
    :raises: ValueError: if any required keys are not found in the dictionary.
    """

    pickle_file_handle = open(pickle_file_name, 'rb')
    result_dict = pickle.load(pickle_file_handle)
    pickle_file_handle.close()

    missing_keys = list(set(REQUIRED_KEYS) - set(result_dict.keys()))
    if len(missing_keys) == 0:
        return result_dict

    error_string = (
        '\n{0:s}\nKeys listed above were expected, but not found, in file '
        '"{1:s}".'
    ).format(str(missing_keys), pickle_file_name)

    raise ValueError(error_string)
