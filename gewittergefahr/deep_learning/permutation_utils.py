"""Helper methods for permutation test.

The permutation test is a way to rank predictor importance for a particular
model.  There are two versions of the permutation test, single-pass
(Breiman 2001) and multi-pass (Lakshmanan et al. 2015).

--- REFERENCES ---

Breiman, L., 2001: "Random forests." Machine Learning, 45 (1), 5-32.

Lakshmanan, V., C. Karstens, J. Krause, K. Elmore, A. Ryzhkov, and S. Berkseth,
    2015: "Which polarimetric variables are important for weather/no-weather
    discrimination?" Journal of Atmospheric and Oceanic Technology, 32 (6),
    1209-1223.
"""

import time
import pickle
import numpy
import keras.utils
from sklearn.metrics import roc_auc_score as sklearn_auc
from gewittergefahr.gg_io import storm_tracking_io as tracking_io
from gewittergefahr.gg_utils import bootstrapping
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.gg_utils import error_checking

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'
MINOR_SEPARATOR_STRING = '\n\n' + '-' * 50 + '\n\n'

DEFAULT_NUM_BOOTSTRAP_REPS = 1
MIN_PROBABILITY = 1e-15
MAX_PROBABILITY = 1. - MIN_PROBABILITY

PREDICTOR_MATRICES_KEY = 'predictor_matrices'
PERMUTED_FLAGS_KEY = 'permuted_flags_by_matrix'
PERMUTED_PREDICTORS_KEY = 'permuted_predictor_names'
PERMUTED_COST_MATRIX_KEY = 'permuted_cost_matrix'
UNPERMUTED_PREDICTORS_KEY = 'unpermuted_predictor_names'
UNPERMUTED_COST_MATRIX_KEY = 'unpermuted_cost_matrix'
BEST_PREDICTOR_KEY = 'best_predictor_name'
BEST_COST_ARRAY_KEY = 'best_cost_array'

BEST_PREDICTORS_KEY = 'best_predictor_names'
BEST_COST_MATRIX_KEY = 'best_cost_matrix'
ORIGINAL_COST_ARRAY_KEY = 'original_cost_array'
STEP1_PREDICTORS_KEY = 'step1_predictor_names'
STEP1_COST_MATRIX_KEY = 'step1_cost_matrix'
BACKWARDS_FLAG = 'backwards_test'

REQUIRED_KEYS = [
    BEST_PREDICTORS_KEY, BEST_COST_MATRIX_KEY, ORIGINAL_COST_ARRAY_KEY,
    STEP1_PREDICTORS_KEY, STEP1_COST_MATRIX_KEY, BACKWARDS_FLAG
]

# Optional keys in output file.
FULL_IDS_KEY = tracking_io.FULL_IDS_KEY
STORM_TIMES_KEY = tracking_io.STORM_TIMES_KEY
MODEL_FILE_KEY = 'model_file_name'
TARGET_VALUES_KEY = 'target_values'


def _unpermute_one_predictor(
        predictor_matrices, clean_predictor_matrices, separate_heights,
        matrix_index, predictor_index):
    """Unpermutes values of one predictor (replaces permuted with clean values).

    Specifically, will unpermute values of the [j]th predictor in the [i]th
    matrix, where i = `matrix_index` and j = `predictor_index`.

    :param predictor_matrices: See doc for `permute_one_predictor`.
    :param clean_predictor_matrices: Clean version of `predictor_matrices` (with
        no values permuted).
    :param separate_heights: See doc for `permute_one_predictor`.
    :param matrix_index: See discussion above.
    :param predictor_index: See discussion above.
    :return: predictor_matrices: Same as input but after unpermuting.
    """

    i = matrix_index
    j = predictor_index
    num_spatial_dim = len(predictor_matrices[i].shape) - 2

    if num_spatial_dim == 3 and separate_heights:
        predictor_matrices[i], original_shape = flatten_last_two_dim(
            predictor_matrices[i]
        )

        clean_predictor_matrices[i] = flatten_last_two_dim(
            clean_predictor_matrices[i]
        )[0]
    else:
        original_shape = None

    predictor_matrices[i][..., j] = clean_predictor_matrices[i][..., j]

    if original_shape is not None:
        predictor_matrices[i] = numpy.reshape(
            predictor_matrices[i], original_shape, order='F'
        )

        clean_predictor_matrices[i] = numpy.reshape(
            clean_predictor_matrices[i], original_shape, order='F'
        )

    return predictor_matrices


def flatten_last_two_dim(data_matrix):
    """Flattens last two dimensions of numpy array.

    :param data_matrix: numpy array.
    :return: data_matrix: Same but with one dimension less.
    :return: orig_shape: numpy array with original dimensions.
    """

    error_checking.assert_is_numpy_array(data_matrix)
    error_checking.assert_is_geq(len(data_matrix.shape), 3)

    orig_shape = numpy.array(data_matrix.shape, dtype=int)

    new_shape = numpy.array(data_matrix.shape, dtype=int)
    new_shape[-2] = new_shape[-2] * new_shape[-1]
    new_shape = new_shape[:-1]

    return numpy.reshape(data_matrix, new_shape, order='F'), orig_shape


def bootstrap_cost(target_values, class_probability_matrix, cost_function,
                   num_replicates):
    """Bootstraps cost for one set of examples.

    E = number of examples
    K = number of classes
    B = number of bootstrap replicates

    :param target_values: length-E numpy array of target values (integers in
        range 0...[K - 1]).
    :param class_probability_matrix: E-by-K numpy array of predicted
        probabilities.
    :param cost_function: Cost function, used to evaluate predicted
        probabilities.  Must be negatively oriented (so that lower values are
        better), with the following inputs and outputs.
    Input: target_values: Same as input to this method.
    Input: class_probability_matrix: Same as input to this method.
    Output: cost: Scalar value.

    :param num_replicates: Number of bootstrap replicates.
    :return: cost_values: length-B numpy array of cost values.
    """

    error_checking.assert_is_integer(num_replicates)
    error_checking.assert_is_geq(num_replicates, 1)
    cost_values = numpy.full(num_replicates, numpy.nan)

    if num_replicates == 1:
        cost_values[0] = cost_function(target_values,
                                       class_probability_matrix)
    else:
        for k in range(num_replicates):
            _, these_indices = bootstrapping.draw_sample(target_values)

            cost_values[k] = cost_function(
                target_values[these_indices],
                class_probability_matrix[these_indices, ...]
            )

    print('Average cost = {0:.4f}'.format(numpy.mean(cost_values)))
    return cost_values


def permute_one_predictor(
        predictor_matrices, separate_heights, matrix_index, predictor_index,
        permuted_values=None):
    """Permutes values of one predictor.

    Specifically, will permute values of the [j]th predictor in the [i]th
    matrix, where i = `matrix_index` and j = `predictor_index`.

    T = number of input tensors to the model
    E = number of examples

    :param predictor_matrices: length-T list of numpy arrays, where the first
        axis of each has length E.
    :param separate_heights: Boolean flag.  If True, for arrays with 3 spatial
        dimensions, each predictor/height pair will be shuffled independently.
        If False, for arrays with 3 spatial dimensions, each predictor will be
        shuffled independently.
    :param matrix_index: See discussion above.
    :param predictor_index: See discussion above.
    :param permuted_values: numpy array of permuted values with which to replace
        clean values.  If None, permuted values will be created randomly on the
        fly.
    :return: predictor_matrices: Same as input but after permutation.
    :return: permuted_values: numpy array of permuted values with which clean
        values were replaced.
    """

    # Check input args.
    error_checking.assert_is_list(predictor_matrices)
    for this_matrix in predictor_matrices:
        error_checking.assert_is_numpy_array_without_nan(this_matrix)

    error_checking.assert_is_boolean(separate_heights)
    error_checking.assert_is_integer(matrix_index)
    error_checking.assert_is_geq(matrix_index, 0)
    error_checking.assert_is_integer(predictor_index)
    error_checking.assert_is_geq(predictor_index, 0)

    if permuted_values is not None:
        error_checking.assert_is_numpy_array_without_nan(permuted_values)

    # Do dirty work.
    i = matrix_index
    j = predictor_index
    num_spatial_dim = len(predictor_matrices[i].shape) - 2

    if num_spatial_dim == 3 and separate_heights:
        predictor_matrices[i], original_shape = flatten_last_two_dim(
            predictor_matrices[i]
        )
    else:
        original_shape = None

    if permuted_values is None:
        random_indices = numpy.random.permutation(
            predictor_matrices[i].shape[0]
        )
        predictor_matrices[i][..., j] = (
            predictor_matrices[i][random_indices, ..., j]
        )

        # predictor_matrices[i][..., j] = numpy.take(
        #     predictor_matrices[i][..., j],
        #     indices=numpy.random.permutation(predictor_matrices[i].shape[0]),
        #     axis=0
        # )
    else:
        predictor_matrices[i][..., j] = permuted_values

    permuted_values = predictor_matrices[i][..., j]

    if original_shape is not None:
        predictor_matrices[i] = numpy.reshape(
            predictor_matrices[i], original_shape, order='F'
        )

    return predictor_matrices, permuted_values


def run_forward_test_one_step(
        model_object, predictor_matrices, predictor_names_by_matrix,
        target_values, prediction_function, cost_function, separate_heights,
        num_bootstrap_reps, step_num, permuted_flags_by_matrix):
    """Runs one step of the forward permutation test.

    T = number of input tensors to the model
    E = number of examples
    K = number of classes
    B = number of bootstrap replicates
    P_i = number of predictors available to permute in [i]th input tensor
    P = total number of predictors to permute in this step

    :param model_object: Trained model (instance of `keras.models.Model` or
        `keras.models.Sequential`).
    :param predictor_matrices: length-T list of numpy arrays, where the first
        axis of each has length E.
    :param predictor_names_by_matrix: length-T list of lists, where the [i]th
        list contains names of predictors to be permuted in the [i]th input
        tensor.  The [i]th list should have length P_i.
    :param target_values: length-E numpy array of target values (integers in
        range 0...[K - 1]).
    :param prediction_function: Function used to generate predictions from the
        model.  Should have the following format.
    Input: model_object: Same as input to this method.
    Input: predictor_matrices: Same as input to this method.
    Output: class_probability_matrix: E-by-K numpy array of predicted
        probabilities.

    :param cost_function: Cost function, used to evaluate predicted
        probabilities.  Must be negatively oriented (so that lower values are
        better), with the following inputs and outputs.
    Input: target_values: Same as input to this method.
    Input: class_probability_matrix: E-by-K numpy array of predicted
        probabilities.
    Output: cost: Scalar value.

    :param separate_heights: Boolean flag.  If True, for arrays with 3 spatial
        dimensions, each predictor/height pair will be shuffled independently.
        If False, for arrays with 3 spatial dimensions, each predictor will be
        shuffled independently.
    :param num_bootstrap_reps: Number of bootstrap replicates.
    :param step_num: Current step (integer) in overall permutation test.
    :param permuted_flags_by_matrix: length-T list of numpy arrays, where the
        [i]th array contains Boolean flags, indicating which predictors in the
        [i]th tensor have yet been permuted.  The [i]th array should have length
        P_i.
    :return: forward_step_dict: Dictionary with the following keys.
    forward_step_dict["predictor_matrices"]: Same as input but maybe with
        different values.
    forward_step_dict["permuted_flags_by_matrix"]: Same as input but maybe with
        different values.
    forward_step_dict["permuted_predictor_names"]: length-P list with names of
        predictors permuted in this step.
    forward_step_dict["permuted_cost_matrix"]: P-by-B numpy array of costs after
        permutation.
    forward_step_dict["best_predictor_name"]: Name of best predictor in this
        step.
    forward_step_dict["best_cost_array"]: length-B numpy array of costs after
        permutation of best predictor.
    """

    best_matrix_index = -1
    best_predictor_index = -1
    best_permuted_values = None

    permuted_predictor_names = []
    permuted_cost_matrix = numpy.full((0, num_bootstrap_reps), numpy.nan)
    best_cost_array = numpy.full(num_bootstrap_reps, -numpy.inf)

    num_matrices = len(predictor_matrices)

    for i in range(num_matrices):
        this_num_predictors = len(predictor_names_by_matrix[i])

        for j in range(this_num_predictors):
            if permuted_flags_by_matrix[i][j]:
                continue

            print('Permuting predictor "{0:s}" at step {1:d}...'.format(
                predictor_names_by_matrix[i][j], step_num
            ))
            exec_start_time_unix_sec = time.time()

            these_predictor_matrices, these_permuted_values = (
                permute_one_predictor(
                    predictor_matrices=[a + 0. for a in predictor_matrices],
                    separate_heights=separate_heights,
                    matrix_index=i, predictor_index=j)
            )

            elapsed_time_sec = time.time() - exec_start_time_unix_sec
            print('Time elapsed = {0:.2f} seconds'.format(elapsed_time_sec))

            this_probability_matrix = prediction_function(
                model_object, these_predictor_matrices)
            print(MINOR_SEPARATOR_STRING)

            this_cost_array = bootstrap_cost(
                target_values=target_values,
                class_probability_matrix=this_probability_matrix,
                cost_function=cost_function, num_replicates=num_bootstrap_reps)

            this_cost_matrix = numpy.reshape(
                this_cost_array, (1, len(this_cost_array))
            )

            permuted_predictor_names.append(predictor_names_by_matrix[i][j])
            permuted_cost_matrix = numpy.concatenate(
                (permuted_cost_matrix, this_cost_matrix), axis=0
            )

            if numpy.mean(this_cost_array) < numpy.mean(best_cost_array):
                continue

            best_matrix_index = i + 0
            best_predictor_index = j + 0
            best_permuted_values = these_permuted_values + 0.

            best_cost_array = this_cost_array + 0.

    if len(permuted_predictor_names) == 0:
        return None

    i = best_matrix_index + 0
    j = best_predictor_index + 0
    best_predictor_name = predictor_names_by_matrix[i][j]
    permuted_flags_by_matrix[i][j] = True

    exec_start_time_unix_sec = time.time()
    predictor_matrices = permute_one_predictor(
        predictor_matrices=predictor_matrices,
        separate_heights=separate_heights,
        matrix_index=i, predictor_index=j, permuted_values=best_permuted_values
    )[0]
    elapsed_time_sec = time.time() - exec_start_time_unix_sec

    print('Best predictor = "{0:s}" ... cost = {1:.4f}'.format(
        best_predictor_name, numpy.mean(best_cost_array)
    ))

    print((
        'Time elapsed in permanently permuting best predictor = {0:.2f} seconds'
    ).format(
        elapsed_time_sec
    ))

    return {
        PREDICTOR_MATRICES_KEY: predictor_matrices,
        PERMUTED_FLAGS_KEY: permuted_flags_by_matrix,
        PERMUTED_PREDICTORS_KEY: permuted_predictor_names,
        PERMUTED_COST_MATRIX_KEY: permuted_cost_matrix,
        BEST_PREDICTOR_KEY: best_predictor_name,
        BEST_COST_ARRAY_KEY: best_cost_array
    }


def run_backwards_test_one_step(
        model_object, predictor_matrices, clean_predictor_matrices,
        predictor_names_by_matrix, target_values, prediction_function,
        cost_function, separate_heights, num_bootstrap_reps, step_num,
        permuted_flags_by_matrix):
    """Runs one step of the backwards permutation test.

    B = number of bootstrap replicates
    P = total number of predictors to permute in this step

    :param model_object: See doc for `run_forward_test_one_step`.
    :param predictor_matrices: Same.
    :param clean_predictor_matrices: Clean version of `predictor_matrices` (with
        no values permuted).
    :param predictor_names_by_matrix: See doc for `run_forward_test_one_step`.
    :param target_values: Same.
    :param prediction_function: Same.
    :param cost_function: Same.
    :param separate_heights: Same.
    :param num_bootstrap_reps: Same.
    :param step_num: Same.
    :param permuted_flags_by_matrix: Same.

    :return: backwards_step_dict: Dictionary with the following keys.
    backwards_step_dict["predictor_matrices"]: Same as input but maybe with
        different values.
    backwards_step_dict["permuted_flags_by_matrix"]: Same as input but maybe
        with different values.
    backwards_step_dict["unpermuted_predictor_names"]: length-P list with names
        of predictors unpermuted in this step.
    backwards_step_dict["unpermuted_cost_matrix"]: P-by-B numpy array of costs
        after unpermutation.
    backwards_step_dict["best_predictor_name"]: Name of best predictor in this
        step.
    backwards_step_dict["best_cost_array"]: length-B numpy array of costs after
        unpermutation of best predictor.
    """

    best_matrix_index = -1
    best_predictor_index = -1

    unpermuted_predictor_names = []
    unpermuted_cost_matrix = numpy.full((0, num_bootstrap_reps), numpy.nan)
    best_cost_array = numpy.full(num_bootstrap_reps, numpy.inf)

    num_matrices = len(predictor_matrices)

    for i in range(num_matrices):
        this_num_predictors = len(predictor_names_by_matrix[i])

        for j in range(this_num_predictors):
            if not permuted_flags_by_matrix[i][j]:
                continue

            print('Unpermuting predictor "{0:s}" at step {1:d}...'.format(
                predictor_names_by_matrix[i][j], step_num
            ))
            exec_start_time_unix_sec = time.time()

            these_predictor_matrices = _unpermute_one_predictor(
                predictor_matrices=[a + 0. for a in predictor_matrices],
                clean_predictor_matrices=clean_predictor_matrices,
                separate_heights=separate_heights,
                matrix_index=i, predictor_index=j
            )

            elapsed_time_sec = time.time() - exec_start_time_unix_sec
            print('Time elapsed = {0:.2f} seconds'.format(elapsed_time_sec))

            this_probability_matrix = prediction_function(
                model_object, these_predictor_matrices)
            print(MINOR_SEPARATOR_STRING)

            this_cost_array = bootstrap_cost(
                target_values=target_values,
                class_probability_matrix=this_probability_matrix,
                cost_function=cost_function, num_replicates=num_bootstrap_reps)

            this_cost_matrix = numpy.reshape(
                this_cost_array, (1, len(this_cost_array))
            )

            unpermuted_predictor_names.append(predictor_names_by_matrix[i][j])
            unpermuted_cost_matrix = numpy.concatenate(
                (unpermuted_cost_matrix, this_cost_matrix), axis=0
            )

            if numpy.mean(this_cost_array) > numpy.mean(best_cost_array):
                continue

            best_matrix_index = i + 0
            best_predictor_index = j + 0
            best_cost_array = this_cost_array + 0.

    if len(unpermuted_predictor_names) == 0:
        return None

    i = best_matrix_index + 0
    j = best_predictor_index + 0
    best_predictor_name = predictor_names_by_matrix[i][j]
    permuted_flags_by_matrix[i][j] = False

    exec_start_time_unix_sec = time.time()
    predictor_matrices = _unpermute_one_predictor(
        predictor_matrices=predictor_matrices,
        clean_predictor_matrices=clean_predictor_matrices,
        separate_heights=separate_heights,
        matrix_index=i, predictor_index=j
    )
    elapsed_time_sec = time.time() - exec_start_time_unix_sec

    print('Best predictor = "{0:s}" ... cost = {1:.4f}'.format(
        best_predictor_name, numpy.mean(best_cost_array)
    ))

    print((
        'Time elapsed in permanently unpermuting best predictor = {0:.2f} '
        'seconds'
    ).format(
        elapsed_time_sec
    ))

    return {
        PREDICTOR_MATRICES_KEY: predictor_matrices,
        PERMUTED_FLAGS_KEY: permuted_flags_by_matrix,
        UNPERMUTED_PREDICTORS_KEY: unpermuted_predictor_names,
        UNPERMUTED_COST_MATRIX_KEY: unpermuted_cost_matrix,
        BEST_PREDICTOR_KEY: best_predictor_name,
        BEST_COST_ARRAY_KEY: best_cost_array
    }


def cross_entropy_function(
        target_values, class_probability_matrix, test_mode=False):
    """Cross-entropy cost function.

    This function works for binary or multi-class classification.

    E = number of examples
    K = number of classes

    :param target_values: length-E numpy array of target values (integers in
        range 0...[K - 1]).
    :param class_probability_matrix: E-by-K numpy array of predicted
        probabilities.
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
    """Computes negative AUC (area under the ROC curve).

    This function works only for binary classification!

    :param target_values: length-E numpy array of target values (integers in
        0...1).
    :param class_probability_matrix: E-by-2 numpy array of predicted
        probabilities.
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


def write_results(result_dict, pickle_file_name):
    """Writes results to Pickle file.

    :param result_dict: Dictionary created by `run_forward_test` or
        `run_backwards_test`, maybe with additional keys.
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
    :return: result_dict: Dictionary created by `run_forward_test` or
        `run_backwards_test`, maybe with additional keys.
    :raises: ValueError: if any required keys are not found in the dictionary.
    """

    pickle_file_handle = open(pickle_file_name, 'rb')
    result_dict = pickle.load(pickle_file_handle)
    pickle_file_handle.close()

    if BACKWARDS_FLAG not in result_dict:
        result_dict[BACKWARDS_FLAG] = False

    missing_keys = list(set(REQUIRED_KEYS) - set(result_dict.keys()))
    if len(missing_keys) == 0:
        return result_dict

    error_string = (
        '\n{0:s}\nKeys listed above were expected, but not found, in file '
        '"{1:s}".'
    ).format(str(missing_keys), pickle_file_name)

    raise ValueError(error_string)
