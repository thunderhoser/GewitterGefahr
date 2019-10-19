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
from gewittergefahr.gg_utils import radar_utils
from gewittergefahr.gg_utils import soundings
from gewittergefahr.gg_utils import bootstrapping
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.gg_utils import error_checking
from gewittergefahr.deep_learning import cnn
from gewittergefahr.deep_learning import training_validation_io as trainval_io
from gewittergefahr.plotting import radar_plotting

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'
MINOR_SEPARATOR_STRING = '\n\n' + '-' * 50 + '\n\n'

DEFAULT_NUM_BOOTSTRAP_REPS = 1
MIN_PROBABILITY = 1e-15
MAX_PROBABILITY = 1. - MIN_PROBABILITY

PREDICTOR_MATRICES_KEY = 'predictor_matrices'
PERMUTED_FLAGS_KEY = 'permuted_flags_by_matrix'
PERMUTED_PREDICTORS_KEY = 'permuted_predictor_names'
PERMUTED_COST_MATRIX_KEY = 'permuted_cost_matrix'
BEST_PREDICTOR_KEY = 'best_predictor_name'
BEST_COST_ARRAY_KEY = 'best_cost_array'

# Mandatory keys in result dictionary (see `write_results`).
BEST_PREDICTORS_KEY = 'best_predictor_names'
BEST_COST_MATRIX_KEY = 'best_cost_matrix'
ORIGINAL_COST_ARRAY_KEY = 'original_cost_array'
STEP1_PREDICTORS_KEY = 'step1_predictor_names'
STEP1_COST_MATRIX_KEY = 'step1_cost_matrix'

REQUIRED_KEYS = [
    BEST_PREDICTORS_KEY, BEST_COST_MATRIX_KEY, ORIGINAL_COST_ARRAY_KEY,
    STEP1_PREDICTORS_KEY, STEP1_COST_MATRIX_KEY
]

# Optional keys in result dictionary (see `write_results`).
FULL_IDS_KEY = tracking_io.FULL_IDS_KEY
STORM_TIMES_KEY = tracking_io.STORM_TIMES_KEY
MODEL_FILE_KEY = 'model_file_name'
TARGET_VALUES_KEY = 'target_values'


def _bootstrap_cost(target_values, class_probability_matrix, cost_function,
                    num_replicates):
    """Bootstraps cost for one set of prediction-observation pairs.

    B = number of bootstrap replicates

    :param target_values: See doc for `run_permutation_test`.
    :param class_probability_matrix: Same.
    :param cost_function: Same.
    :param num_replicates: B in the above discussion.
    :return: cost_values: length-B numpy array of cost values.
    """

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


def _permute_one_predictor(predictor_matrices, separate_radar_heights,
                           matrix_index, predictor_index, permuted_values=None):
    """Permutes values of one predictor.

    Will permute values of [j]th predictor in [i]th matrix, where
    i = `matrix_index` and j = `predictor_index`.

    :param predictor_matrices: See doc for `run_permutation`.
    :param separate_radar_heights: Same.
    :param matrix_index: See discussion above.
    :param predictor_index: Same.
    :param permuted_values: numpy array of permuted values to use.  If this is
        None, permuted values will be created randomly.
    :return: predictor_matrices: Same as input but with permuted values.
    :return: permuted_values: numpy array with only values that were just
        permuted.
    """

    i = matrix_index
    j = predictor_index
    num_spatial_dim = len(predictor_matrices[i].shape) - 2

    if num_spatial_dim == 3 and separate_radar_heights:
        predictor_matrices[i], original_shape = flatten_last_two_dim(
            predictor_matrices[i]
        )
    else:
        original_shape = None

    if permuted_values is None:
        predictor_matrices[i][..., j] = numpy.take(
            predictor_matrices[i][..., j],
            indices=numpy.random.permutation(predictor_matrices[i].shape[0]),
            axis=0
        )
    else:
        predictor_matrices[i][..., j] = permuted_values

    permuted_values = predictor_matrices[i][..., j]

    if original_shape is not None:
        predictor_matrices[i] = numpy.reshape(
            predictor_matrices[i], original_shape, order='F'
        )

    return predictor_matrices, permuted_values


def _run_permutation_one_step(
        model_object, predictor_matrices, predictor_names_by_matrix,
        target_values, prediction_function, cost_function,
        separate_radar_heights, num_bootstrap_reps, step_num,
        permuted_flags_by_matrix):
    """Does one step of permutation test.

    T = number of input tensors to the model
    B = number of bootstrap replicates
    P = number of predictors permuted in this step

    :param model_object: See doc for `run_permutation_test`.
    :param predictor_matrices: Same.
    :param predictor_names_by_matrix: Same.
    :param target_values: Same.
    :param prediction_function: Same.
    :param cost_function: Same.
    :param separate_radar_heights: Same.
    :param num_bootstrap_reps: Same.
    :param step_num: Number of current step.
    :param permuted_flags_by_matrix: length-T list.  The [i]th item is a numpy
        array of Boolean flags, indicating which predictors have already been
        permuted (and thus need not be permuted again).

    :return: one_step_dict: Dictionary with the following keys.
    one_step_dict["predictor_matrices"]: Same as input but maybe with different
        values.
    one_step_dict["permuted_flags_by_matrix"]: Same as input but maybe with
        different values.
    one_step_dict["permuted_predictor_names"]: length-P list with names of
        predictors permuted in this step.
    one_step_dict["permuted_cost_matrix"]: P-by-B numpy array of costs after
        permutation.
    one_step_dict["best_predictor_name"]: Name of best predictor in this step.
    one_step_dict["best_cost_array"]: length-B numpy array of costs after
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

            these_predictor_matrices, these_permuted_values = (
                _permute_one_predictor(
                    predictor_matrices=copy.deepcopy(predictor_matrices),
                    separate_radar_heights=separate_radar_heights,
                    matrix_index=i, predictor_index=j)
            )

            this_probability_matrix = prediction_function(
                model_object, these_predictor_matrices)

            this_cost_array = _bootstrap_cost(
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

    predictor_matrices = _permute_one_predictor(
        predictor_matrices=predictor_matrices,
        separate_radar_heights=separate_radar_heights,
        matrix_index=i, predictor_index=j, permuted_values=best_permuted_values
    )[0]

    print('Best predictor = "{0:s}" ... cost = {1:.4f}'.format(
        best_predictor_name, numpy.mean(best_cost_array)
    ))

    return {
        PREDICTOR_MATRICES_KEY: predictor_matrices,
        PERMUTED_FLAGS_KEY: permuted_flags_by_matrix,
        PERMUTED_PREDICTORS_KEY: permuted_predictor_names,
        PERMUTED_COST_MATRIX_KEY: permuted_cost_matrix,
        BEST_PREDICTOR_KEY: best_predictor_name,
        BEST_COST_ARRAY_KEY: best_cost_array
    }


def create_nice_predictor_names(
        predictor_matrices, cnn_metadata_dict, separate_radar_heights=False):
    """Creates list of nice (human-readable) predictor names for each matrix.

    T = number of input tensors to the CNN

    :param predictor_matrices: See doc for `run_permutation_test`.
    :param cnn_metadata_dict: Same.
    :param separate_radar_heights: Same.
    :return: predictor_names_by_matrix: length-T list, where the [i]th element
        is a list of predictor names for the [i]th matrix.
    """

    error_checking.assert_is_boolean(separate_radar_heights)

    myrorss_2d3d = cnn_metadata_dict[cnn.CONV_2D3D_KEY]
    layer_operation_dicts = cnn_metadata_dict[cnn.LAYER_OPERATIONS_KEY]

    training_option_dict = cnn_metadata_dict[cnn.TRAINING_OPTION_DICT_KEY]
    radar_field_names = training_option_dict[trainval_io.RADAR_FIELDS_KEY]
    radar_heights_m_agl = training_option_dict[trainval_io.RADAR_HEIGHTS_KEY]

    nice_radar_field_names = [
        radar_utils.field_name_to_verbose(field_name=f, include_units=False)
        for f in radar_field_names
    ]

    num_matrices = len(predictor_matrices)
    predictor_names_by_matrix = [[]] * num_matrices

    if myrorss_2d3d:
        if separate_radar_heights:
            these_field_names = (
                [radar_utils.REFL_NAME] * len(radar_heights_m_agl)
            )

            predictor_names_by_matrix[0] = (
                radar_plotting.fields_and_heights_to_names(
                    field_names=these_field_names,
                    heights_m_agl=radar_heights_m_agl, include_units=False)
            )

            predictor_names_by_matrix[0] = [
                n.replace('\n', ' ') for n in predictor_names_by_matrix[0]
            ]
        else:
            predictor_names_by_matrix[0] = ['Reflectivity']

        predictor_names_by_matrix[1] += nice_radar_field_names
    else:
        if separate_radar_heights:
            height_matrix_m_agl, field_name_matrix = numpy.meshgrid(
                radar_heights_m_agl, numpy.array(radar_field_names)
            )

            these_field_names = numpy.ravel(field_name_matrix).tolist()
            these_heights_m_agl = numpy.round(
                numpy.ravel(height_matrix_m_agl)
            ).astype(int)

            predictor_names_by_matrix[0] = (
                radar_plotting.fields_and_heights_to_names(
                    field_names=these_field_names,
                    heights_m_agl=these_heights_m_agl, include_units=False)
            )

            predictor_names_by_matrix[0] = [
                n.replace('\n', ' ') for n in predictor_names_by_matrix[0]
            ]

        elif layer_operation_dicts is not None:
            _, predictor_names_by_matrix[0] = (
                radar_plotting.layer_operations_to_names(
                    list_of_layer_operation_dicts=layer_operation_dicts,
                    include_units=False)
            )

            predictor_names_by_matrix[0] = [
                n.replace('\n', '; ') for n in predictor_names_by_matrix[0]
            ]
        else:
            predictor_names_by_matrix[0] = nice_radar_field_names

    sounding_field_names = training_option_dict[trainval_io.SOUNDING_FIELDS_KEY]

    if sounding_field_names is not None:
        predictor_names_by_matrix[-1] = [
            soundings.field_name_to_verbose(field_name=f, include_units=False)
            for f in sounding_field_names
        ]

    return predictor_names_by_matrix


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
        sounding_matrix=sounding_matrix, verbose=True)


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
        sounding_matrix=sounding_matrix, verbose=True)


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

    num_input_matrices = len(list_of_input_matrices)
    first_num_dimensions = len(list_of_input_matrices[0].shape)
    upsample_refl = first_num_dimensions == 5

    if upsample_refl:
        if num_input_matrices == 2:
            sounding_matrix = list_of_input_matrices[-1]
        else:
            sounding_matrix = None

        return cnn.apply_2d_or_3d_cnn(
            model_object=model_object,
            radar_image_matrix=list_of_input_matrices[0],
            sounding_matrix=sounding_matrix, verbose=True)

    if num_input_matrices == 3:
        sounding_matrix = list_of_input_matrices[-1]
    else:
        sounding_matrix = None

    return cnn.apply_2d3d_cnn(
        model_object=model_object,
        reflectivity_matrix_dbz=list_of_input_matrices[0],
        azimuthal_shear_matrix_s01=list_of_input_matrices[1],
        sounding_matrix=sounding_matrix, verbose=True)


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
        model_object, predictor_matrices, target_values, cnn_metadata_dict,
        cost_function, separate_radar_heights=False,
        num_bootstrap_reps=DEFAULT_NUM_BOOTSTRAP_REPS):
    """Runs the permutation test.

    T = number of input tensors to the model
    E = number of examples
    K = number of classes
    P = number of predictors to permute
    B = number of bootstrap replicates

    :param model_object: Trained model (instance of `keras.models.Model` or
        `keras.models.Sequential`).
    :param cnn_metadata_dict: Dictionary returned by `cnn.read_model_metadata`.
    :param predictor_matrices: length-T list of predictor matrices (numpy
        arrays), in the order used for training.  The first axis of each array
        should have length E.
    :param target_values: length-E numpy array of target values (integers from
        0...[K - 1]).

    :param cost_function: Function used to evaluate model predictions.  Must be
        negatively oriented (lower is better), with the following inputs and
        outputs.
    Input: target_values: Same as input to this method.
    Input: class_probability_matrix: E-by-K numpy array of class probabilities.
    Output: cost: Scalar value.

    :param separate_radar_heights: Boolean flag.  If True, 3-D radar fields will
        be separated by height, so each step will involve permuting one variable
        at one height.  If False, each step will involve permuting one variable
        at all heights.
    :param num_bootstrap_reps: Number of bootstrap replicates.  If you do not
        want to use bootstrapping, make this <= 1.

    :return: result_dict: Dictionary with the following keys.
    result_dict["best_predictor_names"]: length-P list of best predictors.  The
        [j]th element is the name of the [j]th predictor to be permanently
        permuted.
    result_dict["best_cost_matrix"]: P-by-B numpy array of costs after
        permutation.
    result_dict["original_cost_array"]: length-B numpy array of costs before
        permutation.
    result_dict["step1_predictor_names"]: length-P list of predictors in the
        order that they were permuted in step 1.
    result_dict["step1_cost_matrix"]: P-by-B numpy array of costs after
        permutation in step 1.
    """

    error_checking.assert_is_integer_numpy_array(target_values)
    error_checking.assert_is_geq_numpy_array(target_values, 0)
    error_checking.assert_is_integer(num_bootstrap_reps)
    num_bootstrap_reps = max([num_bootstrap_reps, 1])

    if cnn_metadata_dict[cnn.CONV_2D3D_KEY]:
        prediction_function = prediction_function_2d3d_cnn
    else:
        num_radar_dimensions = len(predictor_matrices[0].shape) - 2

        if num_radar_dimensions == 2:
            prediction_function = prediction_function_2d_cnn
        else:
            prediction_function = prediction_function_3d_cnn

    predictor_names_by_matrix = create_nice_predictor_names(
        predictor_matrices=predictor_matrices,
        cnn_metadata_dict=cnn_metadata_dict,
        separate_radar_heights=separate_radar_heights)

    num_matrices = len(predictor_names_by_matrix)
    for i in range(num_matrices):
        print('Predictors in {0:d}th matrix:\n{1:s}\n'.format(
            i + 1, str(predictor_names_by_matrix[i])
        ))

    print(SEPARATOR_STRING)

    # Find original cost (before permutation).
    print('Finding original cost (before permutation)...')
    class_probability_matrix = prediction_function(
        model_object, predictor_matrices)

    original_cost_array = _bootstrap_cost(
        target_values=target_values,
        class_probability_matrix=class_probability_matrix,
        cost_function=cost_function, num_replicates=num_bootstrap_reps)

    # Do the permutation part.
    permuted_flags_by_matrix = [
        numpy.full(len(n), 0, dtype=bool)
        for n in predictor_names_by_matrix
    ]

    step_num = 0

    step1_predictor_names = None
    step1_cost_matrix = None
    best_predictor_names = []
    best_cost_matrix = numpy.full((0, num_bootstrap_reps), numpy.nan)

    while True:
        print(MINOR_SEPARATOR_STRING)
        step_num += 1

        this_dict = _run_permutation_one_step(
            model_object=model_object, predictor_matrices=predictor_matrices,
            predictor_names_by_matrix=predictor_names_by_matrix,
            target_values=target_values,
            prediction_function=prediction_function,
            cost_function=cost_function,
            separate_radar_heights=separate_radar_heights,
            num_bootstrap_reps=num_bootstrap_reps, step_num=step_num,
            permuted_flags_by_matrix=permuted_flags_by_matrix)

        if this_dict is None:
            break

        predictor_matrices = this_dict[PREDICTOR_MATRICES_KEY]
        permuted_flags_by_matrix = this_dict[PERMUTED_FLAGS_KEY]
        best_predictor_names.append(this_dict[BEST_PREDICTOR_KEY])

        this_best_cost_array = this_dict[BEST_COST_ARRAY_KEY]
        this_best_cost_matrix = numpy.reshape(
            this_best_cost_array, (1, len(this_best_cost_array))
        )
        best_cost_matrix = numpy.concatenate(
            (best_cost_matrix, this_best_cost_matrix), axis=0
        )

        if step_num == 1:
            step1_predictor_names = this_dict[PERMUTED_PREDICTORS_KEY]
            step1_cost_matrix = this_dict[PERMUTED_COST_MATRIX_KEY]

    return {
        BEST_PREDICTORS_KEY: best_predictor_names,
        BEST_COST_MATRIX_KEY: best_cost_matrix,
        ORIGINAL_COST_ARRAY_KEY: original_cost_array,
        STEP1_PREDICTORS_KEY: step1_predictor_names,
        STEP1_COST_MATRIX_KEY: step1_cost_matrix
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
