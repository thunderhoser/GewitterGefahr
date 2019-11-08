"""Deals with permutation test for GewitterGefahr models."""

import copy
import numpy
from gewittergefahr.gg_utils import radar_utils
from gewittergefahr.gg_utils import soundings
from gewittergefahr.gg_utils import error_checking
from gewittergefahr.deep_learning import cnn
from gewittergefahr.deep_learning import permutation_utils
from gewittergefahr.deep_learning import training_validation_io as trainval_io
from gewittergefahr.plotting import radar_plotting

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'
MINOR_SEPARATOR_STRING = '\n\n' + '-' * 50 + '\n\n'

DEFAULT_NUM_BOOTSTRAP_REPS = 1

PREDICTOR_MATRICES_KEY = permutation_utils.PREDICTOR_MATRICES_KEY
PERMUTED_FLAGS_KEY = permutation_utils.PERMUTED_FLAGS_KEY
PERMUTED_PREDICTORS_KEY = permutation_utils.PERMUTED_PREDICTORS_KEY
PERMUTED_COST_MATRIX_KEY = permutation_utils.PERMUTED_COST_MATRIX_KEY
UNPERMUTED_PREDICTORS_KEY = permutation_utils.UNPERMUTED_PREDICTORS_KEY
UNPERMUTED_COST_MATRIX_KEY = permutation_utils.UNPERMUTED_COST_MATRIX_KEY
BEST_PREDICTOR_KEY = permutation_utils.BEST_PREDICTOR_KEY
BEST_COST_ARRAY_KEY = permutation_utils.BEST_COST_ARRAY_KEY


def create_nice_predictor_names(
        predictor_matrices, cnn_metadata_dict, separate_radar_heights=False):
    """Creates list of nice (human-readable) predictor names for each matrix.

    T = number of input tensors to the CNN

    :param predictor_matrices: See doc for `run_forward_test` or
        `run_backwards_test`.
    :param cnn_metadata_dict: Same.
    :param separate_radar_heights: Same.
    :return: predictor_names_by_matrix: length-T list, where the [i]th element
        is a list of predictor names for the [i]th matrix.
    """

    error_checking.assert_is_boolean(separate_radar_heights)

    myrorss_2d3d = cnn_metadata_dict[cnn.CONV_2D3D_KEY]
    training_option_dict = cnn_metadata_dict[cnn.TRAINING_OPTION_DICT_KEY]
    upsample_refl = (
        myrorss_2d3d and
        training_option_dict[trainval_io.UPSAMPLE_REFLECTIVITY_KEY]
    )

    layer_operation_dicts = cnn_metadata_dict[cnn.LAYER_OPERATIONS_KEY]
    radar_field_names = training_option_dict[trainval_io.RADAR_FIELDS_KEY]
    radar_heights_m_agl = training_option_dict[trainval_io.RADAR_HEIGHTS_KEY]

    nice_radar_field_names = [
        radar_utils.field_name_to_verbose(field_name=f, include_units=False)
        for f in radar_field_names
    ]

    num_matrices = len(predictor_matrices)
    predictor_names_by_matrix = [[]] * num_matrices

    if upsample_refl:
        # TODO(thunderhoser): This hacky bullshit is terrible.  Nets that
        # upsample reflectivity, should take reflectivity in a separate tensor
        # and then concat the reflectivity and az-shear tensors before the first
        # conv layer.

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

        predictor_names_by_matrix[0] += nice_radar_field_names

    elif myrorss_2d3d:
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


def prediction_function_2d_cnn(model_object, list_of_input_matrices):
    """Prediction function for CNN with 2 spatial dimensions.

    T = number of input tensors to the model
    E = number of examples
    K = number of classes

    :param model_object: Trained model (instance of `keras.models.Model` or
        `keras.models.Sequential`).
    :param list_of_input_matrices: length-T list of predictor matrices (numpy
        arrays), in the order used for training.  The first axis of each array
        should have length E.
    :return: class_probability_matrix: E-by-K numpy array of class
        probabilities.
    """

    if len(list_of_input_matrices) == 2:
        sounding_matrix = list_of_input_matrices[1]
    else:
        sounding_matrix = None

    return cnn.apply_2d_or_3d_cnn(
        model_object=model_object, radar_image_matrix=list_of_input_matrices[0],
        sounding_matrix=sounding_matrix, verbose=True)


def prediction_function_3d_cnn(model_object, list_of_input_matrices):
    """Prediction function for CNN with 3 spatial dimensions.

    :param model_object: See doc for `prediction_function_2d_cnn`.
    :param list_of_input_matrices: Same.
    :return: class_probability_matrix: Same.
    """

    if len(list_of_input_matrices) == 2:
        sounding_matrix = list_of_input_matrices[1]
    else:
        sounding_matrix = None

    return cnn.apply_2d_or_3d_cnn(
        model_object=model_object, radar_image_matrix=list_of_input_matrices[0],
        sounding_matrix=sounding_matrix, verbose=True)


def prediction_function_2d3d_cnn(model_object, list_of_input_matrices):
    """Prediction function for hybrid 2-and-3-D CNN.

    :param model_object: See doc for `prediction_function_2d_cnn`.
    :param list_of_input_matrices: Same.
    :return: class_probability_matrix: Same.
    """

    num_input_matrices = len(list_of_input_matrices)
    upsample_refl = not (
        num_input_matrices > 1 and len(list_of_input_matrices[1].shape) == 4
    )

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


def run_forward_test(
        model_object, predictor_matrices, target_values, cnn_metadata_dict,
        cost_function, separate_radar_heights=False,
        num_bootstrap_reps=DEFAULT_NUM_BOOTSTRAP_REPS):
    """Runs forward permutation test.

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
    result_dict["best_predictor_names"]: length-P list of best predictors.
        The [j]th element is the name of the [j]th predictor to be permanently
        permuted.
    result_dict["best_cost_matrix"]: P-by-B numpy array of costs after
        permutation.
    result_dict["original_cost_array"]: length-B numpy array of costs
        before permutation.
    result_dict["step1_predictor_names"]: length-P list of predictors in
        the order that they were permuted in step 1.
    result_dict["step1_cost_matrix"]: P-by-B numpy array of costs after
        permutation in step 1.
    result_dict["backwards_test"]: Boolean flag (always False).
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
    print(MINOR_SEPARATOR_STRING)

    original_cost_array = permutation_utils.bootstrap_cost(
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

        this_dict = permutation_utils.run_forward_test_one_step(
            model_object=model_object, predictor_matrices=predictor_matrices,
            predictor_names_by_matrix=predictor_names_by_matrix,
            target_values=target_values,
            prediction_function=prediction_function,
            cost_function=cost_function,
            separate_heights=separate_radar_heights,
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
        permutation_utils.BEST_PREDICTORS_KEY: best_predictor_names,
        permutation_utils.BEST_COST_MATRIX_KEY: best_cost_matrix,
        permutation_utils.ORIGINAL_COST_ARRAY_KEY: original_cost_array,
        permutation_utils.STEP1_PREDICTORS_KEY: step1_predictor_names,
        permutation_utils.STEP1_COST_MATRIX_KEY: step1_cost_matrix,
        permutation_utils.BACKWARDS_FLAG: False
    }


def run_backwards_test(
        model_object, predictor_matrices, target_values, cnn_metadata_dict,
        cost_function, separate_radar_heights=False,
        num_bootstrap_reps=DEFAULT_NUM_BOOTSTRAP_REPS):
    """Runs backwards permutation test.

    P = number of predictors to unpermute
    B = number of bootstrap replicates

    :param model_object: See doc for `run_forward_test`.
    :param predictor_matrices: Same.
    :param target_values: Same.
    :param cnn_metadata_dict: Same.
    :param cost_function: Same.
    :param separate_radar_heights: Same.
    :param num_bootstrap_reps: Same.
    :return: result_dict: Dictionary with the following keys.
    result_dict["best_predictor_names"]: length-P list of best
        predictors.  The [j]th element is the name of the [j]th predictor to be
        permanently unpermuted.
    result_dict["best_cost_matrix"]: P-by-B numpy array of costs after
        unpermutation.
    result_dict["original_cost_array"]: length-B numpy array of costs
        before unpermutation.
    result_dict["step1_predictor_names"]: length-P list of predictors in
        the order that they were unpermuted in step 1.
    result_dict["step1_cost_matrix"]: P-by-B numpy array of costs after
        unpermutation in step 1.
    result_dict["backwards_test"]: Boolean flag (always True).
    """

    # Deal with input args.
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

    # Permute all predictors.
    num_matrices = len(predictor_matrices)
    clean_predictor_matrices = copy.deepcopy(predictor_matrices)

    for i in range(num_matrices):
        this_num_predictors = len(predictor_names_by_matrix[i])

        for j in range(this_num_predictors):
            predictor_matrices = permutation_utils.permute_one_predictor(
                predictor_matrices=predictor_matrices,
                separate_heights=separate_radar_heights,
                matrix_index=i, predictor_index=j
            )[0]

    # Find original cost (before unpermutation).
    print('Finding original cost (before unpermutation)...')
    class_probability_matrix = prediction_function(
        model_object, predictor_matrices)
    print(MINOR_SEPARATOR_STRING)

    original_cost_array = permutation_utils.bootstrap_cost(
        target_values=target_values,
        class_probability_matrix=class_probability_matrix,
        cost_function=cost_function, num_replicates=num_bootstrap_reps)

    # Do the dirty work.
    permuted_flags_by_matrix = [
        numpy.full(len(n), 1, dtype=bool)
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

        this_dict = permutation_utils.run_backwards_test_one_step(
            model_object=model_object, predictor_matrices=predictor_matrices,
            clean_predictor_matrices=clean_predictor_matrices,
            predictor_names_by_matrix=predictor_names_by_matrix,
            target_values=target_values,
            prediction_function=prediction_function,
            cost_function=cost_function,
            separate_heights=separate_radar_heights,
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
            step1_predictor_names = this_dict[UNPERMUTED_PREDICTORS_KEY]
            step1_cost_matrix = this_dict[UNPERMUTED_COST_MATRIX_KEY]

    return {
        permutation_utils.BEST_PREDICTORS_KEY: best_predictor_names,
        permutation_utils.BEST_COST_MATRIX_KEY: best_cost_matrix,
        permutation_utils.ORIGINAL_COST_ARRAY_KEY: original_cost_array,
        permutation_utils.STEP1_PREDICTORS_KEY: step1_predictor_names,
        permutation_utils.STEP1_COST_MATRIX_KEY: step1_cost_matrix,
        permutation_utils.BACKWARDS_FLAG: True
    }
