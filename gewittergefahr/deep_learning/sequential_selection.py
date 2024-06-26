"""Methods for running sequential selection.

Sequential (forward or backward) selection is explained in Section xxx of Webb (yyyy).
"""

import copy
import pickle
import numpy
import keras.utils
import sklearn.base
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.gg_utils import error_checking

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

DEFAULT_NUM_STEPS_FOR_LOSS_DECREASE = 5

MIN_DECREASE_KEY = 'min_loss_decrease'
MIN_PERCENT_DECREASE_KEY = 'min_percentage_loss_decrease'
NUM_STEPS_FOR_DECREASE_KEY = 'num_steps_for_loss_decrease'
SELECTED_PREDICTORS_KEY = 'selected_predictor_name_by_step'
LOWEST_COSTS_KEY = 'lowest_cost_by_step'


def _check_input_args(
        list_of_training_matrices, training_target_values,
        list_of_validation_matrices, validation_target_values,
        predictor_names_by_matrix):
    """Error-checks input arguments for sequential selection.

    N = number of input matrices
    T = number of training examples
    V = number of validation examples
    C_q = number of channels (predictors) in the [q]th matrix

    :param list_of_training_matrices: length-N list of matrices (numpy arrays).
        The first axis of each matrix should have length T.
    :param training_target_values: length-T numpy array of target values
        (integer class labels).
    :param list_of_validation_matrices: length-N list of numpy arrays.  The
        first axis of each matrix should have length V; otherwise,
        list_of_validation_matrices[q] should have the same dimensions as
        list_of_training_matrices[q].
    :param validation_target_values: length-V numpy array of target values
        (integer class labels).
    :param predictor_names_by_matrix: length-N list of lists.  The [q]th list
        should be a list of predictor variables in the [q]th matrix, with length
        C_q.
    :raises: ValueError: if length of `list_of_training_matrices` != length of
        `list_of_validation_matrices`.
    :raises: ValueError: if length of `list_of_training_matrices` != length of
        `predictor_names_by_matrix`.
    :raises: ValueError: if any input matrix has < 3 dimensions.
    """

    error_checking.assert_is_integer_numpy_array(training_target_values)
    error_checking.assert_is_geq_numpy_array(training_target_values, 0)
    error_checking.assert_is_integer_numpy_array(validation_target_values)
    error_checking.assert_is_geq_numpy_array(validation_target_values, 0)

    num_input_matrices = len(list_of_training_matrices)

    if len(list_of_validation_matrices) != num_input_matrices:
        error_string = (
            'Number of training matrices ({0:d}) should equal number of '
            'validation matrices ({1:d}).'
        ).format(num_input_matrices, len(list_of_validation_matrices))

        raise ValueError(error_string)

    if len(predictor_names_by_matrix) != num_input_matrices:
        error_string = (
            'Number of predictor-name lists ({0:d}) should equal number of '
            'validation matrices ({1:d}).'
        ).format(num_input_matrices, len(predictor_names_by_matrix))

        raise ValueError(error_string)

    num_training_examples = len(training_target_values)
    num_validation_examples = len(validation_target_values)

    for q in range(num_input_matrices):
        error_checking.assert_is_numpy_array_without_nan(
            list_of_training_matrices[q])
        error_checking.assert_is_numpy_array_without_nan(
            list_of_validation_matrices[q])

        this_num_dimensions = len(list_of_training_matrices[q].shape)
        if this_num_dimensions < 3:
            error_string = (
                '{0:d}th training matrix has {1:d} dimensions.  Should have at '
                'least 3.'
            ).format(q + 1, this_num_dimensions)

            raise ValueError(error_string)

        this_num_dimensions = len(list_of_validation_matrices[q].shape)
        if this_num_dimensions < 3:
            error_string = (
                '{0:d}th validation matrix has {1:d} dimensions.  Should have '
                'at least 3.'
            ).format(q + 1, this_num_dimensions)

            raise ValueError(error_string)

        error_checking.assert_is_string_list(predictor_names_by_matrix[q])
        this_num_predictors = len(predictor_names_by_matrix[q])

        these_expected_dimensions = (
            (num_training_examples,) +
            list_of_training_matrices[q].shape[1:-1] +
            (this_num_predictors,)
        )
        these_expected_dimensions = numpy.array(
            these_expected_dimensions, dtype=int)

        error_checking.assert_is_numpy_array(
            list_of_training_matrices[q],
            exact_dimensions=these_expected_dimensions)

        these_expected_dimensions = (
            (num_validation_examples,) +
            list_of_validation_matrices[q].shape[1:-1] +
            (this_num_predictors,)
        )
        these_expected_dimensions = numpy.array(
            these_expected_dimensions, dtype=int)

        error_checking.assert_is_numpy_array(
            list_of_validation_matrices[q],
            exact_dimensions=these_expected_dimensions)


def _subset_input_matrices(
        list_of_training_matrices, list_of_validation_matrices,
        predictor_names_by_matrix, desired_predictor_names):
    """Subsets input matrices (retaining only the desired predictors).

    :param list_of_training_matrices: See doc for `_check_input_args`.
    :param list_of_validation_matrices: Same.
    :param predictor_names_by_matrix: Same.
    :param desired_predictor_names: 1-D list with names of desired predictors.
    :return: desired_training_matrices: Same as input
        `list_of_training_matrices`, except that the last axis (channel axis) of
        each matrix may be shorter.
    :return: desired_validation_matrices: Same as input
        `list_of_validation_matrices`, except that the last axis (channel axis)
        of each matrix may be shorter.
    """

    num_input_matrices = len(list_of_training_matrices)
    desired_indices_by_matrix = (
        [numpy.array([], dtype=int)] * num_input_matrices
    )

    for this_predictor_name in desired_predictor_names:
        for q in range(num_input_matrices):
            if this_predictor_name not in predictor_names_by_matrix[q]:
                continue

            this_index = predictor_names_by_matrix[q].index(this_predictor_name)
            desired_indices_by_matrix[q] = numpy.concatenate((
                desired_indices_by_matrix[q], numpy.array([this_index])
            ))

    desired_training_matrices = [None] * num_input_matrices
    desired_validation_matrices = [None] * num_input_matrices

    for q in range(num_input_matrices):
        desired_training_matrices[q] = list_of_training_matrices[q][
            ..., desired_indices_by_matrix[q]]
        desired_validation_matrices[q] = list_of_validation_matrices[q][
            ..., desired_indices_by_matrix[q]]

    return desired_training_matrices, desired_validation_matrices


def _eval_sfs_stopping_criterion(
        min_loss_decrease, min_percentage_loss_decrease,
        num_steps_for_loss_decrease, lowest_cost_by_step):
    """Evaluates stopping criterion for sequential forward selection (SFS).

    :param min_loss_decrease: If the loss has decreased by less than
        `min_loss_decrease` over the last `num_steps_for_loss_decrease` steps,
        the algorithm will stop.
    :param min_percentage_loss_decrease:
        [used only if `min_loss_decrease is None`]
        If the loss has decreased by less than `min_percentage_loss_decrease`
        over the last `num_steps_for_loss_decrease` steps, the algorithm will
        stop.
    :param num_steps_for_loss_decrease: See above.
    :param lowest_cost_by_step: 1-D numpy array, where the [i]th value is the
        cost after the [i]th step.  The last step is the current one, so the
        current cost is lowest_cost_by_step[-1].
    :return: stopping_criterion: Boolean flag.
    :raises: ValueError: if both `min_loss_decrease` and
        `min_percentage_loss_decrease` are None.
    """

    if min_loss_decrease is None and min_percentage_loss_decrease is None:
        raise ValueError('Either min_loss_decrease or '
                         'min_percentage_loss_decrease must be specified.')

    if min_loss_decrease is None:
        error_checking.assert_is_greater(min_percentage_loss_decrease, 0.)
        error_checking.assert_is_less_than(min_percentage_loss_decrease, 100.)
    else:
        min_percentage_loss_decrease = None
        error_checking.assert_is_greater(min_loss_decrease, 0.)

    error_checking.assert_is_integer(num_steps_for_loss_decrease)
    error_checking.assert_is_greater(num_steps_for_loss_decrease, 0)

    if len(lowest_cost_by_step) <= num_steps_for_loss_decrease:
        return False

    previous_loss = lowest_cost_by_step[-(num_steps_for_loss_decrease + 1)]
    if min_loss_decrease is None:
        min_loss_decrease = previous_loss * min_percentage_loss_decrease / 100

    max_new_loss = previous_loss - min_loss_decrease

    print((
        'Previous loss ({0:d} steps ago) = {1:.4e} ... minimum loss decrease = '
        '{2:.4e} ... thus, max new loss = {3:.4e} ... actual new loss = {4:.4e}'
    ).format(
        num_steps_for_loss_decrease, previous_loss, min_loss_decrease,
        max_new_loss, lowest_cost_by_step[-1]
    ))

    return lowest_cost_by_step[-1] > max_new_loss


def create_training_function(num_training_examples_per_batch, num_epochs):
    """Creates training function.

    This function satisfies the requirements for the input arg
    `training_function` to `run_sfs`.

    :param num_training_examples_per_batch: Number of training examples per
        weight update.
    :param num_epochs: Number of epochs (cycles through all training examples).
    :return: training_function: Function (see below).
    """

    error_checking.assert_is_integer(num_training_examples_per_batch)
    error_checking.assert_is_geq(num_training_examples_per_batch, 32)
    error_checking.assert_is_integer(num_epochs)
    error_checking.assert_is_geq(num_epochs, 3)

    def training_function(
            model_object, list_of_training_matrices, training_target_values,
            list_of_validation_matrices, validation_target_values):
        """Trains CNN with the given data.

        :param model_object: Untrained instance of `keras.models.Model` or
            `keras.models.Sequential`.
        :param list_of_training_matrices: See doc for `_check_input_args`.
        :param training_target_values: Same.
        :param list_of_validation_matrices: Same.
        :param validation_target_values: Same.
        :return: history_object: Instance of `keras.callbacks.History`.
        """

        num_output_neurons = (
            model_object.layers[-1].output.get_shape().as_list()[-1]
        )

        if num_output_neurons == 1:
            training_target_array = training_target_values + 0
            validation_target_array = validation_target_values + 0
        else:
            training_target_array = keras.utils.to_categorical(
                y=training_target_values, num_classes=num_output_neurons)
            validation_target_array = keras.utils.to_categorical(
                y=validation_target_values, num_classes=num_output_neurons)

        return model_object.fit(
            x=list_of_training_matrices, y=training_target_array,
            batch_size=num_training_examples_per_batch,
            epochs=num_epochs, verbose=1,
            validation_data=(list_of_validation_matrices,
                             validation_target_array),
            shuffle=False, initial_epoch=0)

    return training_function


def run_sfs(
        list_of_training_matrices, training_target_values,
        list_of_validation_matrices, validation_target_values,
        predictor_names_by_matrix, model_builder, training_function,
        min_loss_decrease=None, min_percentage_loss_decrease=None,
        num_steps_for_loss_decrease=DEFAULT_NUM_STEPS_FOR_LOSS_DECREASE):
    """Runs sequential forward selection (SFS).

    :param list_of_training_matrices: See doc for `_check_input_args`.
    :param training_target_values: Same.
    :param list_of_validation_matrices: Same.
    :param validation_target_values: Same.
    :param predictor_names_by_matrix: Same.
    :param model_builder: Function used to create model architecture for the
        given number of predictors.  Should have the following inputs and
        outputs.
    Input: list_of_input_matrices: Same as input `list_of_training_matrices` or
        `list_of_validation_matrices` to this method, except that in general the
        last axis (channel axis) of each matrix will be shorter.
    Output: model_object: Untrained instance of `keras.models.Model` or
        `keras.models.Sequential`.

    :param training_function: Function used to train model.  Should have the
        following inputs and outputs.
    Input: model_object: Model created by `model_builder`.
    Input: list_of_training_matrices: See doc for `_check_input_args`.
    Input: training_target_values: Same.
    Input: list_of_validation_matrices: Same.
    Input: validation_target_values: Same.
    Output: history_object: Instance of `keras.callbacks.History`.

    :param min_loss_decrease: Used to determine stopping criterion.  If the loss
        has decreased by less than `min_loss_decrease` over the last
        `num_steps_for_loss_decrease` steps of sequential selection, the
        algorithm will stop.
    :param min_percentage_loss_decrease:
        [used only if `min_loss_decrease is None`]
        Used to determine stopping criterion.  If the loss has decreased by less
        than `min_percentage_loss_decrease` over the last
        `num_steps_for_loss_decrease` steps of sequential selection, the
        algorithm will stop.
    :param num_steps_for_loss_decrease: See above.

    :return: result_dict: Dictionary with the following keys.  P = number of
        predictors in final model.
    result_dict['min_loss_decrease']: Same as input arg (to save metadata).
    result_dict['min_percentage_loss_decrease']: Same as input arg (to save
        metadata).
    result_dict['num_steps_for_loss_decrease']: Same as input arg (to save
        metadata).
    result_dict['selected_predictor_name_by_step']: length-P list with name of
        predictor selected at each step.
    result_dict['lowest_cost_by_step']: length-P numpy array with corresponding
        cost at each step.
    """

    _check_input_args(
        list_of_training_matrices=list_of_training_matrices,
        training_target_values=training_target_values,
        list_of_validation_matrices=list_of_validation_matrices,
        validation_target_values=validation_target_values,
        predictor_names_by_matrix=predictor_names_by_matrix)

    remaining_predictor_names_by_matrix = copy.deepcopy(
        predictor_names_by_matrix)

    selected_predictor_name_by_step = []
    lowest_cost_by_step = []

    step_num = 0
    num_input_matrices = len(list_of_training_matrices)

    while True:
        print('\n')
        step_num += 1

        lowest_cost = numpy.inf
        best_matrix_index = None
        best_predictor_name = None

        any_predictors_left = False

        for q in range(num_input_matrices):
            if len(remaining_predictor_names_by_matrix[q]) == 0:
                continue

            for this_predictor_name in remaining_predictor_names_by_matrix[q]:
                any_predictors_left = True

                print((
                    'Trying predictor "{0:s}" at step {1:d} of SFS... '
                ).format(this_predictor_name, step_num))
                print(SEPARATOR_STRING)

                these_training_matrices, these_validation_matrices = (
                    _subset_input_matrices(
                        list_of_training_matrices=list_of_training_matrices,
                        list_of_validation_matrices=list_of_validation_matrices,
                        predictor_names_by_matrix=predictor_names_by_matrix,
                        desired_predictor_names=
                        selected_predictor_name_by_step + [this_predictor_name]
                    )
                )

                this_model_object = model_builder(these_training_matrices)

                this_history_object = training_function(
                    model_object=this_model_object,
                    list_of_training_matrices=these_training_matrices,
                    training_target_values=training_target_values,
                    list_of_validation_matrices=these_validation_matrices,
                    validation_target_values=validation_target_values)
                print(SEPARATOR_STRING)

                this_cost = numpy.nanmin(
                    this_history_object.history['val_loss']
                )

                print('Validation loss after adding "{0:s}" = {1:.4e}'.format(
                    this_predictor_name, this_cost))
                print(SEPARATOR_STRING)

                if this_cost > lowest_cost:
                    continue

                lowest_cost = this_cost + 0.
                best_matrix_index = q + 0
                best_predictor_name = this_predictor_name + ''

        if not any_predictors_left:
            break

        stopping_criterion = _eval_sfs_stopping_criterion(
            min_loss_decrease=min_loss_decrease,
            min_percentage_loss_decrease=min_percentage_loss_decrease,
            num_steps_for_loss_decrease=num_steps_for_loss_decrease,
            lowest_cost_by_step=lowest_cost_by_step + [lowest_cost])
        if stopping_criterion:
            break

        selected_predictor_name_by_step.append(best_predictor_name)
        lowest_cost_by_step.append(lowest_cost)

        remaining_predictor_names_by_matrix[best_matrix_index].remove(
            best_predictor_name)

        print('Best predictor = "{0:s}" ... new cost = {1:.4e}'.format(
            best_predictor_name, lowest_cost))

    return {
        MIN_DECREASE_KEY: min_loss_decrease,
        MIN_PERCENT_DECREASE_KEY: min_percentage_loss_decrease,
        NUM_STEPS_FOR_DECREASE_KEY: num_steps_for_loss_decrease,
        SELECTED_PREDICTORS_KEY: selected_predictor_name_by_step,
        LOWEST_COSTS_KEY: lowest_cost_by_step
    }


def run_sfs_on_sklearn_model(
        training_predictor_matrix, training_target_values,
        validation_predictor_matrix, validation_target_values, predictor_names,
        model_object, cost_function, min_loss_decrease=None,
        min_percentage_loss_decrease=None,
        num_steps_for_loss_decrease=DEFAULT_NUM_STEPS_FOR_LOSS_DECREASE):
    """Runs sequential forward selection (SFS) on scikit-learn model.

    T = number of training examples
    V = number of validation examples
    P = number of predictors

    :param training_predictor_matrix: T-by-P numpy array of predictor values.
    :param training_target_values: length-T numpy array of target values
        (integer class labels, since this method supports only classification).
    :param validation_predictor_matrix: V-by-P numpy array of predictor values.
    :param validation_target_values: length-V numpy array of target values.
    :param predictor_names: length-P list with names of predictor variables.
    :param model_object: Instance of scikit-learn model.  Must implement the
        methods `fit` and `predict_proba`.

    :param cost_function: Cost function (used to assess model on validation
        data).  Should have the following inputs and outputs.
    Input: target_values: Same as input `validation_target_values` for this
        method.
    Input: class_probability_matrix: V-by-K matrix of class probabilities, where
        K = number of classes.  class_probability_matrix[i, k] is the predicted
        probability that the [i]th example belongs to the [k]th class.
    Output: cost: Scalar value.

    :param min_loss_decrease: Used to determine stopping criterion.  If the loss
        has decreased by less than `min_loss_decrease` over the last
        `num_steps_for_loss_decrease` steps of sequential selection, the
        algorithm will stop.
    :param min_percentage_loss_decrease:
        [used only if `min_loss_decrease is None`]
        Used to determine stopping criterion.  If the loss has decreased by less
        than `min_percentage_loss_decrease` over the last
        `num_steps_for_loss_decrease` steps of sequential selection, the
        algorithm will stop.
    :param num_steps_for_loss_decrease: See above.

    :return: result_dict: See documentation for `run_sfs`.
    """

    # TODO(thunderhoser): This method does not involve deep learning, so
    # shouldn't really be in this file.

    # Check input args.
    error_checking.assert_is_numpy_array_without_nan(training_predictor_matrix)
    error_checking.assert_is_numpy_array(
        training_predictor_matrix, num_dimensions=2)

    num_training_examples = training_predictor_matrix.shape[0]
    num_predictors = training_predictor_matrix.shape[1]

    error_checking.assert_is_integer_numpy_array(training_target_values)
    error_checking.assert_is_geq_numpy_array(training_target_values, 0)
    error_checking.assert_is_numpy_array(
        training_target_values,
        exact_dimensions=numpy.array([num_training_examples])
    )

    error_checking.assert_is_numpy_array_without_nan(
        validation_predictor_matrix)
    num_validation_examples = validation_predictor_matrix.shape[0]
    error_checking.assert_is_numpy_array(
        validation_predictor_matrix,
        exact_dimensions=numpy.array([num_validation_examples, num_predictors])
    )

    error_checking.assert_is_integer_numpy_array(validation_target_values)
    error_checking.assert_is_geq_numpy_array(validation_target_values, 0)
    error_checking.assert_is_numpy_array(
        validation_target_values,
        exact_dimensions=numpy.array([num_validation_examples])
    )

    error_checking.assert_is_string_list(predictor_names)
    error_checking.assert_is_numpy_array(
        numpy.array(predictor_names),
        exact_dimensions=numpy.array([num_predictors])
    )

    # Create climatological model.
    num_classes = 1 + max(
        [numpy.max(training_target_values), numpy.max(validation_target_values)]
    )

    climo_validation_prob_matrix = numpy.full(
        (num_validation_examples, num_classes), numpy.nan)
    for k in range(num_classes):
        climo_validation_prob_matrix[..., k] = numpy.mean(
            training_target_values == k)

    climo_cost = cost_function(validation_target_values,
                               climo_validation_prob_matrix)

    print('Cost of climatological model: {0:.4e}\n'.format(climo_cost))

    # Do dirty work.
    remaining_predictor_names = predictor_names + []
    selected_predictor_name_by_step = []
    lowest_cost_by_step = []

    step_num = 0

    while len(remaining_predictor_names) > 0:
        print('\n')
        step_num += 1

        lowest_cost = numpy.inf
        best_predictor_name = None

        for this_predictor_name in remaining_predictor_names:
            print((
                'Trying predictor "{0:s}" at step {1:d} of SFS... '
            ).format(this_predictor_name, step_num))

            these_indices = [
                predictor_names.index(s)
                for s in selected_predictor_name_by_step
            ]
            these_indices.append(predictor_names.index(this_predictor_name))
            these_indices = numpy.array(these_indices, dtype=int)

            this_training_matrix = training_predictor_matrix[..., these_indices]
            this_validation_matrix = validation_predictor_matrix[
                ..., these_indices]

            new_model_object = sklearn.base.clone(model_object)
            new_model_object.fit(this_training_matrix, training_target_values)

            this_validation_prob_matrix = new_model_object.predict_proba(
                this_validation_matrix)
            this_cost = cost_function(validation_target_values,
                                      this_validation_prob_matrix)

            print('Validation loss after adding "{0:s}" = {1:.4e}\n'.format(
                this_predictor_name, this_cost))

            if this_cost > lowest_cost:
                continue

            lowest_cost = this_cost + 0.
            best_predictor_name = this_predictor_name + ''

        stopping_criterion = _eval_sfs_stopping_criterion(
            min_loss_decrease=min_loss_decrease,
            min_percentage_loss_decrease=min_percentage_loss_decrease,
            num_steps_for_loss_decrease=num_steps_for_loss_decrease,
            lowest_cost_by_step=lowest_cost_by_step + [lowest_cost])

        if stopping_criterion:
            break

        selected_predictor_name_by_step.append(best_predictor_name)
        lowest_cost_by_step.append(lowest_cost)
        remaining_predictor_names.remove(best_predictor_name)

        print('Best predictor = "{0:s}" ... new cost = {1:.4e}'.format(
            best_predictor_name, lowest_cost))
        print(SEPARATOR_STRING)

    return {
        MIN_DECREASE_KEY: min_loss_decrease,
        MIN_PERCENT_DECREASE_KEY: min_percentage_loss_decrease,
        NUM_STEPS_FOR_DECREASE_KEY: num_steps_for_loss_decrease,
        SELECTED_PREDICTORS_KEY: selected_predictor_name_by_step,
        LOWEST_COSTS_KEY: lowest_cost_by_step
    }


def write_results(result_dict, pickle_file_name):
    """Writes results to Pickle file.

    :param result_dict: Dictionary created by `run_sfs`.
    :param pickle_file_name: Path to output file.
    """

    file_system_utils.mkdir_recursive_if_necessary(file_name=pickle_file_name)
    pickle_file_handle = open(pickle_file_name, 'wb')
    pickle.dump(result_dict, pickle_file_handle)
    pickle_file_handle.close()


def read_results(pickle_file_name):
    """Reads results from Pickle file.

    :param pickle_file_name: Path to input file.
    :return: result_dict: Dictionary created by `run_sfs`.
    """

    pickle_file_handle = open(pickle_file_name, 'rb')
    result_dict = pickle.load(pickle_file_handle)
    pickle_file_handle.close()

    return result_dict
