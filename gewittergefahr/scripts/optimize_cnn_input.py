"""Finds optimal input for the given class, neurons, or channels of a CNN.

CNN = convolutional neural network
"""

import os.path
import argparse
import numpy
from keras import backend as K
import keras.models
from gewittergefahr.gg_utils import general_utils
from gewittergefahr.gg_utils import soundings
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.gg_utils import error_checking
from gewittergefahr.deep_learning import cnn
from gewittergefahr.deep_learning import deep_learning_utils as dl_utils
from gewittergefahr.deep_learning import model_interpretation
from gewittergefahr.deep_learning import feature_optimization
from gewittergefahr.deep_learning import training_validation_io as trainval_io

K.set_session(K.tf.Session(config=K.tf.ConfigProto(
    intra_op_parallelism_threads=1, inter_op_parallelism_threads=1
)))

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

CLASS_COMPONENT_TYPE_STRING = model_interpretation.CLASS_COMPONENT_TYPE_STRING
NEURON_COMPONENT_TYPE_STRING = model_interpretation.NEURON_COMPONENT_TYPE_STRING
CHANNEL_COMPONENT_TYPE_STRING = (
    model_interpretation.CHANNEL_COMPONENT_TYPE_STRING)

SWIRLNET_FIELD_MEANS = numpy.array([20.745745, -0.718525, 1.929636])
SWIRLNET_FIELD_STANDARD_DEVIATIONS = numpy.array(
    [17.947071, 4.343980, 4.969537])
VALID_SWIRLNET_FUNCTION_NAMES = [
    feature_optimization.GAUSSIAN_INIT_FUNCTION_NAME,
    feature_optimization.UNIFORM_INIT_FUNCTION_NAME,
    feature_optimization.CONSTANT_INIT_FUNCTION_NAME
]

SOUNDING_HEIGHTS_M_AGL = soundings.DEFAULT_HEIGHT_LEVELS_M_AGL + 0
VALID_GG_FUNCTION_NAMES = (
    VALID_SWIRLNET_FUNCTION_NAMES +
    [feature_optimization.CLIMO_INIT_FUNCTION_NAME]
)

MODEL_FILE_ARG_NAME = 'model_file_name'
IS_SWIRLNET_ARG_NAME = 'is_model_swirlnet'
COMPONENT_TYPE_ARG_NAME = 'component_type_string'
TARGET_CLASS_ARG_NAME = 'target_class'
NUM_ITERATIONS_ARG_NAME = 'num_iterations'
LEARNING_RATE_ARG_NAME = 'learning_rate'
INIT_FUNCTION_ARG_NAME = 'init_function_name'
LAYER_NAME_ARG_NAME = 'layer_name'
IDEAL_ACTIVATION_ARG_NAME = 'ideal_activation'
NEURON_INDICES_ARG_NAME = 'neuron_indices'
CHANNEL_INDICES_ARG_NAME = 'channel_indices'
OUTPUT_FILE_ARG_NAME = 'output_file_name'

MODEL_FILE_HELP_STRING = (
    'Path to input file, containing trained CNN.  Will be read by '
    '`cnn.read_model`.')

IS_SWIRLNET_HELP_STRING = (
    'Boolean flag.  If 1, this script will assume that `{0:s}` contains a '
    'Swirlnet model from D.J. Gagne.  If 0, will assume that `{0:s}` contains a'
    ' GewitterGefahr model.  This determines how the model will be read, and '
    'the wrong assumption will cause the script to crash.'
).format(MODEL_FILE_ARG_NAME)

COMPONENT_TYPE_HELP_STRING = (
    'Component type.  Input data may be optimized for class prediction, neuron '
    'activation, or channel activation.  Valid options are listed below.\n{0:s}'
).format(str(model_interpretation.VALID_COMPONENT_TYPE_STRINGS))

TARGET_CLASS_HELP_STRING = (
    '[used only if {0:s} = "{1:s}"] Input data will be optimized for prediction'
    ' of class k, where k = `{2:s}`.'
).format(COMPONENT_TYPE_ARG_NAME, CLASS_COMPONENT_TYPE_STRING,
         TARGET_CLASS_ARG_NAME)

NUM_ITERATIONS_HELP_STRING = 'Number of iterations for optimization procedure.'

LEARNING_RATE_HELP_STRING = 'Learning rate for optimization procedure.'

INIT_FUNCTION_HELP_STRING = (
    'Initialization function, used to initialize model inputs before '
    'optimization.  Must be in the following list.\n{0:s}'
).format(str(VALID_GG_FUNCTION_NAMES))

LAYER_NAME_HELP_STRING = (
    '[used only if {0:s} = "{1:s}" or "{2:s}"] Name of layer with neuron or '
    'channel whose activation is to be maximized.'
).format(COMPONENT_TYPE_ARG_NAME, NEURON_COMPONENT_TYPE_STRING,
         CLASS_COMPONENT_TYPE_STRING)

IDEAL_ACTIVATION_HELP_STRING = (
    '[used only if {0:s} = "{1:s}" or "{2:s}"] The loss function will be '
    '(neuron_activation - ideal_activation)**2 or [max(channel_activations) - '
    'ideal_activation]**2.  If {3:s} = -1, the loss function will be '
    '-sign(neuron_activation) * neuron_activation**2 or '
    '-sign(max(channel_activations)) * max(channel_activations)**2.'
).format(COMPONENT_TYPE_ARG_NAME, NEURON_COMPONENT_TYPE_STRING,
         CHANNEL_COMPONENT_TYPE_STRING, IDEAL_ACTIVATION_ARG_NAME)

NEURON_INDICES_HELP_STRING = (
    '[used only if {0:s} = "{1:s}"] Indices for each neuron whose activation is'
    ' to be maximized.  For example, to maximize activation for neuron '
    '(0, 0, 2), this argument should be "0 0 2".  To maximize activations for '
    'neurons (0, 0, 2) and (1, 1, 2), this list should be "0 0 2 -1 1 1 2".  In'
    ' other words, use -1 to separate neurons.'
).format(COMPONENT_TYPE_ARG_NAME, NEURON_COMPONENT_TYPE_STRING)

CHANNEL_INDICES_HELP_STRING = (
    '[used only if {0:s} = "{1:s}"] Index for each channel whose activation is '
    'to be maximized.'
).format(COMPONENT_TYPE_ARG_NAME, CHANNEL_COMPONENT_TYPE_STRING)

OUTPUT_FILE_HELP_STRING = (
    'Path to output file (will be written by '
    '`feature_optimization.write_file`).')

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + MODEL_FILE_ARG_NAME, type=str, required=True,
    help=MODEL_FILE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + IS_SWIRLNET_ARG_NAME, type=int, required=True,
    help=IS_SWIRLNET_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + COMPONENT_TYPE_ARG_NAME, type=str, required=True,
    help=COMPONENT_TYPE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + TARGET_CLASS_ARG_NAME, type=int, required=False, default=1,
    help=TARGET_CLASS_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + NUM_ITERATIONS_ARG_NAME, type=int, required=False,
    default=feature_optimization.DEFAULT_NUM_ITERATIONS,
    help=NUM_ITERATIONS_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + LEARNING_RATE_ARG_NAME, type=float, required=False,
    default=feature_optimization.DEFAULT_LEARNING_RATE,
    help=LEARNING_RATE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + INIT_FUNCTION_ARG_NAME, type=str, required=False,
    default=feature_optimization.CLIMO_INIT_FUNCTION_NAME,
    help=INIT_FUNCTION_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + LAYER_NAME_ARG_NAME, type=str, required=False, default='',
    help=LAYER_NAME_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + NEURON_INDICES_ARG_NAME, type=int, nargs='+', required=False,
    default=[-1], help=NEURON_INDICES_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + IDEAL_ACTIVATION_ARG_NAME, type=float, required=False,
    default=feature_optimization.DEFAULT_IDEAL_ACTIVATION,
    help=IDEAL_ACTIVATION_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + CHANNEL_INDICES_ARG_NAME, type=int, nargs='+', required=False,
    default=[-1], help=CHANNEL_INDICES_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_FILE_ARG_NAME, type=str, required=True,
    help=OUTPUT_FILE_HELP_STRING)


def _brier_score_keras(observation_tensor, class_probability_tensor):
    """Returns Brier score.

    E = number of examples
    K = number of output classes

    :param observation_tensor: E-by-K tensor of observed classes.  If
        observation_tensor[i, k] = 1, the [i]th example belongs to the [k]th
        class.
    :param class_probability_tensor: E-by-K tensor of forecast probabilities.
        class_probability_tensor[i, k] = forecast probability that the [i]th
        example belongs to the [k]th class.
    :return: brier_score: Brier score.
    """

    return K.mean((class_probability_tensor - observation_tensor) ** 2)


def _brier_skill_score_keras(observation_tensor, class_probability_tensor):
    """Returns Brier skill score.

    :param observation_tensor: See doc for `brier_score_keras`.
    :param class_probability_tensor: Same.
    :return: brier_skill_score: Brier skill score.
    """

    uncertainty_tensor = K.mean(
        (observation_tensor - K.mean(observation_tensor)) ** 2)
    return (
        1. -
        _brier_score_keras(observation_tensor, class_probability_tensor) /
        uncertainty_tensor
    )


def _denormalize_swirlnet_data(input_matrix):
    """Denormalizes input data for a Swirlnet model.

    E = number of examples
    M = number of grid rows
    N = number of grid columns
    F = number of radar fields

    :param input_matrix: E-by-M-by-N-by-F numpy array.
    :return: input_matrix: Denormalized version of input (same dimensions).
    """

    num_fields = input_matrix.shape[-1]
    for j in range(num_fields):
        input_matrix[..., j] = (
            SWIRLNET_FIELD_MEANS[j] +
            input_matrix[..., j] * SWIRLNET_FIELD_STANDARD_DEVIATIONS[j]
        )

    return input_matrix


def _check_init_function(init_function_name, is_model_swirlnet):
    """Ensures that initialization function is valid.

    :param init_function_name: See documentation at top of file.
    :param is_model_swirlnet: Same.
    :raises: ValueError: if init function is not in the accepted list.
    """

    if is_model_swirlnet:
        valid_function_names = VALID_SWIRLNET_FUNCTION_NAMES
    else:
        valid_function_names = VALID_GG_FUNCTION_NAMES

    if init_function_name not in valid_function_names:
        error_string = (
            '\n\n{0:s}\nValid init functions (listed above) do not include '
            '"{1:s}".'
        ).format(str(valid_function_names), init_function_name)
        raise ValueError(error_string)


def _create_swirlnet_initializer(init_function_name):
    """Creates initialization function for Swirlnet model.

    :param init_function_name: See documentation at top of file.
    :return: init_function: Initialization function.
    """

    if init_function_name == feature_optimization.CONSTANT_INIT_FUNCTION_NAME:
        return feature_optimization.create_constant_initializer(0.)

    if init_function_name == feature_optimization.UNIFORM_INIT_FUNCTION_NAME:
        return feature_optimization.create_uniform_random_initializer(
            min_value=-1., max_value=1.)

    return feature_optimization.create_gaussian_initializer(
        mean=0., standard_deviation=1.)


def _create_gg_initializer(init_function_name, model_file_name):
    """Creates initialization function for GewitterGefahr model.

    :param init_function_name: See documentation at top of file.
    :param model_file_name: Same.
    :return: init_function: Initialization function.
    """

    metadata_file_name = '{0:s}/model_metadata.p'.format(
        os.path.split(model_file_name)[0])

    print 'Reading metadata from: "{0:s}"...'.format(metadata_file_name)
    model_metadata_dict = cnn.read_model_metadata(metadata_file_name)
    training_option_dict = model_metadata_dict[cnn.TRAINING_OPTION_DICT_KEY]

    used_minmax_norm = (
        training_option_dict[trainval_io.NORMALIZATION_TYPE_KEY] ==
        dl_utils.MINMAX_NORMALIZATION_TYPE_STRING
    )

    if init_function_name == feature_optimization.CONSTANT_INIT_FUNCTION_NAME:
        if used_minmax_norm:
            return feature_optimization.create_constant_initializer(
                (training_option_dict[trainval_io.MAX_NORMALIZED_VALUE_KEY] -
                 training_option_dict[trainval_io.MIN_NORMALIZED_VALUE_KEY])
                / 2)

        return feature_optimization.create_constant_initializer(0.)

    if init_function_name == feature_optimization.UNIFORM_INIT_FUNCTION_NAME:
        if used_minmax_norm:
            return feature_optimization.create_uniform_random_initializer(
                min_value=training_option_dict[
                    trainval_io.MIN_NORMALIZED_VALUE_KEY],
                max_value=training_option_dict[
                    trainval_io.MAX_NORMALIZED_VALUE_KEY])

        return feature_optimization.create_uniform_random_initializer(
            min_value=-3., max_value=3.)

    if init_function_name == feature_optimization.GAUSSIAN_INIT_FUNCTION_NAME:
        if used_minmax_norm:
            return feature_optimization.create_gaussian_initializer(
                mean=
                (training_option_dict[trainval_io.MAX_NORMALIZED_VALUE_KEY] -
                 training_option_dict[trainval_io.MIN_NORMALIZED_VALUE_KEY])
                / 2,
                standard_deviation=
                (training_option_dict[trainval_io.MAX_NORMALIZED_VALUE_KEY] -
                 training_option_dict[trainval_io.MIN_NORMALIZED_VALUE_KEY]) / 6
            )

        return feature_optimization.create_gaussian_initializer(
            mean=0., standard_deviation=1.)

    return feature_optimization.create_climo_initializer(
        training_option_dict=training_option_dict,
        myrorss_2d3d=model_metadata_dict[cnn.USE_2D3D_CONVOLUTION_KEY])


def _run(
        model_file_name, is_model_swirlnet, component_type_string, target_class,
        num_iterations, learning_rate, init_function_name, layer_name,
        ideal_activation, neuron_indices_flattened, channel_indices,
        output_file_name):
    """Finds optimal input for one class, neuron, or channel of a CNN.

    This is effectively the main method.

    :param model_file_name: See documentation at top of file.
    :param is_model_swirlnet: Same.
    :param component_type_string: Same.
    :param target_class: Same.
    :param num_iterations: Same.
    :param learning_rate: Same.
    :param init_function_name: Same.
    :param layer_name: Same.
    :param ideal_activation: Same.
    :param neuron_indices_flattened: Same.
    :param channel_indices: Same.
    :param output_file_name: Same.
    """

    # Check input args.
    _check_init_function(init_function_name=init_function_name,
                         is_model_swirlnet=is_model_swirlnet)
    model_interpretation.check_component_type(component_type_string)

    if ideal_activation <= 0:
        ideal_activation = None

    if component_type_string == NEURON_COMPONENT_TYPE_STRING:
        neuron_indices_flattened = neuron_indices_flattened.astype(float)
        neuron_indices_flattened[neuron_indices_flattened < 0] = numpy.nan

        neuron_indices_2d_list = general_utils.split_array_by_nan(
            neuron_indices_flattened)
        neuron_index_matrix = numpy.array(neuron_indices_2d_list, dtype=int)
    else:
        neuron_index_matrix = None

    if component_type_string == CHANNEL_COMPONENT_TYPE_STRING:
        error_checking.assert_is_geq_numpy_array(channel_indices, 0)

    file_system_utils.mkdir_recursive_if_necessary(file_name=output_file_name)

    # Read model.
    print 'Reading model from: "{0:s}"...'.format(model_file_name)

    if is_model_swirlnet:
        custom_dict = {'brier_skill_score_keras': _brier_skill_score_keras}
        model_object = keras.models.load_model(
            model_file_name, custom_objects=custom_dict)

        init_function = _create_swirlnet_initializer(init_function_name)
    else:
        model_object = cnn.read_model(model_file_name)
        init_function = _create_gg_initializer(
            init_function_name=init_function_name,
            model_file_name=model_file_name)

    # Do feature optimization.
    print SEPARATOR_STRING
    list_of_optimized_input_matrices = None

    if component_type_string == CLASS_COMPONENT_TYPE_STRING:
        print 'Optimizing inputs for target class {0:d}...'.format(
            target_class)

        list_of_optimized_input_matrices = (
            feature_optimization.optimize_input_for_class(
                model_object=model_object, target_class=target_class,
                init_function=init_function, num_iterations=num_iterations,
                learning_rate=learning_rate)
        )

    elif component_type_string == NEURON_COMPONENT_TYPE_STRING:
        for j in range(neuron_index_matrix.shape[0]):
            print (
                'Optimizing inputs for neuron {0:s} in layer "{1:s}"...'
            ).format(str(neuron_index_matrix[j, :]), layer_name)

            these_matrices = (
                feature_optimization.optimize_input_for_neuron_activation(
                    model_object=model_object, layer_name=layer_name,
                    neuron_indices=neuron_index_matrix[j, :],
                    init_function=init_function, num_iterations=num_iterations,
                    learning_rate=learning_rate,
                    ideal_activation=ideal_activation)
            )

            if list_of_optimized_input_matrices is None:
                list_of_optimized_input_matrices = these_matrices + []
            else:
                for k in range(len(list_of_optimized_input_matrices)):
                    list_of_optimized_input_matrices[k] = numpy.concatenate(
                        (list_of_optimized_input_matrices[k],
                         these_matrices[k]), axis=0)

    else:
        for this_channel_index in channel_indices:
            print (
                'Optimizing inputs for channel {0:d} in layer "{1:s}"...'
            ).format(this_channel_index, layer_name)

            these_matrices = (
                feature_optimization.optimize_input_for_channel_activation(
                    model_object=model_object, layer_name=layer_name,
                    channel_index=this_channel_index,
                    init_function=init_function,
                    stat_function_for_neuron_activations=K.max,
                    num_iterations=num_iterations, learning_rate=learning_rate,
                    ideal_activation=ideal_activation)
            )

            if list_of_optimized_input_matrices is None:
                list_of_optimized_input_matrices = these_matrices + []
            else:
                for k in range(len(list_of_optimized_input_matrices)):
                    list_of_optimized_input_matrices[k] = numpy.concatenate(
                        (list_of_optimized_input_matrices[k],
                         these_matrices[k]), axis=0)

    print SEPARATOR_STRING
    print 'Writing optimized input matrices to file: "{0:s}"...'.format(
        output_file_name)

    feature_optimization.write_file(
        pickle_file_name=output_file_name,
        list_of_optimized_input_matrices=list_of_optimized_input_matrices,
        model_file_name=model_file_name, num_iterations=num_iterations,
        learning_rate=learning_rate, init_function_name=init_function_name,
        component_type_string=component_type_string, target_class=target_class,
        layer_name=layer_name, ideal_activation=ideal_activation,
        neuron_index_matrix=neuron_index_matrix,
        channel_indices=channel_indices)


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        model_file_name=getattr(INPUT_ARG_OBJECT, MODEL_FILE_ARG_NAME),
        is_model_swirlnet=bool(getattr(INPUT_ARG_OBJECT, IS_SWIRLNET_ARG_NAME)),
        component_type_string=getattr(
            INPUT_ARG_OBJECT, COMPONENT_TYPE_ARG_NAME),
        target_class=getattr(INPUT_ARG_OBJECT, TARGET_CLASS_ARG_NAME),
        num_iterations=getattr(INPUT_ARG_OBJECT, NUM_ITERATIONS_ARG_NAME),
        learning_rate=getattr(INPUT_ARG_OBJECT, LEARNING_RATE_ARG_NAME),
        init_function_name=getattr(INPUT_ARG_OBJECT, INIT_FUNCTION_ARG_NAME),
        layer_name=getattr(INPUT_ARG_OBJECT, LAYER_NAME_ARG_NAME),
        ideal_activation=getattr(INPUT_ARG_OBJECT, IDEAL_ACTIVATION_ARG_NAME),
        neuron_indices_flattened=numpy.array(
            getattr(INPUT_ARG_OBJECT, NEURON_INDICES_ARG_NAME), dtype=int),
        channel_indices=numpy.array(
            getattr(INPUT_ARG_OBJECT, CHANNEL_INDICES_ARG_NAME), dtype=int),
        output_file_name=getattr(INPUT_ARG_OBJECT, OUTPUT_FILE_ARG_NAME)
    )
