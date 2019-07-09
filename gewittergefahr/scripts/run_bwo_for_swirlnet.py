"""Runs backwards optimization on Swirlnet model."""

import argparse
import numpy
from keras import backend as K
from keras.models import load_model as load_keras_model
from gewittergefahr.deep_learning import model_interpretation
from gewittergefahr.deep_learning import backwards_optimization as backwards_opt

# random.seed(6695)
# numpy.random.seed(6695)

K.set_session(K.tf.Session(config=K.tf.ConfigProto(
    intra_op_parallelism_threads=1, inter_op_parallelism_threads=1,
    allow_soft_placement=False
)))

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

SWIRLNET_FIELD_MEANS = numpy.array([20.745745, -0.718525, 1.929636])
SWIRLNET_FIELD_STANDARD_DEVIATIONS = numpy.array([
    17.947071, 4.343980, 4.969537
])

CLASS_COMPONENT_TYPE_STRING = model_interpretation.CLASS_COMPONENT_TYPE_STRING
NEURON_COMPONENT_TYPE_STRING = model_interpretation.NEURON_COMPONENT_TYPE_STRING
CHANNEL_COMPONENT_TYPE_STRING = (
    model_interpretation.CHANNEL_COMPONENT_TYPE_STRING)

MODEL_FILE_ARG_NAME = 'input_model_file_name'
INIT_FUNCTION_ARG_NAME = 'init_function_name'
COMPONENT_TYPE_ARG_NAME = 'component_type_string'
TARGET_CLASS_ARG_NAME = 'target_class'
LAYER_NAME_ARG_NAME = 'layer_name'
NEURON_INDICES_ARG_NAME = 'neuron_indices'
CHANNEL_INDEX_ARG_NAME = 'channel_index'
IDEAL_ACTIVATION_ARG_NAME = 'ideal_activation'
NUM_ITERATIONS_ARG_NAME = 'num_iterations'
LEARNING_RATE_ARG_NAME = 'learning_rate'
OUTPUT_FILE_ARG_NAME = 'output_file_name'

MODEL_FILE_HELP_STRING = (
    'Path to file with trained Swirlnet model.  Will be read by '
    '`keras.models.load_model`.')

INIT_FUNCTION_HELP_STRING = (
    'Initialization function (used to create initial input matrices for '
    'gradient descent).  Must be accepted by '
    '`backwards_opt.check_init_function`.')

COMPONENT_HELP_STRING = (
    'Determines model component for which activation will be maximized.  See '
    '`model_interpretation.check_component_metadata` for details.')

IDEAL_ACTIVATION_HELP_STRING = (
    '[used only if {0:s} = "{1:s}" or "{2:s}"] See '
    '`backwards_opt.optimize_input_for_neuron` or '
    '`backwards_opt.optimize_input_for_channel` for details.'
).format(COMPONENT_TYPE_ARG_NAME, NEURON_COMPONENT_TYPE_STRING,
         CLASS_COMPONENT_TYPE_STRING)

NUM_ITERATIONS_HELP_STRING = 'Number of iterations for backwards optimization.'

LEARNING_RATE_HELP_STRING = 'Learning rate for backwards optimization.'

OUTPUT_FILE_HELP_STRING = (
    'Path to output file (will be written by '
    '`backwards_opt.write_standard_file`).')

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + MODEL_FILE_ARG_NAME, type=str, required=True,
    help=MODEL_FILE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + INIT_FUNCTION_ARG_NAME, type=str, required=False,
    default=backwards_opt.CONSTANT_INIT_FUNCTION_NAME,
    help=INIT_FUNCTION_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + COMPONENT_TYPE_ARG_NAME, type=str, required=True,
    help=COMPONENT_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + TARGET_CLASS_ARG_NAME, type=int, required=False, default=-1,
    help=COMPONENT_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + LAYER_NAME_ARG_NAME, type=str, required=False, default='',
    help=COMPONENT_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + NEURON_INDICES_ARG_NAME, type=int, nargs='+', required=False,
    default=[-1], help=COMPONENT_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + CHANNEL_INDEX_ARG_NAME, type=int, required=False, default=-1,
    help=COMPONENT_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + IDEAL_ACTIVATION_ARG_NAME, type=float, required=False,
    default=backwards_opt.DEFAULT_IDEAL_ACTIVATION,
    help=IDEAL_ACTIVATION_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + NUM_ITERATIONS_ARG_NAME, type=int, required=False,
    default=backwards_opt.DEFAULT_NUM_ITERATIONS,
    help=NUM_ITERATIONS_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + LEARNING_RATE_ARG_NAME, type=float, required=False,
    default=backwards_opt.DEFAULT_LEARNING_RATE,
    help=LEARNING_RATE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_FILE_ARG_NAME, type=str, required=True,
    help=OUTPUT_FILE_HELP_STRING)


def _brier_score_keras(observation_tensor, class_probability_tensor):
    """Returns Brier score.

    E = number of examples
    K = number of target classes

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
        (observation_tensor - K.mean(observation_tensor)) ** 2
    )

    return (
        1. -
        _brier_score_keras(observation_tensor, class_probability_tensor) /
        uncertainty_tensor
    )


def _denormalize_data(input_matrix):
    """Denormalizes Swirlnet input data.

    E = number of examples
    M = number of rows in grid
    M = number of columns in grid
    F = number of radar fields

    :param input_matrix: E-by-M-by-N-by-F numpy array (normalized).
    :return: input_matrix: Same as input but denormalized.
    """

    num_fields = input_matrix.shape[-1]

    for j in range(num_fields):
        input_matrix[..., j] = (
            SWIRLNET_FIELD_MEANS[j] +
            input_matrix[..., j] * SWIRLNET_FIELD_STANDARD_DEVIATIONS[j]
        )

    return input_matrix


def _create_initializer(init_function_name):
    """Creates initializer function.

    :param init_function_name: See documentation at top of file.
    :return: init_function: Initializer function.
    """

    if init_function_name == backwards_opt.CONSTANT_INIT_FUNCTION_NAME:
        return backwards_opt.create_constant_initializer(0.)

    if init_function_name == backwards_opt.UNIFORM_INIT_FUNCTION_NAME:
        return backwards_opt.create_uniform_random_initializer(
            min_value=-1., max_value=1.)

    return backwards_opt.create_gaussian_initializer(
        mean=0., standard_deviation=1.)


def _run(model_file_name, init_function_name, component_type_string,
         target_class, layer_name, neuron_indices, channel_index,
         ideal_activation, num_iterations, learning_rate, output_file_name):
    """Runs backwards optimization on a trained CNN.

    This is effectively the main method.

    :param model_file_name: See documentation at top of file.
    :param init_function_name: Same.
    :param component_type_string: Same.
    :param target_class: Same.
    :param layer_name: Same.
    :param neuron_indices: Same.
    :param channel_index: Same.
    :param ideal_activation: Same.
    :param num_iterations: Same.
    :param learning_rate: Same.
    :param output_file_name: Same.
    """

    model_interpretation.check_component_type(component_type_string)
    if ideal_activation <= 0:
        ideal_activation = None

    print('Reading model from: "{0:s}"...'.format(model_file_name))
    custom_dict = {'brier_skill_score_keras': _brier_skill_score_keras}
    model_object = load_keras_model(model_file_name, custom_objects=custom_dict)

    init_function = _create_initializer(init_function_name)
    print(SEPARATOR_STRING)

    if component_type_string == CLASS_COMPONENT_TYPE_STRING:
        print('Optimizing image for target class {0:d}...'.format(target_class))

        list_of_optimized_matrices, initial_activation, final_activation = (
            backwards_opt.optimize_input_for_class(
                model_object=model_object, target_class=target_class,
                init_function_or_matrices=init_function,
                num_iterations=num_iterations, learning_rate=learning_rate)
        )

    elif component_type_string == NEURON_COMPONENT_TYPE_STRING:
        print('Optimizing image for neuron {0:s} in layer "{1:s}"...'.format(
            str(neuron_indices), layer_name
        ))

        list_of_optimized_matrices, initial_activation, final_activation = (
            backwards_opt.optimize_input_for_neuron(
                model_object=model_object, layer_name=layer_name,
                neuron_indices=neuron_indices,
                init_function_or_matrices=init_function,
                num_iterations=num_iterations, learning_rate=learning_rate,
                ideal_activation=ideal_activation)
        )

    else:
        print('Optimizing image for channel {0:d} in layer "{1:s}"...'.format(
            channel_index, layer_name))

        list_of_optimized_matrices, initial_activation, final_activation = (
            backwards_opt.optimize_input_for_channel(
                model_object=model_object, layer_name=layer_name,
                channel_index=channel_index,
                init_function_or_matrices=init_function,
                stat_function_for_neuron_activations=K.max,
                num_iterations=num_iterations, learning_rate=learning_rate,
                ideal_activation=ideal_activation)
        )

    print(SEPARATOR_STRING)

    print('Denormalizing optimized examples...')
    list_of_optimized_matrices[0] = _denormalize_data(
        list_of_optimized_matrices[0]
    )

    print('Writing results to: "{0:s}"...'.format(output_file_name))
    backwards_opt.write_standard_file(
        pickle_file_name=output_file_name,
        list_of_optimized_matrices=list_of_optimized_matrices,
        initial_activations=numpy.array([initial_activation]),
        final_activations=numpy.array([final_activation]),
        model_file_name=model_file_name,
        init_function_name_or_matrices=init_function_name,
        num_iterations=num_iterations, learning_rate=learning_rate,
        component_type_string=component_type_string, target_class=target_class,
        layer_name=layer_name, neuron_indices=neuron_indices,
        channel_index=channel_index, ideal_activation=ideal_activation)


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        model_file_name=getattr(INPUT_ARG_OBJECT, MODEL_FILE_ARG_NAME),
        init_function_name=getattr(INPUT_ARG_OBJECT, INIT_FUNCTION_ARG_NAME),
        component_type_string=getattr(
            INPUT_ARG_OBJECT, COMPONENT_TYPE_ARG_NAME),
        target_class=getattr(INPUT_ARG_OBJECT, TARGET_CLASS_ARG_NAME),
        layer_name=getattr(INPUT_ARG_OBJECT, LAYER_NAME_ARG_NAME),
        neuron_indices=numpy.array(
            getattr(INPUT_ARG_OBJECT, NEURON_INDICES_ARG_NAME), dtype=int),
        channel_index=getattr(INPUT_ARG_OBJECT, CHANNEL_INDEX_ARG_NAME),
        ideal_activation=getattr(INPUT_ARG_OBJECT, IDEAL_ACTIVATION_ARG_NAME),
        num_iterations=getattr(INPUT_ARG_OBJECT, NUM_ITERATIONS_ARG_NAME),
        learning_rate=getattr(INPUT_ARG_OBJECT, LEARNING_RATE_ARG_NAME),
        output_file_name=getattr(INPUT_ARG_OBJECT, OUTPUT_FILE_ARG_NAME)
    )
