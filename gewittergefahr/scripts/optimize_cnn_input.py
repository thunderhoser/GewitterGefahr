"""Finds optimal input for the given class, neurons, or channels of a CNN.

CNN = convolutional neural network
"""

import copy
import os.path
import argparse
import numpy
from keras import backend as K
import keras.models
from gewittergefahr.gg_utils import general_utils
from gewittergefahr.gg_utils import radar_utils
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.gg_utils import error_checking
from gewittergefahr.deep_learning import cnn
from gewittergefahr.deep_learning import storm_images
from gewittergefahr.deep_learning import model_interpretation
from gewittergefahr.deep_learning import feature_optimization
from gewittergefahr.deep_learning import deep_learning_utils as dl_utils

# TODO(thunderhoser): Allow optimization with more than one initialization.
# TODO(thunderhoser): Screw with initialization in general.

K.set_session(K.tf.Session(config=K.tf.ConfigProto(
    intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)))

MINOR_SEPARATOR_STRING = '\n\n' + '-' * 50 + '\n\n'

SWIRLNET_FIELD_MEANS = numpy.array([20.745745, -0.718525, 1.929636])
SWIRLNET_FIELD_STANDARD_DEVIATIONS = numpy.array(
    [17.947071, 4.343980, 4.969537])

MODEL_FILE_ARG_NAME = 'model_file_name'
IS_SWIRLNET_ARG_NAME = 'is_model_swirlnet'
COMPONENT_TYPE_ARG_NAME = 'component_type_string'
TARGET_CLASS_ARG_NAME = 'target_class'
OPTIMIZE_FOR_PROBABILITY_ARG_NAME = 'optimize_for_probability'
NUM_ITERATIONS_ARG_NAME = 'num_iterations'
LEARNING_RATE_ARG_NAME = 'learning_rate'
IDEAL_LOGIT_ARG_NAME = 'ideal_logit'
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
).format(COMPONENT_TYPE_ARG_NAME,
         model_interpretation.CLASS_COMPONENT_TYPE_STRING,
         TARGET_CLASS_ARG_NAME)
OPTIMIZE_FOR_PROBABILITY_HELP_STRING = (
    '[used only if {0:s} = "{1:s}"] Boolean flag.  If 1, input data will be '
    'optimized for the predicted probability of class `{2:s}`.  If 0, input '
    'data will be optimized for the pre-softmax logit of class `{2:s}`.'
).format(COMPONENT_TYPE_ARG_NAME,
         model_interpretation.CLASS_COMPONENT_TYPE_STRING,
         TARGET_CLASS_ARG_NAME)
NUM_ITERATIONS_HELP_STRING = 'Number of iterations for optimization procedure.'
LEARNING_RATE_HELP_STRING = 'Learning rate for optimization procedure.'
IDEAL_LOGIT_HELP_STRING = (
    '[used only if {0:s} = "{1:s}" and {2:s} = 0] The loss function will be '
    '(logit[k] - {3:s}) ** 2, where logit[k] is the logit for the target class.'
    '  If {3:s} = -1, the loss function will be -sign(logit[k]) * logit[k]**2, '
    'or the negative signed square of logit[k], so that loss always decreases '
    'as logit[k] increases.'
).format(COMPONENT_TYPE_ARG_NAME,
         model_interpretation.CLASS_COMPONENT_TYPE_STRING,
         OPTIMIZE_FOR_PROBABILITY_ARG_NAME, IDEAL_LOGIT_ARG_NAME)
LAYER_NAME_HELP_STRING = (
    '[used only if {0:s} = "{1:s}" or "{2:s}"] Name of layer with neuron or '
    'channel whose activation is to be maximized.'
).format(COMPONENT_TYPE_ARG_NAME,
         model_interpretation.NEURON_COMPONENT_TYPE_STRING,
         model_interpretation.CLASS_COMPONENT_TYPE_STRING)
IDEAL_ACTIVATION_HELP_STRING = (
    '[used only if {0:s} = "{1:s}" or "{2:s}"] The loss function will be '
    '(neuron_activation - ideal_activation)**2 or [max(channel_activations) - '
    'ideal_activation]**2.  If {3:s} = -1, the loss function will be '
    '-sign(neuron_activation) * neuron_activation**2 or '
    '-sign(max(channel_activations)) * max(channel_activations)**2.'
).format(COMPONENT_TYPE_ARG_NAME,
         model_interpretation.NEURON_COMPONENT_TYPE_STRING,
         model_interpretation.CHANNEL_COMPONENT_TYPE_STRING,
         IDEAL_ACTIVATION_ARG_NAME)
NEURON_INDICES_HELP_STRING = (
    '[used only if {0:s} = "{1:s}"] Indices for each neuron whose activation is'
    ' to be maximized.  For example, to maximize activation for neuron '
    '(0, 0, 2), this argument should be "0 0 2".  To maximize activations for '
    'neurons (0, 0, 2) and (1, 1, 2), this list should be "0 0 2 -1 1 1 2".  In'
    ' other words, use -1 to separate neurons.'
).format(COMPONENT_TYPE_ARG_NAME,
         model_interpretation.NEURON_COMPONENT_TYPE_STRING)
CHANNEL_INDICES_HELP_STRING = (
    '[used only if {0:s} = "{1:s}"] Index for each channel whose activation is '
    'to be maximized.'
).format(COMPONENT_TYPE_ARG_NAME,
         model_interpretation.CHANNEL_COMPONENT_TYPE_STRING)
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
    '--' + OPTIMIZE_FOR_PROBABILITY_ARG_NAME, type=int, required=False,
    default=1, help=OPTIMIZE_FOR_PROBABILITY_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + NUM_ITERATIONS_ARG_NAME, type=int, required=False,
    default=feature_optimization.DEFAULT_NUM_ITERATIONS,
    help=NUM_ITERATIONS_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + LEARNING_RATE_ARG_NAME, type=float, required=False,
    default=feature_optimization.DEFAULT_LEARNING_RATE,
    help=LEARNING_RATE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + IDEAL_LOGIT_ARG_NAME, type=float, required=False,
    default=feature_optimization.DEFAULT_IDEAL_LOGIT,
    help=IDEAL_LOGIT_HELP_STRING)

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

    climatology_tensor = K.mean(
        (observation_tensor - K.mean(observation_tensor)) ** 2)
    return (
        1. -
        _brier_score_keras(observation_tensor, class_probability_tensor) /
        climatology_tensor
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
            input_matrix[..., j] * SWIRLNET_FIELD_STANDARD_DEVIATIONS[j])

    return input_matrix


def _denormalize_gg_data(list_of_input_matrices, model_metadata_dict):
    """Denormalizes input data for a GewitterGefahr model.

    :param list_of_input_matrices: length-T list of input matrices (numpy
        arrays), where T = number of input tensors to the model.
    :param model_metadata_dict: Dictionary with metadata for the GewitterGefahr
        model, created by `cnn.read_model_metadata`.
    :return: list_of_input_matrices: Denormalized version of input (same
        dimensions).
    """

    if model_metadata_dict[cnn.USE_2D3D_CONVOLUTION_KEY]:
        radar_field_names = model_metadata_dict[cnn.RADAR_FIELD_NAMES_KEY]
        azimuthal_shear_indices = numpy.where(numpy.array(
            [f in radar_utils.SHEAR_NAMES for f in radar_field_names]))[0]
        azimuthal_shear_field_names = [
            radar_field_names[j] for j in azimuthal_shear_indices]

        print 'Denormalizing reflectivity fields...'
        list_of_input_matrices[0] = dl_utils.denormalize_radar_images(
            radar_image_matrix=list_of_input_matrices[0],
            field_names=[radar_utils.REFL_NAME],
            normalization_dict=model_metadata_dict[
                cnn.RADAR_NORMALIZATION_DICT_KEY])

        print 'Denormalizing azimuthal-shear fields...'
        list_of_input_matrices[1] = dl_utils.denormalize_radar_images(
            radar_image_matrix=list_of_input_matrices[1],
            field_names=azimuthal_shear_field_names,
            normalization_dict=model_metadata_dict[
                cnn.RADAR_NORMALIZATION_DICT_KEY])
    else:
        radar_file_name_matrix = model_metadata_dict[
            cnn.TRAINING_FILE_NAMES_KEY]
        num_channels = radar_file_name_matrix.shape[1]
        field_name_by_channel = [''] * num_channels

        for j in range(num_channels):
            if len(radar_file_name_matrix.shape) == 3:
                field_name_by_channel[j] = (
                    storm_images.image_file_name_to_field(
                        radar_file_name_matrix[0, j, 0]))
            else:
                field_name_by_channel[j] = (
                    storm_images.image_file_name_to_field(
                        radar_file_name_matrix[0, j]))

        print 'Denormalizing radar fields...'
        list_of_input_matrices[0] = dl_utils.denormalize_radar_images(
            radar_image_matrix=list_of_input_matrices[0],
            field_names=field_name_by_channel,
            normalization_dict=model_metadata_dict[
                cnn.RADAR_NORMALIZATION_DICT_KEY])

    if model_metadata_dict[cnn.SOUNDING_FIELD_NAMES_KEY] is not None:
        print 'Denormalizing soundings...'
        list_of_input_matrices[-1] = dl_utils.denormalize_soundings(
            sounding_matrix=list_of_input_matrices[-1],
            pressureless_field_names=model_metadata_dict[
                cnn.SOUNDING_FIELD_NAMES_KEY],
            normalization_dict=model_metadata_dict[
                cnn.SOUNDING_NORMALIZATION_DICT_KEY])

    return list_of_input_matrices


def _run(
        model_file_name, is_model_swirlnet, component_type_string,
        target_class, optimize_for_probability, num_iterations, learning_rate,
        ideal_logit, layer_name, ideal_activation, neuron_indices_flattened,
        channel_indices, output_file_name):
    """Finds optimal input for one class, neuron, or channel of a CNN.

    This is effectively the main method.

    :param model_file_name: See documentation at top of file.
    :param is_model_swirlnet: Same.
    :param component_type_string: Same.
    :param target_class: Same.
    :param optimize_for_probability: Same.
    :param num_iterations: Same.
    :param learning_rate: Same.
    :param ideal_logit: Same.
    :param layer_name: Same.
    :param ideal_activation: Same.
    :param neuron_indices_flattened: Same.
    :param channel_indices: Same.
    :param output_file_name: Same.
    """

    # Check input args.
    print num_iterations

    file_system_utils.mkdir_recursive_if_necessary(file_name=output_file_name)
    model_interpretation.check_component_type(component_type_string)
    if ideal_logit <= 0:
        ideal_logit = None
    if ideal_activation <= 0:
        ideal_activation = None

    if (component_type_string ==
            model_interpretation.NEURON_COMPONENT_TYPE_STRING):
        neuron_indices_flattened = neuron_indices_flattened.astype(float)
        neuron_indices_flattened[neuron_indices_flattened < 0] = numpy.nan

        neuron_indices_2d_list = general_utils.split_array_by_nan(
            neuron_indices_flattened)
        neuron_index_matrix = numpy.array(neuron_indices_2d_list, dtype=int)
    else:
        neuron_index_matrix = None

    if (component_type_string ==
            model_interpretation.CHANNEL_COMPONENT_TYPE_STRING):
        error_checking.assert_is_geq_numpy_array(channel_indices, 0)

    # Read model.
    print 'Reading model from: "{0:s}"...'.format(model_file_name)
    if is_model_swirlnet:
        model_object = keras.models.load_model(
            model_file_name,
            custom_objects={
                'brier_skill_score_keras': _brier_skill_score_keras})
        init_function = feature_optimization.create_constant_initializer(0.)
    else:
        model_object = cnn.read_model(model_file_name)
        init_function = feature_optimization.create_constant_initializer(0.5)

        metadata_file_name = '{0:s}/model_metadata.p'.format(
            os.path.split(model_file_name)[0])
        print 'Reading metadata from: "{0:s}"...'.format(metadata_file_name)
        model_metadata_dict = cnn.read_model_metadata(metadata_file_name)

    # Do feature optimization.
    print MINOR_SEPARATOR_STRING
    list_of_optimized_input_matrices = None

    if (component_type_string ==
            model_interpretation.CLASS_COMPONENT_TYPE_STRING):

        print '\nOptimizing inputs for target class {0:d}...'.format(
            target_class)
        list_of_optimized_input_matrices = (
            feature_optimization.optimize_input_for_class(
                model_object=model_object, target_class=target_class,
                optimize_for_probability=optimize_for_probability,
                init_function=init_function, num_iterations=num_iterations,
                learning_rate=learning_rate, ideal_logit=ideal_logit))

    elif (component_type_string ==
          model_interpretation.NEURON_COMPONENT_TYPE_STRING):

        for j in range(neuron_index_matrix.shape[0]):
            print (
                '\nOptimizing inputs for neuron {0:s} in layer "{1:s}"...'
            ).format(str(neuron_index_matrix[j, :]), layer_name)

            these_matrices = (
                feature_optimization.optimize_input_for_neuron_activation(
                    model_object=model_object, layer_name=layer_name,
                    neuron_indices=neuron_index_matrix[j, :],
                    init_function=init_function, num_iterations=num_iterations,
                    learning_rate=learning_rate,
                    ideal_activation=ideal_activation))

            if list_of_optimized_input_matrices is None:
                list_of_optimized_input_matrices = copy.deepcopy(these_matrices)
            else:
                for k in range(len(list_of_optimized_input_matrices)):
                    list_of_optimized_input_matrices[k] = numpy.concatenate(
                        (list_of_optimized_input_matrices[k],
                         these_matrices[k]), axis=0)

    else:
        for this_channel_index in channel_indices:
            print (
                '\nOptimizing inputs for channel {0:d} in layer "{1:s}"...'
            ).format(this_channel_index, layer_name)

            these_matrices = (
                feature_optimization.optimize_input_for_channel_activation(
                    model_object=model_object, layer_name=layer_name,
                    channel_index=this_channel_index,
                    init_function=init_function,
                    stat_function_for_neuron_activations=K.max,
                    num_iterations=num_iterations, learning_rate=learning_rate,
                    ideal_activation=ideal_activation))

            if list_of_optimized_input_matrices is None:
                list_of_optimized_input_matrices = copy.deepcopy(these_matrices)
            else:
                for k in range(len(list_of_optimized_input_matrices)):
                    list_of_optimized_input_matrices[k] = numpy.concatenate(
                        (list_of_optimized_input_matrices[k],
                         these_matrices[k]), axis=0)

    print MINOR_SEPARATOR_STRING

    if is_model_swirlnet:
        print 'Denormalizing Swirlnet data...'
        list_of_optimized_input_matrices[0] = _denormalize_swirlnet_data(
            list_of_optimized_input_matrices[0])
    else:
        list_of_optimized_input_matrices = _denormalize_gg_data(
            list_of_input_matrices=list_of_optimized_input_matrices,
            model_metadata_dict=model_metadata_dict)

    print 'Writing optimized input matrices to file: "{0:s}"...'.format(
        output_file_name)
    feature_optimization.write_file(
        pickle_file_name=output_file_name,
        list_of_optimized_input_matrices=list_of_optimized_input_matrices,
        model_file_name=model_file_name, num_iterations=num_iterations,
        learning_rate=learning_rate,
        component_type_string=component_type_string, target_class=target_class,
        optimize_for_probability=optimize_for_probability,
        ideal_logit=ideal_logit, layer_name=layer_name,
        ideal_activation=ideal_activation,
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
        optimize_for_probability=bool(
            getattr(INPUT_ARG_OBJECT, OPTIMIZE_FOR_PROBABILITY_ARG_NAME)),
        num_iterations=getattr(INPUT_ARG_OBJECT, NUM_ITERATIONS_ARG_NAME),
        learning_rate=getattr(INPUT_ARG_OBJECT, LEARNING_RATE_ARG_NAME),
        ideal_logit=getattr(INPUT_ARG_OBJECT, IDEAL_LOGIT_ARG_NAME),
        layer_name=getattr(INPUT_ARG_OBJECT, LAYER_NAME_ARG_NAME),
        ideal_activation=getattr(INPUT_ARG_OBJECT, IDEAL_ACTIVATION_ARG_NAME),
        neuron_indices_flattened=numpy.array(
            getattr(INPUT_ARG_OBJECT, NEURON_INDICES_ARG_NAME), dtype=int),
        channel_indices=numpy.array(
            getattr(INPUT_ARG_OBJECT, CHANNEL_INDICES_ARG_NAME), dtype=int),
        output_file_name=getattr(INPUT_ARG_OBJECT, OUTPUT_FILE_ARG_NAME))
