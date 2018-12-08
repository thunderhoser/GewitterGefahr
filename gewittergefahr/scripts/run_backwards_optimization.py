"""Runs backwards optimization on a trained CNN."""

import os.path
import argparse
import numpy
from keras import backend as K
from gewittergefahr.deep_learning import cnn
from gewittergefahr.deep_learning import deep_learning_utils as dl_utils
from gewittergefahr.deep_learning import model_interpretation
from gewittergefahr.deep_learning import backwards_optimization as backwards_opt
from gewittergefahr.deep_learning import testing_io
from gewittergefahr.deep_learning import training_validation_io as trainval_io
from gewittergefahr.scripts import get_cnn_saliency_maps

# random.seed(6695)
# numpy.random.seed(6695)

K.set_session(K.tf.Session(config=K.tf.ConfigProto(
    intra_op_parallelism_threads=1, inter_op_parallelism_threads=1
)))

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

CLASS_COMPONENT_TYPE_STRING = model_interpretation.CLASS_COMPONENT_TYPE_STRING
NEURON_COMPONENT_TYPE_STRING = model_interpretation.NEURON_COMPONENT_TYPE_STRING
CHANNEL_COMPONENT_TYPE_STRING = (
    model_interpretation.CHANNEL_COMPONENT_TYPE_STRING)

MODEL_FILE_ARG_NAME = 'input_model_file_name'
INIT_FUNCTION_ARG_NAME = 'init_function_name'
STORM_METAFILE_ARG_NAME = 'input_storm_metafile_name'
EXAMPLE_DIR_ARG_NAME = 'input_example_dir_name'
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
    'Path to file with trained CNN.  Will be read by `cnn.read_model`.')

INIT_FUNCTION_HELP_STRING = (
    'Initialization function (used to create initial input matrices for '
    'gradient descent).  Must be accepted by '
    '`backwards_opt.check_init_function`.  To initialize with real '
    'dataset examples, leave this argument alone.')

STORM_METAFILE_HELP_STRING = (
    '[used only if `{0:s}` is empty] Path to file with storm metadata.  Will be'
    ' read by `_read_storm_metadata` and used to find dataset examples for '
    'initialization.'
).format(INIT_FUNCTION_ARG_NAME)

EXAMPLE_DIR_HELP_STRING = (
    '[used only if `{0:s}` is empty] Name of top-level directory with dataset '
    'examples.  Files therein will be read by '
    '`testing_io.read_specific_examples`, where the "specific examples" '
    'correspond to the storm IDs and times specified in `{1:s}`.'
).format(INIT_FUNCTION_ARG_NAME, STORM_METAFILE_ARG_NAME)

COMPONENT_HELP_STRING = (
    'Determines model component for which activation will be maximized.  See '
    '`model_interpretation.check_component_metadata` for details.')

IDEAL_ACTIVATION_HELP_STRING = (
    '[used only if {0:s} = "{1:s}" or "{2:s}"] See '
    '`backwards_opt.optimize_input_for_neuron_activation` or '
    '`backwards_opt.optimize_input_for_channel_activation` for details.'
).format(COMPONENT_TYPE_ARG_NAME, NEURON_COMPONENT_TYPE_STRING,
         CLASS_COMPONENT_TYPE_STRING)

NUM_ITERATIONS_HELP_STRING = 'Number of iterations for backwards optimization.'

LEARNING_RATE_HELP_STRING = 'Learning rate for backwards optimization.'

OUTPUT_FILE_HELP_STRING = (
    'Path to output file (will be written by `backwards_opt.write_results`).')

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + MODEL_FILE_ARG_NAME, type=str, required=True,
    help=MODEL_FILE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + INIT_FUNCTION_ARG_NAME, type=str, required=False, default='',
    help=INIT_FUNCTION_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + STORM_METAFILE_ARG_NAME, type=str, required=False, default='',
    help=STORM_METAFILE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + EXAMPLE_DIR_ARG_NAME, type=str, required=False, default='',
    help=EXAMPLE_DIR_HELP_STRING)

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


def _create_initializer(init_function_name, model_metadata_dict):
    """Creates initialization function.

    :param init_function_name: See documentation at top of file.
    :param model_metadata_dict: Dictionary returned by
        `cnn.read_model_metadata`.
    :return: init_function: Function (see below).
    """

    backwards_opt.check_init_function(init_function_name)
    training_option_dict = model_metadata_dict[cnn.TRAINING_OPTION_DICT_KEY]

    used_minmax_norm = (
        training_option_dict[trainval_io.NORMALIZATION_TYPE_KEY] ==
        dl_utils.MINMAX_NORMALIZATION_TYPE_STRING
    )

    if init_function_name == backwards_opt.CONSTANT_INIT_FUNCTION_NAME:
        if used_minmax_norm:
            return backwards_opt.create_constant_initializer(
                (training_option_dict[trainval_io.MAX_NORMALIZED_VALUE_KEY] -
                 training_option_dict[trainval_io.MIN_NORMALIZED_VALUE_KEY])
                / 2)

        return backwards_opt.create_constant_initializer(0.)

    if init_function_name == backwards_opt.UNIFORM_INIT_FUNCTION_NAME:
        if used_minmax_norm:
            return backwards_opt.create_uniform_random_initializer(
                min_value=training_option_dict[
                    trainval_io.MIN_NORMALIZED_VALUE_KEY],
                max_value=training_option_dict[
                    trainval_io.MAX_NORMALIZED_VALUE_KEY])

        return backwards_opt.create_uniform_random_initializer(
            min_value=-3., max_value=3.)

    if init_function_name == backwards_opt.GAUSSIAN_INIT_FUNCTION_NAME:
        if used_minmax_norm:
            return backwards_opt.create_gaussian_initializer(
                mean=
                (training_option_dict[trainval_io.MAX_NORMALIZED_VALUE_KEY] -
                 training_option_dict[trainval_io.MIN_NORMALIZED_VALUE_KEY])
                / 2,
                standard_deviation=
                (training_option_dict[trainval_io.MAX_NORMALIZED_VALUE_KEY] -
                 training_option_dict[trainval_io.MIN_NORMALIZED_VALUE_KEY]) / 6
            )

        return backwards_opt.create_gaussian_initializer(
            mean=0., standard_deviation=1.)

    return backwards_opt.create_climo_initializer(
        training_option_dict=training_option_dict,
        myrorss_2d3d=model_metadata_dict[cnn.USE_2D3D_CONVOLUTION_KEY])


def _run(model_file_name, init_function_name, storm_metafile_name,
         top_example_dir_name, component_type_string, target_class, layer_name,
         neuron_indices, channel_index, num_iterations, ideal_activation,
         learning_rate, output_file_name):
    """Runs backwards optimization on a trained CNN.

    This is effectively the main method.

    :param model_file_name: See documentation at top of file.
    :param init_function_name: Same.
    :param storm_metafile_name: Same.
    :param top_example_dir_name: Same.
    :param component_type_string: Same.
    :param target_class: Same.
    :param layer_name: Same.
    :param neuron_indices: Same.
    :param channel_index: Same.
    :param num_iterations: Same.
    :param ideal_activation: Same.
    :param learning_rate: Same.
    :param output_file_name: Same.
    """

    if ideal_activation <= 0:
        ideal_activation = None
    if init_function_name in ['', 'None']:
        init_function_name = None

    model_interpretation.check_component_type(component_type_string)

    model_metafile_name = '{0:s}/model_metadata.p'.format(
        os.path.split(model_file_name)[0]
    )

    print 'Reading metadata from: "{0:s}"...'.format(model_metafile_name)
    model_metadata_dict = cnn.read_model_metadata(model_metafile_name)

    if init_function_name is None:
        # TODO(thunderhoser): Get rid of call to private method.

        print 'Reading metadata from: "{0:s}"...'.format(storm_metafile_name)
        storm_ids, storm_times_unix_sec = (
            get_cnn_saliency_maps._read_storm_metadata(storm_metafile_name)
        )

        # TODO(thunderhoser): Get rid of this HACK.
        storm_ids = storm_ids[:10]
        storm_times_unix_sec = storm_times_unix_sec[:10]

        list_of_init_matrices = testing_io.read_specific_examples(
            desired_storm_ids=storm_ids,
            desired_times_unix_sec=storm_times_unix_sec,
            training_option_dict=model_metadata_dict[
                cnn.TRAINING_OPTION_DICT_KEY],
            top_example_dir_name=top_example_dir_name)

        num_examples = list_of_init_matrices[0].shape[0]
        print SEPARATOR_STRING

    else:
        init_function = _create_initializer(
            init_function_name=init_function_name,
            model_metadata_dict=model_metadata_dict)
        num_examples = 1

    print 'Reading model from: "{0:s}"...'.format(model_file_name)
    model_object = cnn.read_model(model_file_name)

    list_of_optimized_matrices = None

    for i in range(num_examples):
        if init_function_name is None:
            this_init_arg = [a[[i], ...] for a in list_of_init_matrices]
        else:
            this_init_arg = init_function

        if component_type_string == CLASS_COMPONENT_TYPE_STRING:
            print (
                '\nOptimizing {0:d}th of {1:d} images for target class {2:d}...'
            ).format(i + 1, num_examples, target_class)

            these_optimized_matrices = backwards_opt.optimize_input_for_class(
                model_object=model_object, target_class=target_class,
                init_function_or_matrices=this_init_arg,
                num_iterations=num_iterations, learning_rate=learning_rate)

        elif component_type_string == NEURON_COMPONENT_TYPE_STRING:
            print (
                '\nOptimizing {0:d}th of {1:d} images for neuron {2:s} in layer'
                ' "{3:s}"...'
            ).format(i + 1, num_examples, str(neuron_indices), layer_name)

            these_optimized_matrices = backwards_opt.optimize_input_for_neuron(
                model_object=model_object, layer_name=layer_name,
                neuron_indices=neuron_indices,
                init_function_or_matrices=this_init_arg,
                num_iterations=num_iterations, learning_rate=learning_rate,
                ideal_activation=ideal_activation)

        else:
            print (
                '\nOptimizing {0:d}th of {1:d} images for channel {2:d} in '
                'layer "{3:s}"...'
            ).format(i + 1, num_examples, channel_index, layer_name)

            these_optimized_matrices = backwards_opt.optimize_input_for_channel(
                model_object=model_object, layer_name=layer_name,
                channel_index=channel_index,
                init_function_or_matrices=this_init_arg,
                stat_function_for_neuron_activations=K.max,
                num_iterations=num_iterations, learning_rate=learning_rate,
                ideal_activation=ideal_activation)

        if list_of_optimized_matrices is None:
            num_matrices = len(these_optimized_matrices)
            list_of_optimized_matrices = [None] * num_matrices

        for k in range(len(list_of_optimized_matrices)):
            if list_of_optimized_matrices[k] is None:
                list_of_optimized_matrices[k] = these_optimized_matrices[k] + 0.
            else:
                list_of_optimized_matrices[k] = numpy.concatenate(
                    (list_of_optimized_matrices[k],
                     these_optimized_matrices[k]),
                    axis=0)

    print SEPARATOR_STRING

    if init_function_name is None:
        this_init_arg = list_of_init_matrices
    else:
        this_init_arg = init_function_name + ''

    print 'Writing results to: "{0:s}"...'.format(output_file_name)
    backwards_opt.write_results(
        pickle_file_name=output_file_name,
        list_of_optimized_input_matrices=list_of_optimized_matrices,
        model_file_name=model_file_name,
        init_function_name_or_matrices=this_init_arg,
        num_iterations=num_iterations, learning_rate=learning_rate,
        component_type_string=component_type_string, target_class=target_class,
        layer_name=layer_name, neuron_indices=neuron_indices,
        channel_index=channel_index, ideal_activation=ideal_activation)


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        model_file_name=getattr(INPUT_ARG_OBJECT, MODEL_FILE_ARG_NAME),
        init_function_name=getattr(INPUT_ARG_OBJECT, INIT_FUNCTION_ARG_NAME),
        storm_metafile_name=getattr(INPUT_ARG_OBJECT, STORM_METAFILE_ARG_NAME),
        top_example_dir_name=getattr(INPUT_ARG_OBJECT, EXAMPLE_DIR_ARG_NAME),
        component_type_string=getattr(
            INPUT_ARG_OBJECT, COMPONENT_TYPE_ARG_NAME),
        target_class=getattr(INPUT_ARG_OBJECT, TARGET_CLASS_ARG_NAME),
        layer_name=getattr(INPUT_ARG_OBJECT, LAYER_NAME_ARG_NAME),
        neuron_indices=numpy.array(
            getattr(INPUT_ARG_OBJECT, NEURON_INDICES_ARG_NAME), dtype=int),
        channel_index=getattr(INPUT_ARG_OBJECT, CHANNEL_INDEX_ARG_NAME),
        num_iterations=getattr(INPUT_ARG_OBJECT, NUM_ITERATIONS_ARG_NAME),
        ideal_activation=getattr(INPUT_ARG_OBJECT, IDEAL_ACTIVATION_ARG_NAME),
        learning_rate=getattr(INPUT_ARG_OBJECT, LEARNING_RATE_ARG_NAME),
        output_file_name=getattr(INPUT_ARG_OBJECT, OUTPUT_FILE_ARG_NAME)
    )
