"""Evaluates predictions from CNN."""

import random
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import argparse
import numpy
from keras import backend as K
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.deep_learning import cnn
from gewittergefahr.deep_learning import input_examples
from gewittergefahr.deep_learning import testing_io
from gewittergefahr.deep_learning import training_validation_io as trainval_io
from gewittergefahr.scripts import model_evaluation_helper as model_eval_helper

random.seed(6695)
numpy.random.seed(6695)

K.set_session(K.tf.Session(config=K.tf.ConfigProto(
    intra_op_parallelism_threads=1, inter_op_parallelism_threads=1,
    allow_soft_placement=False
)))

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

MODEL_FILE_ARG_NAME = 'input_model_file_name'
EXAMPLE_DIR_ARG_NAME = 'input_example_dir_name'
FIRST_SPC_DATE_ARG_NAME = 'first_spc_date_string'
LAST_SPC_DATE_ARG_NAME = 'last_spc_date_string'
NUM_EXAMPLES_ARG_NAME = 'num_examples'
NUM_BOOTSTRAP_ARG_NAME = 'num_bootstrap_reps'
CONFIDENCE_LEVEL_ARG_NAME = 'confidence_level'
CLASS_FRACTION_KEYS_ARG_NAME = 'class_fraction_keys'
CLASS_FRACTION_VALUES_ARG_NAME = 'class_fraction_values'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

MODEL_FILE_HELP_STRING = (
    'File path for trained CNN.  Will be read by `cnn.read_model`.')

EXAMPLE_DIR_HELP_STRING = (
    'Name of top-level directory with input examples.  Files therein will be '
    'found by `input_examples.find_example_file` and read by '
    '`input_examples.read_example_file`.')

SPC_DATE_HELP_STRING = (
    'SPC date (format "yyyymmdd").  This script will evaluate predictions on '
    'examples from the period `{0:s}`...`{1:s}`.'
).format(FIRST_SPC_DATE_ARG_NAME, LAST_SPC_DATE_ARG_NAME)

NUM_EXAMPLES_HELP_STRING = 'Number of examples to use.'

NUM_BOOTSTRAP_HELP_STRING = (
    'Number of bootstrap replicates.  If you do not want bootstrapping, leave '
    'this alone.')

CONFIDENCE_LEVEL_HELP_STRING = (
    '[used only if `{0:s}` > 1] Confidence level for bootstrapping, in range '
    '0...1.'
).format(NUM_BOOTSTRAP_ARG_NAME)

CLASS_FRACTION_KEYS_HELP_STRING = (
    'List of keys used to create input `class_to_sampling_fraction_dict` for '
    '`deep_learning_utils.sample_by_class`.  If you do not want class-'
    'conditional sampling, leave this alone.'
)

CLASS_FRACTION_VALUES_HELP_STRING = (
    'List of values used to create input `class_to_sampling_fraction_dict` for '
    '`deep_learning_utils.sample_by_class`.  If you do not want class-'
    'conditional sampling, leave this alone.'
)

OUTPUT_DIR_HELP_STRING = (
    'Name of output directory.  Results will be saved here.')

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + MODEL_FILE_ARG_NAME, type=str, required=True,
    help=MODEL_FILE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + EXAMPLE_DIR_ARG_NAME, type=str, required=True,
    help=EXAMPLE_DIR_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + FIRST_SPC_DATE_ARG_NAME, type=str, required=True,
    help=SPC_DATE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + LAST_SPC_DATE_ARG_NAME, type=str, required=True,
    help=SPC_DATE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + NUM_EXAMPLES_ARG_NAME, type=int, required=True,
    help=NUM_EXAMPLES_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + NUM_BOOTSTRAP_ARG_NAME, type=int, required=False, default=1,
    help=NUM_BOOTSTRAP_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + CONFIDENCE_LEVEL_ARG_NAME, type=float, required=False, default=0.95,
    help=CONFIDENCE_LEVEL_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + CLASS_FRACTION_KEYS_ARG_NAME, type=int, nargs='+',
    required=False, default=[0], help=CLASS_FRACTION_KEYS_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + CLASS_FRACTION_VALUES_ARG_NAME, type=float, nargs='+',
    required=False, default=[0.], help=CLASS_FRACTION_VALUES_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING)


def _run(model_file_name, top_example_dir_name, first_spc_date_string,
         last_spc_date_string, num_examples, num_bootstrap_reps,
         confidence_level, class_fraction_keys, class_fraction_values,
         output_dir_name):
    """Evaluates CNN (convolutional neural net) predictions.

    This is effectively the main method.

    :param model_file_name: See documentation at top of file.
    :param top_example_dir_name: Same.
    :param first_spc_date_string: Same.
    :param last_spc_date_string: Same.
    :param num_examples: Same.
    :param num_bootstrap_reps: Same.
    :param confidence_level: Same.
    :param class_fraction_keys: Same.
    :param class_fraction_values: Same.
    :param output_dir_name: Same.
    :raises: ValueError: if the model does multi-class classification.
    """

    print('Reading model from: "{0:s}"...'.format(model_file_name))
    model_object = cnn.read_model(model_file_name)

    num_output_neurons = (
        model_object.layers[-1].output.get_shape().as_list()[-1]
    )

    if num_output_neurons > 2:
        error_string = (
            'The model has {0:d} output neurons, which suggests {0:d}-class '
            'classification.  This script handles only binary classification.'
        ).format(num_output_neurons)

        raise ValueError(error_string)

    soundings_only = False

    if isinstance(model_object.input, list):
        list_of_input_tensors = model_object.input
    else:
        list_of_input_tensors = [model_object.input]

    if len(list_of_input_tensors) == 1:
        these_spatial_dim = numpy.array(
            list_of_input_tensors[0].get_shape().as_list()[1:-1], dtype=int
        )
        soundings_only = len(these_spatial_dim) == 1

    model_directory_name, _ = os.path.split(model_file_name)
    model_metafile_name = '{0:s}/model_metadata.p'.format(model_directory_name)

    print('Reading metadata from: "{0:s}"...'.format(model_metafile_name))
    model_metadata_dict = cnn.read_model_metadata(model_metafile_name)
    training_option_dict = model_metadata_dict[cnn.TRAINING_OPTION_DICT_KEY]
    training_option_dict[trainval_io.SAMPLING_FRACTIONS_KEY] = None

    if len(class_fraction_keys) > 1:
        downsampling_dict = dict(list(zip(
            class_fraction_keys, class_fraction_values
        )))
    else:
        downsampling_dict = None

    example_file_names = input_examples.find_many_example_files(
        top_directory_name=top_example_dir_name, shuffled=False,
        first_spc_date_string=first_spc_date_string,
        last_spc_date_string=last_spc_date_string,
        raise_error_if_any_missing=False)

    training_option_dict[trainval_io.EXAMPLE_FILES_KEY] = example_file_names
    training_option_dict[trainval_io.FIRST_STORM_TIME_KEY] = (
        time_conversion.get_start_of_spc_date(first_spc_date_string)
    )
    training_option_dict[trainval_io.LAST_STORM_TIME_KEY] = (
        time_conversion.get_end_of_spc_date(last_spc_date_string)
    )

    if soundings_only:
        generator_object = testing_io.sounding_generator(
            option_dict=training_option_dict, num_examples_total=num_examples)

    elif model_metadata_dict[cnn.LAYER_OPERATIONS_KEY] is not None:
        generator_object = testing_io.gridrad_generator_2d_reduced(
            option_dict=training_option_dict,
            list_of_operation_dicts=model_metadata_dict[
                cnn.LAYER_OPERATIONS_KEY],
            num_examples_total=num_examples
        )

    elif model_metadata_dict[cnn.CONV_2D3D_KEY]:
        generator_object = testing_io.myrorss_generator_2d3d(
            option_dict=training_option_dict, num_examples_total=num_examples)
    else:
        generator_object = testing_io.generator_2d_or_3d(
            option_dict=training_option_dict, num_examples_total=num_examples)

    include_soundings = (
        training_option_dict[trainval_io.SOUNDING_FIELDS_KEY] is not None
    )

    forecast_probabilities = numpy.array([])
    observed_labels = numpy.array([], dtype=int)

    for _ in range(len(example_file_names)):
        try:
            this_storm_object_dict = next(generator_object)
            print(SEPARATOR_STRING)
        except StopIteration:
            break

        observed_labels = numpy.concatenate((
            observed_labels, this_storm_object_dict[testing_io.TARGET_ARRAY_KEY]
        ))

        if soundings_only:
            these_predictor_matrices = [
                this_storm_object_dict[testing_io.SOUNDING_MATRIX_KEY]
            ]
        else:
            these_predictor_matrices = this_storm_object_dict[
                testing_io.INPUT_MATRICES_KEY]

        if include_soundings:
            this_sounding_matrix = these_predictor_matrices[-1]
        else:
            this_sounding_matrix = None

        if soundings_only:
            this_probability_matrix = cnn.apply_cnn_soundings_only(
                model_object=model_object, sounding_matrix=this_sounding_matrix,
                verbose=True)
        elif model_metadata_dict[cnn.CONV_2D3D_KEY]:
            if training_option_dict[trainval_io.UPSAMPLE_REFLECTIVITY_KEY]:
                this_probability_matrix = cnn.apply_2d_or_3d_cnn(
                    model_object=model_object,
                    radar_image_matrix=these_predictor_matrices[0],
                    sounding_matrix=this_sounding_matrix, verbose=True)
            else:
                this_probability_matrix = cnn.apply_2d3d_cnn(
                    model_object=model_object,
                    reflectivity_matrix_dbz=these_predictor_matrices[0],
                    azimuthal_shear_matrix_s01=these_predictor_matrices[1],
                    sounding_matrix=this_sounding_matrix, verbose=True)
        else:
            this_probability_matrix = cnn.apply_2d_or_3d_cnn(
                model_object=model_object,
                radar_image_matrix=these_predictor_matrices[0],
                sounding_matrix=this_sounding_matrix, verbose=True)

        print(SEPARATOR_STRING)

        forecast_probabilities = numpy.concatenate((
            forecast_probabilities, this_probability_matrix[:, -1]
        ))

    model_eval_helper.run_evaluation(
        forecast_probabilities=forecast_probabilities,
        observed_labels=observed_labels, downsampling_dict=downsampling_dict,
        num_bootstrap_reps=num_bootstrap_reps,
        confidence_level=confidence_level, output_dir_name=output_dir_name)


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        model_file_name=getattr(INPUT_ARG_OBJECT, MODEL_FILE_ARG_NAME),
        top_example_dir_name=getattr(INPUT_ARG_OBJECT, EXAMPLE_DIR_ARG_NAME),
        first_spc_date_string=getattr(
            INPUT_ARG_OBJECT, FIRST_SPC_DATE_ARG_NAME),
        last_spc_date_string=getattr(INPUT_ARG_OBJECT, LAST_SPC_DATE_ARG_NAME),
        num_examples=getattr(INPUT_ARG_OBJECT, NUM_EXAMPLES_ARG_NAME),
        num_bootstrap_reps=getattr(INPUT_ARG_OBJECT, NUM_BOOTSTRAP_ARG_NAME),
        confidence_level=getattr(INPUT_ARG_OBJECT, CONFIDENCE_LEVEL_ARG_NAME),
        class_fraction_keys=numpy.array(
            getattr(INPUT_ARG_OBJECT, CLASS_FRACTION_KEYS_ARG_NAME), dtype=int),
        class_fraction_values=numpy.array(
            getattr(INPUT_ARG_OBJECT, CLASS_FRACTION_VALUES_ARG_NAME),
            dtype=float),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
