"""Runs Grad-CAM (gradient-weighted class-activation maps)."""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import argparse
import numpy
from keras import backend as K
from gewittergefahr.gg_io import storm_tracking_io as tracking_io
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.deep_learning import cnn
from gewittergefahr.deep_learning import testing_io
from gewittergefahr.deep_learning import training_validation_io as trainval_io
from gewittergefahr.deep_learning import model_interpretation
from gewittergefahr.deep_learning import gradcam

K.set_session(K.tf.Session(config=K.tf.ConfigProto(
    intra_op_parallelism_threads=1, inter_op_parallelism_threads=1,
    allow_soft_placement=False
)))

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

MODEL_FILE_ARG_NAME = 'model_file_name'
TARGET_CLASS_ARG_NAME = 'target_class'
TARGET_LAYER_ARG_NAME = 'target_layer_name'
EXAMPLE_DIR_ARG_NAME = 'input_example_dir_name'
STORM_METAFILE_ARG_NAME = 'input_storm_metafile_name'
NUM_EXAMPLES_ARG_NAME = 'num_examples'
OUTPUT_FILE_ARG_NAME = 'output_file_name'

MODEL_FILE_HELP_STRING = (
    'Path to file with trained CNN.  Will be read by `cnn.read_model`.')

TARGET_CLASS_HELP_STRING = (
    'Activation maps will be created for this class.  Must be in 0...(K - 1), '
    'where K = number of classes.')

TARGET_LAYER_HELP_STRING = (
    'Name of target layer.  Neuron-importance weights will be based on '
    'activations in this layer.')

EXAMPLE_DIR_HELP_STRING = (
    'Name of top-level directory with input examples.  Files therein will be '
    'found by `input_examples.find_example_file` and read by '
    '`input_examples.read_example_file`.')

STORM_METAFILE_HELP_STRING = (
    'Path to Pickle file with storm IDs and times.  Will be read by '
    '`storm_tracking_io.read_ids_and_times`.')

NUM_EXAMPLES_HELP_STRING = (
    'Number of examples (storm objects) to read from `{0:s}`.  If you want to '
    'read all examples, make this non-positive.'
).format(STORM_METAFILE_ARG_NAME)

OUTPUT_FILE_HELP_STRING = (
    'Path to output file (will be written by `gradcam.write_standard_file`).')

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + MODEL_FILE_ARG_NAME, type=str, required=True,
    help=MODEL_FILE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + TARGET_CLASS_ARG_NAME, type=int, required=False, default=1,
    help=TARGET_CLASS_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + TARGET_LAYER_ARG_NAME, type=str, required=True,
    help=TARGET_LAYER_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + EXAMPLE_DIR_ARG_NAME, type=str, required=True,
    help=EXAMPLE_DIR_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + STORM_METAFILE_ARG_NAME, type=str, required=True,
    help=STORM_METAFILE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + NUM_EXAMPLES_ARG_NAME, type=int, required=False, default=-1,
    help=NUM_EXAMPLES_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_FILE_ARG_NAME, type=str, required=True,
    help=OUTPUT_FILE_HELP_STRING)


def _run(model_file_name, target_class, target_layer_name, top_example_dir_name,
         storm_metafile_name, num_examples, output_file_name):
    """Runs Grad-CAM (gradient-weighted class-activation maps).

    This is effectively the main method.

    :param model_file_name: See documentation at top of file.
    :param target_class: Same.
    :param target_layer_name: Same.
    :param top_example_dir_name: Same.
    :param storm_metafile_name: Same.
    :param num_examples: Same.
    :param output_file_name: Same.
    """

    file_system_utils.mkdir_recursive_if_necessary(file_name=output_file_name)

    # Read model and metadata.
    print('Reading model from: "{0:s}"...'.format(model_file_name))
    model_object = cnn.read_model(model_file_name)

    model_metafile_name = '{0:s}/model_metadata.p'.format(
        os.path.split(model_file_name)[0]
    )

    print('Reading model metadata from: "{0:s}"...'.format(model_metafile_name))
    model_metadata_dict = cnn.read_model_metadata(model_metafile_name)
    training_option_dict = model_metadata_dict[cnn.TRAINING_OPTION_DICT_KEY]
    training_option_dict[trainval_io.REFLECTIVITY_MASK_KEY] = None

    print('Reading storm metadata from: "{0:s}"...'.format(storm_metafile_name))
    full_storm_id_strings, storm_times_unix_sec = (
        tracking_io.read_ids_and_times(storm_metafile_name)
    )

    print(SEPARATOR_STRING)

    if 0 < num_examples < len(full_storm_id_strings):
        full_storm_id_strings = full_storm_id_strings[:num_examples]
        storm_times_unix_sec = storm_times_unix_sec[:num_examples]

    example_dict = testing_io.read_predictors_specific_examples(
        top_example_dir_name=top_example_dir_name,
        desired_full_id_strings=full_storm_id_strings,
        desired_times_unix_sec=storm_times_unix_sec,
        option_dict=training_option_dict,
        layer_operation_dicts=model_metadata_dict[cnn.LAYER_OPERATIONS_KEY]
    )
    print(SEPARATOR_STRING)

    predictor_matrices = example_dict[testing_io.INPUT_MATRICES_KEY]
    sounding_pressure_matrix_pa = (
        example_dict[testing_io.SOUNDING_PRESSURES_KEY]
    )

    num_matrices = len(predictor_matrices)
    num_examples = len(full_storm_id_strings)

    cam_matrices = [None] * num_matrices
    guided_cam_matrices = [None] * num_matrices
    new_model_object = None

    for i in range(num_examples):
        print('Running Grad-CAM for example {0:d} of {1:d}...'.format(
            i + 1, num_examples
        ))

        these_predictor_matrices = [a[[i], ...] for a in predictor_matrices]
        these_cam_matrices = gradcam.run_gradcam(
            model_object=model_object,
            list_of_input_matrices=these_predictor_matrices,
            target_class=target_class, target_layer_name=target_layer_name
        )

        print('Running guided Grad-CAM for example {0:d} of {1:d}...'.format(
            i + 1, num_examples
        ))

        these_guided_cam_matrices, new_model_object = (
            gradcam.run_guided_gradcam(
                orig_model_object=model_object,
                list_of_input_matrices=these_predictor_matrices,
                target_layer_name=target_layer_name,
                list_of_cam_matrices=these_cam_matrices,
                new_model_object=new_model_object)
        )

        if all([a is None for a in cam_matrices]):
            for k in range(num_matrices):
                if these_cam_matrices[k] is None:
                    continue

                these_dim = numpy.array(
                    (num_examples,) + these_cam_matrices[k].shape[1:], dtype=int
                )
                cam_matrices[k] = numpy.full(these_dim, numpy.nan)

                these_dim = numpy.array(
                    (num_examples,) + these_guided_cam_matrices[k].shape[1:],
                    dtype=int
                )
                guided_cam_matrices[k] = numpy.full(these_dim, numpy.nan)

        for k in range(num_matrices):
            if these_cam_matrices[k] is None:
                continue

            cam_matrices[k][i, ...] = these_cam_matrices[k][0, ...]
            guided_cam_matrices[k][i, ...] = (
                these_guided_cam_matrices[k][0, ...]
            )

    print(SEPARATOR_STRING)
    upsample_refl = training_option_dict[trainval_io.UPSAMPLE_REFLECTIVITY_KEY]

    if upsample_refl:
        cam_matrices[0] = numpy.expand_dims(cam_matrices[0], axis=-1)

        num_channels = predictor_matrices[0].shape[-1]
        cam_matrices[0] = numpy.repeat(
            a=cam_matrices[0], repeats=num_channels, axis=-1
        )

        cam_matrices = trainval_io.separate_shear_and_reflectivity(
            list_of_input_matrices=cam_matrices,
            training_option_dict=training_option_dict
        )

        cam_matrices[0] = cam_matrices[0][..., 0]
        cam_matrices[1] = cam_matrices[1][..., 0]

    guided_cam_matrices = trainval_io.separate_shear_and_reflectivity(
        list_of_input_matrices=guided_cam_matrices,
        training_option_dict=training_option_dict
    )

    print('Denormalizing predictors...')
    predictor_matrices = trainval_io.separate_shear_and_reflectivity(
        list_of_input_matrices=predictor_matrices,
        training_option_dict=training_option_dict
    )
    predictor_matrices = model_interpretation.denormalize_data(
        list_of_input_matrices=predictor_matrices,
        model_metadata_dict=model_metadata_dict
    )

    print('Writing class-activation maps to file: "{0:s}"...'.format(
        output_file_name
    ))
    gradcam.write_standard_file(
        pickle_file_name=output_file_name,
        denorm_predictor_matrices=predictor_matrices,
        cam_matrices=cam_matrices,
        guided_cam_matrices=guided_cam_matrices,
        full_storm_id_strings=full_storm_id_strings,
        storm_times_unix_sec=storm_times_unix_sec,
        model_file_name=model_file_name, target_class=target_class,
        target_layer_name=target_layer_name,
        sounding_pressure_matrix_pa=sounding_pressure_matrix_pa
    )


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        model_file_name=getattr(INPUT_ARG_OBJECT, MODEL_FILE_ARG_NAME),
        target_class=getattr(INPUT_ARG_OBJECT, TARGET_CLASS_ARG_NAME),
        target_layer_name=getattr(INPUT_ARG_OBJECT, TARGET_LAYER_ARG_NAME),
        top_example_dir_name=getattr(INPUT_ARG_OBJECT, EXAMPLE_DIR_ARG_NAME),
        storm_metafile_name=getattr(INPUT_ARG_OBJECT, STORM_METAFILE_ARG_NAME),
        num_examples=getattr(INPUT_ARG_OBJECT, NUM_EXAMPLES_ARG_NAME),
        output_file_name=getattr(INPUT_ARG_OBJECT, OUTPUT_FILE_ARG_NAME)
    )
