"""Uses trained CNN to make predictions for specific examples."""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import argparse
import numpy
from keras import backend as K
from gewittergefahr.gg_io import storm_tracking_io as tracking_io
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.deep_learning import cnn
from gewittergefahr.deep_learning import input_examples
from gewittergefahr.deep_learning import testing_io
from gewittergefahr.deep_learning import prediction_io
from gewittergefahr.deep_learning import training_validation_io as trainval_io

# TODO(thunderhoser): Merge with apply_cnn.py.

K.set_session(K.tf.Session(config=K.tf.ConfigProto(
    intra_op_parallelism_threads=1, inter_op_parallelism_threads=1,
    allow_soft_placement=False
)))

NUM_EXAMPLES_PER_BATCH = 1000
SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

MODEL_FILE_ARG_NAME = 'input_model_file_name'
EXAMPLE_DIR_ARG_NAME = 'input_example_dir_name'
STORM_METAFILE_ARG_NAME = 'input_storm_metafile_name'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

MODEL_FILE_HELP_STRING = (
    'Path to trained CNN.  Will be read by `cnn.read_model`.')

EXAMPLE_DIR_HELP_STRING = (
    'Name of top-level directory with input examples.  Files therein will be '
    'found by `input_examples.find_example_file` and read by '
    '`input_examples.read_example_file`.')

STORM_METAFILE_HELP_STRING = (
    'Path to file with storm metadata (IDs and times), to be read by '
    '`storm_tracking_io.read_ids_and_times`.  The CNN will be applied only to '
    'these examples.')

OUTPUT_DIR_HELP_STRING = (
    'Name of output directory.  Results will be written by '
    '`prediction_io.write_ungridded_predictions`, to a location therein '
    'determined by `prediction_io.find_ungridded_file`.')

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + MODEL_FILE_ARG_NAME, type=str, required=True,
    help=MODEL_FILE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + EXAMPLE_DIR_ARG_NAME, type=str, required=True,
    help=EXAMPLE_DIR_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + STORM_METAFILE_ARG_NAME, type=str, required=True,
    help=STORM_METAFILE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING)


def _run(model_file_name, top_example_dir_name, storm_metafile_name,
         output_dir_name):
    """Uses trained CNN to make predictions for specific examples.

    This is effectively the main method.

    :param model_file_name: See documentation at top of file.
    :param top_example_dir_name: Same.
    :param storm_metafile_name: Same.
    :param output_dir_name: Same.
    :raises: ValueError: if the model does multi-class classification.
    """

    print('Reading CNN from: "{0:s}"...'.format(model_file_name))
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

    cnn_metafile_name = cnn.find_metafile(
        model_file_name=model_file_name, raise_error_if_missing=True
    )
    print('Reading CNN metadata from: "{0:s}"...'.format(cnn_metafile_name))
    cnn_metadata_dict = cnn.read_model_metadata(cnn_metafile_name)

    print('Reading storm metadata from: "{0:s}"...'.format(storm_metafile_name))
    desired_full_id_strings, desired_times_unix_sec = (
        tracking_io.read_ids_and_times(storm_metafile_name)
    )

    unique_spc_date_strings = list(set([
        time_conversion.time_to_spc_date_string(t)
        for t in desired_times_unix_sec
    ]))

    example_file_names = [
        input_examples.find_example_file(
            top_directory_name=top_example_dir_name, shuffled=False,
            spc_date_string=d, raise_error_if_missing=True
        ) for d in unique_spc_date_strings
    ]

    first_spc_date_string = time_conversion.time_to_spc_date_string(
        numpy.min(desired_times_unix_sec)
    )
    last_spc_date_string = time_conversion.time_to_spc_date_string(
        numpy.max(desired_times_unix_sec)
    )

    training_option_dict = cnn_metadata_dict[cnn.TRAINING_OPTION_DICT_KEY]
    training_option_dict[trainval_io.EXAMPLE_FILES_KEY] = example_file_names
    training_option_dict[trainval_io.FIRST_STORM_TIME_KEY] = (
        time_conversion.get_start_of_spc_date(first_spc_date_string)
    )
    training_option_dict[trainval_io.LAST_STORM_TIME_KEY] = (
        time_conversion.get_end_of_spc_date(last_spc_date_string)
    )
    training_option_dict[trainval_io.NUM_EXAMPLES_PER_BATCH_KEY] = (
        NUM_EXAMPLES_PER_BATCH
    )

    if soundings_only:
        generator_object = testing_io.sounding_generator(
            option_dict=training_option_dict,
            desired_full_id_strings=desired_full_id_strings,
            desired_times_unix_sec=desired_times_unix_sec)

    elif cnn_metadata_dict[cnn.LAYER_OPERATIONS_KEY] is not None:
        generator_object = testing_io.gridrad_generator_2d_reduced(
            option_dict=training_option_dict,
            desired_full_id_strings=desired_full_id_strings,
            desired_times_unix_sec=desired_times_unix_sec,
            list_of_operation_dicts=cnn_metadata_dict[
                cnn.LAYER_OPERATIONS_KEY]
        )

    elif cnn_metadata_dict[cnn.CONV_2D3D_KEY]:
        generator_object = testing_io.myrorss_generator_2d3d(
            option_dict=training_option_dict,
            desired_full_id_strings=desired_full_id_strings,
            desired_times_unix_sec=desired_times_unix_sec)
    else:
        generator_object = testing_io.generator_2d_or_3d(
            option_dict=training_option_dict,
            desired_full_id_strings=desired_full_id_strings,
            desired_times_unix_sec=desired_times_unix_sec)

    include_soundings = (
        training_option_dict[trainval_io.SOUNDING_FIELDS_KEY] is not None
    )

    full_storm_id_strings = []
    storm_times_unix_sec = numpy.array([], dtype=int)
    observed_labels = numpy.array([], dtype=int)
    class_probability_matrix = None

    while True:
        try:
            this_storm_object_dict = next(generator_object)
            print(SEPARATOR_STRING)
        except StopIteration:
            break

        full_storm_id_strings += this_storm_object_dict[testing_io.FULL_IDS_KEY]
        storm_times_unix_sec = numpy.concatenate((
            storm_times_unix_sec,
            this_storm_object_dict[testing_io.STORM_TIMES_KEY]
        ))
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
        elif cnn_metadata_dict[cnn.CONV_2D3D_KEY]:
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

        if class_probability_matrix is None:
            class_probability_matrix = this_probability_matrix + 0.
        else:
            class_probability_matrix = numpy.concatenate(
                (class_probability_matrix, this_probability_matrix), axis=0
            )

    output_file_name = prediction_io.find_ungridded_file(
        directory_name=output_dir_name, raise_error_if_missing=False)

    print('Writing results to: "{0:s}"...'.format(output_file_name))

    prediction_io.write_ungridded_predictions(
        netcdf_file_name=output_file_name,
        class_probability_matrix=class_probability_matrix,
        observed_labels=observed_labels, storm_ids=full_storm_id_strings,
        storm_times_unix_sec=storm_times_unix_sec,
        target_name=training_option_dict[trainval_io.TARGET_NAME_KEY],
        model_file_name=model_file_name
    )


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        model_file_name=getattr(INPUT_ARG_OBJECT, MODEL_FILE_ARG_NAME),
        top_example_dir_name=getattr(INPUT_ARG_OBJECT, EXAMPLE_DIR_ARG_NAME),
        storm_metafile_name=getattr(INPUT_ARG_OBJECT, STORM_METAFILE_ARG_NAME),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
