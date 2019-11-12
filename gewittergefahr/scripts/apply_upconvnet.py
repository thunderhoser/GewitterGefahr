"""Makes predictions from trained upconvnet."""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import copy
import argparse
import numpy
from keras import backend as K
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.deep_learning import cnn
from gewittergefahr.deep_learning import upconvnet
from gewittergefahr.deep_learning import input_examples
from gewittergefahr.deep_learning import testing_io
from gewittergefahr.deep_learning import model_interpretation
from gewittergefahr.deep_learning import training_validation_io as trainval_io

EARLY_TIME_UNIX_SEC = 0
LATE_TIME_UNIX_SEC = int(1e12)
NUM_EXAMPLES_PER_BATCH = 1000

K.set_session(K.tf.Session(config=K.tf.ConfigProto(
    intra_op_parallelism_threads=1, inter_op_parallelism_threads=1,
    allow_soft_placement=False
)))

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'
MINOR_SEPARATOR_STRING = '\n\n' + '-' * 50 + '\n\n'

UPCONVNET_FILE_ARG_NAME = 'input_upconvnet_file_name'
EXAMPLE_DIR_ARG_NAME = 'input_example_dir_name'
FIRST_DATE_ARG_NAME = 'first_spc_date_string'
LAST_DATE_ARG_NAME = 'last_spc_date_string'
NUM_EXAMPLES_ARG_NAME = 'num_examples_per_date'
DOWNSAMPLING_KEYS_ARG_NAME = 'downsampling_keys'
DOWNSAMPLING_VALUES_ARG_NAME = 'downsampling_values'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

UPCONVNET_FILE_HELP_STRING = (
    'Path to file with trained upconvnet (will be read by `cnn.read_model`).')

EXAMPLE_DIR_HELP_STRING = (
    'Name of top-level directory with examples (containing actual radar '
    'images).  Files therein will be found by '
    '`input_examples.find_example_file` and read by '
    '`input_examples.read_example_file`.')

SPC_DATE_HELP_STRING = (
    'SPC date (format "yyyymmdd").  Will reconstruct radar images for all '
    'examples in the period `{0:s}`...`{1:s}`.'
).format(FIRST_DATE_ARG_NAME, LAST_DATE_ARG_NAME)

NUM_EXAMPLES_HELP_STRING = (
    'Number of examples to use (radar images to reconstruct) per SPC date.  If '
    'you want to use all examples, leave this argument alone.')

DOWNSAMPLING_KEYS_HELP_STRING = (
    'Keys (class labels) used to create dictionary for '
    '`deep_learning_utils.sample_by_class`.  If you do not want downsampling by'
    ' class, leave this alone.')

DOWNSAMPLING_VALUES_HELP_STRING = (
    'Values (class fractions) used to create dictionary for '
    '`deep_learning_utils.sample_by_class`.  If you do not want downsampling by'
    ' class, leave this alone.')

# TODO(thunderhoser): Still need to implement these methods.
OUTPUT_DIR_HELP_STRING = (
    'Name of output directory.  Reconstructed images will be written by '
    '`upconvnet.write_predictions`, to locations therein determined by '
    '`upconvnet.find_prediction_file`.')

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + UPCONVNET_FILE_ARG_NAME, type=str, required=True,
    help=UPCONVNET_FILE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + EXAMPLE_DIR_ARG_NAME, type=str, required=True,
    help=EXAMPLE_DIR_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + FIRST_DATE_ARG_NAME, type=str, required=True,
    help=SPC_DATE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + LAST_DATE_ARG_NAME, type=str, required=True,
    help=SPC_DATE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + NUM_EXAMPLES_ARG_NAME, type=int, required=False, default=int(1e12),
    help=NUM_EXAMPLES_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + DOWNSAMPLING_KEYS_ARG_NAME, type=int, nargs='+',
    required=False, default=[0], help=DOWNSAMPLING_KEYS_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + DOWNSAMPLING_VALUES_ARG_NAME, type=float, nargs='+',
    required=False, default=[0.], help=DOWNSAMPLING_VALUES_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING)


def _apply_upconvnet_one_file(
        example_file_name, num_examples, upconvnet_model_object,
        cnn_model_object, cnn_metadata_dict, cnn_feature_layer_name,
        upconvnet_file_name, top_output_dir_name):
    """Applies upconvnet to examples from one file.

    :param example_file_name: Path to input file (will be read by
        `input_examples.read_example_file`).
    :param num_examples: Number of examples to read.
    :param upconvnet_model_object: Trained upconvnet (instance of
        `keras.models.Model` or `keras.models.Sequential`).
    :param cnn_model_object: Trained CNN (instance of
        `keras.models.Model` or `keras.models.Sequential`).
    :param cnn_metadata_dict: Dictionary returned by `cnn.read_model_metadata`.
    param cnn_feature_layer_name: Name of CNN layer whose output is the feature
        vector, which is the input to the upconvnet.
    :param upconvnet_file_name: See documentation at top of file.
    :param top_output_dir_name: Same.
    """

    # Do housekeeping.
    training_option_dict = cnn_metadata_dict[cnn.TRAINING_OPTION_DICT_KEY]
    training_option_dict[trainval_io.EXAMPLE_FILES_KEY] = [example_file_name]

    if cnn_metadata_dict[cnn.LAYER_OPERATIONS_KEY] is not None:
        generator_object = testing_io.gridrad_generator_2d_reduced(
            option_dict=training_option_dict,
            desired_num_examples=num_examples,
            list_of_operation_dicts=cnn_metadata_dict[cnn.LAYER_OPERATIONS_KEY]
        )

    elif cnn_metadata_dict[cnn.CONV_2D3D_KEY]:
        generator_object = testing_io.myrorss_generator_2d3d(
            option_dict=training_option_dict,
            desired_num_examples=num_examples)
    else:
        generator_object = testing_io.generator_2d_or_3d(
            option_dict=training_option_dict,
            desired_num_examples=num_examples)

    # Apply upconvnet.
    full_storm_id_strings = []
    storm_times_unix_sec = numpy.array([], dtype=int)
    reconstructed_radar_matrix = None
    mse_by_example = numpy.array([], dtype=float)

    while True:
        try:
            this_storm_object_dict = next(generator_object)
            print('\n')
        except StopIteration:
            break

        full_storm_id_strings += this_storm_object_dict[testing_io.FULL_IDS_KEY]
        storm_times_unix_sec = numpy.concatenate((
            storm_times_unix_sec,
            this_storm_object_dict[testing_io.STORM_TIMES_KEY]
        ))

        these_input_matrices = this_storm_object_dict[
            testing_io.INPUT_MATRICES_KEY]
        this_actual_matrix = these_input_matrices[0]

        this_reconstructed_matrix = upconvnet.apply_upconvnet(
            cnn_input_matrices=these_input_matrices,
            cnn_model_object=cnn_model_object,
            cnn_feature_layer_name=cnn_feature_layer_name,
            ucn_model_object=upconvnet_model_object, verbose=True)
        print(MINOR_SEPARATOR_STRING)

        if reconstructed_radar_matrix is None:
            reconstructed_radar_matrix = this_reconstructed_matrix + 0.
        else:
            reconstructed_radar_matrix = numpy.concatenate(
                (reconstructed_radar_matrix, this_reconstructed_matrix), axis=0
            )
        
        num_dimensions = len(this_actual_matrix.shape)
        all_axes_except_first = numpy.linspace(
            1, num_dimensions - 1, num=num_dimensions - 1, dtype=int
        ).tolist()

        these_mse = numpy.mean(
            (this_actual_matrix - this_reconstructed_matrix) ** 2,
            axis=tuple(all_axes_except_first)
        )
        mse_by_example = numpy.concatenate((mse_by_example, these_mse))

    print(MINOR_SEPARATOR_STRING)
    print('Mean sqaured error = {0:.3e}'.format(numpy.mean(mse_by_example)))

    # Denormalize reconstructed images.
    print('Denormalizing reconstructed radar images...')

    metadata_dict_no_soundings = copy.deepcopy(cnn_metadata_dict)
    metadata_dict_no_soundings[cnn.TRAINING_OPTION_DICT_KEY][
        trainval_io.SOUNDING_FIELDS_KEY
    ] = None
    option_dict_no_soundings = metadata_dict_no_soundings[
        cnn.TRAINING_OPTION_DICT_KEY
    ]

    list_of_recon_matrices = trainval_io.separate_shear_and_reflectivity(
        list_of_input_matrices=[reconstructed_radar_matrix],
        training_option_dict=option_dict_no_soundings
    )

    list_of_recon_matrices = model_interpretation.denormalize_data(
        list_of_input_matrices=list_of_recon_matrices,
        model_metadata_dict=metadata_dict_no_soundings)

    # TODO(thunderhoser): UGH, this code is very hacky.
    if len(list_of_recon_matrices) > 1:
        this_refl_matrix_dbz = trainval_io.upsample_reflectivity(
            list_of_recon_matrices[0][..., 0]
        )

        reconstructed_radar_matrix = numpy.concatenate(
            (this_refl_matrix_dbz, list_of_recon_matrices[1]), axis=-1
        )
    else:
        reconstructed_radar_matrix = list_of_recon_matrices[0]

    # Write reconstructed images.
    spc_date_string = time_conversion.time_to_spc_date_string(
        numpy.median(storm_times_unix_sec)
    )

    output_file_name = upconvnet.find_prediction_file(
        top_directory_name=top_output_dir_name,
        spc_date_string=spc_date_string, raise_error_if_missing=False)

    print('Writing predictions to: "{0:s}"...'.format(output_file_name))

    upconvnet.write_predictions(
        netcdf_file_name=output_file_name,
        denorm_recon_radar_matrix=reconstructed_radar_matrix,
        full_storm_id_strings=full_storm_id_strings,
        storm_times_unix_sec=storm_times_unix_sec,
        mse_by_example=mse_by_example, upconvnet_file_name=upconvnet_file_name)


def _run(upconvnet_file_name, top_example_dir_name, first_spc_date_string,
         last_spc_date_string, num_examples_per_date, downsampling_keys,
         downsampling_values, top_output_dir_name):
    """Makes predictions from trained upconvnet.

    This is effectively the main method.

    :param upconvnet_file_name: See documentation at top of file.
    :param top_example_dir_name: Same.
    :param first_spc_date_string: Same.
    :param last_spc_date_string: Same.
    :param num_examples_per_date: Same.
    :param downsampling_keys: Same.
    :param downsampling_values: Same.
    :param top_output_dir_name: Same.
    """

    # Process input args.
    print('Reading upconvnet from: "{0:s}"...'.format(upconvnet_file_name))
    upconvnet_model_object = cnn.read_model(upconvnet_file_name)
    upconvnet_metafile_name = cnn.find_metafile(upconvnet_file_name)

    print('Reading upconvnet metadata from: "{0:s}"...'.format(
        upconvnet_metafile_name
    ))
    upconvnet_metadata_dict = upconvnet.read_model_metadata(
        upconvnet_metafile_name
    )
    cnn_file_name = upconvnet_metadata_dict[upconvnet.CNN_FILE_KEY]

    print('Reading CNN from: "{0:s}"...'.format(cnn_file_name))
    cnn_model_object = cnn.read_model(cnn_file_name)
    cnn_metafile_name = cnn.find_metafile(cnn_file_name)

    print('Reading CNN metadata from: "{0:s}"...'.format(cnn_metafile_name))
    cnn_metadata_dict = cnn.read_model_metadata(cnn_metafile_name)
    training_option_dict = cnn_metadata_dict[cnn.TRAINING_OPTION_DICT_KEY]

    if len(downsampling_keys) > 1:
        downsampling_dict = dict(list(zip(
            downsampling_keys, downsampling_values
        )))
    else:
        downsampling_dict = None

    training_option_dict[trainval_io.SAMPLING_FRACTIONS_KEY] = downsampling_dict

    training_option_dict[trainval_io.NUM_EXAMPLES_PER_BATCH_KEY] = (
        NUM_EXAMPLES_PER_BATCH
    )
    training_option_dict[trainval_io.FIRST_STORM_TIME_KEY] = EARLY_TIME_UNIX_SEC
    training_option_dict[trainval_io.LAST_STORM_TIME_KEY] = LATE_TIME_UNIX_SEC

    # Find example files.
    example_file_names = input_examples.find_many_example_files(
        top_directory_name=top_example_dir_name, shuffled=False,
        first_spc_date_string=first_spc_date_string,
        last_spc_date_string=last_spc_date_string,
        raise_error_if_any_missing=False)

    # Do dirty work.
    for this_example_file_name in example_file_names:
        _apply_upconvnet_one_file(
            example_file_name=this_example_file_name,
            num_examples=num_examples_per_date,
            upconvnet_model_object=upconvnet_model_object,
            cnn_model_object=cnn_model_object,
            cnn_metadata_dict=cnn_metadata_dict,
            cnn_feature_layer_name=
            upconvnet_metadata_dict[upconvnet.CNN_FEATURE_LAYER_KEY],
            upconvnet_file_name=upconvnet_file_name,
            top_output_dir_name=top_output_dir_name)

        print(SEPARATOR_STRING)


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        upconvnet_file_name=getattr(INPUT_ARG_OBJECT, UPCONVNET_FILE_ARG_NAME),
        top_example_dir_name=getattr(INPUT_ARG_OBJECT, EXAMPLE_DIR_ARG_NAME),
        first_spc_date_string=getattr(
            INPUT_ARG_OBJECT, FIRST_DATE_ARG_NAME),
        last_spc_date_string=getattr(INPUT_ARG_OBJECT, LAST_DATE_ARG_NAME),
        num_examples_per_date=getattr(INPUT_ARG_OBJECT, NUM_EXAMPLES_ARG_NAME),
        downsampling_keys=numpy.array(
            getattr(INPUT_ARG_OBJECT, DOWNSAMPLING_KEYS_ARG_NAME), dtype=int),
        downsampling_values=numpy.array(
            getattr(INPUT_ARG_OBJECT, DOWNSAMPLING_VALUES_ARG_NAME),
            dtype=float
        ),
        top_output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
