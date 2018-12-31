"""Trains CNN with 2-D and 3-D MYRORSS images.

The 2-D images contain azimuthal shear, and the 3-D images contain reflectivity.
"""

import argparse
import numpy
import keras
from keras import backend as K
from gewittergefahr.gg_utils import soundings
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.deep_learning import cnn
from gewittergefahr.deep_learning import cnn_setup
from gewittergefahr.deep_learning import input_examples
from gewittergefahr.deep_learning import training_validation_io as trainval_io
from gewittergefahr.scripts import deep_learning_helper as dl_helper

K.set_session(K.tf.Session(config=K.tf.ConfigProto(
    intra_op_parallelism_threads=7, inter_op_parallelism_threads=7
)))

TIME_FORMAT = '%Y-%m-%d-%H%M%S'
SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

FIRST_BATCH_NUMBER = 0
LAST_BATCH_NUMBER = int(1e12)

REFLECTIVITY_HEIGHTS_M_AGL = numpy.linspace(1000, 12000, num=12, dtype=int)
SOUNDING_HEIGHTS_M_AGL = soundings.DEFAULT_HEIGHT_LEVELS_M_AGL + 0

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER = dl_helper.add_input_args(argument_parser=INPUT_ARG_PARSER)


def _run(input_model_file_name, sounding_field_names, normalization_type_string,
         normalization_param_file_name, min_normalized_value,
         max_normalized_value, downsampling_keys, downsampling_fractions,
         monitor_string, weight_loss_function, num_translations,
         max_translation_pixels, num_rotations, max_rotation_angle_deg,
         num_noisings, max_noise_standard_deviation, top_training_dir_name,
         first_training_time_string, last_training_time_string,
         top_validation_dir_name, first_validation_time_string,
         last_validation_time_string, num_examples_per_batch, num_epochs,
         num_training_batches_per_epoch, num_validation_batches_per_epoch,
         output_dir_name):
    """Trains CNN with native (3-D) GridRad images.

    This is effectively the main method.

    :param input_model_file_name: See documentation at top of file.
    :param sounding_field_names: Same.
    :param normalization_type_string: Same.
    :param normalization_param_file_name: Same.
    :param min_normalized_value: Same.
    :param max_normalized_value: Same.
    :param downsampling_keys: Same.
    :param downsampling_fractions: Same.
    :param monitor_string: Same.
    :param weight_loss_function: Same.
    :param num_translations: Same.
    :param max_translation_pixels: Same.
    :param num_rotations: Same.
    :param max_rotation_angle_deg: Same.
    :param num_noisings: Same.
    :param max_noise_standard_deviation: Same.
    :param top_training_dir_name: Same.
    :param first_training_time_string: Same.
    :param last_training_time_string: Same.
    :param top_validation_dir_name: Same.
    :param first_validation_time_string: Same.
    :param last_validation_time_string: Same.
    :param num_examples_per_batch: Same.
    :param num_epochs: Same.
    :param num_training_batches_per_epoch: Same.
    :param num_validation_batches_per_epoch: Same.
    :param output_dir_name: Same.
    """

    # Process input args.
    first_training_time_unix_sec = time_conversion.string_to_unix_sec(
        first_training_time_string, TIME_FORMAT)
    last_training_time_unix_sec = time_conversion.string_to_unix_sec(
        last_training_time_string, TIME_FORMAT)

    first_validation_time_unix_sec = time_conversion.string_to_unix_sec(
        first_validation_time_string, TIME_FORMAT)
    last_validation_time_unix_sec = time_conversion.string_to_unix_sec(
        last_validation_time_string, TIME_FORMAT)

    if sounding_field_names[0] in ['', 'None']:
        sounding_field_names = None

    if len(downsampling_keys) > 1:
        class_to_sampling_fraction_dict = dict(zip(
            downsampling_keys, downsampling_fractions
        ))
    else:
        class_to_sampling_fraction_dict = None

    # Set output locations.
    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=output_dir_name)

    output_model_file_name = '{0:s}/model.h5'.format(output_dir_name)
    history_file_name = '{0:s}/model_history.csv'.format(output_dir_name)
    tensorboard_dir_name = '{0:s}/tensorboard'.format(output_dir_name)
    model_metafile_name = '{0:s}/model_metadata.p'.format(output_dir_name)

    # Find training and validation files.
    training_file_names = input_examples.find_many_example_files(
        top_directory_name=top_training_dir_name, shuffled=True,
        first_batch_number=FIRST_BATCH_NUMBER,
        last_batch_number=LAST_BATCH_NUMBER, raise_error_if_any_missing=False)

    validation_file_names = input_examples.find_many_example_files(
        top_directory_name=top_validation_dir_name, shuffled=True,
        first_batch_number=FIRST_BATCH_NUMBER,
        last_batch_number=LAST_BATCH_NUMBER, raise_error_if_any_missing=False)

    # Read architecture.
    print 'Reading architecture from: "{0:s}"...'.format(input_model_file_name)
    model_object = cnn.read_model(input_model_file_name)
    model_object = keras.models.clone_model(model_object)

    # TODO(thunderhoser): This is a HACK.
    model_object.compile(
        loss=keras.losses.binary_crossentropy,
        optimizer=keras.optimizers.Adam(),
        metrics=cnn_setup.DEFAULT_METRIC_FUNCTION_LIST)

    print SEPARATOR_STRING
    model_object.summary()
    print SEPARATOR_STRING

    # Write metadata.
    this_example_dict = input_examples.read_example_file(
        netcdf_file_name=training_file_names[0], metadata_only=True)
    target_name = this_example_dict[input_examples.TARGET_NAME_KEY]

    metadata_dict = {
        cnn.TARGET_NAME_KEY: target_name,
        cnn.NUM_EPOCHS_KEY: num_epochs,
        cnn.NUM_TRAINING_BATCHES_KEY: num_training_batches_per_epoch,
        cnn.NUM_VALIDATION_BATCHES_KEY: num_validation_batches_per_epoch,
        cnn.MONITOR_STRING_KEY: monitor_string,
        cnn.WEIGHT_LOSS_FUNCTION_KEY: weight_loss_function,
        cnn.USE_2D3D_CONVOLUTION_KEY: True,
        cnn.VALIDATION_FILES_KEY: validation_file_names,
        cnn.FIRST_VALIDN_TIME_KEY: first_validation_time_unix_sec,
        cnn.LAST_VALIDN_TIME_KEY: last_validation_time_unix_sec
    }

    input_tensor = model_object.input
    if isinstance(input_tensor, list):
        input_tensor = input_tensor[0]

    num_grid_rows = input_tensor.get_shape().as_list()[1]
    num_grid_columns = input_tensor.get_shape().as_list()[2]

    training_option_dict = {
        trainval_io.EXAMPLE_FILES_KEY: training_file_names,
        trainval_io.FIRST_STORM_TIME_KEY: first_training_time_unix_sec,
        trainval_io.LAST_STORM_TIME_KEY: last_training_time_unix_sec,
        trainval_io.NUM_EXAMPLES_PER_BATCH_KEY: num_examples_per_batch,
        trainval_io.RADAR_FIELDS_KEY:
            input_examples.AZIMUTHAL_SHEAR_FIELD_NAMES,
        trainval_io.RADAR_HEIGHTS_KEY: REFLECTIVITY_HEIGHTS_M_AGL,
        trainval_io.SOUNDING_FIELDS_KEY: sounding_field_names,
        trainval_io.SOUNDING_HEIGHTS_KEY: SOUNDING_HEIGHTS_M_AGL,
        trainval_io.NUM_ROWS_KEY: num_grid_rows,
        trainval_io.NUM_COLUMNS_KEY: num_grid_columns,
        trainval_io.NORMALIZATION_TYPE_KEY: normalization_type_string,
        trainval_io.NORMALIZATION_FILE_KEY: normalization_param_file_name,
        trainval_io.MIN_NORMALIZED_VALUE_KEY: min_normalized_value,
        trainval_io.MAX_NORMALIZED_VALUE_KEY: max_normalized_value,
        trainval_io.BINARIZE_TARGET_KEY: False,
        trainval_io.SAMPLING_FRACTIONS_KEY: class_to_sampling_fraction_dict,
        trainval_io.LOOP_ONCE_KEY: False,
        trainval_io.NUM_TRANSLATIONS_KEY: num_translations,
        trainval_io.MAX_TRANSLATION_KEY: max_translation_pixels,
        trainval_io.NUM_ROTATIONS_KEY: num_rotations,
        trainval_io.MAX_ROTATION_KEY: max_rotation_angle_deg,
        trainval_io.NUM_NOISINGS_KEY: num_noisings,
        trainval_io.MAX_NOISE_KEY: max_noise_standard_deviation
    }

    print 'Writing metadata to: "{0:s}"...'.format(model_metafile_name)
    cnn.write_model_metadata(
        pickle_file_name=model_metafile_name, metadata_dict=metadata_dict,
        training_option_dict=training_option_dict)

    cnn.train_cnn_2d3d_myrorss(
        model_object=model_object, model_file_name=output_model_file_name,
        history_file_name=history_file_name,
        tensorboard_dir_name=tensorboard_dir_name, num_epochs=num_epochs,
        num_training_batches_per_epoch=num_training_batches_per_epoch,
        training_option_dict=training_option_dict,
        monitor_string=monitor_string,
        weight_loss_function=weight_loss_function,
        num_validation_batches_per_epoch=num_validation_batches_per_epoch,
        validation_file_names=validation_file_names,
        first_validn_time_unix_sec=first_validation_time_unix_sec,
        last_validn_time_unix_sec=last_validation_time_unix_sec)


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        input_model_file_name=getattr(
            INPUT_ARG_OBJECT, dl_helper.INPUT_MODEL_FILE_ARG_NAME),
        sounding_field_names=getattr(
            INPUT_ARG_OBJECT, dl_helper.SOUNDING_FIELDS_ARG_NAME),
        normalization_type_string=getattr(
            INPUT_ARG_OBJECT, dl_helper.NORMALIZATION_TYPE_ARG_NAME),
        normalization_param_file_name=getattr(
            INPUT_ARG_OBJECT, dl_helper.NORMALIZATION_FILE_ARG_NAME),
        min_normalized_value=getattr(
            INPUT_ARG_OBJECT, dl_helper.MIN_NORM_VALUE_ARG_NAME),
        max_normalized_value=getattr(
            INPUT_ARG_OBJECT, dl_helper.MAX_NORM_VALUE_ARG_NAME),
        downsampling_keys=numpy.array(
            getattr(INPUT_ARG_OBJECT, dl_helper.DOWNSAMPLING_KEYS_ARG_NAME),
            dtype=int),
        downsampling_fractions=numpy.array(
            getattr(INPUT_ARG_OBJECT,
                    dl_helper.DOWNSAMPLING_FRACTIONS_ARG_NAME),
            dtype=float),
        monitor_string=getattr(INPUT_ARG_OBJECT, dl_helper.MONITOR_ARG_NAME),
        weight_loss_function=bool(getattr(
            INPUT_ARG_OBJECT, dl_helper.WEIGHT_LOSS_ARG_NAME)),
        num_translations=getattr(
            INPUT_ARG_OBJECT, dl_helper.NUM_TRANSLATIONS_ARG_NAME),
        max_translation_pixels=getattr(
            INPUT_ARG_OBJECT, dl_helper.MAX_TRANSLATION_ARG_NAME),
        num_rotations=getattr(
            INPUT_ARG_OBJECT, dl_helper.NUM_ROTATIONS_ARG_NAME),
        max_rotation_angle_deg=getattr(
            INPUT_ARG_OBJECT, dl_helper.MAX_ROTATION_ARG_NAME),
        num_noisings=getattr(INPUT_ARG_OBJECT, dl_helper.NUM_NOISINGS_ARG_NAME),
        max_noise_standard_deviation=getattr(
            INPUT_ARG_OBJECT, dl_helper.MAX_NOISE_ARG_NAME),
        top_training_dir_name=getattr(
            INPUT_ARG_OBJECT, dl_helper.TRAINING_DIR_ARG_NAME),
        first_training_time_string=getattr(
            INPUT_ARG_OBJECT, dl_helper.FIRST_TRAINING_TIME_ARG_NAME),
        last_training_time_string=getattr(
            INPUT_ARG_OBJECT, dl_helper.LAST_TRAINING_TIME_ARG_NAME),
        top_validation_dir_name=getattr(
            INPUT_ARG_OBJECT, dl_helper.VALIDATION_DIR_ARG_NAME),
        first_validation_time_string=getattr(
            INPUT_ARG_OBJECT, dl_helper.FIRST_VALIDATION_TIME_ARG_NAME),
        last_validation_time_string=getattr(
            INPUT_ARG_OBJECT, dl_helper.LAST_VALIDATION_TIME_ARG_NAME),
        num_examples_per_batch=getattr(
            INPUT_ARG_OBJECT, dl_helper.NUM_EX_PER_BATCH_ARG_NAME),
        num_epochs=getattr(INPUT_ARG_OBJECT, dl_helper.NUM_EPOCHS_ARG_NAME),
        num_training_batches_per_epoch=getattr(
            INPUT_ARG_OBJECT, dl_helper.NUM_TRAINING_BATCHES_ARG_NAME),
        num_validation_batches_per_epoch=getattr(
            INPUT_ARG_OBJECT, dl_helper.NUM_VALIDATION_BATCHES_ARG_NAME),
        output_dir_name=getattr(INPUT_ARG_OBJECT, dl_helper.OUTPUT_DIR_ARG_NAME)
    )
