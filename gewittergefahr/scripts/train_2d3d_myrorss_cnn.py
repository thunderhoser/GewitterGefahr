"""Trains convolutional neural net with 2-D and 3-D MYRORSS images.

The 2-D images contain azimuthal shear, and the 3-D images contain reflectivity.
"""

import argparse
import numpy
from keras import backend as K
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.gg_utils import labels
from gewittergefahr.gg_utils import soundings
from gewittergefahr.gg_utils import error_checking
from gewittergefahr.deep_learning import cnn
from gewittergefahr.deep_learning import cnn_architecture
from gewittergefahr.deep_learning import input_examples
from gewittergefahr.deep_learning import training_validation_io as trainval_io
from gewittergefahr.scripts import deep_learning_helper as dl_helper

K.set_session(K.tf.Session(config=K.tf.ConfigProto(
    intra_op_parallelism_threads=7, inter_op_parallelism_threads=7)))

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

FIRST_BATCH_NUMBER = 0
LAST_BATCH_NUMBER = int(1e12)

SOUNDING_HEIGHTS_M_AGL = soundings.DEFAULT_HEIGHT_LEVELS_M_AGL + 0
REFLECTIVITY_HEIGHTS_M_AGL = numpy.linspace(1000, 12000, num=12, dtype=int)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER = dl_helper.add_input_arguments(
    argument_parser_object=INPUT_ARG_PARSER)

NUM_REFL_FILTERS_ARG_NAME = 'first_num_radar_filters'
NUM_SHEAR_FILTERS_ARG_NAME = 'first_num_shear_filters'
FINAL_CONV_LAYER_SETS_ARG_NAME = 'num_final_conv_layer_sets'
NORMALIZATION_FILE_ARG_NAME = 'normalization_param_file_name'

NUM_REFL_FILTERS_HELP_STRING = (
    'Number of filters in first convolutional layer for reflectivity images.  '
    'If you make this <= 0, it will be automatically determined by '
    'cnn_architecture.py.')

NUM_SHEAR_FILTERS_HELP_STRING = (
    'Number of filters in first convolutional layer for azimuthal-shear images.'
    '  If you make this <= 0, it will be automatically determined by '
    'cnn_architecture.py.')

FINAL_CONV_LAYER_SETS_HELP_STRING = (
    'Number of convolution-layer sets for combined filters (after reflectivity '
    'and az-shear filters have been concatenated).')

NORMALIZATION_FILE_HELP_STRING = (
    'Path to file with normalization params.  Will be read by '
    '`deep_learning_utils.read_normalization_params_from_file` and used by '
    '`deep_learning_utils.normalize_radar_images` and '
    '`deep_learning_utils.normalize_soundings`.')

DEFAULT_NORMALIZATION_FILE_NAME = ''

INPUT_ARG_PARSER.add_argument(
    '--' + NUM_REFL_FILTERS_ARG_NAME, type=int, required=False,
    default=-1, help=NUM_REFL_FILTERS_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + NUM_SHEAR_FILTERS_ARG_NAME, type=int, required=False,
    default=-1, help=NUM_SHEAR_FILTERS_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + FINAL_CONV_LAYER_SETS_ARG_NAME, type=int, required=False,
    default=2, help=FINAL_CONV_LAYER_SETS_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + NORMALIZATION_FILE_ARG_NAME, type=str, required=False,
    default=DEFAULT_NORMALIZATION_FILE_NAME,
    help=NORMALIZATION_FILE_HELP_STRING)


def _run(output_model_dir_name, num_epochs, num_training_batches_per_epoch,
         top_training_dir_name, first_training_time_string,
         last_training_time_string, num_validation_batches_per_epoch,
         top_validation_dir_name, first_validation_time_string,
         last_validation_time_string, monitor_string, weight_loss_function,
         num_examples_per_batch, sounding_field_names, num_grid_rows,
         num_grid_columns, normalization_type_string,
         normalization_param_file_name, min_normalized_value,
         max_normalized_value, binarize_target, sampling_fraction_keys,
         sampling_fraction_values, num_translations, max_translation_pixels,
         num_rotations, max_absolute_rotation_angle_deg, num_noisings,
         max_noise_standard_deviation, num_conv_layers_per_set,
         pooling_type_string, activation_function_string, alpha_for_elu,
         alpha_for_relu, use_batch_normalization, conv_layer_dropout_fraction,
         dense_layer_dropout_fraction, l2_weight, first_num_sounding_filters,
         first_num_refl_filters, first_num_shear_filters,
         num_final_conv_layer_sets):
    """Trains convolutional neural net with 2-D and 3-D MYRORSS images.

    This is effectively the main method.

    :param output_model_dir_name: See documentation at top of
        deep_learning_helper.py.
    :param num_epochs: Same.
    :param num_training_batches_per_epoch: Same.
    :param top_training_dir_name: Same.
    :param first_training_time_string: Same.
    :param last_training_time_string: Same.
    :param num_validation_batches_per_epoch: Same.
    :param top_validation_dir_name: Same.
    :param first_validation_time_string: Same.
    :param last_validation_time_string: Same.
    :param monitor_string: Same.
    :param weight_loss_function: Same.
    :param num_examples_per_batch: Same.
    :param sounding_field_names: Same.
    :param num_grid_rows: Same.
    :param num_grid_columns: Same.
    :param normalization_type_string: Same.
    :param normalization_param_file_name: See documentation at top of this file.
    :param min_normalized_value: See documentation at top of
        deep_learning_helper.py.
    :param max_normalized_value: Same.
    :param binarize_target: Same.
    :param sampling_fraction_keys: Same.
    :param sampling_fraction_values: Same.
    :param num_translations: Same.
    :param max_translation_pixels: Same.
    :param num_rotations: Same.
    :param max_absolute_rotation_angle_deg: Same.
    :param num_noisings: Same.
    :param max_noise_standard_deviation: Same.
    :param num_conv_layers_per_set: Same.
    :param pooling_type_string: Same.
    :param activation_function_string: Same.
    :param alpha_for_elu: Same.
    :param alpha_for_relu: Same.
    :param use_batch_normalization: Same.
    :param conv_layer_dropout_fraction: Same.
    :param dense_layer_dropout_fraction: Same.
    :param l2_weight: Same.
    :param first_num_sounding_filters: Same.
    :param first_num_refl_filters: See documentation at top of this file.
    :param first_num_shear_filters: Same.
    :param num_final_conv_layer_sets: Same.
    """

    # Process input args.
    first_training_time_unix_sec = time_conversion.string_to_unix_sec(
        first_training_time_string, dl_helper.INPUT_TIME_FORMAT)
    last_training_time_unix_sec = time_conversion.string_to_unix_sec(
        last_training_time_string, dl_helper.INPUT_TIME_FORMAT)
    error_checking.assert_is_greater(
        last_training_time_unix_sec, first_training_time_unix_sec)

    first_validn_time_unix_sec = time_conversion.string_to_unix_sec(
        first_validation_time_string, dl_helper.INPUT_TIME_FORMAT)
    last_validn_time_unix_sec = time_conversion.string_to_unix_sec(
        last_validation_time_string, dl_helper.INPUT_TIME_FORMAT)
    error_checking.assert_is_greater(
        last_validn_time_unix_sec, first_validn_time_unix_sec)

    if sounding_field_names[0] == 'None':
        sounding_field_names = None
        num_sounding_fields = None
    else:
        num_sounding_fields = len(sounding_field_names)

    sampling_fraction_keys = numpy.array(sampling_fraction_keys, dtype=int)
    sampling_fraction_values = numpy.array(
        sampling_fraction_values, dtype=float)

    if len(sampling_fraction_keys) > 1:
        class_to_sampling_fraction_dict = dict(zip(
            sampling_fraction_keys, sampling_fraction_values))
    else:
        class_to_sampling_fraction_dict = None

    if conv_layer_dropout_fraction <= 0:
        conv_layer_dropout_fraction = None
    if dense_layer_dropout_fraction <= 0:
        dense_layer_dropout_fraction = None
    if l2_weight <= 0:
        l2_weight = None
    if first_num_refl_filters <= 0:
        first_num_refl_filters = None
    if first_num_shear_filters <= 0:
        first_num_shear_filters = None

    # Set output locations.
    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=output_model_dir_name)

    model_file_name = '{0:s}/model.h5'.format(output_model_dir_name)
    history_file_name = '{0:s}/model_history.csv'.format(output_model_dir_name)
    tensorboard_dir_name = '{0:s}/tensorboard'.format(output_model_dir_name)
    metadata_file_name = '{0:s}/model_metadata.p'.format(output_model_dir_name)

    # Find training and validation files.
    training_file_names = input_examples.find_many_example_files(
        top_directory_name=top_training_dir_name, shuffled=True,
        first_batch_number=FIRST_BATCH_NUMBER,
        last_batch_number=LAST_BATCH_NUMBER, raise_error_if_any_missing=False)

    if num_validation_batches_per_epoch > 0:
        validation_file_names = input_examples.find_many_example_files(
            top_directory_name=top_validation_dir_name, shuffled=True,
            first_batch_number=FIRST_BATCH_NUMBER,
            last_batch_number=LAST_BATCH_NUMBER,
            raise_error_if_any_missing=False)
    else:
        validation_file_names = None
        first_validn_time_unix_sec = None
        last_validn_time_unix_sec = None

    # Write metadata.
    metadata_dict = {
        cnn.NUM_EPOCHS_KEY: num_epochs,
        cnn.NUM_TRAINING_BATCHES_KEY: num_training_batches_per_epoch,
        cnn.NUM_VALIDATION_BATCHES_KEY: num_validation_batches_per_epoch,
        cnn.MONITOR_STRING_KEY: monitor_string,
        cnn.WEIGHT_LOSS_FUNCTION_KEY: weight_loss_function,
        cnn.USE_2D3D_CONVOLUTION_KEY: True,
        cnn.VALIDATION_FILES_KEY: validation_file_names,
        cnn.FIRST_VALIDN_TIME_KEY: first_validn_time_unix_sec,
        cnn.LAST_VALIDN_TIME_KEY: last_validn_time_unix_sec
    }

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
        trainval_io.BINARIZE_TARGET_KEY: binarize_target,
        trainval_io.SAMPLING_FRACTIONS_KEY: class_to_sampling_fraction_dict,
        trainval_io.LOOP_ONCE_KEY: False,
        trainval_io.NUM_TRANSLATIONS_KEY: num_translations,
        trainval_io.MAX_TRANSLATION_KEY: max_translation_pixels,
        trainval_io.NUM_ROTATIONS_KEY: num_rotations,
        trainval_io.MAX_ROTATION_KEY: max_absolute_rotation_angle_deg,
        trainval_io.NUM_NOISINGS_KEY: num_noisings,
        trainval_io.MAX_NOISE_KEY: max_noise_standard_deviation
    }

    print 'Writing metadata to: "{0:s}"...'.format(metadata_file_name)
    cnn.write_model_metadata(
        pickle_file_name=metadata_file_name, metadata_dict=metadata_dict,
        training_option_dict=training_option_dict)

    this_example_dict = input_examples.read_example_file(
        netcdf_file_name=training_file_names[0], metadata_only=True)
    target_name = this_example_dict[input_examples.TARGET_NAME_KEY]

    if binarize_target:
        num_classes_to_predict = 2
    else:
        num_classes_to_predict = labels.column_name_to_num_classes(
            column_name=target_name, include_dead_storms=False)

    reflectivity_option_dict = {
        cnn_architecture.NUM_ROWS_KEY: num_grid_rows,
        cnn_architecture.NUM_COLUMNS_KEY: num_grid_columns,
        cnn_architecture.NUM_HEIGHTS_KEY: len(REFLECTIVITY_HEIGHTS_M_AGL),
        cnn_architecture.NUM_CLASSES_KEY: num_classes_to_predict,
        cnn_architecture.NUM_CONV_LAYERS_PER_SET_KEY: num_conv_layers_per_set,
        cnn_architecture.NUM_DENSE_LAYERS_KEY: 2,
        cnn_architecture.ACTIVATION_FUNCTION_KEY: activation_function_string,
        cnn_architecture.ALPHA_FOR_ELU_KEY: alpha_for_elu,
        cnn_architecture.ALPHA_FOR_RELU_KEY: alpha_for_relu,
        cnn_architecture.USE_BATCH_NORM_KEY: use_batch_normalization,
        cnn_architecture.FIRST_NUM_FILTERS_KEY: first_num_refl_filters,
        cnn_architecture.CONV_LAYER_DROPOUT_KEY: conv_layer_dropout_fraction,
        cnn_architecture.DENSE_LAYER_DROPOUT_KEY: dense_layer_dropout_fraction,
        cnn_architecture.L2_WEIGHT_KEY: l2_weight
    }

    if num_sounding_fields is None:
        sounding_option_dict = None
    else:
        sounding_option_dict = {
            cnn_architecture.NUM_FIELDS_KEY: len(sounding_field_names),
            cnn_architecture.NUM_HEIGHTS_KEY: len(SOUNDING_HEIGHTS_M_AGL),
            cnn_architecture.FIRST_NUM_FILTERS_KEY: first_num_sounding_filters
        }

    model_object = cnn_architecture.get_2d3d_swirlnet_architecture(
        reflectivity_option_dict=reflectivity_option_dict,
        num_azimuthal_shear_fields=len(
            input_examples.AZIMUTHAL_SHEAR_FIELD_NAMES),
        num_final_conv_layer_sets=num_final_conv_layer_sets,
        first_num_az_shear_filters=first_num_shear_filters,
        sounding_option_dict=sounding_option_dict,
        pooling_type_string=pooling_type_string)
    print SEPARATOR_STRING

    cnn.train_cnn_2d3d_myrorss(
        model_object=model_object, model_file_name=model_file_name,
        history_file_name=history_file_name,
        tensorboard_dir_name=tensorboard_dir_name, num_epochs=num_epochs,
        num_training_batches_per_epoch=num_training_batches_per_epoch,
        training_option_dict=training_option_dict,
        monitor_string=monitor_string,
        weight_loss_function=weight_loss_function,
        num_validation_batches_per_epoch=num_validation_batches_per_epoch,
        validation_file_names=validation_file_names,
        first_validn_time_unix_sec=first_validn_time_unix_sec,
        last_validn_time_unix_sec=last_validn_time_unix_sec)


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        output_model_dir_name=getattr(
            INPUT_ARG_OBJECT, dl_helper.MODEL_DIRECTORY_ARG_NAME),
        num_epochs=getattr(INPUT_ARG_OBJECT, dl_helper.NUM_EPOCHS_ARG_NAME),
        num_training_batches_per_epoch=getattr(
            INPUT_ARG_OBJECT, dl_helper.NUM_TRAINING_BATCHES_ARG_NAME),
        top_training_dir_name=getattr(
            INPUT_ARG_OBJECT, dl_helper.TRAINING_DIR_ARG_NAME),
        first_training_time_string=getattr(
            INPUT_ARG_OBJECT, dl_helper.FIRST_TRAINING_TIME_ARG_NAME),
        last_training_time_string=getattr(
            INPUT_ARG_OBJECT, dl_helper.LAST_TRAINING_TIME_ARG_NAME),
        num_validation_batches_per_epoch=getattr(
            INPUT_ARG_OBJECT, dl_helper.NUM_VALIDN_BATCHES_ARG_NAME),
        top_validation_dir_name=getattr(
            INPUT_ARG_OBJECT, dl_helper.VALIDATION_DIR_ARG_NAME),
        first_validation_time_string=getattr(
            INPUT_ARG_OBJECT, dl_helper.FIRST_VALIDN_TIME_ARG_NAME),
        last_validation_time_string=getattr(
            INPUT_ARG_OBJECT, dl_helper.LAST_VALIDN_TIME_ARG_NAME),
        monitor_string=getattr(
            INPUT_ARG_OBJECT, dl_helper.MONITOR_STRING_ARG_NAME),
        weight_loss_function=bool(getattr(
            INPUT_ARG_OBJECT, dl_helper.WEIGHT_LOSS_ARG_NAME)),
        num_examples_per_batch=getattr(
            INPUT_ARG_OBJECT, dl_helper.NUM_EXAMPLES_PER_BATCH_ARG_NAME),
        sounding_field_names=getattr(
            INPUT_ARG_OBJECT, dl_helper.SOUNDING_FIELDS_ARG_NAME),
        num_grid_rows=getattr(INPUT_ARG_OBJECT, dl_helper.NUM_ROWS_ARG_NAME),
        num_grid_columns=getattr(
            INPUT_ARG_OBJECT, dl_helper.NUM_COLUMNS_ARG_NAME),
        normalization_type_string=getattr(
            INPUT_ARG_OBJECT, dl_helper.NORMALIZATION_TYPE_ARG_NAME),
        normalization_param_file_name=getattr(
            INPUT_ARG_OBJECT, NORMALIZATION_FILE_ARG_NAME),
        min_normalized_value=getattr(
            INPUT_ARG_OBJECT, dl_helper.MIN_NORM_VALUE_ARG_NAME),
        max_normalized_value=getattr(
            INPUT_ARG_OBJECT, dl_helper.MAX_NORM_VALUE_ARG_NAME),
        binarize_target=bool(getattr(
            INPUT_ARG_OBJECT, dl_helper.BINARIZE_TARGET_ARG_NAME)),
        sampling_fraction_keys=getattr(
            INPUT_ARG_OBJECT, dl_helper.SAMPLING_FRACTION_KEYS_ARG_NAME),
        sampling_fraction_values=getattr(
            INPUT_ARG_OBJECT, dl_helper.SAMPLING_FRACTION_VALUES_ARG_NAME),
        num_translations=getattr(
            INPUT_ARG_OBJECT, dl_helper.NUM_TRANSLATIONS_ARG_NAME),
        max_translation_pixels=getattr(
            INPUT_ARG_OBJECT, dl_helper.MAX_TRANSLATION_ARG_NAME),
        num_rotations=getattr(
            INPUT_ARG_OBJECT, dl_helper.NUM_ROTATIONS_ARG_NAME),
        max_absolute_rotation_angle_deg=getattr(
            INPUT_ARG_OBJECT, dl_helper.MAX_ROTATION_ARG_NAME),
        num_noisings=getattr(INPUT_ARG_OBJECT, dl_helper.NUM_NOISINGS_ARG_NAME),
        max_noise_standard_deviation=getattr(
            INPUT_ARG_OBJECT, dl_helper.MAX_NOISE_ARG_NAME),
        num_conv_layers_per_set=getattr(
            INPUT_ARG_OBJECT, dl_helper.NUM_CONV_LAYERS_PER_SET_ARG_NAME),
        pooling_type_string=getattr(
            INPUT_ARG_OBJECT, dl_helper.POOLING_TYPE_ARG_NAME),
        activation_function_string=getattr(
            INPUT_ARG_OBJECT, dl_helper.ACTIVATION_FUNCTION_ARG_NAME),
        alpha_for_elu=getattr(
            INPUT_ARG_OBJECT, dl_helper.ALPHA_FOR_ELU_ARG_NAME),
        alpha_for_relu=getattr(
            INPUT_ARG_OBJECT, dl_helper.ALPHA_FOR_RELU_ARG_NAME),
        use_batch_normalization=bool(getattr(
            INPUT_ARG_OBJECT, dl_helper.USE_BATCH_NORM_ARG_NAME)),
        conv_layer_dropout_fraction=getattr(
            INPUT_ARG_OBJECT, dl_helper.CONV_LAYER_DROPOUT_ARG_NAME),
        dense_layer_dropout_fraction=getattr(
            INPUT_ARG_OBJECT, dl_helper.DENSE_LAYER_DROPOUT_ARG_NAME),
        l2_weight=getattr(INPUT_ARG_OBJECT, dl_helper.L2_WEIGHT_ARG_NAME),
        first_num_sounding_filters=getattr(
            INPUT_ARG_OBJECT, dl_helper.NUM_SOUNDING_FILTERS_ARG_NAME),
        first_num_refl_filters=getattr(
            INPUT_ARG_OBJECT, NUM_REFL_FILTERS_ARG_NAME),
        first_num_shear_filters=getattr(
            INPUT_ARG_OBJECT, NUM_SHEAR_FILTERS_ARG_NAME),
        num_final_conv_layer_sets=getattr(
            INPUT_ARG_OBJECT, FINAL_CONV_LAYER_SETS_ARG_NAME)
    )
