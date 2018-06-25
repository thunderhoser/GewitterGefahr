"""Trains CNN with 3-D GridRad images."""

import argparse
import numpy
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.gg_utils import labels
from gewittergefahr.gg_utils import radar_utils
from gewittergefahr.gg_utils import gridrad_utils
from gewittergefahr.deep_learning import cnn
from gewittergefahr.deep_learning import training_validation_io as trainval_io
from gewittergefahr.deep_learning import deep_learning_utils as dl_utils
from gewittergefahr.scripts import deep_learning_helper as dl_helper

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

NUM_RADAR_ROWS = 16
NUM_RADAR_COLUMNS = 16
NUM_SOUNDING_HEIGHTS = 37
RADAR_HEIGHTS_M_ASL = numpy.linspace(1000, 12000, num=12, dtype=int)
RADAR_NORMALIZATION_DICT = dl_utils.DEFAULT_NORMALIZATION_DICT
SOUNDING_NORMALIZATION_DICT = dl_utils.DEFAULT_SOUNDING_NORMALIZATION_DICT

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER = dl_helper.add_input_arguments(
    argument_parser_object=INPUT_ARG_PARSER)

NUM_RADAR_FILTERS_ARG_NAME = 'num_radar_filters_in_first_conv_layer'
NUM_RADAR_FILTERS_HELP_STRING = (
    'Number of radar filters in first convolutional layer.  Number of filters '
    'will double for each successive layer convolving over radar images.')

INPUT_ARG_PARSER.add_argument(
    '--' + NUM_RADAR_FILTERS_ARG_NAME, type=int, required=False,
    default=16, help=NUM_RADAR_FILTERS_HELP_STRING)


def _train_cnn(
        output_model_dir_name, num_epochs, num_examples_per_batch,
        num_examples_per_file_time, num_training_batches_per_epoch,
        top_storm_radar_image_dir_name, one_file_per_time_step,
        first_training_time_string, last_training_time_string,
        radar_field_names, target_name, top_target_dir_name, binarize_target,
        num_radar_filters_in_first_conv_layer, dropout_fraction, l2_weight,
        sampling_fraction_dict_keys, sampling_fraction_dict_values,
        weight_loss_function, num_validation_batches_per_epoch,
        first_validation_time_string, last_validation_time_string,
        sounding_field_names, top_sounding_dir_name,
        sounding_lag_time_for_convective_contamination_sec,
        num_sounding_filters_in_first_conv_layer):
    """Trains CNN with 3-D GridRad images.

    :param output_model_dir_name: See documentation at the top of
        'scripts/deep_learning.py'.
    :param num_epochs: Same.
    :param num_examples_per_batch: Same.
    :param num_examples_per_file_time: Same.
    :param num_training_batches_per_epoch: Same.
    :param top_storm_radar_image_dir_name: Same.
    :param one_file_per_time_step: Same.
    :param first_training_time_string: Same.
    :param last_training_time_string: Same.
    :param radar_field_names: Same.
    :param target_name: Same.
    :param top_target_dir_name: Same.
    :param binarize_target: Same.
    :param num_radar_filters_in_first_conv_layer: See documentation at the top
        of this file.
    :param dropout_fraction: See documentation at the top of
        'scripts/deep_learning.py'.
    :param l2_weight: Same.
    :param sampling_fraction_dict_keys: Same.
    :param sampling_fraction_dict_values: Same.
    :param weight_loss_function: Same.
    :param num_validation_batches_per_epoch: Same.
    :param first_validation_time_string: Same.
    :param last_validation_time_string: Same.
    :param sounding_field_names: Same.
    :param top_sounding_dir_name: Same.
    :param sounding_lag_time_for_convective_contamination_sec: Same.
    :param num_sounding_filters_in_first_conv_layer: Same.
    """

    # Convert inputs.
    if dropout_fraction <= 0.:
        dropout_fraction = None
    if l2_weight <= 0.:
        l2_weight = None

    sampling_fraction_dict_keys = numpy.array(
        sampling_fraction_dict_keys, dtype=int)
    sampling_fraction_dict_values = numpy.array(
        sampling_fraction_dict_values, dtype=float)

    if len(sampling_fraction_dict_keys) > 1:
        sampling_fraction_by_class_dict = dict(zip(
            sampling_fraction_dict_keys, sampling_fraction_dict_values))
    else:
        sampling_fraction_by_class_dict = None

    first_train_time_unix_sec = time_conversion.string_to_unix_sec(
        first_training_time_string, dl_helper.INPUT_TIME_FORMAT)
    last_train_time_unix_sec = time_conversion.string_to_unix_sec(
        last_training_time_string, dl_helper.INPUT_TIME_FORMAT)

    if num_validation_batches_per_epoch <= 0:
        num_validation_batches_per_epoch = None
        first_validn_time_unix_sec = None
        last_validn_time_unix_sec = None
    else:
        first_validn_time_unix_sec = time_conversion.string_to_unix_sec(
            first_validation_time_string, dl_helper.INPUT_TIME_FORMAT)
        last_validn_time_unix_sec = time_conversion.string_to_unix_sec(
            last_validation_time_string, dl_helper.INPUT_TIME_FORMAT)

    if sounding_field_names[0] == '':
        sounding_field_names = None
        num_sounding_fields = None
    else:
        num_sounding_fields = len(sounding_field_names)

    # Error-checking.
    gridrad_utils.fields_and_refl_heights_to_pairs(
        field_names=radar_field_names, heights_m_asl=RADAR_HEIGHTS_M_ASL)

    # Set locations of output files.
    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=output_model_dir_name)
    model_file_name = '{0:s}/model.h5'.format(output_model_dir_name)
    history_file_name = '{0:s}/model_history.csv'.format(output_model_dir_name)
    tensorboard_dir_name = '{0:s}/tensorboard'.format(output_model_dir_name)
    metadata_file_name = '{0:s}/model_metadata.p'.format(output_model_dir_name)

    # Find input files for training.
    radar_file_name_matrix_for_training, _ = trainval_io.find_radar_files_3d(
        top_directory_name=top_storm_radar_image_dir_name,
        radar_source=radar_utils.GRIDRAD_SOURCE_ID,
        radar_field_names=radar_field_names,
        radar_heights_m_asl=RADAR_HEIGHTS_M_ASL,
        first_file_time_unix_sec=first_train_time_unix_sec,
        last_file_time_unix_sec=last_train_time_unix_sec,
        one_file_per_time_step=one_file_per_time_step)
    print SEPARATOR_STRING

    if num_validation_batches_per_epoch is None:
        radar_file_name_matrix_for_validn = None
    else:
        radar_file_name_matrix_for_validn, _ = trainval_io.find_radar_files_3d(
            top_directory_name=top_storm_radar_image_dir_name,
            radar_source=radar_utils.GRIDRAD_SOURCE_ID,
            radar_field_names=radar_field_names,
            radar_heights_m_asl=RADAR_HEIGHTS_M_ASL,
            first_file_time_unix_sec=first_validn_time_unix_sec,
            last_file_time_unix_sec=last_validn_time_unix_sec,
            one_file_per_time_step=one_file_per_time_step)
        print SEPARATOR_STRING

    print 'Writing metadata to: "{0:s}"...\n'.format(metadata_file_name)
    cnn.write_model_metadata(
        pickle_file_name=metadata_file_name, num_epochs=num_epochs,
        num_examples_per_batch=num_examples_per_batch,
        num_examples_per_file_time=num_examples_per_file_time,
        num_training_batches_per_epoch=num_training_batches_per_epoch,
        radar_file_name_matrix_for_training=radar_file_name_matrix_for_training,
        weight_loss_function=weight_loss_function, target_name=target_name,
        binarize_target=binarize_target,
        radar_normalization_dict=RADAR_NORMALIZATION_DICT,
        training_fraction_by_class_dict=sampling_fraction_by_class_dict,
        num_validation_batches_per_epoch=num_validation_batches_per_epoch,
        validation_fraction_by_class_dict=sampling_fraction_by_class_dict,
        radar_file_name_matrix_for_validn=radar_file_name_matrix_for_validn,
        sounding_field_names=sounding_field_names,
        top_sounding_dir_name=top_sounding_dir_name,
        sounding_lag_time_for_convective_contamination_sec=
        sounding_lag_time_for_convective_contamination_sec,
        sounding_normalization_dict=SOUNDING_NORMALIZATION_DICT)

    num_classes = labels.column_name_to_num_classes(target_name)
    model_object = cnn.get_3d_swilrnet_architecture(
        num_radar_rows=NUM_RADAR_ROWS, num_radar_columns=NUM_RADAR_COLUMNS,
        num_radar_heights=len(RADAR_HEIGHTS_M_ASL),
        num_radar_fields=len(radar_field_names),
        num_classes=num_classes,
        num_radar_filters_in_first_conv_layer=
        num_radar_filters_in_first_conv_layer,
        dropout_fraction=dropout_fraction, l2_weight=l2_weight,
        num_sounding_heights=NUM_SOUNDING_HEIGHTS,
        num_sounding_fields=num_sounding_fields,
        num_sounding_filters_in_first_conv_layer=
        num_sounding_filters_in_first_conv_layer)
    print SEPARATOR_STRING

    cnn.train_3d_cnn(
        model_object=model_object, model_file_name=model_file_name,
        history_file_name=history_file_name,
        tensorboard_dir_name=tensorboard_dir_name,
        num_epochs=num_epochs, num_examples_per_batch=num_examples_per_batch,
        num_examples_per_file_time=num_examples_per_file_time,
        num_training_batches_per_epoch=num_training_batches_per_epoch,
        radar_file_name_matrix_for_training=radar_file_name_matrix_for_training,
        target_name=target_name, top_target_directory_name=top_target_dir_name,
        binarize_target=binarize_target,
        weight_loss_function=weight_loss_function,
        training_fraction_by_class_dict=sampling_fraction_by_class_dict,
        num_validation_batches_per_epoch=num_validation_batches_per_epoch,
        validation_fraction_by_class_dict=sampling_fraction_by_class_dict,
        radar_file_name_matrix_for_validn=radar_file_name_matrix_for_validn,
        sounding_field_names=sounding_field_names,
        top_sounding_dir_name=top_sounding_dir_name,
        sounding_lag_time_for_convective_contamination_sec=
        sounding_lag_time_for_convective_contamination_sec)


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _train_cnn(
        output_model_dir_name=getattr(
            INPUT_ARG_OBJECT, dl_helper.MODEL_DIRECTORY_ARG_NAME),
        num_epochs=getattr(
            INPUT_ARG_OBJECT, dl_helper.NUM_EPOCHS_ARG_NAME),
        num_examples_per_batch=getattr(
            INPUT_ARG_OBJECT, dl_helper.NUM_EXAMPLES_PER_BATCH_ARG_NAME),
        num_examples_per_file_time=getattr(
            INPUT_ARG_OBJECT, dl_helper.NUM_EXAMPLES_PER_FILE_TIME_ARG_NAME),
        num_training_batches_per_epoch=getattr(
            INPUT_ARG_OBJECT, dl_helper.NUM_TRAIN_BATCHES_ARG_NAME),
        top_storm_radar_image_dir_name=getattr(
            INPUT_ARG_OBJECT, dl_helper.RADAR_DIRECTORY_ARG_NAME),
        one_file_per_time_step=bool(getattr(
            INPUT_ARG_OBJECT, dl_helper.ONE_FILE_PER_TIME_STEP_ARG_NAME)),
        first_training_time_string=getattr(
            INPUT_ARG_OBJECT, dl_helper.FIRST_TRAINING_TIME_ARG_NAME),
        last_training_time_string=getattr(
            INPUT_ARG_OBJECT, dl_helper.LAST_TRAINING_TIME_ARG_NAME),
        radar_field_names=getattr(
            INPUT_ARG_OBJECT, dl_helper.RADAR_FIELD_NAMES_ARG_NAME),
        target_name=getattr(
            INPUT_ARG_OBJECT, dl_helper.TARGET_NAME_ARG_NAME),
        top_target_dir_name=getattr(
            INPUT_ARG_OBJECT, dl_helper.TARGET_DIRECTORY_ARG_NAME),
        binarize_target=bool(getattr(
            INPUT_ARG_OBJECT, dl_helper.BINARIZE_TARGET_ARG_NAME)),
        num_radar_filters_in_first_conv_layer=getattr(
            INPUT_ARG_OBJECT, NUM_RADAR_FILTERS_ARG_NAME),
        dropout_fraction=getattr(
            INPUT_ARG_OBJECT, dl_helper.DROPOUT_FRACTION_ARG_NAME),
        l2_weight=getattr(
            INPUT_ARG_OBJECT, dl_helper.L2_WEIGHT_ARG_NAME),
        sampling_fraction_dict_keys=getattr(
            INPUT_ARG_OBJECT, dl_helper.SAMPLING_FRACTION_KEYS_ARG_NAME),
        sampling_fraction_dict_values=getattr(
            INPUT_ARG_OBJECT, dl_helper.SAMPLING_FRACTION_VALUES_ARG_NAME),
        weight_loss_function=bool(getattr(
            INPUT_ARG_OBJECT, dl_helper.WEIGHT_LOSS_ARG_NAME)),
        num_validation_batches_per_epoch=getattr(
            INPUT_ARG_OBJECT, dl_helper.NUM_VALIDN_BATCHES_ARG_NAME),
        first_validation_time_string=getattr(
            INPUT_ARG_OBJECT, dl_helper.FIRST_VALIDATION_TIME_ARG_NAME),
        last_validation_time_string=getattr(
            INPUT_ARG_OBJECT, dl_helper.LAST_VALIDATION_TIME_ARG_NAME),
        sounding_field_names=getattr(
            INPUT_ARG_OBJECT, dl_helper.SOUNDING_FIELD_NAMES_ARG_NAME),
        top_sounding_dir_name=getattr(
            INPUT_ARG_OBJECT, dl_helper.SOUNDING_DIRECTORY_ARG_NAME),
        sounding_lag_time_for_convective_contamination_sec=getattr(
            INPUT_ARG_OBJECT, dl_helper.SOUNDING_LAG_TIME_ARG_NAME),
        num_sounding_filters_in_first_conv_layer=getattr(
            INPUT_ARG_OBJECT, dl_helper.NUM_SOUNDING_FILTERS_ARG_NAME))
