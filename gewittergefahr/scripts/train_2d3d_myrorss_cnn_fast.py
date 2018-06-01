"""Trains hybrid 2D/3D convolutional neural net with MYRORSS data."""

import copy
import argparse
import numpy
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.gg_utils import labels
from gewittergefahr.gg_utils import radar_utils
from gewittergefahr.gg_utils import myrorss_and_mrms_utils
from gewittergefahr.deep_learning import cnn
from gewittergefahr.deep_learning import training_validation_io as trainval_io
from gewittergefahr.deep_learning import deep_learning_utils as dl_utils
from gewittergefahr.scripts import deep_learning as dl_script_helper

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

NORMALIZE_BY_BATCH = False
NORMALIZATION_DICT = copy.deepcopy(dl_utils.DEFAULT_NORMALIZATION_DICT)

AZIMUTHAL_SHEAR_FIELD_NAMES = radar_utils.SHEAR_NAMES
RADAR_FIELD_NAMES = [radar_utils.REFL_NAME] + AZIMUTHAL_SHEAR_FIELD_NAMES
REFLECTIVITY_HEIGHTS_M_ASL = numpy.linspace(1000, 12000, num=12, dtype=int)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER = dl_script_helper.add_input_arguments(
    argument_parser_object=INPUT_ARG_PARSER)

FIRST_NUM_REFL_FILTERS_ARG_NAME = 'first_num_reflectivity_filters'
FIRST_NUM_REFL_FILTERS_HELP_STRING = (
    'Number of reflectivity filters in first convolution layer.  For more '
    'details, see `cnn.get_architecture_for_2d3d_myrorss`.')

INPUT_ARG_PARSER.add_argument(
    '--' + FIRST_NUM_REFL_FILTERS_ARG_NAME, type=int, required=True,
    help=FIRST_NUM_REFL_FILTERS_HELP_STRING)


def _train_2d3d_myrorss_cnn(
        output_model_dir_name, num_epochs, num_training_batches_per_epoch,
        input_storm_image_dir_name, input_target_dir_name,
        num_examples_per_batch, training_start_time_string,
        training_end_time_string, target_name, weight_loss_function,
        binarize_target, class_fraction_dict_keys, class_fraction_dict_values,
        num_validation_batches_per_epoch, validation_start_time_string,
        validation_end_time_string, dropout_fraction, l2_weight,
        first_num_reflectivity_filters):
    """Trains hybrid 2D/3D convolutional neural net with MYRORSS data.

    :param output_model_dir_name: See documentation at the top of
        `scripts/deep_learning.py`.
    :param num_epochs: Same.
    :param num_training_batches_per_epoch: Same.
    :param input_storm_image_dir_name: Same.
    :param input_target_dir_name: Same.
    :param num_examples_per_batch: Same.
    :param training_start_time_string: Same.
    :param training_end_time_string: Same.
    :param target_name: Same.
    :param weight_loss_function: Same.
    :param binarize_target: Same.
    :param class_fraction_dict_keys: Same.
    :param class_fraction_dict_values: Same.
    :param num_validation_batches_per_epoch: Same.
    :param validation_start_time_string: Same.
    :param validation_end_time_string: Same.
    :param dropout_fraction: Same.
    :param l2_weight: Same.
    :param first_num_reflectivity_filters: Same.
    """

    # Make output directory; determine names of output files.
    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=output_model_dir_name)
    model_file_name = '{0:s}/model.h5'.format(output_model_dir_name)
    history_file_name = '{0:s}/model_history.csv'.format(output_model_dir_name)
    tensorboard_dir_name = '{0:s}/tensorboard'.format(output_model_dir_name)
    metadata_file_name = '{0:s}/model_metadata.p'.format(output_model_dir_name)

    # Verify radar fields.
    myrorss_and_mrms_utils.fields_and_refl_heights_to_pairs(
        field_names=RADAR_FIELD_NAMES,
        data_source=radar_utils.MYRORSS_SOURCE_ID,
        refl_heights_m_asl=REFLECTIVITY_HEIGHTS_M_ASL)

    # Verify and convert other inputs.
    class_fraction_dict_keys = numpy.array(class_fraction_dict_keys, dtype=int)
    class_fraction_dict_values = numpy.array(
        class_fraction_dict_values, dtype=float)
    class_fraction_dict = dict(zip(
        class_fraction_dict_keys, class_fraction_dict_values))
    print 'Class fractions for sampling = {0:s}'.format(
        str(class_fraction_dict))

    first_train_time_unix_sec = time_conversion.string_to_unix_sec(
        training_start_time_string, dl_script_helper.INPUT_TIME_FORMAT)
    last_train_time_unix_sec = time_conversion.string_to_unix_sec(
        training_end_time_string, dl_script_helper.INPUT_TIME_FORMAT)

    if num_validation_batches_per_epoch <= 0:
        num_validation_batches_per_epoch = None

    if num_validation_batches_per_epoch is None:
        first_validn_time_unix_sec = None
        last_validn_time_unix_sec = None
    else:
        first_validn_time_unix_sec = time_conversion.string_to_unix_sec(
            validation_start_time_string, dl_script_helper.INPUT_TIME_FORMAT)
        last_validn_time_unix_sec = time_conversion.string_to_unix_sec(
            validation_end_time_string, dl_script_helper.INPUT_TIME_FORMAT)

    # Write metadata.
    print 'Writing metadata to: "{0:s}"...\n'.format(metadata_file_name)

    cnn.write_model_metadata(
        num_epochs=num_epochs, num_examples_per_batch=num_examples_per_batch,
        num_examples_per_file_time=2,
        num_training_batches_per_epoch=num_training_batches_per_epoch,
        first_train_time_unix_sec=first_train_time_unix_sec,
        last_train_time_unix_sec=last_train_time_unix_sec,
        num_validation_batches_per_epoch=num_validation_batches_per_epoch,
        first_validn_time_unix_sec=first_validn_time_unix_sec,
        last_validn_time_unix_sec=last_validn_time_unix_sec,
        radar_source=radar_utils.MYRORSS_SOURCE_ID,
        radar_field_names=RADAR_FIELD_NAMES, radar_heights_m_asl=None,
        reflectivity_heights_m_asl=REFLECTIVITY_HEIGHTS_M_ASL,
        target_name=target_name, normalize_by_batch=NORMALIZE_BY_BATCH,
        normalization_dict=NORMALIZATION_DICT,
        percentile_offset_for_normalization=None,
        class_fraction_dict=class_fraction_dict,
        sounding_statistic_names=None, binarize_target=binarize_target,
        use_2d3d_convolution=True, pickle_file_name=metadata_file_name)

    # Set up model architecture.
    if binarize_target:
        num_classes_to_predict = 2
    else:
        num_classes_to_predict = labels.column_name_to_num_classes(
            column_name=target_name, include_dead_storms=False)

    model_object = cnn.get_architecture_for_2d3d_myrorss(
        num_classes=num_classes_to_predict, dropout_fraction=dropout_fraction,
        first_num_reflectivity_filters=first_num_reflectivity_filters,
        num_azimuthal_shear_fields=len(AZIMUTHAL_SHEAR_FIELD_NAMES),
        l2_weight=l2_weight)

    print '\nFinding training files...'
    training_file_name_matrix, _ = trainval_io.find_2d_input_files(
        top_directory_name=input_storm_image_dir_name,
        radar_source=radar_utils.MYRORSS_SOURCE_ID,
        radar_field_names=RADAR_FIELD_NAMES,
        first_image_time_unix_sec=first_train_time_unix_sec,
        last_image_time_unix_sec=last_train_time_unix_sec,
        reflectivity_heights_m_asl=REFLECTIVITY_HEIGHTS_M_ASL,
        one_file_per_time_step=True)

    if num_validation_batches_per_epoch is None:
        validation_file_name_matrix = None
    else:
        print 'Finding validation files...'
        validation_file_name_matrix, _ = trainval_io.find_2d_input_files(
            top_directory_name=input_storm_image_dir_name,
            radar_source=radar_utils.MYRORSS_SOURCE_ID,
            radar_field_names=RADAR_FIELD_NAMES,
            first_image_time_unix_sec=first_validn_time_unix_sec,
            last_image_time_unix_sec=last_validn_time_unix_sec,
            reflectivity_heights_m_asl=REFLECTIVITY_HEIGHTS_M_ASL,
            one_file_per_time_step=True)

    print SEPARATOR_STRING

    cnn.train_2d3d_cnn_with_myrorss(
        model_object=model_object, model_file_name=model_file_name,
        history_file_name=history_file_name,
        tensorboard_dir_name=tensorboard_dir_name, num_epochs=num_epochs,
        num_training_batches_per_epoch=num_training_batches_per_epoch,
        train_image_file_name_matrix=training_file_name_matrix,
        top_target_directory_name=input_target_dir_name,
        num_examples_per_batch=num_examples_per_batch,
        num_examples_per_file_time=1, use_fast_generator=True,
        target_name=target_name, binarize_target=binarize_target,
        weight_loss_function=weight_loss_function,
        training_class_fraction_dict=class_fraction_dict,
        num_validation_batches_per_epoch=num_validation_batches_per_epoch,
        validn_image_file_name_matrix=validation_file_name_matrix,
        validation_class_fraction_dict=class_fraction_dict)


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _train_2d3d_myrorss_cnn(
        output_model_dir_name=getattr(
            INPUT_ARG_OBJECT, dl_script_helper.MODEL_DIRECTORY_ARG_NAME),
        num_epochs=getattr(
            INPUT_ARG_OBJECT, dl_script_helper.NUM_EPOCHS_ARG_NAME),
        num_training_batches_per_epoch=getattr(
            INPUT_ARG_OBJECT, dl_script_helper.NUM_TRAIN_BATCHES_ARG_NAME),
        input_storm_image_dir_name=getattr(
            INPUT_ARG_OBJECT, dl_script_helper.STORM_IMAGE_DIR_ARG_NAME),
        input_target_dir_name=getattr(
            INPUT_ARG_OBJECT, dl_script_helper.TARGET_DIR_ARG_NAME),
        num_examples_per_batch=getattr(
            INPUT_ARG_OBJECT, dl_script_helper.NUM_EXAMPLES_PER_BATCH_ARG_NAME),
        training_start_time_string=getattr(
            INPUT_ARG_OBJECT, dl_script_helper.TRAINING_START_TIME_ARG_NAME),
        training_end_time_string=getattr(
            INPUT_ARG_OBJECT, dl_script_helper.TRAINING_END_TIME_ARG_NAME),
        target_name=getattr(
            INPUT_ARG_OBJECT, dl_script_helper.TARGET_NAME_ARG_NAME),
        weight_loss_function=bool(getattr(
            INPUT_ARG_OBJECT, dl_script_helper.WEIGHT_LOSS_ARG_NAME)),
        binarize_target=bool(getattr(
            INPUT_ARG_OBJECT, dl_script_helper.BINARIZE_TARGET_ARG_NAME)),
        class_fraction_dict_keys=getattr(
            INPUT_ARG_OBJECT,
            dl_script_helper.CLASS_FRACTION_DICT_KEYS_ARG_NAME),
        class_fraction_dict_values=getattr(
            INPUT_ARG_OBJECT,
            dl_script_helper.CLASS_FRACTION_DICT_VALUES_ARG_NAME),
        num_validation_batches_per_epoch=getattr(
            INPUT_ARG_OBJECT, dl_script_helper.NUM_VALIDN_BATCHES_ARG_NAME),
        validation_start_time_string=getattr(
            INPUT_ARG_OBJECT, dl_script_helper.VALIDATION_START_TIME_ARG_NAME),
        validation_end_time_string=getattr(
            INPUT_ARG_OBJECT, dl_script_helper.VALIDATION_END_TIME_ARG_NAME),
        dropout_fraction=getattr(
            INPUT_ARG_OBJECT, dl_script_helper.DROPOUT_FRACTION_ARG_NAME),
        l2_weight=getattr(
            INPUT_ARG_OBJECT, dl_script_helper.L2_WEIGHT_ARG_NAME),
        first_num_reflectivity_filters=getattr(
            INPUT_ARG_OBJECT, FIRST_NUM_REFL_FILTERS_ARG_NAME))
