"""Trains 3-D CNN (one that performs 3-D convolution) with GridRad data."""

import copy
import argparse
import numpy
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.gg_utils import labels
from gewittergefahr.gg_utils import radar_utils
from gewittergefahr.gg_utils import gridrad_utils
from gewittergefahr.deep_learning import cnn
from gewittergefahr.deep_learning import training_validation_io
from gewittergefahr.deep_learning import deep_learning_utils as dl_utils
from gewittergefahr.scripts import deep_learning as dl_script_helper

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

RADAR_HEIGHTS_M_ASL = numpy.linspace(1000, 12000, num=12, dtype=int)
NORMALIZE_BY_BATCH = False
NORMALIZATION_DICT = copy.deepcopy(dl_utils.DEFAULT_NORMALIZATION_DICT)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER = dl_script_helper.add_input_arguments(
    argument_parser_object=INPUT_ARG_PARSER)


def _train_3d_gridrad_cnn(
        output_model_dir_name, num_epochs, num_training_batches_per_epoch,
        input_storm_image_dir_name, radar_field_names, num_examples_per_batch,
        num_examples_per_time, training_start_time_string,
        training_end_time_string, target_name, weight_loss_function,
        class_fractions_to_sample, num_validation_batches_per_epoch,
        validation_start_time_string, validation_end_time_string):
    """Trains 3-D CNN (one that performs 3-D convolution) with GridRad data.

    :param output_model_dir_name: See documentation at the top of
        `scripts/deep_learning.py`.
    :param num_epochs: Same.
    :param num_training_batches_per_epoch: Same.
    :param input_storm_image_dir_name: Same.
    :param radar_field_names: Same.
    :param num_examples_per_batch: Same.
    :param num_examples_per_time: Same.
    :param training_start_time_string: Same.
    :param training_end_time_string: Same.
    :param target_name: Same.
    :param weight_loss_function: Same.
    :param class_fractions_to_sample: Same.
    :param num_validation_batches_per_epoch: Same.
    :param validation_start_time_string: Same.
    :param validation_end_time_string: Same.
    """

    # Make output directory; determine names of output files.
    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=output_model_dir_name)
    model_file_name = '{0:s}/model.h5'.format(output_model_dir_name)
    history_file_name = '{0:s}/model_history.csv'.format(output_model_dir_name)
    tensorboard_dir_name = '{0:s}/tensorboard'.format(output_model_dir_name)
    metadata_file_name = '{0:s}/model_metadata.p'.format(output_model_dir_name)

    # Verify radar fields.
    gridrad_utils.fields_and_refl_heights_to_pairs(
        field_names=radar_field_names, heights_m_asl=RADAR_HEIGHTS_M_ASL)

    # Verify and convert other inputs.
    class_fractions_to_sample = numpy.array(class_fractions_to_sample)
    if len(class_fractions_to_sample) == 1:
        class_fractions_to_sample = None

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
        num_examples_per_time=num_examples_per_time,
        num_training_batches_per_epoch=num_training_batches_per_epoch,
        first_train_time_unix_sec=first_train_time_unix_sec,
        last_train_time_unix_sec=last_train_time_unix_sec,
        num_validation_batches_per_epoch=num_validation_batches_per_epoch,
        first_validn_time_unix_sec=first_validn_time_unix_sec,
        last_validn_time_unix_sec=last_validn_time_unix_sec,
        radar_source=radar_utils.GRIDRAD_SOURCE_ID,
        radar_field_names=radar_field_names,
        radar_heights_m_asl=RADAR_HEIGHTS_M_ASL,
        reflectivity_heights_m_asl=None, target_name=target_name,
        normalize_by_batch=NORMALIZE_BY_BATCH,
        normalization_dict=NORMALIZATION_DICT,
        percentile_offset_for_normalization=None,
        class_fractions=class_fractions_to_sample,
        pickle_file_name=metadata_file_name)

    # Determine number of classes.
    target_param_dict = labels.column_name_to_label_params(target_name)
    wind_speed_class_cutoffs_kt = target_param_dict[
        labels.WIND_SPEED_CLASS_CUTOFFS_KEY]
    if wind_speed_class_cutoffs_kt is None:
        num_classes = 2
    else:
        num_classes = len(wind_speed_class_cutoffs_kt) + 1

    # Create model architecture.
    model_object = cnn.get_3d_swirlnet_architecture(
        num_classes=num_classes, num_input_channels=len(radar_field_names))

    print '\nFinding training files...'
    training_file_name_matrix, _ = training_validation_io.find_3d_input_files(
        top_directory_name=input_storm_image_dir_name,
        radar_source=radar_utils.GRIDRAD_SOURCE_ID,
        radar_field_names=radar_field_names,
        radar_heights_m_asl=RADAR_HEIGHTS_M_ASL,
        first_image_time_unix_sec=first_train_time_unix_sec,
        last_image_time_unix_sec=last_train_time_unix_sec)

    if num_validation_batches_per_epoch is None:
        validation_file_name_matrix = None
    else:
        print 'Finding validation files...'
        validation_file_name_matrix, _ = (
            training_validation_io.find_3d_input_files(
                top_directory_name=input_storm_image_dir_name,
                radar_source=radar_utils.GRIDRAD_SOURCE_ID,
                radar_field_names=radar_field_names,
                radar_heights_m_asl=RADAR_HEIGHTS_M_ASL,
                first_image_time_unix_sec=first_validn_time_unix_sec,
                last_image_time_unix_sec=last_validn_time_unix_sec))

    print SEPARATOR_STRING

    # Train model.
    cnn.train_3d_cnn(
        model_object=model_object, model_file_name=model_file_name,
        history_file_name=history_file_name,
        tensorboard_dir_name=tensorboard_dir_name, num_epochs=num_epochs,
        num_training_batches_per_epoch=num_training_batches_per_epoch,
        training_file_name_matrix=training_file_name_matrix,
        num_examples_per_batch=num_examples_per_batch,
        num_examples_per_time=num_examples_per_time, target_name=target_name,
        normalize_by_batch=NORMALIZE_BY_BATCH,
        normalization_dict=NORMALIZATION_DICT,
        weight_loss_function=weight_loss_function,
        class_fractions_to_sample=class_fractions_to_sample,
        num_validation_batches_per_epoch=num_validation_batches_per_epoch,
        validation_file_name_matrix=validation_file_name_matrix)


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _train_3d_gridrad_cnn(
        output_model_dir_name=getattr(
            INPUT_ARG_OBJECT, dl_script_helper.MODEL_DIRECTORY_ARG_NAME),
        num_epochs=getattr(
            INPUT_ARG_OBJECT, dl_script_helper.NUM_EPOCHS_ARG_NAME),
        num_training_batches_per_epoch=getattr(
            INPUT_ARG_OBJECT, dl_script_helper.NUM_TRAIN_BATCHES_ARG_NAME),
        input_storm_image_dir_name=getattr(
            INPUT_ARG_OBJECT, dl_script_helper.STORM_IMAGE_DIR_ARG_NAME),
        radar_field_names=getattr(
            INPUT_ARG_OBJECT, dl_script_helper.RADAR_FIELD_NAMES_ARG_NAME),
        num_examples_per_batch=getattr(
            INPUT_ARG_OBJECT, dl_script_helper.NUM_EXAMPLES_PER_BATCH_ARG_NAME),
        num_examples_per_time=getattr(
            INPUT_ARG_OBJECT, dl_script_helper.NUM_EXAMPLES_PER_TIME_ARG_NAME),
        training_start_time_string=getattr(
            INPUT_ARG_OBJECT, dl_script_helper.TRAINING_START_TIME_ARG_NAME),
        training_end_time_string=getattr(
            INPUT_ARG_OBJECT, dl_script_helper.TRAINING_END_TIME_ARG_NAME),
        target_name=getattr(
            INPUT_ARG_OBJECT, dl_script_helper.TARGET_NAME_ARG_NAME),
        weight_loss_function=bool(getattr(
            INPUT_ARG_OBJECT, dl_script_helper.WEIGHT_LOSS_ARG_NAME)),
        class_fractions_to_sample=getattr(
            INPUT_ARG_OBJECT, dl_script_helper.CLASS_FRACTIONS_ARG_NAME),
        num_validation_batches_per_epoch=getattr(
            INPUT_ARG_OBJECT, dl_script_helper.NUM_VALIDN_BATCHES_ARG_NAME),
        validation_start_time_string=getattr(
            INPUT_ARG_OBJECT, dl_script_helper.VALIDATION_START_TIME_ARG_NAME),
        validation_end_time_string=getattr(
            INPUT_ARG_OBJECT, dl_script_helper.VALIDATION_END_TIME_ARG_NAME))
