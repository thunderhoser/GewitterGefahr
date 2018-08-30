"""Trains CNN with 2-D and 3-D images from MYRORSS.

CNN = convolutional neural net
MYRORSS = Multi-year Reanalysis of Remotely Sensed Storms
Field in 2-D images = azimuthal shear
Field in 3-D images = reflectivity
"""

import argparse
import numpy
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.gg_utils import labels
from gewittergefahr.gg_utils import radar_utils
from gewittergefahr.deep_learning import cnn
from gewittergefahr.deep_learning import training_validation_io as trainval_io
from gewittergefahr.scripts import deep_learning_helper as dl_helper

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

NUM_REFLECTIVITY_ROWS = 32
NUM_REFLECTIVITY_COLUMNS = 32
NUM_SOUNDING_HEIGHTS = 49
REFLECTIVITY_HEIGHTS_M_AGL = numpy.linspace(1000, 12000, num=12, dtype=int)

AZIMUTHAL_SHEAR_FIELD_NAMES = radar_utils.SHEAR_NAMES
RADAR_FIELD_NAMES = [radar_utils.REFL_NAME] + AZIMUTHAL_SHEAR_FIELD_NAMES

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER = dl_helper.add_input_arguments(
    argument_parser_object=INPUT_ARG_PARSER)

NUM_REFL_FILTERS_ARG_NAME = 'num_refl_filters_in_first_layer'
NUM_SHEAR_FILTERS_ARG_NAME = 'num_shear_filters_in_first_layer'
NORMALIZATION_FILE_ARG_NAME = 'normalization_param_file_name'

NUM_REFL_FILTERS_HELP_STRING = (
    'Number of reflectivity filters in first convolutional layer.  Number of '
    'filters will double for each successive layer convolving over reflectivity'
    ' images.')
NUM_SHEAR_FILTERS_HELP_STRING = (
    'Number of azimuthal-shear filters in first convolutional layer.  Number of'
    ' filters will double for each successive layer convolving over '
    'azimuthal-shear images.')
NORMALIZATION_FILE_HELP_STRING = (
    '[used only if {0:s} is not empty] Path to file with normalization params.'
    '  Will be read by `deep_learning_utils.read_normalization_params_from_file'
    '`.'
).format(dl_helper.NORMALIZATION_TYPE_ARG_NAME)

DEFAULT_NORMALIZATION_FILE_NAME = ''

INPUT_ARG_PARSER.add_argument(
    '--' + NUM_REFL_FILTERS_ARG_NAME, type=int, required=False,
    default=8, help=NUM_REFL_FILTERS_HELP_STRING)
INPUT_ARG_PARSER.add_argument(
    '--' + NUM_SHEAR_FILTERS_ARG_NAME, type=int, required=False,
    default=16, help=NUM_SHEAR_FILTERS_HELP_STRING)
INPUT_ARG_PARSER.add_argument(
    '--' + NORMALIZATION_FILE_ARG_NAME, type=str, required=False,
    default=DEFAULT_NORMALIZATION_FILE_NAME,
    help=NORMALIZATION_FILE_HELP_STRING)


def _train_cnn(
        output_model_dir_name, num_epochs, num_examples_per_batch,
        num_examples_per_file, num_training_batches_per_epoch,
        top_storm_radar_image_dir_name,
        top_storm_radar_image_dir_name_pos_targets_only,
        first_train_spc_date_string, last_train_spc_date_string, monitor_string,
        target_name, top_target_dir_name, binarize_target, num_rows_to_keep,
        num_columns_to_keep, num_radar_conv_layer_sets, num_conv_layers_per_set,
        pooling_type_string, conv_layer_activation_func_string, alpha_for_elu,
        alpha_for_relu, use_batch_normalization, conv_layer_dropout_fraction,
        dense_layer_dropout_fraction, num_refl_filters_in_first_layer,
        num_shear_filters_in_first_layer, normalization_type_string,
        normalization_param_file_name, min_normalized_value,
        max_normalized_value, l2_weight, sampling_fraction_dict_keys,
        sampling_fraction_dict_values, weight_loss_function,
        num_validation_batches_per_epoch, first_validn_spc_date_string,
        last_validn_spc_date_string, sounding_field_names,
        top_sounding_dir_name,
        sounding_lag_time_for_convective_contamination_sec,
        num_sounding_filters_in_first_layer):
    """Trains CNN with 2-D and 3-D images from MYRORSS.

    :param output_model_dir_name: See documentation at the top of
        deep_learning_helper.py.
    :param num_epochs: Same.
    :param num_examples_per_batch: Same.
    :param num_examples_per_file: Same.
    :param num_training_batches_per_epoch: Same.
    :param top_storm_radar_image_dir_name: Same.
    :param top_storm_radar_image_dir_name_pos_targets_only: Same.
    :param first_train_spc_date_string: Same.
    :param last_train_spc_date_string: Same.
    :param monitor_string: Same.
    :param target_name: Same.
    :param top_target_dir_name: Same.
    :param binarize_target: Same.
    :param num_rows_to_keep: Same.
    :param num_columns_to_keep: Same.
    :param num_radar_conv_layer_sets: Same.
    :param num_conv_layers_per_set: Same.
    :param pooling_type_string: Same.
    :param conv_layer_activation_func_string: Same.
    :param alpha_for_elu: Same.
    :param alpha_for_relu: Same.
    :param use_batch_normalization: Same.
    :param conv_layer_dropout_fraction: Same.
    :param dense_layer_dropout_fraction: Same.
    :param num_refl_filters_in_first_layer: See documentation at the top of this
        file.
    :param num_shear_filters_in_first_layer: Same.
    :param normalization_type_string: See documentation at the top of
        deep_learning_helper.py.
    :param normalization_param_file_name: See documentation at the top of this
        file.
    :param min_normalized_value: See documentation at the top of
        deep_learning_helper.py.
    :param max_normalized_value: Same.
    :param l2_weight: Same.
    :param sampling_fraction_dict_keys: Same.
    :param sampling_fraction_dict_values: Same.
    :param weight_loss_function: Same.
    :param num_validation_batches_per_epoch: Same.
    :param first_validn_spc_date_string: Same.
    :param last_validn_spc_date_string: Same.
    :param sounding_field_names: Same.
    :param top_sounding_dir_name: Same.
    :param sounding_lag_time_for_convective_contamination_sec: Same.
    :param num_sounding_filters_in_first_layer: Same.
    """

    # Verify and convert input args.
    if conv_layer_dropout_fraction <= 0:
        conv_layer_dropout_fraction = None
    if dense_layer_dropout_fraction <= 0:
        dense_layer_dropout_fraction = None
    if l2_weight <= 0:
        l2_weight = None
    if num_rows_to_keep <= 0:
        num_rows_to_keep = None
    if num_columns_to_keep <= 0:
        num_columns_to_keep = None

    sampling_fraction_dict_keys = numpy.array(
        sampling_fraction_dict_keys, dtype=int)
    sampling_fraction_dict_values = numpy.array(
        sampling_fraction_dict_values, dtype=float)

    if len(sampling_fraction_dict_keys) > 1:
        sampling_fraction_by_class_dict = dict(zip(
            sampling_fraction_dict_keys, sampling_fraction_dict_values))
    else:
        sampling_fraction_by_class_dict = None

    first_train_time_unix_sec = time_conversion.spc_date_string_to_unix_sec(
        first_train_spc_date_string)
    last_train_time_unix_sec = time_conversion.spc_date_string_to_unix_sec(
        last_train_spc_date_string)

    if num_validation_batches_per_epoch <= 0:
        num_validation_batches_per_epoch = None
        first_validn_time_unix_sec = None
        last_validn_time_unix_sec = None
    else:
        first_validn_time_unix_sec = (
            time_conversion.spc_date_string_to_unix_sec(
                first_validn_spc_date_string)
        )
        last_validn_time_unix_sec = (
            time_conversion.spc_date_string_to_unix_sec(
                last_validn_spc_date_string)
        )

    if sounding_field_names[0] == 'None':
        sounding_field_names = None
        num_sounding_fields = None
    else:
        num_sounding_fields = len(sounding_field_names)

    # Set locations of output files.
    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=output_model_dir_name)
    model_file_name = '{0:s}/model.h5'.format(output_model_dir_name)
    history_file_name = '{0:s}/model_history.csv'.format(output_model_dir_name)
    tensorboard_dir_name = '{0:s}/tensorboard'.format(output_model_dir_name)
    metadata_file_name = '{0:s}/model_metadata.p'.format(output_model_dir_name)

    # Find input files for training.
    if top_storm_radar_image_dir_name_pos_targets_only == 'None':
        top_storm_radar_image_dir_name_pos_targets_only = None

    (radar_fn_matrix_training,
     radar_fn_matrix_training_pos_targets_only
    ) = trainval_io.find_radar_files_2d(
        top_directory_name=top_storm_radar_image_dir_name,
        radar_source=radar_utils.MYRORSS_SOURCE_ID,
        radar_field_names=RADAR_FIELD_NAMES,
        first_file_time_unix_sec=first_train_time_unix_sec,
        last_file_time_unix_sec=last_train_time_unix_sec,
        one_file_per_time_step=False,
        top_directory_name_pos_targets_only=
        top_storm_radar_image_dir_name_pos_targets_only,
        reflectivity_heights_m_agl=REFLECTIVITY_HEIGHTS_M_AGL)
    print SEPARATOR_STRING

    if num_validation_batches_per_epoch is None:
        radar_fn_matrix_validation = None
        radar_fn_matrix_validation_pos_targets_only = None
    else:
        (radar_fn_matrix_validation,
         radar_fn_matrix_validation_pos_targets_only
        ) = trainval_io.find_radar_files_2d(
            top_directory_name=top_storm_radar_image_dir_name,
            radar_source=radar_utils.MYRORSS_SOURCE_ID,
            radar_field_names=RADAR_FIELD_NAMES,
            first_file_time_unix_sec=first_validn_time_unix_sec,
            last_file_time_unix_sec=last_validn_time_unix_sec,
            one_file_per_time_step=False,
            top_directory_name_pos_targets_only=
            top_storm_radar_image_dir_name_pos_targets_only,
            reflectivity_heights_m_agl=REFLECTIVITY_HEIGHTS_M_AGL)
        print SEPARATOR_STRING

    print 'Writing metadata to: "{0:s}"...\n'.format(metadata_file_name)
    cnn.write_model_metadata(
        pickle_file_name=metadata_file_name, num_epochs=num_epochs,
        num_examples_per_batch=num_examples_per_batch,
        num_examples_per_file=num_examples_per_file,
        num_training_batches_per_epoch=num_training_batches_per_epoch,
        radar_fn_matrix_training=radar_fn_matrix_training,
        weight_loss_function=weight_loss_function,
        monitor_string=monitor_string, target_name=target_name,
        binarize_target=binarize_target, num_rows_to_keep=num_rows_to_keep,
        num_columns_to_keep=num_columns_to_keep,
        normalization_type_string=normalization_type_string,
        use_2d3d_convolution=True, radar_source=radar_utils.MYRORSS_SOURCE_ID,
        radar_field_names=RADAR_FIELD_NAMES,
        reflectivity_heights_m_agl=REFLECTIVITY_HEIGHTS_M_AGL,
        min_normalized_value=min_normalized_value,
        max_normalized_value=max_normalized_value,
        normalization_param_file_name=normalization_param_file_name,
        radar_fn_matrix_training_pos_targets_only=
        radar_fn_matrix_training_pos_targets_only,
        training_fraction_by_class_dict=sampling_fraction_by_class_dict,
        validation_fraction_by_class_dict=sampling_fraction_by_class_dict,
        num_validation_batches_per_epoch=num_validation_batches_per_epoch,
        radar_fn_matrix_validation=radar_fn_matrix_validation,
        radar_fn_matrix_validation_pos_targets_only=
        radar_fn_matrix_validation_pos_targets_only,
        sounding_field_names=sounding_field_names,
        sounding_lag_time_for_convective_contamination_sec=
        sounding_lag_time_for_convective_contamination_sec)

    if binarize_target:
        num_classes_to_predict = 2
    else:
        num_classes_to_predict = labels.column_name_to_num_classes(
            column_name=target_name, include_dead_storms=False)

    model_object = cnn.get_2d3d_swirlnet_architecture(
        num_reflectivity_rows=NUM_REFLECTIVITY_ROWS,
        num_reflectivity_columns=NUM_REFLECTIVITY_COLUMNS,
        num_reflectivity_heights=len(REFLECTIVITY_HEIGHTS_M_AGL),
        num_azimuthal_shear_fields=len(AZIMUTHAL_SHEAR_FIELD_NAMES),
        num_radar_conv_layer_sets=num_radar_conv_layer_sets,
        num_conv_layers_per_set=num_conv_layers_per_set,
        pooling_type_string=pooling_type_string,
        num_classes=num_classes_to_predict,
        conv_layer_activation_func_string=conv_layer_activation_func_string,
        alpha_for_elu=alpha_for_elu, alpha_for_relu=alpha_for_relu,
        use_batch_normalization=use_batch_normalization,
        num_refl_filters_in_first_layer=num_refl_filters_in_first_layer,
        num_shear_filters_in_first_layer=num_shear_filters_in_first_layer,
        conv_layer_dropout_fraction=conv_layer_dropout_fraction,
        dense_layer_dropout_fraction=dense_layer_dropout_fraction,
        l2_weight=l2_weight, num_sounding_heights=NUM_SOUNDING_HEIGHTS,
        num_sounding_fields=num_sounding_fields,
        num_sounding_filters_in_first_layer=num_sounding_filters_in_first_layer)
    print SEPARATOR_STRING

    cnn.train_2d3d_cnn(
        model_object=model_object, model_file_name=model_file_name,
        history_file_name=history_file_name,
        tensorboard_dir_name=tensorboard_dir_name,
        num_epochs=num_epochs, num_examples_per_batch=num_examples_per_batch,
        num_examples_per_file=num_examples_per_file,
        num_training_batches_per_epoch=num_training_batches_per_epoch,
        radar_fn_matrix_training=radar_fn_matrix_training,
        target_name=target_name, top_target_directory_name=top_target_dir_name,
        num_rows_to_keep=num_rows_to_keep,
        num_columns_to_keep=num_columns_to_keep,
        normalization_type_string=normalization_type_string,
        normalization_param_file_name=normalization_param_file_name,
        min_normalized_value=min_normalized_value,
        max_normalized_value=max_normalized_value,
        radar_fn_matrix_training_pos_targets_only=
        radar_fn_matrix_training_pos_targets_only,
        monitor_string=monitor_string, binarize_target=binarize_target,
        weight_loss_function=weight_loss_function,
        training_fraction_by_class_dict=sampling_fraction_by_class_dict,
        num_validation_batches_per_epoch=num_validation_batches_per_epoch,
        validation_fraction_by_class_dict=sampling_fraction_by_class_dict,
        radar_fn_matrix_validation=radar_fn_matrix_validation,
        radar_fn_matrix_validation_pos_targets_only=
        radar_fn_matrix_validation_pos_targets_only,
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
        num_examples_per_file=getattr(
            INPUT_ARG_OBJECT, dl_helper.NUM_EXAMPLES_PER_FILE_ARG_NAME),
        num_training_batches_per_epoch=getattr(
            INPUT_ARG_OBJECT, dl_helper.NUM_TRAIN_BATCHES_ARG_NAME),
        top_storm_radar_image_dir_name=getattr(
            INPUT_ARG_OBJECT, dl_helper.RADAR_DIRECTORY_ARG_NAME),
        top_storm_radar_image_dir_name_pos_targets_only=getattr(
            INPUT_ARG_OBJECT, dl_helper.RADAR_DIRECTORY_POS_TARGETS_ARG_NAME),
        first_train_spc_date_string=getattr(
            INPUT_ARG_OBJECT, dl_helper.FIRST_TRAINING_DATE_ARG_NAME),
        last_train_spc_date_string=getattr(
            INPUT_ARG_OBJECT, dl_helper.LAST_TRAINING_DATE_ARG_NAME),
        monitor_string=getattr(
            INPUT_ARG_OBJECT, dl_helper.MONITOR_STRING_ARG_NAME),
        target_name=getattr(
            INPUT_ARG_OBJECT, dl_helper.TARGET_NAME_ARG_NAME),
        top_target_dir_name=getattr(
            INPUT_ARG_OBJECT, dl_helper.TARGET_DIRECTORY_ARG_NAME),
        binarize_target=bool(getattr(
            INPUT_ARG_OBJECT, dl_helper.BINARIZE_TARGET_ARG_NAME)),
        num_rows_to_keep=getattr(
            INPUT_ARG_OBJECT, dl_helper.NUM_ROWS_TO_KEEP_ARG_NAME),
        num_columns_to_keep=getattr(
            INPUT_ARG_OBJECT, dl_helper.NUM_COLUMNS_TO_KEEP_ARG_NAME),
        num_radar_conv_layer_sets=getattr(
            INPUT_ARG_OBJECT, dl_helper.NUM_CONV_LAYER_SETS_ARG_NAME),
        num_conv_layers_per_set=getattr(
            INPUT_ARG_OBJECT, dl_helper.NUM_CONV_LAYERS_PER_SET_ARG_NAME),
        pooling_type_string=getattr(
            INPUT_ARG_OBJECT, dl_helper.POOLING_TYPE_ARG_NAME),
        conv_layer_activation_func_string=getattr(
            INPUT_ARG_OBJECT, dl_helper.ACTIVATION_FUNC_ARG_NAME),
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
        num_refl_filters_in_first_layer=getattr(
            INPUT_ARG_OBJECT, NUM_REFL_FILTERS_ARG_NAME),
        num_shear_filters_in_first_layer=getattr(
            INPUT_ARG_OBJECT, NUM_SHEAR_FILTERS_ARG_NAME),
        normalization_type_string=getattr(
            INPUT_ARG_OBJECT, dl_helper.NORMALIZATION_TYPE_ARG_NAME),
        normalization_param_file_name=getattr(
            INPUT_ARG_OBJECT, NORMALIZATION_FILE_ARG_NAME),
        min_normalized_value=getattr(
            INPUT_ARG_OBJECT, dl_helper.MIN_NORMALIZED_VALUE_ARG_NAME),
        max_normalized_value=getattr(
            INPUT_ARG_OBJECT, dl_helper.MAX_NORMALIZED_VALUE_ARG_NAME),
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
        first_validn_spc_date_string=getattr(
            INPUT_ARG_OBJECT, dl_helper.FIRST_VALIDATION_DATE_ARG_NAME),
        last_validn_spc_date_string=getattr(
            INPUT_ARG_OBJECT, dl_helper.LAST_VALIDATION_DATE_ARG_NAME),
        sounding_field_names=getattr(
            INPUT_ARG_OBJECT, dl_helper.SOUNDING_FIELD_NAMES_ARG_NAME),
        top_sounding_dir_name=getattr(
            INPUT_ARG_OBJECT, dl_helper.SOUNDING_DIRECTORY_ARG_NAME),
        sounding_lag_time_for_convective_contamination_sec=getattr(
            INPUT_ARG_OBJECT, dl_helper.SOUNDING_LAG_TIME_ARG_NAME),
        num_sounding_filters_in_first_layer=getattr(
            INPUT_ARG_OBJECT, dl_helper.NUM_SOUNDING_FILTERS_ARG_NAME))
