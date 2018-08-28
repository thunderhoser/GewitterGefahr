"""Handles input args for deep-learning scripts."""

from gewittergefahr.gg_utils import soundings
from gewittergefahr.deep_learning import cnn
from gewittergefahr.deep_learning import cnn_utils
from gewittergefahr.deep_learning import deep_learning_utils as dl_utils

MODEL_DIRECTORY_ARG_NAME = 'output_model_dir_name'
NUM_EPOCHS_ARG_NAME = 'num_epochs'
NUM_EXAMPLES_PER_BATCH_ARG_NAME = 'num_examples_per_batch'
NUM_EXAMPLES_PER_FILE_ARG_NAME = 'num_examples_per_file'
NUM_TRAIN_BATCHES_ARG_NAME = 'num_training_batches_per_epoch'
RADAR_DIRECTORY_ARG_NAME = 'input_storm_radar_image_dir_name'
RADAR_DIRECTORY_POS_TARGETS_ARG_NAME = 'input_radar_dir_name_pos_targets_only'
FIRST_TRAINING_DATE_ARG_NAME = 'first_train_spc_date_string'
LAST_TRAINING_DATE_ARG_NAME = 'last_train_spc_date_string'
MONITOR_STRING_ARG_NAME = 'monitor_string'
RADAR_FIELD_NAMES_ARG_NAME = 'radar_field_names'
TARGET_NAME_ARG_NAME = 'target_name'
TARGET_DIRECTORY_ARG_NAME = 'input_target_dir_name'
BINARIZE_TARGET_ARG_NAME = 'binarize_target'
NUM_ROWS_TO_KEEP_ARG_NAME = 'num_rows_to_keep'
NUM_COLUMNS_TO_KEEP_ARG_NAME = 'num_columns_to_keep'
NUM_CONV_LAYER_SETS_ARG_NAME = 'num_radar_conv_layer_sets'
NUM_CONV_LAYERS_PER_SET_ARG_NAME = 'num_conv_layers_per_set'
POOLING_TYPE_ARG_NAME = 'pooling_type_string'
ACTIVATION_FUNC_ARG_NAME = 'conv_layer_activation_func_string'
ALPHA_FOR_ELU_ARG_NAME = 'alpha_for_elu'
ALPHA_FOR_RELU_ARG_NAME = 'alpha_for_relu'
USE_BATCH_NORM_ARG_NAME = 'use_batch_normalization'
CONV_LAYER_DROPOUT_ARG_NAME = 'conv_layer_dropout_fraction'
DENSE_LAYER_DROPOUT_ARG_NAME = 'dense_layer_dropout_fraction'
NORMALIZATION_TYPE_ARG_NAME = 'normalization_type_string'
MIN_NORMALIZED_VALUE_ARG_NAME = 'min_normalized_value'
MAX_NORMALIZED_VALUE_ARG_NAME = 'max_normalized_value'
L2_WEIGHT_ARG_NAME = 'l2_weight'
SAMPLING_FRACTION_KEYS_ARG_NAME = 'sampling_fraction_keys'
SAMPLING_FRACTION_VALUES_ARG_NAME = 'sampling_fraction_values'
WEIGHT_LOSS_ARG_NAME = 'weight_loss_function'
NUM_VALIDN_BATCHES_ARG_NAME = 'num_validation_batches_per_epoch'
FIRST_VALIDATION_DATE_ARG_NAME = 'first_validn_spc_date_string'
LAST_VALIDATION_DATE_ARG_NAME = 'last_validn_spc_date_string'
SOUNDING_FIELD_NAMES_ARG_NAME = 'sounding_field_names'
SOUNDING_DIRECTORY_ARG_NAME = 'input_sounding_dir_name'
SOUNDING_LAG_TIME_ARG_NAME = (
    'sounding_lag_time_for_convective_contamination_sec')
NUM_SOUNDING_FILTERS_ARG_NAME = 'num_sounding_filters_in_first_layer'

DEFAULT_NUM_EPOCHS = 25
DEFAULT_NUM_EXAMPLES_PER_BATCH = 1024
DEFAULT_NUM_TRAIN_BATCHES_PER_EPOCH = 32
DEFAULT_MONITOR_STRING = cnn.LOSS_AS_MONITOR_STRING
DEFAULT_BINARIZE_TARGET_FLAG = 0
DEFAULT_NUM_CONV_LAYER_SETS = 3
DEFAULT_NUM_CONV_LAYERS_PER_SET = 1
DEFAULT_POOLING_TYPE_STRING = cnn_utils.MAX_POOLING_TYPE
DEFAULT_CONV_LAYER_ACTIVATION_FUNC_STRING = cnn_utils.RELU_FUNCTION_STRING
DEFAULT_ALPHA_FOR_ELU = cnn_utils.DEFAULT_ALPHA_FOR_ELU
DEFAULT_ALPHA_FOR_RELU = cnn_utils.DEFAULT_ALPHA_FOR_RELU
DEFAULT_USE_BATCH_NORM_FLAG = 0
DEFAULT_CONV_LAYER_DROPOUT_FRACTION = cnn.DEFAULT_CONV_LAYER_DROPOUT_FRACTION
DEFAULT_DENSE_LAYER_DROPOUT_FRACTION = cnn.DEFAULT_DENSE_LAYER_DROPOUT_FRACTION
DEFAULT_WEIGHT_LOSS_FLAG = 0
DEFAULT_L2_WEIGHT = 1e-3
DEFAULT_NUM_VALIDN_BATCHES_PER_EPOCH = 16
DEFAULT_SOUNDING_FIELD_NAMES = [
    soundings.RELATIVE_HUMIDITY_NAME, soundings.SPECIFIC_HUMIDITY_NAME,
    soundings.U_WIND_NAME, soundings.V_WIND_NAME,
    soundings.VIRTUAL_POTENTIAL_TEMPERATURE_NAME
]
DEFAULT_SOUNDING_LAG_TIME_SEC = 1800
DEFAULT_NUM_SOUNDING_FILTERS = 48

MODEL_DIRECTORY_HELP_STRING = (
    'Name of output directory.  The model, training history, and TensorBoard '
    'log files will be saved here after every epoch.')

NUM_EPOCHS_HELP_STRING = 'Number of epochs.'
NUM_EXAMPLES_PER_BATCH_HELP_STRING = (
    'Number of examples (storm objects) per batch.')

NUM_EXAMPLES_PER_FILE_HELP_STRING = (
    'Number of examples (storm objects) per file.  This should be significantly'
    ' lower than `{0:s}` and `{1:s}`, to ensure diversity within each batch.'
).format(NUM_TRAIN_BATCHES_ARG_NAME, NUM_VALIDN_BATCHES_ARG_NAME)

NUM_TRAIN_BATCHES_HELP_STRING = 'Number of training batches per epoch.'
RADAR_DIRECTORY_HELP_STRING = (
    'Name of top-level directory with storm-centered radar images.  Files '
    'therein will be found by `training_validation_io.find_radar_files_2d` or '
    '`training_validation_io.find_radar_files_3d`.')

RADAR_DIRECTORY_POS_TARGETS_HELP_STRING = (
    'Same as `{0:s}`, except that this directory contains files only for storm '
    'objects with positive target values.  This may be left as "None", in which'
    ' case storm-centered radar images will be read only from `{0:s}`.  For '
    'details on how to separate storm objects with positive target values, see '
    'separate_myrorss_images_with_positive_targets.py.'
).format(RADAR_DIRECTORY_ARG_NAME)

TRAINING_DATE_HELP_STRING = (
    'SPC (Storm Prediction Center) date in format "yyyymmdd".  The model will '
    'be trained with storm objects from `{0:s}`...`{1:s}`.'
).format(FIRST_TRAINING_DATE_ARG_NAME, LAST_TRAINING_DATE_ARG_NAME)

MONITOR_STRING_HELP_STRING = (
    'Evaluation function reported after each epoch.  If `{0:s}` > 0, after each'
    ' epoch `{1:s}` will be computed for validation data and the new model will'
    ' be saved only if `{1:s}` has improved.  If `{0:s}` = 0, after each epoch '
    '`{1:s}` will be computed for training data and the new model will be saved'
    ' regardless.  Valid options for `{1:s}` are listed below.\n{2:s}'
).format(NUM_VALIDN_BATCHES_ARG_NAME, MONITOR_STRING_ARG_NAME,
         str(cnn.VALID_MONITOR_STRINGS))

RADAR_FIELD_NAMES_HELP_STRING = (
    'List with names of radar fields.  Each name must be accepted by '
    '`radar_utils.check_field_name`.')

TARGET_NAME_HELP_STRING = (
    'Name of target variable (must be accepted by '
    '`labels.column_name_to_label_params`).')

TARGET_DIRECTORY_HELP_STRING = (
    'Name of top-level directory with labels (target values).  Files therein '
    'will be found by `labels.find_label_file`.')

BINARIZE_TARGET_HELP_STRING = (
    'Boolean flag.  If 1, will binarize target variable, so that the highest '
    'class becomes 1 and all other classes become 0.')

NUM_ROWS_TO_KEEP_HELP_STRING = (
    'Number of rows to keep from each storm-centered radar image.  If less than'
    ' total number of rows, images will be cropped around the center.  To use '
    'the full images, leave this alone.')

NUM_COLUMNS_TO_KEEP_HELP_STRING = (
    'Number of columns to keep from each storm-centered radar image.  If less '
    'than total number of columns, images will be cropped around the center.  '
    'To use the full images, leave this alone.')

NUM_CONV_LAYER_SETS_HELP_STRING = (
    'Number of sets of conv layers for radar data.  Each successive conv-layer '
    'set will halve the dimensions of the radar image (example -- from'
    '32 x 32 x 12 to 16 x 16 x 6, then 8 x 8 x 3, then 4 x 4 x 1).')

NUM_CONV_LAYERS_PER_SET_HELP_STRING = 'Number of conv layers in each set.'
POOLING_TYPE_HELP_STRING = (
    'Pooling type.  Must be in the following list.\n{0:s}'
).format(str(cnn_utils.VALID_POOLING_TYPES))

ACTIVATION_FUNC_HELP_STRING = (
    'Activation function for conv layers.  Must be in the following list.'
    '\n{0:s}'
).format(str(cnn_utils.VALID_CONV_LAYER_ACTIV_FUNC_STRINGS))

ALPHA_FOR_ELU_HELP_STRING = (
    'Slope for negative inputs to eLU (exponential linear unit) activation '
    'function.')

ALPHA_FOR_RELU_HELP_STRING = (
    'Slope for negative inputs to ReLU (rectified linear unit) activation '
    'function.')

USE_BATCH_NORM_HELP_STRING = (
    'Boolean flag.  If 1, a batch-normalization layer will be included after'
    ' each conv or fully connected layer.')

CONV_LAYER_DROPOUT_HELP_STRING = 'Dropout fraction for convolutional layers.'
DENSE_LAYER_DROPOUT_HELP_STRING = (
    'Dropout fraction for dense (fully connected) layer.')

NORMALIZATION_TYPE_HELP_STRING = (
    'Normalization type.  If you do not want to normalize, use the empty string'
    '.  If you want to normalize, this must be in the following list.\n{0:s}'
).format(str(dl_utils.VALID_NORMALIZATION_TYPE_STRINGS))

MIN_NORMALIZED_VALUE_HELP_STRING = (
    '[used only if {0:s} = "{1:s}"] Minimum normalized value.'
).format(NORMALIZATION_TYPE_ARG_NAME, dl_utils.MINMAX_NORMALIZATION_TYPE_STRING)

MAX_NORMALIZED_VALUE_HELP_STRING = (
    '[used only if {0:s} = "{1:s}"] Max normalized value.'
).format(NORMALIZATION_TYPE_ARG_NAME, dl_utils.MINMAX_NORMALIZATION_TYPE_STRING)

L2_WEIGHT_HELP_STRING = (
    'L2-regularization weight.  Will be applied to each convolutional layer.  '
    'Set to -1 for no L2 regularization.')

SAMPLING_FRACTION_KEYS_HELP_STRING = (
    'List of keys.  Each key is the integer representing a class (-2 for '
    '"dead storm").  Corresponding values in `{0:s}` are sampling fractions, '
    'which will be applied to each batch.  This can be used to achieve over- '
    'and undersampling of different classes.  Leave default for no special '
    'sampling.'
).format(SAMPLING_FRACTION_VALUES_ARG_NAME)

SAMPLING_FRACTION_VALUES_HELP_STRING = 'See documentation for `{0:s}`.'.format(
    SAMPLING_FRACTION_KEYS_ARG_NAME)

WEIGHT_LOSS_HELP_STRING = (
    'Boolean flag.  If 0, all classes will be weighted equally in the loss '
    'function.  If 1, each class will be weighted differently (inversely '
    'proportional with its sampling fraction in `{0:s}`).'
).format(SAMPLING_FRACTION_VALUES_ARG_NAME)

NUM_VALIDN_BATCHES_HELP_STRING = (
    'Number of validation batches per epoch.  If you make this 0, on-the-fly '
    'validation will be done with training data.')

VALIDATION_DATE_HELP_STRING = (
    'SPC (Storm Prediction Center) date in format "yyyymmdd".  The model will '
    'be validated on the fly (monitored) with storm objects from `{0:s}`...'
    '`{1:s}`.'
).format(FIRST_VALIDATION_DATE_ARG_NAME, LAST_VALIDATION_DATE_ARG_NAME)

SOUNDING_FIELD_NAMES_HELP_STRING = (
    'List with names of sounding fields.  Each name must be accepted by '
    '`soundings.check_field_name`.  To train without soundings, make this a'
    '1-item list with the string "None".')

SOUNDING_DIRECTORY_HELP_STRING = (
    'Name of top-level directory with storm-centered soundings.  Files therein '
    'will be found by `training_validation_io.find_sounding_files`.')

SOUNDING_LAG_TIME_HELP_STRING = (
    'Lag time for soundings (see `soundings.interp_soundings_to_storm_objects` '
    'for details).  This will be used to locate sounding files.')

NUM_SOUNDING_FILTERS_HELP_STRING = (
    'Number of sounding filters in first convolutional layer.  Number of '
    'filters will double for each successive layer convolving over soundings.')


def add_input_arguments(argument_parser_object):
    """Adds deep-learning input args to ArgumentParser object.

    :param argument_parser_object: `argparse.ArgumentParser` object, which may
        or may not already contain input args.
    :return: argument_parser_object: Same as input object, but with new input
        args added.
    """

    argument_parser_object.add_argument(
        '--' + MODEL_DIRECTORY_ARG_NAME, type=str, required=True,
        help=MODEL_DIRECTORY_HELP_STRING)

    argument_parser_object.add_argument(
        '--' + NUM_EPOCHS_ARG_NAME, type=int, required=False,
        default=DEFAULT_NUM_EPOCHS, help=NUM_EPOCHS_HELP_STRING)

    argument_parser_object.add_argument(
        '--' + NUM_EXAMPLES_PER_BATCH_ARG_NAME, type=int, required=False,
        default=DEFAULT_NUM_EXAMPLES_PER_BATCH,
        help=NUM_EXAMPLES_PER_BATCH_HELP_STRING)

    argument_parser_object.add_argument(
        '--' + NUM_EXAMPLES_PER_FILE_ARG_NAME, type=int, required=True,
        help=NUM_EXAMPLES_PER_FILE_HELP_STRING)

    argument_parser_object.add_argument(
        '--' + NUM_TRAIN_BATCHES_ARG_NAME, type=int, required=False,
        default=DEFAULT_NUM_TRAIN_BATCHES_PER_EPOCH,
        help=NUM_TRAIN_BATCHES_HELP_STRING)

    argument_parser_object.add_argument(
        '--' + RADAR_DIRECTORY_ARG_NAME, type=str, required=True,
        help=RADAR_DIRECTORY_HELP_STRING)

    argument_parser_object.add_argument(
        '--' + RADAR_DIRECTORY_POS_TARGETS_ARG_NAME, type=str, required=False,
        default='None', help=RADAR_DIRECTORY_POS_TARGETS_HELP_STRING)

    argument_parser_object.add_argument(
        '--' + FIRST_TRAINING_DATE_ARG_NAME, type=str, required=True,
        help=TRAINING_DATE_HELP_STRING)

    argument_parser_object.add_argument(
        '--' + LAST_TRAINING_DATE_ARG_NAME, type=str, required=True,
        help=TRAINING_DATE_HELP_STRING)

    argument_parser_object.add_argument(
        '--' + MONITOR_STRING_ARG_NAME, type=str, required=False,
        default=DEFAULT_MONITOR_STRING, help=MONITOR_STRING_HELP_STRING)

    argument_parser_object.add_argument(
        '--' + RADAR_FIELD_NAMES_ARG_NAME, type=str, nargs='+', required=False,
        default=[''], help=RADAR_FIELD_NAMES_HELP_STRING)

    argument_parser_object.add_argument(
        '--' + TARGET_NAME_ARG_NAME, type=str, required=True,
        help=TARGET_NAME_HELP_STRING)

    argument_parser_object.add_argument(
        '--' + TARGET_DIRECTORY_ARG_NAME, type=str, required=True,
        help=TARGET_DIRECTORY_HELP_STRING)

    argument_parser_object.add_argument(
        '--' + BINARIZE_TARGET_ARG_NAME, type=int, required=False,
        default=DEFAULT_BINARIZE_TARGET_FLAG, help=BINARIZE_TARGET_HELP_STRING)

    argument_parser_object.add_argument(
        '--' + NUM_ROWS_TO_KEEP_ARG_NAME, type=int, required=False,
        default=-1, help=NUM_ROWS_TO_KEEP_HELP_STRING)

    argument_parser_object.add_argument(
        '--' + NUM_COLUMNS_TO_KEEP_ARG_NAME, type=int, required=False,
        default=-1, help=NUM_COLUMNS_TO_KEEP_HELP_STRING)

    argument_parser_object.add_argument(
        '--' + NUM_CONV_LAYER_SETS_ARG_NAME, type=int, required=False,
        default=DEFAULT_NUM_CONV_LAYER_SETS,
        help=NUM_CONV_LAYER_SETS_HELP_STRING)

    argument_parser_object.add_argument(
        '--' + NUM_CONV_LAYERS_PER_SET_ARG_NAME, type=int, required=False,
        default=DEFAULT_NUM_CONV_LAYERS_PER_SET,
        help=NUM_CONV_LAYERS_PER_SET_HELP_STRING)

    argument_parser_object.add_argument(
        '--' + POOLING_TYPE_ARG_NAME, type=str, required=False,
        default=DEFAULT_POOLING_TYPE_STRING, help=POOLING_TYPE_HELP_STRING)

    argument_parser_object.add_argument(
        '--' + ACTIVATION_FUNC_ARG_NAME, type=str, required=False,
        default=DEFAULT_CONV_LAYER_ACTIVATION_FUNC_STRING,
        help=ACTIVATION_FUNC_HELP_STRING)

    argument_parser_object.add_argument(
        '--' + ALPHA_FOR_ELU_ARG_NAME, type=float, required=False,
        default=DEFAULT_ALPHA_FOR_ELU, help=ALPHA_FOR_ELU_HELP_STRING)

    argument_parser_object.add_argument(
        '--' + ALPHA_FOR_RELU_ARG_NAME, type=float, required=False,
        default=DEFAULT_ALPHA_FOR_RELU, help=ALPHA_FOR_RELU_HELP_STRING)

    argument_parser_object.add_argument(
        '--' + USE_BATCH_NORM_ARG_NAME, type=int, required=False,
        default=DEFAULT_USE_BATCH_NORM_FLAG, help=USE_BATCH_NORM_HELP_STRING)

    argument_parser_object.add_argument(
        '--' + CONV_LAYER_DROPOUT_ARG_NAME, type=float, required=False,
        default=DEFAULT_CONV_LAYER_DROPOUT_FRACTION,
        help=CONV_LAYER_DROPOUT_HELP_STRING)

    argument_parser_object.add_argument(
        '--' + DENSE_LAYER_DROPOUT_ARG_NAME, type=float, required=False,
        default=DEFAULT_DENSE_LAYER_DROPOUT_FRACTION,
        help=DENSE_LAYER_DROPOUT_HELP_STRING)

    argument_parser_object.add_argument(
        '--' + NORMALIZATION_TYPE_ARG_NAME, type=str, required=False,
        default=dl_utils.MINMAX_NORMALIZATION_TYPE_STRING,
        help=NORMALIZATION_TYPE_HELP_STRING)

    argument_parser_object.add_argument(
        '--' + MIN_NORMALIZED_VALUE_ARG_NAME, type=float, required=False,
        default=dl_utils.DEFAULT_MIN_NORMALIZED_VALUE,
        help=MIN_NORMALIZED_VALUE_HELP_STRING)

    argument_parser_object.add_argument(
        '--' + MAX_NORMALIZED_VALUE_ARG_NAME, type=float, required=False,
        default=dl_utils.DEFAULT_MAX_NORMALIZED_VALUE,
        help=MAX_NORMALIZED_VALUE_HELP_STRING)

    argument_parser_object.add_argument(
        '--' + L2_WEIGHT_ARG_NAME, type=float, required=False,
        default=DEFAULT_L2_WEIGHT, help=L2_WEIGHT_HELP_STRING)

    argument_parser_object.add_argument(
        '--' + SAMPLING_FRACTION_KEYS_ARG_NAME, type=int, nargs='+',
        required=False, default=[0], help=SAMPLING_FRACTION_KEYS_HELP_STRING)

    argument_parser_object.add_argument(
        '--' + SAMPLING_FRACTION_VALUES_ARG_NAME, type=float, nargs='+',
        required=False, default=[0.], help=SAMPLING_FRACTION_VALUES_HELP_STRING)

    argument_parser_object.add_argument(
        '--' + WEIGHT_LOSS_ARG_NAME, type=int, required=False,
        default=DEFAULT_WEIGHT_LOSS_FLAG, help=WEIGHT_LOSS_HELP_STRING)

    argument_parser_object.add_argument(
        '--' + NUM_VALIDN_BATCHES_ARG_NAME, type=int, required=False,
        default=DEFAULT_NUM_VALIDN_BATCHES_PER_EPOCH,
        help=NUM_VALIDN_BATCHES_HELP_STRING)

    argument_parser_object.add_argument(
        '--' + FIRST_VALIDATION_DATE_ARG_NAME, type=str, required=False,
        default='', help=VALIDATION_DATE_HELP_STRING)

    argument_parser_object.add_argument(
        '--' + LAST_VALIDATION_DATE_ARG_NAME, type=str, required=False,
        default='', help=VALIDATION_DATE_HELP_STRING)

    argument_parser_object.add_argument(
        '--' + SOUNDING_FIELD_NAMES_ARG_NAME, type=str, nargs='+',
        required=False, default=DEFAULT_SOUNDING_FIELD_NAMES,
        help=SOUNDING_FIELD_NAMES_HELP_STRING)

    argument_parser_object.add_argument(
        '--' + SOUNDING_DIRECTORY_ARG_NAME, type=str, required=False,
        default='', help=SOUNDING_DIRECTORY_HELP_STRING)

    argument_parser_object.add_argument(
        '--' + SOUNDING_LAG_TIME_ARG_NAME, type=int, required=False,
        default=DEFAULT_SOUNDING_LAG_TIME_SEC,
        help=SOUNDING_LAG_TIME_HELP_STRING)

    argument_parser_object.add_argument(
        '--' + NUM_SOUNDING_FILTERS_ARG_NAME, type=int, required=False,
        default=DEFAULT_NUM_SOUNDING_FILTERS,
        help=NUM_SOUNDING_FILTERS_HELP_STRING)

    return argument_parser_object
