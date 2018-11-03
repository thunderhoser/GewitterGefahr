"""Handles input args for deep-learning scripts."""

from gewittergefahr.gg_utils import soundings
from gewittergefahr.deep_learning import cnn
from gewittergefahr.deep_learning import architecture_utils
from gewittergefahr.deep_learning import deep_learning_utils as dl_utils
from gewittergefahr.deep_learning import training_validation_io as trainval_io

# TODO(thunderhoser): Allow architecture to be read from a file.  Actually,
# maybe this should be the only way to specify architecture.

INPUT_TIME_FORMAT = '%Y-%m-%d-%H%M%S'

MODEL_DIRECTORY_ARG_NAME = 'output_model_dir_name'
NUM_EPOCHS_ARG_NAME = 'num_epochs'
NUM_TRAINING_BATCHES_ARG_NAME = 'num_training_batches_per_epoch'
TRAINING_DIR_ARG_NAME = 'input_training_dir_name'
FIRST_TRAINING_TIME_ARG_NAME = 'first_training_time_string'
LAST_TRAINING_TIME_ARG_NAME = 'last_training_time_string'
NUM_VALIDN_BATCHES_ARG_NAME = 'num_validation_batches_per_epoch'
VALIDATION_DIR_ARG_NAME = 'input_validation_dir_name'
FIRST_VALIDN_TIME_ARG_NAME = 'first_validation_time_string'
LAST_VALIDN_TIME_ARG_NAME = 'last_validation_time_string'
MONITOR_STRING_ARG_NAME = 'monitor_string'
WEIGHT_LOSS_ARG_NAME = 'weight_loss_function'

NUM_EXAMPLES_PER_BATCH_ARG_NAME = 'num_examples_per_batch'
RADAR_FIELDS_ARG_NAME = 'radar_field_names'
SOUNDING_FIELDS_ARG_NAME = 'sounding_field_names'
NUM_ROWS_ARG_NAME = 'num_grid_rows'
NUM_COLUMNS_ARG_NAME = 'num_grid_columns'
NORMALIZATION_TYPE_ARG_NAME = 'normalization_type_string'
MIN_NORM_VALUE_ARG_NAME = 'min_normalized_value'
MAX_NORM_VALUE_ARG_NAME = 'max_normalized_value'
BINARIZE_TARGET_ARG_NAME = 'binarize_target'
SAMPLING_FRACTION_KEYS_ARG_NAME = 'sampling_fraction_keys'
SAMPLING_FRACTION_VALUES_ARG_NAME = 'sampling_fraction_values'
NUM_TRANSLATIONS_ARG_NAME = 'num_translations'
MAX_TRANSLATION_ARG_NAME = 'max_translation_pixels'
NUM_ROTATIONS_ARG_NAME = 'num_rotations'
MAX_ROTATION_ARG_NAME = 'max_absolute_rotation_angle_deg'
NUM_NOISINGS_ARG_NAME = 'num_noisings'
MAX_NOISE_ARG_NAME = 'max_noise_standard_deviation'

NUM_CONV_LAYER_SETS_ARG_NAME = 'num_radar_conv_layer_sets'
NUM_CONV_LAYERS_PER_SET_ARG_NAME = 'num_conv_layers_per_set'
POOLING_TYPE_ARG_NAME = 'pooling_type_string'
ACTIVATION_FUNCTION_ARG_NAME = 'activation_function_string'
ALPHA_FOR_ELU_ARG_NAME = 'alpha_for_elu'
ALPHA_FOR_RELU_ARG_NAME = 'alpha_for_relu'
USE_BATCH_NORM_ARG_NAME = 'use_batch_normalization'
CONV_LAYER_DROPOUT_ARG_NAME = 'conv_layer_dropout_fraction'
DENSE_LAYER_DROPOUT_ARG_NAME = 'dense_layer_dropout_fraction'
L2_WEIGHT_ARG_NAME = 'l2_weight'
NUM_SOUNDING_FILTERS_ARG_NAME = 'first_num_sounding_filters'

DEFAULT_NUM_EPOCHS = 100
DEFAULT_NUM_TRAIN_BATCHES_PER_EPOCH = 32
DEFAULT_NUM_VALIDN_BATCHES_PER_EPOCH = 16
DEFAULT_MONITOR_STRING = cnn.LOSS_FUNCTION_STRING
DEFAULT_WEIGHT_LOSS_FLAG = 0

DEFAULT_NUM_EXAMPLES_PER_BATCH = 1024
DEFAULT_SOUNDING_FIELD_NAMES = [
    soundings.RELATIVE_HUMIDITY_NAME, soundings.SPECIFIC_HUMIDITY_NAME,
    soundings.U_WIND_NAME, soundings.V_WIND_NAME,
    soundings.VIRTUAL_POTENTIAL_TEMPERATURE_NAME
]
DEFAULT_NORMALIZATION_TYPE_STRING = dl_utils.Z_NORMALIZATION_TYPE_STRING
DEFAULT_MIN_NORM_VALUE = -1.
DEFAULT_MAX_NORM_VALUE = 1.
DEFAULT_BINARIZE_TARGET_FLAG = 0

DEFAULT_NUM_TRANSLATIONS = trainval_io.DEFAULT_AUGMENTATION_OPTION_DICT[
    trainval_io.NUM_TRANSLATIONS_KEY]
DEFAULT_MAX_TRANSLATION_PIXELS = trainval_io.DEFAULT_AUGMENTATION_OPTION_DICT[
    trainval_io.MAX_TRANSLATION_KEY]
DEFAULT_NUM_ROTATIONS = trainval_io.DEFAULT_AUGMENTATION_OPTION_DICT[
    trainval_io.NUM_ROTATIONS_KEY]
DEFAULT_MAX_ABS_ROTATION_ANGLE_DEG = (
    trainval_io.DEFAULT_AUGMENTATION_OPTION_DICT[trainval_io.MAX_ROTATION_KEY])
DEFAULT_NUM_NOISINGS = trainval_io.DEFAULT_AUGMENTATION_OPTION_DICT[
    trainval_io.NUM_NOISINGS_KEY]
DEFAULT_MAX_NOISE_STDEV = trainval_io.DEFAULT_AUGMENTATION_OPTION_DICT[
    trainval_io.MAX_NOISE_KEY]

DEFAULT_NUM_CONV_LAYER_SETS = 2
DEFAULT_NUM_CONV_LAYERS_PER_SET = 1
DEFAULT_POOLING_TYPE_STRING = architecture_utils.MAX_POOLING_TYPE
DEFAULT_ACTIVATION_FUNCTION_STRING = architecture_utils.RELU_FUNCTION_STRING
DEFAULT_ALPHA_FOR_ELU = architecture_utils.DEFAULT_ALPHA_FOR_ELU
DEFAULT_ALPHA_FOR_RELU = architecture_utils.DEFAULT_ALPHA_FOR_RELU
DEFAULT_USE_BATCH_NORM_FLAG = 0
DEFAULT_CONV_LAYER_DROPOUT_FRACTION = -1.
DEFAULT_DENSE_LAYER_DROPOUT_FRACTION = 0.25

DEFAULT_L2_WEIGHT = 0.001
DEFAULT_NUM_SOUNDING_FILTERS = 16

MODEL_DIRECTORY_HELP_STRING = (
    'Name of output directory.  The model, training history, and TensorBoard '
    'files will be saved here after each epoch.')

NUM_EPOCHS_HELP_STRING = 'Number of training epochs.'

NUM_TRAINING_BATCHES_HELP_STRING = 'Number of training batches per epoch.'

TRAINING_DIR_HELP_STRING = (
    'Name of top-level directory with training examples.  Shuffled files '
    'therein will be found by `input_examples.find_example_file` and read by '
    '`input_examples.read_example_file`.')

TRAINING_TIME_HELP_STRING = (
    'First training time (format "yyyy-mm-dd-HHMMSS").  Only examples in the '
    'period `{0:s}`...`{1:s}` will be used for training.'
).format(FIRST_TRAINING_TIME_ARG_NAME, LAST_TRAINING_TIME_ARG_NAME)

NUM_VALIDN_BATCHES_HELP_STRING = 'Number of validation batches per epoch.'

VALIDATION_DIR_HELP_STRING = (
    'Name of top-level directory with validation examples.  Shuffled files '
    'therein will be found by `input_examples.find_example_file` and read by '
    '`input_examples.read_example_file`.')

VALIDATION_TIME_HELP_STRING = (
    'First validation time (format "yyyy-mm-dd-HHMMSS").  Only examples in the '
    'period `{0:s}`...`{1:s}` will be used for validation.'
).format(FIRST_TRAINING_TIME_ARG_NAME, LAST_TRAINING_TIME_ARG_NAME)

MONITOR_STRING_HELP_STRING = (
    'Performance metric used to monitor performance (must be accepted by '
    '`cnn._get_checkpoint_object`).  If `{0:s}` > 0, this performance metric '
    'will be computed on validation data after each epoch, and the new model '
    'will be saved only if validation performance is better than all previous '
    'epochs.  If `{0:s}` = 0, `{1:s}` will be reported after each epoch but the'
    ' new model will be saved regardless.'
).format(NUM_VALIDN_BATCHES_ARG_NAME, MONITOR_STRING_ARG_NAME)

WEIGHT_LOSS_HELP_STRING = (
    'Boolean flag.  If 0, all classes will be weighted equally in the loss '
    'function.  If 1, each class will be weighted differently (by the inverse '
    'of its sampling fraction, specified by `{0:s}` and `{1:s}`).'
).format(SAMPLING_FRACTION_KEYS_ARG_NAME, SAMPLING_FRACTION_VALUES_ARG_NAME)

NUM_EXAMPLES_PER_BATCH_HELP_STRING = 'Number of examples per batch.'

RADAR_FIELDS_HELP_STRING = (
    'List of radar fields to use for training.  Each field must be accepted by '
    '`radar_utils.check_field_name`.')

SOUNDING_FIELDS_HELP_STRING = (
    'List of sounding fields to use for training.  Each field must be accepted '
    'by `soundings.check_field_name`.  If you do not want to use soundings, '
    'make this a list with only one item ("None").')

NUM_ROWS_HELP_STRING = (
    'Number of rows in each storm-centered radar image.  This will be doubled '
    'for MYRORSS azimuthal shear.')

NUM_COLUMNS_HELP_STRING = 'Same as `{0:s}` but for columns.'.format(
    NUM_ROWS_ARG_NAME)

NORMALIZATION_TYPE_HELP_STRING = (
    'Normalization type (used for each radar and sounding variable).  Must be '
    'accepted by `deep_learning_utils._check_normalization_type`.  For no '
    'normalization, make this the empty string ("").')

MIN_NORM_VALUE_HELP_STRING = (
    '[used only if `{0:s}` = "{1:s}"] Minimum value after normalization.'
).format(NORMALIZATION_TYPE_ARG_NAME, dl_utils.MINMAX_NORMALIZATION_TYPE_STRING)

MAX_NORM_VALUE_HELP_STRING = (
    '[used only if `{0:s}` = "{1:s}"] Max value after normalization.'
).format(NORMALIZATION_TYPE_ARG_NAME, dl_utils.MINMAX_NORMALIZATION_TYPE_STRING)

BINARIZE_TARGET_HELP_STRING = (
    'Boolean flag.  If 1, target variable will be binarized, so that highest '
    'class is positive and all others are negative.')

SAMPLING_FRACTION_KEYS_HELP_STRING = (
    'List of keys.  Each key is the integer encoding a class (-2 for "dead '
    'storm").  Corresponding values in `{0:s}` are sampling fractions and will '
    'be used for downsampling.  If you want no downsampling, leave this arg '
    'alone.'
).format(SAMPLING_FRACTION_VALUES_ARG_NAME)

SAMPLING_FRACTION_VALUES_HELP_STRING = 'See doc for `{0:s}`.'.format(
    SAMPLING_FRACTION_KEYS_ARG_NAME)

NUM_TRANSLATIONS_HELP_STRING = (
    'Number of translations for each storm-centered radar image.  See '
    '`data_augmentation.get_translations` for more details.')

MAX_TRANSLATION_HELP_STRING = (
    'Max translation for each storm-centered radar image.  See '
    '`data_augmentation.get_translations` for more details.')

NUM_ROTATIONS_HELP_STRING = (
    'Number of rotations for each storm-centered radar image.  See '
    '`data_augmentation.get_rotations` for more details.')

MAX_ROTATION_HELP_STRING = (
    'Max rotation for each storm-centered radar image.  See '
    '`data_augmentation.get_rotations` for more details.')

NUM_NOISINGS_HELP_STRING = (
    'Number of noisings for each storm-centered radar image.  See '
    '`data_augmentation.get_noisings` for more details.')

MAX_NOISE_HELP_STRING = (
    'Max noising for each storm-centered radar image.  See '
    '`data_augmentation.get_noisings` for more details.')

NUM_CONV_LAYER_SETS_HELP_STRING = (
    'Number of convolution-layer sets for radar data.  Each set consists of '
    '`{0:s}` layers, uninterrupted by pooling.'
).format(NUM_CONV_LAYERS_PER_SET_ARG_NAME)

NUM_CONV_LAYERS_PER_SET_HELP_STRING = (
    'Number of convolution layers in each set.')

POOLING_TYPE_HELP_STRING = (
    'Pooling type (must be accepted by '
    '`architecture_utils._check_input_args_for_pooling_layer`).')

ACTIVATION_FUNCTION_HELP_STRING = (
    'Activation function for all layers other than the output (must be accepted'
    ' by `architecture_utils.check_activation_function`).')

ALPHA_FOR_ELU_HELP_STRING = (
    'Slope (alpha parameter) for activation function "{0:s}".'
).format(architecture_utils.ELU_FUNCTION_STRING)

ALPHA_FOR_RELU_HELP_STRING = (
    'Slope (alpha parameter) for activation function "{0:s}".'
).format(architecture_utils.RELU_FUNCTION_STRING)

USE_BATCH_NORM_HELP_STRING = (
    'Boolean flag.  If 1, batch normalization will be used after each '
    'convolution and dense layer.')

CONV_LAYER_DROPOUT_HELP_STRING = (
    'Dropout fraction for convolution layers.  Make this <= 0 for no dropout.')

DENSE_LAYER_DROPOUT_HELP_STRING = (
    'Dropout fraction for dense layers.  Make this <= 0 for no dropout.')

L2_WEIGHT_HELP_STRING = (
    'L2 regularization weight for all convolution layers.  Make this <= 0 for '
    'no regularization.')

NUM_SOUNDING_FILTERS_HELP_STRING = (
    'Number of filters in first convolution layer for soundings.')


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
        '--' + NUM_TRAINING_BATCHES_ARG_NAME, type=int, required=False,
        default=DEFAULT_NUM_TRAIN_BATCHES_PER_EPOCH,
        help=NUM_TRAINING_BATCHES_HELP_STRING)

    argument_parser_object.add_argument(
        '--' + TRAINING_DIR_ARG_NAME, type=str, required=True,
        help=TRAINING_DIR_HELP_STRING)

    argument_parser_object.add_argument(
        '--' + FIRST_TRAINING_TIME_ARG_NAME, type=str, required=True,
        help=TRAINING_TIME_HELP_STRING)

    argument_parser_object.add_argument(
        '--' + LAST_TRAINING_TIME_ARG_NAME, type=str, required=True,
        help=TRAINING_TIME_HELP_STRING)

    argument_parser_object.add_argument(
        '--' + NUM_VALIDN_BATCHES_ARG_NAME, type=int, required=False,
        default=DEFAULT_NUM_VALIDN_BATCHES_PER_EPOCH,
        help=NUM_VALIDN_BATCHES_HELP_STRING)

    argument_parser_object.add_argument(
        '--' + VALIDATION_DIR_ARG_NAME, type=str, required=True,
        help=VALIDATION_DIR_HELP_STRING)

    argument_parser_object.add_argument(
        '--' + FIRST_VALIDN_TIME_ARG_NAME, type=str, required=True,
        help=VALIDATION_TIME_HELP_STRING)

    argument_parser_object.add_argument(
        '--' + LAST_VALIDN_TIME_ARG_NAME, type=str, required=True,
        help=VALIDATION_TIME_HELP_STRING)

    argument_parser_object.add_argument(
        '--' + MONITOR_STRING_ARG_NAME, type=str, required=False,
        default=DEFAULT_MONITOR_STRING, help=MONITOR_STRING_HELP_STRING)

    argument_parser_object.add_argument(
        '--' + WEIGHT_LOSS_ARG_NAME, type=int, required=False,
        default=DEFAULT_WEIGHT_LOSS_FLAG, help=WEIGHT_LOSS_HELP_STRING)

    argument_parser_object.add_argument(
        '--' + NUM_EXAMPLES_PER_BATCH_ARG_NAME, type=int, required=False,
        default=DEFAULT_NUM_EXAMPLES_PER_BATCH,
        help=NUM_EXAMPLES_PER_BATCH_HELP_STRING)

    argument_parser_object.add_argument(
        '--' + RADAR_FIELDS_ARG_NAME, type=str, nargs='+', required=False,
        default=['None'], help=RADAR_FIELDS_HELP_STRING)

    argument_parser_object.add_argument(
        '--' + SOUNDING_FIELDS_ARG_NAME, type=str, nargs='+', required=False,
        default=DEFAULT_SOUNDING_FIELD_NAMES, help=SOUNDING_FIELDS_HELP_STRING)

    argument_parser_object.add_argument(
        '--' + NUM_ROWS_ARG_NAME, type=int, required=True,
        help=NUM_ROWS_HELP_STRING)

    argument_parser_object.add_argument(
        '--' + NUM_COLUMNS_ARG_NAME, type=int, required=True,
        help=NUM_COLUMNS_HELP_STRING)

    argument_parser_object.add_argument(
        '--' + NORMALIZATION_TYPE_ARG_NAME, type=str, required=False,
        default=DEFAULT_NORMALIZATION_TYPE_STRING,
        help=NORMALIZATION_TYPE_HELP_STRING)

    argument_parser_object.add_argument(
        '--' + MIN_NORM_VALUE_ARG_NAME, type=int, required=False,
        default=DEFAULT_MIN_NORM_VALUE, help=MIN_NORM_VALUE_HELP_STRING)

    argument_parser_object.add_argument(
        '--' + MAX_NORM_VALUE_ARG_NAME, type=int, required=False,
        default=DEFAULT_MAX_NORM_VALUE, help=MAX_NORM_VALUE_HELP_STRING)

    argument_parser_object.add_argument(
        '--' + BINARIZE_TARGET_ARG_NAME, type=int, required=False,
        default=DEFAULT_BINARIZE_TARGET_FLAG, help=BINARIZE_TARGET_HELP_STRING)

    argument_parser_object.add_argument(
        '--' + SAMPLING_FRACTION_KEYS_ARG_NAME, type=int, nargs='+',
        required=False, default=[0], help=SAMPLING_FRACTION_KEYS_HELP_STRING)

    argument_parser_object.add_argument(
        '--' + SAMPLING_FRACTION_VALUES_ARG_NAME, type=float, nargs='+',
        required=False, default=[0.], help=SAMPLING_FRACTION_VALUES_HELP_STRING)

    argument_parser_object.add_argument(
        '--' + NUM_TRANSLATIONS_ARG_NAME, type=int, required=False,
        default=DEFAULT_NUM_TRANSLATIONS, help=NUM_TRANSLATIONS_HELP_STRING)

    argument_parser_object.add_argument(
        '--' + MAX_TRANSLATION_ARG_NAME, type=int, required=False,
        default=DEFAULT_MAX_TRANSLATION_PIXELS,
        help=MAX_TRANSLATION_HELP_STRING)

    argument_parser_object.add_argument(
        '--' + NUM_ROTATIONS_ARG_NAME, type=int, required=False,
        default=DEFAULT_NUM_ROTATIONS, help=NUM_ROTATIONS_HELP_STRING)

    argument_parser_object.add_argument(
        '--' + MAX_ROTATION_ARG_NAME, type=float, required=False,
        default=DEFAULT_MAX_ABS_ROTATION_ANGLE_DEG,
        help=MAX_ROTATION_HELP_STRING)

    argument_parser_object.add_argument(
        '--' + NUM_NOISINGS_ARG_NAME, type=int, required=False,
        default=DEFAULT_NUM_NOISINGS, help=NUM_NOISINGS_HELP_STRING)

    argument_parser_object.add_argument(
        '--' + MAX_NOISE_ARG_NAME, type=float, required=False,
        default=DEFAULT_MAX_NOISE_STDEV, help=MAX_NOISE_HELP_STRING)

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
        '--' + ACTIVATION_FUNCTION_ARG_NAME, type=str, required=False,
        default=DEFAULT_ACTIVATION_FUNCTION_STRING,
        help=ACTIVATION_FUNCTION_HELP_STRING)

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
        '--' + L2_WEIGHT_ARG_NAME, type=float, required=False,
        default=DEFAULT_L2_WEIGHT, help=L2_WEIGHT_HELP_STRING)

    argument_parser_object.add_argument(
        '--' + NUM_SOUNDING_FILTERS_ARG_NAME, type=int, required=False,
        default=DEFAULT_NUM_SOUNDING_FILTERS,
        help=NUM_SOUNDING_FILTERS_HELP_STRING)

    return argument_parser_object
