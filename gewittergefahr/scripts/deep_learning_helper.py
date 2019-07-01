"""Handles input args for training deep-learning models."""

import numpy
from gewittergefahr.gg_utils import soundings
from gewittergefahr.deep_learning import cnn
from gewittergefahr.deep_learning import deep_learning_utils as dl_utils

TIME_FORMAT = '%Y-%m-%d-%H%M%S'
SOUNDING_HEIGHTS_M_AGL = soundings.DEFAULT_HEIGHT_LEVELS_M_AGL + 0

INPUT_MODEL_FILE_ARG_NAME = 'input_model_file_name'
SOUNDING_FIELDS_ARG_NAME = 'sounding_field_names'

NORMALIZATION_TYPE_ARG_NAME = 'normalization_type_string'
NORMALIZATION_FILE_ARG_NAME = 'normalization_param_file_name'
MIN_NORM_VALUE_ARG_NAME = 'min_normalized_value'
MAX_NORM_VALUE_ARG_NAME = 'max_normalized_value'

TARGET_NAME_ARG_NAME = 'target_name'
DOWNSAMPLING_CLASSES_ARG_NAME = 'downsampling_classes'
DOWNSAMPLING_FRACTIONS_ARG_NAME = 'downsampling_fractions'
MONITOR_ARG_NAME = 'monitor_string'
WEIGHT_LOSS_ARG_NAME = 'weight_loss_function'

X_TRANSLATIONS_ARG_NAME = 'x_translations_px'
Y_TRANSLATIONS_ARG_NAME = 'y_translations_px'
ROTATION_ANGLES_ARG_NAME = 'ccw_rotation_angles_deg'
NOISE_STDEV_ARG_NAME = 'noise_standard_deviation'
NUM_NOISINGS_ARG_NAME = 'num_noisings'
FLIP_X_ARG_NAME = 'flip_in_x'
FLIP_Y_ARG_NAME = 'flip_in_y'

TRAINING_DIR_ARG_NAME = 'input_training_dir_name'
FIRST_TRAINING_TIME_ARG_NAME = 'first_training_time_string'
LAST_TRAINING_TIME_ARG_NAME = 'last_training_time_string'
VALIDATION_DIR_ARG_NAME = 'input_validation_dir_name'
FIRST_VALIDATION_TIME_ARG_NAME = 'first_validation_time_string'
LAST_VALIDATION_TIME_ARG_NAME = 'last_validation_time_string'
NUM_EX_PER_BATCH_ARG_NAME = 'num_examples_per_batch'
NUM_EPOCHS_ARG_NAME = 'num_epochs'
NUM_TRAINING_BATCHES_ARG_NAME = 'num_training_batches_per_epoch'
NUM_VALIDATION_BATCHES_ARG_NAME = 'num_validation_batches_per_epoch'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

INPUT_MODEL_FILE_HELP_STRING = (
    'Path to input file (containing either trained or untrained CNN).  Will be '
    'read by `cnn.read_model`.  The architecture of this CNN will be copied.')

SOUNDING_FIELDS_HELP_STRING = (
    'List of sounding fields.  Each must be accepted by '
    '`soundings.check_field_name`.  Input will contain each sounding field at '
    'each of the following heights (metres AGL).  If you do not want to train '
    'with soundings, make this a list with one empty string ("").\n{0:s}'
).format(str(SOUNDING_HEIGHTS_M_AGL))

NORMALIZATION_TYPE_HELP_STRING = (
    'Normalization type (used for both radar images and soundings).  See doc '
    'for `deep_learning_utils.normalize_radar_images` or '
    '`deep_learning_utils.normalize_soundings`.')

NORMALIZATION_FILE_HELP_STRING = (
    'Path to file with normalization params (used for both radar images and '
    'soundings).  See doc for `deep_learning_utils.normalize_radar_images` or '
    '`deep_learning_utils.normalize_soundings`.')

MIN_NORM_VALUE_HELP_STRING = (
    'Minimum value for min-max normalization (used for both radar images and '
    'soundings).  See doc for `deep_learning_utils.normalize_radar_images` or '
    '`deep_learning_utils.normalize_soundings`.')

MAX_NORM_VALUE_HELP_STRING = (
    'Max value for min-max normalization (used for both radar images and '
    'soundings).  See doc for `deep_learning_utils.normalize_radar_images` or '
    '`deep_learning_utils.normalize_soundings`.')

TARGET_NAME_HELP_STRING = 'Name of target variable.'

DOWNSAMPLING_CLASSES_HELP_STRING = (
    'List of classes (integer labels) for downsampling.  If you do not want '
    'downsampling, leave this alone.')

DOWNSAMPLING_FRACTIONS_HELP_STRING = (
    'List of downsampling fractions.  The [k]th downsampling fraction goes with'
    ' the [k]th class in `{0:s}`, and the sum of all downsampling fractions '
    'must be 1.0.  If you do not want downsampling, leave this alone.'
).format(DOWNSAMPLING_CLASSES_ARG_NAME)

MONITOR_HELP_STRING = (
    'Function used to monitor validation performance (and implement early '
    'stopping).  Must be in the following list.\n{0:s}'
).format(str(cnn.VALID_MONITOR_STRINGS))

WEIGHT_LOSS_HELP_STRING = (
    'Boolean flag.  If 1, each class in the loss function will be weighted by '
    'the inverse of its frequency in training data.  If 0, no such weighting '
    'will be done.')

X_TRANSLATIONS_HELP_STRING = (
    'x-translations for data augmentation (pixel units).  See doc for '
    '`data_augmentation.shift_radar_images`.  If you do not want translation '
    'augmentation, leave this alone.')

Y_TRANSLATIONS_HELP_STRING = (
    'y-translations for data augmentation (pixel units).  See doc for '
    '`data_augmentation.shift_radar_images`.  If you do not want translation '
    'augmentation, leave this alone.')

ROTATION_ANGLES_HELP_STRING = (
    'Counterclockwise rotation angles for data augmentation.  See doc for '
    '`data_augmentation.rotate_radar_images`.  If you do not want rotation '
    'augmentation, leave this alone.')

NOISE_STDEV_HELP_STRING = (
    'Standard deviation for Gaussian noise.  See doc for '
    '`data_augmentation.noise_radar_images`.  If you do not want noising '
    'augmentation, leave this alone.')

NUM_NOISINGS_HELP_STRING = (
    'Number of times to replicate each example with noise.  See doc for '
    '`data_augmentation.noise_radar_images`.  If you do not want noising '
    'augmentation, leave this alone.')

FLIP_X_HELP_STRING = (
    'Boolean flag.  If 1, will flip each radar image in the x-direction.')

FLIP_Y_HELP_STRING = (
    'Boolean flag.  If 1, will flip each radar image in the y-direction.')

TRAINING_DIR_HELP_STRING = (
    'Name of directory with training data.  Files therein will be found by '
    '`input_examples.find_many_example_files` (with shuffled = True) and read '
    'by `input_examples.read_example_file`.')

TRAINING_TIME_HELP_STRING = (
    'Time (format "yyyy-mm-dd-HHMMSS").  Only examples from the period '
    '`{0:s}`...`{1:s}` will be used for training.'
).format(FIRST_TRAINING_TIME_ARG_NAME, LAST_TRAINING_TIME_ARG_NAME)

VALIDATION_DIR_HELP_STRING = (
    'Same as `{0:s}` but for on-the-fly validation.  If you do not want '
    'validation, leave this alone.'
).format(TRAINING_DIR_ARG_NAME)

VALIDATION_TIME_HELP_STRING = (
    'Time (format "yyyy-mm-dd-HHMMSS").  Only examples from the period '
    '`{0:s}`...`{1:s}` will be used for validation.  If you do not want '
    'validation, leave this alone.'
).format(FIRST_VALIDATION_TIME_ARG_NAME, LAST_VALIDATION_TIME_ARG_NAME)

NUM_EX_PER_BATCH_HELP_STRING = (
    'Number of examples in each training or validation batch.')

NUM_EPOCHS_HELP_STRING = 'Number of training epochs.'

NUM_TRAINING_BATCHES_HELP_STRING = 'Number of training batches in each epoch.'

NUM_VALIDATION_BATCHES_HELP_STRING = (
    'Number of validation batches in each epoch.')

OUTPUT_DIR_HELP_STRING = (
    'Path to output directory.  The newly trained CNN and metafiles will be '
    'saved here.')

DEFAULT_SOUNDING_FIELD_NAMES = [
    soundings.RELATIVE_HUMIDITY_NAME, soundings.SPECIFIC_HUMIDITY_NAME,
    soundings.VIRTUAL_POTENTIAL_TEMPERATURE_NAME,
    soundings.U_WIND_NAME, soundings.V_WIND_NAME
]

DEFAULT_NORM_TYPE_STRING = dl_utils.Z_NORMALIZATION_TYPE_STRING + ''
DEFAULT_MIN_NORM_VALUE = -1.
DEFAULT_MAX_NORM_VALUE = 1.

DEFAULT_DOWNSAMPLING_CLASSES = numpy.array([0, 1], dtype=int)
DEFAULT_DOWNSAMPLING_FRACTIONS = numpy.array([0.5, 0.5])
DEFAULT_MONITOR_STRING = cnn.LOSS_FUNCTION_STRING + ''
DEFAULT_WEIGHT_LOSS_FLAG = 0

DEFAULT_X_TRANSLATIONS_PX = numpy.array([0], dtype=int)
DEFAULT_Y_TRANSLATIONS_PX = numpy.array([0], dtype=int)
DEFAULT_CCW_ROTATION_ANGLES_DEG = numpy.array([0], dtype=float)
DEFAULT_NOISE_STDEV = 0.05
DEFAULT_NUM_NOISINGS = 0
DEFAULT_FLIP_X_FLAG = 0
DEFAULT_FLIP_Y_FLAG = 0

DEFAULT_NUM_EXAMPLES_PER_BATCH = 512
DEFAULT_NUM_EPOCHS = 100
DEFAULT_NUM_TRAINING_BATCHES_PER_EPOCH = 32
DEFAULT_NUM_VALIDATION_BATCHES_PER_EPOCH = 16


def add_input_args(argument_parser):
    """Adds input args to ArgumentParser object.

    :param argument_parser: Instance of `argparse.ArgumentParser` (may already
        contain some input args).
    :return: argument_parser: Same as input but with new args added.
    """

    argument_parser.add_argument(
        '--' + INPUT_MODEL_FILE_ARG_NAME, type=str, required=True,
        help=INPUT_MODEL_FILE_HELP_STRING)

    argument_parser.add_argument(
        '--' + SOUNDING_FIELDS_ARG_NAME, type=str, nargs='+', required=False,
        default=DEFAULT_SOUNDING_FIELD_NAMES, help=SOUNDING_FIELDS_HELP_STRING)

    argument_parser.add_argument(
        '--' + NORMALIZATION_TYPE_ARG_NAME, type=str, required=False,
        default=DEFAULT_NORM_TYPE_STRING, help=NORMALIZATION_TYPE_HELP_STRING)

    argument_parser.add_argument(
        '--' + NORMALIZATION_FILE_ARG_NAME, type=str, required=True,
        help=NORMALIZATION_FILE_HELP_STRING)

    argument_parser.add_argument(
        '--' + MIN_NORM_VALUE_ARG_NAME, type=float, required=False,
        default=DEFAULT_MIN_NORM_VALUE, help=MIN_NORM_VALUE_HELP_STRING)

    argument_parser.add_argument(
        '--' + MAX_NORM_VALUE_ARG_NAME, type=float, required=False,
        default=DEFAULT_MAX_NORM_VALUE, help=MAX_NORM_VALUE_HELP_STRING)

    argument_parser.add_argument(
        '--' + TARGET_NAME_ARG_NAME, type=str, required=True,
        help=TARGET_NAME_HELP_STRING)

    argument_parser.add_argument(
        '--' + DOWNSAMPLING_CLASSES_ARG_NAME, type=int, nargs='+',
        required=False, default=DEFAULT_DOWNSAMPLING_CLASSES,
        help=DOWNSAMPLING_CLASSES_HELP_STRING)

    argument_parser.add_argument(
        '--' + DOWNSAMPLING_FRACTIONS_ARG_NAME, type=float, nargs='+',
        required=False, default=DEFAULT_DOWNSAMPLING_FRACTIONS,
        help=DOWNSAMPLING_FRACTIONS_HELP_STRING)

    argument_parser.add_argument(
        '--' + MONITOR_ARG_NAME, type=str, required=False,
        default=DEFAULT_MONITOR_STRING, help=MONITOR_HELP_STRING)

    argument_parser.add_argument(
        '--' + WEIGHT_LOSS_ARG_NAME, type=int, required=False,
        default=DEFAULT_WEIGHT_LOSS_FLAG, help=WEIGHT_LOSS_HELP_STRING)

    argument_parser.add_argument(
        '--' + X_TRANSLATIONS_ARG_NAME, type=int, nargs='+', required=False,
        default=DEFAULT_X_TRANSLATIONS_PX, help=X_TRANSLATIONS_HELP_STRING)

    argument_parser.add_argument(
        '--' + Y_TRANSLATIONS_ARG_NAME, type=int, nargs='+', required=False,
        default=DEFAULT_Y_TRANSLATIONS_PX, help=Y_TRANSLATIONS_HELP_STRING)

    argument_parser.add_argument(
        '--' + ROTATION_ANGLES_ARG_NAME, type=float, nargs='+', required=False,
        default=DEFAULT_CCW_ROTATION_ANGLES_DEG,
        help=ROTATION_ANGLES_HELP_STRING)

    argument_parser.add_argument(
        '--' + NOISE_STDEV_ARG_NAME, type=float, required=False,
        default=DEFAULT_NOISE_STDEV, help=NOISE_STDEV_HELP_STRING)

    argument_parser.add_argument(
        '--' + NUM_NOISINGS_ARG_NAME, type=int, required=False,
        default=DEFAULT_NUM_NOISINGS, help=NUM_NOISINGS_HELP_STRING)

    argument_parser.add_argument(
        '--' + FLIP_X_ARG_NAME, type=int, required=False,
        default=DEFAULT_FLIP_X_FLAG, help=FLIP_X_HELP_STRING)

    argument_parser.add_argument(
        '--' + FLIP_Y_ARG_NAME, type=int, required=False,
        default=DEFAULT_FLIP_Y_FLAG, help=FLIP_Y_HELP_STRING)

    argument_parser.add_argument(
        '--' + TRAINING_DIR_ARG_NAME, type=str, required=True,
        help=TRAINING_DIR_HELP_STRING)

    argument_parser.add_argument(
        '--' + FIRST_TRAINING_TIME_ARG_NAME, type=str, required=True,
        help=TRAINING_TIME_HELP_STRING)

    argument_parser.add_argument(
        '--' + LAST_TRAINING_TIME_ARG_NAME, type=str, required=True,
        help=TRAINING_TIME_HELP_STRING)

    argument_parser.add_argument(
        '--' + VALIDATION_DIR_ARG_NAME, type=str, required=False, default='',
        help=VALIDATION_DIR_HELP_STRING)

    argument_parser.add_argument(
        '--' + FIRST_VALIDATION_TIME_ARG_NAME, type=str, required=False,
        default='', help=VALIDATION_TIME_HELP_STRING)

    argument_parser.add_argument(
        '--' + LAST_VALIDATION_TIME_ARG_NAME, type=str, required=False,
        default='', help=VALIDATION_TIME_HELP_STRING)

    argument_parser.add_argument(
        '--' + NUM_EX_PER_BATCH_ARG_NAME, type=int, required=False,
        default=DEFAULT_NUM_EXAMPLES_PER_BATCH,
        help=NUM_EX_PER_BATCH_HELP_STRING)

    argument_parser.add_argument(
        '--' + NUM_EPOCHS_ARG_NAME, type=int, required=False,
        default=DEFAULT_NUM_EPOCHS, help=NUM_EPOCHS_HELP_STRING)

    argument_parser.add_argument(
        '--' + NUM_TRAINING_BATCHES_ARG_NAME, type=int, required=False,
        default=DEFAULT_NUM_TRAINING_BATCHES_PER_EPOCH,
        help=NUM_TRAINING_BATCHES_HELP_STRING)

    argument_parser.add_argument(
        '--' + NUM_VALIDATION_BATCHES_ARG_NAME, type=int, required=False,
        default=DEFAULT_NUM_VALIDATION_BATCHES_PER_EPOCH,
        help=NUM_VALIDATION_BATCHES_HELP_STRING)

    argument_parser.add_argument(
        '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
        help=OUTPUT_DIR_HELP_STRING)

    return argument_parser
