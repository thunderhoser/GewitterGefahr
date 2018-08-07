"""Handles input args for deep-learning scripts."""

from gewittergefahr.gg_utils import soundings_only
from gewittergefahr.deep_learning import cnn

INPUT_TIME_FORMAT = '%Y-%m-%d-%H%M%S'

MODEL_DIRECTORY_ARG_NAME = 'output_model_dir_name'
NUM_EPOCHS_ARG_NAME = 'num_epochs'
NUM_EXAMPLES_PER_BATCH_ARG_NAME = 'num_examples_per_batch'
NUM_EXAMPLES_PER_FILE_TIME_ARG_NAME = 'num_examples_per_file_time'
NUM_TRAIN_BATCHES_ARG_NAME = 'num_training_batches_per_epoch'
RADAR_DIRECTORY_ARG_NAME = 'input_storm_radar_image_dir_name'
RADAR_DIRECTORY_POS_TARGETS_ARG_NAME = (
    'input_storm_radar_image_dir_name_pos_targets_only')
ONE_FILE_PER_TIME_STEP_ARG_NAME = 'one_file_per_time_step'
FIRST_TRAINING_TIME_ARG_NAME = 'first_training_time_string'
LAST_TRAINING_TIME_ARG_NAME = 'last_training_time_string'
MONITOR_STRING_ARG_NAME = 'monitor_string'
RADAR_FIELD_NAMES_ARG_NAME = 'radar_field_names'
NUM_RADAR_CONV_LAYERS_ARG_NAME = 'num_radar_conv_layers'
TARGET_NAME_ARG_NAME = 'target_name'
TARGET_DIRECTORY_ARG_NAME = 'input_target_dir_name'
BINARIZE_TARGET_ARG_NAME = 'binarize_target'
DROPOUT_FRACTION_ARG_NAME = 'dropout_fraction'
L2_WEIGHT_ARG_NAME = 'l2_weight'
SAMPLING_FRACTION_KEYS_ARG_NAME = 'sampling_fraction_keys'
SAMPLING_FRACTION_VALUES_ARG_NAME = 'sampling_fraction_values'
WEIGHT_LOSS_ARG_NAME = 'weight_loss_function'
NUM_VALIDN_BATCHES_ARG_NAME = 'num_validation_batches_per_epoch'
FIRST_VALIDATION_TIME_ARG_NAME = 'first_validation_time_string'
LAST_VALIDATION_TIME_ARG_NAME = 'last_validation_time_string'
SOUNDING_FIELD_NAMES_ARG_NAME = 'sounding_field_names'
SOUNDING_DIRECTORY_ARG_NAME = 'input_sounding_dir_name'
SOUNDING_LAG_TIME_ARG_NAME = (
    'sounding_lag_time_for_convective_contamination_sec')
NUM_SOUNDING_FILTERS_ARG_NAME = 'num_sounding_filters_in_first_layer'

DEFAULT_NUM_EPOCHS = 100
DEFAULT_NUM_EXAMPLES_PER_BATCH = 1024
DEFAULT_NUM_TRAIN_BATCHES_PER_EPOCH = 32
DEFAULT_ONE_FILE_PER_TIME_STEP_FLAG = 0
DEFAULT_MONITOR_STRING = cnn.LOSS_AS_MONITOR_STRING
DEFAULT_NUM_RADAR_CONV_LAYERS = 3
DEFAULT_BINARIZE_TARGET_FLAG = 0
DEFAULT_WEIGHT_LOSS_FLAG = 0
DEFAULT_DROPOUT_FRACTION = 0.25
DEFAULT_L2_WEIGHT = 1e-3
DEFAULT_NUM_VALIDN_BATCHES_PER_EPOCH = 16
DEFAULT_SOUNDING_FIELD_NAMES = [
    soundings_only.TEMPERATURE_NAME,
    soundings_only.VIRTUAL_POTENTIAL_TEMPERATURE_NAME,
    soundings_only.RELATIVE_HUMIDITY_NAME,
    soundings_only.SPECIFIC_HUMIDITY_NAME,
    soundings_only.U_WIND_NAME, soundings_only.V_WIND_NAME
]
DEFAULT_SOUNDING_LAG_TIME_SEC = 1800
DEFAULT_NUM_SOUNDING_FILTERS = 48

MODEL_DIRECTORY_HELP_STRING = (
    'Name of output directory.  The model, training history, and TensorBoard '
    'log files will be saved here after every epoch.')
NUM_EPOCHS_HELP_STRING = 'Number of epochs.'
NUM_EXAMPLES_PER_BATCH_HELP_STRING = (
    'Number of examples (storm objects) per batch.')
NUM_EXAMPLES_PER_FILE_TIME_HELP_STRING = (
    'Number of examples (storm objects) per file time.  If `{0:s}` = True, this'
    ' is the number of examples per time step.  If `{0:s}` = False, this is '
    'number of examples per SPC date.'
).format(ONE_FILE_PER_TIME_STEP_ARG_NAME)
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
ONE_FILE_PER_TIME_STEP_HELP_STRING = (
    'Boolean flag.  If 1 (0), the model will be trained with one set of files '
    'per time step (SPC date).')
TRAINING_TIME_HELP_STRING = (
    'Training time (format "yyyy-mm-dd-HHMMSS").  The model will be trained '
    'with storm objects from `{0:s}`...`{1:s}`.  If `{2:s}` = False, this can '
    'be any time in the first SPC date.'
).format(FIRST_TRAINING_TIME_ARG_NAME, LAST_TRAINING_TIME_ARG_NAME,
         ONE_FILE_PER_TIME_STEP_ARG_NAME)
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
NUM_RADAR_CONV_LAYERS_HELP_STRING = (
    'Number of convolutional layers for radar data.  Each successive conv layer'
    ' will cut the dimensions of the radar image in half (example: from '
    '32 x 32 x 12 to 16 x 16 x 6, then 8 x 8 x 3, then 4 x 4 x 1).')
TARGET_NAME_HELP_STRING = (
    'Name of target variable (must be accepted by '
    '`labels.column_name_to_label_params`).')
TARGET_DIRECTORY_HELP_STRING = (
    'Name of top-level directory with labels (target values).  Files therein '
    'will be found by `labels.find_label_file`.')
BINARIZE_TARGET_HELP_STRING = (
    'Boolean flag.  If 1, will binarize target variable, so that the highest '
    'class becomes 1 and all other classes become 0.')
DROPOUT_FRACTION_HELP_STRING = (
    'Dropout fraction.  Will be applied to the weights in each convolutional '
    'layer.  Set to -1 for no dropout.')
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
VALIDATION_TIME_HELP_STRING = (
    'Validation time (format "yyyy-mm-dd-HHMMSS").  The model will be '
    'validation on the fly with storm objects from `{0:s}`...`{1:s}`.  If '
    '`{2:s}` = False, this can be any time in the first SPC date.'
).format(FIRST_VALIDATION_TIME_ARG_NAME, LAST_VALIDATION_TIME_ARG_NAME,
         ONE_FILE_PER_TIME_STEP_ARG_NAME)
SOUNDING_FIELD_NAMES_HELP_STRING = (
    'List with names of sounding fields.  Each name must be accepted by '
    '`soundings_only.check_pressureless_field_name`.  To train without '
    'soundings, make this a 1-item list with the string "None".')
SOUNDING_DIRECTORY_HELP_STRING = (
    'Name of top-level directory with storm-centered soundings.  Files therein '
    'will be found by `training_validation_io.find_sounding_files`.')
SOUNDING_LAG_TIME_HELP_STRING = (
    'Lag time for soundings (see '
    '`soundings_only.interp_soundings_to_storm_objects` for details).  This '
    'will be used to locate sounding files.')
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
        '--' + NUM_EXAMPLES_PER_FILE_TIME_ARG_NAME, type=int, required=True,
        help=NUM_EXAMPLES_PER_FILE_TIME_HELP_STRING)

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
        '--' + ONE_FILE_PER_TIME_STEP_ARG_NAME, type=int, required=False,
        default=DEFAULT_ONE_FILE_PER_TIME_STEP_FLAG,
        help=ONE_FILE_PER_TIME_STEP_HELP_STRING)

    argument_parser_object.add_argument(
        '--' + FIRST_TRAINING_TIME_ARG_NAME, type=str, required=True,
        help=TRAINING_TIME_HELP_STRING)

    argument_parser_object.add_argument(
        '--' + LAST_TRAINING_TIME_ARG_NAME, type=str, required=True,
        help=TRAINING_TIME_HELP_STRING)

    argument_parser_object.add_argument(
        '--' + MONITOR_STRING_ARG_NAME, type=str, required=False,
        default=DEFAULT_MONITOR_STRING, help=MONITOR_STRING_HELP_STRING)

    argument_parser_object.add_argument(
        '--' + RADAR_FIELD_NAMES_ARG_NAME, type=str, nargs='+', required=False,
        default=[''], help=RADAR_FIELD_NAMES_HELP_STRING)

    argument_parser_object.add_argument(
        '--' + NUM_RADAR_CONV_LAYERS_ARG_NAME, type=int, required=True,
        default=DEFAULT_NUM_RADAR_CONV_LAYERS,
        help=NUM_RADAR_CONV_LAYERS_HELP_STRING)

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
        '--' + DROPOUT_FRACTION_ARG_NAME, type=float, required=False,
        default=DEFAULT_DROPOUT_FRACTION, help=DROPOUT_FRACTION_HELP_STRING)

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
        '--' + FIRST_VALIDATION_TIME_ARG_NAME, type=str, required=False,
        default='', help=VALIDATION_TIME_HELP_STRING)

    argument_parser_object.add_argument(
        '--' + LAST_VALIDATION_TIME_ARG_NAME, type=str, required=False,
        default='', help=VALIDATION_TIME_HELP_STRING)

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
