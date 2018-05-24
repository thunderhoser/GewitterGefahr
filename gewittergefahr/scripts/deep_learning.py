"""Helper methods for deep-learning scripts."""

INPUT_TIME_FORMAT = '%Y-%m-%d-%H%M%S'

MODEL_DIRECTORY_ARG_NAME = 'output_model_dir_name'
NUM_EPOCHS_ARG_NAME = 'num_epochs'
NUM_TRAIN_BATCHES_ARG_NAME = 'num_training_batches_per_epoch'
STORM_IMAGE_DIR_ARG_NAME = 'input_storm_image_dir_name'
RADAR_FIELD_NAMES_ARG_NAME = 'radar_field_names'
NUM_EXAMPLES_PER_BATCH_ARG_NAME = 'num_examples_per_batch'
NUM_EXAMPLES_PER_FILE_TIME_ARG_NAME = 'num_examples_per_file_time'
TRAINING_START_TIME_ARG_NAME = 'training_start_time_string'
TRAINING_END_TIME_ARG_NAME = 'training_end_time_string'
TARGET_NAME_ARG_NAME = 'target_name'
WEIGHT_LOSS_ARG_NAME = 'weight_loss_function'
CLASS_FRACTION_DICT_KEYS_ARG_NAME = 'class_fraction_dict_keys'
CLASS_FRACTION_DICT_VALUES_ARG_NAME = 'class_fraction_dict_values'
NUM_VALIDN_BATCHES_ARG_NAME = 'num_validation_batches_per_epoch'
VALIDATION_START_TIME_ARG_NAME = 'validation_start_time_string'
VALIDATION_END_TIME_ARG_NAME = 'validation_end_time_string'

DEFAULT_NUM_EPOCHS = 100
DEFAULT_NUM_TRAIN_BATCHES_PER_EPOCH = 32
DEFAULT_NUM_EXAMPLES_PER_BATCH = 512
DEFAULT_NUM_EXAMPLES_PER_FILE_TIME = 64
DEFAULT_WEIGHT_LOSS_FLAG = 0
DEFAULT_NUM_VALIDN_BATCHES_PER_EPOCH = 16

MODEL_DIRECTORY_HELP_STRING = (
    'Path to output directory.  The model, training history, and TensorBoard '
    'log files will be saved here after every epoch.')
NUM_EPOCHS_HELP_STRING = 'Number of training epochs.'
NUM_TRAIN_BATCHES_HELP_STRING = 'Number of training batches per epoch.'
STORM_IMAGE_DIR_HELP_STRING = (
    'Name of top-level directory with storm-centered radar images (one file per'
    ' training or validation time, radar field, and height) (readable by'
    ' `storm_images.read_storm_images`).')
RADAR_FIELD_NAMES_HELP_STRING = (
    'List with names of radar fields.  For more details, see documentation for'
    ' `training_validation_io.storm_image_generator_2d` or'
    ' `training_validation_io.storm_image_generator_3d`.')
NUM_EXAMPLES_PER_BATCH_HELP_STRING = (
    'Number of examples per (training or validation) batch.')
NUM_EXAMPLES_PER_FILE_TIME_HELP_STRING = (
    'Number of examples (storm objects) per file time, which may be either one '
    'time step or one SPC date.')
TRAINING_TIME_HELP_STRING = (
    'Time (format "yyyy-mm-dd-HHMMSS").  Training examples will be drawn '
    'randomly from `{0:s}`...`{1:s}`.'
).format(TRAINING_START_TIME_ARG_NAME, TRAINING_END_TIME_ARG_NAME)
TARGET_NAME_HELP_STRING = 'Name of target variable.'
WEIGHT_LOSS_HELP_STRING = (
    'Boolean flag.  If 0, classes will be weighted equally in the loss function'
    '.  If 1, classes will be weighted differently in the loss function '
    '(inversely proportional with `{0:s}`).'
).format(CLASS_FRACTION_DICT_VALUES_ARG_NAME)
CLASS_FRACTION_DICT_KEYS_HELP_STRING = (
    '1-D list of class IDs (integers).  These and `{0:s}` will be used to '
    'create `class_fraction_dict`, which is used to oversample or undersample '
    'different classes.'
).format(CLASS_FRACTION_DICT_VALUES_ARG_NAME)
CLASS_FRACTION_DICT_VALUES_HELP_STRING = (
    '1-D list of class fractions.  These and `{0:s}` will be used to create '
    '`class_fraction_dict`, which is used to oversample or undersample '
    'different classes.'
).format(CLASS_FRACTION_DICT_KEYS_ARG_NAME)
NUM_VALIDN_BATCHES_HELP_STRING = (
    'Number of validation batches per epoch.  If you do not want on-the-fly '
    'validation, make this 0.')
VALIDATION_TIME_HELP_STRING = (
    'Time (format "yyyy-mm-dd-HHMMSS").  Validation examples will be drawn '
    'randomly from `{0:s}`...`{1:s}`.'
).format(VALIDATION_START_TIME_ARG_NAME, VALIDATION_END_TIME_ARG_NAME)


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
        '--' + NUM_TRAIN_BATCHES_ARG_NAME, type=int, required=False,
        default=DEFAULT_NUM_TRAIN_BATCHES_PER_EPOCH,
        help=NUM_TRAIN_BATCHES_HELP_STRING)

    argument_parser_object.add_argument(
        '--' + STORM_IMAGE_DIR_ARG_NAME, type=str, required=True,
        help=STORM_IMAGE_DIR_HELP_STRING)

    argument_parser_object.add_argument(
        '--' + RADAR_FIELD_NAMES_ARG_NAME, type=str, nargs='+', required=True,
        help=RADAR_FIELD_NAMES_HELP_STRING)

    argument_parser_object.add_argument(
        '--' + NUM_EXAMPLES_PER_BATCH_ARG_NAME, type=int, required=False,
        default=DEFAULT_NUM_EXAMPLES_PER_BATCH,
        help=NUM_EXAMPLES_PER_BATCH_HELP_STRING)

    argument_parser_object.add_argument(
        '--' + NUM_EXAMPLES_PER_FILE_TIME_ARG_NAME, type=int, required=False,
        default=DEFAULT_NUM_EXAMPLES_PER_FILE_TIME,
        help=NUM_EXAMPLES_PER_FILE_TIME_HELP_STRING)

    argument_parser_object.add_argument(
        '--' + TRAINING_START_TIME_ARG_NAME, type=str, required=True,
        help=TRAINING_TIME_HELP_STRING)

    argument_parser_object.add_argument(
        '--' + TRAINING_END_TIME_ARG_NAME, type=str, required=True,
        help=TRAINING_TIME_HELP_STRING)

    argument_parser_object.add_argument(
        '--' + TARGET_NAME_ARG_NAME, type=str, required=True,
        help=TARGET_NAME_HELP_STRING)

    argument_parser_object.add_argument(
        '--' + WEIGHT_LOSS_ARG_NAME, type=int, required=False,
        default=DEFAULT_WEIGHT_LOSS_FLAG, help=WEIGHT_LOSS_HELP_STRING)

    argument_parser_object.add_argument(
        '--' + CLASS_FRACTION_DICT_KEYS_ARG_NAME, type=int, nargs='+',
        required=False, default=[0], help=CLASS_FRACTION_DICT_KEYS_HELP_STRING)

    argument_parser_object.add_argument(
        '--' + CLASS_FRACTION_DICT_VALUES_ARG_NAME, type=float, nargs='+',
        required=False, default=[0.],
        help=CLASS_FRACTION_DICT_VALUES_HELP_STRING)

    argument_parser_object.add_argument(
        '--' + NUM_VALIDN_BATCHES_ARG_NAME, type=int, required=False,
        default=DEFAULT_NUM_VALIDN_BATCHES_PER_EPOCH,
        help=NUM_VALIDN_BATCHES_HELP_STRING)

    argument_parser_object.add_argument(
        '--' + VALIDATION_START_TIME_ARG_NAME, type=str, required=False,
        default='None', help=VALIDATION_TIME_HELP_STRING)

    argument_parser_object.add_argument(
        '--' + VALIDATION_END_TIME_ARG_NAME, type=str, required=False,
        default='None', help=VALIDATION_TIME_HELP_STRING)

    return argument_parser_object
