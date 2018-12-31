"""Trains CNN with 2-D GridRad images.

These 2-D images are produced by applying layer operations to the native 3-D
images.
"""

import argparse
import numpy
import keras.losses
import keras.optimizers
from keras.models import clone_model
from gewittergefahr.gg_utils import radar_utils
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.gg_utils import error_checking
from gewittergefahr.deep_learning import cnn
from gewittergefahr.deep_learning import input_examples
from gewittergefahr.deep_learning import cnn_architecture
from gewittergefahr.deep_learning import deep_learning_utils as dl_utils
from gewittergefahr.deep_learning import training_validation_io as trainval_io

# TODO(thunderhoser): Allow this script to handle MYRORSS data.
# TODO(thunderhoser): Allow this script to handle soundings.

TIME_FORMAT = '%Y-%m-%d-%H%M%S'
SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

FIRST_BATCH_NUMBER = 0
LAST_BATCH_NUMBER = int(1e12)

MONITOR_STRING = cnn.LOSS_FUNCTION_STRING + ''
WEIGHT_LOSS_FUNCTION = False
KM_TO_METRES = 1000

NORMALIZATION_TYPE_STRING = dl_utils.Z_NORMALIZATION_TYPE_STRING + ''
NORMALIZATION_PARAM_FILE_NAME = (
    '/condo/swatcommon/common/gridrad_final/myrorss_format/tracks/'
    'correct_echo_tops/reanalyzed/downsampled/for_training/input_examples/'
    'shuffled/single_pol_2011-2015/normalization_params.p')

INPUT_MODEL_FILE_ARG_NAME = 'input_model_file_name'
NUM_ROWS_ARG_NAME = 'num_grid_rows'
NUM_COLUMNS_ARG_NAME = 'num_grid_columns'
DOWNSAMPLING_KEYS_ARG_NAME = 'downsampling_keys'
DOWNSAMPLING_FRACTIONS_ARG_NAME = 'downsampling_fractions'
RADAR_FIELDS_ARG_NAME = 'radar_field_name_by_channel'
LAYER_OPERATIONS_ARG_NAME = 'layer_op_name_by_channel'
MIN_HEIGHTS_ARG_NAME = 'min_height_by_channel_m_agl'
MAX_HEIGHTS_ARG_NAME = 'max_height_by_channel_m_agl'
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

NUM_ROWS_HELP_STRING = (
    'Number of rows in each example (storm-centered radar image).')

NUM_COLUMNS_HELP_STRING = (
    'Number of columns in each example (storm-centered radar image).')

DOWNSAMPLING_KEYS_HELP_STRING = (
    'Keys used to create downsampling dictionary.  Each key is the integer '
    'encoding for a class (-2 for "dead storm"), and each value in `{0:s}` is '
    'the corresponding frequency in training data.  If you do not want '
    'downsampling, make this a one-item list.'
).format(DOWNSAMPLING_FRACTIONS_ARG_NAME)

DOWNSAMPLING_FRACTIONS_HELP_STRING = 'See doc for `{0:s}`.'.format(
    DOWNSAMPLING_KEYS_ARG_NAME)

RADAR_FIELDS_HELP_STRING = (
    'List of radar fields (one for each layer operation).  Each field must be '
    'accepted by `radar_utils.check_field_name`.')

LAYER_OPERATIONS_HELP_STRING = (
    'List of layer operations.  Each string must be in the following list.'
    '\n{0:s}'
).format(str(input_examples.VALID_LAYER_OPERATION_NAMES))

MIN_HEIGHTS_HELP_STRING = (
    'List of minimum heights (one for each layer operations).')

MAX_HEIGHTS_HELP_STRING = 'List of max heights (one for each layer operations).'

TRAINING_DIR_HELP_STRING = (
    'Name of directory with training data.  Files therein will be found by '
    '`input_examples.find_many_example_files` (with shuffled = True) and read '
    'by `input_examples.read_example_file`.')

TRAINING_TIME_HELP_STRING = (
    'Time (format "yyyy-mm-dd-HHMMSS").  Only examples from the period '
    '`{0:s}`...`{1:s}` will be used for training.'
).format(FIRST_TRAINING_TIME_ARG_NAME, LAST_TRAINING_TIME_ARG_NAME)

VALIDATION_DIR_HELP_STRING = (
    'Same as `{0:s}` but for on-the-fly validation.'
).format(TRAINING_DIR_ARG_NAME)

VALIDATION_TIME_HELP_STRING = (
    'Time (format "yyyy-mm-dd-HHMMSS").  Only examples from the period '
    '`{0:s}`...`{1:s}` will be used for validation.'
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

DEFAULT_NUM_GRID_ROWS = 32
DEFAULT_NUM_GRID_COLUMNS = 32
DEFAULT_DOWNSAMPLING_KEYS = numpy.array([0, 1], dtype=int)
DEFAULT_DOWNSAMPLING_FRACTIONS = numpy.array([0.5, 0.5])

# DEFAULT_RADAR_FIELD_NAMES = (
#     [radar_utils.REFL_NAME] * 3 +
#     [radar_utils.DIFFERENTIAL_REFL_NAME] * 3 +
#     [radar_utils.SPECTRUM_WIDTH_NAME] * 3 +
#     [radar_utils.VORTICITY_NAME] * 3 +
#     [radar_utils.VORTICITY_NAME] * 3 +
#     [radar_utils.DIFFERENTIAL_REFL_NAME] * 3
# )
#
# DEFAULT_LAYER_OPERATION_NAMES = [
#     input_examples.MIN_OPERATION_NAME, input_examples.MEAN_OPERATION_NAME,
#     input_examples.MAX_OPERATION_NAME
# ] * 6
#
# DEFAULT_MIN_HEIGHTS_M_AGL = numpy.array(
#     [0] * 3 + [0] * 3 + [0] * 3 + [2000] * 3 + [5000] * 3 + [5000] * 3,
#     dtype=int
# )
#
# DEFAULT_MAX_HEIGHTS_M_AGL = numpy.array(
#     [3000] * 3 + [3000] * 3 + [3000] * 3 + [4000] * 3 + [8000] * 3 +
#     [8000] * 3,
#     dtype=int
# )

DEFAULT_RADAR_FIELD_NAMES = (
    [radar_utils.REFL_NAME] * 3 +
    [radar_utils.SPECTRUM_WIDTH_NAME] * 3 +
    [radar_utils.VORTICITY_NAME] * 3 +
    [radar_utils.VORTICITY_NAME] * 3
)

DEFAULT_LAYER_OPERATION_NAMES = [
    input_examples.MIN_OPERATION_NAME, input_examples.MEAN_OPERATION_NAME,
    input_examples.MAX_OPERATION_NAME
] * 4

DEFAULT_MIN_HEIGHTS_M_AGL = numpy.array(
    [0] * 3 + [0] * 3 + [2000] * 3 + [5000] * 3,
    dtype=int
)

DEFAULT_MAX_HEIGHTS_M_AGL = numpy.array(
    [3000] * 3 + [3000] * 3 + [4000] * 3 + [8000] * 3,
    dtype=int
)

DEFAULT_TOP_TRAINING_DIR_NAME = (
    '/condo/swatcommon/common/gridrad_final/myrorss_format/tracks/'
    'correct_echo_tops/reanalyzed/downsampled/for_training/input_examples/'
    'shuffled/single_pol_2011-2015'
)
DEFAULT_FIRST_TRAINING_TIME_STRING = '2011-01-01-000000'
DEFAULT_LAST_TRAINING_TIME_STRING = '2013-12-31-120000'

DEFAULT_TOP_VALIDATION_DIR_NAME = (
    '/condo/swatcommon/common/gridrad_final/myrorss_format/tracks/'
    'correct_echo_tops/reanalyzed/downsampled/for_training/input_examples/'
    'shuffled/single_pol_2011-2015'
)
DEFAULT_FIRST_VALIDN_TIME_STRING = '2014-01-01-120000'
DEFAULT_LAST_VALIDN_TIME_STRING = '2015-12-31-120000'

DEFAULT_NUM_EXAMPLES_PER_BATCH = 1024
DEFAULT_NUM_EPOCHS = 100
DEFAULT_NUM_TRAINING_BATCHES_PER_EPOCH = 32
DEFAULT_NUM_VALIDATION_BATCHES_PER_EPOCH = 16

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + INPUT_MODEL_FILE_ARG_NAME, type=str, required=True,
    help=INPUT_MODEL_FILE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + NUM_ROWS_ARG_NAME, type=int, required=False,
    default=DEFAULT_NUM_GRID_ROWS, help=NUM_ROWS_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + NUM_COLUMNS_ARG_NAME, type=int, required=False,
    default=DEFAULT_NUM_GRID_COLUMNS, help=NUM_COLUMNS_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + DOWNSAMPLING_KEYS_ARG_NAME, type=int, nargs='+', required=False,
    default=DEFAULT_DOWNSAMPLING_KEYS, help=DOWNSAMPLING_KEYS_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + DOWNSAMPLING_FRACTIONS_ARG_NAME, type=float, nargs='+',
    required=False, default=DEFAULT_DOWNSAMPLING_FRACTIONS,
    help=DOWNSAMPLING_FRACTIONS_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + RADAR_FIELDS_ARG_NAME, type=str, nargs='+', required=False,
    default=DEFAULT_RADAR_FIELD_NAMES, help=RADAR_FIELDS_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + LAYER_OPERATIONS_ARG_NAME, type=str, nargs='+', required=False,
    default=DEFAULT_LAYER_OPERATION_NAMES, help=LAYER_OPERATIONS_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + MIN_HEIGHTS_ARG_NAME, type=int, nargs='+', required=False,
    default=DEFAULT_MIN_HEIGHTS_M_AGL, help=MIN_HEIGHTS_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + MAX_HEIGHTS_ARG_NAME, type=int, nargs='+', required=False,
    default=DEFAULT_MAX_HEIGHTS_M_AGL, help=MAX_HEIGHTS_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + TRAINING_DIR_ARG_NAME, type=str, required=False,
    default=DEFAULT_TOP_TRAINING_DIR_NAME, help=TRAINING_DIR_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + FIRST_TRAINING_TIME_ARG_NAME, type=str, required=False,
    default=DEFAULT_FIRST_TRAINING_TIME_STRING, help=TRAINING_TIME_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + LAST_TRAINING_TIME_ARG_NAME, type=str, required=False,
    default=DEFAULT_LAST_TRAINING_TIME_STRING, help=TRAINING_TIME_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + VALIDATION_DIR_ARG_NAME, type=str, required=False,
    default=DEFAULT_TOP_VALIDATION_DIR_NAME, help=VALIDATION_DIR_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + FIRST_VALIDATION_TIME_ARG_NAME, type=str, required=False,
    default=DEFAULT_FIRST_VALIDN_TIME_STRING, help=VALIDATION_TIME_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + LAST_VALIDATION_TIME_ARG_NAME, type=str, required=False,
    default=DEFAULT_LAST_VALIDN_TIME_STRING, help=VALIDATION_TIME_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + NUM_EX_PER_BATCH_ARG_NAME, type=int, required=False,
    default=DEFAULT_NUM_EXAMPLES_PER_BATCH, help=NUM_EX_PER_BATCH_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + NUM_EPOCHS_ARG_NAME, type=int, required=False,
    default=DEFAULT_NUM_EPOCHS, help=NUM_EPOCHS_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + NUM_TRAINING_BATCHES_ARG_NAME, type=int, required=False,
    default=DEFAULT_NUM_TRAINING_BATCHES_PER_EPOCH,
    help=NUM_TRAINING_BATCHES_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + NUM_VALIDATION_BATCHES_ARG_NAME, type=int, required=False,
    default=DEFAULT_NUM_VALIDATION_BATCHES_PER_EPOCH,
    help=NUM_VALIDATION_BATCHES_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING)


def _run(input_model_file_name, num_grid_rows, num_grid_columns,
         downsampling_keys, downsampling_fractions, radar_field_name_by_channel,
         layer_op_name_by_channel, min_height_by_channel_m_agl,
         max_height_by_channel_m_agl, top_training_dir_name,
         first_training_time_string, last_training_time_string,
         top_validation_dir_name, first_validation_time_string,
         last_validation_time_string, num_examples_per_batch, num_epochs,
         num_training_batches_per_epoch, num_validation_batches_per_epoch,
         output_dir_name):
    """Trains CNN with 2-D GridRad images.

    This is effectively the main method.

    :param input_model_file_name: See documentation at top of file.
    :param num_grid_rows: Same.
    :param num_grid_columns: Same.
    :param downsampling_keys: Same.
    :param downsampling_fractions: Same.
    :param radar_field_name_by_channel: Same.
    :param layer_op_name_by_channel: Same.
    :param min_height_by_channel_m_agl: Same.
    :param max_height_by_channel_m_agl: Same.
    :param top_training_dir_name: Same.
    :param first_training_time_string: Same.
    :param last_training_time_string: Same.
    :param top_validation_dir_name: Same.
    :param first_validation_time_string: Same.
    :param last_validation_time_string: Same.
    :param num_examples_per_batch: Same.
    :param num_epochs: Same.
    :param num_training_batches_per_epoch: Same.
    :param num_validation_batches_per_epoch: Same.
    :param output_dir_name: Same.
    """

    # Process input args.
    first_training_time_unix_sec = time_conversion.string_to_unix_sec(
        first_training_time_string, TIME_FORMAT)
    last_training_time_unix_sec = time_conversion.string_to_unix_sec(
        last_training_time_string, TIME_FORMAT)

    first_validation_time_unix_sec = time_conversion.string_to_unix_sec(
        first_validation_time_string, TIME_FORMAT)
    last_validation_time_unix_sec = time_conversion.string_to_unix_sec(
        last_validation_time_string, TIME_FORMAT)

    if len(downsampling_keys) > 1:
        class_to_sampling_fraction_dict = dict(zip(
            downsampling_keys, downsampling_fractions
        ))
    else:
        class_to_sampling_fraction_dict = None

    num_channels = len(radar_field_name_by_channel)
    expected_dimensions = numpy.array([num_channels], dtype=int)

    error_checking.assert_is_numpy_array(
        numpy.array(layer_op_name_by_channel),
        exact_dimensions=expected_dimensions)

    error_checking.assert_is_numpy_array(
        min_height_by_channel_m_agl, exact_dimensions=expected_dimensions)
    error_checking.assert_is_numpy_array(
        max_height_by_channel_m_agl, exact_dimensions=expected_dimensions)

    list_of_layer_operation_dicts = [{}] * num_channels
    for m in range(num_channels):
        list_of_layer_operation_dicts[m] = {
            input_examples.RADAR_FIELD_KEY: radar_field_name_by_channel[m],
            input_examples.OPERATION_NAME_KEY: layer_op_name_by_channel[m],
            input_examples.MIN_HEIGHT_KEY: min_height_by_channel_m_agl[m],
            input_examples.MAX_HEIGHT_KEY: max_height_by_channel_m_agl[m]
        }

    # Set output locations.
    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=output_dir_name)

    output_model_file_name = '{0:s}/model.h5'.format(output_dir_name)
    history_file_name = '{0:s}/model_history.csv'.format(output_dir_name)
    tensorboard_dir_name = '{0:s}/tensorboard'.format(output_dir_name)
    model_metafile_name = '{0:s}/model_metadata.p'.format(output_dir_name)

    # Find training and validation files.
    training_file_names = input_examples.find_many_example_files(
        top_directory_name=top_training_dir_name, shuffled=True,
        first_batch_number=FIRST_BATCH_NUMBER,
        last_batch_number=LAST_BATCH_NUMBER, raise_error_if_any_missing=False)

    validation_file_names = input_examples.find_many_example_files(
        top_directory_name=top_validation_dir_name, shuffled=True,
        first_batch_number=FIRST_BATCH_NUMBER,
        last_batch_number=LAST_BATCH_NUMBER, raise_error_if_any_missing=False)

    this_example_dict = input_examples.read_example_file(
        netcdf_file_name=training_file_names[0], metadata_only=True)
    target_name = this_example_dict[input_examples.TARGET_NAME_KEY]

    metadata_dict = {
        cnn.TARGET_NAME_KEY: target_name,
        cnn.NUM_EPOCHS_KEY: num_epochs,
        cnn.NUM_TRAINING_BATCHES_KEY: num_training_batches_per_epoch,
        cnn.NUM_VALIDATION_BATCHES_KEY: num_validation_batches_per_epoch,
        cnn.MONITOR_STRING_KEY: MONITOR_STRING,
        cnn.WEIGHT_LOSS_FUNCTION_KEY: WEIGHT_LOSS_FUNCTION,
        cnn.USE_2D3D_CONVOLUTION_KEY: False,
        cnn.VALIDATION_FILES_KEY: validation_file_names,
        cnn.FIRST_VALIDN_TIME_KEY: first_validation_time_unix_sec,
        cnn.LAST_VALIDN_TIME_KEY: last_validation_time_unix_sec
    }

    # TODO(thunderhoser): Find more elegant way to do this.
    unique_radar_field_names = list(set(radar_field_name_by_channel))

    min_overall_height_m_agl = numpy.min(min_height_by_channel_m_agl)
    max_overall_height_m_agl = numpy.max(min_height_by_channel_m_agl)
    num_overall_heights = 1 + int(numpy.round(
        (max_overall_height_m_agl - min_overall_height_m_agl) / KM_TO_METRES
    ))

    unique_radar_heights_m_agl = numpy.linspace(
        min_overall_height_m_agl, max_overall_height_m_agl,
        num=num_overall_heights, dtype=int)

    print 'Unique radar heights (m AGL):\n{0:s}\n'.format(
        str(unique_radar_heights_m_agl))

    training_option_dict = {
        trainval_io.EXAMPLE_FILES_KEY: training_file_names,
        trainval_io.FIRST_STORM_TIME_KEY: first_training_time_unix_sec,
        trainval_io.LAST_STORM_TIME_KEY: last_training_time_unix_sec,
        trainval_io.NUM_EXAMPLES_PER_BATCH_KEY: num_examples_per_batch,
        trainval_io.RADAR_FIELDS_KEY: unique_radar_field_names,
        trainval_io.RADAR_HEIGHTS_KEY: unique_radar_heights_m_agl,
        trainval_io.SOUNDING_FIELDS_KEY: None,
        trainval_io.SOUNDING_HEIGHTS_KEY: None,
        trainval_io.NUM_ROWS_KEY: num_grid_rows,
        trainval_io.NUM_COLUMNS_KEY: num_grid_columns,
        trainval_io.NORMALIZATION_TYPE_KEY: NORMALIZATION_TYPE_STRING,
        trainval_io.NORMALIZATION_FILE_KEY: NORMALIZATION_PARAM_FILE_NAME,
        trainval_io.SAMPLING_FRACTIONS_KEY: class_to_sampling_fraction_dict,
        trainval_io.REFLECTIVITY_MASK_KEY: None
    }

    training_option_dict = trainval_io.check_generator_input_args(
        training_option_dict)

    print 'Writing metadata to: "{0:s}"...'.format(model_metafile_name)
    cnn.write_model_metadata(
        pickle_file_name=model_metafile_name, metadata_dict=metadata_dict,
        training_option_dict=training_option_dict,
        list_of_layer_operation_dicts=list_of_layer_operation_dicts)

    print 'Reading architecture from: "{0:s}"...'.format(input_model_file_name)
    model_object = cnn.read_model(input_model_file_name)
    model_object = clone_model(model_object)

    # TODO(thunderhoser): This is a HACK.
    model_object.compile(
        loss=keras.losses.categorical_crossentropy,
        optimizer=keras.optimizers.Adam(),
        metrics=cnn_architecture.DEFAULT_METRIC_FUNCTION_LIST)

    print SEPARATOR_STRING
    model_object.summary()
    print SEPARATOR_STRING

    cnn.train_cnn_gridrad_2d_reduced(
        model_object=model_object, model_file_name=output_model_file_name,
        history_file_name=history_file_name,
        tensorboard_dir_name=tensorboard_dir_name,
        num_epochs=num_epochs,
        num_training_batches_per_epoch=num_training_batches_per_epoch,
        training_option_dict=training_option_dict,
        list_of_layer_operation_dicts=list_of_layer_operation_dicts,
        monitor_string=MONITOR_STRING,
        weight_loss_function=WEIGHT_LOSS_FUNCTION,
        num_validation_batches_per_epoch=num_validation_batches_per_epoch,
        validation_file_names=validation_file_names,
        first_validn_time_unix_sec=first_validation_time_unix_sec,
        last_validn_time_unix_sec=last_validation_time_unix_sec)


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        input_model_file_name=getattr(
            INPUT_ARG_OBJECT, INPUT_MODEL_FILE_ARG_NAME),
        num_grid_rows=getattr(INPUT_ARG_OBJECT, NUM_ROWS_ARG_NAME),
        num_grid_columns=getattr(INPUT_ARG_OBJECT, NUM_COLUMNS_ARG_NAME),
        downsampling_keys=numpy.array(getattr(
            INPUT_ARG_OBJECT, DOWNSAMPLING_KEYS_ARG_NAME), dtype=int),
        downsampling_fractions=numpy.array(getattr(
            INPUT_ARG_OBJECT, DOWNSAMPLING_FRACTIONS_ARG_NAME), dtype=float),
        radar_field_name_by_channel=getattr(
            INPUT_ARG_OBJECT, RADAR_FIELDS_ARG_NAME),
        layer_op_name_by_channel=getattr(
            INPUT_ARG_OBJECT, LAYER_OPERATIONS_ARG_NAME),
        min_height_by_channel_m_agl=numpy.array(getattr(
            INPUT_ARG_OBJECT, MIN_HEIGHTS_ARG_NAME), dtype=int),
        max_height_by_channel_m_agl=numpy.array(getattr(
            INPUT_ARG_OBJECT, MAX_HEIGHTS_ARG_NAME), dtype=int),
        top_training_dir_name=getattr(INPUT_ARG_OBJECT, TRAINING_DIR_ARG_NAME),
        first_training_time_string=getattr(
            INPUT_ARG_OBJECT, FIRST_TRAINING_TIME_ARG_NAME),
        last_training_time_string=getattr(
            INPUT_ARG_OBJECT, LAST_TRAINING_TIME_ARG_NAME),
        top_validation_dir_name=getattr(
            INPUT_ARG_OBJECT, VALIDATION_DIR_ARG_NAME),
        first_validation_time_string=getattr(
            INPUT_ARG_OBJECT, FIRST_VALIDATION_TIME_ARG_NAME),
        last_validation_time_string=getattr(
            INPUT_ARG_OBJECT, LAST_VALIDATION_TIME_ARG_NAME),
        num_examples_per_batch=getattr(
            INPUT_ARG_OBJECT, NUM_EX_PER_BATCH_ARG_NAME),
        num_epochs=getattr(INPUT_ARG_OBJECT, NUM_EPOCHS_ARG_NAME),
        num_training_batches_per_epoch=getattr(
            INPUT_ARG_OBJECT, NUM_TRAINING_BATCHES_ARG_NAME),
        num_validation_batches_per_epoch=getattr(
            INPUT_ARG_OBJECT, NUM_VALIDATION_BATCHES_ARG_NAME),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
