"""Trains CNN with 2-D GridRad images.

These 2-D images are produced by applying layer operations to the native 3-D
images.
"""

import argparse
import numpy
import keras.losses
import keras.optimizers
from keras import backend as K
from gewittergefahr.gg_utils import soundings
from gewittergefahr.gg_utils import radar_utils
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.gg_utils import error_checking
from gewittergefahr.deep_learning import cnn
from gewittergefahr.deep_learning import input_examples
from gewittergefahr.deep_learning import cnn_setup
from gewittergefahr.deep_learning import training_validation_io as trainval_io
from gewittergefahr.scripts import deep_learning_helper as dl_helper

K.set_session(K.tf.Session(config=K.tf.ConfigProto(
    intra_op_parallelism_threads=7, inter_op_parallelism_threads=7
)))

TIME_FORMAT = '%Y-%m-%d-%H%M%S'
SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

FIRST_BATCH_NUMBER = 0
LAST_BATCH_NUMBER = int(1e12)
SOUNDING_HEIGHTS_M_AGL = soundings.DEFAULT_HEIGHT_LEVELS_M_AGL + 0

RADAR_FIELDS_ARG_NAME = 'radar_field_name_by_channel'
LAYER_OPERATIONS_ARG_NAME = 'layer_op_name_by_channel'
MIN_HEIGHTS_ARG_NAME = 'min_height_by_channel_m_agl'
MAX_HEIGHTS_ARG_NAME = 'max_height_by_channel_m_agl'

RADAR_FIELDS_HELP_STRING = (
    'List of radar fields (one for each layer operation).  Each field must be '
    'accepted by `radar_utils.check_field_name`.')

LAYER_OPERATIONS_HELP_STRING = (
    'List of layer operations.  Each string must be in the following list.'
    '\n{0:s}'
).format(str(input_examples.VALID_LAYER_OPERATION_NAMES))

MIN_HEIGHTS_HELP_STRING = (
    'List of minimum heights (one for each layer operation).')

MAX_HEIGHTS_HELP_STRING = 'List of max heights (one for each layer operation).'

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
#     [1000] * 3 + [1000] * 3 + [1000] * 3 + [2000] * 3 + [5000] * 3 +
#     [5000] * 3,
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
    [1000] * 3 + [1000] * 3 + [2000] * 3 + [5000] * 3,
    dtype=int
)

DEFAULT_MAX_HEIGHTS_M_AGL = numpy.array(
    [3000] * 3 + [3000] * 3 + [4000] * 3 + [8000] * 3,
    dtype=int
)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER = dl_helper.add_input_args(argument_parser=INPUT_ARG_PARSER)

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


def _run(input_model_file_name, radar_field_name_by_channel,
         layer_op_name_by_channel, min_height_by_channel_m_agl,
         max_height_by_channel_m_agl, sounding_field_names,
         normalization_type_string, normalization_param_file_name,
         min_normalized_value, max_normalized_value, target_name,
         shuffle_target, downsampling_classes, downsampling_fractions,
         monitor_string, weight_loss_function, x_translations_pixels,
         y_translations_pixels, ccw_rotation_angles_deg,
         noise_standard_deviation, num_noisings, flip_in_x, flip_in_y,
         top_training_dir_name, first_training_time_string,
         last_training_time_string, num_examples_per_train_batch,
         top_validation_dir_name, first_validation_time_string,
         last_validation_time_string, num_examples_per_validn_batch, num_epochs,
         num_training_batches_per_epoch, num_validation_batches_per_epoch,
         output_dir_name):
    """Trains CNN with 2-D GridRad images.

    This is effectively the main method.

    :param input_model_file_name: See documentation at top of file.
    :param radar_field_name_by_channel: Same.
    :param layer_op_name_by_channel: Same.
    :param min_height_by_channel_m_agl: Same.
    :param max_height_by_channel_m_agl: Same.
    :param sounding_field_names: Same.
    :param normalization_type_string: Same.
    :param normalization_param_file_name: Same.
    :param min_normalized_value: Same.
    :param max_normalized_value: Same.
    :param target_name: Same.
    :param shuffle_target: Same.
    :param downsampling_classes: Same.
    :param downsampling_fractions: Same.
    :param monitor_string: Same.
    :param weight_loss_function: Same.
    :param x_translations_pixels: Same.
    :param y_translations_pixels: Same.
    :param ccw_rotation_angles_deg: Same.
    :param noise_standard_deviation: Same.
    :param num_noisings: Same.
    :param flip_in_x: Same.
    :param flip_in_y: Same.
    :param top_training_dir_name: Same.
    :param first_training_time_string: Same.
    :param last_training_time_string: Same.
    :param num_examples_per_train_batch: Same.
    :param top_validation_dir_name: Same.
    :param first_validation_time_string: Same.
    :param last_validation_time_string: Same.
    :param num_examples_per_validn_batch: Same.
    :param num_epochs: Same.
    :param num_training_batches_per_epoch: Same.
    :param num_validation_batches_per_epoch: Same.
    :param output_dir_name: Same.
    """

    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=output_dir_name)

    # argument_file_name = '{0:s}/input_args.p'.format(output_dir_name)
    # print('Writing input args to: "{0:s}"...'.format(argument_file_name))
    #
    # argument_file_handle = open(argument_file_name, 'wb')
    # pickle.dump(INPUT_ARG_OBJECT.__dict__, argument_file_handle)
    # argument_file_handle.close()
    #
    # return

    # Process input args.
    first_training_time_unix_sec = time_conversion.string_to_unix_sec(
        first_training_time_string, TIME_FORMAT)
    last_training_time_unix_sec = time_conversion.string_to_unix_sec(
        last_training_time_string, TIME_FORMAT)

    if top_validation_dir_name in ['', 'None']:
        top_validation_dir_name = None
        num_validation_batches_per_epoch = 0
        first_validation_time_unix_sec = 0
        last_validation_time_unix_sec = 0
    else:
        first_validation_time_unix_sec = time_conversion.string_to_unix_sec(
            first_validation_time_string, TIME_FORMAT)
        last_validation_time_unix_sec = time_conversion.string_to_unix_sec(
            last_validation_time_string, TIME_FORMAT)

    if sounding_field_names[0] in ['', 'None']:
        sounding_field_names = None

    if len(downsampling_classes) > 1:
        class_to_sampling_fraction_dict = dict(list(zip(
            downsampling_classes, downsampling_fractions
        )))
    else:
        class_to_sampling_fraction_dict = None

    if (len(x_translations_pixels) == 1 and
            x_translations_pixels + y_translations_pixels == 0
       ):
        x_translations_pixels = None
        y_translations_pixels = None

    if len(ccw_rotation_angles_deg) == 1 and ccw_rotation_angles_deg[0] == 0:
        ccw_rotation_angles_deg = None

    if num_noisings <= 0:
        num_noisings = 0
        noise_standard_deviation = None

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
    output_model_file_name = '{0:s}/model.h5'.format(output_dir_name)
    history_file_name = '{0:s}/model_history.csv'.format(output_dir_name)
    tensorboard_dir_name = '{0:s}/tensorboard'.format(output_dir_name)
    model_metafile_name = '{0:s}/model_metadata.p'.format(output_dir_name)

    # Find training and validation files.
    training_file_names = input_examples.find_many_example_files(
        top_directory_name=top_training_dir_name, shuffled=True,
        first_batch_number=FIRST_BATCH_NUMBER,
        last_batch_number=LAST_BATCH_NUMBER, raise_error_if_any_missing=False)

    if top_validation_dir_name is None:
        validation_file_names = []
    else:
        validation_file_names = input_examples.find_many_example_files(
            top_directory_name=top_validation_dir_name, shuffled=True,
            first_batch_number=FIRST_BATCH_NUMBER,
            last_batch_number=LAST_BATCH_NUMBER,
            raise_error_if_any_missing=False)

    # Read architecture.
    print('Reading architecture from: "{0:s}"...'.format(input_model_file_name))
    model_object = cnn.read_model(input_model_file_name)
    # model_object = clone_model(model_object)

    # TODO(thunderhoser): This is a HACK.
    model_object.compile(
        loss=keras.losses.binary_crossentropy,
        optimizer=keras.optimizers.Adam(),
        metrics=cnn_setup.DEFAULT_METRIC_FUNCTION_LIST)

    print(SEPARATOR_STRING)
    model_object.summary()
    print(SEPARATOR_STRING)

    # print(K.eval(
    #     model_object.get_layer(name='radar_conv2d_2').weights[0]
    # ))
    # print(SEPARATOR_STRING)

    # Write metadata.
    metadata_dict = {
        cnn.NUM_EPOCHS_KEY: num_epochs,
        cnn.NUM_TRAINING_BATCHES_KEY: num_training_batches_per_epoch,
        cnn.NUM_VALIDATION_BATCHES_KEY: num_validation_batches_per_epoch,
        cnn.MONITOR_STRING_KEY: monitor_string,
        cnn.WEIGHT_LOSS_FUNCTION_KEY: weight_loss_function,
        cnn.CONV_2D3D_KEY: False,
        cnn.VALIDATION_FILES_KEY: validation_file_names,
        cnn.FIRST_VALIDN_TIME_KEY: first_validation_time_unix_sec,
        cnn.LAST_VALIDN_TIME_KEY: last_validation_time_unix_sec,
        cnn.NUM_EX_PER_VALIDN_BATCH_KEY: num_examples_per_validn_batch
    }

    input_tensor = model_object.input
    if isinstance(input_tensor, list):
        input_tensor = input_tensor[0]

    num_grid_rows = input_tensor.get_shape().as_list()[1]
    num_grid_columns = input_tensor.get_shape().as_list()[2]

    training_option_dict = {
        trainval_io.EXAMPLE_FILES_KEY: training_file_names,
        trainval_io.TARGET_NAME_KEY: target_name,
        trainval_io.SHUFFLE_TARGET_KEY: shuffle_target,
        trainval_io.FIRST_STORM_TIME_KEY: first_training_time_unix_sec,
        trainval_io.LAST_STORM_TIME_KEY: last_training_time_unix_sec,
        trainval_io.NUM_EXAMPLES_PER_BATCH_KEY: num_examples_per_train_batch,
        trainval_io.SOUNDING_FIELDS_KEY: sounding_field_names,
        trainval_io.SOUNDING_HEIGHTS_KEY: SOUNDING_HEIGHTS_M_AGL,
        trainval_io.NUM_ROWS_KEY: num_grid_rows,
        trainval_io.NUM_COLUMNS_KEY: num_grid_columns,
        trainval_io.NORMALIZATION_TYPE_KEY: normalization_type_string,
        trainval_io.NORMALIZATION_FILE_KEY: normalization_param_file_name,
        trainval_io.MIN_NORMALIZED_VALUE_KEY: min_normalized_value,
        trainval_io.MAX_NORMALIZED_VALUE_KEY: max_normalized_value,
        trainval_io.BINARIZE_TARGET_KEY: False,
        trainval_io.SAMPLING_FRACTIONS_KEY: class_to_sampling_fraction_dict,
        trainval_io.LOOP_ONCE_KEY: False,
        trainval_io.X_TRANSLATIONS_KEY: x_translations_pixels,
        trainval_io.Y_TRANSLATIONS_KEY: y_translations_pixels,
        trainval_io.ROTATION_ANGLES_KEY: ccw_rotation_angles_deg,
        trainval_io.NOISE_STDEV_KEY: noise_standard_deviation,
        trainval_io.NUM_NOISINGS_KEY: num_noisings,
        trainval_io.FLIP_X_KEY: flip_in_x,
        trainval_io.FLIP_Y_KEY: flip_in_y
    }

    print('Writing metadata to: "{0:s}"...'.format(model_metafile_name))
    cnn.write_model_metadata(
        pickle_file_name=model_metafile_name, metadata_dict=metadata_dict,
        training_option_dict=training_option_dict,
        list_of_layer_operation_dicts=list_of_layer_operation_dicts)

    cnn.train_cnn_gridrad_2d_reduced(
        model_object=model_object, model_file_name=output_model_file_name,
        history_file_name=history_file_name,
        tensorboard_dir_name=tensorboard_dir_name,
        num_epochs=num_epochs,
        num_training_batches_per_epoch=num_training_batches_per_epoch,
        training_option_dict=training_option_dict,
        list_of_layer_operation_dicts=list_of_layer_operation_dicts,
        monitor_string=monitor_string,
        weight_loss_function=weight_loss_function,
        num_validation_batches_per_epoch=num_validation_batches_per_epoch,
        validation_file_names=validation_file_names,
        first_validn_time_unix_sec=first_validation_time_unix_sec,
        last_validn_time_unix_sec=last_validation_time_unix_sec,
        num_examples_per_validn_batch=num_examples_per_validn_batch)


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        input_model_file_name=getattr(
            INPUT_ARG_OBJECT, dl_helper.INPUT_MODEL_FILE_ARG_NAME),
        radar_field_name_by_channel=getattr(
            INPUT_ARG_OBJECT, RADAR_FIELDS_ARG_NAME),
        layer_op_name_by_channel=getattr(
            INPUT_ARG_OBJECT, LAYER_OPERATIONS_ARG_NAME),
        min_height_by_channel_m_agl=numpy.array(
            getattr(INPUT_ARG_OBJECT, MIN_HEIGHTS_ARG_NAME), dtype=int
        ),
        max_height_by_channel_m_agl=numpy.array(
            getattr(INPUT_ARG_OBJECT, MAX_HEIGHTS_ARG_NAME), dtype=int
        ),
        sounding_field_names=getattr(
            INPUT_ARG_OBJECT, dl_helper.SOUNDING_FIELDS_ARG_NAME),
        normalization_type_string=getattr(
            INPUT_ARG_OBJECT, dl_helper.NORMALIZATION_TYPE_ARG_NAME),
        normalization_param_file_name=getattr(
            INPUT_ARG_OBJECT, dl_helper.NORMALIZATION_FILE_ARG_NAME),
        min_normalized_value=getattr(
            INPUT_ARG_OBJECT, dl_helper.MIN_NORM_VALUE_ARG_NAME),
        max_normalized_value=getattr(
            INPUT_ARG_OBJECT, dl_helper.MAX_NORM_VALUE_ARG_NAME),
        target_name=getattr(INPUT_ARG_OBJECT, dl_helper.TARGET_NAME_ARG_NAME),
        shuffle_target=bool(getattr(
            INPUT_ARG_OBJECT, dl_helper.SHUFFLE_TARGET_ARG_NAME)),
        downsampling_classes=numpy.array(
            getattr(INPUT_ARG_OBJECT, dl_helper.DOWNSAMPLING_CLASSES_ARG_NAME),
            dtype=int
        ),
        downsampling_fractions=numpy.array(
            getattr(INPUT_ARG_OBJECT,
                    dl_helper.DOWNSAMPLING_FRACTIONS_ARG_NAME),
            dtype=float
        ),
        monitor_string=getattr(INPUT_ARG_OBJECT, dl_helper.MONITOR_ARG_NAME),
        weight_loss_function=bool(getattr(
            INPUT_ARG_OBJECT, dl_helper.WEIGHT_LOSS_ARG_NAME
        )),
        x_translations_pixels=numpy.array(
            getattr(INPUT_ARG_OBJECT, dl_helper.X_TRANSLATIONS_ARG_NAME),
            dtype=int),
        y_translations_pixels=numpy.array(
            getattr(INPUT_ARG_OBJECT, dl_helper.Y_TRANSLATIONS_ARG_NAME),
            dtype=int),
        ccw_rotation_angles_deg=numpy.array(
            getattr(INPUT_ARG_OBJECT, dl_helper.ROTATION_ANGLES_ARG_NAME),
            dtype=float),
        noise_standard_deviation=getattr(
            INPUT_ARG_OBJECT, dl_helper.NOISE_STDEV_ARG_NAME),
        num_noisings=getattr(INPUT_ARG_OBJECT, dl_helper.NUM_NOISINGS_ARG_NAME),
        flip_in_x=bool(getattr(INPUT_ARG_OBJECT, dl_helper.FLIP_X_ARG_NAME)),
        flip_in_y=bool(getattr(INPUT_ARG_OBJECT, dl_helper.FLIP_Y_ARG_NAME)),
        top_training_dir_name=getattr(
            INPUT_ARG_OBJECT, dl_helper.TRAINING_DIR_ARG_NAME),
        first_training_time_string=getattr(
            INPUT_ARG_OBJECT, dl_helper.FIRST_TRAINING_TIME_ARG_NAME),
        last_training_time_string=getattr(
            INPUT_ARG_OBJECT, dl_helper.LAST_TRAINING_TIME_ARG_NAME),
        num_examples_per_train_batch=getattr(
            INPUT_ARG_OBJECT, dl_helper.NUM_EX_PER_TRAIN_ARG_NAME),
        top_validation_dir_name=getattr(
            INPUT_ARG_OBJECT, dl_helper.VALIDATION_DIR_ARG_NAME),
        first_validation_time_string=getattr(
            INPUT_ARG_OBJECT, dl_helper.FIRST_VALIDATION_TIME_ARG_NAME),
        last_validation_time_string=getattr(
            INPUT_ARG_OBJECT, dl_helper.LAST_VALIDATION_TIME_ARG_NAME),
        num_examples_per_validn_batch=getattr(
            INPUT_ARG_OBJECT, dl_helper.NUM_EX_PER_VALIDN_ARG_NAME),
        num_epochs=getattr(INPUT_ARG_OBJECT, dl_helper.NUM_EPOCHS_ARG_NAME),
        num_training_batches_per_epoch=getattr(
            INPUT_ARG_OBJECT, dl_helper.NUM_TRAINING_BATCHES_ARG_NAME),
        num_validation_batches_per_epoch=getattr(
            INPUT_ARG_OBJECT, dl_helper.NUM_VALIDATION_BATCHES_ARG_NAME),
        output_dir_name=getattr(INPUT_ARG_OBJECT, dl_helper.OUTPUT_DIR_ARG_NAME)
    )
