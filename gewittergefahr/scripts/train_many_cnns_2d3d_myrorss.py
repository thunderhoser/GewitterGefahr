"""Trains many CNNs with 2-D and 3-D MYRORSS images.

The 2-D images contain azimuthal shear, and the 3-D images contain reflectivity.
"""

import os
import pickle
import argparse
import traceback
from multiprocessing import Pool, Manager
import numpy
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import soundings
from gewittergefahr.gg_utils import file_system_utils

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

NUM_GPU_PER_NODE = 8
TIME_FORMAT = '%Y-%m-%d-%H%M%S'

FIRST_BATCH_NUMBER = 0
LAST_BATCH_NUMBER = int(1e12)
REFLECTIVITY_HEIGHTS_M_AGL = numpy.linspace(1000, 12000, num=12, dtype=int)
SOUNDING_HEIGHTS_M_AGL = soundings.DEFAULT_HEIGHT_LEVELS_M_AGL + 0

ARGUMENT_FILES_ARG_NAME = 'argument_file_names'
ARGUMENT_FILES_HELP_STRING = (
    '1-D list of paths to input files, each containing a dictionary of '
    'arguments for the single-CNN script train_cnn_2d3d_myrorss.py.  Each file '
    'should be a Pickle file, containing only said dictionary.')

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + ARGUMENT_FILES_ARG_NAME, type=str, nargs='+', required=True,
    help=ARGUMENT_FILES_HELP_STRING)


def _write_metadata_one_cnn(model_object, argument_dict):
    """Writes metadata for one CNN to file.

    :param model_object: Untrained CNN (instance of `keras.models.Model` or
        `keras.models.Sequential`).
    :param argument_dict: See doc for `_train_one_cnn`.
    :return: metadata_dict: See doc for `cnn.write_model_metadata`.
    :return: training_option_dict: Same.
    """

    from gewittergefahr.deep_learning import cnn
    from gewittergefahr.deep_learning import input_examples
    from gewittergefahr.deep_learning import \
        training_validation_io as trainval_io
    from gewittergefahr.scripts import deep_learning_helper as dl_helper

    # Read input args.
    sounding_field_names = argument_dict[dl_helper.SOUNDING_FIELDS_ARG_NAME]
    if sounding_field_names[0] in ['', 'None']:
        sounding_field_names = None

    normalization_type_string = argument_dict[
        dl_helper.NORMALIZATION_TYPE_ARG_NAME]
    normalization_file_name = argument_dict[
        dl_helper.NORMALIZATION_FILE_ARG_NAME]
    min_normalized_value = argument_dict[dl_helper.MIN_NORM_VALUE_ARG_NAME]
    max_normalized_value = argument_dict[dl_helper.MAX_NORM_VALUE_ARG_NAME]

    target_name = argument_dict[dl_helper.TARGET_NAME_ARG_NAME]
    downsampling_classes = numpy.array(
        argument_dict[dl_helper.DOWNSAMPLING_CLASSES_ARG_NAME],
        dtype=int
    )
    downsampling_fractions = numpy.array(
        argument_dict[dl_helper.DOWNSAMPLING_FRACTIONS_ARG_NAME],
        dtype=float
    )

    monitor_string = argument_dict[dl_helper.MONITOR_ARG_NAME]
    weight_loss_function = bool(argument_dict[dl_helper.WEIGHT_LOSS_ARG_NAME])

    x_translations_pixels = numpy.array(
        argument_dict[dl_helper.X_TRANSLATIONS_ARG_NAME], dtype=int
    )
    y_translations_pixels = numpy.array(
        argument_dict[dl_helper.Y_TRANSLATIONS_ARG_NAME], dtype=int
    )
    ccw_rotation_angles_deg = numpy.array(
        argument_dict[dl_helper.ROTATION_ANGLES_ARG_NAME], dtype=float
    )
    noise_standard_deviation = argument_dict[dl_helper.NOISE_STDEV_ARG_NAME]
    num_noisings = argument_dict[dl_helper.NUM_NOISINGS_ARG_NAME]
    flip_in_x = bool(argument_dict[dl_helper.FLIP_X_ARG_NAME])
    flip_in_y = bool(argument_dict[dl_helper.FLIP_Y_ARG_NAME])

    top_training_dir_name = argument_dict[dl_helper.TRAINING_DIR_ARG_NAME]
    first_training_time_string = argument_dict[
        dl_helper.FIRST_TRAINING_TIME_ARG_NAME]
    last_training_time_string = argument_dict[
        dl_helper.LAST_TRAINING_TIME_ARG_NAME]
    num_examples_per_train_batch = argument_dict[
        dl_helper.NUM_EX_PER_TRAIN_ARG_NAME]

    top_validation_dir_name = argument_dict[
        dl_helper.VALIDATION_DIR_ARG_NAME]
    first_validation_time_string = argument_dict[
        dl_helper.FIRST_VALIDATION_TIME_ARG_NAME]
    last_validation_time_string = argument_dict[
        dl_helper.LAST_VALIDATION_TIME_ARG_NAME]
    num_examples_per_validn_batch = argument_dict[
        dl_helper.NUM_EX_PER_VALIDN_ARG_NAME]

    num_epochs = argument_dict[dl_helper.NUM_EPOCHS_ARG_NAME]
    num_training_batches_per_epoch = argument_dict[
        dl_helper.NUM_TRAINING_BATCHES_ARG_NAME]
    num_validation_batches_per_epoch = argument_dict[
        dl_helper.NUM_VALIDATION_BATCHES_ARG_NAME]
    output_dir_name = argument_dict[dl_helper.OUTPUT_DIR_ARG_NAME]

    # Process input args.
    first_training_time_unix_sec = time_conversion.string_to_unix_sec(
        first_training_time_string, TIME_FORMAT)
    last_training_time_unix_sec = time_conversion.string_to_unix_sec(
        last_training_time_string, TIME_FORMAT)

    first_validation_time_unix_sec = time_conversion.string_to_unix_sec(
        first_validation_time_string, TIME_FORMAT)
    last_validation_time_unix_sec = time_conversion.string_to_unix_sec(
        last_validation_time_string, TIME_FORMAT)

    if len(downsampling_classes) > 1:
        downsampling_dict = dict(list(zip(
            downsampling_classes, downsampling_fractions
        )))
    else:
        downsampling_dict = None

    translate_flag = (
        len(x_translations_pixels) > 1
        or x_translations_pixels[0] != 0 or y_translations_pixels[0] != 0
    )

    if not translate_flag:
        x_translations_pixels = None
        y_translations_pixels = None

    if len(ccw_rotation_angles_deg) == 1 and ccw_rotation_angles_deg[0] == 0:
        ccw_rotation_angles_deg = None

    if num_noisings <= 0:
        num_noisings = 0
        noise_standard_deviation = None

    # Find training and validation files.
    training_file_names = input_examples.find_many_example_files(
        top_directory_name=top_training_dir_name, shuffled=True,
        first_batch_number=FIRST_BATCH_NUMBER,
        last_batch_number=LAST_BATCH_NUMBER,
        raise_error_if_any_missing=False)

    validation_file_names = input_examples.find_many_example_files(
        top_directory_name=top_validation_dir_name, shuffled=True,
        first_batch_number=FIRST_BATCH_NUMBER,
        last_batch_number=LAST_BATCH_NUMBER,
        raise_error_if_any_missing=False)

    # Write metadata.
    metadata_dict = {
        cnn.NUM_EPOCHS_KEY: num_epochs,
        cnn.NUM_TRAINING_BATCHES_KEY: num_training_batches_per_epoch,
        cnn.NUM_VALIDATION_BATCHES_KEY: num_validation_batches_per_epoch,
        cnn.MONITOR_STRING_KEY: monitor_string,
        cnn.WEIGHT_LOSS_FUNCTION_KEY: weight_loss_function,
        cnn.CONV_2D3D_KEY: True,
        cnn.VALIDATION_FILES_KEY: validation_file_names,
        cnn.FIRST_VALIDN_TIME_KEY: first_validation_time_unix_sec,
        cnn.LAST_VALIDN_TIME_KEY: last_validation_time_unix_sec,
        cnn.NUM_EX_PER_VALIDN_BATCH_KEY: num_examples_per_validn_batch
    }

    if isinstance(model_object.input, list):
        list_of_input_tensors = model_object.input
    else:
        list_of_input_tensors = [model_object.input]

    upsample_refl = len(list_of_input_tensors) == 2
    num_grid_rows = list_of_input_tensors[0].get_shape().as_list()[1]
    num_grid_columns = list_of_input_tensors[0].get_shape().as_list()[2]

    if upsample_refl:
        num_grid_rows = int(numpy.round(num_grid_rows / 2))
        num_grid_columns = int(numpy.round(num_grid_columns / 2))

    training_option_dict = {
        trainval_io.EXAMPLE_FILES_KEY: training_file_names,
        trainval_io.TARGET_NAME_KEY: target_name,
        trainval_io.FIRST_STORM_TIME_KEY: first_training_time_unix_sec,
        trainval_io.LAST_STORM_TIME_KEY: last_training_time_unix_sec,
        trainval_io.NUM_EXAMPLES_PER_BATCH_KEY: num_examples_per_train_batch,
        trainval_io.RADAR_FIELDS_KEY:
            input_examples.AZIMUTHAL_SHEAR_FIELD_NAMES,
        trainval_io.RADAR_HEIGHTS_KEY: REFLECTIVITY_HEIGHTS_M_AGL,
        trainval_io.SOUNDING_FIELDS_KEY: sounding_field_names,
        trainval_io.SOUNDING_HEIGHTS_KEY: SOUNDING_HEIGHTS_M_AGL,
        trainval_io.NUM_ROWS_KEY: num_grid_rows,
        trainval_io.NUM_COLUMNS_KEY: num_grid_columns,
        trainval_io.NORMALIZATION_TYPE_KEY: normalization_type_string,
        trainval_io.NORMALIZATION_FILE_KEY: normalization_file_name,
        trainval_io.MIN_NORMALIZED_VALUE_KEY: min_normalized_value,
        trainval_io.MAX_NORMALIZED_VALUE_KEY: max_normalized_value,
        trainval_io.BINARIZE_TARGET_KEY: False,
        trainval_io.SAMPLING_FRACTIONS_KEY: downsampling_dict,
        trainval_io.LOOP_ONCE_KEY: False,
        trainval_io.X_TRANSLATIONS_KEY: x_translations_pixels,
        trainval_io.Y_TRANSLATIONS_KEY: y_translations_pixels,
        trainval_io.ROTATION_ANGLES_KEY: ccw_rotation_angles_deg,
        trainval_io.NOISE_STDEV_KEY: noise_standard_deviation,
        trainval_io.NUM_NOISINGS_KEY: num_noisings,
        trainval_io.FLIP_X_KEY: flip_in_x,
        trainval_io.FLIP_Y_KEY: flip_in_y,
        trainval_io.UPSAMPLE_REFLECTIVITY_KEY: upsample_refl
    }

    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=output_dir_name)
    metafile_name = '{0:s}/model_metadata.p'.format(output_dir_name)

    print('Writing metadata to: "{0:s}"...'.format(metafile_name))
    cnn.write_model_metadata(
        pickle_file_name=metafile_name, metadata_dict=metadata_dict,
        training_option_dict=training_option_dict)

    return metadata_dict, training_option_dict


def _train_one_cnn(gpu_queue, argument_dict):
    """Trains single CNN with 2-D and 3-D MYRORSS images.

    :param gpu_queue: GPU queue (instance of `multiprocessing.Manager.Queue`).
    :param argument_dict: Dictionary of CNN arguments, where each key is an
        input arg to the script train_cnn_2d3d_myrorss.py.
    """

    import keras
    from keras import backend as K
    import tensorflow
    from gewittergefahr.deep_learning import cnn
    from gewittergefahr.deep_learning import cnn_setup
    from gewittergefahr.scripts import deep_learning_helper as dl_helper

    gpu_index = -1

    try:
        # Deal with GPU business.
        gpu_index = int(gpu_queue.get())
        os.environ['CUDA_VISIBLE_DEVICES'] = '{0:d}'.format(gpu_index)

        session_object = tensorflow.Session(
            config=tensorflow.ConfigProto(
                intra_op_parallelism_threads=7, inter_op_parallelism_threads=7,
                allow_soft_placement=False, log_device_placement=False,
                gpu_options=tensorflow.GPUOptions(allow_growth=True)
            )
        )

        K.set_session(session_object)

        # Read untrained model.
        untrained_model_file_name = argument_dict[
            dl_helper.INPUT_MODEL_FILE_ARG_NAME]

        with tensorflow.device('/gpu:0'):
            print('Reading untrained model from: "{0:s}"...'.format(
                untrained_model_file_name
            ))
            model_object = cnn.read_model(untrained_model_file_name)

        model_object.compile(
            loss=keras.losses.binary_crossentropy,
            optimizer=keras.optimizers.Adam(),
            metrics=cnn_setup.DEFAULT_METRIC_FUNCTION_LIST)

        print(SEPARATOR_STRING)
        model_object.summary()
        print(SEPARATOR_STRING)

        # Write metadata.
        metadata_dict, training_option_dict = _write_metadata_one_cnn(
            model_object=model_object, argument_dict=argument_dict)

        print('Training CNN on GPU {0:d}...'.format(gpu_index))
        print(SEPARATOR_STRING)

        # Train CNN.
        output_dir_name = argument_dict[dl_helper.OUTPUT_DIR_ARG_NAME]
        output_model_file_name = '{0:s}/model.h5'.format(output_dir_name)
        history_file_name = '{0:s}/model_history.csv'.format(output_dir_name)
        tensorboard_dir_name = '{0:s}/tensorboard'.format(output_dir_name)

        cnn.train_cnn_2d3d_myrorss(
            model_object=model_object, model_file_name=output_model_file_name,
            history_file_name=history_file_name,
            tensorboard_dir_name=tensorboard_dir_name,
            num_epochs=metadata_dict[cnn.NUM_EPOCHS_KEY],
            num_training_batches_per_epoch=metadata_dict[
                cnn.NUM_TRAINING_BATCHES_KEY],
            training_option_dict=training_option_dict,
            monitor_string=metadata_dict[cnn.MONITOR_STRING_KEY],
            weight_loss_function=metadata_dict[cnn.WEIGHT_LOSS_FUNCTION_KEY],
            num_validation_batches_per_epoch=metadata_dict[
                cnn.NUM_VALIDATION_BATCHES_KEY],
            validation_file_names=metadata_dict[cnn.VALIDATION_FILES_KEY],
            first_validn_time_unix_sec=metadata_dict[cnn.FIRST_VALIDN_TIME_KEY],
            last_validn_time_unix_sec=metadata_dict[cnn.LAST_VALIDN_TIME_KEY],
            num_examples_per_validn_batch=metadata_dict[
                cnn.NUM_EX_PER_VALIDN_BATCH_KEY]
        )

        session_object.close()
        del session_object
        gpu_queue.put(gpu_index)

    except Exception as this_exception:
        if gpu_index >= 0:
            gpu_queue.put(gpu_index)

        print(traceback.format_exc())
        raise this_exception


def _run(argument_file_names):
    """Trains many CNNs with 2-D and 3-D MYRORSS images.

    This is effectively the main method.

    :param argument_file_names: See documentation at top of file.
    """

    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([
        '{0:d}'.format(j) for j in range(NUM_GPU_PER_NODE)
    ])

    gpu_manager = Manager()
    gpu_queue = gpu_manager.Queue()
    gpu_pool = Pool(NUM_GPU_PER_NODE, maxtasksperchild=1)

    for j in range(NUM_GPU_PER_NODE):
        gpu_queue.put(j)

    for this_arg_file_name in argument_file_names:
        print('Reading single-CNN input args from: "{0:s}"...'.format(
            this_arg_file_name
        ))

        this_file_handle = open(this_arg_file_name, 'rb')
        this_argument_dict = pickle.load(this_file_handle)
        this_file_handle.close()

        gpu_pool.apply_async(
            func=_train_one_cnn, args=(gpu_queue, this_argument_dict)
        )

    gpu_pool.close()
    gpu_pool.join()

    del gpu_pool
    del gpu_queue
    del gpu_manager


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        argument_file_names=getattr(INPUT_ARG_OBJECT, ARGUMENT_FILES_ARG_NAME)
    )
