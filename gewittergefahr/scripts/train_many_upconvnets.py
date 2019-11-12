"""Trains many upconvolutional networks."""

import os
import pickle
import argparse
import traceback
from multiprocessing import Pool, Manager
from gewittergefahr.gg_utils import time_conversion

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

NUM_GPU_PER_NODE = 8
TIME_FORMAT = '%Y-%m-%d-%H%M%S'

FIRST_BATCH_NUMBER = 0
LAST_BATCH_NUMBER = int(1e12)

ARGUMENT_FILES_ARG_NAME = 'argument_file_names'
ARGUMENT_FILES_HELP_STRING = (
    '1-D list of paths to input files, each containing a dictionary of '
    'arguments for the single-model script train_upconvnet.py.  Each file '
    'should be a Pickle file, containing only said dictionary.')

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + ARGUMENT_FILES_ARG_NAME, type=str, nargs='+', required=True,
    help=ARGUMENT_FILES_HELP_STRING)


def _write_metadata_one_model(argument_dict):
    """Writes metadata for one upconvnet to file.

    :param argument_dict: See doc for `_train_one_upconvnet`.
    :return: metadata_dict: See doc for `upconvnet.write_model_metadata`.
    """

    from gewittergefahr.deep_learning import cnn
    from gewittergefahr.deep_learning import upconvnet
    from gewittergefahr.deep_learning import input_examples
    from gewittergefahr.scripts import train_upconvnet

    # Read input args.
    cnn_file_name = argument_dict[train_upconvnet.CNN_FILE_ARG_NAME]
    cnn_feature_layer_name = argument_dict[
        train_upconvnet.FEATURE_LAYER_ARG_NAME]

    top_training_dir_name = argument_dict[train_upconvnet.TRAINING_DIR_ARG_NAME]
    first_training_time_string = argument_dict[
        train_upconvnet.FIRST_TRAINING_TIME_ARG_NAME
    ]
    last_training_time_string = argument_dict[
        train_upconvnet.LAST_TRAINING_TIME_ARG_NAME]

    top_validation_dir_name = argument_dict[
        train_upconvnet.VALIDATION_DIR_ARG_NAME
    ]
    first_validation_time_string = argument_dict[
        train_upconvnet.FIRST_VALIDATION_TIME_ARG_NAME
    ]
    last_validation_time_string = argument_dict[
        train_upconvnet.LAST_VALIDATION_TIME_ARG_NAME]

    num_examples_per_batch = argument_dict[
        train_upconvnet.NUM_EX_PER_BATCH_ARG_NAME
    ]
    num_epochs = argument_dict[train_upconvnet.NUM_EPOCHS_ARG_NAME]
    num_training_batches_per_epoch = argument_dict[
        train_upconvnet.NUM_TRAINING_BATCHES_ARG_NAME
    ]
    num_validation_batches_per_epoch = argument_dict[
        train_upconvnet.NUM_VALIDATION_BATCHES_ARG_NAME
    ]
    output_dir_name = argument_dict[train_upconvnet.OUTPUT_DIR_ARG_NAME]

    # Process input args.
    first_training_time_unix_sec = time_conversion.string_to_unix_sec(
        first_training_time_string, TIME_FORMAT)
    last_training_time_unix_sec = time_conversion.string_to_unix_sec(
        last_training_time_string, TIME_FORMAT)

    first_validation_time_unix_sec = time_conversion.string_to_unix_sec(
        first_validation_time_string, TIME_FORMAT)
    last_validation_time_unix_sec = time_conversion.string_to_unix_sec(
        last_validation_time_string, TIME_FORMAT)

    # Find training and validation files.
    training_file_names = input_examples.find_many_example_files(
        top_directory_name=top_training_dir_name, shuffled=True,
        first_batch_number=FIRST_BATCH_NUMBER,
        last_batch_number=LAST_BATCH_NUMBER, raise_error_if_any_missing=False)

    validation_file_names = input_examples.find_many_example_files(
        top_directory_name=top_validation_dir_name, shuffled=True,
        first_batch_number=FIRST_BATCH_NUMBER,
        last_batch_number=LAST_BATCH_NUMBER, raise_error_if_any_missing=False)

    # Write metadata.
    upconvnet_metafile_name = cnn.find_metafile(
        model_file_name='{0:s}/foo.h5'.format(output_dir_name),
        raise_error_if_missing=False
    )
    print('Writing upconvnet metadata to: "{0:s}"...'.format(
        upconvnet_metafile_name
    ))

    return upconvnet.write_model_metadata(
        cnn_file_name=cnn_file_name,
        cnn_feature_layer_name=cnn_feature_layer_name, num_epochs=num_epochs,
        num_examples_per_batch=num_examples_per_batch,
        num_training_batches_per_epoch=num_training_batches_per_epoch,
        training_example_file_names=training_file_names,
        first_training_time_unix_sec=first_training_time_unix_sec,
        last_training_time_unix_sec=last_training_time_unix_sec,
        num_validation_batches_per_epoch=num_validation_batches_per_epoch,
        validation_example_file_names=validation_file_names,
        first_validation_time_unix_sec=first_validation_time_unix_sec,
        last_validation_time_unix_sec=last_validation_time_unix_sec,
        pickle_file_name=upconvnet_metafile_name)


def _train_one_upconvnet(gpu_queue, argument_dict):
    """Trains one upconvolutional network.

    :param gpu_queue: GPU queue (instance of `multiprocessing.Manager.Queue`).
    :param argument_dict: Dictionary of CNN arguments, where each key is an
        input arg to the script train_upconvnet.py.
    """

    import keras
    from keras import backend as K
    import tensorflow
    from gewittergefahr.deep_learning import cnn
    from gewittergefahr.deep_learning import upconvnet
    from gewittergefahr.scripts import train_upconvnet

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

        # Write metadata.
        upconvnet_metadata_dict = _write_metadata_one_model(argument_dict)

        # Read trained CNN.
        cnn_file_name = argument_dict[train_upconvnet.CNN_FILE_ARG_NAME]
        cnn_metafile_name = cnn.find_metafile(cnn_file_name)

        print('Reading CNN metadata from: "{0:s}"...'.format(cnn_metafile_name))
        cnn_metadata_dict = cnn.read_model_metadata(cnn_metafile_name)

        upconvnet_template_file_name = argument_dict[
            train_upconvnet.UPCONVNET_FILE_ARG_NAME]

        with tensorflow.device('/gpu:0'):
            print('Reading trained CNN from: "{0:s}"...'.format(cnn_file_name))
            cnn_model_object = cnn.read_model(cnn_file_name)

            print('Reading upconvnet architecture from: "{0:s}"...'.format(
                upconvnet_template_file_name
            ))
            upconvnet_model_object = cnn.read_model(
                upconvnet_template_file_name)

        upconvnet_model_object.compile(
            loss=keras.losses.mean_squared_error,
            optimizer=keras.optimizers.Adam()
        )

        print(SEPARATOR_STRING)
        upconvnet_model_object.summary()
        print(SEPARATOR_STRING)

        # Train upconvnet.
        print('Training upconvnet on GPU {0:d}...'.format(gpu_index))
        print(SEPARATOR_STRING)

        upconvnet.train_upconvnet(
            upconvnet_model_object=upconvnet_model_object,
            output_dir_name=argument_dict[train_upconvnet.OUTPUT_DIR_ARG_NAME],
            cnn_model_object=cnn_model_object,
            cnn_metadata_dict=cnn_metadata_dict,
            cnn_feature_layer_name=
            upconvnet_metadata_dict[upconvnet.CNN_FEATURE_LAYER_KEY],
            num_epochs=upconvnet_metadata_dict[upconvnet.NUM_EPOCHS_KEY],
            num_examples_per_batch=
            upconvnet_metadata_dict[upconvnet.NUM_EXAMPLES_PER_BATCH_KEY],
            num_training_batches_per_epoch=
            upconvnet_metadata_dict[upconvnet.NUM_TRAINING_BATCHES_KEY],
            training_example_file_names=
            upconvnet_metadata_dict[upconvnet.TRAINING_FILES_KEY],
            first_training_time_unix_sec=
            upconvnet_metadata_dict[upconvnet.FIRST_TRAINING_TIME_KEY],
            last_training_time_unix_sec=
            upconvnet_metadata_dict[upconvnet.LAST_TRAINING_TIME_KEY],
            num_validation_batches_per_epoch=
            upconvnet_metadata_dict[upconvnet.NUM_VALIDATION_BATCHES_KEY],
            validation_example_file_names=
            upconvnet_metadata_dict[upconvnet.VALIDATION_FILES_KEY],
            first_validation_time_unix_sec=
            upconvnet_metadata_dict[upconvnet.FIRST_VALIDATION_TIME_KEY],
            last_validation_time_unix_sec=
            upconvnet_metadata_dict[upconvnet.LAST_VALIDATION_TIME_KEY]
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
    """Trains many upconvolutional networks.

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
        print('Reading single-model input args from: "{0:s}"...'.format(
            this_arg_file_name
        ))

        this_file_handle = open(this_arg_file_name, 'rb')
        this_argument_dict = pickle.load(this_file_handle)
        this_file_handle.close()

        gpu_pool.apply_async(
            func=_train_one_upconvnet, args=(gpu_queue, this_argument_dict)
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
