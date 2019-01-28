"""Trains upconvnet."""

import os.path
import argparse
import keras
from keras import backend as K
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.deep_learning import cnn
from gewittergefahr.deep_learning import upconvnet
from gewittergefahr.deep_learning import input_examples

K.set_session(K.tf.Session(config=K.tf.ConfigProto(
    intra_op_parallelism_threads=7, inter_op_parallelism_threads=7
)))

TIME_FORMAT = '%Y-%m-%d-%H%M%S'
SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

FIRST_BATCH_NUMBER = 0
LAST_BATCH_NUMBER = int(1e12)

CNN_FILE_ARG_NAME = 'input_cnn_file_name'
UPCONVNET_FILE_ARG_NAME = 'input_upconvnet_file_name'
FEATURE_LAYER_ARG_NAME = 'cnn_feature_layer_name'
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
OUTPUT_FILE_ARG_NAME = 'output_model_file_name'

CNN_FILE_HELP_STRING = (
    'Path to file with trained CNN (will be read by `cnn.read_model`).  The CNN'
    ' will be used as the encoder, to transform images into feature vectors.')

UPCONVNET_FILE_HELP_STRING = (
    'Path to file with upconvnet (may be trained or untrained).  This file will'
    ' be read by `cnn.read_model`, and its architecture will be copied to train'
    ' a new upconvnet.')

FEATURE_LAYER_HELP_STRING = (
    'Name of feature-generating layer in CNN.  For each input image, the '
    'feature vector will be the output from this layer.')

TRAINING_DIR_HELP_STRING = (
    'Name of top-level directory with training images (inputs to CNN).  Files '
    'therein will be found by `input_examples.find_example_file` and read by '
    '`input_examples.read_example_file`.')

TRAINING_TIME_HELP_STRING = (
    'Training time.  Only examples in the period `{0:s}`...`{1:s}` will be used'
    ' for training.'
).format(FIRST_TRAINING_TIME_ARG_NAME, LAST_TRAINING_TIME_ARG_NAME)

VALIDATION_DIR_HELP_STRING = (
    'Name of top-level directory with validation images (inputs to CNN).  Files'
    ' therein will be found by `input_examples.find_example_file` and read by '
    '`input_examples.read_example_file`.')

VALIDATION_TIME_HELP_STRING = (
    'Validation time.  Only examples in the period `{0:s}`...`{1:s}` will be '
    'used for validation.'
).format(FIRST_VALIDATION_TIME_ARG_NAME, LAST_VALIDATION_TIME_ARG_NAME)

NUM_EX_PER_BATCH_HELP_STRING = (
    'Number of examples in each training or validation batch.')

NUM_EPOCHS_HELP_STRING = 'Number of epochs.'

NUM_TRAINING_BATCHES_HELP_STRING = (
    'Number of training batches furnished to model in each epoch.')

NUM_VALIDATION_BATCHES_HELP_STRING = (
    'Number of validation batches furnished to model in each epoch.')

OUTPUT_FILE_HELP_STRING = (
    'Path to output file.  The trained upconvnet will be saved here as an HDF5 '
    'file (extension ".h5" is recommended but not enforced).')

DEFAULT_NUM_EXAMPLES_PER_BATCH = 1024
DEFAULT_NUM_EPOCHS = 100
DEFAULT_NUM_TRAINING_BATCHES = 32
DEFAULT_NUM_VALIDATION_BATCHES = 16

DEFAULT_FIRST_TRAINING_TIME_STRING = '2011-01-01-000000'
DEFAULT_LAST_TRAINING_TIME_STRING = '2013-12-31-120000'
DEFAULT_FIRST_VALIDN_TIME_STRING = '2014-01-01-120000'
DEFAULT_LAST_VALIDN_TIME_STRING = '2015-12-31-120000'

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + CNN_FILE_ARG_NAME, type=str, required=True,
    help=CNN_FILE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + UPCONVNET_FILE_ARG_NAME, type=str, required=True,
    help=UPCONVNET_FILE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + FEATURE_LAYER_ARG_NAME, type=str, required=True,
    help=FEATURE_LAYER_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + TRAINING_DIR_ARG_NAME, type=str, required=True,
    help=TRAINING_DIR_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + FIRST_TRAINING_TIME_ARG_NAME, type=str, required=False,
    default=DEFAULT_FIRST_TRAINING_TIME_STRING, help=TRAINING_TIME_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + LAST_TRAINING_TIME_ARG_NAME, type=str, required=False,
    default=DEFAULT_LAST_TRAINING_TIME_STRING, help=TRAINING_TIME_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + VALIDATION_DIR_ARG_NAME, type=str, required=True,
    help=VALIDATION_DIR_HELP_STRING)

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
    default=DEFAULT_NUM_TRAINING_BATCHES, help=NUM_TRAINING_BATCHES_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + NUM_VALIDATION_BATCHES_ARG_NAME, type=int, required=False,
    default=DEFAULT_NUM_VALIDATION_BATCHES,
    help=NUM_VALIDATION_BATCHES_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_FILE_ARG_NAME, type=str, required=True,
    help=OUTPUT_FILE_HELP_STRING)


def _run(input_cnn_file_name, input_upconvnet_file_name, cnn_feature_layer_name,
         top_training_dir_name, first_training_time_string,
         last_training_time_string, top_validation_dir_name,
         first_validation_time_string, last_validation_time_string,
         num_examples_per_batch, num_epochs, num_training_batches_per_epoch,
         num_validation_batches_per_epoch, output_model_file_name):
    """Trains upconvnet.

    This is effectively the main method.

    :param input_cnn_file_name: See documentation at top of file.
    :param input_upconvnet_file_name: Same.
    :param cnn_feature_layer_name: Same.
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
    :param output_model_file_name: Same.
    """

    # Find training and validation files.
    first_training_time_unix_sec = time_conversion.string_to_unix_sec(
        first_training_time_string, TIME_FORMAT)
    last_training_time_unix_sec = time_conversion.string_to_unix_sec(
        last_training_time_string, TIME_FORMAT)

    first_validation_time_unix_sec = time_conversion.string_to_unix_sec(
        first_validation_time_string, TIME_FORMAT)
    last_validation_time_unix_sec = time_conversion.string_to_unix_sec(
        last_validation_time_string, TIME_FORMAT)

    training_file_names = input_examples.find_many_example_files(
        top_directory_name=top_training_dir_name, shuffled=True,
        first_batch_number=FIRST_BATCH_NUMBER,
        last_batch_number=LAST_BATCH_NUMBER, raise_error_if_any_missing=False)

    validation_file_names = input_examples.find_many_example_files(
        top_directory_name=top_validation_dir_name, shuffled=True,
        first_batch_number=FIRST_BATCH_NUMBER,
        last_batch_number=LAST_BATCH_NUMBER, raise_error_if_any_missing=False)

    print 'Reading trained CNN from: "{0:s}"...'.format(input_cnn_file_name)
    cnn_model_object = cnn.read_model(input_cnn_file_name)
    cnn_metafile_name = '{0:s}/model_metadata.p'.format(
        os.path.split(input_cnn_file_name)[0]
    )

    print 'Reading CNN metadata from: "{0:s}"...'.format(cnn_metafile_name)
    cnn_metadata_dict = cnn.read_model_metadata(cnn_metafile_name)

    print 'Reading upconvnet architecture from: "{0:s}"...'.format(
        input_upconvnet_file_name)
    upconvnet_model_object = cnn.read_model(input_upconvnet_file_name)
    upconvnet_model_object = keras.models.clone_model(upconvnet_model_object)

    # TODO(thunderhoser): This is a HACK.
    upconvnet_model_object.compile(
        loss=keras.losses.mean_squared_error,
        optimizer=keras.optimizers.Adam()
    )

    print SEPARATOR_STRING
    upconvnet_model_object.summary()
    print SEPARATOR_STRING

    upconvnet_metafile_name = '{0:s}/model_metadata.p'.format(
        os.path.split(output_model_file_name)[0]
    )

    print 'Writing upconvnet metadata to: "{0:s}"...'.format(
        upconvnet_metafile_name)

    upconvnet.write_model_metadata(
        cnn_file_name=input_cnn_file_name,
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

    print SEPARATOR_STRING

    upconvnet.train_upconvnet(
        upconvnet_model_object=upconvnet_model_object,
        output_model_file_name=output_model_file_name,
        cnn_model_object=cnn_model_object,
        cnn_feature_layer_name=cnn_feature_layer_name,
        cnn_metadata_dict=cnn_metadata_dict, num_epochs=num_epochs,
        num_examples_per_batch=num_examples_per_batch,
        num_training_batches_per_epoch=num_training_batches_per_epoch,
        training_example_file_names=training_file_names,
        first_training_time_unix_sec=first_training_time_unix_sec,
        last_training_time_unix_sec=last_training_time_unix_sec,
        num_validation_batches_per_epoch=num_validation_batches_per_epoch,
        validation_example_file_names=validation_file_names,
        first_validation_time_unix_sec=first_validation_time_unix_sec,
        last_validation_time_unix_sec=last_validation_time_unix_sec)


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        input_cnn_file_name=getattr(INPUT_ARG_OBJECT, CNN_FILE_ARG_NAME),
        input_upconvnet_file_name=getattr(
            INPUT_ARG_OBJECT, UPCONVNET_FILE_ARG_NAME),
        cnn_feature_layer_name=getattr(
            INPUT_ARG_OBJECT, FEATURE_LAYER_ARG_NAME),
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
        output_model_file_name=getattr(INPUT_ARG_OBJECT, OUTPUT_FILE_ARG_NAME)
    )
