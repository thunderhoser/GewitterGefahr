"""Methods for training and applying a CNN (convolutional neural network).

--- NOTATION ---

In this module, the following letters will be used to denote matrix dimensions.

K = number of classes (possible values of target variable)
E = number of examples
M = number of pixel rows per image
N = number of pixel columns per image
D = number of pixel depths per image
C = number of channels (predictor variables) per image
"""

import os.path
import pickle
import numpy
import keras.losses
import keras.optimizers
from keras.models import Sequential, load_model
import keras.callbacks
from gewittergefahr.deep_learning import deep_learning_utils as dl_utils
from gewittergefahr.deep_learning import cnn_utils
from gewittergefahr.deep_learning import keras_metrics
from gewittergefahr.deep_learning import training_validation_io
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.gg_utils import error_checking

DEFAULT_NUM_INPUT_ROWS = 32
DEFAULT_NUM_INPUT_COLUMNS = 32
DEFAULT_NUM_INPUT_DEPTHS = 12

CUSTOM_OBJECT_DICT_FOR_READING_MODEL = {
    'accuracy': keras_metrics.accuracy,
    'binary_accuracy': keras_metrics.binary_accuracy,
    'binary_csi': keras_metrics.binary_csi,
    'binary_frequency_bias': keras_metrics.binary_frequency_bias,
    'binary_pod': keras_metrics.binary_pod,
    'binary_pofd': keras_metrics.binary_pofd,
    'binary_success_ratio': keras_metrics.binary_success_ratio,
    'binary_focn': keras_metrics.binary_focn
}

LIST_OF_METRIC_FUNCTIONS = [
    keras_metrics.accuracy, keras_metrics.binary_accuracy,
    keras_metrics.binary_csi, keras_metrics.binary_frequency_bias,
    keras_metrics.binary_pod, keras_metrics.binary_pofd,
    keras_metrics.binary_success_ratio, keras_metrics.binary_focn]

NUM_EPOCHS_KEY = 'num_epochs'
NUM_EXAMPLES_PER_BATCH_KEY = 'num_examples_per_batch'
NUM_EXAMPLES_PER_TIME_KEY = 'num_examples_per_time'
NUM_TRAINING_BATCHES_PER_EPOCH_KEY = 'num_training_batches_per_epoch'
FIRST_TRAINING_TIME_KEY = 'first_train_time_unix_sec'
LAST_TRAINING_TIME_KEY = 'last_train_time_unix_sec'
NUM_VALIDATION_BATCHES_PER_EPOCH_KEY = 'num_validation_batches_per_epoch'
FIRST_VALIDATION_TIME_KEY = 'first_validn_time_unix_sec'
LAST_VALIDATION_TIME_KEY = 'last_validn_time_unix_sec'
RADAR_SOURCE_KEY = 'radar_source'
RADAR_FIELD_NAMES_KEY = 'radar_field_names'
RADAR_HEIGHTS_KEY = 'radar_heights_m_asl'
REFLECTIVITY_HEIGHTS_KEY = 'reflectivity_heights_m_asl'
TARGET_NAME_KEY = 'target_name'
NORMALIZE_BY_BATCH_KEY = 'normalize_by_batch'
NORMALIZATION_DICT_KEY = 'normalization_dict'
PERCENTILE_OFFSET_KEY = 'percentile_offset_for_normalization'
CLASS_FRACTIONS_KEY = 'class_fractions'

MODEL_METADATA_KEYS = [
    NUM_EPOCHS_KEY, NUM_EXAMPLES_PER_BATCH_KEY, NUM_EXAMPLES_PER_TIME_KEY,
    NUM_TRAINING_BATCHES_PER_EPOCH_KEY, FIRST_TRAINING_TIME_KEY,
    LAST_TRAINING_TIME_KEY, NUM_VALIDATION_BATCHES_PER_EPOCH_KEY,
    FIRST_VALIDATION_TIME_KEY, LAST_VALIDATION_TIME_KEY, RADAR_SOURCE_KEY,
    RADAR_FIELD_NAMES_KEY, RADAR_HEIGHTS_KEY, REFLECTIVITY_HEIGHTS_KEY,
    TARGET_NAME_KEY, NORMALIZE_BY_BATCH_KEY, NORMALIZATION_DICT_KEY,
    PERCENTILE_OFFSET_KEY, CLASS_FRACTIONS_KEY]


# TODO(thunderhoser): should play with Adam optimizer, per DJ Gagne.


def _check_training_args(
        num_epochs, num_training_batches_per_epoch,
        num_validation_batches_per_epoch, weight_loss_function,
        class_fractions_to_sample, model_file_name, history_file_name,
        tensorboard_dir_name):
    """Error-checks input arguments for training method.

    :param num_epochs: Number of training epochs.
    :param num_training_batches_per_epoch: Number of training batches per epoch.
    :param num_validation_batches_per_epoch: Number of validation batches per
        epoch.
    :param weight_loss_function: Boolean flag.  If False, classes will be
        weighted equally in the loss function.  If True, classes will be
        weighted differently in the loss function (inversely proportional with
        `class_fractions_to_sample`).
    :param class_fractions_to_sample: See documentation for
        `training_validation_io.storm_image_generator_2d`.
    :param model_file_name: Path to output file (HDF5 format).  The model will
        be saved here after every epoch.
    :param history_file_name: Path to output file (CSV format).  Training
        history will be saved here after every epoch.
    :param tensorboard_dir_name: Path to output directory for TensorBoard log
        files.
    :return: class_weight_dict: See doc for
        `dl_utils.class_fractions_to_weights`.
    """

    error_checking.assert_is_integer(num_epochs)
    error_checking.assert_is_geq(num_epochs, 1)
    error_checking.assert_is_integer(num_training_batches_per_epoch)
    error_checking.assert_is_geq(num_training_batches_per_epoch, 1)
    if num_validation_batches_per_epoch is not None:
        error_checking.assert_is_integer(num_validation_batches_per_epoch)
        error_checking.assert_is_geq(num_validation_batches_per_epoch, 1)

    error_checking.assert_is_boolean(weight_loss_function)
    if weight_loss_function:
        class_weight_dict = dl_utils.class_fractions_to_weights(
            class_fractions_to_sample)
    else:
        class_weight_dict = None

    file_system_utils.mkdir_recursive_if_necessary(file_name=model_file_name)
    file_system_utils.mkdir_recursive_if_necessary(file_name=history_file_name)
    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=tensorboard_dir_name)

    return class_weight_dict


def get_mnist_architecture(num_classes, num_input_channels=3):
    """Creates CNN with architecture used in the following example.

    This is a 2-D CNN, which means that it performs 2-D convolution.

    https://github.com/keras-team/keras/blob/master/examples/mnist_cnn.py

    :param num_classes: Number of target classes.
    :param num_input_channels: Number of input channels (predictor variables).
    :return: model_object: Instance of `keras.models.Sequential` with the
        aforementioned architecture.
    """

    error_checking.assert_is_integer(num_classes)
    error_checking.assert_is_geq(num_classes, 2)

    model_object = Sequential()

    layer_object = cnn_utils.get_2d_conv_layer(
        num_output_filters=32, num_kernel_rows=3, num_kernel_columns=3,
        num_rows_per_stride=1, num_columns_per_stride=1,
        padding_type=cnn_utils.NO_PADDING_TYPE, activation_function='relu',
        is_first_layer=True, num_input_rows=DEFAULT_NUM_INPUT_ROWS,
        num_input_columns=DEFAULT_NUM_INPUT_COLUMNS,
        num_input_channels=num_input_channels)
    model_object.add(layer_object)

    layer_object = cnn_utils.get_2d_conv_layer(
        num_output_filters=64, num_kernel_rows=3, num_kernel_columns=3,
        num_rows_per_stride=1, num_columns_per_stride=1,
        padding_type=cnn_utils.NO_PADDING_TYPE, activation_function='relu')
    model_object.add(layer_object)

    layer_object = cnn_utils.get_2d_pooling_layer(
        num_rows_in_window=2, num_columns_in_window=2,
        pooling_type=cnn_utils.MAX_POOLING_TYPE, num_rows_per_stride=2,
        num_columns_per_stride=2)
    model_object.add(layer_object)

    layer_object = cnn_utils.get_dropout_layer(dropout_fraction=0.25)
    model_object.add(layer_object)

    layer_object = cnn_utils.get_flattening_layer()
    model_object.add(layer_object)

    layer_object = cnn_utils.get_fully_connected_layer(
        num_output_units=128, activation_function='relu')
    model_object.add(layer_object)

    layer_object = cnn_utils.get_dropout_layer(dropout_fraction=0.5)
    model_object.add(layer_object)

    layer_object = cnn_utils.get_fully_connected_layer(
        num_output_units=num_classes, activation_function='softmax')
    model_object.add(layer_object)

    model_object.compile(
        loss=keras.losses.categorical_crossentropy,
        optimizer=keras.optimizers.Adadelta(), metrics=LIST_OF_METRIC_FUNCTIONS)

    model_object.summary()
    return model_object


def get_2d_swirlnet_architecture(num_classes, num_input_channels=3):
    """Creates 2-D CNN (one that performs 2-D convolution).

    Architecture is similar to the following example:

    https://github.com/djgagne/swirlnet/blob/master/notebooks/
    deep_swirl_tutorial.ipynb

    :param num_classes: Number of target classes.
    :param num_input_channels: Number of input channels (predictor variables).
    :return: model_object: Instance of `keras.models.Sequential` with the
        aforementioned architecture.
    """

    error_checking.assert_is_integer(num_classes)
    error_checking.assert_is_geq(num_classes, 2)

    model_object = Sequential()
    regularizer_object = cnn_utils.get_weight_regularizer(
        l1_penalty=0, l2_penalty=0.01)

    # Input to this layer is E x 32 x 32 x C.
    layer_object = cnn_utils.get_2d_conv_layer(
        num_output_filters=16, num_kernel_rows=5, num_kernel_columns=5,
        num_rows_per_stride=1, num_columns_per_stride=1,
        padding_type=cnn_utils.YES_PADDING_TYPE,
        kernel_weight_regularizer=regularizer_object,
        activation_function='relu', is_first_layer=True,
        num_input_rows=DEFAULT_NUM_INPUT_ROWS,
        num_input_columns=DEFAULT_NUM_INPUT_COLUMNS,
        num_input_channels=num_input_channels)
    model_object.add(layer_object)

    # Input to this layer is E x 32 x 32 x 16.
    layer_object = cnn_utils.get_dropout_layer(dropout_fraction=0.1)
    model_object.add(layer_object)

    # Input to this layer is E x 32 x 32 x 16.
    layer_object = cnn_utils.get_2d_pooling_layer(
        num_rows_in_window=2, num_columns_in_window=2,
        pooling_type=cnn_utils.MEAN_POOLING_TYPE, num_rows_per_stride=2,
        num_columns_per_stride=2)
    model_object.add(layer_object)

    # Input to this layer is E x 16 x 16 x 16.
    layer_object = cnn_utils.get_2d_conv_layer(
        num_output_filters=32, num_kernel_rows=5, num_kernel_columns=5,
        num_rows_per_stride=1, num_columns_per_stride=1,
        padding_type=cnn_utils.YES_PADDING_TYPE,
        kernel_weight_regularizer=regularizer_object,
        activation_function='relu')
    model_object.add(layer_object)

    # Input to this layer is E x 16 x 16 x 32.
    layer_object = cnn_utils.get_dropout_layer(dropout_fraction=0.1)
    model_object.add(layer_object)

    # Input to this layer is E x 16 x 16 x 32.
    layer_object = cnn_utils.get_2d_pooling_layer(
        num_rows_in_window=2, num_columns_in_window=2,
        pooling_type=cnn_utils.MEAN_POOLING_TYPE, num_rows_per_stride=2,
        num_columns_per_stride=2)
    model_object.add(layer_object)

    # Input to this layer is E x 8 x 8 x 32.
    layer_object = cnn_utils.get_2d_conv_layer(
        num_output_filters=64, num_kernel_rows=3, num_kernel_columns=3,
        num_rows_per_stride=1, num_columns_per_stride=1,
        padding_type=cnn_utils.YES_PADDING_TYPE,
        kernel_weight_regularizer=regularizer_object,
        activation_function='relu')
    model_object.add(layer_object)

    # Input to this layer is E x 8 x 8 x 64.
    layer_object = cnn_utils.get_dropout_layer(dropout_fraction=0.1)
    model_object.add(layer_object)

    # Input to this layer is E x 8 x 8 x 64.
    layer_object = cnn_utils.get_2d_pooling_layer(
        num_rows_in_window=2, num_columns_in_window=2,
        pooling_type=cnn_utils.MEAN_POOLING_TYPE, num_rows_per_stride=2,
        num_columns_per_stride=2)
    model_object.add(layer_object)

    # Input to this layer is E x 4 x 4 x 64.
    layer_object = cnn_utils.get_flattening_layer()
    model_object.add(layer_object)

    # Input to this layer is length-1024.
    layer_object = cnn_utils.get_fully_connected_layer(
        num_output_units=num_classes, activation_function='softmax')
    model_object.add(layer_object)

    model_object.compile(
        loss=keras.losses.categorical_crossentropy,
        optimizer=keras.optimizers.Adadelta(), metrics=LIST_OF_METRIC_FUNCTIONS)

    model_object.summary()
    return model_object


def get_3d_swirlnet_architecture(num_classes, num_input_channels=3):
    """Creates 3-D CNN (one that performs 3-D convolution).

    Architecture is similar to the following example:

    https://github.com/djgagne/swirlnet/blob/master/notebooks/
    deep_swirl_tutorial.ipynb

    :param num_classes: Number of target classes.
    :param num_input_channels: Number of input channels (predictor variables).
    :return: model_object: Instance of `keras.models.Sequential` with the
        aforementioned architecture.
    """

    error_checking.assert_is_integer(num_classes)
    error_checking.assert_is_geq(num_classes, 2)

    model_object = Sequential()
    regularizer_object = cnn_utils.get_weight_regularizer(
        l1_penalty=0, l2_penalty=0.01)

    # Input to this layer is E x 32 x 32 x 12 x C.
    layer_object = cnn_utils.get_3d_conv_layer(
        num_output_filters=8*num_input_channels, num_kernel_rows=5,
        num_kernel_columns=5, num_kernel_depths=3, num_rows_per_stride=1,
        num_columns_per_stride=1, num_depths_per_stride=1,
        padding_type=cnn_utils.YES_PADDING_TYPE,
        kernel_weight_regularizer=regularizer_object,
        activation_function='relu', is_first_layer=True,
        num_input_rows=DEFAULT_NUM_INPUT_ROWS,
        num_input_columns=DEFAULT_NUM_INPUT_COLUMNS,
        num_input_depths=DEFAULT_NUM_INPUT_DEPTHS,
        num_input_channels=num_input_channels)
    model_object.add(layer_object)

    # Input to this layer is E x 32 x 32 x 12 x 8C.
    layer_object = cnn_utils.get_dropout_layer(dropout_fraction=0.1)
    model_object.add(layer_object)

    # Input to this layer is E x 32 x 32 x 12 x 8C.
    layer_object = cnn_utils.get_3d_pooling_layer(
        num_rows_in_window=2, num_columns_in_window=2, num_depths_in_window=2,
        pooling_type=cnn_utils.MEAN_POOLING_TYPE, num_rows_per_stride=2,
        num_columns_per_stride=2, num_depths_per_stride=2)
    model_object.add(layer_object)

    # Input to this layer is E x 16 x 16 x 6 x 8C.
    layer_object = cnn_utils.get_3d_conv_layer(
        num_output_filters=16*num_input_channels, num_kernel_rows=5,
        num_kernel_columns=5, num_kernel_depths=3, num_rows_per_stride=1,
        num_columns_per_stride=1, num_depths_per_stride=1,
        padding_type=cnn_utils.YES_PADDING_TYPE,
        kernel_weight_regularizer=regularizer_object,
        activation_function='relu')
    model_object.add(layer_object)

    # Input to this layer is E x 16 x 16 x 6 x 16C.
    layer_object = cnn_utils.get_dropout_layer(dropout_fraction=0.1)
    model_object.add(layer_object)

    # Input to this layer is E x 16 x 16 x 6 x 16C.
    layer_object = cnn_utils.get_3d_pooling_layer(
        num_rows_in_window=2, num_columns_in_window=2, num_depths_in_window=2,
        pooling_type=cnn_utils.MEAN_POOLING_TYPE, num_rows_per_stride=2,
        num_columns_per_stride=2, num_depths_per_stride=2)
    model_object.add(layer_object)

    # Input to this layer is E x 8 x 8 x 3 x 16C.
    layer_object = cnn_utils.get_3d_conv_layer(
        num_output_filters=32*num_input_channels, num_kernel_rows=3,
        num_kernel_columns=3, num_kernel_depths=3, num_rows_per_stride=1,
        num_columns_per_stride=1, num_depths_per_stride=1,
        padding_type=cnn_utils.YES_PADDING_TYPE,
        kernel_weight_regularizer=regularizer_object,
        activation_function='relu')
    model_object.add(layer_object)

    # Input to this layer is E x 8 x 8 x 3 x 32C.
    layer_object = cnn_utils.get_dropout_layer(dropout_fraction=0.1)
    model_object.add(layer_object)

    # Input to this layer is E x 8 x 8 x 3 x 32C.
    layer_object = cnn_utils.get_3d_pooling_layer(
        num_rows_in_window=2, num_columns_in_window=2, num_depths_in_window=3,
        pooling_type=cnn_utils.MEAN_POOLING_TYPE, num_rows_per_stride=2,
        num_columns_per_stride=2, num_depths_per_stride=3)
    model_object.add(layer_object)

    # Input to this layer is E x 4 x 4 x 1 x 32C.
    layer_object = cnn_utils.get_flattening_layer()
    model_object.add(layer_object)

    # Input to this layer is length-512C.
    layer_object = cnn_utils.get_fully_connected_layer(
        num_output_units=num_classes, activation_function='softmax')
    model_object.add(layer_object)

    model_object.compile(
        loss=keras.losses.categorical_crossentropy,
        optimizer=keras.optimizers.Adadelta(), metrics=LIST_OF_METRIC_FUNCTIONS)

    model_object.summary()
    return model_object


def read_model(hdf5_file_name):
    """Reads model from HDF5 file.

    :param hdf5_file_name: Path to input file.
    :return: model_object: Instance of `keras.models.Model`.
    """

    error_checking.assert_file_exists(hdf5_file_name)
    return load_model(
        hdf5_file_name, custom_objects=CUSTOM_OBJECT_DICT_FOR_READING_MODEL)


def write_model_metadata(
        num_epochs, num_examples_per_batch, num_examples_per_time,
        num_training_batches_per_epoch, first_train_time_unix_sec,
        last_train_time_unix_sec, num_validation_batches_per_epoch,
        first_validn_time_unix_sec, last_validn_time_unix_sec,
        radar_source, radar_field_names, radar_heights_m_asl,
        reflectivity_heights_m_asl, target_name, normalize_by_batch,
        normalization_dict, percentile_offset_for_normalization,
        class_fractions, pickle_file_name):
    """Writes metadata to Pickle file.

    :param num_epochs: See documentation for `train_2d_cnn`.
    :param num_examples_per_batch: Same.
    :param num_examples_per_time: Same.
    :param num_training_batches_per_epoch: Same.
    :param first_train_time_unix_sec: Same.
    :param last_train_time_unix_sec: Same.
    :param num_validation_batches_per_epoch: Same.
    :param first_validn_time_unix_sec: Same.
    :param last_validn_time_unix_sec: Same.
    :param radar_source: Same.
    :param radar_field_names: Same.
    :param radar_heights_m_asl: Same.
    :param reflectivity_heights_m_asl: Same.
    :param target_name: Same.
    :param normalize_by_batch: Same.
    :param normalization_dict: Same.
    :param percentile_offset_for_normalization: Same.
    :param class_fractions: Same.
    :param pickle_file_name: Path to output file.
    """

    model_metadata_dict = {
        NUM_EPOCHS_KEY: num_epochs,
        NUM_EXAMPLES_PER_BATCH_KEY: num_examples_per_batch,
        NUM_EXAMPLES_PER_TIME_KEY: num_examples_per_time,
        NUM_TRAINING_BATCHES_PER_EPOCH_KEY: num_training_batches_per_epoch,
        FIRST_TRAINING_TIME_KEY: first_train_time_unix_sec,
        LAST_TRAINING_TIME_KEY: last_train_time_unix_sec,
        NUM_VALIDATION_BATCHES_PER_EPOCH_KEY: num_validation_batches_per_epoch,
        FIRST_VALIDATION_TIME_KEY: first_validn_time_unix_sec,
        LAST_VALIDATION_TIME_KEY: last_validn_time_unix_sec,
        RADAR_SOURCE_KEY: radar_source,
        RADAR_FIELD_NAMES_KEY: radar_field_names,
        RADAR_HEIGHTS_KEY: radar_heights_m_asl,
        REFLECTIVITY_HEIGHTS_KEY: reflectivity_heights_m_asl,
        TARGET_NAME_KEY: target_name,
        NORMALIZE_BY_BATCH_KEY: normalize_by_batch,
        NORMALIZATION_DICT_KEY: normalization_dict,
        PERCENTILE_OFFSET_KEY: percentile_offset_for_normalization,
        CLASS_FRACTIONS_KEY: class_fractions
    }

    file_system_utils.mkdir_recursive_if_necessary(file_name=pickle_file_name)
    pickle_file_handle = open(pickle_file_name, 'wb')
    pickle.dump(model_metadata_dict, pickle_file_handle)
    pickle_file_handle.close()


def read_model_metadata(pickle_file_name):
    """Reads metadata from Pickle file.

    :param pickle_file_name: Path to input file.
    :return: model_metadata_dict: Dictionary with all keys in the list
        `MODEL_METADATA_KEYS`.
    :raises: ValueError: if dictionary does not contain all keys in the list
        `MODEL_METADATA_KEYS`.
    """

    pickle_file_handle = open(pickle_file_name, 'rb')
    model_metadata_dict = pickle.load(pickle_file_handle)
    pickle_file_handle.close()

    expected_keys_as_set = set(MODEL_METADATA_KEYS)
    actual_keys_as_set = set(model_metadata_dict.keys())
    if not set(expected_keys_as_set).issubset(actual_keys_as_set):
        error_string = (
            '\n\n{0:s}\nExpected keys are listed above.  Keys found in file '
            '("{1:s}") are listed below.  Some expected keys were not found.'
            '\n{2:s}\n').format(MODEL_METADATA_KEYS, pickle_file_name,
                                model_metadata_dict.keys())

        raise ValueError(error_string)

    return model_metadata_dict


def train_2d_cnn(
        model_object, model_file_name, history_file_name, tensorboard_dir_name,
        num_epochs, num_training_batches_per_epoch, top_input_dir_name,
        radar_source, radar_field_names, num_examples_per_batch,
        num_examples_per_time, first_train_time_unix_sec,
        last_train_time_unix_sec, target_name, radar_heights_m_asl=None,
        reflectivity_heights_m_asl=None, normalize_by_batch=False,
        normalization_dict=dl_utils.DEFAULT_NORMALIZATION_DICT,
        percentile_offset_for_normalization=
        dl_utils.DEFAULT_PERCENTILE_OFFSET_FOR_NORMALIZATION,
        weight_loss_function=False, class_fractions_to_sample=None,
        num_validation_batches_per_epoch=None, first_validn_time_unix_sec=None,
        last_validn_time_unix_sec=None):
    """Trains 2-D CNN (one that performs 2-D convolution).

    :param model_object: Instance of `keras.models.Sequential`.
    :param model_file_name: See documentation for `_check_training_args`.
    :param history_file_name: Same.
    :param tensorboard_dir_name: Same.
    :param num_epochs: Same.
    :param num_training_batches_per_epoch: Same.
    :param top_input_dir_name: See doc for
        `training_validation_io.storm_image_generator_2d`.
    :param radar_source: Same.
    :param radar_field_names: Same.
    :param num_examples_per_batch: Same.
    :param num_examples_per_time: Same.
    :param first_train_time_unix_sec: First image time for training.
        Examples will be created for random times in
        `first_train_time_unix_sec`...`last_train_time_unix_sec`.
    :param last_train_time_unix_sec: See above.
    :param target_name: See doc for
        `training_validation_io.storm_image_generator_2d`.
    :param radar_heights_m_asl: Same.
    :param reflectivity_heights_m_asl: Same.
    :param normalize_by_batch: Same.
    :param normalization_dict: Same.
    :param percentile_offset_for_normalization: Same.
    :param weight_loss_function: See doc for `_check_training_args`.
    :param class_fractions_to_sample: Same.
    :param num_validation_batches_per_epoch: Same.
    :param first_validn_time_unix_sec: First image time for validation.
        Examples will be created for random times in
        `first_validn_time_unix_sec`...`last_validn_time_unix_sec`.
    :param last_validn_time_unix_sec: See above.
    """

    class_weight_dict = _check_training_args(
        num_epochs=num_epochs,
        num_training_batches_per_epoch=num_training_batches_per_epoch,
        num_validation_batches_per_epoch=num_validation_batches_per_epoch,
        weight_loss_function=weight_loss_function,
        class_fractions_to_sample=class_fractions_to_sample,
        model_file_name=model_file_name, history_file_name=history_file_name,
        tensorboard_dir_name=tensorboard_dir_name)

    history_object = keras.callbacks.CSVLogger(
        filename=history_file_name, separator=',', append=False)

    embedding_layer_names = [
        this_layer.name for this_layer in model_object.layers if
        this_layer.name.startswith('dense') or
        this_layer.name.startswith('conv')]

    tensorboard_object = keras.callbacks.TensorBoard(
        log_dir=tensorboard_dir_name, histogram_freq=0,
        batch_size=num_examples_per_batch, write_graph=True, write_grads=True,
        write_images=True, embeddings_freq=1,
        embeddings_layer_names=embedding_layer_names)

    training_file_name_matrix = training_validation_io.find_2d_input_files(
        top_directory_name=top_input_dir_name, radar_source=radar_source,
        radar_field_names=radar_field_names,
        first_image_time_unix_sec=first_train_time_unix_sec,
        last_image_time_unix_sec=last_train_time_unix_sec,
        radar_heights_m_asl=radar_heights_m_asl,
        reflectivity_heights_m_asl=reflectivity_heights_m_asl)

    if num_validation_batches_per_epoch is None:
        checkpoint_object = keras.callbacks.ModelCheckpoint(
            filepath=model_file_name, monitor='loss', verbose=1,
            save_best_only=False, save_weights_only=False, mode='min', period=1)

        model_object.fit_generator(
            generator=training_validation_io.storm_image_generator_2d(
                image_file_name_matrix=training_file_name_matrix,
                num_examples_per_batch=num_examples_per_batch,
                num_examples_per_image_time=num_examples_per_time,
                target_name=target_name, normalize_by_batch=normalize_by_batch,
                normalization_dict=normalization_dict,
                percentile_offset_for_normalization=
                percentile_offset_for_normalization,
                class_fractions_to_sample=class_fractions_to_sample),
            steps_per_epoch=num_training_batches_per_epoch, epochs=num_epochs,
            verbose=1, class_weight=class_weight_dict,
            callbacks=[checkpoint_object, history_object, tensorboard_object])

    else:
        validation_file_name_matrix = (
            training_validation_io.find_2d_input_files(
                top_directory_name=top_input_dir_name,
                radar_source=radar_source,
                radar_field_names=radar_field_names,
                first_image_time_unix_sec=first_validn_time_unix_sec,
                last_image_time_unix_sec=last_validn_time_unix_sec,
                radar_heights_m_asl=radar_heights_m_asl,
                reflectivity_heights_m_asl=reflectivity_heights_m_asl))

        checkpoint_object = keras.callbacks.ModelCheckpoint(
            filepath=model_file_name, monitor='val_loss', verbose=1,
            save_best_only=True, save_weights_only=False, mode='min', period=1)

        model_object.fit_generator(
            generator=training_validation_io.storm_image_generator_2d(
                image_file_name_matrix=training_file_name_matrix,
                num_examples_per_batch=num_examples_per_batch,
                num_examples_per_image_time=num_examples_per_time,
                target_name=target_name,
                normalize_by_batch=normalize_by_batch,
                normalization_dict=normalization_dict,
                percentile_offset_for_normalization=
                percentile_offset_for_normalization,
                class_fractions_to_sample=class_fractions_to_sample),
            steps_per_epoch=num_training_batches_per_epoch, epochs=num_epochs,
            verbose=1, class_weight=class_weight_dict,
            callbacks=[checkpoint_object, history_object, tensorboard_object],
            validation_data=training_validation_io.storm_image_generator_2d(
                image_file_name_matrix=validation_file_name_matrix,
                num_examples_per_batch=num_examples_per_batch,
                num_examples_per_image_time=num_examples_per_time,
                target_name=target_name,
                normalize_by_batch=normalize_by_batch,
                normalization_dict=normalization_dict,
                percentile_offset_for_normalization=
                percentile_offset_for_normalization,
                class_fractions_to_sample=class_fractions_to_sample),
            validation_steps=num_validation_batches_per_epoch)


def train_2d_cnn_with_dynamic_sampling(
        model_object, model_file_name, num_epochs,
        num_training_batches_per_epoch, top_input_dir_name, radar_source,
        radar_field_names, num_examples_per_batch, num_examples_per_time,
        first_train_time_unix_sec, last_train_time_unix_sec, target_name,
        class_fractions_by_epoch_matrix, radar_heights_m_asl=None,
        reflectivity_heights_m_asl=None, normalize_by_batch=False,
        normalization_dict=dl_utils.DEFAULT_NORMALIZATION_DICT,
        percentile_offset_for_normalization=
        dl_utils.DEFAULT_PERCENTILE_OFFSET_FOR_NORMALIZATION,
        weight_loss_function=True, num_validation_batches_per_epoch=None,
        first_validn_time_unix_sec=None, last_validn_time_unix_sec=None):
    """Trains 2-D CNN with dynamic class-conditional sampling.

    :param model_object: See documentation for `train_2d_cnn`.
    :param model_file_name: Same.
    :param num_epochs: Same.
    :param num_training_batches_per_epoch: Same.
    :param top_input_dir_name: Same.
    :param radar_source: Same.
    :param radar_field_names: Same.
    :param num_examples_per_batch: Same.
    :param num_examples_per_time: Same.
    :param first_train_time_unix_sec: Same.
    :param last_train_time_unix_sec: Same.
    :param target_name: Same.
    :param class_fractions_by_epoch_matrix:
        See doc for `train_3d_cnn_with_dynamic_sampling`.
    :param radar_heights_m_asl: See doc for `train_2d_cnn`.
    :param reflectivity_heights_m_asl: Same.
    :param normalize_by_batch: Same.
    :param normalization_dict: Same.
    :param percentile_offset_for_normalization: Same.
    :param weight_loss_function: Same.
    :param num_validation_batches_per_epoch: Same.
    :param first_validn_time_unix_sec: Same.
    :param last_validn_time_unix_sec: Same.
    """

    error_checking.assert_is_integer(num_epochs)
    error_checking.assert_is_greater(num_epochs, 0)

    num_classes = class_fractions_by_epoch_matrix.shape[1]
    error_checking.assert_is_numpy_array(
        class_fractions_by_epoch_matrix,
        exact_dimensions=numpy.array([num_epochs, num_classes]))

    file_system_utils.mkdir_recursive_if_necessary(file_name=model_file_name)
    model_directory_name, _ = os.path.split(model_file_name)
    history_file_names = [''] * num_epochs
    tensorboard_dir_names = [''] * num_epochs

    for i in range(num_epochs):
        history_file_names[i] = '{0:s}/model_history_epoch{1:04d}.csv'.format(
            model_directory_name, i)
        tensorboard_dir_names[i] = '{0:s}/tensorboard_epoch{1:04d}.csv'.format(
            model_directory_name, i)

        _check_training_args(
            num_epochs=num_epochs,
            num_training_batches_per_epoch=num_training_batches_per_epoch,
            num_validation_batches_per_epoch=num_validation_batches_per_epoch,
            weight_loss_function=weight_loss_function,
            class_fractions_to_sample=class_fractions_by_epoch_matrix[i, :],
            model_file_name=model_file_name,
            history_file_name=history_file_names[i],
            tensorboard_dir_name=tensorboard_dir_names[i])

    for i in range(num_epochs):
        train_2d_cnn(
            model_object=model_object, model_file_name=model_file_name,
            history_file_name=history_file_names[i],
            tensorboard_dir_name=tensorboard_dir_names[i], num_epochs=1,
            num_training_batches_per_epoch=num_training_batches_per_epoch,
            top_input_dir_name=top_input_dir_name, radar_source=radar_source,
            radar_field_names=radar_field_names,
            num_examples_per_batch=num_examples_per_batch,
            num_examples_per_time=num_examples_per_time,
            first_train_time_unix_sec=first_train_time_unix_sec,
            last_train_time_unix_sec=last_train_time_unix_sec,
            target_name=target_name, radar_heights_m_asl=radar_heights_m_asl,
            reflectivity_heights_m_asl=reflectivity_heights_m_asl,
            normalize_by_batch=normalize_by_batch,
            normalization_dict=normalization_dict,
            percentile_offset_for_normalization=
            percentile_offset_for_normalization,
            weight_loss_function=weight_loss_function,
            class_fractions_to_sample=class_fractions_by_epoch_matrix[i, :],
            num_validation_batches_per_epoch=num_validation_batches_per_epoch,
            first_validn_time_unix_sec=first_validn_time_unix_sec,
            last_validn_time_unix_sec=last_validn_time_unix_sec)


def train_3d_cnn(
        model_object, model_file_name, history_file_name, tensorboard_dir_name,
        num_epochs, num_training_batches_per_epoch, top_input_dir_name,
        radar_source, radar_field_names, radar_heights_m_asl,
        num_examples_per_batch, num_examples_per_time,
        first_train_time_unix_sec, last_train_time_unix_sec, target_name,
        normalize_by_batch=False,
        normalization_dict=dl_utils.DEFAULT_NORMALIZATION_DICT,
        percentile_offset_for_normalization=
        dl_utils.DEFAULT_PERCENTILE_OFFSET_FOR_NORMALIZATION,
        weight_loss_function=False, class_fractions_to_sample=None,
        num_validation_batches_per_epoch=None, first_validn_time_unix_sec=None,
        last_validn_time_unix_sec=None):
    """Trains 3-D CNN (one that performs 3-D convolution).

    :param model_object: Instance of `keras.models.Sequential`.
    :param model_file_name: See documentation for `_check_training_args`.
    :param history_file_name: Same.
    :param tensorboard_dir_name: Same.
    :param num_epochs: Same.
    :param num_training_batches_per_epoch: Same.
    :param top_input_dir_name: See doc for
        `training_validation_io.storm_image_generator_3d`.
    :param radar_source: Same.
    :param radar_field_names: Same.
    :param radar_heights_m_asl: Same.
    :param num_examples_per_batch: Same.
    :param num_examples_per_time: Same.
    :param first_train_time_unix_sec: First image time for training.
        Examples will be created for random times in
        `first_train_time_unix_sec`...`last_train_time_unix_sec`.
    :param last_train_time_unix_sec: See above.
    :param target_name: See doc for
        `training_validation_io.storm_image_generator_3d`.
    :param normalize_by_batch: Same.
    :param normalization_dict: Same.
    :param percentile_offset_for_normalization: Same.
    :param weight_loss_function: See doc for `_check_training_args`.
    :param class_fractions_to_sample: Same.
    :param num_validation_batches_per_epoch: Same.
    :param first_validn_time_unix_sec: First image time for validation.
        Examples will be created for random times in
        `first_validn_time_unix_sec`...`last_validn_time_unix_sec`.
    :param last_validn_time_unix_sec: See above.
    """

    class_weight_dict = _check_training_args(
        num_epochs=num_epochs,
        num_training_batches_per_epoch=num_training_batches_per_epoch,
        num_validation_batches_per_epoch=num_validation_batches_per_epoch,
        weight_loss_function=weight_loss_function,
        class_fractions_to_sample=class_fractions_to_sample,
        model_file_name=model_file_name, history_file_name=history_file_name,
        tensorboard_dir_name=tensorboard_dir_name)

    history_object = keras.callbacks.CSVLogger(
        filename=history_file_name, separator=',', append=False)

    embedding_layer_names = [
        this_layer.name for this_layer in model_object.layers if
        this_layer.name.startswith('dense') or
        this_layer.name.startswith('conv')]

    tensorboard_object = keras.callbacks.TensorBoard(
        log_dir=tensorboard_dir_name, histogram_freq=0,
        batch_size=num_examples_per_batch, write_graph=True, write_grads=True,
        write_images=True, embeddings_freq=1,
        embeddings_layer_names=embedding_layer_names)

    training_file_name_matrix = training_validation_io.find_3d_input_files(
        top_directory_name=top_input_dir_name,
        radar_source=radar_source, radar_field_names=radar_field_names,
        radar_heights_m_asl=radar_heights_m_asl,
        first_image_time_unix_sec=first_train_time_unix_sec,
        last_image_time_unix_sec=last_train_time_unix_sec)

    if num_validation_batches_per_epoch is None:
        checkpoint_object = keras.callbacks.ModelCheckpoint(
            filepath=model_file_name, monitor='loss', verbose=1,
            save_best_only=False, save_weights_only=False, mode='min', period=1)

        model_object.fit_generator(
            generator=training_validation_io.storm_image_generator_3d(
                image_file_name_matrix=training_file_name_matrix,
                num_examples_per_batch=num_examples_per_batch,
                num_examples_per_image_time=num_examples_per_time,
                target_name=target_name, normalize_by_batch=normalize_by_batch,
                normalization_dict=normalization_dict,
                percentile_offset_for_normalization=
                percentile_offset_for_normalization,
                class_fractions_to_sample=class_fractions_to_sample),
            steps_per_epoch=num_training_batches_per_epoch, epochs=num_epochs,
            verbose=1, class_weight=class_weight_dict,
            callbacks=[checkpoint_object, history_object, tensorboard_object])

    else:
        validation_file_name_matrix = (
            training_validation_io.find_3d_input_files(
                top_directory_name=top_input_dir_name,
                radar_source=radar_source, radar_field_names=radar_field_names,
                radar_heights_m_asl=radar_heights_m_asl,
                first_image_time_unix_sec=first_validn_time_unix_sec,
                last_image_time_unix_sec=last_validn_time_unix_sec))

        checkpoint_object = keras.callbacks.ModelCheckpoint(
            filepath=model_file_name, monitor='val_loss', verbose=1,
            save_best_only=True, save_weights_only=False, mode='min', period=1)

        model_object.fit_generator(
            generator=training_validation_io.storm_image_generator_3d(
                image_file_name_matrix=training_file_name_matrix,
                num_examples_per_batch=num_examples_per_batch,
                num_examples_per_image_time=num_examples_per_time,
                target_name=target_name, normalize_by_batch=normalize_by_batch,
                normalization_dict=normalization_dict,
                percentile_offset_for_normalization=
                percentile_offset_for_normalization,
                class_fractions_to_sample=class_fractions_to_sample),
            steps_per_epoch=num_training_batches_per_epoch, epochs=num_epochs,
            verbose=1, class_weight=class_weight_dict,
            callbacks=[checkpoint_object, history_object, tensorboard_object],
            validation_data=training_validation_io.storm_image_generator_3d(
                image_file_name_matrix=validation_file_name_matrix,
                num_examples_per_batch=num_examples_per_batch,
                num_examples_per_image_time=num_examples_per_time,
                target_name=target_name, normalize_by_batch=normalize_by_batch,
                normalization_dict=normalization_dict,
                percentile_offset_for_normalization=
                percentile_offset_for_normalization,
                class_fractions_to_sample=class_fractions_to_sample),
            validation_steps=num_validation_batches_per_epoch)


def train_3d_cnn_with_dynamic_sampling(
        model_object, model_file_name, num_epochs,
        num_training_batches_per_epoch, top_input_dir_name, radar_source,
        radar_field_names, radar_heights_m_asl, num_examples_per_batch,
        num_examples_per_time, first_train_time_unix_sec,
        last_train_time_unix_sec, target_name, class_fractions_by_epoch_matrix,
        normalize_by_batch=False,
        normalization_dict=dl_utils.DEFAULT_NORMALIZATION_DICT,
        percentile_offset_for_normalization=
        dl_utils.DEFAULT_PERCENTILE_OFFSET_FOR_NORMALIZATION,
        weight_loss_function=True, num_validation_batches_per_epoch=None,
        first_validn_time_unix_sec=None, last_validn_time_unix_sec=None):
    """Trains 3-D CNN with dynamic class-conditional sampling.

    K = number of classes for target variable
    L = number of epochs

    :param model_object: See documentation for `train_3d_cnn`.
    :param model_file_name: Same.
    :param num_epochs: Same.
    :param num_training_batches_per_epoch: Same.
    :param top_input_dir_name: Same.
    :param radar_source: Same.
    :param radar_field_names: Same.
    :param radar_heights_m_asl: Same.
    :param num_examples_per_batch: Same.
    :param num_examples_per_time: Same.
    :param first_train_time_unix_sec: Same.
    :param last_train_time_unix_sec: Same.
    :param target_name: Same.
    :param class_fractions_by_epoch_matrix: L-by-K numpy array, where
        class_fractions_by_epoch_matrix[i, j] is the fraction of data points
        from the [j]th class to use at the [i]th epoch.
    :param normalize_by_batch: See documentation for `train_3d_cnn`.
    :param normalization_dict: Same.
    :param percentile_offset_for_normalization: Same.
    :param weight_loss_function: Same.
    :param num_validation_batches_per_epoch: Same.
    :param first_validn_time_unix_sec: Same.
    :param last_validn_time_unix_sec: Same.
    """

    error_checking.assert_is_integer(num_epochs)
    error_checking.assert_is_greater(num_epochs, 0)

    num_classes = class_fractions_by_epoch_matrix.shape[1]
    error_checking.assert_is_numpy_array(
        class_fractions_by_epoch_matrix,
        exact_dimensions=numpy.array([num_epochs, num_classes]))

    file_system_utils.mkdir_recursive_if_necessary(file_name=model_file_name)
    model_directory_name, _ = os.path.split(model_file_name)
    history_file_names = [''] * num_epochs
    tensorboard_dir_names = [''] * num_epochs

    for i in range(num_epochs):
        history_file_names[i] = '{0:s}/model_history_epoch{1:04d}.csv'.format(
            model_directory_name, i)
        tensorboard_dir_names[i] = '{0:s}/tensorboard_epoch{1:04d}.csv'.format(
            model_directory_name, i)

        _check_training_args(
            num_epochs=num_epochs,
            num_training_batches_per_epoch=num_training_batches_per_epoch,
            num_validation_batches_per_epoch=num_validation_batches_per_epoch,
            weight_loss_function=weight_loss_function,
            class_fractions_to_sample=class_fractions_by_epoch_matrix[i, :],
            model_file_name=model_file_name,
            history_file_name=history_file_names[i],
            tensorboard_dir_name=tensorboard_dir_names[i])

    for i in range(num_epochs):
        train_3d_cnn(
            model_object=model_object, model_file_name=model_file_name,
            history_file_name=history_file_names[i],
            tensorboard_dir_name=tensorboard_dir_names[i], num_epochs=1,
            num_training_batches_per_epoch=num_training_batches_per_epoch,
            top_input_dir_name=top_input_dir_name, radar_source=radar_source,
            radar_field_names=radar_field_names,
            radar_heights_m_asl=radar_heights_m_asl,
            num_examples_per_batch=num_examples_per_batch,
            num_examples_per_time=num_examples_per_time,
            first_train_time_unix_sec=first_train_time_unix_sec,
            last_train_time_unix_sec=last_train_time_unix_sec,
            target_name=target_name, normalize_by_batch=normalize_by_batch,
            normalization_dict=normalization_dict,
            percentile_offset_for_normalization=
            percentile_offset_for_normalization,
            weight_loss_function=weight_loss_function,
            class_fractions_to_sample=class_fractions_by_epoch_matrix[i, :],
            num_validation_batches_per_epoch=num_validation_batches_per_epoch,
            first_validn_time_unix_sec=first_validn_time_unix_sec,
            last_validn_time_unix_sec=last_validn_time_unix_sec)


def apply_2d_cnn(model_object, predictor_matrix):
    """Applies 2-D CNN (one that performs 2-D convolution) to new examples.

    :param model_object: Instance of `keras.models.Sequential`.
    :param predictor_matrix: E-by-M-by-N-by-C numpy array of storm-centered
        radar images.
    :return: class_probability_matrix: E-by-K numpy array of forecast
        probabilities.  class_probability_matrix[i, k] is the forecast
        probability the [i]th example belongs to the [k]th class.
    """

    dl_utils.check_predictor_matrix(
        predictor_matrix=predictor_matrix, min_num_dimensions=4,
        max_num_dimensions=4)

    num_examples = predictor_matrix.shape[0]
    return model_object.predict(predictor_matrix, batch_size=num_examples)


def apply_3d_cnn(model_object, predictor_matrix):
    """Applies 3-D CNN (one that performs 3-D convolution) to new examples.

    :param model_object: Instance of `keras.models.Sequential`.
    :param predictor_matrix: E-by-M-by-N-by-D-by-C numpy array of storm-centered
        radar images.
    :return: class_probability_matrix: E-by-K numpy array of forecast
        probabilities.  class_probability_matrix[i, k] is the forecast
        probability the [i]th example belongs to the [k]th class.
    """

    dl_utils.check_predictor_matrix(
        predictor_matrix=predictor_matrix, min_num_dimensions=5,
        max_num_dimensions=5)

    num_examples = predictor_matrix.shape[0]
    return model_object.predict(predictor_matrix, batch_size=num_examples)
