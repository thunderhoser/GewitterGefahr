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
import keras.losses
import keras.optimizers
import keras.models
import keras.callbacks
from gewittergefahr.deep_learning import deep_learning_utils as dl_utils
from gewittergefahr.deep_learning import cnn_utils
from gewittergefahr.deep_learning import keras_metrics
from gewittergefahr.deep_learning import training_validation_io as trainval_io
from gewittergefahr.gg_utils import soundings
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.gg_utils import error_checking

NUM_INPUT_ROWS = 32
NUM_INPUT_COLUMNS = 32
NUM_INPUT_DEPTHS = 12
NUM_INPUT_ROWS_FOR_AZ_SHEAR = 64
NUM_INPUT_COLUMNS_FOR_AZ_SHEAR = 64
DEFAULT_L2_WEIGHT = 1e-3

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
NUM_EXAMPLES_PER_FILE_TIME_KEY = 'num_examples_per_time'
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
CLASS_FRACTION_DICT_KEY = 'class_fraction_dict'
SOUNDING_STAT_NAMES_KEY = 'sounding_statistic_names'
BINARIZE_TARGET_KEY = 'binarize_target'
USE_2D3D_CONVOLUTION_KEY = 'use_2d3d_convolution'

MODEL_METADATA_KEYS = [
    NUM_EPOCHS_KEY, NUM_EXAMPLES_PER_BATCH_KEY, NUM_EXAMPLES_PER_FILE_TIME_KEY,
    NUM_TRAINING_BATCHES_PER_EPOCH_KEY, FIRST_TRAINING_TIME_KEY,
    LAST_TRAINING_TIME_KEY, NUM_VALIDATION_BATCHES_PER_EPOCH_KEY,
    FIRST_VALIDATION_TIME_KEY, LAST_VALIDATION_TIME_KEY, RADAR_SOURCE_KEY,
    RADAR_FIELD_NAMES_KEY, RADAR_HEIGHTS_KEY, REFLECTIVITY_HEIGHTS_KEY,
    TARGET_NAME_KEY, NORMALIZE_BY_BATCH_KEY, NORMALIZATION_DICT_KEY,
    PERCENTILE_OFFSET_KEY, CLASS_FRACTION_DICT_KEY, SOUNDING_STAT_NAMES_KEY,
    BINARIZE_TARGET_KEY, USE_2D3D_CONVOLUTION_KEY
]


def _check_architecture_args(
        num_classes, num_input_channels, num_sounding_stats,
        dropout_fraction=None, l2_weight=None):
    """Error-checks input arguments for architecture-creation method.

    :param num_classes: Number of classes for target variable.
    :param num_input_channels: Number of input channels (predictor variables).
    :param num_sounding_stats: Number of sounding statistics.  If the model does
        not take sounding statistics as input, set this to 0.
    :param dropout_fraction: Dropout fraction.  This will be applied to each
        dropout layer, of which there is one after each convolutional layer.
    :param l2_weight: L2-regularization weight.  This will be used for each
        convolutional layer.
    """

    error_checking.assert_is_integer(num_classes)
    error_checking.assert_is_geq(num_classes, 2)
    error_checking.assert_is_integer(num_input_channels)
    error_checking.assert_is_geq(num_input_channels, 1)
    error_checking.assert_is_integer(num_sounding_stats)
    error_checking.assert_is_geq(num_sounding_stats, 0)

    if dropout_fraction is not None:
        error_checking.assert_is_greater(dropout_fraction, 0.)
        error_checking.assert_is_less_than(dropout_fraction, 1.)

    if l2_weight is not None:
        error_checking.assert_is_greater(l2_weight, 0.)


def _check_training_args(
        num_epochs, num_training_batches_per_epoch,
        num_validation_batches_per_epoch, weight_loss_function, binarize_target,
        training_class_fraction_dict, model_file_name, history_file_name,
        tensorboard_dir_name):
    """Error-checks input arguments for training method.

    :param num_epochs: Number of training epochs.
    :param num_training_batches_per_epoch: Number of training batches per epoch.
    :param num_validation_batches_per_epoch: Number of validation batches per
        epoch.
    :param weight_loss_function: Boolean flag.  If False, classes will be
        weighted equally in the loss function.  If True, classes will be
        weighted differently in the loss function (inversely proportional with
        `class_fraction_dict`).
    :param binarize_target: Boolean flag.  If 0, the model will be trained to
        predict all classes of the target variable.  If 1, the highest class
        will be taken as the positive class and all other classes will be taken
        as negative, turning the problem into binary classification.
    :param training_class_fraction_dict: Dictionary, where each key is a
        class integer (-2 for dead storms) and each value is the corresponding
        sampling fraction for training data.
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
    error_checking.assert_is_boolean(binarize_target)
    if weight_loss_function:
        class_weight_dict = dl_utils.class_fractions_to_weights(
            class_fraction_dict=training_class_fraction_dict,
            binarize_target=binarize_target)
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

    _check_architecture_args(
        num_classes=num_classes, num_input_channels=num_input_channels,
        num_sounding_stats=0)

    model_object = keras.models.Sequential()
    layer_object = cnn_utils.get_2d_conv_layer(
        num_output_filters=32, num_kernel_rows=3, num_kernel_columns=3,
        num_rows_per_stride=1, num_columns_per_stride=1,
        padding_type=cnn_utils.NO_PADDING_TYPE, activation_function='relu',
        is_first_layer=True, num_input_rows=NUM_INPUT_ROWS,
        num_input_columns=NUM_INPUT_COLUMNS,
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
        optimizer=keras.optimizers.Adam(), metrics=LIST_OF_METRIC_FUNCTIONS)

    model_object.summary()
    return model_object


def get_2d_swirlnet_architecture(
        num_classes, dropout_fraction, num_input_channels=3,
        num_sounding_stats=0, l2_weight=DEFAULT_L2_WEIGHT):
    """Creates 2-D CNN (one that performs 2-D convolution).

    Architecture is similar to the following example:

    https://github.com/djgagne/swirlnet/blob/master/notebooks/
    deep_swirl_tutorial.ipynb

    :param num_classes: Number of target classes.
    :param dropout_fraction: Dropout fraction.  This will be applied to each
        dropout layer, of which there is one after each conv layer.
    :param num_input_channels: Number of input channels (predictor variables).
    :param num_sounding_stats: Number of sounding statistics.  If the model is
        not being trained with sounding statistics, leave this as 0.
    :param l2_weight: L2-regularization weight.  This will be used for all
        convolutional layers.
    :return: model_object: Instance of `keras.models.Sequential` with the
        aforementioned architecture.
    """

    _check_architecture_args(
        num_classes=num_classes, num_input_channels=num_input_channels,
        num_sounding_stats=num_sounding_stats,
        dropout_fraction=dropout_fraction, l2_weight=l2_weight)

    regularizer_object = cnn_utils.get_weight_regularizer(
        l1_penalty=0, l2_penalty=l2_weight)

    radar_input_layer_object = keras.layers.Input(shape=(
        NUM_INPUT_ROWS, NUM_INPUT_COLUMNS, num_input_channels))

    # Input to this layer is E x 32 x 32 x C.
    layer_object = cnn_utils.get_2d_conv_layer(
        num_output_filters=16, num_kernel_rows=5, num_kernel_columns=5,
        num_rows_per_stride=1, num_columns_per_stride=1,
        padding_type=cnn_utils.YES_PADDING_TYPE,
        kernel_weight_regularizer=regularizer_object,
        activation_function='relu', is_first_layer=False)(
            radar_input_layer_object)

    # Input to this layer is E x 32 x 32 x 16.
    layer_object = cnn_utils.get_dropout_layer(
        dropout_fraction=dropout_fraction)(layer_object)

    # Input to this layer is E x 32 x 32 x 16.
    layer_object = cnn_utils.get_2d_pooling_layer(
        num_rows_in_window=2, num_columns_in_window=2,
        pooling_type=cnn_utils.MEAN_POOLING_TYPE, num_rows_per_stride=2,
        num_columns_per_stride=2)(layer_object)

    # Input to this layer is E x 16 x 16 x 16.
    layer_object = cnn_utils.get_2d_conv_layer(
        num_output_filters=32, num_kernel_rows=5, num_kernel_columns=5,
        num_rows_per_stride=1, num_columns_per_stride=1,
        padding_type=cnn_utils.YES_PADDING_TYPE,
        kernel_weight_regularizer=regularizer_object,
        activation_function='relu')(layer_object)

    # Input to this layer is E x 16 x 16 x 32.
    layer_object = cnn_utils.get_dropout_layer(
        dropout_fraction=dropout_fraction)(layer_object)

    # Input to this layer is E x 16 x 16 x 32.
    layer_object = cnn_utils.get_2d_pooling_layer(
        num_rows_in_window=2, num_columns_in_window=2,
        pooling_type=cnn_utils.MEAN_POOLING_TYPE, num_rows_per_stride=2,
        num_columns_per_stride=2)(layer_object)

    # Input to this layer is E x 8 x 8 x 32.
    layer_object = cnn_utils.get_2d_conv_layer(
        num_output_filters=64, num_kernel_rows=3, num_kernel_columns=3,
        num_rows_per_stride=1, num_columns_per_stride=1,
        padding_type=cnn_utils.YES_PADDING_TYPE,
        kernel_weight_regularizer=regularizer_object,
        activation_function='relu')(layer_object)

    # Input to this layer is E x 8 x 8 x 64.
    layer_object = cnn_utils.get_dropout_layer(
        dropout_fraction=dropout_fraction)(layer_object)

    # Input to this layer is E x 8 x 8 x 64.
    layer_object = cnn_utils.get_2d_pooling_layer(
        num_rows_in_window=2, num_columns_in_window=2,
        pooling_type=cnn_utils.MEAN_POOLING_TYPE, num_rows_per_stride=2,
        num_columns_per_stride=2)(layer_object)

    # Input to this layer is E x 4 x 4 x 64.
    layer_object = cnn_utils.get_flattening_layer()(layer_object)

    if num_sounding_stats > 0:
        sounding_input_layer_object = keras.layers.Input(
            shape=(num_sounding_stats,))
        layer_object = keras.layers.concatenate(
            [layer_object, sounding_input_layer_object])

    # Input to this layer is E x (1024 + num_sounding_stats).
    layer_object = cnn_utils.get_fully_connected_layer(
        num_output_units=num_classes, activation_function='softmax')(
            layer_object)

    if num_sounding_stats > 0:
        model_object = keras.models.Model(
            inputs=[radar_input_layer_object, sounding_input_layer_object],
            outputs=layer_object)
    else:
        model_object = keras.models.Model(
            inputs=radar_input_layer_object, outputs=layer_object)

    model_object.compile(
        loss=keras.losses.categorical_crossentropy,
        optimizer=keras.optimizers.Adam(), metrics=LIST_OF_METRIC_FUNCTIONS)

    model_object.summary()
    return model_object


def get_3d_swirlnet_architecture(
        num_classes, dropout_fraction, num_input_channels=3,
        num_sounding_stats=0, l2_weight=DEFAULT_L2_WEIGHT):
    """Creates 2-D CNN (one that performs 2-D convolution).

    Architecture is similar to the following example:

    https://github.com/djgagne/swirlnet/blob/master/notebooks/
    deep_swirl_tutorial.ipynb

    :param num_classes: See doc for `get_2d_swirlnet_architecture`.
    :param dropout_fraction: Same.
    :param num_input_channels: Same.
    :param num_sounding_stats: Same.
    :param l2_weight: Same.
    :return: model_object: Same.
    """

    _check_architecture_args(
        num_classes=num_classes, num_input_channels=num_input_channels,
        num_sounding_stats=num_sounding_stats,
        dropout_fraction=dropout_fraction, l2_weight=l2_weight)

    regularizer_object = cnn_utils.get_weight_regularizer(
        l1_penalty=0, l2_penalty=l2_weight)

    radar_input_layer_object = keras.layers.Input(shape=(
        NUM_INPUT_ROWS, NUM_INPUT_COLUMNS, NUM_INPUT_DEPTHS, num_input_channels
    ))

    # Input to this layer is E x 32 x 32 x 12 x C.
    layer_object = cnn_utils.get_3d_conv_layer(
        num_output_filters=8*num_input_channels, num_kernel_rows=5,
        num_kernel_columns=5, num_kernel_depths=3, num_rows_per_stride=1,
        num_columns_per_stride=1, num_depths_per_stride=1,
        padding_type=cnn_utils.YES_PADDING_TYPE,
        kernel_weight_regularizer=regularizer_object,
        activation_function='relu', is_first_layer=False)(
            radar_input_layer_object)

    # Input to this layer is E x 32 x 32 x 12 x 8C.
    layer_object = cnn_utils.get_dropout_layer(
        dropout_fraction=dropout_fraction)(layer_object)

    # Input to this layer is E x 32 x 32 x 12 x 8C.
    layer_object = cnn_utils.get_3d_pooling_layer(
        num_rows_in_window=2, num_columns_in_window=2, num_depths_in_window=2,
        pooling_type=cnn_utils.MEAN_POOLING_TYPE, num_rows_per_stride=2,
        num_columns_per_stride=2, num_depths_per_stride=2)(layer_object)

    # Input to this layer is E x 16 x 16 x 6 x 8C.
    layer_object = cnn_utils.get_3d_conv_layer(
        num_output_filters=16*num_input_channels, num_kernel_rows=5,
        num_kernel_columns=5, num_kernel_depths=3, num_rows_per_stride=1,
        num_columns_per_stride=1, num_depths_per_stride=1,
        padding_type=cnn_utils.YES_PADDING_TYPE,
        kernel_weight_regularizer=regularizer_object,
        activation_function='relu')(layer_object)

    # Input to this layer is E x 16 x 16 x 6 x 16C.
    layer_object = cnn_utils.get_dropout_layer(
        dropout_fraction=dropout_fraction)(layer_object)

    # Input to this layer is E x 16 x 16 x 6 x 16C.
    layer_object = cnn_utils.get_3d_pooling_layer(
        num_rows_in_window=2, num_columns_in_window=2, num_depths_in_window=2,
        pooling_type=cnn_utils.MEAN_POOLING_TYPE, num_rows_per_stride=2,
        num_columns_per_stride=2, num_depths_per_stride=2)(layer_object)

    # Input to this layer is E x 8 x 8 x 3 x 16C.
    layer_object = cnn_utils.get_3d_conv_layer(
        num_output_filters=32*num_input_channels, num_kernel_rows=3,
        num_kernel_columns=3, num_kernel_depths=3, num_rows_per_stride=1,
        num_columns_per_stride=1, num_depths_per_stride=1,
        padding_type=cnn_utils.YES_PADDING_TYPE,
        kernel_weight_regularizer=regularizer_object,
        activation_function='relu')(layer_object)

    # Input to this layer is E x 8 x 8 x 3 x 32C.
    layer_object = cnn_utils.get_dropout_layer(
        dropout_fraction=dropout_fraction)(layer_object)

    # Input to this layer is E x 8 x 8 x 3 x 32C.
    layer_object = cnn_utils.get_3d_pooling_layer(
        num_rows_in_window=2, num_columns_in_window=2, num_depths_in_window=3,
        pooling_type=cnn_utils.MEAN_POOLING_TYPE, num_rows_per_stride=2,
        num_columns_per_stride=2, num_depths_per_stride=3)(layer_object)

    # Input to this layer is E x 4 x 4 x 1 x 32C.
    layer_object = cnn_utils.get_flattening_layer()(layer_object)

    if num_sounding_stats > 0:
        sounding_input_layer_object = keras.layers.Input(
            shape=(num_sounding_stats,))
        layer_object = keras.layers.concatenate(
            [layer_object, sounding_input_layer_object])

    # Input to this layer is length-512C.
    layer_object = cnn_utils.get_fully_connected_layer(
        num_output_units=num_classes, activation_function='softmax')(
            layer_object)

    if num_sounding_stats > 0:
        model_object = keras.models.Model(
            inputs=[radar_input_layer_object, sounding_input_layer_object],
            outputs=layer_object)
    else:
        model_object = keras.models.Model(
            inputs=radar_input_layer_object, outputs=layer_object)

    model_object.compile(
        loss=keras.losses.categorical_crossentropy,
        optimizer=keras.optimizers.Adam(), metrics=LIST_OF_METRIC_FUNCTIONS)

    model_object.summary()
    return model_object


def get_architecture_for_2d3d_myrorss(
        num_classes, dropout_fraction, first_num_reflectivity_filters,
        num_azimuthal_shear_fields=2, l2_weight=DEFAULT_L2_WEIGHT):
    """Creates hybrid 2D-3D CNN for MYRORSS data.

    This CNN will perform 3-D convolution over reflectivity fields and 2-D
    convolution over azimuthal-shear fields.

    F = number of azimuthal-shear fields

    :param num_classes: Number of output classes.
    :param dropout_fraction: Dropout fraction.  This will be applied to each
        dropout layer, of which there is one after each conv layer.
    :param first_num_reflectivity_filters: Number of reflectivity filters in
        first layer.
    :param num_azimuthal_shear_fields: Number of azimuthal-shear fields.
    :param l2_weight: L2-regularization weight.  This will be used for all
        convolutional layers.
    :return: model_object: Instance of `keras.models.Sequential` with the
        aforementioned architecture.
    """

    _check_architecture_args(
        num_classes=num_classes, num_input_channels=2, num_sounding_stats=0,
        dropout_fraction=dropout_fraction, l2_weight=l2_weight)

    error_checking.assert_is_integer(first_num_reflectivity_filters)
    error_checking.assert_is_geq(first_num_reflectivity_filters, 2)
    error_checking.assert_is_integer(num_azimuthal_shear_fields)
    error_checking.assert_is_geq(num_azimuthal_shear_fields, 1)

    regularizer_object = cnn_utils.get_weight_regularizer(
        l1_penalty=0, l2_penalty=l2_weight)

    reflectivity_input_layer_object = keras.layers.Input(shape=(
        NUM_INPUT_ROWS, NUM_INPUT_COLUMNS, NUM_INPUT_DEPTHS, 1))
    az_shear_input_layer_object = keras.layers.Input(shape=(
        NUM_INPUT_ROWS_FOR_AZ_SHEAR, NUM_INPUT_COLUMNS_FOR_AZ_SHEAR,
        num_azimuthal_shear_fields))

    # Input to this layer is E x 32 x 32 x 12 x 1.
    reflectivity_layer_object = cnn_utils.get_3d_conv_layer(
        num_output_filters=first_num_reflectivity_filters, num_kernel_rows=5,
        num_kernel_columns=5, num_kernel_depths=3, num_rows_per_stride=1,
        num_columns_per_stride=1, num_depths_per_stride=1,
        padding_type=cnn_utils.YES_PADDING_TYPE,
        kernel_weight_regularizer=regularizer_object,
        activation_function='relu', is_first_layer=False)(
            reflectivity_input_layer_object)

    # Input to this layer is E x 32 x 32 x 12 x
    # `first_num_reflectivity_filters`.
    reflectivity_layer_object = cnn_utils.get_dropout_layer(
        dropout_fraction=dropout_fraction)(reflectivity_layer_object)

    # Input to this layer is E x 32 x 32 x 12 x
    # `first_num_reflectivity_filters`.
    reflectivity_layer_object = cnn_utils.get_3d_pooling_layer(
        num_rows_in_window=1, num_columns_in_window=1, num_depths_in_window=2,
        pooling_type=cnn_utils.MEAN_POOLING_TYPE, num_rows_per_stride=1,
        num_columns_per_stride=1, num_depths_per_stride=2)(
            reflectivity_layer_object)

    # Input to this layer is E x 32 x 32 x 6 x
    # `first_num_reflectivity_filters`.
    reflectivity_layer_object = cnn_utils.get_3d_conv_layer(
        num_output_filters=2 * first_num_reflectivity_filters,
        num_kernel_rows=5, num_kernel_columns=5, num_kernel_depths=3,
        num_rows_per_stride=1, num_columns_per_stride=1,
        num_depths_per_stride=1, padding_type=cnn_utils.YES_PADDING_TYPE,
        kernel_weight_regularizer=regularizer_object,
        activation_function='relu')(reflectivity_layer_object)

    # Input to this layer is E x 32 x 32 x 6 x
    # `2 * first_num_reflectivity_filters`.
    reflectivity_layer_object = cnn_utils.get_dropout_layer(
        dropout_fraction=dropout_fraction)(reflectivity_layer_object)

    # Input to this layer is E x 32 x 32 x 6 x
    # `2 * first_num_reflectivity_filters`.
    reflectivity_layer_object = cnn_utils.get_3d_pooling_layer(
        num_rows_in_window=1, num_columns_in_window=1, num_depths_in_window=2,
        pooling_type=cnn_utils.MEAN_POOLING_TYPE, num_rows_per_stride=1,
        num_columns_per_stride=1, num_depths_per_stride=2)(
            reflectivity_layer_object)

    # Input to this layer is E x 32 x 32 x 3 x
    # `2 * first_num_reflectivity_filters`.
    reflectivity_layer_object = cnn_utils.get_3d_conv_layer(
        num_output_filters=4 * first_num_reflectivity_filters,
        num_kernel_rows=5, num_kernel_columns=5, num_kernel_depths=3,
        num_rows_per_stride=1, num_columns_per_stride=1,
        num_depths_per_stride=1, padding_type=cnn_utils.YES_PADDING_TYPE,
        kernel_weight_regularizer=regularizer_object,
        activation_function='relu')(reflectivity_layer_object)

    # Input to this layer is E x 32 x 32 x 3 x
    # `4 * first_num_reflectivity_filters`.
    reflectivity_layer_object = cnn_utils.get_dropout_layer(
        dropout_fraction=dropout_fraction)(reflectivity_layer_object)

    # Input to this layer is E x 32 x 32 x 3 x
    # `4 * first_num_reflectivity_filters`.
    reflectivity_layer_object = cnn_utils.get_3d_pooling_layer(
        num_rows_in_window=1, num_columns_in_window=1, num_depths_in_window=3,
        pooling_type=cnn_utils.MEAN_POOLING_TYPE, num_rows_per_stride=1,
        num_columns_per_stride=1, num_depths_per_stride=3)(
            reflectivity_layer_object)

    # Input to this layer is E x 32 x 32 x 1 x
    # `4 * first_num_reflectivity_filters`.  Output is E x 32 x 32 x
    # `4 * first_num_reflectivity_filters`.
    this_target_shape = (
        NUM_INPUT_ROWS, NUM_INPUT_COLUMNS, 4 * first_num_reflectivity_filters)
    reflectivity_layer_object = keras.layers.Reshape(
        target_shape=this_target_shape)(reflectivity_layer_object)

    # Input to this layer is E x 64 x 64 x F.
    az_shear_layer_object = cnn_utils.get_2d_conv_layer(
        num_output_filters=16, num_kernel_rows=5, num_kernel_columns=5,
        num_rows_per_stride=1, num_columns_per_stride=1,
        padding_type=cnn_utils.YES_PADDING_TYPE,
        kernel_weight_regularizer=regularizer_object,
        activation_function='relu', is_first_layer=False)(
            az_shear_input_layer_object)

    # Input to this layer is E x 64 x 64 x 16.
    az_shear_layer_object = cnn_utils.get_dropout_layer(
        dropout_fraction=dropout_fraction)(az_shear_layer_object)

    # Input to this layer is E x 64 x 64 x 16.  Output is E x 32 x 32 x 16.
    az_shear_layer_object = cnn_utils.get_2d_pooling_layer(
        num_rows_in_window=2, num_columns_in_window=2,
        pooling_type=cnn_utils.MEAN_POOLING_TYPE, num_rows_per_stride=2,
        num_columns_per_stride=2)(az_shear_layer_object)

    # Output of this layer is E x 32 x 32 x
    # `16 + 4 * first_num_reflectivity_filters`.
    layer_object = keras.layers.concatenate(
        [reflectivity_layer_object, az_shear_layer_object], axis=-1)

    # Input to this layer is E x 32 x 32 x
    # `16 + 4 * first_num_reflectivity_filters`.
    layer_object = cnn_utils.get_2d_conv_layer(
        num_output_filters=32 + 8 * first_num_reflectivity_filters,
        num_kernel_rows=5, num_kernel_columns=5, num_rows_per_stride=1,
        num_columns_per_stride=1, padding_type=cnn_utils.YES_PADDING_TYPE,
        kernel_weight_regularizer=regularizer_object,
        activation_function='relu')(layer_object)

    # Input to this layer is E x 32 x 32 x
    # `32 + 8 * first_num_reflectivity_filters`.
    layer_object = cnn_utils.get_dropout_layer(
        dropout_fraction=dropout_fraction)(layer_object)

    # Input to this layer is E x 32 x 32 x
    # `32 + 8 * first_num_reflectivity_filters`.
    layer_object = cnn_utils.get_2d_pooling_layer(
        num_rows_in_window=2, num_columns_in_window=2,
        pooling_type=cnn_utils.MEAN_POOLING_TYPE, num_rows_per_stride=2,
        num_columns_per_stride=2)(layer_object)

    # Input to this layer is E x 16 x 16 x
    # `32 + 8 * first_num_reflectivity_filters`.
    layer_object = cnn_utils.get_2d_conv_layer(
        num_output_filters=64 + 16 * first_num_reflectivity_filters,
        num_kernel_rows=5, num_kernel_columns=5, num_rows_per_stride=1,
        num_columns_per_stride=1, padding_type=cnn_utils.YES_PADDING_TYPE,
        kernel_weight_regularizer=regularizer_object,
        activation_function='relu')(layer_object)

    # Input to this layer is E x 16 x 16 x
    # `64 + 16 * first_num_reflectivity_filters`.
    layer_object = cnn_utils.get_dropout_layer(
        dropout_fraction=dropout_fraction)(layer_object)

    # Input to this layer is E x 16 x 16 x
    # `64 + 16 * first_num_reflectivity_filters`.
    layer_object = cnn_utils.get_2d_pooling_layer(
        num_rows_in_window=2, num_columns_in_window=2,
        pooling_type=cnn_utils.MEAN_POOLING_TYPE, num_rows_per_stride=2,
        num_columns_per_stride=2)(layer_object)

    # Input to this layer is E x 8 x 8 x
    # `64 + 16 * first_num_reflectivity_filters`.
    layer_object = cnn_utils.get_2d_conv_layer(
        num_output_filters=128 + 32 * first_num_reflectivity_filters,
        num_kernel_rows=5, num_kernel_columns=5, num_rows_per_stride=1,
        num_columns_per_stride=1, padding_type=cnn_utils.YES_PADDING_TYPE,
        kernel_weight_regularizer=regularizer_object,
        activation_function='relu')(layer_object)

    # Input to this layer is E x 8 x 8 x
    # `128 + 32 * first_num_reflectivity_filters`.
    layer_object = cnn_utils.get_dropout_layer(
        dropout_fraction=dropout_fraction)(layer_object)

    # Input to this layer is E x 8 x 8 x
    # `128 + 32 * first_num_reflectivity_filters`.
    layer_object = cnn_utils.get_2d_pooling_layer(
        num_rows_in_window=2, num_columns_in_window=2,
        pooling_type=cnn_utils.MEAN_POOLING_TYPE, num_rows_per_stride=2,
        num_columns_per_stride=2)(layer_object)

    # Input to this layer is E x 4 x 4 x
    # `128 + 32 * first_num_reflectivity_filters`.
    layer_object = cnn_utils.get_flattening_layer()(layer_object)

    # Input to this layer is E x `2048 + 512 * first_num_reflectivity_filters`.
    layer_object = cnn_utils.get_fully_connected_layer(
        num_output_units=num_classes, activation_function='softmax')(
            layer_object)

    model_object = keras.models.Model(
        inputs=[reflectivity_input_layer_object, az_shear_input_layer_object],
        outputs=layer_object)

    model_object.compile(
        loss=keras.losses.categorical_crossentropy,
        optimizer=keras.optimizers.Adam(), metrics=LIST_OF_METRIC_FUNCTIONS)

    model_object.summary()
    return model_object


def read_model(hdf5_file_name):
    """Reads model from HDF5 file.

    :param hdf5_file_name: Path to input file.
    :return: model_object: Instance of `keras.models.Model`.
    """

    error_checking.assert_file_exists(hdf5_file_name)
    return keras.models.load_model(
        hdf5_file_name, custom_objects=CUSTOM_OBJECT_DICT_FOR_READING_MODEL)


def write_model_metadata(
        num_epochs, num_examples_per_batch, num_examples_per_file_time,
        num_training_batches_per_epoch, first_train_time_unix_sec,
        last_train_time_unix_sec, num_validation_batches_per_epoch,
        first_validn_time_unix_sec, last_validn_time_unix_sec,
        radar_source, radar_field_names, radar_heights_m_asl,
        reflectivity_heights_m_asl, target_name, normalize_by_batch,
        normalization_dict, percentile_offset_for_normalization,
        class_fraction_dict, sounding_statistic_names,
        binarize_target, use_2d3d_convolution, pickle_file_name):
    """Writes metadata to Pickle file.

    :param num_epochs: See documentation for `train_2d_cnn` or `train_3d_cnn`.
    :param num_examples_per_batch: Same.
    :param num_examples_per_file_time: Same.
    :param num_training_batches_per_epoch: Same.
    :param first_train_time_unix_sec: See doc for
        `training_validation_io.find_2d_input_files` or
        `training_validation_io.find_3d_input_files`.
    :param last_train_time_unix_sec: Same.
    :param num_validation_batches_per_epoch: See doc for `train_2d_cnn` or
        `train_3d_cnn`.
    :param first_validn_time_unix_sec: See doc for
        `training_validation_io.find_2d_input_files` or
        `training_validation_io.find_3d_input_files`.
    :param last_validn_time_unix_sec: Same.
    :param radar_source: See doc for
        `training_validation_io.find_2d_input_files` or
        `training_validation_io.find_3d_input_files`.
    :param radar_field_names: Same.
    :param radar_heights_m_asl: Same.
    :param reflectivity_heights_m_asl: Same.
    :param target_name: See doc for `train_2d_cnn` or `train_3d_cnn`.
    :param normalize_by_batch: Same.
    :param normalization_dict: Same.
    :param percentile_offset_for_normalization: Same.
    :param class_fraction_dict: Same.
    :param sounding_statistic_names: Same.
    :param binarize_target: Same.
    :param use_2d3d_convolution: Boolean flag.  If True, the model convolves
        over both 3-D reflectivity and 2-D azimuthal-shear fields.
    :param pickle_file_name: Path to output file.
    """

    # TODO(thunderhoser): This needs to be updated.  I have more hyperparams
    # now.

    model_metadata_dict = {
        NUM_EPOCHS_KEY: num_epochs,
        NUM_EXAMPLES_PER_BATCH_KEY: num_examples_per_batch,
        NUM_EXAMPLES_PER_FILE_TIME_KEY: num_examples_per_file_time,
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
        CLASS_FRACTION_DICT_KEY: class_fraction_dict,
        SOUNDING_STAT_NAMES_KEY: sounding_statistic_names,
        BINARIZE_TARGET_KEY: binarize_target,
        USE_2D3D_CONVOLUTION_KEY: use_2d3d_convolution
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

    if SOUNDING_STAT_NAMES_KEY not in model_metadata_dict:
        model_metadata_dict.update({SOUNDING_STAT_NAMES_KEY: None})
    if BINARIZE_TARGET_KEY not in model_metadata_dict:
        model_metadata_dict.update({BINARIZE_TARGET_KEY: False})
    if USE_2D3D_CONVOLUTION_KEY not in model_metadata_dict:
        model_metadata_dict.update({USE_2D3D_CONVOLUTION_KEY: False})
    if CLASS_FRACTION_DICT_KEY not in model_metadata_dict:
        model_metadata_dict.update({CLASS_FRACTION_DICT_KEY: None})

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
        num_epochs, num_training_batches_per_epoch,
        train_image_file_name_matrix, top_target_directory_name,
        num_examples_per_batch, num_examples_per_file_time, target_name,
        binarize_target=False, sounding_statistic_names=None,
        train_sounding_stat_file_names=None, weight_loss_function=False,
        training_class_fraction_dict=None,
        num_validation_batches_per_epoch=None,
        validn_image_file_name_matrix=None,
        validn_sounding_stat_file_names=None,
        validation_class_fraction_dict=None):
    """Trains 2-D CNN (one that performs 2-D convolution).

    T = number of storm times (initial times) for training
    V = number of storm times for validation
    C = number of channels = num predictor variables = num field/height pairs

    :param model_object: Instance of `keras.models.Sequential`.
    :param model_file_name: See documentation for `_check_training_args`.
    :param history_file_name: Same.
    :param tensorboard_dir_name: Same.
    :param num_epochs: Same.
    :param num_training_batches_per_epoch: Same.
    :param train_image_file_name_matrix: T-by-C numpy array of paths to training
        files with radar images.  This array should be created by
        `training_validation_io.find_2d_input_files`.
    :param top_target_directory_name: See doc for
        `training_validation_io.storm_image_generator_2d`.
    :param num_examples_per_batch: Same.
    :param num_examples_per_file_time: Same.
    :param target_name: Same.
    :param binarize_target: Same.
    :param sounding_statistic_names: Same.
    :param train_sounding_stat_file_names: length-T list of paths to training
        files with sounding statistics.  This list should be created by
        `training_validation_io.find_sounding_statistic_files`.
    :param weight_loss_function: See doc for `_check_training_args`.
    :param training_class_fraction_dict: Same.
    :param num_validation_batches_per_epoch: Same.
    :param validn_image_file_name_matrix: V-by-C numpy array of paths to
        validation files with radar images.  This array should be created by
        `training_validation_io.find_2d_input_files`.
    :param validn_sounding_stat_file_names: length-V list of paths to validation
        files with sounding statistics.  This list should be created by
        `training_validation_io.find_sounding_statistic_files`.
    :param validation_class_fraction_dict: See doc for `_check_training_args`.
    """

    class_weight_dict = _check_training_args(
        num_epochs=num_epochs,
        num_training_batches_per_epoch=num_training_batches_per_epoch,
        num_validation_batches_per_epoch=num_validation_batches_per_epoch,
        weight_loss_function=weight_loss_function,
        binarize_target=binarize_target,
        training_class_fraction_dict=training_class_fraction_dict,
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

    if sounding_statistic_names is None:
        sounding_stat_metadata_table = None
    else:
        sounding_stat_metadata_table = soundings.read_metadata_for_statistics()

    if num_validation_batches_per_epoch is None:
        checkpoint_object = keras.callbacks.ModelCheckpoint(
            filepath=model_file_name, monitor='loss', verbose=1,
            save_best_only=False, save_weights_only=False, mode='min', period=1)

        model_object.fit_generator(
            generator=trainval_io.storm_image_generator_2d(
                image_file_name_matrix=train_image_file_name_matrix,
                top_target_directory_name=top_target_directory_name,
                num_examples_per_batch=num_examples_per_batch,
                num_examples_per_file_time=num_examples_per_file_time,
                target_name=target_name, binarize_target=binarize_target,
                class_fraction_dict=training_class_fraction_dict,
                sounding_statistic_file_names=train_sounding_stat_file_names,
                sounding_statistic_names=sounding_statistic_names,
                sounding_stat_metadata_table=sounding_stat_metadata_table),
            steps_per_epoch=num_training_batches_per_epoch, epochs=num_epochs,
            verbose=1, class_weight=class_weight_dict,
            callbacks=[checkpoint_object, history_object, tensorboard_object])

    else:
        checkpoint_object = keras.callbacks.ModelCheckpoint(
            filepath=model_file_name, monitor='val_loss', verbose=1,
            save_best_only=True, save_weights_only=False, mode='min', period=1)

        model_object.fit_generator(
            generator=trainval_io.storm_image_generator_2d(
                image_file_name_matrix=train_image_file_name_matrix,
                top_target_directory_name=top_target_directory_name,
                num_examples_per_batch=num_examples_per_batch,
                num_examples_per_file_time=num_examples_per_file_time,
                target_name=target_name, binarize_target=binarize_target,
                class_fraction_dict=training_class_fraction_dict,
                sounding_statistic_file_names=train_sounding_stat_file_names,
                sounding_statistic_names=sounding_statistic_names,
                sounding_stat_metadata_table=sounding_stat_metadata_table),
            steps_per_epoch=num_training_batches_per_epoch, epochs=num_epochs,
            verbose=1, class_weight=class_weight_dict,
            callbacks=[checkpoint_object, history_object, tensorboard_object],
            validation_data=trainval_io.storm_image_generator_2d(
                image_file_name_matrix=validn_image_file_name_matrix,
                top_target_directory_name=top_target_directory_name,
                num_examples_per_batch=num_examples_per_batch,
                num_examples_per_file_time=num_examples_per_file_time,
                target_name=target_name, binarize_target=binarize_target,
                class_fraction_dict=validation_class_fraction_dict,
                sounding_statistic_file_names=validn_sounding_stat_file_names,
                sounding_statistic_names=sounding_statistic_names,
                sounding_stat_metadata_table=sounding_stat_metadata_table),
            validation_steps=num_validation_batches_per_epoch)


def train_2d_cnn_with_dynamic_sampling(
        model_object, model_file_name, num_epochs,
        num_training_batches_per_epoch, train_image_file_name_matrix,
        top_target_directory_name, num_examples_per_batch,
        num_examples_per_file_time, target_name,
        train_class_fraction_dict_by_epoch, binarize_target=False,
        sounding_statistic_names=None, train_sounding_stat_file_names=None,
        weight_loss_function=True, num_validation_batches_per_epoch=None,
        validn_image_file_name_matrix=None,
        validn_sounding_stat_file_names=None,
        validn_class_fraction_dict_by_epoch=None):
    """Trains 2-D CNN with dynamic class-conditional sampling.

    L = number of epochs

    :param model_object: See documentation for `train_2d_cnn`.
    :param model_file_name: Same.
    :param num_epochs: Same.
    :param num_training_batches_per_epoch: Same.
    :param train_image_file_name_matrix: Same.
    :param top_target_directory_name: Same.
    :param num_examples_per_batch: Same.
    :param num_examples_per_file_time: Same.
    :param target_name: Same.
    :param train_class_fraction_dict_by_epoch: length-L numpy array, where the
        [i]th item is a dictionary of sampling frequencies for training data at
        the [i]th epoch.  For more details, see doc for
        `training_class_fraction_dict` in `_check_training_args`.
    :param binarize_target: See documentation for `train_2d_cnn`.
    :param sounding_statistic_names: See doc for `train_2d_cnn`.
    :param train_sounding_stat_file_names: Same.
    :param weight_loss_function: Same.
    :param num_validation_batches_per_epoch: Same.
    :param validn_image_file_name_matrix: Same.
    :param validn_sounding_stat_file_names: Same.
    :param validn_class_fraction_dict_by_epoch: Same as
        `train_class_fraction_dict_by_epoch`, but for validation data.
    """

    error_checking.assert_is_integer(num_epochs)
    error_checking.assert_is_greater(num_epochs, 0)
    error_checking.assert_is_geq(
        len(train_class_fraction_dict_by_epoch), num_epochs)
    error_checking.assert_is_leq(
        len(train_class_fraction_dict_by_epoch), num_epochs)

    if validn_class_fraction_dict_by_epoch is not None:
        error_checking.assert_is_geq(
            len(validn_class_fraction_dict_by_epoch), num_epochs)
        error_checking.assert_is_leq(
            len(validn_class_fraction_dict_by_epoch), num_epochs)

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
            binarize_target=binarize_target,
            training_class_fraction_dict=train_class_fraction_dict_by_epoch[i],
            model_file_name=model_file_name,
            history_file_name=history_file_names[i],
            tensorboard_dir_name=tensorboard_dir_names[i])

    for i in range(num_epochs):
        train_2d_cnn(
            model_object=model_object, model_file_name=model_file_name,
            history_file_name=history_file_names[i],
            tensorboard_dir_name=tensorboard_dir_names[i], num_epochs=1,
            num_training_batches_per_epoch=num_training_batches_per_epoch,
            train_image_file_name_matrix=train_image_file_name_matrix,
            top_target_directory_name=top_target_directory_name,
            num_examples_per_batch=num_examples_per_batch,
            num_examples_per_file_time=num_examples_per_file_time,
            target_name=target_name, binarize_target=binarize_target,
            sounding_statistic_names=sounding_statistic_names,
            train_sounding_stat_file_names=train_sounding_stat_file_names,
            weight_loss_function=weight_loss_function,
            training_class_fraction_dict=train_class_fraction_dict_by_epoch[i],
            num_validation_batches_per_epoch=num_validation_batches_per_epoch,
            validn_image_file_name_matrix=validn_image_file_name_matrix,
            validn_sounding_stat_file_names=validn_sounding_stat_file_names,
            validation_class_fraction_dict=
            validn_class_fraction_dict_by_epoch[i])


def train_2d3d_cnn_with_myrorss(
        model_object, model_file_name, history_file_name, tensorboard_dir_name,
        num_epochs, num_training_batches_per_epoch,
        train_image_file_name_matrix, top_target_directory_name,
        num_examples_per_batch, num_examples_per_file_time, target_name,
        binarize_target=False, weight_loss_function=False,
        training_class_fraction_dict=None,
        num_validation_batches_per_epoch=None,
        validn_image_file_name_matrix=None,
        validation_class_fraction_dict=None):
    """Trains hybrid 2D/3D CNN with MYRORSS data.

    T = number of storm times (initial times) for training
    V = number of storm times for validation
    D = number of reflectivity heights
    F = number of azimuthal-shear fields

    :param model_object: See documentation for `train_2d_cnn`.
    :param model_file_name: Same.
    :param history_file_name: Same.
    :param tensorboard_dir_name: Same.
    :param num_epochs: Same.
    :param num_training_batches_per_epoch: Same.
    :param train_image_file_name_matrix: T-by-(F + D) numpy array of paths to
        training files with radar images.  This array should be created by
        `training_validation_io.find_2d_input_files`.
    :param top_target_directory_name: See doc for `train_2d_cnn`.
    :param num_examples_per_batch: Same.
    :param num_examples_per_file_time: Same.
    :param target_name: Same.
    :param binarize_target: Same.
    :param weight_loss_function: Same.
    :param training_class_fraction_dict: Same.
    :param num_validation_batches_per_epoch: Same.
    :param validn_image_file_name_matrix: V-by-(F + D) numpy array of paths to
        validation files with radar images.  This array should be created by
        `training_validation_io.find_2d_input_files`.
    :param validation_class_fraction_dict: See doc for `train_2d_cnn`.
    """

    class_weight_dict = _check_training_args(
        num_epochs=num_epochs,
        num_training_batches_per_epoch=num_training_batches_per_epoch,
        num_validation_batches_per_epoch=num_validation_batches_per_epoch,
        weight_loss_function=weight_loss_function,
        binarize_target=binarize_target,
        training_class_fraction_dict=training_class_fraction_dict,
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

    if num_validation_batches_per_epoch is None:
        checkpoint_object = keras.callbacks.ModelCheckpoint(
            filepath=model_file_name, monitor='loss', verbose=1,
            save_best_only=False, save_weights_only=False, mode='min', period=1)

        model_object.fit_generator(
            generator=trainval_io.storm_image_generator_2d3d_myrorss(
                image_file_name_matrix=train_image_file_name_matrix,
                top_target_directory_name=top_target_directory_name,
                num_examples_per_batch=num_examples_per_batch,
                num_examples_per_file_time=num_examples_per_file_time,
                target_name=target_name, binarize_target=binarize_target,
                class_fraction_dict=training_class_fraction_dict),
            steps_per_epoch=num_training_batches_per_epoch, epochs=num_epochs,
            verbose=1, class_weight=class_weight_dict,
            callbacks=[checkpoint_object, history_object, tensorboard_object])

    else:
        checkpoint_object = keras.callbacks.ModelCheckpoint(
            filepath=model_file_name, monitor='val_loss', verbose=1,
            save_best_only=True, save_weights_only=False, mode='min', period=1)

        model_object.fit_generator(
            generator=trainval_io.storm_image_generator_2d3d_myrorss(
                image_file_name_matrix=train_image_file_name_matrix,
                top_target_directory_name=top_target_directory_name,
                num_examples_per_batch=num_examples_per_batch,
                num_examples_per_file_time=num_examples_per_file_time,
                target_name=target_name, binarize_target=binarize_target,
                class_fraction_dict=training_class_fraction_dict),
            steps_per_epoch=num_training_batches_per_epoch, epochs=num_epochs,
            verbose=1, class_weight=class_weight_dict,
            callbacks=[checkpoint_object, history_object, tensorboard_object],
            validation_data=
            trainval_io.storm_image_generator_2d3d_myrorss(
                image_file_name_matrix=validn_image_file_name_matrix,
                top_target_directory_name=top_target_directory_name,
                num_examples_per_batch=num_examples_per_batch,
                num_examples_per_file_time=num_examples_per_file_time,
                target_name=target_name, binarize_target=binarize_target,
                class_fraction_dict=validation_class_fraction_dict),
            validation_steps=num_validation_batches_per_epoch)


def train_2d3d_cnn_with_dynamic_sampling(
        model_object, model_file_name, num_epochs,
        num_training_batches_per_epoch, train_image_file_name_matrix,
        top_target_directory_name, num_examples_per_batch,
        num_examples_per_file_time, target_name,
        train_class_fraction_dict_by_epoch, binarize_target=False,
        weight_loss_function=True, num_validation_batches_per_epoch=None,
        validn_image_file_name_matrix=None,
        validn_class_fraction_dict_by_epoch=None):
    """Trains hybrid 2D/3D CNN with dynamic class-conditional sampling.

    L = number of epochs

    :param model_object: See documentation for `train_2d3d_cnn_with_myrorss`.
    :param model_file_name: Same.
    :param num_epochs: Same.
    :param num_training_batches_per_epoch: Same.
    :param train_image_file_name_matrix: Same.
    :param top_target_directory_name: Same.
    :param num_examples_per_batch: Same.
    :param num_examples_per_file_time: Same.
    :param target_name: Same.
    :param train_class_fraction_dict_by_epoch: length-L numpy array, where the
        [i]th item is a dictionary of sampling frequencies for training data at
        the [i]th epoch.  For more details, see doc for
        `training_class_fraction_dict` in `_check_training_args`.
    :param binarize_target: See documentation for `train_2d3d_cnn_with_myrorss`.
    :param weight_loss_function: See doc for `train_2d3d_cnn_with_myrorss`.
    :param num_validation_batches_per_epoch: Same.
    :param validn_image_file_name_matrix: Same.
    :param validn_class_fraction_dict_by_epoch: Same as
        `train_class_fraction_dict_by_epoch`, but for validation data.
    """

    error_checking.assert_is_integer(num_epochs)
    error_checking.assert_is_greater(num_epochs, 0)
    error_checking.assert_is_geq(
        len(train_class_fraction_dict_by_epoch), num_epochs)
    error_checking.assert_is_leq(
        len(train_class_fraction_dict_by_epoch), num_epochs)

    if validn_class_fraction_dict_by_epoch is None:
        error_checking.assert_is_geq(
            len(validn_class_fraction_dict_by_epoch), num_epochs)
        error_checking.assert_is_leq(
            len(validn_class_fraction_dict_by_epoch), num_epochs)

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
            binarize_target=binarize_target,
            training_class_fraction_dict=train_class_fraction_dict_by_epoch[i],
            model_file_name=model_file_name,
            history_file_name=history_file_names[i],
            tensorboard_dir_name=tensorboard_dir_names[i])

    for i in range(num_epochs):
        train_2d3d_cnn_with_myrorss(
            model_object=model_object, model_file_name=model_file_name,
            history_file_name=history_file_names[i],
            tensorboard_dir_name=tensorboard_dir_names[i], num_epochs=1,
            num_training_batches_per_epoch=num_training_batches_per_epoch,
            train_image_file_name_matrix=train_image_file_name_matrix,
            top_target_directory_name=top_target_directory_name,
            num_examples_per_batch=num_examples_per_batch,
            num_examples_per_file_time=num_examples_per_file_time,
            target_name=target_name, binarize_target=binarize_target,
            weight_loss_function=weight_loss_function,
            training_class_fraction_dict=train_class_fraction_dict_by_epoch[i],
            num_validation_batches_per_epoch=num_validation_batches_per_epoch,
            validn_image_file_name_matrix=validn_image_file_name_matrix,
            validation_class_fraction_dict=
            validn_class_fraction_dict_by_epoch[i])


def train_3d_cnn(
        model_object, model_file_name, history_file_name, tensorboard_dir_name,
        num_epochs, num_training_batches_per_epoch,
        train_image_file_name_matrix, top_target_directory_name,
        num_examples_per_batch, num_examples_per_file_time, target_name,
        binarize_target=False, sounding_statistic_names=None,
        train_sounding_stat_file_names=None, weight_loss_function=False,
        training_class_fraction_dict=None,
        num_validation_batches_per_epoch=None,
        validn_image_file_name_matrix=None,
        validn_sounding_stat_file_names=None,
        validation_class_fraction_dict=None):
    """Trains 3-D CNN (one that performs 3-D convolution).

    T = number of storm times (initial times) for training
    V = number of storm times for validation
    F = number of radar fields
    D = number of radar heights

    :param model_object: See documentation for `train_2d_cnn`.
    :param model_file_name: Same.
    :param history_file_name: Same.
    :param tensorboard_dir_name: Same.
    :param num_epochs: Same.
    :param num_training_batches_per_epoch: Same.
    :param train_image_file_name_matrix: T-by-F-by-D numpy array of paths to
        training files with radar images.  This array should be created by
        `training_validation_io.find_3d_input_files`.
    :param top_target_directory_name: See doc for `train_2d_cnn`.
    :param num_examples_per_batch: Same.
    :param num_examples_per_file_time: Same.
    :param target_name: Same.
    :param binarize_target: Same.
    :param sounding_statistic_names: Same.
    :param train_sounding_stat_file_names: Same.
    :param weight_loss_function: Same.
    :param training_class_fraction_dict: Same.
    :param num_validation_batches_per_epoch: Same.
    :param validn_image_file_name_matrix: V-by-F-by-D numpy array of paths to
        validation files with radar images.  This array should be created by
        `training_validation_io.find_3d_input_files`.
    :param validn_sounding_stat_file_names: See doc for `train_2d_cnn`.
    :param validation_class_fraction_dict: See doc for `train_2d_cnn`.
    """

    class_weight_dict = _check_training_args(
        num_epochs=num_epochs,
        num_training_batches_per_epoch=num_training_batches_per_epoch,
        num_validation_batches_per_epoch=num_validation_batches_per_epoch,
        weight_loss_function=weight_loss_function,
        binarize_target=binarize_target,
        training_class_fraction_dict=training_class_fraction_dict,
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

    if sounding_statistic_names is None:
        sounding_stat_metadata_table = None
    else:
        sounding_stat_metadata_table = soundings.read_metadata_for_statistics()

    if num_validation_batches_per_epoch is None:
        checkpoint_object = keras.callbacks.ModelCheckpoint(
            filepath=model_file_name, monitor='loss', verbose=1,
            save_best_only=False, save_weights_only=False, mode='min', period=1)

        model_object.fit_generator(
            generator=trainval_io.storm_image_generator_3d(
                image_file_name_matrix=train_image_file_name_matrix,
                top_target_directory_name=top_target_directory_name,
                num_examples_per_batch=num_examples_per_batch,
                num_examples_per_file_time=num_examples_per_file_time,
                target_name=target_name, binarize_target=binarize_target,
                class_fraction_dict=training_class_fraction_dict,
                sounding_statistic_file_names=train_sounding_stat_file_names,
                sounding_statistic_names=sounding_statistic_names,
                sounding_stat_metadata_table=sounding_stat_metadata_table),
            steps_per_epoch=num_training_batches_per_epoch, epochs=num_epochs,
            verbose=1, class_weight=class_weight_dict,
            callbacks=[checkpoint_object, history_object, tensorboard_object])

    else:
        checkpoint_object = keras.callbacks.ModelCheckpoint(
            filepath=model_file_name, monitor='val_loss', verbose=1,
            save_best_only=True, save_weights_only=False, mode='min', period=1)

        model_object.fit_generator(
            generator=trainval_io.storm_image_generator_3d(
                image_file_name_matrix=train_image_file_name_matrix,
                top_target_directory_name=top_target_directory_name,
                num_examples_per_batch=num_examples_per_batch,
                num_examples_per_file_time=num_examples_per_file_time,
                target_name=target_name, binarize_target=binarize_target,
                class_fraction_dict=training_class_fraction_dict,
                sounding_statistic_file_names=train_sounding_stat_file_names,
                sounding_statistic_names=sounding_statistic_names,
                sounding_stat_metadata_table=sounding_stat_metadata_table),
            steps_per_epoch=num_training_batches_per_epoch, epochs=num_epochs,
            verbose=1, class_weight=class_weight_dict,
            callbacks=[checkpoint_object, history_object, tensorboard_object],
            validation_data=trainval_io.storm_image_generator_3d(
                image_file_name_matrix=validn_image_file_name_matrix,
                top_target_directory_name=top_target_directory_name,
                num_examples_per_batch=num_examples_per_batch,
                num_examples_per_file_time=num_examples_per_file_time,
                target_name=target_name, binarize_target=binarize_target,
                class_fraction_dict=validation_class_fraction_dict,
                sounding_statistic_file_names=validn_sounding_stat_file_names,
                sounding_statistic_names=sounding_statistic_names,
                sounding_stat_metadata_table=sounding_stat_metadata_table),
            validation_steps=num_validation_batches_per_epoch)


def train_3d_cnn_with_dynamic_sampling(
        model_object, model_file_name, num_epochs,
        num_training_batches_per_epoch, train_image_file_name_matrix,
        top_target_directory_name, num_examples_per_batch,
        num_examples_per_file_time, target_name,
        train_class_fraction_dict_by_epoch, binarize_target=False,
        sounding_statistic_names=None, train_sounding_stat_file_names=None,
        weight_loss_function=True, num_validation_batches_per_epoch=None,
        validn_image_file_name_matrix=None,
        validn_sounding_stat_file_names=None,
        validn_class_fraction_dict_by_epoch=None):
    """Trains 3-D CNN with dynamic class-conditional sampling.

    L = number of epochs

    :param model_object: See documentation for `train_3d_cnn`.
    :param model_file_name: Same.
    :param num_epochs: Same.
    :param num_training_batches_per_epoch: Same.
    :param train_image_file_name_matrix: Same.
    :param top_target_directory_name: Same.
    :param num_examples_per_batch: Same.
    :param num_examples_per_file_time: Same.
    :param target_name: Same.
    :param train_class_fraction_dict_by_epoch: length-L numpy array, where the
        [i]th item is a dictionary of sampling frequencies for training data at
        the [i]th epoch.  For more details, see doc for
        `training_class_fraction_dict` in `_check_training_args`.
    :param binarize_target: See doc for `train_3d_cnn`.
    :param sounding_statistic_names: Same.
    :param train_sounding_stat_file_names: Same.
    :param weight_loss_function: Same.
    :param num_validation_batches_per_epoch: Same.
    :param validn_image_file_name_matrix: Same.
    :param validn_sounding_stat_file_names: Same.
    :param validn_class_fraction_dict_by_epoch: Same as
        `train_class_fraction_dict_by_epoch`, but for validation data.
    """

    error_checking.assert_is_integer(num_epochs)
    error_checking.assert_is_greater(num_epochs, 0)
    error_checking.assert_is_geq(
        len(train_class_fraction_dict_by_epoch), num_epochs)
    error_checking.assert_is_leq(
        len(train_class_fraction_dict_by_epoch), num_epochs)

    if validn_class_fraction_dict_by_epoch is not None:
        error_checking.assert_is_geq(
            len(validn_class_fraction_dict_by_epoch), num_epochs)
        error_checking.assert_is_leq(
            len(validn_class_fraction_dict_by_epoch), num_epochs)

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
            binarize_target=binarize_target,
            training_class_fraction_dict=train_class_fraction_dict_by_epoch[i],
            model_file_name=model_file_name,
            history_file_name=history_file_names[i],
            tensorboard_dir_name=tensorboard_dir_names[i])

    for i in range(num_epochs):
        train_3d_cnn(
            model_object=model_object, model_file_name=model_file_name,
            history_file_name=history_file_names[i],
            tensorboard_dir_name=tensorboard_dir_names[i], num_epochs=1,
            num_training_batches_per_epoch=num_training_batches_per_epoch,
            train_image_file_name_matrix=train_image_file_name_matrix,
            top_target_directory_name=top_target_directory_name,
            num_examples_per_batch=num_examples_per_batch,
            num_examples_per_file_time=num_examples_per_file_time,
            target_name=target_name, binarize_target=binarize_target,
            sounding_statistic_names=sounding_statistic_names,
            train_sounding_stat_file_names=train_sounding_stat_file_names,
            weight_loss_function=weight_loss_function,
            training_class_fraction_dict=train_class_fraction_dict_by_epoch[i],
            num_validation_batches_per_epoch=num_validation_batches_per_epoch,
            validn_image_file_name_matrix=validn_image_file_name_matrix,
            validn_sounding_stat_file_names=validn_sounding_stat_file_names,
            validation_class_fraction_dict=
            validn_class_fraction_dict_by_epoch[i])


def apply_2d_cnn(model_object, image_matrix, sounding_stat_matrix=None):
    """Applies 2-D CNN (one that performs 2-D convolution) to new examples.

    :param model_object: Instance of `keras.models.Sequential`.
    :param image_matrix: E-by-M-by-N-by-C numpy array of storm-centered radar
        images.
    :param sounding_stat_matrix: E-by-S numpy array of sounding statistics.
    :return: class_probability_matrix: E-by-K numpy array of forecast
        probabilities.  class_probability_matrix[i, k] is the forecast
        probability the [i]th example belongs to the [k]th class.
    """

    dl_utils.check_predictor_matrix(
        predictor_matrix=image_matrix, min_num_dimensions=4,
        max_num_dimensions=4)

    num_examples = image_matrix.shape[0]
    if sounding_stat_matrix is None:
        return model_object.predict(image_matrix, batch_size=num_examples)

    dl_utils.check_sounding_stat_matrix(
        sounding_stat_matrix=sounding_stat_matrix, num_examples=num_examples)
    return model_object.predict(
        [image_matrix, sounding_stat_matrix], batch_size=num_examples)


def apply_3d_cnn(model_object, image_matrix, sounding_stat_matrix=None):
    """Applies 3-D CNN (one that performs 3-D convolution) to new examples.

    :param model_object: Instance of `keras.models.Sequential`.
    :param image_matrix: E-by-M-by-N-by-D-by-C numpy array of storm-centered
        radar images.
    :param sounding_stat_matrix: E-by-S numpy array of sounding statistics.
    :return: class_probability_matrix: E-by-K numpy array of forecast
        probabilities.  class_probability_matrix[i, k] is the forecast
        probability the [i]th example belongs to the [k]th class.
    """

    dl_utils.check_predictor_matrix(
        predictor_matrix=image_matrix, min_num_dimensions=5,
        max_num_dimensions=5)

    num_examples = image_matrix.shape[0]
    if sounding_stat_matrix is None:
        return model_object.predict(image_matrix, batch_size=num_examples)

    dl_utils.check_sounding_stat_matrix(
        sounding_stat_matrix=sounding_stat_matrix, num_examples=num_examples)
    return model_object.predict(
        [image_matrix, sounding_stat_matrix], batch_size=num_examples)


def apply_2d3d_cnn_to_myrorss(
        model_object, reflectivity_image_matrix_dbz, az_shear_image_matrix_s01):
    """Applies hybrid 2D/3D CNN to MYRORSS data.

    m = number of pixel rows per reflectivity image
    n = number of pixel columns per reflectivity image
    D = number of pixel heights (depths) per reflectivity image
    M = number of pixel rows per azimuthal-shear image
    N = number of pixel columns per azimuthal-shear image
    F = number of azimuthal-shear fields

    :param model_object: Instance of `keras.models.Sequential`.
    :param reflectivity_image_matrix_dbz: E-by-m-by-n-by-D-by-1 numpy array of
        storm-centered reflectivity images.
    :param az_shear_image_matrix_s01: E-by-M-by-N-by-F numpy array of storm-
        centered azimuthal-shear images.
    :return: class_probability_matrix: E-by-K numpy array of forecast
        probabilities.  class_probability_matrix[i, k] is the forecast
        probability the [i]th example belongs to the [k]th class.
    """

    dl_utils.check_predictor_matrix(
        predictor_matrix=reflectivity_image_matrix_dbz, min_num_dimensions=5,
        max_num_dimensions=5)
    dl_utils.check_predictor_matrix(
        predictor_matrix=az_shear_image_matrix_s01, min_num_dimensions=4,
        max_num_dimensions=4)

    expected_dimensions = reflectivity_image_matrix_dbz.shape[:-1] + (1,)
    error_checking.assert_is_numpy_array(
        reflectivity_image_matrix_dbz, exact_dimensions=expected_dimensions)

    num_examples = reflectivity_image_matrix_dbz.shape[0]
    expected_dimensions = (num_examples,) + az_shear_image_matrix_s01[1:]
    error_checking.assert_is_numpy_array(
        az_shear_image_matrix_s01, exact_dimensions=expected_dimensions)

    return model_object.predict(
        [reflectivity_image_matrix_dbz, az_shear_image_matrix_s01],
        batch_size=num_examples)
