"""Methods for creating, training, and applying a CNN*.

* convolutional neural network

--- NOTATION ---

The following letters will be used throughout this module.

E = number of examples (storm objects)
M = number of rows per radar image
N = number of columns per radar image
H_r = number of heights per radar image
F_r = number of radar fields (not including different heights)
H_s = number of vertical levels per sounding
F_s = number of sounding fields (not including different vertical levels)
C = number of field/height pairs per radar image
K = number of classes for target variable
T = number of file times (time steps or SPC dates)
"""

import pickle
import numpy
import keras.losses
import keras.optimizers
import keras.models
import keras.callbacks
from gewittergefahr.deep_learning import deep_learning_utils as dl_utils
from gewittergefahr.deep_learning import cnn_utils
from gewittergefahr.deep_learning import keras_metrics
from gewittergefahr.deep_learning import training_validation_io as trainval_io
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.gg_utils import error_checking

VALID_NUMBERS_OF_RADAR_ROWS = numpy.array([16, 32, 64], dtype=int)
VALID_NUMBERS_OF_RADAR_COLUMNS = numpy.array([16, 32, 64], dtype=int)
VALID_NUMBERS_OF_RADAR_HEIGHTS = numpy.array([12], dtype=int)
VALID_NUMBERS_OF_SOUNDING_HEIGHTS = numpy.array([37], dtype=int)

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
    keras_metrics.binary_success_ratio, keras_metrics.binary_focn
]

NUM_EPOCHS_KEY = 'num_epochs'
NUM_EXAMPLES_PER_BATCH_KEY = 'num_examples_per_batch'
NUM_EXAMPLES_PER_FILE_TIME_KEY = 'num_examples_per_time'
NUM_TRAINING_BATCHES_KEY = 'num_training_batches_per_epoch'
FIRST_TRAINING_TIME_KEY = 'first_train_time_unix_sec'
LAST_TRAINING_TIME_KEY = 'last_train_time_unix_sec'
WEIGHT_LOSS_FUNCTION_KEY = 'weight_loss_function'
TARGET_NAME_KEY = 'target_name'
BINARIZE_TARGET_KEY = 'binarize_target'
RADAR_SOURCE_KEY = 'radar_source'
RADAR_FIELD_NAMES_KEY = 'radar_field_names'
RADAR_NORMALIZATION_DICT_KEY = 'radar_normalization_dict'
RADAR_HEIGHTS_KEY = 'radar_heights_m_asl'
REFLECTIVITY_HEIGHTS_KEY = 'reflectivity_heights_m_asl'
TRAINING_FRACTION_BY_CLASS_KEY = 'training_fraction_by_class_dict'
VALIDATION_FRACTION_BY_CLASS_KEY = 'validation_fraction_by_class_dict'
NUM_VALIDATION_BATCHES_KEY = 'num_validation_batches_per_epoch'
FIRST_VALIDATION_TIME_KEY = 'first_validn_time_unix_sec'
LAST_VALIDATION_TIME_KEY = 'last_validn_time_unix_sec'
SOUNDING_FIELD_NAMES_KEY = 'sounding_field_names'
SOUNDING_NORMALIZATION_DICT_KEY = 'sounding_normalization_dict'
SOUNDING_LAG_TIME_KEY = 'sounding_lag_time_for_convective_contamination_sec'

MODEL_METADATA_KEYS = [
    NUM_EPOCHS_KEY, NUM_EXAMPLES_PER_BATCH_KEY, NUM_EXAMPLES_PER_FILE_TIME_KEY,
    NUM_TRAINING_BATCHES_KEY, FIRST_TRAINING_TIME_KEY, LAST_TRAINING_TIME_KEY,
    WEIGHT_LOSS_FUNCTION_KEY, TARGET_NAME_KEY, BINARIZE_TARGET_KEY,
    RADAR_SOURCE_KEY, RADAR_FIELD_NAMES_KEY, RADAR_NORMALIZATION_DICT_KEY,
    RADAR_HEIGHTS_KEY, REFLECTIVITY_HEIGHTS_KEY, TRAINING_FRACTION_BY_CLASS_KEY,
    VALIDATION_FRACTION_BY_CLASS_KEY, NUM_VALIDATION_BATCHES_KEY,
    FIRST_VALIDATION_TIME_KEY, LAST_VALIDATION_TIME_KEY,
    SOUNDING_FIELD_NAMES_KEY, SOUNDING_NORMALIZATION_DICT_KEY,
    SOUNDING_LAG_TIME_KEY
]


def _check_architecture_args(
        num_radar_rows, num_radar_columns, num_radar_dimensions, num_classes,
        num_radar_channels=None, num_radar_fields=None, num_radar_heights=None,
        dropout_fraction=None, l2_weight=None, num_sounding_heights=None,
        num_sounding_fields=None):
    """Error-checking of input args for model architecture.

    :param num_radar_rows: Number of pixel rows per storm-centered radar image.
    :param num_radar_columns: Number of pixel columns per radar image.
    :param num_radar_dimensions: Number of dimensions per radar image.
    :param num_classes: Number of classes for target variable.
    :param num_radar_channels: [used only if num_radar_dimensions = 2]
        Number of channels (field/height pairs) per radar image.
    :param num_radar_fields: [used only if num_radar_dimensions = 3]
        Number of fields per radar image.
    :param num_radar_heights: [used only if num_radar_dimensions = 3]
        Number of pixel heights per radar image.
    :param dropout_fraction: Dropout fraction.  Will be used for each dropout
        layer, of which there is one after each convolutional layer.
    :param l2_weight: L2-regularization weight.  Will be used for each
        convolutional layer.
    :param num_sounding_heights: Number of heights per storm-centered sounding.
    :param num_sounding_fields: Number of fields per sounding.
    :raises: ValueError: if `num_radar_rows not in VALID_NUMBERS_OF_RADAR_ROWS`.
    :raises: ValueError: if
        `num_radar_columns not in VALID_NUMBERS_OF_RADAR_COLUMNS`.
    :raises: ValueError: if
        `num_radar_heights not in VALID_NUMBERS_OF_RADAR_HEIGHTS`.
    :raises: ValueError: if
        `num_sounding_heights not in VALID_NUMBERS_OF_SOUNDING_HEIGHTS`.
    """

    error_checking.assert_is_integer(num_radar_rows)
    if num_radar_rows not in VALID_NUMBERS_OF_RADAR_ROWS:
        error_string = (
            '\n\n{0:s}\nValid numbers of radar rows (listed above) do not '
            'include {1:d}.'
        ).format(str(VALID_NUMBERS_OF_RADAR_ROWS), num_radar_rows)
        raise ValueError(error_string)

    error_checking.assert_is_integer(num_radar_columns)
    if num_radar_columns not in VALID_NUMBERS_OF_RADAR_COLUMNS:
        error_string = (
            '\n\n{0:s}\nValid numbers of radar columns (listed above) do not '
            'include {1:d}.'
        ).format(str(VALID_NUMBERS_OF_RADAR_COLUMNS), num_radar_columns)
        raise ValueError(error_string)

    error_checking.assert_is_integer(num_classes)
    error_checking.assert_is_geq(num_classes, 2)
    error_checking.assert_is_integer(num_radar_dimensions)
    error_checking.assert_is_geq(num_radar_dimensions, 2)
    error_checking.assert_is_leq(num_radar_dimensions, 3)

    if num_radar_dimensions == 2:
        error_checking.assert_is_integer(num_radar_channels)
        error_checking.assert_is_geq(num_radar_channels, 1)
    else:
        error_checking.assert_is_integer(num_radar_fields)
        error_checking.assert_is_geq(num_radar_fields, 1)
        error_checking.assert_is_integer(num_radar_heights)

        if num_radar_heights not in VALID_NUMBERS_OF_RADAR_HEIGHTS:
            error_string = (
                '\n\n{0:s}\nValid numbers of radar heights (listed above) do '
                'not include {1:d}.'
            ).format(str(VALID_NUMBERS_OF_RADAR_HEIGHTS), num_radar_heights)
            raise ValueError(error_string)

    if dropout_fraction is not None:
        error_checking.assert_is_greater(dropout_fraction, 0.)
        error_checking.assert_is_less_than(dropout_fraction, 1.)

    if l2_weight is not None:
        error_checking.assert_is_greater(l2_weight, 0.)

    if num_sounding_heights is not None or num_sounding_fields is not None:
        error_checking.assert_is_integer(num_sounding_fields)
        error_checking.assert_is_geq(num_sounding_fields, 1)
        error_checking.assert_is_integer(num_sounding_heights)

        if num_sounding_heights not in VALID_NUMBERS_OF_SOUNDING_HEIGHTS:
            error_string = (
                '\n\n{0:s}\nValid numbers of sounding heights (listed above) do'
                ' not include {1:d}.'
            ).format(str(VALID_NUMBERS_OF_SOUNDING_HEIGHTS),
                     num_sounding_heights)
            raise ValueError(error_string)


def _check_training_args(
        num_epochs, num_training_batches_per_epoch,
        num_validation_batches_per_epoch, weight_loss_function,
        binarize_target, training_fraction_by_class_dict, model_file_name,
        history_file_name, tensorboard_dir_name):
    """Error-checking of input args for training.

    :param num_epochs: Number of epochs.
    :param num_training_batches_per_epoch: Number of training batches per epoch.
    :param num_validation_batches_per_epoch: Number of validation batches per
        epoch.
    :param weight_loss_function: Boolean flag.  If False, classes will be
        weighted equally in the loss function.  If True, classes will be
        weighted differently in the loss function (inversely proportional to
        their sampling fractions, specified in
        `training_fraction_by_class_dict`).
    :param binarize_target: Boolean flag.  If True, will binarize target
        variable, so that the highest class becomes 1 and all other classes
        become 0.
    :param training_fraction_by_class_dict: See doc for
        `training_validation_io.storm_image_generator_2d`.
    :param model_file_name: Path to output file (HDF5 format).  Model will be
        saved here after each epoch.
    :param history_file_name: Path to output file (CSV format).  Training
        history will be saved here after each epoch.
    :param tensorboard_dir_name: Path to output directory for TensorBoard log
        files.
    :return: loss_function_weight_by_class_dict: See doc for
        `deep_learning_utils.class_fractions_to_weights`.
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
        loss_function_weight_by_class_dict = (
            dl_utils.class_fractions_to_weights(
                class_fraction_dict=training_fraction_by_class_dict,
                binarize_target=binarize_target))
    else:
        loss_function_weight_by_class_dict = None

    file_system_utils.mkdir_recursive_if_necessary(file_name=model_file_name)
    file_system_utils.mkdir_recursive_if_necessary(file_name=history_file_name)
    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=tensorboard_dir_name)

    return loss_function_weight_by_class_dict


def _get_sounding_layers(
        num_sounding_heights, num_sounding_fields,
        num_filters_in_first_conv_layer, dropout_fraction, regularizer_object):
    """Returns the part of a CNN that convolves over sounding data.

    :param num_sounding_heights: See doc for `_check_architecture_args`.
    :param num_sounding_fields: Same.
    :param num_filters_in_first_conv_layer: See doc for
        `get_2d_swirlnet_architecture`.
    :param dropout_fraction: See doc for `_check_architecture_args`.
    :param regularizer_object: Instance of `keras.regularizers.l1_l2`.  Will be
        applied to each convolutional layer.
    :return: input_layer_object: Instance of `keras.layers.Input`.
    :return: flattening_layer_object: Instance of `keras.layers.Flatten`.
    """

    sounding_input_layer_object = keras.layers.Input(shape=(
        num_sounding_heights, num_sounding_fields))

    sounding_layer_object = cnn_utils.get_1d_conv_layer(
        num_output_filters=num_filters_in_first_conv_layer, num_kernel_pixels=3,
        num_pixels_per_stride=1, padding_type=cnn_utils.YES_PADDING_TYPE,
        kernel_weight_regularizer=regularizer_object,
        activation_function='relu', is_first_layer=False
    )(sounding_input_layer_object)

    if dropout_fraction is not None:
        sounding_layer_object = cnn_utils.get_dropout_layer(
            dropout_fraction=dropout_fraction
        )(sounding_layer_object)

    sounding_layer_object = cnn_utils.get_1d_pooling_layer(
        num_pixels_in_window=2, pooling_type=cnn_utils.MEAN_POOLING_TYPE,
        num_pixels_per_stride=2
    )(sounding_layer_object)

    sounding_layer_object = cnn_utils.get_1d_conv_layer(
        num_output_filters=2 * num_filters_in_first_conv_layer,
        num_kernel_pixels=3, num_pixels_per_stride=1,
        padding_type=cnn_utils.YES_PADDING_TYPE,
        kernel_weight_regularizer=regularizer_object,
        activation_function='relu', is_first_layer=False
    )(sounding_layer_object)

    if dropout_fraction is not None:
        sounding_layer_object = cnn_utils.get_dropout_layer(
            dropout_fraction=dropout_fraction
        )(sounding_layer_object)

    sounding_layer_object = cnn_utils.get_1d_pooling_layer(
        num_pixels_in_window=2, pooling_type=cnn_utils.MEAN_POOLING_TYPE,
        num_pixels_per_stride=2
    )(sounding_layer_object)

    sounding_layer_object = cnn_utils.get_1d_conv_layer(
        num_output_filters=4 * num_filters_in_first_conv_layer,
        num_kernel_pixels=3, num_pixels_per_stride=1,
        padding_type=cnn_utils.YES_PADDING_TYPE,
        kernel_weight_regularizer=regularizer_object,
        activation_function='relu', is_first_layer=False
    )(sounding_layer_object)

    if dropout_fraction is not None:
        sounding_layer_object = cnn_utils.get_dropout_layer(
            dropout_fraction=dropout_fraction
        )(sounding_layer_object)

    sounding_layer_object = cnn_utils.get_1d_pooling_layer(
        num_pixels_in_window=2, pooling_type=cnn_utils.MEAN_POOLING_TYPE,
        num_pixels_per_stride=2
    )(sounding_layer_object)

    sounding_layer_object = cnn_utils.get_flattening_layer()(
        sounding_layer_object)
    return sounding_input_layer_object, sounding_layer_object


def get_2d_mnist_architecture(
        num_radar_rows, num_radar_columns, num_radar_channels, num_classes,
        num_filters_in_first_conv_layer=32, l2_weight=None):
    """Creates CNN with architecture similar to the following example.

    https://github.com/keras-team/keras/blob/master/examples/mnist_cnn.py

    This CNN does 2-D convolution only.

    :param num_radar_rows: See doc for `_check_architecture_args`.
    :param num_radar_columns: Same.
    :param num_radar_channels: Same.
    :param num_classes: Same.
    :param num_filters_in_first_conv_layer: Number of filters in first
        convolutional layer.  Number of filters will double for each successive
        convolutional layer.
    :param l2_weight: See doc for `_check_architecture_args`.
    :return: model_object: `keras.models.Sequential` object with the
        aforementioned architecture.
    """

    _check_architecture_args(
        num_radar_rows=num_radar_rows, num_radar_columns=num_radar_columns,
        num_radar_dimensions=2, num_classes=num_classes,
        num_radar_channels=num_radar_channels, l2_weight=l2_weight)

    if l2_weight is None:
        regularizer_object = None
    else:
        regularizer_object = cnn_utils.get_weight_regularizer(
            l1_penalty=0, l2_penalty=l2_weight)

    model_object = keras.models.Sequential()
    layer_object = cnn_utils.get_2d_conv_layer(
        num_output_filters=num_filters_in_first_conv_layer, num_kernel_rows=3,
        num_kernel_columns=3, num_rows_per_stride=1, num_columns_per_stride=1,
        padding_type=cnn_utils.NO_PADDING_TYPE, activation_function='relu',
        kernel_weight_regularizer=regularizer_object,
        is_first_layer=True, num_input_rows=num_radar_rows,
        num_input_columns=num_radar_columns,
        num_input_channels=num_radar_channels)
    model_object.add(layer_object)

    layer_object = cnn_utils.get_2d_conv_layer(
        num_output_filters=2 * num_filters_in_first_conv_layer,
        num_kernel_rows=3, num_kernel_columns=3, num_rows_per_stride=1,
        num_columns_per_stride=1, padding_type=cnn_utils.NO_PADDING_TYPE,
        activation_function='relu',
        kernel_weight_regularizer=regularizer_object)
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


def get_2d_swilrnet_architecture(
        num_radar_rows, num_radar_columns, num_radar_channels, num_classes,
        num_radar_filters_in_first_conv_layer=16, dropout_fraction=0.5,
        l2_weight=DEFAULT_L2_WEIGHT, num_sounding_heights=None,
        num_sounding_fields=None, num_sounding_filters_in_first_conv_layer=48):
    """Creates CNN with architecture similar to the following example.

    https://github.com/djgagne/swirlnet/blob/master/notebooks/
    deep_swirl_tutorial.ipynb

    This CNN does 2-D convolution only.

    :param num_radar_rows: See doc for `_check_architecture_args`.
    :param num_radar_columns: Same.
    :param num_radar_channels: Same.
    :param num_classes: Same.
    :param num_radar_filters_in_first_conv_layer: Number of filters in first
        convolutional layer for radar images.  Number of filters will double for
        each successive layer convolving over radar images.
    :param dropout_fraction: See doc for `_check_architecture_args`.
    :param l2_weight: Same.
    :param num_sounding_heights: Same.
    :param num_sounding_fields: Same.
    :param num_sounding_filters_in_first_conv_layer: Number of filters in first
        convolutional layer for soundings.  Number of filters will double for
        each successive layer convolving over soundings.
    :return: model_object: `keras.models.Sequential` object with the
        aforementioned architecture.
    """

    _check_architecture_args(
        num_radar_rows=num_radar_rows, num_radar_columns=num_radar_columns,
        num_radar_dimensions=2, num_classes=num_classes,
        num_radar_channels=num_radar_channels,
        dropout_fraction=dropout_fraction, l2_weight=l2_weight,
        num_sounding_heights=num_sounding_heights,
        num_sounding_fields=num_sounding_fields)

    if l2_weight is None:
        regularizer_object = None
    else:
        regularizer_object = cnn_utils.get_weight_regularizer(
            l1_penalty=0, l2_penalty=l2_weight)

    radar_input_layer_object = keras.layers.Input(
        shape=(num_radar_rows, num_radar_columns, num_radar_channels))
    this_num_output_filters = num_radar_filters_in_first_conv_layer + 0

    for i in range(3):
        if i == 0:
            this_input_layer_object = radar_input_layer_object
        else:
            this_input_layer_object = radar_layer_object
            this_num_output_filters *= 2

        radar_layer_object = cnn_utils.get_2d_conv_layer(
            num_output_filters=this_num_output_filters,
            num_kernel_rows=5, num_kernel_columns=5, num_rows_per_stride=1,
            num_columns_per_stride=1, padding_type=cnn_utils.YES_PADDING_TYPE,
            kernel_weight_regularizer=regularizer_object,
            activation_function='relu', is_first_layer=False
        )(this_input_layer_object)

        if dropout_fraction is not None:
            radar_layer_object = cnn_utils.get_dropout_layer(
                dropout_fraction=dropout_fraction
            )(radar_layer_object)

        radar_layer_object = cnn_utils.get_2d_pooling_layer(
            num_rows_in_window=2, num_columns_in_window=2,
            pooling_type=cnn_utils.MEAN_POOLING_TYPE, num_rows_per_stride=2,
            num_columns_per_stride=2
        )(radar_layer_object)

    radar_layer_object = cnn_utils.get_flattening_layer()(radar_layer_object)

    if num_sounding_fields is None:
        layer_object = radar_layer_object
    else:
        (sounding_input_layer_object, sounding_layer_object
        ) = _get_sounding_layers(
            num_sounding_heights=num_sounding_heights,
            num_sounding_fields=num_sounding_fields,
            num_filters_in_first_conv_layer=
            num_sounding_filters_in_first_conv_layer,
            dropout_fraction=dropout_fraction,
            regularizer_object=regularizer_object)

        layer_object = keras.layers.concatenate(
            [radar_layer_object, sounding_layer_object])

    layer_object = cnn_utils.get_fully_connected_layer(
        num_output_units=num_classes, activation_function='softmax'
    )(layer_object)

    if num_sounding_fields is None:
        model_object = keras.models.Model(
            inputs=radar_input_layer_object, outputs=layer_object)
    else:
        model_object = keras.models.Model(
            inputs=[radar_input_layer_object, sounding_input_layer_object],
            outputs=layer_object)

    model_object.compile(
        loss=keras.losses.categorical_crossentropy,
        optimizer=keras.optimizers.Adam(), metrics=LIST_OF_METRIC_FUNCTIONS)

    model_object.summary()
    return model_object


def get_3d_swilrnet_architecture(
        num_radar_rows, num_radar_columns, num_radar_heights, num_radar_fields,
        num_classes, num_radar_filters_in_first_conv_layer=16,
        dropout_fraction=0.5, l2_weight=DEFAULT_L2_WEIGHT,
        num_sounding_heights=None, num_sounding_fields=None,
        num_sounding_filters_in_first_conv_layer=48):
    """Creates CNN with architecture similar to the following example.

    https://github.com/djgagne/swirlnet/blob/master/notebooks/
    deep_swirl_tutorial.ipynb

    This CNN does 3-D convolution only.

    :param num_radar_rows: See doc for `_check_architecture_args`.
    :param num_radar_columns: Same.
    :param num_radar_heights: Same.
    :param num_radar_fields: Same.
    :param num_classes: Same.
    :param num_radar_filters_in_first_conv_layer: See doc for
        `get_2d_swilrnet_architecture`.
    :param dropout_fraction: See doc for `_check_architecture_args`.
    :param l2_weight: Same.
    :param num_sounding_heights: Same.
    :param num_sounding_fields: Same.
    :param num_sounding_filters_in_first_conv_layer: See doc for
        `get_2d_swilrnet_architecture`.
    :return: model_object: `keras.models.Sequential` object with the
        aforementioned architecture.
    """

    _check_architecture_args(
        num_radar_rows=num_radar_rows, num_radar_columns=num_radar_columns,
        num_radar_heights=num_radar_heights, num_radar_fields=num_radar_fields,
        num_radar_dimensions=3, num_classes=num_classes,
        dropout_fraction=dropout_fraction, l2_weight=l2_weight,
        num_sounding_heights=num_sounding_heights,
        num_sounding_fields=num_sounding_fields)

    if l2_weight is None:
        regularizer_object = None
    else:
        regularizer_object = cnn_utils.get_weight_regularizer(
            l1_penalty=0, l2_penalty=l2_weight)

    radar_input_layer_object = keras.layers.Input(
        shape=(num_radar_rows, num_radar_columns, num_radar_heights,
               num_radar_fields))
    this_num_output_filters = num_radar_filters_in_first_conv_layer + 0

    for i in range(3):
        if i == 0:
            this_input_layer_object = radar_input_layer_object
        else:
            this_input_layer_object = radar_layer_object
            this_num_output_filters *= 2

        radar_layer_object = cnn_utils.get_3d_conv_layer(
            num_output_filters=this_num_output_filters,
            num_kernel_rows=5, num_kernel_columns=5, num_kernel_depths=3,
            num_rows_per_stride=1, num_columns_per_stride=1,
            num_depths_per_stride=1, padding_type=cnn_utils.YES_PADDING_TYPE,
            kernel_weight_regularizer=regularizer_object,
            activation_function='relu', is_first_layer=False
        )(this_input_layer_object)

        if dropout_fraction is not None:
            radar_layer_object = cnn_utils.get_dropout_layer(
                dropout_fraction=dropout_fraction
            )(radar_layer_object)

        radar_layer_object = cnn_utils.get_3d_pooling_layer(
            num_rows_in_window=2, num_columns_in_window=2, num_depths_in_window=2,
            pooling_type=cnn_utils.MEAN_POOLING_TYPE, num_rows_per_stride=2,
            num_columns_per_stride=2, num_depths_per_stride=2
        )(radar_layer_object)

    radar_layer_object = cnn_utils.get_flattening_layer()(radar_layer_object)

    if num_sounding_fields is None:
        layer_object = radar_layer_object
    else:
        (sounding_input_layer_object, sounding_layer_object
        ) = _get_sounding_layers(
            num_sounding_heights=num_sounding_heights,
            num_sounding_fields=num_sounding_fields,
            num_filters_in_first_conv_layer=
            num_sounding_filters_in_first_conv_layer,
            dropout_fraction=dropout_fraction,
            regularizer_object=regularizer_object)

        layer_object = keras.layers.concatenate(
            [radar_layer_object, sounding_layer_object])

    layer_object = cnn_utils.get_fully_connected_layer(
        num_output_units=num_classes, activation_function='softmax'
    )(layer_object)

    if num_sounding_fields is None:
        model_object = keras.models.Model(
            inputs=radar_input_layer_object, outputs=layer_object)
    else:
        model_object = keras.models.Model(
            inputs=[radar_input_layer_object, sounding_input_layer_object],
            outputs=layer_object)

    model_object.compile(
        loss=keras.losses.categorical_crossentropy,
        optimizer=keras.optimizers.Adam(), metrics=LIST_OF_METRIC_FUNCTIONS)

    model_object.summary()
    return model_object


def get_2d3d_swirlnet_architecture(
        num_reflectivity_rows, num_reflectivity_columns,
        num_reflectivity_heights, num_azimuthal_shear_fields, num_classes,
        num_refl_filters_in_first_conv_layer=8,
        num_az_shear_filters_in_first_conv_layer=16, dropout_fraction=0.5,
        l2_weight=DEFAULT_L2_WEIGHT, num_sounding_heights=None,
        num_sounding_fields=None, num_sounding_filters_in_first_conv_layer=48):
    """Creates CNN with architecture similar to the following example.

    https://github.com/djgagne/swirlnet/blob/master/notebooks/
    deep_swirl_tutorial.ipynb

    This CNN does 2-D convolution over azimuthal-shear fields and 3-D
    convolution over reflectivity fields.

    This method assumes that azimuthal-shear fields have twice the horizontal
    resolution as reflectivity fields.  In other words:

    num_azimuthal_shear_rows = 2 * num_reflectivity_rows
    num_azimuthal_shear_columns = 2 * num_reflectivity_columns

    :param num_reflectivity_rows: Number of pixel rows in each reflectivity
        image.
    :param num_reflectivity_columns: Number of pixel columns in each
        reflectivity image.
    :param num_reflectivity_heights: Number of pixel heights in each
        reflectivity image.
    :param num_azimuthal_shear_fields: Number of azimuthal-shear fields
        (channels).
    :param num_classes: Number of classes for target variable.
    :param num_refl_filters_in_first_conv_layer: Number of filters in first
        convolutional layer for reflectivity images.  Number of filters will
        double for each successive layer convolving over reflectivity images.
    :param num_az_shear_filters_in_first_conv_layer: Same, but for azimuthal
        shear.
    :param dropout_fraction: See doc for `_check_architecture_args`.
    :param l2_weight: Same.
    :param num_sounding_heights: Same.
    :param num_sounding_fields: Same.
    :param num_sounding_filters_in_first_conv_layer: See doc for
        `get_2d_swirlnet_architecture`.
    :return: model_object: `keras.models.Sequential` object with the
        aforementioned architecture.
    """

    # Error-checking for reflectivity part.
    _check_architecture_args(
        num_radar_rows=num_reflectivity_rows,
        num_radar_columns=num_reflectivity_columns,
        num_radar_heights=num_reflectivity_heights, num_radar_fields=1,
        num_radar_dimensions=3, num_classes=num_classes,
        dropout_fraction=dropout_fraction, l2_weight=l2_weight,
        num_sounding_heights=num_sounding_heights,
        num_sounding_fields=num_sounding_fields)

    # Error-checking for azimuthal-shear part.
    num_azimuthal_shear_rows = 2 * num_reflectivity_rows
    num_azimuthal_shear_columns = 2 * num_reflectivity_columns

    _check_architecture_args(
        num_radar_rows=num_azimuthal_shear_rows,
        num_radar_columns=num_azimuthal_shear_columns,
        num_radar_channels=num_azimuthal_shear_fields, num_radar_dimensions=2,
        num_classes=num_classes, dropout_fraction=dropout_fraction,
        l2_weight=l2_weight, num_sounding_heights=num_sounding_heights,
        num_sounding_fields=num_sounding_fields)

    if l2_weight is None:
        regularizer_object = None
    else:
        regularizer_object = cnn_utils.get_weight_regularizer(
            l1_penalty=0, l2_penalty=l2_weight)

    refl_input_layer_object = keras.layers.Input(
        shape=(num_reflectivity_rows, num_reflectivity_columns,
               num_reflectivity_heights, 1))
    az_shear_input_layer_object = keras.layers.Input(
        shape=(num_azimuthal_shear_rows, num_azimuthal_shear_columns,
               num_azimuthal_shear_fields))
    this_num_refl_filters = num_refl_filters_in_first_conv_layer + 0

    for i in range(3):
        if i == 0:
            this_input_layer_object = refl_input_layer_object
        else:
            this_num_refl_filters *= 2
            this_input_layer_object = reflectivity_layer_object

        reflectivity_layer_object = cnn_utils.get_3d_conv_layer(
            num_output_filters=this_num_refl_filters,
            num_kernel_rows=5, num_kernel_columns=5, num_kernel_depths=3,
            num_rows_per_stride=1, num_columns_per_stride=1,
            num_depths_per_stride=1, padding_type=cnn_utils.YES_PADDING_TYPE,
            kernel_weight_regularizer=regularizer_object,
            activation_function='relu', is_first_layer=False
        )(this_input_layer_object)

        if dropout_fraction is not None:
            reflectivity_layer_object = cnn_utils.get_dropout_layer(
                dropout_fraction=dropout_fraction
            )(reflectivity_layer_object)

        reflectivity_layer_object = cnn_utils.get_3d_pooling_layer(
            num_rows_in_window=1, num_columns_in_window=1, num_depths_in_window=2,
            pooling_type=cnn_utils.MEAN_POOLING_TYPE, num_rows_per_stride=1,
            num_columns_per_stride=1, num_depths_per_stride=2
        )(reflectivity_layer_object)

    this_target_shape = (
        num_reflectivity_rows, num_reflectivity_columns, this_num_refl_filters)
    reflectivity_layer_object = keras.layers.Reshape(
        target_shape=this_target_shape
    )(reflectivity_layer_object)

    azimuthal_shear_layer_object = cnn_utils.get_2d_conv_layer(
        num_output_filters=num_az_shear_filters_in_first_conv_layer,
        num_kernel_rows=5, num_kernel_columns=5, num_rows_per_stride=1,
        num_columns_per_stride=1, padding_type=cnn_utils.YES_PADDING_TYPE,
        kernel_weight_regularizer=regularizer_object,
        activation_function='relu', is_first_layer=False
    )(az_shear_input_layer_object)

    if dropout_fraction is not None:
        azimuthal_shear_layer_object = cnn_utils.get_dropout_layer(
            dropout_fraction=dropout_fraction
        )(azimuthal_shear_layer_object)

    azimuthal_shear_layer_object = cnn_utils.get_2d_pooling_layer(
        num_rows_in_window=2, num_columns_in_window=2,
        pooling_type=cnn_utils.MEAN_POOLING_TYPE, num_rows_per_stride=2,
        num_columns_per_stride=2
    )(azimuthal_shear_layer_object)

    radar_layer_object = keras.layers.concatenate(
        [reflectivity_layer_object, azimuthal_shear_layer_object], axis=-1)

    this_num_output_filters = (
        2 * num_az_shear_filters_in_first_conv_layer +
        2 * this_num_refl_filters
    )

    for i in range(3):
        if i != 0:
            this_num_output_filters *= 2

        radar_layer_object = cnn_utils.get_2d_conv_layer(
            num_output_filters=this_num_output_filters, num_kernel_rows=5,
            num_kernel_columns=5, num_rows_per_stride=1, num_columns_per_stride=1,
            padding_type=cnn_utils.YES_PADDING_TYPE,
            kernel_weight_regularizer=regularizer_object,
            activation_function='relu'
        )(radar_layer_object)

        if dropout_fraction is not None:
            radar_layer_object = cnn_utils.get_dropout_layer(
                dropout_fraction=dropout_fraction
            )(radar_layer_object)

        radar_layer_object = cnn_utils.get_2d_pooling_layer(
            num_rows_in_window=2, num_columns_in_window=2,
            pooling_type=cnn_utils.MEAN_POOLING_TYPE, num_rows_per_stride=2,
            num_columns_per_stride=2
        )(radar_layer_object)

    radar_layer_object = cnn_utils.get_flattening_layer()(radar_layer_object)

    if num_sounding_fields is None:
        layer_object = radar_layer_object
    else:
        (sounding_input_layer_object, sounding_layer_object
        ) = _get_sounding_layers(
            num_sounding_heights=num_sounding_heights,
            num_sounding_fields=num_sounding_fields,
            num_filters_in_first_conv_layer=
            num_sounding_filters_in_first_conv_layer,
            dropout_fraction=dropout_fraction,
            regularizer_object=regularizer_object)

        layer_object = keras.layers.concatenate(
            [radar_layer_object, sounding_layer_object])

    layer_object = cnn_utils.get_fully_connected_layer(
        num_output_units=num_classes, activation_function='softmax'
    )(layer_object)

    if num_sounding_fields is None:
        model_object = keras.models.Model(
            inputs=[refl_input_layer_object, az_shear_input_layer_object],
            outputs=layer_object)
    else:
        model_object = keras.models.Model(
            inputs=[refl_input_layer_object, az_shear_input_layer_object,
                    sounding_input_layer_object],
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
        pickle_file_name, num_epochs, num_examples_per_batch,
        num_examples_per_file_time, num_training_batches_per_epoch,
        first_train_time_unix_sec, last_train_time_unix_sec,
        weight_loss_function, target_name, binarize_target, radar_source,
        radar_field_names, radar_normalization_dict, radar_heights_m_asl=None,
        reflectivity_heights_m_asl=None, training_fraction_by_class_dict=None,
        validation_fraction_by_class_dict=None,
        num_validation_batches_per_epoch=None, first_validn_time_unix_sec=None,
        last_validn_time_unix_sec=None, sounding_field_names=None,
        sounding_normalization_dict=None,
        sounding_lag_time_for_convective_contamination_sec=None):
    """Writes metadata for CNN to Pickle file.

    :param pickle_file_name: Path to output file.
    :param num_epochs: Number of epochs.
    :param num_examples_per_batch: Number of examples (storm objects) per batch.
    :param num_examples_per_file_time: Number of examples (storm objects) per
        file time.  One "file time" may be either one time step or one SPC date.
    :param num_training_batches_per_epoch: Number of training batches per epoch.
    :param first_train_time_unix_sec: First file time in training period.  If
        files contain one SPC date each, this can be any time on the first SPC
        date.
    :param last_train_time_unix_sec: Last file time in training period.
    :param weight_loss_function: See doc for `_check_training_args`.
    :param target_name: Name of target variable.
    :param binarize_target: Boolean flag.  If True, target variable was
        binarized, so that the highest class becomes 1 and all other classes
        become 0.
    :param radar_source: Data source (must be accepted by
        `radar_utils.check_data_source`).
    :param radar_field_names: 1-D list with names of radar fields used in
        training.  Each field must accepted by `radar_utils.check_field_name`.
    :param radar_normalization_dict: Used to normalize radar data (see
        `deep_learning_utils.normalize_predictor_matrix` for details).
    :param radar_heights_m_asl: [needed only if CNN does 3-D convolution]
        1-D numpy array of heights (metres above sea level), which apply to each
        radar field.
    :param reflectivity_heights_m_asl: [needed only if CNN does 2-D convolution]
        1-D numpy array of heights (metres above sea level), which apply only to
        "reflectivity_dbz".
    :param training_fraction_by_class_dict: Used to sample training data (see
        `training_validation_io.storm_image_generator_2d` for details).
    :param validation_fraction_by_class_dict:
        [needed only if CNN did validation on the fly]
        Same as `training_fraction_by_class_dict`, but for validation data.
    :param num_validation_batches_per_epoch:
        [needed only if CNN did validation on the fly]
        Number of validation batches per epoch.
    :param first_validn_time_unix_sec:
        [needed only if CNN did validation on the fly]
        First file time in validation period.
    :param last_validn_time_unix_sec:
        [needed only if CNN did validation on the fly]
        Last file time in validation period.
    :param sounding_field_names: [needed only if CNN takes soundings]
        1-D list with names of sounding fields.  Each must be accepted by
        `soundings_only.check_pressureless_field_name`.
    :param sounding_normalization_dict: Used to normalize soundings (see
        `deep_learning_utils.normalize_sounding_matrix` for details).
    :param sounding_lag_time_for_convective_contamination_sec: See doc for
        `training_validation_io.find_sounding_files`.
    """

    model_metadata_dict = {
        NUM_EPOCHS_KEY: num_epochs,
        NUM_EXAMPLES_PER_BATCH_KEY: num_examples_per_batch,
        NUM_EXAMPLES_PER_FILE_TIME_KEY: num_examples_per_file_time,
        NUM_TRAINING_BATCHES_KEY: num_training_batches_per_epoch,
        FIRST_TRAINING_TIME_KEY: first_train_time_unix_sec,
        LAST_TRAINING_TIME_KEY: last_train_time_unix_sec,
        WEIGHT_LOSS_FUNCTION_KEY: weight_loss_function,
        TARGET_NAME_KEY: target_name,
        BINARIZE_TARGET_KEY: binarize_target,
        RADAR_SOURCE_KEY: radar_source,
        RADAR_FIELD_NAMES_KEY: radar_field_names,
        RADAR_NORMALIZATION_DICT_KEY: radar_normalization_dict,
        RADAR_HEIGHTS_KEY: radar_heights_m_asl,
        REFLECTIVITY_HEIGHTS_KEY: reflectivity_heights_m_asl,
        TRAINING_FRACTION_BY_CLASS_KEY: training_fraction_by_class_dict,
        VALIDATION_FRACTION_BY_CLASS_KEY: validation_fraction_by_class_dict,
        NUM_VALIDATION_BATCHES_KEY: num_validation_batches_per_epoch,
        FIRST_VALIDATION_TIME_KEY: first_validn_time_unix_sec,
        LAST_VALIDATION_TIME_KEY: last_validn_time_unix_sec,
        SOUNDING_FIELD_NAMES_KEY: sounding_field_names,
        SOUNDING_NORMALIZATION_DICT_KEY: sounding_normalization_dict,
        SOUNDING_LAG_TIME_KEY:
            sounding_lag_time_for_convective_contamination_sec
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
        num_epochs, num_examples_per_batch, num_examples_per_file_time,
        num_training_batches_per_epoch, radar_file_name_matrix_for_training,
        target_name, top_target_directory_name, binarize_target=False,
        weight_loss_function=False, training_fraction_by_class_dict=None,
        num_validation_batches_per_epoch=None,
        validation_fraction_by_class_dict=None,
        radar_file_name_matrix_for_validn=None, sounding_field_names=None,
        top_sounding_dir_name=None,
        sounding_lag_time_for_convective_contamination_sec=None):
    """Trains CNN with 2-D radar images.

    T = number of storm times (initial times) for training
    V = number of storm times (initial times) for validation

    :param model_object: Instance of `keras.models.Sequential`.
    :param model_file_name: See doc for `_check_training_args`.
    :param history_file_name: Same.
    :param tensorboard_dir_name: Same.
    :param num_epochs: Number of epochs.
    :param num_examples_per_batch: Number of examples (storm objects) per batch.
    :param num_examples_per_file_time: Number of examples (storm objects) per
        file.  If each file contains one time step (SPC date), this will be
        number of examples per time step (SPC date).
    :param num_training_batches_per_epoch: Number of training batches per epoch.
    :param radar_file_name_matrix_for_training: T-by-C numpy array of paths to
        radar files.  Should be created by
        `training_validation_io.find_radar_files_2d`.
    :param target_name: Name of target variable.
    :param top_target_directory_name: Name of top-level directory with target
        values (storm-hazard labels).  Files within this directory should be
        findable by `labels.find_label_file`.
    :param binarize_target: See doc for `_check_training_args`.
    :param weight_loss_function: See doc for `_check_training_args`.
    :param training_fraction_by_class_dict: See doc for `_check_training_args`.
    :param num_validation_batches_per_epoch: Number of validation batches per
        epoch.
    :param validation_fraction_by_class_dict: Same as
        `training_fraction_by_class_dict`, but for validation data.
    :param radar_file_name_matrix_for_validn: V-by-C numpy array of paths to
        radar files.  Should be created by
        `training_validation_io.find_radar_files_2d`.
    :param sounding_field_names: list (length F_s) with names of sounding
        fields.  Each must be accepted by
        `soundings_only.check_pressureless_field_name`.
    :param top_sounding_dir_name: See doc for
        `training_validation_io.find_sounding_files`.
    :param sounding_lag_time_for_convective_contamination_sec: Same.
    """

    loss_function_weight_by_class_dict = _check_training_args(
        num_epochs=num_epochs,
        num_training_batches_per_epoch=num_training_batches_per_epoch,
        num_validation_batches_per_epoch=num_validation_batches_per_epoch,
        weight_loss_function=weight_loss_function,
        binarize_target=binarize_target,
        training_fraction_by_class_dict=training_fraction_by_class_dict,
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
            generator=trainval_io.storm_image_generator_2d(
                radar_file_name_matrix=radar_file_name_matrix_for_training,
                top_target_directory_name=top_target_directory_name,
                num_examples_per_batch=num_examples_per_batch,
                num_examples_per_file_time=num_examples_per_file_time,
                target_name=target_name, binarize_target=binarize_target,
                sampling_fraction_by_class_dict=training_fraction_by_class_dict,
                sounding_field_names=sounding_field_names,
                top_sounding_dir_name=top_sounding_dir_name,
                sounding_lag_time_for_convective_contamination_sec=
                sounding_lag_time_for_convective_contamination_sec),
            steps_per_epoch=num_training_batches_per_epoch, epochs=num_epochs,
            verbose=1, class_weight=loss_function_weight_by_class_dict,
            callbacks=[checkpoint_object, history_object, tensorboard_object])

    else:
        checkpoint_object = keras.callbacks.ModelCheckpoint(
            filepath=model_file_name, monitor='val_loss', verbose=1,
            save_best_only=True, save_weights_only=False, mode='min', period=1)

        model_object.fit_generator(
            generator=trainval_io.storm_image_generator_2d(
                radar_file_name_matrix=radar_file_name_matrix_for_training,
                top_target_directory_name=top_target_directory_name,
                num_examples_per_batch=num_examples_per_batch,
                num_examples_per_file_time=num_examples_per_file_time,
                target_name=target_name, binarize_target=binarize_target,
                sampling_fraction_by_class_dict=training_fraction_by_class_dict,
                sounding_field_names=sounding_field_names,
                top_sounding_dir_name=top_sounding_dir_name,
                sounding_lag_time_for_convective_contamination_sec=
                sounding_lag_time_for_convective_contamination_sec),
            steps_per_epoch=num_training_batches_per_epoch, epochs=num_epochs,
            verbose=1, class_weight=loss_function_weight_by_class_dict,
            callbacks=[checkpoint_object, history_object, tensorboard_object],
            validation_data=trainval_io.storm_image_generator_2d(
                radar_file_name_matrix=radar_file_name_matrix_for_validn,
                top_target_directory_name=top_target_directory_name,
                num_examples_per_batch=num_examples_per_batch,
                num_examples_per_file_time=num_examples_per_file_time,
                target_name=target_name, binarize_target=binarize_target,
                sampling_fraction_by_class_dict=
                validation_fraction_by_class_dict,
                sounding_field_names=sounding_field_names,
                top_sounding_dir_name=top_sounding_dir_name,
                sounding_lag_time_for_convective_contamination_sec=
                sounding_lag_time_for_convective_contamination_sec),
            validation_steps=num_validation_batches_per_epoch)


def train_3d_cnn(
        model_object, model_file_name, history_file_name, tensorboard_dir_name,
        num_epochs, num_examples_per_batch, num_examples_per_file_time,
        num_training_batches_per_epoch, radar_file_name_matrix_for_training,
        target_name, top_target_directory_name, binarize_target=False,
        weight_loss_function=False, training_fraction_by_class_dict=None,
        num_validation_batches_per_epoch=None,
        validation_fraction_by_class_dict=None,
        radar_file_name_matrix_for_validn=None, sounding_field_names=None,
        top_sounding_dir_name=None,
        sounding_lag_time_for_convective_contamination_sec=None):
    """Trains CNN with 3-D radar images.

    T = number of storm times (initial times) for training
    V = number of storm times (initial times) for validation

    :param model_object: See doc for `train_2d_cnn`.
    :param model_file_name: Same.
    :param history_file_name: Same.
    :param tensorboard_dir_name: Same.
    :param num_epochs: Same.
    :param num_examples_per_batch: Same.
    :param num_examples_per_file_time: Same.
    :param num_training_batches_per_epoch: Same.
    :param radar_file_name_matrix_for_training: T-by-F-by-H numpy array of paths
        to radar files.  Should be created by
        `training_validation_io.find_radar_files_3d`.
    :param target_name: See doc for `train_2d_cnn`.
    :param top_target_directory_name: Same.
    :param binarize_target: Same.
    :param weight_loss_function: Same.
    :param training_fraction_by_class_dict: Same.
    :param num_validation_batches_per_epoch: Same.
    :param validation_fraction_by_class_dict: Same.
    :param radar_file_name_matrix_for_validn: V-by-F-by-H numpy array of paths
        to radar files.  Should be created by
        `training_validation_io.find_radar_files_3d`.
    :param sounding_field_names: See doc for `train_2d_cnn`.
    :param top_sounding_dir_name: Same.
    :param sounding_lag_time_for_convective_contamination_sec: Same.
    """

    loss_function_weight_by_class_dict = _check_training_args(
        num_epochs=num_epochs,
        num_training_batches_per_epoch=num_training_batches_per_epoch,
        num_validation_batches_per_epoch=num_validation_batches_per_epoch,
        weight_loss_function=weight_loss_function,
        binarize_target=binarize_target,
        training_fraction_by_class_dict=training_fraction_by_class_dict,
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
            generator=trainval_io.storm_image_generator_3d(
                radar_file_name_matrix=radar_file_name_matrix_for_training,
                top_target_directory_name=top_target_directory_name,
                num_examples_per_batch=num_examples_per_batch,
                num_examples_per_file_time=num_examples_per_file_time,
                target_name=target_name, binarize_target=binarize_target,
                sampling_fraction_by_class_dict=training_fraction_by_class_dict,
                sounding_field_names=sounding_field_names,
                top_sounding_dir_name=top_sounding_dir_name,
                sounding_lag_time_for_convective_contamination_sec=
                sounding_lag_time_for_convective_contamination_sec),
            steps_per_epoch=num_training_batches_per_epoch, epochs=num_epochs,
            verbose=1, class_weight=loss_function_weight_by_class_dict,
            callbacks=[checkpoint_object, history_object, tensorboard_object])

    else:
        checkpoint_object = keras.callbacks.ModelCheckpoint(
            filepath=model_file_name, monitor='val_loss', verbose=1,
            save_best_only=True, save_weights_only=False, mode='min', period=1)

        model_object.fit_generator(
            generator=trainval_io.storm_image_generator_3d(
                radar_file_name_matrix=radar_file_name_matrix_for_training,
                top_target_directory_name=top_target_directory_name,
                num_examples_per_batch=num_examples_per_batch,
                num_examples_per_file_time=num_examples_per_file_time,
                target_name=target_name, binarize_target=binarize_target,
                sampling_fraction_by_class_dict=training_fraction_by_class_dict,
                sounding_field_names=sounding_field_names,
                top_sounding_dir_name=top_sounding_dir_name,
                sounding_lag_time_for_convective_contamination_sec=
                sounding_lag_time_for_convective_contamination_sec),
            steps_per_epoch=num_training_batches_per_epoch, epochs=num_epochs,
            verbose=1, class_weight=loss_function_weight_by_class_dict,
            callbacks=[checkpoint_object, history_object, tensorboard_object],
            validation_data=trainval_io.storm_image_generator_3d(
                radar_file_name_matrix=radar_file_name_matrix_for_validn,
                top_target_directory_name=top_target_directory_name,
                num_examples_per_batch=num_examples_per_batch,
                num_examples_per_file_time=num_examples_per_file_time,
                target_name=target_name, binarize_target=binarize_target,
                sampling_fraction_by_class_dict=
                validation_fraction_by_class_dict,
                sounding_field_names=sounding_field_names,
                top_sounding_dir_name=top_sounding_dir_name,
                sounding_lag_time_for_convective_contamination_sec=
                sounding_lag_time_for_convective_contamination_sec),
            validation_steps=num_validation_batches_per_epoch)


def train_2d3d_cnn(
        model_object, model_file_name, history_file_name, tensorboard_dir_name,
        num_epochs, num_examples_per_batch, num_examples_per_file_time,
        num_training_batches_per_epoch, radar_file_name_matrix_for_training,
        target_name, top_target_directory_name, binarize_target=False,
        weight_loss_function=False, training_fraction_by_class_dict=None,
        num_validation_batches_per_epoch=None,
        validation_fraction_by_class_dict=None,
        radar_file_name_matrix_for_validn=None, sounding_field_names=None,
        top_sounding_dir_name=None,
        sounding_lag_time_for_convective_contamination_sec=None):
    """Trains CNN with 2-D azimuthal-shear images and 3-D reflectivity images.

    T = number of storm times (initial times) for training
    V = number of storm times (initial times) for validation
    F_a = number of azimuthal-shear fields
    H_r = number of heights per reflectivity image

    :param model_object: See doc for `train_2d_cnn`.
    :param model_file_name: Same.
    :param history_file_name: Same.
    :param tensorboard_dir_name: Same.
    :param num_epochs: Same.
    :param num_examples_per_batch: Same.
    :param num_examples_per_file_time: Same.
    :param num_training_batches_per_epoch: Same.
    :param radar_file_name_matrix_for_training: numpy array (T x [H_r + F_a]) of
        paths to image files.  This should be created by `find_radar_files_2d`.
    :param target_name: See doc for `train_2d_cnn`.
    :param top_target_directory_name: Same.
    :param binarize_target: Same.
    :param weight_loss_function: Same.
    :param training_fraction_by_class_dict: Same.
    :param num_validation_batches_per_epoch: Same.
    :param validation_fraction_by_class_dict: Same.
    :param radar_file_name_matrix_for_validn: numpy array (V x [H_r + F_a]) of
        paths to image files.  This should be created by `find_radar_files_2d`.
    :param sounding_field_names: See doc for `train_2d_cnn`.
    :param top_sounding_dir_name: Same.
    :param sounding_lag_time_for_convective_contamination_sec: Same.
    """

    loss_function_weight_by_class_dict = _check_training_args(
        num_epochs=num_epochs,
        num_training_batches_per_epoch=num_training_batches_per_epoch,
        num_validation_batches_per_epoch=num_validation_batches_per_epoch,
        weight_loss_function=weight_loss_function,
        binarize_target=binarize_target,
        training_fraction_by_class_dict=training_fraction_by_class_dict,
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
                radar_file_name_matrix=radar_file_name_matrix_for_training,
                top_target_directory_name=top_target_directory_name,
                num_examples_per_batch=num_examples_per_batch,
                num_examples_per_file_time=num_examples_per_file_time,
                target_name=target_name, binarize_target=binarize_target,
                sampling_fraction_by_class_dict=training_fraction_by_class_dict,
                sounding_field_names=sounding_field_names,
                top_sounding_dir_name=top_sounding_dir_name,
                sounding_lag_time_for_convective_contamination_sec=
                sounding_lag_time_for_convective_contamination_sec),
            steps_per_epoch=num_training_batches_per_epoch, epochs=num_epochs,
            verbose=1, class_weight=loss_function_weight_by_class_dict,
            callbacks=[checkpoint_object, history_object, tensorboard_object])

    else:
        checkpoint_object = keras.callbacks.ModelCheckpoint(
            filepath=model_file_name, monitor='val_loss', verbose=1,
            save_best_only=True, save_weights_only=False, mode='min', period=1)

        model_object.fit_generator(
            generator=trainval_io.storm_image_generator_2d3d_myrorss(
                radar_file_name_matrix=radar_file_name_matrix_for_training,
                top_target_directory_name=top_target_directory_name,
                num_examples_per_batch=num_examples_per_batch,
                num_examples_per_file_time=num_examples_per_file_time,
                target_name=target_name, binarize_target=binarize_target,
                sampling_fraction_by_class_dict=training_fraction_by_class_dict,
                sounding_field_names=sounding_field_names,
                top_sounding_dir_name=top_sounding_dir_name,
                sounding_lag_time_for_convective_contamination_sec=
                sounding_lag_time_for_convective_contamination_sec),
            steps_per_epoch=num_training_batches_per_epoch, epochs=num_epochs,
            verbose=1, class_weight=loss_function_weight_by_class_dict,
            callbacks=[checkpoint_object, history_object, tensorboard_object],
            validation_data=trainval_io.storm_image_generator_2d3d_myrorss(
                radar_file_name_matrix=radar_file_name_matrix_for_validn,
                top_target_directory_name=top_target_directory_name,
                num_examples_per_batch=num_examples_per_batch,
                num_examples_per_file_time=num_examples_per_file_time,
                target_name=target_name, binarize_target=binarize_target,
                sampling_fraction_by_class_dict=
                validation_fraction_by_class_dict,
                sounding_field_names=sounding_field_names,
                top_sounding_dir_name=top_sounding_dir_name,
                sounding_lag_time_for_convective_contamination_sec=
                sounding_lag_time_for_convective_contamination_sec),
            validation_steps=num_validation_batches_per_epoch)


def apply_2d_cnn(model_object, radar_image_matrix, sounding_matrix=None):
    """Applies CNN to 2-D radar images.

    :param model_object: Instance of `keras.models.Sequential`.
    :param radar_image_matrix: E-by-M-by-N-by-C numpy array of storm-centered
        radar images.
    :param sounding_matrix: [may be None]
        numpy array (E x H_s x F_s) of storm-centered soundings.
    :return: class_probability_matrix: E-by-K numpy array of class
        probabilities.  class_probability_matrix[i, k] is the forecast
        probability that the [i]th storm object belongs to the [k]th class.
        Classes are mutually exclusive and collectively exhaustive, so the sum
        across each row is 1.
    """

    dl_utils.check_predictor_matrix(
        predictor_matrix=radar_image_matrix, min_num_dimensions=4,
        max_num_dimensions=4)

    num_examples = radar_image_matrix.shape[0]
    if sounding_matrix is None:
        return model_object.predict(radar_image_matrix, batch_size=num_examples)

    dl_utils.check_sounding_matrix(
        sounding_matrix=sounding_matrix, num_examples=num_examples)
    return model_object.predict(
        [radar_image_matrix, sounding_matrix], batch_size=num_examples)


def apply_3d_cnn(model_object, radar_image_matrix, sounding_matrix=None):
    """Applies CNN to 3-D radar images.

    :param model_object: Instance of `keras.models.Sequential`.
    :param radar_image_matrix: numpy array (E x M x N x H_r x F_r) of storm-
        centered radar images.
    :param sounding_matrix: See doc for `apply_2d_cnn`.
    :return: class_probability_matrix: Same.
    """

    dl_utils.check_predictor_matrix(
        predictor_matrix=radar_image_matrix, min_num_dimensions=5,
        max_num_dimensions=5)

    num_examples = radar_image_matrix.shape[0]
    if sounding_matrix is None:
        return model_object.predict(radar_image_matrix, batch_size=num_examples)

    dl_utils.check_sounding_matrix(
        sounding_matrix=sounding_matrix, num_examples=num_examples)
    return model_object.predict(
        [radar_image_matrix, sounding_matrix], batch_size=num_examples)


def apply_2d3d_cnn(
        model_object, reflectivity_image_matrix_dbz,
        azimuthal_shear_image_matrix_s01, sounding_matrix=None):
    """Applies CNN to 2-D azimuthal-shear images and 3-D reflectivity images.

    :param model_object: Instance of `keras.models.Sequential`.
    :param reflectivity_image_matrix_dbz: numpy array (E x m x n x H_r x 1) of
        storm-centered reflectivity images.
    :param azimuthal_shear_image_matrix_s01: numpy array (E x M x N x F_a) of
        storm-centered azimuthal-shear images.
    :param sounding_matrix: See doc for `apply_2d_cnn`.
    :return: class_probability_matrix: Same.
    """

    dl_utils.check_predictor_matrix(
        predictor_matrix=reflectivity_image_matrix_dbz, min_num_dimensions=5,
        max_num_dimensions=5)
    dl_utils.check_predictor_matrix(
        predictor_matrix=azimuthal_shear_image_matrix_s01, min_num_dimensions=4,
        max_num_dimensions=4)

    expected_dimensions = numpy.array(
        reflectivity_image_matrix_dbz.shape[:-1] + (1,))
    error_checking.assert_is_numpy_array(
        reflectivity_image_matrix_dbz, exact_dimensions=expected_dimensions)

    num_examples = reflectivity_image_matrix_dbz.shape[0]
    expected_dimensions = numpy.array(
        (num_examples,) + azimuthal_shear_image_matrix_s01[1:])
    error_checking.assert_is_numpy_array(
        azimuthal_shear_image_matrix_s01, exact_dimensions=expected_dimensions)

    if sounding_matrix is None:
        return model_object.predict(
            [reflectivity_image_matrix_dbz, azimuthal_shear_image_matrix_s01],
            batch_size=num_examples)

    dl_utils.check_sounding_matrix(
        sounding_matrix=sounding_matrix, num_examples=num_examples)
    return model_object.predict(
        [reflectivity_image_matrix_dbz, azimuthal_shear_image_matrix_s01,
         sounding_matrix],
        batch_size=num_examples)
