"""Methods for creating, training, and applying a CNN*.

* convolutional neural network

Some variables in this module contain the string "pos_targets_only", which means
that they pertain only to storm objects with positive target values.  For
details on how to separate storm objects with positive target values, see
separate_myrorss_images_with_positive_targets.py.

--- NOTATION ---

The following letters will be used throughout this module.

E = number of examples (storm objects)
M = number of rows per radar image
N = number of columns per radar image
H_r = number of heights per radar image
F_r = number of radar fields (not including different heights)
H_s = number of height levels per sounding
F_s = number of sounding fields (not including different heights)
C = number of field/height pairs per radar image
K = number of classes for target variable
T = number of file times (time steps or SPC dates)
"""

import copy
import pickle
import numpy
import netCDF4
import keras.losses
import keras.optimizers
import keras.models
import keras.callbacks
from gewittergefahr.deep_learning import deep_learning_utils as dl_utils
from gewittergefahr.deep_learning import cnn_utils
from gewittergefahr.deep_learning import keras_metrics
from gewittergefahr.deep_learning import training_validation_io as trainval_io
from gewittergefahr.gg_io import netcdf_io
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.gg_utils import error_checking

LOSS_AS_MONITOR_STRING = 'loss'
PEIRCE_SCORE_AS_MONITOR_STRING = 'binary_peirce_score'
VALID_MONITOR_STRINGS = [LOSS_AS_MONITOR_STRING, PEIRCE_SCORE_AS_MONITOR_STRING]
VALID_NUMBERS_OF_SOUNDING_HEIGHTS = numpy.array([49], dtype=int)

DEFAULT_L2_WEIGHT = 1e-3
DEFAULT_CONV_LAYER_DROPOUT_FRACTION = 0.25
DEFAULT_DENSE_LAYER_DROPOUT_FRACTION = 0.5

DEFAULT_NUM_RADAR_FILTERS = 32
DEFAULT_NUM_SOUNDING_FILTERS = 16
DEFAULT_NUM_SOUNDING_CONV_LAYER_SETS = 3

CUSTOM_OBJECT_DICT_FOR_READING_MODEL = {
    'accuracy': keras_metrics.accuracy,
    'binary_accuracy': keras_metrics.binary_accuracy,
    'binary_csi': keras_metrics.binary_csi,
    'binary_frequency_bias': keras_metrics.binary_frequency_bias,
    'binary_pod': keras_metrics.binary_pod,
    'binary_pofd': keras_metrics.binary_pofd,
    'binary_peirce_score': keras_metrics.binary_peirce_score,
    'binary_success_ratio': keras_metrics.binary_success_ratio,
    'binary_focn': keras_metrics.binary_focn
}

LIST_OF_METRIC_FUNCTIONS = [
    keras_metrics.accuracy, keras_metrics.binary_accuracy,
    keras_metrics.binary_csi, keras_metrics.binary_frequency_bias,
    keras_metrics.binary_pod, keras_metrics.binary_pofd,
    keras_metrics.binary_peirce_score, keras_metrics.binary_success_ratio,
    keras_metrics.binary_focn
]

NUM_EPOCHS_KEY = 'num_epochs'
NUM_TRAINING_BATCHES_KEY = 'num_training_batches_per_epoch'
MONITOR_STRING_KEY = 'monitor_string'
WEIGHT_LOSS_FUNCTION_KEY = 'weight_loss_function'
NUM_VALIDATION_BATCHES_KEY = 'num_validation_batches_per_epoch'
VALIDATION_FILES_KEY = 'validn_file_name_matrix'
POSITIVE_VALIDATION_FILES_KEY = 'validn_file_name_matrix_pos_targets_only'
TRAINING_FILES_KEY = 'training_file_name_matrix'
POSITIVE_TRAINING_FILES_KEY = 'training_file_name_matrix_pos_targets_only'
USE_2D3D_CONVOLUTION_KEY = 'use_2d3d_convolution'
NUM_EXAMPLES_PER_BATCH_KEY = 'num_examples_per_batch'
NUM_EXAMPLES_PER_FILE_KEY = 'num_examples_per_file'
TARGET_NAME_KEY = 'target_name'
NUM_ROWS_TO_KEEP_KEY = 'num_rows_to_keep'
NUM_COLUMNS_TO_KEEP_KEY = 'num_columns_to_keep'
NORMALIZATION_TYPE_KEY = 'normalization_type_string'
MIN_NORMALIZED_VALUE_KEY = 'min_normalized_value'
MAX_NORMALIZED_VALUE_KEY = 'max_normalized_value'
NORMALIZATION_FILE_KEY = 'normalization_param_file_name'
BINARIZE_TARGET_KEY = 'binarize_target'
SAMPLING_FRACTIONS_KEY = 'sampling_fraction_by_class_dict'
SOUNDING_FIELD_NAMES_KEY = 'sounding_field_names'
SOUNDING_LAG_TIME_KEY = 'sounding_lag_time_sec'
NUM_TRANSLATIONS_KEY = 'num_translations'
MAX_TRANSLATION_KEY = 'max_translation_pixels'
NUM_ROTATIONS_KEY = 'num_rotations'
MAX_ROTATION_KEY = 'max_absolute_rotation_angle_deg'
NUM_NOISINGS_KEY = 'num_noisings'
MAX_NOISE_KEY = 'max_noise_standard_deviation'
REFL_MASKING_THRESHOLD_KEY = 'refl_masking_threshold_dbz'
RADAR_SOURCE_KEY = 'radar_source'
RADAR_FIELDS_KEY = 'radar_field_names'
RADAR_HEIGHTS_KEY = 'radar_heights_m_agl'
REFLECTIVITY_HEIGHTS_KEY = 'reflectivity_heights_m_agl'
SOUNDING_HEIGHTS_KEY = 'sounding_heights_m_agl'

MODEL_METADATA_KEYS = [
    NUM_EPOCHS_KEY, NUM_TRAINING_BATCHES_KEY, MONITOR_STRING_KEY,
    WEIGHT_LOSS_FUNCTION_KEY, NUM_VALIDATION_BATCHES_KEY, VALIDATION_FILES_KEY,
    POSITIVE_VALIDATION_FILES_KEY, TRAINING_FILES_KEY,
    POSITIVE_TRAINING_FILES_KEY, USE_2D3D_CONVOLUTION_KEY,
    NUM_EXAMPLES_PER_BATCH_KEY, NUM_EXAMPLES_PER_FILE_KEY, TARGET_NAME_KEY,
    NUM_ROWS_TO_KEEP_KEY, NUM_COLUMNS_TO_KEEP_KEY, NORMALIZATION_TYPE_KEY,
    MIN_NORMALIZED_VALUE_KEY, MAX_NORMALIZED_VALUE_KEY, NORMALIZATION_FILE_KEY,
    BINARIZE_TARGET_KEY, SAMPLING_FRACTIONS_KEY, SOUNDING_FIELD_NAMES_KEY,
    SOUNDING_LAG_TIME_KEY, NUM_TRANSLATIONS_KEY, MAX_TRANSLATION_KEY,
    NUM_ROTATIONS_KEY, MAX_ROTATION_KEY, NUM_NOISINGS_KEY, MAX_NOISE_KEY,
    REFL_MASKING_THRESHOLD_KEY, RADAR_SOURCE_KEY, RADAR_FIELDS_KEY,
    RADAR_HEIGHTS_KEY, REFLECTIVITY_HEIGHTS_KEY, SOUNDING_HEIGHTS_KEY
]

DEFAULT_METADATA_DICT = {
    VALIDATION_FILES_KEY: None,
    POSITIVE_VALIDATION_FILES_KEY: None,
    POSITIVE_TRAINING_FILES_KEY: None,
    SOUNDING_LAG_TIME_KEY: None,
    REFL_MASKING_THRESHOLD_KEY: None,
    RADAR_HEIGHTS_KEY: None,
    REFLECTIVITY_HEIGHTS_KEY: None,
    SOUNDING_HEIGHTS_KEY: None
}

DEFAULT_METADATA_DICT.update(trainval_io.DEFAULT_AUGMENTATION_OPTION_DICT)

STORM_OBJECT_DIMENSION_KEY = 'storm_object'
FEATURE_DIMENSION_KEY = 'feature'
SPATIAL_DIMENSION_KEYS = [
    'spatial_dimension1', 'spatial_dimension2', 'spatial_dimension3']

FEATURE_MATRIX_KEY = 'feature_matrix'
TARGET_VALUES_KEY = 'target_values'
NUM_CLASSES_KEY = 'num_classes'


def _check_architecture_args(
        num_radar_rows, num_radar_columns, num_radar_dimensions,
        num_radar_conv_layer_sets, num_conv_layers_per_set,
        num_radar_filters_in_first_layer, num_classes,
        conv_layer_activation_func_string, use_batch_normalization,
        alpha_for_elu=None, alpha_for_relu=None, num_radar_channels=None,
        num_radar_fields=None, num_radar_heights=None,
        conv_layer_dropout_fraction=None, dense_layer_dropout_fraction=None,
        l2_weight=None, num_sounding_fields=None, num_sounding_heights=None,
        num_sounding_conv_layer_sets=None,
        num_sounding_filters_in_first_layer=None):
    """Error-checking of input args for model architecture.

    :param num_radar_rows: Number of pixel rows per storm-centered radar image.
    :param num_radar_columns: Number of pixel columns per radar image.
    :param num_radar_dimensions: Number of dimensions per radar image.
    :param num_radar_conv_layer_sets: Number of sets of conv layers for radar
        data.  Each successive conv-layer set will halve the dimensions of the
        radar image (example: from 32 x 32 x 12 to 16 x 16 x 6, then
        8 x 8 x 3, then 4 x 4 x 1).
    :param num_conv_layers_per_set: Number of conv layers in each set.
    :param num_radar_filters_in_first_layer: Number of filters produced by the
        first conv layer for radar data.
    :param num_classes: Number of classes for target variable.
    :param conv_layer_activation_func_string: Activation function for conv
        layers.  Must be accepted by `cnn_utils.check_activation_function`.
    :param use_batch_normalization: Boolean flag.  If True, a batch-
        normalization layer will be included after each convolutional or fully
        connected layer.
    :param alpha_for_elu: Slope for negative inputs to eLU (exponential linear
        unit) activation function.
    :param alpha_for_relu: Slope for negative inputs to ReLU (rectified linear
        unit) activation function.
    :param num_radar_channels: [used only if num_radar_dimensions = 2]
        Number of channels (field/height pairs) per radar image.
    :param num_radar_fields: [used only if num_radar_dimensions = 3]
        Number of fields per radar image.
    :param num_radar_heights: [used only if num_radar_dimensions = 3]
        Number of pixel heights per radar image.
    :param conv_layer_dropout_fraction: Dropout fraction for convolutional
        layers.
    :param dense_layer_dropout_fraction: Dropout fraction for dense (fully
        connected layers).
    :param l2_weight: L2-regularization weight.  Will be used for each
        convolutional layer.
    :param num_sounding_fields: Number of fields per storm-centered sounding.
    :param num_sounding_heights: Number of heights per sounding.
    :param num_sounding_conv_layer_sets: Number of sets of conv layers for
        sounding data.
    :param num_sounding_filters_in_first_layer: Number of filters produced by the
        first conv layer for sounding data.
    :raises: ValueError: if
        `num_sounding_heights not in VALID_NUMBERS_OF_SOUNDING_HEIGHTS`.
    """

    cnn_utils.check_activation_function(
        activation_function_string=conv_layer_activation_func_string,
        alpha_for_elu=alpha_for_elu, alpha_for_relu=alpha_for_relu)
    error_checking.assert_is_boolean(use_batch_normalization)

    error_checking.assert_is_integer(num_radar_rows)
    error_checking.assert_is_integer(num_radar_columns)
    error_checking.assert_is_integer(num_radar_dimensions)
    error_checking.assert_is_geq(num_radar_dimensions, 2)
    error_checking.assert_is_leq(num_radar_dimensions, 3)

    if num_radar_dimensions == 2:
        error_checking.assert_is_integer(num_radar_channels)
        error_checking.assert_is_geq(num_radar_channels, 1)
    else:
        error_checking.assert_is_integer(num_radar_fields)
        error_checking.assert_is_geq(num_radar_fields, 1)

    error_checking.assert_is_integer(num_radar_conv_layer_sets)
    error_checking.assert_is_geq(num_radar_conv_layer_sets, 2)
    error_checking.assert_is_integer(num_conv_layers_per_set)
    error_checking.assert_is_geq(num_conv_layers_per_set, 1)
    error_checking.assert_is_integer(num_radar_filters_in_first_layer)
    error_checking.assert_is_geq(num_radar_filters_in_first_layer, 1)

    num_rows_after_last_conv = float(num_radar_rows)
    num_columns_after_last_conv = float(num_radar_columns)
    if num_radar_dimensions == 3:
        num_heights_after_last_conv = float(num_radar_heights)

    for _ in range(num_radar_conv_layer_sets):
        num_rows_after_last_conv = numpy.floor(num_rows_after_last_conv / 2)
        num_columns_after_last_conv = numpy.floor(
            num_columns_after_last_conv / 2)
        if num_radar_dimensions == 3:
            num_heights_after_last_conv = numpy.floor(
                num_heights_after_last_conv / 2)

    error_checking.assert_is_geq(num_rows_after_last_conv, 1)
    error_checking.assert_is_geq(num_columns_after_last_conv, 1)
    if num_radar_dimensions == 3:
        error_checking.assert_is_geq(num_heights_after_last_conv, 1)

    error_checking.assert_is_integer(num_classes)
    error_checking.assert_is_geq(num_classes, 2)

    if conv_layer_dropout_fraction is not None:
        error_checking.assert_is_greater(conv_layer_dropout_fraction, 0.)
        error_checking.assert_is_less_than(conv_layer_dropout_fraction, 1.)
    if dense_layer_dropout_fraction is not None:
        error_checking.assert_is_greater(dense_layer_dropout_fraction, 0.)
        error_checking.assert_is_less_than(dense_layer_dropout_fraction, 1.)

    if l2_weight is not None:
        error_checking.assert_is_greater(l2_weight, 0.)

    if num_sounding_fields is not None:
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

        error_checking.assert_is_integer(num_sounding_conv_layer_sets)
        error_checking.assert_is_geq(num_sounding_conv_layer_sets, 2)
        error_checking.assert_is_integer(num_sounding_filters_in_first_layer)
        error_checking.assert_is_geq(num_sounding_filters_in_first_layer, 1)

        num_heights_after_last_conv = float(num_sounding_heights)
        for _ in range(num_sounding_conv_layer_sets):
            num_heights_after_last_conv = numpy.floor(
                num_heights_after_last_conv / 2)

        error_checking.assert_is_geq(num_heights_after_last_conv, 1)


def _check_training_args(
        num_epochs, num_training_batches_per_epoch,
        num_validation_batches_per_epoch, weight_loss_function, target_name,
        binarize_target, training_fraction_by_class_dict, model_file_name,
        history_file_name, tensorboard_dir_name):
    """Error-checks input args for training.

    :param num_epochs: Number of epochs.
    :param num_training_batches_per_epoch: Number of training batches per epoch.
    :param num_validation_batches_per_epoch: Number of validation batches per
        epoch.
    :param weight_loss_function: Boolean flag.  If False, classes will be
        equally weighted in the loss function.  If True, classes will be
        weighted differently (inversely proportional to their sampling
        fractions, specified in `training_fraction_by_class_dict`).
    :param target_name: Name of target variable.
    :param binarize_target: Boolean flag.  If True, target variable will be
        binarized so that the highest class = 1 and all other classes = 0.
    :param training_fraction_by_class_dict: See doc for
        `deep_learning_utils.class_fractions_to_weights`.
    :param model_file_name: Path to output file (HDF5 format).  Model will be
        saved here after each epoch.
    :param history_file_name: Path to output file (CSV format).  Training
        history will be saved here after each epoch.
    :param tensorboard_dir_name: Path to output directory for TensorBoard log
        files.
    :return: lf_weight_by_class_dict: Dictionary created by
        `deep_learning_utils.class_fractions_to_weights`.
    """

    error_checking.assert_is_integer(num_epochs)
    error_checking.assert_is_geq(num_epochs, 1)
    error_checking.assert_is_integer(num_training_batches_per_epoch)
    error_checking.assert_is_geq(num_training_batches_per_epoch, 1)
    error_checking.assert_is_integer(num_validation_batches_per_epoch)
    error_checking.assert_is_geq(num_validation_batches_per_epoch, 0)

    error_checking.assert_is_boolean(weight_loss_function)
    error_checking.assert_is_boolean(binarize_target)
    if weight_loss_function:
        lf_weight_by_class_dict = dl_utils.class_fractions_to_weights(
            sampling_fraction_by_class_dict=training_fraction_by_class_dict,
            target_name=target_name, binarize_target=binarize_target)
    else:
        lf_weight_by_class_dict = None

    file_system_utils.mkdir_recursive_if_necessary(file_name=model_file_name)
    file_system_utils.mkdir_recursive_if_necessary(file_name=history_file_name)
    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=tensorboard_dir_name)

    return lf_weight_by_class_dict


def _get_sounding_layers(
        num_sounding_heights, num_sounding_fields, num_filters_in_first_layer,
        num_conv_layer_sets, num_conv_layers_per_set, pooling_type_string,
        dropout_fraction, regularizer_object, conv_layer_activation_func_string,
        alpha_for_elu, alpha_for_relu, use_batch_normalization):
    """Returns the part of a CNN that convolves over sounding data.

    :param num_sounding_heights: See doc for `_check_architecture_args`.
    :param num_sounding_fields: Same.
    :param num_filters_in_first_layer: Same.
    :param num_conv_layer_sets: Same.
    :param num_conv_layers_per_set: Same.
    :param pooling_type_string: Same.
    :param dropout_fraction: Same.
    :param regularizer_object: Instance of `keras.regularizers.l1_l2`.  Will be
        applied to each convolutional layer.
    :param conv_layer_activation_func_string: See doc for
        `_check_architecture_args`.
    :param alpha_for_elu: Same.
    :param alpha_for_relu: Same.
    :param use_batch_normalization: Same.
    :return: input_layer_object: Instance of `keras.layers.Input`.
    :return: flattening_layer_object: Instance of `keras.layers.Flatten`.
    """

    sounding_input_layer_object = keras.layers.Input(shape=(
        num_sounding_heights, num_sounding_fields))

    sounding_layer_object = None
    this_num_output_filters = num_filters_in_first_layer + 0

    for _ in range(num_conv_layer_sets):
        if sounding_layer_object is None:
            sounding_layer_object = sounding_input_layer_object
        else:
            this_num_output_filters *= 2

        for _ in range(num_conv_layers_per_set):
            sounding_layer_object = cnn_utils.get_1d_conv_layer(
                num_output_filters=this_num_output_filters,
                num_kernel_pixels=3, num_pixels_per_stride=1,
                padding_type=cnn_utils.YES_PADDING_TYPE,
                kernel_weight_regularizer=regularizer_object,
                is_first_layer=False
            )(sounding_layer_object)

            sounding_layer_object = cnn_utils.get_activation_layer(
                activation_function_string=conv_layer_activation_func_string,
                alpha_for_elu=alpha_for_elu, alpha_for_relu=alpha_for_relu
            )(sounding_layer_object)

            if use_batch_normalization:
                sounding_layer_object = (
                    cnn_utils.get_batch_normalization_layer()(
                        sounding_layer_object))

            if dropout_fraction is not None:
                sounding_layer_object = cnn_utils.get_dropout_layer(
                    dropout_fraction=dropout_fraction
                )(sounding_layer_object)

        sounding_layer_object = cnn_utils.get_1d_pooling_layer(
            num_pixels_in_window=2, pooling_type=pooling_type_string,
            num_pixels_per_stride=2
        )(sounding_layer_object)

    sounding_layer_object = cnn_utils.get_flattening_layer()(
        sounding_layer_object)
    return sounding_input_layer_object, sounding_layer_object


def _get_output_layer_and_loss_function(num_classes):
    """Returns output layer and loss function.

    :param num_classes: Number of classes.
    :return: dense_layer_object: Instance of `keras.layers.Dense`, with no
        activation.
    :return: activation_layer_object: Instance of `keras.layers.Activation`,
        `keras.layers.ELU`, or `keras.layers.LeakyReLU`.
    :return: loss_function: Instance of `keras.losses.binary_crossentropy` or
        `keras.losses.categorical_crossentropy`.
    """

    if num_classes == 2:
        num_output_units = 1
        loss_function = keras.losses.binary_crossentropy
        activation_function_string = cnn_utils.SIGMOID_FUNCTION_STRING
    else:
        num_output_units = num_classes
        loss_function = keras.losses.categorical_crossentropy
        activation_function_string = cnn_utils.SOFTMAX_FUNCTION_STRING

    dense_layer_object = cnn_utils.get_fully_connected_layer(
        num_output_units=num_output_units)
    activation_layer_object = cnn_utils.get_activation_layer(
        activation_function_string=activation_function_string)

    return dense_layer_object, activation_layer_object, loss_function


def _get_checkpoint_object(
        output_model_file_name, monitor_string, use_validation):
    """Creates checkpoint object for Keras model.

    The checkpoint object determines, after each epoch, whether or not the new
    model will be saved.  If the model is saved, the checkpoint object also
    determines *where* the model will be saved.

    :param output_model_file_name: Path to output file (HDF5 format).
    :param monitor_string: Evaluation function reported after each epoch.  Valid
        options are in the list `VALID_MONITOR_STRINGS`.
    :param use_validation: Boolean flag.  If True, after each epoch
        `monitor_string` will be computed for the validation data and the new
        model will be saved only if `monitor_string` has improved.  If False,
        after each epoch `monitor_string` will be computed for the training
        data and the new model will be saved regardless of the value of
        `monitor_string`.
    :return: checkpoint_object: Instance of `keras.callbacks.ModelCheckpoint`.
    :raises: ValueError: if `monitor_string not in VALID_MONITOR_STRINGS`.
    """

    error_checking.assert_is_string(monitor_string)
    if monitor_string not in VALID_MONITOR_STRINGS:
        error_string = (
            '\n\n{0:s}\nValid monitors (listed above) do not include "{1:s}".'
        ).format(str(VALID_MONITOR_STRINGS), monitor_string)
        raise ValueError(error_string)

    if monitor_string == LOSS_AS_MONITOR_STRING:
        mode_string = 'min'
    else:
        mode_string = 'max'

    if use_validation:
        return keras.callbacks.ModelCheckpoint(
            filepath=output_model_file_name,
            monitor='val_{0:s}'.format(monitor_string), verbose=1,
            save_best_only=True, save_weights_only=False, mode=mode_string,
            period=1)

    return keras.callbacks.ModelCheckpoint(
        filepath=output_model_file_name, monitor=monitor_string, verbose=1,
        save_best_only=False, save_weights_only=False, mode=mode_string,
        period=1)


def model_to_feature_generator(model_object, output_layer_name):
    """Reduces Keras model from predictor to feature-generator.

    Specifically, this method turns an intermediate layer H into the output
    layer, removing all layers after H.  The output of the new ("intermediate")
    model will consist of activations from layer H, rather than predictions.

    :param model_object: Instance of `keras.models.Model`.
    :param output_layer_name: Name of new output layer.
    :return: intermediate_model_object: Same as input, except that all layers
        after H are removed.
    """

    error_checking.assert_is_string(output_layer_name)
    return keras.models.Model(
        inputs=model_object.input,
        outputs=model_object.get_layer(name=output_layer_name).output)


def get_2d_mnist_architecture(
        num_radar_rows, num_radar_columns, num_radar_channels, num_classes,
        num_filters_in_first_layer=32, l2_weight=None):
    """Creates CNN with architecture similar to the following example.

    https://github.com/keras-team/keras/blob/master/examples/mnist_cnn.py

    This CNN does 2-D convolution only.

    :param num_radar_rows: See doc for `_check_architecture_args`.
    :param num_radar_columns: Same.
    :param num_radar_channels: Same.
    :param num_classes: Same.
    :param num_filters_in_first_layer: Same.
    :param l2_weight: Same.
    :return: model_object: `keras.models.Sequential` object with the
        aforementioned architecture.
    """

    _check_architecture_args(
        num_radar_rows=num_radar_rows, num_radar_columns=num_radar_columns,
        num_radar_channels=num_radar_channels, num_radar_dimensions=2,
        num_radar_conv_layer_sets=2, num_conv_layers_per_set=1,
        num_radar_filters_in_first_layer=num_filters_in_first_layer,
        num_classes=num_classes, conv_layer_activation_func_string='relu',
        use_batch_normalization=False, l2_weight=l2_weight,
        alpha_for_relu=cnn_utils.DEFAULT_ALPHA_FOR_RELU)

    if l2_weight is None:
        regularizer_object = None
    else:
        regularizer_object = cnn_utils.get_weight_regularizer(
            l1_penalty=0, l2_penalty=l2_weight)

    model_object = keras.models.Sequential()
    layer_object = cnn_utils.get_2d_conv_layer(
        num_output_filters=num_filters_in_first_layer, num_kernel_rows=3,
        num_kernel_columns=3, num_rows_per_stride=1, num_columns_per_stride=1,
        padding_type=cnn_utils.NO_PADDING_TYPE,
        kernel_weight_regularizer=regularizer_object, is_first_layer=True,
        num_input_rows=num_radar_rows, num_input_columns=num_radar_columns,
        num_input_channels=num_radar_channels)
    model_object.add(layer_object)

    model_object.add(cnn_utils.get_activation_layer(
        activation_function_string='relu'))

    layer_object = cnn_utils.get_2d_conv_layer(
        num_output_filters=2 * num_filters_in_first_layer, num_kernel_rows=3,
        num_kernel_columns=3, num_rows_per_stride=1, num_columns_per_stride=1,
        padding_type=cnn_utils.NO_PADDING_TYPE,
        kernel_weight_regularizer=regularizer_object)
    model_object.add(layer_object)

    model_object.add(cnn_utils.get_activation_layer(
        activation_function_string='relu'))

    layer_object = cnn_utils.get_2d_pooling_layer(
        num_rows_in_window=2, num_columns_in_window=2,
        pooling_type=cnn_utils.MAX_POOLING_TYPE, num_rows_per_stride=2,
        num_columns_per_stride=2)
    model_object.add(layer_object)

    layer_object = cnn_utils.get_dropout_layer(dropout_fraction=0.25)
    model_object.add(layer_object)

    layer_object = cnn_utils.get_flattening_layer()
    model_object.add(layer_object)

    layer_object = cnn_utils.get_fully_connected_layer(num_output_units=128)
    model_object.add(layer_object)
    model_object.add(cnn_utils.get_activation_layer(
        activation_function_string='relu'))

    layer_object = cnn_utils.get_dropout_layer(dropout_fraction=0.5)
    model_object.add(layer_object)

    (dense_layer_object, activation_layer_object, loss_function
    ) = _get_output_layer_and_loss_function(num_classes)
    model_object.add(dense_layer_object)
    model_object.add(activation_layer_object)

    model_object.compile(
        loss=loss_function, optimizer=keras.optimizers.Adam(),
        metrics=LIST_OF_METRIC_FUNCTIONS)

    model_object.summary()
    return model_object


def get_2d_swirlnet_architecture(
        num_radar_rows, num_radar_columns, num_radar_channels,
        num_radar_conv_layer_sets, num_conv_layers_per_set, pooling_type_string,
        num_classes,
        conv_layer_activation_func_string=cnn_utils.RELU_FUNCTION_STRING,
        alpha_for_elu=cnn_utils.DEFAULT_ALPHA_FOR_ELU,
        alpha_for_relu=cnn_utils.DEFAULT_ALPHA_FOR_RELU,
        use_batch_normalization=True,
        num_radar_filters_in_first_layer=DEFAULT_NUM_RADAR_FILTERS,
        conv_layer_dropout_fraction=DEFAULT_CONV_LAYER_DROPOUT_FRACTION,
        dense_layer_dropout_fraction=DEFAULT_DENSE_LAYER_DROPOUT_FRACTION,
        l2_weight=DEFAULT_L2_WEIGHT, num_sounding_heights=None,
        num_sounding_fields=None,
        num_sounding_conv_layer_sets=DEFAULT_NUM_SOUNDING_CONV_LAYER_SETS,
        num_sounding_filters_in_first_layer=DEFAULT_NUM_SOUNDING_FILTERS):
    """Creates CNN with architecture similar to the following example.

    https://github.com/djgagne/swirlnet/blob/master/notebooks/
    deep_swirl_tutorial.ipynb

    This CNN does 2-D convolution only.

    :param num_radar_rows: See doc for `_check_architecture_args`.
    :param num_radar_columns: Same.
    :param num_radar_channels: Same.
    :param num_radar_conv_layer_sets: Same.
    :param num_conv_layers_per_set: Same.
    :param pooling_type_string: Same.
    :param num_classes: Same.
    :param conv_layer_activation_func_string: Same.
    :param alpha_for_elu: Same.
    :param alpha_for_relu: Same.
    :param use_batch_normalization: Same.
    :param num_radar_filters_in_first_layer: Same.
    :param conv_layer_dropout_fraction: Same.
    :param dense_layer_dropout_fraction: Same.
    :param l2_weight: Same.
    :param num_sounding_heights: Same.
    :param num_sounding_fields: Same.
    :param num_sounding_conv_layer_sets: Same.
    :param num_sounding_filters_in_first_layer: Same.
    :return: model_object: `keras.models.Sequential` object with the
        aforementioned architecture.
    """

    _check_architecture_args(
        num_radar_rows=num_radar_rows, num_radar_columns=num_radar_columns,
        num_radar_channels=num_radar_channels, num_radar_dimensions=2,
        num_radar_conv_layer_sets=num_radar_conv_layer_sets,
        num_conv_layers_per_set=num_conv_layers_per_set,
        num_radar_filters_in_first_layer=num_radar_filters_in_first_layer,
        num_classes=num_classes,
        conv_layer_activation_func_string=conv_layer_activation_func_string,
        alpha_for_elu=alpha_for_elu, alpha_for_relu=alpha_for_relu,
        use_batch_normalization=use_batch_normalization,
        conv_layer_dropout_fraction=conv_layer_dropout_fraction,
        dense_layer_dropout_fraction=dense_layer_dropout_fraction,
        l2_weight=l2_weight, num_sounding_heights=num_sounding_heights,
        num_sounding_fields=num_sounding_fields,
        num_sounding_conv_layer_sets=num_sounding_conv_layer_sets,
        num_sounding_filters_in_first_layer=num_sounding_filters_in_first_layer)

    if l2_weight is None:
        regularizer_object = None
    else:
        regularizer_object = cnn_utils.get_weight_regularizer(
            l1_penalty=0, l2_penalty=l2_weight)

    radar_input_layer_object = keras.layers.Input(
        shape=(num_radar_rows, num_radar_columns, num_radar_channels))

    radar_layer_object = None
    this_num_output_filters = num_radar_filters_in_first_layer + 0

    for _ in range(num_radar_conv_layer_sets):
        if radar_layer_object is None:
            radar_layer_object = radar_input_layer_object
        else:
            this_num_output_filters *= 2

        for _ in range(num_conv_layers_per_set):
            radar_layer_object = cnn_utils.get_2d_conv_layer(
                num_output_filters=this_num_output_filters,
                num_kernel_rows=5, num_kernel_columns=5, num_rows_per_stride=1,
                num_columns_per_stride=1,
                padding_type=cnn_utils.YES_PADDING_TYPE,
                kernel_weight_regularizer=regularizer_object,
                is_first_layer=False
            )(radar_layer_object)

            radar_layer_object = cnn_utils.get_activation_layer(
                activation_function_string=conv_layer_activation_func_string,
                alpha_for_elu=alpha_for_elu, alpha_for_relu=alpha_for_relu
            )(radar_layer_object)

            if use_batch_normalization:
                radar_layer_object = cnn_utils.get_batch_normalization_layer()(
                    radar_layer_object)

            if conv_layer_dropout_fraction is not None:
                radar_layer_object = cnn_utils.get_dropout_layer(
                    dropout_fraction=conv_layer_dropout_fraction
                )(radar_layer_object)

        radar_layer_object = cnn_utils.get_2d_pooling_layer(
            num_rows_in_window=2, num_columns_in_window=2,
            pooling_type=pooling_type_string, num_rows_per_stride=2,
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
            num_filters_in_first_layer=num_sounding_filters_in_first_layer,
            num_conv_layer_sets=num_sounding_conv_layer_sets,
            num_conv_layers_per_set=num_conv_layers_per_set,
            pooling_type_string=pooling_type_string,
            dropout_fraction=conv_layer_dropout_fraction,
            regularizer_object=regularizer_object,
            conv_layer_activation_func_string=conv_layer_activation_func_string,
            alpha_for_elu=alpha_for_elu, alpha_for_relu=alpha_for_relu,
            use_batch_normalization=use_batch_normalization)

        layer_object = keras.layers.concatenate(
            [radar_layer_object, sounding_layer_object])

    (dense_layer_object, activation_layer_object, loss_function
    ) = _get_output_layer_and_loss_function(num_classes)
    layer_object = dense_layer_object(layer_object)

    if use_batch_normalization:
        layer_object = cnn_utils.get_batch_normalization_layer()(layer_object)
    if dense_layer_dropout_fraction is not None:
        layer_object = cnn_utils.get_dropout_layer(
            dropout_fraction=dense_layer_dropout_fraction)(layer_object)

    layer_object = activation_layer_object(layer_object)

    if num_sounding_fields is None:
        model_object = keras.models.Model(
            inputs=radar_input_layer_object, outputs=layer_object)
    else:
        model_object = keras.models.Model(
            inputs=[radar_input_layer_object, sounding_input_layer_object],
            outputs=layer_object)

    model_object.compile(
        loss=loss_function, optimizer=keras.optimizers.Adam(),
        metrics=LIST_OF_METRIC_FUNCTIONS)

    model_object.summary()
    return model_object


def get_3d_swirlnet_architecture(
        num_radar_rows, num_radar_columns, num_radar_heights, num_radar_fields,
        num_radar_conv_layer_sets, num_conv_layers_per_set, pooling_type_string,
        num_classes,
        conv_layer_activation_func_string=cnn_utils.RELU_FUNCTION_STRING,
        alpha_for_elu=cnn_utils.DEFAULT_ALPHA_FOR_ELU,
        alpha_for_relu=cnn_utils.DEFAULT_ALPHA_FOR_RELU,
        use_batch_normalization=True,
        num_radar_filters_in_first_layer=DEFAULT_NUM_RADAR_FILTERS,
        conv_layer_dropout_fraction=DEFAULT_CONV_LAYER_DROPOUT_FRACTION,
        dense_layer_dropout_fraction=DEFAULT_DENSE_LAYER_DROPOUT_FRACTION,
        l2_weight=DEFAULT_L2_WEIGHT, num_sounding_heights=None,
        num_sounding_fields=None,
        num_sounding_conv_layer_sets=DEFAULT_NUM_SOUNDING_CONV_LAYER_SETS,
        num_sounding_filters_in_first_layer=DEFAULT_NUM_SOUNDING_FILTERS):
    """Creates CNN with architecture similar to the following example.

    https://github.com/djgagne/swirlnet/blob/master/notebooks/
    deep_swirl_tutorial.ipynb

    This CNN does 3-D convolution only.

    :param num_radar_rows: See doc for `_check_architecture_args`.
    :param num_radar_columns: Same.
    :param num_radar_heights: Same.
    :param num_radar_fields: Same.
    :param num_radar_conv_layer_sets: Same.
    :param num_conv_layers_per_set: Same.
    :param pooling_type_string: Same.
    :param num_classes: Same.
    :param conv_layer_activation_func_string: Same.
    :param alpha_for_elu: Same.
    :param alpha_for_relu: Same.
    :param use_batch_normalization: Same.
    :param num_radar_filters_in_first_layer: Same.
    :param conv_layer_dropout_fraction: Same.
    :param dense_layer_dropout_fraction: Same.
    :param l2_weight: Same.
    :param num_sounding_heights: Same.
    :param num_sounding_fields: Same.
    :param num_sounding_conv_layer_sets: Same.
    :param num_sounding_filters_in_first_layer: Same.
    :return: model_object: `keras.models.Sequential` object with the
        aforementioned architecture.
    """

    _check_architecture_args(
        num_radar_rows=num_radar_rows, num_radar_columns=num_radar_columns,
        num_radar_heights=num_radar_heights, num_radar_fields=num_radar_fields,
        num_radar_dimensions=3,
        num_radar_conv_layer_sets=num_radar_conv_layer_sets,
        num_conv_layers_per_set=num_conv_layers_per_set,
        num_radar_filters_in_first_layer=num_radar_filters_in_first_layer,
        num_classes=num_classes,
        conv_layer_activation_func_string=conv_layer_activation_func_string,
        alpha_for_elu=alpha_for_elu, alpha_for_relu=alpha_for_relu,
        use_batch_normalization=use_batch_normalization,
        conv_layer_dropout_fraction=conv_layer_dropout_fraction,
        dense_layer_dropout_fraction=dense_layer_dropout_fraction,
        l2_weight=l2_weight, num_sounding_heights=num_sounding_heights,
        num_sounding_fields=num_sounding_fields,
        num_sounding_conv_layer_sets=num_sounding_conv_layer_sets,
        num_sounding_filters_in_first_layer=num_sounding_filters_in_first_layer)

    if l2_weight is None:
        regularizer_object = None
    else:
        regularizer_object = cnn_utils.get_weight_regularizer(
            l1_penalty=0, l2_penalty=l2_weight)

    radar_input_layer_object = keras.layers.Input(
        shape=(num_radar_rows, num_radar_columns, num_radar_heights,
               num_radar_fields))

    radar_layer_object = None
    this_num_output_filters = num_radar_filters_in_first_layer + 0

    for _ in range(num_radar_conv_layer_sets):
        if radar_layer_object is None:
            radar_layer_object = radar_input_layer_object
        else:
            this_num_output_filters *= 2

        for _ in range(num_conv_layers_per_set):
            radar_layer_object = cnn_utils.get_3d_conv_layer(
                num_output_filters=this_num_output_filters,
                num_kernel_rows=3, num_kernel_columns=3, num_kernel_depths=3,
                num_rows_per_stride=1, num_columns_per_stride=1,
                num_depths_per_stride=1,
                padding_type=cnn_utils.YES_PADDING_TYPE,
                kernel_weight_regularizer=regularizer_object,
                is_first_layer=False
            )(radar_layer_object)

            radar_layer_object = cnn_utils.get_activation_layer(
                activation_function_string=conv_layer_activation_func_string,
                alpha_for_elu=alpha_for_elu, alpha_for_relu=alpha_for_relu
            )(radar_layer_object)

            if use_batch_normalization:
                radar_layer_object = cnn_utils.get_batch_normalization_layer()(
                    radar_layer_object)

            if conv_layer_dropout_fraction is not None:
                radar_layer_object = cnn_utils.get_dropout_layer(
                    dropout_fraction=conv_layer_dropout_fraction
                )(radar_layer_object)

        radar_layer_object = cnn_utils.get_3d_pooling_layer(
            num_rows_in_window=2, num_columns_in_window=2,
            num_depths_in_window=2, pooling_type=pooling_type_string,
            num_rows_per_stride=2, num_columns_per_stride=2,
            num_depths_per_stride=2
        )(radar_layer_object)

    radar_layer_object = cnn_utils.get_flattening_layer()(radar_layer_object)

    if num_sounding_fields is None:
        layer_object = radar_layer_object
    else:
        (sounding_input_layer_object, sounding_layer_object
        ) = _get_sounding_layers(
            num_sounding_heights=num_sounding_heights,
            num_sounding_fields=num_sounding_fields,
            num_filters_in_first_layer=num_sounding_filters_in_first_layer,
            num_conv_layer_sets=num_sounding_conv_layer_sets,
            num_conv_layers_per_set=num_conv_layers_per_set,
            pooling_type_string=pooling_type_string,
            dropout_fraction=conv_layer_dropout_fraction,
            regularizer_object=regularizer_object,
            conv_layer_activation_func_string=conv_layer_activation_func_string,
            alpha_for_elu=alpha_for_elu, alpha_for_relu=alpha_for_relu,
            use_batch_normalization=use_batch_normalization)

        layer_object = keras.layers.concatenate(
            [radar_layer_object, sounding_layer_object])

    (dense_layer_object, activation_layer_object, loss_function
    ) = _get_output_layer_and_loss_function(num_classes)
    layer_object = dense_layer_object(layer_object)

    if use_batch_normalization:
        layer_object = cnn_utils.get_batch_normalization_layer()(layer_object)
    if dense_layer_dropout_fraction is not None:
        layer_object = cnn_utils.get_dropout_layer(
            dropout_fraction=dense_layer_dropout_fraction)(layer_object)

    layer_object = activation_layer_object(layer_object)

    if num_sounding_fields is None:
        model_object = keras.models.Model(
            inputs=radar_input_layer_object, outputs=layer_object)
    else:
        model_object = keras.models.Model(
            inputs=[radar_input_layer_object, sounding_input_layer_object],
            outputs=layer_object)

    model_object.compile(
        loss=loss_function, optimizer=keras.optimizers.Adam(),
        metrics=LIST_OF_METRIC_FUNCTIONS)

    model_object.summary()
    return model_object


def get_2d3d_swirlnet_architecture(
        num_reflectivity_rows, num_reflectivity_columns,
        num_reflectivity_heights, num_azimuthal_shear_fields,
        num_radar_conv_layer_sets, num_conv_layers_per_set, pooling_type_string,
        num_classes,
        conv_layer_activation_func_string=cnn_utils.RELU_FUNCTION_STRING,
        alpha_for_elu=cnn_utils.DEFAULT_ALPHA_FOR_ELU,
        alpha_for_relu=cnn_utils.DEFAULT_ALPHA_FOR_RELU,
        use_batch_normalization=True, num_refl_filters_in_first_layer=8,
        num_shear_filters_in_first_layer=16,
        conv_layer_dropout_fraction=DEFAULT_CONV_LAYER_DROPOUT_FRACTION,
        dense_layer_dropout_fraction=DEFAULT_DENSE_LAYER_DROPOUT_FRACTION,
        l2_weight=DEFAULT_L2_WEIGHT, num_sounding_heights=None,
        num_sounding_fields=None,
        num_sounding_conv_layer_sets=DEFAULT_NUM_SOUNDING_CONV_LAYER_SETS,
        num_sounding_filters_in_first_layer=DEFAULT_NUM_SOUNDING_FILTERS):
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
    :param num_radar_conv_layer_sets: Number of conv-layer sets for radar data
        (after reflectivity and azimuthal shear have been joined).
    :param num_conv_layers_per_set: See doc for `_check_architecture_args`.
    :param pooling_type_string: Same.
    :param num_classes: Same.
    :param conv_layer_activation_func_string: Same.
    :param alpha_for_elu: Same.
    :param alpha_for_relu: Same.
    :param use_batch_normalization: Same.
    :param num_refl_filters_in_first_layer: Number of filters in first
        convolutional layer for reflectivity images.  Number of filters will
        double for each successive layer convolving over reflectivity images.
    :param num_shear_filters_in_first_layer: Same, but for azimuthal
        shear.
    :param conv_layer_dropout_fraction: See doc for `_check_architecture_args`.
    :param dense_layer_dropout_fraction: Same.
    :param l2_weight: Same.
    :param num_sounding_heights: Same.
    :param num_sounding_fields: Same.
    :param num_sounding_conv_layer_sets: Same.
    :param num_sounding_filters_in_first_layer: Same.
    :return: model_object: `keras.models.Sequential` object with the
        aforementioned architecture.
    """

    # Check args for the part that convolves over reflectivity.
    _check_architecture_args(
        num_radar_rows=num_reflectivity_rows,
        num_radar_columns=num_reflectivity_columns,
        num_radar_heights=num_reflectivity_heights, num_radar_fields=1,
        num_radar_dimensions=3, num_radar_conv_layer_sets=3,
        num_conv_layers_per_set=num_conv_layers_per_set,
        num_radar_filters_in_first_layer=num_refl_filters_in_first_layer,
        num_classes=num_classes,
        conv_layer_activation_func_string=conv_layer_activation_func_string,
        alpha_for_elu=alpha_for_elu, alpha_for_relu=alpha_for_relu,
        use_batch_normalization=use_batch_normalization,
        conv_layer_dropout_fraction=conv_layer_dropout_fraction,
        dense_layer_dropout_fraction=dense_layer_dropout_fraction,
        l2_weight=l2_weight, num_sounding_heights=num_sounding_heights,
        num_sounding_fields=num_sounding_fields,
        num_sounding_conv_layer_sets=num_sounding_conv_layer_sets,
        num_sounding_filters_in_first_layer=num_sounding_filters_in_first_layer)

    # Check args for the part that convolves over azimuthal shear.
    num_azimuthal_shear_rows = 2 * num_reflectivity_rows
    num_azimuthal_shear_columns = 2 * num_reflectivity_columns

    _check_architecture_args(
        num_radar_rows=num_azimuthal_shear_rows,
        num_radar_columns=num_azimuthal_shear_columns,
        num_radar_channels=num_azimuthal_shear_fields, num_radar_dimensions=2,
        num_radar_conv_layer_sets=1,
        num_radar_filters_in_first_layer=num_shear_filters_in_first_layer,
        num_conv_layers_per_set=num_conv_layers_per_set,
        num_classes=num_classes,
        conv_layer_activation_func_string=conv_layer_activation_func_string,
        alpha_for_elu=alpha_for_elu, alpha_for_relu=alpha_for_relu,
        use_batch_normalization=use_batch_normalization)

    # Check args for the part that convolves over reflectivity and az shear.
    num_radar_channels = (
        num_shear_filters_in_first_layer + num_refl_filters_in_first_layer * 8)

    _check_architecture_args(
        num_radar_rows=num_reflectivity_rows,
        num_radar_columns=num_reflectivity_columns,
        num_radar_channels=num_radar_channels, num_radar_dimensions=2,
        num_radar_conv_layer_sets=num_radar_conv_layer_sets,
        num_conv_layers_per_set=num_conv_layers_per_set,
        num_radar_filters_in_first_layer=num_radar_channels * 2,
        num_classes=num_classes,
        conv_layer_activation_func_string=conv_layer_activation_func_string,
        alpha_for_elu=alpha_for_elu, alpha_for_relu=alpha_for_relu,
        use_batch_normalization=use_batch_normalization)

    # Flatten reflectivity images from 3-D to 2-D (by convolving over height).
    if l2_weight is None:
        regularizer_object = None
    else:
        regularizer_object = cnn_utils.get_weight_regularizer(
            l1_penalty=0, l2_penalty=l2_weight)

    refl_input_layer_object = keras.layers.Input(
        shape=(num_reflectivity_rows, num_reflectivity_columns,
               num_reflectivity_heights, 1))

    reflectivity_layer_object = None
    this_num_refl_filters = num_refl_filters_in_first_layer + 0

    for _ in range(3):
        if reflectivity_layer_object is None:
            this_target_shape = (
                num_reflectivity_rows * num_reflectivity_columns,
                num_reflectivity_heights, 1)
            reflectivity_layer_object = keras.layers.Reshape(
                target_shape=this_target_shape
            )(refl_input_layer_object)
        else:
            this_num_refl_filters *= 2

        for _ in range(num_conv_layers_per_set):
            reflectivity_layer_object = cnn_utils.get_2d_conv_layer(
                num_output_filters=this_num_refl_filters,
                num_kernel_rows=1, num_kernel_columns=3, num_rows_per_stride=1,
                num_columns_per_stride=1,
                padding_type=cnn_utils.YES_PADDING_TYPE,
                kernel_weight_regularizer=regularizer_object,
                is_first_layer=False
            )(reflectivity_layer_object)

            reflectivity_layer_object = cnn_utils.get_activation_layer(
                activation_function_string=conv_layer_activation_func_string,
                alpha_for_elu=alpha_for_elu, alpha_for_relu=alpha_for_relu
            )(reflectivity_layer_object)

            if use_batch_normalization:
                reflectivity_layer_object = (
                    cnn_utils.get_batch_normalization_layer()(
                        reflectivity_layer_object))

            if conv_layer_dropout_fraction is not None:
                reflectivity_layer_object = cnn_utils.get_dropout_layer(
                    dropout_fraction=conv_layer_dropout_fraction
                )(reflectivity_layer_object)

        reflectivity_layer_object = cnn_utils.get_2d_pooling_layer(
            num_rows_in_window=1, num_columns_in_window=2,
            pooling_type=pooling_type_string, num_rows_per_stride=1,
            num_columns_per_stride=2
        )(reflectivity_layer_object)

    this_target_shape = (
        num_reflectivity_rows, num_reflectivity_columns, this_num_refl_filters)
    reflectivity_layer_object = keras.layers.Reshape(
        target_shape=this_target_shape
    )(reflectivity_layer_object)

    # Halve the size of azimuthal-shear images (to make them match
    # reflectivity).
    az_shear_input_layer_object = keras.layers.Input(
        shape=(num_azimuthal_shear_rows, num_azimuthal_shear_columns,
               num_azimuthal_shear_fields))

    azimuthal_shear_layer_object = None
    for _ in range(num_conv_layers_per_set):
        if azimuthal_shear_layer_object is None:
            azimuthal_shear_layer_object = az_shear_input_layer_object

        azimuthal_shear_layer_object = cnn_utils.get_2d_conv_layer(
            num_output_filters=num_shear_filters_in_first_layer,
            num_kernel_rows=3, num_kernel_columns=3, num_rows_per_stride=1,
            num_columns_per_stride=1, padding_type=cnn_utils.YES_PADDING_TYPE,
            kernel_weight_regularizer=regularizer_object, is_first_layer=False
        )(azimuthal_shear_layer_object)

        azimuthal_shear_layer_object = cnn_utils.get_activation_layer(
            activation_function_string=conv_layer_activation_func_string,
            alpha_for_elu=alpha_for_elu, alpha_for_relu=alpha_for_relu
        )(azimuthal_shear_layer_object)

        if use_batch_normalization:
            azimuthal_shear_layer_object = (
                cnn_utils.get_batch_normalization_layer()(
                    azimuthal_shear_layer_object))

        if conv_layer_dropout_fraction is not None:
            azimuthal_shear_layer_object = cnn_utils.get_dropout_layer(
                dropout_fraction=conv_layer_dropout_fraction
            )(azimuthal_shear_layer_object)

    azimuthal_shear_layer_object = cnn_utils.get_2d_pooling_layer(
        num_rows_in_window=2, num_columns_in_window=2,
        pooling_type=pooling_type_string, num_rows_per_stride=2,
        num_columns_per_stride=2
    )(azimuthal_shear_layer_object)

    # Concatenate reflectivity and az-shear filters.
    radar_layer_object = keras.layers.concatenate(
        [reflectivity_layer_object, azimuthal_shear_layer_object], axis=-1)

    # Convolve over reflectivity and az-shear filters.
    this_num_output_filters = 2 * (
        num_shear_filters_in_first_layer + this_num_refl_filters)

    for i in range(num_radar_conv_layer_sets):
        if i != 0:
            this_num_output_filters *= 2

        for _ in range(num_conv_layers_per_set):
            radar_layer_object = cnn_utils.get_2d_conv_layer(
                num_output_filters=this_num_output_filters, num_kernel_rows=3,
                num_kernel_columns=3, num_rows_per_stride=1,
                num_columns_per_stride=1,
                padding_type=cnn_utils.YES_PADDING_TYPE,
                kernel_weight_regularizer=regularizer_object
            )(radar_layer_object)

            radar_layer_object = cnn_utils.get_activation_layer(
                activation_function_string=conv_layer_activation_func_string,
                alpha_for_elu=alpha_for_elu, alpha_for_relu=alpha_for_relu
            )(radar_layer_object)

            if use_batch_normalization:
                radar_layer_object = cnn_utils.get_batch_normalization_layer()(
                    radar_layer_object)

            if conv_layer_dropout_fraction is not None:
                radar_layer_object = cnn_utils.get_dropout_layer(
                    dropout_fraction=conv_layer_dropout_fraction
                )(radar_layer_object)

        radar_layer_object = cnn_utils.get_2d_pooling_layer(
            num_rows_in_window=2, num_columns_in_window=2,
            pooling_type=pooling_type_string, num_rows_per_stride=2,
            num_columns_per_stride=2
        )(radar_layer_object)

    radar_layer_object = cnn_utils.get_flattening_layer()(radar_layer_object)

    # Convolve over soundings.
    if num_sounding_fields is None:
        layer_object = radar_layer_object
    else:
        (sounding_input_layer_object, sounding_layer_object
        ) = _get_sounding_layers(
            num_sounding_heights=num_sounding_heights,
            num_sounding_fields=num_sounding_fields,
            num_filters_in_first_layer=num_sounding_filters_in_first_layer,
            num_conv_layer_sets=num_sounding_conv_layer_sets,
            num_conv_layers_per_set=num_conv_layers_per_set,
            pooling_type_string=pooling_type_string,
            dropout_fraction=conv_layer_dropout_fraction,
            regularizer_object=regularizer_object,
            conv_layer_activation_func_string=conv_layer_activation_func_string,
            alpha_for_elu=alpha_for_elu, alpha_for_relu=alpha_for_relu,
            use_batch_normalization=use_batch_normalization)

        layer_object = keras.layers.concatenate(
            [radar_layer_object, sounding_layer_object])

    (dense_layer_object, activation_layer_object, loss_function
    ) = _get_output_layer_and_loss_function(num_classes)
    layer_object = dense_layer_object(layer_object)

    if use_batch_normalization:
        layer_object = cnn_utils.get_batch_normalization_layer()(layer_object)
    if dense_layer_dropout_fraction is not None:
        layer_object = cnn_utils.get_dropout_layer(
            dropout_fraction=dense_layer_dropout_fraction)(layer_object)

    layer_object = activation_layer_object(layer_object)

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
        loss=loss_function, optimizer=keras.optimizers.Adam(),
        metrics=LIST_OF_METRIC_FUNCTIONS)

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


def write_model_metadata(pickle_file_name, metadata_dict):
    """Writes metadata for CNN to Pickle file.

    :param pickle_file_name: Path to output file.
    :param metadata_dict: Dictionary with the following keys.
    metadata_dict['num_epochs']: Number of epochs.
    metadata_dict['num_training_batches_per_epoch']: Number of training batches
        per epoch.
    metadata_dict['monitor_string']: See doc for `_get_checkpoint_object`.
    metadata_dict['weight_loss_function']: See doc for `_check_training_args`.
    metadata_dict['num_validation_batches_per_epoch']: Number of validation
        batches per epoch.
    metadata_dict['validn_file_name_matrix']: numpy array created by
        `training_validation_io.find_radar_files_2d` or
        `training_validation_io.find_radar_files_3d`, for validation only.
    metadata_dict['validn_file_name_matrix_pos_targets_only']: Same as
        "validn_file_name_matrix", but containing only examples with target
        class = 1.
    metadata_dict['training_file_name_matrix']: numpy array created by
        `training_validation_io.find_radar_files_2d` or
        `training_validation_io.find_radar_files_3d`, for training only.
    metadata_dict['training_file_name_matrix_pos_targets_only']: Same as
        "training_file_name_matrix", but containing only examples with target
        class = 1.
    metadata_dict['use_2d3d_convolution']: Boolean flag.  If True, the network
        performs a hybrid of 2-D and 3-D convolution, so the data-generator is
        `training_validation_io.storm_image_generator_2d3d_myrorss`.
    metadata_dict['num_examples_per_batch']: Number of examples (storm objects)
        per batch.
    metadata_dict['num_examples_per_file']: Number of examples (storm objects)
        per file.
    metadata_dict['target_name']: Name of target variable.
    metadata_dict['num_rows_to_keep']: See doc for
        `training_validation_io.storm_image_generator_2d`.
    metadata_dict['num_columns_to_keep']: Same.
    metadata_dict['normalization_type_string']: Same.
    metadata_dict['min_normalized_value']: Same.
    metadata_dict['max_normalized_value']: Same.
    metadata_dict['normalization_param_file_name']: Same.
    metadata_dict['binarize_target']: Same.
    metadata_dict['sampling_fraction_by_class_dict']: Same.
    metadata_dict['sounding_field_names']: Same.
    metadata_dict['sounding_lag_time_sec']: Same.
    metadata_dict['num_translations']: Same.
    metadata_dict['max_translation_pixels']: Same.
    metadata_dict['num_rotations']: Same.
    metadata_dict['max_absolute_rotation_angle_deg']: Same.
    metadata_dict['num_noisings']: Same.
    metadata_dict['max_noise_standard_deviation']: Same.
    metadata_dict['refl_masking_threshold_dbz']: See doc for
        `training_validation_io.storm_image_generator_3d`.
    metadata_dict['radar_source']: Data source (must be accepted by
        `radar_utils.check_field_name`).
    metadata_dict['radar_field_names']: See doc for
        `training_validation_io.find_radar_files_2d` or
        `training_validation_io.find_radar_files_3d`.
    metadata_dict['radar_heights_m_agl']: Same.
    metadata_dict['reflectivity_heights_m_agl']: Same.
    metadata_dict['sounding_heights_m_agl']: 1-D numpy array of sounding heights
        (integer metres above ground level).

    :raises: ValueError: if any expected keys are missing from `metadata_dict`.
    """

    orig_metadata_dict = metadata_dict.copy()
    metadata_dict = DEFAULT_METADATA_DICT.copy()
    metadata_dict.update(orig_metadata_dict)

    missing_keys = list(set(MODEL_METADATA_KEYS) - set(metadata_dict.keys()))
    if len(missing_keys):
        error_string = (
            'The following expected keys are missing from the dictionary.'
            '\n{0:s}'
        ).format(str(missing_keys))
        raise ValueError(error_string)

    file_system_utils.mkdir_recursive_if_necessary(file_name=pickle_file_name)
    pickle_file_handle = open(pickle_file_name, 'wb')
    pickle.dump(metadata_dict, pickle_file_handle)
    pickle_file_handle.close()


def read_model_metadata(pickle_file_name):
    """Reads metadata from Pickle file.

    :param pickle_file_name: Path to input file.
    :return: metadata_dict: See doc for `write_model_metadata`.
    :raises: ValueError: if any expected keys are missing from `metadata_dict`.
    """

    pickle_file_handle = open(pickle_file_name, 'rb')
    metadata_dict = pickle.load(pickle_file_handle)
    pickle_file_handle.close()

    missing_keys = list(set(MODEL_METADATA_KEYS) - set(metadata_dict.keys()))
    if len(missing_keys):
        error_string = (
            'The following expected keys are missing from the dictionary.'
            '\n{0:s}'
        ).format(str(missing_keys))
        raise ValueError(error_string)

    return metadata_dict


def train_2d_cnn(
        model_object, model_file_name, history_file_name, tensorboard_dir_name,
        num_epochs, num_training_batches_per_epoch,
        monitor_string=LOSS_AS_MONITOR_STRING, weight_loss_function=False,
        training_option_dict=None, num_validation_batches_per_epoch=0,
        validn_file_name_matrix=None,
        validn_file_name_matrix_pos_targets_only=None):
    """Trains CNN with 2-D radar images.

    :param model_object: Instance of `keras.models.Sequential`.
    :param model_file_name: See doc for `_check_training_args`.
    :param history_file_name: Same.
    :param tensorboard_dir_name: Same.
    :param num_epochs: Number of epochs.
    :param num_training_batches_per_epoch: Number of training batches per epoch.
    :param monitor_string: Number of training batches per epoch.
    :param weight_loss_function: See doc for `_check_training_args`.
    :param training_option_dict: See doc for
        `training_validation_io.storm_image_generator_2d`.  This exact
        dictionary will be used to generate training data.  If
        `num_validation_batches_per_epoch > 0`, the next 3 arguments will be
        used to change the dictionary to generate validation data.
    :param num_validation_batches_per_epoch: Number of validation batches per
        epoch.
    :param validn_file_name_matrix:
        [used iff num_validation_batches_per_epoch > 0]
        See doc for `radar_file_name_matrix` in
        `training_validation_io.storm_image_generator_2d`.
    :param validn_file_name_matrix_pos_targets_only:
        [used iff num_validation_batches_per_epoch > 0]
        See doc for `radar_file_name_matrix_pos_targets_only` in
        `training_validation_io.storm_image_generator_2d`.
    """

    lf_weight_by_class_dict = _check_training_args(
        num_epochs=num_epochs,
        num_training_batches_per_epoch=num_training_batches_per_epoch,
        num_validation_batches_per_epoch=num_validation_batches_per_epoch,
        weight_loss_function=weight_loss_function,
        target_name=training_option_dict[trainval_io.TARGET_NAME_KEY],
        binarize_target=training_option_dict[trainval_io.BINARIZE_TARGET_KEY],
        training_fraction_by_class_dict=training_option_dict[
            trainval_io.SAMPLING_FRACTIONS_KEY],
        model_file_name=model_file_name, history_file_name=history_file_name,
        tensorboard_dir_name=tensorboard_dir_name)

    history_object = keras.callbacks.CSVLogger(
        filename=history_file_name, separator=',', append=False)

    embedding_layer_names = [
        this_layer.name for this_layer in model_object.layers if
        this_layer.name.startswith('dense') or
        this_layer.name.startswith('conv')
    ]

    tensorboard_object = keras.callbacks.TensorBoard(
        log_dir=tensorboard_dir_name, histogram_freq=0,
        batch_size=training_option_dict[
            trainval_io.NUM_EXAMPLES_PER_BATCH_KEY],
        write_graph=True, write_grads=True, write_images=True,
        embeddings_freq=1, embeddings_layer_names=embedding_layer_names)

    checkpoint_object = _get_checkpoint_object(
        output_model_file_name=model_file_name, monitor_string=monitor_string,
        use_validation=num_validation_batches_per_epoch is not None)

    if num_validation_batches_per_epoch > 0:
        model_object.fit_generator(
            generator=trainval_io.storm_image_generator_2d(
                training_option_dict),
            steps_per_epoch=num_training_batches_per_epoch, epochs=num_epochs,
            verbose=1, class_weight=lf_weight_by_class_dict,
            callbacks=[checkpoint_object, history_object, tensorboard_object])

    else:
        validation_option_dict = copy.deepcopy(training_option_dict)
        validation_option_dict[
            trainval_io.RADAR_FILE_NAMES_KEY] = validn_file_name_matrix
        validation_option_dict[
            trainval_io.POSITIVE_RADAR_FILE_NAMES_KEY
        ] = validn_file_name_matrix_pos_targets_only

        model_object.fit_generator(
            generator=trainval_io.storm_image_generator_2d(
                training_option_dict),
            steps_per_epoch=num_training_batches_per_epoch, epochs=num_epochs,
            verbose=1, class_weight=lf_weight_by_class_dict,
            callbacks=[checkpoint_object, history_object, tensorboard_object],
            validation_data=trainval_io.storm_image_generator_2d(
                validation_option_dict),
            validation_steps=num_validation_batches_per_epoch)


def train_3d_cnn(
        model_object, model_file_name, history_file_name, tensorboard_dir_name,
        num_epochs, num_training_batches_per_epoch,
        monitor_string=LOSS_AS_MONITOR_STRING, weight_loss_function=False,
        training_option_dict=None, num_validation_batches_per_epoch=0,
        validn_file_name_matrix=None,
        validn_file_name_matrix_pos_targets_only=None):
    """Trains CNN with 3-D radar images.

    :param model_object: Instance of `keras.models.Sequential`.
    :param model_file_name: See doc for `_check_training_args`.
    :param history_file_name: Same.
    :param tensorboard_dir_name: Same.
    :param num_epochs: Number of epochs.
    :param num_training_batches_per_epoch: Number of training batches per epoch.
    :param monitor_string: Number of training batches per epoch.
    :param weight_loss_function: See doc for `_check_training_args`.
    :param training_option_dict: See doc for
        `training_validation_io.storm_image_generator_3d`.  This exact
        dictionary will be used to generate training data.  If
        `num_validation_batches_per_epoch > 0`, the next 3 arguments will be
        used to change the dictionary to generate validation data.
    :param num_validation_batches_per_epoch: Number of validation batches per
        epoch.
    :param validn_file_name_matrix:
        [used iff num_validation_batches_per_epoch > 0]
        See doc for `radar_file_name_matrix` in
        `training_validation_io.storm_image_generator_3d`.
    :param validn_file_name_matrix_pos_targets_only:
        [used iff num_validation_batches_per_epoch > 0]
        See doc for `radar_file_name_matrix_pos_targets_only` in
        `training_validation_io.storm_image_generator_3d`.
    """

    lf_weight_by_class_dict = _check_training_args(
        num_epochs=num_epochs,
        num_training_batches_per_epoch=num_training_batches_per_epoch,
        num_validation_batches_per_epoch=num_validation_batches_per_epoch,
        weight_loss_function=weight_loss_function,
        target_name=training_option_dict[trainval_io.TARGET_NAME_KEY],
        binarize_target=training_option_dict[trainval_io.BINARIZE_TARGET_KEY],
        training_fraction_by_class_dict=training_option_dict[
            trainval_io.SAMPLING_FRACTIONS_KEY],
        model_file_name=model_file_name, history_file_name=history_file_name,
        tensorboard_dir_name=tensorboard_dir_name)

    history_object = keras.callbacks.CSVLogger(
        filename=history_file_name, separator=',', append=False)

    embedding_layer_names = [
        this_layer.name for this_layer in model_object.layers if
        this_layer.name.startswith('dense') or
        this_layer.name.startswith('conv')
    ]

    tensorboard_object = keras.callbacks.TensorBoard(
        log_dir=tensorboard_dir_name, histogram_freq=0,
        batch_size=training_option_dict[
            trainval_io.NUM_EXAMPLES_PER_BATCH_KEY],
        write_graph=True, write_grads=True, write_images=True,
        embeddings_freq=1, embeddings_layer_names=embedding_layer_names)

    checkpoint_object = _get_checkpoint_object(
        output_model_file_name=model_file_name, monitor_string=monitor_string,
        use_validation=num_validation_batches_per_epoch is not None)

    if num_validation_batches_per_epoch is None:
        model_object.fit_generator(
            generator=trainval_io.storm_image_generator_3d(
                training_option_dict),
            steps_per_epoch=num_training_batches_per_epoch, epochs=num_epochs,
            verbose=1, class_weight=lf_weight_by_class_dict,
            callbacks=[checkpoint_object, history_object, tensorboard_object])

    else:
        validation_option_dict = copy.deepcopy(training_option_dict)
        validation_option_dict[
            trainval_io.RADAR_FILE_NAMES_KEY] = validn_file_name_matrix
        validation_option_dict[
            trainval_io.POSITIVE_RADAR_FILE_NAMES_KEY
        ] = validn_file_name_matrix_pos_targets_only

        model_object.fit_generator(
            generator=trainval_io.storm_image_generator_3d(
                training_option_dict),
            steps_per_epoch=num_training_batches_per_epoch, epochs=num_epochs,
            verbose=1, class_weight=lf_weight_by_class_dict,
            callbacks=[checkpoint_object, history_object, tensorboard_object],
            validation_data=trainval_io.storm_image_generator_3d(
                validation_option_dict),
            validation_steps=num_validation_batches_per_epoch)


def train_2d3d_cnn(
        model_object, model_file_name, history_file_name, tensorboard_dir_name,
        num_epochs, num_training_batches_per_epoch,
        monitor_string=LOSS_AS_MONITOR_STRING, weight_loss_function=False,
        training_option_dict=None, num_validation_batches_per_epoch=0,
        validn_file_name_matrix=None,
        validn_file_name_matrix_pos_targets_only=None):
    """Trains CNN with 2-D and 3-D radar images.

    :param model_object: Instance of `keras.models.Sequential`.
    :param model_file_name: See doc for `_check_training_args`.
    :param history_file_name: Same.
    :param tensorboard_dir_name: Same.
    :param num_epochs: Number of epochs.
    :param num_training_batches_per_epoch: Number of training batches per epoch.
    :param monitor_string: Number of training batches per epoch.
    :param weight_loss_function: See doc for `_check_training_args`.
    :param training_option_dict: See doc for
        `training_validation_io.storm_image_generator_2d3d_myrorss`.  This exact
        dictionary will be used to generate training data.  If
        `num_validation_batches_per_epoch > 0`, the next 3 arguments will be
        used to change the dictionary to generate validation data.
    :param num_validation_batches_per_epoch: Number of validation batches per
        epoch.
    :param validn_file_name_matrix:
        [used iff num_validation_batches_per_epoch > 0]
        See doc for `radar_file_name_matrix` in
        `training_validation_io.storm_image_generator_2d3d_myrorss`.
    :param validn_file_name_matrix_pos_targets_only:
        [used iff num_validation_batches_per_epoch > 0]
        See doc for `radar_file_name_matrix_pos_targets_only` in
        `training_validation_io.storm_image_generator_2d3d_myrorss`.
    """

    lf_weight_by_class_dict = _check_training_args(
        num_epochs=num_epochs,
        num_training_batches_per_epoch=num_training_batches_per_epoch,
        num_validation_batches_per_epoch=num_validation_batches_per_epoch,
        weight_loss_function=weight_loss_function,
        target_name=training_option_dict[trainval_io.TARGET_NAME_KEY],
        binarize_target=training_option_dict[trainval_io.BINARIZE_TARGET_KEY],
        training_fraction_by_class_dict=training_option_dict[
            trainval_io.SAMPLING_FRACTIONS_KEY],
        model_file_name=model_file_name, history_file_name=history_file_name,
        tensorboard_dir_name=tensorboard_dir_name)

    history_object = keras.callbacks.CSVLogger(
        filename=history_file_name, separator=',', append=False)

    embedding_layer_names = [
        this_layer.name for this_layer in model_object.layers if
        this_layer.name.startswith('dense') or
        this_layer.name.startswith('conv')
    ]

    tensorboard_object = keras.callbacks.TensorBoard(
        log_dir=tensorboard_dir_name, histogram_freq=0,
        batch_size=training_option_dict[
            trainval_io.NUM_EXAMPLES_PER_BATCH_KEY],
        write_graph=True, write_grads=True, write_images=True,
        embeddings_freq=1, embeddings_layer_names=embedding_layer_names)

    checkpoint_object = _get_checkpoint_object(
        output_model_file_name=model_file_name, monitor_string=monitor_string,
        use_validation=num_validation_batches_per_epoch is not None)

    if num_validation_batches_per_epoch is None:
        model_object.fit_generator(
            generator=trainval_io.storm_image_generator_2d3d_myrorss(
                training_option_dict),
            steps_per_epoch=num_training_batches_per_epoch, epochs=num_epochs,
            verbose=1, class_weight=lf_weight_by_class_dict,
            callbacks=[checkpoint_object, history_object, tensorboard_object])

    else:
        validation_option_dict = copy.deepcopy(training_option_dict)
        validation_option_dict[
            trainval_io.RADAR_FILE_NAMES_KEY] = validn_file_name_matrix
        validation_option_dict[
            trainval_io.POSITIVE_RADAR_FILE_NAMES_KEY
        ] = validn_file_name_matrix_pos_targets_only

        model_object.fit_generator(
            generator=trainval_io.storm_image_generator_2d3d_myrorss(
                training_option_dict),
            steps_per_epoch=num_training_batches_per_epoch, epochs=num_epochs,
            verbose=1, class_weight=lf_weight_by_class_dict,
            callbacks=[checkpoint_object, history_object, tensorboard_object],
            validation_data=trainval_io.storm_image_generator_2d3d_myrorss(
                validation_option_dict),
            validation_steps=num_validation_batches_per_epoch)


def apply_2d_cnn(
        model_object, radar_image_matrix, sounding_matrix=None,
        return_features=False, output_layer_name=None):
    """Applies CNN to 2-D radar images.

    If return_features = True, this method will return only features
    (activations from an intermediate layer H).  If return_features = False,
    this method will return predictions.

    :param model_object: Instance of `keras.models.Sequential`.
    :param radar_image_matrix: E-by-M-by-N-by-C numpy array of storm-centered
        radar images.
    :param sounding_matrix: [may be None]
        numpy array (E x H_s x F_s) of storm-centered soundings.
    :param return_features: See general discussion above.
    :param output_layer_name: [used only if return_features = True]
        Name of intermediate layer from which activations will be returned.

    If return_features = True...

    :return: feature_matrix: numpy array of features.

    If return_features = False...

    :return: class_probability_matrix: E-by-K numpy array of class
        probabilities.  class_probability_matrix[i, k] is the forecast
        probability that the [i]th storm object belongs to the [k]th class.
        Classes are mutually exclusive and collectively exhaustive, so the sum
        across each row is 1.
    """

    dl_utils.check_radar_images(
        radar_image_matrix=radar_image_matrix, min_num_dimensions=4,
        max_num_dimensions=4)
    num_examples = radar_image_matrix.shape[0]

    error_checking.assert_is_boolean(return_features)
    if sounding_matrix is not None:
        dl_utils.check_soundings(
            sounding_matrix=sounding_matrix, num_examples=num_examples)

    if return_features:
        intermediate_model_object = model_to_feature_generator(
            model_object=model_object, output_layer_name=output_layer_name)
        if sounding_matrix is None:
            return intermediate_model_object.predict(
                radar_image_matrix, batch_size=num_examples)

        return intermediate_model_object.predict(
            [radar_image_matrix, sounding_matrix], batch_size=num_examples)

    if sounding_matrix is None:
        these_probabilities = model_object.predict(
            radar_image_matrix, batch_size=num_examples)
    else:
        these_probabilities = model_object.predict(
            [radar_image_matrix, sounding_matrix], batch_size=num_examples)

    if these_probabilities.shape[-1] > 1:
        return these_probabilities

    these_probabilities = numpy.reshape(
        these_probabilities, (len(these_probabilities), 1))
    return numpy.hstack((1. - these_probabilities, these_probabilities))


def apply_3d_cnn(
        model_object, radar_image_matrix, sounding_matrix=None,
        return_features=False, output_layer_name=None):
    """Applies CNN to 3-D radar images.

    :param model_object: Instance of `keras.models.Sequential`.
    :param radar_image_matrix: numpy array (E x M x N x H_r x F_r) of storm-
        centered radar images.
    :param sounding_matrix: See doc for `apply_2d_cnn`.
    :param return_features: Same.
    :param output_layer_name: Same.

    If return_features = True...

    :return: feature_matrix: See doc for `apply_2d_cnn`.

    If return_features = False...

    :return: class_probability_matrix: See doc for `apply_2d_cnn`.
    """

    dl_utils.check_radar_images(
        radar_image_matrix=radar_image_matrix, min_num_dimensions=5,
        max_num_dimensions=5)
    num_examples = radar_image_matrix.shape[0]

    if sounding_matrix is not None:
        dl_utils.check_soundings(
            sounding_matrix=sounding_matrix, num_examples=num_examples)

    error_checking.assert_is_boolean(return_features)
    if return_features:
        intermediate_model_object = model_to_feature_generator(
            model_object=model_object, output_layer_name=output_layer_name)
        if sounding_matrix is None:
            return intermediate_model_object.predict(
                radar_image_matrix, batch_size=num_examples)

        return intermediate_model_object.predict(
            [radar_image_matrix, sounding_matrix], batch_size=num_examples)

    if sounding_matrix is None:
        these_probabilities = model_object.predict(
            radar_image_matrix, batch_size=num_examples)
    else:
        these_probabilities = model_object.predict(
            [radar_image_matrix, sounding_matrix], batch_size=num_examples)

    if these_probabilities.shape[-1] > 1:
        return these_probabilities

    these_probabilities = numpy.reshape(
        these_probabilities, (len(these_probabilities), 1))
    return numpy.hstack((1. - these_probabilities, these_probabilities))


def apply_2d3d_cnn(
        model_object, reflectivity_image_matrix_dbz,
        azimuthal_shear_image_matrix_s01, sounding_matrix=None,
        return_features=False, output_layer_name=None):
    """Applies CNN to 2-D azimuthal-shear images and 3-D reflectivity images.

    :param model_object: Instance of `keras.models.Sequential`.
    :param reflectivity_image_matrix_dbz: numpy array (E x m x n x H_r x 1) of
        storm-centered reflectivity images.
    :param azimuthal_shear_image_matrix_s01: numpy array (E x M x N x F_a) of
        storm-centered azimuthal-shear images.
    :param sounding_matrix: See doc for `apply_2d_cnn`.
    :param return_features: Same.
    :param output_layer_name: Same.

    If return_features = True...

    :return: feature_matrix: See doc for `apply_2d_cnn`.

    If return_features = False...

    :return: class_probability_matrix: See doc for `apply_2d_cnn`.
    """

    dl_utils.check_radar_images(
        radar_image_matrix=reflectivity_image_matrix_dbz, min_num_dimensions=5,
        max_num_dimensions=5)
    dl_utils.check_radar_images(
        radar_image_matrix=azimuthal_shear_image_matrix_s01,
        min_num_dimensions=4, max_num_dimensions=4)

    expected_dimensions = numpy.array(
        reflectivity_image_matrix_dbz.shape[:-1] + (1,))
    error_checking.assert_is_numpy_array(
        reflectivity_image_matrix_dbz, exact_dimensions=expected_dimensions)

    num_examples = reflectivity_image_matrix_dbz.shape[0]
    expected_dimensions = numpy.array(
        (num_examples,) + azimuthal_shear_image_matrix_s01[1:])
    error_checking.assert_is_numpy_array(
        azimuthal_shear_image_matrix_s01, exact_dimensions=expected_dimensions)

    error_checking.assert_is_boolean(return_features)
    if sounding_matrix is not None:
        dl_utils.check_soundings(
            sounding_matrix=sounding_matrix, num_examples=num_examples)

    if return_features:
        intermediate_model_object = model_to_feature_generator(
            model_object=model_object, output_layer_name=output_layer_name)
        if sounding_matrix is None:
            return intermediate_model_object.predict(
                [reflectivity_image_matrix_dbz,
                 azimuthal_shear_image_matrix_s01],
                batch_size=num_examples)

        return intermediate_model_object.predict(
            [reflectivity_image_matrix_dbz, azimuthal_shear_image_matrix_s01,
             sounding_matrix],
            batch_size=num_examples)

    if sounding_matrix is None:
        these_probabilities = model_object.predict(
            [reflectivity_image_matrix_dbz, azimuthal_shear_image_matrix_s01],
            batch_size=num_examples)
    else:
        these_probabilities = model_object.predict(
            [reflectivity_image_matrix_dbz, azimuthal_shear_image_matrix_s01,
             sounding_matrix],
            batch_size=num_examples)

    if these_probabilities.shape[-1] > 1:
        return these_probabilities

    these_probabilities = numpy.reshape(
        these_probabilities, (len(these_probabilities), 1))
    return numpy.hstack((1. - these_probabilities, these_probabilities))


def write_features(
        netcdf_file_name, feature_matrix, target_values, num_classes,
        append_to_file=False):
    """Writes features (activations of intermediate layer) to NetCDF file.

    :param netcdf_file_name: Path to output file.
    :param feature_matrix: numpy array of features.  Must have >= 2 dimensions,
        where the first dimension (length E) represents examples and the last
        dimension represents channels (transformed input variables).
    :param target_values: length-E numpy array of target values.  Must all be
        integers in 0...(K - 1), where K = number of classes.
    :param num_classes: Number of classes.
    :param append_to_file: Boolean flag.  If True, will append to existing file.
        If False, will create new file.
    """

    error_checking.assert_is_boolean(append_to_file)
    error_checking.assert_is_numpy_array(feature_matrix)
    num_storm_objects = feature_matrix.shape[0]

    dl_utils.check_target_array(
        target_array=target_values, num_dimensions=1, num_classes=num_classes)
    error_checking.assert_is_numpy_array(
        target_values, exact_dimensions=numpy.array([num_storm_objects]))

    if append_to_file:
        error_checking.assert_is_string(netcdf_file_name)
        netcdf_dataset = netCDF4.Dataset(
            netcdf_file_name, 'a', format='NETCDF3_64BIT_OFFSET')

        prev_num_storm_objects = len(numpy.array(
            netcdf_dataset.variables[TARGET_VALUES_KEY][:]))
        netcdf_dataset.variables[FEATURE_MATRIX_KEY][
            prev_num_storm_objects:(prev_num_storm_objects + num_storm_objects),
            ...
        ] = feature_matrix
        netcdf_dataset.variables[TARGET_VALUES_KEY][
            prev_num_storm_objects:(prev_num_storm_objects + num_storm_objects)
        ] = target_values

    else:
        file_system_utils.mkdir_recursive_if_necessary(
            file_name=netcdf_file_name)
        netcdf_dataset = netCDF4.Dataset(
            netcdf_file_name, 'w', format='NETCDF3_64BIT_OFFSET')

        netcdf_dataset.setncattr(NUM_CLASSES_KEY, num_classes)
        netcdf_dataset.createDimension(STORM_OBJECT_DIMENSION_KEY, None)
        netcdf_dataset.createDimension(
            FEATURE_DIMENSION_KEY, feature_matrix.shape[1])

        num_spatial_dimensions = len(feature_matrix.shape) - 2
        tuple_of_dimension_keys = (STORM_OBJECT_DIMENSION_KEY,)

        for i in range(num_spatial_dimensions):
            netcdf_dataset.createDimension(
                SPATIAL_DIMENSION_KEYS[i], feature_matrix.shape[i + 1])
            tuple_of_dimension_keys += (SPATIAL_DIMENSION_KEYS[i],)

        tuple_of_dimension_keys += (FEATURE_DIMENSION_KEY,)
        netcdf_dataset.createVariable(
            FEATURE_MATRIX_KEY, datatype=numpy.float32,
            dimensions=tuple_of_dimension_keys)
        netcdf_dataset.variables[FEATURE_MATRIX_KEY][:] = feature_matrix

        netcdf_dataset.createVariable(
            TARGET_VALUES_KEY, datatype=numpy.int32,
            dimensions=STORM_OBJECT_DIMENSION_KEY)
        netcdf_dataset.variables[TARGET_VALUES_KEY][:] = target_values

    netcdf_dataset.close()


def read_features(netcdf_file_name):
    """Reads features (activations of intermediate layer) from NetCDF file.

    :param netcdf_file_name: Path to input file.
    :return: feature_matrix: E-by-Z numpy array of features.
    :return: target_values: length-E numpy array of target values.  All in
        0...(K - 1), where K = number of classes.
    :return: num_classes: Number of classes.
    """

    netcdf_dataset = netcdf_io.open_netcdf(
        netcdf_file_name=netcdf_file_name, raise_error_if_fails=True)

    feature_matrix = numpy.array(
        netcdf_dataset.variables[FEATURE_MATRIX_KEY][:])
    target_values = numpy.array(
        netcdf_dataset.variables[TARGET_VALUES_KEY][:], dtype=int)
    num_classes = getattr(netcdf_dataset, NUM_CLASSES_KEY)
    netcdf_dataset.close()

    return feature_matrix, target_values, num_classes
