"""Helper methods for building CNN architecture."""

import numpy
import keras
from gewittergefahr.gg_utils import error_checking

KERNEL_INITIALIZER_NAME = 'glorot_uniform'
BIAS_INITIALIZER_NAME = 'zeros'

MAX_POOLING_STRING = 'max'
MEAN_POOLING_STRING = 'mean'
VALID_POOLING_TYPE_STRINGS = [MAX_POOLING_STRING, MEAN_POOLING_STRING]

YES_PADDING_STRING = 'same'
NO_PADDING_STRING = 'valid'
VALID_PADDING_TYPE_STRINGS = [YES_PADDING_STRING, NO_PADDING_STRING]

DEFAULT_L1_WEIGHT = 0.
DEFAULT_L2_WEIGHT = 0.001

DEFAULT_ALPHA_FOR_ELU = 1.
DEFAULT_ALPHA_FOR_RELU = 0.2

ELU_FUNCTION_STRING = 'elu'
RELU_FUNCTION_STRING = 'relu'
SELU_FUNCTION_STRING = 'selu'
SOFTMAX_FUNCTION_STRING = 'softmax'
SIGMOID_FUNCTION_STRING = 'sigmoid'

VALID_CONV_ACTIV_FUNC_STRINGS = [
    ELU_FUNCTION_STRING, RELU_FUNCTION_STRING, SELU_FUNCTION_STRING
]
VALID_ACTIVATION_FUNCTION_STRINGS = VALID_CONV_ACTIV_FUNC_STRINGS + [
    SOFTMAX_FUNCTION_STRING, SIGMOID_FUNCTION_STRING
]


def _check_convolution_options(
        num_kernel_rows, num_rows_per_stride, padding_type_string,
        num_filters, num_kernel_dimensions, num_kernel_columns=None,
        num_columns_per_stride=None, num_kernel_heights=None,
        num_heights_per_stride=None):
    """Checks input args for 1-D, 2-D, or 3-D convolution layer.

    :param num_kernel_rows: Number of rows in kernel.
    :param num_rows_per_stride: Number of rows per stride (number of rows moved
        by the kernel at once).
    :param padding_type_string: Padding type (must be in
        `VALID_PADDING_TYPE_STRINGS`).
    :param num_filters: Number of output filters (channels).
    :param num_kernel_dimensions: Number of dimensions in kernel.
    :param num_kernel_columns: [used only if num_kernel_dimensions > 1]
        Number of columns in kernel.
    :param num_columns_per_stride: [used only if num_kernel_dimensions > 1]
        Number of columns per stride.
    :param num_kernel_heights: [used only if num_kernel_dimensions = 3]
        Number of heights in kernel.
    :param num_heights_per_stride: [used only if num_kernel_dimensions = 3]
        Number of heights per stride.
    :raises: ValueError: if
        `padding_type_string not in VALID_PADDING_TYPE_STRINGS`.
    """

    error_checking.assert_is_integer(num_kernel_rows)
    error_checking.assert_is_geq(num_kernel_rows, 3)
    error_checking.assert_is_integer(num_rows_per_stride)
    error_checking.assert_is_geq(num_rows_per_stride, 1)
    error_checking.assert_is_leq(num_rows_per_stride, num_kernel_rows)

    error_checking.assert_is_string(padding_type_string)
    if padding_type_string not in VALID_PADDING_TYPE_STRINGS:
        error_string = (
            '\n{0:s}\nValid padding types (listed above) do not include '
            '"{1:s}".'
        ).format(str(VALID_PADDING_TYPE_STRINGS), padding_type_string)

        raise ValueError(error_string)

    error_checking.assert_is_integer(num_filters)
    error_checking.assert_is_geq(num_filters, 1)
    error_checking.assert_is_integer(num_kernel_dimensions)
    error_checking.assert_is_geq(num_kernel_dimensions, 1)
    error_checking.assert_is_leq(num_kernel_dimensions, 3)

    if num_kernel_dimensions >= 2:
        error_checking.assert_is_integer(num_kernel_columns)
        error_checking.assert_is_geq(num_kernel_columns, 3)
        error_checking.assert_is_integer(num_columns_per_stride)
        error_checking.assert_is_geq(num_columns_per_stride, 1)
        error_checking.assert_is_leq(num_columns_per_stride, num_kernel_columns)

    if num_kernel_dimensions == 3:
        error_checking.assert_is_integer(num_kernel_heights)
        error_checking.assert_is_geq(num_kernel_heights, 3)
        error_checking.assert_is_integer(num_heights_per_stride)
        error_checking.assert_is_geq(num_heights_per_stride, 1)
        error_checking.assert_is_leq(num_heights_per_stride, num_kernel_heights)


def _check_pooling_options(
        num_rows_in_window, num_rows_per_stride, pooling_type_string,
        num_dimensions, num_columns_in_window=None, num_columns_per_stride=None,
        num_heights_in_window=None, num_heights_per_stride=None):
    """Checks input args for 1-D, 2-D, or 3-D pooling layer.

    :param num_rows_in_window: Number of rows in pooling window.
    :param num_rows_per_stride: Number of rows per stride (number of rows moved
        by the window at once).
    :param pooling_type_string: Pooling type (must be in
        `VALID_POOLING_TYPE_STRINGS`).
    :param num_dimensions: Number of dimensions in pooling window.
    :param num_columns_in_window: [used only if num_dimensions > 1]
        Number of columns in window.
    :param num_columns_per_stride: [used only if num_dimensions > 1]
        Number of columns per stride.
    :param num_heights_in_window: [used only if num_dimensions = 3]
        Number of heights in window.
    :param num_heights_per_stride: [used only if num_dimensions = 3]
        Number of heights per stride.
    :raises: ValueError: if
        `pooling_type_string not in VALID_POOLING_TYPE_STRINGS`.
    """

    error_checking.assert_is_integer(num_rows_in_window)
    error_checking.assert_is_geq(num_rows_in_window, 2)
    error_checking.assert_is_integer(num_rows_per_stride)
    error_checking.assert_is_geq(num_rows_per_stride, 1)
    error_checking.assert_is_leq(num_rows_per_stride, num_rows_in_window)

    error_checking.assert_is_string(pooling_type_string)
    if pooling_type_string not in VALID_POOLING_TYPE_STRINGS:
        error_string = (
            '\n{0:s}\nValid pooling types (listed above) do not include '
            '"{1:s}".'
        ).format(str(VALID_POOLING_TYPE_STRINGS), pooling_type_string)

        raise ValueError(error_string)

    error_checking.assert_is_integer(num_dimensions)
    error_checking.assert_is_geq(num_dimensions, 1)
    error_checking.assert_is_leq(num_dimensions, 3)

    if num_dimensions >= 2:
        error_checking.assert_is_integer(num_columns_in_window)
        error_checking.assert_is_geq(num_columns_in_window, 2)
        error_checking.assert_is_integer(num_columns_per_stride)
        error_checking.assert_is_geq(num_columns_per_stride, 1)
        error_checking.assert_is_leq(num_columns_per_stride,
                                     num_columns_in_window)

    if num_dimensions == 3:
        error_checking.assert_is_integer(num_heights_in_window)
        error_checking.assert_is_geq(num_heights_in_window, 2)
        error_checking.assert_is_integer(num_heights_per_stride)
        error_checking.assert_is_geq(num_heights_per_stride, 1)
        error_checking.assert_is_leq(num_heights_per_stride,
                                     num_heights_in_window)


def check_activation_function(
        activation_function_string, alpha_for_elu=DEFAULT_ALPHA_FOR_ELU,
        alpha_for_relu=DEFAULT_ALPHA_FOR_RELU, for_conv_layer=False):
    """Error-checks activation function.

    :param activation_function_string: Name of activation function.
    :param alpha_for_elu: [used only if activation function is "elu"]
        Slope for negative inputs to activation function.
    :param alpha_for_relu: [used only if activation function is "relu"]
        Slope for negative inputs to activation function.
    :param for_conv_layer: Boolean flag.  If True, activation function is for a
        convolution layer.
    :raises: ValueError: if
        `activation_function_string not in VALID_ACTIVATION_FUNCTION_STRINGS`.
    """

    error_checking.assert_is_boolean(for_conv_layer)
    error_checking.assert_is_string(activation_function_string)

    if for_conv_layer:
        these_valid_strings = VALID_CONV_ACTIV_FUNC_STRINGS
    else:
        these_valid_strings = VALID_ACTIVATION_FUNCTION_STRINGS

    if activation_function_string not in these_valid_strings:
        error_string = (
            '\n{0:s}\nValid activation functions (listed above) do not '
            'include "{1:s}".'
        ).format(str(these_valid_strings), activation_function_string)

        raise ValueError(error_string)

    if activation_function_string == ELU_FUNCTION_STRING:
        error_checking.assert_is_geq(alpha_for_elu, 0.)

    if activation_function_string == RELU_FUNCTION_STRING:
        error_checking.assert_is_geq(alpha_for_relu, 0.)


def get_dense_layer_dimensions(num_input_units, num_classes, num_dense_layers):
    """Returns number of input and output neurons for each dense layer.

    D = number of dense layers

    :param num_input_units: Number of inputs to first dense layer.
    :param num_classes: Number of classes for target variable.
    :param num_dense_layers: Number of dense layers.
    :return: num_inputs_by_layer: length-D numpy array with number of input
        neurons for each layer.
    :return: num_outputs_by_layer: length-D numpy array with number of output
        neurons for each layer.
    """

    error_checking.assert_is_integer(num_classes)
    error_checking.assert_is_greater(num_classes, 0)
    error_checking.assert_is_integer(num_input_units)
    error_checking.assert_is_greater(num_input_units, num_classes)
    error_checking.assert_is_integer(num_dense_layers)
    error_checking.assert_is_greater(num_dense_layers, 0)

    if num_classes == 2:
        num_output_units = 1
    else:
        num_output_units = num_classes + 0

    e_folding_param = (
        float(-1 * num_dense_layers) /
        numpy.log(float(num_output_units) / num_input_units)
    )

    dense_layer_indices = numpy.linspace(
        0, num_dense_layers - 1, num=num_dense_layers, dtype=float
    )
    num_inputs_by_layer = num_input_units * numpy.exp(
        -1 * dense_layer_indices / e_folding_param
    )
    num_inputs_by_layer = numpy.round(num_inputs_by_layer).astype(int)

    num_outputs_by_layer = numpy.concatenate((
        num_inputs_by_layer[1:],
        numpy.array([num_output_units], dtype=int)
    ))

    return num_inputs_by_layer, num_outputs_by_layer


def get_weight_regularizer(l1_weight=DEFAULT_L1_WEIGHT,
                           l2_weight=DEFAULT_L2_WEIGHT):
    """Creates regularizer for network weights.

    :param l1_weight: L1 regularization weight.  This "weight" is not to be
        confused with those being regularized (weights learned by the net).
    :param l2_weight: L2 regularization weight.
    :return: regularizer_object: Instance of `keras.regularizers.l1_l2`.
    """

    error_checking.assert_is_geq(l1_weight, 0.)
    error_checking.assert_is_geq(l2_weight, 0.)
    return keras.regularizers.l1_l2(l1=l1_weight, l2=l2_weight)


def get_1d_conv_layer(
        num_kernel_rows, num_rows_per_stride, num_filters,
        padding_type_string=NO_PADDING_STRING, weight_regularizer=None):
    """Creates layer for 1-D convolution.

    :param num_kernel_rows: See doc for `_check_convolution_options`.
    :param num_rows_per_stride: Same.
    :param num_filters: Same.
    :param padding_type_string: Same.
    :param weight_regularizer: Will be used to regularize weights in the new
        layer.  This may be instance of `keras.regularizers` or None (if you
        want no regularization).
    :return: layer_object: Instance of `keras.layers.Conv1D`.
    """

    _check_convolution_options(
        num_kernel_rows=num_kernel_rows,
        num_rows_per_stride=num_rows_per_stride,
        padding_type_string=padding_type_string,
        num_filters=num_filters, num_kernel_dimensions=1)

    return keras.layers.Conv1D(
        filters=num_filters, kernel_size=(num_kernel_rows,),
        strides=(num_rows_per_stride,), padding=padding_type_string,
        dilation_rate=(1,), activation=None, use_bias=True,
        kernel_initializer=KERNEL_INITIALIZER_NAME,
        bias_initializer=BIAS_INITIALIZER_NAME,
        kernel_regularizer=weight_regularizer,
        bias_regularizer=weight_regularizer)


def get_1d_separable_conv_layer(
        num_kernel_rows, num_rows_per_stride, num_spatial_filters,
        num_non_spatial_filters, padding_type_string=NO_PADDING_STRING,
        weight_regularizer=None):
    """Creates layer for 1-D convolution.

    Total number of filters = `num_spatial_filters * num_non_spatial_filters`

    :param num_kernel_rows: See doc for `_check_convolution_options`.
    :param num_rows_per_stride: Same.
    :param num_spatial_filters: Number of spatial filters (applied independently
        to each input channel).
    :param num_non_spatial_filters: Number of non-spatial filters (applied
        independently to each spatial position).
    :param padding_type_string: See doc for `_check_convolution_options`.
    :param weight_regularizer: See doc for `get_1d_conv_layer`.
    :return: layer_object: Instance of `keras.layers.SeparableConv1D`.
    """

    error_checking.assert_is_integer(num_spatial_filters)
    error_checking.assert_is_greater(num_non_spatial_filters, 0)
    num_total_filters = num_spatial_filters * num_non_spatial_filters

    _check_convolution_options(
        num_kernel_rows=num_kernel_rows,
        num_rows_per_stride=num_rows_per_stride,
        padding_type_string=padding_type_string,
        num_filters=num_total_filters, num_kernel_dimensions=1)

    return keras.layers.SeparableConv1D(
        filters=num_total_filters, kernel_size=num_kernel_rows,
        strides=num_rows_per_stride, depth_multiplier=num_non_spatial_filters,
        padding=padding_type_string, dilation_rate=(1,),
        activation=None, use_bias=True,
        depthwise_initializer=KERNEL_INITIALIZER_NAME,
        pointwise_initializer=KERNEL_INITIALIZER_NAME,
        bias_initializer=BIAS_INITIALIZER_NAME,
        depthwise_regularizer=weight_regularizer,
        pointwise_regularizer=weight_regularizer,
        bias_regularizer=weight_regularizer)


def get_2d_conv_layer(
        num_kernel_rows, num_kernel_columns, num_rows_per_stride,
        num_columns_per_stride, num_filters,
        padding_type_string=NO_PADDING_STRING, weight_regularizer=None):
    """Creates layer for 2-D convolution.

    :param num_kernel_rows: See doc for `_check_convolution_options`.
    :param num_kernel_columns: Same.
    :param num_rows_per_stride: Same.
    :param num_columns_per_stride: Same.
    :param num_filters: Same.
    :param padding_type_string: Same.
    :param weight_regularizer: See doc for `get_1d_conv_layer`.
    :return: layer_object: Instance of `keras.layers.Conv2D`.
    """

    _check_convolution_options(
        num_kernel_rows=num_kernel_rows, num_kernel_columns=num_kernel_columns,
        num_rows_per_stride=num_rows_per_stride,
        num_columns_per_stride=num_columns_per_stride,
        padding_type_string=padding_type_string,
        num_filters=num_filters, num_kernel_dimensions=2)

    return keras.layers.Conv2D(
        filters=num_filters, kernel_size=(num_kernel_rows, num_kernel_columns),
        strides=(num_rows_per_stride, num_columns_per_stride),
        padding=padding_type_string, dilation_rate=(1, 1),
        activation=None, use_bias=True,
        kernel_initializer=KERNEL_INITIALIZER_NAME,
        bias_initializer=BIAS_INITIALIZER_NAME,
        kernel_regularizer=weight_regularizer,
        bias_regularizer=weight_regularizer)


def get_2d_separable_conv_layer(
        num_kernel_rows, num_kernel_columns, num_rows_per_stride,
        num_columns_per_stride, num_spatial_filters, num_non_spatial_filters,
        padding_type_string=NO_PADDING_STRING, weight_regularizer=None):
    """Creates layer for 2-D convolution.

    Total number of filters = `num_spatial_filters * num_non_spatial_filters`

    :param num_kernel_rows: See doc for `_check_convolution_options`.
    :param num_kernel_columns: Same.
    :param num_rows_per_stride: Same.
    :param num_columns_per_stride: Same.
    :param num_spatial_filters: See doc for `get_1d_separable_conv_layer`.
    :param num_non_spatial_filters: Same.
    :param padding_type_string: See doc for `_check_convolution_options`.
    :param weight_regularizer: See doc for `get_1d_conv_layer`.
    :return: layer_object: Instance of `keras.layers.SeparableConv1D`.
    """

    error_checking.assert_is_integer(num_spatial_filters)
    error_checking.assert_is_greater(num_non_spatial_filters, 0)
    num_total_filters = num_spatial_filters * num_non_spatial_filters

    _check_convolution_options(
        num_kernel_rows=num_kernel_rows, num_kernel_columns=num_kernel_columns,
        num_rows_per_stride=num_rows_per_stride,
        num_columns_per_stride=num_columns_per_stride,
        padding_type_string=padding_type_string,
        num_filters=num_total_filters, num_kernel_dimensions=2)

    return keras.layers.SeparableConv2D(
        filters=num_total_filters,
        kernel_size=(num_kernel_rows, num_kernel_columns),
        strides=(num_rows_per_stride, num_columns_per_stride),
        depth_multiplier=num_non_spatial_filters,
        padding=padding_type_string, dilation_rate=(1, 1),
        activation=None, use_bias=True,
        depthwise_initializer=KERNEL_INITIALIZER_NAME,
        pointwise_initializer=KERNEL_INITIALIZER_NAME,
        bias_initializer=BIAS_INITIALIZER_NAME,
        depthwise_regularizer=weight_regularizer,
        pointwise_regularizer=weight_regularizer,
        bias_regularizer=weight_regularizer)


def get_3d_conv_layer(
        num_kernel_rows, num_kernel_columns, num_kernel_heights,
        num_rows_per_stride, num_columns_per_stride, num_heights_per_stride,
        num_filters, padding_type_string=NO_PADDING_STRING,
        weight_regularizer=None):
    """Creates layer for 2-D convolution.

    :param num_kernel_rows: See doc for `_check_convolution_options`.
    :param num_kernel_columns: Same.
    :param num_kernel_heights: Same.
    :param num_rows_per_stride: Same.
    :param num_columns_per_stride: Same.
    :param num_heights_per_stride: Same.
    :param num_filters: Same.
    :param padding_type_string: Same.
    :param weight_regularizer: See doc for `get_1d_conv_layer`.
    :return: layer_object: Instance of `keras.layers.Conv2D`.
    """

    _check_convolution_options(
        num_kernel_rows=num_kernel_rows, num_kernel_columns=num_kernel_columns,
        num_kernel_heights=num_kernel_heights,
        num_rows_per_stride=num_rows_per_stride,
        num_columns_per_stride=num_columns_per_stride,
        num_heights_per_stride=num_heights_per_stride,
        padding_type_string=padding_type_string,
        num_filters=num_filters, num_kernel_dimensions=2)

    return keras.layers.Conv3D(
        filters=num_filters,
        kernel_size=(num_kernel_rows, num_kernel_columns, num_kernel_heights),
        strides=(num_rows_per_stride, num_columns_per_stride,
                 num_heights_per_stride),
        padding=padding_type_string, dilation_rate=(1, 1, 1),
        activation=None, use_bias=True,
        kernel_initializer=KERNEL_INITIALIZER_NAME,
        bias_initializer=BIAS_INITIALIZER_NAME,
        kernel_regularizer=weight_regularizer,
        bias_regularizer=weight_regularizer)


def get_1d_pooling_layer(num_rows_in_window, num_rows_per_stride,
                         pooling_type_string=MAX_POOLING_STRING):
    """Creates layer for 1-D pooling.

    :param num_rows_in_window: See doc for `_check_pooling_options`.
    :param num_rows_per_stride: Same.
    :param pooling_type_string: Same.
    :return: layer_object: Instance of `keras.layers.MaxPooling1D` or
        `keras.layers.AveragePooling1D`.
    """

    _check_pooling_options(
        num_rows_in_window=num_rows_in_window,
        num_rows_per_stride=num_rows_per_stride,
        pooling_type_string=pooling_type_string, num_dimensions=1)

    if pooling_type_string == MAX_POOLING_STRING:
        return keras.layers.MaxPooling1D(
            pool_size=num_rows_in_window, strides=num_rows_per_stride,
            padding=NO_PADDING_STRING)

    return keras.layers.AveragePooling1D(
        pool_size=num_rows_in_window, strides=num_rows_per_stride,
        padding=NO_PADDING_STRING)


def get_2d_pooling_layer(
        num_rows_in_window, num_columns_in_window, num_rows_per_stride,
        num_columns_per_stride, pooling_type_string=MAX_POOLING_STRING):
    """Creates layer for 2-D pooling.

    :param num_rows_in_window: See doc for `_check_pooling_options`.
    :param num_columns_in_window: Same.
    :param num_rows_per_stride: Same.
    :param num_columns_per_stride: Same.
    :param pooling_type_string: Same.
    :return: layer_object: Instance of `keras.layers.MaxPooling2D` or
        `keras.layers.AveragePooling2D`.
    """

    _check_pooling_options(
        num_rows_in_window=num_rows_in_window,
        num_columns_in_window=num_columns_in_window,
        num_rows_per_stride=num_rows_per_stride,
        num_columns_per_stride=num_columns_per_stride,
        pooling_type_string=pooling_type_string, num_dimensions=2)

    if pooling_type_string == MAX_POOLING_STRING:
        return keras.layers.MaxPooling2D(
            pool_size=(num_rows_in_window, num_columns_in_window),
            strides=(num_rows_per_stride, num_columns_per_stride),
            padding=NO_PADDING_STRING)

    return keras.layers.AveragePooling2D(
        pool_size=(num_rows_in_window, num_columns_in_window),
        strides=(num_rows_per_stride, num_columns_per_stride),
        padding=NO_PADDING_STRING)


def get_3d_pooling_layer(
        num_rows_in_window, num_columns_in_window, num_heights_in_window,
        num_rows_per_stride, num_columns_per_stride, num_heights_per_stride,
        pooling_type_string=MAX_POOLING_STRING):
    """Creates layer for 2-D pooling.

    :param num_rows_in_window: See doc for `_check_pooling_options`.
    :param num_columns_in_window: Same.
    :param num_heights_in_window: Same.
    :param num_rows_per_stride: Same.
    :param num_columns_per_stride: Same.
    :param num_heights_per_stride: Same.
    :param pooling_type_string: Same.
    :return: layer_object: Instance of `keras.layers.MaxPooling2D` or
        `keras.layers.AveragePooling2D`.
    """

    _check_pooling_options(
        num_rows_in_window=num_rows_in_window,
        num_columns_in_window=num_columns_in_window,
        num_heights_in_window=num_heights_in_window,
        num_rows_per_stride=num_rows_per_stride,
        num_columns_per_stride=num_columns_per_stride,
        num_heights_per_stride=num_heights_per_stride,
        pooling_type_string=pooling_type_string, num_dimensions=3)

    if pooling_type_string == MAX_POOLING_STRING:
        return keras.layers.MaxPooling3D(
            pool_size=(num_rows_in_window, num_columns_in_window,
                       num_heights_in_window),
            strides=(num_rows_per_stride, num_columns_per_stride,
                     num_heights_per_stride),
            padding=NO_PADDING_STRING)

    return keras.layers.AveragePooling3D(
        pool_size=(num_rows_in_window, num_columns_in_window,
                   num_heights_in_window),
        strides=(num_rows_per_stride, num_columns_per_stride,
                 num_heights_per_stride),
        padding=NO_PADDING_STRING)


def get_dense_layer(num_output_units, weight_regularizer=None):
    """Creates dense (fully connected) layer.

    :param num_output_units: Number of output units (or "features" or
        "neurons").
    :param weight_regularizer: See doc for `get_1d_conv_layer`.
    :return: layer_object: Instance of `keras.layers.Dense`.
    """

    error_checking.assert_is_integer(num_output_units)
    error_checking.assert_is_greater(num_output_units, 0)

    return keras.layers.Dense(
        num_output_units, activation=None, use_bias=True,
        kernel_initializer=KERNEL_INITIALIZER_NAME,
        bias_initializer=BIAS_INITIALIZER_NAME,
        kernel_regularizer=weight_regularizer,
        bias_regularizer=weight_regularizer)


def get_activation_layer(
        activation_function_string, alpha_for_elu=DEFAULT_ALPHA_FOR_ELU,
        alpha_for_relu=DEFAULT_ALPHA_FOR_RELU):
    """Creates activation layer.

    :param activation_function_string: See doc for `check_activation_function`.
    :param alpha_for_elu: Same.
    :param alpha_for_relu: Same.
    :return: layer_object: Instance of `keras.layers.Activation`,
        `keras.layers.ELU`, or `keras.layers.LeakyReLU`.
    """

    check_activation_function(
        activation_function_string=activation_function_string,
        alpha_for_elu=alpha_for_elu, alpha_for_relu=alpha_for_relu)

    if activation_function_string == ELU_FUNCTION_STRING:
        return keras.layers.ELU(alpha=alpha_for_elu)

    if activation_function_string == RELU_FUNCTION_STRING:
        return keras.layers.LeakyReLU(alpha=alpha_for_relu)

    return keras.layers.Activation(activation_function_string)


def get_dropout_layer(dropout_fraction):
    """Creates dropout layer.

    :param dropout_fraction: Fraction of weights to drop.
    :return: layer_object: Instance of `keras.layers.Dropout`.
    """

    error_checking.assert_is_greater(dropout_fraction, 0.)
    error_checking.assert_is_less_than(dropout_fraction, 1.)

    return keras.layers.Dropout(rate=dropout_fraction)


def get_batch_norm_layer():
    """Creates batch-normalization layer.

    :return: Instance of `keras.layers.BatchNormalization`.
    """

    return keras.layers.BatchNormalization(
        axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True)


def get_flattening_layer():
    """Creates flattening layer.

    :return: layer_object: Instance of `keras.layers.Flatten`.
    """

    return keras.layers.Flatten()
