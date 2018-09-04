"""Helper methods for CNN (convolutional neural network) architecture.

--- NOTATION ---

In this module, the following letters will be used to denote matrix dimensions.

K = number of classes (possible values of target variable)
E = number of examples
M = number of pixel rows per image
N = number of pixel columns per image
D = number of pixel depths per image
C = number of channels (predictor variables) per image

--- REFERENCES ---

Ioffe, S., and C. Szegedy (2015): "Batch normalization: Accelerating deep
    network training by reducing internal covariate shift". International
    Conference on Machine Learning, 448-456.

Lagerquist, R. (2016): "Using machine learning to predict damaging straight-line
    convective winds". Master's thesis, School of Meteorology, University of
    Oklahoma.
"""

import keras
import keras.initializers
import keras.optimizers
import keras.layers
from gewittergefahr.gg_utils import error_checking

MAX_POOLING_TYPE = 'max'
MEAN_POOLING_TYPE = 'mean'
VALID_POOLING_TYPES = [MAX_POOLING_TYPE, MEAN_POOLING_TYPE]

YES_PADDING_TYPE = 'same'
NO_PADDING_TYPE = 'valid'
VALID_PADDING_TYPES = [YES_PADDING_TYPE, NO_PADDING_TYPE]

DEFAULT_MIN_INIT_WEIGHT = -0.05
DEFAULT_MAX_INIT_WEIGHT = 0.05
DEFAULT_DROPOUT_FRACTION = 0.5

DEFAULT_L1_PENALTY = 0.
DEFAULT_L2_PENALTY = 0.01

DEFAULT_ALPHA_FOR_ELU = 1.
DEFAULT_ALPHA_FOR_RELU = 0.
ELU_FUNCTION_STRING = 'elu'
RELU_FUNCTION_STRING = 'relu'
SELU_FUNCTION_STRING = 'selu'
SOFTMAX_FUNCTION_STRING = 'softmax'
SIGMOID_FUNCTION_STRING = 'sigmoid'

VALID_CONV_LAYER_ACTIV_FUNC_STRINGS = [
    ELU_FUNCTION_STRING, RELU_FUNCTION_STRING, SELU_FUNCTION_STRING
]
VALID_ACTIVATION_FUNCTION_STRINGS = VALID_CONV_LAYER_ACTIV_FUNC_STRINGS + [
    SOFTMAX_FUNCTION_STRING, SIGMOID_FUNCTION_STRING
]


def _check_input_args_for_conv_layer(
        num_output_filters, num_kernel_rows, num_rows_per_stride, padding_type,
        is_first_layer, num_kernel_columns=None, num_kernel_depths=None,
        num_columns_per_stride=None, num_depths_per_stride=None,
        num_input_rows=None, num_input_channels=None, num_input_columns=None,
        num_input_depths=None):
    """Checks input args for 1-D, 2-D, or 3-D-convolution layer.

    :param num_output_filters: See documentation for `get_3d_conv_layer`.
    :param num_kernel_rows: Same.
    :param num_rows_per_stride: Same.
    :param padding_type: Same.
    :param is_first_layer: Same.
    :param num_kernel_columns: [used only for 2-D or 3-D convolution] Same.
    :param num_kernel_depths: [used only for 3-D conv] Same.
    :param num_columns_per_stride: [used only for 2-D or 3-D conv] Same.
    :param num_depths_per_stride: [used only for 3-D conv] Same.
    :param num_input_rows: Same.
    :param num_input_channels: Same.
    :param num_input_columns:
        [used only for 2-D or 3-D conv when is_first_layer = True]
        Same.
    :param num_input_depths:
        [used only for 3-D conv when is_first_layer = True]
        Same.
    :raises: ValueError: if `padding_type not in VALID_PADDING_TYPES`.
    """

    error_checking.assert_is_integer(num_output_filters)
    error_checking.assert_is_geq(num_output_filters, 2)
    error_checking.assert_is_integer(num_kernel_rows)
    error_checking.assert_is_geq(num_kernel_rows, 1)
    error_checking.assert_is_integer(num_rows_per_stride)
    error_checking.assert_is_greater(num_rows_per_stride, 0)
    error_checking.assert_is_leq(num_rows_per_stride, num_kernel_rows)

    error_checking.assert_is_string(padding_type)
    if padding_type not in VALID_PADDING_TYPES:
        error_string = (
            '\n\n{0:s}\nValid padding types (listed above) do not include '
            '"{1:s}".').format(VALID_PADDING_TYPES, padding_type)
        raise ValueError(error_string)

    error_checking.assert_is_boolean(is_first_layer)
    if num_kernel_columns is None:
        num_dimensions = 1
    elif num_kernel_depths is None:
        num_dimensions = 2
    else:
        num_dimensions = 3

    if num_dimensions >= 2:
        error_checking.assert_is_integer(num_kernel_columns)
        error_checking.assert_is_geq(num_kernel_columns, 1)
        error_checking.assert_is_integer(num_columns_per_stride)
        error_checking.assert_is_greater(num_columns_per_stride, 0)
        error_checking.assert_is_leq(
            num_columns_per_stride, num_kernel_columns)

    if num_dimensions == 3:
        error_checking.assert_is_integer(num_kernel_depths)
        error_checking.assert_is_geq(num_kernel_depths, 1)
        error_checking.assert_is_integer(num_depths_per_stride)
        error_checking.assert_is_greater(num_depths_per_stride, 0)
        error_checking.assert_is_leq(
            num_depths_per_stride, num_kernel_depths)

    if is_first_layer:
        error_checking.assert_is_integer(num_input_rows)
        error_checking.assert_is_geq(num_input_rows, 3)
        error_checking.assert_is_integer(num_input_channels)
        error_checking.assert_is_geq(num_input_channels, 1)

        if num_dimensions >= 2:
            error_checking.assert_is_integer(num_input_columns)
            error_checking.assert_is_geq(num_input_columns, 3)

        if num_dimensions == 3:
            error_checking.assert_is_integer(num_input_depths)
            error_checking.assert_is_geq(num_input_depths, 3)


def _check_input_args_for_pooling_layer(
        num_rows_in_window, pooling_type, num_rows_per_stride,
        num_columns_in_window=None, num_depths_in_window=None,
        num_columns_per_stride=None, num_depths_per_stride=None):
    """Checks input arguments for 2-D-pooling or 3-D-pooling layer.

    :param num_rows_in_window: See documentation for `get_3d_pooling_layer`.
    :param pooling_type: Same.
    :param num_rows_per_stride: Same.
    :param num_columns_in_window: [used only for 2-D or 3-D pooling] Same.
    :param num_depths_in_window: [used only for 3-D pooling] Same.
    :param num_columns_per_stride: [used only for 2-D or 3-D pooling] Same.
    :param num_depths_per_stride: [used only for 3-D pooling] Same.
    :raises: ValueError: if `pooling_type not in VALID_POOLING_TYPES`.
    """

    error_checking.assert_is_integer(num_rows_in_window)
    error_checking.assert_is_geq(num_rows_in_window, 1)
    error_checking.assert_is_string(pooling_type)

    if pooling_type not in VALID_POOLING_TYPES:
        error_string = (
            '\n\n{0:s}\nValid pooling types (listed above) do not include '
            '"{1:s}".').format(VALID_POOLING_TYPES, pooling_type)
        raise ValueError(error_string)

    error_checking.assert_is_integer(num_rows_per_stride)
    error_checking.assert_is_greater(num_rows_per_stride, 0)
    error_checking.assert_is_leq(num_rows_per_stride, num_rows_in_window)

    if num_columns_in_window is not None:
        error_checking.assert_is_integer(num_columns_in_window)
        error_checking.assert_is_geq(num_columns_in_window, 1)
        error_checking.assert_is_integer(num_columns_per_stride)
        error_checking.assert_is_greater(num_columns_per_stride, 0)
        error_checking.assert_is_leq(
            num_columns_per_stride, num_columns_in_window)

    if num_depths_in_window is not None:
        error_checking.assert_is_integer(num_depths_in_window)
        error_checking.assert_is_geq(num_depths_in_window, 1)
        error_checking.assert_is_integer(num_depths_per_stride)
        error_checking.assert_is_greater(num_depths_per_stride, 0)
        error_checking.assert_is_leq(
            num_depths_per_stride, num_depths_in_window)


def check_activation_function(
        activation_function_string, alpha_for_elu=None, alpha_for_relu=None,
        for_conv_layer=False):
    """Ensures that activation function is valid.

    :param activation_function_string: Activation function.
    :param alpha_for_elu: [used only if activation_function_string = "elu"]
        Slope for negative inputs to activation function.
    :param alpha_for_relu: [used only if activation_function_string = "relu"]
        Slope for negative inputs to activation function.
    :param for_conv_layer: Boolean flag.  If True, this method ensures that the
        activation function is valid for a convolutional layers.  If False,
        ensures validity for any layer.
    :raises: ValueError: if `activation_function_string not in
        VALID_ACTIVATION_FUNCTION_STRINGS`.
    """

    error_checking.assert_is_boolean(for_conv_layer)
    error_checking.assert_is_string(activation_function_string)
    if for_conv_layer:
        these_valid_strings = VALID_CONV_LAYER_ACTIV_FUNC_STRINGS
    else:
        these_valid_strings = VALID_ACTIVATION_FUNCTION_STRINGS

    if activation_function_string not in these_valid_strings:
        error_string = (
            '\n\n{0:s}\nValid activation functions (listed above) do not '
            'include "{1:s}".'
        ).format(str(these_valid_strings), activation_function_string)
        raise ValueError(error_string)

    if activation_function_string == ELU_FUNCTION_STRING:
        error_checking.assert_is_geq(alpha_for_elu, 0.)
    if activation_function_string == RELU_FUNCTION_STRING:
        error_checking.assert_is_geq(alpha_for_relu, 0.)


def get_weight_regularizer(
        l1_penalty=DEFAULT_L1_PENALTY, l2_penalty=DEFAULT_L2_PENALTY):
    """Creates regularizer for network weights.

    :param l1_penalty: L1 regularization penalty (see Equation 4.3 of Lagerquist
        2016).
    :param l2_penalty: L2 regularization penalty (see Equation 4.3 of Lagerquist
        2016).
    :return: regularizer_object: Instance of `keras.regularizers.l1_l2`.
    """

    error_checking.assert_is_geq(l1_penalty, 0.)
    error_checking.assert_is_geq(l2_penalty, 0.)
    return keras.regularizers.l1_l2(l1=l1_penalty, l2=l2_penalty)


def get_random_uniform_initializer(
        min_value=DEFAULT_MIN_INIT_WEIGHT, max_value=DEFAULT_MAX_INIT_WEIGHT):
    """Creates weight-initializer with uniform random distribution.

    :param min_value: Bottom of uniform distribution.
    :param max_value: Top of uniform distribution.
    :return: initializer_object: Instance of `keras.initializers.Initializer`.
    """

    error_checking.assert_is_geq(min_value, -1.)
    error_checking.assert_is_leq(max_value, 1.)
    error_checking.assert_is_greater(max_value, min_value)

    return keras.initializers.RandomUniform(minval=min_value, maxval=max_value)


def get_1d_conv_layer(
        num_output_filters, num_kernel_pixels, num_pixels_per_stride,
        padding_type=YES_PADDING_TYPE, kernel_weight_regularizer=None,
        bias_weight_regularizer=None, is_first_layer=False,
        num_input_pixels=None, num_input_channels=None):
    """Creates 1-D-convolution layer.

    In other words, this layer will convolve over 1-D feature maps.

    To add this layer to a Keras model, use the following code.

    model_object = keras.models.Sequential()
    ... # May add other layers here.
    layer_object = get_1d_conv_layer(...)
    model_object.add(layer_object)

    :param num_output_filters: See documentation for `get_3d_conv_layer`.
    :param num_kernel_pixels: Same.
    :param num_pixels_per_stride: Same.
    :param padding_type: Same.
    :param kernel_weight_regularizer: Same.
    :param bias_weight_regularizer: Same.
    :param is_first_layer: Same.
    :param num_input_pixels: Same.
    :param num_input_channels: Same.
    :return: layer_object: Instance of `keras.layers.Conv1D`.
    """

    _check_input_args_for_conv_layer(
        num_output_filters=num_output_filters,
        num_kernel_rows=num_kernel_pixels,
        num_rows_per_stride=num_pixels_per_stride, padding_type=padding_type,
        is_first_layer=is_first_layer, num_input_rows=num_input_pixels,
        num_input_channels=num_input_channels)

    if is_first_layer:
        return keras.layers.Conv1D(
            filters=num_output_filters, kernel_size=num_kernel_pixels,
            strides=num_pixels_per_stride, padding=padding_type,
            dilation_rate=(1,), activation=None, use_bias=True,
            kernel_initializer='glorot_uniform', bias_initializer='zeros',
            kernel_regularizer=kernel_weight_regularizer,
            bias_regularizer=bias_weight_regularizer,
            input_shape=(num_input_pixels, num_input_channels))

    return keras.layers.Conv1D(
        filters=num_output_filters, kernel_size=(num_kernel_pixels,),
        strides=(num_pixels_per_stride,), padding=padding_type,
        dilation_rate=(1,), activation=None, use_bias=True,
        kernel_initializer='glorot_uniform', bias_initializer='zeros',
        kernel_regularizer=kernel_weight_regularizer,
        bias_regularizer=bias_weight_regularizer)


def get_2d_conv_layer(
        num_output_filters, num_kernel_rows, num_kernel_columns,
        num_rows_per_stride, num_columns_per_stride,
        padding_type=YES_PADDING_TYPE, kernel_weight_regularizer=None,
        bias_weight_regularizer=None, is_first_layer=False, num_input_rows=None,
        num_input_columns=None, num_input_channels=None):
    """Creates 2-D-convolution layer.

    In other words, this layer will convolve over 2-D feature maps.

    To add this layer to a Keras model, use the following code.

    model_object = keras.models.Sequential()
    ... # May add other layers here.
    layer_object = get_2d_conv_layer(...)
    model_object.add(layer_object)

    :param num_output_filters: See documentation for `get_3d_conv_layer`.
    :param num_kernel_rows: Same.
    :param num_kernel_columns: Same.
    :param num_rows_per_stride: Same.
    :param num_columns_per_stride: Same.
    :param padding_type: Same.
    :param kernel_weight_regularizer: Same.
    :param bias_weight_regularizer: Same.
    :param is_first_layer: Same.
    :param num_input_rows: Same.
    :param num_input_columns: Same.
    :param num_input_channels: Same.
    :return: layer_object: Instance of `keras.layers.Conv2D`.
    """

    _check_input_args_for_conv_layer(
        num_output_filters=num_output_filters, num_kernel_rows=num_kernel_rows,
        num_kernel_columns=num_kernel_columns,
        num_rows_per_stride=num_rows_per_stride,
        num_columns_per_stride=num_columns_per_stride,
        padding_type=padding_type, is_first_layer=is_first_layer,
        num_input_rows=num_input_rows, num_input_columns=num_input_columns,
        num_input_channels=num_input_channels)

    if is_first_layer:
        return keras.layers.Conv2D(
            filters=num_output_filters,
            kernel_size=(num_kernel_rows, num_kernel_columns),
            strides=(num_rows_per_stride, num_columns_per_stride),
            padding=padding_type, data_format='channels_last',
            dilation_rate=(1, 1), activation=None, use_bias=True,
            kernel_initializer='glorot_uniform', bias_initializer='zeros',
            kernel_regularizer=kernel_weight_regularizer,
            bias_regularizer=bias_weight_regularizer,
            input_shape=(num_input_rows, num_input_columns, num_input_channels))

    return keras.layers.Conv2D(
        filters=num_output_filters,
        kernel_size=(num_kernel_rows, num_kernel_columns),
        strides=(num_rows_per_stride, num_columns_per_stride),
        padding=padding_type, data_format='channels_last', dilation_rate=(1, 1),
        activation=None, use_bias=True, kernel_initializer='glorot_uniform',
        bias_initializer='zeros', kernel_regularizer=kernel_weight_regularizer,
        bias_regularizer=bias_weight_regularizer)


def get_3d_conv_layer(
        num_output_filters, num_kernel_rows, num_kernel_columns,
        num_kernel_depths, num_rows_per_stride, num_columns_per_stride,
        num_depths_per_stride, padding_type=YES_PADDING_TYPE,
        kernel_weight_regularizer=None, bias_weight_regularizer=None,
        is_first_layer=False, num_input_rows=None, num_input_columns=None,
        num_input_depths=None, num_input_channels=None):
    """Creates 3-D-convolution layer.

    In other words, this layer will convolve over 3-D feature maps.

    To add this layer to a Keras model, use the following code.

    model_object = keras.models.Sequential()
    ... # May add other layers here.
    layer_object = get_3d_conv_layer(...)
    model_object.add(layer_object)

    :param num_output_filters: Number of output filters.
    :param num_kernel_rows: Number of rows in kernel.
    :param num_kernel_columns: Number of columns in kernel.
    :param num_kernel_depths: Number of depths in kernel.
    :param num_rows_per_stride: Stride length in y-direction.
    :param num_columns_per_stride: Stride length in x-direction.
    :param num_depths_per_stride: Stride length in z-direction.
    :param padding_type: Either "same" or "valid".  If "same", each output
        feature map will have the same dimensions (M x N x D) as each input
        feature map.  If "valid", each dimension will be reduced by l - 1, where
        l is the length of the kernel in said dimension.  For example, if the
        kernel is m x n x d, the size of each output feature map will be
        (M - m + 1) x (N - n + 1) x (D - d + 1).  To be even more concrete, if
        the input feature maps are 32 x 32 x 8 and the kernel is 5 x 5 x 3, the
        output feature maps will be 28 x 28 x 6.
    :param kernel_weight_regularizer: Instance of
        `keras.regularizers.Regularizer`.  This may be created, for example, by
        `get_weight_regularizer`.
    :param bias_weight_regularizer: See above.
    :param is_first_layer: Boolean flag.  If True, this is the first layer in
        the network (convolves over raw images rather than feature maps, which
        means that dimensions of raw images must be known).
    :param num_input_rows: [used only if is_first_layer = True]
        Number of pixel rows (M) in each raw image.
    :param num_input_columns: [used only if is_first_layer = True]
        Number of pixel columns (N) in each raw image.
    :param num_input_depths: [used only if is_first_layer = True]
        Number of pixel depths (D) in each raw image.
    :param num_input_channels: [used only if is_first_layer = True]
        Number of channels (C) -- or predictor variables -- in each raw image.
    :return: layer_object: Instance of `keras.layers.Conv3D`.
    """

    _check_input_args_for_conv_layer(
        num_output_filters=num_output_filters, num_kernel_rows=num_kernel_rows,
        num_kernel_columns=num_kernel_columns,
        num_kernel_depths=num_kernel_depths,
        num_rows_per_stride=num_rows_per_stride,
        num_columns_per_stride=num_columns_per_stride,
        num_depths_per_stride=num_depths_per_stride, padding_type=padding_type,
        is_first_layer=is_first_layer, num_input_rows=num_input_rows,
        num_input_columns=num_input_columns, num_input_depths=num_input_depths,
        num_input_channels=num_input_channels)

    if is_first_layer:
        return keras.layers.Conv3D(
            filters=num_output_filters,
            kernel_size=(num_kernel_rows, num_kernel_columns,
                         num_kernel_depths),
            strides=(num_rows_per_stride, num_columns_per_stride,
                     num_depths_per_stride),
            padding=padding_type, data_format='channels_last',
            dilation_rate=(1, 1, 1), activation=None, use_bias=True,
            kernel_initializer='glorot_uniform', bias_initializer='zeros',
            kernel_regularizer=kernel_weight_regularizer,
            bias_regularizer=bias_weight_regularizer,
            input_shape=(num_input_rows, num_input_columns, num_input_depths,
                         num_input_channels))

    return keras.layers.Conv3D(
        filters=num_output_filters,
        kernel_size=(
            num_kernel_rows, num_kernel_columns, num_kernel_depths),
        strides=(
            num_rows_per_stride, num_columns_per_stride, num_depths_per_stride),
        padding=padding_type, data_format='channels_last',
        dilation_rate=(1, 1, 1), activation=None, use_bias=True,
        kernel_initializer='glorot_uniform', bias_initializer='zeros',
        kernel_regularizer=kernel_weight_regularizer,
        bias_regularizer=bias_weight_regularizer)


def get_dropout_layer(dropout_fraction=DEFAULT_DROPOUT_FRACTION):
    """Creates dropout layer.

    To add this layer to a Keras model, use the following code.

    model_object = keras.models.Sequential()
    ... # Add other layers here.
    layer_object = get_dropout_layer(...)
    model_object.add(layer_object)

    :param dropout_fraction: Fraction of input units to drop.
    :return: layer_object: Instance of `keras.layers.Dropout`.
    """

    error_checking.assert_is_greater(dropout_fraction, 0.)
    error_checking.assert_is_less_than(dropout_fraction, 1.)
    return keras.layers.Dropout(rate=dropout_fraction)


def get_1d_pooling_layer(
        num_pixels_in_window, num_pixels_per_stride,
        pooling_type=MAX_POOLING_TYPE):
    """Creates 1-D-pooling layer.

    In other words, this layer will pool over 1-D feature maps.

    model_object = keras.models.Sequential()
    ... # May add other layers here.
    layer_object = get_1d_pooling_layer(...)
    model_object.add(layer_object)

    :param num_pixels_in_window: See doc for `get_3d_pooling_layer`.
    :param num_pixels_per_stride: Same.
    :param pooling_type: Same.
    :return: layer_object: Instance of `keras.layers.MaxPooling2D` or
        `keras.layers.AveragePooling2D`.
    """

    _check_input_args_for_pooling_layer(
        num_rows_in_window=num_pixels_in_window, pooling_type=pooling_type,
        num_rows_per_stride=num_pixels_per_stride)

    if pooling_type == MAX_POOLING_TYPE:
        return keras.layers.MaxPooling1D(
            pool_size=num_pixels_in_window, strides=num_pixels_per_stride,
            padding=NO_PADDING_TYPE)

    return keras.layers.AveragePooling1D(
        pool_size=num_pixels_in_window, strides=num_pixels_per_stride,
        padding=NO_PADDING_TYPE)


def get_2d_pooling_layer(
        num_rows_in_window, num_columns_in_window, num_rows_per_stride,
        num_columns_per_stride, pooling_type=MAX_POOLING_TYPE):
    """Creates 2-D-pooling layer.

    In other words, this layer will pool over 2-D feature maps.

    model_object = keras.models.Sequential()
    ... # May add other layers here.
    layer_object = get_2d_pooling_layer(...)
    model_object.add(layer_object)

    :param num_rows_in_window: See doc for `get_3d_pooling_layer`.
    :param num_columns_in_window: Same.
    :param pooling_type: Same.
    :param num_rows_per_stride: Same.
    :param num_columns_per_stride: Same.
    :return: layer_object: Instance of `keras.layers.MaxPooling2D` or
        `keras.layers.AveragePooling2D`.
    """

    _check_input_args_for_pooling_layer(
        num_rows_in_window=num_rows_in_window,
        num_columns_in_window=num_columns_in_window, pooling_type=pooling_type,
        num_rows_per_stride=num_rows_per_stride,
        num_columns_per_stride=num_columns_per_stride)

    if pooling_type == MAX_POOLING_TYPE:
        return keras.layers.MaxPooling2D(
            pool_size=(num_rows_in_window, num_columns_in_window),
            strides=(num_rows_per_stride, num_columns_per_stride),
            padding=NO_PADDING_TYPE, data_format='channels_last')

    return keras.layers.AveragePooling2D(
        pool_size=(num_rows_in_window, num_columns_in_window),
        strides=(num_rows_per_stride, num_columns_per_stride),
        padding=NO_PADDING_TYPE, data_format='channels_last')


def get_3d_pooling_layer(
        num_rows_in_window, num_columns_in_window, num_depths_in_window,
        num_rows_per_stride, num_columns_per_stride, num_depths_per_stride,
        pooling_type=MAX_POOLING_TYPE):
    """Creates 3-D-pooling layer.

    In other words, this layer will pool over 3-D feature maps.

    model_object = keras.models.Sequential()
    ... # May add other layers here.
    layer_object = get_3d_pooling_layer(...)
    model_object.add(layer_object)

    :param num_rows_in_window: Number of rows in pooling window.
    :param num_columns_in_window: Number of columns in pooling window.
    :param num_depths_in_window: Number of depths in pooling window.
    :param pooling_type: Pooling type (either "max" or "mean").
    :param num_rows_per_stride: Stride length in y-direction.  Default is
        `num_rows_in_window`, which means that adjacent positions of the window
        are abutting but non-overlapping.
    :param num_columns_per_stride: Stride length in x-direction.  Default is
        `num_columns_in_window`, which means that adjacent positions of the
        window are abutting but non-overlapping.
    :param num_depths_per_stride: Stride length in z-direction.  Default is
        `num_depths_in_window`, which means that adjacent positions of the
        window are abutting but non-overlapping.
    :return: layer_object: Instance of `keras.layers.MaxPooling3D` or
        `keras.layers.AveragePooling3D`.
    """

    _check_input_args_for_pooling_layer(
        num_rows_in_window=num_rows_in_window,
        num_columns_in_window=num_columns_in_window,
        num_depths_in_window=num_depths_in_window, pooling_type=pooling_type,
        num_rows_per_stride=num_rows_per_stride,
        num_columns_per_stride=num_columns_per_stride,
        num_depths_per_stride=num_depths_per_stride)

    if pooling_type == MAX_POOLING_TYPE:
        return keras.layers.MaxPooling3D(
            pool_size=(num_rows_in_window, num_columns_in_window,
                       num_depths_in_window),
            strides=(num_rows_per_stride, num_columns_per_stride,
                     num_depths_per_stride),
            padding=NO_PADDING_TYPE, data_format='channels_last')

    return keras.layers.AveragePooling3D(
        pool_size=(num_rows_in_window, num_columns_in_window,
                   num_depths_in_window),
        strides=(num_rows_per_stride, num_columns_per_stride,
                 num_depths_per_stride),
        padding=NO_PADDING_TYPE, data_format='channels_last')


def get_batch_normalization_layer():
    """Creates batch-normalization layer.

    :return: layer_object: Instance of `keras.layers.BatchNormalization`.
    """

    return keras.layers.BatchNormalization(
        axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True)


def get_fully_connected_layer(
        num_output_units, kernel_weight_regularizer=None,
        bias_weight_regularizer=None):
    """Creates fully connected layer (traditional neural-net layer).

    :param num_output_units: Number of output units ("neurons").
    :param kernel_weight_regularizer: See doc for `get_3d_conv_layer`.
    :param bias_weight_regularizer: Same.
    :return: layer_object: Instance of `keras.layers.Dense`.
    """

    error_checking.assert_is_integer(num_output_units)
    error_checking.assert_is_greater(num_output_units, 0)

    return keras.layers.Dense(
        num_output_units, activation=None, use_bias=True,
        kernel_initializer='glorot_uniform', bias_initializer='zeros',
        kernel_regularizer=kernel_weight_regularizer,
        bias_regularizer=bias_weight_regularizer)


def get_flattening_layer():
    """Creates flattening layer.

    A "flattening layer" concatenates input feature maps into a single 1-D
    vector.

    :return: layer_object: Instance of `keras.layers.Flatten`.
    """

    return keras.layers.Flatten()


def get_activation_layer(
        activation_function_string, alpha_for_elu=DEFAULT_ALPHA_FOR_ELU,
        alpha_for_relu=DEFAULT_ALPHA_FOR_RELU):
    """Creates activation layer.

    :param activation_function_string: Activation function (must be accepted by
        `check_activation_function`).
    :param alpha_for_elu: [used only if activation_function_string = "elu"]
        Slope for negative inputs to activation function.
    :param alpha_for_relu: [used only if activation_function_string = "relu"]
        Slope for negative inputs to activation function.
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
