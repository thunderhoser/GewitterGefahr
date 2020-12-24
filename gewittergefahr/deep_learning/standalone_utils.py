"""Stand-alone Keras operations.

--- NOTATION ---

The following letters will be used throughout this module.

M = number of rows before operation
N = number of columns before operation
H = number of heights before operation
C = number of channels before operation
"""

import numpy
import tensorflow.keras.layers as layers
import tensorflow.python.keras.backend as K
from gewittergefahr.gg_utils import error_checking
from gewittergefahr.deep_learning import architecture_utils

MAX_POOLING_TYPE_STRING = 'max'
MEAN_POOLING_TYPE_STRING = 'avg'
VALID_POOLING_TYPE_STRINGS = [MAX_POOLING_TYPE_STRING, MEAN_POOLING_TYPE_STRING]


def _check_pooling_type(pooling_type_string):
    """Error-checks pooling type.

    :param pooling_type_string: Pooling type ("max" or "avg").
    :raises: ValueError: if pooling type is unrecognized.
    """

    error_checking.assert_is_string(pooling_type_string)

    if pooling_type_string not in VALID_POOLING_TYPE_STRINGS:
        error_string = (
            '\n\n{0:s}\nValid pooling types (listed above) do not include '
            '"{1:s}".'
        ).format(str(VALID_POOLING_TYPE_STRINGS), pooling_type_string)

        raise ValueError(error_string)


def do_2d_convolution(
        feature_matrix, kernel_matrix, pad_edges=False, stride_length_px=1):
    """Convolves 2-D feature maps with 2-D kernel.

    m = number of rows in kernel
    n = number of columns in kernel
    c = number of output feature maps (channels)

    :param feature_matrix: Input feature maps (numpy array).  Dimensions must be
        M x N x C or 1 x M x N x C.
    :param kernel_matrix: Kernel as numpy array.  Dimensions must be
        m x n x C x c.
    :param pad_edges: Boolean flag.  If True, edges of input feature maps will
        be zero-padded during convolution, so spatial dimensions of the output
        feature maps will be the same (M x N).  If False, dimensions
        of the output maps will be (M - m + 1) x (N - n + 1).
    :param stride_length_px: Stride length (pixels).  The kernel will move by
        this many rows or columns at a time as it slides over each input feature
        map.
    :return: feature_matrix: Output feature maps (numpy array).  Dimensions will
        be 1 x M x N x c or 1 x (M - m + 1) x (N - n + 1) x c, depending on
        whether or not edges are padded.
    """

    error_checking.assert_is_numpy_array_without_nan(feature_matrix)
    error_checking.assert_is_numpy_array_without_nan(kernel_matrix)
    error_checking.assert_is_numpy_array(kernel_matrix, num_dimensions=4)
    error_checking.assert_is_boolean(pad_edges)
    error_checking.assert_is_integer(stride_length_px)
    error_checking.assert_is_geq(stride_length_px, 1)

    if len(feature_matrix.shape) == 3:
        feature_matrix = numpy.expand_dims(feature_matrix, axis=0)

    error_checking.assert_is_numpy_array(feature_matrix, num_dimensions=4)

    if pad_edges:
        padding_string = 'same'
    else:
        padding_string = 'valid'

    feature_tensor = K.conv2d(
        x=K.variable(feature_matrix), kernel=K.variable(kernel_matrix),
        strides=(stride_length_px, stride_length_px), padding=padding_string,
        data_format='channels_last'
    )

    return feature_tensor.numpy()


def do_3d_convolution(
        feature_matrix, kernel_matrix, pad_edges=False, stride_length_px=1):
    """Convolves 3-D feature maps with 3-D kernel.

    m = number of rows in kernel
    n = number of columns in kernel
    h = number of height in kernel
    c = number of output feature maps (channels)

    :param feature_matrix: Input feature maps (numpy array).  Dimensions must be
        M x N x H x C or 1 x M x N x H x C.
    :param kernel_matrix: Kernel as numpy array.  Dimensions must be
        m x n x h x C x c.
    :param pad_edges: See doc for `do_2d_convolution`.
    :param stride_length_px: See doc for `do_2d_convolution`.
    :return: feature_matrix: Output feature maps (numpy array).  Dimensions will
        be 1 x M x N x H x c or
        1 x (M - m + 1) x (N - n + 1) x (H - h + 1) x c, depending on
        whether or not edges are padded.
    """

    error_checking.assert_is_numpy_array_without_nan(feature_matrix)
    error_checking.assert_is_numpy_array_without_nan(kernel_matrix)
    error_checking.assert_is_numpy_array(kernel_matrix, num_dimensions=5)
    error_checking.assert_is_boolean(pad_edges)
    error_checking.assert_is_integer(stride_length_px)
    error_checking.assert_is_geq(stride_length_px, 1)

    if len(feature_matrix.shape) == 4:
        feature_matrix = numpy.expand_dims(feature_matrix, axis=0)

    error_checking.assert_is_numpy_array(feature_matrix, num_dimensions=5)

    if pad_edges:
        padding_string = 'same'
    else:
        padding_string = 'valid'

    feature_tensor = K.conv3d(
        x=K.variable(feature_matrix), kernel=K.variable(kernel_matrix),
        strides=(stride_length_px, stride_length_px, stride_length_px),
        padding=padding_string, data_format='channels_last'
    )

    return feature_tensor.numpy()


def do_activation(input_values, function_name, alpha=0.2):
    """Runs input array through activation function.

    :param input_values: numpy array (any shape).
    :param function_name: Name of activation function (must be accepted by ``).
    :param alpha: Slope parameter (alpha) for activation function.  This applies
        only for eLU and ReLU.
    :return: output_values: Same as `input_values` but post-activation.
    """

    architecture_utils.check_activation_function(
        activation_function_string=function_name, alpha_for_elu=alpha,
        alpha_for_relu=alpha
    )

    input_object = K.placeholder()

    if function_name == architecture_utils.ELU_FUNCTION_STRING:
        function_object = K.function(
            [input_object],
            [layers.ELU(alpha=alpha)(input_object)]
        )
    elif function_name == architecture_utils.RELU_FUNCTION_STRING:
        function_object = K.function(
            [input_object],
            [layers.LeakyReLU(alpha=alpha)(input_object)]
        )
    else:
        function_object = K.function(
            [input_object],
            [layers.Activation(function_name)(input_object)]
        )

    return function_object([input_values])[0]


def do_2d_upsampling(feature_matrix, upsampling_factor=2,
                     use_linear_interp=True):
    """Upsamples 2-D feature maps.

    m = number of rows after upsampling
    n = number of columns after upsampling

    :param feature_matrix: Input feature maps (numpy array).  Dimensions must be
        M x N x C or 1 x M x N x C.
    :param upsampling_factor: Upsampling factor (integer > 1).
    :param use_linear_interp: Boolean flag.  If True (False), will use linear
        (nearest-neighbour) interpolation.
    :return: feature_matrix: Output feature maps (numpy array).  Dimensions will
        be 1 x m x n x C.
    """

    error_checking.assert_is_numpy_array_without_nan(feature_matrix)
    error_checking.assert_is_integer(upsampling_factor)
    error_checking.assert_is_geq(upsampling_factor, 2)
    error_checking.assert_is_boolean(use_linear_interp)

    if len(feature_matrix.shape) == 3:
        feature_matrix = numpy.expand_dims(feature_matrix, axis=0)

    error_checking.assert_is_numpy_array(feature_matrix, num_dimensions=4)


def do_2d_pooling(feature_matrix, stride_length_px=2,
                  pooling_type_string=MAX_POOLING_TYPE_STRING):
    """Pools 2-D feature maps.

    m = number of rows after pooling
    n = number of columns after pooling

    :param feature_matrix: Input feature maps (numpy array).  Dimensions must be
        M x N x C or 1 x M x N x C.
    :param stride_length_px: Stride length (pixels).  The pooling window will
        move by this many rows or columns at a time as it slides over each input
        feature map.
    :param pooling_type_string: Pooling type (must be accepted by
        `_check_pooling_type`).
    :return: feature_matrix: Output feature maps (numpy array).  Dimensions will
        be 1 x m x n x C.
    """

    error_checking.assert_is_numpy_array_without_nan(feature_matrix)
    error_checking.assert_is_integer(stride_length_px)
    error_checking.assert_is_geq(stride_length_px, 2)
    _check_pooling_type(pooling_type_string)

    if len(feature_matrix.shape) == 3:
        feature_matrix = numpy.expand_dims(feature_matrix, axis=0)

    error_checking.assert_is_numpy_array(feature_matrix, num_dimensions=4)

    feature_tensor = K.pool2d(
        x=K.variable(feature_matrix), pool_mode=pooling_type_string,
        pool_size=(stride_length_px, stride_length_px),
        strides=(stride_length_px, stride_length_px), padding='valid',
        data_format='channels_last'
    )

    return feature_tensor.numpy()


def do_3d_pooling(feature_matrix, stride_length_px=2,
                  pooling_type_string=MAX_POOLING_TYPE_STRING):
    """Pools 3-D feature maps.

    :param feature_matrix: Input feature maps (numpy array).  Dimensions must be
        M x N x H x C or 1 x M x N x H x C.
    :param stride_length_px: See doc for `do_2d_pooling`.
    :param pooling_type_string: Pooling type (must be accepted by
        `_check_pooling_type`).
    :return: feature_matrix: Output feature maps (numpy array).  Dimensions will
        be 1 x m x n x h x C.
    """

    error_checking.assert_is_numpy_array_without_nan(feature_matrix)
    error_checking.assert_is_integer(stride_length_px)
    error_checking.assert_is_geq(stride_length_px, 2)
    _check_pooling_type(pooling_type_string)

    if len(feature_matrix.shape) == 4:
        feature_matrix = numpy.expand_dims(feature_matrix, axis=0)

    error_checking.assert_is_numpy_array(feature_matrix, num_dimensions=5)

    feature_tensor = K.pool3d(
        x=K.variable(feature_matrix), pool_mode=pooling_type_string,
        pool_size=(stride_length_px, stride_length_px, stride_length_px),
        strides=(stride_length_px, stride_length_px, stride_length_px),
        padding='valid', data_format='channels_last'
    )

    return feature_tensor.numpy()


def do_batch_normalization(
        feature_matrix, scale_parameter=1., shift_parameter=0.):
    """Performs batch normalization on each feature in the batch.

    E = number of examples in batch

    :param feature_matrix: Input feature maps (numpy array).  Dimensions must be
        E x M x N x C or E x M x N x H x C.
    :param scale_parameter: Scale parameter (beta in the equation on page 3 of
        Ioffe and Szegedy 2015).
    :param shift_parameter: Shift parameter (gamma in the equation).
    :return: feature_matrix: Feature maps after batch normalization (same
        dimensions).
    """

    error_checking.assert_is_numpy_array_without_nan(feature_matrix)
    error_checking.assert_is_greater(scale_parameter, 0.)

    num_dimensions = len(feature_matrix.shape)
    error_checking.assert_is_geq(num_dimensions, 4)
    error_checking.assert_is_leq(num_dimensions, 5)

    stdev_matrix = numpy.std(feature_matrix, axis=0, ddof=1)
    stdev_matrix = numpy.expand_dims(stdev_matrix, axis=0)
    stdev_matrix = numpy.repeat(stdev_matrix, feature_matrix.shape[0], axis=0)

    mean_matrix = numpy.mean(feature_matrix, axis=0)
    mean_matrix = numpy.expand_dims(mean_matrix, axis=0)
    mean_matrix = numpy.repeat(mean_matrix, feature_matrix.shape[0], axis=0)

    return shift_parameter + scale_parameter * (
        (feature_matrix - mean_matrix) / (stdev_matrix + K.epsilon())
    )
