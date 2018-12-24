"""Methods for setting up, training, and applying upconvolution networks."""

import numpy
import keras
from gewittergefahr.gg_utils import error_checking

L1_WEIGHT = 0.
L2_WEIGHT = 0.001
SLOPE_FOR_RELU = 0.2
NUM_CONV_FILTER_ROWS = 3
NUM_CONV_FILTER_COLUMNS = 3
NUM_SMOOTHING_FILTER_ROWS = 5
NUM_SMOOTHING_FILTER_COLUMNS = 5


def _create_smoothing_filter(
        smoothing_radius_px, num_filter_rows, num_filter_columns, num_channels):
    """Creates convolution filter for Gaussian smoothing.

    M = number of rows in filter
    N = number of columns in filter
    C = number of channels (or "variables" or "features") to smooth.  Each
        channel will be smoothed independently.

    :param smoothing_radius_px: e-folding radius (pixels).
    :param num_filter_rows: M in the above discussion.
    :param num_filter_columns: N in the above discussion.
    :param num_channels: C in the above discussion.
    :return: weight_matrix: M-by-N-by-C-by-C numpy array of convolution weights.
    """


def create_net(
        num_input_features, first_num_rows, first_num_columns,
        upsampling_factors, num_output_channels,
        use_activation_for_out_layer=False, use_bn_for_out_layer=True,
        use_transposed_conv=False, smoothing_radius_px=None):
    """Creates (but does not train) upconvnet.
    
    L = number of conv or deconv layers
    
    :param num_input_features: Number of input features.
    :param first_num_rows: Number of rows in input to first deconv layer.  The
        input features will be reshaped into a grid with this many rows.
    :param first_num_columns: Same but for columns.
    :param upsampling_factors: length-L numpy array of upsampling factors.  Must
        all be positive integers.
    :param num_output_channels: Number of channels in output images.
    :param use_activation_for_out_layer: Boolean flag.  If True, activation will
        be applied to output layer.
    :param use_bn_for_out_layer: Boolean flag.  If True, batch normalization
        will be applied to output layer.
    :param use_transposed_conv: Boolean flag.  If True, upsampling will be done
        with transposed-convolution layers.  If False, each upsampling will be
        done with an upsampling layer followed by a conv layer.
    :param smoothing_radius_px: Smoothing radius (pixels).  Gaussian smoothing
        with this e-folding radius will be done after each upsampling.  If
        `smoothing_radius_px is None`, no smoothing will be done.
    :return: model_object: Untrained instance of `keras.models.Model`.
    """

    error_checking.assert_is_integer(num_input_features)
    error_checking.assert_is_integer(first_num_rows)
    error_checking.assert_is_integer(first_num_columns)
    error_checking.assert_is_integer(num_output_channels)

    error_checking.assert_is_greater(num_input_features, 0)
    error_checking.assert_is_greater(first_num_rows, 0)
    error_checking.assert_is_greater(first_num_columns, 0)
    error_checking.assert_is_greater(num_output_channels, 0)

    error_checking.assert_is_integer_numpy_array(upsampling_factors)
    error_checking.assert_is_numpy_array(upsampling_factors, num_dimensions=1)
    error_checking.assert_is_geq_numpy_array(upsampling_factors, 1)

    error_checking.assert_is_boolean(use_activation_for_out_layer)
    error_checking.assert_is_boolean(use_bn_for_out_layer)
    error_checking.assert_is_boolean(use_transposed_conv)

    if smoothing_radius_px is not None:
        error_checking.assert_is_integer(smoothing_radius_px)
        error_checking.assert_is_greater(smoothing_radius_px, 0)

    regularizer_object = keras.regularizers.l1_l2(l1=L1_WEIGHT, l2=L2_WEIGHT)
    input_layer_object = keras.layers.Input(shape=(num_input_features,))

    current_num_filters = int(numpy.round(
        num_input_features / (first_num_rows * first_num_columns)
    ))

    layer_object = keras.layers.Reshape(
        target_shape=(first_num_rows, first_num_columns, current_num_filters)
    )(input_layer_object)

    num_main_layers = len(upsampling_factors)

    for i in range(num_main_layers):
        this_upsampling_factor = upsampling_factors[i]

        if i == num_main_layers - 1:
            current_num_filters = num_output_channels + 0
        elif this_upsampling_factor == 1:
            current_num_filters = int(numpy.round(current_num_filters / 2))

        if use_transposed_conv:
            if this_upsampling_factor > 1:
                this_padding_arg = 'same'
            else:
                this_padding_arg = 'valid'

            layer_object = keras.layers.Conv2DTranspose(
                filters=current_num_filters,
                kernel_size=(NUM_CONV_FILTER_ROWS, NUM_CONV_FILTER_COLUMNS),
                strides=(this_upsampling_factor, this_upsampling_factor),
                padding=this_padding_arg, data_format='channels_last',
                dilation_rate=(1, 1), activation=None, use_bias=True,
                kernel_initializer='glorot_uniform', bias_initializer='zeros',
                kernel_regularizer=regularizer_object
            )(layer_object)

        else:
            if this_upsampling_factor > 1:
                layer_object = keras.layers.UpSampling2D(
                    size=(this_upsampling_factor, this_upsampling_factor),
                    data_format='channels_last', interpolation='nearest'
                )(layer_object)

            layer_object = keras.layers.Conv2D(
                filters=current_num_filters,
                kernel_size=(NUM_CONV_FILTER_ROWS, NUM_CONV_FILTER_COLUMNS),
                strides=(1, 1), padding='same', data_format='channels_last',
                dilation_rate=(1, 1), activation=None, use_bias=True,
                kernel_initializer='glorot_uniform', bias_initializer='zeros',
                kernel_regularizer=regularizer_object
            )(layer_object)

        if i < num_main_layers - 1 or use_activation_for_out_layer:
            layer_object = keras.layers.LeakyReLU(
                alpha=SLOPE_FOR_RELU
            )(layer_object)

        if i < num_main_layers - 1 or use_bn_for_out_layer:
            layer_object = keras.layers.BatchNormalization(
                axis=-1, center=True, scale=True
            )(layer_object)

    model_object = keras.models.Model(
        inputs=input_layer_object, outputs=layer_object)
    model_object.compile(
        loss=keras.losses.mean_squared_error, optimizer=keras.optimizers.Adam())

    model_object.summary()
    return model_object
