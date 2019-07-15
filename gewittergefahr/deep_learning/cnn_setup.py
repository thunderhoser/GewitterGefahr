"""CNN setup (building the architecture and compiling)."""

import numpy
import keras.layers
from gewittergefahr.gg_utils import error_checking
from gewittergefahr.deep_learning import architecture_utils
from gewittergefahr.deep_learning import keras_metrics

NUM_ROWS_KEY = 'num_grid_rows'
NUM_COLUMNS_KEY = 'num_grid_columns'
NUM_DIMENSIONS_KEY = 'num_dimensions'
NUM_CLASSES_KEY = 'num_classes'
NUM_CONV_LAYER_SETS_KEY = 'num_conv_layer_sets'
NUM_CONV_LAYERS_PER_SET_KEY = 'num_conv_layers_per_set'
FIRST_NUM_FILTERS_KEY = 'first_num_filters'
CONV_LAYER_DROPOUT_KEY = 'conv_layer_dropout_fraction'
NUM_DENSE_LAYERS_KEY = 'num_dense_layers'
DENSE_LAYER_DROPOUT_KEY = 'dense_layer_dropout_fraction'
NUM_CHANNELS_KEY = 'num_channels'
NUM_FIELDS_KEY = 'num_fields'
NUM_HEIGHTS_KEY = 'num_heights'
L1_WEIGHT_KEY = 'l1_weight'
L2_WEIGHT_KEY = 'l2_weight'
ACTIVATION_FUNCTION_KEY = 'activation_function_string'
ALPHA_FOR_ELU_KEY = 'alpha_for_elu'
ALPHA_FOR_RELU_KEY = 'alpha_for_relu'
USE_BATCH_NORM_KEY = 'use_batch_normalization'

NUM_CONV_KERNEL_ROWS = 3
NUM_CONV_KERNEL_COLUMNS = 3
NUM_CONV_KERNEL_HEIGHTS = 3

DEFAULT_NUM_CONV_LAYER_SETS = 2
DEFAULT_NUM_CONV_LAYERS_PER_SET = 2
DEFAULT_CONV_LAYER_DROPOUT_FRACTION = None
DEFAULT_NUM_DENSE_LAYERS = 3
DEFAULT_DENSE_LAYER_DROPOUT_FRACTION = 0.5
DEFAULT_L1_WEIGHT = 0.
DEFAULT_L2_WEIGHT = 0.001
DEFAULT_USE_BATCH_NORM_FLAG = True
DEFAULT_ACTIVATION_FUNCTION_STRING = architecture_utils.RELU_FUNCTION_STRING

VALID_NUMBERS_OF_SOUNDING_HEIGHTS = numpy.array([49], dtype=int)

DEFAULT_METRIC_FUNCTION_LIST = [
    keras_metrics.accuracy, keras_metrics.binary_accuracy,
    keras_metrics.binary_csi, keras_metrics.binary_frequency_bias,
    keras_metrics.binary_pod, keras_metrics.binary_pofd,
    keras_metrics.binary_peirce_score, keras_metrics.binary_success_ratio,
    keras_metrics.binary_focn
]


def _create_sounding_layers(
        num_fields, num_heights, num_conv_layer_sets,
        num_conv_layers_per_set, padding_type_string, l1_weight, l2_weight,
        pooling_type_string, activation_function_string, alpha_for_elu,
        alpha_for_relu, use_batch_normalization, dropout_fraction,
        do_separable_conv, first_num_filters=None,
        first_num_spatial_filters=None, num_non_spatial_filters=None):
    """Creates convolution and pooling layers to process sounding data.

    :param num_fields: See doc for `check_sounding_options`.
    :param num_heights: Same.
    :param num_conv_layer_sets: See doc for `check_radar_options`.
    :param num_conv_layers_per_set: Same.
    :param padding_type_string: Padding type (see doc for
        `architecture_utils._check_convolution_options`).
    :param l1_weight: See doc for `check_radar_options`.
    :param l2_weight: Same.
    :param pooling_type_string: Same.
    :param activation_function_string: Same.
    :param alpha_for_elu: Same.
    :param alpha_for_relu: Same.
    :param use_batch_normalization: Same.
    :param dropout_fraction: Same.
    :param do_separable_conv: See doc for `check_sounding_options`.
    :param first_num_filters: Same.
    :param first_num_spatial_filters: Same.
    :param num_non_spatial_filters: Same.
    :return: input_layer_object: Input layer (instance of `keras.layers.Input`),
        used to feed soundings into the CNN.
    :return: flattening_layer_object: Flattening layer (instance of
        `keras.layers.Flatten`), which outputs scalar features.
    :return: num_scalar_features: Number of scalar features output by
        flattening layer.
    """

    first_num_filters, num_non_spatial_filters = check_sounding_options(
        num_fields=num_fields, num_heights=num_heights,
        do_separable_conv=do_separable_conv,
        first_num_filters=first_num_filters,
        first_num_spatial_filters=first_num_spatial_filters,
        num_non_spatial_filters=num_non_spatial_filters)

    input_layer_object = keras.layers.Input(
        shape=(num_heights, num_fields)
    )

    regularizer_object = architecture_utils.get_weight_regularizer(
        l1_weight=l1_weight, l2_weight=l2_weight)

    current_layer_object = None
    current_num_filters = None
    current_num_spatial_filters = None

    for i in range(num_conv_layer_sets):
        # if current_layer_object is None:
        #     if do_separable_conv:
        #         current_num_spatial_filters = first_num_spatial_filters + 0
        #     else:
        #         current_num_filters = first_num_filters + 0
        # else:
        #     if do_separable_conv:
        #         current_num_spatial_filters *= 2
        #     else:
        #         current_num_filters *= 2

        for _ in range(num_conv_layers_per_set):
            if current_layer_object is None:
                this_input_layer_object = input_layer_object

                if do_separable_conv:
                    current_num_spatial_filters = first_num_spatial_filters + 0
                else:
                    current_num_filters = first_num_filters + 0
            else:
                this_input_layer_object = current_layer_object

                if do_separable_conv:
                    current_num_spatial_filters *= 2
                else:
                    current_num_filters *= 2

            if do_separable_conv:
                current_layer_object = (
                    architecture_utils.get_1d_separable_conv_layer(
                        num_kernel_rows=NUM_CONV_KERNEL_HEIGHTS,
                        num_rows_per_stride=1,
                        num_spatial_filters=current_num_spatial_filters,
                        num_non_spatial_filters=num_non_spatial_filters,
                        padding_type_string=architecture_utils.YES_PADDING_STRING,
                        weight_regularizer=regularizer_object
                    )(this_input_layer_object)
                )
            else:
                current_layer_object = architecture_utils.get_1d_conv_layer(
                    num_kernel_rows=NUM_CONV_KERNEL_HEIGHTS,
                    num_rows_per_stride=1, num_filters=current_num_filters,
                    padding_type_string=architecture_utils.YES_PADDING_STRING,
                    weight_regularizer=regularizer_object
                )(this_input_layer_object)

            current_layer_object = architecture_utils.get_activation_layer(
                activation_function_string=activation_function_string,
                alpha_for_elu=alpha_for_elu, alpha_for_relu=alpha_for_relu
            )(current_layer_object)

            if dropout_fraction is not None:
                current_layer_object = architecture_utils.get_dropout_layer(
                    dropout_fraction
                )(current_layer_object)

            if use_batch_normalization:
                current_layer_object = (
                    architecture_utils.get_batch_norm_layer()(
                        current_layer_object)
                )

        # TODO(thunderhoser): This condition is a HACK.
        current_num_heights = current_layer_object.shape[1]

        if not (current_num_heights < 8 and i == num_conv_layer_sets - 1):
            current_layer_object = architecture_utils.get_1d_pooling_layer(
                num_rows_in_window=2, num_rows_per_stride=2,
                pooling_type_string=pooling_type_string
            )(current_layer_object)

    these_dimensions = numpy.array(
        current_layer_object.get_shape().as_list()[1:], dtype=int)
    num_scalar_features = numpy.prod(these_dimensions)

    current_layer_object = architecture_utils.get_flattening_layer()(
        current_layer_object)
    return input_layer_object, current_layer_object, num_scalar_features


def _get_output_layer_and_loss(num_classes):
    """Creates output layer and loss function.

    :param num_classes: Number of classes for target variable.
    :return: dense_layer_object: Last dense layer (instance of
        `keras.layers.Dense` with no activation).
    :return: activation_layer_object: Last activation layer (to go after the
        dense layer).
    :return: loss_function: Loss function (instance of
        `keras.losses.binary_crossentropy` or
        `keras.losses.categorical_crossentropy`).
    """

    if num_classes == 2:
        num_output_units = 1
        loss_function = keras.losses.binary_crossentropy
        activation_function_string = architecture_utils.SIGMOID_FUNCTION_STRING
    else:
        num_output_units = num_classes
        loss_function = keras.losses.categorical_crossentropy
        activation_function_string = architecture_utils.SOFTMAX_FUNCTION_STRING

    dense_layer_object = architecture_utils.get_dense_layer(
        num_output_units=num_output_units)

    activation_layer_object = architecture_utils.get_activation_layer(
        activation_function_string=activation_function_string)

    return dense_layer_object, activation_layer_object, loss_function


def _create_dense_layers(
        flattening_layer_object, num_scalar_features, num_classes,
        num_dense_layers, dropout_fraction, activation_function_string,
        alpha_for_elu, alpha_for_relu, use_batch_normalization):
    """Creates dense layers (to go between flattening layer and output).

    :param flattening_layer_object: Input to first dense layer (instance of
        `keras.layers.Flatten`).
    :param num_scalar_features: Number of features output by flattening layer.
    :param num_classes: Number of classes for target variable.
    :param num_dense_layers: Desired number of dense layers.
    :param dropout_fraction: Dropout fraction for each dense layer.
    :param activation_function_string: See doc for `check_radar_options`.
    :param alpha_for_elu: Same.
    :param alpha_for_relu: Same.
    :param use_batch_normalization: Same.
    :return: output_layer_object: Output layer for CNN.
    """

    _, num_outputs_by_dense_layer = (
        architecture_utils.get_dense_layer_dimensions(
            num_input_units=num_scalar_features, num_classes=num_classes,
            num_dense_layers=num_dense_layers)
    )

    layer_object = None

    for k in range(num_dense_layers - 1):
        if layer_object is None:
            this_input_layer_object = flattening_layer_object
        else:
            this_input_layer_object = layer_object

        layer_object = architecture_utils.get_dense_layer(
            num_output_units=num_outputs_by_dense_layer[k]
        )(this_input_layer_object)

        layer_object = architecture_utils.get_activation_layer(
            activation_function_string=activation_function_string,
            alpha_for_elu=alpha_for_elu, alpha_for_relu=alpha_for_relu
        )(layer_object)

        if dropout_fraction is not None:
            layer_object = architecture_utils.get_dropout_layer(
                dropout_fraction=dropout_fraction
            )(layer_object)

        if use_batch_normalization:
            layer_object = (
                architecture_utils.get_batch_norm_layer()(layer_object)
            )

    dense_layer_object, activation_layer_object = _get_output_layer_and_loss(
        num_classes
    )[:2]

    layer_object = dense_layer_object(layer_object)

    if dropout_fraction is not None and num_dense_layers == 1:
        layer_object = architecture_utils.get_dropout_layer(
            dropout_fraction=dropout_fraction
        )(layer_object)

    return activation_layer_object(layer_object)


def check_radar_options(
        num_grid_rows, num_grid_columns, num_dimensions, num_classes,
        num_conv_layer_sets, num_conv_layers_per_set,
        conv_layer_dropout_fraction, num_dense_layers,
        dense_layer_dropout_fraction, l1_weight, l2_weight,
        activation_function_string, alpha_for_elu, alpha_for_relu,
        use_batch_normalization, do_separable_conv, first_num_filters=None,
        first_num_spatial_filters=None, num_non_spatial_filters=None,
        num_channels=None, num_fields=None, num_heights=None):
    """Checks options for part of CNN that handles radar data.

    :param num_grid_rows: Number of rows in each grid.
    :param num_grid_columns: Number of columns in each grid.
    :param num_dimensions: Number of dimensions in each grid.
    :param num_classes: Number of classes for target variable.
    :param num_conv_layer_sets: Number of sets of convolution layers
        (uninterrupted by pooling).
    :param num_conv_layers_per_set: Number of convolution layers in each set.
    :param conv_layer_dropout_fraction: Dropout fraction for convolution layers.
        This may be None.
    :param num_dense_layers: Number of dense (fully connected) layers at the
        end.
    :param dense_layer_dropout_fraction: Dropout fraction for dense layers.
        This may be None.  Also, dropout will *not* be used for the last dense
        layer.
    :param l1_weight: L1 regularization weight (used for each conv or dense
        layer).
    :param l2_weight: L2 regularization weight (used for each conv or dense
        layer).
    :param activation_function_string: Activation function (used for each conv
        or dense layer, except the last dense layer).  See doc for
        `architecture_utils.get_activation_layer`.
    :param alpha_for_elu: See doc for
        `architecture_utils.get_activation_layer`.
    :param alpha_for_relu: Same.
    :param use_batch_normalization: Boolean flag.  If True, will use batch
        normalization after each conv or dense layer.
    :param do_separable_conv: Boolean flag.  If True, will do depthwise-
        separable convolution.  If False, will do traditional convolution.
    :param first_num_filters: [used only if `do_separable_conv = False`]
        Number of filters produced by first convolution layer.  The number of
        filters in each successive conv layer will double.
    :param first_num_spatial_filters: [used only if `do_separable_conv = True`]
        Number of spatial filters (applied independently to each input channel)
        produced by first convolution layer.  Will double with each successive
        conv layer.
    :param num_non_spatial_filters: [used only if `do_separable_conv = True`]
        Number of non-spatial filters (applied independently to each spatial
        position) produced by each conv layer.
    :param num_channels: [used only if num_dimensions = 2]
        Number of channels (field/height pairs).
    :param num_fields: [used only if num_dimensions = 3]
        Number of fields.
    :param num_heights: [used only if num_dimensions = 3]
        Number of heights.

    :return: first_num_filters: Same as input, except value may have been
        replaced with default.
    :return: num_non_spatial_filters: Same.
    """

    error_checking.assert_is_integer(num_grid_rows)
    error_checking.assert_is_geq(num_grid_rows, 8)
    error_checking.assert_is_integer(num_grid_columns)
    error_checking.assert_is_geq(num_grid_columns, 8)
    error_checking.assert_is_integer(num_dimensions)
    error_checking.assert_is_geq(num_dimensions, 2)
    error_checking.assert_is_leq(num_dimensions, 3)
    error_checking.assert_is_integer(num_classes)
    error_checking.assert_is_geq(num_classes, 2)

    if num_dimensions == 2:
        error_checking.assert_is_integer(num_channels)
        error_checking.assert_is_geq(num_channels, 1)
    else:
        error_checking.assert_is_integer(num_fields)
        error_checking.assert_is_geq(num_fields, 1)
        error_checking.assert_is_integer(num_heights)
        error_checking.assert_is_geq(num_heights, 4)

    error_checking.assert_is_integer(num_conv_layer_sets)
    error_checking.assert_is_geq(num_conv_layer_sets, 1)
    error_checking.assert_is_integer(num_conv_layers_per_set)
    error_checking.assert_is_geq(num_conv_layers_per_set, 1)

    if conv_layer_dropout_fraction is not None:
        error_checking.assert_is_greater(conv_layer_dropout_fraction, 0.)
        error_checking.assert_is_less_than(conv_layer_dropout_fraction, 1.)

    error_checking.assert_is_integer(num_dense_layers)
    error_checking.assert_is_geq(num_dense_layers, 1)

    if dense_layer_dropout_fraction is not None:
        error_checking.assert_is_greater(dense_layer_dropout_fraction, 0.)
        error_checking.assert_is_less_than(dense_layer_dropout_fraction, 1.)

    error_checking.assert_is_geq(l1_weight, 0.)
    error_checking.assert_is_geq(l2_weight, 0.)

    architecture_utils.check_activation_function(
        activation_function_string=activation_function_string,
        alpha_for_elu=alpha_for_elu, alpha_for_relu=alpha_for_relu)

    error_checking.assert_is_boolean(use_batch_normalization)
    error_checking.assert_is_boolean(do_separable_conv)

    if not do_separable_conv:
        if first_num_filters is None:
            if num_dimensions == 2:
                first_num_filters = 8 * num_channels
            else:
                first_num_filters = 8 * num_fields

        error_checking.assert_is_integer(first_num_filters)
        error_checking.assert_is_geq(first_num_filters, 4)

        return first_num_filters, None

    if num_non_spatial_filters is None:
        if num_dimensions == 2:
            num_non_spatial_filters = num_channels + 0
        else:
            num_non_spatial_filters = num_fields + 0

    error_checking.assert_is_integer(first_num_spatial_filters)
    error_checking.assert_is_greater(first_num_spatial_filters, 0)
    error_checking.assert_is_integer(num_non_spatial_filters)
    error_checking.assert_is_greater(num_non_spatial_filters, 0)
    error_checking.assert_is_geq(
        first_num_spatial_filters * num_non_spatial_filters, 4)

    return None, num_non_spatial_filters


def check_sounding_options(
        num_fields, num_heights, do_separable_conv, first_num_filters=None,
        first_num_spatial_filters=None, num_non_spatial_filters=None):
    """Checks options for part of CNN that handles sounding data.

    :param num_fields: Number of sounding fields (variables).
    :param num_heights: Number of sounding heights.
    :param do_separable_conv: See doc for `check_radar_options`.
    :param first_num_filters: Same.
    :param first_num_spatial_filters: Same.
    :param num_non_spatial_filters: Same.
    :return: first_num_filters: Same as input, except value may have been
        replaced with default.
    :return: num_non_spatial_filters: Same.
    """

    error_checking.assert_is_integer(num_fields)
    error_checking.assert_is_geq(num_fields, 1)
    error_checking.assert_is_integer(num_heights)

    if num_heights not in VALID_NUMBERS_OF_SOUNDING_HEIGHTS:
        error_string = (
            '\n{0:s}\nValid numbers of sounding heights (listed above) do not '
            'include {1:d}.'
        ).format(str(VALID_NUMBERS_OF_SOUNDING_HEIGHTS), num_heights)

        raise ValueError(error_string)

    error_checking.assert_is_boolean(do_separable_conv)
    if not do_separable_conv:
        if first_num_filters is None:
            first_num_filters = 8 * num_fields

        error_checking.assert_is_integer(first_num_filters)
        error_checking.assert_is_geq(first_num_filters, 4)

        return first_num_filters, None

    if num_non_spatial_filters is None:
        num_non_spatial_filters = num_fields + 0

    error_checking.assert_is_integer(first_num_spatial_filters)
    error_checking.assert_is_greater(first_num_spatial_filters, 0)
    error_checking.assert_is_integer(num_non_spatial_filters)
    error_checking.assert_is_greater(num_non_spatial_filters, 0)
    error_checking.assert_is_geq(
        first_num_spatial_filters * num_non_spatial_filters, 4)

    return None, num_non_spatial_filters


def create_2d_cnn(
        num_radar_rows, num_radar_columns, num_radar_channels, num_classes,
        num_sounding_fields, num_sounding_heights, use_padding=False,
        pooling_type_string=architecture_utils.MAX_POOLING_STRING,
        num_conv_layer_sets=DEFAULT_NUM_CONV_LAYER_SETS,
        num_conv_layers_per_set=DEFAULT_NUM_CONV_LAYERS_PER_SET,
        conv_layer_dropout_fraction=DEFAULT_CONV_LAYER_DROPOUT_FRACTION,
        num_dense_layers=DEFAULT_NUM_DENSE_LAYERS,
        dense_layer_dropout_fraction=DEFAULT_DENSE_LAYER_DROPOUT_FRACTION,
        l1_weight=DEFAULT_L1_WEIGHT, l2_weight=DEFAULT_L2_WEIGHT,
        activation_function_string=DEFAULT_ACTIVATION_FUNCTION_STRING,
        alpha_for_elu=architecture_utils.DEFAULT_ALPHA_FOR_ELU,
        alpha_for_relu=architecture_utils.DEFAULT_ALPHA_FOR_RELU,
        use_batch_normalization=DEFAULT_USE_BATCH_NORM_FLAG,
        list_of_metric_functions=DEFAULT_METRIC_FUNCTION_LIST,
        first_num_radar_filters=None, first_num_sounding_filters=None):
    """Creates CNN to take in 2-D radar data (and possibly soundings).

    This CNN does *not* use depthwise-separable convolution.

    :param num_radar_rows: See doc for `check_radar_options`.
    :param num_radar_columns: Same.
    :param num_radar_channels: Same.
    :param num_classes: Same.
    :param num_sounding_fields: See doc for `check_sounding_options`.
    :param num_sounding_heights: Same.
    :param use_padding: Boolean flag.  If True, will use edge-padding so that
        convolution does not change image size.  If False, will not use edge-
        padding.
    :param pooling_type_string: Pooling type (see doc for
        `architecture_utils._check_pooling_options`).
    :param num_conv_layer_sets: See doc for `check_radar_options`.
    :param num_conv_layers_per_set: Same.
    :param conv_layer_dropout_fraction: Same.
    :param num_dense_layers: Same.
    :param dense_layer_dropout_fraction: Same.
    :param l1_weight: Same.
    :param l2_weight: Same.
    :param activation_function_string: Same.
    :param alpha_for_elu: Same.
    :param alpha_for_relu: Same.
    :param use_batch_normalization: Same.
    :param first_num_radar_filters: Same.
    :param first_num_sounding_filters: See doc for `check_sounding_options`.
    :param list_of_metric_functions: List of Keras metrics.  Will be used to
        report performance during training.
    :return: cnn_model_object: Untrained instance of `keras.models.Model`.
    """

    error_checking.assert_is_boolean(use_padding)
    padding_type_string = (
        architecture_utils.YES_PADDING_STRING if use_padding
        else architecture_utils.NO_PADDING_STRING
    )

    error_checking.assert_is_integer(num_sounding_fields)
    error_checking.assert_is_integer(num_sounding_heights)
    include_soundings = num_sounding_fields > 0 or num_sounding_heights > 0

    first_num_radar_filters, _ = check_radar_options(
        num_grid_rows=num_radar_rows, num_grid_columns=num_radar_columns,
        num_dimensions=2, num_classes=num_classes,
        num_conv_layer_sets=num_conv_layer_sets,
        num_conv_layers_per_set=num_conv_layers_per_set,
        conv_layer_dropout_fraction=conv_layer_dropout_fraction,
        num_dense_layers=num_dense_layers,
        dense_layer_dropout_fraction=dense_layer_dropout_fraction,
        l1_weight=l1_weight, l2_weight=l2_weight,
        activation_function_string=activation_function_string,
        alpha_for_elu=alpha_for_elu, alpha_for_relu=alpha_for_relu,
        use_batch_normalization=use_batch_normalization,
        num_channels=num_radar_channels, do_separable_conv=False,
        first_num_filters=first_num_radar_filters)

    radar_input_layer_object = keras.layers.Input(
        shape=(num_radar_rows, num_radar_columns, num_radar_channels)
    )

    regularizer_object = architecture_utils.get_weight_regularizer(
        l1_weight=l1_weight, l2_weight=l2_weight)

    radar_layer_object = None
    current_num_filters = None

    for _ in range(num_conv_layer_sets):
        if current_num_filters is None:
            current_num_filters = first_num_radar_filters + 0
        else:
            current_num_filters *= 2

        for _ in range(num_conv_layers_per_set):
            if radar_layer_object is None:
                this_input_layer_object = radar_input_layer_object
                # current_num_filters = first_num_radar_filters + 0
            else:
                this_input_layer_object = radar_layer_object
                # current_num_filters *= 2

            radar_layer_object = architecture_utils.get_2d_conv_layer(
                num_filters=current_num_filters,
                num_kernel_rows=NUM_CONV_KERNEL_ROWS,
                num_kernel_columns=NUM_CONV_KERNEL_COLUMNS,
                num_rows_per_stride=1, num_columns_per_stride=1,
                padding_type_string=padding_type_string,
                weight_regularizer=regularizer_object
            )(this_input_layer_object)

            radar_layer_object = architecture_utils.get_activation_layer(
                activation_function_string=activation_function_string,
                alpha_for_elu=alpha_for_elu, alpha_for_relu=alpha_for_relu
            )(radar_layer_object)

            if conv_layer_dropout_fraction is not None:
                radar_layer_object = architecture_utils.get_dropout_layer(
                    dropout_fraction=conv_layer_dropout_fraction
                )(radar_layer_object)

            if use_batch_normalization:
                radar_layer_object = (
                    architecture_utils.get_batch_norm_layer()(
                        radar_layer_object)
                )

        radar_layer_object = architecture_utils.get_2d_pooling_layer(
            num_rows_in_window=2, num_columns_in_window=2,
            num_rows_per_stride=2, num_columns_per_stride=2,
            pooling_type_string=pooling_type_string
        )(radar_layer_object)

    these_dimensions = numpy.array(
        radar_layer_object.get_shape().as_list()[1:], dtype=int)
    num_radar_features = numpy.prod(these_dimensions)

    radar_layer_object = architecture_utils.get_flattening_layer()(
        radar_layer_object)

    if include_soundings:
        (sounding_input_layer_object, sounding_layer_object,
         num_sounding_features
        ) = _create_sounding_layers(
            num_fields=num_sounding_fields, num_heights=num_sounding_heights,
            num_conv_layer_sets=num_conv_layer_sets,
            num_conv_layers_per_set=num_conv_layers_per_set,
            padding_type_string=padding_type_string, l1_weight=l1_weight,
            l2_weight=l2_weight, pooling_type_string=pooling_type_string,
            activation_function_string=activation_function_string,
            alpha_for_elu=alpha_for_elu, alpha_for_relu=alpha_for_relu,
            use_batch_normalization=use_batch_normalization,
            dropout_fraction=conv_layer_dropout_fraction,
            do_separable_conv=False,
            first_num_filters=first_num_sounding_filters)

        layer_object = keras.layers.concatenate(
            [radar_layer_object, sounding_layer_object]
        )

    else:
        layer_object = radar_layer_object
        num_sounding_features = 0

    layer_object = _create_dense_layers(
        flattening_layer_object=layer_object,
        num_scalar_features=num_radar_features + num_sounding_features,
        num_classes=num_classes, num_dense_layers=num_dense_layers,
        dropout_fraction=dense_layer_dropout_fraction,
        activation_function_string=activation_function_string,
        alpha_for_elu=alpha_for_elu, alpha_for_relu=alpha_for_relu,
        use_batch_normalization=use_batch_normalization)

    if include_soundings:
        model_object = keras.models.Model(
            inputs=[radar_input_layer_object, sounding_input_layer_object],
            outputs=layer_object)
    else:
        model_object = keras.models.Model(
            inputs=radar_input_layer_object, outputs=layer_object)

    loss_function = _get_output_layer_and_loss(num_classes)[-1]
    model_object.compile(
        loss=loss_function, optimizer=keras.optimizers.Adam(),
        metrics=list_of_metric_functions)

    model_object.summary()
    return model_object


def create_cnn_soundings_only(
        num_fields, num_heights, first_num_filters, num_classes,
        use_padding=False,
        pooling_type_string=architecture_utils.MAX_POOLING_STRING,
        num_conv_layer_sets=DEFAULT_NUM_CONV_LAYER_SETS,
        num_conv_layers_per_set=DEFAULT_NUM_CONV_LAYERS_PER_SET,
        conv_layer_dropout_fraction=DEFAULT_CONV_LAYER_DROPOUT_FRACTION,
        num_dense_layers=DEFAULT_NUM_DENSE_LAYERS,
        dense_layer_dropout_fraction=DEFAULT_DENSE_LAYER_DROPOUT_FRACTION,
        l1_weight=DEFAULT_L1_WEIGHT, l2_weight=DEFAULT_L2_WEIGHT,
        activation_function_string=DEFAULT_ACTIVATION_FUNCTION_STRING,
        alpha_for_elu=architecture_utils.DEFAULT_ALPHA_FOR_ELU,
        alpha_for_relu=architecture_utils.DEFAULT_ALPHA_FOR_RELU,
        use_batch_normalization=DEFAULT_USE_BATCH_NORM_FLAG,
        list_of_metric_functions=DEFAULT_METRIC_FUNCTION_LIST):
    """Creates CNN that convolves over soundings only.

    :param num_fields: Number of sounding fields.
    :param num_heights: Number of sounding heights.
    :param first_num_filters: Number of filters produced by first conv layer.
        Will double with each successive conv layer.
    :param num_classes: See doc for `create_2d_cnn`.
    :param use_padding: Same.
    :param pooling_type_string: Same.
    :param num_conv_layer_sets: Same.
    :param num_conv_layers_per_set: Same.
    :param conv_layer_dropout_fraction: Same.
    :param num_dense_layers: Same.
    :param dense_layer_dropout_fraction: Same.
    :param l1_weight: Same.
    :param l2_weight: Same.
    :param activation_function_string: Same.
    :param alpha_for_elu: Same.
    :param alpha_for_relu: Same.
    :param use_batch_normalization: Same.
    :param list_of_metric_functions: Same.
    :return: cnn_model_object: Untrained instance of `keras.models.Model`.
    """

    error_checking.assert_is_boolean(use_padding)
    padding_type_string = (
        architecture_utils.YES_PADDING_STRING if use_padding
        else architecture_utils.NO_PADDING_STRING
    )

    check_radar_options(
        num_grid_rows=32, num_grid_columns=32, num_dimensions=2,
        num_channels=32, first_num_filters=32, num_classes=num_classes,
        num_conv_layer_sets=num_conv_layer_sets,
        num_conv_layers_per_set=num_conv_layers_per_set,
        conv_layer_dropout_fraction=conv_layer_dropout_fraction,
        num_dense_layers=num_dense_layers,
        dense_layer_dropout_fraction=dense_layer_dropout_fraction,
        l1_weight=l1_weight, l2_weight=l2_weight,
        activation_function_string=activation_function_string,
        alpha_for_elu=alpha_for_elu, alpha_for_relu=alpha_for_relu,
        use_batch_normalization=use_batch_normalization,
        do_separable_conv=False)

    check_sounding_options(
        num_fields=num_fields, num_heights=num_heights,
        first_num_filters=first_num_filters, do_separable_conv=False)

    input_layer_object, layer_object, num_features = _create_sounding_layers(
        num_fields=num_fields, num_heights=num_heights,
        first_num_filters=first_num_filters,
        num_conv_layer_sets=num_conv_layer_sets,
        num_conv_layers_per_set=num_conv_layers_per_set,
        padding_type_string=padding_type_string, l1_weight=l1_weight,
        l2_weight=l2_weight, pooling_type_string=pooling_type_string,
        activation_function_string=activation_function_string,
        alpha_for_elu=alpha_for_elu, alpha_for_relu=alpha_for_relu,
        use_batch_normalization=use_batch_normalization,
        dropout_fraction=conv_layer_dropout_fraction, do_separable_conv=False)

    layer_object = _create_dense_layers(
        flattening_layer_object=layer_object, num_scalar_features=num_features,
        num_classes=num_classes, num_dense_layers=num_dense_layers,
        dropout_fraction=dense_layer_dropout_fraction,
        activation_function_string=activation_function_string,
        alpha_for_elu=alpha_for_elu, alpha_for_relu=alpha_for_relu,
        use_batch_normalization=use_batch_normalization)

    model_object = keras.models.Model(
        inputs=input_layer_object, outputs=layer_object)

    loss_function = _get_output_layer_and_loss(num_classes)[-1]
    model_object.compile(
        loss=loss_function, optimizer=keras.optimizers.Adam(),
        metrics=list_of_metric_functions)

    model_object.summary()
    return model_object


def create_separable_2d_cnn(
        num_radar_rows, num_radar_columns, num_radar_channels, num_classes,
        num_sounding_fields, num_sounding_heights, use_padding=False,
        pooling_type_string=architecture_utils.MAX_POOLING_STRING,
        num_conv_layer_sets=DEFAULT_NUM_CONV_LAYER_SETS,
        num_conv_layers_per_set=DEFAULT_NUM_CONV_LAYERS_PER_SET,
        conv_layer_dropout_fraction=DEFAULT_CONV_LAYER_DROPOUT_FRACTION,
        num_dense_layers=DEFAULT_NUM_DENSE_LAYERS,
        dense_layer_dropout_fraction=DEFAULT_DENSE_LAYER_DROPOUT_FRACTION,
        l1_weight=DEFAULT_L1_WEIGHT, l2_weight=DEFAULT_L2_WEIGHT,
        activation_function_string=DEFAULT_ACTIVATION_FUNCTION_STRING,
        alpha_for_elu=architecture_utils.DEFAULT_ALPHA_FOR_ELU,
        alpha_for_relu=architecture_utils.DEFAULT_ALPHA_FOR_RELU,
        use_batch_normalization=DEFAULT_USE_BATCH_NORM_FLAG,
        list_of_metric_functions=DEFAULT_METRIC_FUNCTION_LIST,
        first_num_spatial_radar_filters=None,
        num_non_spatial_radar_filters=None,
        first_num_spatial_sounding_filters=None,
        num_non_spatial_sounding_filters=None):
    """Creates CNN to take in 2-D radar data (and possibly soundings).

    This CNN does depthwise-separable convolution.

    :param num_radar_rows: See doc for `check_radar_options`.
    :param num_radar_columns: Same.
    :param num_radar_channels: Same.
    :param num_classes: Same.
    :param num_sounding_fields: See doc for `check_sounding_options`.
    :param num_sounding_heights: Same.
    :param use_padding: See doc for `create_2d_cnn`.
    :param pooling_type_string: Pooling type (see doc for
        `architecture_utils._check_pooling_options`).
    :param num_conv_layer_sets: See doc for `check_radar_options`.
    :param num_conv_layers_per_set: Same.
    :param conv_layer_dropout_fraction: Same.
    :param num_dense_layers: Same.
    :param dense_layer_dropout_fraction: Same.
    :param l1_weight: Same.
    :param l2_weight: Same.
    :param activation_function_string: Same.
    :param alpha_for_elu: Same.
    :param alpha_for_relu: Same.
    :param use_batch_normalization: Same.
    :param first_num_spatial_radar_filters: Same.
    :param num_non_spatial_radar_filters: Same.
    :param first_num_spatial_sounding_filters: See doc for
        `check_sounding_options`.
    :param num_non_spatial_sounding_filters: Same.
    :param list_of_metric_functions: List of Keras metrics.  Will be used to
        report performance during training.
    :return: cnn_model_object: Untrained instance of `keras.models.Model`.
    """

    error_checking.assert_is_boolean(use_padding)
    padding_type_string = (
        architecture_utils.YES_PADDING_STRING if use_padding
        else architecture_utils.NO_PADDING_STRING
    )

    error_checking.assert_is_integer(num_sounding_fields)
    error_checking.assert_is_integer(num_sounding_heights)
    include_soundings = num_sounding_fields > 0 or num_sounding_heights > 0

    _, num_non_spatial_radar_filters = check_radar_options(
        num_grid_rows=num_radar_rows, num_grid_columns=num_radar_columns,
        num_dimensions=2, num_classes=num_classes,
        num_conv_layer_sets=num_conv_layer_sets,
        num_conv_layers_per_set=num_conv_layers_per_set,
        conv_layer_dropout_fraction=conv_layer_dropout_fraction,
        num_dense_layers=num_dense_layers,
        dense_layer_dropout_fraction=dense_layer_dropout_fraction,
        l1_weight=l1_weight, l2_weight=l2_weight,
        activation_function_string=activation_function_string,
        alpha_for_elu=alpha_for_elu, alpha_for_relu=alpha_for_relu,
        use_batch_normalization=use_batch_normalization,
        num_channels=num_radar_channels, do_separable_conv=True,
        first_num_spatial_filters=first_num_spatial_radar_filters,
        num_non_spatial_filters=num_non_spatial_radar_filters)

    radar_input_layer_object = keras.layers.Input(
        shape=(num_radar_rows, num_radar_columns, num_radar_channels)
    )

    regularizer_object = architecture_utils.get_weight_regularizer(
        l1_weight=l1_weight, l2_weight=l2_weight)

    radar_layer_object = None
    current_num_spatial_filters = None

    for _ in range(num_conv_layer_sets):
        for _ in range(num_conv_layers_per_set):
            if radar_layer_object is None:
                this_input_layer_object = radar_input_layer_object
                current_num_spatial_filters = (
                    first_num_spatial_radar_filters + 0)
            else:
                this_input_layer_object = radar_layer_object
                current_num_spatial_filters *= 2

            radar_layer_object = architecture_utils.get_2d_separable_conv_layer(
                num_spatial_filters=current_num_spatial_filters,
                num_kernel_rows=NUM_CONV_KERNEL_ROWS,
                num_kernel_columns=NUM_CONV_KERNEL_COLUMNS,
                num_rows_per_stride=1, num_columns_per_stride=1,
                num_non_spatial_filters=num_non_spatial_radar_filters,
                padding_type_string=padding_type_string,
                weight_regularizer=regularizer_object
            )(this_input_layer_object)

            radar_layer_object = architecture_utils.get_activation_layer(
                activation_function_string=activation_function_string,
                alpha_for_elu=alpha_for_elu, alpha_for_relu=alpha_for_relu
            )(radar_layer_object)

            if conv_layer_dropout_fraction is not None:
                radar_layer_object = architecture_utils.get_dropout_layer(
                    dropout_fraction=conv_layer_dropout_fraction
                )(radar_layer_object)

            if use_batch_normalization:
                radar_layer_object = (
                    architecture_utils.get_batch_norm_layer()(
                        radar_layer_object)
                )

        radar_layer_object = architecture_utils.get_2d_pooling_layer(
            num_rows_in_window=2, num_columns_in_window=2,
            num_rows_per_stride=2, num_columns_per_stride=2,
            pooling_type_string=pooling_type_string
        )(radar_layer_object)

    these_dimensions = numpy.array(
        radar_layer_object.get_shape().as_list()[1:], dtype=int)
    num_radar_features = numpy.prod(these_dimensions)

    radar_layer_object = architecture_utils.get_flattening_layer()(
        radar_layer_object)

    if include_soundings:
        (sounding_input_layer_object, sounding_layer_object,
         num_sounding_features
        ) = _create_sounding_layers(
            num_fields=num_sounding_fields, num_heights=num_sounding_heights,
            num_conv_layer_sets=num_conv_layer_sets,
            num_conv_layers_per_set=num_conv_layers_per_set,
            padding_type_string=padding_type_string, l1_weight=l1_weight,
            l2_weight=l2_weight, pooling_type_string=pooling_type_string,
            activation_function_string=activation_function_string,
            alpha_for_elu=alpha_for_elu, alpha_for_relu=alpha_for_relu,
            use_batch_normalization=use_batch_normalization,
            dropout_fraction=conv_layer_dropout_fraction,
            do_separable_conv=True,
            first_num_spatial_filters=first_num_spatial_sounding_filters,
            num_non_spatial_filters=num_non_spatial_sounding_filters)

        layer_object = keras.layers.concatenate(
            [radar_layer_object, sounding_layer_object]
        )

    else:
        layer_object = radar_layer_object
        num_sounding_features = 0

    layer_object = _create_dense_layers(
        flattening_layer_object=layer_object,
        num_scalar_features=num_radar_features + num_sounding_features,
        num_classes=num_classes, num_dense_layers=num_dense_layers,
        dropout_fraction=dense_layer_dropout_fraction,
        activation_function_string=activation_function_string,
        alpha_for_elu=alpha_for_elu, alpha_for_relu=alpha_for_relu,
        use_batch_normalization=use_batch_normalization)

    if include_soundings:
        model_object = keras.models.Model(
            inputs=[radar_input_layer_object, sounding_input_layer_object],
            outputs=layer_object)
    else:
        model_object = keras.models.Model(
            inputs=radar_input_layer_object, outputs=layer_object)

    loss_function = _get_output_layer_and_loss(num_classes)[-1]
    model_object.compile(
        loss=loss_function, optimizer=keras.optimizers.Adam(),
        metrics=list_of_metric_functions)

    model_object.summary()
    return model_object


def create_2d3d_cnn(
        num_refl_rows, num_refl_columns, num_refl_heights, num_az_shear_fields,
        num_classes, num_sounding_fields, num_sounding_heights,
        use_padding=False,
        pooling_type_string=architecture_utils.MAX_POOLING_STRING,
        num_conv_layer_sets=DEFAULT_NUM_CONV_LAYER_SETS,
        num_conv_layers_per_set=DEFAULT_NUM_CONV_LAYERS_PER_SET,
        first_num_refl_filters=None, first_num_shear_filters=None,
        first_num_sounding_filters=None,
        conv_layer_dropout_fraction=DEFAULT_CONV_LAYER_DROPOUT_FRACTION,
        num_dense_layers=DEFAULT_NUM_DENSE_LAYERS,
        dense_layer_dropout_fraction=DEFAULT_DENSE_LAYER_DROPOUT_FRACTION,
        l1_weight=DEFAULT_L1_WEIGHT, l2_weight=DEFAULT_L2_WEIGHT,
        activation_function_string=DEFAULT_ACTIVATION_FUNCTION_STRING,
        alpha_for_elu=architecture_utils.DEFAULT_ALPHA_FOR_ELU,
        alpha_for_relu=architecture_utils.DEFAULT_ALPHA_FOR_RELU,
        use_batch_normalization=DEFAULT_USE_BATCH_NORM_FLAG,
        list_of_metric_functions=DEFAULT_METRIC_FUNCTION_LIST):
    """Creates CNN to take in 2-D and 3-D radar data (and possibly soundings).

    In this case, 3-D radar fields contain reflectivity and 2-D fields contain
    azimuthal shear.

    :param num_refl_rows: Number of rows in reflectivity grid.
    :param num_refl_columns: Number of columns in reflectivity grid.
    :param num_refl_heights: Number of heights in reflectivity grid.
    :param num_az_shear_fields: Number of azimuthal-shear fields.
    See doc for `check_radar_options`.
    :param num_sounding_fields: See doc for `check_sounding_options`.
    :param num_sounding_heights: Same.
    :param use_padding: See doc for `create_2d_cnn`.
    :param pooling_type_string: Pooling type (see doc for
        `architecture_utils._check_pooling_options`).
    :param num_conv_layer_sets: See doc for `check_radar_options`.
    :param num_conv_layers_per_set: Same.
    :param first_num_refl_filters: Number of filters created by first
        convolution layer for reflectivity data.  This will double with
        ach successive conv layer.
    :param first_num_shear_filters: Same but for azimuthal shear.
    :param first_num_sounding_filters: Same but for sounding.
    :param conv_layer_dropout_fraction: See doc for `check_radar_options`.
    :param num_dense_layers: Same.
    :param dense_layer_dropout_fraction: Same.
    :param l1_weight: Same.
    :param l2_weight: Same.
    :param activation_function_string: Same.
    :param alpha_for_elu: Same.
    :param alpha_for_relu: Same.
    :param use_batch_normalization: Same.
    :param list_of_metric_functions: List of Keras metrics.  Will be used to
        report performance during training.
    :return: cnn_model_object: Untrained instance of `keras.models.Model`.
    """

    error_checking.assert_is_boolean(use_padding)
    padding_type_string = (
        architecture_utils.YES_PADDING_STRING if use_padding
        else architecture_utils.NO_PADDING_STRING
    )

    error_checking.assert_is_integer(num_sounding_fields)
    error_checking.assert_is_integer(num_sounding_heights)
    include_soundings = num_sounding_fields > 0 or num_sounding_heights > 0

    first_num_refl_filters, _ = check_radar_options(
        num_grid_rows=num_refl_rows, num_grid_columns=num_refl_columns,
        num_heights=num_refl_heights, num_fields=1, num_dimensions=3,
        num_classes=num_classes, num_conv_layer_sets=num_conv_layer_sets,
        num_conv_layers_per_set=num_conv_layers_per_set,
        conv_layer_dropout_fraction=conv_layer_dropout_fraction,
        num_dense_layers=num_dense_layers,
        dense_layer_dropout_fraction=dense_layer_dropout_fraction,
        l1_weight=l1_weight, l2_weight=l2_weight,
        activation_function_string=activation_function_string,
        alpha_for_elu=alpha_for_elu, alpha_for_relu=alpha_for_relu,
        use_batch_normalization=use_batch_normalization,
        do_separable_conv=False, first_num_filters=first_num_refl_filters)

    first_num_shear_filters, _ = check_radar_options(
        num_grid_rows=2 * num_refl_rows, num_grid_columns=2 * num_refl_columns,
        num_channels=num_az_shear_fields, num_dimensions=2,
        num_classes=num_classes, num_conv_layer_sets=num_conv_layer_sets,
        num_conv_layers_per_set=num_conv_layers_per_set,
        conv_layer_dropout_fraction=conv_layer_dropout_fraction,
        num_dense_layers=num_dense_layers,
        dense_layer_dropout_fraction=dense_layer_dropout_fraction,
        l1_weight=l1_weight, l2_weight=l2_weight,
        activation_function_string=activation_function_string,
        alpha_for_elu=alpha_for_elu, alpha_for_relu=alpha_for_relu,
        use_batch_normalization=use_batch_normalization,
        do_separable_conv=False, first_num_filters=first_num_shear_filters)

    shear_input_layer_object = keras.layers.Input(
        shape=(2 * num_refl_rows, 2 * num_refl_columns, num_az_shear_fields)
    )

    regularizer_object = architecture_utils.get_weight_regularizer(
        l1_weight=l1_weight, l2_weight=l2_weight)

    shear_layer_object = None

    for _ in range(num_conv_layers_per_set):
        if shear_layer_object is None:
            this_input_layer_object = shear_input_layer_object
        else:
            this_input_layer_object = shear_layer_object

        shear_layer_object = architecture_utils.get_2d_conv_layer(
            num_filters=first_num_shear_filters,
            num_kernel_rows=NUM_CONV_KERNEL_ROWS,
            num_kernel_columns=NUM_CONV_KERNEL_COLUMNS,
            num_rows_per_stride=1, num_columns_per_stride=1,
            padding_type_string=architecture_utils.YES_PADDING_STRING,
            weight_regularizer=regularizer_object
        )(this_input_layer_object)

        shear_layer_object = architecture_utils.get_activation_layer(
            activation_function_string=activation_function_string,
            alpha_for_elu=alpha_for_elu, alpha_for_relu=alpha_for_relu
        )(shear_layer_object)

        if conv_layer_dropout_fraction is not None:
            shear_layer_object = architecture_utils.get_dropout_layer(
                dropout_fraction=conv_layer_dropout_fraction
            )(shear_layer_object)

        if use_batch_normalization:
            shear_layer_object = architecture_utils.get_batch_norm_layer()(
                shear_layer_object
            )

    shear_layer_object = architecture_utils.get_2d_pooling_layer(
        num_rows_in_window=2, num_columns_in_window=2,
        num_rows_per_stride=2, num_columns_per_stride=2,
        pooling_type_string=pooling_type_string
    )(shear_layer_object)

    refl_input_layer_object = keras.layers.Input(
        shape=(num_refl_rows, num_refl_columns, num_refl_heights, 1)
    )

    refl_layer_object = keras.layers.Reshape(
        target_shape=(num_refl_rows, num_refl_columns, num_refl_heights)
    )(refl_input_layer_object)

    radar_layer_object = keras.layers.concatenate(
        [refl_layer_object, shear_layer_object], axis=-1
    )

    current_num_filters = None

    for _ in range(num_conv_layer_sets):
        if current_num_filters is None:
            current_num_filters = (
                first_num_shear_filters + first_num_refl_filters
            )
        else:
            current_num_filters *= 2

        for _ in range(num_conv_layers_per_set):
            this_input_layer_object = radar_layer_object

            radar_layer_object = architecture_utils.get_2d_conv_layer(
                num_filters=current_num_filters,
                num_kernel_rows=NUM_CONV_KERNEL_ROWS,
                num_kernel_columns=NUM_CONV_KERNEL_COLUMNS,
                num_rows_per_stride=1, num_columns_per_stride=1,
                padding_type_string=padding_type_string,
                weight_regularizer=regularizer_object
            )(this_input_layer_object)

            radar_layer_object = architecture_utils.get_activation_layer(
                activation_function_string=activation_function_string,
                alpha_for_elu=alpha_for_elu, alpha_for_relu=alpha_for_relu
            )(radar_layer_object)

            if conv_layer_dropout_fraction is not None:
                radar_layer_object = architecture_utils.get_dropout_layer(
                    dropout_fraction=conv_layer_dropout_fraction
                )(radar_layer_object)

            if use_batch_normalization:
                radar_layer_object = (
                    architecture_utils.get_batch_norm_layer()(
                        radar_layer_object)
                )

        radar_layer_object = architecture_utils.get_2d_pooling_layer(
            num_rows_in_window=2, num_columns_in_window=2,
            num_rows_per_stride=2, num_columns_per_stride=2,
            pooling_type_string=pooling_type_string
        )(radar_layer_object)

    these_dimensions = numpy.array(
        radar_layer_object.get_shape().as_list()[1:], dtype=int
    )
    num_radar_features = numpy.prod(these_dimensions)

    radar_layer_object = architecture_utils.get_flattening_layer()(
        radar_layer_object)

    if include_soundings:
        (sounding_input_layer_object, sounding_layer_object,
         num_sounding_features
        ) = _create_sounding_layers(
            num_fields=num_sounding_fields, num_heights=num_sounding_heights,
            num_conv_layer_sets=num_conv_layer_sets,
            num_conv_layers_per_set=num_conv_layers_per_set,
            padding_type_string=padding_type_string, l1_weight=l1_weight,
            l2_weight=l2_weight, pooling_type_string=pooling_type_string,
            activation_function_string=activation_function_string,
            alpha_for_elu=alpha_for_elu, alpha_for_relu=alpha_for_relu,
            use_batch_normalization=use_batch_normalization,
            dropout_fraction=conv_layer_dropout_fraction,
            do_separable_conv=False,
            first_num_filters=first_num_sounding_filters)

        layer_object = keras.layers.concatenate(
            [radar_layer_object, sounding_layer_object]
        )

    else:
        layer_object = radar_layer_object
        num_sounding_features = 0

    layer_object = _create_dense_layers(
        flattening_layer_object=layer_object,
        num_scalar_features=num_radar_features + num_sounding_features,
        num_classes=num_classes, num_dense_layers=num_dense_layers,
        dropout_fraction=dense_layer_dropout_fraction,
        activation_function_string=activation_function_string,
        alpha_for_elu=alpha_for_elu, alpha_for_relu=alpha_for_relu,
        use_batch_normalization=use_batch_normalization)

    input_layer_objects = [refl_input_layer_object, shear_input_layer_object]
    if include_soundings:
        input_layer_objects.append(sounding_input_layer_object)

    model_object = keras.models.Model(
        inputs=input_layer_objects, outputs=layer_object)

    loss_function = _get_output_layer_and_loss(num_classes)[-1]
    model_object.compile(
        loss=loss_function, optimizer=keras.optimizers.Adam(),
        metrics=list_of_metric_functions)

    model_object.summary()
    return model_object


def create_3d_cnn(
        num_radar_rows, num_radar_columns, num_radar_heights, num_radar_fields,
        num_classes, num_sounding_fields, num_sounding_heights,
        use_padding=False,
        pooling_type_string=architecture_utils.MAX_POOLING_STRING,
        num_conv_layer_sets=DEFAULT_NUM_CONV_LAYER_SETS,
        num_conv_layers_per_set=DEFAULT_NUM_CONV_LAYERS_PER_SET,
        first_num_radar_filters=None, first_num_sounding_filters=None,
        conv_layer_dropout_fraction=DEFAULT_CONV_LAYER_DROPOUT_FRACTION,
        num_dense_layers=DEFAULT_NUM_DENSE_LAYERS,
        dense_layer_dropout_fraction=DEFAULT_DENSE_LAYER_DROPOUT_FRACTION,
        l1_weight=DEFAULT_L1_WEIGHT, l2_weight=DEFAULT_L2_WEIGHT,
        activation_function_string=DEFAULT_ACTIVATION_FUNCTION_STRING,
        alpha_for_elu=architecture_utils.DEFAULT_ALPHA_FOR_ELU,
        alpha_for_relu=architecture_utils.DEFAULT_ALPHA_FOR_RELU,
        use_batch_normalization=DEFAULT_USE_BATCH_NORM_FLAG,
        list_of_metric_functions=DEFAULT_METRIC_FUNCTION_LIST):
    """Creates CNN to take in 3-D radar data (and possibly soundings).

    :param num_radar_rows: See doc for `check_radar_options`.
    :param num_radar_columns: Same.
    :param num_radar_heights: Same.
    :param num_radar_fields: Same.
    :param num_classes: Same.
    :param num_sounding_fields: See doc for `check_sounding_options`.
    :param num_sounding_heights: Same.
    :param use_padding: See doc for `create_2d_cnn`.
    :param pooling_type_string: Pooling type (see doc for
        `architecture_utils._check_pooling_options`).
    :param num_conv_layer_sets: See doc for `check_radar_options`.
    :param num_conv_layers_per_set: Same.
    :param first_num_radar_filters: Same.
    :param first_num_sounding_filters: See doc for `check_sounding_options`.
    :param conv_layer_dropout_fraction: See doc for `check_radar_options`.
    :param num_dense_layers: Same.
    :param dense_layer_dropout_fraction: Same.
    :param l1_weight: Same.
    :param l2_weight: Same.
    :param activation_function_string: Same.
    :param alpha_for_elu: Same.
    :param alpha_for_relu: Same.
    :param use_batch_normalization: Same.
    :param list_of_metric_functions: List of Keras metrics.  Will be used to
        report performance during training.
    :return: cnn_model_object: Untrained instance of `keras.models.Model`.
    """

    error_checking.assert_is_boolean(use_padding)
    padding_type_string = (
        architecture_utils.YES_PADDING_STRING if use_padding
        else architecture_utils.NO_PADDING_STRING
    )

    error_checking.assert_is_integer(num_sounding_fields)
    error_checking.assert_is_integer(num_sounding_heights)
    include_soundings = num_sounding_fields > 0 or num_sounding_heights > 0

    first_num_radar_filters, _ = check_radar_options(
        num_grid_rows=num_radar_rows, num_grid_columns=num_radar_columns,
        num_dimensions=3, num_classes=num_classes,
        num_conv_layer_sets=num_conv_layer_sets,
        num_conv_layers_per_set=num_conv_layers_per_set,
        conv_layer_dropout_fraction=conv_layer_dropout_fraction,
        num_dense_layers=num_dense_layers,
        dense_layer_dropout_fraction=dense_layer_dropout_fraction,
        l1_weight=l1_weight, l2_weight=l2_weight,
        activation_function_string=activation_function_string,
        alpha_for_elu=alpha_for_elu, alpha_for_relu=alpha_for_relu,
        use_batch_normalization=use_batch_normalization,
        num_heights=num_radar_heights, num_fields=num_radar_fields,
        do_separable_conv=False, first_num_filters=first_num_radar_filters)

    radar_input_layer_object = keras.layers.Input(
        shape=(num_radar_rows, num_radar_columns, num_radar_heights,
               num_radar_fields)
    )

    regularizer_object = architecture_utils.get_weight_regularizer(
        l1_weight=l1_weight, l2_weight=l2_weight)

    radar_layer_object = None
    current_num_filters = None

    for _ in range(num_conv_layer_sets):
        for _ in range(num_conv_layers_per_set):
            if current_num_filters is None:
                current_num_filters = first_num_radar_filters + 0
                this_input_layer_object = radar_input_layer_object
            else:
                current_num_filters *= 2
                this_input_layer_object = radar_layer_object

            radar_layer_object = architecture_utils.get_3d_conv_layer(
                num_filters=current_num_filters,
                num_kernel_rows=NUM_CONV_KERNEL_ROWS,
                num_kernel_columns=NUM_CONV_KERNEL_COLUMNS,
                num_kernel_heights=NUM_CONV_KERNEL_HEIGHTS,
                num_rows_per_stride=1, num_columns_per_stride=1,
                num_heights_per_stride=1,
                padding_type_string=padding_type_string,
                weight_regularizer=regularizer_object
            )(this_input_layer_object)

            radar_layer_object = architecture_utils.get_activation_layer(
                activation_function_string=activation_function_string,
                alpha_for_elu=alpha_for_elu, alpha_for_relu=alpha_for_relu
            )(radar_layer_object)

            if conv_layer_dropout_fraction is not None:
                radar_layer_object = architecture_utils.get_dropout_layer(
                    dropout_fraction=conv_layer_dropout_fraction
                )(radar_layer_object)

            if use_batch_normalization:
                radar_layer_object = (
                    architecture_utils.get_batch_norm_layer()(
                        radar_layer_object)
                )

        radar_layer_object = architecture_utils.get_3d_pooling_layer(
            num_rows_in_window=2, num_columns_in_window=2,
            num_heights_in_window=2, num_rows_per_stride=2,
            num_columns_per_stride=2, num_heights_per_stride=2,
            pooling_type_string=pooling_type_string
        )(radar_layer_object)

    these_dimensions = numpy.array(
        radar_layer_object.get_shape().as_list()[1:], dtype=int)
    num_radar_features = numpy.prod(these_dimensions)

    radar_layer_object = architecture_utils.get_flattening_layer()(
        radar_layer_object)

    if include_soundings:
        (sounding_input_layer_object, sounding_layer_object,
         num_sounding_features
        ) = _create_sounding_layers(
            num_fields=num_sounding_fields, num_heights=num_sounding_heights,
            num_conv_layer_sets=num_conv_layer_sets,
            num_conv_layers_per_set=num_conv_layers_per_set,
            padding_type_string=padding_type_string, l1_weight=l1_weight,
            l2_weight=l2_weight, pooling_type_string=pooling_type_string,
            activation_function_string=activation_function_string,
            alpha_for_elu=alpha_for_elu, alpha_for_relu=alpha_for_relu,
            use_batch_normalization=use_batch_normalization,
            dropout_fraction=conv_layer_dropout_fraction,
            do_separable_conv=False,
            first_num_filters=first_num_sounding_filters)

        layer_object = keras.layers.concatenate(
            [radar_layer_object, sounding_layer_object]
        )

    else:
        layer_object = radar_layer_object
        num_sounding_features = 0

    layer_object = _create_dense_layers(
        flattening_layer_object=layer_object,
        num_scalar_features=num_radar_features + num_sounding_features,
        num_classes=num_classes, num_dense_layers=num_dense_layers,
        dropout_fraction=dense_layer_dropout_fraction,
        activation_function_string=activation_function_string,
        alpha_for_elu=alpha_for_elu, alpha_for_relu=alpha_for_relu,
        use_batch_normalization=use_batch_normalization)

    if include_soundings:
        model_object = keras.models.Model(
            inputs=[radar_input_layer_object, sounding_input_layer_object],
            outputs=layer_object)
    else:
        model_object = keras.models.Model(
            inputs=radar_input_layer_object, outputs=layer_object)

    loss_function = _get_output_layer_and_loss(num_classes)[-1]
    model_object.compile(
        loss=loss_function, optimizer=keras.optimizers.Adam(),
        metrics=list_of_metric_functions)

    model_object.summary()
    return model_object


def _create_sounding_layers_resnet(
        num_fields, num_heights, num_conv_layer_sets,
        num_conv_layers_per_set, padding_type_string, l1_weight, l2_weight,
        pooling_type_string, activation_function_string, alpha_for_elu,
        alpha_for_relu, use_batch_normalization, dropout_fraction,
        first_num_filters):
    """Creates conv and pooling layers, with residual blocks, for sounding data.

    :param num_fields: See doc for `_create_sounding_layers`.
    :param num_heights: Same.
    :param num_conv_layer_sets: Same.
    :param num_conv_layers_per_set: Same.
    :param padding_type_string: Same.
    :param l1_weight: Same.
    :param l2_weight: Same.
    :param pooling_type_string: Same.
    :param activation_function_string: Same.
    :param alpha_for_elu: Same.
    :param alpha_for_relu: Same.
    :param use_batch_normalization: Same.
    :param dropout_fraction: Same.
    :param first_num_filters: Same.
    :return: input_layer_object: Same.
    :return: flattening_layer_object: Same.
    :return: num_scalar_features: Same.
    """

    # TODO(thunderhoser): This is my first attempt at res-nets.  I don't know
    # what the best architecture is, especially vis-a-vis where activation and
    # batch-norm layers should go.

    first_num_filters, _ = check_sounding_options(
        num_fields=num_fields, num_heights=num_heights, do_separable_conv=False,
        first_num_filters=first_num_filters)

    input_layer_object = keras.layers.Input(
        shape=(num_heights, num_fields)
    )

    regularizer_object = architecture_utils.get_weight_regularizer(
        l1_weight=l1_weight, l2_weight=l2_weight)

    current_layer_object = None
    current_num_filters = None

    for _ in range(num_conv_layer_sets):
        if current_layer_object is None:
            this_input_layer_object = input_layer_object
            current_num_filters = first_num_filters + 0
        else:
            this_input_layer_object = current_layer_object
            current_num_filters *= 2

        current_layer_object = architecture_utils.get_1d_conv_layer(
            num_kernel_rows=NUM_CONV_KERNEL_HEIGHTS,
            num_rows_per_stride=1, num_filters=current_num_filters,
            padding_type_string=padding_type_string,
            weight_regularizer=regularizer_object
        )(this_input_layer_object)

        current_layer_object = architecture_utils.get_activation_layer(
            activation_function_string=activation_function_string,
            alpha_for_elu=alpha_for_elu, alpha_for_relu=alpha_for_relu
        )(current_layer_object)

        if dropout_fraction is not None:
            current_layer_object = architecture_utils.get_dropout_layer(
                dropout_fraction
            )(current_layer_object)

        if use_batch_normalization:
            current_layer_object = architecture_utils.get_batch_norm_layer()(
                current_layer_object)

        new_layer_object = None

        for _ in range(num_conv_layers_per_set):
            if new_layer_object is None:
                this_input_layer_object = current_layer_object
            else:
                this_input_layer_object = new_layer_object

            new_layer_object = architecture_utils.get_1d_conv_layer(
                num_kernel_rows=NUM_CONV_KERNEL_HEIGHTS,
                num_rows_per_stride=1, num_filters=current_num_filters,
                padding_type_string=padding_type_string,
                weight_regularizer=regularizer_object
            )(this_input_layer_object)

            new_layer_object = architecture_utils.get_activation_layer(
                activation_function_string=activation_function_string,
                alpha_for_elu=alpha_for_elu, alpha_for_relu=alpha_for_relu
            )(new_layer_object)

            if dropout_fraction is not None:
                new_layer_object = architecture_utils.get_dropout_layer(
                    dropout_fraction
                )(new_layer_object)

            if use_batch_normalization:
                new_layer_object = (
                    architecture_utils.get_batch_norm_layer()(new_layer_object)
                )

        current_layer_object = keras.layers.add(
            [current_layer_object, new_layer_object]
        )

        current_layer_object = architecture_utils.get_1d_pooling_layer(
            num_rows_in_window=2, num_rows_per_stride=2,
            pooling_type_string=pooling_type_string
        )(current_layer_object)

    these_dimensions = numpy.array(
        current_layer_object.get_shape().as_list()[1:], dtype=int)
    num_scalar_features = numpy.prod(these_dimensions)

    current_layer_object = architecture_utils.get_flattening_layer()(
        current_layer_object)
    return input_layer_object, current_layer_object, num_scalar_features


def create_3d_resnet(
        num_radar_rows, num_radar_columns, num_radar_heights, num_radar_fields,
        num_classes, num_sounding_fields, num_sounding_heights,
        use_padding=False,
        pooling_type_string=architecture_utils.MAX_POOLING_STRING,
        num_conv_layer_sets=DEFAULT_NUM_CONV_LAYER_SETS,
        num_conv_layers_per_set=DEFAULT_NUM_CONV_LAYERS_PER_SET,
        first_num_radar_filters=None, first_num_sounding_filters=None,
        conv_layer_dropout_fraction=DEFAULT_CONV_LAYER_DROPOUT_FRACTION,
        num_dense_layers=DEFAULT_NUM_DENSE_LAYERS,
        dense_layer_dropout_fraction=DEFAULT_DENSE_LAYER_DROPOUT_FRACTION,
        l1_weight=DEFAULT_L1_WEIGHT, l2_weight=DEFAULT_L2_WEIGHT,
        activation_function_string=DEFAULT_ACTIVATION_FUNCTION_STRING,
        alpha_for_elu=architecture_utils.DEFAULT_ALPHA_FOR_ELU,
        alpha_for_relu=architecture_utils.DEFAULT_ALPHA_FOR_RELU,
        use_batch_normalization=DEFAULT_USE_BATCH_NORM_FLAG,
        list_of_metric_functions=DEFAULT_METRIC_FUNCTION_LIST):
    """Creates res-net to take in 3-D radar data (and possibly soundings).

    :param num_radar_rows: See doc for `create_3d_cnn`.
    :param num_radar_columns: Same.
    :param num_radar_heights: Same.
    :param num_radar_fields: Same.
    :param num_classes: Same.
    :param num_sounding_fields: Same.
    :param num_sounding_heights: Same.
    :param use_padding: Same.
    :param pooling_type_string: Same.
    :param num_conv_layer_sets: Same.
    :param num_conv_layers_per_set: Same.
    :param first_num_radar_filters: Same.
    :param first_num_sounding_filters: Same.
    :param conv_layer_dropout_fraction: Same.
    :param num_dense_layers: Same.
    :param dense_layer_dropout_fraction: Same.
    :param l1_weight: Same.
    :param l2_weight: Same.
    :param activation_function_string: Same.
    :param alpha_for_elu: Same.
    :param alpha_for_relu: Same.
    :param use_batch_normalization: Same.
    :param list_of_metric_functions: Same.
    :return: resnet_model_object: Untrained instance of `keras.models.Model`.
    """

    # TODO(thunderhoser): This is my first attempt at res-nets.  I don't know
    # what the best architecture is, especially vis-a-vis where activation and
    # batch-norm layers should go.

    error_checking.assert_is_boolean(use_padding)
    padding_type_string = (
        architecture_utils.YES_PADDING_STRING if use_padding
        else architecture_utils.NO_PADDING_STRING
    )

    error_checking.assert_is_integer(num_sounding_fields)
    error_checking.assert_is_integer(num_sounding_heights)
    include_soundings = num_sounding_fields > 0 or num_sounding_heights > 0

    first_num_radar_filters, _ = check_radar_options(
        num_grid_rows=num_radar_rows, num_grid_columns=num_radar_columns,
        num_dimensions=3, num_classes=num_classes,
        num_conv_layer_sets=num_conv_layer_sets,
        num_conv_layers_per_set=num_conv_layers_per_set,
        conv_layer_dropout_fraction=conv_layer_dropout_fraction,
        num_dense_layers=num_dense_layers,
        dense_layer_dropout_fraction=dense_layer_dropout_fraction,
        l1_weight=l1_weight, l2_weight=l2_weight,
        activation_function_string=activation_function_string,
        alpha_for_elu=alpha_for_elu, alpha_for_relu=alpha_for_relu,
        use_batch_normalization=use_batch_normalization,
        num_heights=num_radar_heights, num_fields=num_radar_fields,
        do_separable_conv=False, first_num_filters=first_num_radar_filters)

    radar_input_layer_object = keras.layers.Input(
        shape=(num_radar_rows, num_radar_columns, num_radar_heights,
               num_radar_fields)
    )

    regularizer_object = architecture_utils.get_weight_regularizer(
        l1_weight=l1_weight, l2_weight=l2_weight)

    radar_layer_object = None
    current_num_filters = None

    for i in range(num_conv_layer_sets):
        if current_num_filters is None:
            current_num_filters = first_num_radar_filters + 0
            this_input_layer_object = radar_input_layer_object
        else:
            current_num_filters *= 2
            this_input_layer_object = radar_layer_object

        radar_layer_object = architecture_utils.get_3d_conv_layer(
            num_filters=current_num_filters,
            num_kernel_rows=NUM_CONV_KERNEL_ROWS,
            num_kernel_columns=NUM_CONV_KERNEL_COLUMNS,
            num_kernel_heights=NUM_CONV_KERNEL_HEIGHTS,
            num_rows_per_stride=1, num_columns_per_stride=1,
            num_heights_per_stride=1, padding_type_string=padding_type_string,
            weight_regularizer=regularizer_object
        )(this_input_layer_object)

        radar_layer_object = architecture_utils.get_activation_layer(
            activation_function_string=activation_function_string,
            alpha_for_elu=alpha_for_elu, alpha_for_relu=alpha_for_relu
        )(radar_layer_object)

        if conv_layer_dropout_fraction is not None:
            radar_layer_object = architecture_utils.get_dropout_layer(
                dropout_fraction=conv_layer_dropout_fraction
            )(radar_layer_object)

        if use_batch_normalization:
            radar_layer_object = architecture_utils.get_batch_norm_layer()(
                radar_layer_object)

        new_layer_object = None

        for _ in range(num_conv_layers_per_set):
            if new_layer_object is None:
                this_input_layer_object = radar_layer_object
            else:
                this_input_layer_object = new_layer_object

            new_layer_object = architecture_utils.get_3d_conv_layer(
                num_filters=current_num_filters,
                num_kernel_rows=NUM_CONV_KERNEL_ROWS,
                num_kernel_columns=NUM_CONV_KERNEL_COLUMNS,
                num_kernel_heights=NUM_CONV_KERNEL_HEIGHTS,
                num_rows_per_stride=1, num_columns_per_stride=1,
                num_heights_per_stride=1,
                padding_type_string=padding_type_string,
                weight_regularizer=regularizer_object
            )(this_input_layer_object)

            new_layer_object = architecture_utils.get_activation_layer(
                activation_function_string=activation_function_string,
                alpha_for_elu=alpha_for_elu, alpha_for_relu=alpha_for_relu
            )(new_layer_object)

            if conv_layer_dropout_fraction is not None:
                new_layer_object = architecture_utils.get_dropout_layer(
                    dropout_fraction=conv_layer_dropout_fraction
                )(new_layer_object)

            if use_batch_normalization:
                new_layer_object = (
                    architecture_utils.get_batch_norm_layer()(new_layer_object)
                )

        radar_layer_object = keras.layers.add(
            [radar_layer_object, new_layer_object]
        )

        if i == num_conv_layer_sets - 1:
            this_height_factor = 1
        else:
            this_height_factor = 2

        radar_layer_object = architecture_utils.get_3d_pooling_layer(
            num_rows_in_window=2, num_columns_in_window=2,
            num_heights_in_window=this_height_factor, num_rows_per_stride=2,
            num_columns_per_stride=2, num_heights_per_stride=this_height_factor,
            pooling_type_string=pooling_type_string
        )(radar_layer_object)

    these_dimensions = numpy.array(
        radar_layer_object.get_shape().as_list()[1:], dtype=int)
    num_radar_features = numpy.prod(these_dimensions)

    radar_layer_object = architecture_utils.get_flattening_layer()(
        radar_layer_object)

    if include_soundings:
        (sounding_input_layer_object, sounding_layer_object,
         num_sounding_features
        ) = _create_sounding_layers_resnet(
            num_fields=num_sounding_fields, num_heights=num_sounding_heights,
            num_conv_layer_sets=num_conv_layer_sets,
            num_conv_layers_per_set=num_conv_layers_per_set,
            padding_type_string=padding_type_string, l1_weight=l1_weight,
            l2_weight=l2_weight, pooling_type_string=pooling_type_string,
            activation_function_string=activation_function_string,
            alpha_for_elu=alpha_for_elu, alpha_for_relu=alpha_for_relu,
            use_batch_normalization=use_batch_normalization,
            dropout_fraction=conv_layer_dropout_fraction,
            first_num_filters=first_num_sounding_filters)

        layer_object = keras.layers.concatenate(
            [radar_layer_object, sounding_layer_object]
        )

    else:
        layer_object = radar_layer_object
        num_sounding_features = 0

    layer_object = _create_dense_layers(
        flattening_layer_object=layer_object,
        num_scalar_features=num_radar_features + num_sounding_features,
        num_classes=num_classes, num_dense_layers=num_dense_layers,
        dropout_fraction=dense_layer_dropout_fraction,
        activation_function_string=activation_function_string,
        alpha_for_elu=alpha_for_elu, alpha_for_relu=alpha_for_relu,
        use_batch_normalization=use_batch_normalization)

    if include_soundings:
        model_object = keras.models.Model(
            inputs=[radar_input_layer_object, sounding_input_layer_object],
            outputs=layer_object)
    else:
        model_object = keras.models.Model(
            inputs=radar_input_layer_object, outputs=layer_object)

    loss_function = _get_output_layer_and_loss(num_classes)[-1]
    model_object.compile(
        loss=loss_function, optimizer=keras.optimizers.Adam(),
        metrics=list_of_metric_functions)

    model_object.summary()
    return model_object
