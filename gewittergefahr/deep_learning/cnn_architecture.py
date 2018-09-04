"""Methods for creating a CNN (building the architecture)."""

import numpy
import keras.layers
from gewittergefahr.gg_utils import error_checking
from gewittergefahr.deep_learning import architecture_utils
from gewittergefahr.deep_learning import keras_metrics

DEFAULT_L2_WEIGHT = 1e-3
DEFAULT_CONV_LAYER_DROPOUT_FRACTION = 0.25
DEFAULT_DENSE_LAYER_DROPOUT_FRACTION = 0.5

INIT_NUM_RADAR_FILTERS_DEFAULT = 32
INIT_NUM_SOUNDING_FILTERS_DEFAULT = 16
DEFAULT_NUM_SOUNDING_CONV_LAYER_SETS = 3

VALID_NUMBERS_OF_SOUNDING_HEIGHTS = numpy.array([49], dtype=int)

DEFAULT_METRIC_FUNCTION_LIST = [
    keras_metrics.accuracy, keras_metrics.binary_accuracy,
    keras_metrics.binary_csi, keras_metrics.binary_frequency_bias,
    keras_metrics.binary_pod, keras_metrics.binary_pofd,
    keras_metrics.binary_peirce_score, keras_metrics.binary_success_ratio,
    keras_metrics.binary_focn
]


def _check_input_args(
        num_radar_rows, num_radar_columns, num_radar_dimensions,
        num_radar_conv_layer_sets, num_conv_layers_per_set,
        init_num_radar_filters, num_classes, conv_activation_function_string,
        use_batch_normalization, alpha_for_elu=None, alpha_for_relu=None,
        num_radar_channels=None, num_radar_fields=None, num_radar_heights=None,
        conv_layer_dropout_fraction=None, dense_layer_dropout_fraction=None,
        l2_weight=None, num_sounding_fields=None, num_sounding_heights=None,
        num_sounding_conv_layer_sets=None, init_num_sounding_filters=None):
    """Error-checks input arguments for model architecture.

    :param num_radar_rows: Number of horizontal rows in each radar image.
    :param num_radar_columns: Number of horizontal columns in each radar image.
    :param num_radar_dimensions: Number of dimensions in each radar image.
    :param num_radar_conv_layer_sets: Number of sets of conv layers for radar
        data.  Each successive conv-layer set will halve the dimensions of the
        image.  Conv layers in the same set will *not* change the dimensions.
    :param num_conv_layers_per_set: Number of conv layers in each set.
    :param init_num_radar_filters: Initial number of radar filters (in the first
        conv-layer set for radar data).  Each successive conv-layer set will
        double the number of filters.
    :param num_classes: Number of output classes (i.e., number of classes that
        the net will predict).
    :param conv_activation_function_string: Activation function for conv layers.
        Must be accepted by `architecture_utils.check_activation_function`.
    :param use_batch_normalization: Boolean flag.  If True, the net will include
        a batch-normalization layer after each conv layer and each dense layer.
    :param alpha_for_elu: Slope for negative inputs to eLU (exponential linear
        unit) activation function.
    :param alpha_for_relu: Slope for negative inputs to ReLU (rectified linear
        unit) activation function.
    :param num_radar_channels: [used only if num_radar_dimensions = 2]
        Number of channels (field/height pairs) in each radar image.
    :param num_radar_fields: [used only if num_radar_dimensions = 3]
        Number of fields in each radar image.
    :param num_radar_heights: [used only if num_radar_dimensions = 3]
        Number of pixel heights in each radar image.
    :param conv_layer_dropout_fraction: Dropout fraction (will be applied to
        each conv layer).  If None, there will be no dropout after conv layers.
    :param dense_layer_dropout_fraction: Dropout fraction (will be applied to
        each dense ["fully connected"] layer).  If None, there will be no
        dropout after dernse layers.
    :param l2_weight: L2-regularization weight (will be applied to each conv
        layer).  If None, there will be no L2 regularization.
    :param num_sounding_fields: Number of fields in each sounding.  If the model
        input does not include soundings, leave this as None.
    :param num_sounding_heights: [used iff `num_sounding_fields is not None`]
        Number of height levels in each sounding.
    :param num_sounding_conv_layer_sets:
        [used iff `num_sounding_fields is not None`]
        Number of sets of conv layers for sounding data.  See
            `num_radar_conv_layer_sets` for more details on how these work.
    :param init_num_sounding_filters:
        [used iff `num_sounding_fields is not None`]
        Initial number of sounding filters.  See `init_num_radar_filters`
        for more details on how these work.
    :raises: ValueError: if
        `num_sounding_heights not in VALID_NUMBERS_OF_SOUNDING_HEIGHTS`.
    """

    architecture_utils.check_activation_function(
        activation_function_string=conv_activation_function_string,
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
    error_checking.assert_is_integer(init_num_radar_filters)
    error_checking.assert_is_geq(init_num_radar_filters, 1)

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
        error_checking.assert_is_integer(init_num_sounding_filters)
        error_checking.assert_is_geq(init_num_sounding_filters, 1)

        num_heights_after_last_conv = float(num_sounding_heights)
        for _ in range(num_sounding_conv_layer_sets):
            num_heights_after_last_conv = numpy.floor(
                num_heights_after_last_conv / 2)

        error_checking.assert_is_geq(num_heights_after_last_conv, 1)


def _get_sounding_layers(
        num_sounding_heights, num_sounding_fields, init_num_filters,
        num_conv_layer_sets, num_conv_layers_per_set, pooling_type_string,
        dropout_fraction, regularizer_object, conv_activation_function_string,
        alpha_for_elu, alpha_for_relu, use_batch_normalization):
    """Creates CNN layers to convolve over sounding data.

    :param num_sounding_heights: See doc for `_check_input_args`.
    :param num_sounding_fields: Same.
    :param init_num_filters: Same.
    :param num_conv_layer_sets: Same.
    :param num_conv_layers_per_set: Same.
    :param pooling_type_string: Same.
    :param dropout_fraction: Same.
    :param regularizer_object: Instance of `keras.regularizers.l1_l2`.  Will be
        applied to each convolutional layer.
    :param conv_activation_function_string: See doc for `_check_input_args`.
    :param alpha_for_elu: Same.
    :param alpha_for_relu: Same.
    :param use_batch_normalization: Same.
    :return: input_layer_object: Instance of `keras.layers.Input`.
    :return: flattening_layer_object: Instance of `keras.layers.Flatten`.
    """

    sounding_input_layer_object = keras.layers.Input(shape=(
        num_sounding_heights, num_sounding_fields))

    sounding_layer_object = None
    this_num_output_filters = init_num_filters + 0

    for _ in range(num_conv_layer_sets):
        if sounding_layer_object is None:
            sounding_layer_object = sounding_input_layer_object
        else:
            this_num_output_filters *= 2

        for _ in range(num_conv_layers_per_set):
            sounding_layer_object = architecture_utils.get_1d_conv_layer(
                num_output_filters=this_num_output_filters,
                num_kernel_pixels=3, num_pixels_per_stride=1,
                padding_type=architecture_utils.YES_PADDING_TYPE,
                kernel_weight_regularizer=regularizer_object,
                is_first_layer=False
            )(sounding_layer_object)

            sounding_layer_object = architecture_utils.get_activation_layer(
                activation_function_string=conv_activation_function_string,
                alpha_for_elu=alpha_for_elu, alpha_for_relu=alpha_for_relu
            )(sounding_layer_object)

            if use_batch_normalization:
                sounding_layer_object = (
                    architecture_utils.get_batch_normalization_layer()(
                        sounding_layer_object)
                )

            if dropout_fraction is not None:
                sounding_layer_object = architecture_utils.get_dropout_layer(
                    dropout_fraction=dropout_fraction
                )(sounding_layer_object)

        sounding_layer_object = architecture_utils.get_1d_pooling_layer(
            num_pixels_in_window=2, pooling_type=pooling_type_string,
            num_pixels_per_stride=2
        )(sounding_layer_object)

    sounding_layer_object = architecture_utils.get_flattening_layer()(
        sounding_layer_object)
    return sounding_input_layer_object, sounding_layer_object


def _get_output_layer_and_loss_function(num_classes):
    """Creates output layer and loss function.

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
        activation_function_string = architecture_utils.SIGMOID_FUNCTION_STRING
    else:
        num_output_units = num_classes
        loss_function = keras.losses.categorical_crossentropy
        activation_function_string = architecture_utils.SOFTMAX_FUNCTION_STRING

    dense_layer_object = architecture_utils.get_fully_connected_layer(
        num_output_units=num_output_units)
    activation_layer_object = architecture_utils.get_activation_layer(
        activation_function_string=activation_function_string)

    return dense_layer_object, activation_layer_object, loss_function


def get_2d_mnist_architecture(
        num_radar_rows, num_radar_columns, num_radar_channels, num_classes,
        init_num_filters, l2_weight=None,
        list_of_metric_functions=DEFAULT_METRIC_FUNCTION_LIST):
    """Creates 2-D CNN with architecture inspired by the following example.

    https://github.com/keras-team/keras/blob/master/examples/mnist_cnn.py

    :param num_radar_rows: See doc for `_check_input_args`.
    :param num_radar_columns: Same.
    :param num_radar_channels: Same.
    :param num_classes: Same.
    :param init_num_filters: Same.
    :param l2_weight: Same.
    :param list_of_metric_functions: 1-D list of metric functions.  These will
        be used during training and printed to the log file after each batch.
        Each metric must take one target tensor and one prediction tensor and
        return a single number.  See `keras_metrics.binary_accuracy` for an
        example.
    :return: model_object: `keras.models.Sequential` object with the
        aforementioned architecture.
    """

    _check_input_args(
        num_radar_rows=num_radar_rows, num_radar_columns=num_radar_columns,
        num_radar_channels=num_radar_channels, num_radar_dimensions=2,
        num_radar_conv_layer_sets=2, num_conv_layers_per_set=1,
        init_num_radar_filters=init_num_filters, num_classes=num_classes,
        conv_activation_function_string='relu', use_batch_normalization=False,
        l2_weight=l2_weight,
        alpha_for_relu=architecture_utils.DEFAULT_ALPHA_FOR_RELU)

    if l2_weight is None:
        regularizer_object = None
    else:
        regularizer_object = architecture_utils.get_weight_regularizer(
            l1_penalty=0, l2_penalty=l2_weight)

    model_object = keras.models.Sequential()
    layer_object = architecture_utils.get_2d_conv_layer(
        num_output_filters=init_num_filters, num_kernel_rows=3,
        num_kernel_columns=3, num_rows_per_stride=1, num_columns_per_stride=1,
        padding_type=architecture_utils.NO_PADDING_TYPE,
        kernel_weight_regularizer=regularizer_object, is_first_layer=True,
        num_input_rows=num_radar_rows, num_input_columns=num_radar_columns,
        num_input_channels=num_radar_channels)
    model_object.add(layer_object)

    model_object.add(architecture_utils.get_activation_layer(
        activation_function_string='relu'))

    layer_object = architecture_utils.get_2d_conv_layer(
        num_output_filters=2 * init_num_filters, num_kernel_rows=3,
        num_kernel_columns=3, num_rows_per_stride=1, num_columns_per_stride=1,
        padding_type=architecture_utils.NO_PADDING_TYPE,
        kernel_weight_regularizer=regularizer_object)
    model_object.add(layer_object)

    model_object.add(architecture_utils.get_activation_layer(
        activation_function_string='relu'))

    layer_object = architecture_utils.get_2d_pooling_layer(
        num_rows_in_window=2, num_columns_in_window=2,
        pooling_type=architecture_utils.MAX_POOLING_TYPE, num_rows_per_stride=2,
        num_columns_per_stride=2)
    model_object.add(layer_object)

    layer_object = architecture_utils.get_dropout_layer(dropout_fraction=0.25)
    model_object.add(layer_object)

    layer_object = architecture_utils.get_flattening_layer()
    model_object.add(layer_object)

    layer_object = architecture_utils.get_fully_connected_layer(
        num_output_units=128)
    model_object.add(layer_object)
    model_object.add(architecture_utils.get_activation_layer(
        activation_function_string='relu'))

    layer_object = architecture_utils.get_dropout_layer(dropout_fraction=0.5)
    model_object.add(layer_object)

    (dense_layer_object, activation_layer_object, loss_function
    ) = _get_output_layer_and_loss_function(num_classes)
    model_object.add(dense_layer_object)
    model_object.add(activation_layer_object)

    model_object.compile(
        loss=loss_function, optimizer=keras.optimizers.Adam(),
        metrics=list_of_metric_functions)

    model_object.summary()
    return model_object


def get_2d_swirlnet_architecture(
        num_radar_rows, num_radar_columns, num_radar_channels,
        num_radar_conv_layer_sets, num_conv_layers_per_set, pooling_type_string,
        num_classes,
        conv_activation_function_string=architecture_utils.RELU_FUNCTION_STRING,
        alpha_for_elu=architecture_utils.DEFAULT_ALPHA_FOR_ELU,
        alpha_for_relu=architecture_utils.DEFAULT_ALPHA_FOR_RELU,
        use_batch_normalization=True,
        init_num_radar_filters=INIT_NUM_RADAR_FILTERS_DEFAULT,
        conv_layer_dropout_fraction=DEFAULT_CONV_LAYER_DROPOUT_FRACTION,
        dense_layer_dropout_fraction=DEFAULT_DENSE_LAYER_DROPOUT_FRACTION,
        l2_weight=DEFAULT_L2_WEIGHT, num_sounding_fields=None,
        num_sounding_heights=None,
        num_sounding_conv_layer_sets=DEFAULT_NUM_SOUNDING_CONV_LAYER_SETS,
        init_num_sounding_filters=INIT_NUM_SOUNDING_FILTERS_DEFAULT,
        list_of_metric_functions=DEFAULT_METRIC_FUNCTION_LIST):
    """Creates 2-D CNN with architecture inspired by the following example.

    https://github.com/djgagne/swirlnet/blob/master/notebooks/
    deep_swirl_tutorial.ipynb

    :param num_radar_rows: See doc for `_check_input_args`.
    :param num_radar_columns: Same.
    :param num_radar_channels: Same.
    :param num_radar_conv_layer_sets: Same.
    :param num_conv_layers_per_set: Same.
    :param pooling_type_string: Same.
    :param num_classes: Same.
    :param conv_activation_function_string: Same.
    :param alpha_for_elu: Same.
    :param alpha_for_relu: Same.
    :param use_batch_normalization: Same.
    :param init_num_radar_filters: Same.
    :param conv_layer_dropout_fraction: Same.
    :param dense_layer_dropout_fraction: Same.
    :param l2_weight: Same.
    :param num_sounding_heights: Same.
    :param num_sounding_fields: Same.
    :param num_sounding_conv_layer_sets: Same.
    :param init_num_sounding_filters: Same.
    :param list_of_metric_functions: See doc for `get_2d_mnist_architecture`.
    :return: model_object: `keras.models.Sequential` object with the
        aforementioned architecture.
    """

    _check_input_args(
        num_radar_rows=num_radar_rows, num_radar_columns=num_radar_columns,
        num_radar_channels=num_radar_channels, num_radar_dimensions=2,
        num_radar_conv_layer_sets=num_radar_conv_layer_sets,
        num_conv_layers_per_set=num_conv_layers_per_set,
        init_num_radar_filters=init_num_radar_filters, num_classes=num_classes,
        conv_activation_function_string=conv_activation_function_string,
        alpha_for_elu=alpha_for_elu, alpha_for_relu=alpha_for_relu,
        use_batch_normalization=use_batch_normalization,
        conv_layer_dropout_fraction=conv_layer_dropout_fraction,
        dense_layer_dropout_fraction=dense_layer_dropout_fraction,
        l2_weight=l2_weight, num_sounding_fields=num_sounding_fields,
        num_sounding_heights=num_sounding_heights,
        num_sounding_conv_layer_sets=num_sounding_conv_layer_sets,
        init_num_sounding_filters=init_num_sounding_filters)

    if l2_weight is None:
        regularizer_object = None
    else:
        regularizer_object = architecture_utils.get_weight_regularizer(
            l1_penalty=0, l2_penalty=l2_weight)

    radar_input_layer_object = keras.layers.Input(
        shape=(num_radar_rows, num_radar_columns, num_radar_channels)
    )

    radar_layer_object = None
    this_num_output_filters = init_num_radar_filters + 0

    for _ in range(num_radar_conv_layer_sets):
        if radar_layer_object is None:
            radar_layer_object = radar_input_layer_object
        else:
            this_num_output_filters *= 2

        for _ in range(num_conv_layers_per_set):
            radar_layer_object = architecture_utils.get_2d_conv_layer(
                num_output_filters=this_num_output_filters,
                num_kernel_rows=5, num_kernel_columns=5, num_rows_per_stride=1,
                num_columns_per_stride=1,
                padding_type=architecture_utils.YES_PADDING_TYPE,
                kernel_weight_regularizer=regularizer_object,
                is_first_layer=False
            )(radar_layer_object)

            radar_layer_object = architecture_utils.get_activation_layer(
                activation_function_string=conv_activation_function_string,
                alpha_for_elu=alpha_for_elu, alpha_for_relu=alpha_for_relu
            )(radar_layer_object)

            if use_batch_normalization:
                radar_layer_object = (
                    architecture_utils.get_batch_normalization_layer()(
                        radar_layer_object)
                )

            if conv_layer_dropout_fraction is not None:
                radar_layer_object = architecture_utils.get_dropout_layer(
                    dropout_fraction=conv_layer_dropout_fraction
                )(radar_layer_object)

        radar_layer_object = architecture_utils.get_2d_pooling_layer(
            num_rows_in_window=2, num_columns_in_window=2,
            pooling_type=pooling_type_string, num_rows_per_stride=2,
            num_columns_per_stride=2
        )(radar_layer_object)

    radar_layer_object = architecture_utils.get_flattening_layer()(
        radar_layer_object)

    if num_sounding_fields is None:
        layer_object = radar_layer_object
    else:
        (sounding_input_layer_object, sounding_layer_object
        ) = _get_sounding_layers(
            num_sounding_heights=num_sounding_heights,
            num_sounding_fields=num_sounding_fields,
            init_num_filters=init_num_sounding_filters,
            num_conv_layer_sets=num_sounding_conv_layer_sets,
            num_conv_layers_per_set=num_conv_layers_per_set,
            pooling_type_string=pooling_type_string,
            dropout_fraction=conv_layer_dropout_fraction,
            regularizer_object=regularizer_object,
            conv_activation_function_string=conv_activation_function_string,
            alpha_for_elu=alpha_for_elu, alpha_for_relu=alpha_for_relu,
            use_batch_normalization=use_batch_normalization)

        layer_object = keras.layers.concatenate(
            [radar_layer_object, sounding_layer_object])

    (dense_layer_object, activation_layer_object, loss_function
    ) = _get_output_layer_and_loss_function(num_classes)
    layer_object = dense_layer_object(layer_object)

    if use_batch_normalization:
        layer_object = architecture_utils.get_batch_normalization_layer()(
            layer_object)
    if dense_layer_dropout_fraction is not None:
        layer_object = architecture_utils.get_dropout_layer(
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
        metrics=list_of_metric_functions)

    model_object.summary()
    return model_object


def get_3d_swirlnet_architecture(
        num_radar_rows, num_radar_columns, num_radar_heights, num_radar_fields,
        num_radar_conv_layer_sets, num_conv_layers_per_set, pooling_type_string,
        num_classes,
        conv_activation_function_string=architecture_utils.RELU_FUNCTION_STRING,
        alpha_for_elu=architecture_utils.DEFAULT_ALPHA_FOR_ELU,
        alpha_for_relu=architecture_utils.DEFAULT_ALPHA_FOR_RELU,
        use_batch_normalization=True,
        init_num_radar_filters=INIT_NUM_RADAR_FILTERS_DEFAULT,
        conv_layer_dropout_fraction=DEFAULT_CONV_LAYER_DROPOUT_FRACTION,
        dense_layer_dropout_fraction=DEFAULT_DENSE_LAYER_DROPOUT_FRACTION,
        l2_weight=DEFAULT_L2_WEIGHT, num_sounding_fields=None,
        num_sounding_heights=None,
        num_sounding_conv_layer_sets=DEFAULT_NUM_SOUNDING_CONV_LAYER_SETS,
        init_num_sounding_filters=INIT_NUM_SOUNDING_FILTERS_DEFAULT,
        list_of_metric_functions=DEFAULT_METRIC_FUNCTION_LIST):
    """Creates 3-D CNN with architecture inspired by the following example.

    https://github.com/djgagne/swirlnet/blob/master/notebooks/
    deep_swirl_tutorial.ipynb

    :param num_radar_rows: See doc for `_check_input_args`.
    :param num_radar_columns: Same.
    :param num_radar_heights: Same.
    :param num_radar_fields: Same.
    :param num_radar_conv_layer_sets: Same.
    :param num_conv_layers_per_set: Same.
    :param pooling_type_string: Same.
    :param num_classes: Same.
    :param conv_activation_function_string: Same.
    :param alpha_for_elu: Same.
    :param alpha_for_relu: Same.
    :param use_batch_normalization: Same.
    :param init_num_radar_filters: Same.
    :param conv_layer_dropout_fraction: Same.
    :param dense_layer_dropout_fraction: Same.
    :param l2_weight: Same.
    :param num_sounding_heights: Same.
    :param num_sounding_fields: Same.
    :param num_sounding_conv_layer_sets: Same.
    :param init_num_sounding_filters: Same.
    :param list_of_metric_functions: See doc for `get_2d_mnist_architecture`.
    :return: model_object: `keras.models.Sequential` object with the
        aforementioned architecture.
    """

    _check_input_args(
        num_radar_rows=num_radar_rows, num_radar_columns=num_radar_columns,
        num_radar_heights=num_radar_heights, num_radar_fields=num_radar_fields,
        num_radar_dimensions=3,
        num_radar_conv_layer_sets=num_radar_conv_layer_sets,
        num_conv_layers_per_set=num_conv_layers_per_set,
        init_num_radar_filters=init_num_radar_filters, num_classes=num_classes,
        conv_activation_function_string=conv_activation_function_string,
        alpha_for_elu=alpha_for_elu, alpha_for_relu=alpha_for_relu,
        use_batch_normalization=use_batch_normalization,
        conv_layer_dropout_fraction=conv_layer_dropout_fraction,
        dense_layer_dropout_fraction=dense_layer_dropout_fraction,
        l2_weight=l2_weight, num_sounding_heights=num_sounding_heights,
        num_sounding_fields=num_sounding_fields,
        num_sounding_conv_layer_sets=num_sounding_conv_layer_sets,
        init_num_sounding_filters=init_num_sounding_filters)

    if l2_weight is None:
        regularizer_object = None
    else:
        regularizer_object = architecture_utils.get_weight_regularizer(
            l1_penalty=0, l2_penalty=l2_weight)

    radar_input_layer_object = keras.layers.Input(
        shape=(num_radar_rows, num_radar_columns, num_radar_heights,
               num_radar_fields)
    )

    radar_layer_object = None
    this_num_output_filters = init_num_radar_filters + 0

    for _ in range(num_radar_conv_layer_sets):
        if radar_layer_object is None:
            radar_layer_object = radar_input_layer_object
        else:
            this_num_output_filters *= 2

        for _ in range(num_conv_layers_per_set):
            radar_layer_object = architecture_utils.get_3d_conv_layer(
                num_output_filters=this_num_output_filters,
                num_kernel_rows=3, num_kernel_columns=3, num_kernel_depths=3,
                num_rows_per_stride=1, num_columns_per_stride=1,
                num_depths_per_stride=1,
                padding_type=architecture_utils.YES_PADDING_TYPE,
                kernel_weight_regularizer=regularizer_object,
                is_first_layer=False
            )(radar_layer_object)

            radar_layer_object = architecture_utils.get_activation_layer(
                activation_function_string=conv_activation_function_string,
                alpha_for_elu=alpha_for_elu, alpha_for_relu=alpha_for_relu
            )(radar_layer_object)

            if use_batch_normalization:
                radar_layer_object = (
                    architecture_utils.get_batch_normalization_layer()(
                        radar_layer_object)
                )

            if conv_layer_dropout_fraction is not None:
                radar_layer_object = architecture_utils.get_dropout_layer(
                    dropout_fraction=conv_layer_dropout_fraction
                )(radar_layer_object)

        radar_layer_object = architecture_utils.get_3d_pooling_layer(
            num_rows_in_window=2, num_columns_in_window=2,
            num_depths_in_window=2, pooling_type=pooling_type_string,
            num_rows_per_stride=2, num_columns_per_stride=2,
            num_depths_per_stride=2
        )(radar_layer_object)

    radar_layer_object = architecture_utils.get_flattening_layer()(
        radar_layer_object)

    if num_sounding_fields is None:
        layer_object = radar_layer_object
    else:
        (sounding_input_layer_object, sounding_layer_object
        ) = _get_sounding_layers(
            num_sounding_heights=num_sounding_heights,
            num_sounding_fields=num_sounding_fields,
            init_num_filters=init_num_sounding_filters,
            num_conv_layer_sets=num_sounding_conv_layer_sets,
            num_conv_layers_per_set=num_conv_layers_per_set,
            pooling_type_string=pooling_type_string,
            dropout_fraction=conv_layer_dropout_fraction,
            regularizer_object=regularizer_object,
            conv_activation_function_string=conv_activation_function_string,
            alpha_for_elu=alpha_for_elu, alpha_for_relu=alpha_for_relu,
            use_batch_normalization=use_batch_normalization)

        layer_object = keras.layers.concatenate(
            [radar_layer_object, sounding_layer_object])

    (dense_layer_object, activation_layer_object, loss_function
    ) = _get_output_layer_and_loss_function(num_classes)
    layer_object = dense_layer_object(layer_object)

    if use_batch_normalization:
        layer_object = architecture_utils.get_batch_normalization_layer()(
            layer_object)
    if dense_layer_dropout_fraction is not None:
        layer_object = architecture_utils.get_dropout_layer(
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
        metrics=list_of_metric_functions)

    model_object.summary()
    return model_object


def get_2d3d_swirlnet_architecture(
        num_reflectivity_rows, num_reflectivity_columns,
        num_reflectivity_heights, num_azimuthal_shear_fields,
        num_radar_conv_layer_sets, num_conv_layers_per_set, pooling_type_string,
        num_classes,
        conv_activation_function_string=architecture_utils.RELU_FUNCTION_STRING,
        alpha_for_elu=architecture_utils.DEFAULT_ALPHA_FOR_ELU,
        alpha_for_relu=architecture_utils.DEFAULT_ALPHA_FOR_RELU,
        use_batch_normalization=True, init_num_reflectivity_filters=8,
        init_num_shear_filters=16,
        conv_layer_dropout_fraction=DEFAULT_CONV_LAYER_DROPOUT_FRACTION,
        dense_layer_dropout_fraction=DEFAULT_DENSE_LAYER_DROPOUT_FRACTION,
        l2_weight=DEFAULT_L2_WEIGHT, num_sounding_fields=None,
        num_sounding_heights=None,
        num_sounding_conv_layer_sets=DEFAULT_NUM_SOUNDING_CONV_LAYER_SETS,
        init_num_sounding_filters=INIT_NUM_SOUNDING_FILTERS_DEFAULT,
        list_of_metric_functions=DEFAULT_METRIC_FUNCTION_LIST):
    """Creates CNN with architecture inspired by the following example.

    https://github.com/djgagne/swirlnet/blob/master/notebooks/
    deep_swirl_tutorial.ipynb

    This CNN does 2-D convolution over azimuthal-shear fields and 3-D
    convolution over reflectivity fields.

    This method assumes that azimuthal-shear fields have twice the horizontal
    resolution as reflectivity fields.  In other words:

    num_azimuthal_shear_rows = 2 * num_reflectivity_rows
    num_azimuthal_shear_columns = 2 * num_reflectivity_columns

    :param num_reflectivity_rows: Number of horizontal rows in each reflectivity
        image.
    :param num_reflectivity_columns: Number of horizontal columns in each
        reflectivity image.
    :param num_reflectivity_heights: Number of heights in each reflectivity
        image.
    :param num_azimuthal_shear_fields: Number of azimuthal-shear fields
        (channels).
    :param num_radar_conv_layer_sets: Number of conv-layer sets for radar data
        (after joining reflectivity and azimuthal-shear filters).
    :param num_conv_layers_per_set: See doc for `_check_input_args`.
    :param pooling_type_string: Same.
    :param num_classes: Same.
    :param conv_activation_function_string: Same.
    :param alpha_for_elu: Same.
    :param alpha_for_relu: Same.
    :param use_batch_normalization: Same.
    :param init_num_reflectivity_filters: Initial number of reflectivity filters
        (in the first conv-layer set for reflectivity data).  Each successive
        conv-layer set will double the number of filters.
    :param init_num_shear_filters: Same, but for azimuthal shear.
    :param conv_layer_dropout_fraction: See doc for `_check_input_args`.
    :param dense_layer_dropout_fraction: Same.
    :param l2_weight: Same.
    :param num_sounding_heights: Same.
    :param num_sounding_fields: Same.
    :param num_sounding_conv_layer_sets: Same.
    :param init_num_sounding_filters: Same.
    :param list_of_metric_functions: See doc for `get_2d_mnist_architecture`.
    :return: model_object: `keras.models.Sequential` object with the
        aforementioned architecture.
    """

    # Check input args for the reflectivity part of the net.
    _check_input_args(
        num_radar_rows=num_reflectivity_rows,
        num_radar_columns=num_reflectivity_columns,
        num_radar_heights=num_reflectivity_heights, num_radar_fields=1,
        num_radar_dimensions=3, num_radar_conv_layer_sets=3,
        num_conv_layers_per_set=num_conv_layers_per_set,
        init_num_radar_filters=init_num_reflectivity_filters,
        num_classes=num_classes,
        conv_activation_function_string=conv_activation_function_string,
        alpha_for_elu=alpha_for_elu, alpha_for_relu=alpha_for_relu,
        use_batch_normalization=use_batch_normalization,
        conv_layer_dropout_fraction=conv_layer_dropout_fraction,
        dense_layer_dropout_fraction=dense_layer_dropout_fraction,
        l2_weight=l2_weight, num_sounding_fields=num_sounding_fields,
        num_sounding_heights=num_sounding_heights,
        num_sounding_conv_layer_sets=num_sounding_conv_layer_sets,
        init_num_sounding_filters=init_num_sounding_filters)

    # Check input args for the azimuthal-shear part of the net.
    num_azimuthal_shear_rows = 2 * num_reflectivity_rows
    num_azimuthal_shear_columns = 2 * num_reflectivity_columns

    _check_input_args(
        num_radar_rows=num_azimuthal_shear_rows,
        num_radar_columns=num_azimuthal_shear_columns,
        num_radar_channels=num_azimuthal_shear_fields, num_radar_dimensions=2,
        num_radar_conv_layer_sets=1,
        init_num_radar_filters=init_num_shear_filters,
        num_conv_layers_per_set=num_conv_layers_per_set,
        num_classes=num_classes,
        conv_activation_function_string=conv_activation_function_string,
        alpha_for_elu=alpha_for_elu, alpha_for_relu=alpha_for_relu,
        use_batch_normalization=use_batch_normalization)

    # Check input args for the part that handles reflectivity and az shear.
    num_radar_channels = (
        init_num_shear_filters + init_num_reflectivity_filters * 8)

    _check_input_args(
        num_radar_rows=num_reflectivity_rows,
        num_radar_columns=num_reflectivity_columns,
        num_radar_channels=num_radar_channels, num_radar_dimensions=2,
        num_radar_conv_layer_sets=num_radar_conv_layer_sets,
        num_conv_layers_per_set=num_conv_layers_per_set,
        init_num_radar_filters=num_radar_channels * 2, num_classes=num_classes,
        conv_activation_function_string=conv_activation_function_string,
        alpha_for_elu=alpha_for_elu, alpha_for_relu=alpha_for_relu,
        use_batch_normalization=use_batch_normalization)

    # Flatten reflectivity images from 3-D to 2-D (by convolving over height).
    if l2_weight is None:
        regularizer_object = None
    else:
        regularizer_object = architecture_utils.get_weight_regularizer(
            l1_penalty=0, l2_penalty=l2_weight)

    refl_input_layer_object = keras.layers.Input(
        shape=(num_reflectivity_rows, num_reflectivity_columns,
               num_reflectivity_heights, 1)
    )

    reflectivity_layer_object = None
    this_num_refl_filters = init_num_reflectivity_filters + 0

    for _ in range(3):
        if reflectivity_layer_object is None:
            this_target_shape = (
                num_reflectivity_rows * num_reflectivity_columns,
                num_reflectivity_heights, 1
            )
            reflectivity_layer_object = keras.layers.Reshape(
                target_shape=this_target_shape
            )(refl_input_layer_object)
        else:
            this_num_refl_filters *= 2

        for _ in range(num_conv_layers_per_set):
            reflectivity_layer_object = architecture_utils.get_2d_conv_layer(
                num_output_filters=this_num_refl_filters,
                num_kernel_rows=1, num_kernel_columns=3, num_rows_per_stride=1,
                num_columns_per_stride=1,
                padding_type=architecture_utils.YES_PADDING_TYPE,
                kernel_weight_regularizer=regularizer_object,
                is_first_layer=False
            )(reflectivity_layer_object)

            reflectivity_layer_object = architecture_utils.get_activation_layer(
                activation_function_string=conv_activation_function_string,
                alpha_for_elu=alpha_for_elu, alpha_for_relu=alpha_for_relu
            )(reflectivity_layer_object)

            if use_batch_normalization:
                reflectivity_layer_object = (
                    architecture_utils.get_batch_normalization_layer()(
                        reflectivity_layer_object)
                )

            if conv_layer_dropout_fraction is not None:
                reflectivity_layer_object = (
                    architecture_utils.get_dropout_layer(
                        dropout_fraction=conv_layer_dropout_fraction
                    )(reflectivity_layer_object)
                )

        reflectivity_layer_object = architecture_utils.get_2d_pooling_layer(
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
               num_azimuthal_shear_fields)
    )

    azimuthal_shear_layer_object = None
    for _ in range(num_conv_layers_per_set):
        if azimuthal_shear_layer_object is None:
            azimuthal_shear_layer_object = az_shear_input_layer_object

        azimuthal_shear_layer_object = architecture_utils.get_2d_conv_layer(
            num_output_filters=init_num_shear_filters,
            num_kernel_rows=3, num_kernel_columns=3, num_rows_per_stride=1,
            num_columns_per_stride=1,
            padding_type=architecture_utils.YES_PADDING_TYPE,
            kernel_weight_regularizer=regularizer_object, is_first_layer=False
        )(azimuthal_shear_layer_object)

        azimuthal_shear_layer_object = architecture_utils.get_activation_layer(
            activation_function_string=conv_activation_function_string,
            alpha_for_elu=alpha_for_elu, alpha_for_relu=alpha_for_relu
        )(azimuthal_shear_layer_object)

        if use_batch_normalization:
            azimuthal_shear_layer_object = (
                architecture_utils.get_batch_normalization_layer()(
                    azimuthal_shear_layer_object)
            )

        if conv_layer_dropout_fraction is not None:
            azimuthal_shear_layer_object = architecture_utils.get_dropout_layer(
                dropout_fraction=conv_layer_dropout_fraction
            )(azimuthal_shear_layer_object)

    azimuthal_shear_layer_object = architecture_utils.get_2d_pooling_layer(
        num_rows_in_window=2, num_columns_in_window=2,
        pooling_type=pooling_type_string, num_rows_per_stride=2,
        num_columns_per_stride=2
    )(azimuthal_shear_layer_object)

    # Concatenate reflectivity and az-shear filters.
    radar_layer_object = keras.layers.concatenate(
        [reflectivity_layer_object, azimuthal_shear_layer_object], axis=-1)

    # Convolve over reflectivity and az-shear filters.
    this_num_output_filters = 2 * (
        init_num_shear_filters + this_num_refl_filters)

    for i in range(num_radar_conv_layer_sets):
        if i != 0:
            this_num_output_filters *= 2

        for _ in range(num_conv_layers_per_set):
            radar_layer_object = architecture_utils.get_2d_conv_layer(
                num_output_filters=this_num_output_filters, num_kernel_rows=3,
                num_kernel_columns=3, num_rows_per_stride=1,
                num_columns_per_stride=1,
                padding_type=architecture_utils.YES_PADDING_TYPE,
                kernel_weight_regularizer=regularizer_object
            )(radar_layer_object)

            radar_layer_object = architecture_utils.get_activation_layer(
                activation_function_string=conv_activation_function_string,
                alpha_for_elu=alpha_for_elu, alpha_for_relu=alpha_for_relu
            )(radar_layer_object)

            if use_batch_normalization:
                radar_layer_object = (
                    architecture_utils.get_batch_normalization_layer()(
                        radar_layer_object)
                )

            if conv_layer_dropout_fraction is not None:
                radar_layer_object = architecture_utils.get_dropout_layer(
                    dropout_fraction=conv_layer_dropout_fraction
                )(radar_layer_object)

        radar_layer_object = architecture_utils.get_2d_pooling_layer(
            num_rows_in_window=2, num_columns_in_window=2,
            pooling_type=pooling_type_string, num_rows_per_stride=2,
            num_columns_per_stride=2
        )(radar_layer_object)

    radar_layer_object = architecture_utils.get_flattening_layer()(
        radar_layer_object)

    # Convolve over soundings.
    if num_sounding_fields is None:
        layer_object = radar_layer_object
    else:
        (sounding_input_layer_object, sounding_layer_object
        ) = _get_sounding_layers(
            num_sounding_heights=num_sounding_heights,
            num_sounding_fields=num_sounding_fields,
            init_num_filters=init_num_sounding_filters,
            num_conv_layer_sets=num_sounding_conv_layer_sets,
            num_conv_layers_per_set=num_conv_layers_per_set,
            pooling_type_string=pooling_type_string,
            dropout_fraction=conv_layer_dropout_fraction,
            regularizer_object=regularizer_object,
            conv_activation_function_string=conv_activation_function_string,
            alpha_for_elu=alpha_for_elu, alpha_for_relu=alpha_for_relu,
            use_batch_normalization=use_batch_normalization)

        layer_object = keras.layers.concatenate(
            [radar_layer_object, sounding_layer_object])

    (dense_layer_object, activation_layer_object, loss_function
    ) = _get_output_layer_and_loss_function(num_classes)
    layer_object = dense_layer_object(layer_object)

    if use_batch_normalization:
        layer_object = architecture_utils.get_batch_normalization_layer()(
            layer_object)
    if dense_layer_dropout_fraction is not None:
        layer_object = architecture_utils.get_dropout_layer(
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
        metrics=list_of_metric_functions)

    model_object.summary()
    return model_object
