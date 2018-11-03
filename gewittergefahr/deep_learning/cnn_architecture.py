"""Methods for creating a CNN (building the architecture)."""

import copy
import numpy
import keras.layers
from gewittergefahr.gg_utils import error_checking
from gewittergefahr.deep_learning import architecture_utils
from gewittergefahr.deep_learning import keras_metrics

VALID_NUMBERS_OF_SOUNDING_HEIGHTS = numpy.array([49], dtype=int)

NUM_ROWS_KEY = 'num_rows'
NUM_COLUMNS_KEY = 'num_columns'
NUM_DIMENSIONS_KEY = 'num_dimensions'
NUM_CHANNELS_KEY = 'num_channels'
NUM_FIELDS_KEY = 'num_fields'
NUM_HEIGHTS_KEY = 'num_heights'
NUM_CLASSES_KEY = 'num_classes'

NUM_CONV_LAYER_SETS_KEY = 'num_conv_layer_sets'
NUM_CONV_LAYERS_PER_SET_KEY = 'num_conv_layers_per_set'
NUM_DENSE_LAYERS_KEY = 'num_dense_layers'
ACTIVATION_FUNCTION_KEY = 'activation_function_string'
ALPHA_FOR_ELU_KEY = 'alpha_for_elu'
ALPHA_FOR_RELU_KEY = 'alpha_for_relu'
USE_BATCH_NORM_KEY = 'use_batch_normalization'
FIRST_NUM_FILTERS_KEY = 'first_num_filters'
CONV_LAYER_DROPOUT_KEY = 'conv_layer_dropout_fraction'
DENSE_LAYER_DROPOUT_KEY = 'dense_layer_dropout_fraction'
L2_WEIGHT_KEY = 'l2_weight'

DEFAULT_RADAR_OPTION_DICT = {
    NUM_CONV_LAYERS_PER_SET_KEY: 1,
    NUM_DENSE_LAYERS_KEY: 2,
    ACTIVATION_FUNCTION_KEY: architecture_utils.RELU_FUNCTION_STRING,
    ALPHA_FOR_ELU_KEY: architecture_utils.DEFAULT_ALPHA_FOR_ELU,
    ALPHA_FOR_RELU_KEY: architecture_utils.DEFAULT_ALPHA_FOR_RELU,
    USE_BATCH_NORM_KEY: True,
    FIRST_NUM_FILTERS_KEY: None,
    CONV_LAYER_DROPOUT_KEY: None,
    DENSE_LAYER_DROPOUT_KEY: 0.25,
    L2_WEIGHT_KEY: 0.001
}

DEFAULT_SOUNDING_OPTION_DICT = {
    NUM_CONV_LAYER_SETS_KEY: 3,
    FIRST_NUM_FILTERS_KEY: None
}

DEFAULT_METRIC_FUNCTION_LIST = [
    keras_metrics.accuracy, keras_metrics.binary_accuracy,
    keras_metrics.binary_csi, keras_metrics.binary_frequency_bias,
    keras_metrics.binary_pod, keras_metrics.binary_pofd,
    keras_metrics.binary_peirce_score, keras_metrics.binary_success_ratio,
    keras_metrics.binary_focn
]


def _get_sounding_layers(
        radar_option_dict, sounding_option_dict, pooling_type_string,
        regularizer_object):
    """Returns convolution and pooling layers for sounding data.

    :param radar_option_dict: See doc for `_check_radar_input_args`.
    :param sounding_option_dict: See doc for `_check_sounding_input_args`.
    :param pooling_type_string: Pooling type (must be in
        `architecture_utils.VALID_POOLING_TYPES`).
    :param regularizer_object: Instance of `keras.regularizers.l1_l2` (used for
        all convolution layers).
    :return: input_layer_object: Instance of `keras.layers.Input` (used to feed
        soundings into CNN).
    :return: flattening_layer_object: Instance of `keras.layers.Flatten`
        (returns scalar features created from soundings).
    :return: num_scalar_features: Number of scalar features (outputs of
        `flattening_layer_object`).
    """

    num_fields = sounding_option_dict[NUM_FIELDS_KEY]
    num_heights = sounding_option_dict[NUM_HEIGHTS_KEY]
    first_num_filters = sounding_option_dict[FIRST_NUM_FILTERS_KEY]
    num_conv_layer_sets = sounding_option_dict[NUM_CONV_LAYER_SETS_KEY]

    num_conv_layers_per_set = radar_option_dict[NUM_CONV_LAYERS_PER_SET_KEY]
    activation_function_string = radar_option_dict[ACTIVATION_FUNCTION_KEY]
    alpha_for_elu = radar_option_dict[ALPHA_FOR_ELU_KEY]
    alpha_for_relu = radar_option_dict[ALPHA_FOR_RELU_KEY]
    use_batch_normalization = radar_option_dict[USE_BATCH_NORM_KEY]
    dropout_fraction = radar_option_dict[CONV_LAYER_DROPOUT_KEY]

    input_layer_object = keras.layers.Input(shape=(num_heights, num_fields))

    sounding_layer_object = None
    this_num_filters = first_num_filters + 0

    for _ in range(num_conv_layer_sets):
        if sounding_layer_object is None:
            sounding_layer_object = input_layer_object
        else:
            this_num_filters *= 2

        for _ in range(num_conv_layers_per_set):
            sounding_layer_object = architecture_utils.get_1d_conv_layer(
                num_output_filters=this_num_filters,
                num_kernel_pixels=3, num_pixels_per_stride=1,
                padding_type=architecture_utils.NO_PADDING_TYPE,
                kernel_weight_regularizer=regularizer_object,
                is_first_layer=False
            )(sounding_layer_object)

            sounding_layer_object = architecture_utils.get_activation_layer(
                activation_function_string=activation_function_string,
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

    these_dimensions = numpy.array(
        sounding_layer_object.get_shape().as_list()[1:], dtype=int)
    num_scalar_features = numpy.prod(these_dimensions)

    sounding_layer_object = architecture_utils.get_flattening_layer()(
        sounding_layer_object)
    return input_layer_object, sounding_layer_object, num_scalar_features


def _get_output_layer_and_loss(num_classes):
    """Creates output layer and loss function.

    :param num_classes: Number of classes.
    :return: dense_layer_object: Output layer (instance of `keras.layers.Dense`
        with no activation function).
    :return: activation_layer_object: Activation layer (instance of
        `keras.layers.Activation`, `keras.layers.ELU`, or
        `keras.layers.LeakyReLU` -- to succeed the output layer).
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


def check_radar_input_args(option_dict):
    """Error-checks input arguments for radar data.

    :param option_dict: Dictionary with the following keys.
    option_dict['num_rows']: Number of rows in each radar image.
    option_dict['num_columns']: Number of columns in each radar image.
    option_dict['num_dimensions']: Number of dimensions (2 or 3) in each radar
        image.
    option_dict['num_conv_layer_sets']: Number of sets of convolution layers.
        Each conv-layer set is uninterrupted by pooling, so all conv layers in
        the same set process images at the same resolution.
    option_dict['num_conv_layers_per_set']: Number of convolution layers in each
        set.
    option_dict['num_dense_layers']: Number of dense layers at the end.
    option_dict['first_num_filters']: Number of filters produced by the first
        convolution layer.  This will double with each successive *set* of
        convolution layers.
    option_dict['num_classes']: Number of output classes (possible values of
        target variable).
    option_dict['activation_function_string']: Activation function for all
        layers except the output (i.e., all convolution layers and non-terminal
        dense layers).  Must be accepted by
        `architecture_utils.check_activation_function`.
    option_dict['use_batch_normalization']: Boolean flag.  If True, batch
        normalization will be applied to all layers except the output (i.e., all
        convolution layers and non-terminal dense layers).
    option_dict['alpha_for_elu']: Slope (alpha parameter) for eLU activation
        function.
    option_dict['alpha_for_relu']: Slope (alpha parameter) for ReLU activation
        function.
    option_dict['num_channels']: [used only if num_dimensions = 2]
        Number of channels (variables).
    option_dict['num_fields']: [used only if num_dimensions = 3]
        Number of radar fields.
    option_dict['num_heights']: [used only if num_dimensions = 3]
        Number of radar heights.
    option_dict['conv_layer_dropout_fraction']: Dropout fraction (applied to all
        convolution layers).
    option_dict['dense_layer_dropout_fraction']: Dropout fraction (applied to
        all dense layers).
    option_dict['l2_weight']: L2 regularization weight (applied to all
        convolution layers).

    :return: option_dict: Same as input, except that any missing default args
        have been set.
    """

    orig_option_dict = option_dict.copy()
    option_dict = DEFAULT_RADAR_OPTION_DICT.copy()
    option_dict.update(orig_option_dict)

    activation_function_string = option_dict[ACTIVATION_FUNCTION_KEY]
    alpha_for_elu = option_dict[ALPHA_FOR_ELU_KEY]
    alpha_for_relu = option_dict[ALPHA_FOR_RELU_KEY]
    use_batch_normalization = option_dict[USE_BATCH_NORM_KEY]

    architecture_utils.check_activation_function(
        activation_function_string=activation_function_string,
        alpha_for_elu=alpha_for_elu, alpha_for_relu=alpha_for_relu)
    error_checking.assert_is_boolean(use_batch_normalization)

    num_rows = option_dict[NUM_ROWS_KEY]
    num_columns = option_dict[NUM_COLUMNS_KEY]
    num_dimensions = option_dict[NUM_DIMENSIONS_KEY]

    error_checking.assert_is_integer(num_rows)
    error_checking.assert_is_geq(num_rows, 8)
    error_checking.assert_is_integer(num_columns)
    error_checking.assert_is_geq(num_columns, 8)
    error_checking.assert_is_integer(num_dimensions)
    error_checking.assert_is_geq(num_dimensions, 2)
    error_checking.assert_is_leq(num_dimensions, 3)

    if num_dimensions == 2:
        num_channels = option_dict[NUM_CHANNELS_KEY]
        error_checking.assert_is_integer(num_channels)
        error_checking.assert_is_geq(num_channels, 1)
    else:
        num_fields = option_dict[NUM_FIELDS_KEY]
        num_heights = option_dict[NUM_HEIGHTS_KEY]
        error_checking.assert_is_integer(num_fields)
        error_checking.assert_is_geq(num_fields, 1)
        error_checking.assert_is_integer(num_heights)
        error_checking.assert_is_geq(num_heights, 4)

    num_conv_layer_sets = option_dict[NUM_CONV_LAYER_SETS_KEY]
    num_conv_layers_per_set = option_dict[NUM_CONV_LAYERS_PER_SET_KEY]
    num_dense_layers = option_dict[NUM_DENSE_LAYERS_KEY]

    if option_dict[FIRST_NUM_FILTERS_KEY] is None:
        if num_dimensions == 2:
            option_dict[FIRST_NUM_FILTERS_KEY] = 8 * num_channels
        else:
            option_dict[FIRST_NUM_FILTERS_KEY] = 8 * num_fields

    first_num_filters = option_dict[FIRST_NUM_FILTERS_KEY]

    error_checking.assert_is_integer(num_conv_layer_sets)
    error_checking.assert_is_geq(num_conv_layer_sets, 1)
    error_checking.assert_is_integer(num_conv_layers_per_set)
    error_checking.assert_is_geq(num_conv_layers_per_set, 1)
    error_checking.assert_is_integer(num_dense_layers)
    error_checking.assert_is_geq(num_dense_layers, 1)
    error_checking.assert_is_integer(first_num_filters)
    error_checking.assert_is_geq(first_num_filters, 4)

    num_classes = option_dict[NUM_CLASSES_KEY]
    conv_layer_dropout_fraction = option_dict[CONV_LAYER_DROPOUT_KEY]
    dense_layer_dropout_fraction = option_dict[DENSE_LAYER_DROPOUT_KEY]
    l2_weight = option_dict[L2_WEIGHT_KEY]

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

    return option_dict


def check_sounding_input_args(option_dict):
    """Error-checks input arguments for sounding data.

    :param option_dict: Dictionary with the following keys.
    option_dict['num_fields']: Number of sounding fields.
    option_dict['num_heights']: Number of sounding heights.
    option_dict['num_conv_layer_sets']: Number of sets of convolution layers.
        Each conv-layer set is uninterrupted by pooling, so all conv layers in
        the same set process images at the same resolution.
    option_dict['first_num_filters']: Number of filters produced by the first
        convolution layer.  This will double with each successive *set* of
        convolution layers.

    :return: option_dict: Same as input, except that any missing default args
        have been set.
    :raises: ValueError: if
        `num_sounding_heights not in VALID_NUMBERS_OF_SOUNDING_HEIGHTS`.
    """

    orig_option_dict = option_dict.copy()
    option_dict = DEFAULT_SOUNDING_OPTION_DICT.copy()
    option_dict.update(orig_option_dict)

    num_conv_layer_sets = option_dict[NUM_CONV_LAYER_SETS_KEY]
    num_fields = option_dict[NUM_FIELDS_KEY]
    num_heights = option_dict[NUM_HEIGHTS_KEY]

    if option_dict[FIRST_NUM_FILTERS_KEY] is None:
        option_dict[FIRST_NUM_FILTERS_KEY] = 8 * num_fields
    first_num_filters = option_dict[FIRST_NUM_FILTERS_KEY]

    error_checking.assert_is_integer(num_conv_layer_sets)
    error_checking.assert_is_geq(num_conv_layer_sets, 2)
    error_checking.assert_is_integer(first_num_filters)
    error_checking.assert_is_geq(first_num_filters, 4)
    error_checking.assert_is_integer(num_fields)
    error_checking.assert_is_geq(num_fields, 1)
    error_checking.assert_is_integer(num_heights)

    if num_heights not in VALID_NUMBERS_OF_SOUNDING_HEIGHTS:
        error_string = (
            '\n\n{0:s}\nValid numbers of sounding heights (listed above) do'
            ' not include {1:d}.'
        ).format(str(VALID_NUMBERS_OF_SOUNDING_HEIGHTS), num_heights)
        raise ValueError(error_string)

    return option_dict


def get_2d_mnist_architecture(
        radar_option_dict,
        list_of_metric_functions=DEFAULT_METRIC_FUNCTION_LIST):
    """Creates CNN for 2-D radar images.

    This architecture is inspired by the following example:

    https://github.com/keras-team/keras/blob/master/examples/mnist_cnn.py

    :param radar_option_dict: See doc for `_check_radar_input_args`.
    :param list_of_metric_functions: 1-D list of performance metrics (will be
        printed to the log file after each batch).  Each metric must take one
        target tensor and one prediction tensor and return a scalar.
    :return: model_object: Untrained instance of `keras.models.Sequential` with
        the aforementioned architecture.
    """

    radar_option_dict[NUM_DIMENSIONS_KEY] = 2
    radar_option_dict[NUM_CLASSES_KEY] = 2
    radar_option_dict[NUM_CONV_LAYER_SETS_KEY] = 2
    radar_option_dict[NUM_CONV_LAYERS_PER_SET_KEY] = 1
    radar_option_dict[NUM_DENSE_LAYERS_KEY] = 2
    radar_option_dict[
        ACTIVATION_FUNCTION_KEY] = architecture_utils.RELU_FUNCTION_STRING
    radar_option_dict[
        ALPHA_FOR_RELU_KEY] = architecture_utils.DEFAULT_ALPHA_FOR_RELU

    radar_option_dict = check_radar_input_args(radar_option_dict)

    l2_weight = radar_option_dict[L2_WEIGHT_KEY]
    first_num_filters = radar_option_dict[FIRST_NUM_FILTERS_KEY]
    num_rows = radar_option_dict[NUM_ROWS_KEY]
    num_columns = radar_option_dict[NUM_COLUMNS_KEY]
    num_channels = radar_option_dict[NUM_CHANNELS_KEY]
    activation_function_string = radar_option_dict[ACTIVATION_FUNCTION_KEY]
    alpha_for_relu = radar_option_dict[ALPHA_FOR_RELU_KEY]
    num_classes = radar_option_dict[NUM_CLASSES_KEY]

    if l2_weight is None:
        regularizer_object = None
    else:
        regularizer_object = architecture_utils.get_weight_regularizer(
            l1_penalty=0, l2_penalty=l2_weight)

    model_object = keras.models.Sequential()
    layer_object = architecture_utils.get_2d_conv_layer(
        num_output_filters=first_num_filters, num_kernel_rows=3,
        num_kernel_columns=3, num_rows_per_stride=1, num_columns_per_stride=1,
        padding_type=architecture_utils.NO_PADDING_TYPE,
        kernel_weight_regularizer=regularizer_object, is_first_layer=True,
        num_input_rows=num_rows, num_input_columns=num_columns,
        num_input_channels=num_channels)
    model_object.add(layer_object)

    activation_layer_object = architecture_utils.get_activation_layer(
        activation_function_string=activation_function_string,
        alpha_for_relu=alpha_for_relu)
    model_object.add(activation_layer_object)

    layer_object = architecture_utils.get_2d_conv_layer(
        num_output_filters=2 * first_num_filters, num_kernel_rows=3,
        num_kernel_columns=3, num_rows_per_stride=1, num_columns_per_stride=1,
        padding_type=architecture_utils.NO_PADDING_TYPE,
        kernel_weight_regularizer=regularizer_object)
    model_object.add(layer_object)
    model_object.add(activation_layer_object)

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
    model_object.add(activation_layer_object)

    layer_object = architecture_utils.get_dropout_layer(dropout_fraction=0.5)
    model_object.add(layer_object)

    dense_layer_object, activation_layer_object, loss_function = (
        _get_output_layer_and_loss(num_classes)
    )
    model_object.add(dense_layer_object)
    model_object.add(activation_layer_object)

    model_object.compile(
        loss=loss_function, optimizer=keras.optimizers.Adam(),
        metrics=list_of_metric_functions)

    model_object.summary()
    return model_object


def get_2d_swirlnet_architecture(
        radar_option_dict, sounding_option_dict=None,
        pooling_type_string=architecture_utils.MAX_POOLING_TYPE,
        list_of_metric_functions=DEFAULT_METRIC_FUNCTION_LIST):
    """Creates CNN for 2-D radar images (and possibly soundings).

    This architecture is inspired by the following example:

    https://github.com/djgagne/swirlnet/blob/master/notebooks/
    deep_swirl_tutorial.ipynb

    :param radar_option_dict: See doc for `check_radar_input_args`.
    :param sounding_option_dict: See doc for `check_sounding_input_args`.  If
        you do not want soundings, leave this as None.
    :param pooling_type_string: Pooling type (must be in
        `architecture_utils.VALID_POOLING_TYPES`).
    :param list_of_metric_functions: See doc for `get_2d_mnist_architecture`.
    :return: model_object: Untrained instance of `keras.models.Sequential` with
        the aforementioned architecture.
    """

    radar_option_dict[NUM_DIMENSIONS_KEY] = 2
    radar_option_dict = check_radar_input_args(radar_option_dict)

    l2_weight = radar_option_dict[L2_WEIGHT_KEY]
    num_radar_rows = radar_option_dict[NUM_ROWS_KEY]
    num_radar_columns = radar_option_dict[NUM_COLUMNS_KEY]
    num_radar_channels = radar_option_dict[NUM_CHANNELS_KEY]
    first_num_radar_filters = radar_option_dict[FIRST_NUM_FILTERS_KEY]
    num_radar_conv_layer_sets = radar_option_dict[NUM_CONV_LAYER_SETS_KEY]
    num_conv_layers_per_set = radar_option_dict[NUM_CONV_LAYERS_PER_SET_KEY]
    num_dense_layers = radar_option_dict[NUM_DENSE_LAYERS_KEY]

    activation_function_string = radar_option_dict[ACTIVATION_FUNCTION_KEY]
    alpha_for_elu = radar_option_dict[ALPHA_FOR_ELU_KEY]
    alpha_for_relu = radar_option_dict[ALPHA_FOR_RELU_KEY]
    use_batch_normalization = radar_option_dict[USE_BATCH_NORM_KEY]
    conv_layer_dropout_fraction = radar_option_dict[CONV_LAYER_DROPOUT_KEY]
    dense_layer_dropout_fraction = radar_option_dict[DENSE_LAYER_DROPOUT_KEY]
    num_classes = radar_option_dict[NUM_CLASSES_KEY]

    if l2_weight is None:
        regularizer_object = None
    else:
        regularizer_object = architecture_utils.get_weight_regularizer(
            l1_penalty=0, l2_penalty=l2_weight)

    radar_input_layer_object = keras.layers.Input(
        shape=(num_radar_rows, num_radar_columns, num_radar_channels))

    radar_layer_object = None
    this_num_filters = first_num_radar_filters + 0

    for _ in range(num_radar_conv_layer_sets):
        if radar_layer_object is None:
            radar_layer_object = radar_input_layer_object
        else:
            this_num_filters *= 2

        for _ in range(num_conv_layers_per_set):
            radar_layer_object = architecture_utils.get_2d_conv_layer(
                num_output_filters=this_num_filters,
                num_kernel_rows=5, num_kernel_columns=5, num_rows_per_stride=1,
                num_columns_per_stride=1,
                padding_type=architecture_utils.NO_PADDING_TYPE,
                kernel_weight_regularizer=regularizer_object,
                is_first_layer=False
            )(radar_layer_object)

            radar_layer_object = architecture_utils.get_activation_layer(
                activation_function_string=activation_function_string,
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

    these_dimensions = numpy.array(
        radar_layer_object.get_shape().as_list()[1:], dtype=int)
    num_radar_features = numpy.prod(these_dimensions)

    radar_layer_object = architecture_utils.get_flattening_layer()(
        radar_layer_object)

    if sounding_option_dict is None:
        layer_object = radar_layer_object
        num_sounding_features = 0
    else:
        sounding_option_dict = check_sounding_input_args(sounding_option_dict)
        (sounding_input_layer_object, sounding_layer_object,
         num_sounding_features
        ) = _get_sounding_layers(
            radar_option_dict=radar_option_dict,
            sounding_option_dict=sounding_option_dict,
            pooling_type_string=pooling_type_string,
            regularizer_object=regularizer_object)

        layer_object = keras.layers.concatenate(
            [radar_layer_object, sounding_layer_object])

    _, num_outputs_by_dense_layer = (
        architecture_utils.get_dense_layer_dimensions(
            num_input_units=num_radar_features + num_sounding_features,
            num_classes=num_classes, num_dense_layers=num_dense_layers)
    )

    for k in range(num_dense_layers - 1):
        layer_object = architecture_utils.get_fully_connected_layer(
            num_output_units=num_outputs_by_dense_layer[k]
        )(layer_object)

        layer_object = architecture_utils.get_activation_layer(
            activation_function_string=activation_function_string,
            alpha_for_elu=alpha_for_elu, alpha_for_relu=alpha_for_relu
        )(layer_object)

        if use_batch_normalization:
            layer_object = architecture_utils.get_batch_normalization_layer()(
                layer_object)

        if dense_layer_dropout_fraction is not None:
            layer_object = architecture_utils.get_dropout_layer(
                dropout_fraction=dense_layer_dropout_fraction
            )(layer_object)

    dense_layer_object, activation_layer_object, loss_function = (
        _get_output_layer_and_loss(num_classes)
    )
    layer_object = dense_layer_object(layer_object)
    layer_object = activation_layer_object(layer_object)

    if dense_layer_dropout_fraction is not None:
        layer_object = architecture_utils.get_dropout_layer(
            dropout_fraction=dense_layer_dropout_fraction
        )(layer_object)

    if sounding_option_dict is None:
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
        radar_option_dict, sounding_option_dict=None,
        pooling_type_string=architecture_utils.MAX_POOLING_TYPE,
        list_of_metric_functions=DEFAULT_METRIC_FUNCTION_LIST):
    """Creates CNN for 3-D radar images (and possibly soundings).

    This architecture is inspired by the following example:

    https://github.com/djgagne/swirlnet/blob/master/notebooks/
    deep_swirl_tutorial.ipynb

    :param radar_option_dict: See doc for `check_radar_input_args`.
    :param sounding_option_dict: See doc for `check_sounding_input_args`.  If
        you do not want soundings, leave this as None.
    :param pooling_type_string: Pooling type (must be in
        `architecture_utils.VALID_POOLING_TYPES`).
    :param list_of_metric_functions: See doc for `get_2d_mnist_architecture`.
    :return: model_object: Untrained instance of `keras.models.Sequential` with
        the aforementioned architecture.
    """

    radar_option_dict[NUM_DIMENSIONS_KEY] = 3
    radar_option_dict = check_radar_input_args(radar_option_dict)

    l2_weight = radar_option_dict[L2_WEIGHT_KEY]
    num_radar_rows = radar_option_dict[NUM_ROWS_KEY]
    num_radar_columns = radar_option_dict[NUM_COLUMNS_KEY]
    num_radar_heights = radar_option_dict[NUM_HEIGHTS_KEY]
    num_radar_fields = radar_option_dict[NUM_FIELDS_KEY]
    first_num_radar_filters = radar_option_dict[FIRST_NUM_FILTERS_KEY]
    num_radar_conv_layer_sets = radar_option_dict[NUM_CONV_LAYER_SETS_KEY]
    num_conv_layers_per_set = radar_option_dict[NUM_CONV_LAYERS_PER_SET_KEY]
    num_dense_layers = radar_option_dict[NUM_DENSE_LAYERS_KEY]

    activation_function_string = radar_option_dict[ACTIVATION_FUNCTION_KEY]
    alpha_for_elu = radar_option_dict[ALPHA_FOR_ELU_KEY]
    alpha_for_relu = radar_option_dict[ALPHA_FOR_RELU_KEY]
    use_batch_normalization = radar_option_dict[USE_BATCH_NORM_KEY]
    conv_layer_dropout_fraction = radar_option_dict[CONV_LAYER_DROPOUT_KEY]
    dense_layer_dropout_fraction = radar_option_dict[DENSE_LAYER_DROPOUT_KEY]
    num_classes = radar_option_dict[NUM_CLASSES_KEY]

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
    this_num_filters = first_num_radar_filters + 0

    for _ in range(num_radar_conv_layer_sets):
        if radar_layer_object is None:
            radar_layer_object = radar_input_layer_object
        else:
            this_num_filters *= 2

        for _ in range(num_conv_layers_per_set):
            radar_layer_object = architecture_utils.get_3d_conv_layer(
                num_output_filters=this_num_filters,
                num_kernel_rows=3, num_kernel_columns=3, num_kernel_depths=3,
                num_rows_per_stride=1, num_columns_per_stride=1,
                num_depths_per_stride=1,
                padding_type=architecture_utils.NO_PADDING_TYPE,
                kernel_weight_regularizer=regularizer_object,
                is_first_layer=False
            )(radar_layer_object)

            radar_layer_object = architecture_utils.get_activation_layer(
                activation_function_string=activation_function_string,
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

    these_dimensions = numpy.array(
        radar_layer_object.get_shape().as_list()[1:], dtype=int)
    num_radar_features = numpy.prod(these_dimensions)

    radar_layer_object = architecture_utils.get_flattening_layer()(
        radar_layer_object)

    if sounding_option_dict is None:
        layer_object = radar_layer_object
        num_sounding_features = 0
    else:
        sounding_option_dict = check_sounding_input_args(sounding_option_dict)
        (sounding_input_layer_object, sounding_layer_object,
         num_sounding_features
        ) = _get_sounding_layers(
            radar_option_dict=radar_option_dict,
            sounding_option_dict=sounding_option_dict,
            pooling_type_string=pooling_type_string,
            regularizer_object=regularizer_object)

        layer_object = keras.layers.concatenate(
            [radar_layer_object, sounding_layer_object])

    _, num_outputs_by_dense_layer = (
        architecture_utils.get_dense_layer_dimensions(
            num_input_units=num_radar_features + num_sounding_features,
            num_classes=num_classes, num_dense_layers=num_dense_layers)
    )

    for k in range(num_dense_layers - 1):
        layer_object = architecture_utils.get_fully_connected_layer(
            num_output_units=num_outputs_by_dense_layer[k]
        )(layer_object)

        layer_object = architecture_utils.get_activation_layer(
            activation_function_string=activation_function_string,
            alpha_for_elu=alpha_for_elu, alpha_for_relu=alpha_for_relu
        )(layer_object)

        if use_batch_normalization:
            layer_object = architecture_utils.get_batch_normalization_layer()(
                layer_object)

        if dense_layer_dropout_fraction is not None:
            layer_object = architecture_utils.get_dropout_layer(
                dropout_fraction=dense_layer_dropout_fraction
            )(layer_object)

    dense_layer_object, activation_layer_object, loss_function = (
        _get_output_layer_and_loss(num_classes)
    )
    layer_object = dense_layer_object(layer_object)
    layer_object = activation_layer_object(layer_object)

    if dense_layer_dropout_fraction is not None:
        layer_object = architecture_utils.get_dropout_layer(
            dropout_fraction=dense_layer_dropout_fraction
        )(layer_object)

    if sounding_option_dict is None:
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
        reflectivity_option_dict, num_azimuthal_shear_fields,
        num_final_conv_layer_sets, first_num_az_shear_filters=None,
        sounding_option_dict=None,
        pooling_type_string=architecture_utils.MAX_POOLING_TYPE,
        list_of_metric_functions=DEFAULT_METRIC_FUNCTION_LIST):
    """Creates CNN for 2-D and 3-D radar images (and possibly soundings).

    The 2-D radar images contain azimuthal shear, and the 3-D images contain
    reflectivity.

    This architecture is inspired by the following example:

    https://github.com/djgagne/swirlnet/blob/master/notebooks/
    deep_swirl_tutorial.ipynb

    :param reflectivity_option_dict: See doc for `check_radar_input_args`.
    :param num_azimuthal_shear_fields: Number of azimuthal-shear fields.
    :param num_final_conv_layer_sets: Number of convolution-layer sets for
        combined filters (after reflectivity and az-shear filters have been
        concatenated).
    :param first_num_az_shear_filters: Number of azimuthal-shear filters
        produced by first convolution layer.
    :param sounding_option_dict: See doc for `check_sounding_input_args`.  If
        you do not want soundings, leave this as None.
    :param pooling_type_string: Pooling type (must be in
        `architecture_utils.VALID_POOLING_TYPES`).
    :param list_of_metric_functions: See doc for `get_2d_mnist_architecture`.
    :return: model_object: Untrained instance of `keras.models.Sequential` with
        the aforementioned architecture.
    """

    error_checking.assert_is_integer(num_final_conv_layer_sets)
    error_checking.assert_is_greater(num_final_conv_layer_sets, 0)

    reflectivity_option_dict[NUM_DIMENSIONS_KEY] = 3
    reflectivity_option_dict[NUM_FIELDS_KEY] = 1
    reflectivity_option_dict[NUM_CONV_LAYER_SETS_KEY] = 2
    reflectivity_option_dict = check_radar_input_args(reflectivity_option_dict)

    az_shear_option_dict = copy.deepcopy(reflectivity_option_dict)
    az_shear_option_dict[NUM_ROWS_KEY] *= 2
    az_shear_option_dict[NUM_COLUMNS_KEY] *= 2
    az_shear_option_dict[NUM_DIMENSIONS_KEY] = 2
    az_shear_option_dict[NUM_CHANNELS_KEY] = num_azimuthal_shear_fields
    az_shear_option_dict[NUM_CONV_LAYER_SETS_KEY] = 1
    az_shear_option_dict[FIRST_NUM_FILTERS_KEY] = first_num_az_shear_filters
    az_shear_option_dict = check_radar_input_args(az_shear_option_dict)

    l2_weight = reflectivity_option_dict[L2_WEIGHT_KEY]
    num_reflectivity_rows = reflectivity_option_dict[NUM_ROWS_KEY]
    num_reflectivity_columns = reflectivity_option_dict[NUM_COLUMNS_KEY]
    num_reflectivity_heights = reflectivity_option_dict[NUM_HEIGHTS_KEY]
    first_num_refl_filters = reflectivity_option_dict[FIRST_NUM_FILTERS_KEY]
    num_refl_conv_layer_sets = reflectivity_option_dict[NUM_CONV_LAYER_SETS_KEY]
    num_conv_layers_per_set = reflectivity_option_dict[
        NUM_CONV_LAYERS_PER_SET_KEY]
    num_dense_layers = reflectivity_option_dict[NUM_DENSE_LAYERS_KEY]

    activation_function_string = reflectivity_option_dict[
        ACTIVATION_FUNCTION_KEY]
    alpha_for_elu = reflectivity_option_dict[ALPHA_FOR_ELU_KEY]
    alpha_for_relu = reflectivity_option_dict[ALPHA_FOR_RELU_KEY]
    use_batch_normalization = reflectivity_option_dict[USE_BATCH_NORM_KEY]
    conv_layer_dropout_fraction = reflectivity_option_dict[
        CONV_LAYER_DROPOUT_KEY]
    dense_layer_dropout_fraction = reflectivity_option_dict[
        DENSE_LAYER_DROPOUT_KEY]
    num_classes = reflectivity_option_dict[NUM_CLASSES_KEY]

    if l2_weight is None:
        regularizer_object = None
    else:
        regularizer_object = architecture_utils.get_weight_regularizer(
            l1_penalty=0, l2_penalty=l2_weight)

    # Convolve over reflectivity images (to get rid of height dimension).
    refl_input_layer_object = keras.layers.Input(
        shape=(num_reflectivity_rows, num_reflectivity_columns,
               num_reflectivity_heights, 1)
    )

    reflectivity_layer_object = None
    this_num_refl_filters = first_num_refl_filters + 0

    for _ in range(num_refl_conv_layer_sets):
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
                padding_type=architecture_utils.NO_PADDING_TYPE,
                kernel_weight_regularizer=regularizer_object,
                is_first_layer=False
            )(reflectivity_layer_object)

            reflectivity_layer_object = architecture_utils.get_activation_layer(
                activation_function_string=activation_function_string,
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

    # Convolve over azimuthal-shear images (to decrease resolution and make it
    # consistent with reflectivity).
    num_azimuthal_shear_rows = az_shear_option_dict[NUM_ROWS_KEY]
    num_azimuthal_shear_columns = az_shear_option_dict[NUM_COLUMNS_KEY]
    first_num_az_shear_filters = az_shear_option_dict[FIRST_NUM_FILTERS_KEY]

    az_shear_input_layer_object = keras.layers.Input(
        shape=(num_azimuthal_shear_rows, num_azimuthal_shear_columns,
               num_azimuthal_shear_fields)
    )

    azimuthal_shear_layer_object = None
    for _ in range(num_conv_layers_per_set):
        if azimuthal_shear_layer_object is None:
            azimuthal_shear_layer_object = az_shear_input_layer_object

        azimuthal_shear_layer_object = architecture_utils.get_2d_conv_layer(
            num_output_filters=first_num_az_shear_filters,
            num_kernel_rows=3, num_kernel_columns=3, num_rows_per_stride=1,
            num_columns_per_stride=1,
            padding_type=architecture_utils.YES_PADDING_TYPE,
            kernel_weight_regularizer=regularizer_object, is_first_layer=False
        )(azimuthal_shear_layer_object)

        azimuthal_shear_layer_object = architecture_utils.get_activation_layer(
            activation_function_string=activation_function_string,
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

    # Convolve over both reflectivity and az-shear filters.
    this_num_filters = 2 * (
        first_num_az_shear_filters + this_num_refl_filters)

    for i in range(num_final_conv_layer_sets):
        if i != 0:
            this_num_filters *= 2

        for _ in range(num_conv_layers_per_set):
            radar_layer_object = architecture_utils.get_2d_conv_layer(
                num_output_filters=this_num_filters, num_kernel_rows=3,
                num_kernel_columns=3, num_rows_per_stride=1,
                num_columns_per_stride=1,
                padding_type=architecture_utils.NO_PADDING_TYPE,
                kernel_weight_regularizer=regularizer_object
            )(radar_layer_object)

            radar_layer_object = architecture_utils.get_activation_layer(
                activation_function_string=activation_function_string,
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

    these_dimensions = numpy.array(
        radar_layer_object.get_shape().as_list()[1:], dtype=int)
    num_radar_features = numpy.prod(these_dimensions)

    radar_layer_object = architecture_utils.get_flattening_layer()(
        radar_layer_object)

    # Convolve over soundings.
    if sounding_option_dict is None:
        layer_object = radar_layer_object
        num_sounding_features = 0
    else:
        sounding_option_dict = check_sounding_input_args(sounding_option_dict)
        (sounding_input_layer_object, sounding_layer_object,
         num_sounding_features
        ) = _get_sounding_layers(
            radar_option_dict=reflectivity_option_dict,
            sounding_option_dict=sounding_option_dict,
            pooling_type_string=pooling_type_string,
            regularizer_object=regularizer_object)

        layer_object = keras.layers.concatenate(
            [radar_layer_object, sounding_layer_object])

    _, num_outputs_by_dense_layer = (
        architecture_utils.get_dense_layer_dimensions(
            num_input_units=num_radar_features + num_sounding_features,
            num_classes=num_classes, num_dense_layers=num_dense_layers)
    )

    for k in range(num_dense_layers - 1):
        layer_object = architecture_utils.get_fully_connected_layer(
            num_output_units=num_outputs_by_dense_layer[k]
        )(layer_object)

        layer_object = architecture_utils.get_activation_layer(
            activation_function_string=activation_function_string,
            alpha_for_elu=alpha_for_elu, alpha_for_relu=alpha_for_relu
        )(layer_object)

        if use_batch_normalization:
            layer_object = architecture_utils.get_batch_normalization_layer()(
                layer_object)

        if dense_layer_dropout_fraction is not None:
            layer_object = architecture_utils.get_dropout_layer(
                dropout_fraction=dense_layer_dropout_fraction
            )(layer_object)

    dense_layer_object, activation_layer_object, loss_function = (
        _get_output_layer_and_loss(num_classes)
    )
    layer_object = dense_layer_object(layer_object)
    layer_object = activation_layer_object(layer_object)

    if dense_layer_dropout_fraction is not None:
        layer_object = architecture_utils.get_dropout_layer(
            dropout_fraction=dense_layer_dropout_fraction
        )(layer_object)

    if sounding_option_dict is None:
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
