"""Methods for setting up, training, and applying upconvolution networks."""

import copy
import pickle
import os.path
import numpy
import keras
import netCDF4
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.gg_utils import error_checking
from gewittergefahr.deep_learning import cnn
from gewittergefahr.deep_learning import testing_io
from gewittergefahr.deep_learning import architecture_utils
from gewittergefahr.deep_learning import training_validation_io as trainval_io

PATHLESS_FILE_NAME_PREFIX = 'upconvnet_predictions'

CONV_FILTER_SIZE = 3
NUM_EX_PER_TESTING_BATCH = 1000

DEFAULT_L1_WEIGHT = 0.
DEFAULT_L2_WEIGHT = 0.001
DEFAULT_ALPHA_FOR_ELU = architecture_utils.DEFAULT_ALPHA_FOR_ELU
DEFAULT_ALPHA_FOR_RELU = architecture_utils.DEFAULT_ALPHA_FOR_RELU

DEFAULT_SMOOTHING_RADIUS_PX = 1.
DEFAULT_HALF_SMOOTHING_ROWS = 2
DEFAULT_HALF_SMOOTHING_COLUMNS = 2

PLATEAU_PATIENCE_EPOCHS = 3
PLATEAU_LEARNING_RATE_MULTIPLIER = 0.5
PLATEAU_COOLDOWN_EPOCHS = 0
EARLY_STOPPING_PATIENCE_EPOCHS = 15
MSE_PATIENCE = 0.005

CNN_FILE_KEY = 'cnn_file_name'
CNN_FEATURE_LAYER_KEY = 'cnn_feature_layer_name'
NUM_EPOCHS_KEY = 'num_epochs'
NUM_EXAMPLES_PER_BATCH_KEY = 'num_examples_per_batch'
NUM_TRAINING_BATCHES_KEY = 'num_training_batches_per_epoch'
TRAINING_FILES_KEY = 'training_example_file_names'
FIRST_TRAINING_TIME_KEY = 'first_training_time_unix_sec'
LAST_TRAINING_TIME_KEY = 'last_training_time_unix_sec'
NUM_VALIDATION_BATCHES_KEY = 'num_validation_batches_per_epoch'
VALIDATION_FILES_KEY = 'validation_example_file_names'
FIRST_VALIDATION_TIME_KEY = 'first_validation_time_unix_sec'
LAST_VALIDATION_TIME_KEY = 'last_validation_time_unix_sec'

METADATA_KEYS = [
    CNN_FILE_KEY, CNN_FEATURE_LAYER_KEY, NUM_EPOCHS_KEY,
    NUM_EXAMPLES_PER_BATCH_KEY, NUM_TRAINING_BATCHES_KEY, TRAINING_FILES_KEY,
    FIRST_TRAINING_TIME_KEY, LAST_TRAINING_TIME_KEY, NUM_VALIDATION_BATCHES_KEY,
    VALIDATION_FILES_KEY, FIRST_VALIDATION_TIME_KEY, LAST_VALIDATION_TIME_KEY
]

RECON_IMAGE_MATRIX_KEY = 'denorm_recon_radar_matrix'
MEAN_SQUARED_ERRORS_KEY = 'mse_by_example'
FULL_STORM_IDS_KEY = 'full_storm_id_strings'
STORM_TIMES_KEY = 'storm_times_unix_sec'
UPCONVNET_FILE_KEY = 'upconvnet_file_name'

EXAMPLE_DIMENSION_KEY = 'example'
ROW_DIMENSION_KEY = 'grid_row'
COLUMN_DIMENSION_KEY = 'grid_column'
HEIGHT_DIMENSION_KEY = 'grid_height'
CHANNEL_DIMENSION_KEY = 'channel'
ID_CHAR_DIMENSION_KEY = 'storm_id_char'


def create_smoothing_filter(
        num_channels, smoothing_radius_px=DEFAULT_SMOOTHING_RADIUS_PX,
        num_half_filter_rows=DEFAULT_HALF_SMOOTHING_ROWS,
        num_half_filter_columns=DEFAULT_HALF_SMOOTHING_COLUMNS):
    """Creates convolution filter for Gaussian smoothing.

    M = number of rows in filter
    N = number of columns in filter
    C = number of channels (or "variables" or "features") to smooth.  Each
        channel will be smoothed independently.

    :param num_channels: C in the above discussion.
    :param smoothing_radius_px: e-folding radius (pixels).
    :param num_half_filter_rows: Number of rows in one half of filter.  Total
        number of rows will be 2 * `num_half_filter_rows` + 1.
    :param num_half_filter_columns: Same but for columns.
    :return: weight_matrix: M-by-N-by-C-by-C numpy array of convolution weights.
    """

    error_checking.assert_is_integer(num_channels)
    error_checking.assert_is_greater(num_channels, 0)
    error_checking.assert_is_greater(smoothing_radius_px, 0.)
    error_checking.assert_is_integer(num_half_filter_rows)
    error_checking.assert_is_greater(num_half_filter_rows, 0)
    error_checking.assert_is_integer(num_half_filter_columns)
    error_checking.assert_is_greater(num_half_filter_columns, 0)

    num_filter_rows = 2 * num_half_filter_rows + 1
    num_filter_columns = 2 * num_half_filter_columns + 1

    row_offsets_unique = numpy.linspace(
        -num_half_filter_rows, num_half_filter_rows, num=num_filter_rows,
        dtype=float)
    column_offsets_unique = numpy.linspace(
        -num_half_filter_columns, num_half_filter_columns,
        num=num_filter_columns, dtype=float)

    column_offset_matrix, row_offset_matrix = numpy.meshgrid(
        column_offsets_unique, row_offsets_unique)

    pixel_offset_matrix = numpy.sqrt(
        row_offset_matrix ** 2 + column_offset_matrix ** 2)

    small_weight_matrix = numpy.exp(
        -pixel_offset_matrix ** 2 / (2 * smoothing_radius_px ** 2)
    )
    small_weight_matrix = small_weight_matrix / numpy.sum(small_weight_matrix)

    weight_matrix = numpy.zeros(
        (num_filter_rows, num_filter_columns, num_channels, num_channels)
    )

    for k in range(num_channels):
        weight_matrix[..., k, k] = small_weight_matrix

    return weight_matrix


def create_3d_net(
        num_input_features, first_spatial_dimensions, rowcol_upsampling_factors,
        height_upsampling_factors, num_output_channels,
        l1_weight=DEFAULT_L1_WEIGHT, l2_weight=DEFAULT_L2_WEIGHT,
        use_transposed_conv=True, activation_function_name=None,
        alpha_for_elu=DEFAULT_ALPHA_FOR_ELU,
        alpha_for_relu=DEFAULT_ALPHA_FOR_RELU,
        use_activn_for_last_layer=False,
        use_batch_norm=True, use_batch_norm_for_last_layer=True):
    """Creates (but does not train) upconvnet with 3 spatial dimensions.

    L = number of main (transposed-conv or upsampling) layers

    :param num_input_features: Length of input feature vector.
    :param first_spatial_dimensions: length-3 numpy array of dimensions in first
        main layer.  The order should be (num_rows, num_columns, num_heights).
        Before it is passed to the first main layer, the feature vector will be
        reshaped into a grid with these dimensions.
    :param rowcol_upsampling_factors: length-L numpy array of upsampling factors
        for horizontal dimensions.
    :param height_upsampling_factors: length-L numpy array of upsampling factors
        for vertical dimension.
    :param num_output_channels: Number of channels in output image.
    :param l1_weight: Weight of L1 regularization for conv and transposed-conv
        layers.
    :param l2_weight: Same but for L2 regularization.
    :param use_transposed_conv: Boolean flag.  If True, each upsampling will be
        done with a transposed-conv layer.  If False, each upsampling will be
        done with an upsampling layer followed by a normal conv layer.
    :param activation_function_name: Activation function.  If you do not want
        activation, make this None.  Otherwise, must be accepted by
        `architecture_utils.check_activation_function`.
    :param alpha_for_elu: See doc for
        `architecture_utils.check_activation_function`.
    :param alpha_for_relu: Same.
    :param use_activn_for_last_layer: Boolean flag.  If True, will apply
        activation function to output image.
    :param use_batch_norm: Boolean flag.  If True, will apply batch
        normalization to conv and transposed-conv layers.
    :param use_batch_norm_for_last_layer: Boolean flag.  If True, will apply
        batch normalization to output image.
    :return: model_object: Untrained model (instance of `keras.models.Model`).
    """

    # TODO(thunderhoser): This method assumes that the original CNN does
    # edge-padding.

    # Check input args.
    error_checking.assert_is_integer(num_input_features)
    error_checking.assert_is_greater(num_input_features, 0)
    error_checking.assert_is_integer(num_output_channels)
    error_checking.assert_is_greater(num_output_channels, 0)
    error_checking.assert_is_geq(l1_weight, 0.)
    error_checking.assert_is_geq(l2_weight, 0.)

    error_checking.assert_is_boolean(use_transposed_conv)
    error_checking.assert_is_boolean(use_activn_for_last_layer)
    error_checking.assert_is_boolean(use_batch_norm)
    error_checking.assert_is_boolean(use_batch_norm_for_last_layer)

    error_checking.assert_is_numpy_array(
        first_spatial_dimensions, exact_dimensions=numpy.array([3], dtype=int)
    )
    error_checking.assert_is_integer_numpy_array(first_spatial_dimensions)
    error_checking.assert_is_greater_numpy_array(first_spatial_dimensions, 0)

    error_checking.assert_is_numpy_array(
        rowcol_upsampling_factors, num_dimensions=1
    )
    error_checking.assert_is_integer_numpy_array(rowcol_upsampling_factors)
    error_checking.assert_is_geq_numpy_array(rowcol_upsampling_factors, 1)

    num_main_layers = len(rowcol_upsampling_factors)
    these_expected_dim = numpy.array([num_main_layers], dtype=int)

    error_checking.assert_is_numpy_array(
        height_upsampling_factors, exact_dimensions=these_expected_dim
    )
    error_checking.assert_is_integer_numpy_array(height_upsampling_factors)
    error_checking.assert_is_geq_numpy_array(height_upsampling_factors, 1)

    # Set up CNN architecture.
    regularizer_object = keras.regularizers.l1_l2(l1=l1_weight, l2=l2_weight)
    input_layer_object = keras.layers.Input(shape=(num_input_features,))

    current_num_filters = int(numpy.round(
        num_input_features / numpy.prod(first_spatial_dimensions)
    ))
    first_dimensions = numpy.concatenate((
        first_spatial_dimensions, numpy.array([current_num_filters], dtype=int)
    ))
    layer_object = keras.layers.Reshape(
        target_shape=first_dimensions
    )(input_layer_object)

    kernel_size_tuple = (CONV_FILTER_SIZE, CONV_FILTER_SIZE, CONV_FILTER_SIZE)

    for i in range(num_main_layers):
        if i == num_main_layers - 1:
            current_num_filters = num_output_channels + 0
        elif rowcol_upsampling_factors[i] == 1:
            current_num_filters = int(numpy.round(current_num_filters / 2))

        this_stride_tuple = (
            rowcol_upsampling_factors[i], rowcol_upsampling_factors[i],
            height_upsampling_factors[i]
        )

        if use_transposed_conv:
            layer_object = keras.layers.Conv3DTranspose(
                filters=current_num_filters, kernel_size=kernel_size_tuple,
                strides=this_stride_tuple, padding='same',
                data_format='channels_last', dilation_rate=(1, 1, 1),
                activation=None, use_bias=True,
                kernel_initializer='glorot_uniform', bias_initializer='zeros',
                kernel_regularizer=regularizer_object
            )(layer_object)
        else:
            if rowcol_upsampling_factors[i] > 1:
                try:
                    layer_object = keras.layers.UpSampling3D(
                        size=this_stride_tuple, data_format='channels_last',
                        interpolation='bilinear'
                    )(layer_object)
                except:
                    layer_object = keras.layers.UpSampling3D(
                        size=this_stride_tuple, data_format='channels_last'
                    )(layer_object)

            layer_object = keras.layers.Conv3D(
                filters=current_num_filters, kernel_size=kernel_size_tuple,
                strides=(1, 1, 1), padding='same', data_format='channels_last',
                dilation_rate=(1, 1, 1), activation=None, use_bias=True,
                kernel_initializer='glorot_uniform', bias_initializer='zeros',
                kernel_regularizer=regularizer_object
            )(layer_object)

        use_activation_here = (
            activation_function_name is not None and
            (i < num_main_layers - 1 or use_activn_for_last_layer)
        )

        if use_activation_here:
            layer_object = architecture_utils.get_activation_layer(
                activation_function_string=activation_function_name,
                alpha_for_elu=alpha_for_elu, alpha_for_relu=alpha_for_relu
            )(layer_object)

        use_batch_norm_here = (
            use_batch_norm and
            (i < num_main_layers - 1 or use_batch_norm_for_last_layer)
        )

        if use_batch_norm_here:
            layer_object = (
                architecture_utils.get_batch_norm_layer()(layer_object)
            )

    # Compile CNN.
    model_object = keras.models.Model(
        inputs=input_layer_object, outputs=layer_object)
    model_object.compile(
        loss=keras.losses.mean_squared_error, optimizer=keras.optimizers.Adam()
    )

    model_object.summary()
    return model_object


def create_2d_net(
        num_input_features, first_spatial_dimensions, upsampling_factors,
        num_output_channels,
        l1_weight=DEFAULT_L1_WEIGHT, l2_weight=DEFAULT_L2_WEIGHT,
        use_transposed_conv=True, activation_function_name=None,
        alpha_for_elu=DEFAULT_ALPHA_FOR_ELU,
        alpha_for_relu=DEFAULT_ALPHA_FOR_RELU,
        use_activn_for_last_layer=False,
        use_batch_norm=True, use_batch_norm_for_last_layer=True):
    """Creates (but does not train) upconvnet with 2 spatial dimensions.

    L = number of main (transposed-conv or upsampling) layers

    :param num_input_features: Length of input feature vector.
    :param first_spatial_dimensions: length-2 numpy array of dimensions in first
        main layer.  The order should be (num_rows, num_columns).  Before it is
        passed to the first main layer, the feature vector will be reshaped into
        a grid with these dimensions.
    :param upsampling_factors: length-L numpy array of upsampling factors.
    :param num_output_channels: See doc for `create_3d_net`.
    :param l1_weight: Same.
    :param l2_weight: Same.
    :param use_transposed_conv: Same.
    :param activation_function_name: Same.
    :param alpha_for_elu: Same.
    :param alpha_for_relu: Same.
    :param use_activn_for_last_layer: Same.
    :param use_batch_norm: Same.
    :param use_batch_norm_for_last_layer: Same.
    :return: model_object: Same.
    """

    # TODO(thunderhoser): This method assumes that the original CNN does
    # edge-padding.

    # Check input args.
    error_checking.assert_is_integer(num_input_features)
    error_checking.assert_is_greater(num_input_features, 0)
    error_checking.assert_is_integer(num_output_channels)
    error_checking.assert_is_greater(num_output_channels, 0)
    error_checking.assert_is_geq(l1_weight, 0.)
    error_checking.assert_is_geq(l2_weight, 0.)

    error_checking.assert_is_boolean(use_transposed_conv)
    error_checking.assert_is_boolean(use_activn_for_last_layer)
    error_checking.assert_is_boolean(use_batch_norm)
    error_checking.assert_is_boolean(use_batch_norm_for_last_layer)

    error_checking.assert_is_numpy_array(
        first_spatial_dimensions, exact_dimensions=numpy.array([2], dtype=int)
    )
    error_checking.assert_is_integer_numpy_array(first_spatial_dimensions)
    error_checking.assert_is_greater_numpy_array(first_spatial_dimensions, 0)

    error_checking.assert_is_numpy_array(upsampling_factors, num_dimensions=1)
    error_checking.assert_is_integer_numpy_array(upsampling_factors)
    error_checking.assert_is_geq_numpy_array(upsampling_factors, 1)

    # Set up CNN architecture.
    regularizer_object = keras.regularizers.l1_l2(l1=l1_weight, l2=l2_weight)
    input_layer_object = keras.layers.Input(shape=(num_input_features,))

    current_num_filters = int(numpy.round(
        num_input_features / numpy.prod(first_spatial_dimensions)
    ))
    first_dimensions = numpy.concatenate((
        first_spatial_dimensions, numpy.array([current_num_filters], dtype=int)
    ))
    layer_object = keras.layers.Reshape(
        target_shape=first_dimensions
    )(input_layer_object)

    num_main_layers = len(upsampling_factors)
    kernel_size_tuple = (CONV_FILTER_SIZE, CONV_FILTER_SIZE)

    for i in range(num_main_layers):
        if i == num_main_layers - 1:
            current_num_filters = num_output_channels + 0

            # layer_object = keras.layers.ZeroPadding2D(
            #     padding=((1, 0), (1, 0)), data_format='channels_last'
            # )(layer_object)

        elif upsampling_factors[i] == 1:
            current_num_filters = int(numpy.round(current_num_filters / 2))

        this_stride_tuple = (upsampling_factors[i], upsampling_factors[i])

        if use_transposed_conv:
            layer_object = keras.layers.Conv2DTranspose(
                filters=current_num_filters, kernel_size=kernel_size_tuple,
                strides=this_stride_tuple, padding='same',
                data_format='channels_last', dilation_rate=(1, 1),
                activation=None, use_bias=True,
                kernel_initializer='glorot_uniform', bias_initializer='zeros',
                kernel_regularizer=regularizer_object
            )(layer_object)
        else:
            if upsampling_factors[i] > 1:
                try:
                    layer_object = keras.layers.UpSampling2D(
                        size=this_stride_tuple, data_format='channels_last',
                        interpolation='bilinear'
                    )(layer_object)
                except:
                    layer_object = keras.layers.UpSampling2D(
                        size=this_stride_tuple, data_format='channels_last'
                    )(layer_object)

            layer_object = keras.layers.Conv2D(
                filters=current_num_filters, kernel_size=kernel_size_tuple,
                strides=(1, 1), padding='same', data_format='channels_last',
                dilation_rate=(1, 1), activation=None, use_bias=True,
                kernel_initializer='glorot_uniform', bias_initializer='zeros',
                kernel_regularizer=regularizer_object
            )(layer_object)

        use_activation_here = (
            activation_function_name is not None and
            (i < num_main_layers - 1 or use_activn_for_last_layer)
        )

        if use_activation_here:
            layer_object = architecture_utils.get_activation_layer(
                activation_function_string=activation_function_name,
                alpha_for_elu=alpha_for_elu, alpha_for_relu=alpha_for_relu
            )(layer_object)

        use_batch_norm_here = (
            use_batch_norm and
            (i < num_main_layers - 1 or use_batch_norm_for_last_layer)
        )

        if use_batch_norm_here:
            layer_object = (
                architecture_utils.get_batch_norm_layer()(layer_object)
            )

    # Compile CNN.
    model_object = keras.models.Model(
        inputs=input_layer_object, outputs=layer_object)
    model_object.compile(
        loss=keras.losses.mean_squared_error, optimizer=keras.optimizers.Adam()
    )

    model_object.summary()
    return model_object


def trainval_generator(cnn_model_object, cnn_metadata_dict,
                       cnn_feature_layer_name):
    """Generates training or validation examples for upconvnet.

    E = number of examples per batch
    Z = number of features in vector

    :param cnn_model_object: Trained CNN (instance of `keras.models.Model` or
        `keras.models.Sequential`), used to convert images to feature vectors.
        The upconvnet tries to convert feature vectors back to original images.
    :param cnn_metadata_dict: See doc for `cnn.read_model_metadata`.
    :param cnn_feature_layer_name: Name of feature-generating layer in CNN.  The
        feature vector will be the output of this layer.
    :return: feature_matrix: E-by-Z numpy array of scalar features.  Each row is
        the feature vector for one example.  These are inputs (predictors) for
        the upconvnet.
    :return: radar_matrix: numpy array of radar images.  These are targets
        (desired outputs) for the upconvnet.  If radar images have 3 spatial
        dimensions, this should be a 5-D array; if 2 spatial dimensions, this
        should be a 4-D array.
    """

    training_option_dict = cnn_metadata_dict[cnn.TRAINING_OPTION_DICT_KEY]
    training_option_dict[trainval_io.SHUFFLE_TARGET_KEY] = False
    training_option_dict[trainval_io.LOOP_ONCE_KEY] = False
    training_option_dict[trainval_io.X_TRANSLATIONS_KEY] = None
    training_option_dict[trainval_io.Y_TRANSLATIONS_KEY] = None
    training_option_dict[trainval_io.ROTATION_ANGLES_KEY] = None
    training_option_dict[trainval_io.NOISE_STDEV_KEY] = None
    training_option_dict[trainval_io.NUM_NOISINGS_KEY] = 0
    training_option_dict[trainval_io.FLIP_X_KEY] = False
    training_option_dict[trainval_io.FLIP_Y_KEY] = False

    partial_cnn_model_object = cnn.model_to_feature_generator(
        model_object=cnn_model_object,
        feature_layer_name=cnn_feature_layer_name)

    conv_2d3d = cnn_metadata_dict[cnn.CONV_2D3D_KEY]
    layer_operation_dicts = cnn_metadata_dict[cnn.LAYER_OPERATIONS_KEY]

    if conv_2d3d:
        cnn_generator = trainval_io.myrorss_generator_2d3d(training_option_dict)
    elif layer_operation_dicts is None:
        cnn_generator = trainval_io.generator_2d_or_3d(training_option_dict)
    else:
        cnn_generator = trainval_io.gridrad_generator_2d_reduced(
            option_dict=training_option_dict,
            list_of_operation_dicts=layer_operation_dicts)

    while True:
        try:
            these_image_matrices = next(cnn_generator)[0]
        except StopIteration:
            break

        if isinstance(these_image_matrices, list):
            radar_matrix = these_image_matrices[0]
        else:
            radar_matrix = these_image_matrices

        feature_matrix = partial_cnn_model_object.predict(
            these_image_matrices, batch_size=radar_matrix.shape[0]
        )

        yield (feature_matrix, radar_matrix)


def testing_generator(cnn_model_object, cnn_metadata_dict,
                      cnn_feature_layer_name, num_examples_total):
    """Generates testing examples for upconvnet.

    :param cnn_model_object: See doc for `trainval_generator`.
    :param cnn_metadata_dict: Same.
    :param cnn_feature_layer_name: Same.
    :param num_examples_total: Number of examples to read.
    :return: feature_matrix: See doc for `trainval_generator`.
    :return: radar_matrix: Same.
    """

    training_option_dict = copy.deepcopy(
        cnn_metadata_dict[cnn.TRAINING_OPTION_DICT_KEY]
    )
    training_option_dict[trainval_io.NUM_EXAMPLES_PER_BATCH_KEY] = (
        NUM_EX_PER_TESTING_BATCH
    )

    partial_cnn_model_object = cnn.model_to_feature_generator(
        model_object=cnn_model_object,
        feature_layer_name=cnn_feature_layer_name)

    conv_2d3d = cnn_metadata_dict[cnn.CONV_2D3D_KEY]
    layer_operation_dicts = cnn_metadata_dict[cnn.LAYER_OPERATIONS_KEY]

    if conv_2d3d:
        cnn_generator = testing_io.myrorss_generator_2d3d(
            option_dict=training_option_dict,
            desired_num_examples=num_examples_total)
    elif layer_operation_dicts is None:
        cnn_generator = testing_io.generator_2d_or_3d(
            option_dict=training_option_dict,
            desired_num_examples=num_examples_total)
    else:
        cnn_generator = testing_io.gridrad_generator_2d_reduced(
            option_dict=training_option_dict,
            list_of_operation_dicts=layer_operation_dicts,
            desired_num_examples=num_examples_total)

    while True:
        try:
            this_storm_object_dict = next(cnn_generator)
        except StopIteration:
            break

        these_image_matrices = this_storm_object_dict[
            testing_io.INPUT_MATRICES_KEY]

        if isinstance(these_image_matrices, list):
            radar_matrix = these_image_matrices[0]
        else:
            radar_matrix = these_image_matrices

        feature_matrix = partial_cnn_model_object.predict(
            these_image_matrices, batch_size=radar_matrix.shape[0]
        )

        yield (feature_matrix, radar_matrix)


def train_upconvnet(
        upconvnet_model_object, output_dir_name, cnn_model_object,
        cnn_metadata_dict, cnn_feature_layer_name, num_epochs,
        num_examples_per_batch, num_training_batches_per_epoch,
        training_example_file_names, first_training_time_unix_sec,
        last_training_time_unix_sec, num_validation_batches_per_epoch,
        validation_example_file_names, first_validation_time_unix_sec,
        last_validation_time_unix_sec):
    """Trains upconvnet.

    :param upconvnet_model_object: Upconvnet to be trained (instance of
        `keras.models.Model` or `keras.models.Sequential`).
    :param output_dir_name: Name of output directory (model and training history
        will be saved here).
    :param cnn_model_object: See doc for `trainval_generator`.
    :param cnn_metadata_dict: Same.
    :param cnn_feature_layer_name: Same.
    :param num_epochs: Number of epochs.
    :param num_examples_per_batch: Number of examples in each training or
        validation batch.
    :param num_training_batches_per_epoch: Number of training batches furnished
        to model in each epoch.
    :param training_example_file_names: 1-D list of paths to files with training
        examples (inputs to CNN).  These files will be read by
        `input_examples.read_example_file`.
    :param first_training_time_unix_sec: First time in training period.
    :param last_training_time_unix_sec: Last time in training period.
    :param num_validation_batches_per_epoch: Number of validation batches
        furnished to model in each epoch.
    :param validation_example_file_names: 1-D list of paths to files with
        validation examples (inputs to CNN).  These files will be read by
        `input_examples.read_example_file`.
    :param first_validation_time_unix_sec: First time in validation period.
    :param last_validation_time_unix_sec: Last time in validation period.
    """

    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=output_dir_name)

    upconvnet_file_name = '{0:s}/upconvnet_model.h5'.format(output_dir_name)
    history_file_name = '{0:s}/upconvnet_model_history.csv'.format(
        output_dir_name)

    training_metadata_dict = copy.deepcopy(cnn_metadata_dict)
    this_option_dict = training_metadata_dict[cnn.TRAINING_OPTION_DICT_KEY]

    this_option_dict[
        trainval_io.EXAMPLE_FILES_KEY] = training_example_file_names
    this_option_dict[
        trainval_io.NUM_EXAMPLES_PER_BATCH_KEY] = num_examples_per_batch
    this_option_dict[
        trainval_io.FIRST_STORM_TIME_KEY] = first_training_time_unix_sec
    this_option_dict[
        trainval_io.LAST_STORM_TIME_KEY] = last_training_time_unix_sec

    training_metadata_dict[cnn.TRAINING_OPTION_DICT_KEY] = this_option_dict
    training_generator = trainval_generator(
        cnn_model_object=cnn_model_object,
        cnn_metadata_dict=training_metadata_dict,
        cnn_feature_layer_name=cnn_feature_layer_name)

    validation_metadata_dict = copy.deepcopy(cnn_metadata_dict)
    this_option_dict = validation_metadata_dict[cnn.TRAINING_OPTION_DICT_KEY]

    this_option_dict[
        trainval_io.EXAMPLE_FILES_KEY] = validation_example_file_names
    this_option_dict[
        trainval_io.NUM_EXAMPLES_PER_BATCH_KEY] = num_examples_per_batch
    this_option_dict[
        trainval_io.FIRST_STORM_TIME_KEY] = first_validation_time_unix_sec
    this_option_dict[
        trainval_io.LAST_STORM_TIME_KEY] = last_validation_time_unix_sec

    validation_metadata_dict[cnn.TRAINING_OPTION_DICT_KEY] = this_option_dict
    validation_generator = trainval_generator(
        cnn_model_object=cnn_model_object,
        cnn_metadata_dict=validation_metadata_dict,
        cnn_feature_layer_name=cnn_feature_layer_name)

    history_object = keras.callbacks.CSVLogger(
        filename=history_file_name, separator=',', append=False)

    checkpoint_object = keras.callbacks.ModelCheckpoint(
        filepath=upconvnet_file_name, monitor='val_loss', verbose=1,
        save_best_only=True, save_weights_only=False, mode='min', period=1)

    early_stopping_object = keras.callbacks.EarlyStopping(
        monitor='val_loss', min_delta=MSE_PATIENCE,
        patience=EARLY_STOPPING_PATIENCE_EPOCHS, verbose=1, mode='min')

    plateau_object = keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=PLATEAU_LEARNING_RATE_MULTIPLIER,
        patience=PLATEAU_PATIENCE_EPOCHS, verbose=1, mode='min',
        min_delta=MSE_PATIENCE, cooldown=PLATEAU_COOLDOWN_EPOCHS)

    list_of_callback_objects = [
        history_object, checkpoint_object, early_stopping_object, plateau_object
    ]

    upconvnet_model_object.fit_generator(
        generator=training_generator,
        steps_per_epoch=num_training_batches_per_epoch, epochs=num_epochs,
        verbose=1, callbacks=list_of_callback_objects,
        validation_data=validation_generator,
        validation_steps=num_validation_batches_per_epoch, workers=0)


def write_model_metadata(
        cnn_file_name, cnn_feature_layer_name, num_epochs,
        num_examples_per_batch, num_training_batches_per_epoch,
        training_example_file_names, first_training_time_unix_sec,
        last_training_time_unix_sec, num_validation_batches_per_epoch,
        validation_example_file_names, first_validation_time_unix_sec,
        last_validation_time_unix_sec, pickle_file_name):
    """Writes metadata to Pickle file.

    :param cnn_file_name: Path to file with trained CNN (readable by
        `cnn.read_model`).
    :param cnn_feature_layer_name: See doc for `train_upconvnet`.
    :param num_epochs: Same.
    :param num_examples_per_batch: Same.
    :param num_training_batches_per_epoch: Same.
    :param training_example_file_names: Same.
    :param first_training_time_unix_sec: Same.
    :param last_training_time_unix_sec: Same.
    :param num_validation_batches_per_epoch: Same.
    :param validation_example_file_names: Same.
    :param first_validation_time_unix_sec: Same.
    :param last_validation_time_unix_sec: Same.
    :param pickle_file_name: Path to output file.

    :return: metadata_dict: Dictionary with the following keys.
    metadata_dict['cnn_file_name']: See doc for `read_model_metadata`.
    metadata_dict['cnn_feature_layer_name']: Same.
    metadata_dict['num_epochs']: Same.
    metadata_dict['num_examples_per_batch']: Same.
    metadata_dict['num_training_batches_per_epoch']: Same.
    metadata_dict['training_example_file_names']: Same.
    metadata_dict['first_training_time_unix_sec']: Same.
    metadata_dict['last_training_time_unix_sec']: Same.
    metadata_dict['num_validation_batches_per_epoch']: Same.
    metadata_dict['validation_example_file_names']: Same.
    metadata_dict['first_validation_time_unix_sec']: Same.
    metadata_dict['last_validation_time_unix_sec']: Same.
    """

    error_checking.assert_is_string(cnn_file_name)
    error_checking.assert_is_string(cnn_feature_layer_name)
    error_checking.assert_is_integer(num_epochs)
    error_checking.assert_is_greater(num_epochs, 0)
    error_checking.assert_is_integer(num_examples_per_batch)
    error_checking.assert_is_greater(num_examples_per_batch, 0)
    error_checking.assert_is_integer(num_training_batches_per_epoch)
    error_checking.assert_is_greater(num_training_batches_per_epoch, 0)
    error_checking.assert_is_integer(num_validation_batches_per_epoch)
    error_checking.assert_is_greater(num_validation_batches_per_epoch, 0)

    error_checking.assert_is_string_list(training_example_file_names)
    error_checking.assert_is_numpy_array(
        numpy.array(training_example_file_names), num_dimensions=1)

    error_checking.assert_is_integer(first_training_time_unix_sec)
    error_checking.assert_is_integer(last_training_time_unix_sec)
    error_checking.assert_is_greater(
        last_training_time_unix_sec, first_training_time_unix_sec)

    error_checking.assert_is_string_list(validation_example_file_names)
    error_checking.assert_is_numpy_array(
        numpy.array(validation_example_file_names), num_dimensions=1)

    error_checking.assert_is_integer(first_validation_time_unix_sec)
    error_checking.assert_is_integer(last_validation_time_unix_sec)
    error_checking.assert_is_greater(
        last_validation_time_unix_sec, first_validation_time_unix_sec)

    metadata_dict = {
        CNN_FILE_KEY: cnn_file_name,
        CNN_FEATURE_LAYER_KEY: cnn_feature_layer_name,
        NUM_EPOCHS_KEY: num_epochs,
        NUM_EXAMPLES_PER_BATCH_KEY: num_examples_per_batch,
        NUM_TRAINING_BATCHES_KEY: num_training_batches_per_epoch,
        TRAINING_FILES_KEY: training_example_file_names,
        FIRST_TRAINING_TIME_KEY: first_training_time_unix_sec,
        LAST_TRAINING_TIME_KEY: last_training_time_unix_sec,
        NUM_VALIDATION_BATCHES_KEY: num_validation_batches_per_epoch,
        VALIDATION_FILES_KEY: validation_example_file_names,
        FIRST_VALIDATION_TIME_KEY: first_validation_time_unix_sec,
        LAST_VALIDATION_TIME_KEY: last_validation_time_unix_sec
    }

    file_system_utils.mkdir_recursive_if_necessary(file_name=pickle_file_name)

    pickle_file_handle = open(pickle_file_name, 'wb')
    pickle.dump(metadata_dict, pickle_file_handle)
    pickle_file_handle.close()

    return metadata_dict


def read_model_metadata(pickle_file_name):
    """Reads metadata from Pickle file.

    :param pickle_file_name: Path to input file.
    :return: metadata_dict: See doc for `write_model_metadata`.
    :raises: ValueError: if any expected key is missing from the dictionary.
    """

    pickle_file_handle = open(pickle_file_name, 'rb')
    metadata_dict = pickle.load(pickle_file_handle)
    pickle_file_handle.close()

    missing_keys = list(set(METADATA_KEYS) - set(metadata_dict.keys()))
    if len(missing_keys) == 0:
        return metadata_dict

    error_string = (
        '\n{0:s}\nKeys listed above were expected, but not found, in file '
        '"{1:s}".'
    ).format(str(missing_keys), pickle_file_name)

    raise ValueError(error_string)


def apply_upconvnet(
        radar_matrix, cnn_model_object, cnn_feature_layer_name,
        ucn_model_object, num_examples_per_batch=1000, verbose=True):
    """Applies upconvnet to new radar images.

    :param radar_matrix: numpy array of original images (inputs to CNN).
    :param cnn_model_object: Trained CNN (instance of `keras.models.Model` or
        `keras.models.Sequential`).  Will be used to convert images to feature
        vectors.
    :param cnn_feature_layer_name: Name of feature-generating layer in CNN.
        Feature vectors (inputs to upconvnet) will be outputs from this layer.
    :param ucn_model_object: Trained upconvnet (instance of `keras.models.Model`
        or `keras.models.Sequential`).  Will be used to convert feature vectors
        back to images (ideally back to original images).
    :param num_examples_per_batch: Number of examples per batch.
    :param verbose: Boolean flag.  If True, will print progress messages to
        command window.
    :return: reconstructed_radar_matrix: Reconstructed version of input.
    """

    partial_cnn_model_object = cnn.model_to_feature_generator(
        model_object=cnn_model_object,
        feature_layer_name=cnn_feature_layer_name)

    error_checking.assert_is_boolean(verbose)
    error_checking.assert_is_numpy_array_without_nan(radar_matrix)

    num_dimensions = len(radar_matrix.shape)
    error_checking.assert_is_geq(num_dimensions, 4)
    error_checking.assert_is_leq(num_dimensions, 5)

    num_examples = radar_matrix.shape[0]
    if num_examples_per_batch is None:
        num_examples_per_batch = num_examples + 0

    error_checking.assert_is_integer(num_examples_per_batch)
    error_checking.assert_is_greater(num_examples_per_batch, 0)
    num_examples_per_batch = min([num_examples_per_batch, num_examples])

    reconstructed_radar_matrix = None

    for i in range(0, num_examples, num_examples_per_batch):
        j = i
        k = min([i + num_examples_per_batch - 1, num_examples - 1])
        these_example_indices = numpy.linspace(j, k, num=k - j + 1, dtype=int)

        if verbose:
            print((
                'Applying upconvnet to examples {0:d}-{1:d} of {2:d}...'
            ).format(
                numpy.min(these_example_indices) + 1,
                numpy.max(these_example_indices) + 1,
                num_examples
            ))

        this_feature_matrix = partial_cnn_model_object.predict(
            radar_matrix[these_example_indices, ...],
            batch_size=len(these_example_indices)
        )

        this_reconstructed_matrix = ucn_model_object.predict(
            this_feature_matrix, batch_size=len(these_example_indices)
        )

    if reconstructed_radar_matrix is None:
        dimensions = numpy.array(
            (num_examples,) + this_reconstructed_matrix.shape[1:], dtype=int
        )
        reconstructed_radar_matrix = numpy.full(dimensions, numpy.nan)

    reconstructed_radar_matrix[
        these_example_indices, ...] = this_reconstructed_matrix

    print('Have applied upconvnet to all {0:d} examples!'.format(num_examples))
    return reconstructed_radar_matrix


def find_prediction_file(top_directory_name, spc_date_string,
                         raise_error_if_missing=False):
    """Finds file with upconvnet predictions (reconstructed radar images).

    :param top_directory_name: Name of top-level directory with upconvnet
        predictions.
    :param spc_date_string: SPC date (format "yyyymmdd").
    :param raise_error_if_missing: Boolean flag.  If file is missing and
        `raise_error_if_missing = True`, this method will error out.
    :return: prediction_file_name: Path to prediction file.  If file is missing
        and `raise_error_if_missing = False`, this will be the expected path.
    :raises: ValueError: if file is missing and `raise_error_if_missing = True`.
    """

    error_checking.assert_is_string(top_directory_name)
    error_checking.assert_is_boolean(raise_error_if_missing)
    time_conversion.spc_date_string_to_unix_sec(spc_date_string)

    prediction_file_name = (
        '{0:s}/{1:s}/{2:s}_{3:s}.nc'
    ).format(
        top_directory_name, spc_date_string[:4], PATHLESS_FILE_NAME_PREFIX,
        spc_date_string
    )

    if raise_error_if_missing and not os.path.isfile(prediction_file_name):
        error_string = 'Cannot find file.  Expected at: "{0:s}"'.format(
            prediction_file_name)
        raise ValueError(error_string)

    return prediction_file_name


def write_predictions(
        netcdf_file_name, denorm_recon_radar_matrix, full_storm_id_strings,
        storm_times_unix_sec, mse_by_example, upconvnet_file_name):
    """Writes predictions (reconstructed radar images) to NetCDF file.

    E = number of examples

    :param netcdf_file_name: Path to output file.
    :param denorm_recon_radar_matrix: numpy array of denormalized, reconstructed
        radar images.  Must be 4-D or 5-D numpy array where first axis has
        length E.
    :param full_storm_id_strings: length-E list of storm IDs.
    :param storm_times_unix_sec: length-E numpy array of valid times.
    :param mse_by_example: length-E numpy array of mean squared errors (in
        normalized, not physical, units).
    :param upconvnet_file_name: Path to upconvnet that generated the
        reconstructed images (readable by `cnn.read_model`).
    """

    error_checking.assert_is_numpy_array_without_nan(denorm_recon_radar_matrix)
    num_dimensions = len(denorm_recon_radar_matrix.shape)
    error_checking.assert_is_geq(num_dimensions, 4)
    error_checking.assert_is_leq(num_dimensions, 5)

    num_spatial_dim = len(denorm_recon_radar_matrix.shape) - 2
    num_examples = denorm_recon_radar_matrix.shape[0]
    these_expected_dim = numpy.array([num_examples], dtype=int)

    error_checking.assert_is_string_list(full_storm_id_strings)
    error_checking.assert_is_numpy_array(
        numpy.array(full_storm_id_strings), exact_dimensions=these_expected_dim
    )

    error_checking.assert_is_integer(storm_times_unix_sec)
    error_checking.assert_is_numpy_array(
        storm_times_unix_sec, exact_dimensions=these_expected_dim)

    error_checking.assert_is_geq_numpy_array(mse_by_example, 0.)
    error_checking.assert_is_numpy_array(
        mse_by_example, exact_dimensions=these_expected_dim)

    error_checking.assert_is_string(netcdf_file_name)
    error_checking.assert_is_string(upconvnet_file_name)

    # Open NetCDF file.
    file_system_utils.mkdir_recursive_if_necessary(file_name=netcdf_file_name)
    dataset_object = netCDF4.Dataset(
        netcdf_file_name, 'w', format='NETCDF3_64BIT_OFFSET')

    dataset_object.setncattr(UPCONVNET_FILE_KEY, upconvnet_file_name)

    # Set up dimensions.
    dataset_object.createDimension(EXAMPLE_DIMENSION_KEY, num_examples)
    dataset_object.createDimension(
        ROW_DIMENSION_KEY, denorm_recon_radar_matrix.shape[1]
    )
    dataset_object.createDimension(
        COLUMN_DIMENSION_KEY, denorm_recon_radar_matrix.shape[2]
    )
    dataset_object.createDimension(
        CHANNEL_DIMENSION_KEY, denorm_recon_radar_matrix.shape[-1]
    )
    spatial_dimensions = (
        EXAMPLE_DIMENSION_KEY, ROW_DIMENSION_KEY, COLUMN_DIMENSION_KEY
    )

    if num_spatial_dim == 3:
        dataset_object.createDimension(
            HEIGHT_DIMENSION_KEY, denorm_recon_radar_matrix.shape[3]
        )
        spatial_dimensions += (HEIGHT_DIMENSION_KEY,)

    num_id_characters = numpy.max(numpy.array([
        len(id) for id in full_storm_id_strings
    ]))
    dataset_object.createDimension(ID_CHAR_DIMENSION_KEY, num_id_characters)

    # Add reconstructed images.
    dataset_object.createVariable(
        RECON_IMAGE_MATRIX_KEY, datatype=numpy.float32,
        dimensions=spatial_dimensions + (CHANNEL_DIMENSION_KEY,)
    )
    dataset_object.variables[RECON_IMAGE_MATRIX_KEY][:] = (
        denorm_recon_radar_matrix
    )

    # Add mean squared errors.
    dataset_object.createVariable(
        MEAN_SQUARED_ERRORS_KEY, datatype=numpy.float32,
        dimensions=EXAMPLE_DIMENSION_KEY
    )
    dataset_object.variables[MEAN_SQUARED_ERRORS_KEY][:] = mse_by_example

    # Add storm IDs.
    this_string_format = 'S{0:d}'.format(num_id_characters)
    full_storm_ids_char_array = netCDF4.stringtochar(numpy.array(
        full_storm_id_strings, dtype=this_string_format
    ))

    dataset_object.createVariable(
        FULL_STORM_IDS_KEY, datatype='S1',
        dimensions=(EXAMPLE_DIMENSION_KEY, ID_CHAR_DIMENSION_KEY)
    )
    dataset_object.variables[FULL_STORM_IDS_KEY][:] = numpy.array(
        full_storm_ids_char_array)

    # Add storm times.
    dataset_object.createVariable(
        STORM_TIMES_KEY, datatype=numpy.int32, dimensions=EXAMPLE_DIMENSION_KEY
    )
    dataset_object.variables[STORM_TIMES_KEY][:] = storm_times_unix_sec

    dataset_object.close()


def read_predictions(netcdf_file_name):
    """Reads predictions (reconstructed radar images) from NetCDF file.

    :param netcdf_file_name: Path to input file.
    :return: prediction_dict: Dictionary with the following keys.
    prediction_dict["denorm_recon_radar_matrix"]: See doc for
        `write_predictions`.
    prediction_dict["mse_by_example"]: Same.
    prediction_dict["full_storm_id_strings"]: Same.
    prediction_dict["storm_times_unix_sec"]: Same.
    prediction_dict["upconvnet_file_name"]: Same.
    """

    dataset_object = netCDF4.Dataset(netcdf_file_name)

    prediction_dict = {
        RECON_IMAGE_MATRIX_KEY: numpy.array(
            dataset_object.variables[RECON_IMAGE_MATRIX_KEY][:]
        ),
        MEAN_SQUARED_ERRORS_KEY: numpy.array(
            dataset_object.variables[MEAN_SQUARED_ERRORS_KEY][:]
        ),
        STORM_TIMES_KEY: numpy.array(
            dataset_object.variables[STORM_TIMES_KEY][:], dtype=int
        ),
        FULL_STORM_IDS_KEY: [
            str(s) for s in netCDF4.chartostring(
                dataset_object.variables[FULL_STORM_IDS_KEY][:]
            )
        ],
        UPCONVNET_FILE_KEY: str(getattr(dataset_object, UPCONVNET_FILE_KEY))
    }

    dataset_object.close()
    return prediction_dict
