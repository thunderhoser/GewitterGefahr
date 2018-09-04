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
from gewittergefahr.deep_learning import keras_metrics
from gewittergefahr.deep_learning import training_validation_io as trainval_io
from gewittergefahr.gg_io import netcdf_io
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.gg_utils import error_checking

LOSS_AS_MONITOR_STRING = 'loss'
PEIRCE_SCORE_AS_MONITOR_STRING = 'binary_peirce_score'
VALID_MONITOR_STRINGS = [LOSS_AS_MONITOR_STRING, PEIRCE_SCORE_AS_MONITOR_STRING]

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

NUM_EPOCHS_KEY = 'num_epochs'
NUM_TRAINING_BATCHES_KEY = 'num_training_batches_per_epoch'
MONITOR_STRING_KEY = 'monitor_string'
WEIGHT_LOSS_FUNCTION_KEY = 'weight_loss_function'
NUM_VALIDATION_BATCHES_KEY = 'num_validation_batches_per_epoch'
VALIDATION_FILES_KEY = 'validn_file_name_matrix'
POSITIVE_VALIDATION_FILES_KEY = 'validn_file_name_matrix_pos_targets_only'
USE_2D3D_CONVOLUTION_KEY = 'use_2d3d_convolution'
RADAR_SOURCE_KEY = 'radar_source'
RADAR_FIELDS_KEY = 'radar_field_names'
RADAR_HEIGHTS_KEY = 'radar_heights_m_agl'
REFLECTIVITY_HEIGHTS_KEY = 'reflectivity_heights_m_agl'
SOUNDING_HEIGHTS_KEY = 'sounding_heights_m_agl'
TRAINING_OPTION_DICT_KEY = 'training_option_dict'

REQUIRED_METADATA_KEYS = [
    NUM_EPOCHS_KEY, NUM_TRAINING_BATCHES_KEY, MONITOR_STRING_KEY,
    WEIGHT_LOSS_FUNCTION_KEY, NUM_VALIDATION_BATCHES_KEY, VALIDATION_FILES_KEY,
    POSITIVE_VALIDATION_FILES_KEY, USE_2D3D_CONVOLUTION_KEY, RADAR_SOURCE_KEY,
    RADAR_FIELDS_KEY, RADAR_HEIGHTS_KEY, REFLECTIVITY_HEIGHTS_KEY,
    SOUNDING_HEIGHTS_KEY
]

DEFAULT_METADATA_DICT = {
    VALIDATION_FILES_KEY: None,
    POSITIVE_VALIDATION_FILES_KEY: None,
    RADAR_HEIGHTS_KEY: None,
    REFLECTIVITY_HEIGHTS_KEY: None,
    SOUNDING_HEIGHTS_KEY: None
}

DEFAULT_METADATA_DICT.update(trainval_io.DEFAULT_AUGMENTATION_OPTION_DICT)

STORM_OBJECT_DIMENSION_KEY = 'storm_object'
FEATURE_DIMENSION_KEY = 'feature'
SPATIAL_DIMENSION_KEYS = [
    'spatial_dimension1', 'spatial_dimension2', 'spatial_dimension3'
]

FEATURE_MATRIX_KEY = 'feature_matrix'
TARGET_VALUES_KEY = 'target_values'
NUM_CLASSES_KEY = 'num_classes'


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


def read_model(hdf5_file_name):
    """Reads model from HDF5 file.

    :param hdf5_file_name: Path to input file.
    :return: model_object: Instance of `keras.models.Model`.
    """

    error_checking.assert_file_exists(hdf5_file_name)
    return keras.models.load_model(
        hdf5_file_name, custom_objects=CUSTOM_OBJECT_DICT_FOR_READING_MODEL)


def write_model_metadata(pickle_file_name, metadata_dict, training_option_dict):
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
    metadata_dict['use_2d3d_convolution']: Boolean flag.  If True, the network
        performs a hybrid of 2-D and 3-D convolution, so the data-generator is
        `training_validation_io.storm_image_generator_2d3d_myrorss`.
    metadata_dict['radar_source']: Data source (must be accepted by
        `radar_utils.check_field_name`).
    metadata_dict['radar_field_names']: See doc for
        `training_validation_io.find_radar_files_2d` or
        `training_validation_io.find_radar_files_3d`.
    metadata_dict['radar_heights_m_agl']: Same.
    metadata_dict['reflectivity_heights_m_agl']: Same.
    metadata_dict['sounding_heights_m_agl']: 1-D numpy array of sounding heights
        (integer metres above ground level).

    :param training_option_dict: Input to generator (e.g.,
        `training_validation_io.storm_image_generator_2d`) for training data.
    :raises: ValueError: if any expected keys are missing from `metadata_dict`.
    """

    orig_metadata_dict = metadata_dict.copy()
    metadata_dict = DEFAULT_METADATA_DICT.copy()
    metadata_dict.update(orig_metadata_dict)

    missing_keys = list(set(REQUIRED_METADATA_KEYS) - set(metadata_dict.keys()))
    if len(missing_keys):
        error_string = (
            'The following expected keys are missing from the dictionary.'
            '\n{0:s}'
        ).format(str(missing_keys))
        raise ValueError(error_string)

    orig_training_option_dict = training_option_dict.copy()
    training_option_dict = trainval_io.DEFAULT_GENERATOR_OPTION_DICT.copy()
    training_option_dict.update(orig_training_option_dict)
    metadata_dict.update({TRAINING_OPTION_DICT_KEY: training_option_dict})

    file_system_utils.mkdir_recursive_if_necessary(file_name=pickle_file_name)
    pickle_file_handle = open(pickle_file_name, 'wb')
    pickle.dump(metadata_dict, pickle_file_handle)
    pickle_file_handle.close()


def read_model_metadata(pickle_file_name):
    """Reads metadata from Pickle file.

    :param pickle_file_name: Path to input file.
    :return: metadata_dict: See doc for `write_model_metadata`.
    """

    pickle_file_handle = open(pickle_file_name, 'rb')
    metadata_dict = pickle.load(pickle_file_handle)
    pickle_file_handle.close()

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

    checkpoint_object = _get_checkpoint_object(
        output_model_file_name=model_file_name, monitor_string=monitor_string,
        use_validation=num_validation_batches_per_epoch is not None)

    if num_validation_batches_per_epoch > 0:
        model_object.fit_generator(
            generator=trainval_io.storm_image_generator_2d(
                training_option_dict),
            steps_per_epoch=num_training_batches_per_epoch, epochs=num_epochs,
            verbose=1, class_weight=lf_weight_by_class_dict,
            callbacks=[checkpoint_object, history_object])

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
            callbacks=[checkpoint_object, history_object],
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

    checkpoint_object = _get_checkpoint_object(
        output_model_file_name=model_file_name, monitor_string=monitor_string,
        use_validation=num_validation_batches_per_epoch is not None)

    if num_validation_batches_per_epoch is None:
        model_object.fit_generator(
            generator=trainval_io.storm_image_generator_3d(
                training_option_dict),
            steps_per_epoch=num_training_batches_per_epoch, epochs=num_epochs,
            verbose=1, class_weight=lf_weight_by_class_dict,
            callbacks=[checkpoint_object, history_object])

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
            callbacks=[checkpoint_object, history_object],
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

    checkpoint_object = _get_checkpoint_object(
        output_model_file_name=model_file_name, monitor_string=monitor_string,
        use_validation=num_validation_batches_per_epoch is not None)

    if num_validation_batches_per_epoch is None:
        model_object.fit_generator(
            generator=trainval_io.storm_image_generator_2d3d_myrorss(
                training_option_dict),
            steps_per_epoch=num_training_batches_per_epoch, epochs=num_epochs,
            verbose=1, class_weight=lf_weight_by_class_dict,
            callbacks=[checkpoint_object, history_object])

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
            callbacks=[checkpoint_object, history_object],
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
