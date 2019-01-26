"""Methods for novelty detection.

--- REFERENCES ---

Wagstaff, K., and J. Lee: "Interpretable discovery in large image data sets."
    arXiv e-prints, 1806, https://arxiv.org/abs/1806.08340.
"""

import pickle
import numpy
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.gg_utils import error_checking
from gewittergefahr.deep_learning import cnn
from gewittergefahr.deep_learning import deep_learning_utils as dl_utils

# TODO(thunderhoser): Fix IO methods.

DEFAULT_PCT_VARIANCE_TO_KEEP = 97.5
NUM_EXAMPLES_PER_CNN_BATCH = 1000

EOF_MATRIX_KEY = 'eof_matrix'
FEATURE_MEANS_KEY = 'feature_means'
FEATURE_STDEVS_KEY = 'feature_standard_deviations'

NOVEL_IMAGES_ACTUAL_KEY = 'novel_image_matrix_actual'
NOVEL_IMAGES_UPCONV_KEY = 'novel_image_matrix_upconv'
NOVEL_IMAGES_UPCONV_SVD_KEY = 'novel_image_matrix_upconv_svd'

BASELINE_IMAGES_KEY = 'baseline_image_matrix'
TEST_IMAGES_KEY = 'test_image_matrix'
UCN_FILE_NAME_KEY = 'ucn_file_name'
PERCENT_VARIANCE_KEY = 'percent_svd_variance_to_keep'
NORM_FUNCTION_KEY = 'norm_function_name'
DENORM_FUNCTION_KEY = 'denorm_function_name'

NOVEL_EXAMPLES_ACTUAL_KEY = 'list_of_novel_input_matrices'
CNN_FEATURE_LAYER_KEY = 'cnn_feature_layer_name'
MULTIPASS_KEY = 'multipass'

REQUIRED_KEYS = [
    NOVEL_IMAGES_ACTUAL_KEY, NOVEL_IMAGES_UPCONV_KEY,
    NOVEL_IMAGES_UPCONV_SVD_KEY, BASELINE_IMAGES_KEY, TEST_IMAGES_KEY,
    UCN_FILE_NAME_KEY, PERCENT_VARIANCE_KEY, NORM_FUNCTION_KEY,
    DENORM_FUNCTION_KEY
]


def _normalize_features(feature_matrix, feature_means=None,
                        feature_standard_deviations=None):
    """Normalizes scalar features to z-scores.

    E = number of examples (storm objects)
    Z = number of features

    :param feature_matrix: E-by-Z numpy array of features.
    :param feature_means: length-Z numpy array of mean values.  If
        `feature_means is None`, these will be computed on the fly from
        `feature_matrix`.
    :param feature_standard_deviations: Same but with standard deviations.
    :return: feature_matrix: Normalized version of input.
    :return: feature_means: See input doc.
    :return: feature_standard_deviations: See input doc.
    """

    if feature_means is None or feature_standard_deviations is None:
        feature_means = numpy.mean(feature_matrix, axis=0)
        feature_standard_deviations = numpy.std(feature_matrix, axis=0, ddof=1)

    num_examples = feature_matrix.shape[0]
    num_features = feature_matrix.shape[1]

    mean_matrix = numpy.reshape(feature_means, (1, num_features))
    mean_matrix = numpy.repeat(mean_matrix, repeats=num_examples, axis=0)

    stdev_matrix = numpy.reshape(feature_standard_deviations, (1, num_features))
    stdev_matrix = numpy.repeat(stdev_matrix, repeats=num_examples, axis=0)

    feature_matrix = (feature_matrix - mean_matrix) / stdev_matrix
    return feature_matrix, feature_means, feature_standard_deviations


def _fit_svd(baseline_feature_matrix, test_feature_matrix,
             percent_variance_to_keep):
    """Fits SVD (singular-value decomposition) model.

    B = number of baseline examples (storm objects)
    T = number of testing examples (storm objects)
    Z = number of scalar features (produced by dense layer of a CNN)
    K = number of modes (top eigenvectors) retained

    The SVD model will be fit only to the baseline set, but both the baseline
    and testing sets will be used to compute normalization parameters (means and
    standard deviations).  Before, when only the baseline set was used to
    compute normalization params, the testing set had huge standard deviations,
    which caused the results of novelty detection to be physically unrealistic.

    :param baseline_feature_matrix: B-by-Z numpy array of features.
    :param test_feature_matrix: T-by-Z numpy array of features.
    :param percent_variance_to_keep: Percentage of variance to keep.  Determines
        how many eigenvectors (K in the above discussion) will be used in the
        SVD model.

    :return: svd_dictionary: Dictionary with the following keys.
    svd_dictionary['eof_matrix']: Z-by-K numpy array, where each column is an
        EOF (empirical orthogonal function).
    svd_dictionary['feature_means']: length-Z numpy array with mean value of
        each feature (before transformation).
    svd_dictionary['feature_standard_deviations']: length-Z numpy array with
        standard deviation of each feature (before transformation).
    """

    error_checking.assert_is_greater(percent_variance_to_keep, 0.)
    error_checking.assert_is_leq(percent_variance_to_keep, 100.)

    combined_feature_matrix = numpy.concatenate(
        (baseline_feature_matrix, test_feature_matrix), axis=0)

    combined_feature_matrix, feature_means, feature_standard_deviations = (
        _normalize_features(feature_matrix=combined_feature_matrix)
    )

    num_features = baseline_feature_matrix.shape[1]
    num_baseline_examples = baseline_feature_matrix.shape[0]
    baseline_feature_matrix = combined_feature_matrix[
        :num_baseline_examples, ...]

    eigenvalues, eof_matrix = numpy.linalg.svd(baseline_feature_matrix)[1:]
    eigenvalues = eigenvalues ** 2

    explained_variances = eigenvalues / numpy.sum(eigenvalues)
    cumulative_explained_variances = numpy.cumsum(explained_variances)

    fraction_of_variance_to_keep = 0.01 * percent_variance_to_keep
    these_indices = numpy.where(
        cumulative_explained_variances >= fraction_of_variance_to_keep
    )[0]

    if len(these_indices) == 0:
        these_indices = numpy.array([num_features - 1], dtype=int)
    num_modes_to_keep = 1 + these_indices[0]

    print (
        'Number of modes required to explain {0:f}% of variance: {1:d}'
    ).format(percent_variance_to_keep, num_modes_to_keep)

    return {
        EOF_MATRIX_KEY: numpy.transpose(eof_matrix)[..., :num_modes_to_keep],
        FEATURE_MEANS_KEY: feature_means,
        FEATURE_STDEVS_KEY: feature_standard_deviations
    }


def _apply_svd(feature_vector, svd_dictionary):
    """Applies SVD (singular-value decomposition) model to new example.

    Z = number of features

    :param feature_vector: length-Z numpy array with feature values for one
        example (storm object).
    :param svd_dictionary: Dictionary created by `_fit_svd`.
    :return: reconstructed_feature_vector: Reconstructed version of input.
    """

    this_matrix = numpy.dot(
        svd_dictionary[EOF_MATRIX_KEY],
        numpy.transpose(svd_dictionary[EOF_MATRIX_KEY])
    )

    feature_vector_norm = (
        (feature_vector - svd_dictionary[FEATURE_MEANS_KEY]) /
        svd_dictionary[FEATURE_STDEVS_KEY]
    )

    reconstructed_feature_vector_norm = numpy.dot(
        this_matrix, feature_vector_norm)

    return (
        svd_dictionary[FEATURE_MEANS_KEY] +
        reconstructed_feature_vector_norm * svd_dictionary[FEATURE_STDEVS_KEY]
    )


def _apply_cnn(cnn_model_object, list_of_predictor_matrices, output_layer_name,
               verbose=True):
    """Applies trained CNN to new data.

    T = number of input tensors to the model
    E = number of examples

    :param cnn_model_object: Trained instance of `keras.models.Model`.
    :param list_of_predictor_matrices: length-T list of numpy arrays, where the
        [i]th array is the [i]th input matrix to the model.  The first axis of
        each array must have length E.
    :param output_layer_name: Will return output from this layer.
    :param verbose: Boolean flag.  If True, will print progress messages.
    :return: feature_matrix: numpy array of features (outputs from the given
        layer).  There is no guarantee on the shape of this array, except that
        the first axis has length E.
    """

    intermediate_model_object = cnn.model_to_feature_generator(
        model_object=cnn_model_object, output_layer_name=output_layer_name)

    num_examples = list_of_predictor_matrices[0].shape[0]
    feature_matrix = None

    for i in range(0, num_examples, NUM_EXAMPLES_PER_CNN_BATCH):
        this_first_index = i
        this_last_index = min(
            [i + NUM_EXAMPLES_PER_CNN_BATCH - 1, num_examples - 1]
        )

        if verbose:
            print('Applying CNN to examples {0:d}-{1:d} of {2:d}...'.format(
                this_first_index, this_last_index, num_examples))

        these_indices = numpy.linspace(
            this_first_index, this_last_index,
            num=this_last_index - this_first_index + 1, dtype=int)

        if len(list_of_predictor_matrices) == 1:
            this_feature_matrix = intermediate_model_object.predict(
                list_of_predictor_matrices[0][these_indices, ...],
                batch_size=NUM_EXAMPLES_PER_CNN_BATCH)
        else:
            this_feature_matrix = intermediate_model_object.predict(
                [a[these_indices, ...] for a in list_of_predictor_matrices],
                batch_size=NUM_EXAMPLES_PER_CNN_BATCH)

        if feature_matrix is None:
            feature_matrix = this_feature_matrix + 0.
        else:
            feature_matrix = numpy.concatenate(
                (feature_matrix, this_feature_matrix), axis=0)

    return feature_matrix


def gg_norm_function(radar_field_names, normalization_type_string,
                     normalization_param_file_name, min_normalized_value=0.,
                     max_normalized_value=1.):
    """Creates normalization function for GewitterGefahr radar data.

    This function satisfies the requirements for the input arg `norm_function`
    to the method `do_novelty_detection`.

    :param radar_field_names: See doc for
        `deep_learning_utils.normalize_radar_images`.
    :param normalization_type_string: Same.
    :param normalization_param_file_name: Same.
    :param min_normalized_value: Same.
    :param max_normalized_value: Same.
    :return: norm_function: Function (see below).
    """

    def norm_function(radar_image_matrix):
        """Normalizes GewitterGefahr radar data.

        :param radar_image_matrix: See doc for
            `deep_learning_utils.normalize_radar_images`.
        :return: radar_image_matrix: Normalized version of input.
        """

        return dl_utils.normalize_radar_images(
            radar_image_matrix=radar_image_matrix,
            field_names=radar_field_names,
            normalization_type_string=normalization_type_string,
            normalization_param_file_name=normalization_param_file_name,
            min_normalized_value=min_normalized_value,
            max_normalized_value=max_normalized_value)

    return norm_function


def gg_denorm_function(radar_field_names, normalization_type_string,
                       normalization_param_file_name, min_normalized_value=0.,
                       max_normalized_value=1.):
    """Creates *de*normalization function for GewitterGefahr radar data.

    This function satisfies the requirements for the input arg `denorm_function`
    to the method `do_novelty_detection`.

    :param radar_field_names: See doc for
        `deep_learning_utils.denormalize_radar_images`.
    :param normalization_type_string: Same.
    :param normalization_param_file_name: Same.
    :param min_normalized_value: Same.
    :param max_normalized_value: Same.
    :return: denorm_function: Function (see below).
    """

    def denorm_function(radar_image_matrix):
        """Denormalizes GewitterGefahr radar data.

        :param radar_image_matrix: See doc for
            `deep_learning_utils.denormalize_radar_images`.
        :return: radar_image_matrix: Denormalized version of input.
        """

        return dl_utils.denormalize_radar_images(
            radar_image_matrix=radar_image_matrix,
            field_names=radar_field_names,
            normalization_type_string=normalization_type_string,
            normalization_param_file_name=normalization_param_file_name,
            min_normalized_value=min_normalized_value,
            max_normalized_value=max_normalized_value)

    return denorm_function


def do_novelty_detection_old(
        baseline_image_matrix, test_image_matrix, cnn_model_object,
        cnn_feature_layer_name, ucn_model_object, num_novel_test_images,
        norm_function, denorm_function,
        percent_svd_variance_to_keep=DEFAULT_PCT_VARIANCE_TO_KEEP):
    """Does novelty detection.

    Specifically, this method follows the procedure in Wagstaff et al. (2018)
    to determine which images in the test set are most novel with respect to the
    baseline set.

    B = number of baseline examples
    T = number of test examples
    M = number of rows in each image
    N = number of columns in each image
    H = number of heights in each image (optional)
    C = number of channels (predictor variables)

    If `norm_function is None` and `denorm_function is None`, this method will
    assume that the input images (`baseline_image_matrix` and
    `test_image_matrix`) are normalized and will return normalized images.
    Otherwise, will assume that input images are *de*normalized and will return
    *de*normalized images.

    :param baseline_image_matrix: numpy array (either B x M x N x C or
        B x M x N x H x C) of baseline images.
    :param test_image_matrix: numpy array (either T x M x N x C or
        T x M x N x H x C) of baseline images.
    :param cnn_model_object: Trained CNN model (instance of
        `keras.models.Model`).  Will be used to turn images into scalar
        features.
    :param cnn_feature_layer_name: The "scalar features" will be the set of
        activations from this layer.
    :param ucn_model_object: Trained UCN model (instance of
        `keras.models.Model`).  Will be used to turn scalar features into
        images.
    :param num_novel_test_images: Number of novel test images to find.
    :param norm_function: Function used to normalize images.  Must have the
        following inputs and outputs.
    Input: image_matrix: numpy array (either E x M x N x C or E x M x N x H x C,
        where E = number of examples) of denormalized images.
    Output: image_matrix_norm: numpy array (equivalent shape) of normalized
        images.

    :param denorm_function: Function used to *de*normalize images.  Must have
        the following inputs and outputs.
    Input: image_matrix: numpy array (either E x M x N x C or E x M x N x H x C,
        where E = number of examples) of normalized images.
    Output: image_matrix_norm: numpy array (equivalent shape) of denormalized
        images.

    :param percent_svd_variance_to_keep: See doc for `_fit_svd`.

    :return: novelty_dict: Dictionary with the following keys.  In the following
        discussion, Q = number of novel test images found.
    novelty_dict['novel_image_matrix_actual']: Q-by-M-by-N-by-C numpy array of
        novel test images.
    novelty_dict['novel_image_matrix_upconv']: Same as
        "novel_image_matrix_actual" but reconstructed by the upconvnet.
    novelty_dict['novel_image_matrix_upconv_svd']: Same as
        "novel_image_matrix_actual" but reconstructed by SVD (singular-value
        decomposition) and the upconvnet.

    novelty_dict['baseline_image_matrix']: Same as input.
    novelty_dict['test_image_matrix']: Same as input.
    novelty_dict['percent_svd_variance_to_keep']: Same as input.
    novelty_dict['norm_function_name']: Name of input `norm_function`.
    novelty_dict['denorm_function_name']: Name of input `denorm_function`.
    """

    # TODO(thunderhoser): Move this method to GeneralExam repository.

    error_checking.assert_is_numpy_array_without_nan(baseline_image_matrix)
    num_dimensions = len(baseline_image_matrix.shape)
    error_checking.assert_is_geq(num_dimensions, 4)
    error_checking.assert_is_leq(num_dimensions, 5)

    error_checking.assert_is_numpy_array_without_nan(test_image_matrix)

    num_test_examples = test_image_matrix.shape[0]
    expected_dimensions = numpy.array(
        (num_test_examples,) + baseline_image_matrix.shape[1:], dtype=int)
    error_checking.assert_is_numpy_array(
        test_image_matrix, exact_dimensions=expected_dimensions)

    error_checking.assert_is_integer(num_novel_test_images)
    error_checking.assert_is_geq(num_novel_test_images, 1)
    error_checking.assert_is_leq(num_novel_test_images, num_test_examples)

    inputs_normalized = norm_function is None and denorm_function is None

    if inputs_normalized:
        baseline_image_matrix_norm = baseline_image_matrix
        test_image_matrix_norm = test_image_matrix
    else:
        baseline_image_matrix_norm = norm_function(baseline_image_matrix)
        test_image_matrix_norm = norm_function(test_image_matrix)

    baseline_feature_matrix = _apply_cnn(
        cnn_model_object=cnn_model_object,
        list_of_predictor_matrices=[baseline_image_matrix_norm],
        output_layer_name=cnn_feature_layer_name, verbose=False)

    test_feature_matrix = _apply_cnn(
        cnn_model_object=cnn_model_object,
        list_of_predictor_matrices=[test_image_matrix_norm],
        output_layer_name=cnn_feature_layer_name, verbose=False)

    novel_indices = []
    novel_image_matrix_upconv = None
    novel_image_matrix_upconv_svd = None

    for k in range(num_novel_test_images):
        print('Finding {0:d}th of {1:d} novel test images...'.format(
            k + 1, num_novel_test_images))

        if len(novel_indices) == 0:
            this_baseline_feature_matrix = baseline_feature_matrix + 0.
            this_test_feature_matrix = test_feature_matrix + 0.
        else:
            novel_indices_numpy = numpy.array(novel_indices, dtype=int)
            this_baseline_feature_matrix = numpy.concatenate(
                (baseline_feature_matrix,
                 test_feature_matrix[novel_indices_numpy, ...]),
                axis=0)

            this_test_feature_matrix = numpy.delete(
                test_feature_matrix, obj=novel_indices_numpy, axis=0)

        svd_dictionary = _fit_svd(
            baseline_feature_matrix=this_baseline_feature_matrix,
            test_feature_matrix=this_test_feature_matrix,
            percent_variance_to_keep=percent_svd_variance_to_keep)

        svd_errors = numpy.full(num_test_examples, numpy.nan)
        test_feature_matrix_svd = numpy.full(
            test_feature_matrix.shape, numpy.nan)

        for i in range(num_test_examples):
            if i in novel_indices:
                continue

            test_feature_matrix_svd[i, ...] = _apply_svd(
                feature_vector=test_feature_matrix[i, ...],
                svd_dictionary=svd_dictionary)

            svd_errors[i] = numpy.linalg.norm(
                test_feature_matrix_svd[i, ...] - test_feature_matrix[i, ...]
            )

        new_novel_index = numpy.nanargmax(svd_errors)
        novel_indices.append(new_novel_index)

        new_image_matrix_upconv = ucn_model_object.predict(
            test_feature_matrix[[new_novel_index], ...], batch_size=1)

        new_image_matrix_upconv_svd = ucn_model_object.predict(
            test_feature_matrix_svd[[new_novel_index], ...], batch_size=1)

        if novel_image_matrix_upconv is None:
            novel_image_matrix_upconv = new_image_matrix_upconv + 0.
            novel_image_matrix_upconv_svd = new_image_matrix_upconv_svd + 0.
        else:
            novel_image_matrix_upconv = numpy.concatenate(
                (novel_image_matrix_upconv, new_image_matrix_upconv), axis=0)
            novel_image_matrix_upconv_svd = numpy.concatenate(
                (novel_image_matrix_upconv_svd, new_image_matrix_upconv_svd),
                axis=0)

    novel_indices = numpy.array(novel_indices, dtype=int)

    if inputs_normalized:
        norm_function_name = None
        denorm_function_name = None
    else:
        novel_image_matrix_upconv = denorm_function(novel_image_matrix_upconv)
        novel_image_matrix_upconv_svd = denorm_function(
            novel_image_matrix_upconv_svd)

        norm_function_name = norm_function.__name__
        denorm_function_name = denorm_function.__name__

    return {
        NOVEL_IMAGES_ACTUAL_KEY: test_image_matrix[novel_indices, ...],
        NOVEL_IMAGES_UPCONV_KEY: novel_image_matrix_upconv,
        NOVEL_IMAGES_UPCONV_SVD_KEY: novel_image_matrix_upconv_svd,
        BASELINE_IMAGES_KEY: baseline_image_matrix,
        TEST_IMAGES_KEY: test_image_matrix,
        PERCENT_VARIANCE_KEY: percent_svd_variance_to_keep,
        NORM_FUNCTION_KEY: norm_function_name,
        DENORM_FUNCTION_KEY: denorm_function_name
    }


def do_novelty_detection(
        list_of_baseline_input_matrices, list_of_trial_input_matrices,
        cnn_model_object, cnn_feature_layer_name, upconvnet_model_object,
        num_novel_examples, multipass=False,
        percent_svd_variance_to_keep=DEFAULT_PCT_VARIANCE_TO_KEEP):
    """Runs novelty detection.

    I = number of input tensors to the CNN
    B = number of baseline examples
    T = number of trial examples

    This method assumes that both `list_of_baseline_input_matrices` and
    `list_of_trial_input_matrices` are normalized.

    :param list_of_baseline_input_matrices: length-I list of numpy arrays, where
        the [i]th array is the [i]th input matrix to the CNN.  The first axis of
        each array must have length B.
    :param list_of_trial_input_matrices: Same, except the first axis of each
        array must have length T.
    :param cnn_model_object: Trained CNN (instance of `keras.models.Model` or
        `keras.models.Sequential`).
    :param cnn_feature_layer_name: Name of feature layer in CNN.  Outputs of
        this layer will be inputs to the upconvnet.
    :param upconvnet_model_object: Trained upconvnet (instance of
        `keras.models.Model` or `keras.models.Sequential`).
    :param num_novel_examples: Number of novel trial examples to find.  This
        method will find the N most novel trial examples, where N =
        `num_novel_examples`.
    :param multipass: Boolean flag.  If True, will run multi-pass version.  If
        False, will run single-pass version.  In the multi-pass version,
        whenever the next-most novel trial example is found, it is used to fit a
        new SVD model.  In other words, after finding the [k]th-most novel trial
        example, a new SVD model is fit on all baseline examples and the k most
        novel trial examples.
    :param percent_svd_variance_to_keep: See doc for `_fit_svd`.
    :return: novelty_dict: Dictionary with the following keys, letting
        Q = `num_novel_examples`.
    novelty_dict['novel_indices']: length-Q numpy array with indices of novel
        examples, where novel_indices[k] is the index of the [k]th-most novel.
        These are indices into the first axis of each array in
        `list_of_trial_input_matrices`.
    novelty_dict['novel_image_matrix_upconv']: numpy array with upconvnet
        reconstructions of novel examples.  The first axis has length Q.
    novelty_dict['novel_image_matrix_upconv_svd']: Same as
        "novel_image_matrix_upconv", except that images were reconstructed by
        SVD and then the upconvnet.
    novelty_dict['percent_svd_variance_to_keep']: Same as input.
    novelty_dict['cnn_feature_layer_name']: Same as input.
    novelty_dict['multipass']: Same as input.
    """

    baseline_feature_matrix = _apply_cnn(
        cnn_model_object=cnn_model_object,
        list_of_predictor_matrices=list_of_baseline_input_matrices,
        output_layer_name=cnn_feature_layer_name, verbose=True)
    print '\n'

    trial_feature_matrix = _apply_cnn(
        cnn_model_object=cnn_model_object,
        list_of_predictor_matrices=list_of_trial_input_matrices,
        output_layer_name=cnn_feature_layer_name, verbose=True)
    print '\n'

    num_trial_examples = trial_feature_matrix.shape[0]

    error_checking.assert_is_integer(num_novel_examples)
    error_checking.assert_is_greater(num_novel_examples, 0)
    error_checking.assert_is_leq(num_trial_examples, 0)
    error_checking.assert_is_boolean(multipass)

    svd_dictionary = None
    novel_indices = numpy.array([], dtype=int)
    novel_image_matrix_upconv = None
    novel_image_matrix_upconv_svd = None

    for k in range(num_novel_examples):
        print 'Finding {0:d}th-most novel trial example...'.format(
            k + 1, num_novel_examples)

        fit_new_svd = multipass or k == 0

        if fit_new_svd:
            this_baseline_feature_matrix = numpy.concatenate(
                (baseline_feature_matrix,
                 trial_feature_matrix[novel_indices, ...]),
                axis=0)

            this_trial_feature_matrix = numpy.delete(
                trial_feature_matrix, obj=novel_indices, axis=0)

            svd_dictionary = _fit_svd(
                baseline_feature_matrix=this_baseline_feature_matrix,
                test_feature_matrix=this_trial_feature_matrix,
                percent_variance_to_keep=percent_svd_variance_to_keep)

        trial_svd_errors = numpy.full(num_trial_examples, numpy.nan)
        trial_feature_matrix_svd = numpy.full(
            trial_feature_matrix.shape, numpy.nan)

        for i in range(num_trial_examples):
            if i in novel_indices:
                continue

            trial_feature_matrix_svd[i, ...] = _apply_svd(
                feature_vector=trial_feature_matrix[i, ...],
                svd_dictionary=svd_dictionary)

            trial_svd_errors[i] = numpy.linalg.norm(
                trial_feature_matrix_svd[i, ...] - trial_feature_matrix[i, ...]
            )

        these_novel_indices = numpy.full(1, numpy.nanargmax(trial_svd_errors))
        novel_indices = numpy.concatenate((novel_indices, these_novel_indices))

        this_image_matrix_upconv = upconvnet_model_object.predict(
            trial_feature_matrix[these_novel_indices, ...], batch_size=1)

        this_image_matrix_upconv_svd = upconvnet_model_object.predict(
            trial_feature_matrix_svd[these_novel_indices, ...], batch_size=1)

        if novel_image_matrix_upconv is None:
            novel_image_matrix_upconv = this_image_matrix_upconv + 0.
            novel_image_matrix_upconv_svd = this_image_matrix_upconv_svd + 0.
        else:
            novel_image_matrix_upconv = numpy.concatenate(
                (novel_image_matrix_upconv, this_image_matrix_upconv), axis=0)
            novel_image_matrix_upconv_svd = numpy.concatenate(
                (novel_image_matrix_upconv_svd, this_image_matrix_upconv_svd),
                axis=0)

    return {
        NOVEL_EXAMPLES_ACTUAL_KEY:
            [a[novel_indices, ...] for a in list_of_trial_input_matrices],
        NOVEL_IMAGES_UPCONV_KEY: novel_image_matrix_upconv,
        NOVEL_IMAGES_UPCONV_SVD_KEY: novel_image_matrix_upconv_svd,
        PERCENT_VARIANCE_KEY: percent_svd_variance_to_keep,
        CNN_FEATURE_LAYER_KEY: cnn_feature_layer_name,
        MULTIPASS_KEY: multipass
    }


def write_results(novelty_dict, pickle_file_name):
    """Writes novelty-detection results to Pickle file.

    :param novelty_dict: Dictionary created by `do_novelty_detection`, plus one
        extra key.
    novelty_dict['ucn_file_name']: Path to file with upconvnet used for novelty
        detection.  This should be an HDF5 file, readable by
        `keras.models.load_model`.

    :param pickle_file_name: Path to output file.
    :raises: ValueError: if any expected key is not found.
    """

    missing_keys = list(
        set(REQUIRED_KEYS) - set(novelty_dict.keys())
    )

    if len(missing_keys) > 0:
        error_string = 'Cannot find the following expected keys.\n{0:s}'.format(
            str(missing_keys))
        raise ValueError(error_string)

    file_system_utils.mkdir_recursive_if_necessary(file_name=pickle_file_name)

    pickle_file_handle = open(pickle_file_name, 'wb')
    pickle.dump(novelty_dict, pickle_file_handle)
    pickle_file_handle.close()


def read_results(pickle_file_name):
    """Reads novelty-detection results from Pickle file.

    :param pickle_file_name: Path to input file.
    :return: novelty_dict: See doc for `write_results`.
    """

    pickle_file_handle = open(pickle_file_name, 'rb')
    novelty_dict = pickle.load(pickle_file_handle)
    pickle_file_handle.close()

    missing_keys = list(
        set(REQUIRED_KEYS) - set(novelty_dict.keys())
    )

    if len(missing_keys) == 0:
        return novelty_dict

    error_string = 'Cannot find the following expected keys.\n{0:s}'.format(
        str(missing_keys))
    raise ValueError(error_string)
