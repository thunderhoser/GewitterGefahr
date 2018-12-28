"""Methods for novelty detection.

--- REFERENCES ---

Wagstaff, K., and J. Lee: "Interpretable discovery in large image data sets."
    arXiv e-prints, 1806, https://arxiv.org/abs/1806.08340.
"""

import numpy
from gewittergefahr.gg_utils import error_checking
from gewittergefahr.deep_learning import cnn
from gewittergefahr.deep_learning import deep_learning_utils as dl_utils

# TODO(thunderhoser): This file deals only with models that have one input
# tensor (e.g., models trained with only radar images, rather than radar images
# and soundings).

DEFAULT_NUM_SVD_MODES = 10
NUM_EXAMPLES_PER_CNN_BATCH = 1000

EOF_MATRIX_KEY = 'eof_matrix'
FEATURE_MEANS_KEY = 'feature_means'
FEATURE_STDEVS_KEY = 'feature_standard_deviations'

NOVEL_IMAGES_ACTUAL_KEY = 'novel_image_matrix_actual'
NOVEL_IMAGES_UPCONV_KEY = 'novel_image_matrix_upconv'
NOVEL_IMAGES_UPCONV_SVD_KEY = 'novel_image_matrix_upconv_svd'


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


def _fit_svd(baseline_feature_matrix, test_feature_matrix, num_modes_to_keep):
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
    :param num_modes_to_keep: Number of modes (top eigenvectors) to use in SVD
        model.  This is K in the above discussion.
    :return: svd_dictionary: Dictionary with the following keys.
    svd_dictionary['eof_matrix']: Z-by-K numpy array, where each column is an
        EOF (empirical orthogonal function).
    svd_dictionary['feature_means']: length-Z numpy array with mean value of
        each feature (before transformation).
    svd_dictionary['feature_standard_deviations']: length-Z numpy array with
        standard deviation of each feature (before transformation).
    """

    error_checking.assert_is_integer(num_modes_to_keep)
    error_checking.assert_is_geq(num_modes_to_keep, 1)
    error_checking.assert_is_leq(
        num_modes_to_keep, baseline_feature_matrix.shape[1])

    combined_feature_matrix = numpy.concatenate(
        (baseline_feature_matrix, test_feature_matrix), axis=0)

    combined_feature_matrix, feature_means, feature_standard_deviations = (
        _normalize_features(feature_matrix=combined_feature_matrix)
    )

    num_baseline_examples = baseline_feature_matrix.shape[0]
    baseline_feature_matrix = combined_feature_matrix[
        :num_baseline_examples, ...]

    eof_matrix = numpy.linalg.svd(baseline_feature_matrix)[-1]

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


def _apply_cnn(cnn_model_object, predictor_matrix_norm, output_layer_name,
               verbose=True):
    """Applies trained CNN (convolutional neural net) to new data.

    E = number of examples

    :param cnn_model_object: Trained instance of `keras.models.Model`.
    :param predictor_matrix_norm: numpy array of predictor values.  Must be
        normalized in the same way as training data for the CNN.  Must have the
        same shape as training data for the CNN.  Length of first axis must be
        E.
    :param output_layer_name: Will return output from this layer.
    :param verbose: Boolean flag.  If True, will print progress messages.
    :return: feature_matrix: numpy array of features (outputs from the given
        layer).  There is no guarantee on the shape of this array, except that
        the first axis has length E.
    """

    intermediate_model_object = cnn.model_to_feature_generator(
        model_object=cnn_model_object, output_layer_name=output_layer_name)

    num_examples = predictor_matrix_norm.shape[0]
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

        this_feature_matrix = intermediate_model_object.predict(
            predictor_matrix_norm[these_indices, ...],
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


def do_novelty_detection(
        baseline_image_matrix, test_image_matrix, cnn_model_object,
        cnn_feature_layer_name, ucn_model_object, num_novel_test_images,
        norm_function, denorm_function,
        num_svd_modes_to_keep=DEFAULT_NUM_SVD_MODES):
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

    :param num_svd_modes_to_keep: Number of modes to keep in SVD (singular-value
        decomposition) of scalar features.  See `_fit_svd` for more details.

    :return: novelty_dict: Dictionary with the following keys.  In the following
        discussion, Q = number of novel test images found.
    novelty_dict['novel_image_matrix_actual']: Q-by-M-by-N-by-C numpy array of
        novel test images.
    novelty_dict['novel_image_matrix_upconv']: Same as
        "novel_image_matrix_actual" but reconstructed by the upconvnet.
    novelty_dict['novel_image_matrix_upconv_svd']: Same as
        "novel_image_matrix_actual" but reconstructed by SVD (singular-value
        decomposition) and the upconvnet.
    """

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
        predictor_matrix_norm=baseline_image_matrix_norm,
        output_layer_name=cnn_feature_layer_name, verbose=False)

    test_feature_matrix = _apply_cnn(
        cnn_model_object=cnn_model_object,
        predictor_matrix_norm=test_image_matrix_norm,
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
            num_modes_to_keep=num_svd_modes_to_keep)

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

    if not inputs_normalized:
        novel_image_matrix_upconv = denorm_function(novel_image_matrix_upconv)
        novel_image_matrix_upconv_svd = denorm_function(
            novel_image_matrix_upconv_svd)

    return {
        NOVEL_IMAGES_ACTUAL_KEY: test_image_matrix[novel_indices, ...],
        NOVEL_IMAGES_UPCONV_KEY: novel_image_matrix_upconv,
        NOVEL_IMAGES_UPCONV_SVD_KEY: novel_image_matrix_upconv_svd
    }
