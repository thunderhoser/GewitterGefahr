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

DEFAULT_PCT_VARIANCE_TO_KEEP = 97.5
NUM_EXAMPLES_PER_BATCH = 1000

EOF_MATRIX_KEY = 'eof_matrix'
FEATURE_MEANS_KEY = 'feature_means'
FEATURE_STDEVS_KEY = 'feature_standard_deviations'

BASELINE_INPUTS_KEY = 'list_of_baseline_input_matrices'
TRIAL_INPUTS_KEY = 'list_of_trial_input_matrices'
NOVEL_INDICES_KEY = 'novel_indices'
NOVEL_IMAGES_UPCONV_KEY = 'novel_image_matrix_upconv'
NOVEL_IMAGES_UPCONV_SVD_KEY = 'novel_image_matrix_upconv_svd'
PERCENT_VARIANCE_KEY = 'percent_svd_variance_to_keep'
CNN_FEATURE_LAYER_KEY = 'cnn_feature_layer_name'
MULTIPASS_KEY = 'multipass'

BASELINE_STORM_IDS_KEY = 'baseline_storm_ids'
BASELINE_STORM_TIMES_KEY = 'baseline_storm_times_unix_sec'
TRIAL_STORM_IDS_KEY = 'trial_storm_ids'
TRIAL_STORM_TIMES_KEY = 'trial_storm_times_unix_sec'
CNN_FILE_KEY = 'cnn_file_name'
UPCONVNET_FILE_KEY = 'upconvnet_file_name'

STANDARD_FILE_KEYS = [
    BASELINE_INPUTS_KEY, TRIAL_INPUTS_KEY, NOVEL_INDICES_KEY,
    NOVEL_IMAGES_UPCONV_KEY, NOVEL_IMAGES_UPCONV_SVD_KEY, PERCENT_VARIANCE_KEY,
    CNN_FEATURE_LAYER_KEY, MULTIPASS_KEY,
    BASELINE_STORM_IDS_KEY, BASELINE_STORM_TIMES_KEY, TRIAL_STORM_IDS_KEY,
    TRIAL_STORM_TIMES_KEY, CNN_FILE_KEY, UPCONVNET_FILE_KEY
]

MEAN_NOVEL_IMAGE_KEY = 'mean_novel_image_matrix'
MEAN_NOVEL_IMAGE_UPCONV_KEY = 'mean_novel_image_matrix_upconv'
MEAN_NOVEL_IMAGE_UPCONV_SVD_KEY = 'mean_novel_image_matrix_upconv_svd'
THRESHOLD_COUNTS_KEY = 'threshold_count_matrix'
STANDARD_FILE_NAME_KEY = 'standard_novelty_file_name'
PMM_METADATA_KEY = 'pmm_metadata_dict'

PMM_FILE_KEYS = [
    MEAN_NOVEL_IMAGE_KEY, MEAN_NOVEL_IMAGE_UPCONV_KEY,
    MEAN_NOVEL_IMAGE_UPCONV_SVD_KEY, THRESHOLD_COUNTS_KEY,
    STANDARD_FILE_NAME_KEY, PMM_METADATA_KEY
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
        model_object=cnn_model_object, feature_layer_name=output_layer_name)

    num_examples = list_of_predictor_matrices[0].shape[0]
    feature_matrix = None

    for i in range(0, num_examples, NUM_EXAMPLES_PER_BATCH):
        this_first_index = i
        this_last_index = min(
            [i + NUM_EXAMPLES_PER_BATCH - 1, num_examples - 1]
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
                batch_size=NUM_EXAMPLES_PER_BATCH)
        else:
            this_feature_matrix = intermediate_model_object.predict(
                [a[these_indices, ...] for a in list_of_predictor_matrices],
                batch_size=NUM_EXAMPLES_PER_BATCH)

        if feature_matrix is None:
            feature_matrix = this_feature_matrix + 0.
        else:
            feature_matrix = numpy.concatenate(
                (feature_matrix, this_feature_matrix), axis=0)

    return feature_matrix


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
    novelty_dict['list_of_baseline_input_matrices']: Same as input.
    novelty_dict['list_of_trial_input_matrices']: Same as input.
    novelty_dict['novel_indices']: length-Q numpy array with indices of novel
        examples, where novel_indices[k] is the index of the [k]th-most novel
        example.  These are indices into the first axis of each array in
        `list_of_trial_input_matrices`.
    novelty_dict['novel_image_matrix_upconv']: numpy array with upconvnet
        reconstructions of the most novel examples.  The first axis has length
        Q.
    novelty_dict['novel_image_matrix_upconv_svd']: numpy array with upconvnet
        reconstructions of SVD reconstructions of the most novel examples.
        Same dimensions as `novel_image_matrix_upconv`.
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
    error_checking.assert_is_leq(num_novel_examples, num_trial_examples)
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
        BASELINE_INPUTS_KEY: list_of_baseline_input_matrices,
        TRIAL_INPUTS_KEY: list_of_trial_input_matrices,
        NOVEL_INDICES_KEY: novel_indices,
        NOVEL_IMAGES_UPCONV_KEY: novel_image_matrix_upconv,
        NOVEL_IMAGES_UPCONV_SVD_KEY: novel_image_matrix_upconv_svd,
        PERCENT_VARIANCE_KEY: percent_svd_variance_to_keep,
        CNN_FEATURE_LAYER_KEY: cnn_feature_layer_name,
        MULTIPASS_KEY: multipass
    }


def add_metadata(
        novelty_dict, baseline_storm_ids, baseline_storm_times_unix_sec,
        trial_storm_ids, trial_storm_times_unix_sec, cnn_file_name,
        upconvnet_file_name):
    """Adds metadata to novelty-detection results.

    B = number of baseline examples
    T = number of trial examples

    :param novelty_dict: Dictionary created by `do_novelty_detection`.
    :param baseline_storm_ids: length-B list of storm IDs (strings) for baseline
        examples.
    :param baseline_storm_times_unix_sec: length-B numpy array of valid times
        for baseline examples.
    :param trial_storm_ids: length-T list of storm IDs (strings) for trial
        examples.
    :param trial_storm_times_unix_sec: length-T numpy array of valid times for
        baseline examples.
    :param cnn_file_name: Path to file with CNN used for novelty detection
        (readable by `cnn.read_model`).
    :param upconvnet_file_name: Path to file with upconvnet used for novelty
        detection (readable by `cnn.read_model`).

    :return: novelty_dict: Dictionary with the following keys.
    novelty_dict['list_of_baseline_input_matrices']: See doc for
        `do_novelty_detection`.
    novelty_dict['list_of_trial_input_matrices']: Same.
    novelty_dict['novel_indices']: Same.
    novelty_dict['novel_image_matrix_upconv']: Same.
    novelty_dict['novel_image_matrix_upconv_svd']: Same.
    novelty_dict['percent_svd_variance_to_keep']: Same.
    novelty_dict['cnn_feature_layer_name']: Same.
    novelty_dict['multipass']: Same.
    novelty_dict['baseline_storm_ids']: See input doc for this method.
    novelty_dict['baseline_storm_times_unix_sec']: Same.
    novelty_dict['trial_storm_ids']: Same.
    novelty_dict['trial_storm_times_unix_sec']: Same.
    novelty_dict['cnn_file_name']: Same.
    novelty_dict['upconvnet_file_name']: Same.
    """

    num_baseline_examples = novelty_dict[BASELINE_INPUTS_KEY][0].shape[0]
    these_expected_dim = numpy.array([num_baseline_examples], dtype=int)

    error_checking.assert_is_string_list(baseline_storm_ids)
    error_checking.assert_is_numpy_array(
        numpy.array(baseline_storm_ids), exact_dimensions=these_expected_dim)
    error_checking.assert_is_integer_numpy_array(baseline_storm_times_unix_sec)
    error_checking.assert_is_numpy_array(
        baseline_storm_times_unix_sec, exact_dimensions=these_expected_dim)

    num_trial_examples = novelty_dict[TRIAL_INPUTS_KEY][0].shape[0]
    these_expected_dim = numpy.array([num_trial_examples], dtype=int)

    error_checking.assert_is_string_list(trial_storm_ids)
    error_checking.assert_is_numpy_array(
        numpy.array(trial_storm_ids), exact_dimensions=these_expected_dim)
    error_checking.assert_is_integer_numpy_array(trial_storm_times_unix_sec)
    error_checking.assert_is_numpy_array(
        trial_storm_times_unix_sec, exact_dimensions=these_expected_dim)

    error_checking.assert_is_string(cnn_file_name)
    error_checking.assert_is_string(upconvnet_file_name)

    novelty_dict.update({
        BASELINE_STORM_IDS_KEY: baseline_storm_ids,
        BASELINE_STORM_TIMES_KEY: baseline_storm_times_unix_sec,
        TRIAL_STORM_IDS_KEY: trial_storm_ids,
        TRIAL_STORM_TIMES_KEY: trial_storm_times_unix_sec,
        CNN_FILE_KEY: cnn_file_name,
        UPCONVNET_FILE_KEY: upconvnet_file_name
    })

    return novelty_dict


def write_standard_file(novelty_dict, pickle_file_name):
    """Writes novelty-detection results to Pickle file.

    :param novelty_dict: Dictionary created by `add_metadata`.
    :param pickle_file_name: Path to output file.
    :raises: ValueError: if any expected key is missing from the dictionary.
    """

    missing_keys = list(
        set(STANDARD_FILE_KEYS) - set(novelty_dict.keys())
    )

    if len(missing_keys) > 0:
        error_string = (
            '\n{0:s}\nKeys listed above were expected, but not found, in the '
            'dictionary.'
        ).format(str(missing_keys))

        raise ValueError(error_string)

    file_system_utils.mkdir_recursive_if_necessary(file_name=pickle_file_name)

    pickle_file_handle = open(pickle_file_name, 'wb')
    pickle.dump(novelty_dict, pickle_file_handle)
    pickle_file_handle.close()


def read_standard_file(pickle_file_name):
    """Reads novelty-detection results from Pickle file.

    :param pickle_file_name: Path to input file.
    :return: novelty_dict: Dictionary with keys listed in output doc for
        `add_metadata`.
    :raises: ValueError: if any expected key is missing from the dictionary.
    """

    pickle_file_handle = open(pickle_file_name, 'rb')
    novelty_dict = pickle.load(pickle_file_handle)
    pickle_file_handle.close()

    missing_keys = list(set(STANDARD_FILE_KEYS) - set(novelty_dict.keys()))
    if len(missing_keys) == 0:
        return novelty_dict

    error_string = (
        '\n{0:s}\nKeys listed above were expected, but not found, in file '
        '"{1:s}".'
    ).format(str(missing_keys), pickle_file_name)

    raise ValueError(error_string)


def write_pmm_file(
        pickle_file_name, mean_novel_image_matrix,
        mean_novel_image_matrix_upconv, mean_novel_image_matrix_upconv_svd,
        threshold_count_matrix, standard_novelty_file_name, pmm_metadata_dict):
    """Writes mean novelty-detection results to Pickle file.

    This is a mean over many examples, created by PMM (probability-matched
    means).

    :param pickle_file_name: Path to output file.
    :param mean_novel_image_matrix: numpy array with mean image over all trial
        examples.
    :param mean_novel_image_matrix_upconv: numpy array (same dimensions as
        `mean_novel_image_matrix`) with mean upconvnet-reconstructed image over
        all trial examples.
    :param mean_novel_image_matrix_upconv_svd: numpy array (same dimensions as
        `mean_novel_image_matrix`) with mean upconvnet-and-SVD-reconstructed
        image over all trial examples.
    :param threshold_count_matrix: See doc for
        `prob_matched_means.run_pmm_many_variables`.
    :param standard_novelty_file_name: Path to file with standard
        novelty-detection output (readable by `read_standard_file`).
    :param pmm_metadata_dict: Dictionary created by
        `prob_matched_means.check_input_args`.
    """

    error_checking.assert_is_string(standard_novelty_file_name)

    error_checking.assert_is_numpy_array_without_nan(mean_novel_image_matrix)
    error_checking.assert_is_geq(len(mean_novel_image_matrix.shape), 2)
    these_expected_dim = numpy.array(mean_novel_image_matrix.shape, dtype=int)

    error_checking.assert_is_numpy_array_without_nan(
        mean_novel_image_matrix_upconv)
    error_checking.assert_is_numpy_array(
        mean_novel_image_matrix_upconv, exact_dimensions=these_expected_dim)

    error_checking.assert_is_numpy_array_without_nan(
        mean_novel_image_matrix_upconv_svd)
    error_checking.assert_is_numpy_array(
        mean_novel_image_matrix_upconv_svd, exact_dimensions=these_expected_dim)

    if threshold_count_matrix is not None:
        error_checking.assert_is_integer_numpy_array(threshold_count_matrix)
        error_checking.assert_is_geq_numpy_array(threshold_count_matrix, 0)

        spatial_dimensions = numpy.array(
            mean_novel_image_matrix.shape[:-1], dtype=int)
        error_checking.assert_is_numpy_array(
            threshold_count_matrix, exact_dimensions=spatial_dimensions)

    mean_novelty_dict = {
        MEAN_NOVEL_IMAGE_KEY: mean_novel_image_matrix,
        MEAN_NOVEL_IMAGE_UPCONV_KEY: mean_novel_image_matrix_upconv,
        MEAN_NOVEL_IMAGE_UPCONV_SVD_KEY: mean_novel_image_matrix_upconv_svd,
        THRESHOLD_COUNTS_KEY: threshold_count_matrix,
        STANDARD_FILE_NAME_KEY: standard_novelty_file_name,
        PMM_METADATA_KEY: pmm_metadata_dict
    }

    file_system_utils.mkdir_recursive_if_necessary(file_name=pickle_file_name)
    pickle_file_handle = open(pickle_file_name, 'wb')
    pickle.dump(mean_novelty_dict, pickle_file_handle)
    pickle_file_handle.close()


def read_pmm_file(pickle_file_name):
    """Reads mean novelty-detection results from Pickle file.

    :param pickle_file_name: Path to input file.
    :return: mean_novelty_dict: Dictionary with the following keys.
    mean_novelty_dict['mean_novel_image_matrix']: See doc for `write_pmm_file`.
    mean_novelty_dict['mean_novel_image_matrix_upconv']: Same.
    mean_novelty_dict['mean_novel_image_matrix_upconv_svd']: Same.
    mean_novelty_dict['threshold_count_matrix']: Same.
    mean_novelty_dict['standard_novelty_file_name']: Same.
    mean_novelty_dict['pmm_metadata_dict']: Same.

    :raises: ValueError: if any of the aforelisted keys are missing from the
        dictionary.
    """

    pickle_file_handle = open(pickle_file_name, 'rb')
    mean_novelty_dict = pickle.load(pickle_file_handle)
    pickle_file_handle.close()

    missing_keys = list(set(PMM_FILE_KEYS) - set(mean_novelty_dict.keys()))
    if len(missing_keys) == 0:
        return mean_novelty_dict

    error_string = (
        '\n{0:s}\nKeys listed above were expected, but not found, in file '
        '"{1:s}".'
    ).format(str(missing_keys), pickle_file_name)

    raise ValueError(error_string)
