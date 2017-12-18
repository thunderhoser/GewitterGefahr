"""Feature-transformation methods.

Currently the only method in this file is SVD (singular-value decomposition).
However, I may add more.
"""

import numpy
import pandas
from gewittergefahr.gg_utils import error_checking

DEFAULT_FRACTION_OF_VARIANCE_TO_KEEP = 0.875

FEATURE_NAMES_KEY = 'feature_names'
ORIGINAL_MEANS_KEY = 'original_means'
ORIGINAL_STDEVIATIONS_KEY = 'original_standard_deviations'

PC_MATRIX_KEY = 'principal_component_matrix'
EIGENVALUE_MATRIX_KEY = 'eigenvalue_matrix'
EOF_MATRIX_KEY = 'eof_matrix'


def _reorder_standardization_dict(standardization_dict, new_feature_names):
    """Reorders arrays in standardization dictionary.

    :param standardization_dict: Dictionary created by _standardize_features.
    :param new_feature_names: 1-D list of feature names.  Arrays in
        `standardization_dict` will be put into this order.
    :return: standardization_dict: Same as input, except arrays may be in
        different order.
    :raises: ValueError: if any element of `feature_names` is not in
        `standardization_dict`.
    """

    try:
        sort_indices = numpy.array(
            [standardization_dict[FEATURE_NAMES_KEY].index(f) for
             f in new_feature_names])
    except ValueError as _:
        error_string = (
            str(standardization_dict[FEATURE_NAMES_KEY]) +
            '\n\nFeature names in standardization_dict (shown above) do not '
            'span those in new_feature_names (shown below).\n\n' +
            str(new_feature_names))
        raise ValueError(error_string)

    for this_key in standardization_dict.keys():
        if isinstance(standardization_dict[this_key], numpy.ndarray):
            standardization_dict[this_key] = standardization_dict[
                this_key][sort_indices]
        else:
            standardization_dict[this_key] = numpy.array(
                standardization_dict[this_key])[sort_indices].tolist()

    return standardization_dict


def _standardize_features(feature_table, standardization_dict=None):
    """Standardizes each feature by converting to z-score.

    N = number of examples
    M = number of features (input variables)

    For both input and output, `standardization_dict` is a dictionary with the
    following keys.

    standardization_dict['feature_name']: Name of feature.
    standardization_dict['original_mean']: Original mean (before
        standardization).
    standardization_dict['original_stdev']: Original standard deviation (before
        standardization).

    :param feature_table: pandas DataFrame with N rows and M columns.  Column
        names are feature names.
    :param standardization_dict: Dictionary with keys listed above.  If None,
        feature_table contains training data and a new dictionary will be
        created.  If specified, feature_table contains non-training data and
        will be standardized using the means and deviations from training data.
    :return: standardized_feature_table: Same as input, except that each column
        is standardized.
    :return: standardization_dict: Dictionary with keys listed above.
    :raises: ValueError: if any column of feature_table has < 2 real values
        (not NaN).
    """

    num_real_values_by_feature = numpy.sum(
        numpy.invert(numpy.isnan(feature_table.as_matrix())), axis=0)
    if numpy.any(num_real_values_by_feature < 2):
        raise ValueError('Each column of feature_table must have >= 2 real '
                         'values (not NaN).')

    if standardization_dict is None:
        feature_means = numpy.nanmean(feature_table.as_matrix(), axis=0)
        feature_standard_deviations = numpy.nanstd(
            feature_table.as_matrix(), axis=0, ddof=1)

        standardization_dict = {FEATURE_NAMES_KEY: list(feature_table),
                                ORIGINAL_MEANS_KEY: feature_means,
                                ORIGINAL_STDEVIATIONS_KEY: feature_standard_deviations}

    else:
        standardization_dict = _reorder_standardization_dict(
            standardization_dict, list(feature_table))

    feature_names = list(feature_table)
    num_features = len(feature_names)
    standardized_feature_table = None

    for j in range(num_features):
        these_standardized_values = (
            (feature_table[feature_names[j]].values -
             standardization_dict[ORIGINAL_MEANS_KEY][j]) /
            standardization_dict[ORIGINAL_STDEVIATIONS_KEY][j])

        nan_indices = numpy.where(numpy.isnan(these_standardized_values))[0]
        these_standardized_values[nan_indices] = 0.

        if standardized_feature_table is None:
            standardized_feature_table = pandas.DataFrame.from_dict(
                {feature_names[j]: these_standardized_values})
        else:
            standardized_feature_table = standardized_feature_table.assign(
                **{feature_names[j]: these_standardized_values})

    return standardized_feature_table, standardization_dict


def perform_svd(feature_table):
    """Performs SVD (singular-value decomposition) on feature matrix.

    N = number of examples
    M = number of features (input variables)

    :param feature_table: pandas DataFrame with N rows and M columns.  Column
        names are feature names.  Should contain raw (not standardized) values.
        This method calls _standardize_features before performing SVD.
    :return: standardization_dict: Dictionary returned by _standardize_features.
    :return: svd_dictionary: Dictionary with the following keys.
    svd_dictionary['principal_component_matrix']: N-by-N numpy array, where each
        column is a principal-component time series.
    svd_dictionary['eigenvalue_matrix']: M-by-M numpy array (diagonal matrix of
        eigenvalues).
    svd_dictionary['eof_matrix']: M-by-M numpy array, where each row is an
        empirical orthogonal function.
    """

    standardized_feature_table, standardization_dict = _standardize_features(
        feature_table)
    principal_component_matrix, amplitude_vector, eof_matrix = numpy.linalg.svd(
        standardized_feature_table.as_matrix())

    svd_dictionary = {
        PC_MATRIX_KEY: principal_component_matrix,
        EIGENVALUE_MATRIX_KEY: numpy.diag(amplitude_vector ** 2),
        EOF_MATRIX_KEY: numpy.transpose(eof_matrix)
    }
    return standardization_dict, svd_dictionary


def filter_svd_by_explained_variance(
        svd_dictionary,
        fraction_of_variance_to_keep=DEFAULT_FRACTION_OF_VARIANCE_TO_KEEP):
    """Filters SVD results by explained variance.

    :param svd_dictionary: Dictionary returned by perform_svd.
    :param fraction_of_variance_to_keep: Fraction of variance to keep.  Will
        select modes in descending order until they explain >=
        `fraction_of_variance_to_keep` of total variance in dataset.
    :return: svd_dictionary: Same as input, except that arrays may be shorter.
    """

    error_checking.assert_is_greater(fraction_of_variance_to_keep, 0.)
    error_checking.assert_is_less_than(fraction_of_variance_to_keep, 1.)

    eigenvalue_by_mode = numpy.diag(svd_dictionary[EIGENVALUE_MATRIX_KEY])
    explained_variance_by_mode = (
        eigenvalue_by_mode / numpy.sum(eigenvalue_by_mode))
    cumul_explained_variance_by_mode = numpy.cumsum(explained_variance_by_mode)

    num_modes_to_keep = 1 + numpy.where(
        cumul_explained_variance_by_mode >= fraction_of_variance_to_keep)[0][0]

    svd_dictionary[PC_MATRIX_KEY] = (
        svd_dictionary[PC_MATRIX_KEY][:, :num_modes_to_keep])
    svd_dictionary[EIGENVALUE_MATRIX_KEY] = (
        svd_dictionary[EIGENVALUE_MATRIX_KEY][:num_modes_to_keep,
                                              :num_modes_to_keep])
    svd_dictionary[EOF_MATRIX_KEY] = (
        svd_dictionary[EOF_MATRIX_KEY][:, :num_modes_to_keep])

    return svd_dictionary


def transform_features_via_svd(
        feature_table, standardization_dict, svd_dictionary):
    """Uses results of SVD (singular-value decomposition) to transform features.

    N = number of examples
    M = number of original features
    K = number of transformed features

    SVD results should be based on training data, whereas `feature_table` may
    contain training or non-training data.

    :param feature_table: DataFrame with N rows and M columns.  Column
        names are feature names.  Should contain raw (not standardized) values.
    :param standardization_dict: Dictionary created by _standardize_features
        from training data.
    :param svd_dictionary: Dictionary created from training data by perform_svd
        or filter_svd_by_explained_variance.
    :return: transformed_feature_matrix: N-by-K numpy array of transformed
        features (original features projected onto the first K empirical
        orthogonal functions).
    """

    standardized_feature_table, _ = _standardize_features(
        feature_table, standardization_dict=standardization_dict)
    return numpy.dot(standardized_feature_table.as_matrix(),
                     svd_dictionary[EOF_MATRIX_KEY])
