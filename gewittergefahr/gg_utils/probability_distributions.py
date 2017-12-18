"""Methods for handling probability distributions."""

import numpy
import pandas
import scipy.stats
from gewittergefahr.gg_utils import error_checking

FEATURE_NAMES_KEY = 'feature_names'
FEATURE_MEANS_KEY = 'feature_means'
COVARIANCE_MATRIX_KEY = 'covariance_matrix'
COVAR_MATRIX_INVERSE_KEY = 'covar_matrix_inverse'
COVAR_MATRIX_DETERMINANT_KEY = 'covar_matrix_determinant'

PRIOR_CLASS_PROBABILITY_KEY = 'prior_class_probability'
ORIG_FEATURE_TABLE_KEY = 'orig_feature_table'

MIN_CUMUL_DENSITY_FOR_NORMAL_DIST = 1e-6
MAX_CUMUL_DENSITY_FOR_NORMAL_DIST = 1. - 1e-6


def _transform_each_marginal_to_uniform(
        new_feature_table, orig_feature_table=None):
    """Transforms marginal distribution of each feature to uniform distribution.

    This method transforms data in `new_feature_table` only.

    If `orig_feature_table` is None, the transformation for feature "x" in the
    [i]th example will be based on the percentile score of
    new_feature_table["x"].values[i] in new_feature_table["x"].values.

    If `orig_feature_table` is specified, the transformation for feature "x" in
    the [i]th example will be based on the percentile score of
    new_feature_table["x"].values[i] in orig_feature_table["x"].values.

    P = number of original examples
    Q = number of new examples
    M = number of features

    :param new_feature_table: pandas DataFrame with Q rows and M columns.
        Column names are feature names.
    :param orig_feature_table: pandas DataFrame with P rows and M columns.
        Column names are feature names.
    :return: transformed_new_feature_table: Same as input, except that the
        marginal distribution of each column is uniform.
    """

    # TODO(thunderhoser): I could probably make this faster for cases where
    # `orig_feature_table` is specified.

    feature_names = list(new_feature_table)
    new_feature_matrix = new_feature_table.as_matrix(columns=feature_names)

    if orig_feature_table is not None:
        error_checking.assert_columns_in_dataframe(
            orig_feature_table, feature_names)
        orig_feature_matrix = orig_feature_table.as_matrix(
            columns=feature_names)

    num_features = len(feature_names)
    num_new_examples = new_feature_matrix.shape[0]
    transformed_new_feature_table = None

    for j in range(num_features):
        new_indices_to_use = numpy.where(
            numpy.invert(numpy.isnan(new_feature_matrix[:, j])))[0]
        transformed_values = numpy.full(num_new_examples, 0.5)

        if orig_feature_table is None:
            these_ranks = scipy.stats.rankdata(
                new_feature_matrix[new_indices_to_use, j], method='average')
            transformed_values[new_indices_to_use] = (
                these_ranks / len(new_indices_to_use))
        else:
            orig_indices_to_use = numpy.where(
                numpy.invert(numpy.isnan(orig_feature_matrix[:, j])))[0]

            for i in new_indices_to_use:
                transformed_values[i] = scipy.stats.percentileofscore(
                    orig_feature_matrix[orig_indices_to_use, j],
                    new_feature_matrix[i, j], kind='weak') / 100

        if transformed_new_feature_table is None:
            transformed_new_feature_table = pandas.DataFrame.from_dict(
                {feature_names[j]: transformed_values})
        else:
            transformed_new_feature_table = (
                transformed_new_feature_table.assign(
                    **{feature_names[j]: transformed_values}))

    return transformed_new_feature_table


def _transform_each_marginal_to_normal(
        new_feature_table, orig_feature_table=None):
    """Transforms marginal distribution of each feature to normal distribution.

    To learn about the roles of `new_feature_table` and `orig_feature_table`,
    see documentation for _transform_each_marginal_to_uniform.

    :param new_feature_table: See doc for _transform_each_marginal_to_uniform.
    :param orig_feature_table: See doc for _transform_each_marginal_to_uniform.
    :return: transformed_new_feature_table: Same as input, except that the
        marginal distribution of each column is normal.
    """

    transformed_new_feature_table = _transform_each_marginal_to_uniform(
        new_feature_table, orig_feature_table)

    for this_column in list(transformed_new_feature_table):
        these_values = transformed_new_feature_table[this_column].values

        these_values[these_values < MIN_CUMUL_DENSITY_FOR_NORMAL_DIST] = (
            MIN_CUMUL_DENSITY_FOR_NORMAL_DIST)
        these_values[these_values > MAX_CUMUL_DENSITY_FOR_NORMAL_DIST] = (
            MAX_CUMUL_DENSITY_FOR_NORMAL_DIST)
        these_values = scipy.stats.norm.ppf(these_values, loc=0, scale=1)

        transformed_new_feature_table = transformed_new_feature_table.assign(
            **{this_column: these_values})

    return transformed_new_feature_table


def _normalize_class_probabilities(class_probability_matrix):
    """Normalizes class probabilities, so that probs for each example sum to 1.

    N = number of examples
    K = number of classes

    :param class_probability_matrix: N-by-K numpy array of class probabilities.
    :return: class_probability_matrix: Same as input, except that each row sums
        to 1.
    """

    error_checking.assert_is_geq_numpy_array(
        class_probability_matrix, 0., allow_nan=True)

    row_with_nan_flags = numpy.any(
        numpy.isnan(class_probability_matrix), axis=1)
    row_with_nan_indices = numpy.where(row_with_nan_flags)[0]
    class_probability_matrix[row_with_nan_indices, :] = numpy.nan

    max_original_prob = numpy.nanmax(class_probability_matrix)
    min_original_prob = numpy.nanmin(class_probability_matrix)
    class_probability_matrix = (
        (class_probability_matrix - min_original_prob) /
        (max_original_prob - min_original_prob))

    row_without_nan_indices = numpy.where(numpy.invert(row_with_nan_flags))[0]
    for this_row in row_without_nan_indices:
        class_probability_matrix[this_row, :] = (
            class_probability_matrix[this_row, :] /
            numpy.sum(class_probability_matrix[this_row, :]))

    return class_probability_matrix


def _get_feature_means(feature_matrix):
    """Computes mean of each feature.

    N = number of examples
    M = number of features (input variables)

    :param feature_matrix: N-by-M numpy array.  feature_matrix[i, j] is the
        value of the [j]th feature for the [i]th example.
    :return: feature_means: length-M numpy array with mean value of each
        feature.
    """

    return numpy.nanmean(feature_matrix, axis=0)


def _get_covariance_matrix(feature_matrix, assume_diagonal=False):
    """Computes covariance matrix.

    N = number of examples
    M = number of features (input variables)

    :param feature_matrix: N-by-M numpy array.  feature_matrix[i, j] is the
        value of the [j]th feature for the [i]th example.
    :param assume_diagonal: Boolean flag.  If True, assumes diagonal covariance
        matrix.
    :return: covariance_matrix: M-by-M numpy array.  Entry [i, j] is the
        covariance between the [i]th and [j]th features.
    :return: feature_means: length-M numpy array with mean value of each
        feature.
    """

    num_features = feature_matrix.shape[1]
    feature_means = _get_feature_means(feature_matrix)

    if assume_diagonal:
        covariance_matrix = numpy.diag(
            numpy.nanvar(feature_matrix, axis=0, ddof=1))
    else:
        covariance_matrix = numpy.full((num_features, num_features), numpy.nan)

        for j in range(num_features):
            for k in range(j + 1):
                nan_flags = numpy.logical_or(
                    numpy.isnan(feature_matrix[:, j]),
                    numpy.isnan(feature_matrix[:, k]))
                not_nan_indices = numpy.where(numpy.invert(nan_flags))[0]

                covariance_matrix[j, k] = numpy.sum(
                    (feature_matrix[not_nan_indices, j] - feature_means[j]) *
                    (feature_matrix[not_nan_indices, k] - feature_means[k]))
                covariance_matrix[j, k] = (
                    (1. / (len(not_nan_indices) - 1)) * covariance_matrix[j, k])
                covariance_matrix[k, j] = covariance_matrix[j, k]

    return covariance_matrix, feature_means


def fit_multivariate_normal(feature_table, assume_diagonal_covar_matrix=False):
    """Fits data to a multivariate normal distribution.

    N = number of examples
    M = number of features (input variables)

    :param feature_table: pandas DataFrame with N rows and M columns.  Column
        names are feature names.
    :param assume_diagonal_covar_matrix: Boolean flag.  If True, will assume a
        diagonal covariance matrix (where all off-diagonal entries are zero).
        This eliminates the risk of a singular (non-invertible) covariance
        matrix.
    :return: multivariate_normal_dict: Dictionary with the following keys.
    multivariate_normal_dict['feature_names']: length-M list of feature names.
    multivariate_normal_dict['feature_means']: length-M numpy array with mean
        value of each feature.
    multivariate_normal_dict['covariance_matrix']: M-by-M numpy array.  Entry
        [i, j] is the covariance between the [i]th and [j]th features.
    multivariate_normal_dict['covar_matrix_inverse']: Inverse of covariance
        matrix.
    multivariate_normal_dict['covar_matrix_determinant']: Determinant of
        covariance matrix.
    :raises: ValueError: if any column of feature_matrix has < 2 real values
        (not NaN).
    """

    error_checking.assert_is_boolean(assume_diagonal_covar_matrix)

    num_real_values_by_feature = numpy.sum(
        numpy.invert(numpy.isnan(feature_table.as_matrix())), axis=0)
    if numpy.any(num_real_values_by_feature < 2):
        raise ValueError('Each column of feature_table must have >= 2 real '
                         'values (not NaN).')

    covariance_matrix, feature_means = _get_covariance_matrix(
        _transform_each_marginal_to_normal(feature_table).as_matrix(),
        assume_diagonal=assume_diagonal_covar_matrix)

    return {FEATURE_NAMES_KEY: list(feature_table),
            FEATURE_MEANS_KEY: feature_means,
            COVARIANCE_MATRIX_KEY: covariance_matrix,
            COVAR_MATRIX_INVERSE_KEY: numpy.linalg.inv(covariance_matrix),
            COVAR_MATRIX_DETERMINANT_KEY: numpy.linalg.det(covariance_matrix)}


def fit_mvn_for_each_class(feature_table, class_labels, num_classes,
                           assume_diagonal_covar_matrix=False):
    """For each class, fits data to a multivariate normal distribution.

    N = number of examples
    M = number of features (input variables)
    K = number of classes

    :param feature_table: pandas DataFrame with N rows and M columns.  Column
        names are feature names.
    :param class_labels: length-N numpy array of class labels.  Should be
        integers ranging from 0...[num_classes - 1].
    :param num_classes: Number of classes.
    :param assume_diagonal_covar_matrix: See documentation for
        fit_multivariate_normal.
    :return: list_of_mvn_dictionaries: length-K list of dictionaries, each with
        the following keys.
    list_of_mvn_dictionaries[k]['prior_class_probability']: Prior probability of
        [k]th class.  This is the frequency of value (k - 1) in `class_labels`.
    list_of_mvn_dictionaries[k]['orig_feature_table']: Original feature table
        (before transforming marginals to normal distribution) for [k]th class.
    list_of_mvn_dictionaries[k]['feature_names']: length-M list of feature names
        (same for each class).
    list_of_mvn_dictionaries[k]['feature_means']: length-M numpy array with mean
        value of each feature, given the [k]th class.
    list_of_mvn_dictionaries[k]['covariance_matrix']: M-by-M numpy array.
        Covariance matrix, given the [k]th class.
    list_of_mvn_dictionaries[k]['covar_matrix_inverse']: Inverse of covariance
        matrix for [k]th class.
    list_of_mvn_dictionaries[k]['covar_matrix_determinant']: Determinant of
        covariance matrix for [k]th class.
    :raises: ValueError: if any class is not represented in `class_labels`.
    """

    num_examples = len(feature_table.index)

    error_checking.assert_is_integer(num_classes)
    error_checking.assert_is_geq(num_classes, 2)
    error_checking.assert_is_integer_numpy_array(class_labels)
    error_checking.assert_is_numpy_array(
        class_labels, exact_dimensions=numpy.array([num_examples]))
    error_checking.assert_is_geq_numpy_array(class_labels, 0)
    error_checking.assert_is_less_than_numpy_array(class_labels, num_classes)

    list_of_mvn_dictionaries = []
    for k in range(num_classes):
        these_flags = class_labels == k
        if not numpy.any(these_flags):
            error_string = ('Class {0:d} (label {1:d}) does not exist in the '
                            'input data.').format(k + 1, k)
            raise ValueError(error_string)

        these_indices = numpy.where(these_flags)[0]
        this_dict = fit_multivariate_normal(
            feature_table.iloc[these_indices],
            assume_diagonal_covar_matrix=assume_diagonal_covar_matrix)

        this_dict.update({PRIOR_CLASS_PROBABILITY_KEY:
                              float(len(these_indices)) / num_examples})
        this_dict.update(
            {ORIG_FEATURE_TABLE_KEY: feature_table.iloc[these_indices]})

        list_of_mvn_dictionaries.append(this_dict)

    return list_of_mvn_dictionaries


def apply_mvn_for_each_class(feature_table, list_of_mvn_dictionaries):
    """Uses multivariate normal distributions to predict class probabilities.

    In other words, this method applies previously fit MVN distributions to
    predict new examples.  Thus, `feature_matrix` may not be the same feature
    matrix used to fit the distributions (i.e., used to create
    `list_of_mvn_dictionaries`).

    N = number of examples
    M = number of features (input variables)
    K = number of classes

    :param feature_table: pandas DataFrame with N rows and M columns.  Column
        names are feature names.
    :param list_of_mvn_dictionaries: length-K list of dictionaries created by
        fit_mvn_for_each_class.
    :return: forecast_prob_matrix: N-by-K numpy array of forecast probabilities.
        Entry [i, k] is the forecast probability that the [i]th example comes
        from the [k]th class.
    :raises: ValueError: if list_of_mvn_dictionaries[k]["feature_names"] does
        not match columns of feature_table for any class k.
    """

    num_classes = len(list_of_mvn_dictionaries)
    feature_names = list(feature_table)
    num_features = len(feature_names)

    for k in range(num_classes):
        features_match = (
            len(list_of_mvn_dictionaries[k][FEATURE_NAMES_KEY]) == num_features
            and list_of_mvn_dictionaries[k][FEATURE_NAMES_KEY] == feature_names)

        if not features_match:
            error_string = (
                str(feature_names) + '\n\nFeature names in feature_table ' +
                '(shown above) and the MVN dictionary for class ' + str(k + 1) +
                ' (shown below) do not match.\n\n' +
                str(list_of_mvn_dictionaries[k][FEATURE_NAMES_KEY]))
            raise ValueError(error_string)

    num_examples = len(feature_table.index)
    forecast_prob_matrix = numpy.full((num_examples, num_classes), numpy.nan)

    for k in range(num_classes):
        transformed_feature_table = _transform_each_marginal_to_normal(
            feature_table, orig_feature_table=
            list_of_mvn_dictionaries[k][ORIG_FEATURE_TABLE_KEY])
        transformed_feature_matrix = transformed_feature_table.as_matrix()

        for i in range(num_examples):
            this_deviation_vector = (
                transformed_feature_matrix[i, :] -
                list_of_mvn_dictionaries[k][FEATURE_MEANS_KEY])
            this_deviation_vector = numpy.reshape(
                this_deviation_vector, (num_features, 1))

            this_matrix_product = numpy.dot(
                list_of_mvn_dictionaries[k][COVAR_MATRIX_INVERSE_KEY],
                this_deviation_vector)
            this_matrix_product = numpy.dot(
                numpy.transpose(this_deviation_vector), this_matrix_product)[
                    0, 0]

            forecast_prob_matrix[i, k] = -0.5 * (
                numpy.log(
                    list_of_mvn_dictionaries[k][COVAR_MATRIX_DETERMINANT_KEY]) +
                this_matrix_product)

    forecast_prob_matrix = numpy.exp(
        forecast_prob_matrix - 0.5 * num_classes * numpy.log(2 * numpy.pi))
    for k in range(num_classes):
        forecast_prob_matrix[:, k] = (
            forecast_prob_matrix[:, k] *
            list_of_mvn_dictionaries[k][PRIOR_CLASS_PROBABILITY_KEY])

    return _normalize_class_probabilities(forecast_prob_matrix)


class MultivariateNormalDist(object):
    """Class for multivariate normal distributions.

    Has `fit` and `predict` methods, just like scikit-learn models.
    """

    DEFAULT_NUM_CLASSES = 2

    def __init__(self, num_classes=DEFAULT_NUM_CLASSES,
                 assume_diagonal_covar_matrix=False):
        """Constructor.

        :param num_classes: Number of classes for target variable.  See
            documentation for fit_mvn_for_each_class.
        :param assume_diagonal_covar_matrix: See documentation for
            fit_multivariate_normal.
        """

        self.list_of_mvn_dictionaries = None
        self.num_classes = num_classes
        self.assume_diagonal_covar_matrix = assume_diagonal_covar_matrix

    def fit(self, feature_table, class_labels):
        """Fits data to a multivariate normal distribution for each class.

        :param feature_table: See documentation for fit_mvn_for_each_class.
        :param class_labels: See documentation for fit_mvn_for_each_class.
        """

        self.list_of_mvn_dictionaries = fit_mvn_for_each_class(
            feature_table, class_labels, num_classes=self.num_classes,
            assume_diagonal_covar_matrix=self.assume_diagonal_covar_matrix)

    def predict(self, feature_table):
        """Uses multivariate normal distributions to predict class probs.

        :param feature_table: See documentation for apply_mvn_for_each_class.
        :return: forecast_prob_matrix: See documentation for
            apply_mvn_for_each_class.
        """

        return apply_mvn_for_each_class(
            feature_table, self.list_of_mvn_dictionaries)
