"""Deals with correlations."""

import numpy
from scipy.stats import pearsonr
from gewittergefahr.gg_utils import error_checking
from gewittergefahr.deep_learning import permutation

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'


def _linear_idx_to_matrix_channel_idxs(linear_index, num_predictors_by_matrix):
    """Converts linear predictor index to matrix channel indices.

    T = number of input tensors to the model

    :param linear_index: Linear predictor index.
    :param num_predictors_by_matrix: length-T numpy array with number of
        predictors (channels) in each input matrix.
    :return: matrix_index: Matrix index.
    :return: channel_index: Channel index.
    """

    cumsum_predictors_by_matrix = numpy.cumsum(num_predictors_by_matrix)
    matrix_index = numpy.where(linear_index < cumsum_predictors_by_matrix)[0][0]

    if matrix_index == 0:
        channel_index = linear_index
    else:
        channel_index = (
            linear_index - cumsum_predictors_by_matrix[matrix_index - 1]
        )

    return matrix_index, channel_index


def _take_spatial_mean(data_matrix):
    """Takes spatial mean over data matrix.

    E = number of examples

    :param data_matrix: numpy array, where the first axis has length E and all
        other axes represent spatial dimensions.
    :return: mean_values: length-E numpy array of means.
    """

    num_spatial_dim = len(data_matrix.shape) - 1
    these_axes = numpy.linspace(
        1, num_spatial_dim, num=num_spatial_dim, dtype=int
    ).tolist()

    return numpy.mean(data_matrix, axis=tuple(these_axes))


def get_pearson_correlations(predictor_matrices, cnn_metadata_dict,
                             separate_radar_heights=False):
    """Computes Pearson correlation between each pair of predictors.

    P = total number of predictors (over all matrices)

    :param predictor_matrices: See doc for `permutation.run_forward_test`.
    :param cnn_metadata_dict: Same.
    :param separate_radar_heights: Same.
    :return: correlation_matrix: P-by-P numpy array of Pearson correlations.
    :return: predictor_names: length-P list of predictor names.
    """

    error_checking.assert_is_boolean(separate_radar_heights)

    predictor_names_by_matrix = permutation.create_nice_predictor_names(
        predictor_matrices=predictor_matrices,
        cnn_metadata_dict=cnn_metadata_dict,
        separate_radar_heights=separate_radar_heights)

    num_matrices = len(predictor_names_by_matrix)
    for i in range(num_matrices):
        print('Predictors in {0:d}th matrix:\n{1:s}\n'.format(
            i + 1, str(predictor_names_by_matrix[i])
        ))

    print(SEPARATOR_STRING)

    predictor_names = sum(predictor_names_by_matrix, [])
    num_predictors = len(predictor_names)
    correlation_matrix = numpy.full((num_predictors, num_predictors), numpy.nan)

    predictor_matrices_to_use = [
        permutation.flatten_last_two_dim(a)[0] for a in predictor_matrices
    ]
    num_predictors_by_matrix = numpy.array(
        [a.shape[-1] for a in predictor_matrices_to_use], dtype=int
    )

    for i in range(num_predictors):
        for j in range(i, num_predictors):
            if i == j:
                correlation_matrix[i, j] = 1.
                continue

            i_matrix, i_channel = _linear_idx_to_matrix_channel_idxs(
                linear_index=i,
                num_predictors_by_matrix=num_predictors_by_matrix)

            j_matrix, j_channel = _linear_idx_to_matrix_channel_idxs(
                linear_index=j,
                num_predictors_by_matrix=num_predictors_by_matrix)

            these_first_values = _take_spatial_mean(
                predictor_matrices_to_use[i_matrix][..., i_channel]
            )
            these_second_values = _take_spatial_mean(
                predictor_matrices_to_use[j_matrix][..., j_channel]
            )

            correlation_matrix[i, j] = pearsonr(
                these_first_values, these_second_values
            )[0]
            correlation_matrix[j, i] = correlation_matrix[i, j]

    return correlation_matrix, predictor_names
