"""Deals with correlations."""

import numpy
from scipy.stats import pearsonr
from gewittergefahr.gg_utils import radar_utils
from gewittergefahr.gg_utils import error_checking
from gewittergefahr.deep_learning import cnn
from gewittergefahr.deep_learning import training_validation_io as trainval_io


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


def get_pearson_correlations(denorm_predictor_matrices, cnn_metadata_dict,
                             separate_radar_heights=False):
    """Computes Pearson correlation between each pair of predictors.

    T = number of input tensors to the model
    P = total number of predictors (over all tensors)

    :param denorm_predictor_matrices: length-T list of denormalized predictor
        matrices.  Each item must be a numpy array.
    :param cnn_metadata_dict: Dictionary with CNN metadata (see doc for
        `cnn.read_model_metadata`).
    :param separate_radar_heights: Boolean flag.  If the inputs include 3-D
        radar fields and `separate_radar_heights == True`, will consider each
        variable/height pair separately, rather than each variable separately.
    :return: correlation_matrix: P-by-P numpy array of Pearson correlations.
    :return: predictor_names: length-P list of predictor names.
    """

    # TODO(thunderhoser): Deal with layer operations.

    # Check input args.
    num_spatial_dim_by_matrix = numpy.array(
        [len(a.shape) - 2 for a in denorm_predictor_matrices], dtype=int
    )

    error_checking.assert_is_geq_numpy_array(num_spatial_dim_by_matrix, 1)
    error_checking.assert_is_boolean(separate_radar_heights)

    separate_radar_heights = (
        separate_radar_heights and numpy.any(num_spatial_dim_by_matrix == 3)
    )

    # Create list of predictor names.
    myrorss_2d3d = cnn_metadata_dict[cnn.CONV_2D3D_KEY]
    training_option_dict = cnn_metadata_dict[cnn.TRAINING_OPTION_DICT_KEY]
    radar_field_names = training_option_dict[trainval_io.RADAR_FIELDS_KEY]
    radar_heights_m_agl = training_option_dict[trainval_io.RADAR_HEIGHTS_KEY]

    if myrorss_2d3d:
        if separate_radar_heights:
            predictor_names = [
                '{0:s}_{1:d}metres'.format(radar_utils.REFL_NAME, h)
                for h in radar_heights_m_agl
            ]

            denorm_predictor_matrices[0] = denorm_predictor_matrices[0][..., 0]
        else:
            predictor_names = [radar_utils.REFL_NAME]

        predictor_names += radar_field_names

    else:
        if separate_radar_heights:
            height_matrix_m_agl, field_name_matrix = numpy.meshgrid(
                numpy.array(radar_heights_m_agl),
                numpy.array(radar_field_names)
            )

            radar_field_names = numpy.ravel(field_name_matrix)
            radar_heights_m_agl = numpy.round(
                numpy.ravel(height_matrix_m_agl)
            ).astype(int)

            predictor_names = [
                '{0:s}_{1:d}metres'.format(f, h)
                for f, h in zip(radar_field_names, radar_heights_m_agl)
            ]

            new_spatial_dim = numpy.array(
                denorm_predictor_matrices[0].shape, dtype=int
            )
            new_spatial_dim[-2] = new_spatial_dim[-2] * new_spatial_dim[-1]
            new_spatial_dim = new_spatial_dim[:-1]

            denorm_predictor_matrices[0] = numpy.reshape(
                denorm_predictor_matrices[0], new_spatial_dim
            )
        else:
            predictor_names = radar_field_names

    sounding_field_names = training_option_dict[trainval_io.SOUNDING_FIELDS_KEY]
    if sounding_field_names is not None:
        predictor_names += sounding_field_names

    num_predictors_by_matrix = numpy.array(
        [a.shape[-1] for a in denorm_predictor_matrices], dtype=int
    )

    num_predictors = numpy.sum(num_predictors_by_matrix)
    correlation_matrix = numpy.full((num_predictors, num_predictors), numpy.nan)

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
                denorm_predictor_matrices[i_matrix][..., i_channel]
            )
            these_second_values = _take_spatial_mean(
                denorm_predictor_matrices[j_matrix][..., j_channel]
            )

            correlation_matrix[i, j] = pearsonr(
                these_first_values, these_second_values
            )[0]
            correlation_matrix[j, i] = correlation_matrix[i, j]

    return correlation_matrix, predictor_names
