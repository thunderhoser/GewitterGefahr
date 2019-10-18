"""Unit tests for correlation.py."""

import copy
import unittest
import numpy
from gewittergefahr.gg_utils import correlation
from gewittergefahr.gg_utils import radar_utils
from gewittergefahr.gg_utils import soundings
from gewittergefahr.deep_learning import cnn
from gewittergefahr.deep_learning import training_validation_io as trainval_io

TOLERANCE = 1e-6

# The following constants are used to test _linear_idx_to_matrix_channel_idxs.
NUM_PREDICTORS_BY_MATRIX = numpy.array([4, 2, 5], dtype=int)
LINEAR_INDICES = numpy.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=int)
MATRIX_INDICES = numpy.array([0, 0, 0, 0, 1, 1, 2, 2, 2, 2, 2], dtype=int)
CHANNEL_INDICES = numpy.array([0, 1, 2, 3, 0, 1, 0, 1, 2, 3, 4], dtype=int)

# The following constants are used to test _take_spatial_mean.
SPATIAL_DATA_MATRIX = numpy.array([
    [0, 1, 2, 3],
    [4, 5, 6, 7],
    [8, 9, 10, 11]
], dtype=float)

SPATIAL_MEANS = numpy.array([1.5, 5.5, 9.5])

# The following constants are used to test get_pearson_correlations.
RADAR_HEIGHTS_M_AGL = numpy.array([2000, 6000, 10000], dtype=int)

THIS_LOW_LEVEL_MATRIX = numpy.array([
    [0, 0, 0, 1, 1, 0, 0, 0],
    [0, 0, 1, 1, 1, 1, 0, 0],
    [0, 1, 1, 1, 1, 1, 1, 0],
    [0, 1, 1, 1, 1, 1, 1, 0],
    [0, 0, 1, 1, 1, 1, 0, 0],
    [0, 0, 0, 1, 1, 0, 0, 0]
], dtype=float)

THIS_MID_LEVEL_MATRIX = THIS_LOW_LEVEL_MATRIX * 10
THIS_UPPER_LEVEL_MATRIX = THIS_LOW_LEVEL_MATRIX * 5
THIS_FIRST_EXAMPLE_MATRIX = numpy.stack(
    (THIS_LOW_LEVEL_MATRIX, THIS_MID_LEVEL_MATRIX, THIS_UPPER_LEVEL_MATRIX),
    axis=-1
)

THIS_SECOND_EXAMPLE_MATRIX = THIS_FIRST_EXAMPLE_MATRIX + 10
THIS_THIRD_EXAMPLE_MATRIX = THIS_FIRST_EXAMPLE_MATRIX + 20
THIS_FOURTH_EXAMPLE_MATRIX = THIS_FIRST_EXAMPLE_MATRIX + 30
REFLECTIVITY_MATRIX_DBZ = numpy.stack((
    THIS_FIRST_EXAMPLE_MATRIX, THIS_SECOND_EXAMPLE_MATRIX,
    THIS_THIRD_EXAMPLE_MATRIX, THIS_FOURTH_EXAMPLE_MATRIX
), axis=0)

LOW_LEVEL_SHEAR_MATRIX_S01 = 0.001 * REFLECTIVITY_MATRIX_DBZ[..., 0]
MID_LEVEL_SHEAR_MATRIX_S01 = 0.001 * REFLECTIVITY_MATRIX_DBZ[..., 1]
VORTICITY_MATRIX_S01 = 0.001 * REFLECTIVITY_MATRIX_DBZ

AZIMUTHAL_SHEAR_MATRIX_S01 = numpy.stack(
    (LOW_LEVEL_SHEAR_MATRIX_S01, MID_LEVEL_SHEAR_MATRIX_S01), axis=-1
)
AZ_SHEAR_FIELD_NAMES = [
    radar_utils.LOW_LEVEL_SHEAR_NAME, radar_utils.MID_LEVEL_SHEAR_NAME
]

RADAR_MATRIX_3D = numpy.stack(
    (REFLECTIVITY_MATRIX_DBZ, VORTICITY_MATRIX_S01), axis=-1
)
RADAR_FIELD_NAMES_3D = [radar_utils.REFL_NAME, radar_utils.VORTICITY_NAME]

REFLECTIVITY_MATRIX_DBZ = numpy.expand_dims(REFLECTIVITY_MATRIX_DBZ, axis=-1)

THIS_TEMPERATURE_MATRIX_CELSIUS = numpy.array([
    [5, 4, 3, 2, 1, 0, -1],
    [12, 11, 10, 9, 8, 7, 6],
    [19, 18, 17, 16, 15, 14, 13],
    [26, 25, 24, 23, 22, 21, 20]
], dtype=float)

THIS_TEMPERATURE_MATRIX_KELVINS = 273.15 + THIS_TEMPERATURE_MATRIX_CELSIUS
THIS_RH_MATRIX_UNITLESS = 0.01 * (THIS_TEMPERATURE_MATRIX_CELSIUS + 1.)

SOUNDING_MATRIX = numpy.stack(
    (THIS_TEMPERATURE_MATRIX_KELVINS, THIS_RH_MATRIX_UNITLESS), axis=-1
)
SOUNDING_FIELD_NAMES = [
    soundings.TEMPERATURE_NAME, soundings.RELATIVE_HUMIDITY_NAME
]

THIS_TRAINING_OPTION_DICT = {
    trainval_io.RADAR_FIELDS_KEY:
        [radar_utils.LOW_LEVEL_SHEAR_NAME, radar_utils.MID_LEVEL_SHEAR_NAME],
    trainval_io.RADAR_HEIGHTS_KEY: RADAR_HEIGHTS_M_AGL,
    trainval_io.SOUNDING_FIELDS_KEY: SOUNDING_FIELD_NAMES
}

FIRST_METADATA_DICT = {
    cnn.CONV_2D3D_KEY: True,
    cnn.TRAINING_OPTION_DICT_KEY: THIS_TRAINING_OPTION_DICT
}

FIRST_PREDICTOR_MATRICES = [
    REFLECTIVITY_MATRIX_DBZ, AZIMUTHAL_SHEAR_MATRIX_S01, SOUNDING_MATRIX
]

FIRST_PREDICTOR_NAMES_HGTS_TOGETHER = (
    [radar_utils.REFL_NAME] + AZ_SHEAR_FIELD_NAMES + SOUNDING_FIELD_NAMES
)

FIRST_PREDICTOR_NAMES_HGTS_SEPARATE = [
    '{0:s}_2000metres'.format(radar_utils.REFL_NAME),
    '{0:s}_6000metres'.format(radar_utils.REFL_NAME),
    '{0:s}_10000metres'.format(radar_utils.REFL_NAME)
]

FIRST_PREDICTOR_NAMES_HGTS_SEPARATE += (
    AZ_SHEAR_FIELD_NAMES + SOUNDING_FIELD_NAMES
)

THIS_TRAINING_OPTION_DICT = {
    trainval_io.RADAR_FIELDS_KEY:
        [radar_utils.LOW_LEVEL_SHEAR_NAME, radar_utils.MID_LEVEL_SHEAR_NAME],
    trainval_io.RADAR_HEIGHTS_KEY: RADAR_HEIGHTS_M_AGL,
    trainval_io.SOUNDING_FIELDS_KEY: None
}

SECOND_METADATA_DICT = {
    cnn.CONV_2D3D_KEY: True,
    cnn.TRAINING_OPTION_DICT_KEY: THIS_TRAINING_OPTION_DICT
}

SECOND_PREDICTOR_MATRICES = [
    REFLECTIVITY_MATRIX_DBZ, AZIMUTHAL_SHEAR_MATRIX_S01
]

SECOND_PREDICTOR_NAMES_HGTS_TOGETHER = (
    [radar_utils.REFL_NAME] + AZ_SHEAR_FIELD_NAMES
)

SECOND_PREDICTOR_NAMES_HGTS_SEPARATE = [
    '{0:s}_2000metres'.format(radar_utils.REFL_NAME),
    '{0:s}_6000metres'.format(radar_utils.REFL_NAME),
    '{0:s}_10000metres'.format(radar_utils.REFL_NAME)
]

SECOND_PREDICTOR_NAMES_HGTS_SEPARATE += AZ_SHEAR_FIELD_NAMES

THIS_TRAINING_OPTION_DICT = {
    trainval_io.RADAR_FIELDS_KEY: RADAR_FIELD_NAMES_3D,
    trainval_io.RADAR_HEIGHTS_KEY: RADAR_HEIGHTS_M_AGL,
    trainval_io.SOUNDING_FIELDS_KEY: SOUNDING_FIELD_NAMES
}

THIRD_METADATA_DICT = {
    cnn.CONV_2D3D_KEY: False,
    cnn.TRAINING_OPTION_DICT_KEY: THIS_TRAINING_OPTION_DICT
}

THIRD_PREDICTOR_MATRICES = [RADAR_MATRIX_3D, SOUNDING_MATRIX]

THIRD_PREDICTOR_NAMES_HGTS_TOGETHER = (
    RADAR_FIELD_NAMES_3D + SOUNDING_FIELD_NAMES
)

THIRD_PREDICTOR_NAMES_HGTS_SEPARATE = [
    '{0:s}_2000metres'.format(radar_utils.REFL_NAME),
    '{0:s}_6000metres'.format(radar_utils.REFL_NAME),
    '{0:s}_10000metres'.format(radar_utils.REFL_NAME),
    '{0:s}_2000metres'.format(radar_utils.VORTICITY_NAME),
    '{0:s}_6000metres'.format(radar_utils.VORTICITY_NAME),
    '{0:s}_10000metres'.format(radar_utils.VORTICITY_NAME)
]

THIRD_PREDICTOR_NAMES_HGTS_SEPARATE += SOUNDING_FIELD_NAMES


class CorrelationTests(unittest.TestCase):
    """Each method is a unit test for correlation.py."""

    def test_linear_idx_to_matrix_channel_idxs(self):
        """Ensures correct output from _linear_idx_to_matrix_channel_idxs."""

        for (i, j, k) in zip(MATRIX_INDICES, CHANNEL_INDICES, LINEAR_INDICES):
            this_i, this_j = correlation._linear_idx_to_matrix_channel_idxs(
                linear_index=k,
                num_predictors_by_matrix=NUM_PREDICTORS_BY_MATRIX)

            self.assertTrue(this_i == i)
            self.assertTrue(this_j == j)

    def test_take_spatial_mean(self):
        """Ensures correct output from _take_spatial_mean."""

        these_means = correlation._take_spatial_mean(SPATIAL_DATA_MATRIX)

        self.assertTrue(numpy.allclose(
            these_means, SPATIAL_MEANS, atol=TOLERANCE
        ))

    def test_get_pearson_correlations_first_hgts_together(self):
        """Ensures correct output from get_pearson_correlations.

        In this case, using first set of inputs with
        `separate_radar_heights = False`.
        """

        this_correlation_matrix, these_predictor_names = (
            correlation.get_pearson_correlations(
                denorm_predictor_matrices=copy.deepcopy(
                    FIRST_PREDICTOR_MATRICES),
                cnn_metadata_dict=copy.deepcopy(FIRST_METADATA_DICT),
                separate_radar_heights=False
            )
        )

        self.assertTrue(
            these_predictor_names == FIRST_PREDICTOR_NAMES_HGTS_TOGETHER
        )

        num_predictors = len(these_predictor_names)
        expected_dim = (num_predictors, num_predictors)
        self.assertTrue(this_correlation_matrix.shape == expected_dim)

    def test_get_pearson_correlations_first_hgts_separate(self):
        """Ensures correct output from get_pearson_correlations.

        In this case, using first set of inputs with
        `separate_radar_heights = True`.
        """

        this_correlation_matrix, these_predictor_names = (
            correlation.get_pearson_correlations(
                denorm_predictor_matrices=copy.deepcopy(
                    FIRST_PREDICTOR_MATRICES),
                cnn_metadata_dict=copy.deepcopy(FIRST_METADATA_DICT),
                separate_radar_heights=True
            )
        )

        self.assertTrue(
            these_predictor_names == FIRST_PREDICTOR_NAMES_HGTS_SEPARATE
        )

        num_predictors = len(these_predictor_names)
        expected_dim = (num_predictors, num_predictors)
        self.assertTrue(this_correlation_matrix.shape == expected_dim)

    def test_get_pearson_correlations_second_hgts_together(self):
        """Ensures correct output from get_pearson_correlations.

        In this case, using second set of inputs with
        `separate_radar_heights = False`.
        """

        this_correlation_matrix, these_predictor_names = (
            correlation.get_pearson_correlations(
                denorm_predictor_matrices=copy.deepcopy(
                    SECOND_PREDICTOR_MATRICES),
                cnn_metadata_dict=copy.deepcopy(SECOND_METADATA_DICT),
                separate_radar_heights=False
            )
        )

        self.assertTrue(
            these_predictor_names == SECOND_PREDICTOR_NAMES_HGTS_TOGETHER
        )

        num_predictors = len(these_predictor_names)
        expected_dim = (num_predictors, num_predictors)
        self.assertTrue(this_correlation_matrix.shape == expected_dim)

    def test_get_pearson_correlations_second_hgts_separate(self):
        """Ensures correct output from get_pearson_correlations.

        In this case, using second set of inputs with
        `separate_radar_heights = True`.
        """

        this_correlation_matrix, these_predictor_names = (
            correlation.get_pearson_correlations(
                denorm_predictor_matrices=copy.deepcopy(
                    SECOND_PREDICTOR_MATRICES),
                cnn_metadata_dict=copy.deepcopy(SECOND_METADATA_DICT),
                separate_radar_heights=True
            )
        )

        self.assertTrue(
            these_predictor_names == SECOND_PREDICTOR_NAMES_HGTS_SEPARATE
        )

        num_predictors = len(these_predictor_names)
        expected_dim = (num_predictors, num_predictors)
        self.assertTrue(this_correlation_matrix.shape == expected_dim)

    def test_get_pearson_correlations_third_hgts_together(self):
        """Ensures correct output from get_pearson_correlations.

        In this case, using third set of inputs with
        `separate_radar_heights = False`.
        """

        this_correlation_matrix, these_predictor_names = (
            correlation.get_pearson_correlations(
                denorm_predictor_matrices=copy.deepcopy(
                    THIRD_PREDICTOR_MATRICES),
                cnn_metadata_dict=copy.deepcopy(THIRD_METADATA_DICT),
                separate_radar_heights=False
            )
        )

        self.assertTrue(
            these_predictor_names == THIRD_PREDICTOR_NAMES_HGTS_TOGETHER
        )

        num_predictors = len(these_predictor_names)
        expected_dim = (num_predictors, num_predictors)
        self.assertTrue(this_correlation_matrix.shape == expected_dim)

    def test_get_pearson_correlations_third_hgts_separate(self):
        """Ensures correct output from get_pearson_correlations.

        In this case, using third set of inputs with
        `separate_radar_heights = True`.
        """

        this_correlation_matrix, these_predictor_names = (
            correlation.get_pearson_correlations(
                denorm_predictor_matrices=copy.deepcopy(
                    THIRD_PREDICTOR_MATRICES),
                cnn_metadata_dict=copy.deepcopy(THIRD_METADATA_DICT),
                separate_radar_heights=True
            )
        )

        self.assertTrue(
            these_predictor_names == THIRD_PREDICTOR_NAMES_HGTS_SEPARATE
        )

        num_predictors = len(these_predictor_names)
        expected_dim = (num_predictors, num_predictors)
        self.assertTrue(this_correlation_matrix.shape == expected_dim)


if __name__ == '__main__':
    unittest.main()
