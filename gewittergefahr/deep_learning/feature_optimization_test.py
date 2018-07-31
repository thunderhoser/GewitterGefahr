"""Unit tests for feature_optimization.py."""

import unittest
import numpy
from gewittergefahr.gg_utils import radar_utils
from gewittergefahr.gg_utils import soundings_only
from gewittergefahr.deep_learning import feature_optimization

TOLERANCE = 1e-6

ARRAY_DIMENSIONS_2D = numpy.array([1, 2], dtype=int)
ARRAY_DIMENSIONS_3D = numpy.array([1, 37, 6], dtype=int)
ARRAY_DIMENSIONS_4D = numpy.array([1, 32, 32, 12], dtype=int)
ARRAY_DIMENSIONS_5D = numpy.array([1, 16, 16, 12, 4], dtype=int)

# The following constants are used to test create_gaussian_initializer.
GAUSSIAN_MEAN = 0.
GAUSSIAN_STANDARD_DEVIATION = 0.
GAUSSIAN_MATRIX_3D = numpy.full(ARRAY_DIMENSIONS_3D, 0.)
GAUSSIAN_MATRIX_4D = numpy.full(ARRAY_DIMENSIONS_4D, 0.)
GAUSSIAN_MATRIX_5D = numpy.full(ARRAY_DIMENSIONS_5D, 0.)

# The following constants are used to test create_uniform_random_initializer.
MIN_UNIFORM_VALUE = 1.
MAX_UNIFORM_VALUE = 1.
UNIFORM_MATRIX_3D = numpy.full(ARRAY_DIMENSIONS_3D, 1.)
UNIFORM_MATRIX_4D = numpy.full(ARRAY_DIMENSIONS_4D, 1.)
UNIFORM_MATRIX_5D = numpy.full(ARRAY_DIMENSIONS_5D, 1.)

# The following constants are used to test create_constant_initializer.
CONSTANT_VALUE = 2.
CONSTANT_MATRIX_3D = numpy.full(ARRAY_DIMENSIONS_3D, 2.)
CONSTANT_MATRIX_4D = numpy.full(ARRAY_DIMENSIONS_4D, 2.)
CONSTANT_MATRIX_5D = numpy.full(ARRAY_DIMENSIONS_5D, 2.)

# The following constants are used to test create_climo_initializer.
RADAR_FIELD_NAMES = [radar_utils.REFL_NAME, radar_utils.SPECTRUM_WIDTH_NAME]
RADAR_HEIGHTS_M_ASL = numpy.array([1000, 2000, 3000], dtype=int)

RADAR_FIELD_NAME_BY_CHANNEL = [
    radar_utils.REFL_NAME, radar_utils.REFL_NAME, radar_utils.REFL_NAME,
    radar_utils.REFL_COLUMN_MAX_NAME]
RADAR_HEIGHT_BY_CHANNEL_M_ASL = numpy.array([1000, 2000, 3000, 250], dtype=int)

SOUNDING_FIELD_NAMES = [
    soundings_only.TEMPERATURE_NAME,
    soundings_only.VIRTUAL_POTENTIAL_TEMPERATURE_NAME
]
SOUNDING_PRESSURES_MB = numpy.array([500, 700, 850, 1000], dtype=int)

MEAN_RADAR_VALUE_DICT = {}
MEAN_RADAR_VALUE_DICT[radar_utils.REFL_NAME, 1000] = 8.65
MEAN_RADAR_VALUE_DICT[radar_utils.REFL_NAME, 2000] = 15.5
MEAN_RADAR_VALUE_DICT[radar_utils.REFL_NAME, 3000] = 18.3
MEAN_RADAR_VALUE_DICT[radar_utils.SPECTRUM_WIDTH_NAME, 1000] = 0.169
MEAN_RADAR_VALUE_DICT[radar_utils.SPECTRUM_WIDTH_NAME, 2000] = 0.423
MEAN_RADAR_VALUE_DICT[radar_utils.SPECTRUM_WIDTH_NAME, 3000] = 0.81
MEAN_RADAR_VALUE_DICT[radar_utils.REFL_COLUMN_MAX_NAME, 250] = 20.7

MEAN_SOUNDING_VALUE_DICT = {}
MEAN_SOUNDING_VALUE_DICT[soundings_only.TEMPERATURE_NAME, 500] = 262.
MEAN_SOUNDING_VALUE_DICT[soundings_only.TEMPERATURE_NAME, 700] = 280.
MEAN_SOUNDING_VALUE_DICT[soundings_only.TEMPERATURE_NAME, 850] = 289.
MEAN_SOUNDING_VALUE_DICT[soundings_only.TEMPERATURE_NAME, 1000] = 297.
MEAN_SOUNDING_VALUE_DICT[
    soundings_only.VIRTUAL_POTENTIAL_TEMPERATURE_NAME, 500] = 319.
MEAN_SOUNDING_VALUE_DICT[
    soundings_only.VIRTUAL_POTENTIAL_TEMPERATURE_NAME, 700] = 311.
MEAN_SOUNDING_VALUE_DICT[
    soundings_only.VIRTUAL_POTENTIAL_TEMPERATURE_NAME, 850] = 305.
MEAN_SOUNDING_VALUE_DICT[
    soundings_only.VIRTUAL_POTENTIAL_TEMPERATURE_NAME, 1000] = 299.

SOUNDING_DIMENSIONS = numpy.array([1, 4, 2], dtype=int)
RADAR_DIMENSIONS_4D = numpy.array([1, 16, 32, 4], dtype=int)
RADAR_DIMENSIONS_5D = numpy.array([1, 16, 32, 3, 2], dtype=int)

INIT_SOUNDING_MATRIX = numpy.array([[262, 319],
                                    [280, 311],
                                    [289, 305],
                                    [297, 299]], dtype=float)
INIT_SOUNDING_MATRIX = numpy.expand_dims(INIT_SOUNDING_MATRIX, axis=0)

THIS_MATRIX_FIELD1 = numpy.full(
    (RADAR_DIMENSIONS_4D[1], RADAR_DIMENSIONS_4D[2]), 8.65)
THIS_MATRIX_FIELD2 = numpy.full(
    (RADAR_DIMENSIONS_4D[1], RADAR_DIMENSIONS_4D[2]), 15.5)
THIS_MATRIX_FIELD3 = numpy.full(
    (RADAR_DIMENSIONS_4D[1], RADAR_DIMENSIONS_4D[2]), 18.3)
THIS_MATRIX_FIELD4 = numpy.full(
    (RADAR_DIMENSIONS_4D[1], RADAR_DIMENSIONS_4D[2]), 20.7)
INIT_RADAR_MATRIX_4D = numpy.stack(
    (THIS_MATRIX_FIELD1, THIS_MATRIX_FIELD2, THIS_MATRIX_FIELD3,
     THIS_MATRIX_FIELD4), axis=-1)
INIT_RADAR_MATRIX_4D = numpy.expand_dims(INIT_RADAR_MATRIX_4D, axis=0)

THIS_MATRIX_FIELD1_HEIGHT1 = numpy.full(
    (RADAR_DIMENSIONS_5D[1], RADAR_DIMENSIONS_5D[2]), 8.65)
THIS_MATRIX_FIELD1_HEIGHT2 = numpy.full(
    (RADAR_DIMENSIONS_5D[1], RADAR_DIMENSIONS_5D[2]), 15.5)
THIS_MATRIX_FIELD1_HEIGHT3 = numpy.full(
    (RADAR_DIMENSIONS_5D[1], RADAR_DIMENSIONS_5D[2]), 18.3)
THIS_MATRIX_FIELD1 = numpy.stack(
    (THIS_MATRIX_FIELD1_HEIGHT1, THIS_MATRIX_FIELD1_HEIGHT2,
     THIS_MATRIX_FIELD1_HEIGHT3), axis=-1)

THIS_MATRIX_FIELD2_HEIGHT1 = numpy.full(
    (RADAR_DIMENSIONS_5D[1], RADAR_DIMENSIONS_5D[2]), 0.169)
THIS_MATRIX_FIELD2_HEIGHT2 = numpy.full(
    (RADAR_DIMENSIONS_5D[1], RADAR_DIMENSIONS_5D[2]), 0.423)
THIS_MATRIX_FIELD2_HEIGHT3 = numpy.full(
    (RADAR_DIMENSIONS_5D[1], RADAR_DIMENSIONS_5D[2]), 0.81)
THIS_MATRIX_FIELD2 = numpy.stack(
    (THIS_MATRIX_FIELD2_HEIGHT1, THIS_MATRIX_FIELD2_HEIGHT2,
     THIS_MATRIX_FIELD2_HEIGHT3), axis=-1)

INIT_RADAR_MATRIX_5D = numpy.stack(
    (THIS_MATRIX_FIELD1, THIS_MATRIX_FIELD2), axis=-1)
INIT_RADAR_MATRIX_5D = numpy.expand_dims(INIT_RADAR_MATRIX_5D, axis=0)


class FeatureOptimizationTests(unittest.TestCase):
    """Each method is a unit test for feature_optimization.py."""

    def test_create_gaussian_initializer_3d(self):
        """Ensures correct output from create_gaussian_initializer.

        In this case, the desired matrix is 3-D.
        """

        this_init_function = feature_optimization.create_gaussian_initializer(
            mean=GAUSSIAN_MEAN, standard_deviation=GAUSSIAN_STANDARD_DEVIATION)
        this_matrix = this_init_function(ARRAY_DIMENSIONS_3D)
        self.assertTrue(numpy.allclose(
            this_matrix, GAUSSIAN_MATRIX_3D, atol=TOLERANCE))

    def test_create_gaussian_initializer_4d(self):
        """Ensures correct output from create_gaussian_initializer.

        In this case, the desired matrix is 4-D.
        """

        this_init_function = feature_optimization.create_gaussian_initializer(
            mean=GAUSSIAN_MEAN, standard_deviation=GAUSSIAN_STANDARD_DEVIATION)
        this_matrix = this_init_function(ARRAY_DIMENSIONS_4D)
        self.assertTrue(numpy.allclose(
            this_matrix, GAUSSIAN_MATRIX_4D, atol=TOLERANCE))

    def test_create_gaussian_initializer_5d(self):
        """Ensures correct output from create_gaussian_initializer.

        In this case, the desired matrix is 5-D.
        """

        this_init_function = feature_optimization.create_gaussian_initializer(
            mean=GAUSSIAN_MEAN, standard_deviation=GAUSSIAN_STANDARD_DEVIATION)
        this_matrix = this_init_function(ARRAY_DIMENSIONS_5D)
        self.assertTrue(numpy.allclose(
            this_matrix, GAUSSIAN_MATRIX_5D, atol=TOLERANCE))

    def test_create_uniform_random_initializer_3d(self):
        """Ensures correct output from create_uniform_random_initializer.

        In this case, the desired matrix is 3-D.
        """

        this_init_function = (
            feature_optimization.create_uniform_random_initializer(
                min_value=MIN_UNIFORM_VALUE, max_value=MAX_UNIFORM_VALUE))
        this_matrix = this_init_function(ARRAY_DIMENSIONS_3D)
        self.assertTrue(numpy.allclose(
            this_matrix, UNIFORM_MATRIX_3D, atol=TOLERANCE))

    def test_create_uniform_random_initializer_4d(self):
        """Ensures correct output from create_uniform_random_initializer.

        In this case, the desired matrix is 4-D.
        """

        this_init_function = (
            feature_optimization.create_uniform_random_initializer(
                min_value=MIN_UNIFORM_VALUE, max_value=MAX_UNIFORM_VALUE))
        this_matrix = this_init_function(ARRAY_DIMENSIONS_4D)
        self.assertTrue(numpy.allclose(
            this_matrix, UNIFORM_MATRIX_4D, atol=TOLERANCE))

    def test_create_uniform_random_initializer_5d(self):
        """Ensures correct output from create_uniform_random_initializer.

        In this case, the desired matrix is 5-D.
        """

        this_init_function = (
            feature_optimization.create_uniform_random_initializer(
                min_value=MIN_UNIFORM_VALUE, max_value=MAX_UNIFORM_VALUE))
        this_matrix = this_init_function(ARRAY_DIMENSIONS_5D)
        self.assertTrue(numpy.allclose(
            this_matrix, UNIFORM_MATRIX_5D, atol=TOLERANCE))

    def test_create_constant_initializer_3d(self):
        """Ensures correct output from create_constant_initializer.

        In this case, the desired matrix is 3-D.
        """

        this_init_function = feature_optimization.create_constant_initializer(
            constant_value=CONSTANT_VALUE)
        this_matrix = this_init_function(ARRAY_DIMENSIONS_3D)
        self.assertTrue(numpy.allclose(
            this_matrix, CONSTANT_MATRIX_3D, atol=TOLERANCE))

    def test_create_constant_initializer_4d(self):
        """Ensures correct output from create_constant_initializer.

        In this case, the desired matrix is 4-D.
        """

        this_init_function = feature_optimization.create_constant_initializer(
            constant_value=CONSTANT_VALUE)
        this_matrix = this_init_function(ARRAY_DIMENSIONS_4D)
        self.assertTrue(numpy.allclose(
            this_matrix, CONSTANT_MATRIX_4D, atol=TOLERANCE))

    def test_create_constant_initializer_5d(self):
        """Ensures correct output from create_constant_initializer.

        In this case, the desired matrix is 5-D.
        """

        this_init_function = feature_optimization.create_constant_initializer(
            constant_value=CONSTANT_VALUE)
        this_matrix = this_init_function(ARRAY_DIMENSIONS_5D)
        self.assertTrue(numpy.allclose(
            this_matrix, CONSTANT_MATRIX_5D, atol=TOLERANCE))

    def test_create_climo_initializer_3d(self):
        """Ensures correct output from create_climo_initializer.

        In this case, the desired matrix is 3-D.
        """

        this_init_function = feature_optimization.create_climo_initializer(
            mean_radar_value_dict=MEAN_RADAR_VALUE_DICT,
            mean_sounding_value_dict=MEAN_SOUNDING_VALUE_DICT,
            sounding_field_names=SOUNDING_FIELD_NAMES,
            sounding_pressures_mb=SOUNDING_PRESSURES_MB,
            radar_field_names=RADAR_FIELD_NAMES,
            radar_heights_m_asl=RADAR_HEIGHTS_M_ASL,
            radar_field_name_by_channel=RADAR_FIELD_NAME_BY_CHANNEL,
            radar_height_by_channel_m_asl=RADAR_HEIGHT_BY_CHANNEL_M_ASL)
        this_matrix = this_init_function(SOUNDING_DIMENSIONS)
        self.assertTrue(numpy.allclose(
            this_matrix, INIT_SOUNDING_MATRIX, atol=TOLERANCE))

    def test_create_climo_initializer_4d(self):
        """Ensures correct output from create_climo_initializer.

        In this case, the desired matrix is 4-D.
        """

        this_init_function = feature_optimization.create_climo_initializer(
            mean_radar_value_dict=MEAN_RADAR_VALUE_DICT,
            mean_sounding_value_dict=MEAN_SOUNDING_VALUE_DICT,
            sounding_field_names=SOUNDING_FIELD_NAMES,
            sounding_pressures_mb=SOUNDING_PRESSURES_MB,
            radar_field_names=RADAR_FIELD_NAMES,
            radar_heights_m_asl=RADAR_HEIGHTS_M_ASL,
            radar_field_name_by_channel=RADAR_FIELD_NAME_BY_CHANNEL,
            radar_height_by_channel_m_asl=RADAR_HEIGHT_BY_CHANNEL_M_ASL)
        this_matrix = this_init_function(RADAR_DIMENSIONS_4D)
        self.assertTrue(numpy.allclose(
            this_matrix, INIT_RADAR_MATRIX_4D, atol=TOLERANCE))

    def test_create_climo_initializer_5d(self):
        """Ensures correct output from create_climo_initializer.

        In this case, the desired matrix is 5-D.
        """

        this_init_function = feature_optimization.create_climo_initializer(
            mean_radar_value_dict=MEAN_RADAR_VALUE_DICT,
            mean_sounding_value_dict=MEAN_SOUNDING_VALUE_DICT,
            sounding_field_names=SOUNDING_FIELD_NAMES,
            sounding_pressures_mb=SOUNDING_PRESSURES_MB,
            radar_field_names=RADAR_FIELD_NAMES,
            radar_heights_m_asl=RADAR_HEIGHTS_M_ASL,
            radar_field_name_by_channel=RADAR_FIELD_NAME_BY_CHANNEL,
            radar_height_by_channel_m_asl=RADAR_HEIGHT_BY_CHANNEL_M_ASL)
        this_matrix = this_init_function(RADAR_DIMENSIONS_5D)
        self.assertTrue(numpy.allclose(
            this_matrix, INIT_RADAR_MATRIX_5D, atol=TOLERANCE))

    def test_create_climo_initializer_2d(self):
        """Ensures correct output from create_climo_initializer.

        In this case, the desired matrix is 2-D (invalid).
        """

        this_init_function = feature_optimization.create_climo_initializer(
            mean_radar_value_dict=MEAN_RADAR_VALUE_DICT,
            mean_sounding_value_dict=MEAN_SOUNDING_VALUE_DICT,
            sounding_field_names=SOUNDING_FIELD_NAMES,
            sounding_pressures_mb=SOUNDING_PRESSURES_MB,
            radar_field_names=RADAR_FIELD_NAMES,
            radar_heights_m_asl=RADAR_HEIGHTS_M_ASL,
            radar_field_name_by_channel=RADAR_FIELD_NAME_BY_CHANNEL,
            radar_height_by_channel_m_asl=RADAR_HEIGHT_BY_CHANNEL_M_ASL)
        this_matrix = this_init_function(ARRAY_DIMENSIONS_2D)
        self.assertTrue(this_matrix is None)


if __name__ == '__main__':
    unittest.main()
