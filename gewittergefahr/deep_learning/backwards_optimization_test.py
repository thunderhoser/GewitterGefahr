"""Unit tests for backwards_optimization.py."""

import unittest
import numpy
import pandas
from gewittergefahr.gg_utils import radar_utils
from gewittergefahr.gg_utils import soundings
from gewittergefahr.deep_learning import cnn
from gewittergefahr.deep_learning import backwards_optimization as backwards_opt
from gewittergefahr.deep_learning import deep_learning_utils as dl_utils
from gewittergefahr.deep_learning import training_validation_io as trainval_io

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
SOUNDING_FIELD_NAMES = [
    soundings.TEMPERATURE_NAME, soundings.VIRTUAL_POTENTIAL_TEMPERATURE_NAME
]
SOUNDING_HEIGHTS_M_AGL = numpy.array([0, 500, 1000, 1500], dtype=int)

THESE_FIELD_NAMES = [radar_utils.REFL_NAME, radar_utils.SPECTRUM_WIDTH_NAME]
THESE_HEIGHTS_M_AGL = numpy.array([1000, 2000, 3000], dtype=int)

TRAINING_OPTION_DICT_3D = {
    trainval_io.RADAR_FIELDS_KEY: THESE_FIELD_NAMES,
    trainval_io.RADAR_HEIGHTS_KEY: THESE_HEIGHTS_M_AGL,
    trainval_io.SOUNDING_FIELDS_KEY: SOUNDING_FIELD_NAMES,
    trainval_io.SOUNDING_HEIGHTS_KEY: SOUNDING_HEIGHTS_M_AGL,
    trainval_io.NORMALIZATION_TYPE_KEY:
        dl_utils.MINMAX_NORMALIZATION_TYPE_STRING,
    trainval_io.MIN_NORMALIZED_VALUE_KEY: 0.,
    trainval_io.MAX_NORMALIZED_VALUE_KEY: 1.,
    trainval_io.NORMALIZATION_FILE_KEY: 'foo'
}

THESE_FIELD_NAMES = [
    radar_utils.REFL_NAME, radar_utils.REFL_NAME, radar_utils.REFL_NAME,
    radar_utils.REFL_COLUMN_MAX_NAME
]
THESE_HEIGHTS_M_AGL = numpy.array([1000, 2000, 3000, 250], dtype=int)

TRAINING_OPTION_DICT_2D = {
    trainval_io.RADAR_FIELDS_KEY: THESE_FIELD_NAMES,
    trainval_io.RADAR_HEIGHTS_KEY: THESE_HEIGHTS_M_AGL,
    trainval_io.SOUNDING_FIELDS_KEY: SOUNDING_FIELD_NAMES,
    trainval_io.SOUNDING_HEIGHTS_KEY: SOUNDING_HEIGHTS_M_AGL,
    trainval_io.NORMALIZATION_TYPE_KEY:
        dl_utils.MINMAX_NORMALIZATION_TYPE_STRING,
    trainval_io.MIN_NORMALIZED_VALUE_KEY: 0.,
    trainval_io.MAX_NORMALIZED_VALUE_KEY: 1.,
    trainval_io.NORMALIZATION_FILE_KEY: 'foo'
}

RADAR_NORMALIZATION_DICT = {
    (radar_utils.REFL_NAME, 1000): numpy.array([8.65, 1, 0, 0]),
    (radar_utils.REFL_NAME, 2000): numpy.array([15.5, 1, 0, 0]),
    (radar_utils.REFL_NAME, 3000): numpy.array([18.3, 1, 0, 0]),
    (radar_utils.SPECTRUM_WIDTH_NAME, 1000): numpy.array([0.169, 1, 0, 0]),
    (radar_utils.SPECTRUM_WIDTH_NAME, 2000): numpy.array([0.423, 1, 0, 0]),
    (radar_utils.SPECTRUM_WIDTH_NAME, 3000): numpy.array([0.81, 1, 0, 0]),
    (radar_utils.REFL_COLUMN_MAX_NAME, 250): numpy.array([20.7, 1, 0, 0]),
    radar_utils.REFL_NAME: numpy.array([0., 0, 0, 20]),
    radar_utils.SPECTRUM_WIDTH_NAME: numpy.array([0, 0, 0.1, 1.1]),
    radar_utils.REFL_COLUMN_MAX_NAME: numpy.array([0., 0, 0, 50])
}
RADAR_NORMALIZATION_TABLE = pandas.DataFrame.from_dict(
    RADAR_NORMALIZATION_DICT, orient='index')

COLUMN_DICT_OLD_TO_NEW = {
    0: dl_utils.MEAN_VALUE_COLUMN,
    1: dl_utils.STANDARD_DEVIATION_COLUMN,
    2: dl_utils.MIN_VALUE_COLUMN,
    3: dl_utils.MAX_VALUE_COLUMN
}
RADAR_NORMALIZATION_TABLE.rename(columns=COLUMN_DICT_OLD_TO_NEW, inplace=True)

SOUNDING_NORMALIZATION_DICT = {
    (soundings.TEMPERATURE_NAME, 0): numpy.array([262., 1, 0, 0]),
    (soundings.TEMPERATURE_NAME, 500): numpy.array([280., 1, 0, 0]),
    (soundings.TEMPERATURE_NAME, 1000): numpy.array([289., 1, 0, 0]),
    (soundings.TEMPERATURE_NAME, 1500): numpy.array([297., 1, 0, 0]),
    (soundings.VIRTUAL_POTENTIAL_TEMPERATURE_NAME, 0):
        numpy.array([319., 1, 0, 0]),
    (soundings.VIRTUAL_POTENTIAL_TEMPERATURE_NAME, 500):
        numpy.array([311., 1, 0, 0]),
    (soundings.VIRTUAL_POTENTIAL_TEMPERATURE_NAME, 1000):
        numpy.array([305., 1, 0, 0]),
    (soundings.VIRTUAL_POTENTIAL_TEMPERATURE_NAME, 1500):
        numpy.array([299., 1, 0, 0]),
    soundings.TEMPERATURE_NAME: numpy.array([0., 0, 210, 310]),
    soundings.VIRTUAL_POTENTIAL_TEMPERATURE_NAME: numpy.array([0., 0, 280, 380])
}

SOUNDING_NORMALIZATION_TABLE = pandas.DataFrame.from_dict(
    SOUNDING_NORMALIZATION_DICT, orient='index')
SOUNDING_NORMALIZATION_TABLE.rename(
    columns=COLUMN_DICT_OLD_TO_NEW, inplace=True)

SOUNDING_DIMENSIONS = numpy.array([1, 4, 2], dtype=int)
RADAR_DIMENSIONS_4D = numpy.array([1, 16, 32, 4], dtype=int)
RADAR_DIMENSIONS_5D = numpy.array([1, 16, 32, 3, 2], dtype=int)

INIT_SOUNDING_MATRIX = numpy.array([[262, 319],
                                    [280, 311],
                                    [289, 305],
                                    [297, 299]], dtype=float)
INIT_SOUNDING_MATRIX[:, 0] = (INIT_SOUNDING_MATRIX[:, 0] - 210) / 100
INIT_SOUNDING_MATRIX[:, 1] = (INIT_SOUNDING_MATRIX[:, 1] - 280) / 100
INIT_SOUNDING_MATRIX = numpy.expand_dims(INIT_SOUNDING_MATRIX, axis=0)

THIS_MATRIX_FIELD1 = numpy.full(
    (RADAR_DIMENSIONS_4D[1], RADAR_DIMENSIONS_4D[2]), 8.65 / 20)
THIS_MATRIX_FIELD2 = numpy.full(
    (RADAR_DIMENSIONS_4D[1], RADAR_DIMENSIONS_4D[2]), 15.5 / 20)
THIS_MATRIX_FIELD3 = numpy.full(
    (RADAR_DIMENSIONS_4D[1], RADAR_DIMENSIONS_4D[2]), 18.3 / 20)
THIS_MATRIX_FIELD4 = numpy.full(
    (RADAR_DIMENSIONS_4D[1], RADAR_DIMENSIONS_4D[2]), 20.7 / 50)
INIT_RADAR_MATRIX_4D = numpy.stack(
    (THIS_MATRIX_FIELD1, THIS_MATRIX_FIELD2, THIS_MATRIX_FIELD3,
     THIS_MATRIX_FIELD4), axis=-1)
INIT_RADAR_MATRIX_4D = numpy.expand_dims(INIT_RADAR_MATRIX_4D, axis=0)

THIS_MATRIX_FIELD1_HEIGHT1 = numpy.full(
    (RADAR_DIMENSIONS_5D[1], RADAR_DIMENSIONS_5D[2]), 8.65 / 20)
THIS_MATRIX_FIELD1_HEIGHT2 = numpy.full(
    (RADAR_DIMENSIONS_5D[1], RADAR_DIMENSIONS_5D[2]), 15.5 / 20)
THIS_MATRIX_FIELD1_HEIGHT3 = numpy.full(
    (RADAR_DIMENSIONS_5D[1], RADAR_DIMENSIONS_5D[2]), 18.3 / 20)
THIS_MATRIX_FIELD1 = numpy.stack(
    (THIS_MATRIX_FIELD1_HEIGHT1, THIS_MATRIX_FIELD1_HEIGHT2,
     THIS_MATRIX_FIELD1_HEIGHT3), axis=-1)

THIS_MATRIX_FIELD2_HEIGHT1 = numpy.full(
    (RADAR_DIMENSIONS_5D[1], RADAR_DIMENSIONS_5D[2]), 0.169 - 0.1)
THIS_MATRIX_FIELD2_HEIGHT2 = numpy.full(
    (RADAR_DIMENSIONS_5D[1], RADAR_DIMENSIONS_5D[2]), 0.423 - 0.1)
THIS_MATRIX_FIELD2_HEIGHT3 = numpy.full(
    (RADAR_DIMENSIONS_5D[1], RADAR_DIMENSIONS_5D[2]), 0.81 - 0.1)
THIS_MATRIX_FIELD2 = numpy.stack(
    (THIS_MATRIX_FIELD2_HEIGHT1, THIS_MATRIX_FIELD2_HEIGHT2,
     THIS_MATRIX_FIELD2_HEIGHT3), axis=-1)

INIT_RADAR_MATRIX_5D = numpy.stack(
    (THIS_MATRIX_FIELD1, THIS_MATRIX_FIELD2), axis=-1)
INIT_RADAR_MATRIX_5D = numpy.expand_dims(INIT_RADAR_MATRIX_5D, axis=0)


class FeatureOptimizationTests(unittest.TestCase):
    """Each method is a unit test for backwards_optimization.py."""

    def test_create_gaussian_initializer_3d(self):
        """Ensures correct output from create_gaussian_initializer.

        In this case, the desired matrix is 3-D.
        """

        this_init_function = backwards_opt.create_gaussian_initializer(
            mean=GAUSSIAN_MEAN, standard_deviation=GAUSSIAN_STANDARD_DEVIATION)

        this_matrix = this_init_function(ARRAY_DIMENSIONS_3D)
        self.assertTrue(numpy.allclose(
            this_matrix, GAUSSIAN_MATRIX_3D, atol=TOLERANCE))

    def test_create_gaussian_initializer_4d(self):
        """Ensures correct output from create_gaussian_initializer.

        In this case, the desired matrix is 4-D.
        """

        this_init_function = backwards_opt.create_gaussian_initializer(
            mean=GAUSSIAN_MEAN, standard_deviation=GAUSSIAN_STANDARD_DEVIATION)

        this_matrix = this_init_function(ARRAY_DIMENSIONS_4D)
        self.assertTrue(numpy.allclose(
            this_matrix, GAUSSIAN_MATRIX_4D, atol=TOLERANCE))

    def test_create_gaussian_initializer_5d(self):
        """Ensures correct output from create_gaussian_initializer.

        In this case, the desired matrix is 5-D.
        """

        this_init_function = backwards_opt.create_gaussian_initializer(
            mean=GAUSSIAN_MEAN, standard_deviation=GAUSSIAN_STANDARD_DEVIATION)

        this_matrix = this_init_function(ARRAY_DIMENSIONS_5D)
        self.assertTrue(numpy.allclose(
            this_matrix, GAUSSIAN_MATRIX_5D, atol=TOLERANCE))

    def test_create_uniform_random_initializer_3d(self):
        """Ensures correct output from create_uniform_random_initializer.

        In this case, the desired matrix is 3-D.
        """

        this_init_function = backwards_opt.create_uniform_random_initializer(
            min_value=MIN_UNIFORM_VALUE, max_value=MAX_UNIFORM_VALUE)

        this_matrix = this_init_function(ARRAY_DIMENSIONS_3D)
        self.assertTrue(numpy.allclose(
            this_matrix, UNIFORM_MATRIX_3D, atol=TOLERANCE))

    def test_create_uniform_random_initializer_4d(self):
        """Ensures correct output from create_uniform_random_initializer.

        In this case, the desired matrix is 4-D.
        """

        this_init_function = backwards_opt.create_uniform_random_initializer(
            min_value=MIN_UNIFORM_VALUE, max_value=MAX_UNIFORM_VALUE)

        this_matrix = this_init_function(ARRAY_DIMENSIONS_4D)
        self.assertTrue(numpy.allclose(
            this_matrix, UNIFORM_MATRIX_4D, atol=TOLERANCE))

    def test_create_uniform_random_initializer_5d(self):
        """Ensures correct output from create_uniform_random_initializer.

        In this case, the desired matrix is 5-D.
        """

        this_init_function = backwards_opt.create_uniform_random_initializer(
            min_value=MIN_UNIFORM_VALUE, max_value=MAX_UNIFORM_VALUE)

        this_matrix = this_init_function(ARRAY_DIMENSIONS_5D)
        self.assertTrue(numpy.allclose(
            this_matrix, UNIFORM_MATRIX_5D, atol=TOLERANCE))

    def test_create_constant_initializer_3d(self):
        """Ensures correct output from create_constant_initializer.

        In this case, the desired matrix is 3-D.
        """

        this_init_function = backwards_opt.create_constant_initializer(
            constant_value=CONSTANT_VALUE)

        this_matrix = this_init_function(ARRAY_DIMENSIONS_3D)
        self.assertTrue(numpy.allclose(
            this_matrix, CONSTANT_MATRIX_3D, atol=TOLERANCE))

    def test_create_constant_initializer_4d(self):
        """Ensures correct output from create_constant_initializer.

        In this case, the desired matrix is 4-D.
        """

        this_init_function = backwards_opt.create_constant_initializer(
            constant_value=CONSTANT_VALUE)

        this_matrix = this_init_function(ARRAY_DIMENSIONS_4D)
        self.assertTrue(numpy.allclose(
            this_matrix, CONSTANT_MATRIX_4D, atol=TOLERANCE))

    def test_create_constant_initializer_5d(self):
        """Ensures correct output from create_constant_initializer.

        In this case, the desired matrix is 5-D.
        """

        this_init_function = backwards_opt.create_constant_initializer(
            constant_value=CONSTANT_VALUE)

        this_matrix = this_init_function(ARRAY_DIMENSIONS_5D)
        self.assertTrue(numpy.allclose(
            this_matrix, CONSTANT_MATRIX_5D, atol=TOLERANCE))

    def test_create_climo_initializer_3d(self):
        """Ensures correct output from create_climo_initializer.

        In this case, the desired matrix is 3-D.
        """

        this_metadata_dict = {
            cnn.TRAINING_OPTION_DICT_KEY: TRAINING_OPTION_DICT_3D,
            cnn.USE_2D3D_CONVOLUTION_KEY: False
        }

        this_init_function = backwards_opt.create_climo_initializer(
            model_metadata_dict=this_metadata_dict, test_mode=True,
            radar_normalization_table=RADAR_NORMALIZATION_TABLE,
            sounding_normalization_table=SOUNDING_NORMALIZATION_TABLE)

        this_matrix = this_init_function(SOUNDING_DIMENSIONS)
        self.assertTrue(numpy.allclose(
            this_matrix, INIT_SOUNDING_MATRIX, atol=TOLERANCE))

    def test_create_climo_initializer_4d(self):
        """Ensures correct output from create_climo_initializer.

        In this case, the desired matrix is 4-D.
        """

        this_metadata_dict = {
            cnn.TRAINING_OPTION_DICT_KEY: TRAINING_OPTION_DICT_2D,
            cnn.USE_2D3D_CONVOLUTION_KEY: False,
            cnn.LAYER_OPERATIONS_KEY: None
        }

        this_init_function = backwards_opt.create_climo_initializer(
            model_metadata_dict=this_metadata_dict, test_mode=True,
            radar_normalization_table=RADAR_NORMALIZATION_TABLE,
            sounding_normalization_table=SOUNDING_NORMALIZATION_TABLE)

        this_matrix = this_init_function(RADAR_DIMENSIONS_4D)
        self.assertTrue(numpy.allclose(
            this_matrix, INIT_RADAR_MATRIX_4D, atol=TOLERANCE))

    def test_create_climo_initializer_5d(self):
        """Ensures correct output from create_climo_initializer.

        In this case, the desired matrix is 5-D.
        """

        this_metadata_dict = {
            cnn.TRAINING_OPTION_DICT_KEY: TRAINING_OPTION_DICT_3D,
            cnn.USE_2D3D_CONVOLUTION_KEY: False
        }

        this_init_function = backwards_opt.create_climo_initializer(
            model_metadata_dict=this_metadata_dict, test_mode=True,
            radar_normalization_table=RADAR_NORMALIZATION_TABLE,
            sounding_normalization_table=SOUNDING_NORMALIZATION_TABLE)

        this_matrix = this_init_function(RADAR_DIMENSIONS_5D)
        self.assertTrue(numpy.allclose(
            this_matrix, INIT_RADAR_MATRIX_5D, atol=TOLERANCE))

    def test_create_climo_initializer_2d(self):
        """Ensures correct output from create_climo_initializer.

        In this case, the desired matrix is 2-D (invalid).
        """

        this_metadata_dict = {
            cnn.TRAINING_OPTION_DICT_KEY: TRAINING_OPTION_DICT_3D,
            cnn.USE_2D3D_CONVOLUTION_KEY: False
        }

        this_init_function = backwards_opt.create_climo_initializer(
            model_metadata_dict=this_metadata_dict, test_mode=True,
            radar_normalization_table=RADAR_NORMALIZATION_TABLE,
            sounding_normalization_table=SOUNDING_NORMALIZATION_TABLE)

        this_matrix = this_init_function(ARRAY_DIMENSIONS_2D)
        self.assertTrue(this_matrix is None)


if __name__ == '__main__':
    unittest.main()
