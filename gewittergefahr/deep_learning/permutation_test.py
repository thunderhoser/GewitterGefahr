"""Unit tests for permutation.py."""

import unittest
import numpy
from gewittergefahr.gg_utils import radar_utils
from gewittergefahr.gg_utils import soundings
from gewittergefahr.deep_learning import cnn
from gewittergefahr.deep_learning import permutation
from gewittergefahr.deep_learning import input_examples
from gewittergefahr.deep_learning import training_validation_io as trainval_io

# The following constants are used to test different methods.
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

LOW_LEVEL_SHEAR_MATRIX_S01 = 0.002 * REFLECTIVITY_MATRIX_DBZ[..., 0]
MID_LEVEL_SHEAR_MATRIX_S01 = 0.002 * REFLECTIVITY_MATRIX_DBZ[..., 1]
VORTICITY_MATRIX_S01 = 0.001 * REFLECTIVITY_MATRIX_DBZ

AZIMUTHAL_SHEAR_MATRIX_S01 = numpy.stack(
    (LOW_LEVEL_SHEAR_MATRIX_S01, MID_LEVEL_SHEAR_MATRIX_S01), axis=-1
)
RADAR_MATRIX_3D = numpy.stack(
    (REFLECTIVITY_MATRIX_DBZ, VORTICITY_MATRIX_S01), axis=-1
)
RADAR_MATRIX_3D_FLATTENED = numpy.concatenate(
    (REFLECTIVITY_MATRIX_DBZ, VORTICITY_MATRIX_S01), axis=-1
)

AZ_SHEAR_FIELD_NAMES = [
    radar_utils.LOW_LEVEL_SHEAR_NAME, radar_utils.MID_LEVEL_SHEAR_NAME
]
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
    trainval_io.SOUNDING_FIELDS_KEY: SOUNDING_FIELD_NAMES,
    trainval_io.UPSAMPLE_REFLECTIVITY_KEY: False
}

FIRST_METADATA_DICT = {
    cnn.CONV_2D3D_KEY: True,
    cnn.LAYER_OPERATIONS_KEY: None,
    cnn.TRAINING_OPTION_DICT_KEY: THIS_TRAINING_OPTION_DICT
}
FIRST_PREDICTOR_MATRICES = [
    REFLECTIVITY_MATRIX_DBZ, AZIMUTHAL_SHEAR_MATRIX_S01, SOUNDING_MATRIX
]
FIRST_SEPARATE_HEIGHTS_FLAG = False
FIRST_NICE_PREDICTOR_NAMES = [
    'Reflectivity', 'Low-level shear', 'Mid-level shear', 'Temperature',
    'Relative humidity'
]

SECOND_METADATA_DICT = FIRST_METADATA_DICT
SECOND_PREDICTOR_MATRICES = FIRST_PREDICTOR_MATRICES
SECOND_SEPARATE_HEIGHTS_FLAG = True
SECOND_NICE_PREDICTOR_NAMES = [
    'Reflectivity at 2000 m AGL', 'Reflectivity at 6000 m AGL',
    'Reflectivity at 10000 m AGL', 'Low-level shear', 'Mid-level shear',
    'Temperature', 'Relative humidity'
]

THIS_TRAINING_OPTION_DICT = {
    trainval_io.RADAR_FIELDS_KEY:
        [radar_utils.LOW_LEVEL_SHEAR_NAME, radar_utils.MID_LEVEL_SHEAR_NAME],
    trainval_io.RADAR_HEIGHTS_KEY: RADAR_HEIGHTS_M_AGL,
    trainval_io.SOUNDING_FIELDS_KEY: None,
    trainval_io.UPSAMPLE_REFLECTIVITY_KEY: False
}

THIRD_METADATA_DICT = {
    cnn.CONV_2D3D_KEY: True,
    cnn.LAYER_OPERATIONS_KEY: None,
    cnn.TRAINING_OPTION_DICT_KEY: THIS_TRAINING_OPTION_DICT
}
THIRD_PREDICTOR_MATRICES = [
    REFLECTIVITY_MATRIX_DBZ, AZIMUTHAL_SHEAR_MATRIX_S01
]
THIRD_SEPARATE_HEIGHTS_FLAG = False
THIRD_NICE_PREDICTOR_NAMES = [
    'Reflectivity', 'Low-level shear', 'Mid-level shear'
]

FOURTH_METADATA_DICT = THIRD_METADATA_DICT
FOURTH_PREDICTOR_MATRICES = THIRD_PREDICTOR_MATRICES
FOURTH_SEPARATE_HEIGHTS_FLAG = True
FOURTH_NICE_PREDICTOR_NAMES = [
    'Reflectivity at 2000 m AGL', 'Reflectivity at 6000 m AGL',
    'Reflectivity at 10000 m AGL', 'Low-level shear', 'Mid-level shear'
]

THIS_TRAINING_OPTION_DICT = {
    trainval_io.RADAR_FIELDS_KEY:
        [radar_utils.LOW_LEVEL_SHEAR_NAME, radar_utils.MID_LEVEL_SHEAR_NAME],
    trainval_io.RADAR_HEIGHTS_KEY: RADAR_HEIGHTS_M_AGL,
    trainval_io.SOUNDING_FIELDS_KEY: None,
    trainval_io.UPSAMPLE_REFLECTIVITY_KEY: True
}

FIFTH_METADATA_DICT = {
    cnn.CONV_2D3D_KEY: True,
    cnn.LAYER_OPERATIONS_KEY: None,
    cnn.TRAINING_OPTION_DICT_KEY: THIS_TRAINING_OPTION_DICT
}
FIFTH_PREDICTOR_MATRICES = [
    REFLECTIVITY_MATRIX_DBZ, AZIMUTHAL_SHEAR_MATRIX_S01
]
FIFTH_SEPARATE_HEIGHTS_FLAG = False
FIFTH_NICE_PREDICTOR_NAMES = [
    'Reflectivity at 2000 m AGL', 'Reflectivity at 6000 m AGL',
    'Reflectivity at 10000 m AGL', 'Low-level shear', 'Mid-level shear'
]

SIXTH_METADATA_DICT = FIFTH_METADATA_DICT
SIXTH_PREDICTOR_MATRICES = FIFTH_PREDICTOR_MATRICES
SIXTH_SEPARATE_HEIGHTS_FLAG = True
SIXTH_NICE_PREDICTOR_NAMES = FIFTH_NICE_PREDICTOR_NAMES

THIS_TRAINING_OPTION_DICT = {
    trainval_io.RADAR_FIELDS_KEY: RADAR_FIELD_NAMES_3D,
    trainval_io.RADAR_HEIGHTS_KEY: RADAR_HEIGHTS_M_AGL,
    trainval_io.SOUNDING_FIELDS_KEY: SOUNDING_FIELD_NAMES
}

SEVENTH_METADATA_DICT = {
    cnn.CONV_2D3D_KEY: False,
    cnn.LAYER_OPERATIONS_KEY: None,
    cnn.TRAINING_OPTION_DICT_KEY: THIS_TRAINING_OPTION_DICT
}
SEVENTH_PREDICTOR_MATRICES = [RADAR_MATRIX_3D, SOUNDING_MATRIX]
SEVENTH_SEPARATE_HEIGHTS_FLAG = False
SEVENTH_NICE_PREDICTOR_NAMES = [
    'Reflectivity', 'Vorticity', 'Temperature', 'Relative humidity'
]

EIGHTH_METADATA_DICT = SEVENTH_METADATA_DICT
EIGHTH_PREDICTOR_MATRICES = SEVENTH_PREDICTOR_MATRICES
EIGHTH_SEPARATE_HEIGHTS_FLAG = True
EIGHTH_NICE_PREDICTOR_NAMES = [
    'Reflectivity at 2000 m AGL', 'Reflectivity at 6000 m AGL',
    'Reflectivity at 10000 m AGL', 'Vorticity at 2000 m AGL',
    'Vorticity at 6000 m AGL', 'Vorticity at 10000 m AGL',
    'Temperature', 'Relative humidity'
]

THIS_MIN_VORTICITY_MATRIX_S01 = numpy.max(
    VORTICITY_MATRIX_S01[..., 1:], axis=-1
)
THIS_MAX_REFL_MATRIX_DBZ = numpy.max(
    REFLECTIVITY_MATRIX_DBZ[..., :2, 0], axis=-1
)

RADAR_MATRIX_2D = numpy.stack(
    (THIS_MIN_VORTICITY_MATRIX_S01, THIS_MAX_REFL_MATRIX_DBZ), axis=-1
)

THIS_FIRST_DICT = {
    input_examples.RADAR_FIELD_KEY: radar_utils.VORTICITY_NAME,
    input_examples.OPERATION_NAME_KEY: input_examples.MIN_OPERATION_NAME,
    input_examples.MIN_HEIGHT_KEY: 6000,
    input_examples.MAX_HEIGHT_KEY: 10000
}

THIS_SECOND_DICT = {
    input_examples.RADAR_FIELD_KEY: radar_utils.REFL_NAME,
    input_examples.OPERATION_NAME_KEY: input_examples.MAX_OPERATION_NAME,
    input_examples.MIN_HEIGHT_KEY: 2000,
    input_examples.MAX_HEIGHT_KEY: 6000
}

LAYER_OPERATION_DICTS = [THIS_FIRST_DICT, THIS_SECOND_DICT]

THIS_TRAINING_OPTION_DICT = {
    trainval_io.RADAR_FIELDS_KEY: RADAR_FIELD_NAMES_3D,
    trainval_io.RADAR_HEIGHTS_KEY: RADAR_HEIGHTS_M_AGL,
    trainval_io.SOUNDING_FIELDS_KEY: SOUNDING_FIELD_NAMES
}

NINTH_METADATA_DICT = {
    cnn.CONV_2D3D_KEY: False,
    cnn.LAYER_OPERATIONS_KEY: LAYER_OPERATION_DICTS,
    cnn.TRAINING_OPTION_DICT_KEY: THIS_TRAINING_OPTION_DICT
}
NINTH_PREDICTOR_MATRICES = [RADAR_MATRIX_2D, SOUNDING_MATRIX]
NINTH_NICE_PREDICTOR_NAMES = [
    'Vorticity; MIN from 6000-10000 m AGL',
    'Reflectivity; MAX from 2000-6000 m AGL',
    'Temperature', 'Relative humidity'
]


class PermutationTests(unittest.TestCase):
    """Each method is a unit test for permutation.py."""

    def test_create_nice_predictor_names_first(self):
        """Ensures correct output from create_nice_predictor_names.

        In this case, using first set of inputs.
        """

        these_names_by_matrix = permutation.create_nice_predictor_names(
            predictor_matrices=FIRST_PREDICTOR_MATRICES,
            cnn_metadata_dict=FIRST_METADATA_DICT,
            separate_radar_heights=FIRST_SEPARATE_HEIGHTS_FLAG)

        these_predictor_names = sum(these_names_by_matrix, [])
        self.assertTrue(these_predictor_names == FIRST_NICE_PREDICTOR_NAMES)

    def test_create_nice_predictor_names_second(self):
        """Ensures correct output from create_nice_predictor_names.

        In this case, using second set of inputs.
        """

        these_names_by_matrix = permutation.create_nice_predictor_names(
            predictor_matrices=SECOND_PREDICTOR_MATRICES,
            cnn_metadata_dict=SECOND_METADATA_DICT,
            separate_radar_heights=SECOND_SEPARATE_HEIGHTS_FLAG)

        these_predictor_names = sum(these_names_by_matrix, [])
        self.assertTrue(these_predictor_names == SECOND_NICE_PREDICTOR_NAMES)

    def test_create_nice_predictor_names_third(self):
        """Ensures correct output from create_nice_predictor_names.

        In this case, using third set of inputs.
        """

        these_names_by_matrix = permutation.create_nice_predictor_names(
            predictor_matrices=THIRD_PREDICTOR_MATRICES,
            cnn_metadata_dict=THIRD_METADATA_DICT,
            separate_radar_heights=THIRD_SEPARATE_HEIGHTS_FLAG)

        these_predictor_names = sum(these_names_by_matrix, [])
        self.assertTrue(these_predictor_names == THIRD_NICE_PREDICTOR_NAMES)

    def test_create_nice_predictor_names_fourth(self):
        """Ensures correct output from create_nice_predictor_names.

        In this case, using fourth set of inputs.
        """

        these_names_by_matrix = permutation.create_nice_predictor_names(
            predictor_matrices=FOURTH_PREDICTOR_MATRICES,
            cnn_metadata_dict=FOURTH_METADATA_DICT,
            separate_radar_heights=FOURTH_SEPARATE_HEIGHTS_FLAG)

        these_predictor_names = sum(these_names_by_matrix, [])
        self.assertTrue(these_predictor_names == FOURTH_NICE_PREDICTOR_NAMES)

    def test_create_nice_predictor_names_fifth(self):
        """Ensures correct output from create_nice_predictor_names.

        In this case, using fifth set of inputs.
        """

        these_names_by_matrix = permutation.create_nice_predictor_names(
            predictor_matrices=FIFTH_PREDICTOR_MATRICES,
            cnn_metadata_dict=FIFTH_METADATA_DICT,
            separate_radar_heights=FIFTH_SEPARATE_HEIGHTS_FLAG)

        these_predictor_names = sum(these_names_by_matrix, [])
        self.assertTrue(these_predictor_names == FIFTH_NICE_PREDICTOR_NAMES)

    def test_create_nice_predictor_names_sixth(self):
        """Ensures correct output from create_nice_predictor_names.

        In this case, using sixth set of inputs.
        """

        these_names_by_matrix = permutation.create_nice_predictor_names(
            predictor_matrices=SIXTH_PREDICTOR_MATRICES,
            cnn_metadata_dict=SIXTH_METADATA_DICT,
            separate_radar_heights=SIXTH_SEPARATE_HEIGHTS_FLAG)

        these_predictor_names = sum(these_names_by_matrix, [])
        self.assertTrue(these_predictor_names == SIXTH_NICE_PREDICTOR_NAMES)

    def test_create_nice_predictor_names_seventh(self):
        """Ensures correct output from create_nice_predictor_names.

        In this case, using seventh set of inputs.
        """

        these_names_by_matrix = permutation.create_nice_predictor_names(
            predictor_matrices=SEVENTH_PREDICTOR_MATRICES,
            cnn_metadata_dict=SEVENTH_METADATA_DICT,
            separate_radar_heights=SEVENTH_SEPARATE_HEIGHTS_FLAG)

        these_predictor_names = sum(these_names_by_matrix, [])
        self.assertTrue(these_predictor_names == SEVENTH_NICE_PREDICTOR_NAMES)

    def test_create_nice_predictor_names_eighth(self):
        """Ensures correct output from create_nice_predictor_names.

        In this case, using eighth set of inputs.
        """

        these_names_by_matrix = permutation.create_nice_predictor_names(
            predictor_matrices=EIGHTH_PREDICTOR_MATRICES,
            cnn_metadata_dict=EIGHTH_METADATA_DICT,
            separate_radar_heights=EIGHTH_SEPARATE_HEIGHTS_FLAG)

        these_predictor_names = sum(these_names_by_matrix, [])
        self.assertTrue(these_predictor_names == EIGHTH_NICE_PREDICTOR_NAMES)

    def test_create_nice_predictor_names_ninth(self):
        """Ensures correct output from create_nice_predictor_names.

        In this case, using ninth set of inputs.
        """

        these_names_by_matrix = permutation.create_nice_predictor_names(
            predictor_matrices=NINTH_PREDICTOR_MATRICES,
            cnn_metadata_dict=NINTH_METADATA_DICT)

        these_predictor_names = sum(these_names_by_matrix, [])
        self.assertTrue(these_predictor_names == NINTH_NICE_PREDICTOR_NAMES)


if __name__ == '__main__':
    unittest.main()
