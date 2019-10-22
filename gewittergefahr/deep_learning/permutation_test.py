"""Unit tests for permutation.py."""

import copy
import unittest
import numpy
from sklearn.metrics import log_loss as sklearn_cross_entropy
from gewittergefahr.gg_utils import radar_utils
from gewittergefahr.gg_utils import soundings
from gewittergefahr.deep_learning import cnn
from gewittergefahr.deep_learning import permutation
from gewittergefahr.deep_learning import input_examples
from gewittergefahr.deep_learning import training_validation_io as trainval_io

TOLERANCE = 1e-6
XENTROPY_TOLERANCE = 1e-4

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

# The following constants are used to test _permute_one_predictor.
PREDICTOR_MATRICES_FOR_PERMUTN = copy.deepcopy([
    RADAR_MATRIX_3D, SOUNDING_MATRIX
])

FIRST_SEPARATE_FLAG_FOR_PERMUTN = True
FIRST_MATRIX_INDEX_FOR_PERMUTN = 0
FIRST_PRED_INDEX_FOR_PERMUTN = 1

SECOND_SEPARATE_FLAG_FOR_PERMUTN = True
SECOND_MATRIX_INDEX_FOR_PERMUTN = 1
SECOND_PRED_INDEX_FOR_PERMUTN = 0

THIRD_SEPARATE_FLAG_FOR_PERMUTN = False
THIRD_MATRIX_INDEX_FOR_PERMUTN = 0
THIRD_PRED_INDEX_FOR_PERMUTN = 1

# The following constants are used to test cross_entropy_function,
# negative_auc_function, and _bootstrap_cost.
BINARY_TARGET_VALUES = numpy.array([0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0], dtype=int)
BINARY_PROBABILITY_MATRIX = numpy.array([
    [0, 1],
    [1, 0],
    [0, 1],
    [1, 0],
    [0.5, 0.5],
    [0.5, 0.5],
    [0, 1],
    [1, 0],
    [0, 1],
    [1, 0],
    [0.5, 0.5]
])

NEGATIVE_AUC = -0.75
BINARY_CROSS_ENTROPY = sklearn_cross_entropy(
    BINARY_TARGET_VALUES, BINARY_PROBABILITY_MATRIX[:, 1]
)

TERNARY_TARGET_VALUES = numpy.array(
    [2, 0, 1, 1, 2, 1, 2, 0, 2, 0, 0], dtype=int
)

TERNARY_PROBABILITY_MATRIX = numpy.array([
    [0, 0, 1],
    [1, 0, 0],
    [0, 1, 0],
    [1. / 3, 1. / 3, 1. / 3],
    [0.5, 0.25, 0.25],
    [0.25, 0.5, 0.25],
    [1. / 3, 0, 2. / 3],
    [1, 0, 0],
    [0, 0.5, 0.5],
    [0.75, 0, 0.25],
    [0.6, 0.2, 0.2]
])

TERNARY_CROSS_ENTROPY = sklearn_cross_entropy(
    TERNARY_TARGET_VALUES, TERNARY_PROBABILITY_MATRIX)


def _compare_before_and_after_permutn(
        orig_predictor_matrices, new_predictor_matrices, separate_radar_heights,
        matrix_index, predictor_index):
    """Compares predictor matrices before and after permutation.

    T = number of predictor matrices

    :param orig_predictor_matrices: length-T list of numpy arrays.
    :param new_predictor_matrices: length-T list of numpy arrays with matching
        dimensions.
    :param separate_radar_heights: See doc for
        `permutation._permute_one_predictor`.
    :param matrix_index: Same.
    :param predictor_index: Same.
    :return: all_good: Boolean flag, indicating whether or not results are as
        expected.
    """

    num_matrices = len(orig_predictor_matrices)
    all_good = True

    for i in range(num_matrices):
        if i == 0 and separate_radar_heights:
            this_orig_matrix = permutation.flatten_last_two_dim(
                orig_predictor_matrices[i]
            )[0]
            this_new_matrix = permutation.flatten_last_two_dim(
                new_predictor_matrices[i]
            )[0]
        else:
            this_orig_matrix = orig_predictor_matrices[i]
            this_new_matrix = new_predictor_matrices[i]

        this_num_predictors = this_orig_matrix.shape[-1]

        for j in range(this_num_predictors):
            if i == matrix_index and j == predictor_index:
                all_good = all_good and not numpy.allclose(
                    this_orig_matrix[..., j], this_new_matrix[..., j],
                    atol=TOLERANCE
                )
            else:
                all_good = all_good and numpy.allclose(
                    this_orig_matrix[..., j], this_new_matrix[..., j],
                    atol=TOLERANCE
                )

            if not all_good:
                return False

    return all_good


class PermutationTests(unittest.TestCase):
    """Each method is a unit test for permutation.py."""

    def test_bootstrap_cost_1replicate(self):
        """Ensures correct output from _bootstrap_cost.

        In this case there is only one bootstrap replicate.
        """

        this_cost_array = permutation._bootstrap_cost(
            target_values=BINARY_TARGET_VALUES,
            class_probability_matrix=BINARY_PROBABILITY_MATRIX,
            cost_function=permutation.negative_auc_function, num_replicates=1)

        self.assertTrue(numpy.allclose(
            this_cost_array, numpy.array([NEGATIVE_AUC]), atol=TOLERANCE
        ))

    def test_bootstrap_cost_1000replicates(self):
        """Ensures correct output from _bootstrap_cost.

        In this case there are 1000 replicates.
        """

        this_cost_array = permutation._bootstrap_cost(
            target_values=BINARY_TARGET_VALUES,
            class_probability_matrix=BINARY_PROBABILITY_MATRIX,
            cost_function=permutation.cross_entropy_function,
            num_replicates=1000)

        self.assertTrue(len(this_cost_array) == 1000)

    def test_permute_one_predictor_first(self):
        """Ensures correct output from _permute_one_predictor.

        In this case, using first set of inputs.
        """

        these_new_matrices, these_permuted_values = (
            permutation._permute_one_predictor(
                predictor_matrices=copy.deepcopy(
                    PREDICTOR_MATRICES_FOR_PERMUTN),
                separate_radar_heights=FIRST_SEPARATE_FLAG_FOR_PERMUTN,
                matrix_index=FIRST_MATRIX_INDEX_FOR_PERMUTN,
                predictor_index=FIRST_PRED_INDEX_FOR_PERMUTN,
                permuted_values=None)
        )

        self.assertTrue(_compare_before_and_after_permutn(
            orig_predictor_matrices=PREDICTOR_MATRICES_FOR_PERMUTN,
            new_predictor_matrices=these_new_matrices,
            separate_radar_heights=FIRST_SEPARATE_FLAG_FOR_PERMUTN,
            matrix_index=FIRST_MATRIX_INDEX_FOR_PERMUTN,
            predictor_index=FIRST_PRED_INDEX_FOR_PERMUTN
        ))

        these_newnew_matrices = permutation._permute_one_predictor(
            predictor_matrices=copy.deepcopy(these_new_matrices),
            separate_radar_heights=FIRST_SEPARATE_FLAG_FOR_PERMUTN,
            matrix_index=FIRST_MATRIX_INDEX_FOR_PERMUTN,
            predictor_index=FIRST_PRED_INDEX_FOR_PERMUTN,
            permuted_values=these_permuted_values
        )[0]

        for i in range(len(these_new_matrices)):
            self.assertTrue(numpy.allclose(
                these_newnew_matrices[i], these_new_matrices[i],
                atol=TOLERANCE
            ))

    def test_unpermute_one_predictor_first(self):
        """Ensures correct output from _unpermute_one_predictor.

        In this case, using first set of inputs.
        """

        these_new_matrices = permutation._permute_one_predictor(
            predictor_matrices=copy.deepcopy(PREDICTOR_MATRICES_FOR_PERMUTN),
            separate_radar_heights=FIRST_SEPARATE_FLAG_FOR_PERMUTN,
            matrix_index=FIRST_MATRIX_INDEX_FOR_PERMUTN,
            predictor_index=FIRST_PRED_INDEX_FOR_PERMUTN,
            permuted_values=None
        )[0]

        these_orig_matrices = permutation._unpermute_one_predictor(
            predictor_matrices=copy.deepcopy(these_new_matrices),
            clean_predictor_matrices=PREDICTOR_MATRICES_FOR_PERMUTN,
            separate_radar_heights=FIRST_SEPARATE_FLAG_FOR_PERMUTN,
            matrix_index=FIRST_MATRIX_INDEX_FOR_PERMUTN,
            predictor_index=FIRST_PRED_INDEX_FOR_PERMUTN)

        for i in range(len(these_new_matrices)):
            self.assertTrue(numpy.allclose(
                these_orig_matrices[i], PREDICTOR_MATRICES_FOR_PERMUTN[i],
                atol=TOLERANCE
            ))

    def test_permute_one_predictor_second(self):
        """Ensures correct output from _permute_one_predictor.

        In this case, using second set of inputs.
        """

        these_new_matrices, these_permuted_values = (
            permutation._permute_one_predictor(
                predictor_matrices=copy.deepcopy(
                    PREDICTOR_MATRICES_FOR_PERMUTN),
                separate_radar_heights=SECOND_SEPARATE_FLAG_FOR_PERMUTN,
                matrix_index=SECOND_MATRIX_INDEX_FOR_PERMUTN,
                predictor_index=SECOND_PRED_INDEX_FOR_PERMUTN,
                permuted_values=None)
        )

        self.assertTrue(_compare_before_and_after_permutn(
            orig_predictor_matrices=PREDICTOR_MATRICES_FOR_PERMUTN,
            new_predictor_matrices=these_new_matrices,
            separate_radar_heights=SECOND_SEPARATE_FLAG_FOR_PERMUTN,
            matrix_index=SECOND_MATRIX_INDEX_FOR_PERMUTN,
            predictor_index=SECOND_PRED_INDEX_FOR_PERMUTN
        ))

        these_newnew_matrices = permutation._permute_one_predictor(
            predictor_matrices=copy.deepcopy(these_new_matrices),
            separate_radar_heights=SECOND_SEPARATE_FLAG_FOR_PERMUTN,
            matrix_index=SECOND_MATRIX_INDEX_FOR_PERMUTN,
            predictor_index=SECOND_PRED_INDEX_FOR_PERMUTN,
            permuted_values=these_permuted_values
        )[0]

        for i in range(len(these_new_matrices)):
            self.assertTrue(numpy.allclose(
                these_newnew_matrices[i], these_new_matrices[i],
                atol=TOLERANCE
            ))

    def test_unpermute_one_predictor_second(self):
        """Ensures correct output from _unpermute_one_predictor.

        In this case, using second set of inputs.
        """

        these_new_matrices = permutation._permute_one_predictor(
            predictor_matrices=copy.deepcopy(PREDICTOR_MATRICES_FOR_PERMUTN),
            separate_radar_heights=SECOND_SEPARATE_FLAG_FOR_PERMUTN,
            matrix_index=SECOND_MATRIX_INDEX_FOR_PERMUTN,
            predictor_index=SECOND_PRED_INDEX_FOR_PERMUTN,
            permuted_values=None
        )[0]

        these_orig_matrices = permutation._unpermute_one_predictor(
            predictor_matrices=copy.deepcopy(these_new_matrices),
            clean_predictor_matrices=PREDICTOR_MATRICES_FOR_PERMUTN,
            separate_radar_heights=SECOND_SEPARATE_FLAG_FOR_PERMUTN,
            matrix_index=SECOND_MATRIX_INDEX_FOR_PERMUTN,
            predictor_index=SECOND_PRED_INDEX_FOR_PERMUTN)

        for i in range(len(these_new_matrices)):
            self.assertTrue(numpy.allclose(
                these_orig_matrices[i], PREDICTOR_MATRICES_FOR_PERMUTN[i],
                atol=TOLERANCE
            ))

    def test_permute_one_predictor_third(self):
        """Ensures correct output from _permute_one_predictor.

        In this case, using third set of inputs.
        """

        these_new_matrices, these_permuted_values = (
            permutation._permute_one_predictor(
                predictor_matrices=copy.deepcopy(
                    PREDICTOR_MATRICES_FOR_PERMUTN),
                separate_radar_heights=THIRD_SEPARATE_FLAG_FOR_PERMUTN,
                matrix_index=THIRD_MATRIX_INDEX_FOR_PERMUTN,
                predictor_index=THIRD_PRED_INDEX_FOR_PERMUTN,
                permuted_values=None)
        )

        self.assertTrue(_compare_before_and_after_permutn(
            orig_predictor_matrices=PREDICTOR_MATRICES_FOR_PERMUTN,
            new_predictor_matrices=these_new_matrices,
            separate_radar_heights=THIRD_SEPARATE_FLAG_FOR_PERMUTN,
            matrix_index=THIRD_MATRIX_INDEX_FOR_PERMUTN,
            predictor_index=THIRD_PRED_INDEX_FOR_PERMUTN
        ))

        these_newnew_matrices = permutation._permute_one_predictor(
            predictor_matrices=copy.deepcopy(these_new_matrices),
            separate_radar_heights=THIRD_SEPARATE_FLAG_FOR_PERMUTN,
            matrix_index=THIRD_MATRIX_INDEX_FOR_PERMUTN,
            predictor_index=THIRD_PRED_INDEX_FOR_PERMUTN,
            permuted_values=these_permuted_values
        )[0]

        for i in range(len(these_new_matrices)):
            self.assertTrue(numpy.allclose(
                these_newnew_matrices[i], these_new_matrices[i],
                atol=TOLERANCE
            ))

    def test_unpermute_one_predictor_third(self):
        """Ensures correct output from _unpermute_one_predictor.

        In this case, using third set of inputs.
        """

        these_new_matrices = permutation._permute_one_predictor(
            predictor_matrices=copy.deepcopy(PREDICTOR_MATRICES_FOR_PERMUTN),
            separate_radar_heights=THIRD_SEPARATE_FLAG_FOR_PERMUTN,
            matrix_index=THIRD_MATRIX_INDEX_FOR_PERMUTN,
            predictor_index=THIRD_PRED_INDEX_FOR_PERMUTN,
            permuted_values=None
        )[0]

        these_orig_matrices = permutation._unpermute_one_predictor(
            predictor_matrices=copy.deepcopy(these_new_matrices),
            clean_predictor_matrices=PREDICTOR_MATRICES_FOR_PERMUTN,
            separate_radar_heights=THIRD_SEPARATE_FLAG_FOR_PERMUTN,
            matrix_index=THIRD_MATRIX_INDEX_FOR_PERMUTN,
            predictor_index=THIRD_PRED_INDEX_FOR_PERMUTN)

        for i in range(len(these_new_matrices)):
            self.assertTrue(numpy.allclose(
                these_orig_matrices[i], PREDICTOR_MATRICES_FOR_PERMUTN[i],
                atol=TOLERANCE
            ))

    def test_flatten_last_two_dim(self):
        """Ensures correct output from flatten_last_two_dim."""

        this_radar_matrix_4d, this_orig_shape = (
            permutation.flatten_last_two_dim(RADAR_MATRIX_3D)
        )

        self.assertTrue(numpy.allclose(
            this_radar_matrix_4d, RADAR_MATRIX_3D_FLATTENED, atol=TOLERANCE
        ))
        self.assertTrue(numpy.array_equal(
            this_orig_shape, numpy.array(RADAR_MATRIX_3D.shape, dtype=int)
        ))

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

    def test_cross_entropy_function_binary(self):
        """Ensures correct output from cross_entropy_function.

        In this case there are 2 target classes (binary).
        """

        this_cross_entropy = permutation.cross_entropy_function(
            target_values=BINARY_TARGET_VALUES,
            class_probability_matrix=BINARY_PROBABILITY_MATRIX, test_mode=True)

        self.assertTrue(numpy.isclose(
            this_cross_entropy, BINARY_CROSS_ENTROPY, atol=XENTROPY_TOLERANCE
        ))

    def test_cross_entropy_function_ternary(self):
        """Ensures correct output from cross_entropy_function.

        In this case there are 3 target classes (ternary).
        """

        this_cross_entropy = permutation.cross_entropy_function(
            target_values=TERNARY_TARGET_VALUES,
            class_probability_matrix=TERNARY_PROBABILITY_MATRIX, test_mode=True)

        self.assertTrue(numpy.isclose(
            this_cross_entropy, TERNARY_CROSS_ENTROPY, atol=XENTROPY_TOLERANCE
        ))

    def test_negative_auc_function_binary(self):
        """Ensures correct output from negative_auc_function.

        In this case there are 2 target classes (binary).
        """

        this_negative_auc = permutation.negative_auc_function(
            target_values=BINARY_TARGET_VALUES,
            class_probability_matrix=BINARY_PROBABILITY_MATRIX)

        self.assertTrue(numpy.isclose(
            this_negative_auc, NEGATIVE_AUC, atol=TOLERANCE
        ))

    def test_negative_auc_function_ternary(self):
        """Ensures correct output from negative_auc_function.

        In this case there are 3 target classes (ternary).
        """

        with self.assertRaises(TypeError):
            permutation.negative_auc_function(
                target_values=TERNARY_TARGET_VALUES,
                class_probability_matrix=TERNARY_PROBABILITY_MATRIX)


if __name__ == '__main__':
    unittest.main()
