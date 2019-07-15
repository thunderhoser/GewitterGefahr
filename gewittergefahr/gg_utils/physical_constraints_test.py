"""Unit tests for physical_constraints.py."""

import copy
import unittest
import numpy
import pandas
from keras import backend as K
from gewittergefahr.gg_utils import physical_constraints
from gewittergefahr.gg_utils import radar_utils
from gewittergefahr.gg_utils import soundings
from gewittergefahr.deep_learning import cnn
from gewittergefahr.deep_learning import input_examples
from gewittergefahr.deep_learning import deep_learning_utils as dl_utils
from gewittergefahr.deep_learning import training_validation_io as trainval_io

TOLERANCE = 1e-6

# The following constants are used to test _find_constrained_radar_channels.
THESE_FIELD_NAMES = (
    [radar_utils.REFL_NAME] * 3 +
    [radar_utils.SPECTRUM_WIDTH_NAME] * 3 +
    [radar_utils.VORTICITY_NAME] * 3 +
    [radar_utils.VORTICITY_NAME] * 3
)

THESE_OPERATION_NAMES = 4 * [
    input_examples.MIN_OPERATION_NAME, input_examples.MEAN_OPERATION_NAME,
    input_examples.MAX_OPERATION_NAME
]

THESE_MIN_HEIGHTS_M_AGL = numpy.array(
    [1000] * 3 + [1000] * 3 + [2000] * 3 + [5000] * 3,
    dtype=int
)

THESE_MAX_HEIGHTS_M_AGL = numpy.array(
    [3000] * 3 + [3000] * 3 + [4000] * 3 + [8000] * 3,
    dtype=int
)

FIRST_LIST_OF_OPERATION_DICTS = [
    {
        input_examples.RADAR_FIELD_KEY: THESE_FIELD_NAMES[m],
        input_examples.OPERATION_NAME_KEY: THESE_OPERATION_NAMES[m],
        input_examples.MIN_HEIGHT_KEY: THESE_MIN_HEIGHTS_M_AGL[m],
        input_examples.MAX_HEIGHT_KEY: THESE_MAX_HEIGHTS_M_AGL[m]
    }
    for m in range(len(THESE_MIN_HEIGHTS_M_AGL))
]

FIRST_GREATER_INDICES = numpy.array([1, 2, 4, 5, 7, 8, 10, 11], dtype=int)
FIRST_LESS_INDICES = numpy.array([0, 1, 3, 4, 6, 7, 9, 10], dtype=int)

THESE_FIELD_NAMES = (
    [radar_utils.REFL_NAME] * 2 +
    [radar_utils.SPECTRUM_WIDTH_NAME] * 2 +
    [radar_utils.VORTICITY_NAME] * 2 +
    [radar_utils.VORTICITY_NAME] * 2
)

THESE_OPERATION_NAMES = [
    input_examples.MEAN_OPERATION_NAME, input_examples.MIN_OPERATION_NAME,
    input_examples.MEAN_OPERATION_NAME, input_examples.MAX_OPERATION_NAME,
    input_examples.MIN_OPERATION_NAME, input_examples.MEAN_OPERATION_NAME,
    input_examples.MAX_OPERATION_NAME, input_examples.MIN_OPERATION_NAME
]

THESE_MIN_HEIGHTS_M_AGL = numpy.array(
    [1000] * 2 + [1000] * 2 + [2000] * 2 + [5000] * 2,
    dtype=int
)

THESE_MAX_HEIGHTS_M_AGL = numpy.array(
    [3000] * 2 + [3000] * 2 + [4000] * 2 + [8000] * 2,
    dtype=int
)

SECOND_LIST_OF_OPERATION_DICTS = [
    {
        input_examples.RADAR_FIELD_KEY: THESE_FIELD_NAMES[m],
        input_examples.OPERATION_NAME_KEY: THESE_OPERATION_NAMES[m],
        input_examples.MIN_HEIGHT_KEY: THESE_MIN_HEIGHTS_M_AGL[m],
        input_examples.MAX_HEIGHT_KEY: THESE_MAX_HEIGHTS_M_AGL[m]
    }
    for m in range(len(THESE_MIN_HEIGHTS_M_AGL))
]

SECOND_GREATER_INDICES = numpy.array([0, 3, 5, 6], dtype=int)
SECOND_LESS_INDICES = numpy.array([1, 2, 4, 7], dtype=int)

THESE_FIELD_NAMES = (
    [radar_utils.REFL_NAME] +
    [radar_utils.SPECTRUM_WIDTH_NAME] * 2 +
    [radar_utils.VORTICITY_NAME] +
    [radar_utils.VORTICITY_NAME]
)

THESE_OPERATION_NAMES = [
    input_examples.MEAN_OPERATION_NAME,
    input_examples.MEAN_OPERATION_NAME, input_examples.MAX_OPERATION_NAME,
    input_examples.MIN_OPERATION_NAME,
    input_examples.MAX_OPERATION_NAME
]

THESE_MIN_HEIGHTS_M_AGL = numpy.array(
    [1000] + [1000] * 2 + [2000] + [5000],
    dtype=int
)

THESE_MAX_HEIGHTS_M_AGL = numpy.array(
    [3000] + [3000] * 2 + [4000] + [8000],
    dtype=int
)

THIRD_LIST_OF_OPERATION_DICTS = [
    {
        input_examples.RADAR_FIELD_KEY: THESE_FIELD_NAMES[m],
        input_examples.OPERATION_NAME_KEY: THESE_OPERATION_NAMES[m],
        input_examples.MIN_HEIGHT_KEY: THESE_MIN_HEIGHTS_M_AGL[m],
        input_examples.MAX_HEIGHT_KEY: THESE_MAX_HEIGHTS_M_AGL[m]
    }
    for m in range(len(THESE_MIN_HEIGHTS_M_AGL))
]

THIRD_GREATER_INDICES = numpy.array([2], dtype=int)
THIRD_LESS_INDICES = numpy.array([1], dtype=int)

THESE_FIELD_NAMES = [
    radar_utils.REFL_NAME, radar_utils.SPECTRUM_WIDTH_NAME,
    radar_utils.VORTICITY_NAME, radar_utils.VORTICITY_NAME
]

THESE_OPERATION_NAMES = [
    input_examples.MEAN_OPERATION_NAME, input_examples.MEAN_OPERATION_NAME,
    input_examples.MIN_OPERATION_NAME, input_examples.MAX_OPERATION_NAME
]

THESE_MIN_HEIGHTS_M_AGL = numpy.array([1000, 1000, 2000, 5000], dtype=int)
THESE_MAX_HEIGHTS_M_AGL = numpy.array([3000, 3000, 4000, 8000], dtype=int)

FOURTH_LIST_OF_OPERATION_DICTS = [
    {
        input_examples.RADAR_FIELD_KEY: THESE_FIELD_NAMES[m],
        input_examples.OPERATION_NAME_KEY: THESE_OPERATION_NAMES[m],
        input_examples.MIN_HEIGHT_KEY: THESE_MIN_HEIGHTS_M_AGL[m],
        input_examples.MAX_HEIGHT_KEY: THESE_MAX_HEIGHTS_M_AGL[m]
    }
    for m in range(len(THESE_MIN_HEIGHTS_M_AGL))
]

FOURTH_GREATER_INDICES = numpy.array([], dtype=int)
FOURTH_LESS_INDICES = numpy.array([], dtype=int)

# The following constants are used to test radar_constraints_to_loss_fn.
NUM_EXAMPLES = 10
NUM_RADAR_ROWS = 32
NUM_RADAR_COLUMNS = 32

# The following constants are used to test _normalize_minima_and_maxima.
NUM_REFL_HEIGHTS = 12
NUM_SOUNDING_HEIGHTS = 49
MIN_NORMALIZED_VALUE = 0.
MAX_NORMALIZED_VALUE = 1.

AZ_SHEAR_FIELD_NAMES = [
    radar_utils.LOW_LEVEL_SHEAR_NAME, radar_utils.MID_LEVEL_SHEAR_NAME
]

SOUNDING_FIELD_NAMES = [
    soundings.VIRTUAL_POTENTIAL_TEMPERATURE_NAME,
    soundings.SPECIFIC_HUMIDITY_NAME, soundings.RELATIVE_HUMIDITY_NAME,
    soundings.U_WIND_NAME, soundings.V_WIND_NAME
]

THIS_DICT = {
    radar_utils.REFL_NAME: numpy.array([22, 15, 0, 77.5]),
    radar_utils.SPECTRUM_WIDTH_NAME: numpy.array([3, 1.5, 0, 10]),
    radar_utils.VORTICITY_NAME: numpy.array([2e-4, 3e-4, 0, 0.02]),
    radar_utils.DIVERGENCE_NAME: numpy.array([2e-4, 2e-4, 0, 0.015]),
    radar_utils.LOW_LEVEL_SHEAR_NAME: numpy.array([2e-4, 3e-4, 0, 0.02]),
    radar_utils.MID_LEVEL_SHEAR_NAME: numpy.array([2e-4, 2e-4, 0, 0.015])
}

RADAR_NORMALIZATION_TABLE = pandas.DataFrame.from_dict(
    THIS_DICT, orient='index')

THIS_RENAMING_DICT = {
    0: dl_utils.MEAN_VALUE_COLUMN,
    1: dl_utils.STANDARD_DEVIATION_COLUMN,
    2: dl_utils.MIN_VALUE_COLUMN,
    3: dl_utils.MAX_VALUE_COLUMN
}

RADAR_NORMALIZATION_TABLE.rename(columns=THIS_RENAMING_DICT, inplace=True)

THIS_DICT = {
    soundings.VIRTUAL_POTENTIAL_TEMPERATURE_NAME:
        numpy.array([300, 50, 250, 500.5]),
    soundings.SPECIFIC_HUMIDITY_NAME: numpy.array([0.004, 0.003, 0, 0.02]),
    soundings.RELATIVE_HUMIDITY_NAME: numpy.array([0.7, 0.2, 0, 1]),
    soundings.U_WIND_NAME: numpy.array([-0.5, 5, -30, 30]),
    soundings.V_WIND_NAME: numpy.array([0.5, 5, -30, 30])
}

SOUNDING_NORMALIZATION_TABLE = pandas.DataFrame.from_dict(
    THIS_DICT, orient='index')
SOUNDING_NORMALIZATION_TABLE.rename(columns=THIS_RENAMING_DICT, inplace=True)

THIS_REFLECTIVITY_TENSOR = K.placeholder(
    shape=(NUM_EXAMPLES, NUM_RADAR_ROWS, NUM_RADAR_COLUMNS, NUM_REFL_HEIGHTS,
           1),
    dtype=float
)

THIS_AZ_SHEAR_TENSOR = K.placeholder(
    shape=(
        NUM_EXAMPLES, NUM_RADAR_ROWS, NUM_RADAR_COLUMNS,
        len(AZ_SHEAR_FIELD_NAMES)
    ), dtype=float
)

THIS_SOUNDING_TENSOR = K.placeholder(
    shape=(NUM_EXAMPLES, NUM_SOUNDING_HEIGHTS, len(SOUNDING_FIELD_NAMES)),
    dtype=float
)

FIRST_LIST_OF_INPUT_TENSORS = [
    THIS_REFLECTIVITY_TENSOR, THIS_AZ_SHEAR_TENSOR, THIS_SOUNDING_TENSOR
]

FIRST_METADATA_DICT = {
    cnn.CONV_2D3D_KEY: True,
    cnn.LAYER_OPERATIONS_KEY: None,
    cnn.TRAINING_OPTION_DICT_KEY: {
        trainval_io.RADAR_FIELDS_KEY: AZ_SHEAR_FIELD_NAMES,
        trainval_io.SOUNDING_FIELDS_KEY: SOUNDING_FIELD_NAMES,
        trainval_io.MIN_NORMALIZED_VALUE_KEY: MIN_NORMALIZED_VALUE,
        trainval_io.MAX_NORMALIZED_VALUE_KEY: MAX_NORMALIZED_VALUE
    }
}

FIRST_MIN_VALUES_Z_NORM = [
    numpy.array([-22. / 15]),
    numpy.full(len(AZ_SHEAR_FIELD_NAMES), numpy.nan),
    numpy.array([-6, -4. / 3, -3.5, numpy.nan, numpy.nan])
]

FIRST_MAX_VALUES_Z_NORM = [
    numpy.full(1, numpy.nan),
    numpy.full(len(AZ_SHEAR_FIELD_NAMES), numpy.nan),
    numpy.array([numpy.nan, 332, 1.5, numpy.nan, numpy.nan])
]

FIRST_MIN_VALUES_MINMAX_NORM = [
    numpy.array([0.]),
    numpy.full(len(AZ_SHEAR_FIELD_NAMES), numpy.nan),
    numpy.array([0, 0, 0, numpy.nan, numpy.nan])
]

FIRST_MAX_VALUES_MINMAX_NORM = [
    numpy.full(1, numpy.nan),
    numpy.full(len(AZ_SHEAR_FIELD_NAMES), numpy.nan),
    numpy.array([numpy.nan, 1, 1, numpy.nan, numpy.nan])
]

SECOND_LIST_OF_INPUT_TENSORS = [THIS_AZ_SHEAR_TENSOR, THIS_SOUNDING_TENSOR]

SECOND_METADATA_DICT = {
    cnn.CONV_2D3D_KEY: False,
    cnn.LAYER_OPERATIONS_KEY: None,
    cnn.TRAINING_OPTION_DICT_KEY: {
        trainval_io.RADAR_FIELDS_KEY: AZ_SHEAR_FIELD_NAMES,
        trainval_io.SOUNDING_FIELDS_KEY: SOUNDING_FIELD_NAMES,
        trainval_io.MIN_NORMALIZED_VALUE_KEY: MIN_NORMALIZED_VALUE,
        trainval_io.MAX_NORMALIZED_VALUE_KEY: MAX_NORMALIZED_VALUE
    }
}

SECOND_MIN_VALUES_Z_NORM = [
    numpy.full(len(AZ_SHEAR_FIELD_NAMES), numpy.nan),
    numpy.array([-6, -4. / 3, -3.5, numpy.nan, numpy.nan])
]

SECOND_MAX_VALUES_Z_NORM = [
    numpy.full(len(AZ_SHEAR_FIELD_NAMES), numpy.nan),
    numpy.array([numpy.nan, 332, 1.5, numpy.nan, numpy.nan])
]

SECOND_MIN_VALUES_MINMAX_NORM = [
    numpy.full(len(AZ_SHEAR_FIELD_NAMES), numpy.nan),
    numpy.array([0, 0, 0, numpy.nan, numpy.nan])
]

SECOND_MAX_VALUES_MINMAX_NORM = [
    numpy.full(len(AZ_SHEAR_FIELD_NAMES), numpy.nan),
    numpy.array([numpy.nan, 1, 1, numpy.nan, numpy.nan])
]

THIS_LIST_OF_OPERATION_DICTS = copy.deepcopy(FOURTH_LIST_OF_OPERATION_DICTS)

THIS_RADAR_TENSOR = K.placeholder(
    shape=(
        NUM_EXAMPLES, NUM_RADAR_ROWS, NUM_RADAR_COLUMNS,
        len(THIS_LIST_OF_OPERATION_DICTS)
    ), dtype=float
)

THIRD_LIST_OF_INPUT_TENSORS = [THIS_RADAR_TENSOR, THIS_SOUNDING_TENSOR]

THIRD_METADATA_DICT = {
    cnn.CONV_2D3D_KEY: False,
    cnn.LAYER_OPERATIONS_KEY: THIS_LIST_OF_OPERATION_DICTS,
    cnn.TRAINING_OPTION_DICT_KEY: {
        trainval_io.SOUNDING_FIELDS_KEY: SOUNDING_FIELD_NAMES,
        trainval_io.MIN_NORMALIZED_VALUE_KEY: MIN_NORMALIZED_VALUE,
        trainval_io.MAX_NORMALIZED_VALUE_KEY: MAX_NORMALIZED_VALUE
    }
}

THIRD_MIN_VALUES_Z_NORM = [
    numpy.array([-22. / 15, -2, numpy.nan, numpy.nan]),
    numpy.array([-6, -4. / 3, -3.5, numpy.nan, numpy.nan])
]

THIRD_MAX_VALUES_Z_NORM = [
    numpy.full(len(THIS_LIST_OF_OPERATION_DICTS), numpy.nan),
    numpy.array([numpy.nan, 332, 1.5, numpy.nan, numpy.nan])
]

THIRD_MIN_VALUES_MINMAX_NORM = [
    numpy.array([0, 0, numpy.nan, numpy.nan]),
    numpy.array([0, 0, 0, numpy.nan, numpy.nan])
]

THIRD_MAX_VALUES_MINMAX_NORM = [
    numpy.full(len(THIS_LIST_OF_OPERATION_DICTS), numpy.nan),
    numpy.array([numpy.nan, 1, 1, numpy.nan, numpy.nan])
]

FOURTH_LIST_OF_INPUT_TENSORS = [THIS_RADAR_TENSOR]

FOURTH_METADATA_DICT = {
    cnn.CONV_2D3D_KEY: False,
    cnn.LAYER_OPERATIONS_KEY: THIS_LIST_OF_OPERATION_DICTS,
    cnn.TRAINING_OPTION_DICT_KEY: {
        trainval_io.SOUNDING_FIELDS_KEY: None,
        trainval_io.MIN_NORMALIZED_VALUE_KEY: MIN_NORMALIZED_VALUE,
        trainval_io.MAX_NORMALIZED_VALUE_KEY: MAX_NORMALIZED_VALUE
    }
}

FOURTH_MIN_VALUES_Z_NORM = [
    numpy.array([-22. / 15, -2, numpy.nan, numpy.nan])
]

FOURTH_MAX_VALUES_Z_NORM = [
    numpy.full(len(THIS_LIST_OF_OPERATION_DICTS), numpy.nan)
]

FOURTH_MIN_VALUES_MINMAX_NORM = [
    numpy.array([0, 0, numpy.nan, numpy.nan])
]

FOURTH_MAX_VALUES_MINMAX_NORM = [
    numpy.full(len(THIS_LIST_OF_OPERATION_DICTS), numpy.nan)
]


def _compare_array_lists(first_array_list, second_array_list):
    """Compares two lists of numpy arrays.

    Each list must be 1-D.

    :param first_array_list: First list.
    :param second_array_list: Second list.
    :return: are_lists_equal: Boolean flag.
    """

    num_first_arrays = len(first_array_list)
    num_second_arrays = len(second_array_list)
    if num_first_arrays != num_second_arrays:
        return False

    for i in range(num_first_arrays):
        if not numpy.allclose(
                first_array_list[i], second_array_list[i], atol=TOLERANCE,
                equal_nan=True):
            return False

    return True


class PhysicalConstraintsTests(unittest.TestCase):
    """Each method is a unit test for physical_constraints.py."""

    def test_find_constrained_radar_channels_first(self):
        """Ensures correct output from _find_constrained_radar_channels.

        In this case, using first set of channels.
        """

        these_greater_indices, these_less_indices = (
            physical_constraints._find_constrained_radar_channels(
                FIRST_LIST_OF_OPERATION_DICTS)
        )

        self.assertTrue(numpy.array_equal(
            these_greater_indices, FIRST_GREATER_INDICES
        ))
        self.assertTrue(numpy.array_equal(
            these_less_indices, FIRST_LESS_INDICES
        ))

    def test_find_constrained_radar_channels_second(self):
        """Ensures correct output from _find_constrained_radar_channels.

        In this case, using second set of channels.
        """

        these_greater_indices, these_less_indices = (
            physical_constraints._find_constrained_radar_channels(
                SECOND_LIST_OF_OPERATION_DICTS)
        )

        self.assertTrue(numpy.array_equal(
            these_greater_indices, SECOND_GREATER_INDICES
        ))
        self.assertTrue(numpy.array_equal(
            these_less_indices, SECOND_LESS_INDICES
        ))

    def test_find_constrained_radar_channels_third(self):
        """Ensures correct output from _find_constrained_radar_channels.

        In this case, using third set of channels.
        """

        these_greater_indices, these_less_indices = (
            physical_constraints._find_constrained_radar_channels(
                THIRD_LIST_OF_OPERATION_DICTS)
        )

        self.assertTrue(numpy.array_equal(
            these_greater_indices, THIRD_GREATER_INDICES
        ))
        self.assertTrue(numpy.array_equal(
            these_less_indices, THIRD_LESS_INDICES
        ))

    def test_find_constrained_radar_channels_fourth(self):
        """Ensures correct output from _find_constrained_radar_channels.

        In this case, using fourth set of channels.
        """

        these_greater_indices, these_less_indices = (
            physical_constraints._find_constrained_radar_channels(
                FOURTH_LIST_OF_OPERATION_DICTS)
        )

        self.assertTrue(numpy.array_equal(
            these_greater_indices, FOURTH_GREATER_INDICES
        ))
        self.assertTrue(numpy.array_equal(
            these_less_indices, FOURTH_LESS_INDICES
        ))

    def test_radar_constraints_to_loss_fn_first(self):
        """Ensures correct output from radar_constraints_to_loss_fn.

        In this case, using first set of channels.
        """

        these_dimensions = (
            NUM_EXAMPLES, NUM_RADAR_ROWS, NUM_RADAR_COLUMNS,
            len(FIRST_LIST_OF_OPERATION_DICTS)
        )

        this_radar_tensor = K.placeholder(shape=these_dimensions, dtype=float)

        this_loss_tensor = physical_constraints.radar_constraints_to_loss_fn(
            radar_tensor=this_radar_tensor,
            list_of_layer_operation_dicts=FIRST_LIST_OF_OPERATION_DICTS)

        self.assertTrue(this_loss_tensor is not None)

    def test_radar_constraints_to_loss_fn_second(self):
        """Ensures correct output from radar_constraints_to_loss_fn.

        In this case, using second set of channels.
        """

        these_dimensions = (
            NUM_EXAMPLES, NUM_RADAR_ROWS, NUM_RADAR_COLUMNS,
            len(SECOND_LIST_OF_OPERATION_DICTS)
        )

        this_radar_tensor = K.placeholder(shape=these_dimensions, dtype=float)

        this_loss_tensor = physical_constraints.radar_constraints_to_loss_fn(
            radar_tensor=this_radar_tensor,
            list_of_layer_operation_dicts=SECOND_LIST_OF_OPERATION_DICTS)

        self.assertTrue(this_loss_tensor is not None)

    def test_radar_constraints_to_loss_fn_third(self):
        """Ensures correct output from radar_constraints_to_loss_fn.

        In this case, using third set of channels.
        """

        these_dimensions = (
            NUM_EXAMPLES, NUM_RADAR_ROWS, NUM_RADAR_COLUMNS,
            len(THIRD_LIST_OF_OPERATION_DICTS)
        )

        this_radar_tensor = K.placeholder(shape=these_dimensions, dtype=float)

        this_loss_tensor = physical_constraints.radar_constraints_to_loss_fn(
            radar_tensor=this_radar_tensor,
            list_of_layer_operation_dicts=THIRD_LIST_OF_OPERATION_DICTS)

        self.assertTrue(this_loss_tensor is not None)

    def test_radar_constraints_to_loss_fn_fourth(self):
        """Ensures correct output from radar_constraints_to_loss_fn.

        In this case, using fourth set of channels.
        """

        these_dimensions = (
            NUM_EXAMPLES, NUM_RADAR_ROWS, NUM_RADAR_COLUMNS,
            len(FOURTH_LIST_OF_OPERATION_DICTS)
        )

        this_radar_tensor = K.placeholder(shape=these_dimensions, dtype=float)

        this_loss_tensor = physical_constraints.radar_constraints_to_loss_fn(
            radar_tensor=this_radar_tensor,
            list_of_layer_operation_dicts=FOURTH_LIST_OF_OPERATION_DICTS)

        self.assertTrue(this_loss_tensor is None)

    def test_normalize_minima_and_maxima_first_z(self):
        """Ensures correct output from _normalize_minima_and_maxima.

        In this case, using first set of inputs and z-score normalization.
        """

        FIRST_METADATA_DICT[cnn.TRAINING_OPTION_DICT_KEY][
            trainval_io.NORMALIZATION_TYPE_KEY
        ] = dl_utils.Z_NORMALIZATION_TYPE_STRING

        these_min_values_by_tensor, these_max_values_by_tensor = (
            physical_constraints._normalize_minima_and_maxima(
                list_of_input_tensors=FIRST_LIST_OF_INPUT_TENSORS,
                cnn_metadata_dict=FIRST_METADATA_DICT, test_mode=True,
                radar_normalization_table=RADAR_NORMALIZATION_TABLE,
                sounding_normalization_table=SOUNDING_NORMALIZATION_TABLE)
        )

        self.assertTrue(_compare_array_lists(
            FIRST_MIN_VALUES_Z_NORM, these_min_values_by_tensor
        ))
        self.assertTrue(_compare_array_lists(
            FIRST_MAX_VALUES_Z_NORM, these_max_values_by_tensor
        ))

    def test_normalize_minima_and_maxima_first_minmax(self):
        """Ensures correct output from _normalize_minima_and_maxima.

        In this case, using first set of inputs and minmax normalization.
        """

        FIRST_METADATA_DICT[cnn.TRAINING_OPTION_DICT_KEY][
            trainval_io.NORMALIZATION_TYPE_KEY
        ] = dl_utils.MINMAX_NORMALIZATION_TYPE_STRING

        these_min_values_by_tensor, these_max_values_by_tensor = (
            physical_constraints._normalize_minima_and_maxima(
                list_of_input_tensors=FIRST_LIST_OF_INPUT_TENSORS,
                cnn_metadata_dict=FIRST_METADATA_DICT, test_mode=True,
                radar_normalization_table=RADAR_NORMALIZATION_TABLE,
                sounding_normalization_table=SOUNDING_NORMALIZATION_TABLE)
        )

        self.assertTrue(_compare_array_lists(
            FIRST_MIN_VALUES_MINMAX_NORM, these_min_values_by_tensor
        ))
        self.assertTrue(_compare_array_lists(
            FIRST_MAX_VALUES_MINMAX_NORM, these_max_values_by_tensor
        ))

    def test_normalize_minima_and_maxima_second_z(self):
        """Ensures correct output from _normalize_minima_and_maxima.

        In this case, using second set of inputs and z-score normalization.
        """

        SECOND_METADATA_DICT[cnn.TRAINING_OPTION_DICT_KEY][
            trainval_io.NORMALIZATION_TYPE_KEY
        ] = dl_utils.Z_NORMALIZATION_TYPE_STRING

        these_min_values_by_tensor, these_max_values_by_tensor = (
            physical_constraints._normalize_minima_and_maxima(
                list_of_input_tensors=SECOND_LIST_OF_INPUT_TENSORS,
                cnn_metadata_dict=SECOND_METADATA_DICT, test_mode=True,
                radar_normalization_table=RADAR_NORMALIZATION_TABLE,
                sounding_normalization_table=SOUNDING_NORMALIZATION_TABLE)
        )

        self.assertTrue(_compare_array_lists(
            SECOND_MIN_VALUES_Z_NORM, these_min_values_by_tensor
        ))
        self.assertTrue(_compare_array_lists(
            SECOND_MAX_VALUES_Z_NORM, these_max_values_by_tensor
        ))

    def test_normalize_minima_and_maxima_second_minmax(self):
        """Ensures correct output from _normalize_minima_and_maxima.

        In this case, using second set of inputs and minmax normalization.
        """

        SECOND_METADATA_DICT[cnn.TRAINING_OPTION_DICT_KEY][
            trainval_io.NORMALIZATION_TYPE_KEY
        ] = dl_utils.MINMAX_NORMALIZATION_TYPE_STRING

        these_min_values_by_tensor, these_max_values_by_tensor = (
            physical_constraints._normalize_minima_and_maxima(
                list_of_input_tensors=SECOND_LIST_OF_INPUT_TENSORS,
                cnn_metadata_dict=SECOND_METADATA_DICT, test_mode=True,
                radar_normalization_table=RADAR_NORMALIZATION_TABLE,
                sounding_normalization_table=SOUNDING_NORMALIZATION_TABLE)
        )

        self.assertTrue(_compare_array_lists(
            SECOND_MIN_VALUES_MINMAX_NORM, these_min_values_by_tensor
        ))
        self.assertTrue(_compare_array_lists(
            SECOND_MAX_VALUES_MINMAX_NORM, these_max_values_by_tensor
        ))

    def test_normalize_minima_and_maxima_third_z(self):
        """Ensures correct output from _normalize_minima_and_maxima.

        In this case, using third set of inputs and z-score normalization.
        """

        THIRD_METADATA_DICT[cnn.TRAINING_OPTION_DICT_KEY][
            trainval_io.NORMALIZATION_TYPE_KEY
        ] = dl_utils.Z_NORMALIZATION_TYPE_STRING

        these_min_values_by_tensor, these_max_values_by_tensor = (
            physical_constraints._normalize_minima_and_maxima(
                list_of_input_tensors=THIRD_LIST_OF_INPUT_TENSORS,
                cnn_metadata_dict=THIRD_METADATA_DICT, test_mode=True,
                radar_normalization_table=RADAR_NORMALIZATION_TABLE,
                sounding_normalization_table=SOUNDING_NORMALIZATION_TABLE)
        )

        self.assertTrue(_compare_array_lists(
            THIRD_MIN_VALUES_Z_NORM, these_min_values_by_tensor
        ))
        self.assertTrue(_compare_array_lists(
            THIRD_MAX_VALUES_Z_NORM, these_max_values_by_tensor
        ))

    def test_normalize_minima_and_maxima_third_minmax(self):
        """Ensures correct output from _normalize_minima_and_maxima.

        In this case, using third set of inputs and minmax normalization.
        """

        THIRD_METADATA_DICT[cnn.TRAINING_OPTION_DICT_KEY][
            trainval_io.NORMALIZATION_TYPE_KEY
        ] = dl_utils.MINMAX_NORMALIZATION_TYPE_STRING

        these_min_values_by_tensor, these_max_values_by_tensor = (
            physical_constraints._normalize_minima_and_maxima(
                list_of_input_tensors=THIRD_LIST_OF_INPUT_TENSORS,
                cnn_metadata_dict=THIRD_METADATA_DICT, test_mode=True,
                radar_normalization_table=RADAR_NORMALIZATION_TABLE,
                sounding_normalization_table=SOUNDING_NORMALIZATION_TABLE)
        )

        self.assertTrue(_compare_array_lists(
            THIRD_MIN_VALUES_MINMAX_NORM, these_min_values_by_tensor
        ))
        self.assertTrue(_compare_array_lists(
            THIRD_MAX_VALUES_MINMAX_NORM, these_max_values_by_tensor
        ))

    def test_normalize_minima_and_maxima_fourth_z(self):
        """Ensures correct output from _normalize_minima_and_maxima.

        In this case, using fourth set of inputs and z-score normalization.
        """

        FOURTH_METADATA_DICT[cnn.TRAINING_OPTION_DICT_KEY][
            trainval_io.NORMALIZATION_TYPE_KEY
        ] = dl_utils.Z_NORMALIZATION_TYPE_STRING

        these_min_values_by_tensor, these_max_values_by_tensor = (
            physical_constraints._normalize_minima_and_maxima(
                list_of_input_tensors=FOURTH_LIST_OF_INPUT_TENSORS,
                cnn_metadata_dict=FOURTH_METADATA_DICT, test_mode=True,
                radar_normalization_table=RADAR_NORMALIZATION_TABLE,
                sounding_normalization_table=SOUNDING_NORMALIZATION_TABLE)
        )

        self.assertTrue(_compare_array_lists(
            FOURTH_MIN_VALUES_Z_NORM, these_min_values_by_tensor
        ))
        self.assertTrue(_compare_array_lists(
            FOURTH_MAX_VALUES_Z_NORM, these_max_values_by_tensor
        ))

    def test_normalize_minima_and_maxima_fourth_minmax(self):
        """Ensures correct output from _normalize_minima_and_maxima.

        In this case, using fourth set of inputs and minmax normalization.
        """

        FOURTH_METADATA_DICT[cnn.TRAINING_OPTION_DICT_KEY][
            trainval_io.NORMALIZATION_TYPE_KEY
        ] = dl_utils.MINMAX_NORMALIZATION_TYPE_STRING

        these_min_values_by_tensor, these_max_values_by_tensor = (
            physical_constraints._normalize_minima_and_maxima(
                list_of_input_tensors=FOURTH_LIST_OF_INPUT_TENSORS,
                cnn_metadata_dict=FOURTH_METADATA_DICT, test_mode=True,
                radar_normalization_table=RADAR_NORMALIZATION_TABLE,
                sounding_normalization_table=SOUNDING_NORMALIZATION_TABLE)
        )

        self.assertTrue(_compare_array_lists(
            FOURTH_MIN_VALUES_MINMAX_NORM, these_min_values_by_tensor
        ))
        self.assertTrue(_compare_array_lists(
            FOURTH_MAX_VALUES_MINMAX_NORM, these_max_values_by_tensor
        ))


if __name__ == '__main__':
    unittest.main()
