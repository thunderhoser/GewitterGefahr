"""Unit tests for physical_constraints.py."""

import unittest
import numpy
from keras import backend as K
from gewittergefahr.gg_utils import physical_constraints
from gewittergefahr.gg_utils import radar_utils
from gewittergefahr.deep_learning import input_examples

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
NUM_GRID_ROWS = 32
NUM_GRID_COLUMNS = 32


class PhysicalConstraintsTests(unittest.TestCase):
    """Each method is a unit test for physical_constraints.py."""

    def test__find_constrained_radar_channels_first(self):
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

    def test__find_constrained_radar_channels_second(self):
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

    def test__find_constrained_radar_channels_third(self):
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

    def test__find_constrained_radar_channels_fourth(self):
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
            NUM_EXAMPLES, NUM_GRID_ROWS, NUM_GRID_COLUMNS,
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
            NUM_EXAMPLES, NUM_GRID_ROWS, NUM_GRID_COLUMNS,
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
            NUM_EXAMPLES, NUM_GRID_ROWS, NUM_GRID_COLUMNS,
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
            NUM_EXAMPLES, NUM_GRID_ROWS, NUM_GRID_COLUMNS,
            len(FOURTH_LIST_OF_OPERATION_DICTS)
        )

        this_radar_tensor = K.placeholder(shape=these_dimensions, dtype=float)

        this_loss_tensor = physical_constraints.radar_constraints_to_loss_fn(
            radar_tensor=this_radar_tensor,
            list_of_layer_operation_dicts=FOURTH_LIST_OF_OPERATION_DICTS)

        self.assertTrue(this_loss_tensor is None)


if __name__ == '__main__':
    unittest.main()
