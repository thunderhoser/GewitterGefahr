"""Unit tests for radar_plotting.py."""

import unittest
import numpy
from gewittergefahr.gg_utils import radar_utils
from gewittergefahr.deep_learning import input_examples
from gewittergefahr.plotting import radar_plotting

METRES_TO_KM = 0.001

# The following constants are used to test layer_ops_to_field_and_panel_names.
THESE_FIELD_NAMES = (
    [radar_utils.REFL_NAME] * 3 +
    [radar_utils.SPECTRUM_WIDTH_NAME] * 3 +
    [radar_utils.VORTICITY_NAME] * 3 +
    [radar_utils.VORTICITY_NAME] * 3
)

LAYER_OPERATION_NAMES = [
    input_examples.MIN_OPERATION_NAME, input_examples.MEAN_OPERATION_NAME,
    input_examples.MAX_OPERATION_NAME
] * 4

MIN_HEIGHTS_M_AGL = numpy.array(
    [1000] * 3 + [1000] * 3 + [2000] * 3 + [5000] * 3,
    dtype=int
)

MAX_HEIGHTS_M_AGL = numpy.array(
    [3000] * 3 + [3000] * 3 + [4000] * 3 + [8000] * 3,
    dtype=int
)

FIELD_NAMES_WITH_LAYER_OPS = THESE_FIELD_NAMES + []

NUM_LAYER_OPERATIONS = len(THESE_FIELD_NAMES)
LIST_OF_LAYER_OPERATION_DICTS = [{}] * NUM_LAYER_OPERATIONS
PANEL_NAMES_WITH_LAYER_OPS = [''] * NUM_LAYER_OPERATIONS

for k in range(NUM_LAYER_OPERATIONS):
    LIST_OF_LAYER_OPERATION_DICTS[k] = {
        input_examples.RADAR_FIELD_KEY: THESE_FIELD_NAMES[k],
        input_examples.OPERATION_NAME_KEY: LAYER_OPERATION_NAMES[k],
        input_examples.MIN_HEIGHT_KEY: MIN_HEIGHTS_M_AGL[k],
        input_examples.MAX_HEIGHT_KEY: MAX_HEIGHTS_M_AGL[k]
    }

    PANEL_NAMES_WITH_LAYER_OPS[k] = (
        '{0:s}\n{1:s} from {2:d}-{3:d} km AGL'
    ).format(
        radar_plotting._field_name_to_plotting_units(THESE_FIELD_NAMES[k]),
        LAYER_OPERATION_NAMES[k].upper(),
        int(numpy.round(MIN_HEIGHTS_M_AGL[k] * METRES_TO_KM)),
        int(numpy.round(MAX_HEIGHTS_M_AGL[k] * METRES_TO_KM))
    )

# The following constants are used to test
# radar_fields_and_heights_to_panel_names.
FIELD_NAME_BY_PAIR = [
    radar_utils.REFL_NAME, radar_utils.SPECTRUM_WIDTH_NAME,
    radar_utils.VORTICITY_NAME, radar_utils.REFL_NAME
]
HEIGHT_BY_PAIR_M_AGL = numpy.array([1000, 3000, 2000, 10000])

NUM_FIELD_HEIGHT_PAIRS = len(FIELD_NAME_BY_PAIR)
PANEL_NAMES_WITHOUT_LAYER_OPS = [''] * NUM_FIELD_HEIGHT_PAIRS

for k in range(NUM_FIELD_HEIGHT_PAIRS):
    PANEL_NAMES_WITHOUT_LAYER_OPS[k] = '{0:s}\nat {1:.2f} km AGL'.format(
        radar_plotting._field_name_to_plotting_units(FIELD_NAME_BY_PAIR[k]),
        HEIGHT_BY_PAIR_M_AGL[k] * METRES_TO_KM
    )


class RadarPlottingTests(unittest.TestCase):
    """Each method is a unit test for radar_plotting.py."""

    def test_layer_ops_to_field_and_panel_names(self):
        """Ensures correct output from layer_ops_to_field_and_panel_names."""

        these_field_names, these_panel_names = (
            radar_plotting.layer_ops_to_field_and_panel_names(
                LIST_OF_LAYER_OPERATION_DICTS)
        )

        self.assertTrue(these_field_names == FIELD_NAMES_WITH_LAYER_OPS)
        self.assertTrue(these_panel_names == PANEL_NAMES_WITH_LAYER_OPS)

    def test_radar_fields_and_heights_to_panel_names(self):
        """Ensures correctness of radar_fields_and_heights_to_panel_names."""

        these_panel_names = (
            radar_plotting.radar_fields_and_heights_to_panel_names(
                field_names=FIELD_NAME_BY_PAIR,
                heights_m_agl=HEIGHT_BY_PAIR_M_AGL)
        )

        self.assertTrue(these_panel_names == PANEL_NAMES_WITHOUT_LAYER_OPS)


if __name__ == '__main__':
    unittest.main()
