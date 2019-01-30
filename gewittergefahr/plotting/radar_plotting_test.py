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

for k in range(NUM_LAYER_OPERATIONS):
    LIST_OF_LAYER_OPERATION_DICTS[k] = {
        input_examples.RADAR_FIELD_KEY: THESE_FIELD_NAMES[k],
        input_examples.OPERATION_NAME_KEY: LAYER_OPERATION_NAMES[k],
        input_examples.MIN_HEIGHT_KEY: MIN_HEIGHTS_M_AGL[k],
        input_examples.MAX_HEIGHT_KEY: MAX_HEIGHTS_M_AGL[k]
    }

PANEL_NAMES_WITH_OPS_SANS_UNITS = [
    'Reflectivity\nMIN from 1-3 km AGL', 'Reflectivity\nMEAN from 1-3 km AGL',
    'Reflectivity\nMAX from 1-3 km AGL', 'Spectrum width\nMIN from 1-3 km AGL',
    'Spectrum width\nMEAN from 1-3 km AGL',
    'Spectrum width\nMAX from 1-3 km AGL',
    'Vorticity\nMIN from 2-4 km AGL', 'Vorticity\nMEAN from 2-4 km AGL',
    'Vorticity\nMAX from 2-4 km AGL', 'Vorticity\nMIN from 5-8 km AGL',
    'Vorticity\nMEAN from 5-8 km AGL', 'Vorticity\nMAX from 5-8 km AGL'
]

PANEL_NAMES_WITH_OPS_WITH_UNITS = [
    'Reflectivity (dBZ)\nMIN from 1-3 km AGL',
    'Reflectivity (dBZ)\nMEAN from 1-3 km AGL',
    'Reflectivity (dBZ)\nMAX from 1-3 km AGL',
    'Spectrum width (m s$^{-1}$)\nMIN from 1-3 km AGL',
    'Spectrum width (m s$^{-1}$)\nMEAN from 1-3 km AGL',
    'Spectrum width (m s$^{-1}$)\nMAX from 1-3 km AGL',
    'Vorticity (ks$^{-1}$)\nMIN from 2-4 km AGL',
    'Vorticity (ks$^{-1}$)\nMEAN from 2-4 km AGL',
    'Vorticity (ks$^{-1}$)\nMAX from 2-4 km AGL',
    'Vorticity (ks$^{-1}$)\nMIN from 5-8 km AGL',
    'Vorticity (ks$^{-1}$)\nMEAN from 5-8 km AGL',
    'Vorticity (ks$^{-1}$)\nMAX from 5-8 km AGL'
]

# The following constants are used to test
# radar_fields_and_heights_to_panel_names.
FIELD_NAME_BY_PAIR = [
    radar_utils.REFL_NAME, radar_utils.SPECTRUM_WIDTH_NAME,
    radar_utils.VORTICITY_NAME, radar_utils.REFL_NAME
]
HEIGHT_BY_PAIR_M_AGL = numpy.array([1000, 3000, 2000, 10000])

PANEL_NAMES_SANS_OPS_WITH_UNITS = [
    'Reflectivity (dBZ)\nat 1.00 km AGL',
    'Spectrum width (m s$^{-1}$)\nat 3.00 km AGL',
    'Vorticity (ks$^{-1}$)\nat 2.00 km AGL',
    'Reflectivity (dBZ)\nat 10.00 km AGL'
]

PANEL_NAMES_SANS_OPS_SANS_UNITS = [
    'Reflectivity\nat 1.00 km AGL',
    'Spectrum width\nat 3.00 km AGL',
    'Vorticity\nat 2.00 km AGL',
    'Reflectivity\nat 10.00 km AGL'
]


class RadarPlottingTests(unittest.TestCase):
    """Each method is a unit test for radar_plotting.py."""

    def test_layer_ops_to_field_and_panel_names_with_units(self):
        """Ensures correct output from layer_ops_to_field_and_panel_names.

        In this case, expecting panel names with units.
        """

        these_field_names, these_panel_names = (
            radar_plotting.layer_ops_to_field_and_panel_names(
                list_of_layer_operation_dicts=LIST_OF_LAYER_OPERATION_DICTS,
                include_units=True)
        )

        self.assertTrue(these_field_names == FIELD_NAMES_WITH_LAYER_OPS)
        self.assertTrue(these_panel_names == PANEL_NAMES_WITH_OPS_WITH_UNITS)

    def test_layer_ops_to_field_and_panel_names_sans_units(self):
        """Ensures correct output from layer_ops_to_field_and_panel_names.

        In this case, expecting panel names without units.
        """

        these_field_names, these_panel_names = (
            radar_plotting.layer_ops_to_field_and_panel_names(
                list_of_layer_operation_dicts=LIST_OF_LAYER_OPERATION_DICTS,
                include_units=False)
        )

        self.assertTrue(these_field_names == FIELD_NAMES_WITH_LAYER_OPS)
        self.assertTrue(these_panel_names == PANEL_NAMES_WITH_OPS_SANS_UNITS)

    def test_radar_fields_and_heights_to_panel_names_with_units(self):
        """Ensures correctness of radar_fields_and_heights_to_panel_names.

        In this case, expecting panel names with units.
        """

        these_panel_names = (
            radar_plotting.radar_fields_and_heights_to_panel_names(
                field_names=FIELD_NAME_BY_PAIR,
                heights_m_agl=HEIGHT_BY_PAIR_M_AGL, include_units=True)
        )

        self.assertTrue(these_panel_names == PANEL_NAMES_SANS_OPS_WITH_UNITS)

    def test_radar_fields_and_heights_to_panel_names_sans_units(self):
        """Ensures correctness of radar_fields_and_heights_to_panel_names.

        In this case, expecting panel names without units.
        """

        these_panel_names = (
            radar_plotting.radar_fields_and_heights_to_panel_names(
                field_names=FIELD_NAME_BY_PAIR,
                heights_m_agl=HEIGHT_BY_PAIR_M_AGL, include_units=False)
        )

        self.assertTrue(these_panel_names == PANEL_NAMES_SANS_OPS_SANS_UNITS)


if __name__ == '__main__':
    unittest.main()
