"""Unit tests for radar_utils.py."""

import unittest
import numpy
from gewittergefahr.gg_utils import radar_utils

TOLERANCE = 1e-6
FAKE_SOURCE_NAME = 'poop'

LL_SHEAR_NAME = radar_utils.LOW_LEVEL_SHEAR_NAME
LL_SHEAR_NAME_FAKE = 'poop'
LL_SHEAR_NAME_MYRORSS = radar_utils.FIELD_NAME_TO_MYRORSS_DICT[
    radar_utils.LOW_LEVEL_SHEAR_NAME
]
LL_SHEAR_NAME_MRMS = radar_utils.FIELD_NAME_TO_MRMS_DICT[
    radar_utils.LOW_LEVEL_SHEAR_NAME
]

DIFFERENTIAL_REFL_NAME_GRIDRAD = radar_utils.FIELD_NAME_TO_GRIDRAD_DICT[
    radar_utils.DIFFERENTIAL_REFL_NAME
]

# The following constants are used to test get_valid_heights.
SHEAR_HEIGHTS_M_ASL = numpy.array([radar_utils.SHEAR_HEIGHT_M_ASL])
NON_SHEAR_HEIGHTS_MYRORSS_M_ASL = numpy.array(
    [radar_utils.DEFAULT_HEIGHT_MYRORSS_M_ASL]
)
NON_SHEAR_HEIGHTS_MRMS_M_ASL = numpy.array(
    [radar_utils.DEFAULT_HEIGHT_MRMS_M_ASL]
)

# The following constants are used to test rowcol_to_latlng and
# latlng_to_rowcol.
NW_GRID_POINT_LAT_DEG = 55.
NW_GRID_POINT_LNG_DEG = 230.
LAT_SPACING_DEG = 0.01
LNG_SPACING_DEG = 0.01

GRID_ROW_INDICES = numpy.linspace(0, 10, num=11, dtype=int)
GRID_COLUMN_INDICES = numpy.linspace(0, 10, num=11, dtype=int)
GRID_POINT_LATITUDES_DEG = numpy.array([
    55, 54.99, 54.98, 54.97, 54.96, 54.95, 54.94, 54.93, 54.92, 54.91, 54.9
])
GRID_POINT_LONGITUDES_DEG = numpy.array([
    230, 230.01, 230.02, 230.03, 230.04, 230.05, 230.06, 230.07, 230.08, 230.09,
    230.1
])

# The following constants are used to test get_center_of_grid.
NUM_GRID_ROWS = 3501
NUM_GRID_COLUMNS = 7001
GRID_CENTER_LATITUDE_DEG = 37.5
GRID_CENTER_LONGITUDE_DEG = 265.

# These constants are used to test get_echo_top.
GRID_POINT_HEIGHTS_M_ASL = numpy.array([1000, 2000, 3000], dtype=float)

THIS_REFL_MATRIX_1KM_DBZ = numpy.array([
    [0, 10],
    [20, 0],
    [0, numpy.nan]
])
THIS_REFL_MATRIX_2KM_DBZ = numpy.array([
    [20, 30],
    [40, 10],
    [50, 50]
], dtype=float)
THIS_REFL_MATRIX_3KM_DBZ = numpy.array([
    [40, 50],
    [60, 20],
    [20, 20]
], dtype=float)

REFLECTIVITY_MATRIX_DBZ = numpy.stack((
    THIS_REFL_MATRIX_1KM_DBZ, THIS_REFL_MATRIX_2KM_DBZ, THIS_REFL_MATRIX_3KM_DBZ
), axis=0)

CRIT_REFL_FOR_ECHO_TOPS_DBZ = 40.
ECHO_TOP_MATRIX_M_ASL = numpy.array([
    [3000, 3200],
    [3333.333333, numpy.nan],
    [2333.333333, 2333.333333]
])


class RadarUtilsTests(unittest.TestCase):
    """Each method is a unit test for radar_utils.py."""

    def test_field_name_orig_to_new_myrorss(self):
        """Ensures correct output from field_name_orig_to_new.

        In this case, original field name is from MYRORSS.
        """

        this_field_name = radar_utils.field_name_orig_to_new(
            field_name_orig=LL_SHEAR_NAME_MYRORSS,
            data_source_name=radar_utils.MYRORSS_SOURCE_ID)

        self.assertTrue(this_field_name == LL_SHEAR_NAME)

    def test_field_name_orig_to_new_mrms(self):
        """Ensures correct output from field_name_orig_to_new.

        In this case, original field name is from MRMS.
        """

        this_field_name = radar_utils.field_name_orig_to_new(
            field_name_orig=LL_SHEAR_NAME_MRMS,
            data_source_name=radar_utils.MRMS_SOURCE_ID)

        self.assertTrue(this_field_name == LL_SHEAR_NAME)

    def test_field_name_orig_to_new_gridrad(self):
        """Ensures correct output from field_name_orig_to_new.

        In this case, original field name is from GridRad.
        """

        this_field_name = radar_utils.field_name_orig_to_new(
            field_name_orig=DIFFERENTIAL_REFL_NAME_GRIDRAD,
            data_source_name=radar_utils.GRIDRAD_SOURCE_ID)

        self.assertTrue(this_field_name == radar_utils.DIFFERENTIAL_REFL_NAME)

    def test_field_name_new_to_orig_myrorss_valid(self):
        """Ensures correct output from field_name_new_to_orig.

        In this case, field name is being converted to MYRORSS format.
        """

        this_field_name_myrorss = radar_utils.field_name_new_to_orig(
            field_name=LL_SHEAR_NAME,
            data_source_name=radar_utils.MYRORSS_SOURCE_ID)

        self.assertTrue(this_field_name_myrorss == LL_SHEAR_NAME_MYRORSS)

    def test_field_name_new_to_orig_myrorss_invalid(self):
        """Ensures correct output from field_name_new_to_orig.

        In this case, trying to convert field name to MYRORSS format, but field
        does not exist in MYRORSS.
        """

        with self.assertRaises(KeyError):
            radar_utils.field_name_new_to_orig(
                field_name=radar_utils.DIFFERENTIAL_REFL_NAME,
                data_source_name=radar_utils.MYRORSS_SOURCE_ID)

    def test_field_name_new_to_orig_mrms_valid(self):
        """Ensures correct output from field_name_new_to_orig.

        In this case, field name is being converted to MRMS format.
        """

        this_field_name_mrms = radar_utils.field_name_new_to_orig(
            field_name=LL_SHEAR_NAME,
            data_source_name=radar_utils.MRMS_SOURCE_ID)

        self.assertTrue(this_field_name_mrms == LL_SHEAR_NAME_MRMS)

    def test_field_name_new_to_orig_mrms_invalid(self):
        """Ensures correct output from field_name_new_to_orig.

        In this case, trying to convert field name to MRMS format, but field
        does not exist in MRMS.
        """

        with self.assertRaises(KeyError):
            radar_utils.field_name_new_to_orig(
                field_name=radar_utils.DIFFERENTIAL_REFL_NAME,
                data_source_name=radar_utils.MRMS_SOURCE_ID)

    def test_field_name_new_to_orig_gridrad_valid(self):
        """Ensures correct output from field_name_new_to_orig.

        In this case, field name is being converted to GridRad format.
        """

        this_field_name_gridrad = radar_utils.field_name_new_to_orig(
            field_name=radar_utils.DIFFERENTIAL_REFL_NAME,
            data_source_name=radar_utils.GRIDRAD_SOURCE_ID)

        self.assertTrue(
            this_field_name_gridrad == DIFFERENTIAL_REFL_NAME_GRIDRAD
        )

    def test_field_name_new_to_orig_gridrad_invalid(self):
        """Ensures correct output from field_name_new_to_orig.

        In this case, trying to convert field name to GridRad format, but field
        does not exist in GridRad.
        """

        with self.assertRaises(KeyError):
            radar_utils.field_name_new_to_orig(
                field_name=LL_SHEAR_NAME,
                data_source_name=radar_utils.GRIDRAD_SOURCE_ID)

    def test_field_name_to_echo_top_refl_good(self):
        """Ensures correct output from field_name_to_echo_top_refl.

        In this case the input is actually an echo-top field.
        """

        this_critical_refl_dbz = radar_utils.field_name_to_echo_top_refl(
            radar_utils.ECHO_TOP_25DBZ_NAME)

        self.assertTrue(numpy.isclose(
            this_critical_refl_dbz, 25, atol=TOLERANCE
        ))

    def test_field_name_to_echo_top_refl_bad(self):
        """Ensures correct output from field_name_to_echo_top_refl.

        In this case the input is *not* an echo-top field.
        """

        with self.assertRaises(ValueError):
            radar_utils.field_name_to_echo_top_refl(radar_utils.VIL_NAME)

    def test_get_valid_heights_myrorss_shear(self):
        """Ensures correct output from get_valid_heights.

        In this case, data source and field are azimuthal shear in MYRORSS.
        """

        these_valid_heights_m_asl = radar_utils.get_valid_heights(
            data_source=radar_utils.MYRORSS_SOURCE_ID,
            field_name=radar_utils.MID_LEVEL_SHEAR_NAME)

        self.assertTrue(numpy.array_equal(
            these_valid_heights_m_asl, SHEAR_HEIGHTS_M_ASL
        ))

    def test_get_valid_heights_mrms_shear(self):
        """Ensures correct output from get_valid_heights.

        In this case, data source and field are azimuthal shear in MRMS.
        """

        these_valid_heights_m_asl = radar_utils.get_valid_heights(
            data_source=radar_utils.MRMS_SOURCE_ID,
            field_name=radar_utils.MID_LEVEL_SHEAR_NAME)

        self.assertTrue(numpy.array_equal(
            these_valid_heights_m_asl, SHEAR_HEIGHTS_M_ASL
        ))

    def test_get_valid_heights_myrorss_non_shear(self):
        """Ensures correct output from get_valid_heights.

        In this case, data source is MYRORSS; field is *not* azimuthal shear.
        """

        these_valid_heights_m_asl = radar_utils.get_valid_heights(
            data_source=radar_utils.MYRORSS_SOURCE_ID,
            field_name=radar_utils.REFL_M10CELSIUS_NAME)

        self.assertTrue(numpy.array_equal(
            these_valid_heights_m_asl, NON_SHEAR_HEIGHTS_MYRORSS_M_ASL
        ))

    def test_get_valid_heights_mrms_non_shear(self):
        """Ensures correct output from get_valid_heights.

        In this case, data source is MRMS and field is *not* azimuthal shear.
        """

        these_valid_heights_m_asl = radar_utils.get_valid_heights(
            data_source=radar_utils.MRMS_SOURCE_ID,
            field_name=radar_utils.REFL_M10CELSIUS_NAME)

        self.assertTrue(numpy.array_equal(
            these_valid_heights_m_asl, NON_SHEAR_HEIGHTS_MRMS_M_ASL
        ))

    def test_get_valid_heights_reflectivity(self):
        """Ensures correct output from get_valid_heights.

        In this case, field is reflectivity (defined at many height levels).
        """

        these_valid_heights_m_asl = radar_utils.get_valid_heights(
            data_source=radar_utils.MRMS_SOURCE_ID,
            field_name=radar_utils.REFL_NAME)

        self.assertTrue(len(these_valid_heights_m_asl) > 1)

    def test_get_valid_heights_gridrad(self):
        """Ensures correct output from get_valid_heights.

        In this case, data source is GridRad.
        """

        these_valid_heights_m_asl = radar_utils.get_valid_heights(
            data_source=radar_utils.GRIDRAD_SOURCE_ID)

        self.assertTrue(len(these_valid_heights_m_asl) > 1)

    def test_get_valid_heights_storm_id(self):
        """Ensures correct output from get_valid_heights.

        In this case, field is storm ID, which is not defined at certain
        heights.
        """

        with self.assertRaises(ValueError):
            radar_utils.get_valid_heights(
                data_source=radar_utils.MYRORSS_SOURCE_ID,
                field_name=radar_utils.STORM_ID_NAME)

    def test_check_heights_myrorss_shear_valid(self):
        """Ensures correct output from check_heights.

        In this case, data source and field are azimuthal shear in MYRORSS;
        input height is valid.
        """

        these_heights_m_asl = numpy.array([radar_utils.SHEAR_HEIGHT_M_ASL])

        radar_utils.check_heights(
            data_source=radar_utils.MYRORSS_SOURCE_ID,
            heights_m_asl=these_heights_m_asl,
            field_name=radar_utils.MID_LEVEL_SHEAR_NAME)

    def test_check_heights_myrorss_shear_invalid(self):
        """Ensures correct output from check_heights.

        In this case, data source and field are azimuthal shear in MYRORSS;
        input height is *invalid*.
        """

        these_heights_m_asl = numpy.array([
            radar_utils.SHEAR_HEIGHT_M_ASL + 1.
        ])

        with self.assertRaises(ValueError):
            radar_utils.check_heights(
                data_source=radar_utils.MYRORSS_SOURCE_ID,
                heights_m_asl=these_heights_m_asl,
                field_name=radar_utils.MID_LEVEL_SHEAR_NAME)

    def test_check_heights_myrorss_non_shear_valid(self):
        """Ensures correct output from check_heights.

        In this case, data source is MYRORSS; field is *not* azimuthal shear;
        and input height is valid.
        """

        these_heights_m_asl = numpy.array([
            radar_utils.DEFAULT_HEIGHT_MYRORSS_M_ASL
        ])

        radar_utils.check_heights(
            data_source=radar_utils.MYRORSS_SOURCE_ID,
            heights_m_asl=these_heights_m_asl,
            field_name=radar_utils.REFL_M10CELSIUS_NAME)

    def test_check_heights_myrorss_non_shear_invalid(self):
        """Ensures correct output from check_heights.

        In this case, data source is MYRORSS; field is *not* azimuthal shear;
        and input height is *invalid*.
        """

        these_heights_m_asl = numpy.array([
            radar_utils.DEFAULT_HEIGHT_MYRORSS_M_ASL + 1.
        ])

        with self.assertRaises(ValueError):
            radar_utils.check_heights(
                data_source=radar_utils.MYRORSS_SOURCE_ID,
                heights_m_asl=these_heights_m_asl,
                field_name=radar_utils.REFL_M10CELSIUS_NAME)

    def test_check_heights_myrorss_refl_valid(self):
        """Ensures correct output from check_heights.

        In this case, data source is MYRORSS; field is reflectivity (defined at
        many height levels); and input height is valid.
        """

        radar_utils.check_heights(
            data_source=radar_utils.MYRORSS_SOURCE_ID,
            heights_m_asl=numpy.array([250.]), field_name=radar_utils.REFL_NAME
        )

    def test_check_heights_myrorss_refl_invalid(self):
        """Ensures correct output from check_heights.

        In this case, data source is MYRORSS; field is reflectivity (defined at
        many height levels); and input height is *invalid*.
        """

        with self.assertRaises(ValueError):
            radar_utils.check_heights(
                data_source=radar_utils.MYRORSS_SOURCE_ID,
                heights_m_asl=numpy.array([251.]),
                field_name=radar_utils.REFL_NAME
            )

    def test_check_heights_gridrad_valid(self):
        """Ensures correct output from check_heights.

        In this case, data source is GridRad and input height is valid.
        """

        radar_utils.check_heights(
            data_source=radar_utils.GRIDRAD_SOURCE_ID,
            heights_m_asl=numpy.array([500.])
        )

    def test_check_heights_gridrad_invalid(self):
        """Ensures correct output from check_heights.

        In this case, data source is GridRad and input height is *invalid*.
        """

        with self.assertRaises(ValueError):
            radar_utils.check_heights(
                data_source=radar_utils.GRIDRAD_SOURCE_ID,
                heights_m_asl=numpy.array([501.])
            )

    def test_rowcol_to_latlng(self):
        """Ensures correct output from rowcol_to_latlng."""

        these_latitudes_deg, these_longitudes_deg = (
            radar_utils.rowcol_to_latlng(
                GRID_ROW_INDICES, GRID_COLUMN_INDICES,
                nw_grid_point_lat_deg=NW_GRID_POINT_LAT_DEG,
                nw_grid_point_lng_deg=NW_GRID_POINT_LNG_DEG,
                lat_spacing_deg=LAT_SPACING_DEG,
                lng_spacing_deg=LNG_SPACING_DEG)
        )

        self.assertTrue(numpy.allclose(
            these_latitudes_deg, GRID_POINT_LATITUDES_DEG, atol=TOLERANCE
        ))
        self.assertTrue(numpy.allclose(
            these_longitudes_deg, GRID_POINT_LONGITUDES_DEG, atol=TOLERANCE
        ))

    def test_latlng_to_rowcol(self):
        """Ensures correct output from latlng_to_rowcol."""

        these_row_indices, these_column_indices = radar_utils.latlng_to_rowcol(
            GRID_POINT_LATITUDES_DEG, GRID_POINT_LONGITUDES_DEG,
            nw_grid_point_lat_deg=NW_GRID_POINT_LAT_DEG,
            nw_grid_point_lng_deg=NW_GRID_POINT_LNG_DEG,
            lat_spacing_deg=LAT_SPACING_DEG, lng_spacing_deg=LNG_SPACING_DEG)

        self.assertTrue(numpy.allclose(
            these_row_indices, GRID_ROW_INDICES, atol=TOLERANCE
        ))
        self.assertTrue(numpy.allclose(
            these_column_indices, GRID_COLUMN_INDICES, atol=TOLERANCE
        ))

    def test_get_center_of_grid(self):
        """Ensures correct output from get_center_of_grid."""

        this_center_lat_deg, this_center_lng_deg = (
            radar_utils.get_center_of_grid(
                nw_grid_point_lat_deg=NW_GRID_POINT_LAT_DEG,
                nw_grid_point_lng_deg=NW_GRID_POINT_LNG_DEG,
                lat_spacing_deg=LAT_SPACING_DEG,
                lng_spacing_deg=LNG_SPACING_DEG,
                num_grid_rows=NUM_GRID_ROWS, num_grid_columns=NUM_GRID_COLUMNS)
        )

        self.assertTrue(numpy.isclose(
            this_center_lat_deg, GRID_CENTER_LATITUDE_DEG, atol=TOLERANCE
        ))
        self.assertTrue(numpy.isclose(
            this_center_lng_deg, GRID_CENTER_LONGITUDE_DEG, atol=TOLERANCE
        ))

    def test_get_echo_top_single_column(self):
        """Ensures correct output from get_echo_top_single_column."""

        this_num_rows = THIS_REFL_MATRIX_1KM_DBZ.shape[0]
        this_num_columns = THIS_REFL_MATRIX_1KM_DBZ.shape[1]

        this_echo_top_matrix_m_asl = numpy.full(
            (this_num_rows, this_num_columns), numpy.nan
        )

        for i in range(this_num_rows):
            for j in range(this_num_columns):
                this_echo_top_matrix_m_asl[i, j] = (
                    radar_utils.get_echo_top_single_column(
                        reflectivities_dbz=REFLECTIVITY_MATRIX_DBZ[:, i, j],
                        heights_m_asl=GRID_POINT_HEIGHTS_M_ASL,
                        critical_reflectivity_dbz=CRIT_REFL_FOR_ECHO_TOPS_DBZ,
                        check_args=True)
                )

        self.assertTrue(numpy.allclose(
            this_echo_top_matrix_m_asl, ECHO_TOP_MATRIX_M_ASL, atol=TOLERANCE,
            equal_nan=True
        ))


if __name__ == '__main__':
    unittest.main()
