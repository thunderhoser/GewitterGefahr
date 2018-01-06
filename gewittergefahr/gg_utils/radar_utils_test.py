"""Unit tests for radar_utils.py."""

import unittest
import numpy
from gewittergefahr.gg_utils import radar_utils

TOLERANCE = 1e-6
FAKE_DATA_SOURCE = 'poop'

LL_SHEAR_NAME_MYRORSS = radar_utils.LOW_LEVEL_SHEAR_NAME_MYRORSS
LL_SHEAR_NAME_MRMS = radar_utils.LOW_LEVEL_SHEAR_NAME_MRMS
LL_SHEAR_NAME_NEW = radar_utils.LOW_LEVEL_SHEAR_NAME
LL_SHEAR_NAME_NEW_FAKE = 'poop'

# The following constants are used to test get_valid_heights_for_field.
SHEAR_HEIGHTS_M_ASL = numpy.array([radar_utils.SHEAR_HEIGHT_M_ASL])
NON_SHEAR_HEIGHTS_MYRORSS_M_ASL = numpy.array(
    [radar_utils.DEFAULT_HEIGHT_MYRORSS_M_ASL])
NON_SHEAR_HEIGHTS_MRMS_M_ASL = numpy.array(
    [radar_utils.DEFAULT_HEIGHT_MRMS_M_ASL])

# The following constants are used to test check_reflectivity_heights.
REFL_HEIGHTS_M_ASL = numpy.array([500., 1000., 2000., 3000., 5000., 10000.])
REFL_HEIGHTS_ONE_BAD_M_ASL = numpy.array(
    [500., 1000., 2000., 3456., 5000., 10000.])

# The following constants are used to test field_and_height_arrays_to_dict.
UNIQUE_FIELD_NAMES = [
    'echo_top_50dbz_km', 'low_level_shear_s01', 'reflectivity_dbz',
    'reflectivity_column_max_dbz']
UNIQUE_REFLECTIVITY_HEIGHTS_M_ASL = numpy.array(
    [250., 500., 750., 1000., 5000., 10000., 20000.])

FIELD_TO_HEIGHTS_DICT_MYRORSS_M_ASL = {
    'echo_top_50dbz_km': numpy.array([250]),
    'low_level_shear_s01': numpy.array([radar_utils.SHEAR_HEIGHT_M_ASL]),
    'reflectivity_dbz': UNIQUE_REFLECTIVITY_HEIGHTS_M_ASL,
    'reflectivity_column_max_dbz': numpy.array([250])}
FIELD_TO_HEIGHTS_DICT_MRMS_M_ASL = {
    'echo_top_50dbz_km': numpy.array([500]),
    'low_level_shear_s01': numpy.array([radar_utils.SHEAR_HEIGHT_M_ASL]),
    'reflectivity_dbz': UNIQUE_REFLECTIVITY_HEIGHTS_M_ASL,
    'reflectivity_column_max_dbz': numpy.array([500])}

# The following constants are used to test unique_fields_and_heights_to_pairs.
FIELD_NAME_BY_PAIR = [
    'echo_top_50dbz_km', 'low_level_shear_s01', 'reflectivity_dbz',
    'reflectivity_dbz', 'reflectivity_dbz', 'reflectivity_dbz',
    'reflectivity_dbz', 'reflectivity_dbz', 'reflectivity_dbz',
    'reflectivity_column_max_dbz']
HEIGHT_BY_PAIR_MYRORSS_M_ASL = numpy.array(
    [radar_utils.DEFAULT_HEIGHT_MYRORSS_M_ASL, radar_utils.SHEAR_HEIGHT_M_ASL,
     250., 500., 750., 1000., 5000., 10000., 20000.,
     radar_utils.DEFAULT_HEIGHT_MYRORSS_M_ASL])
HEIGHT_BY_PAIR_MRMS_M_ASL = numpy.array(
    [radar_utils.DEFAULT_HEIGHT_MRMS_M_ASL, radar_utils.SHEAR_HEIGHT_M_ASL,
     250., 500., 750., 1000., 5000., 10000., 20000.,
     radar_utils.DEFAULT_HEIGHT_MRMS_M_ASL])

# The following constants are used to test rowcol_to_latlng and
# latlng_to_rowcol.
NW_GRID_POINT_LAT_DEG = 55.
NW_GRID_POINT_LNG_DEG = 230.
LAT_SPACING_DEG = 0.01
LNG_SPACING_DEG = 0.01

GRID_ROW_INDICES = numpy.linspace(0, 10, num=11, dtype=int)
GRID_COLUMN_INDICES = numpy.linspace(0, 10, num=11, dtype=int)
GRID_POINT_LATITUDES_DEG = numpy.array(
    [55., 54.99, 54.98, 54.97, 54.96, 54.95, 54.94, 54.93, 54.92, 54.91, 54.9])
GRID_POINT_LONGITUDES_DEG = numpy.array(
    [230., 230.01, 230.02, 230.03, 230.04, 230.05, 230.06, 230.07, 230.08,
     230.09, 230.1])

# The following constants are used to test get_center_of_grid.
NUM_GRID_ROWS = 3501
NUM_GRID_COLUMNS = 7001
GRID_CENTER_LATITUDE_DEG = 37.5
GRID_CENTER_LONGITUDE_DEG = 265.

# These constants are used to test get_echo_top.
GRID_POINT_HEIGHTS_M_ASL = numpy.array([1000., 2000., 3000.])

THIS_REFL_MATRIX_1KM_DBZ = numpy.array(
    [[0., 10.], [20., 0., ], [0., numpy.nan]])
THIS_REFL_MATRIX_2KM_DBZ = numpy.array(
    [[20., 30.], [40., 10., ], [50., 50.]])
THIS_REFL_MATRIX_3KM_DBZ = numpy.array(
    [[40., 50.], [60., 20., ], [20., 20.]])
REFLECTIVITY_MATRIX_DBZ = numpy.stack(
    (THIS_REFL_MATRIX_1KM_DBZ, THIS_REFL_MATRIX_2KM_DBZ,
     THIS_REFL_MATRIX_3KM_DBZ), axis=0)

CRIT_REFL_FOR_ECHO_TOPS_DBZ = 40.
ECHO_TOP_MATRIX_M_ASL = numpy.array(
    [[3000., 3200.], [3333.333333, numpy.nan], [2333.333333, 2333.333333]])


class RadarUtilsTests(unittest.TestCase):
    """Each method is a unit test for radar_utils.py."""

    def test_check_data_source_myrorss(self):
        """Ensures correct output from check_data_source.

        In this case, data source is MYRORSS.
        """

        radar_utils.check_data_source(radar_utils.MYRORSS_SOURCE_ID)

    def test_check_data_source_mrms(self):
        """Ensures correct output from check_data_source.

        In this case, data source is MRMS.
        """

        radar_utils.check_data_source(radar_utils.MRMS_SOURCE_ID)

    def test_check_data_source_gridrad(self):
        """Ensures correct output from check_data_source.

        In this case, data source is GridRad.
        """

        radar_utils.check_data_source(radar_utils.GRIDRAD_SOURCE_ID)

    def test_check_data_source_fake(self):
        """Ensures correct output from check_data_source.

        In this case, data source is fake.
        """

        with self.assertRaises(ValueError):
            radar_utils.check_data_source(FAKE_DATA_SOURCE)

    def test_check_field_name_valid(self):
        """Ensures correct output from check_field_name.

        In this case, field name is valid.
        """

        radar_utils.check_field_name(LL_SHEAR_NAME_NEW)

    def test_check_field_name_invalid(self):
        """Ensures correct output from check_field_name.

        In this case, field name is invalid.
        """

        with self.assertRaises(ValueError):
            radar_utils.check_field_name(LL_SHEAR_NAME_NEW_FAKE)

    def test_check_field_name_orig_myrorss_valid(self):
        """Ensures correct output from check_field_name_orig.

        In this case, input is valid field name from MYRORSS.
        """

        radar_utils.check_field_name_orig(
            LL_SHEAR_NAME_MYRORSS, data_source=radar_utils.MYRORSS_SOURCE_ID)

    def test_check_field_name_orig_myrorss_invalid(self):
        """Ensures correct output from check_field_name_orig.

        In this case, input is field name from MRMS (invalid for MYRORSS).
        """

        with self.assertRaises(ValueError):
            radar_utils.check_field_name_orig(
                LL_SHEAR_NAME_MRMS, data_source=radar_utils.MYRORSS_SOURCE_ID)

    def test_check_field_name_orig_mrms_valid(self):
        """Ensures correct output from check_field_name_orig.

        In this case, input is valid field name from MRMS.
        """

        radar_utils.check_field_name_orig(
            LL_SHEAR_NAME_MRMS, data_source=radar_utils.MRMS_SOURCE_ID)

    def test_check_field_name_orig_mrms_invalid(self):
        """Ensures correct output from check_field_name_orig.

        In this case, input is field name from MYRORSS (invalid for MRMS).
        """

        with self.assertRaises(ValueError):
            radar_utils.check_field_name_orig(
                LL_SHEAR_NAME_MYRORSS, data_source=radar_utils.MRMS_SOURCE_ID)

    def test_check_field_name_orig_gridrad_valid(self):
        """Ensures correct output from check_field_name_orig.

        In this case, input is valid field name from GridRad.
        """

        radar_utils.check_field_name_orig(
            radar_utils.DIFFERENTIAL_REFL_NAME_GRIDRAD,
            data_source=radar_utils.GRIDRAD_SOURCE_ID)

    def test_check_field_name_orig_gridrad_invalid(self):
        """Ensures correct output from check_field_name_orig.

        In this case, input is field name from MYRORSS (invalid for GridRad).
        """

        with self.assertRaises(ValueError):
            radar_utils.check_field_name_orig(
                LL_SHEAR_NAME_MYRORSS,
                data_source=radar_utils.GRIDRAD_SOURCE_ID)

    def test_field_name_orig_to_new_myrorss(self):
        """Ensures correct output from field_name_orig_to_new.

        In this case, original field name is from MYRORSS.
        """

        this_field_name = radar_utils.field_name_orig_to_new(
            LL_SHEAR_NAME_MYRORSS, data_source=radar_utils.MYRORSS_SOURCE_ID)
        self.assertTrue(this_field_name == LL_SHEAR_NAME_NEW)

    def test_field_name_orig_to_new_mrms(self):
        """Ensures correct output from field_name_orig_to_new.

        In this case, original field name is from MRMS.
        """

        this_field_name = radar_utils.field_name_orig_to_new(
            LL_SHEAR_NAME_MRMS, data_source=radar_utils.MRMS_SOURCE_ID)
        self.assertTrue(this_field_name == LL_SHEAR_NAME_NEW)

    def test_field_name_orig_to_new_gridrad(self):
        """Ensures correct output from field_name_orig_to_new.

        In this case, original field name is from GridRad.
        """

        this_field_name = radar_utils.field_name_orig_to_new(
            radar_utils.DIFFERENTIAL_REFL_NAME_GRIDRAD,
            data_source=radar_utils.GRIDRAD_SOURCE_ID)
        self.assertTrue(this_field_name == radar_utils.DIFFERENTIAL_REFL_NAME)

    def test_field_name_new_to_orig_myrorss_valid(self):
        """Ensures correct output from field_name_new_to_orig.

        In this case, field name is being converted to MYRORSS format.
        """

        this_field_name_myrorss = radar_utils.field_name_new_to_orig(
            LL_SHEAR_NAME_NEW, data_source=radar_utils.MYRORSS_SOURCE_ID)
        self.assertTrue(this_field_name_myrorss == LL_SHEAR_NAME_MYRORSS)

    def test_field_name_new_to_orig_myrorss_invalid(self):
        """Ensures correct output from field_name_new_to_orig.

        In this case, trying to convert field name to MYRORSS format, but field
        does not exist in MYRORSS.
        """

        with self.assertRaises(ValueError):
            radar_utils.field_name_new_to_orig(
                radar_utils.DIFFERENTIAL_REFL_NAME,
                data_source=radar_utils.MYRORSS_SOURCE_ID)

    def test_field_name_new_to_orig_mrms_valid(self):
        """Ensures correct output from field_name_new_to_orig.

        In this case, field name is being converted to MRMS format.
        """

        this_field_name_mrms = radar_utils.field_name_new_to_orig(
            LL_SHEAR_NAME_NEW, data_source=radar_utils.MRMS_SOURCE_ID)
        self.assertTrue(this_field_name_mrms == LL_SHEAR_NAME_MRMS)

    def test_field_name_new_to_orig_mrms_invalid(self):
        """Ensures correct output from field_name_new_to_orig.

        In this case, trying to convert field name to MRMS format, but field
        does not exist in MRMS.
        """

        with self.assertRaises(ValueError):
            radar_utils.field_name_new_to_orig(
                radar_utils.DIFFERENTIAL_REFL_NAME,
                data_source=radar_utils.MRMS_SOURCE_ID)

    def test_field_name_new_to_orig_gridrad_valid(self):
        """Ensures correct output from field_name_new_to_orig.

        In this case, field name is being converted to GridRad format.
        """

        this_field_name_gridrad = radar_utils.field_name_new_to_orig(
            radar_utils.DIFFERENTIAL_REFL_NAME,
            data_source=radar_utils.GRIDRAD_SOURCE_ID)
        self.assertTrue(this_field_name_gridrad ==
                        radar_utils.DIFFERENTIAL_REFL_NAME_GRIDRAD)

    def test_field_name_new_to_orig_gridrad_invalid(self):
        """Ensures correct output from field_name_new_to_orig.

        In this case, trying to convert field name to GridRad format, but field
        does not exist in GridRad.
        """

        with self.assertRaises(ValueError):
            radar_utils.field_name_new_to_orig(
                LL_SHEAR_NAME_NEW, data_source=radar_utils.GRIDRAD_SOURCE_ID)

    def test_get_valid_heights_for_field_shear_myrorss(self):
        """Ensures correct output from get_valid_heights_for_field.

        In this case, the field is azimuthal shear in MYRORSS.
        """

        these_valid_heights_m_asl = radar_utils.get_valid_heights_for_field(
            radar_utils.MID_LEVEL_SHEAR_NAME,
            data_source=radar_utils.MYRORSS_SOURCE_ID)
        self.assertTrue(
            numpy.array_equal(these_valid_heights_m_asl, SHEAR_HEIGHTS_M_ASL))

    def test_get_valid_heights_for_field_shear_mrms(self):
        """Ensures correct output from get_valid_heights_for_field.

        In this case, the field is azimuthal shear in MRMS.
        """

        these_valid_heights_m_asl = radar_utils.get_valid_heights_for_field(
            radar_utils.MID_LEVEL_SHEAR_NAME,
            data_source=radar_utils.MRMS_SOURCE_ID)
        self.assertTrue(
            numpy.array_equal(these_valid_heights_m_asl, SHEAR_HEIGHTS_M_ASL))

    def test_get_valid_heights_for_field_non_shear_myrorss(self):
        """Ensures correct output from get_valid_heights_for_field.

        In this case, the field is a non-shear field in MYRORSS.
        """

        these_valid_heights_m_asl = radar_utils.get_valid_heights_for_field(
            radar_utils.REFL_M10CELSIUS_NAME,
            data_source=radar_utils.MYRORSS_SOURCE_ID)
        self.assertTrue(numpy.array_equal(these_valid_heights_m_asl,
                                          NON_SHEAR_HEIGHTS_MYRORSS_M_ASL))

    def test_get_valid_heights_for_field_non_shear_mrms(self):
        """Ensures correct output from get_valid_heights_for_field.

        In this case, the field is a non-shear field in MRMS.
        """

        these_valid_heights_m_asl = radar_utils.get_valid_heights_for_field(
            radar_utils.REFL_M10CELSIUS_NAME,
            data_source=radar_utils.MRMS_SOURCE_ID)
        self.assertTrue(numpy.array_equal(these_valid_heights_m_asl,
                                          NON_SHEAR_HEIGHTS_MRMS_M_ASL))

    def test_get_valid_heights_for_field_reflectivity(self):
        """Ensures correct output from get_valid_heights_for_field.

        In this case, the field is simple reflectivity.
        """

        these_valid_heights_m_asl = radar_utils.get_valid_heights_for_field(
            radar_utils.REFL_NAME, data_source=radar_utils.MYRORSS_SOURCE_ID)
        self.assertTrue(len(these_valid_heights_m_asl) > 1)

    def test_get_valid_heights_for_field_gridrad(self):
        """Ensures correct output from get_valid_heights_for_field.

        In this case, desired data source is GridRad, for which
        get_valid_heights_for_field does not work.
        """

        with self.assertRaises(ValueError):
            radar_utils.get_valid_heights_for_field(
                radar_utils.REFL_NAME,
                data_source=radar_utils.GRIDRAD_SOURCE_ID)

    def test_get_valid_heights_for_field_storm_id(self):
        """Ensures correct output from get_valid_heights_for_field.

        In this case, desired field is storm ID, for which
        get_valid_heights_for_field does not work.
        """

        with self.assertRaises(ValueError):
            radar_utils.get_valid_heights_for_field(
                radar_utils.STORM_ID_NAME,
                data_source=radar_utils.MYRORSS_SOURCE_ID)

    def test_check_reflectivity_heights_valid(self):
        """Ensures correct output from check_reflectivity_heights.

        In this case, all heights are valid.
        """

        radar_utils.check_reflectivity_heights(
            REFL_HEIGHTS_M_ASL, data_source=radar_utils.MYRORSS_SOURCE_ID)

    def test_check_reflectivity_heights_invalid(self):
        """Ensures correct output from check_reflectivity_heights.

        In this case, at least one height is invalid.
        """

        with self.assertRaises(ValueError):
            radar_utils.check_reflectivity_heights(
                REFL_HEIGHTS_ONE_BAD_M_ASL,
                data_source=radar_utils.MYRORSS_SOURCE_ID)

    def test_field_and_height_arrays_to_dict_myrorss(self):
        """Ensures correct output from field_and_height_arrays_to_dict.

        In this case, data source is MYRORSS.
        """

        this_field_to_heights_dict_m_asl = (
            radar_utils.field_and_height_arrays_to_dict(
                field_names=UNIQUE_FIELD_NAMES,
                refl_heights_m_asl=UNIQUE_REFLECTIVITY_HEIGHTS_M_ASL,
                data_source=radar_utils.MYRORSS_SOURCE_ID))
        self.assertTrue(this_field_to_heights_dict_m_asl ==
                        FIELD_TO_HEIGHTS_DICT_MYRORSS_M_ASL)

    def test_field_and_height_arrays_to_dict_mrms(self):
        """Ensures correct output from field_and_height_arrays_to_dict.

        In this case, data source is MRMS.
        """

        this_field_to_heights_dict_m_asl = (
            radar_utils.field_and_height_arrays_to_dict(
                field_names=UNIQUE_FIELD_NAMES,
                refl_heights_m_asl=UNIQUE_REFLECTIVITY_HEIGHTS_M_ASL,
                data_source=radar_utils.MRMS_SOURCE_ID))
        self.assertTrue(this_field_to_heights_dict_m_asl ==
                        FIELD_TO_HEIGHTS_DICT_MRMS_M_ASL)

    def test_unique_fields_and_heights_to_pairs_myrorss(self):
        """Ensures correct output from unique_fields_and_heights_to_pairs.

        In this case, data source is MYRORSS.
        """

        this_field_name_by_pair, this_height_by_pair_m_asl = (
            radar_utils.unique_fields_and_heights_to_pairs(
                unique_field_names=UNIQUE_FIELD_NAMES,
                refl_heights_m_asl=UNIQUE_REFLECTIVITY_HEIGHTS_M_ASL,
                data_source=radar_utils.MYRORSS_SOURCE_ID))

        self.assertTrue(this_field_name_by_pair == FIELD_NAME_BY_PAIR)
        self.assertTrue(numpy.array_equal(
            this_height_by_pair_m_asl, HEIGHT_BY_PAIR_MYRORSS_M_ASL))

    def test_unique_fields_and_heights_to_pairs_mrms(self):
        """Ensures correct output from unique_fields_and_heights_to_pairs.

        In this case, data source is MRMS.
        """

        this_field_name_by_pair, this_height_by_pair_m_asl = (
            radar_utils.unique_fields_and_heights_to_pairs(
                unique_field_names=UNIQUE_FIELD_NAMES,
                refl_heights_m_asl=UNIQUE_REFLECTIVITY_HEIGHTS_M_ASL,
                data_source=radar_utils.MRMS_SOURCE_ID))

        self.assertTrue(this_field_name_by_pair == FIELD_NAME_BY_PAIR)
        self.assertTrue(numpy.array_equal(
            this_height_by_pair_m_asl, HEIGHT_BY_PAIR_MRMS_M_ASL))

    def test_rowcol_to_latlng(self):
        """Ensures correct output from rowcol_to_latlng."""

        (these_latitudes_deg,
         these_longitudes_deg) = radar_utils.rowcol_to_latlng(
             GRID_ROW_INDICES, GRID_COLUMN_INDICES,
             nw_grid_point_lat_deg=NW_GRID_POINT_LAT_DEG,
             nw_grid_point_lng_deg=NW_GRID_POINT_LNG_DEG,
             lat_spacing_deg=LAT_SPACING_DEG, lng_spacing_deg=LNG_SPACING_DEG)

        self.assertTrue(numpy.allclose(
            these_latitudes_deg, GRID_POINT_LATITUDES_DEG, atol=TOLERANCE))
        self.assertTrue(numpy.allclose(
            these_longitudes_deg, GRID_POINT_LONGITUDES_DEG, atol=TOLERANCE))

    def test_latlng_to_rowcol(self):
        """Ensures correct output from latlng_to_rowcol."""

        (these_row_indices, these_column_indices) = radar_utils.latlng_to_rowcol(
            GRID_POINT_LATITUDES_DEG, GRID_POINT_LONGITUDES_DEG,
            nw_grid_point_lat_deg=NW_GRID_POINT_LAT_DEG,
            nw_grid_point_lng_deg=NW_GRID_POINT_LNG_DEG,
            lat_spacing_deg=LAT_SPACING_DEG, lng_spacing_deg=LNG_SPACING_DEG)

        self.assertTrue(numpy.allclose(
            these_row_indices, GRID_ROW_INDICES, atol=TOLERANCE))
        self.assertTrue(numpy.allclose(
            these_column_indices, GRID_COLUMN_INDICES, atol=TOLERANCE))

    def test_get_center_of_grid(self):
        """Ensures correct output from get_center_of_grid."""

        (this_center_lat_deg,
         this_center_lng_deg) = radar_utils.get_center_of_grid(
             nw_grid_point_lat_deg=NW_GRID_POINT_LAT_DEG,
             nw_grid_point_lng_deg=NW_GRID_POINT_LNG_DEG,
             lat_spacing_deg=LAT_SPACING_DEG, lng_spacing_deg=LNG_SPACING_DEG,
             num_grid_rows=NUM_GRID_ROWS, num_grid_columns=NUM_GRID_COLUMNS)

        self.assertTrue(numpy.isclose(
            this_center_lat_deg, GRID_CENTER_LATITUDE_DEG, atol=TOLERANCE))
        self.assertTrue(numpy.isclose(
            this_center_lng_deg, GRID_CENTER_LONGITUDE_DEG, atol=TOLERANCE))

    def test_get_echo_top_single_column(self):
        """Ensures correct output from get_echo_top_single_column."""

        this_num_rows = THIS_REFL_MATRIX_1KM_DBZ.shape[0]
        this_num_columns = THIS_REFL_MATRIX_1KM_DBZ.shape[1]
        this_echo_top_matrix_m_asl = numpy.full(
            (this_num_rows, this_num_columns), numpy.nan)

        for i in range(this_num_rows):
            for j in range(this_num_columns):
                this_echo_top_matrix_m_asl[i, j] = (
                    radar_utils.get_echo_top_single_column(
                        reflectivities_dbz=REFLECTIVITY_MATRIX_DBZ[:, i, j],
                        heights_m_asl=GRID_POINT_HEIGHTS_M_ASL,
                        critical_reflectivity_dbz=CRIT_REFL_FOR_ECHO_TOPS_DBZ,
                        check_args=True))

        self.assertTrue(numpy.allclose(
            this_echo_top_matrix_m_asl, ECHO_TOP_MATRIX_M_ASL, atol=TOLERANCE,
            equal_nan=True))


if __name__ == '__main__':
    unittest.main()
