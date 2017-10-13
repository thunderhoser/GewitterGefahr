"""Unit tests for nwp_soundings.py."""

import copy
import unittest
import numpy
import pandas
from gewittergefahr.gg_utils import rap_model_utils
from gewittergefahr.gg_utils import nwp_soundings

TOLERANCE = 1e-6

# The following constants are used to test _check_model_name.
FAKE_MODEL_NAME = 'ecmwf'
MINIMUM_PRESSURE_MB = 950.

# The following constants are used to test _fahrenheit_to_kelvins.
TEMPERATURES_FAHRENHEIT = numpy.array(
    [-130., -76., -40., -4., 32., 68., numpy.nan])
TEMPERATURES_KELVINS = numpy.array(
    [183.15, 213.15, 233.15, 253.15, 273.15, 293.15, numpy.nan])

# The following constants are used to test _get_sounding_columns.
RAP_SOUNDING_COLUMNS = [
    'temperature_kelvins_950mb', 'temperature_kelvins_975mb',
    'temperature_kelvins_1000mb', rap_model_utils.LOWEST_TEMPERATURE_COLUMN,
    'relative_humidity_950mb', 'relative_humidity_975mb',
    'relative_humidity_1000mb', rap_model_utils.LOWEST_RH_COLUMN,
    'geopotential_height_metres_950mb', 'geopotential_height_metres_975mb',
    'geopotential_height_metres_1000mb', rap_model_utils.LOWEST_GPH_COLUMN,
    'u_wind_m_s01_950mb', 'u_wind_m_s01_975mb', 'u_wind_m_s01_1000mb',
    rap_model_utils.LOWEST_U_WIND_COLUMN,
    'v_wind_m_s01_950mb', 'v_wind_m_s01_975mb', 'v_wind_m_s01_1000mb',
    rap_model_utils.LOWEST_V_WIND_COLUMN]

RAP_SOUNDING_COLUMNS_ORIG = [
    'TMP:950 mb', 'TMP:975 mb', 'TMP:1000 mb',
    rap_model_utils.LOWEST_TEMPERATURE_COLUMN_ORIG,
    'RH:950 mb', 'RH:975 mb', 'RH:1000 mb',
    rap_model_utils.LOWEST_RH_COLUMN_ORIG,
    'HGT:950 mb', 'HGT:975 mb', 'HGT:1000 mb',
    rap_model_utils.LOWEST_GPH_COLUMN_ORIG,
    'UGRD:950 mb', 'UGRD:975 mb', 'UGRD:1000 mb',
    rap_model_utils.LOWEST_U_WIND_COLUMN_ORIG,
    'VGRD:950 mb', 'VGRD:975 mb', 'VGRD:1000 mb',
    rap_model_utils.LOWEST_V_WIND_COLUMN_ORIG]

NARR_SOUNDING_COLUMNS = [
    'temperature_kelvins_950mb', 'temperature_kelvins_975mb',
    'temperature_kelvins_1000mb', 'specific_humidity_950mb',
    'specific_humidity_975mb', 'specific_humidity_1000mb',
    'geopotential_height_metres_950mb', 'geopotential_height_metres_975mb',
    'geopotential_height_metres_1000mb', 'u_wind_m_s01_950mb',
    'u_wind_m_s01_975mb', 'u_wind_m_s01_1000mb', 'v_wind_m_s01_950mb',
    'v_wind_m_s01_975mb', 'v_wind_m_s01_1000mb']

NARR_SOUNDING_COLUMNS_ORIG = [
    'TMP:950 mb', 'TMP:975 mb', 'TMP:1000 mb',
    'SPFH:950 mb', 'SPFH:975 mb', 'SPFH:1000 mb',
    'HGT:950 mb', 'HGT:975 mb', 'HGT:1000 mb',
    'UGRD:950 mb', 'UGRD:975 mb', 'UGRD:1000 mb',
    'VGRD:950 mb', 'VGRD:975 mb', 'VGRD:1000 mb']

# The following constants are used to test _remove_subsurface_sounding_data.
SURFACE_ROW = 3
THESE_PRESSURES_MB = numpy.array(
    [1000., 965., 962., 936.87, 936.8669, 925., 904.95, 889., 873.9, 877.])
THESE_HEIGHTS_M_ASL = numpy.array(
    [34., 350., 377.51, 610., 610.0001, 722., 914., 1069.78, 1219., 1188.26])
THESE_TEMPERATURES_DEG_C = numpy.array(
    [30., 27.8, 27.4, 25.51, 25.51, 24.6, 23.05, 21.8, 22.02, 22.2])
THESE_DEWPOINTS_DEG_C = numpy.array(
    [24.5, 23.8, 22.8, 21.72, 21.72, 21.2, 20.43, 19.8, 16.98, 17.3])
THESE_U_WINDS_KT = numpy.array(
    [-8.45, -11.5, -13.2, -17.2, -17.2, -17., -14.36, -11.65, -4.36, -8.34])
THESE_V_WINDS_KT = numpy.array(
    [18.13, 19.92, 22.86, 24.57, 24.57, 29.44, 39.47, 43.47, 49.81, 47.27])

THIS_SOUNDING_DICT = {
    nwp_soundings.PRESSURE_COLUMN_FOR_SOUNDING_INDICES: THESE_PRESSURES_MB,
    nwp_soundings.HEIGHT_COLUMN_FOR_SOUNDING_INDICES: THESE_HEIGHTS_M_ASL,
    nwp_soundings.TEMPERATURE_COLUMN_FOR_SOUNDING_INDICES:
        THESE_TEMPERATURES_DEG_C,
    nwp_soundings.DEWPOINT_COLUMN_FOR_SOUNDING_INDICES: THESE_DEWPOINTS_DEG_C,
    nwp_soundings.U_WIND_COLUMN_FOR_SOUNDING_INDICES: THESE_U_WINDS_KT,
    nwp_soundings.V_WIND_COLUMN_FOR_SOUNDING_INDICES: THESE_V_WINDS_KT
}
SOUNDING_TABLE_ORIG = pandas.DataFrame.from_dict(THIS_SOUNDING_DICT)

SENTINEL_VALUE = nwp_soundings.SENTINEL_VALUE_FOR_SOUNDING_INDICES
THESE_TEMPERATURES_DEG_C = numpy.array(
    [SENTINEL_VALUE, SENTINEL_VALUE, SENTINEL_VALUE,
     25.51, 25.51, 24.6, 23.05, 21.8, 22.02, 22.2])
THESE_DEWPOINTS_DEG_C = numpy.array(
    [SENTINEL_VALUE, SENTINEL_VALUE, SENTINEL_VALUE,
     21.72, 21.72, 21.2, 20.43, 19.8, 16.98, 17.3])
THESE_U_WINDS_KT = numpy.array(
    [SENTINEL_VALUE, SENTINEL_VALUE, SENTINEL_VALUE,
     -17.2, -17.2, -17., -14.36, -11.65, -4.36, -8.34])
THESE_V_WINDS_KT = numpy.array(
    [SENTINEL_VALUE, SENTINEL_VALUE, SENTINEL_VALUE,
     24.57, 24.57, 29.44, 39.47, 43.47, 49.81, 47.27])

THIS_SOUNDING_DICT = {
    nwp_soundings.PRESSURE_COLUMN_FOR_SOUNDING_INDICES: THESE_PRESSURES_MB,
    nwp_soundings.HEIGHT_COLUMN_FOR_SOUNDING_INDICES: THESE_HEIGHTS_M_ASL,
    nwp_soundings.TEMPERATURE_COLUMN_FOR_SOUNDING_INDICES:
        THESE_TEMPERATURES_DEG_C,
    nwp_soundings.DEWPOINT_COLUMN_FOR_SOUNDING_INDICES: THESE_DEWPOINTS_DEG_C,
    nwp_soundings.U_WIND_COLUMN_FOR_SOUNDING_INDICES: THESE_U_WINDS_KT,
    nwp_soundings.V_WIND_COLUMN_FOR_SOUNDING_INDICES: THESE_V_WINDS_KT
}
SOUNDING_TABLE_WITH_SENTINELS = pandas.DataFrame.from_dict(THIS_SOUNDING_DICT)

# The following constants are used to test _sort_sounding_table_by_height.
THESE_PRESSURES_MB = numpy.array(
    [1000., 965., 962., 936.87, 936.8669, 925., 904.95, 889., 877., 873.9])
THESE_HEIGHTS_M_ASL = numpy.array(
    [34., 350., 377.51, 610., 610.0001, 722., 914., 1069.78, 1188.26, 1219.])
THESE_TEMPERATURES_DEG_C = numpy.array(
    [SENTINEL_VALUE, SENTINEL_VALUE, SENTINEL_VALUE,
     25.51, 25.51, 24.6, 23.05, 21.8, 22.2, 22.02])
THESE_DEWPOINTS_DEG_C = numpy.array(
    [SENTINEL_VALUE, SENTINEL_VALUE, SENTINEL_VALUE,
     21.72, 21.72, 21.2, 20.43, 19.8, 17.3, 16.98])
THESE_U_WINDS_KT = numpy.array(
    [SENTINEL_VALUE, SENTINEL_VALUE, SENTINEL_VALUE,
     -17.2, -17.2, -17., -14.36, -11.65, -8.34, -4.36])
THESE_V_WINDS_KT = numpy.array(
    [SENTINEL_VALUE, SENTINEL_VALUE, SENTINEL_VALUE,
     24.57, 24.57, 29.44, 39.47, 43.47, 47.27, 49.81])

THIS_SOUNDING_DICT = {
    nwp_soundings.PRESSURE_COLUMN_FOR_SOUNDING_INDICES: THESE_PRESSURES_MB,
    nwp_soundings.HEIGHT_COLUMN_FOR_SOUNDING_INDICES: THESE_HEIGHTS_M_ASL,
    nwp_soundings.TEMPERATURE_COLUMN_FOR_SOUNDING_INDICES:
        THESE_TEMPERATURES_DEG_C,
    nwp_soundings.DEWPOINT_COLUMN_FOR_SOUNDING_INDICES: THESE_DEWPOINTS_DEG_C,
    nwp_soundings.U_WIND_COLUMN_FOR_SOUNDING_INDICES: THESE_U_WINDS_KT,
    nwp_soundings.V_WIND_COLUMN_FOR_SOUNDING_INDICES: THESE_V_WINDS_KT
}
SOUNDING_TABLE_WITH_SENTINELS_SORTED = pandas.DataFrame.from_dict(
    THIS_SOUNDING_DICT)

# The following constants are used to test _remove_redundant_sounding_data.
THESE_PRESSURES_MB = numpy.array(
    [1000., 965., 962., 936.87, 936.8669, 925., 904.95, 889., 877., 873.9])
THESE_HEIGHTS_M_ASL = numpy.array(
    [34., 350., 377.51, 610., 610.0001, 722., 914., 1069.78, 1188.26, 1219.])
THESE_TEMPERATURES_DEG_C = numpy.array(
    [SENTINEL_VALUE, SENTINEL_VALUE, SENTINEL_VALUE, SENTINEL_VALUE,
     25.51, 24.6, 23.05, 21.8, 22.2, 22.02])
THESE_DEWPOINTS_DEG_C = numpy.array(
    [SENTINEL_VALUE, SENTINEL_VALUE, SENTINEL_VALUE, SENTINEL_VALUE,
     21.72, 21.2, 20.43, 19.8, 17.3, 16.98])
THESE_U_WINDS_KT = numpy.array(
    [SENTINEL_VALUE, SENTINEL_VALUE, SENTINEL_VALUE, SENTINEL_VALUE,
     -17.2, -17., -14.36, -11.65, -8.34, -4.36])
THESE_V_WINDS_KT = numpy.array(
    [SENTINEL_VALUE, SENTINEL_VALUE, SENTINEL_VALUE, SENTINEL_VALUE,
     24.57, 29.44, 39.47, 43.47, 47.27, 49.81])

THIS_SOUNDING_DICT = {
    nwp_soundings.PRESSURE_COLUMN_FOR_SOUNDING_INDICES: THESE_PRESSURES_MB,
    nwp_soundings.HEIGHT_COLUMN_FOR_SOUNDING_INDICES: THESE_HEIGHTS_M_ASL,
    nwp_soundings.TEMPERATURE_COLUMN_FOR_SOUNDING_INDICES:
        THESE_TEMPERATURES_DEG_C,
    nwp_soundings.DEWPOINT_COLUMN_FOR_SOUNDING_INDICES: THESE_DEWPOINTS_DEG_C,
    nwp_soundings.U_WIND_COLUMN_FOR_SOUNDING_INDICES: THESE_U_WINDS_KT,
    nwp_soundings.V_WIND_COLUMN_FOR_SOUNDING_INDICES: THESE_V_WINDS_KT
}
SOUNDING_TABLE_NO_REDUNDANT = pandas.DataFrame.from_dict(THIS_SOUNDING_DICT)

# The following constants are used to test _remove_subsurface_sounding_data.
THESE_PRESSURES_MB = numpy.array(
    [1000., 936.87, 936.8669, 925., 904.95, 889., 877., 873.9])
THESE_HEIGHTS_M_ASL = numpy.array(
    [34., 610., 610.0001, 722., 914., 1069.78, 1188.26, 1219.])
THESE_TEMPERATURES_DEG_C = numpy.array(
    [SENTINEL_VALUE, SENTINEL_VALUE, 25.51, 24.6, 23.05, 21.8, 22.2, 22.02])
THESE_DEWPOINTS_DEG_C = numpy.array(
    [SENTINEL_VALUE, SENTINEL_VALUE, 21.72, 21.2, 20.43, 19.8, 17.3, 16.98])
THESE_U_WINDS_KT = numpy.array(
    [SENTINEL_VALUE, SENTINEL_VALUE, -17.2, -17., -14.36, -11.65, -8.34, -4.36])
THESE_V_WINDS_KT = numpy.array(
    [SENTINEL_VALUE, SENTINEL_VALUE, 24.57, 29.44, 39.47, 43.47, 47.27, 49.81])

THIS_SOUNDING_DICT = {
    nwp_soundings.PRESSURE_COLUMN_FOR_SOUNDING_INDICES: THESE_PRESSURES_MB,
    nwp_soundings.HEIGHT_COLUMN_FOR_SOUNDING_INDICES: THESE_HEIGHTS_M_ASL,
    nwp_soundings.TEMPERATURE_COLUMN_FOR_SOUNDING_INDICES:
        THESE_TEMPERATURES_DEG_C,
    nwp_soundings.DEWPOINT_COLUMN_FOR_SOUNDING_INDICES: THESE_DEWPOINTS_DEG_C,
    nwp_soundings.U_WIND_COLUMN_FOR_SOUNDING_INDICES: THESE_U_WINDS_KT,
    nwp_soundings.V_WIND_COLUMN_FOR_SOUNDING_INDICES: THESE_V_WINDS_KT
}
SOUNDING_TABLE_NO_SUBSURFACE = pandas.DataFrame.from_dict(THIS_SOUNDING_DICT)

# The following constants are used to test _split_vector_column.
VECTOR_COLUMN = 'vector'
VECTOR_COLUMN_AS_TABLE = pandas.DataFrame.from_dict(
    {VECTOR_COLUMN: numpy.full(8, numpy.nan)})

THIS_NESTED_ARRAY = VECTOR_COLUMN_AS_TABLE[[
    VECTOR_COLUMN, VECTOR_COLUMN]].values.tolist()
VECTOR_COLUMN_AS_TABLE = VECTOR_COLUMN_AS_TABLE.assign(
    **{VECTOR_COLUMN: THIS_NESTED_ARRAY})

VECTOR_COLUMN_AS_TABLE[VECTOR_COLUMN].values[0] = numpy.array([1., 0.])
VECTOR_COLUMN_AS_TABLE[VECTOR_COLUMN].values[1] = numpy.array([1., 1.])
VECTOR_COLUMN_AS_TABLE[VECTOR_COLUMN].values[2] = numpy.array([0., 1.])
VECTOR_COLUMN_AS_TABLE[VECTOR_COLUMN].values[3] = numpy.array([-1., 1.])
VECTOR_COLUMN_AS_TABLE[VECTOR_COLUMN].values[4] = numpy.array([-1., 0.])
VECTOR_COLUMN_AS_TABLE[VECTOR_COLUMN].values[5] = numpy.array([-1., -1.])
VECTOR_COLUMN_AS_TABLE[VECTOR_COLUMN].values[6] = numpy.array([0., -1.])
VECTOR_COLUMN_AS_TABLE[VECTOR_COLUMN].values[7] = numpy.array([1., -1.])

ROOT2 = numpy.sqrt(2.)
HALF_ROOT2 = ROOT2 / 2

THESE_X_COMPONENTS = numpy.array([1., 1., 0., -1., -1., -1., 0., 1.])
THESE_Y_COMPONENTS = numpy.array([0., 1., 1., 1., 0., -1., -1., -1.])
THESE_MAGNITUDES = numpy.array([1., ROOT2, 1., ROOT2, 1., ROOT2, 1., ROOT2])
THESE_COSINES = numpy.array(
    [1., HALF_ROOT2, 0., -HALF_ROOT2, -1., -HALF_ROOT2, 0., HALF_ROOT2])
THESE_SINES = numpy.array(
    [0., HALF_ROOT2, 1., HALF_ROOT2, 0., -HALF_ROOT2, -1., -HALF_ROOT2])

X_COMPONENT_COLUMN = 'vector_x'
Y_COMPONENT_COLUMN = 'vector_y'
MAGNITUDE_COLUMN = 'vector_magnitude'
COSINE_COLUMN = 'vector_cos'
SINE_COLUMN = 'vector_sin'

VECTOR_COMPONENT_DICT = {
    X_COMPONENT_COLUMN: THESE_X_COMPONENTS,
    Y_COMPONENT_COLUMN: THESE_Y_COMPONENTS, MAGNITUDE_COLUMN: THESE_MAGNITUDES,
    COSINE_COLUMN: THESE_COSINES, SINE_COLUMN: THESE_SINES}
VECTOR_COMPONENT_TABLE = pandas.DataFrame.from_dict(VECTOR_COMPONENT_DICT)

# The following constants are used to test convert_sounding_indices.
CONVECTIVE_TEMPERATURE_NAME = 'convective_temperature_kelvins'
WIND_MEAN_0TO1KM_NAME = 'wind_mean_0to1km_agl_m_s01'
RELATIVE_HUMIDITY_SURFACE_NAME = 'relative_humidity_surface'
DERECHO_COMPOSITE_NAME = 'derecho_composite_param'

SOUNDING_INDEX_METAFILE_NAME = 'sounding_index_metadata.csv'
SOUNDING_INDEX_METADATA_TABLE = (
    nwp_soundings.read_metadata_for_sounding_indices(
        SOUNDING_INDEX_METAFILE_NAME))

SOUNDING_INDEX_NAMES = SOUNDING_INDEX_METADATA_TABLE[
    nwp_soundings.SOUNDING_INDEX_NAME_COLUMN].values
SHARPPY_NAMES = SOUNDING_INDEX_METADATA_TABLE[
    nwp_soundings.SHARPPY_NAME_COLUMN].values

THESE_FLAGS = [s == CONVECTIVE_TEMPERATURE_NAME for s in SOUNDING_INDEX_NAMES]
CONVECTIVE_TEMPERATURE_NAME_SHARPPY = SHARPPY_NAMES[
    numpy.where(THESE_FLAGS)[0][0]]

THESE_FLAGS = [s == WIND_MEAN_0TO1KM_NAME for s in SOUNDING_INDEX_NAMES]
WIND_MEAN_0TO1KM_NAME_SHARPPY = SHARPPY_NAMES[numpy.where(THESE_FLAGS)[0][0]]

THESE_FLAGS = [s == RELATIVE_HUMIDITY_SURFACE_NAME for s in
               SOUNDING_INDEX_NAMES]
RH_SURFACE_NAME_SHARPPY = SHARPPY_NAMES[numpy.where(THESE_FLAGS)[0][0]]

THESE_FLAGS = [s == DERECHO_COMPOSITE_NAME for s in SOUNDING_INDEX_NAMES]
DERECHO_COMPOSITE_NAME_SHARPPY = SHARPPY_NAMES[numpy.where(THESE_FLAGS)[0][0]]

CONVECTIVE_TEMPERATURES_FAHRENHEIT = copy.deepcopy(TEMPERATURES_FAHRENHEIT)
U_WINDS_0TO1KM_AGL_KT = numpy.array([10., 10., 0., -10., -10., -10., 0.])
V_WINDS_0TO1KM_AGL_KT = numpy.array([0., 10., 10., 10., 0., -10., -10.])
RH_SURFACE_PERCENTAGES = numpy.array([40., 50., 60., 70., 80., 90., 100.])
DERECHO_COMPOSITE_PARAMS = numpy.array([0., 5., 10., 15., 20., 25., 30.])

SOUNDING_INDEX_DICT_SHARPPY = {
    CONVECTIVE_TEMPERATURE_NAME_SHARPPY: CONVECTIVE_TEMPERATURES_FAHRENHEIT,
    RH_SURFACE_NAME_SHARPPY: RH_SURFACE_PERCENTAGES,
    DERECHO_COMPOSITE_NAME_SHARPPY: DERECHO_COMPOSITE_PARAMS}
SOUNDING_INDEX_TABLE_SHARPPY = pandas.DataFrame.from_dict(
    SOUNDING_INDEX_DICT_SHARPPY)

THIS_NESTED_ARRAY = SOUNDING_INDEX_TABLE_SHARPPY[[
    RH_SURFACE_NAME_SHARPPY, RH_SURFACE_NAME_SHARPPY]].values.tolist()
SOUNDING_INDEX_TABLE_SHARPPY = SOUNDING_INDEX_TABLE_SHARPPY.assign(
    **{WIND_MEAN_0TO1KM_NAME_SHARPPY: THIS_NESTED_ARRAY})

for s in range(len(U_WINDS_0TO1KM_AGL_KT)):
    SOUNDING_INDEX_TABLE_SHARPPY[WIND_MEAN_0TO1KM_NAME_SHARPPY].values[s] = (
        numpy.array([U_WINDS_0TO1KM_AGL_KT[s], V_WINDS_0TO1KM_AGL_KT[s]]))

TEN_KT_IN_MPS = 5.144444
ROOT200_KT_IN_MPS = 7.275343
U_WIND_0TO1KM_NAME = 'wind_mean_0to1km_agl_m_s01_x'
V_WIND_0TO1KM_NAME = 'wind_mean_0to1km_agl_m_s01_y'
WIND_SPEED_0TO1KM_NAME = 'wind_mean_0to1km_agl_m_s01_magnitude'
WIND_COS_0TO1KM_NAME = 'wind_mean_0to1km_agl_m_s01_cos'
WIND_SIN_0TO1KM_NAME = 'wind_mean_0to1km_agl_m_s01_sin'

CONVECTIVE_TEMPERATURES_KELVINS = copy.deepcopy(TEMPERATURES_KELVINS)
U_WINDS_M_S01 = numpy.array(
    [TEN_KT_IN_MPS, TEN_KT_IN_MPS, 0., -TEN_KT_IN_MPS, -TEN_KT_IN_MPS,
     -TEN_KT_IN_MPS, 0.])
V_WINDS_M_S01 = numpy.array(
    [0., TEN_KT_IN_MPS, TEN_KT_IN_MPS, TEN_KT_IN_MPS, 0., -TEN_KT_IN_MPS,
     -TEN_KT_IN_MPS])
WIND_SPEEDS_M_S01 = numpy.array(
    [TEN_KT_IN_MPS, ROOT200_KT_IN_MPS, TEN_KT_IN_MPS, ROOT200_KT_IN_MPS,
     TEN_KT_IN_MPS, ROOT200_KT_IN_MPS, TEN_KT_IN_MPS])
WIND_COSINES = numpy.array(
    [1., HALF_ROOT2, 0., -HALF_ROOT2, -1., -HALF_ROOT2, 0.])
WIND_SINES = numpy.array([0., HALF_ROOT2, 1., HALF_ROOT2, 0., -HALF_ROOT2, -1.])
RH_SURFACE_UNITLESS = numpy.array([0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.])

SOUNDING_INDEX_DICT = {
    CONVECTIVE_TEMPERATURE_NAME: CONVECTIVE_TEMPERATURES_KELVINS,
    U_WIND_0TO1KM_NAME: U_WINDS_M_S01, V_WIND_0TO1KM_NAME: V_WINDS_M_S01,
    WIND_SPEED_0TO1KM_NAME: WIND_SPEEDS_M_S01,
    WIND_COS_0TO1KM_NAME: WIND_COSINES, WIND_SIN_0TO1KM_NAME: WIND_SINES,
    RELATIVE_HUMIDITY_SURFACE_NAME: RH_SURFACE_UNITLESS,
    DERECHO_COMPOSITE_NAME: DERECHO_COMPOSITE_PARAMS}
SOUNDING_INDEX_TABLE = pandas.DataFrame.from_dict(SOUNDING_INDEX_DICT)


class NwpSoundingsTests(unittest.TestCase):
    """Each method is a unit test for nwp_soundings.py."""

    def test_check_model_name_rap(self):
        """Ensures correct output from _check_model_name.

        In this case, model is RAP.
        """

        nwp_soundings._check_model_name(nwp_soundings.RAP_MODEL_NAME)

    def test_check_model_name_narr(self):
        """Ensures correct output from _check_model_name.

        In this case, model is NARR.
        """

        nwp_soundings._check_model_name(nwp_soundings.NARR_MODEL_NAME)

    def test_check_model_name_fake(self):
        """Ensures correct output from _check_model_name.

        In this case, model name is not recognized.
        """

        with self.assertRaises(ValueError):
            nwp_soundings._check_model_name(FAKE_MODEL_NAME)

    def test_fahrenheit_to_kelvins(self):
        """Ensures correct output from _fahrenheit_to_kelvins."""

        these_temperatures_kelvins = nwp_soundings._fahrenheit_to_kelvins(
            TEMPERATURES_FAHRENHEIT)
        self.assertTrue(numpy.allclose(
            these_temperatures_kelvins, TEMPERATURES_KELVINS, atol=TOLERANCE,
            equal_nan=True))

    def test_get_sounding_columns_rap(self):
        """Ensures correct output from _get_sounding_columns.

        In this case, model is RAP.
        """

        (these_sounding_columns,
         these_sounding_columns_orig) = nwp_soundings._get_sounding_columns(
             nwp_soundings.RAP_MODEL_NAME,
             minimum_pressure_mb=MINIMUM_PRESSURE_MB)

        self.assertTrue(these_sounding_columns == RAP_SOUNDING_COLUMNS)
        self.assertTrue(
            these_sounding_columns_orig == RAP_SOUNDING_COLUMNS_ORIG)

    def test_get_sounding_columns_narr(self):
        """Ensures correct output from _get_sounding_columns.

        In this case, model is NARR.
        """

        (these_sounding_columns,
         these_sounding_columns_orig) = nwp_soundings._get_sounding_columns(
             nwp_soundings.NARR_MODEL_NAME,
             minimum_pressure_mb=MINIMUM_PRESSURE_MB)

        self.assertTrue(these_sounding_columns == NARR_SOUNDING_COLUMNS)
        self.assertTrue(
            these_sounding_columns_orig == NARR_SOUNDING_COLUMNS_ORIG)

    def test_remove_subsurface_sounding_data_keep_rows(self):
        """Ensures correct output from _remove_subsurface_sounding_data.

        In this case, remove_subsurface_rows = False.
        """

        this_sounding_table = nwp_soundings._remove_subsurface_sounding_data(
            SOUNDING_TABLE_ORIG, surface_row=SURFACE_ROW,
            remove_subsurface_rows=False)
        self.assertTrue(
            this_sounding_table.equals(SOUNDING_TABLE_WITH_SENTINELS))

    def test_sort_sounding_table_by_height(self):
        """Ensures correct output from _sort_sounding_table_by_height."""

        this_input_table = copy.deepcopy(SOUNDING_TABLE_WITH_SENTINELS_SORTED)
        this_sounding_table = nwp_soundings._sort_sounding_table_by_height(
            this_input_table)

        self.assertTrue(
            this_sounding_table.equals(SOUNDING_TABLE_WITH_SENTINELS_SORTED))

    def test_remove_redundant_sounding_data(self):
        """Ensures correct output from _remove_redundant_sounding_data."""

        this_input_table = copy.deepcopy(SOUNDING_TABLE_WITH_SENTINELS_SORTED)
        this_sounding_table = nwp_soundings._remove_redundant_sounding_data(
            this_input_table, SURFACE_ROW)

        self.assertTrue(this_sounding_table.equals(SOUNDING_TABLE_NO_REDUNDANT))

    def test_remove_subsurface_sounding_data_remove_rows(self):
        """Ensures correct output from _remove_subsurface_sounding_data.

        In this case, remove_subsurface_rows = True.
        """

        this_sounding_table = nwp_soundings._remove_subsurface_sounding_data(
            SOUNDING_TABLE_NO_REDUNDANT, surface_row=SURFACE_ROW,
            remove_subsurface_rows=True)
        self.assertTrue(
            this_sounding_table.equals(SOUNDING_TABLE_NO_SUBSURFACE))

    def test_split_vector_column(self):
        """Ensures correct output from _split_vector_column."""

        this_component_dict = nwp_soundings._split_vector_column(
            VECTOR_COLUMN_AS_TABLE)
        this_component_table = pandas.DataFrame.from_dict(this_component_dict)

        self.assertTrue(set(list(this_component_table)) ==
                        set(list(VECTOR_COMPONENT_TABLE)))

        for this_column in list(this_component_table):
            self.assertTrue(numpy.allclose(
                this_component_table[this_column].values,
                VECTOR_COMPONENT_TABLE[this_column].values, atol=TOLERANCE))

    def test_convert_sounding_indices(self):
        """Ensures correct output from convert_sounding_indices."""

        this_sounding_index_table = nwp_soundings.convert_sounding_indices(
            SOUNDING_INDEX_TABLE_SHARPPY, SOUNDING_INDEX_METADATA_TABLE)

        self.assertTrue(set(list(this_sounding_index_table)) ==
                        set(list(SOUNDING_INDEX_TABLE)))

        for this_column in list(this_sounding_index_table):
            self.assertTrue(numpy.allclose(
                this_sounding_index_table[this_column].values,
                SOUNDING_INDEX_TABLE[this_column].values, atol=TOLERANCE,
                equal_nan=True))


if __name__ == '__main__':
    unittest.main()
