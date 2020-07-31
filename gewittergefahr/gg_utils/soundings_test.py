"""Unit tests for soundings.py."""

import copy
import unittest
import numpy
import pandas
from gewittergefahr.gg_utils import soundings
from gewittergefahr.gg_utils import nwp_model_utils
from gewittergefahr.gg_utils import storm_tracking_utils as tracking_utils
from gewittergefahr.gg_utils import temperature_conversions
from gewittergefahr.gg_utils import moisture_conversions

TOLERANCE = 1e-6
TOLERANCE_FOR_CONVERTED_VALUES = 1e-3

# The following constants are used to test _get_nwp_fields_for_sounding.
MINIMUM_PRESSURE_MB = 950.
MODEL_NAME = nwp_model_utils.RAP_MODEL_NAME

SURFACE_HEIGHT_NAME, SURFACE_HEIGHT_NAME_GRIB1 = (
    nwp_model_utils.get_lowest_height_name(MODEL_NAME)
)
SURFACE_TEMP_NAME, SURFACE_TEMP_NAME_GRIB1 = (
    nwp_model_utils.get_lowest_temperature_name(MODEL_NAME)
)
SURFACE_HUMIDITY_NAME, SURFACE_HUMIDITY_NAME_GRIB1 = (
    nwp_model_utils.get_lowest_humidity_name(MODEL_NAME)
)
SURFACE_U_WIND_NAME, SURFACE_U_WIND_NAME_GRIB1 = (
    nwp_model_utils.get_lowest_u_wind_name(MODEL_NAME)
)
SURFACE_V_WIND_NAME, SURFACE_V_WIND_NAME_GRIB1 = (
    nwp_model_utils.get_lowest_v_wind_name(MODEL_NAME)
)
SURFACE_PRESSURE_NAME, SURFACE_PRESSURE_NAME_GRIB1 = (
    nwp_model_utils.get_lowest_pressure_name(MODEL_NAME)
)

FIELD_NAMES_WITH_SURFACE = [
    'geopotential_height_metres_950mb', 'geopotential_height_metres_975mb',
    'geopotential_height_metres_1000mb', SURFACE_HEIGHT_NAME,
    'temperature_kelvins_950mb', 'temperature_kelvins_975mb',
    'temperature_kelvins_1000mb', SURFACE_TEMP_NAME,
    'relative_humidity_percent_950mb', 'relative_humidity_percent_975mb',
    'relative_humidity_percent_1000mb', SURFACE_HUMIDITY_NAME,
    'u_wind_m_s01_950mb', 'u_wind_m_s01_975mb', 'u_wind_m_s01_1000mb',
    SURFACE_U_WIND_NAME,
    'v_wind_m_s01_950mb', 'v_wind_m_s01_975mb', 'v_wind_m_s01_1000mb',
    SURFACE_V_WIND_NAME,
    SURFACE_PRESSURE_NAME
]

FIELD_NAMES_NO_SURFACE = [
    'geopotential_height_metres_950mb', 'geopotential_height_metres_975mb',
    'geopotential_height_metres_1000mb',
    'temperature_kelvins_950mb', 'temperature_kelvins_975mb',
    'temperature_kelvins_1000mb',
    'relative_humidity_percent_950mb', 'relative_humidity_percent_975mb',
    'relative_humidity_percent_1000mb',
    'u_wind_m_s01_950mb', 'u_wind_m_s01_975mb', 'u_wind_m_s01_1000mb',
    'v_wind_m_s01_950mb', 'v_wind_m_s01_975mb', 'v_wind_m_s01_1000mb'
]

FIELD_NAMES_WITH_SURFACE_GRIB1 = [
    'HGT:950 mb', 'HGT:975 mb', 'HGT:1000 mb', SURFACE_HEIGHT_NAME_GRIB1,
    'TMP:950 mb', 'TMP:975 mb', 'TMP:1000 mb', SURFACE_TEMP_NAME_GRIB1,
    'RH:950 mb', 'RH:975 mb', 'RH:1000 mb', SURFACE_HUMIDITY_NAME_GRIB1,
    'UGRD:950 mb', 'UGRD:975 mb', 'UGRD:1000 mb', SURFACE_U_WIND_NAME_GRIB1,
    'VGRD:950 mb', 'VGRD:975 mb', 'VGRD:1000 mb', SURFACE_V_WIND_NAME_GRIB1,
    SURFACE_PRESSURE_NAME_GRIB1
]

FIELD_NAMES_NO_SURFACE_GRIB1 = [
    'HGT:950 mb', 'HGT:975 mb', 'HGT:1000 mb',
    'TMP:950 mb', 'TMP:975 mb', 'TMP:1000 mb',
    'RH:950 mb', 'RH:975 mb', 'RH:1000 mb',
    'UGRD:950 mb', 'UGRD:975 mb', 'UGRD:1000 mb',
    'VGRD:950 mb', 'VGRD:975 mb', 'VGRD:1000 mb'
]

HEIGHT_NAMES_NO_SURFACE = [
    'geopotential_height_metres_950mb', 'geopotential_height_metres_975mb',
    'geopotential_height_metres_1000mb'
]
TEMPERATURE_NAMES_NO_SURFACE = [
    'temperature_kelvins_950mb', 'temperature_kelvins_975mb',
    'temperature_kelvins_1000mb'
]
HUMIDITY_NAMES_NO_SURFACE = [
    'relative_humidity_percent_950mb', 'relative_humidity_percent_975mb',
    'relative_humidity_percent_1000mb'
]
U_WIND_NAMES_NO_SURFACE = [
    'u_wind_m_s01_950mb', 'u_wind_m_s01_975mb', 'u_wind_m_s01_1000mb'
]
V_WIND_NAMES_NO_SURFACE = [
    'v_wind_m_s01_950mb', 'v_wind_m_s01_975mb', 'v_wind_m_s01_1000mb'
]
PRESSURE_LEVELS_NO_SURFACE_MB = numpy.array([950, 975, 1000], dtype=float)

THIS_DICT = {
    soundings.PRESSURE_LEVEL_KEY: numpy.concatenate((
        PRESSURE_LEVELS_NO_SURFACE_MB, numpy.array([numpy.nan])
    )),
    nwp_model_utils.HEIGHT_COLUMN_FOR_SOUNDINGS:
        HEIGHT_NAMES_NO_SURFACE + [SURFACE_HEIGHT_NAME],
    nwp_model_utils.TEMPERATURE_COLUMN_FOR_SOUNDINGS:
        TEMPERATURE_NAMES_NO_SURFACE + [SURFACE_TEMP_NAME],
    nwp_model_utils.RH_COLUMN_FOR_SOUNDINGS:
        HUMIDITY_NAMES_NO_SURFACE + [SURFACE_HUMIDITY_NAME],
    nwp_model_utils.U_WIND_COLUMN_FOR_SOUNDINGS:
        U_WIND_NAMES_NO_SURFACE + [SURFACE_U_WIND_NAME],
    nwp_model_utils.V_WIND_COLUMN_FOR_SOUNDINGS:
        V_WIND_NAMES_NO_SURFACE + [SURFACE_V_WIND_NAME]
}

FIELD_NAME_TABLE_WITH_SURFACE = pandas.DataFrame.from_dict(THIS_DICT)

THIS_DICT = {
    soundings.PRESSURE_LEVEL_KEY: PRESSURE_LEVELS_NO_SURFACE_MB,
    nwp_model_utils.HEIGHT_COLUMN_FOR_SOUNDINGS: HEIGHT_NAMES_NO_SURFACE,
    nwp_model_utils.TEMPERATURE_COLUMN_FOR_SOUNDINGS:
        TEMPERATURE_NAMES_NO_SURFACE,
    nwp_model_utils.RH_COLUMN_FOR_SOUNDINGS: HUMIDITY_NAMES_NO_SURFACE,
    nwp_model_utils.U_WIND_COLUMN_FOR_SOUNDINGS: U_WIND_NAMES_NO_SURFACE,
    nwp_model_utils.V_WIND_COLUMN_FOR_SOUNDINGS: V_WIND_NAMES_NO_SURFACE
}

FIELD_NAME_TABLE_NO_SURFACE = pandas.DataFrame.from_dict(THIS_DICT)

# The following constants are used to test _create_target_points_for_interp.
THESE_FULL_ID_STRINGS = ['A', 'B', 'C', 'A', 'B', 'C']
THESE_TIMES_UNIX_SEC = numpy.array([0, 0, 0, 1, 1, 1], dtype=int)
THESE_LATITUDES_DEG = numpy.array([50, 55, 60, 51, 56, 61], dtype=float)
THESE_LONGITUDES_DEG = numpy.array([250, 260, 270, 251, 261, 271], dtype=float)
THESE_EAST_VELOCITIES_M_S01 = numpy.full(6, 10000, dtype=float)
THESE_NORTH_VELOCITIES_M_S01 = numpy.full(6, 10000, dtype=float)

THIS_DICT = {
    tracking_utils.FULL_ID_COLUMN: THESE_FULL_ID_STRINGS,
    tracking_utils.VALID_TIME_COLUMN: THESE_TIMES_UNIX_SEC,
    tracking_utils.CENTROID_LATITUDE_COLUMN: THESE_LATITUDES_DEG,
    tracking_utils.CENTROID_LONGITUDE_COLUMN: THESE_LONGITUDES_DEG,
    tracking_utils.EAST_VELOCITY_COLUMN: THESE_EAST_VELOCITIES_M_S01,
    tracking_utils.NORTH_VELOCITY_COLUMN: THESE_NORTH_VELOCITIES_M_S01
}
DUMMY_STORM_OBJECT_TABLE = pandas.DataFrame.from_dict(THIS_DICT)

UNIQUE_LEAD_TIMES_SECONDS = numpy.array([0, 1], dtype=int)

THESE_FULL_ID_STRINGS = [
    'A', 'B', 'C', 'A', 'B', 'C', 'A', 'B', 'C', 'A', 'B', 'C'
]
THESE_INIT_TIMES_UNIX_SEC = numpy.array(
    [0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1], dtype=int
)
THESE_LATITUDES_DEG = numpy.array([
    50, 55, 60, 51, 56, 61, 50.08981978, 55.08972691, 60.08963404, 51.08980123,
    56.08970834, 61.08961544
], dtype=float)
THESE_LONGITUDES_DEG = numpy.array([
    250, 260, 270, 251, 261, 271, 250.13973873, 260.15661394, 270.17969769,
    251.14273048, 261.16064721, 271.18533962
], dtype=float)
THESE_VALID_TIMES_UNIX_SEC = numpy.array(
    [0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2], dtype=int
)
THESE_LEAD_TIMES_SECONDS = numpy.array(
    [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1], dtype=int
)
THESE_EAST_VELOCITIES_M_S01 = numpy.full(12, 10000, dtype=float)
THESE_NORTH_VELOCITIES_M_S01 = numpy.full(12, 10000, dtype=float)

THIS_DICT = {
    tracking_utils.FULL_ID_COLUMN: THESE_FULL_ID_STRINGS,
    soundings.INITIAL_TIME_COLUMN: THESE_INIT_TIMES_UNIX_SEC,
    tracking_utils.CENTROID_LATITUDE_COLUMN: THESE_LATITUDES_DEG,
    tracking_utils.CENTROID_LONGITUDE_COLUMN: THESE_LONGITUDES_DEG,
    soundings.FORECAST_TIME_COLUMN: THESE_VALID_TIMES_UNIX_SEC,
    soundings.LEAD_TIME_KEY: THESE_LEAD_TIMES_SECONDS,
    tracking_utils.EAST_VELOCITY_COLUMN: THESE_EAST_VELOCITIES_M_S01,
    tracking_utils.NORTH_VELOCITY_COLUMN: THESE_NORTH_VELOCITIES_M_S01
}
DUMMY_TARGET_POINT_TABLE = pandas.DataFrame.from_dict(THIS_DICT)

# The following constants are used to test _convert_interp_table_to_soundings.
THIS_MATRIX = numpy.array([
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
    [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30]
], dtype=float)

INTERP_TABLE_NO_SURFACE = pandas.DataFrame(THIS_MATRIX)

THESE_FULL_ID_STRINGS = ['a', 'b']
THESE_INIT_TIMES_UNIX_SEC = numpy.array([10, 10], dtype=int)
THESE_LEAD_TIMES_SECONDS = numpy.array([5, 5], dtype=int)

THIS_DICT = {
    tracking_utils.FULL_ID_COLUMN: THESE_FULL_ID_STRINGS,
    soundings.INITIAL_TIME_COLUMN: THESE_INIT_TIMES_UNIX_SEC,
    soundings.LEAD_TIME_KEY: THESE_LEAD_TIMES_SECONDS
}
TARGET_POINT_TABLE = pandas.DataFrame.from_dict(THIS_DICT)

for k in range(len(FIELD_NAMES_NO_SURFACE)):
    INTERP_TABLE_NO_SURFACE.rename(
        columns={k: FIELD_NAMES_NO_SURFACE[k]}, inplace=True
    )

THIS_FIRST_MATRIX = numpy.array([
    [0, 6, 3, 9, 12],
    [1, 7, 4, 10, 13],
    [2, 8, 5, 11, 14]
], dtype=int)

THIS_SECOND_MATRIX = numpy.array([
    [2, 14, 8, 20, 26],
    [4, 16, 10, 22, 28],
    [6, 18, 12, 24, 30]
], dtype=int)

THIS_SOUNDING_MATRIX = numpy.stack(
    (THIS_FIRST_MATRIX, THIS_SECOND_MATRIX), axis=0
)

THESE_PRESSURE_LEVELS_MB = numpy.array([950, 975, 1000])
THESE_FIELD_NAMES = [
    nwp_model_utils.HEIGHT_COLUMN_FOR_SOUNDINGS,
    nwp_model_utils.RH_COLUMN_FOR_SOUNDINGS,
    nwp_model_utils.TEMPERATURE_COLUMN_FOR_SOUNDINGS,
    nwp_model_utils.U_WIND_COLUMN_FOR_SOUNDINGS,
    nwp_model_utils.V_WIND_COLUMN_FOR_SOUNDINGS
]

SOUNDING_DICT_P_COORDS_NO_SURFACE = {
    soundings.FULL_IDS_KEY: THESE_FULL_ID_STRINGS,
    soundings.INITIAL_TIMES_KEY: THESE_INIT_TIMES_UNIX_SEC,
    soundings.LEAD_TIMES_KEY: THESE_LEAD_TIMES_SECONDS,
    soundings.SOUNDING_MATRIX_KEY: THIS_SOUNDING_MATRIX,
    soundings.SURFACE_PRESSURES_KEY: None,
    soundings.PRESSURE_LEVELS_WITH_SFC_KEY: THESE_PRESSURE_LEVELS_MB,
    soundings.FIELD_NAMES_KEY: THESE_FIELD_NAMES
}

THIS_MATRIX = numpy.array([
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
     2000],
    [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40,
     4200]
], dtype=float)

INTERP_TABLE_WITH_SURFACE = pandas.DataFrame(THIS_MATRIX)

for k in range(len(FIELD_NAMES_WITH_SURFACE)):
    INTERP_TABLE_WITH_SURFACE.rename(
        columns={k: FIELD_NAMES_WITH_SURFACE[k]}, inplace=True
    )

THIS_FIRST_MATRIX = numpy.array([
    [0, 8, 4, 12, 16],
    [1, 9, 5, 13, 17],
    [2, 10, 6, 14, 18],
    [3, 11, 7, 15, 19]
], dtype=int)

THIS_SECOND_MATRIX = numpy.array([
    [2, 18, 10, 26, 34],
    [4, 20, 12, 28, 36],
    [6, 22, 14, 30, 38],
    [8, 24, 16, 32, 40]
], dtype=int)

THIS_SOUNDING_MATRIX = numpy.stack(
    (THIS_FIRST_MATRIX, THIS_SECOND_MATRIX), axis=0
)

THESE_PRESSURE_LEVELS_MB = numpy.array([950, 975, 1000, numpy.nan])
THESE_SURFACE_PRESSURES_MB = numpy.array([20, 42])

SOUNDING_DICT_P_COORDS_WITH_SURFACE = {
    soundings.FULL_IDS_KEY: THESE_FULL_ID_STRINGS,
    soundings.INITIAL_TIMES_KEY: THESE_INIT_TIMES_UNIX_SEC,
    soundings.LEAD_TIMES_KEY: THESE_LEAD_TIMES_SECONDS,
    soundings.SOUNDING_MATRIX_KEY: THIS_SOUNDING_MATRIX,
    soundings.SURFACE_PRESSURES_KEY: THESE_SURFACE_PRESSURES_MB,
    soundings.PRESSURE_LEVELS_WITH_SFC_KEY: THESE_PRESSURE_LEVELS_MB,
    soundings.FIELD_NAMES_KEY: THESE_FIELD_NAMES
}

# The following constants are used to test _get_pressures.
PRESSURE_MATRIX_NO_SURFACE_PASCALS = numpy.array([
    [95000, 97500, 100000],
    [95000, 97500, 100000]
])

PRESSURE_MATRIX_WITH_SURFACE_PASCALS = numpy.array([
    [95000, 97500, 100000, 2000],
    [95000, 97500, 100000, 4200]
])

# The following constants are used to test _relative_to_specific_humidity.
THESE_HEIGHTS_METRES = numpy.array([400, 300, 200, 100, 0])
THESE_TEMPERATURES_KELVINS = numpy.array([
    273.15, 278.15, 283.15, 288.15, 298.15
])
THESE_U_WINDS_M_S01 = numpy.array([-10, -5, 0, 5, 10])
THESE_V_WINDS_M_S01 = numpy.array([20, 30, -40, 15, 7.2])
THESE_SPEC_HUMIDITIES_KG_KG01 = 0.001 * numpy.array([0.1, 1., 5., 10., 20.])

THESE_PRESSURES_PASCALS = numpy.array([99000, 100000, 101000, 102000, 103000])
PRESSURE_MATRIX_PASCALS = numpy.reshape(THESE_PRESSURES_PASCALS, (1, 5))

THESE_DEWPOINTS_KELVINS = moisture_conversions.specific_humidity_to_dewpoint(
    specific_humidities_kg_kg01=THESE_SPEC_HUMIDITIES_KG_KG01,
    temperatures_kelvins=THESE_TEMPERATURES_KELVINS,
    total_pressures_pascals=THESE_PRESSURES_PASCALS
)
DEWPOINT_MATRIX_KELVINS = numpy.reshape(THESE_DEWPOINTS_KELVINS, (1, 5))

THESE_RELATIVE_HUMIDITIES = moisture_conversions.dewpoint_to_relative_humidity(
    dewpoints_kelvins=THESE_DEWPOINTS_KELVINS,
    temperatures_kelvins=THESE_TEMPERATURES_KELVINS,
    total_pressures_pascals=THESE_PRESSURES_PASCALS
)
THESE_RELATIVE_HUMIDITIES_PERCENT = 100 * THESE_RELATIVE_HUMIDITIES

THESE_FIELD_NAMES = [
    nwp_model_utils.HEIGHT_COLUMN_FOR_SOUNDINGS,
    nwp_model_utils.RH_COLUMN_FOR_SOUNDINGS,
    nwp_model_utils.TEMPERATURE_COLUMN_FOR_SOUNDINGS,
    nwp_model_utils.U_WIND_COLUMN_FOR_SOUNDINGS,
    nwp_model_utils.V_WIND_COLUMN_FOR_SOUNDINGS
]

THIS_SOUNDING_MATRIX = numpy.full((1, 5, 5), numpy.nan)
THIS_SOUNDING_MATRIX[0, :, 0] = THESE_HEIGHTS_METRES
THIS_SOUNDING_MATRIX[0, :, 1] = THESE_RELATIVE_HUMIDITIES_PERCENT
THIS_SOUNDING_MATRIX[0, :, 2] = THESE_TEMPERATURES_KELVINS
THIS_SOUNDING_MATRIX[0, :, 3] = THESE_U_WINDS_M_S01
THIS_SOUNDING_MATRIX[0, :, 4] = THESE_V_WINDS_M_S01

SOUNDING_DICT_P_COORDS_NO_SPFH = {
    soundings.SOUNDING_MATRIX_KEY: THIS_SOUNDING_MATRIX,
    soundings.FIELD_NAMES_KEY: THESE_FIELD_NAMES
}

THIS_NEW_MATRIX = numpy.reshape(THESE_SPEC_HUMIDITIES_KG_KG01, (1, 5, 1))
THIS_SOUNDING_MATRIX = numpy.concatenate(
    (THIS_SOUNDING_MATRIX, THIS_NEW_MATRIX), axis=-1
)
THIS_SOUNDING_MATRIX[..., 1] = THIS_SOUNDING_MATRIX[..., 1] / 100

THESE_FIELD_NAMES = [
    nwp_model_utils.HEIGHT_COLUMN_FOR_SOUNDINGS,
    soundings.RELATIVE_HUMIDITY_NAME,
    nwp_model_utils.TEMPERATURE_COLUMN_FOR_SOUNDINGS,
    nwp_model_utils.U_WIND_COLUMN_FOR_SOUNDINGS,
    nwp_model_utils.V_WIND_COLUMN_FOR_SOUNDINGS,
    nwp_model_utils.SPFH_COLUMN_FOR_SOUNDINGS
]

SOUNDING_DICT_P_COORDS_WITH_SPFH = {
    soundings.SOUNDING_MATRIX_KEY: THIS_SOUNDING_MATRIX,
    soundings.FIELD_NAMES_KEY: THESE_FIELD_NAMES
}

# The following constants are used to test _specific_to_relative_humidity.
THESE_FIELD_NAMES = [
    nwp_model_utils.HEIGHT_COLUMN_FOR_SOUNDINGS,
    nwp_model_utils.SPFH_COLUMN_FOR_SOUNDINGS,
    nwp_model_utils.TEMPERATURE_COLUMN_FOR_SOUNDINGS,
    nwp_model_utils.U_WIND_COLUMN_FOR_SOUNDINGS,
    nwp_model_utils.V_WIND_COLUMN_FOR_SOUNDINGS
]

THIS_SOUNDING_MATRIX = numpy.full((1, 5, 5), numpy.nan)
THIS_SOUNDING_MATRIX[0, :, 0] = THESE_HEIGHTS_METRES
THIS_SOUNDING_MATRIX[0, :, 1] = THESE_SPEC_HUMIDITIES_KG_KG01
THIS_SOUNDING_MATRIX[0, :, 2] = THESE_TEMPERATURES_KELVINS
THIS_SOUNDING_MATRIX[0, :, 3] = THESE_U_WINDS_M_S01
THIS_SOUNDING_MATRIX[0, :, 4] = THESE_V_WINDS_M_S01

SOUNDING_DICT_P_COORDS_NO_RH = {
    soundings.SOUNDING_MATRIX_KEY: THIS_SOUNDING_MATRIX,
    soundings.FIELD_NAMES_KEY: THESE_FIELD_NAMES
}

THIS_NEW_MATRIX = numpy.reshape(THESE_RELATIVE_HUMIDITIES, (1, 5, 1))
THIS_SOUNDING_MATRIX = numpy.concatenate(
    (THIS_SOUNDING_MATRIX, THIS_NEW_MATRIX), axis=-1
)
NEW_FIELD_NAMES = THESE_FIELD_NAMES + [soundings.RELATIVE_HUMIDITY_NAME]

SOUNDING_DICT_P_COORDS_WITH_RH = {
    soundings.SOUNDING_MATRIX_KEY: THIS_SOUNDING_MATRIX,
    soundings.FIELD_NAMES_KEY: NEW_FIELD_NAMES
}

SOUNDING_DICT_P_COORDS_NO_THETA_V = {
    soundings.SOUNDING_MATRIX_KEY: THIS_SOUNDING_MATRIX,
    soundings.FIELD_NAMES_KEY: NEW_FIELD_NAMES
}

# The following constants are used to test _get_virtual_potential_temperatures.
THESE_VAPOUR_PRESSURES_PASCALS = (
    moisture_conversions.dewpoint_to_vapour_pressure(
        dewpoints_kelvins=THESE_DEWPOINTS_KELVINS,
        temperatures_kelvins=THESE_TEMPERATURES_KELVINS
    )
)

THESE_VIRTUAL_TEMPS_KELVINS = (
    moisture_conversions.temperature_to_virtual_temperature(
        temperatures_kelvins=THESE_TEMPERATURES_KELVINS,
        total_pressures_pascals=THESE_PRESSURES_PASCALS,
        vapour_pressures_pascals=THESE_VAPOUR_PRESSURES_PASCALS
    )
)

THESE_THETA_V_KELVINS = (
    temperature_conversions.temperatures_to_potential_temperatures(
        temperatures_kelvins=THESE_VIRTUAL_TEMPS_KELVINS,
        total_pressures_pascals=THESE_PRESSURES_PASCALS)
)

THIS_NEW_MATRIX = numpy.reshape(THESE_THETA_V_KELVINS, (1, 5, 1))
THIS_SOUNDING_MATRIX = numpy.concatenate(
    (THIS_SOUNDING_MATRIX, THIS_NEW_MATRIX), axis=-1
)

NEW_FIELD_NAMES = THESE_FIELD_NAMES + [
    soundings.RELATIVE_HUMIDITY_NAME,
    soundings.VIRTUAL_POTENTIAL_TEMPERATURE_NAME
]

SOUNDING_DICT_P_COORDS_WITH_THETA_V = {
    soundings.SOUNDING_MATRIX_KEY: THIS_SOUNDING_MATRIX,
    soundings.FIELD_NAMES_KEY: NEW_FIELD_NAMES
}

# The following constants are used to test _fill_nans_in_soundings.
THESE_PRESSURE_LEVELS_MB = numpy.array([
    500, 600, 700, 850, 925, 1000, numpy.nan
])
THESE_SURFACE_PRESSURES_MB = numpy.array([965, 1000])
PRESSURE_MATRIX_FOR_NAN_FILL_PASCALS = numpy.array([
    [50000, 60000, 70000, 85000, 92500, 100000, 96500],
    [50000, 60000, 70000, 85000, 92500, 100000, 100000]
])

THESE_HEIGHTS_METRES = numpy.array([5700, numpy.nan, 3090, 1500, 770, 0, 385])
THESE_U_WINDS_M_S01 = numpy.array([numpy.nan, 30, 20, 10, 0, -10, numpy.nan])
THESE_V_WINDS_M_S01 = numpy.array([30, 25, 20, numpy.nan, 10, 5, 0])
THESE_TEMPERATURES_KELVINS = numpy.array([
    numpy.nan, numpy.nan, 275, numpy.nan, 290, 300, 301
])
THESE_SPEC_HUMIDITIES_KG_KG01 = numpy.array([
    0.001, 0.002, numpy.nan, numpy.nan, 0.005, numpy.nan, numpy.nan
])

THESE_FIELD_NAMES = [
    soundings.GEOPOTENTIAL_HEIGHT_NAME, soundings.U_WIND_NAME,
    soundings.V_WIND_NAME, soundings.TEMPERATURE_NAME,
    soundings.SPECIFIC_HUMIDITY_NAME
]

THIS_FIRST_MATRIX = numpy.transpose(numpy.vstack(
    (THESE_HEIGHTS_METRES, THESE_U_WINDS_M_S01, THESE_V_WINDS_M_S01,
     THESE_TEMPERATURES_KELVINS, THESE_SPEC_HUMIDITIES_KG_KG01)
))

THESE_HEIGHTS_METRES = numpy.array([
    numpy.nan, numpy.nan, numpy.nan, 1600, numpy.nan, numpy.nan, numpy.nan
])
THIS_SECOND_MATRIX = numpy.transpose(numpy.vstack(
    (THESE_HEIGHTS_METRES, THESE_U_WINDS_M_S01, THESE_V_WINDS_M_S01,
     THESE_TEMPERATURES_KELVINS, THESE_SPEC_HUMIDITIES_KG_KG01)
))

THIS_SOUNDING_MATRIX = numpy.stack(
    (THIS_FIRST_MATRIX, THIS_SECOND_MATRIX), axis=0
)

THESE_FULL_ID_STRINGS = ['c', 'd']
THESE_INIT_TIMES_UNIX_SEC = numpy.array([6, 6], dtype=int)
THESE_LEAD_TIMES_SECONDS = numpy.array([1, 1], dtype=int)

SOUNDING_DICT_P_COORDS_WITH_NANS = {
    soundings.FULL_IDS_KEY: THESE_FULL_ID_STRINGS,
    soundings.INITIAL_TIMES_KEY: THESE_INIT_TIMES_UNIX_SEC,
    soundings.LEAD_TIMES_KEY: THESE_LEAD_TIMES_SECONDS,
    soundings.SOUNDING_MATRIX_KEY: THIS_SOUNDING_MATRIX,
    soundings.PRESSURE_LEVELS_WITH_SFC_KEY: THESE_PRESSURE_LEVELS_MB,
    soundings.FIELD_NAMES_KEY: THESE_FIELD_NAMES,
    soundings.SURFACE_PRESSURES_KEY: THESE_SURFACE_PRESSURES_MB
}

THESE_HEIGHTS_METRES = numpy.array([
    5700, 4285.73988745, 3090, 1500, 770, 0, 385
])
THESE_U_WINDS_M_S01 = numpy.array([41.82748964, 30, 20, 10, 0, -10, -5])
THESE_V_WINDS_M_S01 = numpy.array([30, 25, 20, 13.14655172, 10, 5, 0])
THESE_TEMPERATURES_KELVINS = numpy.array([
    258.125, 267.26892314, 275, 285.28017241, 290, 300, 301
])
THESE_SPEC_HUMIDITIES_KG_KG01 = numpy.array([
    0.001, 0.002, 0.00302033, 0.00437709, 0.005, 0.00565705, 0.00532852
])

THIS_SOUNDING_MATRIX = numpy.transpose(numpy.vstack(
    (THESE_HEIGHTS_METRES, THESE_U_WINDS_M_S01, THESE_V_WINDS_M_S01,
     THESE_TEMPERATURES_KELVINS, THESE_SPEC_HUMIDITIES_KG_KG01)
))
THIS_SOUNDING_MATRIX = numpy.expand_dims(THIS_SOUNDING_MATRIX, 0)

THESE_FULL_ID_STRINGS = ['c']
THESE_INIT_TIMES_UNIX_SEC = numpy.array([6], dtype=int)
THESE_LEAD_TIMES_SECONDS = numpy.array([1], dtype=int)
THESE_SURFACE_PRESSURES_MB = numpy.array([965])

SOUNDING_DICT_P_COORDS_NO_NANS = {
    soundings.FULL_IDS_KEY: THESE_FULL_ID_STRINGS,
    soundings.INITIAL_TIMES_KEY: THESE_INIT_TIMES_UNIX_SEC,
    soundings.LEAD_TIMES_KEY: THESE_LEAD_TIMES_SECONDS,
    soundings.SOUNDING_MATRIX_KEY: THIS_SOUNDING_MATRIX,
    soundings.PRESSURE_LEVELS_WITH_SFC_KEY: THESE_PRESSURE_LEVELS_MB,
    soundings.FIELD_NAMES_KEY: THESE_FIELD_NAMES,
    soundings.SURFACE_PRESSURES_KEY: THESE_SURFACE_PRESSURES_MB
}

# The following constants are used to test _pressure_to_height_coords.
THESE_STORM_ELEVATIONS_M_ASL = numpy.array([385])
SOUNDING_DICT_PRESSURE_COORDS = copy.deepcopy(SOUNDING_DICT_P_COORDS_NO_NANS)
SOUNDING_DICT_PRESSURE_COORDS.update({
    soundings.STORM_ELEVATIONS_KEY: THESE_STORM_ELEVATIONS_M_ASL
})

HEIGHT_LEVELS_M_AGL = numpy.array([0, 385, 1115, 1910, 2705])
THESE_PRESSURES_PASCALS = numpy.array([96500, 92500, 85000, 77136.2431, 70000])
THESE_U_WINDS_M_S01 = numpy.array([-5, 0, 10, 15.2425046, 20])
THESE_V_WINDS_M_S01 = numpy.array([0, 10, 13.14655172, 16.7394751, 20])
THESE_TEMPERATURES_KELVINS = numpy.array([
    301, 290, 285.28017241, 279.890787, 275
])
THESE_SPEC_HUMIDITIES_KG_KG01 = numpy.array([
    0.00532852, 0.005, 0.00437709, 0.00366580795, 0.00302033
])

THESE_DEWPOINTS_KELVINS = (
    moisture_conversions.specific_humidity_to_dewpoint(
        specific_humidities_kg_kg01=THESE_SPEC_HUMIDITIES_KG_KG01,
        temperatures_kelvins=THESE_TEMPERATURES_KELVINS,
        total_pressures_pascals=THESE_PRESSURES_PASCALS
    )
)

THESE_RELATIVE_HUMIDITIES = (
    moisture_conversions.dewpoint_to_relative_humidity(
        dewpoints_kelvins=THESE_DEWPOINTS_KELVINS,
        temperatures_kelvins=THESE_TEMPERATURES_KELVINS,
        total_pressures_pascals=THESE_PRESSURES_PASCALS
    )
)

THESE_VAPOUR_PRESSURES_PASCALS = (
    moisture_conversions.dewpoint_to_vapour_pressure(
        dewpoints_kelvins=THESE_DEWPOINTS_KELVINS,
        temperatures_kelvins=THESE_TEMPERATURES_KELVINS
    )
)

THESE_VIRTUAL_TEMPS_KELVINS = (
    moisture_conversions.temperature_to_virtual_temperature(
        temperatures_kelvins=THESE_TEMPERATURES_KELVINS,
        total_pressures_pascals=THESE_PRESSURES_PASCALS,
        vapour_pressures_pascals=THESE_VAPOUR_PRESSURES_PASCALS
    )
)

THESE_THETA_V_KELVINS = (
    temperature_conversions.temperatures_to_potential_temperatures(
        temperatures_kelvins=THESE_VIRTUAL_TEMPS_KELVINS,
        total_pressures_pascals=THESE_PRESSURES_PASCALS)
)

THESE_FIELD_NAMES = [
    soundings.PRESSURE_NAME, soundings.U_WIND_NAME, soundings.V_WIND_NAME,
    soundings.TEMPERATURE_NAME, soundings.SPECIFIC_HUMIDITY_NAME,
    soundings.RELATIVE_HUMIDITY_NAME,
    soundings.VIRTUAL_POTENTIAL_TEMPERATURE_NAME
]

THIS_SOUNDING_MATRIX = numpy.transpose(numpy.vstack(
    (THESE_PRESSURES_PASCALS, THESE_U_WINDS_M_S01, THESE_V_WINDS_M_S01,
     THESE_TEMPERATURES_KELVINS, THESE_SPEC_HUMIDITIES_KG_KG01,
     THESE_RELATIVE_HUMIDITIES, THESE_THETA_V_KELVINS)
))

THIS_SOUNDING_MATRIX = numpy.expand_dims(THIS_SOUNDING_MATRIX, 0)

SOUNDING_DICT_HEIGHT_COORDS = {
    soundings.FULL_IDS_KEY: THESE_FULL_ID_STRINGS,
    soundings.INITIAL_TIMES_KEY: THESE_INIT_TIMES_UNIX_SEC,
    soundings.LEAD_TIMES_KEY: THESE_LEAD_TIMES_SECONDS,
    soundings.STORM_ELEVATIONS_KEY: THESE_STORM_ELEVATIONS_M_ASL,
    soundings.SOUNDING_MATRIX_KEY: THIS_SOUNDING_MATRIX,
    soundings.HEIGHT_LEVELS_KEY: HEIGHT_LEVELS_M_AGL,
    soundings.FIELD_NAMES_KEY: THESE_FIELD_NAMES
}

# The following constants are used to test field_name_to_verbose.
NON_VERBOSE_FIELD_NAMES = [
    'pressure_pascals', 'virtual_potential_temperature_kelvins',
    'relative_humidity_unitless', 'specific_humidity',
    'geopotential_height_metres', 'v_wind_m_s01', 'u_wind_m_s01',
    'temperature_kelvins'
]

VERBOSE_FIELD_NAMES_WITH_UNITS = [
    'Pressure (Pa)', 'Virtual potential temperature (K)',
    'Relative humidity', r'Specific humidity (kg kg$^{-1}$)',
    'Geopotential height (m)', r'$v$-wind (m s$^{-1}$)',
    r'$u$-wind (m s$^{-1}$)', 'Temperature (K)'
]

VERBOSE_FIELD_NAMES_NO_UNITS = [
    'Pressure', 'Virtual potential temperature',
    'Relative humidity', 'Specific humidity',
    'Geopotential height', r'$v$-wind',
    r'$u$-wind', 'Temperature'
]

# The following constants are used to test find_sounding_file.
TOP_DIRECTORY_NAME = 'storm_soundings'
SPC_DATE_STRING = '20180618'
LEAD_TIME_IN_FILES_SEC = 3600
LAG_TIME_IN_FILES_SEC = 1800
FILE_TIME_UNIX_SEC = 1529374673

SOUNDING_FILE_NAME_ONE_TIME = (
    'storm_soundings/2018/20180618/'
    'storm_soundings_2018-06-19-021753_lead-time-03600sec_lag-time-1800sec.nc'
)
SOUNDING_FILE_NAME_ONE_SPC_DATE = (
    'storm_soundings/2018/storm_soundings_20180618_lead-time-03600sec'
    '_lag-time-1800sec.nc'
)


def _compare_target_point_tables(
        first_target_point_table, second_target_point_table):
    """Determines equality of two target-point tables.

    :param first_target_point_table: First table.
    :param second_target_point_table: Second table.
    :return: are_tables_equal: Boolean flag.
    """

    first_column_names = list(first_target_point_table)
    second_column_names = list(second_target_point_table)
    if set(first_column_names) != set(second_column_names):
        return False

    for this_column_name in first_column_names:
        if this_column_name == tracking_utils.FULL_ID_COLUMN:
            are_columns_equal = numpy.array_equal(
                first_target_point_table[this_column_name].values,
                second_target_point_table[this_column_name].values)
        else:
            are_columns_equal = numpy.allclose(
                first_target_point_table[this_column_name].values,
                second_target_point_table[this_column_name].values,
                atol=TOLERANCE)

        if not are_columns_equal:
            return False

    return True


def _compare_sounding_dictionaries(first_sounding_dict, second_sounding_dict):
    """Determines equality of two sounding dictionaries.

    Either both dictionaries must be in pressure coords, or both must be in
    ground-relative height coords.

    :param first_sounding_dict: Dictionary with keys listed in
        `_convert_interp_table_to_soundings` or `_pressure_to_height_coords`.
    :param second_sounding_dict: Same.
    :return: are_dicts_equal: Boolean flag.
    """

    first_keys = list(first_sounding_dict.keys())
    second_keys = list(second_sounding_dict.keys())
    if set(first_keys) != set(second_keys):
        return False

    sort_indices = numpy.argsort(numpy.array(
        first_sounding_dict[soundings.FIELD_NAMES_KEY]
    ))

    first_sounding_dict[soundings.FIELD_NAMES_KEY] = [
        first_sounding_dict[soundings.FIELD_NAMES_KEY][k] for k in sort_indices
    ]
    first_sounding_dict[soundings.SOUNDING_MATRIX_KEY] = (
        first_sounding_dict[soundings.SOUNDING_MATRIX_KEY][..., sort_indices]
    )

    sort_indices = numpy.argsort(numpy.array(
        second_sounding_dict[soundings.FIELD_NAMES_KEY]
    ))

    second_sounding_dict[soundings.FIELD_NAMES_KEY] = [
        second_sounding_dict[soundings.FIELD_NAMES_KEY][k] for k in sort_indices
    ]
    second_sounding_dict[soundings.SOUNDING_MATRIX_KEY] = (
        second_sounding_dict[soundings.SOUNDING_MATRIX_KEY][..., sort_indices]
    )

    for this_key in list(first_sounding_dict.keys()):
        if this_key in [soundings.FIELD_NAMES_KEY, soundings.FULL_IDS_KEY]:
            if first_sounding_dict[this_key] != second_sounding_dict[this_key]:
                print('FOO2')
                return False

        elif (this_key == soundings.SURFACE_PRESSURES_KEY and
              first_sounding_dict[this_key] is None):
            if second_sounding_dict[this_key] is not None:
                print('FOO3')
                return False

        else:
            if not numpy.allclose(
                    first_sounding_dict[this_key],
                    second_sounding_dict[this_key], atol=TOLERANCE,
                    equal_nan=True):
                print(first_sounding_dict[this_key])
                print('\n\n')
                print(second_sounding_dict[this_key])
                print('\n\n\n***\n\n\n')
                return False

    return True


class SoundingsTests(unittest.TestCase):
    """Each method is a unit test for soundings.py."""

    def test_get_nwp_fields_for_sounding_no_table_no_surface(self):
        """Ensures correct output from _get_nwp_fields_for_sounding.

        In this case, return_table = False and include_surface = False.
        """

        these_field_names, these_field_names_grib1 = (
            soundings._get_nwp_fields_for_sounding(
                model_name=MODEL_NAME, return_table=False,
                minimum_pressure_mb=MINIMUM_PRESSURE_MB, include_surface=False
            )[:2]
        )

        self.assertTrue(
            set(these_field_names) == set(FIELD_NAMES_NO_SURFACE)
        )
        self.assertTrue(
            set(these_field_names_grib1) == set(FIELD_NAMES_NO_SURFACE_GRIB1)
        )

    def test_get_nwp_fields_for_sounding_no_table_yes_surface(self):
        """Ensures correct output from _get_nwp_fields_for_sounding.

        In this case, return_table = False and include_surface = True.
        """

        these_field_names, these_field_names_grib1 = (
            soundings._get_nwp_fields_for_sounding(
                model_name=MODEL_NAME, return_table=False,
                minimum_pressure_mb=MINIMUM_PRESSURE_MB, include_surface=True
            )[:2]
        )

        self.assertTrue(
            set(these_field_names) == set(FIELD_NAMES_WITH_SURFACE)
        )
        self.assertTrue(
            set(these_field_names_grib1) == set(FIELD_NAMES_WITH_SURFACE_GRIB1)
        )

    def test_get_nwp_fields_for_sounding_yes_table_no_surface(self):
        """Ensures correct output from _get_nwp_fields_for_sounding.

        In this case, return_table = True and include_surface = False.
        """

        this_field_name_table = soundings._get_nwp_fields_for_sounding(
            model_name=MODEL_NAME, return_table=True,
            minimum_pressure_mb=MINIMUM_PRESSURE_MB, include_surface=False
        )[-1]

        actual_columns = list(this_field_name_table)
        expected_columns = list(FIELD_NAME_TABLE_NO_SURFACE)
        self.assertTrue(set(actual_columns) == set(expected_columns))

        self.assertTrue(this_field_name_table[actual_columns].equals(
            FIELD_NAME_TABLE_NO_SURFACE[actual_columns]
        ))

    def test_get_nwp_fields_for_sounding_yes_table_yes_surface(self):
        """Ensures correct output from _get_nwp_fields_for_sounding.

        In this case, return_table = True and include_surface = True.
        """

        this_field_name_table = soundings._get_nwp_fields_for_sounding(
            model_name=MODEL_NAME, return_table=True,
            minimum_pressure_mb=MINIMUM_PRESSURE_MB, include_surface=True
        )[-1]

        actual_columns = list(this_field_name_table)
        expected_columns = list(FIELD_NAME_TABLE_WITH_SURFACE)
        self.assertTrue(set(actual_columns) == set(expected_columns))

        self.assertTrue(this_field_name_table[actual_columns].equals(
            FIELD_NAME_TABLE_WITH_SURFACE[actual_columns]
        ))

    def test_create_target_points_for_interp(self):
        """Ensures correct output from _create_target_points_for_interp."""

        this_target_point_table = soundings._create_target_points_for_interp(
            storm_object_table=DUMMY_STORM_OBJECT_TABLE,
            lead_times_seconds=UNIQUE_LEAD_TIMES_SECONDS)

        self.assertTrue(_compare_target_point_tables(
            this_target_point_table, DUMMY_TARGET_POINT_TABLE
        ))

    def test_convert_interp_table_to_soundings_no_surface(self):
        """Ensures correct output from _convert_interp_table_to_soundings.

        In this case, include_surface = False.
        """

        this_sounding_dict = soundings._convert_interp_table_to_soundings(
            interp_table=INTERP_TABLE_NO_SURFACE,
            target_point_table=TARGET_POINT_TABLE, model_name=MODEL_NAME,
            include_surface=False, minimum_pressure_mb=MINIMUM_PRESSURE_MB)

        self.assertTrue(_compare_sounding_dictionaries(
            this_sounding_dict, copy.deepcopy(SOUNDING_DICT_P_COORDS_NO_SURFACE)
        ))

    def test_convert_interp_table_to_soundings_with_surface(self):
        """Ensures correct output from _convert_interp_table_to_soundings.

        In this case, include_surface = True.
        """

        this_sounding_dict = soundings._convert_interp_table_to_soundings(
            interp_table=INTERP_TABLE_WITH_SURFACE,
            target_point_table=TARGET_POINT_TABLE, model_name=MODEL_NAME,
            include_surface=True, minimum_pressure_mb=MINIMUM_PRESSURE_MB)

        self.assertTrue(_compare_sounding_dictionaries(
            this_sounding_dict,
            copy.deepcopy(SOUNDING_DICT_P_COORDS_WITH_SURFACE)
        ))

    def test_get_pressures_no_surface(self):
        """Ensures correct output from _get_pressures.

        In this case, soundings do *not* include surface.
        """

        this_pressure_matrix_pascals = soundings._get_pressures(
            SOUNDING_DICT_P_COORDS_NO_SURFACE)

        self.assertTrue(numpy.allclose(
            this_pressure_matrix_pascals, PRESSURE_MATRIX_NO_SURFACE_PASCALS,
            atol=TOLERANCE
        ))

    def test_get_pressures_with_surface(self):
        """Ensures correct output from _get_pressures.

        In this case, soundings include surface.
        """

        this_pressure_matrix_pascals = soundings._get_pressures(
            SOUNDING_DICT_P_COORDS_WITH_SURFACE)

        self.assertTrue(numpy.allclose(
            this_pressure_matrix_pascals, PRESSURE_MATRIX_WITH_SURFACE_PASCALS,
            atol=TOLERANCE
        ))

    def test_relative_to_specific_humidity(self):
        """Ensures correct output from _relative_to_specific_humidity."""

        this_sounding_dict, this_dewpoint_matrix_kelvins = (
            soundings._relative_to_specific_humidity(
                sounding_dict=copy.deepcopy(SOUNDING_DICT_P_COORDS_NO_SPFH),
                pressure_matrix_pascals=PRESSURE_MATRIX_PASCALS)
        )

        self.assertTrue(numpy.allclose(
            this_dewpoint_matrix_kelvins, DEWPOINT_MATRIX_KELVINS,
            atol=TOLERANCE_FOR_CONVERTED_VALUES
        ))

        self.assertTrue(_compare_sounding_dictionaries(
            this_sounding_dict, copy.deepcopy(SOUNDING_DICT_P_COORDS_WITH_SPFH)
        ))

    def test_specific_to_relative_humidity(self):
        """Ensures correct output from _specific_to_relative_humidity."""

        this_sounding_dict, this_dewpoint_matrix_kelvins = (
            soundings._specific_to_relative_humidity(
                sounding_dict=copy.deepcopy(SOUNDING_DICT_P_COORDS_NO_RH),
                pressure_matrix_pascals=PRESSURE_MATRIX_PASCALS)
        )

        self.assertTrue(numpy.allclose(
            this_dewpoint_matrix_kelvins, DEWPOINT_MATRIX_KELVINS,
            atol=TOLERANCE_FOR_CONVERTED_VALUES
        ))

        self.assertTrue(_compare_sounding_dictionaries(
            this_sounding_dict, copy.deepcopy(SOUNDING_DICT_P_COORDS_WITH_RH)
        ))

    def test_get_virtual_potential_temperatures(self):
        """Ensures correct output from _get_virtual_potential_temperatures."""

        this_sounding_dict = soundings._get_virtual_potential_temperatures(
            sounding_dict=copy.deepcopy(SOUNDING_DICT_P_COORDS_NO_THETA_V),
            pressure_matrix_pascals=PRESSURE_MATRIX_PASCALS,
            dewpoint_matrix_kelvins=DEWPOINT_MATRIX_KELVINS)

        self.assertTrue(_compare_sounding_dictionaries(
            this_sounding_dict,
            copy.deepcopy(SOUNDING_DICT_P_COORDS_WITH_THETA_V)
        ))

    def test_fill_nans_in_soundings(self):
        """Ensures correct output from _fill_nans_in_soundings."""

        this_sounding_dict = soundings._fill_nans_in_soundings(
            sounding_dict_pressure_coords=copy.deepcopy(
                SOUNDING_DICT_P_COORDS_WITH_NANS),
            pressure_matrix_pascals=PRESSURE_MATRIX_FOR_NAN_FILL_PASCALS,
            min_num_pressure_levels_without_nan=2)

        self.assertTrue(_compare_sounding_dictionaries(
            this_sounding_dict, copy.deepcopy(SOUNDING_DICT_P_COORDS_NO_NANS)
        ))

    def test_pressure_to_height_coords(self):
        """Ensures correct output from _pressure_to_height_coords."""

        this_sounding_dict = soundings._pressure_to_height_coords(
            sounding_dict_pressure_coords=copy.deepcopy(
                SOUNDING_DICT_PRESSURE_COORDS),
            height_levels_m_agl=HEIGHT_LEVELS_M_AGL)

        self.assertTrue(_compare_sounding_dictionaries(
            this_sounding_dict, copy.deepcopy(SOUNDING_DICT_HEIGHT_COORDS)
        ))

    def test_check_field_name_valid(self):
        """Ensures correct output from check_field_name.

        In this case, field name is valid.
        """

        soundings.check_field_name(soundings.VIRTUAL_POTENTIAL_TEMPERATURE_NAME)

    def test_check_field_name_invalid(self):
        """Ensures correct output from check_field_name.

        In this case, field name is *not* valid.
        """

        with self.assertRaises(ValueError):
            soundings.check_field_name(soundings.STORM_ELEVATIONS_KEY)

    def test_field_name_to_verbose_with_units(self):
        """Ensures correct output from field_name_to_verbose.

        In this case, verbose field names should have units.
        """

        these_verbose_field_names = [
            soundings.field_name_to_verbose(field_name=f, include_units=True)
            for f in NON_VERBOSE_FIELD_NAMES
        ]

        self.assertTrue(
            these_verbose_field_names == VERBOSE_FIELD_NAMES_WITH_UNITS
        )

    def test_field_name_to_verbose_no_units(self):
        """Ensures correct output from field_name_to_verbose.

        In this case, verbose field names should have no units.
        """

        these_verbose_field_names = [
            soundings.field_name_to_verbose(field_name=f, include_units=False)
            for f in NON_VERBOSE_FIELD_NAMES
        ]

        self.assertTrue(
            these_verbose_field_names == VERBOSE_FIELD_NAMES_NO_UNITS
        )

    def test_find_sounding_file_one_time(self):
        """Ensures correct output from find_sounding_file.

        In this case, the file contains soundings for one time step.
        """

        this_file_name = soundings.find_sounding_file(
            top_directory_name=TOP_DIRECTORY_NAME,
            spc_date_string=SPC_DATE_STRING,
            lead_time_seconds=LEAD_TIME_IN_FILES_SEC,
            lag_time_for_convective_contamination_sec=LAG_TIME_IN_FILES_SEC,
            init_time_unix_sec=FILE_TIME_UNIX_SEC, raise_error_if_missing=False)

        self.assertTrue(this_file_name == SOUNDING_FILE_NAME_ONE_TIME)

    def test_find_sounding_file_one_spc_date(self):
        """Ensures correct output from find_sounding_file.

        In this case, the file contains soundings for one SPC date.
        """

        this_file_name = soundings.find_sounding_file(
            top_directory_name=TOP_DIRECTORY_NAME,
            spc_date_string=SPC_DATE_STRING,
            lead_time_seconds=LEAD_TIME_IN_FILES_SEC,
            lag_time_for_convective_contamination_sec=LAG_TIME_IN_FILES_SEC,
            init_time_unix_sec=None, raise_error_if_missing=False)

        self.assertTrue(this_file_name == SOUNDING_FILE_NAME_ONE_SPC_DATE)


if __name__ == '__main__':
    unittest.main()
