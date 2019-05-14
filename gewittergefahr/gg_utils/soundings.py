"""Atmospheric soundings.

--- NOTATION ---

The following letters will be used throughout this module.

T = number of lead times
n = number of storm objects
p = number of pressure levels, not including surface
P = number of pressure levels, including surface
F = number of sounding fields

N = number of soundings = T*n
"""

import os.path
import numpy
import pandas
import netCDF4
from scipy.interpolate import interp1d as scipy_interp1d
from gewittergefahr.gg_io import grib_io
from gewittergefahr.gg_io import netcdf_io
from gewittergefahr.gg_utils import geodetic_utils
from gewittergefahr.gg_utils import nwp_model_utils
from gewittergefahr.gg_utils import storm_tracking_utils as tracking_utils
from gewittergefahr.gg_utils import interp
from gewittergefahr.gg_utils import moisture_conversions
from gewittergefahr.gg_utils import temperature_conversions
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.gg_utils import error_checking

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'
TIME_FORMAT_IN_FILE_NAMES = '%Y-%m-%d-%H%M%S'

MB_TO_PASCALS = 100
PASCALS_TO_MB = 0.01
PERCENT_TO_UNITLESS = 0.01
ELEVATION_DIR_NAME = '/condo/swatwork/ralager/elevation'

PRESSURE_LEVEL_KEY = 'pressure_level_mb'
LEAD_TIME_KEY = 'lead_time_seconds'
LAG_TIME_KEY = 'lag_time_for_convective_contamination_sec'
INITIAL_TIME_COLUMN = 'init_time_unix_sec'
FORECAST_TIME_COLUMN = 'forecast_time_unix_sec'

FULL_IDS_KEY = 'full_storm_id_strings'
INITIAL_TIMES_KEY = 'init_times_unix_sec'
LEAD_TIMES_KEY = 'lead_times_seconds'
STORM_ELEVATIONS_KEY = 'storm_elevations_m_asl'
SOUNDING_MATRIX_KEY = 'sounding_matrix'
SURFACE_PRESSURES_KEY = 'surface_pressures_mb'
PRESSURE_LEVELS_WITH_SFC_KEY = 'pressure_levels_with_surface_mb'
HEIGHT_LEVELS_KEY = 'height_levels_m_agl'
FIELD_NAMES_KEY = 'field_names'

GEOPOTENTIAL_HEIGHT_NAME = nwp_model_utils.HEIGHT_COLUMN_FOR_SOUNDING_TABLES
RELATIVE_HUMIDITY_NAME = 'relative_humidity_unitless'
TEMPERATURE_NAME = nwp_model_utils.TEMPERATURE_COLUMN_FOR_SOUNDING_TABLES
U_WIND_NAME = nwp_model_utils.U_WIND_COLUMN_FOR_SOUNDING_TABLES
V_WIND_NAME = nwp_model_utils.V_WIND_COLUMN_FOR_SOUNDING_TABLES
SPECIFIC_HUMIDITY_NAME = nwp_model_utils.SPFH_COLUMN_FOR_SOUNDING_TABLES
VIRTUAL_POTENTIAL_TEMPERATURE_NAME = 'virtual_potential_temperature_kelvins'
PRESSURE_NAME = 'pressure_pascals'

VALID_FIELD_NAMES = [
    GEOPOTENTIAL_HEIGHT_NAME, RELATIVE_HUMIDITY_NAME, TEMPERATURE_NAME,
    U_WIND_NAME, V_WIND_NAME, SPECIFIC_HUMIDITY_NAME,
    VIRTUAL_POTENTIAL_TEMPERATURE_NAME, PRESSURE_NAME
]

FIELD_NAME_TO_VERBOSE_DICT = {
    GEOPOTENTIAL_HEIGHT_NAME: 'Geopotential height (m)',
    RELATIVE_HUMIDITY_NAME: 'Relative humidity (fraction)',
    TEMPERATURE_NAME: 'Temperature (K)',
    U_WIND_NAME: r'$u$-wind (m s$^{-1}$)',
    V_WIND_NAME: r'$v$-wind (m s$^{-1}$)',
    SPECIFIC_HUMIDITY_NAME: r'Specific humidity (kg kg$^{-1}$)',
    VIRTUAL_POTENTIAL_TEMPERATURE_NAME: 'Virtual potential temperature (K)',
    PRESSURE_NAME: 'Pressure (Pa)'
}

STORM_OBJECT_DIMENSION_KEY = 'storm_object'
FIELD_DIMENSION_KEY = 'field'
HEIGHT_DIMENSION_KEY = 'height_level'
STORM_ID_CHAR_DIMENSION_KEY = 'storm_id_character'
FIELD_NAME_CHAR_DIMENSION_KEY = 'field_name_character'

# Field names for MetPy.
PRESSURE_COLUMN_METPY = 'pressures_mb'
TEMPERATURE_COLUMN_METPY = 'temperatures_deg_c'
DEWPOINT_COLUMN_METPY = 'dewpoints_deg_c'
U_WIND_COLUMN_METPY = 'u_winds_kt'
V_WIND_COLUMN_METPY = 'v_winds_kt'

DEFAULT_LEAD_TIMES_SEC = numpy.array([0], dtype=int)
DEFAULT_LAG_TIME_FOR_CONVECTIVE_CONTAMINATION_SEC = 1800
DEFAULT_HEIGHT_LEVELS_M_AGL = numpy.linspace(0, 12000, num=49, dtype=int)


def _get_nwp_fields_for_sounding(
        model_name, return_table, include_surface=False,
        minimum_pressure_mb=0.):
    """Returns list of NWP fields needed to create sounding.

    :param model_name: Model name (must be accepted by
        `nwp_model_utils.check_model_name`).
    :param return_table: Boolean flag.  See below for how this affects output.
    :param include_surface: Boolean flag.  If True, this method will return the
        "surface" (2-metre or 10-metre) level for each field.
    :param minimum_pressure_mb: Leave this alone.

    :return: sounding_field_names: [None if return_table = True]
        length-F list with names of sounding fields (GewitterGefahr format).
    :return: sounding_field_names_grib1: [None if return_table = True]
        length-F list with names of sounding fields (grib1 format).
    :return: sounding_field_name_table: [None if return_table = False]
        pandas DataFrame with the following columns.  Each row is one pressure
        level.  Only one of "relative_humidity_percent" and "specific_humidity"
        (depending on the NWP model) will be present.
    sounding_field_name_table.geopotential_height_metres: Name of geopotential-
        height field.
    sounding_field_name_table.temperature_kelvins: Name of temperature field.
    sounding_field_name_table.relative_humidity_percent: Name of humidity field.
    sounding_field_name_table.specific_humidity: Name of humidity field.
    sounding_field_name_table.u_wind_m_s01: Name of u-wind field.
    sounding_field_name_table.v_wind_m_s01: Name of v-wind field.
    sounding_field_name_table.pressure_level_mb: Pressure level (millibars).
        The surface is denoted by NaN.
    """

    nwp_model_utils.check_model_name(model_name)
    error_checking.assert_is_boolean(return_table)
    error_checking.assert_is_geq(minimum_pressure_mb, 0.)
    error_checking.assert_is_boolean(include_surface)

    pressure_levels_no_surface_mb = nwp_model_utils.get_pressure_levels(
        model_name=model_name, grid_id=nwp_model_utils.ID_FOR_130GRID
    ).astype(float)

    pressure_levels_no_surface_mb = pressure_levels_no_surface_mb[
        pressure_levels_no_surface_mb >= minimum_pressure_mb
    ]

    if include_surface:
        pressure_levels_with_surface_mb = numpy.concatenate((
            pressure_levels_no_surface_mb, numpy.array([numpy.nan])
        ))
    else:
        pressure_levels_with_surface_mb = pressure_levels_no_surface_mb + 0.

    num_pressure_levels_no_surface = len(pressure_levels_no_surface_mb)
    num_pressure_levels_with_surface = len(pressure_levels_with_surface_mb)

    field_names, field_names_grib1 = (
        nwp_model_utils.get_columns_in_sounding_table(model_name)
    )
    num_fields = len(field_names)

    sounding_field_name_table = None
    sounding_field_names = []
    sounding_field_names_grib1 = []

    if return_table:
        sounding_field_name_dict = {
            PRESSURE_LEVEL_KEY: pressure_levels_with_surface_mb
        }

        list_of_empty_strings = [''] * num_pressure_levels_with_surface
        for j in range(num_fields):
            sounding_field_name_dict.update({
                field_names[j]: list_of_empty_strings
            })

        sounding_field_name_table = pandas.DataFrame.from_dict(
            sounding_field_name_dict)

    for j in range(num_fields):
        for k in range(num_pressure_levels_no_surface):
            this_field_name = '{0:s}_{1:d}mb'.format(
                field_names[j],
                int(numpy.round(pressure_levels_no_surface_mb[k]))
            )

            if return_table:
                sounding_field_name_table[field_names[j]].values[k] = (
                    this_field_name
                )
            else:
                this_field_name_grib1 = '{0:s}:{1:d} mb'.format(
                    field_names_grib1[j],
                    int(numpy.round(pressure_levels_no_surface_mb[k]))
                )

                sounding_field_names.append(this_field_name)
                sounding_field_names_grib1.append(this_field_name_grib1)

        if not include_surface:
            continue

        if field_names[j] == GEOPOTENTIAL_HEIGHT_NAME:
            this_field_name, this_field_name_grib1 = (
                nwp_model_utils.get_lowest_height_name(model_name)
            )

        if field_names[j] == TEMPERATURE_NAME:
            this_field_name, this_field_name_grib1 = (
                nwp_model_utils.get_lowest_temperature_name(model_name)
            )

        if field_names[j] in [nwp_model_utils.RH_COLUMN_FOR_SOUNDING_TABLES,
                              SPECIFIC_HUMIDITY_NAME]:
            this_field_name, this_field_name_grib1 = (
                nwp_model_utils.get_lowest_humidity_name(model_name)
            )

        if field_names[j] == U_WIND_NAME:
            this_field_name, this_field_name_grib1 = (
                nwp_model_utils.get_lowest_u_wind_name(model_name)
            )

        if field_names[j] == V_WIND_NAME:
            this_field_name, this_field_name_grib1 = (
                nwp_model_utils.get_lowest_v_wind_name(model_name)
            )

        if return_table:
            sounding_field_name_table[field_names[j]].values[
                num_pressure_levels_with_surface - 1
            ] = this_field_name
        else:
            sounding_field_names.append(this_field_name)
            sounding_field_names_grib1.append(this_field_name_grib1)

    if return_table or not include_surface:
        return (sounding_field_names, sounding_field_names_grib1,
                sounding_field_name_table)

    this_field_name, this_field_name_grib1 = (
        nwp_model_utils.get_lowest_pressure_name(model_name)
    )

    sounding_field_names.append(this_field_name)
    sounding_field_names_grib1.append(this_field_name_grib1)

    return (sounding_field_names, sounding_field_names_grib1,
            sounding_field_name_table)


def _create_target_points_for_interp(storm_object_table, lead_times_seconds):
    """Creates target points for interpolation.

    Each target point consists of (latitude, longitude, time).

    :param storm_object_table: pandas DataFrame with columns documented in
        `storm_tracking_io.write_file`.
    :param lead_times_seconds: 1-D numpy array of lead times (non-negative
        integers).  For each lead time t, each storm object will be extrapolated
        t seconds into the future, along its estimated motion vector.
    :return: target_point_table: pandas DataFrame with the following columns.
    target_point_table.full_id_string: Full storm ID.
    target_point_table.init_time_unix_sec: Initial time (storm time).  Valid
        time = initial time + lead time.
    target_point_table.centroid_lat_deg: Latitude (deg N) of extrapolated storm
        object's centroid.
    target_point_table.centroid_lng_deg: Longitude (deg E) of extrapolated storm
        object's centroid.
    target_point_table.valid_time_unix_sec: Time of extrapolated storm object.
    target_point_table.lead_time_seconds: Lead time used for extrapolation.
    target_point_table.east_velocity_m_s01: Eastward component (metres per
        second) of estimated storm-motion vector.
    target_point_table.north_velocity_m_s01: Northward component.
    """

    if numpy.any(lead_times_seconds > 0):
        storm_speeds_m_s01, storm_bearings_deg = (
            geodetic_utils.xy_to_scalar_displacements_and_bearings(
                x_displacements_metres=
                storm_object_table[tracking_utils.EAST_VELOCITY_COLUMN].values,
                y_displacements_metres=
                storm_object_table[tracking_utils.NORTH_VELOCITY_COLUMN].values)
        )

    num_storm_objects = len(storm_object_table.index)
    num_lead_times = len(lead_times_seconds)
    list_of_target_point_tables = [None] * num_lead_times

    for i in range(num_lead_times):
        if lead_times_seconds[i] == 0:
            list_of_target_point_tables[i] = storm_object_table[[
                tracking_utils.FULL_ID_COLUMN, tracking_utils.VALID_TIME_COLUMN,
                tracking_utils.CENTROID_LATITUDE_COLUMN,
                tracking_utils.CENTROID_LONGITUDE_COLUMN,
                tracking_utils.EAST_VELOCITY_COLUMN,
                tracking_utils.NORTH_VELOCITY_COLUMN
            ]]

            argument_dict = {
                LEAD_TIME_KEY: numpy.full(num_storm_objects, 0, dtype=int),
                FORECAST_TIME_COLUMN: list_of_target_point_tables[i][
                    tracking_utils.VALID_TIME_COLUMN].values
            }

            list_of_target_point_tables[i] = (
                list_of_target_point_tables[i].assign(**argument_dict)
            )

            continue

        these_extrap_latitudes_deg, these_extrap_longitudes_deg = (
            geodetic_utils.start_points_and_displacements_to_endpoints(
                start_latitudes_deg=storm_object_table[
                    tracking_utils.CENTROID_LATITUDE_COLUMN].values,
                start_longitudes_deg=storm_object_table[
                    tracking_utils.CENTROID_LONGITUDE_COLUMN].values,
                scalar_displacements_metres=
                storm_speeds_m_s01 * lead_times_seconds[i],
                geodetic_bearings_deg=storm_bearings_deg)
        )

        these_times_unix_sec = (
            storm_object_table[tracking_utils.VALID_TIME_COLUMN].values +
            lead_times_seconds[i]
        )

        this_dict = {
            tracking_utils.FULL_ID_COLUMN:
                storm_object_table[tracking_utils.FULL_ID_COLUMN].values,
            tracking_utils.VALID_TIME_COLUMN:
                storm_object_table[tracking_utils.VALID_TIME_COLUMN].values,
            tracking_utils.CENTROID_LATITUDE_COLUMN: these_extrap_latitudes_deg,
            tracking_utils.CENTROID_LONGITUDE_COLUMN:
                these_extrap_longitudes_deg,
            FORECAST_TIME_COLUMN: these_times_unix_sec,
            tracking_utils.EAST_VELOCITY_COLUMN:
                storm_object_table[tracking_utils.EAST_VELOCITY_COLUMN].values,
            tracking_utils.NORTH_VELOCITY_COLUMN:
                storm_object_table[tracking_utils.NORTH_VELOCITY_COLUMN].values,
            LEAD_TIME_KEY: numpy.full(
                num_storm_objects, lead_times_seconds[i], dtype=int)
        }

        list_of_target_point_tables[i] = pandas.DataFrame.from_dict(this_dict)
        if i == 0:
            continue

        list_of_target_point_tables[i] = list_of_target_point_tables[i].align(
            list_of_target_point_tables[0], axis=1
        )[0]

    target_point_table = pandas.concat(
        list_of_target_point_tables, axis=0, ignore_index=True)

    column_dict_old_to_new = {
        tracking_utils.VALID_TIME_COLUMN: INITIAL_TIME_COLUMN
    }

    return target_point_table.rename(
        columns=column_dict_old_to_new, inplace=False)


def _interp_soundings_from_nwp(
        target_point_table, top_grib_directory_name, include_surface,
        model_name, use_all_grids, grid_id, wgrib_exe_name, wgrib2_exe_name,
        raise_error_if_missing):
    """Interpolates soundings from NWP model to target points.

    Each target point consists of (latitude, longitude, time).

    :param target_point_table: pandas DataFrame created by
        `_create_target_points_for_interp`.
    :param top_grib_directory_name: See doc for
        `interp.interp_nwp_from_xy_grid`.
    :param include_surface: See doc for `_get_nwp_fields_for_sounding`.
    :param model_name: See doc for `interp.interp_nwp_from_xy_grid`.
    :param use_all_grids: Same.
    :param grid_id: Same.
    :param wgrib_exe_name: Same.
    :param wgrib2_exe_name: Same.
    :param raise_error_if_missing: Same.
    :return: interp_table: pandas DataFrame, where each column is one field and
        each row is one target point.  Column names are from the list
    """

    sounding_field_names, sounding_field_names_grib1 = (
        _get_nwp_fields_for_sounding(
            model_name=model_name, return_table=False,
            include_surface=include_surface
        )[:2]
    )

    return interp.interp_nwp_from_xy_grid(
        query_point_table=target_point_table, field_names=sounding_field_names,
        field_names_grib1=sounding_field_names_grib1, model_name=model_name,
        top_grib_directory_name=top_grib_directory_name,
        use_all_grids=use_all_grids, grid_id=grid_id,
        temporal_interp_method_string=interp.PREV_NEIGHBOUR_METHOD_STRING,
        spatial_interp_method_string=interp.NEAREST_NEIGHBOUR_METHOD_STRING,
        wgrib_exe_name=wgrib_exe_name, wgrib2_exe_name=wgrib2_exe_name,
        raise_error_if_missing=raise_error_if_missing)


def _convert_interp_table_to_soundings(
        interp_table, target_point_table, model_name, include_surface=False,
        minimum_pressure_mb=0.):
    """Converts table of interpolated values to list of soundings.

    :param interp_table: N-row pandas DataFrame created by
        `_interp_soundings_from_nwp`.
    :param target_point_table: N-row pandas DataFrame created by
        `_create_target_points_for_interp`.
    :param model_name: Model name (must be accepted by
        `nwp_model_utils.check_model_name`).
    :param include_surface: See doc for `_get_nwp_fields_for_sounding`.
    :param minimum_pressure_mb: Same.
    :return: sounding_dict_pressure_coords: Dictionary with the following keys.
    sounding_dict_pressure_coords['full_storm_id_strings']: length-N list of
        full IDs.
    sounding_dict_pressure_coords['init_times_unix_sec']: length-N numpy array
        of initial times (storm times).  Valid time = initial time + lead time.
    sounding_dict_pressure_coords['lead_times_seconds']: length-N numpy array of
        lead times.
    sounding_dict_pressure_coords['sounding_matrix']: N-by-P-by-F numpy array of
        sounding values.
    sounding_dict_pressure_coords['surface_pressures_mb']: length-N numpy array
        with surface pressure (millibars) for each storm object.  If
        `include_surface = False`, this is `None`.
    sounding_dict_pressure_coords['pressure_levels_mb']: length-P numpy array of
        pressure levels (millibars).  The surface is denoted by NaN.
    sounding_dict_pressure_coords['field_names']: length-F list of field names.
    """

    sounding_field_name_table = _get_nwp_fields_for_sounding(
        model_name=model_name, return_table=True,
        include_surface=include_surface,
        minimum_pressure_mb=minimum_pressure_mb
    )[-1]

    if include_surface:
        surface_pressure_name = nwp_model_utils.get_lowest_pressure_name(
            model_name
        )[0]

        surface_pressures_mb = (
            PASCALS_TO_MB * interp_table[surface_pressure_name].values
        )
    else:
        surface_pressures_mb = None

    field_names = list(sounding_field_name_table)
    field_names.remove(PRESSURE_LEVEL_KEY)

    pressure_levels_with_surface_mb = sounding_field_name_table[
        PRESSURE_LEVEL_KEY
    ].values

    num_fields = len(field_names)
    num_pressure_levels = len(sounding_field_name_table.index)
    num_storm_objects = len(interp_table.index)

    sounding_matrix = numpy.full(
        (num_storm_objects, num_pressure_levels, num_fields), numpy.nan
    )

    for j in range(num_pressure_levels):
        for k in range(num_fields):
            this_field_name = (
                sounding_field_name_table[field_names[k]].values[j]
            )
            sounding_matrix[:, j, k] = interp_table[this_field_name].values

    return {
        FULL_IDS_KEY:
            target_point_table[tracking_utils.FULL_ID_COLUMN].values.tolist(),
        INITIAL_TIMES_KEY: target_point_table[INITIAL_TIME_COLUMN].values,
        LEAD_TIMES_KEY: target_point_table[LEAD_TIME_KEY].values,
        SOUNDING_MATRIX_KEY: sounding_matrix,
        SURFACE_PRESSURES_KEY: surface_pressures_mb,
        PRESSURE_LEVELS_WITH_SFC_KEY: pressure_levels_with_surface_mb,
        FIELD_NAMES_KEY: field_names
    }


def _get_pressures(sounding_dict):
    """Returns pressure levels in soundings.

    :param sounding_dict: Dictionary created by
        `_convert_interp_table_to_soundings` or `_pressure_to_height_coords`.
    :return: pressure_matrix_pascals: N-by-P numpy array of pressures.
    """

    if PRESSURE_LEVELS_WITH_SFC_KEY not in sounding_dict:
        pressure_index = sounding_dict[FIELD_NAMES_KEY].index(PRESSURE_NAME)
        return sounding_dict[SOUNDING_MATRIX_KEY][..., pressure_index]

    num_storm_objects = sounding_dict[SOUNDING_MATRIX_KEY].shape[0]
    num_pressure_levels = sounding_dict[SOUNDING_MATRIX_KEY].shape[1]
    pressure_matrix_pascals = numpy.full(
        (num_storm_objects, num_pressure_levels), numpy.nan)

    for i in range(num_storm_objects):
        pressure_matrix_pascals[i, :] = sounding_dict[
            PRESSURE_LEVELS_WITH_SFC_KEY]

    if sounding_dict[SURFACE_PRESSURES_KEY] is not None:
        surface_index = numpy.where(
            numpy.isnan(sounding_dict[PRESSURE_LEVELS_WITH_SFC_KEY]))[0][0]
        pressure_matrix_pascals[:, surface_index] = sounding_dict[
            SURFACE_PRESSURES_KEY]

    return MB_TO_PASCALS * pressure_matrix_pascals


def _relative_to_specific_humidity(sounding_dict, pressure_matrix_pascals):
    """Converts relative to specific humidity in each sounding.

    :param sounding_dict: Dictionary created by
        `_convert_interp_table_to_soundings` or `_pressure_to_height_coords`.
    :param pressure_matrix_pascals: N-by-P numpy array of pressures.
    :return: sounding_dict: Same as input, with the following exceptions.
    [1] contains specific humidity
    [2] relative humidity is in 0...1, rather than a percentage

    :return: dewpoint_matrix_kelvins: N-by-P numpy array of dewpoints.
    """

    field_names = sounding_dict[FIELD_NAMES_KEY]
    sounding_matrix = sounding_dict[SOUNDING_MATRIX_KEY]
    temperature_index = field_names.index(TEMPERATURE_NAME)

    if nwp_model_utils.RH_COLUMN_FOR_SOUNDING_TABLES in field_names:
        relative_humidity_index = field_names.index(
            nwp_model_utils.RH_COLUMN_FOR_SOUNDING_TABLES)

        sounding_matrix[..., relative_humidity_index] = (
            PERCENT_TO_UNITLESS * sounding_matrix[..., relative_humidity_index]
        )

        field_names[relative_humidity_index] = RELATIVE_HUMIDITY_NAME
    else:
        relative_humidity_index = field_names.index(RELATIVE_HUMIDITY_NAME)

    dewpoint_matrix_kelvins = (
        moisture_conversions.relative_humidity_to_dewpoint(
            relative_humidities=sounding_matrix[..., relative_humidity_index],
            temperatures_kelvins=sounding_matrix[..., temperature_index],
            total_pressures_pascals=pressure_matrix_pascals)
    )

    spec_humidity_matrix_kg_kg01 = (
        moisture_conversions.dewpoint_to_specific_humidity(
            dewpoints_kelvins=dewpoint_matrix_kelvins,
            total_pressures_pascals=pressure_matrix_pascals)
    )

    if SPECIFIC_HUMIDITY_NAME in field_names:
        sounding_matrix[
            ..., field_names.index(SPECIFIC_HUMIDITY_NAME)
        ] = spec_humidity_matrix_kg_kg01
    else:
        field_names.append(SPECIFIC_HUMIDITY_NAME)

        spec_humidity_matrix_kg_kg01 = numpy.reshape(
            spec_humidity_matrix_kg_kg01,
            spec_humidity_matrix_kg_kg01.shape + (1,)
        )

        sounding_matrix = numpy.concatenate(
            (sounding_matrix, spec_humidity_matrix_kg_kg01), axis=-1
        )

    sounding_dict[FIELD_NAMES_KEY] = field_names
    sounding_dict[SOUNDING_MATRIX_KEY] = sounding_matrix
    return sounding_dict, dewpoint_matrix_kelvins


def _specific_to_relative_humidity(sounding_dict, pressure_matrix_pascals):
    """Converts specific to relative humidity in each sounding.

    :param sounding_dict: Dictionary created by
        `_convert_interp_table_to_soundings` or `_pressure_to_height_coords`.
    :param pressure_matrix_pascals: N-by-P numpy array of pressures.
    :return: sounding_dict: Same as input, but including relative humidity.
    :return: dewpoint_matrix_kelvins: N-by-P numpy array of dewpoints.
    """

    field_names = sounding_dict[FIELD_NAMES_KEY]
    sounding_matrix = sounding_dict[SOUNDING_MATRIX_KEY]
    temperature_index = field_names.index(TEMPERATURE_NAME)
    specific_humidity_index = field_names.index(SPECIFIC_HUMIDITY_NAME)

    dewpoint_matrix_kelvins = (
        moisture_conversions.specific_humidity_to_dewpoint(
            specific_humidities_kg_kg01=sounding_matrix[
                ..., specific_humidity_index],
            total_pressures_pascals=pressure_matrix_pascals)
    )

    relative_humidity_matrix = (
        moisture_conversions.dewpoint_to_relative_humidity(
            dewpoints_kelvins=dewpoint_matrix_kelvins,
            temperatures_kelvins=sounding_matrix[..., temperature_index],
            total_pressures_pascals=pressure_matrix_pascals)
    )

    if RELATIVE_HUMIDITY_NAME in field_names:
        sounding_matrix[
            ..., field_names.index(RELATIVE_HUMIDITY_NAME)
        ] = relative_humidity_matrix
    else:
        field_names.append(RELATIVE_HUMIDITY_NAME)

        relative_humidity_matrix = numpy.reshape(
            relative_humidity_matrix, relative_humidity_matrix.shape + (1,)
        )

        sounding_matrix = numpy.concatenate(
            (sounding_matrix, relative_humidity_matrix), axis=-1
        )

    sounding_dict[FIELD_NAMES_KEY] = field_names
    sounding_dict[SOUNDING_MATRIX_KEY] = sounding_matrix
    return sounding_dict, dewpoint_matrix_kelvins


def _get_virtual_potential_temperatures(
        sounding_dict, pressure_matrix_pascals, dewpoint_matrix_kelvins):
    """Adds virtual potential temperature to each sounding.

    :param sounding_dict: Dictionary created by
        `_convert_interp_table_to_soundings` or `_pressure_to_height_coords`.
    :param pressure_matrix_pascals: N-by-P numpy array of pressures.
    :param dewpoint_matrix_kelvins: N-by-P numpy array of dewpoints.
    :return: sounding_dict: Same as input, but including virtual potential
        temperature.
    """

    field_names = sounding_dict[FIELD_NAMES_KEY]
    sounding_matrix = sounding_dict[SOUNDING_MATRIX_KEY]
    temperature_index = field_names.index(TEMPERATURE_NAME)

    vapour_pressure_matrix_pascals = (
        moisture_conversions.dewpoint_to_vapour_pressure(
            dewpoint_matrix_kelvins)
    )

    virtual_temperature_matrix_kelvins = (
        moisture_conversions.temperature_to_virtual_temperature(
            temperatures_kelvins=sounding_matrix[..., temperature_index],
            total_pressures_pascals=pressure_matrix_pascals,
            vapour_pressures_pascals=vapour_pressure_matrix_pascals)
    )

    theta_v_matrix_kelvins = (
        temperature_conversions.temperatures_to_potential_temperatures(
            temperatures_kelvins=virtual_temperature_matrix_kelvins,
            total_pressures_pascals=pressure_matrix_pascals)
    )

    if VIRTUAL_POTENTIAL_TEMPERATURE_NAME in field_names:
        sounding_matrix[
            ..., field_names.index(VIRTUAL_POTENTIAL_TEMPERATURE_NAME)
        ] = theta_v_matrix_kelvins
    else:
        field_names.append(VIRTUAL_POTENTIAL_TEMPERATURE_NAME)

        theta_v_matrix_kelvins = numpy.reshape(
            theta_v_matrix_kelvins, theta_v_matrix_kelvins.shape + (1,)
        )

        sounding_matrix = numpy.concatenate(
            (sounding_matrix, theta_v_matrix_kelvins), axis=-1
        )

    sounding_dict[FIELD_NAMES_KEY] = field_names
    sounding_dict[SOUNDING_MATRIX_KEY] = sounding_matrix
    return sounding_dict


def _fill_nans_in_soundings(
        sounding_dict_pressure_coords, pressure_matrix_pascals,
        min_num_pressure_levels_without_nan=15):
    """Interpolates to fill NaN's in each sounding.

    :param sounding_dict_pressure_coords: See doc for
        `_convert_interp_table_to_soundings`.
    :param pressure_matrix_pascals: N-by-P numpy array of pressures.
    :param min_num_pressure_levels_without_nan: Minimum number of pressure
        levels without NaN.  For a given sounding S, if any field has fewer
        pressure levels without NaN, S will be thrown out.
    :return: sounding_dict_pressure_coords: Same as input, with the following
        exceptions.
    [1] maybe fewer soundings
    [2] NaN's have been replaced
    """

    # TODO(thunderhoser): Remove surface pressure of NaN.

    field_names = sounding_dict_pressure_coords[FIELD_NAMES_KEY]
    sounding_matrix = sounding_dict_pressure_coords[SOUNDING_MATRIX_KEY]
    height_index = field_names.index(GEOPOTENTIAL_HEIGHT_NAME)

    num_soundings = sounding_matrix.shape[0]
    keep_sounding_flags = numpy.full(num_soundings, True, dtype=bool)

    field_names_to_interp = [
        GEOPOTENTIAL_HEIGHT_NAME, U_WIND_NAME, V_WIND_NAME, TEMPERATURE_NAME,
        SPECIFIC_HUMIDITY_NAME
    ]

    for i in range(num_soundings):
        for this_field_name in field_names_to_interp:
            this_field_index = field_names.index(this_field_name)
            these_real_flags = numpy.invert(numpy.isnan(
                sounding_matrix[i, :, this_field_index]
            ))

            if numpy.all(these_real_flags):
                continue

            if (numpy.sum(these_real_flags) <
                    min_num_pressure_levels_without_nan):
                keep_sounding_flags[i] = False
                break

            these_nan_indices = numpy.where(numpy.invert(these_real_flags))[0]
            these_real_indices = numpy.where(these_real_flags)[0]

            if this_field_name == GEOPOTENTIAL_HEIGHT_NAME:
                interp_object = scipy_interp1d(
                    x=numpy.log(pressure_matrix_pascals[i, these_real_indices]),
                    y=sounding_matrix[i, these_real_indices, this_field_index],
                    kind='linear', bounds_error=False, fill_value='extrapolate',
                    assume_sorted=False)

                sounding_matrix[i, these_nan_indices, this_field_index] = (
                    interp_object(
                        numpy.log(pressure_matrix_pascals[i, these_nan_indices])
                    )
                )
            else:
                interp_object = scipy_interp1d(
                    x=sounding_matrix[i, these_real_indices, height_index],
                    y=sounding_matrix[i, these_real_indices, this_field_index],
                    kind='linear', bounds_error=False, fill_value='extrapolate',
                    assume_sorted=False)

                sounding_matrix[i, these_nan_indices, this_field_index] = (
                    interp_object(
                        sounding_matrix[i, these_nan_indices, height_index]
                    )
                )

    keep_sounding_indices = numpy.where(keep_sounding_flags)[0]
    sounding_dict_pressure_coords[SOUNDING_MATRIX_KEY] = (
        sounding_matrix[keep_sounding_indices, ...]
    )
    sounding_dict_pressure_coords[FULL_IDS_KEY] = [
        sounding_dict_pressure_coords[FULL_IDS_KEY][i]
        for i in keep_sounding_indices
    ]

    sounding_dict_pressure_coords[INITIAL_TIMES_KEY] = (
        sounding_dict_pressure_coords[INITIAL_TIMES_KEY][keep_sounding_indices]
    )
    sounding_dict_pressure_coords[LEAD_TIMES_KEY] = (
        sounding_dict_pressure_coords[LEAD_TIMES_KEY][keep_sounding_indices]
    )

    if sounding_dict_pressure_coords[SURFACE_PRESSURES_KEY] is not None:
        sounding_dict_pressure_coords[SURFACE_PRESSURES_KEY] = (
            sounding_dict_pressure_coords[SURFACE_PRESSURES_KEY][
                keep_sounding_indices]
        )

    return sounding_dict_pressure_coords


def _convert_fields_and_units(sounding_dict_pressure_coords):
    """Converts fields and units in each sounding.

    :param sounding_dict_pressure_coords: See doc for
        `_convert_interp_table_to_soundings`.
    :return: sounding_dict_pressure_coords: Same as input, but with different
        fields and units.
    """

    pressure_matrix_pascals = _get_pressures(sounding_dict_pressure_coords)
    found_rh = (
        nwp_model_utils.RH_COLUMN_FOR_SOUNDING_TABLES in
        sounding_dict_pressure_coords[FIELD_NAMES_KEY]
    )

    if found_rh:
        sounding_dict_pressure_coords, dewpoint_matrix_kelvins = (
            _relative_to_specific_humidity(
                sounding_dict=sounding_dict_pressure_coords,
                pressure_matrix_pascals=pressure_matrix_pascals)
        )

    sounding_dict_pressure_coords = _fill_nans_in_soundings(
        sounding_dict_pressure_coords=sounding_dict_pressure_coords,
        pressure_matrix_pascals=pressure_matrix_pascals)

    pressure_matrix_pascals = _get_pressures(sounding_dict_pressure_coords)

    if found_rh:
        specific_humidity_index = sounding_dict_pressure_coords[
            FIELD_NAMES_KEY
        ].index(SPECIFIC_HUMIDITY_NAME)

        dewpoint_matrix_kelvins = (
            moisture_conversions.specific_humidity_to_dewpoint(
                specific_humidities_kg_kg01=sounding_dict_pressure_coords[
                    SOUNDING_MATRIX_KEY][..., specific_humidity_index],
                total_pressures_pascals=pressure_matrix_pascals)
        )
    else:
        sounding_dict_pressure_coords, dewpoint_matrix_kelvins = (
            _specific_to_relative_humidity(
                sounding_dict=sounding_dict_pressure_coords,
                pressure_matrix_pascals=pressure_matrix_pascals)
        )

    return _get_virtual_potential_temperatures(
        sounding_dict=sounding_dict_pressure_coords,
        pressure_matrix_pascals=pressure_matrix_pascals,
        dewpoint_matrix_kelvins=dewpoint_matrix_kelvins)


def _pressure_to_height_coords(
        sounding_dict_pressure_coords, height_levels_m_agl):
    """Converts soundings from pressure coords to ground-relative height coords.

    :param sounding_dict_pressure_coords: Dictionary created by
        `_convert_fields_and_units`, but with additional keys listed below.
    sounding_dict_pressure_coords['storm_elevations_m_asl']: length-N numpy
        array of storm elevations (metres above sea level).

    :param height_levels_m_agl: length-H numpy array of height levels (integer
        metres above ground level).  This method will interpolate each sounding
        to said heights, and the output soundings will be in ground-relative
        height coords rather than pressure coords.
    :return: sounding_dict_height_coords: Dictionary with the following keys.
    sounding_dict_height_coords['full_storm_id_strings']: length-N list of full
        IDs.
    sounding_dict_height_coords['init_times_unix_sec']: length-N numpy array of
        initial times (storm times).  Valid time = initial time + lead time.
    sounding_dict_height_coords['lead_times_seconds']: length-N numpy array of
        lead times.
    sounding_dict_height_coords['storm_elevations_m_asl']: length-N numpy array
        of storm elevations (metres above sea level).
    sounding_dict_height_coords['sounding_matrix']: N-by-P-by-F numpy array of
        sounding values.
    sounding_dict_height_coords['height_levels_m_agl']: length-H numpy array of
        height levels (metres above ground level).
    sounding_dict_height_coords['field_names']: length-F list of field names.
    """

    error_checking.assert_is_numpy_array(height_levels_m_agl, num_dimensions=1)
    error_checking.assert_is_geq_numpy_array(height_levels_m_agl, 0)
    height_levels_m_agl = numpy.round(height_levels_m_agl).astype(int)

    sounding_dict_height_coords = {
        FULL_IDS_KEY: sounding_dict_pressure_coords[FULL_IDS_KEY],
        INITIAL_TIMES_KEY: sounding_dict_pressure_coords[INITIAL_TIMES_KEY],
        LEAD_TIMES_KEY: sounding_dict_pressure_coords[LEAD_TIMES_KEY],
        STORM_ELEVATIONS_KEY:
            sounding_dict_pressure_coords[STORM_ELEVATIONS_KEY],
        SOUNDING_MATRIX_KEY: sounding_dict_pressure_coords[SOUNDING_MATRIX_KEY],
        HEIGHT_LEVELS_KEY: height_levels_m_agl,
        FIELD_NAMES_KEY: sounding_dict_pressure_coords[FIELD_NAMES_KEY]
    }

    field_names = sounding_dict_pressure_coords[FIELD_NAMES_KEY]
    orig_sounding_matrix = sounding_dict_pressure_coords[SOUNDING_MATRIX_KEY]
    storm_elevations_m_asl = sounding_dict_pressure_coords[STORM_ELEVATIONS_KEY]

    height_index = field_names.index(GEOPOTENTIAL_HEIGHT_NAME)
    orig_height_matrix_m_asl = orig_sounding_matrix[..., height_index] + 0.

    pressure_matrix_pascals = _get_pressures(sounding_dict_pressure_coords)
    field_names[height_index] = PRESSURE_NAME
    orig_sounding_matrix[..., height_index] = pressure_matrix_pascals + 0.

    field_names_to_interp = [
        PRESSURE_NAME, SPECIFIC_HUMIDITY_NAME, TEMPERATURE_NAME, U_WIND_NAME,
        V_WIND_NAME
    ]

    num_soundings = orig_sounding_matrix.shape[0]
    num_fields = orig_sounding_matrix.shape[-1]
    num_height_levels = len(height_levels_m_agl)

    new_sounding_matrix = numpy.full(
        (num_soundings, num_height_levels, num_fields), numpy.nan
    )

    pressure_index = field_names.index(PRESSURE_NAME)

    for j in range(len(field_names_to_interp)):
        this_field_index = field_names.index(field_names_to_interp[j])

        for i in range(num_soundings):
            if field_names_to_interp[j] == PRESSURE_NAME:
                this_interp_object = scipy_interp1d(
                    x=orig_height_matrix_m_asl[i, ...],
                    y=numpy.log(orig_sounding_matrix[i, ..., this_field_index]),
                    kind='linear', bounds_error=False, fill_value='extrapolate',
                    assume_sorted=False)

                new_sounding_matrix[i, ..., this_field_index] = numpy.exp(
                    this_interp_object(
                        storm_elevations_m_asl[i] + height_levels_m_agl)
                )
            else:
                this_interp_object = scipy_interp1d(
                    x=orig_sounding_matrix[i, ..., pressure_index],
                    y=orig_sounding_matrix[i, ..., this_field_index],
                    kind='linear', bounds_error=False, fill_value='extrapolate',
                    assume_sorted=False)

                new_sounding_matrix[i, ..., this_field_index] = (
                    this_interp_object(
                        new_sounding_matrix[i, ..., pressure_index]
                    )
                )

    sounding_dict_height_coords[FIELD_NAMES_KEY] = field_names
    sounding_dict_height_coords[SOUNDING_MATRIX_KEY] = new_sounding_matrix

    pressure_matrix_pascals = _get_pressures(sounding_dict_height_coords)

    sounding_dict_height_coords, dewpoint_matrix_kelvins = (
        _specific_to_relative_humidity(
            sounding_dict=sounding_dict_height_coords,
            pressure_matrix_pascals=pressure_matrix_pascals)
    )

    return _get_virtual_potential_temperatures(
        sounding_dict=sounding_dict_height_coords,
        pressure_matrix_pascals=pressure_matrix_pascals,
        dewpoint_matrix_kelvins=dewpoint_matrix_kelvins)


def check_field_name(field_name):
    """Error-checks name of sounding field.

    :param field_name: Name of sounding field.
    :raises: ValueError: if `field_name not in VALID_FIELD_NAMES`.
    """

    error_checking.assert_is_string(field_name)

    if field_name not in VALID_FIELD_NAMES:
        error_string = (
            '\n{0:s}\nValid field names (listed above) do not include "{1:s}".'
        ).format(VALID_FIELD_NAMES, field_name)

        raise ValueError(error_string)


def field_name_to_verbose(field_name, include_units=True):
    """Converts field name from underscore-separated format to verbose.

    :param field_name: Field name in default (underscore-separated format).
    :param include_units: Boolean flag.  If True, verbose name will include
        units.
    :return: field_name_verbose: Verbose field name.
    """

    error_checking.assert_is_boolean(include_units)
    field_name_verbose = FIELD_NAME_TO_VERBOSE_DICT[field_name]

    if include_units:
        return field_name_verbose

    return field_name_verbose[:field_name_verbose.find(' (')]


def interp_soundings_to_storm_objects(
        storm_object_table, top_grib_directory_name, model_name,
        use_all_grids=True, grid_id=None,
        height_levels_m_agl=DEFAULT_HEIGHT_LEVELS_M_AGL,
        lead_times_seconds=DEFAULT_LEAD_TIMES_SEC,
        lag_time_for_convective_contamination_sec=
        DEFAULT_LAG_TIME_FOR_CONVECTIVE_CONTAMINATION_SEC,
        wgrib_exe_name=grib_io.WGRIB_EXE_NAME_DEFAULT,
        wgrib2_exe_name=grib_io.WGRIB2_EXE_NAME_DEFAULT,
        raise_error_if_missing=False):
    """Interpolates NWP sounding to each storm object at each lead time.

    :param storm_object_table: pandas DataFrame with columns listed in
        `storm_tracking_io.write_file`.
    :param top_grib_directory_name: Name of top-level directory with grib files
        for the given NWP model.
    :param model_name: Model name (must be accepted by
        `nwp_model_utils.check_grid_id`).
    :param use_all_grids: Boolean flag.  If True, this method will interp from
        the highest-resolution grid available at each model-initialization time.
        If False, will use only `grid_id`.
    :param grid_id: [used only if `use_all_grids = False`]
        Grid ID (must be accepted by `nwp_model_utils.check_grid_id`).
    :param height_levels_m_agl: 1-D numpy array of height levels (metres above
        ground level).  These will be the height levels in each sounding.
    :param lead_times_seconds: length-T numpy array of lead times.
    :param lag_time_for_convective_contamination_sec: Lag time (used to avoid
        convective contamination of soundings, where the sounding for storm S is
        heavily influenced by storm S).  This will be subtracted from each lead
        time.
    :param wgrib_exe_name: Path to wgrib executable.
    :param wgrib2_exe_name: Path to wgrib2 executable.
    :param raise_error_if_missing: Boolean flag.  If any grib file is missing
        and `raise_error_if_missing = True`, this method will error out.  If any
        grib file is missing and `raise_error_if_missing = False`, this method
        will carry on, leaving the affected values as NaN.
    :return: sounding_dict_by_lead_time: length-T list of dictionaries, each
        containing the keys listed in `_pressure_to_height_coords`.
    """

    error_checking.assert_is_integer_numpy_array(lead_times_seconds)
    error_checking.assert_is_numpy_array(lead_times_seconds, num_dimensions=1)
    error_checking.assert_is_geq_numpy_array(lead_times_seconds, 0)
    error_checking.assert_is_integer(lag_time_for_convective_contamination_sec)
    error_checking.assert_is_geq(lag_time_for_convective_contamination_sec, 0)

    good_indices = numpy.where(numpy.invert(numpy.logical_or(
        numpy.isnan(
            storm_object_table[tracking_utils.EAST_VELOCITY_COLUMN].values),
        numpy.isnan(
            storm_object_table[tracking_utils.NORTH_VELOCITY_COLUMN].values)
    )))[0]

    storm_object_table = storm_object_table.iloc[good_indices]

    print (
        'Creating target point for each storm object and lead time ({0:s} '
        'seconds)...'
    ).format(str(lead_times_seconds))

    target_point_table = _create_target_points_for_interp(
        storm_object_table=storm_object_table,
        lead_times_seconds=lead_times_seconds)

    print (
        'Subtracting lag time ({0:d} seconds) from each target point, to '
        'account for convective contamination...'
    ).format(lag_time_for_convective_contamination_sec)

    target_point_table[
        FORECAST_TIME_COLUMN
    ] -= lag_time_for_convective_contamination_sec

    column_dict_old_to_new = {
        tracking_utils.CENTROID_LATITUDE_COLUMN: interp.QUERY_LAT_COLUMN,
        tracking_utils.CENTROID_LONGITUDE_COLUMN: interp.QUERY_LNG_COLUMN,
        FORECAST_TIME_COLUMN: interp.QUERY_TIME_COLUMN
    }

    target_point_table.rename(columns=column_dict_old_to_new, inplace=True)

    print SEPARATOR_STRING
    interp_table = _interp_soundings_from_nwp(
        target_point_table=target_point_table,
        top_grib_directory_name=top_grib_directory_name, include_surface=False,
        model_name=model_name, use_all_grids=use_all_grids, grid_id=grid_id,
        wgrib_exe_name=wgrib_exe_name, wgrib2_exe_name=wgrib2_exe_name,
        raise_error_if_missing=raise_error_if_missing)
    print SEPARATOR_STRING

    print 'Converting interpolated values to soundings...'
    sounding_dict_pressure_coords = _convert_interp_table_to_soundings(
        interp_table=interp_table, target_point_table=target_point_table,
        model_name=model_name, include_surface=False)

    print 'Converting fields and units in each sounding...'
    orig_num_soundings = len(sounding_dict_pressure_coords[FULL_IDS_KEY])
    sounding_dict_pressure_coords = _convert_fields_and_units(
        sounding_dict_pressure_coords)
    num_soundings = len(sounding_dict_pressure_coords[FULL_IDS_KEY])

    print 'Removed {0:d} of {1:d} soundings (too many NaN''s).'.format(
        orig_num_soundings - num_soundings, orig_num_soundings)

    print 'Finding elevation of each storm object...'
    storm_elevations_m_asl = geodetic_utils.get_elevations(
        latitudes_deg=storm_object_table[
            tracking_utils.CENTROID_LATITUDE_COLUMN].values,
        longitudes_deg=storm_object_table[
            tracking_utils.CENTROID_LONGITUDE_COLUMN].values,
        working_dir_name=ELEVATION_DIR_NAME)

    these_indices = tracking_utils.find_storm_objects(
        all_id_strings=storm_object_table[
            tracking_utils.FULL_ID_COLUMN].values.tolist(),
        all_times_unix_sec=storm_object_table[
            tracking_utils.VALID_TIME_COLUMN].values,
        id_strings_to_keep=sounding_dict_pressure_coords[FULL_IDS_KEY],
        times_to_keep_unix_sec=sounding_dict_pressure_coords[INITIAL_TIMES_KEY]
    )

    storm_elevations_m_asl = storm_elevations_m_asl[these_indices]
    sounding_dict_pressure_coords.update({
        STORM_ELEVATIONS_KEY: storm_elevations_m_asl
    })

    print 'Converting soundings from pressure coords to metres AGL...\n'
    sounding_dict_height_coords = _pressure_to_height_coords(
        sounding_dict_pressure_coords=sounding_dict_pressure_coords,
        height_levels_m_agl=height_levels_m_agl)

    num_lead_times = len(lead_times_seconds)
    sounding_dict_by_lead_time = [None] * num_lead_times

    for k in range(num_lead_times):
        print (
            'Creating separate sounding dictionary for {0:d}-second lead '
            'time...'
        ).format(lead_times_seconds[k])

        these_indices = numpy.where(
            sounding_dict_height_coords[LEAD_TIMES_KEY] ==
            lead_times_seconds[k]
        )[0]

        sounding_dict_by_lead_time[k] = {
            FULL_IDS_KEY: [
                sounding_dict_height_coords[FULL_IDS_KEY][i]
                for i in these_indices
            ],
            INITIAL_TIMES_KEY:
                sounding_dict_height_coords[INITIAL_TIMES_KEY][these_indices],
            LEAD_TIMES_KEY:
                sounding_dict_height_coords[LEAD_TIMES_KEY][these_indices],
            STORM_ELEVATIONS_KEY:
                sounding_dict_height_coords[STORM_ELEVATIONS_KEY][
                    these_indices],
            SOUNDING_MATRIX_KEY:
                sounding_dict_height_coords[SOUNDING_MATRIX_KEY][
                    these_indices, ...],
            HEIGHT_LEVELS_KEY: sounding_dict_height_coords[HEIGHT_LEVELS_KEY],
            FIELD_NAMES_KEY: sounding_dict_height_coords[FIELD_NAMES_KEY]
        }

        print (
            'Dictionary for {0:d}-second lead time contains {1:d} of {2:d} '
            'soundings.'
        ).format(lead_times_seconds[k], len(these_indices), num_soundings)

    return sounding_dict_by_lead_time


def write_soundings(
        netcdf_file_name, sounding_dict_height_coords, lead_time_seconds,
        lag_time_for_convective_contamination_sec):
    """Writes soundings to NetCDF file.

    This file may contain soundings with one lead time only.

    :param netcdf_file_name: Path to output file.
    :param sounding_dict_height_coords: Dictionary created by
        `interp_soundings_to_storm_objects`.
    :param lead_time_seconds: Lead time for all soundings.
    :param lag_time_for_convective_contamination_sec: Lag time for all soundings
        (see doc for `interp_soundings_to_storm_objects`).
    :raises: ValueError: if `sounding_dict_height_coords` contains more than one
        unique lead time.
    :raises: ValueError: if lead time in `sounding_dict_height_coords` does not
        match the input arg `lead_time_seconds`.
    """

    error_checking.assert_is_integer(lead_time_seconds)
    error_checking.assert_is_geq(lead_time_seconds, 0)
    error_checking.assert_is_integer(lag_time_for_convective_contamination_sec)
    error_checking.assert_is_geq(lag_time_for_convective_contamination_sec, 0)

    unique_lead_times_seconds = numpy.unique(
        sounding_dict_height_coords[LEAD_TIMES_KEY]
    )

    if not numpy.all(unique_lead_times_seconds == lead_time_seconds):
        error_string = (
            'All lead times in sounding dictionary should be {0:d} seconds.  '
            'Instead, got lead times listed below.\n{1:s}'
        ).format(lead_time_seconds, str(unique_lead_times_seconds))

        raise ValueError(error_string)

    # Create file and set global attributes.
    file_system_utils.mkdir_recursive_if_necessary(file_name=netcdf_file_name)
    netcdf_dataset = netCDF4.Dataset(
        netcdf_file_name, 'w', format='NETCDF3_64BIT_OFFSET')

    netcdf_dataset.setncattr(LEAD_TIME_KEY, lead_time_seconds)
    netcdf_dataset.setncattr(
        LAG_TIME_KEY, lag_time_for_convective_contamination_sec)

    num_storm_objects = len(sounding_dict_height_coords[FULL_IDS_KEY])
    num_height_levels = len(sounding_dict_height_coords[HEIGHT_LEVELS_KEY])
    num_fields = len(sounding_dict_height_coords[FIELD_NAMES_KEY])

    netcdf_dataset.createDimension(
        STORM_OBJECT_DIMENSION_KEY, num_storm_objects)
    netcdf_dataset.createDimension(HEIGHT_DIMENSION_KEY, num_height_levels)
    netcdf_dataset.createDimension(FIELD_DIMENSION_KEY, num_fields)

    storm_id_lengths = [
        len(f) for f in sounding_dict_height_coords[FULL_IDS_KEY]
    ]

    num_storm_id_chars = max(storm_id_lengths + [1])
    netcdf_dataset.createDimension(
        STORM_ID_CHAR_DIMENSION_KEY, num_storm_id_chars)

    num_field_name_chars = max([
        len(f) for f in sounding_dict_height_coords[FIELD_NAMES_KEY]
    ])
    netcdf_dataset.createDimension(
        FIELD_NAME_CHAR_DIMENSION_KEY, num_field_name_chars)

    # Add storm IDs to file.
    netcdf_dataset.createVariable(
        FULL_IDS_KEY, datatype='S1',
        dimensions=(STORM_OBJECT_DIMENSION_KEY, STORM_ID_CHAR_DIMENSION_KEY)
    )

    string_type = 'S{0:d}'.format(num_storm_id_chars)
    storm_ids_as_char_array = netCDF4.stringtochar(numpy.array(
        sounding_dict_height_coords[FULL_IDS_KEY], dtype=string_type
    ))
    netcdf_dataset.variables[FULL_IDS_KEY][:] = numpy.array(
        storm_ids_as_char_array)

    # Add initial times (storm times) to file.
    netcdf_dataset.createVariable(
        INITIAL_TIMES_KEY, datatype=numpy.int32,
        dimensions=STORM_OBJECT_DIMENSION_KEY)
    netcdf_dataset.variables[INITIAL_TIMES_KEY][:] = (
        sounding_dict_height_coords[INITIAL_TIMES_KEY]
    )

    # Add storm elevations to file.
    netcdf_dataset.createVariable(
        STORM_ELEVATIONS_KEY, datatype=numpy.float32,
        dimensions=STORM_OBJECT_DIMENSION_KEY)
    netcdf_dataset.variables[STORM_ELEVATIONS_KEY][:] = (
        sounding_dict_height_coords[STORM_ELEVATIONS_KEY]
    )

    # Add height levels to file.
    netcdf_dataset.createVariable(
        HEIGHT_LEVELS_KEY, datatype=numpy.int32,
        dimensions=HEIGHT_DIMENSION_KEY)
    netcdf_dataset.variables[HEIGHT_LEVELS_KEY][:] = (
        sounding_dict_height_coords[HEIGHT_LEVELS_KEY]
    )

    # Add field names to file.
    netcdf_dataset.createVariable(
        FIELD_NAMES_KEY, datatype='S1',
        dimensions=(FIELD_DIMENSION_KEY, FIELD_NAME_CHAR_DIMENSION_KEY)
    )

    string_type = 'S{0:d}'.format(num_field_name_chars)
    field_names_as_char_array = netCDF4.stringtochar(numpy.array(
        sounding_dict_height_coords[FIELD_NAMES_KEY], dtype=string_type
    ))
    netcdf_dataset.variables[FIELD_NAMES_KEY][:] = numpy.array(
        field_names_as_char_array)

    # Add soundings to file.
    netcdf_dataset.createVariable(
        SOUNDING_MATRIX_KEY, datatype=numpy.float32,
        dimensions=(STORM_OBJECT_DIMENSION_KEY, HEIGHT_DIMENSION_KEY,
                    FIELD_DIMENSION_KEY)
    )

    netcdf_dataset.variables[SOUNDING_MATRIX_KEY][:] = (
        sounding_dict_height_coords[SOUNDING_MATRIX_KEY]
    )
    netcdf_dataset.close()


def read_soundings(
        netcdf_file_name, field_names_to_keep=None,
        full_id_strings_to_keep=None, init_times_to_keep_unix_sec=None):
    """Reads soundings from NetCDF file.

    K = number of storm objects to keep

    If `full_id_strings_to_keep is None or init_times_to_keep_unix_sec is None`,
    this method will return soundings for all storm objects.  Otherwise, will
    return only a subset of storm objects.

    If `field_names_to_keep is None`, this method will return all sounding
    fields.  Otherwise, will return only a subset of fields.

    :param netcdf_file_name: Path to input file.
    :param field_names_to_keep: 1-D list with names of sounding fields.
    :param full_id_strings_to_keep: length-K list of full IDs.
    :param init_times_to_keep_unix_sec: length-K numpy array of initial times
        (storm times).
    :return: sounding_dict_height_coords: Dictionary with keys listed in
        `_pressure_to_height_coords`.
    :return: lag_time_for_convective_contamination_sec: See doc for
        `interp_soundings_to_storm_objects`.
    """

    netcdf_dataset = netcdf_io.open_netcdf(
        netcdf_file_name=netcdf_file_name, raise_error_if_fails=True)

    lead_time_seconds = getattr(netcdf_dataset, LEAD_TIME_KEY)
    lag_time_for_convective_contamination_sec = int(getattr(
        netcdf_dataset, LAG_TIME_KEY
    ))

    height_levels_m_agl = numpy.array(
        netcdf_dataset.variables[HEIGHT_LEVELS_KEY][:], dtype=int
    )
    field_names = netCDF4.chartostring(
        netcdf_dataset.variables[FIELD_NAMES_KEY][:]
    )
    field_names = [str(f) for f in field_names]

    if field_names_to_keep is None:
        field_indices_to_keep = numpy.linspace(
            0, len(field_names) - 1, num=len(field_names), dtype=int
        )
    else:
        error_checking.assert_is_numpy_array(
            numpy.array(field_names_to_keep), num_dimensions=1
        )
        for this_field_name in field_names_to_keep:
            check_field_name(this_field_name)

        field_indices_to_keep = numpy.array(
            [field_names.index(f) for f in field_names_to_keep], dtype=int
        )
        field_names = field_names_to_keep + []

    num_storm_objects = netcdf_dataset.variables[FULL_IDS_KEY].shape[0]
    num_height_levels = len(height_levels_m_agl)
    num_fields = len(field_names)

    if num_storm_objects == 0:
        full_id_strings = []
        init_times_unix_sec = numpy.array([], dtype=int)
        storm_elevations_m_asl = numpy.array([], dtype=float)
        sounding_matrix = numpy.full(
            (num_storm_objects, num_height_levels, num_fields), numpy.nan
        )
    else:
        full_id_strings = netCDF4.chartostring(
            netcdf_dataset.variables[FULL_IDS_KEY][:]
        )
        full_id_strings = [str(this_id) for this_id in full_id_strings]

        init_times_unix_sec = numpy.array(
            netcdf_dataset.variables[INITIAL_TIMES_KEY][:], dtype=int
        )
        storm_elevations_m_asl = numpy.array(
            netcdf_dataset.variables[STORM_ELEVATIONS_KEY][:]
        )
        sounding_matrix = numpy.array(
            netcdf_dataset.variables[SOUNDING_MATRIX_KEY][
                ..., field_indices_to_keep]
        )

    netcdf_dataset.close()

    filter_storm_objects = (
        full_id_strings_to_keep is not None and
        init_times_to_keep_unix_sec is not None and
        num_storm_objects != 0
    )

    if filter_storm_objects:
        these_indices = tracking_utils.find_storm_objects(
            all_id_strings=full_id_strings,
            all_times_unix_sec=init_times_unix_sec,
            id_strings_to_keep=full_id_strings_to_keep,
            times_to_keep_unix_sec=init_times_to_keep_unix_sec,
            allow_missing=True)

        these_indices = these_indices[these_indices != -1]

        full_id_strings = [full_id_strings[i] for i in these_indices]
        init_times_unix_sec = init_times_unix_sec[these_indices]
        storm_elevations_m_asl = storm_elevations_m_asl[these_indices]
        sounding_matrix = sounding_matrix[these_indices, ...]

    num_storm_objects = len(full_id_strings)
    lead_times_seconds = numpy.full(
        num_storm_objects, lead_time_seconds, dtype=int)

    sounding_dict_height_coords = {
        FULL_IDS_KEY: full_id_strings,
        INITIAL_TIMES_KEY: init_times_unix_sec,
        LEAD_TIMES_KEY: lead_times_seconds,
        STORM_ELEVATIONS_KEY: storm_elevations_m_asl,
        SOUNDING_MATRIX_KEY: sounding_matrix,
        HEIGHT_LEVELS_KEY: height_levels_m_agl,
        FIELD_NAMES_KEY: field_names
    }

    return (sounding_dict_height_coords,
            lag_time_for_convective_contamination_sec)


def find_sounding_file(
        top_directory_name, spc_date_string, lead_time_seconds,
        lag_time_for_convective_contamination_sec, init_time_unix_sec=None,
        raise_error_if_missing=True):
    """Finds NetCDF file created by `write_soundings`.

    If `init_time_unix_sec is None`, this method will seek a file with all
    soundings for one SPC date.  Otherwise, will seek a file with soundings for
    one time step.

    :param top_directory_name: Name of top-level directory with sounding files.
    :param spc_date_string: SPC date (format "yyyymmdd").
    :param lead_time_seconds: Lead time.
    :param lag_time_for_convective_contamination_sec: See doc for
        `interp_soundings_to_storm_objects`.
    :param init_time_unix_sec: Initial time (storm time).
    :param raise_error_if_missing: Boolean flag.  If file is missing and
        `raise_error_if_missing = True`, this method will error out.
    :return: sounding_file_name: Path to sounding file.  If file is missing and
        `raise_error_if_missing = False`, this is the *expected* path.
    :raises: ValueError: if file is missing and `raise_error_if_missing = True`.
    """

    error_checking.assert_is_string(top_directory_name)
    time_conversion.spc_date_string_to_unix_sec(spc_date_string)
    error_checking.assert_is_boolean(raise_error_if_missing)

    if init_time_unix_sec is None:
        sounding_file_name = (
            '{0:s}/{1:s}/storm_soundings_{2:s}_lead-time-{3:05d}sec'
            '_lag-time-{4:04d}sec.nc'
        ).format(
            top_directory_name, spc_date_string[:4], spc_date_string,
            lead_time_seconds, lag_time_for_convective_contamination_sec
        )
    else:
        sounding_file_name = (
            '{0:s}/{1:s}/{2:s}/storm_soundings_{3:s}_lead-time-{4:05d}sec'
            '_lag-time-{5:04d}sec.nc'
        ).format(
            top_directory_name, spc_date_string[:4], spc_date_string,
            time_conversion.unix_sec_to_string(
                init_time_unix_sec, TIME_FORMAT_IN_FILE_NAMES),
            lead_time_seconds, lag_time_for_convective_contamination_sec
        )

    if raise_error_if_missing and not os.path.isfile(sounding_file_name):
        error_string = (
            'Cannot find file with soundings interpolated to storm objects.  '
            'Expected at: {0:s}'
        ).format(sounding_file_name)

        raise ValueError(error_string)

    return sounding_file_name
