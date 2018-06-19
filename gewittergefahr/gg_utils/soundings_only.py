"""Methods to handle atmospheric soundings."""

import os.path
import numpy
import pandas
import netCDF4
import scipy.interpolate
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
PERCENT_TO_UNITLESS = 0.01
MB_TO_PASCALS = 100

TEMPORAL_INTERP_METHOD = interp.PREVIOUS_INTERP_METHOD
SPATIAL_INTERP_METHOD = interp.NEAREST_INTERP_METHOD

PRESSURE_LEVEL_KEY = 'pressure_level_mb'
LEAD_TIME_KEY = 'lead_time_seconds'
LAG_TIME_KEY = 'lag_time_for_convective_contamination_sec'
INITIAL_TIME_COLUMN = 'init_time_unix_sec'
VALID_TIME_COLUMN = 'valid_time_unix_sec'

STORM_IDS_KEY = 'storm_ids'
INITIAL_TIMES_KEY = 'init_times_unix_sec'
LEAD_TIMES_KEY = 'lead_times_unix_sec'
SOUNDING_MATRIX_KEY = 'sounding_matrix'
LOWEST_PRESSURES_KEY = 'lowest_pressures_mb'
VERTICAL_LEVELS_KEY = 'vertical_levels_mb'
PRESSURELESS_FIELD_NAMES_KEY = 'pressureless_field_names'

RELATIVE_HUMIDITY_KEY = 'relative_humidity_unitless'
VIRTUAL_POTENTIAL_TEMPERATURE_KEY = 'virtual_potential_temperature_kelvins'

STORM_OBJECT_DIMENSION_KEY = 'storm_object'
PRESSURELESS_FIELD_DIMENSION_KEY = 'pressureless_field'
VERTICAL_DIMENSION_KEY = 'vertical_level'
STORM_ID_CHAR_DIMENSION_KEY = 'storm_id_character'
FIELD_NAME_CHAR_DIMENSION_KEY = 'pressureless_field_name_character'

PRESSURELESS_FIELDS_TO_INTERP = [
    nwp_model_utils.HEIGHT_COLUMN_FOR_SOUNDING_TABLES,
    nwp_model_utils.U_WIND_COLUMN_FOR_SOUNDING_TABLES,
    nwp_model_utils.V_WIND_COLUMN_FOR_SOUNDING_TABLES,
    nwp_model_utils.TEMPERATURE_COLUMN_FOR_SOUNDING_TABLES,
    nwp_model_utils.SPFH_COLUMN_FOR_SOUNDING_TABLES
]

DEFAULT_LEAD_TIMES_SEC = numpy.array([0], dtype=int)
DEFAULT_LAG_TIME_FOR_CONVECTIVE_CONTAMINATION_SEC = 1800


def _get_nwp_fields_for_sounding(
        model_name, return_table, include_surface=False,
        minimum_pressure_mb=0.):
    """Returns list of NWP fields needed to create sounding.

    Each field = one variable at one pressure level.

    H = number of vertical levels
    V = number of sounding variables (pressureless fields)
    F = H*V = number of sounding fields

    :param model_name: Model name (must be accepted by
        `nwp_model_utils.check_model_name`).
    :param return_table: Boolean flag.  See output doc for how this affects
        output.
    :param include_surface: Boolean flag.  If True, this method will include the
        surface or near-surface (2-metre or 10-metre) value of each field.
    :param minimum_pressure_mb: Minimum pressure level (millibars).  Leave this
        at 0 always.

    :return: sounding_field_names: [only if `return_table = False`]
        length-F list with names of sounding fields, in GewitterGefahr format.
    :return: sounding_field_names_grib1: [only if `return_table = False`]
        length-F list with names of sounding fields, in grib1 format.
    :return: sounding_field_name_table: [only if `return_table = True`]
        pandas DataFrame with the following columns.  Each row is one vertical
        level.  One of "relative_humidity_percent" and "specific_humidity" will
        be empty.
    sounding_field_name_table.geopotential_height_metres: Name of geopotential-
        height field.
    sounding_field_name_table.temperature_kelvins: Name of temperature field.
    sounding_field_name_table.relative_humidity_percent: Name of humidity field.
    sounding_field_name_table.specific_humidity: Name of humidity field.
    sounding_field_name_table.u_wind_m_s01: Name of u-wind field.
    sounding_field_name_table.v_wind_m_s01: Name of v-wind field.
    sounding_field_name_table.pressure_level_mb: Pressure level (millibars).
        The surface value is NaN.
    """

    nwp_model_utils.check_model_name(model_name)
    error_checking.assert_is_boolean(return_table)
    error_checking.assert_is_geq(minimum_pressure_mb, 0.)
    error_checking.assert_is_boolean(include_surface)

    pressure_levels_mb = nwp_model_utils.get_pressure_levels(
        model_name=model_name, grid_id=nwp_model_utils.ID_FOR_130GRID
    ).astype(float)
    pressure_levels_mb = pressure_levels_mb[
        pressure_levels_mb >= minimum_pressure_mb]

    if include_surface:
        vertical_levels_mb = numpy.concatenate((
            pressure_levels_mb, numpy.array([numpy.nan])
        ))
    else:
        vertical_levels_mb = pressure_levels_mb + 0.

    num_pressure_levels = len(pressure_levels_mb)
    num_vertical_levels = len(vertical_levels_mb)

    pressureless_field_names, pressureless_field_names_grib1 = (
        nwp_model_utils.get_columns_in_sounding_table(model_name))
    num_pressureless_fields = len(pressureless_field_names)

    sounding_field_name_table = None
    sounding_field_names = []
    sounding_field_names_grib1 = []

    if return_table:
        sounding_field_name_dict = {PRESSURE_LEVEL_KEY: vertical_levels_mb}
        list_of_empty_strings = [''] * num_vertical_levels
        for j in range(num_pressureless_fields):
            sounding_field_name_dict.update(
                {pressureless_field_names[j]: list_of_empty_strings})

        sounding_field_name_table = pandas.DataFrame.from_dict(
            sounding_field_name_dict)

    for j in range(num_pressureless_fields):
        for k in range(num_pressure_levels):
            this_field_name = '{0:s}_{1:d}mb'.format(
                pressureless_field_names[j],
                int(numpy.round(pressure_levels_mb[k])))

            if return_table:
                sounding_field_name_table[
                    pressureless_field_names[j]].values[k] = this_field_name
            else:
                this_field_name_grib1 = '{0:s}:{1:d} mb'.format(
                    pressureless_field_names_grib1[j],
                    int(numpy.round(pressure_levels_mb[k])))

                sounding_field_names.append(this_field_name)
                sounding_field_names_grib1.append(this_field_name_grib1)

        if not include_surface:
            continue

        if (pressureless_field_names[j] ==
                nwp_model_utils.HEIGHT_COLUMN_FOR_SOUNDING_TABLES):
            this_field_name, this_field_name_grib1 = (
                nwp_model_utils.get_lowest_height_name(model_name))

        if (pressureless_field_names[j] ==
                nwp_model_utils.TEMPERATURE_COLUMN_FOR_SOUNDING_TABLES):
            this_field_name, this_field_name_grib1 = (
                nwp_model_utils.get_lowest_temperature_name(model_name))

        if pressureless_field_names[j] in [
                nwp_model_utils.RH_COLUMN_FOR_SOUNDING_TABLES,
                nwp_model_utils.SPFH_COLUMN_FOR_SOUNDING_TABLES]:
            this_field_name, this_field_name_grib1 = (
                nwp_model_utils.get_lowest_humidity_name(model_name))

        if (pressureless_field_names[j] ==
                nwp_model_utils.U_WIND_COLUMN_FOR_SOUNDING_TABLES):
            this_field_name, this_field_name_grib1 = (
                nwp_model_utils.get_lowest_u_wind_name(model_name))

        if (pressureless_field_names[j] ==
                nwp_model_utils.V_WIND_COLUMN_FOR_SOUNDING_TABLES):
            this_field_name, this_field_name_grib1 = (
                nwp_model_utils.get_lowest_v_wind_name(model_name))

        if return_table:
            sounding_field_name_table[pressureless_field_names[j]].values[
                num_vertical_levels - 1] = this_field_name
        else:
            sounding_field_names.append(this_field_name)
            sounding_field_names_grib1.append(this_field_name_grib1)

    if return_table or not include_surface:
        return (sounding_field_names, sounding_field_names_grib1,
                sounding_field_name_table)

    this_field_name, this_field_name_grib1 = (
        nwp_model_utils.get_lowest_pressure_name(model_name))
    sounding_field_names.append(this_field_name)
    sounding_field_names_grib1.append(this_field_name_grib1)

    return (sounding_field_names, sounding_field_names_grib1,
            sounding_field_name_table)


def _create_target_points_for_interp(storm_object_table, lead_times_seconds):
    """Creates target points for interpolation.

    Each target point consists of (latitude, longitude, time).

    T = number of lead times

    :param storm_object_table: pandas DataFrame with columns documented in
        `storm_tracking_io.write_processed_file`.
    :param lead_times_seconds: length-T numpy array of lead times (may include
        0).  For each lead time t, each storm object will be extrapolated t
        seconds into the future, along its estimated motion vector.
    :return: target_point_table: pandas DataFrame with the following columns.
    target_point_table.storm_id: String ID for storm cell.
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
        (storm_speeds_m_s01, storm_bearings_deg
        ) = geodetic_utils.xy_components_to_displacements_and_bearings(
            x_displacements_metres=
            storm_object_table[tracking_utils.EAST_VELOCITY_COLUMN].values,
            y_displacements_metres=
            storm_object_table[tracking_utils.NORTH_VELOCITY_COLUMN].values)

    num_storm_objects = len(storm_object_table.index)
    num_lead_times = len(lead_times_seconds)
    list_of_target_point_tables = [None] * num_lead_times

    for i in range(num_lead_times):
        if lead_times_seconds[i] == 0:
            list_of_target_point_tables[i] = storm_object_table[[
                tracking_utils.STORM_ID_COLUMN, tracking_utils.TIME_COLUMN,
                tracking_utils.CENTROID_LAT_COLUMN,
                tracking_utils.CENTROID_LNG_COLUMN,
                tracking_utils.EAST_VELOCITY_COLUMN,
                tracking_utils.NORTH_VELOCITY_COLUMN
            ]]

            argument_dict = {
                LEAD_TIME_KEY: numpy.full(num_storm_objects, 0, dtype=int),
                VALID_TIME_COLUMN: list_of_target_point_tables[
                    i][tracking_utils.TIME_COLUMN].values
            }
            list_of_target_point_tables[i] = (
                list_of_target_point_tables[i].assign(**argument_dict))
            continue

        (these_extrap_latitudes_deg, these_extrap_longitudes_deg
        ) = geodetic_utils.start_points_and_distances_and_bearings_to_endpoints(
            start_latitudes_deg=storm_object_table[
                tracking_utils.CENTROID_LAT_COLUMN].values,
            start_longitudes_deg=storm_object_table[
                tracking_utils.CENTROID_LNG_COLUMN].values,
            displacements_metres=storm_speeds_m_s01 * lead_times_seconds[i],
            geodetic_bearings_deg=storm_bearings_deg)

        this_dict = {
            tracking_utils.STORM_ID_COLUMN: storm_object_table[
                tracking_utils.STORM_ID_COLUMN].values,
            tracking_utils.TIME_COLUMN: storm_object_table[
                tracking_utils.TIME_COLUMN].values,
            tracking_utils.CENTROID_LAT_COLUMN: these_extrap_latitudes_deg,
            tracking_utils.CENTROID_LNG_COLUMN: these_extrap_longitudes_deg,
            VALID_TIME_COLUMN:
                (storm_object_table[tracking_utils.TIME_COLUMN].values +
                 lead_times_seconds[i]),
            tracking_utils.EAST_VELOCITY_COLUMN: storm_object_table[
                tracking_utils.EAST_VELOCITY_COLUMN].values,
            tracking_utils.NORTH_VELOCITY_COLUMN: storm_object_table[
                tracking_utils.NORTH_VELOCITY_COLUMN].values,
            LEAD_TIME_KEY: numpy.full(
                num_storm_objects, lead_times_seconds[i], dtype=int)
        }

        list_of_target_point_tables[i] = pandas.DataFrame.from_dict(
            this_dict)
        if i == 0:
            continue

        list_of_target_point_tables[i], _ = (
            list_of_target_point_tables[i].align(
                list_of_target_point_tables[0], axis=1))

    target_point_table = pandas.concat(
        list_of_target_point_tables, axis=0, ignore_index=True)

    column_dict_old_to_new = {tracking_utils.TIME_COLUMN: INITIAL_TIME_COLUMN}
    return target_point_table.rename(
        columns=column_dict_old_to_new, inplace=False)


def _interp_soundings_from_nwp(
        target_point_table, top_grib_directory_name, model_name,
        include_surface, grid_id=None,
        wgrib_exe_name=grib_io.WGRIB_EXE_NAME_DEFAULT,
        wgrib2_exe_name=grib_io.WGRIB2_EXE_NAME_DEFAULT,
        raise_error_if_missing=False):
    """Interpolates soundings from NWP model to target points.

    Each target point consists of (latitude, longitude, time).

    :param target_point_table: pandas DataFrame created by
        `_create_target_points_for_interp`.
    :param top_grib_directory_name: Name of top-level directory with grib files
        for the given model.
    :param model_name: Model name (must be accepted by
        `nwp_model_utils.check_model_name`).
    :param include_surface: See documentation for
        `_get_nwp_fields_for_sounding`.
    :param grid_id: Grid ID (must be accepted by
        `nwp_model_utils.check_grid_id`).
    :param wgrib_exe_name: Path to wgrib executable.
    :param wgrib2_exe_name: Path to wgrib2 executable.
    :param raise_error_if_missing: See documentation for
        `interp.interp_nwp_from_xy_grid`.
    :return: interp_table: pandas DataFrame, where each column is one field and
        each row is one target point.  Column names are from the list
        `sounding_field_names` returned by `_get_nwp_fields_for_sounding`.
    """

    sounding_field_names, sounding_field_names_grib1, _ = (
        _get_nwp_fields_for_sounding(
            model_name=model_name, return_table=False,
            include_surface=include_surface))

    return interp.interp_nwp_from_xy_grid(
        query_point_table=target_point_table, model_name=model_name,
        grid_id=grid_id, field_names=sounding_field_names,
        field_names_grib1=sounding_field_names_grib1,
        top_grib_directory_name=top_grib_directory_name,
        temporal_interp_method=TEMPORAL_INTERP_METHOD,
        spatial_interp_method=SPATIAL_INTERP_METHOD,
        wgrib_exe_name=wgrib_exe_name, wgrib2_exe_name=wgrib2_exe_name,
        raise_error_if_missing=raise_error_if_missing)


def _interp_soundings_from_ruc(
        target_point_table, top_grib_directory_name, include_surface,
        wgrib_exe_name=grib_io.WGRIB_EXE_NAME_DEFAULT,
        wgrib2_exe_name=grib_io.WGRIB2_EXE_NAME_DEFAULT,
        raise_error_if_missing=False):
    """Interpolates soundings from RUC model to target points.

    Each target point consists of (latitude, longitude, time).

    This method may interpolate from different grids at different time steps.

    :param target_point_table: See doc for `_interp_soundings_from_nwp`.
    :param top_grib_directory_name: Same.
    :param include_surface: Same.
    :param wgrib_exe_name: Same.
    :param wgrib2_exe_name: Same.
    :param raise_error_if_missing: Same.
    :return: interp_table: Same.
    """

    sounding_field_names, sounding_field_names_grib1, _ = (
        _get_nwp_fields_for_sounding(
            model_name=nwp_model_utils.RUC_MODEL_NAME, return_table=False,
            include_surface=include_surface))

    return interp.interp_ruc_all_grids(
        query_point_table=target_point_table, field_names=sounding_field_names,
        field_names_grib1=sounding_field_names_grib1,
        top_grib_directory_name=top_grib_directory_name,
        temporal_interp_method=TEMPORAL_INTERP_METHOD,
        spatial_interp_method=SPATIAL_INTERP_METHOD,
        wgrib_exe_name=wgrib_exe_name, wgrib2_exe_name=wgrib2_exe_name,
        raise_error_if_missing=raise_error_if_missing)


def _convert_interp_table_to_soundings(
        interp_table, target_point_table, model_name, include_surface=False,
        minimum_pressure_mb=0.):
    """Converts table of interpolated values to list of soundings.

    T = number of lead times
    N = number of storm objects
    P = number of pressure levels
    H = number of vertical levels
    V = number of sounding variables (pressureless fields)

    K = T*N

    :param interp_table: K-row pandas DataFrame created by
        `_interp_soundings_from_nwp` or `_interp_soundings_from_ruc`.
    :param target_point_table: K-row pandas DataFrame created by
        `_create_target_points_for_interp`.
    :param model_name: Model name (must be accepted by
        `nwp_model_utils.check_model_name`).
    :param include_surface: See doc for `_get_nwp_fields_for_sounding`.
    :param minimum_pressure_mb: Same.
    :return: sounding_dict: Dictionary with the following keys.
    sounding_dict['storm_ids']: length-K list of storm IDs (strings).
    sounding_dict['init_times_unix_sec']: length-K list of initial times (storm
        times).  Valid time = initial time + lead time.
    sounding_dict['lead_times_seconds']: length-K list of lead times.
    sounding_dict['sounding_matrix']: K-by-H-by-V numpy array of sounding
        values.
    sounding_dict['lowest_pressures_mb']: length-K numpy array with lowest
        (surface or near-surface) pressure for each storm object.  If
        include_surface = False, this is None.
    sounding_dict['vertical_levels_mb']: length-H numpy array of vertical levels
        (millibars).  The surface value is NaN.
    sounding_dict['pressureless_field_names']: length-V list with names of
        pressureless fields.
    """

    _, _, sounding_field_name_table = _get_nwp_fields_for_sounding(
        model_name=model_name, return_table=True,
        include_surface=include_surface,
        minimum_pressure_mb=minimum_pressure_mb)

    if include_surface:
        lowest_pressure_name, _ = nwp_model_utils.get_lowest_pressure_name(
            model_name)
        lowest_pressures_mb = interp_table[lowest_pressure_name].values
    else:
        lowest_pressures_mb = None

    pressureless_field_names = list(sounding_field_name_table)
    pressureless_field_names.remove(PRESSURE_LEVEL_KEY)
    vertical_levels_mb = sounding_field_name_table[PRESSURE_LEVEL_KEY].values

    num_pressureless_fields = len(pressureless_field_names)
    num_vertical_levels = len(sounding_field_name_table.index)
    num_storm_objects = len(interp_table.index)
    sounding_matrix = numpy.full(
        (num_storm_objects, num_vertical_levels, num_pressureless_fields),
        numpy.nan)

    for j in range(num_vertical_levels):
        for k in range(num_pressureless_fields):
            this_field_name = sounding_field_name_table[
                pressureless_field_names[k]].values[j]
            sounding_matrix[:, j, k] = interp_table[this_field_name].values

    return {
        STORM_IDS_KEY:
            target_point_table[tracking_utils.STORM_ID_COLUMN].values.tolist(),
        INITIAL_TIMES_KEY: target_point_table[INITIAL_TIME_COLUMN].values,
        LEAD_TIMES_KEY: target_point_table[LEAD_TIME_KEY].values,
        SOUNDING_MATRIX_KEY: sounding_matrix,
        LOWEST_PRESSURES_KEY: lowest_pressures_mb,
        VERTICAL_LEVELS_KEY: vertical_levels_mb,
        PRESSURELESS_FIELD_NAMES_KEY: pressureless_field_names
    }


def _get_pressures(sounding_dict):
    """Finds pressure at each vertical level in each sounding.

    N = number of storm objects
    H = number of vertical levels

    :param sounding_dict: Dictionary created by
        `_convert_interp_table_to_soundings`.
    :return: pressure_matrix_pascals: N-by-H numpy array of pressures (Pa).
    """

    num_storm_objects = sounding_dict[SOUNDING_MATRIX_KEY].shape[0]
    num_vertical_levels = sounding_dict[SOUNDING_MATRIX_KEY].shape[1]
    pressure_matrix_pascals = numpy.full(
        (num_storm_objects, num_vertical_levels), numpy.nan)
    for i in range(num_storm_objects):
        pressure_matrix_pascals[i, :] = sounding_dict[VERTICAL_LEVELS_KEY]

    if sounding_dict[LOWEST_PRESSURES_KEY] is not None:
        surface_index = numpy.where(
            numpy.isnan(sounding_dict[VERTICAL_LEVELS_KEY]))[0][0]
        pressure_matrix_pascals[:, surface_index] = sounding_dict[
            LOWEST_PRESSURES_KEY]

    return MB_TO_PASCALS * pressure_matrix_pascals


def _relative_to_specific_humidity(sounding_dict, pressure_matrix_pascals):
    """Converts relative to specific humidity at each level in each sounding.

    N = number of storm objects
    H = number of vertical levels

    :param sounding_dict: Dictionary created by
        `_convert_interp_table_to_soundings`.
    :param pressure_matrix_pascals: N-by-H numpy array of pressures (Pa).
    :return: sounding_dict: Same as input, but now including specific humidity.
        Also, relative humidity is converted from percentage to the 0...1 range.
    :return: dewpoint_matrix_kelvins: N-by-H numpy array of dewpoints (K).
    """

    pressureless_field_names = sounding_dict[PRESSURELESS_FIELD_NAMES_KEY]
    sounding_matrix = sounding_dict[SOUNDING_MATRIX_KEY]
    temperature_index = pressureless_field_names.index(
        nwp_model_utils.TEMPERATURE_COLUMN_FOR_SOUNDING_TABLES)

    relative_humidity_index = pressureless_field_names.index(
        nwp_model_utils.RH_COLUMN_FOR_SOUNDING_TABLES)
    sounding_matrix[..., relative_humidity_index] = (
        PERCENT_TO_UNITLESS * sounding_matrix[..., relative_humidity_index])
    pressureless_field_names[relative_humidity_index] = RELATIVE_HUMIDITY_KEY

    dewpoint_matrix_kelvins = (
        moisture_conversions.relative_humidity_to_dewpoint(
            relative_humidities=sounding_matrix[..., relative_humidity_index],
            temperatures_kelvins=sounding_matrix[..., temperature_index],
            total_pressures_pascals=pressure_matrix_pascals))

    spec_humidity_matrix_kg_kg01 = (
        moisture_conversions.dewpoint_to_specific_humidity(
            dewpoints_kelvins=dewpoint_matrix_kelvins,
            total_pressures_pascals=pressure_matrix_pascals))
    spec_humidity_matrix_kg_kg01 = numpy.reshape(
        spec_humidity_matrix_kg_kg01,
        spec_humidity_matrix_kg_kg01.shape + (1,))

    pressureless_field_names.append(
        nwp_model_utils.SPFH_COLUMN_FOR_SOUNDING_TABLES)
    sounding_matrix = numpy.concatenate(
        (sounding_matrix, spec_humidity_matrix_kg_kg01), axis=-1)

    sounding_dict[PRESSURELESS_FIELD_NAMES_KEY] = pressureless_field_names
    sounding_dict[SOUNDING_MATRIX_KEY] = sounding_matrix
    return sounding_dict, dewpoint_matrix_kelvins


def _specific_to_relative_humidity(sounding_dict, pressure_matrix_pascals):
    """Converts specific to relative humidity at each level in each sounding.

    N = number of storm objects
    H = number of vertical levels

    :param sounding_dict: Dictionary created by
        `_convert_interp_table_to_soundings`.
    :param pressure_matrix_pascals: N-by-H numpy array of pressures (Pa).
    :return: sounding_dict: Same as input, but now including relative humidity.
    :return: dewpoint_matrix_kelvins: N-by-H numpy array of dewpoints (K).
    """

    pressureless_field_names = sounding_dict[PRESSURELESS_FIELD_NAMES_KEY]
    sounding_matrix = sounding_dict[SOUNDING_MATRIX_KEY]

    specific_humidity_index = pressureless_field_names.index(
        nwp_model_utils.SPFH_COLUMN_FOR_SOUNDING_TABLES)
    temperature_index = pressureless_field_names.index(
        nwp_model_utils.TEMPERATURE_COLUMN_FOR_SOUNDING_TABLES)

    dewpoint_matrix_kelvins = (
        moisture_conversions.specific_humidity_to_dewpoint(
            specific_humidities_kg_kg01=sounding_matrix[
                ..., specific_humidity_index],
            total_pressures_pascals=pressure_matrix_pascals))

    relative_humidity_matrix = (
        moisture_conversions.dewpoint_to_relative_humidity(
            dewpoints_kelvins=dewpoint_matrix_kelvins,
            temperatures_kelvins=sounding_matrix[..., temperature_index],
            total_pressures_pascals=pressure_matrix_pascals))
    relative_humidity_matrix = numpy.reshape(
        relative_humidity_matrix, relative_humidity_matrix.shape + (1,))

    pressureless_field_names.append(RELATIVE_HUMIDITY_KEY)
    sounding_matrix = numpy.concatenate(
        (sounding_matrix, relative_humidity_matrix), axis=-1)

    sounding_dict[PRESSURELESS_FIELD_NAMES_KEY] = pressureless_field_names
    sounding_dict[SOUNDING_MATRIX_KEY] = sounding_matrix
    return sounding_dict, dewpoint_matrix_kelvins


def _get_virtual_potential_temperatures(
        sounding_dict, pressure_matrix_pascals, dewpoint_matrix_kelvins):
    """Finds virtual potential temperature at each level in each sounding.

    N = number of storm objects
    H = number of vertical levels

    :param sounding_dict: Dictionary created by
        `_convert_interp_table_to_soundings`.
    :param pressure_matrix_pascals: N-by-H numpy array of pressures (Pa).
    :param dewpoint_matrix_kelvins: N-by-H numpy array of dewpoints (K).
    :return: sounding_dict: Same as input, but now including virtual potential
        temperature.
    """

    pressureless_field_names = sounding_dict[PRESSURELESS_FIELD_NAMES_KEY]
    sounding_matrix = sounding_dict[SOUNDING_MATRIX_KEY]
    temperature_index = pressureless_field_names.index(
        nwp_model_utils.TEMPERATURE_COLUMN_FOR_SOUNDING_TABLES)

    vapour_pressure_matrix_pascals = (
        moisture_conversions.dewpoint_to_vapour_pressure(
            dewpoint_matrix_kelvins))

    virtual_temperature_matrix_kelvins = (
        moisture_conversions.temperature_to_virtual_temperature(
            temperatures_kelvins=sounding_matrix[..., temperature_index],
            total_pressures_pascals=pressure_matrix_pascals,
            vapour_pressures_pascals=vapour_pressure_matrix_pascals))

    theta_v_matrix_kelvins = (
        temperature_conversions.temperatures_to_potential_temperatures(
            temperatures_kelvins=virtual_temperature_matrix_kelvins,
            total_pressures_pascals=pressure_matrix_pascals))
    theta_v_matrix_kelvins = numpy.reshape(
        theta_v_matrix_kelvins, theta_v_matrix_kelvins.shape + (1,))

    pressureless_field_names.append(VIRTUAL_POTENTIAL_TEMPERATURE_KEY)
    sounding_matrix = numpy.concatenate(
        (sounding_matrix, theta_v_matrix_kelvins), axis=-1)

    sounding_dict[PRESSURELESS_FIELD_NAMES_KEY] = pressureless_field_names
    sounding_dict[SOUNDING_MATRIX_KEY] = sounding_matrix
    return sounding_dict


def _fill_nans_in_soundings(
        sounding_dict, pressure_matrix_pascals,
        min_num_vertical_levels_without_nan=15):
    """Fills missing values (NaN's) in each sounding.

    :param sounding_dict: Dictionary created by
        `_convert_interp_table_to_soundings`.
    :param pressure_matrix_pascals: N-by-H numpy array of pressures (Pa).
    :param min_num_vertical_levels_without_nan: Minimum number of vertical
        levels without NaN.  For a given sounding S, if any variable has fewer
        levels without NaN, S will be thrown out (rather than interpolating to
        get rid of the NaN's).  Please keep this argument at the default value.
    :return: sounding_dict: Same as input, but without NaN's for u-wind, v-wind,
        geopotential height, temperature, or specific humidity.
    """

    pressureless_field_names = sounding_dict[PRESSURELESS_FIELD_NAMES_KEY]
    sounding_matrix = sounding_dict[SOUNDING_MATRIX_KEY]
    height_index = pressureless_field_names.index(
        nwp_model_utils.HEIGHT_COLUMN_FOR_SOUNDING_TABLES)

    num_soundings = sounding_matrix.shape[0]
    keep_sounding_flags = numpy.full(num_soundings, True, dtype=bool)

    for i in range(num_soundings):
        for this_field_name in PRESSURELESS_FIELDS_TO_INTERP:
            this_field_index = pressureless_field_names.index(this_field_name)
            these_nan_level_flags = numpy.isnan(
                sounding_matrix[i, :, this_field_index])
            if not numpy.any(these_nan_level_flags):
                continue

            if (numpy.sum(numpy.invert(these_nan_level_flags)) <
                    min_num_vertical_levels_without_nan):
                keep_sounding_flags[i] = False
                break

            these_nan_level_indices = numpy.where(these_nan_level_flags)[0]
            these_real_level_indices = numpy.where(
                numpy.invert(these_nan_level_flags))[0]

            if (this_field_name ==
                    nwp_model_utils.HEIGHT_COLUMN_FOR_SOUNDING_TABLES):
                interp_object = scipy.interpolate.interp1d(
                    x=numpy.log(
                        pressure_matrix_pascals[i, these_real_level_indices]),
                    y=sounding_matrix[
                        i, these_real_level_indices, this_field_index],
                    kind='linear', bounds_error=False, fill_value='extrapolate',
                    assume_sorted=False)

                sounding_matrix[
                    i, these_nan_level_indices, this_field_index
                ] = interp_object(numpy.log(
                    pressure_matrix_pascals[i, these_nan_level_indices]))
            else:
                interp_object = scipy.interpolate.interp1d(
                    x=sounding_matrix[
                        i, these_real_level_indices, height_index],
                    y=sounding_matrix[
                        i, these_real_level_indices, this_field_index],
                    kind='linear', bounds_error=False, fill_value='extrapolate',
                    assume_sorted=False)

                sounding_matrix[
                    i, these_nan_level_indices, this_field_index
                ] = interp_object(
                    sounding_matrix[i, these_nan_level_indices, height_index])

    keep_sounding_indices = numpy.where(keep_sounding_flags)[0]
    sounding_dict[SOUNDING_MATRIX_KEY] = sounding_matrix[
        keep_sounding_indices, ...]
    sounding_dict[STORM_IDS_KEY] = [
        sounding_dict[STORM_IDS_KEY][i] for i in keep_sounding_indices]
    sounding_dict[INITIAL_TIMES_KEY] = sounding_dict[
        INITIAL_TIMES_KEY][keep_sounding_indices]
    sounding_dict[LEAD_TIMES_KEY] = sounding_dict[
        LEAD_TIMES_KEY][keep_sounding_indices]

    if sounding_dict[LOWEST_PRESSURES_KEY] is not None:
        sounding_dict[LOWEST_PRESSURES_KEY] = sounding_dict[
            LOWEST_PRESSURES_KEY][keep_sounding_indices]

    return sounding_dict


def _convert_soundings(sounding_dict):
    """Converts variables and units in each sounding.

    :param sounding_dict: Dictionary created by
        `_convert_interp_table_to_soundings`.
    :return: sounding_dict: Same as input, but with different variables and
        units.
    """

    pressure_matrix_pascals = _get_pressures(sounding_dict)
    found_relative_humidity = (nwp_model_utils.RH_COLUMN_FOR_SOUNDING_TABLES in
                               sounding_dict[PRESSURELESS_FIELD_NAMES_KEY])

    if found_relative_humidity:
        sounding_dict, dewpoint_matrix_kelvins = _relative_to_specific_humidity(
            sounding_dict=sounding_dict,
            pressure_matrix_pascals=pressure_matrix_pascals)

    sounding_dict = _fill_nans_in_soundings(
        sounding_dict=sounding_dict,
        pressure_matrix_pascals=pressure_matrix_pascals)
    pressure_matrix_pascals = _get_pressures(sounding_dict)

    sounding_dict, dewpoint_matrix_kelvins = _specific_to_relative_humidity(
        sounding_dict=sounding_dict,
        pressure_matrix_pascals=pressure_matrix_pascals)

    return _get_virtual_potential_temperatures(
        sounding_dict=sounding_dict,
        pressure_matrix_pascals=pressure_matrix_pascals,
        dewpoint_matrix_kelvins=dewpoint_matrix_kelvins)


def interp_soundings_to_storm_objects(
        storm_object_table, top_grib_directory_name,
        lead_times_seconds=DEFAULT_LEAD_TIMES_SEC,
        lag_time_for_convective_contamination_sec=
        DEFAULT_LAG_TIME_FOR_CONVECTIVE_CONTAMINATION_SEC,
        include_surface=False, all_ruc_grids=True, model_name=None,
        grid_id=None, wgrib_exe_name=grib_io.WGRIB_EXE_NAME_DEFAULT,
        wgrib2_exe_name=grib_io.WGRIB2_EXE_NAME_DEFAULT,
        raise_error_if_missing=False):
    """Creates interpolated NWP sounding for each storm object.

    T = number of lead times
    N = number of storm objects
    P = number of pressure levels
    H = number of vertical levels
    V = number of sounding variables (pressureless fields)

    :param storm_object_table: pandas DataFrame with columns documented in
        `storm_tracking_io.write_processed_file`.
    :param top_grib_directory_name: Name of top-level directory with grib files
        for the NWP model.
    :param lead_times_seconds: length-T numpy array of lead times.  For each
        lead time t, each storm object will be extrapolated t seconds into the
        future along its estimated motion vector.  Thus, one sounding will be
        created for each pair of storm object and lead time.
    :param lag_time_for_convective_contamination_sec: Lag time (used to avoid
        convective contamination of soundings, where the sounding for storm S is
        heavily influenced by storm S).  This will be subtracted from each lead
        time in `lead_times_seconds`.
    :param include_surface: Boolean flag.  If True, will include surface values
        in each sounding.
    :param all_ruc_grids: Boolean flag.  If True, this method will use
        `_interp_soundings_from_ruc`, which interpolates data from the highest-
        resolution RUC grid available at each initialization time (thus,
        different soundings may be interpolated from different grids).  If
        False, will use `_interp_soundings_from_nwp`, which interpolates data
        from the same grid at each initialization time.
    :param model_name: [used only if all_ruc_grids = False]
        Name of model (must be accepted by `nwp_model_utils.check_grid_id`).
    :param grid_id: [used only if all_ruc_grids = False]
        Name of grid (must be accepted by `nwp_model_utils.check_grid_id`).
    :param wgrib_exe_name: Path to wgrib executable.
    :param wgrib2_exe_name: Path to wgrib2 executable.
    :param raise_error_if_missing: Boolean flag.  If any grib file is missing
        and raise_error_if_missing = True, this method will error out.  If any
        grib file is missing and raise_error_if_missing = False, this method
        will carry on, leaving the affected sounding values as NaN.
    :return: sounding_dict_by_lead_time: length-T list of dictionaries.  Each
        dictionary contains the keys listed in
        `_convert_interp_table_to_soundings`.
    """

    error_checking.assert_is_integer_numpy_array(lead_times_seconds)
    error_checking.assert_is_numpy_array(lead_times_seconds, num_dimensions=1)
    error_checking.assert_is_geq_numpy_array(lead_times_seconds, 0)
    error_checking.assert_is_integer(lag_time_for_convective_contamination_sec)
    error_checking.assert_is_geq(lag_time_for_convective_contamination_sec, 0)
    error_checking.assert_is_boolean(include_surface)
    error_checking.assert_is_boolean(all_ruc_grids)

    print (
        'Creating target point for each storm object and lead time ({0:s} '
        'seconds)...'
    ).format(str(lead_times_seconds))

    target_point_table = _create_target_points_for_interp(
        storm_object_table=storm_object_table,
        lead_times_seconds=lead_times_seconds)
    target_point_table[
        VALID_TIME_COLUMN] -= lag_time_for_convective_contamination_sec

    column_dict_old_to_new = {
        tracking_utils.CENTROID_LAT_COLUMN: interp.QUERY_LAT_COLUMN,
        tracking_utils.CENTROID_LNG_COLUMN: interp.QUERY_LNG_COLUMN,
        VALID_TIME_COLUMN: interp.QUERY_TIME_COLUMN}
    target_point_table.rename(columns=column_dict_old_to_new, inplace=True)

    print SEPARATOR_STRING

    if all_ruc_grids:
        model_name = nwp_model_utils.RUC_MODEL_NAME
        interp_table = _interp_soundings_from_ruc(
            target_point_table=target_point_table,
            top_grib_directory_name=top_grib_directory_name,
            include_surface=include_surface, wgrib_exe_name=wgrib_exe_name,
            wgrib2_exe_name=wgrib2_exe_name,
            raise_error_if_missing=raise_error_if_missing)
    else:
        interp_table = _interp_soundings_from_nwp(
            target_point_table=target_point_table,
            top_grib_directory_name=top_grib_directory_name,
            include_surface=include_surface, model_name=model_name,
            grid_id=grid_id, wgrib_exe_name=wgrib_exe_name,
            wgrib2_exe_name=wgrib2_exe_name,
            raise_error_if_missing=raise_error_if_missing)

    print SEPARATOR_STRING
    print 'Converting table of interpolated values to soundings...'
    sounding_dict = _convert_interp_table_to_soundings(
        interp_table=interp_table, target_point_table=target_point_table,
        model_name=model_name, include_surface=include_surface)

    print 'Converting variables and units in each sounding...'
    orig_num_soundings = len(sounding_dict[STORM_IDS_KEY])
    sounding_dict = _convert_soundings(sounding_dict)
    num_soundings = len(sounding_dict[STORM_IDS_KEY])

    print '{0:d} of {1:d} soundings were removed (too many NaN''s).\n'.format(
        orig_num_soundings - num_soundings, orig_num_soundings)

    num_lead_times = len(lead_times_seconds)
    sounding_dict_by_lead_time = [None] * num_lead_times
    for k in range(num_lead_times):
        print (
            'Creating separate sounding dictionary for {0:d}-second lead '
            'time...'
        ).format(lead_times_seconds[k])

        these_indices = numpy.where(
            sounding_dict[LEAD_TIMES_KEY] == lead_times_seconds[k])[0]

        sounding_dict_by_lead_time[k] = {
            STORM_IDS_KEY:
                [sounding_dict[STORM_IDS_KEY][i] for i in these_indices],
            INITIAL_TIMES_KEY: sounding_dict[INITIAL_TIMES_KEY][these_indices],
            LEAD_TIMES_KEY: sounding_dict[LEAD_TIMES_KEY][these_indices],
            SOUNDING_MATRIX_KEY:
                sounding_dict[SOUNDING_MATRIX_KEY][these_indices, ...],
            VERTICAL_LEVELS_KEY: sounding_dict[VERTICAL_LEVELS_KEY],
            PRESSURELESS_FIELD_NAMES_KEY:
                sounding_dict[PRESSURELESS_FIELD_NAMES_KEY]
        }

        if sounding_dict[LOWEST_PRESSURES_KEY] is None:
            sounding_dict_by_lead_time[k].update({LOWEST_PRESSURES_KEY: None})
        else:
            sounding_dict_by_lead_time[k].update(
                {LOWEST_PRESSURES_KEY:
                     sounding_dict[LOWEST_PRESSURES_KEY][these_indices]}
            )

        print (
            'Dictionary for {0:d}-second lead time contains {1:d} of {2:d} '
            'soundings.\n'
        ).format(lead_times_seconds[k], len(these_indices), num_soundings)

    return sounding_dict_by_lead_time


def write_soundings(
        sounding_dict, lead_time_seconds,
        lag_time_for_convective_contamination_sec, netcdf_file_name):
    """Writes soundings to NetCDF file.

    This file should contain soundings for only one lead time.

    :param sounding_dict: Dictionary with keys documented in
        `_convert_interp_table_to_soundings`.
    :param lead_time_seconds: See doc for
        `interp_soundings_to_storm_objects`.
    :param lag_time_for_convective_contamination_sec: Same.
    :param netcdf_file_name: Path to output file.
    :raises: ValueError: if `sounding_dict` contains lead times other than
        `lead_time_seconds`.
    """

    # Check input arguments.
    error_checking.assert_is_integer(lead_time_seconds)
    error_checking.assert_is_geq(lead_time_seconds, 0)
    error_checking.assert_is_integer(lag_time_for_convective_contamination_sec)
    error_checking.assert_is_geq(lag_time_for_convective_contamination_sec, 0)

    if numpy.any(sounding_dict[LEAD_TIMES_KEY] != lead_time_seconds):
        unique_lead_time_seconds = numpy.unique(
            sounding_dict[LEAD_TIMES_KEY])
        error_string = (
            'sounding_dict should contain only soundings with {0:d}-second '
            'lead time.  Instead, contains lead times listed below.\n{1:s}'
        ).format(lead_time_seconds, str(unique_lead_time_seconds))
        raise ValueError(error_string)

    # Create NetCDF file and set global attributes.
    file_system_utils.mkdir_recursive_if_necessary(file_name=netcdf_file_name)
    netcdf_dataset = netCDF4.Dataset(
        netcdf_file_name, 'w', format='NETCDF3_64BIT_OFFSET')

    netcdf_dataset.setncattr(LEAD_TIME_KEY, lead_time_seconds)
    netcdf_dataset.setncattr(
        LAG_TIME_KEY, lag_time_for_convective_contamination_sec)

    # Create dimensions.
    num_storm_objects = len(sounding_dict[STORM_IDS_KEY])
    num_vertical_levels = len(sounding_dict[VERTICAL_LEVELS_KEY])
    num_pressureless_fields = len(sounding_dict[PRESSURELESS_FIELD_NAMES_KEY])

    netcdf_dataset.createDimension(
        STORM_OBJECT_DIMENSION_KEY, num_storm_objects)
    netcdf_dataset.createDimension(VERTICAL_DIMENSION_KEY, num_vertical_levels)
    netcdf_dataset.createDimension(
        PRESSURELESS_FIELD_DIMENSION_KEY, num_pressureless_fields)

    num_storm_id_chars = 1
    for i in range(num_storm_objects):
        num_storm_id_chars = max(
            [num_storm_id_chars, len(sounding_dict[STORM_IDS_KEY][i])])

    netcdf_dataset.createDimension(
        STORM_ID_CHAR_DIMENSION_KEY, num_storm_id_chars)

    num_field_name_chars = 0
    for j in range(num_pressureless_fields):
        num_field_name_chars = max(
            [num_field_name_chars,
             len(sounding_dict[PRESSURELESS_FIELD_NAMES_KEY][j])])

    netcdf_dataset.createDimension(
        FIELD_NAME_CHAR_DIMENSION_KEY, num_field_name_chars)

    # Add storm IDs to file.
    netcdf_dataset.createVariable(
        STORM_IDS_KEY, datatype='S1',
        dimensions=(STORM_OBJECT_DIMENSION_KEY, STORM_ID_CHAR_DIMENSION_KEY))

    string_type = 'S{0:d}'.format(num_storm_id_chars)
    storm_ids_as_char_array = netCDF4.stringtochar(numpy.array(
        sounding_dict[STORM_IDS_KEY], dtype=string_type))
    netcdf_dataset.variables[STORM_IDS_KEY][:] = numpy.array(
        storm_ids_as_char_array)

    # Add initial times (storm times) to file.
    netcdf_dataset.createVariable(
        INITIAL_TIMES_KEY, datatype=numpy.int32,
        dimensions=STORM_OBJECT_DIMENSION_KEY)
    netcdf_dataset.variables[INITIAL_TIMES_KEY][:] = sounding_dict[
        INITIAL_TIMES_KEY]

    # Add vertical levels to file.
    netcdf_dataset.createVariable(
        VERTICAL_LEVELS_KEY, datatype=numpy.int32,
        dimensions=VERTICAL_DIMENSION_KEY)
    netcdf_dataset.variables[VERTICAL_LEVELS_KEY][:] = sounding_dict[
        VERTICAL_LEVELS_KEY]

    # Add field names to file.
    netcdf_dataset.createVariable(
        PRESSURELESS_FIELD_NAMES_KEY, datatype='S1',
        dimensions=(
            PRESSURELESS_FIELD_DIMENSION_KEY, FIELD_NAME_CHAR_DIMENSION_KEY))

    string_type = 'S{0:d}'.format(num_field_name_chars)
    field_names_as_char_array = netCDF4.stringtochar(numpy.array(
        sounding_dict[PRESSURELESS_FIELD_NAMES_KEY], dtype=string_type))
    netcdf_dataset.variables[PRESSURELESS_FIELD_NAMES_KEY][:] = numpy.array(
        field_names_as_char_array)

    # Add lowest pressures to file.
    if sounding_dict[LOWEST_PRESSURES_KEY] is not None:
        netcdf_dataset.createVariable(
            LOWEST_PRESSURES_KEY, datatype=numpy.float32,
            dimensions=STORM_OBJECT_DIMENSION_KEY)
        netcdf_dataset.variables[LOWEST_PRESSURES_KEY][:] = sounding_dict[
            LOWEST_PRESSURES_KEY]

    # Add soundings to file.
    netcdf_dataset.createVariable(
        SOUNDING_MATRIX_KEY, datatype=numpy.float32,
        dimensions=(STORM_OBJECT_DIMENSION_KEY, VERTICAL_DIMENSION_KEY,
                    PRESSURELESS_FIELD_DIMENSION_KEY))

    netcdf_dataset.variables[SOUNDING_MATRIX_KEY][:] = sounding_dict[
        SOUNDING_MATRIX_KEY]
    netcdf_dataset.close()


def read_soundings(netcdf_file_name):
    """Reads soundings from NetCDF file.

    :param netcdf_file_name: Path to input file.
    :return: sounding_dict: Dictionary with keys documented in
        `_convert_interp_table_to_soundings`.
    :return: lag_time_for_convective_contamination_sec: See doc for
        `interp_soundings_to_storm_objects`.
    """

    netcdf_dataset = netcdf_io.open_netcdf(
        netcdf_file_name=netcdf_file_name, raise_error_if_fails=True)

    lead_time_seconds = getattr(netcdf_dataset, LEAD_TIME_KEY)
    lag_time_for_convective_contamination_sec = int(
        getattr(netcdf_dataset, LAG_TIME_KEY))

    storm_ids = netCDF4.chartostring(netcdf_dataset.variables[STORM_IDS_KEY][:])
    storm_ids = [str(s) for s in storm_ids]
    pressureless_field_names = netCDF4.chartostring(
        netcdf_dataset.variables[PRESSURELESS_FIELD_NAMES_KEY][:])
    pressureless_field_names = [str(s) for s in pressureless_field_names]

    init_times_unix_sec = numpy.array(
        netcdf_dataset.variables[INITIAL_TIMES_KEY][:], dtype=int)
    vertical_levels_mb = numpy.array(
        netcdf_dataset.variables[VERTICAL_LEVELS_KEY][:], dtype=int)
    sounding_matrix = numpy.array(netcdf_dataset.variables[SOUNDING_MATRIX_KEY])

    if numpy.any(numpy.isnan(vertical_levels_mb)):
        lowest_pressures_mb = numpy.array(
            netcdf_dataset.variables[LOWEST_PRESSURES_KEY][:], dtype=int)
    else:
        lowest_pressures_mb = None

    netcdf_dataset.close()

    num_storm_objects = len(storm_ids)
    lead_times_seconds = numpy.full(
        num_storm_objects, lead_time_seconds, dtype=int)

    sounding_dict = {
        STORM_IDS_KEY: storm_ids,
        INITIAL_TIMES_KEY: init_times_unix_sec,
        LEAD_TIMES_KEY: lead_times_seconds,
        SOUNDING_MATRIX_KEY: sounding_matrix,
        LOWEST_PRESSURES_KEY: lowest_pressures_mb,
        VERTICAL_LEVELS_KEY: vertical_levels_mb,
        PRESSURELESS_FIELD_NAMES_KEY: pressureless_field_names
    }
    return sounding_dict, lag_time_for_convective_contamination_sec


def find_sounding_file(
        top_directory_name, spc_date_string, lead_time_seconds,
        lag_time_for_convective_contamination_sec,
        init_time_unix_sec=None, raise_error_if_missing=True):
    """Finds file with soundings interpolated to storm objects.

    :param top_directory_name: Name of top-level directory with sounding files.
    :param spc_date_string: SPC date (format "yyyymmdd").
    :param lead_time_seconds: Lead time.
    :param lag_time_for_convective_contamination_sec: See doc for
        `interp_soundings_to_storm_objects`.
    :param init_time_unix_sec: Initial time (should be common to all storm
        objects in the file).  If `init_time_unix_sec is None`, will look for a
        file containing soundings for one SPC date, rather than one time step.
    :param raise_error_if_missing: Boolean flag.  If file is missing and
        raise_error_if_missing = True, this method will error out.
    :return: sounding_file_name: Path to image file.  If file is missing and
        raise_error_if_missing = False, this is the *expected* path.
    :raises: ValueError: if file is missing and raise_error_if_missing = True.
    """

    error_checking.assert_is_string(top_directory_name)
    time_conversion.spc_date_string_to_unix_sec(spc_date_string)
    error_checking.assert_is_boolean(raise_error_if_missing)

    if init_time_unix_sec is None:
        sounding_file_name = (
            '{0:s}/{1:s}/storm_soundings_{2:s}_lead-time-{3:05d}sec'
            '_lag-time-{4:04d}sec.nc'
        ).format(top_directory_name, spc_date_string[:4], spc_date_string,
                 lead_time_seconds, lag_time_for_convective_contamination_sec)
    else:
        sounding_file_name = (
            '{0:s}/{1:s}/{2:s}/storm_soundings_{3:s}_lead-time-{4:05d}sec'
            '_lag-time-{5:04d}sec.nc'
        ).format(top_directory_name, spc_date_string[:4], spc_date_string,
                 time_conversion.unix_sec_to_string(
                     init_time_unix_sec, TIME_FORMAT_IN_FILE_NAMES),
                 lead_time_seconds, lag_time_for_convective_contamination_sec)

    if raise_error_if_missing and not os.path.isfile(sounding_file_name):
        error_string = (
            'Cannot find file with soundings interpolated to storm objects.  '
            'Expected at: {0:s}').format(sounding_file_name)
        raise ValueError(error_string)

    return sounding_file_name
