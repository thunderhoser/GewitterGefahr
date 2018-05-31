"""Methods to handle atmospheric soundings."""

import numpy
import pandas
import scipy.interpolate
from gewittergefahr.gg_io import grib_io
from gewittergefahr.gg_utils import geodetic_utils
from gewittergefahr.gg_utils import nwp_model_utils
from gewittergefahr.gg_utils import storm_tracking_utils as tracking_utils
from gewittergefahr.gg_utils import interp
from gewittergefahr.gg_utils import moisture_conversions
from gewittergefahr.gg_utils import temperature_conversions
from gewittergefahr.gg_utils import error_checking
PERCENT_TO_UNITLESS = 0.01
MB_TO_PASCALS = 100
MIN_NUM_VERTICAL_LEVELS_WITHOUT_NAN = 15

TEMPORAL_INTERP_METHOD = interp.PREVIOUS_INTERP_METHOD
SPATIAL_INTERP_METHOD = interp.NEAREST_INTERP_METHOD

PRESSURE_LEVEL_KEY = 'pressure_level_mb'
LEAD_TIME_COLUMN = 'lead_time_seconds'

SOUNDING_MATRIX_KEY = 'sounding_matrix'
LOWEST_PRESSURES_KEY = 'lowest_pressures_mb'
VERTICAL_LEVELS_KEY = 'vertical_levels_mb'
PRESSURELESS_FIELD_NAMES_KEY = 'pressureless_field_names'
RELATIVE_HUMIDITY_KEY = 'relative_humidity_unitless'
VIRTUAL_POTENTIAL_TEMPERATURE_KEY = 'virtual_potential_temperature_kelvins'

PRESSURELESS_FIELDS_TO_INTERP = [
    nwp_model_utils.HEIGHT_COLUMN_FOR_SOUNDING_TABLES,
    nwp_model_utils.U_WIND_COLUMN_FOR_SOUNDING_TABLES,
    nwp_model_utils.V_WIND_COLUMN_FOR_SOUNDING_TABLES,
    nwp_model_utils.TEMPERATURE_COLUMN_FOR_SOUNDING_TABLES,
    nwp_model_utils.SPFH_COLUMN_FOR_SOUNDING_TABLES
]


def get_nwp_fields_for_sounding(
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
    target_point_table.centroid_lat_deg: Latitude (deg N) of extrapolated storm
        object's centroid.
    target_point_table.centroid_lng_deg: Longitude (deg E) of extrapolated storm
        object's centroid.
    target_point_table.unix_time_sec: Time of extrapolated storm object.
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
                tracking_utils.CENTROID_LAT_COLUMN,
                tracking_utils.CENTROID_LNG_COLUMN, tracking_utils.TIME_COLUMN,
                tracking_utils.EAST_VELOCITY_COLUMN,
                tracking_utils.NORTH_VELOCITY_COLUMN]]

            argument_dict = {
                LEAD_TIME_COLUMN: numpy.full(num_storm_objects, 0, dtype=int)}
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
            tracking_utils.CENTROID_LAT_COLUMN: these_extrap_latitudes_deg,
            tracking_utils.CENTROID_LNG_COLUMN: these_extrap_longitudes_deg,
            tracking_utils.TIME_COLUMN:
                (storm_object_table[tracking_utils.TIME_COLUMN].values +
                 lead_times_seconds[i]),
            tracking_utils.EAST_VELOCITY_COLUMN: storm_object_table[
                tracking_utils.EAST_VELOCITY_COLUMN].values,
            tracking_utils.NORTH_VELOCITY_COLUMN: storm_object_table[
                tracking_utils.NORTH_VELOCITY_COLUMN].values,
            LEAD_TIME_COLUMN: numpy.full(
                num_storm_objects, lead_times_seconds[i], dtype=int)
        }

        list_of_target_point_tables[i] = pandas.DataFrame.from_dict(
            this_dict)
        if i == 0:
            continue

        list_of_target_point_tables[i], _ = (
            list_of_target_point_tables[i].align(
                list_of_target_point_tables[0], axis=1))

    return pandas.concat(list_of_target_point_tables, axis=0, ignore_index=True)


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
    :param include_surface: See documentation for `get_nwp_fields_for_sounding`.
    :param grid_id: Grid ID (must be accepted by
        `nwp_model_utils.check_grid_id`).
    :param wgrib_exe_name: Path to wgrib executable.
    :param wgrib2_exe_name: Path to wgrib2 executable.
    :param raise_error_if_missing: See documentation for
        `interp.interp_nwp_from_xy_grid`.
    :return: interp_table: pandas DataFrame, where each column is one field and
        each row is one target point.  Column names are from the list
        `sounding_field_names` returned by `get_nwp_fields_for_sounding`.
    """

    sounding_field_names, sounding_field_names_grib1, _ = (
        get_nwp_fields_for_sounding(
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
        get_nwp_fields_for_sounding(
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


def _interp_table_to_soundings(
        interp_table, model_name, include_surface=False,
        minimum_pressure_mb=0.):
    """Converts table of interpolated values to list of soundings.

    N = number of storm objects
    P = number of pressure levels
    H = number of vertical levels
    V = number of sounding variables (pressureless fields)

    :param interp_table: N-row pandas DataFrame created by
        `_interp_soundings_from_nwp` or `_interp_soundings_from_ruc`.
    :param model_name: Model name (must be accepted by
        `nwp_model_utils.check_model_name`).
    :param include_surface: See doc for `get_nwp_fields_for_sounding`.
    :param minimum_pressure_mb: Same.
    :return: sounding_dict: Dictionary with the following keys.
    sounding_dict['sounding_matrix']: N-by-H-by-V numpy array of sounding
        values.
    sounding_dict['lowest_pressures_mb']: length-N numpy array with lowest
        (surface or near-surface) pressure for each storm object.  If
        include_surface = False, this is None.
    sounding_dict['vertical_levels_mb']: length-H numpy array of vertical levels
        (millibars).  The surface value is NaN.
    sounding_dict['pressureless_field_names']: length-V list with names of
        pressureless fields.
    """

    _, _, sounding_field_name_table = get_nwp_fields_for_sounding(
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
        SOUNDING_MATRIX_KEY: sounding_matrix,
        LOWEST_PRESSURES_KEY: lowest_pressures_mb,
        VERTICAL_LEVELS_KEY: vertical_levels_mb,
        PRESSURELESS_FIELD_NAMES_KEY: pressureless_field_names
    }


def _get_pressures(sounding_dict):
    """Finds pressure at each vertical level in each sounding.

    N = number of storm objects
    H = number of vertical levels

    :param sounding_dict: Dictionary created by `_interp_table_to_soundings`.
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

    :param sounding_dict: Dictionary created by `_interp_table_to_soundings`.
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

    :param sounding_dict: Dictionary created by `_interp_table_to_soundings`.
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

    :param sounding_dict: Dictionary created by `_interp_table_to_soundings`.
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


def _fill_nans_in_soundings(sounding_dict, pressure_matrix_pascals):
    """Fills missing values (NaN's) in each sounding.

    :param sounding_dict: Dictionary created by `_interp_table_to_soundings`.
    :param pressure_matrix_pascals: N-by-H numpy array of pressures (Pa).
    :return: sounding_dict: Same as input, but without NaN's for u-wind, v-wind,
        geopotential height, temperature, or specific humidity.
    """

    # TODO(thunderhoser): This still needs a unit test!

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
                    MIN_NUM_VERTICAL_LEVELS_WITHOUT_NAN):
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
                    pressure_matrix_pascals[i, these_real_level_indices]))
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
    if sounding_dict[LOWEST_PRESSURES_KEY] is not None:
        sounding_dict[LOWEST_PRESSURES_KEY] = sounding_dict[
            LOWEST_PRESSURES_KEY][keep_sounding_indices]

    return sounding_dict


def _convert_soundings(sounding_dict):
    """Converts variables and units in each sounding.

    :param sounding_dict: Dictionary created by `_interp_table_to_soundings`.
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

    sounding_dict, dewpoint_matrix_kelvins = _specific_to_relative_humidity(
        sounding_dict=sounding_dict,
        pressure_matrix_pascals=pressure_matrix_pascals)

    return _get_virtual_potential_temperatures(
        sounding_dict=sounding_dict,
        pressure_matrix_pascals=pressure_matrix_pascals,
        dewpoint_matrix_kelvins=dewpoint_matrix_kelvins)
