"""Methods to handle atmospheric soundings."""

import numpy
import pandas
from gewittergefahr.gg_io import grib_io
from gewittergefahr.gg_utils import geodetic_utils
from gewittergefahr.gg_utils import nwp_model_utils
from gewittergefahr.gg_utils import storm_tracking_utils as tracking_utils
from gewittergefahr.gg_utils import interp
from gewittergefahr.gg_utils import error_checking

TEMPORAL_INTERP_METHOD = interp.PREVIOUS_INTERP_METHOD
SPATIAL_INTERP_METHOD = interp.NEAREST_INTERP_METHOD

PRESSURE_LEVEL_KEY = 'pressure_level_mb'
LEAD_TIME_COLUMN = 'lead_time_seconds'


def get_nwp_fields_for_sounding(
        model_name, return_table, minimum_pressure_mb=0.,
        include_surface=False):
    """Returns list of NWP fields needed to create sounding.

    Each field = one variable at one pressure level.

    V = number of variables
    P = number of pressure levels
    N = V*P = number of fields

    :param model_name: Model name (must be accepted by
        `nwp_model_utils.check_model_name`).
    :param return_table: Boolean flag.  See output doc for how this affects
        output.
    :param minimum_pressure_mb: Minimum pressure level (millibars).
    :param include_surface: Boolean flag.  If True, this method will include the
        surface or near-surface (2-metre or 10-metre) value of each field.

    :return: sounding_field_names: [only if `return_table = False`]
        length-N list with names of sounding fields, in GewitterGefahr format.
    :return: sounding_field_names_grib1: [only if `return_table = False`]
        length-N list with names of sounding fields, in grib1 format.
    :return: sounding_field_name_table: [only if `return_table = True`]
        pandas DataFrame with the following columns.  Each row is one pressure
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
