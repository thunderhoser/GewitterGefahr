"""Methods for computing sounding statistics."""

import copy
import pickle
import os.path
import numpy
import pandas
import netCDF4
import scipy.interpolate
from sharppy.sharptab import params as sharppy_params
from sharppy.sharptab import winds as sharppy_winds
from sharppy.sharptab import interp as sharppy_interp
from sharppy.sharptab import profile as sharppy_profile
from sharppy.sharptab import utils as sharppy_utils
from gewittergefahr.gg_io import grib_io
from gewittergefahr.gg_io import netcdf_io
from gewittergefahr.gg_utils import storm_tracking_utils as tracking_utils
from gewittergefahr.gg_utils import moisture_conversions
from gewittergefahr.gg_utils import temperature_conversions
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import interp
from gewittergefahr.gg_utils import nwp_model_utils
from gewittergefahr.gg_utils import geodetic_utils
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.gg_utils import error_checking

TIME_FORMAT_IN_FILE_NAMES = '%Y-%m-%d-%H%M%S'
LEAD_TIME_COLUMN = 'lead_time_seconds'
DEFAULT_LEAD_TIMES_SEC = numpy.array([0], dtype=int)

TEMPORAL_INTERP_METHOD = interp.PREVIOUS_INTERP_METHOD
SPATIAL_INTERP_METHOD = interp.NEAREST_INTERP_METHOD
STORM_COLUMNS_TO_KEEP = [
    tracking_utils.STORM_ID_COLUMN, tracking_utils.TIME_COLUMN]

MIN_RELATIVE_HUMIDITY_PERCENT = 1.
SENTINEL_VALUE_FOR_SHARPPY = -9999.
SENTINEL_VALUE_TOLERANCE = 1e-3
REDUNDANT_PRESSURE_TOLERANCE_MB = 1e-3
REDUNDANT_HEIGHT_TOLERANCE_METRES = 1e-3
MIN_PRESSURE_LEVELS_IN_SOUNDING = 15

PERCENT_TO_UNITLESS = 0.01
UNITLESS_TO_PERCENT = 100
PASCALS_TO_MB = 0.01
MB_TO_PASCALS = 100
METRES_PER_SECOND_TO_KT = 3.6 / 1.852
KT_TO_METRES_PER_SECOND = 1.852 / 3.6

INIT_TIME_KEY = 'init_time_unix_sec'
LEAD_TIME_KEY = 'lead_time_sec'
SPC_DATE_KEY = 'spc_date_string'
STORM_DIMENSION_KEY = 'storm'
STATISTIC_DIMENSION_KEY = 'statistic'
STATISTIC_CHAR_DIMENSION_KEY = 'statistic_name_char'
STORM_ID_CHAR_DIMENSION_KEY = 'storm_id_char'
STATISTIC_NAMES_KEY = 'statistic_names'
STORM_IDS_KEY = 'storm_ids'
STATISTIC_MATRIX_KEY = 'sounding_statistic_matrix'

PRESSURE_COLUMN_IN_SHARPPY_SOUNDING = 'pressure_mb'
HEIGHT_COLUMN_IN_SHARPPY_SOUNDING = 'geopotential_height_metres'
TEMPERATURE_COLUMN_IN_SHARPPY_SOUNDING = 'temperature_deg_c'
DEWPOINT_COLUMN_IN_SHARPPY_SOUNDING = 'dewpoint_deg_c'
U_WIND_COLUMN_IN_SHARPPY_SOUNDING = 'u_wind_kt'
V_WIND_COLUMN_IN_SHARPPY_SOUNDING = 'v_wind_kt'
SURFACE_COLUMN_IN_SHARPPY_SOUNDING = 'is_surface'

SHARPPY_SOUNDING_COLUMNS = [
    PRESSURE_COLUMN_IN_SHARPPY_SOUNDING, HEIGHT_COLUMN_IN_SHARPPY_SOUNDING,
    TEMPERATURE_COLUMN_IN_SHARPPY_SOUNDING, DEWPOINT_COLUMN_IN_SHARPPY_SOUNDING,
    U_WIND_COLUMN_IN_SHARPPY_SOUNDING, V_WIND_COLUMN_IN_SHARPPY_SOUNDING,
    SURFACE_COLUMN_IN_SHARPPY_SOUNDING
]

CONVECTIVE_TEMPERATURE_NAME = 'convective_temperature_kelvins'
HELICITY_NAMES_SHARPPY = ['srh1km', 'srh3km', 'left_esrh', 'right_esrh']
STORM_VELOCITY_NAME_SHARPPY = 'storm_velocity_m_s01'
SHARPPY_NAMES_FOR_MASKED_WIND_ARRAYS = [
    'mean_1km', 'mean_3km', 'mean_6km', 'mean_8km', 'mean_lcl_el']

X_COMPONENT_SUFFIX = 'x'
Y_COMPONENT_SUFFIX = 'y'
MAGNITUDE_SUFFIX = 'magnitude'
COSINE_SUFFIX = 'cos'
SINE_SUFFIX = 'sin'
VECTOR_SUFFIXES = [
    X_COMPONENT_SUFFIX, Y_COMPONENT_SUFFIX, MAGNITUDE_SUFFIX, COSINE_SUFFIX,
    SINE_SUFFIX]

STATISTIC_NAME_COLUMN = 'statistic_name'
SHARPPY_STATISTIC_NAME_COLUMN = 'statistic_name_sharppy'
CONVERSION_FACTOR_COLUMN = 'conversion_factor'
IS_VECTOR_COLUMN = 'is_vector'
IN_MUPCL_COLUMN = 'in_mupcl_object'
MIN_VALUES_FOR_NORM_COLUMN = 'min_values_for_normalization'
MAX_VALUES_FOR_NORM_COLUMN = 'max_values_for_normalization'

METAFILE_NAME = os.path.join(
    os.path.dirname(__file__), 'metadata_for_sounding_stats.p')
METADATA_COLUMNS = [
    STATISTIC_NAME_COLUMN, SHARPPY_STATISTIC_NAME_COLUMN,
    CONVERSION_FACTOR_COLUMN, IS_VECTOR_COLUMN, IN_MUPCL_COLUMN,
    MIN_VALUES_FOR_NORM_COLUMN, MAX_VALUES_FOR_NORM_COLUMN
]


def _remove_bad_pressure_levels(sounding_table):
    """Removes bad pressure levels from sounding.

    A "bad pressure level" is for one which any non-humidity value is NaN.

    If not enough good pressure levels are left, this method returns None.

    :param sounding_table: See input doc for `sounding_to_sharppy_units`.
    :return: sounding_table: Same as input, but without bad pressure levels.
    """

    columns_to_check = list(sounding_table)
    if nwp_model_utils.SPFH_COLUMN_FOR_SOUNDING_TABLES in columns_to_check:
        columns_to_check.remove(nwp_model_utils.SPFH_COLUMN_FOR_SOUNDING_TABLES)
    if nwp_model_utils.RH_COLUMN_FOR_SOUNDING_TABLES in columns_to_check:
        columns_to_check.remove(nwp_model_utils.RH_COLUMN_FOR_SOUNDING_TABLES)

    sounding_table = sounding_table.loc[
        sounding_table[columns_to_check].notnull().all(axis=1)]
    if len(sounding_table.index) < MIN_PRESSURE_LEVELS_IN_SOUNDING:
        return None

    return sounding_table


def _fill_missing_humidity(sounding_table):
    """Fills missing humidity values (by interpolation) in missing.

    If there are < 2 non-missing humidity values, this method returns None.

    :param sounding_table: See input doc for `sounding_to_sharppy_units`.
    :return: sounding_table: Same as input, but without missing humidity values.
    """

    if nwp_model_utils.SPFH_COLUMN_FOR_SOUNDING_TABLES in list(sounding_table):
        humidity_column = nwp_model_utils.SPFH_COLUMN_FOR_SOUNDING_TABLES
    else:
        humidity_column = nwp_model_utils.RH_COLUMN_FOR_SOUNDING_TABLES

    humidity_missing_flags = numpy.isnan(sounding_table[humidity_column].values)
    if not numpy.any(humidity_missing_flags):
        return sounding_table

    humidity_not_missing_indices = numpy.where(
        numpy.invert(humidity_missing_flags))[0]
    if len(humidity_not_missing_indices) < 2:
        return None

    control_heights_m_asl = sounding_table[
        nwp_model_utils.HEIGHT_COLUMN_FOR_SOUNDING_TABLES
    ].values[humidity_not_missing_indices]

    if humidity_column == nwp_model_utils.SPFH_COLUMN_FOR_SOUNDING_TABLES:
        control_specific_humidities_kg_kg01 = sounding_table[
            humidity_column].values[humidity_not_missing_indices]
    else:
        control_relative_humidities = (
            sounding_table[humidity_column].values[humidity_not_missing_indices]
            * PERCENT_TO_UNITLESS)
        control_temperatures_kelvins = sounding_table[
            nwp_model_utils.TEMPERATURE_COLUMN_FOR_SOUNDING_TABLES].values[
                humidity_not_missing_indices]
        control_pressures_pascals = (
            sounding_table[PRESSURE_COLUMN_IN_SHARPPY_SOUNDING].values[
                humidity_not_missing_indices] * MB_TO_PASCALS)

        control_dewpoints_kelvins = (
            moisture_conversions.relative_humidity_to_dewpoint(
                relative_humidities=control_relative_humidities,
                temperatures_kelvins=control_temperatures_kelvins,
                total_pressures_pascals=control_pressures_pascals))

        control_specific_humidities_kg_kg01 = (
            moisture_conversions.dewpoint_to_specific_humidity(
                dewpoints_kelvins=control_dewpoints_kelvins,
                total_pressures_pascals=control_pressures_pascals))

    interp_object = scipy.interpolate.interp1d(
        control_heights_m_asl, control_specific_humidities_kg_kg01,
        kind='linear', bounds_error=False, fill_value='extrapolate')

    humidity_missing_indices = numpy.where(humidity_missing_flags)[0]
    query_heights_m_asl = sounding_table[
        nwp_model_utils.HEIGHT_COLUMN_FOR_SOUNDING_TABLES].values[
            humidity_missing_indices]
    query_specific_humidities_kg_kg01 = interp_object(query_heights_m_asl)

    if humidity_column == nwp_model_utils.SPFH_COLUMN_FOR_SOUNDING_TABLES:
        sounding_table[humidity_column].values[
            humidity_missing_indices] = query_specific_humidities_kg_kg01
    else:
        query_pressures_pascals = (
            sounding_table[PRESSURE_COLUMN_IN_SHARPPY_SOUNDING].values[
                humidity_missing_indices] * MB_TO_PASCALS)
        query_temperatures_kelvins = sounding_table[
            nwp_model_utils.TEMPERATURE_COLUMN_FOR_SOUNDING_TABLES].values[
                humidity_missing_indices]

        query_dewpoints_kelvins = (
            moisture_conversions.specific_humidity_to_dewpoint(
                specific_humidities_kg_kg01=query_specific_humidities_kg_kg01,
                total_pressures_pascals=query_pressures_pascals))

        query_relative_humidities_percent = UNITLESS_TO_PERCENT * (
            moisture_conversions.dewpoint_to_relative_humidity(
                dewpoints_kelvins=query_dewpoints_kelvins,
                temperatures_kelvins=query_temperatures_kelvins,
                total_pressures_pascals=query_pressures_pascals))

        sounding_table[humidity_column].values[
            humidity_missing_indices] = query_relative_humidities_percent

    return sounding_table


def _get_dummy_sharppy_statistic_table(
        u_motion_m_s01, v_motion_m_s01, metadata_table):
    """Creates dummy table with SHARPpy sounding statistics.

    In this "dummy table," all statistics except storm motion are NaN.

    :param u_motion_m_s01: Eastward component of storm motion (metres per
        second).
    :param v_motion_m_s01: Northward component of storm motion (metres per
        second).
    :param metadata_table: pandas DataFrame created by
        `read_metadata_for_statistics`.
    :return: statistic_table_sharppy: pandas DataFrame, with the same columns as
        one created by `_compute_sounding_statistics`.
    """

    vector_flags = numpy.array(
        metadata_table[IS_VECTOR_COLUMN].values)
    vector_indices = numpy.where(vector_flags)[0]
    scalar_indices = numpy.where(numpy.invert(vector_flags))[0]

    dummy_dict = {}
    for this_index in scalar_indices:
        this_statistic_name = metadata_table[
            SHARPPY_STATISTIC_NAME_COLUMN].values[this_index]
        dummy_dict.update({this_statistic_name: numpy.full(1, numpy.nan)})

    statistic_table_sharppy = pandas.DataFrame.from_dict(dummy_dict)

    first_statistic_name = list(statistic_table_sharppy)[0]
    nested_array = statistic_table_sharppy[[
        first_statistic_name, first_statistic_name]].values.tolist()

    for this_index in vector_indices:
        this_statistic_name = metadata_table[
            SHARPPY_STATISTIC_NAME_COLUMN].values[this_index]
        statistic_table_sharppy = statistic_table_sharppy.assign(
            **{this_statistic_name: nested_array})

        if this_statistic_name == STORM_VELOCITY_NAME_SHARPPY:
            statistic_table_sharppy[this_statistic_name].values[
                0] = numpy.array([u_motion_m_s01, v_motion_m_s01])
        else:
            statistic_table_sharppy[this_statistic_name].values[
                0] = numpy.full(2, numpy.nan)

    return statistic_table_sharppy


def _get_nwp_fields_for_sounding(
        model_name, minimum_pressure_mb=0., return_dict=False):
    """Returns list of NWP fields needed to create sounding.

    Each field is one variable at one pressure level.

    N = number of fields
    P = number of pressure levels

    :param model_name: Model name (string).  This must be accepted by
        `nwp_model_utils.check_model_name`.
    :param minimum_pressure_mb: Minimum pressure (millibars).
    :param return_dict: Boolean flag.  Determines output variables (see below).

    :return: sounding_field_names: [only if return_dict = False]
        length-N list of field names (GewitterGefahr format).
    :return: sounding_field_names_grib1: [only if return_dict = False]
        length-N list of field names (grib1 format).

    :return: sounding_field_dict: [only if return_dict = True]
        Dictionary with the following keys.
    sounding_field_dict['geopotential_height_metres']: length-P list with names
        of height fields.
    sounding_field_dict['temperature_kelvins']: Same but for temperature fields.
    sounding_field_dict['humidity']: Same but for humidity fields.
    sounding_field_dict['u_wind_m_s01']: Same but for u-wind fields.
    sounding_field_dict['v_wind_m_s01']: Same but for v-wind fields.

    :return: pressure_levels_mb: [only if return_dict = True]
        length-P numpy array of pressure fields (millibars).
    """

    nwp_model_utils.check_model_name(model_name)
    error_checking.assert_is_geq(minimum_pressure_mb, 0.)
    error_checking.assert_is_boolean(return_dict)

    pressure_levels_mb = nwp_model_utils.get_pressure_levels(
        model_name=model_name, grid_id=nwp_model_utils.ID_FOR_130GRID)
    pressure_levels_mb = pressure_levels_mb[
        pressure_levels_mb >= minimum_pressure_mb]
    num_pressure_levels = len(pressure_levels_mb)

    basic_field_names, basic_field_names_grib1 = (
        nwp_model_utils.get_columns_in_sounding_table(model_name))
    num_basic_fields = len(basic_field_names)

    if return_dict:
        sounding_field_dict = {}
        for j in range(num_basic_fields):
            sounding_field_dict.update({basic_field_names[j]: []})

    else:
        sounding_field_names = []
        sounding_field_names_grib1 = []

    for j in range(num_basic_fields):
        for k in range(num_pressure_levels):
            this_field_name = '{0:s}_{1:d}mb'.format(
                basic_field_names[j], int(pressure_levels_mb[k]))

            if return_dict:
                sounding_field_dict[basic_field_names[j]].append(
                    this_field_name)
            else:
                sounding_field_names.append(this_field_name)
                sounding_field_names_grib1.append('{0:s}:{1:d} mb'.format(
                    basic_field_names_grib1[j], int(pressure_levels_mb[k])))

        if (basic_field_names[j] ==
                nwp_model_utils.TEMPERATURE_COLUMN_FOR_SOUNDING_TABLES):

            this_field_name, this_field_name_grib1 = (
                nwp_model_utils.get_lowest_temperature_name(model_name))

        elif basic_field_names[j] in [
                nwp_model_utils.RH_COLUMN_FOR_SOUNDING_TABLES,
                nwp_model_utils.SPFH_COLUMN_FOR_SOUNDING_TABLES]:

            this_field_name, this_field_name_grib1 = (
                nwp_model_utils.get_lowest_humidity_name(model_name))

        elif (basic_field_names[j] ==
              nwp_model_utils.HEIGHT_COLUMN_FOR_SOUNDING_TABLES):

            this_field_name, this_field_name_grib1 = (
                nwp_model_utils.get_lowest_height_name(model_name))

        elif (basic_field_names[j] ==
              nwp_model_utils.U_WIND_COLUMN_FOR_SOUNDING_TABLES):

            this_field_name, this_field_name_grib1 = (
                nwp_model_utils.get_lowest_u_wind_name(model_name))

        elif (basic_field_names[j] ==
              nwp_model_utils.V_WIND_COLUMN_FOR_SOUNDING_TABLES):

            this_field_name, this_field_name_grib1 = (
                nwp_model_utils.get_lowest_v_wind_name(model_name))

        if return_dict:
            sounding_field_dict[basic_field_names[j]].append(this_field_name)
        else:
            sounding_field_names.append(this_field_name)
            sounding_field_names_grib1.append(this_field_name_grib1)

    if return_dict:
        return sounding_field_dict, pressure_levels_mb

    this_field_name, this_field_name_grib1 = (
        nwp_model_utils.get_lowest_pressure_name(model_name))
    sounding_field_names.append(this_field_name)
    sounding_field_names_grib1.append(this_field_name_grib1)

    return sounding_field_names, sounding_field_names_grib1


def _remove_subsurface_pressure_levels(sounding_table_sharppy, delete_rows):
    """Removes subsurface pressure levels from sounding.

    :param sounding_table_sharppy: See input doc for
        `_compute_sounding_statistics`.
    :param delete_rows: Boolean flag.  If True, will delete all subsurface rows
        except 1000 mb.  If False, will replace all subsurface data with
        sentinel values.
    :return: sounding_table_sharppy: Same as input, but maybe with fewer rows or
        with sentinel values.
    """

    surface_flags = sounding_table_sharppy[
        SURFACE_COLUMN_IN_SHARPPY_SOUNDING].values
    if not numpy.any(surface_flags):
        return sounding_table_sharppy

    surface_index = numpy.where(surface_flags)[0][0]
    surface_height_m_asl = sounding_table_sharppy[
        HEIGHT_COLUMN_IN_SHARPPY_SOUNDING].values[surface_index]

    subsurface_flags = (
        sounding_table_sharppy[HEIGHT_COLUMN_IN_SHARPPY_SOUNDING].values <
        surface_height_m_asl)
    subsurface_indices = numpy.where(subsurface_flags)[0]

    if delete_rows:
        pressure_1000mb_flags = (
            sounding_table_sharppy[PRESSURE_COLUMN_IN_SHARPPY_SOUNDING].values ==
            1000)
        bad_row_indices = numpy.where(numpy.logical_and(
            subsurface_flags, numpy.invert(pressure_1000mb_flags)))[0]

        sounding_table_sharppy = sounding_table_sharppy.drop(
            sounding_table_sharppy.index[bad_row_indices], axis=0, inplace=False
        ).reset_index(drop=True)

        subsurface_flags = (
            sounding_table_sharppy[HEIGHT_COLUMN_IN_SHARPPY_SOUNDING].values <
            surface_height_m_asl)
        subsurface_indices = numpy.where(subsurface_flags)[0]

    sounding_table_sharppy[TEMPERATURE_COLUMN_IN_SHARPPY_SOUNDING].values[
        subsurface_indices] = SENTINEL_VALUE_FOR_SHARPPY
    sounding_table_sharppy[DEWPOINT_COLUMN_IN_SHARPPY_SOUNDING].values[
        subsurface_indices] = SENTINEL_VALUE_FOR_SHARPPY
    sounding_table_sharppy[U_WIND_COLUMN_IN_SHARPPY_SOUNDING].values[
        subsurface_indices] = SENTINEL_VALUE_FOR_SHARPPY
    sounding_table_sharppy[V_WIND_COLUMN_IN_SHARPPY_SOUNDING].values[
        subsurface_indices] = SENTINEL_VALUE_FOR_SHARPPY

    return sounding_table_sharppy


def _sort_sounding_by_height(sounding_table_sharppy):
    """Sorts sounding levels by increasing height.

    :param sounding_table_sharppy: See input doc for
        `_compute_sounding_statistics`.
    :return: sounding_table_sharppy: Same as input, but maybe sorted
        differently.
    """

    return sounding_table_sharppy.sort_values(
        HEIGHT_COLUMN_IN_SHARPPY_SOUNDING, axis=0, ascending=True, inplace=False)


def _remove_redundant_pressure_levels(sounding_table_sharppy_sorted):
    """Removes pressure levels very close to the surface.

    :param sounding_table_sharppy_sorted: pandas DataFrame created by
        `_sort_sounding_by_height`.
    :return: sounding_table_sharppy_sorted: Same as input, but maybe with fewer
        rows.
    """

    surface_flags = sounding_table_sharppy_sorted[
        SURFACE_COLUMN_IN_SHARPPY_SOUNDING].values
    if not numpy.any(surface_flags):
        return sounding_table_sharppy_sorted

    surface_index = numpy.where(surface_flags)[0][0]
    bad_indices = []

    heights_m_asl = sounding_table_sharppy_sorted[
        HEIGHT_COLUMN_IN_SHARPPY_SOUNDING].values
    pressures_mb = sounding_table_sharppy_sorted[
        HEIGHT_COLUMN_IN_SHARPPY_SOUNDING].values

    if surface_index != len(surface_flags) - 1:
        height_diff_metres = numpy.absolute(
            heights_m_asl[surface_index + 1] - heights_m_asl[surface_index])
        pressure_diff_mb = numpy.absolute(
            pressures_mb[surface_index + 1] - pressures_mb[surface_index])

        if (height_diff_metres < REDUNDANT_HEIGHT_TOLERANCE_METRES or
                pressure_diff_mb < REDUNDANT_PRESSURE_TOLERANCE_MB):
            bad_indices.append(surface_index + 1)

    if surface_index != 0:
        height_diff_metres = numpy.absolute(
            heights_m_asl[surface_index - 1] - heights_m_asl[surface_index])
        pressure_diff_mb = numpy.absolute(
            pressures_mb[surface_index - 1] - pressures_mb[surface_index])

        if (height_diff_metres < REDUNDANT_HEIGHT_TOLERANCE_METRES or
                pressure_diff_mb < REDUNDANT_PRESSURE_TOLERANCE_MB):
            bad_indices.append(surface_index + 1)

    if not len(bad_indices):
        return sounding_table_sharppy_sorted

    bad_indices = numpy.array(bad_indices, dtype=int)
    return sounding_table_sharppy_sorted.drop(
        sounding_table_sharppy_sorted.index[bad_indices], axis=0, inplace=False
    ).reset_index(drop=True)


def _adjust_srw_for_storm_motion(profile_object, u_motion_kt, v_motion_kt):
    """Adjusts storm-relative winds to account for storm motion.

    :param profile_object: Instance of `sharppy.sharptab.Profile`.
    :param u_motion_kt: Eastward component of storm motion (knots).
    :param v_motion_kt: Northward component of storm motion (knots).
    :return: profile_object: Same as input, but maybe with different storm-
        relative winds.
    """

    surface_pressure_mb = profile_object.pres[profile_object.sfc]
    pressure_at_1km_mb = sharppy_interp.pres(
        profile_object, sharppy_interp.to_msl(profile_object, 1000.))
    pressure_at_2km_mb = sharppy_interp.pres(
        profile_object, sharppy_interp.to_msl(profile_object, 2000.))
    pressure_at_3km_mb = sharppy_interp.pres(
        profile_object, sharppy_interp.to_msl(profile_object, 3000.))
    pressure_at_4km_mb = sharppy_interp.pres(
        profile_object, sharppy_interp.to_msl(profile_object, 4000.))
    pressure_at_5km_mb = sharppy_interp.pres(
        profile_object, sharppy_interp.to_msl(profile_object, 5000.))
    pressure_at_6km_mb = sharppy_interp.pres(
        profile_object, sharppy_interp.to_msl(profile_object, 6000.))
    pressure_at_8km_mb = sharppy_interp.pres(
        profile_object, sharppy_interp.to_msl(profile_object, 8000.))
    pressure_at_9km_mb = sharppy_interp.pres(
        profile_object, sharppy_interp.to_msl(profile_object, 9000.))
    pressure_at_11km_mb = sharppy_interp.pres(
        profile_object, sharppy_interp.to_msl(profile_object, 11000.))

    this_depth_metres = (profile_object.mupcl.elhght - profile_object.ebotm) / 2
    effective_bulk_layer_top_mb = sharppy_interp.pres(
        profile_object, sharppy_interp.to_msl(
            profile_object, profile_object.ebotm + this_depth_metres))

    profile_object.srw_1km = sharppy_winds.sr_wind(
        profile_object, pbot=surface_pressure_mb, ptop=pressure_at_1km_mb,
        stu=u_motion_kt, stv=v_motion_kt)
    profile_object.srw_0_2km = sharppy_winds.sr_wind(
        profile_object, pbot=surface_pressure_mb, ptop=pressure_at_2km_mb,
        stu=u_motion_kt, stv=v_motion_kt)
    profile_object.srw_3km = sharppy_winds.sr_wind(
        profile_object, pbot=surface_pressure_mb, ptop=pressure_at_3km_mb,
        stu=u_motion_kt, stv=v_motion_kt)
    profile_object.srw_6km = sharppy_winds.sr_wind(
        profile_object, pbot=surface_pressure_mb, ptop=pressure_at_6km_mb,
        stu=u_motion_kt, stv=v_motion_kt)
    profile_object.srw_8km = sharppy_winds.sr_wind(
        profile_object, pbot=surface_pressure_mb, ptop=pressure_at_8km_mb,
        stu=u_motion_kt, stv=v_motion_kt)
    profile_object.srw_4_5km = sharppy_winds.sr_wind(
        profile_object, pbot=pressure_at_4km_mb, ptop=pressure_at_5km_mb,
        stu=u_motion_kt, stv=v_motion_kt)
    profile_object.srw_4_6km = sharppy_winds.sr_wind(
        profile_object, pbot=pressure_at_4km_mb, ptop=pressure_at_6km_mb,
        stu=u_motion_kt, stv=v_motion_kt)
    profile_object.srw_9_11km = sharppy_winds.sr_wind(
        profile_object, pbot=pressure_at_9km_mb, ptop=pressure_at_11km_mb,
        stu=u_motion_kt, stv=v_motion_kt)
    profile_object.srw_ebw = sharppy_winds.sr_wind(
        profile_object, pbot=profile_object.ebottom,
        ptop=effective_bulk_layer_top_mb, stu=u_motion_kt, stv=v_motion_kt)
    profile_object.srw_eff = sharppy_winds.sr_wind(
        profile_object, pbot=profile_object.ebottom, ptop=profile_object.etop,
        stu=u_motion_kt, stv=v_motion_kt)
    profile_object.srw_lcl_el = sharppy_winds.sr_wind(
        profile_object, pbot=profile_object.mupcl.lclpres,
        ptop=profile_object.mupcl.elpres, stu=u_motion_kt, stv=v_motion_kt)

    return profile_object


def _adjust_stats_for_storm_motion(profile_object, u_motion_kt, v_motion_kt):
    """Adjusts sounding statistics to account for storm motion.

    :param profile_object: Instance of `sharppy.sharptab.Profile`.
    :param u_motion_kt: Eastward component of storm motion (knots).
    :param v_motion_kt: Northward component of storm motion (knots).
    :return: profile_object: Same as input, but maybe with different statistics.
    """

    profile_object = _adjust_srw_for_storm_motion(
        profile_object, u_motion_kt=u_motion_kt, v_motion_kt=v_motion_kt)

    profile_object.critical_angle = sharppy_winds.critical_angle(
        profile_object, stu=u_motion_kt, stv=v_motion_kt)
    profile_object.srh1km = sharppy_winds.helicity(
        profile_object, 0., 1000., stu=u_motion_kt, stv=v_motion_kt)
    profile_object.srh3km = sharppy_winds.helicity(
        profile_object, 0., 3000., stu=u_motion_kt, stv=v_motion_kt)
    profile_object.ehi1km = sharppy_params.ehi(
        profile_object, profile_object.mupcl, 0., 1000., stu=u_motion_kt,
        stv=v_motion_kt)
    profile_object.ehi3km = sharppy_params.ehi(
        profile_object, profile_object.mupcl, 0., 3000., stu=u_motion_kt,
        stv=v_motion_kt)

    effective_layer_srh_j_kg01 = sharppy_winds.helicity(
        profile_object, profile_object.ebotm, profile_object.etopm,
        stu=u_motion_kt, stv=v_motion_kt)[1]
    profile_object.stp_cin = sharppy_params.stp_cin(
        profile_object.mlpcl.bplus, effective_layer_srh_j_kg01,
        profile_object.ebwspd * KT_TO_METRES_PER_SECOND,
        profile_object.mlpcl.lclhght, profile_object.mlpcl.bminus)

    this_shear_magnitude_m_s01 = KT_TO_METRES_PER_SECOND * numpy.sqrt(
        profile_object.sfc_6km_shear[0] ** 2 +
        profile_object.sfc_6km_shear[1] ** 2)
    profile_object.stp_fixed = sharppy_params.stp_fixed(
        profile_object.sfcpcl.bplus, profile_object.sfcpcl.lclhght,
        profile_object.srh1km[0], this_shear_magnitude_m_s01)

    profile_object.get_traj()  # Recomputes updraft tilt.
    return profile_object


def _fix_sharppy_wind_vector(sharppy_wind_vector):
    """Converts wind vector from SHARPpy format to new format.

    SHARPpy format = masked array with direction and magnitude
    New format = normal numpy array with u- and v-components

    :param sharppy_wind_vector: Wind vector in SHARPpy format.
    :return: wind_vector: Wind vector in new format.
    """

    if not sharppy_wind_vector[0] or not sharppy_wind_vector[1]:
        return numpy.full(2, numpy.nan)

    return numpy.asarray(sharppy_utils.vec2comp(
        sharppy_wind_vector[0], sharppy_wind_vector[1]))


def _fix_all_sharppy_wind_vectors(profile_object):
    """Converts all wind vectors in SHARPpy profile to new format.

    See `_fix_sharppy_wind_vector` for definitions of the two formats.

    :param profile_object: Instance of `sharppy.sharptab.Profile`.
    :return: profile_object: Same as input, but with all wind vectors in new
        format.
    """

    for this_sharppy_name in SHARPPY_NAMES_FOR_MASKED_WIND_ARRAYS:
        setattr(profile_object, this_sharppy_name, _fix_sharppy_wind_vector(
            getattr(profile_object, this_sharppy_name)))

    return profile_object


def _extract_stats_from_sharppy_profile(profile_object, metadata_table):
    """Extracts sounding statistics from SHARPpy profile.

    :param profile_object: Instance of `sharppy.sharptab.Profile`.
    :param metadata_table: pandas DataFrame created by
        `read_metadata_for_statistics`.
    :return: statistic_table_sharppy: pandas DataFrame with sounding statistics
        in SHARPpy format.  In other words, both column names and units are in
        SHARPpy format.
    """

    profile_object.mupcl.brndenom = profile_object.mupcl.brnshear
    profile_object.mupcl.brnshear = numpy.array(
        [profile_object.mupcl.brnu, profile_object.mupcl.brnv])

    is_vector_flags = metadata_table[IS_VECTOR_COLUMN].values
    vector_indices = numpy.where(is_vector_flags)[0]
    scalar_indices = numpy.where(numpy.invert(is_vector_flags))[0]
    statistic_names_sharppy = metadata_table[
        SHARPPY_STATISTIC_NAME_COLUMN].values

    statistic_dict_sharppy = {}
    for this_index in scalar_indices:
        statistic_dict_sharppy.update({
            statistic_names_sharppy[this_index]: numpy.array([numpy.nan])
        })
    statistic_table_sharppy = pandas.DataFrame.from_dict(statistic_dict_sharppy)

    first_statistic_name = list(statistic_table_sharppy)[0]
    nested_array = statistic_table_sharppy[[
        first_statistic_name, first_statistic_name]].values.tolist()

    argument_dict = {}
    for this_index in vector_indices:
        argument_dict.update(
            {statistic_names_sharppy[this_index]: nested_array})
    statistic_table_sharppy = statistic_table_sharppy.assign(**argument_dict)

    num_statistics = len(statistic_names_sharppy)
    for j in range(num_statistics):
        if statistic_names_sharppy[j] in HELICITY_NAMES_SHARPPY:
            this_vector = getattr(profile_object, statistic_names_sharppy[j])
            statistic_table_sharppy[
                statistic_names_sharppy[j]].values[0] = this_vector[1]

        elif metadata_table[IN_MUPCL_COLUMN].values[j]:
            statistic_table_sharppy[statistic_names_sharppy[j]].values[
                0] = getattr(profile_object.mupcl, statistic_names_sharppy[j])
        else:
            statistic_table_sharppy[statistic_names_sharppy[j]].values[
                0] = getattr(profile_object, statistic_names_sharppy[j])

    return statistic_table_sharppy


def _split_vector_column(input_table, conversion_factor=1.):
    """Splits array of 2-D vectors into 5 arrays, listed below.

    - x-component
    - y-component
    - magnitude
    - sine of direction
    - cosine of direction

    N = number of vectors

    :param input_table: pandas DataFrame with one column, where each row is a
        2-D vector.
    :param conversion_factor: Both x- and y-components will be multiplied by
        this factor.
    :return: vector_dict: Dictionary with the following keys (assuming that the
        original column name is "foo").
    vector_dict['foo_x']: length-N numpy array of x-components.
    vector_dict['foo_y']: length-N numpy array of y-components.
    vector_dict['foo_magnitude']: length-N numpy array of magnitudes.
    vector_dict['foo_cos']: length-N numpy array with cosines of vector
        directions.
    vector_dict['foo_sin']: length-N numpy array with sines of vector
        directions.
    """

    input_column = list(input_table)[0]
    num_vectors = len(input_table.index)
    x_components = numpy.full(num_vectors, numpy.nan)
    y_components = numpy.full(num_vectors, numpy.nan)

    for i in range(num_vectors):
        x_components[i] = input_table[input_column].values[i][0]
        y_components[i] = input_table[input_column].values[i][1]

    x_components *= conversion_factor
    y_components *= conversion_factor
    magnitudes = numpy.sqrt(x_components ** 2 + y_components ** 2)
    cosines = x_components / magnitudes
    sines = y_components / magnitudes

    x_component_name = add_vector_suffix_to_stat_name(
        basic_statistic_name=input_column, vector_suffix=X_COMPONENT_SUFFIX)
    y_component_name = add_vector_suffix_to_stat_name(
        basic_statistic_name=input_column, vector_suffix=Y_COMPONENT_SUFFIX)
    magnitude_name = add_vector_suffix_to_stat_name(
        basic_statistic_name=input_column, vector_suffix=MAGNITUDE_SUFFIX)
    cosine_name = add_vector_suffix_to_stat_name(
        basic_statistic_name=input_column, vector_suffix=COSINE_SUFFIX)
    sine_name = add_vector_suffix_to_stat_name(
        basic_statistic_name=input_column, vector_suffix=SINE_SUFFIX)

    return {
        x_component_name: x_components, y_component_name: y_components,
        magnitude_name: magnitudes, cosine_name: cosines, sine_name: sines
    }


def _sounding_to_sharppy_units(sounding_table):
    """Converts sounding table from GewitterGefahr units to SHARPpy units.

    This method overwrites the input table (does not create a new copy).

    :param sounding_table: pandas DataFrame with the following columns.  Must
        contain *either* relative or specific humidity.  Each row is one
        pressure level.
    sounding_table.pressure_mb: Pressure (millibars).
    sounding_table.temperature_kelvins: Temperature.
    sounding_table.relative_humidity_percent: [optional] Relative humidity.
    sounding_table.specific_humidity: [optional] Specific humidity.
    sounding_table.geopotential_height_metres: Geopotential height.
    sounding_table.u_wind_m_s01: u-wind (metres per second).
    sounding_table.v_wind_m_s01: v-wind (metres per second).

    :return: sounding_table_sharppy: pandas DataFrame with the following
        columns. Each row is one pressure level.
    sounding_table_sharppy.pressure_mb: Pressure (millibars).
    sounding_table_sharppy.geopotential_height_metres: Geopotential height.
    sounding_table_sharppy.temperature_deg_c: Temperature.
    sounding_table_sharppy.dewpoint_deg_c: Dewpoint.
    sounding_table_sharppy.u_wind_kt: Eastward wind component (knots).
    sounding_table_sharppy.v_wind_kt: Northward wind component (knots).
    """

    columns_to_drop = [
        nwp_model_utils.TEMPERATURE_COLUMN_FOR_SOUNDING_TABLES,
        nwp_model_utils.U_WIND_COLUMN_FOR_SOUNDING_TABLES,
        nwp_model_utils.V_WIND_COLUMN_FOR_SOUNDING_TABLES
    ]

    if nwp_model_utils.SPFH_COLUMN_FOR_SOUNDING_TABLES in sounding_table:
        columns_to_drop.append(nwp_model_utils.SPFH_COLUMN_FOR_SOUNDING_TABLES)

        dewpoints_kelvins = moisture_conversions.specific_humidity_to_dewpoint(
            specific_humidities_kg_kg01=sounding_table[
                nwp_model_utils.SPFH_COLUMN_FOR_SOUNDING_TABLES].values,
            total_pressures_pascals=MB_TO_PASCALS * sounding_table[
                PRESSURE_COLUMN_IN_SHARPPY_SOUNDING].values)

    else:
        columns_to_drop.append(nwp_model_utils.RH_COLUMN_FOR_SOUNDING_TABLES)

        relative_humidities_percent = sounding_table[
            nwp_model_utils.RH_COLUMN_FOR_SOUNDING_TABLES].values
        relative_humidities_percent[
            relative_humidities_percent < MIN_RELATIVE_HUMIDITY_PERCENT
        ] = MIN_RELATIVE_HUMIDITY_PERCENT
        sounding_table = sounding_table.assign(
            **{nwp_model_utils.RH_COLUMN_FOR_SOUNDING_TABLES:
                   relative_humidities_percent})

        dewpoints_kelvins = moisture_conversions.relative_humidity_to_dewpoint(
            relative_humidities=PERCENT_TO_UNITLESS * sounding_table[
                nwp_model_utils.RH_COLUMN_FOR_SOUNDING_TABLES].values,
            temperatures_kelvins=sounding_table[
                nwp_model_utils.TEMPERATURE_COLUMN_FOR_SOUNDING_TABLES].values,
            total_pressures_pascals=MB_TO_PASCALS * sounding_table[
                PRESSURE_COLUMN_IN_SHARPPY_SOUNDING].values)

    dewpoints_deg_c = temperature_conversions.kelvins_to_celsius(
        dewpoints_kelvins)
    temperatures_deg_c = temperature_conversions.kelvins_to_celsius(
        sounding_table[
            nwp_model_utils.TEMPERATURE_COLUMN_FOR_SOUNDING_TABLES].values)
    u_winds_kt = METRES_PER_SECOND_TO_KT * sounding_table[
        nwp_model_utils.U_WIND_COLUMN_FOR_SOUNDING_TABLES].values
    v_winds_kt = METRES_PER_SECOND_TO_KT * sounding_table[
        nwp_model_utils.V_WIND_COLUMN_FOR_SOUNDING_TABLES].values

    argument_dict = {
        DEWPOINT_COLUMN_IN_SHARPPY_SOUNDING: dewpoints_deg_c,
        TEMPERATURE_COLUMN_IN_SHARPPY_SOUNDING: temperatures_deg_c,
        U_WIND_COLUMN_IN_SHARPPY_SOUNDING: u_winds_kt,
        V_WIND_COLUMN_IN_SHARPPY_SOUNDING: v_winds_kt
    }
    sounding_table = sounding_table.assign(**argument_dict)
    return sounding_table.drop(columns_to_drop, axis=1)


def _create_query_points(storm_object_table, lead_times_sec):
    """Creates set of query points for interpolation.

    Each query point consists of (latitude, longitude, time).

    T = number of lead times

    :param storm_object_table: pandas DataFrame with columns documented in
        `storm_tracking_io.write_processed_file`.
    :param lead_times_sec: length-T numpy array of lead times.  For each lead
        time t, each storm object will be extrapolated t seconds into the
        future, along its estimated motion vector at the initial time.
    :return: query_point_table: pandas DataFrame with the following columns,
        where each row is one extrapolated storm object.
    query_point_table.centroid_lat_deg: Latitude (deg N) of extrapolated
        centroid.
    query_point_table.centroid_lng_deg: Longitude (deg E) of extrapolated
        centroid.
    query_point_table.unix_time_sec: Time of extrapolated point.
    query_point_table.lead_time_seconds: Lead time (amount of time over which
        storm object was extrapolated).
    query_point_table.east_velocity_m_s01: u-component of storm motion (metres
        per second).
    query_point_table.north_velocity_m_s01: v-component of storm motion (metres
        per second).
    """

    if numpy.any(lead_times_sec > 0):
        storm_speeds_m_s01, geodetic_bearings_deg = (
            geodetic_utils.xy_components_to_displacements_and_bearings(
                storm_object_table[tracking_utils.EAST_VELOCITY_COLUMN].values,
                storm_object_table[tracking_utils.NORTH_VELOCITY_COLUMN].values
            ))

    num_storm_objects = len(storm_object_table.index)
    num_lead_times = len(lead_times_sec)
    list_of_query_point_tables = [None] * num_lead_times

    for i in range(num_lead_times):
        if lead_times_sec[i] == 0:
            list_of_query_point_tables[i] = storm_object_table[[
                tracking_utils.CENTROID_LAT_COLUMN,
                tracking_utils.CENTROID_LNG_COLUMN, tracking_utils.TIME_COLUMN,
                tracking_utils.EAST_VELOCITY_COLUMN,
                tracking_utils.NORTH_VELOCITY_COLUMN]]

            argument_dict = {
                LEAD_TIME_COLUMN: numpy.full(num_storm_objects, 0, dtype=int)}
            list_of_query_point_tables[i] = (
                list_of_query_point_tables[i].assign(**argument_dict))

        else:
            these_extrap_latitudes_deg, these_extrap_longitudes_deg = (
                geodetic_utils.start_points_and_distances_and_bearings_to_endpoints(
                    start_latitudes_deg=storm_object_table[
                        tracking_utils.CENTROID_LAT_COLUMN].values,
                    start_longitudes_deg=storm_object_table[
                        tracking_utils.CENTROID_LNG_COLUMN].values,
                    displacements_metres=storm_speeds_m_s01 * lead_times_sec[i],
                    geodetic_bearings_deg=geodetic_bearings_deg))

            print (these_extrap_latitudes_deg - storm_object_table[
                tracking_utils.CENTROID_LAT_COLUMN].values)
            print (these_extrap_longitudes_deg - storm_object_table[
                tracking_utils.CENTROID_LNG_COLUMN].values)

            this_dict = {
                tracking_utils.CENTROID_LAT_COLUMN: these_extrap_latitudes_deg,
                tracking_utils.CENTROID_LNG_COLUMN: these_extrap_longitudes_deg,
                tracking_utils.TIME_COLUMN:
                    (storm_object_table[tracking_utils.TIME_COLUMN].values +
                     lead_times_sec[i]),
                tracking_utils.EAST_VELOCITY_COLUMN: storm_object_table[
                    tracking_utils.EAST_VELOCITY_COLUMN].values,
                tracking_utils.NORTH_VELOCITY_COLUMN: storm_object_table[
                    tracking_utils.NORTH_VELOCITY_COLUMN].values,
                LEAD_TIME_COLUMN: numpy.full(
                    num_storm_objects, lead_times_sec[i], dtype=int)
            }
            list_of_query_point_tables[i] = pandas.DataFrame.from_dict(
                this_dict)

        if i == 0:
            continue

        list_of_query_point_tables[i], _ = list_of_query_point_tables[i].align(
            list_of_query_point_tables[0], axis=1)

    return pandas.concat(list_of_query_point_tables, axis=0, ignore_index=True)


def _interp_soundings_from_nwp(
        query_point_table, model_name, top_grib_directory_name, grid_id=None,
        wgrib_exe_name=grib_io.WGRIB_EXE_NAME_DEFAULT,
        wgrib2_exe_name=grib_io.WGRIB2_EXE_NAME_DEFAULT,
        raise_error_if_missing=False):
    """Interpolates soundings from NWP model to query points.

    Each query point consists of (latitude, longitude, time).

    :param query_point_table: pandas DataFrame created by
        `_create_query_points`.
    :param model_name: Name of NWP model.
    :param top_grib_directory_name: Name of top-level directory with grib data
        from said model.
    :param grid_id: String ID for model grid.
    :param wgrib_exe_name: Path to wgrib executable.
    :param wgrib2_exe_name: Path to wgrib2 executable.
    :param raise_error_if_missing: See documentation for
        `interp.interp_nwp_from_xy_grid`.
    :return: interp_table: pandas DataFrame, where each column is one field and
        each row is one query point.  Column names are given by the list
        `sounding_field_names` returned by `_get_nwp_fields_for_sounding`.
    """

    sounding_field_names, sounding_field_names_grib1 = (
        _get_nwp_fields_for_sounding(model_name=model_name, return_dict=False))

    return interp.interp_nwp_from_xy_grid(
        query_point_table=query_point_table, model_name=model_name,
        grid_id=grid_id, field_names=sounding_field_names,
        field_names_grib1=sounding_field_names_grib1,
        top_grib_directory_name=top_grib_directory_name,
        temporal_interp_method=TEMPORAL_INTERP_METHOD,
        spatial_interp_method=SPATIAL_INTERP_METHOD,
        wgrib_exe_name=wgrib_exe_name, wgrib2_exe_name=wgrib2_exe_name,
        raise_error_if_missing=raise_error_if_missing)


def _interp_soundings_from_ruc(
        query_point_table, top_grib_directory_name,
        wgrib_exe_name=grib_io.WGRIB_EXE_NAME_DEFAULT,
        wgrib2_exe_name=grib_io.WGRIB2_EXE_NAME_DEFAULT,
        raise_error_if_missing=False):
    """Interpolates soundings from RUC model to query points.

    This method may interpolate from different RUC grids at different time
    steps.

    :param query_point_table: See doc for `_interp_soundings_from_nwp`.
    :param top_grib_directory_name: Same.
    :param wgrib_exe_name: Same.
    :param wgrib2_exe_name: Same.
    :param raise_error_if_missing: Same.
    :return: interp_table: Same.
    """

    sounding_field_names, sounding_field_names_grib1 = (
        _get_nwp_fields_for_sounding(
            model_name=nwp_model_utils.RUC_MODEL_NAME, return_dict=False))

    return interp.interp_ruc_all_grids(
        query_point_table, field_names=sounding_field_names,
        field_names_grib1=sounding_field_names_grib1,
        top_grib_directory_name=top_grib_directory_name,
        temporal_interp_method=TEMPORAL_INTERP_METHOD,
        spatial_interp_method=SPATIAL_INTERP_METHOD,
        wgrib_exe_name=wgrib_exe_name, wgrib2_exe_name=wgrib2_exe_name,
        raise_error_if_missing=raise_error_if_missing)


def _create_sharppy_sounding_tables(interp_table, model_name):
    """Converts each input table to a sounding table in SHARPpy format.

    N = number of soundings

    :param interp_table: N-row pandas DataFrame created by
        `_interp_soundings_from_nwp` or `_interp_soundings_from_ruc`.  Each
        column is one NWP field (one variable at one pressure level).
    :param model_name: Name of NWP model used to create `interp_table`.
    :return: list_of_sharppy_sounding_tables: length-N list of sounding tables.
        Each table contains columns listed in input documentation for
        `_compute_sounding_statistics`.
    """

    sounding_field_dict, pressure_levels_mb = _get_nwp_fields_for_sounding(
        model_name=model_name, return_dict=True)
    basic_field_names = sounding_field_dict.keys()
    num_basic_fields = len(basic_field_names)

    lowest_pressure_name, _ = nwp_model_utils.get_lowest_pressure_name(
        model_name)

    pressure_levels_mb = numpy.concatenate((
        pressure_levels_mb, numpy.array([numpy.nan])))
    is_surface_flags = numpy.full(len(pressure_levels_mb), False, dtype=int)
    is_surface_flags[-1] = True

    generic_sounding_dict = {
        PRESSURE_COLUMN_IN_SHARPPY_SOUNDING: pressure_levels_mb,
        SURFACE_COLUMN_IN_SHARPPY_SOUNDING: is_surface_flags}
    generic_sounding_table = pandas.DataFrame.from_dict(generic_sounding_dict)

    num_soundings = len(interp_table.index)
    list_of_sharppy_sounding_tables = [generic_sounding_table] * num_soundings

    for i in range(num_soundings):
        for j in range(num_basic_fields):
            these_sounding_field_names = sounding_field_dict[
                basic_field_names[j]]

            argument_dict = {
                basic_field_names[j]:
                    interp_table[these_sounding_field_names].values[i]}
            list_of_sharppy_sounding_tables[i] = (
                list_of_sharppy_sounding_tables[i].assign(**argument_dict))

        list_of_sharppy_sounding_tables[i][
            PRESSURE_COLUMN_IN_SHARPPY_SOUNDING].values[-1] = (
                PASCALS_TO_MB * interp_table[lowest_pressure_name].values[i])

        list_of_sharppy_sounding_tables[i] = _remove_bad_pressure_levels(
            list_of_sharppy_sounding_tables[i])
        if list_of_sharppy_sounding_tables[i] is None:
            continue

        list_of_sharppy_sounding_tables[i] = _fill_missing_humidity(
            list_of_sharppy_sounding_tables[i])
        if list_of_sharppy_sounding_tables[i] is None:
            continue

        list_of_sharppy_sounding_tables[i] = _sounding_to_sharppy_units(
            list_of_sharppy_sounding_tables[i])

    return list_of_sharppy_sounding_tables


def _compute_sounding_statistics(
        sounding_table_sharppy, u_motion_m_s01, v_motion_m_s01, metadata_table):
    """Uses SHARPpy to compute statistics for a single sounding.

    :param sounding_table_sharppy: pandas DataFrame with the following columns.
        Each row is one pressure level.
    sounding_table_sharppy.pressure_mb: Pressure (millibars).
    sounding_table_sharppy.geopotential_height_metres: Geopotential height.
    sounding_table_sharppy.temperature_deg_c: Temperature.
    sounding_table_sharppy.dewpoint_deg_c: Dewpoint.
    sounding_table_sharppy.u_wind_kt: Eastward wind component (knots).
    sounding_table_sharppy.v_wind_kt: Northward wind component (knots).
    sounding_table_sharppy.is_surface: Boolean flag, indicating which row is the
        surface level.

    :param u_motion_m_s01: Eastward component of storm motion (knots).
    :param v_motion_m_s01: Northward component of storm motion (knots).
    :param metadata_table: pandas DataFrame created by
        `read_metadata_for_statistics`.
    :return: statistic_table_sharpy: pandas DataFrame with sounding
        statistics in SHARPpy format.  In other words, both column names and
        units are in SHARPpy format.
    """

    error_checking.assert_is_not_nan(u_motion_m_s01)
    error_checking.assert_is_not_nan(v_motion_m_s01)

    sounding_table_sharppy = _remove_subsurface_pressure_levels(
        sounding_table_sharppy=sounding_table_sharppy, delete_rows=False)
    sounding_table_sharppy = _sort_sounding_by_height(sounding_table_sharppy)
    sounding_table_sharppy = _remove_redundant_pressure_levels(
        sounding_table_sharppy)

    try:
        profile_object = sharppy_profile.create_profile(
            profile='convective',
            pres=sounding_table_sharppy[
                PRESSURE_COLUMN_IN_SHARPPY_SOUNDING].values,
            hght=sounding_table_sharppy[HEIGHT_COLUMN_IN_SHARPPY_SOUNDING].values,
            tmpc=sounding_table_sharppy[
                TEMPERATURE_COLUMN_IN_SHARPPY_SOUNDING].values,
            dwpc=sounding_table_sharppy[
                DEWPOINT_COLUMN_IN_SHARPPY_SOUNDING].values,
            u=sounding_table_sharppy[U_WIND_COLUMN_IN_SHARPPY_SOUNDING].values,
            v=sounding_table_sharppy[V_WIND_COLUMN_IN_SHARPPY_SOUNDING].values)

    except:
        sounding_table_sharppy = _remove_subsurface_pressure_levels(
            sounding_table_sharppy=sounding_table_sharppy, delete_rows=True)

        profile_object = sharppy_profile.create_profile(
            profile='convective',
            pres=sounding_table_sharppy[
                PRESSURE_COLUMN_IN_SHARPPY_SOUNDING].values,
            hght=sounding_table_sharppy[
                HEIGHT_COLUMN_IN_SHARPPY_SOUNDING].values,
            tmpc=sounding_table_sharppy[
                TEMPERATURE_COLUMN_IN_SHARPPY_SOUNDING].values,
            dwpc=sounding_table_sharppy[
                DEWPOINT_COLUMN_IN_SHARPPY_SOUNDING].values,
            u=sounding_table_sharppy[U_WIND_COLUMN_IN_SHARPPY_SOUNDING].values,
            v=sounding_table_sharppy[V_WIND_COLUMN_IN_SHARPPY_SOUNDING].values)

    setattr(profile_object, STORM_VELOCITY_NAME_SHARPPY,
            numpy.array([u_motion_m_s01, v_motion_m_s01]))

    profile_object.right_esrh = sharppy_winds.helicity(
        profile_object, profile_object.ebotm, profile_object.etopm,
        stu=profile_object.srwind[0], stv=profile_object.srwind[1])
    profile_object.right_ehi = sharppy_params.ehi(
        profile_object, profile_object.mupcl, profile_object.ebotm,
        profile_object.etopm, stu=profile_object.srwind[0],
        stv=profile_object.srwind[1])
    profile_object.left_ehi = sharppy_params.ehi(
        profile_object, profile_object.mupcl, profile_object.ebotm,
        profile_object.etopm, stu=profile_object.srwind[2],
        stv=profile_object.srwind[3])

    profile_object.lhp = sharppy_params.lhp(profile_object)
    profile_object.x_totals = sharppy_params.c_totals(profile_object)
    profile_object.v_totals = sharppy_params.v_totals(profile_object)
    profile_object.sherb = sharppy_params.sherb(
        profile_object, effective=True, ebottom=profile_object.ebottom,
        etop=profile_object.etop, mupcl=profile_object.mupcl)
    profile_object.dcp = sharppy_params.dcp(profile_object)
    profile_object.sweat = sharppy_params.sweat(profile_object)
    profile_object.thetae_diff = sharppy_params.thetae_diff(
        profile_object)

    profile_object.right_scp = sharppy_params.scp(
        profile_object.mupcl.bplus, profile_object.right_esrh[0],
        profile_object.ebwspd * KT_TO_METRES_PER_SECOND)

    boundary_layer_top_mb = sharppy_params.pbl_top(profile_object)
    boundary_layer_top_m_asl = sharppy_interp.hght(
        profile_object, boundary_layer_top_mb)
    profile_object.pbl_depth = sharppy_interp.to_agl(
        profile_object, boundary_layer_top_m_asl)
    profile_object.edepthm = profile_object.etopm - profile_object.ebotm

    profile_object = _adjust_stats_for_storm_motion(
        profile_object=profile_object,
        u_motion_kt=METRES_PER_SECOND_TO_KT * u_motion_m_s01,
        v_motion_kt=METRES_PER_SECOND_TO_KT * v_motion_m_s01)
    profile_object = _fix_all_sharppy_wind_vectors(profile_object)
    return _extract_stats_from_sharppy_profile(
        profile_object=profile_object, metadata_table=metadata_table)


def _sentinels_to_nan(statistic_table_sharppy, metadata_table):
    """Replaces sentinel values with NaN.

    :param statistic_table_sharppy: pandas DataFrame created by
        `_compute_sounding_statistics`.
    :param metadata_table: pandas DataFrame created by
        `read_metadata_for_statistics`.
    :return: statistic_table_sharppy: Same as input, but sentinel values have
        been replaced with NaN.
    """

    statistic_names_sharppy = metadata_table[
        SHARPPY_STATISTIC_NAME_COLUMN].values.tolist()
    num_soundings = len(statistic_table_sharppy.index)

    for this_name in list(statistic_table_sharppy):
        if this_name not in statistic_names_sharppy:
            continue

        this_index = statistic_names_sharppy.index(this_name)
        this_vector_flag = metadata_table[IS_VECTOR_COLUMN].values[this_index]

        if this_vector_flag:
            for i in range(num_soundings):
                these_sentinel_flags = numpy.isclose(
                    statistic_table_sharppy[this_name].values[i],
                    SENTINEL_VALUE_FOR_SHARPPY, atol=SENTINEL_VALUE_TOLERANCE)
                if numpy.any(these_sentinel_flags):
                    statistic_table_sharppy[this_name].values[i] = numpy.full(
                        2, numpy.nan)

        else:
            these_sentinel_flags = numpy.isclose(
                statistic_table_sharppy[this_name].values,
                SENTINEL_VALUE_FOR_SHARPPY, atol=SENTINEL_VALUE_TOLERANCE)
            these_sentinel_indices = numpy.where(these_sentinel_flags)[0]
            statistic_table_sharppy[this_name].values[
                these_sentinel_indices] = numpy.nan

    return statistic_table_sharppy


def _convert_sounding_statistics(statistic_table_sharppy, metadata_table):
    """Converts sounding statistics from SHARPpy to GewitterGefahr format.

    :param statistic_table_sharppy: pandas DataFrame created by
        `_compute_sounding_statistics`.
    :param metadata_table: pandas DataFrame created by
        `read_metadata_for_statistics`.
    :return: sounding_statistic_table: pandas DataFrame with sounding statistics
        in GewitterGefahr format.  In other words, both column names and units
        are in GewitterGefahr format.  Also, vectors are split into one column
        per component.
    """

    sharppy_column_names = list(statistic_table_sharppy)
    sounding_statistic_table = None

    for this_orig_name in sharppy_column_names:
        match_flags = [s == this_orig_name for s in metadata_table[
            SHARPPY_STATISTIC_NAME_COLUMN].values]
        match_index = numpy.where(match_flags)[0][0]

        this_new_name = metadata_table[STATISTIC_NAME_COLUMN].values[
            match_index]
        this_conversion_factor = metadata_table[
            CONVERSION_FACTOR_COLUMN].values[match_index]
        this_vector_flag = metadata_table[IS_VECTOR_COLUMN].values[match_index]

        if this_vector_flag:
            this_column_as_table = statistic_table_sharppy[[this_orig_name]]
            this_column_as_table.rename(
                columns={this_orig_name: this_new_name}, inplace=True)
            argument_dict = _split_vector_column(
                input_table=this_column_as_table,
                conversion_factor=this_conversion_factor)
        else:
            argument_dict = {
                this_new_name: (
                    this_conversion_factor *
                    statistic_table_sharppy[this_orig_name].values)
            }

        if sounding_statistic_table is None:
            sounding_statistic_table = pandas.DataFrame.from_dict(argument_dict)
        else:
            sounding_statistic_table = sounding_statistic_table.assign(
                **argument_dict)

    temperatures_kelvins = temperature_conversions.fahrenheit_to_kelvins(
        sounding_statistic_table[CONVECTIVE_TEMPERATURE_NAME].values)
    argument_dict = {CONVECTIVE_TEMPERATURE_NAME: temperatures_kelvins}
    return sounding_statistic_table.assign(**argument_dict)


def _get_unique_storm_soundings(
        list_of_sharppy_sounding_tables, u_motions_m_s01, v_motions_m_s01):
    """Finds unique "storm soundings" (pairs of sounding and motion vector).

    N = number of storm objects
    U = number of unique "storm soundings"

    :param list_of_sharppy_sounding_tables: length-N list of pandas DataFrames,
        produced by `_create_sharppy_sounding_tables`.
    :param u_motions_m_s01: length-N numpy array with eastward components of
        storm motion (metres per second).
    :param v_motions_m_s01: length-N numpy array with northward components of
        storm motion (metres per second).
    :return: unique_indices: length-U numpy array with indices of unique storm
        soundings.  These are indices into the input arrays.
    :return: orig_to_unique_indices: length-N numpy array.  If
        orig_to_unique_indices[j] = i, the [j]th original storm sounding is an
        instance of the [i]th unique storm sounding.
    """

    num_storm_objects = len(list_of_sharppy_sounding_tables)
    storm_sounding_strings = [''] * num_storm_objects

    for i in range(num_storm_objects):
        if list_of_sharppy_sounding_tables[i] is None:
            storm_sounding_strings[i] = 'None'
        else:
            storm_sounding_strings[i] = '{0:.2f}_{1:.2f}'.format(
                u_motions_m_s01[i], v_motions_m_s01[i])

            for j in range(3):
                for this_column in SHARPPY_SOUNDING_COLUMNS:
                    storm_sounding_strings[i] += '_{0:.2f}'.format(
                        list_of_sharppy_sounding_tables[
                            i][this_column].values[j])

    _, unique_indices, orig_to_unique_indices = numpy.unique(
        numpy.array(storm_sounding_strings), return_index=True,
        return_inverse=True)
    return unique_indices, orig_to_unique_indices


def check_statistic_name(statistic_name, metadata_table):
    """Ensures that statistic name is valid.

    :param statistic_name: Statistic name in GewitterGefahr format.
    :param metadata_table: pandas DataFrame created by
        `read_metadata_for_statistics`.
    :raises: ValueError: if `statistic_name` is invalid.
    """

    error_checking.assert_is_string(statistic_name)
    error_string = '"{0:s}" is not a valid statistic.'.format(statistic_name)

    all_basic_stat_names = metadata_table[STATISTIC_NAME_COLUMN].values
    if statistic_name in all_basic_stat_names:
        return

    statistic_name_parts = statistic_name.split('_')
    if len(statistic_name_parts) < 2:
        raise ValueError(error_string)

    if statistic_name_parts[-1] not in VECTOR_SUFFIXES:
        raise ValueError(error_string)

    basic_statistic_name = '_'.join(statistic_name_parts[:-1])
    if basic_statistic_name not in all_basic_stat_names:
        raise ValueError(error_string)


def remove_vector_suffix_from_stat_name(statistic_name, metadata_table):
    """Removes vector suffix from statistic name.

    :param statistic_name: Statistic name (in GewitterGefahr format).
    :param metadata_table: pandas DataFrame created by
        `read_metadata_for_statistics`.
    :return: basic_statistic_name: Name without vector suffix.  If
        `statistic_name` has no vector suffix, this will be equal to
        `statistic_name`.
    :return: vector_suffix: Vector suffix.  If `statistic_name` has no vector
        suffix, this will be None.
    """

    check_statistic_name(
        statistic_name=statistic_name, metadata_table=metadata_table)

    all_basic_stat_names = metadata_table[STATISTIC_NAME_COLUMN].values
    if statistic_name in all_basic_stat_names:
        return statistic_name, None

    name_parts = statistic_name.split('_')
    return '_'.join(name_parts[:-1]), name_parts[-1]


def add_vector_suffix_to_stat_name(basic_statistic_name, vector_suffix):
    """Adds vector suffix to statistic name.

    :param basic_statistic_name: Statistic name without vector suffix (in
        GewitterGefahr format).
    :param vector_suffix: Vector suffix (must be in list `VECTOR_SUFFIXES`)
    :return: statistic_name: Statistic name with vector suffix.
    :raises: ValueError: if `vector_suffix not in VECTOR_SUFFIXES`.
    """

    error_checking.assert_is_string(vector_suffix)
    error_checking.assert_is_string(basic_statistic_name)

    if vector_suffix not in VECTOR_SUFFIXES:
        error_string = (
            '\n\n{0:s}\nValid vector suffixes (listed above) do not include '
            '"{1:s}".'
        ).format(str(VECTOR_SUFFIXES), vector_suffix)
        raise ValueError(error_string)

    return '{0:s}_{1:s}'.format(basic_statistic_name, vector_suffix)


def check_sounding_statistic_table(
        sounding_statistic_table, require_storm_objects=True):
    """Ensures that table contains sounding statistics.

    :param sounding_statistic_table: pandas DataFrame.
    :param require_storm_objects: Boolean flag.  If True, the table must contain
        columns "storm_id" and "unix_time_sec".
    :return: statistic_column_names: 1-D list with names of columns containing
        sounding statistics.  If there are no such columns, this is None.
    :raises: ValueError: `sounding_statistic_table` contains no columns with
        sounding statistics.
    """

    error_checking.assert_is_boolean(require_storm_objects)

    statistic_column_names = get_statistic_columns(sounding_statistic_table)
    if statistic_column_names is None:
        raise ValueError('Table contains no columns with sounding statistics.')

    if require_storm_objects:
        error_checking.assert_columns_in_dataframe(
            sounding_statistic_table, STORM_COLUMNS_TO_KEEP)

    return statistic_column_names


def get_statistic_columns(sounding_statistic_table):
    """Returns names of columns with sounding statistics.

    :param sounding_statistic_table: pandas DataFrame.
    :return: statistic_column_names: 1-D list with names of columns containing
        sounding statistics.  If there are no such columns, this is None.
    """

    metadata_table = read_metadata_for_statistics()
    column_names = list(sounding_statistic_table)
    statistic_column_names = []

    for this_column_name in column_names:
        try:
            check_statistic_name(
                statistic_name=this_column_name, metadata_table=metadata_table)
            statistic_column_names.append(this_column_name)
        except ValueError:
            pass

    if not len(statistic_column_names):
        return None

    return statistic_column_names


def read_metadata_for_statistics():
    """Reads metadata for sounding statistics.

    :return: metadata_table: pandas DataFrame with the following columns.  Each
        row is one sounding statistic.
    metadata_table.statistic_name: Name of sounding statistic (GewitterGefahr
        format).
    metadata_table.statistic_name_sharppy: Name of sounding statistic (SHARPpy
        format).
    metadata_table.conversion_factor: Multiplier from SHARPpy units to
        GewitterGefahr units.
    metadata_table.is_vector: Boolean flag.  If True, the statistic is a 2-D
        vector.  If False, it is a scalar.
    metadata_table.in_mupcl_object: Boolean flag.  If True, the statistic is an
        attribute of `profile_object.mupcl`, where `profile_object` is an
        instance of `sharppy.sharptab.Profile`.  If False, the statistic is just
        an attribute of `profile_object`.
    metadata_table.min_values_for_normalization: 1-D numpy array with minimum
        values for normalization (to be used in
        `deep_learning_utils.normalize_sounding_statistics`).  For scalar
        statistics, the array has length 1.  For vector statistics, the array
        has length 2, with minimum values for the x- and y-components
        respectively.
    metadata_table.max_values_for_normalization: Same as above, except for max
        values.
    """

    error_checking.assert_file_exists(METAFILE_NAME)
    pickle_file_handle = open(METAFILE_NAME, 'rb')
    metadata_table = pickle.load(pickle_file_handle)
    pickle_file_handle.close()

    error_checking.assert_columns_in_dataframe(metadata_table, METADATA_COLUMNS)
    return metadata_table


def get_sounding_stats_for_storm_objects(
        storm_object_table, top_grib_directory_name,
        lead_times_sec=DEFAULT_LEAD_TIMES_SEC, all_ruc_grids=True,
        model_name=None, grid_id=None,
        wgrib_exe_name=grib_io.WGRIB_EXE_NAME_DEFAULT,
        wgrib2_exe_name=grib_io.WGRIB2_EXE_NAME_DEFAULT,
        raise_error_if_missing=False):
    """Computes sounding statistics for each storm object.

    N = number of storm objects
    T = number of lead times
    K = number of sounding stats (after decomposing vectors into scalars)

    :param storm_object_table: pandas DataFrame with columns documented in
        `storm_tracking_io.write_processed_file`.
    :param top_grib_directory_name: Name of top-level directory with grib files
        from the NWP model.
    :param lead_times_sec: length-T numpy array of lead times.  For each lead
        time t, each storm object will be extrapolated t seconds into the
        future, along its estimated motion vector at the initial time.
    :param all_ruc_grids: Boolean flag.  If True, will use the method
        `_interp_soundings_from_ruc` to interpolate soundings to storm objects.
        If False, will use `_interp_soundings_from_nwp`.
    :param model_name: [used only if `all_ruc_grids = False`]
        Soundings will be interpolated from this NWP model.
    :param grid_id: [used only if `all_ruc_grids = False`]
        String ID for model grid.
    :param wgrib_exe_name: Path to wgrib executable.
    :param wgrib2_exe_name: Path to wgrib2 executable.
    :param raise_error_if_missing: Boolean flag.  If a grib file with NWP data
        is missing and `raise_error_if_missing` = True, this method will error
        out.  If a grib file is missing and `raise_error_if_missing` = False,
        this method will skip the corresponding time step.
    :return: sounding_statistic_table: pandas DataFrame with N*T rows (one for
        each storm object and lead time) and 3 + K columns.  The first 3 columns
        are listed below.  The other K column names can be found by running the
        table through `get_statistic_columns`.
    sounding_statistic_table.storm_id: String ID for storm cell.
    sounding_statistic_table.unix_time_sec: Valid time for storm object.
    sounding_statistic_table.lead_time_seconds: Lead time for sounding
        statistics.  Valid time of sounding statistics is `unix_time_sec +
        lead_time_seconds`.
    """

    error_checking.assert_is_integer_numpy_array(lead_times_sec)
    error_checking.assert_is_numpy_array(lead_times_sec, num_dimensions=1)
    error_checking.assert_is_geq_numpy_array(lead_times_sec, 0)
    error_checking.assert_is_boolean(all_ruc_grids)

    query_point_table = _create_query_points(
        storm_object_table=storm_object_table, lead_times_sec=lead_times_sec)

    column_dict_old_to_new = {
        tracking_utils.CENTROID_LAT_COLUMN: interp.QUERY_LAT_COLUMN,
        tracking_utils.CENTROID_LNG_COLUMN: interp.QUERY_LNG_COLUMN,
        tracking_utils.TIME_COLUMN: interp.QUERY_TIME_COLUMN}
    query_point_table.rename(columns=column_dict_old_to_new, inplace=True)

    if all_ruc_grids:
        model_name = nwp_model_utils.RUC_MODEL_NAME
        interp_table = _interp_soundings_from_ruc(
            query_point_table=query_point_table,
            top_grib_directory_name=top_grib_directory_name,
            wgrib_exe_name=wgrib_exe_name, wgrib2_exe_name=wgrib2_exe_name,
            raise_error_if_missing=raise_error_if_missing)
    else:
        interp_table = _interp_soundings_from_nwp(
            query_point_table=query_point_table, model_name=model_name,
            top_grib_directory_name=top_grib_directory_name, grid_id=grid_id,
            wgrib_exe_name=wgrib_exe_name, wgrib2_exe_name=wgrib2_exe_name,
            raise_error_if_missing=raise_error_if_missing)

    list_of_sharppy_sounding_tables = _create_sharppy_sounding_tables(
        interp_table=interp_table, model_name=model_name)
    unique_indices, orig_to_unique_indices = _get_unique_storm_soundings(
        list_of_sharppy_sounding_tables=list_of_sharppy_sounding_tables,
        u_motions_m_s01=query_point_table[
            tracking_utils.EAST_VELOCITY_COLUMN].values,
        v_motions_m_s01=query_point_table[
            tracking_utils.NORTH_VELOCITY_COLUMN].values)

    num_soundings = len(list_of_sharppy_sounding_tables)
    num_unique_soundings = len(unique_indices)
    print 'Number of unique soundings = {0:d}/{1:d}\n'.format(
        num_unique_soundings, num_soundings)

    list_of_sharppy_statistic_tables = [None] * num_soundings
    metadata_table = read_metadata_for_statistics()

    for i in range(num_unique_soundings):
        print (
            'Computing statistics for {0:d}th of {1:d} unique soundings...'
        ).format(i + 1, num_soundings)

        j = unique_indices[i]
        if list_of_sharppy_sounding_tables[j] is None:
            this_statistic_table = _get_dummy_sharppy_statistic_table(
                u_motion_m_s01=query_point_table[
                    tracking_utils.EAST_VELOCITY_COLUMN].values[j],
                v_motion_m_s01=query_point_table[
                    tracking_utils.NORTH_VELOCITY_COLUMN].values[j],
                metadata_table=metadata_table)

        else:
            this_statistic_table = _compute_sounding_statistics(
                sounding_table_sharppy=list_of_sharppy_sounding_tables[j],
                u_motion_m_s01=query_point_table[
                    tracking_utils.EAST_VELOCITY_COLUMN].values[j],
                v_motion_m_s01=query_point_table[
                    tracking_utils.NORTH_VELOCITY_COLUMN].values[j],
                metadata_table=metadata_table)

        these_orig_indices = numpy.where(orig_to_unique_indices == i)[0]
        for this_orig_index in these_orig_indices:
            list_of_sharppy_statistic_tables[
                this_orig_index] = this_statistic_table

    for i in range(num_soundings):
        if i == 0:
            continue

        list_of_sharppy_statistic_tables[i], _ = (
            list_of_sharppy_statistic_tables[i].align(
                list_of_sharppy_statistic_tables[0], axis=1))

    statistic_table_sharppy = pandas.concat(
        list_of_sharppy_statistic_tables, axis=0, ignore_index=True)
    statistic_table_sharppy = _sentinels_to_nan(
        statistic_table_sharppy=statistic_table_sharppy,
        metadata_table=metadata_table)
    sounding_statistic_table = _convert_sounding_statistics(
        statistic_table_sharppy=statistic_table_sharppy,
        metadata_table=metadata_table)
    sounding_statistic_table = pandas.concat(
        [storm_object_table[STORM_COLUMNS_TO_KEEP],
         sounding_statistic_table], axis=1)

    argument_dict = {
        LEAD_TIME_COLUMN: query_point_table[LEAD_TIME_COLUMN].values}
    return sounding_statistic_table.assign(**argument_dict)


def find_sounding_statistic_file(
        top_directory_name, init_time_unix_sec, lead_time_sec, spc_date_string,
        raise_error_if_missing=True):
    """Locates file with sounding statistics for storm objects.

    This file should be written by `write_sounding_statistics`.

    :param top_directory_name: Name of top-level directory with files containing
        sounding stats.
    :param init_time_unix_sec: Initial time.
    :param lead_time_sec: Lead time for sounding stats.
    :param spc_date_string: SPC date (format "yyyymmdd").
    :param raise_error_if_missing: Boolean flag.  If `raise_error_if_missing`
        and file is missing, this method will error out.
    :return: sounding_statistic_file_name: Path to file with sounding stats.  If
        file is missing and `raise_error_if_missing = False`, this will be the
        *expected* path.
    :raises: ValueError: if `raise_error_if_missing` and file is not found.
    """

    error_checking.assert_is_string(top_directory_name)
    error_checking.assert_is_integer(lead_time_sec)
    time_conversion.spc_date_string_to_unix_sec(spc_date_string)
    error_checking.assert_is_boolean(raise_error_if_missing)

    init_time_string = time_conversion.unix_sec_to_string(
        init_time_unix_sec, TIME_FORMAT_IN_FILE_NAMES)

    sounding_statistic_file_name = (
        '{0:s}/{1:s}/{2:s}/sounding_statistics_{3:s}_lead-time-{4:04d}sec.nc'
    ).format(top_directory_name, spc_date_string[:4], spc_date_string,
             init_time_string, lead_time_sec)

    if raise_error_if_missing and not os.path.isfile(
            sounding_statistic_file_name):
        error_string = (
            'Cannot find file with sounding stats for storm objects.  Expected '
            'at: {0:s}').format(sounding_statistic_file_name)
        raise ValueError(error_string)

    return sounding_statistic_file_name


def write_sounding_statistics(
        top_directory_name, init_time_unix_sec, lead_time_sec, spc_date_string,
        sounding_statistic_table, verbose=False):
    """Writes sounding statistics for storm objects to NetCDF file.

    :param top_directory_name: See documentation for
        `find_sounding_statistic_file`.
    :param init_time_unix_sec: Same.
    :param lead_time_sec: Same.
    :param spc_date_string: Same.
    :param sounding_statistic_table: See output doc for
        `_convert_sounding_statistics`.
    :param verbose: Boolean flag.  If True, will print log message.
    """

    netcdf_file_name = find_sounding_statistic_file(
        top_directory_name=top_directory_name,
        init_time_unix_sec=init_time_unix_sec, lead_time_sec=lead_time_sec,
        spc_date_string=spc_date_string, raise_error_if_missing=False)

    error_checking.assert_is_boolean(verbose)
    if verbose:
        print 'Writing sounding statistics to: "{0:s}"...'.format(
            netcdf_file_name)

    statistic_names = check_sounding_statistic_table(
        sounding_statistic_table, require_storm_objects=True)

    error_checking.assert_equals_numpy_array(
        sounding_statistic_table[tracking_utils.TIME_COLUMN].values,
        init_time_unix_sec)
    error_checking.assert_equals_numpy_array(
        sounding_statistic_table[LEAD_TIME_COLUMN].values, lead_time_sec)

    file_system_utils.mkdir_recursive_if_necessary(file_name=netcdf_file_name)
    netcdf_dataset = netCDF4.Dataset(
        netcdf_file_name, 'w', format='NETCDF3_64BIT_OFFSET')

    netcdf_dataset.setncattr(INIT_TIME_KEY, init_time_unix_sec)
    netcdf_dataset.setncattr(LEAD_TIME_KEY, lead_time_sec)
    netcdf_dataset.setncattr(SPC_DATE_KEY, spc_date_string)

    num_storm_objects = len(sounding_statistic_table.index)
    num_statistics = len(statistic_names)
    num_statistic_chars = 0
    for j in range(num_statistics):
        num_statistic_chars = max(
            [num_statistic_chars, len(statistic_names[j])])

    storm_ids = sounding_statistic_table[tracking_utils.STORM_ID_COLUMN].values
    num_storm_id_chars = 0
    for i in range(num_storm_objects):
        num_storm_id_chars = max([num_storm_id_chars, len(storm_ids[i])])

    netcdf_dataset.createDimension(STORM_DIMENSION_KEY, num_storm_objects)
    netcdf_dataset.createDimension(STATISTIC_DIMENSION_KEY, num_statistics)
    netcdf_dataset.createDimension(
        STATISTIC_CHAR_DIMENSION_KEY, num_statistic_chars)
    netcdf_dataset.createDimension(
        STORM_ID_CHAR_DIMENSION_KEY, num_storm_id_chars)

    netcdf_dataset.createVariable(
        STATISTIC_NAMES_KEY, datatype='S1',
        dimensions=(STATISTIC_DIMENSION_KEY, STATISTIC_CHAR_DIMENSION_KEY))

    string_type = 'S{0:d}'.format(num_statistic_chars)
    statistic_names_as_char_array = netCDF4.stringtochar(numpy.array(
        statistic_names, dtype=string_type))
    netcdf_dataset.variables[STATISTIC_NAMES_KEY][:] = numpy.array(
        statistic_names_as_char_array)

    netcdf_dataset.createVariable(
        STORM_IDS_KEY, datatype='S1',
        dimensions=(STORM_DIMENSION_KEY, STORM_ID_CHAR_DIMENSION_KEY))

    string_type = 'S{0:d}'.format(num_storm_id_chars)
    storm_ids_as_char_array = netCDF4.stringtochar(numpy.array(
        storm_ids, dtype=string_type))
    netcdf_dataset.variables[STORM_IDS_KEY][:] = numpy.array(
        storm_ids_as_char_array)

    chunk_size_tuple = (1, num_statistics)
    netcdf_dataset.createVariable(
        STATISTIC_MATRIX_KEY, datatype=numpy.float32,
        dimensions=(STORM_DIMENSION_KEY, STATISTIC_DIMENSION_KEY),
        chunksizes=chunk_size_tuple)

    sounding_statistic_matrix = sounding_statistic_table.as_matrix(
        columns=statistic_names)
    netcdf_dataset.variables[
        STATISTIC_MATRIX_KEY][:] = sounding_statistic_matrix
    netcdf_dataset.close()


def write_sounding_statistics_many_times(
        top_directory_name, sounding_statistic_table):
    """Writes sounding statistics for storm objects to multiple NetCDF files.

    Specifically, this method writes one NetCDF file for each initial time and
    lead time.

    :param top_directory_name: Name of top-level directory with files containing
        sounding stats.
    :param sounding_statistic_table: pandas DataFrame created by
        `get_sounding_stats_for_storm_objects`.
    """

    init_times_unix_sec = numpy.unique(
        sounding_statistic_table[tracking_utils.TIME_COLUMN].values)
    lead_times_sec = numpy.unique(
        sounding_statistic_table[LEAD_TIME_COLUMN].values)

    num_init_times = len(init_times_unix_sec)
    num_lead_times = len(lead_times_sec)

    for i in range(num_init_times):
        for j in range(num_lead_times):
            these_indices = numpy.where(numpy.logical_and(
                sounding_statistic_table[tracking_utils.TIME_COLUMN].values ==
                init_times_unix_sec[i],
                sounding_statistic_table[LEAD_TIME_COLUMN].values ==
                lead_times_sec[j]))[0]

            write_sounding_statistics(
                top_directory_name=top_directory_name,
                init_time_unix_sec=init_times_unix_sec[i],
                lead_time_sec=lead_times_sec[j],
                spc_date_string=time_conversion.time_to_spc_date_string(
                    init_times_unix_sec[i]),
                sounding_statistic_table=sounding_statistic_table.iloc[
                    these_indices],
                verbose=True)


def read_sounding_statistics(
        netcdf_file_name, storm_ids_to_keep=None, statistic_names_to_keep=None):
    """Reads sounding statistics for storm objects from NetCDF file.

    This file should be written by `write_sounding_statistics`.

    N = number of storm objects
    V = number of sounding statistics

    :param netcdf_file_name: Path to input file.
    :param storm_ids_to_keep: 1-D list of storm IDs to keep (strings).  If None,
        will keep all storm objects.
    :param statistic_names_to_keep: 1-D list with names of statistics to keep.
        If None, will keep all statistics.

    :return: sounding_statistic_dict: Dictionary with the following keys.
    sounding_statistic_dict['init_time_unix_sec']: Initial time.
    sounding_statistic_dict['lead_time_sec']: Lead time.
    sounding_statistic_dict['spc_date_string']: SPC date (format "yyyymmdd").
    sounding_statistic_dict['storm_ids']: length-N list of storm IDs (strings).
    sounding_statistic_dict['statistic_names']: length-V list of statistic names
        (strings).
    sounding_statistic_dict['sounding_statistic_matrix']: N-by-V numpy array of
        sounding stats.
    """

    netcdf_dataset = netcdf_io.open_netcdf(
        netcdf_file_name=netcdf_file_name, raise_error_if_fails=True)

    init_time_unix_sec = getattr(netcdf_dataset, INIT_TIME_KEY)
    lead_time_sec = getattr(netcdf_dataset, LEAD_TIME_KEY)
    spc_date_string = str(getattr(netcdf_dataset, SPC_DATE_KEY))
    storm_ids = netCDF4.chartostring(netcdf_dataset.variables[STORM_IDS_KEY][:])
    statistic_names = netCDF4.chartostring(
        netcdf_dataset.variables[STATISTIC_NAMES_KEY][:])

    storm_ids = [str(s) for s in storm_ids]
    statistic_names = [str(s) for s in statistic_names]

    if storm_ids_to_keep is None:
        sounding_statistic_matrix = numpy.array(
            netcdf_dataset.variables[STATISTIC_MATRIX_KEY][:])
    else:
        error_checking.assert_is_string_list(storm_ids_to_keep)
        error_checking.assert_is_numpy_array(
            numpy.array(storm_ids_to_keep), num_dimensions=1)

        storm_indices_to_keep = numpy.array(
            [storm_ids.index(s) for s in storm_ids_to_keep], dtype=int)
        storm_ids = copy.deepcopy(storm_ids_to_keep)
        sounding_statistic_matrix = numpy.array(
            netcdf_dataset.variables[STATISTIC_MATRIX_KEY][
                storm_indices_to_keep, ...])

    netcdf_dataset.close()

    if statistic_names_to_keep is not None:
        error_checking.assert_is_string_list(statistic_names_to_keep)
        error_checking.assert_is_numpy_array(
            numpy.array(statistic_names_to_keep), num_dimensions=1)

        stat_indices_to_keep = numpy.array(
            [statistic_names.index(s) for s in statistic_names_to_keep],
            dtype=int)
        statistic_names = copy.deepcopy(statistic_names_to_keep)
        sounding_statistic_matrix = sounding_statistic_matrix[
            ..., stat_indices_to_keep]

    return {
        INIT_TIME_KEY: init_time_unix_sec, LEAD_TIME_KEY: lead_time_sec,
        SPC_DATE_KEY: spc_date_string, STORM_IDS_KEY: storm_ids,
        STATISTIC_NAMES_KEY: statistic_names,
        STATISTIC_MATRIX_KEY: sounding_statistic_matrix
    }
