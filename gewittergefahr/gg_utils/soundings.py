"""Methods for computing sounding statistics."""

import os.path
import pickle
import numpy
import pandas
from sharppy.sharptab import params as sharppy_params
from sharppy.sharptab import winds as sharppy_winds
from sharppy.sharptab import interp as sharppy_interp
from sharppy.sharptab import profile as sharppy_profile
from sharppy.sharptab import utils as sharppy_utils
from gewittergefahr.gg_io import grib_io
from gewittergefahr.gg_io import storm_tracking_io as tracking_io
from gewittergefahr.gg_utils import moisture_conversions
from gewittergefahr.gg_utils import temperature_conversions
from gewittergefahr.gg_utils import interp
from gewittergefahr.gg_utils import nwp_model_utils
from gewittergefahr.gg_utils import geodetic_utils
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.gg_utils import error_checking

LEAD_TIME_COLUMN = 'lead_time_seconds'

TEMPORAL_INTERP_METHOD = interp.PREVIOUS_INTERP_METHOD
SPATIAL_INTERP_METHOD = interp.NEAREST_INTERP_METHOD
STORM_COLUMNS_TO_KEEP = [tracking_io.STORM_ID_COLUMN, tracking_io.TIME_COLUMN]

SENTINEL_VALUE_FOR_SHARPPY = -9999.
REDUNDANT_PRESSURE_TOLERANCE_MB = 1e-3
REDUNDANT_HEIGHT_TOLERANCE_METRES = 1e-3
MIN_PRESSURE_LEVELS_IN_SOUNDING = 15

PERCENT_TO_UNITLESS = 0.01
PASCALS_TO_MB = 0.01
MB_TO_PASCALS = 100
METRES_PER_SECOND_TO_KT = 3.6 / 1.852
KT_TO_METRES_PER_SECOND = 1.852 / 3.6

PRESSURE_COLUMN_FOR_SHARPPY_INPUT = 'pressure_mb'
HEIGHT_COLUMN_FOR_SHARPPY_INPUT = 'geopotential_height_metres'
TEMPERATURE_COLUMN_FOR_SHARPPY_INPUT = 'temperature_deg_c'
DEWPOINT_COLUMN_FOR_SHARPPY_INPUT = 'dewpoint_deg_c'
U_WIND_COLUMN_FOR_SHARPPY_INPUT = 'u_wind_kt'
V_WIND_COLUMN_FOR_SHARPPY_INPUT = 'v_wind_kt'
IS_SURFACE_COLUMN_FOR_SHARPPY_INPUT = 'is_surface'

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

# The following constants are used to read metadata for sounding stats.
STAT_NAME_COLUMN_FOR_METADATA = 'statistic_name'
SHARPPY_NAME_COLUMN_FOR_METADATA = 'statistic_name_sharppy'
CONVERSION_FACTOR_COLUMN_FOR_METADATA = 'conversion_factor'
IS_VECTOR_COLUMN_FOR_METADATA = 'is_vector'
IN_MUPCL_OBJECT_COLUMN_FOR_METADATA = 'in_mupcl_object'

METAFILE_NAME = os.path.join(
    os.path.dirname(__file__), 'metadata_for_sounding_stats.csv')
METADATA_COLUMNS = [
    STAT_NAME_COLUMN_FOR_METADATA, SHARPPY_NAME_COLUMN_FOR_METADATA,
    CONVERSION_FACTOR_COLUMN_FOR_METADATA, IS_VECTOR_COLUMN_FOR_METADATA,
    IN_MUPCL_OBJECT_COLUMN_FOR_METADATA]
COLUMN_TYPE_DICT_FOR_METADATA = {
    STAT_NAME_COLUMN_FOR_METADATA: str,
    SHARPPY_NAME_COLUMN_FOR_METADATA: str,
    CONVERSION_FACTOR_COLUMN_FOR_METADATA: numpy.float64,
    IS_VECTOR_COLUMN_FOR_METADATA: bool,
    IN_MUPCL_OBJECT_COLUMN_FOR_METADATA: bool}


def _remove_sounding_rows_with_nan(sounding_table):
    """Removes any row with a NaN from sounding table.

    :param sounding_table: pandas DataFrame (may contain NaN's).
    :return: sounding_table: Same as input, except that rows with NaN have been
        removed.  If there are no rows without NaN, returns None.
    """

    sounding_table = sounding_table.loc[sounding_table.notnull().all(axis=1)]
    if len(sounding_table.index) < MIN_PRESSURE_LEVELS_IN_SOUNDING:
        return None
    return sounding_table


def _get_empty_sharppy_stat_table(eastward_motion_m_s01,
                                  northward_motion_m_s01):
    """Creates pandas DataFrame where all SHARPpy statistics are NaN.

    :param eastward_motion_m_s01: u-component of storm motion (metres per
        second).
    :param northward_motion_m_s01: v-component of storm motion (metres per
        second).
    :return: empty_sharppy_stat_table: pandas DataFrame, where column names are
        SHARPpy names for sounding statistics and all values (except storm
        motion) are NaN.
    """

    metadata_table = read_metadata_for_sounding_stats()
    vector_stat_flags = numpy.array(
        metadata_table[IS_VECTOR_COLUMN_FOR_METADATA].values)
    vector_stat_indices = numpy.where(vector_stat_flags)[0]
    scalar_stat_indices = numpy.where(numpy.invert(vector_stat_flags))[0]

    empty_sharppy_stat_dict = {}
    for this_index in scalar_stat_indices:
        this_sharppy_stat_name = metadata_table[
            SHARPPY_NAME_COLUMN_FOR_METADATA].values[this_index]
        empty_sharppy_stat_dict.update(
            {this_sharppy_stat_name: numpy.full(1, numpy.nan)})

    empty_sharppy_stat_table = pandas.DataFrame.from_dict(
        empty_sharppy_stat_dict)

    first_statistic_name = list(empty_sharppy_stat_table)[0]
    nested_array = empty_sharppy_stat_table[[
        first_statistic_name, first_statistic_name]].values.tolist()

    for this_index in vector_stat_indices:
        this_sharppy_stat_name = metadata_table[
            SHARPPY_NAME_COLUMN_FOR_METADATA].values[this_index]
        empty_sharppy_stat_table = empty_sharppy_stat_table.assign(
            **{this_sharppy_stat_name: nested_array})

        if this_sharppy_stat_name == STORM_VELOCITY_NAME_SHARPPY:
            empty_sharppy_stat_table[this_sharppy_stat_name].values[0] = (
                numpy.array([eastward_motion_m_s01, northward_motion_m_s01]))
        else:
            empty_sharppy_stat_table[this_sharppy_stat_name].values[
                0] = numpy.full(2, numpy.nan)

    return empty_sharppy_stat_table


def _column_name_to_sounding_stat(column_name, valid_statistic_names):
    """Determines sounding statistic from column name.

    This column should be in a pandas DataFrame created by
    convert_sounding_indices_from_sharppy.

    :param column_name: Name of column.
    :param valid_statistic_names: 1-D list with names of valid sounding
        statistics (in GewitterGefahr format, not SHARPpy format).
    :return: statistic_name: Name of sounding statistic.
    """

    if column_name in valid_statistic_names:
        return column_name

    column_name_parts = column_name.split('_')
    if len(column_name_parts) < 2:
        return None

    if column_name_parts[-1] not in VECTOR_SUFFIXES:
        return None

    statistic_name = '_'.join(column_name_parts[:-1])
    if statistic_name in valid_statistic_names:
        return statistic_name

    return None


def _get_nwp_fields_in_sounding(model_name, minimum_pressure_mb=0.,
                                return_dict=False):
    """Returns list of NWP fields needed to create sounding.

    Each field is one variable at one pressure level (e.g., 500-mb height,
    600-mb specific humidity, 700-mb temperature, etc.).

    N = number of sounding fields
    P = number of pressure levels

    :param model_name: Name of model.
    :param minimum_pressure_mb: Minimum pressure (millibars).  All fields from a
        lower pressure will be ignored.
    :param return_dict: Boolean flag.  If True, this method returns
        `sounding_table_to_fields_dict` and `pressure_levels_mb`.  If False,
        returns `sounding_field_names` and `sounding_field_names_grib1`.
    :return: sounding_field_names: length-N list of field names in
        GewitterGefahr format.
    :return: sounding_field_names_grib1: length-N list of field names in grib1
        format.
    :return: sounding_table_to_fields_dict: Dictionary with the following keys.
    sounding_table_to_fields_dict['geopotential_height_metres']: length-P list
        of height fields in `sounding_field_names`.
    sounding_table_to_fields_dict['temperature_kelvins']: length-P list of
        temperature fields in `sounding_field_names`.
    sounding_table_to_fields_dict['humidity']: length-P list of humidity fields
        in `sounding_field_names`.
    sounding_table_to_fields_dict['u_wind_m_s01']: length-P list of u-wind
        fields in `sounding_field_names`.
    sounding_table_to_fields_dict['v_wind_m_s01']: length-P list of v-wind
        fields in `sounding_field_names`.

    :return: pressure_levels_mb: length-P numpy array of pressure levels
        (millibars).
    """

    nwp_model_utils.check_model_name(model_name)
    error_checking.assert_is_geq(minimum_pressure_mb, 0)
    error_checking.assert_is_boolean(return_dict)

    pressure_levels_mb = nwp_model_utils.get_pressure_levels(
        model_name, grid_id=nwp_model_utils.ID_FOR_130GRID)
    pressure_levels_mb = pressure_levels_mb[
        pressure_levels_mb >= minimum_pressure_mb]
    num_pressure_levels = len(pressure_levels_mb)

    sounding_table_columns, sounding_table_columns_grib1 = (
        nwp_model_utils.get_columns_in_sounding_table(model_name))
    num_sounding_table_columns = len(sounding_table_columns)

    if return_dict:
        sounding_table_to_fields_dict = {}
        for j in range(num_sounding_table_columns):
            sounding_table_to_fields_dict.update(
                {sounding_table_columns[j]: []})
    else:
        sounding_field_names = []
        sounding_field_names_grib1 = []

    for j in range(num_sounding_table_columns):
        for k in range(num_pressure_levels):
            this_field_name = '{0:s}_{1:d}mb'.format(
                sounding_table_columns[j], int(pressure_levels_mb[k]))

            if return_dict:
                sounding_table_to_fields_dict[sounding_table_columns[j]].append(
                    this_field_name)
            else:
                sounding_field_names.append(this_field_name)
                sounding_field_names_grib1.append('{0:s}:{1:d} mb'.format(
                    sounding_table_columns_grib1[j],
                    int(pressure_levels_mb[k])))

        if (sounding_table_columns[j] ==
                nwp_model_utils.TEMPERATURE_COLUMN_FOR_SOUNDING_TABLES):

            this_field_name, this_field_name_grib1 = (
                nwp_model_utils.get_lowest_temperature_name(model_name))

        elif sounding_table_columns[j] in [
                nwp_model_utils.RH_COLUMN_FOR_SOUNDING_TABLES,
                nwp_model_utils.SPFH_COLUMN_FOR_SOUNDING_TABLES]:

            this_field_name, this_field_name_grib1 = (
                nwp_model_utils.get_lowest_humidity_name(model_name))

        elif (sounding_table_columns[j] ==
              nwp_model_utils.HEIGHT_COLUMN_FOR_SOUNDING_TABLES):

            this_field_name, this_field_name_grib1 = (
                nwp_model_utils.get_lowest_height_name(model_name))

        elif (sounding_table_columns[j] ==
              nwp_model_utils.U_WIND_COLUMN_FOR_SOUNDING_TABLES):

            this_field_name, this_field_name_grib1 = (
                nwp_model_utils.get_lowest_u_wind_name(model_name))

        elif (sounding_table_columns[j] ==
              nwp_model_utils.V_WIND_COLUMN_FOR_SOUNDING_TABLES):

            this_field_name, this_field_name_grib1 = (
                nwp_model_utils.get_lowest_v_wind_name(model_name))

        if return_dict:
            sounding_table_to_fields_dict[sounding_table_columns[j]].append(
                this_field_name)
        else:
            sounding_field_names.append(this_field_name)
            sounding_field_names_grib1.append(this_field_name_grib1)

    if return_dict:
        return sounding_table_to_fields_dict, pressure_levels_mb

    this_field_name, this_field_name_grib1 = (
        nwp_model_utils.get_lowest_pressure_name(model_name))
    sounding_field_names.append(this_field_name)
    sounding_field_names_grib1.append(this_field_name_grib1)

    return sounding_field_names, sounding_field_names_grib1


def _remove_subsurface_sounding_data(sounding_table, delete_rows=None):
    """Removes sounding data from levels below the surface.

    :param sounding_table: See documentation for
        get_sounding_stats_from_sharppy.
    :param delete_rows: Boolean flag.  If True, will delete all subsurface rows
        except 1000 mb.  If False, will set values at subsurface rows (except
        height and pressure) to sentinel value.
    :return: sounding_table: Same as input, except for the following.
        [1] Some rows may have been removed.
        [2] Some values may have been replaced with sentinel value.
    """

    surface_flags = sounding_table[IS_SURFACE_COLUMN_FOR_SHARPPY_INPUT].values
    if not numpy.any(surface_flags):
        return sounding_table

    surface_row = numpy.where(surface_flags)[0][0]
    surface_height_m_asl = sounding_table[
        HEIGHT_COLUMN_FOR_SHARPPY_INPUT].values[surface_row]

    subsurface_flags = (sounding_table[HEIGHT_COLUMN_FOR_SHARPPY_INPUT].values <
                        surface_height_m_asl)
    subsurface_rows = numpy.where(subsurface_flags)[0]

    if delete_rows:
        pressure_1000mb_flags = (
            sounding_table[PRESSURE_COLUMN_FOR_SHARPPY_INPUT].values == 1000)
        bad_flags = numpy.logical_and(
            subsurface_flags, numpy.invert(pressure_1000mb_flags))

        bad_rows = numpy.where(bad_flags)[0]
        sounding_table = sounding_table.drop(
            sounding_table.index[bad_rows], axis=0, inplace=False).reset_index(
                drop=True)

        subsurface_flags = (
            sounding_table[HEIGHT_COLUMN_FOR_SHARPPY_INPUT].values <
            surface_height_m_asl)
        subsurface_rows = numpy.where(subsurface_flags)[0]

    sounding_table[TEMPERATURE_COLUMN_FOR_SHARPPY_INPUT].values[
        subsurface_rows] = SENTINEL_VALUE_FOR_SHARPPY
    sounding_table[DEWPOINT_COLUMN_FOR_SHARPPY_INPUT].values[
        subsurface_rows] = SENTINEL_VALUE_FOR_SHARPPY
    sounding_table[U_WIND_COLUMN_FOR_SHARPPY_INPUT].values[
        subsurface_rows] = SENTINEL_VALUE_FOR_SHARPPY
    sounding_table[V_WIND_COLUMN_FOR_SHARPPY_INPUT].values[
        subsurface_rows] = SENTINEL_VALUE_FOR_SHARPPY

    return sounding_table


def _sort_sounding_by_height(sounding_table):
    """Sorts rows in sounding table by increasing height.

    :param sounding_table: See documentation for
        get_sounding_stats_from_sharppy.
    :return: sounding_table: Same as input, except row order.
    """

    return sounding_table.sort_values(
        HEIGHT_COLUMN_FOR_SHARPPY_INPUT, axis=0, ascending=True, inplace=False)


def _remove_redundant_sounding_data(sorted_sounding_table):
    """If surface is very close to an adjacent level, removes adjacent level.

    :param sorted_sounding_table: See documentation for
        _sort_sounding_by_height.
    :return: sorted_sounding_table: Same as input, except maybe with fewer rows.
    """

    surface_flags = sorted_sounding_table[
        IS_SURFACE_COLUMN_FOR_SHARPPY_INPUT].values
    if not numpy.any(surface_flags):
        return sorted_sounding_table

    surface_row = numpy.where(surface_flags)[0][0]
    bad_rows = []

    if surface_row != len(surface_flags) - 1:
        height_diff_metres = numpy.absolute(
            sorted_sounding_table[HEIGHT_COLUMN_FOR_SHARPPY_INPUT].values[
                surface_row + 1] -
            sorted_sounding_table[HEIGHT_COLUMN_FOR_SHARPPY_INPUT].values[
                surface_row])
        pressure_diff_mb = numpy.absolute(
            sorted_sounding_table[PRESSURE_COLUMN_FOR_SHARPPY_INPUT].values[
                surface_row + 1] -
            sorted_sounding_table[PRESSURE_COLUMN_FOR_SHARPPY_INPUT].values[
                surface_row])

        if (height_diff_metres < REDUNDANT_HEIGHT_TOLERANCE_METRES or
                pressure_diff_mb < REDUNDANT_PRESSURE_TOLERANCE_MB):
            bad_rows.append(surface_row + 1)

    if surface_row != 0:
        height_diff_metres = numpy.absolute(
            sorted_sounding_table[HEIGHT_COLUMN_FOR_SHARPPY_INPUT].values[
                surface_row - 1] -
            sorted_sounding_table[HEIGHT_COLUMN_FOR_SHARPPY_INPUT].values[
                surface_row])
        pressure_diff_mb = numpy.absolute(
            sorted_sounding_table[PRESSURE_COLUMN_FOR_SHARPPY_INPUT].values[
                surface_row - 1] -
            sorted_sounding_table[PRESSURE_COLUMN_FOR_SHARPPY_INPUT].values[
                surface_row])

        if (height_diff_metres < REDUNDANT_HEIGHT_TOLERANCE_METRES or
                pressure_diff_mb < REDUNDANT_PRESSURE_TOLERANCE_MB):
            bad_rows.append(surface_row - 1)

    if not bad_rows:
        return sorted_sounding_table

    bad_rows = numpy.array(bad_rows, dtype=int)
    return sorted_sounding_table.drop(
        sorted_sounding_table.index[bad_rows], axis=0,
        inplace=False).reset_index(drop=True)


def _adjust_srw_for_storm_motion(profile_object, eastward_motion_kt=None,
                                 northward_motion_kt=None):
    """Adjusts storm-relative winds to account for storm motion.

    :param profile_object: Instance of `sharppy.sharptab.Profile`.
    :param eastward_motion_kt: u-component of storm motion (knots).
    :param northward_motion_kt: v-component of storm motion (knots).
    :return: profile_object: Same as input, but with different storm-relative
        winds.
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
        stu=eastward_motion_kt, stv=northward_motion_kt)
    profile_object.srw_0_2km = sharppy_winds.sr_wind(
        profile_object, pbot=surface_pressure_mb, ptop=pressure_at_2km_mb,
        stu=eastward_motion_kt, stv=northward_motion_kt)
    profile_object.srw_3km = sharppy_winds.sr_wind(
        profile_object, pbot=surface_pressure_mb, ptop=pressure_at_3km_mb,
        stu=eastward_motion_kt, stv=northward_motion_kt)
    profile_object.srw_6km = sharppy_winds.sr_wind(
        profile_object, pbot=surface_pressure_mb, ptop=pressure_at_6km_mb,
        stu=eastward_motion_kt, stv=northward_motion_kt)
    profile_object.srw_8km = sharppy_winds.sr_wind(
        profile_object, pbot=surface_pressure_mb, ptop=pressure_at_8km_mb,
        stu=eastward_motion_kt, stv=northward_motion_kt)
    profile_object.srw_4_5km = sharppy_winds.sr_wind(
        profile_object, pbot=pressure_at_4km_mb, ptop=pressure_at_5km_mb,
        stu=eastward_motion_kt, stv=northward_motion_kt)
    profile_object.srw_4_6km = sharppy_winds.sr_wind(
        profile_object, pbot=pressure_at_4km_mb, ptop=pressure_at_6km_mb,
        stu=eastward_motion_kt, stv=northward_motion_kt)
    profile_object.srw_9_11km = sharppy_winds.sr_wind(
        profile_object, pbot=pressure_at_9km_mb, ptop=pressure_at_11km_mb,
        stu=eastward_motion_kt, stv=northward_motion_kt)
    profile_object.srw_ebw = sharppy_winds.sr_wind(
        profile_object, pbot=profile_object.ebottom,
        ptop=effective_bulk_layer_top_mb, stu=eastward_motion_kt,
        stv=northward_motion_kt)
    profile_object.srw_eff = sharppy_winds.sr_wind(
        profile_object, pbot=profile_object.ebottom, ptop=profile_object.etop,
        stu=eastward_motion_kt, stv=northward_motion_kt)
    profile_object.srw_lcl_el = sharppy_winds.sr_wind(
        profile_object, pbot=profile_object.mupcl.lclpres,
        ptop=profile_object.mupcl.elpres, stu=eastward_motion_kt,
        stv=northward_motion_kt)

    return profile_object


def _adjust_stats_for_storm_motion(profile_object, eastward_motion_kt=None,
                                   northward_motion_kt=None):
    """Adjusts sounding statistics to account for storm motion.

    :param profile_object: Instance of `sharppy.sharptab.Profile`.
    :param eastward_motion_kt: u-component of storm motion (knots).
    :param northward_motion_kt: v-component of storm motion (knots).
    :return: profile_object: Same as input, except for stats that depend on
        storm motion.
    """

    profile_object = _adjust_srw_for_storm_motion(
        profile_object, eastward_motion_kt=eastward_motion_kt,
        northward_motion_kt=northward_motion_kt)

    profile_object.critical_angle = sharppy_winds.critical_angle(
        profile_object, stu=eastward_motion_kt, stv=northward_motion_kt)
    profile_object.srh1km = sharppy_winds.helicity(
        profile_object, 0., 1000., stu=eastward_motion_kt,
        stv=northward_motion_kt)
    profile_object.srh3km = sharppy_winds.helicity(
        profile_object, 0., 3000., stu=eastward_motion_kt,
        stv=northward_motion_kt)
    profile_object.ehi1km = sharppy_params.ehi(
        profile_object, profile_object.mupcl, 0., 1000., stu=eastward_motion_kt,
        stv=northward_motion_kt)
    profile_object.ehi3km = sharppy_params.ehi(
        profile_object, profile_object.mupcl, 0., 3000., stu=eastward_motion_kt,
        stv=northward_motion_kt)

    effective_layer_srh_j_kg01 = sharppy_winds.helicity(
        profile_object, profile_object.ebotm, profile_object.etopm,
        stu=eastward_motion_kt, stv=northward_motion_kt)[1]
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

    SHARPpy format = masked array with direction/magnitude
    New format = normal numpy array with u- and v-components

    SHARPpy creates masked arrays only for mean wind from 0-1 km AGL,
    0-3 km AGL, 0-6 km AGL, 0-8 km AGL, and lifting condensation level
    - equilibrium level.  Why??

    :param sharppy_wind_vector: Wind vector in SHARPpy format.
    :return: wind_vector: Wind vector in new format.
    """

    if not sharppy_wind_vector[0] or not sharppy_wind_vector[1]:
        return numpy.full(2, numpy.nan)

    return numpy.asarray(sharppy_utils.vec2comp(
        sharppy_wind_vector[0], sharppy_wind_vector[1]))


def _fix_sharppy_wind_vectors(profile_object):
    """Converts all wind vectors in profile from SHARPpy format to new format.

    See _fix_sharppy_wind_vector for definitions of the two formats.

    :param profile_object: Instance of `sharppy.sharptab.Profile`.
    :return: profile_object: Same as input, but with all wind vectors in new
        format.
    """

    for this_sharppy_name in SHARPPY_NAMES_FOR_MASKED_WIND_ARRAYS:
        setattr(profile_object, this_sharppy_name, _fix_sharppy_wind_vector(
            getattr(profile_object, this_sharppy_name)))

    return profile_object


def _sharppy_profile_to_table(profile_object, metadata_table):
    """Converts SHARPpy profile to pandas DataFrame.

    :param profile_object: Instance of `sharppy.sharptab.Profile`.
    :param metadata_table: pandas DataFrame created by
        read_metadata_for_sounding_stats.
    :return: sounding_stat_table_sharppy: pandas DataFrame with sounding
        statistics in SHARPpy format.  Column names are SHARPpy names for
        sounding stats, and values are in SHARPpy units.
    """

    profile_object.mupcl.brndenom = profile_object.mupcl.brnshear
    profile_object.mupcl.brnshear = numpy.array(
        [profile_object.mupcl.brnu, profile_object.mupcl.brnv])

    is_vector_flags = metadata_table[IS_VECTOR_COLUMN_FOR_METADATA].values
    vector_rows = numpy.where(is_vector_flags)[0]
    scalar_rows = numpy.where(numpy.invert(is_vector_flags))[0]

    nan_array = numpy.array([numpy.nan])
    statistic_names_sharppy = metadata_table[
        SHARPPY_NAME_COLUMN_FOR_METADATA].values

    sounding_stat_dict_sharppy = {}
    for this_row in scalar_rows:
        sounding_stat_dict_sharppy.update(
            {statistic_names_sharppy[this_row]: nan_array})
    sounding_stat_table_sharppy = pandas.DataFrame.from_dict(
        sounding_stat_dict_sharppy)

    first_statistic_name = list(sounding_stat_table_sharppy)[0]
    nested_array = sounding_stat_table_sharppy[[
        first_statistic_name, first_statistic_name]].values.tolist()

    argument_dict = {}
    for this_row in vector_rows:
        argument_dict.update({statistic_names_sharppy[this_row]: nested_array})
    sounding_stat_table_sharppy = sounding_stat_table_sharppy.assign(
        **argument_dict)

    num_statistics = len(statistic_names_sharppy)
    for j in range(num_statistics):
        if statistic_names_sharppy[j] in HELICITY_NAMES_SHARPPY:
            this_vector = getattr(
                profile_object, statistic_names_sharppy[j])
            sounding_stat_table_sharppy[
                statistic_names_sharppy[j]].values[0] = this_vector[1]

        elif metadata_table[IN_MUPCL_OBJECT_COLUMN_FOR_METADATA].values[j]:
            sounding_stat_table_sharppy[
                statistic_names_sharppy[j]].values[0] = getattr(
                    profile_object.mupcl, statistic_names_sharppy[j])
        else:
            sounding_stat_table_sharppy[
                statistic_names_sharppy[j]].values[0] = getattr(
                    profile_object, statistic_names_sharppy[j])

    return sounding_stat_table_sharppy


def _split_vector_column(input_table, conversion_factor=1.):
    """Splits column of 2-D vectors into 5 columns, listed below.

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
    :return: vector_dict: Dictionary with the following keys (if the original
        column name is "foo").
    vector_dict["foo_x"]: length-N numpy array of x-components.
    vector_dict["foo_y"]: length-N numpy array of y-components.
    vector_dict["foo_magnitude"]: length-N numpy array of magnitudes.
    vector_dict["foo_cos"]: length-N numpy array with cosines of vector
        directions.
    vector_dict["foo_sin"]: length-N numpy array with sines of vector
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

    return {
        '{0:s}_{1:s}'.format(input_column, X_COMPONENT_SUFFIX): x_components,
        '{0:s}_{1:s}'.format(input_column, Y_COMPONENT_SUFFIX): y_components,
        '{0:s}_{1:s}'.format(input_column, MAGNITUDE_SUFFIX): magnitudes,
        '{0:s}_{1:s}'.format(input_column, COSINE_SUFFIX): cosines,
        '{0:s}_{1:s}'.format(input_column, SINE_SUFFIX): sines}


def _sounding_to_sharppy_units(sounding_table):
    """Converts sounding from GewitterGefahr units to SHARPpy units.

    :param sounding_table: pandas DataFrame with the following columns (must
        contain either relative_humidity_percent or specific_humidity; does not
        need both).
    sounding_table.pressure_mb: Pressure (millibars).
    sounding_table.temperature_kelvins: Temperature.
    sounding_table.relative_humidity_percent: [optional] Relative humidity.
    sounding_table.specific_humidity: [optional] Specific humidity (unitless).
    sounding_table.geopotential_height_metres: Geopotential height.
    sounding_table.u_wind_m_s01: u-wind (metres per second).
    sounding_table.v_wind_m_s01: v-wind (metres per second).
    :return: sounding_table: pandas DataFrame required by
        get_sounding_stats_from_sharppy.
    """

    columns_to_drop = [nwp_model_utils.TEMPERATURE_COLUMN_FOR_SOUNDING_TABLES,
                       nwp_model_utils.U_WIND_COLUMN_FOR_SOUNDING_TABLES,
                       nwp_model_utils.V_WIND_COLUMN_FOR_SOUNDING_TABLES]

    if nwp_model_utils.SPFH_COLUMN_FOR_SOUNDING_TABLES in sounding_table:
        columns_to_drop.append(nwp_model_utils.SPFH_COLUMN_FOR_SOUNDING_TABLES)

        dewpoints_kelvins = moisture_conversions.specific_humidity_to_dewpoint(
            sounding_table[
                nwp_model_utils.SPFH_COLUMN_FOR_SOUNDING_TABLES].values,
            sounding_table[
                PRESSURE_COLUMN_FOR_SHARPPY_INPUT].values * MB_TO_PASCALS)
    else:
        columns_to_drop.append(nwp_model_utils.RH_COLUMN_FOR_SOUNDING_TABLES)

        dewpoints_kelvins = moisture_conversions.relative_humidity_to_dewpoint(
            PERCENT_TO_UNITLESS * sounding_table[
                nwp_model_utils.RH_COLUMN_FOR_SOUNDING_TABLES].values,
            sounding_table[
                nwp_model_utils.TEMPERATURE_COLUMN_FOR_SOUNDING_TABLES].values,
            sounding_table[
                PRESSURE_COLUMN_FOR_SHARPPY_INPUT].values * MB_TO_PASCALS)

    dewpoints_deg_c = temperature_conversions.kelvins_to_celsius(
        dewpoints_kelvins)
    temperatures_deg_c = temperature_conversions.kelvins_to_celsius(
        sounding_table[
            nwp_model_utils.TEMPERATURE_COLUMN_FOR_SOUNDING_TABLES].values)
    u_winds_kt = METRES_PER_SECOND_TO_KT * sounding_table[
        nwp_model_utils.U_WIND_COLUMN_FOR_SOUNDING_TABLES].values
    v_winds_kt = METRES_PER_SECOND_TO_KT * sounding_table[
        nwp_model_utils.V_WIND_COLUMN_FOR_SOUNDING_TABLES].values

    argument_dict = {DEWPOINT_COLUMN_FOR_SHARPPY_INPUT: dewpoints_deg_c,
                     TEMPERATURE_COLUMN_FOR_SHARPPY_INPUT: temperatures_deg_c,
                     U_WIND_COLUMN_FOR_SHARPPY_INPUT: u_winds_kt,
                     V_WIND_COLUMN_FOR_SHARPPY_INPUT: v_winds_kt}
    sounding_table = sounding_table.assign(**argument_dict)
    return sounding_table.drop(columns_to_drop, axis=1)


def _create_query_point_table(storm_object_table, lead_times_seconds):
    """Creates table of query points for interpolation.

    Each query point consists of (latitude, longitude, time).

    T = number of lead times

    :param storm_object_table: pandas DataFrame with columns documented in
        `storm_tracking_io.write_processed_file`.  May contain additional
        columns.
    :param lead_times_seconds: length-T numpy array of lead times.  For each
        lead time t, each storm object will be extrapolated t seconds into the
        future, given its motion estimate at the valid time.
    :return: query_point_table: pandas DataFrame with the following columns,
        where each row is an extrapolated storm object.
    query_point_table.centroid_lat_deg: Latitude (deg N) of extrapolated
        centroid.
    query_point_table.centroid_lng_deg: Longitude (deg E) of extrapolated
        centroid.
    query_point_table.unix_time_sec: Time to which position has been
        extrapolated.
    query_point_table.lead_time_seconds: Lead time.
    query_point_table.east_velocity_m_s01: Eastward component of storm motion
        (metres per second).
    query_point_table.north_velocity_m_s01: Northward component of storm motion
        (metres per second).
    """

    if numpy.any(lead_times_seconds > 0):
        storm_speeds_m_s01, geodetic_bearings_deg = (
            geodetic_utils.xy_components_to_displacements_and_bearings(
                storm_object_table[tracking_io.EAST_VELOCITY_COLUMN].values,
                storm_object_table[tracking_io.NORTH_VELOCITY_COLUMN].values))

    num_storm_objects = len(storm_object_table.index)
    num_lead_times = len(lead_times_seconds)
    list_of_query_point_tables = [None] * num_lead_times

    for i in range(num_lead_times):
        if lead_times_seconds[i] == 0:
            list_of_query_point_tables[i] = storm_object_table[[
                tracking_io.CENTROID_LAT_COLUMN,
                tracking_io.CENTROID_LNG_COLUMN, tracking_io.TIME_COLUMN,
                tracking_io.EAST_VELOCITY_COLUMN,
                tracking_io.NORTH_VELOCITY_COLUMN]]

            argument_dict = {
                LEAD_TIME_COLUMN: numpy.full(num_storm_objects, 0, dtype=int)}
            list_of_query_point_tables[i] = (
                list_of_query_point_tables[i].assign(
                    **argument_dict))

        else:
            these_extrap_lats_deg, these_extrap_lngs_deg = (
                geodetic_utils.start_points_and_distances_and_bearings_to_endpoints(
                    start_latitudes_deg=
                    storm_object_table[tracking_io.CENTROID_LAT_COLUMN].values,
                    start_longitudes_deg=
                    storm_object_table[tracking_io.CENTROID_LNG_COLUMN].values,
                    displacements_metres=
                    storm_speeds_m_s01 * lead_times_seconds[i],
                    geodetic_bearings_deg=geodetic_bearings_deg))

            this_dict = {
                tracking_io.CENTROID_LAT_COLUMN: these_extrap_lats_deg,
                tracking_io.CENTROID_LNG_COLUMN: these_extrap_lngs_deg,
                tracking_io.TIME_COLUMN:
                    (storm_object_table[tracking_io.TIME_COLUMN].values +
                     lead_times_seconds[i]),
                tracking_io.EAST_VELOCITY_COLUMN: storm_object_table[
                    tracking_io.EAST_VELOCITY_COLUMN].values,
                tracking_io.NORTH_VELOCITY_COLUMN: storm_object_table[
                    tracking_io.NORTH_VELOCITY_COLUMN].values,
                LEAD_TIME_COLUMN: numpy.full(
                    num_storm_objects, lead_times_seconds[i], dtype=int)
            }
            list_of_query_point_tables[i] = pandas.DataFrame.from_dict(
                this_dict)

        if i == 0:
            continue

        list_of_query_point_tables[i], _ = list_of_query_point_tables[i].align(
            list_of_query_point_tables[0], axis=1)

    return pandas.concat(list_of_query_point_tables, axis=0, ignore_index=True)


def _get_unique_storm_soundings(
        list_of_sounding_tables, eastward_motions_m_s01=None,
        northward_motions_m_s01=None):
    """Finds unique storm soundings (pairs of sounding and motion vector).

    N = number of storm soundings
    U = number of unique storm soundings

    :param list_of_sounding_tables: length-N list of pandas DataFrames.  Some
        entries may be None.
    :param eastward_motions_m_s01: length-N numpy array with eastward components
        of storm motion.
    :param northward_motions_m_s01: length-N numpy array with northward
        components of storm motion.
    :return: unique_indices: length-U numpy array with array indices of unique
        storm soundings.  If unique_indices[i] = j, the [i]th unique storm
        sounding is the [j]th original storm sounding.
    :return: orig_to_unique_indices: length-N numpy array.  If
        orig_to_unique_indices[j] = i, the [j]th original storm sounding is an
        instance of the [i]th unique storm sounding.
    """

    num_soundings = len(list_of_sounding_tables)
    sounding_strings = [''] * num_soundings

    for i in range(num_soundings):
        if list_of_sounding_tables[i] is None:
            sounding_strings[i] = 'None'
        else:
            sounding_strings[i] = '{0:s}_{1:.4f}_{2:.4f}'.format(
                list_of_sounding_tables[i].to_string, eastward_motions_m_s01[i],
                northward_motions_m_s01[i])

    _, unique_indices, orig_to_unique_indices = numpy.unique(
        numpy.asarray(sounding_strings), return_index=True, return_inverse=True)
    return unique_indices, orig_to_unique_indices


def get_sounding_stat_columns(sounding_stat_table):
    """Returns names of columns with sounding statistics.

    :param sounding_stat_table: pandas DataFrame.
    :return: statistic_column_names: 1-D list containing names of columns with
        sounding stats.  If there are no columns with sounding stats, this is
        None.
    """

    metadata_table = read_metadata_for_sounding_stats()
    valid_statistic_names = metadata_table[STAT_NAME_COLUMN_FOR_METADATA].values

    column_names = list(sounding_stat_table)
    statistic_column_names = None

    for this_column_name in column_names:
        this_statistic_name = _column_name_to_sounding_stat(
            this_column_name, valid_statistic_names)
        if this_statistic_name is None:
            continue

        if statistic_column_names is None:
            statistic_column_names = [this_column_name]
        else:
            statistic_column_names.append(this_column_name)

    return statistic_column_names


def check_sounding_stat_table(sounding_stat_table, require_storm_objects=True):
    """Ensures that pandas DataFrame contains sounding statistics.

    :param sounding_stat_table: pandas DataFrame.
    :param require_storm_objects: Boolean flag.  If True, sounding_stat_table
        must have columns "storm_id" and "unix_time_sec".  If False,
        sounding_stat_table does not need these columns.
    :return: statistic_column_names: 1-D list containing names of columns with
        sounding statistics.
    :raises: ValueError: if sounding_stat_table has no columns with sounding
        statistics.
    """

    statistic_column_names = get_sounding_stat_columns(sounding_stat_table)
    if statistic_column_names is None:
        raise ValueError(
            'sounding_stat_table has no columns with sounding statistics.')

    if require_storm_objects:
        error_checking.assert_columns_in_dataframe(
            sounding_stat_table, STORM_COLUMNS_TO_KEEP)

    return statistic_column_names


def read_metadata_for_sounding_stats():
    """Reads metadata for sounding statistics.

    :return: metadata_table: pandas DataFrame with the following columns.
    metadata_table.statistic_name: Name of sounding statistic in GewitterGefahr
        format.
    metadata_table.statistic_name_sharppy: Name of sounding statistic in SHARPpy
        format.
    metadata_table.conversion_factor: Multiplier from SHARPpy units to
        GewitterGefahr units.
    metadata_table.is_vector: Boolean flag.  If True, the corresponding
        statistic is a 2-D vector.  If False, the corresponding stat is a
        scalar.
    metadata_table.in_mupcl_object: Boolean flag.  If True, the
        corresponding statistic is an attribute of profile_object.mupcl, where
        profile_object is an instance of `sharppy.sharptab.Profile`.  If False,
        the corresponding stat is an attribute of just profile_object.
    """

    error_checking.assert_file_exists(METAFILE_NAME)
    return pandas.read_csv(
        METAFILE_NAME, header=0, usecols=METADATA_COLUMNS,
        dtype=COLUMN_TYPE_DICT_FOR_METADATA)


def interp_soundings_from_nwp(
        query_point_table, model_name=None, grid_id=None,
        top_grib_directory_name=None,
        wgrib_exe_name=grib_io.WGRIB_EXE_NAME_DEFAULT,
        wgrib2_exe_name=grib_io.WGRIB2_EXE_NAME_DEFAULT,
        raise_error_if_missing=False):
    """Interpolates soundings from NWP model to query points.

    Each query point consists of (latitude, longitude, time).

    :param query_point_table: pandas DataFrame in format specified by
        `interp.interp_nwp_from_xy_grid`.
    :param model_name: Name of model.
    :param grid_id: String ID for model grid.
    :param top_grib_directory_name: Name of top-level directory with grib files
        for said model.
    :param wgrib_exe_name: Path to wgrib executable.
    :param wgrib2_exe_name: Path to wgrib2 executable.
    :param raise_error_if_missing: See documentation for
        `interp.interp_nwp_from_xy_grid`.
    :return: interp_table: pandas DataFrame, where each column is one field and
        each row is one query point.  Column names are given by the list
        sounding_field_names returned by `_get_nwp_fields_in_sounding`.
    """

    sounding_field_names, sounding_field_names_grib1 = (
        _get_nwp_fields_in_sounding(model_name, return_dict=False))

    return interp.interp_nwp_from_xy_grid(
        query_point_table, model_name=model_name, grid_id=grid_id,
        field_names=sounding_field_names,
        field_names_grib1=sounding_field_names_grib1,
        top_grib_directory_name=top_grib_directory_name,
        temporal_interp_method=TEMPORAL_INTERP_METHOD,
        spatial_interp_method=SPATIAL_INTERP_METHOD,
        wgrib_exe_name=wgrib_exe_name, wgrib2_exe_name=wgrib2_exe_name,
        raise_error_if_missing=raise_error_if_missing)


def interp_soundings_from_ruc_all_grids(
        query_point_table, top_grib_directory_name=None,
        wgrib_exe_name=grib_io.WGRIB_EXE_NAME_DEFAULT,
        wgrib2_exe_name=grib_io.WGRIB2_EXE_NAME_DEFAULT,
        raise_error_if_missing=False):
    """Interpolates RUC soundings from one or more grids to query points.

    :param query_point_table: See documentation for interp_soundings_from_nwp.
    :param top_grib_directory_name: See doc for interp_soundings_from_nwp.
    :param wgrib_exe_name: See doc for interp_soundings_from_nwp.
    :param wgrib2_exe_name: See doc for interp_soundings_from_nwp.
    :param raise_error_if_missing: See doc for interp_soundings_from_nwp.
    :return: interp_table: See doc for interp_soundings_from_nwp.
    """

    sounding_field_names, sounding_field_names_grib1 = (
        _get_nwp_fields_in_sounding(
            nwp_model_utils.RUC_MODEL_NAME, return_dict=False))

    return interp.interp_ruc_all_grids(
        query_point_table, field_names=sounding_field_names,
        field_names_grib1=sounding_field_names_grib1,
        top_grib_directory_name=top_grib_directory_name,
        temporal_interp_method=TEMPORAL_INTERP_METHOD,
        spatial_interp_method=SPATIAL_INTERP_METHOD,
        wgrib_exe_name=wgrib_exe_name, wgrib2_exe_name=wgrib2_exe_name,
        raise_error_if_missing=raise_error_if_missing)


def interp_table_to_sharppy_sounding_tables(interp_table, model_name):
    """Converts each row of interp_table to a sounding table in SHARPpy format.

    N = number of soundings
    P = number of pressure levels in each sounding

    :param interp_table: N-row pandas DataFrame created by
        `interp.interp_nwp_from_xy_grid`, where each column is one model
        field (e.g., 500-mb height or 600-mb temperature).
    :param model_name: Name of NWP model used to create interp_table.
    :return: list_of_sounding_tables: length-N list of sounding tables.  Each
        table is a pandas DataFrame in the format required by
        get_sounding_stats_from_sharppy.
    """

    # TODO(thunderhoser): Might be able to produce one output table and convert
    # units faster?

    sounding_table_to_field_dict, pressure_levels_mb = (
        _get_nwp_fields_in_sounding(model_name, return_dict=True))
    sounding_table_columns = sounding_table_to_field_dict.keys()
    num_sounding_table_columns = len(sounding_table_columns)

    lowest_pressure_name, _ = nwp_model_utils.get_lowest_pressure_name(
        model_name)

    pressure_levels_mb = numpy.concatenate((
        pressure_levels_mb, numpy.array([numpy.nan])))
    is_surface_flags = numpy.full(len(pressure_levels_mb), False, dtype=int)
    is_surface_flags[-1] = True

    base_sounding_dict = {
        PRESSURE_COLUMN_FOR_SHARPPY_INPUT: pressure_levels_mb,
        IS_SURFACE_COLUMN_FOR_SHARPPY_INPUT: is_surface_flags}
    base_sounding_table = pandas.DataFrame.from_dict(base_sounding_dict)

    num_soundings = len(interp_table.index)
    list_of_sounding_tables = [base_sounding_table] * num_soundings

    for i in range(num_soundings):
        for j in range(num_sounding_table_columns):
            these_sounding_field_names = sounding_table_to_field_dict[
                sounding_table_columns[j]]

            argument_dict = {
                sounding_table_columns[j]:
                    interp_table[these_sounding_field_names].values[i]}
            list_of_sounding_tables[i] = list_of_sounding_tables[i].assign(
                **argument_dict)

        list_of_sounding_tables[i][PRESSURE_COLUMN_FOR_SHARPPY_INPUT].values[
            -1] = PASCALS_TO_MB * interp_table[lowest_pressure_name].values[i]
        list_of_sounding_tables[i] = _remove_sounding_rows_with_nan(
            list_of_sounding_tables[i])
        if list_of_sounding_tables[i] is None:
            continue

        list_of_sounding_tables[i] = _sounding_to_sharppy_units(
            list_of_sounding_tables[i])

    return list_of_sounding_tables


def get_sounding_stats_from_sharppy(
        sounding_table, eastward_motion_m_s01=None, northward_motion_m_s01=None,
        metadata_table=None):
    """Uses SHARPpy to compute sounding statistics.

    This method works on a single sounding.

    :param sounding_table: pandas DataFrame with the following columns.
    sounding_table.pressure_mb: Pressure (millibars).
    sounding_table.geopotential_height_metres: Geopotential height.
    sounding_table.temperature_deg_c: Temperature.
    sounding_table.dewpoint_deg_c: Temperature.
    sounding_table.u_wind_kt: u-wind (knots).
    sounding_table.v_wind_kt: v-wind (knots).
    sounding_table.is_surface: Boolean flag, indicating which row is the surface
        level.
    :param eastward_motion_m_s01: u-component of storm motion (metres per
        second).
    :param northward_motion_m_s01: v-component of storm motion (metres per
        second).
    :param metadata_table: pandas DataFrame created by
        read_metadata_for_sounding_stats.
    :return: sounding_stat_table_sharppy: pandas DataFrame with sounding
        statistics in SHARPpy format.  Column names are SHARPpy names for
        sounding stats, and values are in SHARPpy units.
    """

    error_checking.assert_is_not_nan(eastward_motion_m_s01)
    error_checking.assert_is_not_nan(northward_motion_m_s01)

    sounding_table = _remove_subsurface_sounding_data(
        sounding_table, delete_rows=False)
    sounding_table = _sort_sounding_by_height(sounding_table)
    sounding_table = _remove_redundant_sounding_data(sounding_table)

    try:
        profile_object = sharppy_profile.create_profile(
            profile='convective',
            pres=sounding_table[PRESSURE_COLUMN_FOR_SHARPPY_INPUT].values,
            hght=sounding_table[HEIGHT_COLUMN_FOR_SHARPPY_INPUT].values,
            tmpc=sounding_table[TEMPERATURE_COLUMN_FOR_SHARPPY_INPUT].values,
            dwpc=sounding_table[DEWPOINT_COLUMN_FOR_SHARPPY_INPUT].values,
            u=sounding_table[U_WIND_COLUMN_FOR_SHARPPY_INPUT].values,
            v=sounding_table[V_WIND_COLUMN_FOR_SHARPPY_INPUT].values)

    except:
        sounding_table = _remove_subsurface_sounding_data(
            sounding_table, delete_rows=True)

        profile_object = sharppy_profile.create_profile(
            profile='convective',
            pres=sounding_table[PRESSURE_COLUMN_FOR_SHARPPY_INPUT].values,
            hght=sounding_table[HEIGHT_COLUMN_FOR_SHARPPY_INPUT].values,
            tmpc=sounding_table[TEMPERATURE_COLUMN_FOR_SHARPPY_INPUT].values,
            dwpc=sounding_table[DEWPOINT_COLUMN_FOR_SHARPPY_INPUT].values,
            u=sounding_table[U_WIND_COLUMN_FOR_SHARPPY_INPUT].values,
            v=sounding_table[V_WIND_COLUMN_FOR_SHARPPY_INPUT].values)

    setattr(profile_object, STORM_VELOCITY_NAME_SHARPPY,
            numpy.array([eastward_motion_m_s01, northward_motion_m_s01]))

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
        profile_object,
        eastward_motion_kt=METRES_PER_SECOND_TO_KT * eastward_motion_m_s01,
        northward_motion_kt=METRES_PER_SECOND_TO_KT * northward_motion_m_s01)
    profile_object = _fix_sharppy_wind_vectors(profile_object)
    return _sharppy_profile_to_table(profile_object, metadata_table)


def convert_sounding_stats_from_sharppy(sounding_stat_table_sharppy,
                                        metadata_table):
    """Converts names and units of sounding statistics.

    :param sounding_stat_table_sharppy: pandas DataFrame with columns generated
        by get_sounding_stats_from_sharppy.  Each row is one sounding.
    :param metadata_table: pandas DataFrame created by
        read_metadata_for_sounding_stats.
    :return: sounding_stat_table: pandas DataFrame with sounding
        statistics in GewitterGefahr format.  Column names are GewitterGefahr
        names for sounding stats, and values are in GewitterGefahr units.
        Vectors are split into components, so each column is either a scalar
        sounding statistic or one component of a vector.
    """

    orig_column_names = list(sounding_stat_table_sharppy)
    sounding_stat_table = None

    for this_orig_name in orig_column_names:
        match_flags = [s == this_orig_name for s in metadata_table[
            SHARPPY_NAME_COLUMN_FOR_METADATA].values]
        match_index = numpy.where(match_flags)[0][0]

        this_new_name = metadata_table[STAT_NAME_COLUMN_FOR_METADATA].values[
            match_index]
        this_conversion_factor = metadata_table[
            CONVERSION_FACTOR_COLUMN_FOR_METADATA].values[match_index]
        this_vector_flag = metadata_table[IS_VECTOR_COLUMN_FOR_METADATA].values[
            match_index]

        if this_vector_flag:
            this_column_as_table = sounding_stat_table_sharppy[[this_orig_name]]
            this_column_as_table.rename(
                columns={this_orig_name: this_new_name}, inplace=True)
            argument_dict = _split_vector_column(
                this_column_as_table, this_conversion_factor)
        else:
            argument_dict = {
                this_new_name:
                    this_conversion_factor * sounding_stat_table_sharppy[
                        this_orig_name].values}

        if sounding_stat_table is None:
            sounding_stat_table = pandas.DataFrame.from_dict(argument_dict)
        else:
            sounding_stat_table = sounding_stat_table.assign(**argument_dict)

    temperatures_kelvins = temperature_conversions.fahrenheit_to_kelvins(
        sounding_stat_table[CONVECTIVE_TEMPERATURE_NAME].values)
    argument_dict = {CONVECTIVE_TEMPERATURE_NAME: temperatures_kelvins}
    return sounding_stat_table.assign(**argument_dict)


def get_sounding_stats_for_storm_objects(
        storm_object_table, lead_times_seconds=numpy.array([0]),
        all_ruc_grids=False, model_name=None, grid_id=None,
        top_grib_directory_name=None,
        wgrib_exe_name=grib_io.WGRIB_EXE_NAME_DEFAULT,
        wgrib2_exe_name=grib_io.WGRIB2_EXE_NAME_DEFAULT,
        raise_error_if_missing=False):
    """Computes sounding statistics for each storm object.

    N = number of storm objects
    T = number of lead times
    K = number of sounding indices, after decomposing vectors into scalars.

    :param storm_object_table: pandas DataFrame with columns documented in
        `storm_tracking_io.write_processed_file`.
    :param lead_times_seconds: length-T numpy array of lead times.  For each
        lead time t, each storm object will be extrapolated t seconds into the
        future, given its motion estimate at the valid time.
    :param all_ruc_grids: Boolean flag.  If True, this method will use
        `interp_soundings_from_ruc_all_grids` to interpolate soundings to storm
        objects.  If False, will use `interp_soundings_from_nwp`.
    :param model_name: Soundings will be interpolated from this NWP model.
        If all_ruc_grids = True, you can leave this as None.
    :param grid_id: String ID for model grid.  If all_ruc_grids = True, you can
        leave this as None.
    :param top_grib_directory_name: Name of top-level directory with grib files
        for said model.
    :param wgrib_exe_name: Path to wgrib executable.
    :param wgrib2_exe_name: Path to wgrib2 executable.
    :param raise_error_if_missing: See documentation for
        interp_soundings_from_nwp.
    :return: sounding_stat_table_for_storms: pandas DataFrame with N*T rows and
        3 + K columns.  The first 3 columns are listed below.  The last K
        columns are sounding statistics.  Names of the last K columns are from
        the column "statistic_name" of the table returned by
        read_metadata_for_sounding_stats.
    sounding_stat_table_for_storms.storm_id: String ID for storm cell.  Same as
        input column `storm_object_table.storm_id`.
    sounding_stat_table_for_storms.unix_time_sec: Valid time.  Same as input
        column `storm_object_table.unix_time_sec`.
    sounding_stat_table_for_storms.lead_time_seconds: Lead time.
    """

    error_checking.assert_is_integer_numpy_array(lead_times_seconds)
    error_checking.assert_is_numpy_array(lead_times_seconds, num_dimensions=1)
    error_checking.assert_is_geq_numpy_array(lead_times_seconds, 0)
    error_checking.assert_is_boolean(all_ruc_grids)

    query_point_table = _create_query_point_table(
        storm_object_table, lead_times_seconds)

    column_dict_old_to_new = {
        tracking_io.CENTROID_LAT_COLUMN: interp.QUERY_LAT_COLUMN,
        tracking_io.CENTROID_LNG_COLUMN: interp.QUERY_LNG_COLUMN,
        tracking_io.TIME_COLUMN: interp.QUERY_TIME_COLUMN}
    query_point_table.rename(columns=column_dict_old_to_new, inplace=True)

    if all_ruc_grids:
        model_name = nwp_model_utils.RUC_MODEL_NAME
        interp_table = interp_soundings_from_ruc_all_grids(
            query_point_table, top_grib_directory_name=top_grib_directory_name,
            wgrib_exe_name=wgrib_exe_name, wgrib2_exe_name=wgrib2_exe_name,
            raise_error_if_missing=raise_error_if_missing)
    else:
        interp_table = interp_soundings_from_nwp(
            query_point_table, model_name=model_name, grid_id=grid_id,
            top_grib_directory_name=top_grib_directory_name,
            wgrib_exe_name=wgrib_exe_name, wgrib2_exe_name=wgrib2_exe_name,
            raise_error_if_missing=raise_error_if_missing)

    list_of_sounding_tables = interp_table_to_sharppy_sounding_tables(
        interp_table, model_name)
    num_soundings = len(list_of_sounding_tables)

    unique_sounding_indices, orig_to_unique_sounding_indices = (
        _get_unique_storm_soundings(
            list_of_sounding_tables,
            eastward_motions_m_s01=
            query_point_table[tracking_io.EAST_VELOCITY_COLUMN].values,
            northward_motions_m_s01=
            query_point_table[tracking_io.NORTH_VELOCITY_COLUMN].values))

    list_of_sounding_tables = list_of_sounding_tables[unique_sounding_indices]
    num_unique_soundings = len(list_of_sounding_tables)
    list_of_sharppy_stat_tables = [None] * num_soundings
    metadata_table = read_metadata_for_sounding_stats()

    print 'Number of total soundings, unique soundings = {0:d}, {1:d}'.format(
        num_soundings, num_unique_soundings)

    for i in range(num_unique_soundings):
        print 'Computing stats for unique sounding {0:d}/{1:d}...'.format(
            i + 1, num_unique_soundings)

        if list_of_sounding_tables[i] is None:
            this_statistic_table = _get_empty_sharppy_stat_table(
                query_point_table[tracking_io.EAST_VELOCITY_COLUMN].values[i],
                query_point_table[tracking_io.NORTH_VELOCITY_COLUMN].values[i])
        else:
            this_statistic_table = get_sounding_stats_from_sharppy(
                list_of_sounding_tables[i],
                eastward_motion_m_s01=
                query_point_table[tracking_io.EAST_VELOCITY_COLUMN].values[i],
                northward_motion_m_s01=
                query_point_table[tracking_io.NORTH_VELOCITY_COLUMN].values[i],
                metadata_table=metadata_table)

        these_orig_indices = numpy.where(
            orig_to_unique_sounding_indices == i)[0]
        for this_index in these_orig_indices:
            list_of_sharppy_stat_tables[this_index] = this_statistic_table

    for i in range(1, num_soundings):
        list_of_sharppy_stat_tables[i], _ = (
            list_of_sharppy_stat_tables[i].align(
                list_of_sharppy_stat_tables[0], axis=1))

    sounding_stat_table_sharppy = pandas.concat(
        list_of_sharppy_stat_tables, axis=0, ignore_index=True)
    sounding_stat_table_for_storms = convert_sounding_stats_from_sharppy(
        sounding_stat_table_sharppy, metadata_table)
    sounding_stat_table_for_storms = pandas.concat(
        [storm_object_table[STORM_COLUMNS_TO_KEEP],
         sounding_stat_table_for_storms], axis=1)

    argument_dict = {
        LEAD_TIME_COLUMN: query_point_table[LEAD_TIME_COLUMN].values}
    return sounding_stat_table_for_storms.assign(**argument_dict)


def write_sounding_stats_for_storm_objects(
        sounding_stat_table_for_storms, lead_times_seconds=None,
        pickle_file_names=None):
    """Writes sounding stats for storm objects to Pickle files.

    T = number of lead times

    :param sounding_stat_table_for_storms: pandas DataFrame created by
        get_sounding_stats_for_storm_objects.  This method will write columns
        "storm_id", "unix_time_sec", and sounding statistics.
    :param lead_times_seconds: length-T numpy array of lead times.
    :param pickle_file_names: length-T list of paths to output files (one per
        lead time).
    """

    statistic_column_names = check_sounding_stat_table(
        sounding_stat_table_for_storms, require_storm_objects=True)
    error_checking.assert_columns_in_dataframe(
        sounding_stat_table_for_storms, [LEAD_TIME_COLUMN])

    error_checking.assert_is_integer_numpy_array(lead_times_seconds)
    error_checking.assert_is_numpy_array(lead_times_seconds, num_dimensions=1)
    error_checking.assert_is_geq_numpy_array(lead_times_seconds, 0)
    num_lead_times = len(lead_times_seconds)

    error_checking.assert_is_string_list(pickle_file_names)
    error_checking.assert_is_numpy_array(
        numpy.asarray(pickle_file_names),
        exact_dimensions=numpy.array([num_lead_times]))

    columns_to_write = STORM_COLUMNS_TO_KEEP + statistic_column_names
    for i in range(num_lead_times):
        file_system_utils.mkdir_recursive_if_necessary(
            file_name=pickle_file_names[i])

        this_file_handle = open(pickle_file_names[i], 'wb')
        pickle.dump(
            sounding_stat_table_for_storms.loc[
                sounding_stat_table_for_storms[LEAD_TIME_COLUMN] ==
                lead_times_seconds[i]][columns_to_write],
            this_file_handle)
        this_file_handle.close()


def read_sounding_stats_for_storm_objects(pickle_file_name):
    """Reads sounding statistics for storm objects from a Pickle file.

    :param pickle_file_name: Path to input file.
    :return: sounding_stat_table_for_storms: pandas DataFrame with columns
        documented in write_sounding_stats_for_storm_objects.
    """

    pickle_file_handle = open(pickle_file_name, 'rb')
    sounding_stat_table_for_storms = pickle.load(pickle_file_handle)
    pickle_file_handle.close()

    check_sounding_stat_table(
        sounding_stat_table_for_storms, require_storm_objects=True)
    return sounding_stat_table_for_storms
