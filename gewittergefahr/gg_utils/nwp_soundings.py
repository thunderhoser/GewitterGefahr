"""Methods for creating soundings from an NWP model.

--- DEFINITIONS ---

NWP = numerical weather prediction
"""

import numpy
from sharppy.sharptab import params as sharppy_params
from sharppy.sharptab import winds as sharppy_winds
from sharppy.sharptab import interp as sharppy_interp
from sharppy.sharptab import profile as sharppy_profile
from gewittergefahr.gg_utils import narr_utils
from gewittergefahr.gg_utils import rap_model_utils
from gewittergefahr.gg_utils import nwp_model_utils
from gewittergefahr.gg_utils import error_checking

RAP_MODEL_NAME = 'rap'
NARR_MODEL_NAME = 'narr'
MODEL_NAMES = [RAP_MODEL_NAME, NARR_MODEL_NAME]

SENTINEL_VALUE_FOR_SOUNDING_INDICES = -9999.
REDUNDANT_PRESSURE_TOLERANCE_MB = 1e-3
REDUNDANT_HEIGHT_TOLERANCE_METRES = 1e-3

PRESSURE_COLUMN_FOR_SOUNDING_INDICES = 'pressure_mb'
HEIGHT_COLUMN_FOR_SOUNDING_INDICES = 'height_m_asl'
TEMPERATURE_COLUMN_FOR_SOUNDING_INDICES = 'temperature_deg_c'
DEWPOINT_COLUMN_FOR_SOUNDING_INDICES = 'dewpoint_deg_c'
U_WIND_COLUMN_FOR_SOUNDING_INDICES = 'u_wind_kt'
V_WIND_COLUMN_FOR_SOUNDING_INDICES = 'v_wind_kt'

STANDARD_HEIGHTS_M_ASL = numpy.array(
    [1000., 2000., 3000., 4000., 5000., 6000., 8000., 9000., 11000.])

KT_TO_METRES_PER_SECOND = 1.852 / 3.6


def _check_model_name(model_name):
    """Ensures that model name is valid.

    :param model_name: Name of model (either "rap" or "narr" -- this list will
        be expanded if/when more NWP models are included in GewitterGefahr).
    :raises: ValueError: model_name is not in list of valid names.
    """

    error_checking.assert_is_string(model_name)
    if model_name not in MODEL_NAMES:
        error_string = (
            '\n\n' + str(MODEL_NAMES) +
            '\n\nValid model names (listed above) do not include "' +
            model_name + '".')
        raise ValueError(error_string)


def _get_sounding_columns(model_name, minimum_pressure_mb=0.):
    """Returns list of sounding variables in NWP model.

    N = number of sounding variables

    :param model_name: Name of model.
    :param minimum_pressure_mb: Minimum pressure (millibars).  All sounding
        variables from a lower pressure will be ignored (not included in the
        list).
    :return: sounding_columns: length-N list with names of sounding variables.
    :return: sounding_columns_orig: length-N list with original names of
        sounding variables (names used in grib/grib2 files).
    """

    _check_model_name(model_name)
    error_checking.assert_is_geq(minimum_pressure_mb, 0)

    if model_name == RAP_MODEL_NAME:
        pressure_levels_mb = rap_model_utils.PRESSURE_LEVELS_MB[
            rap_model_utils.PRESSURE_LEVELS_MB >= minimum_pressure_mb]
        main_sounding_columns = rap_model_utils.MAIN_SOUNDING_COLUMNS
        main_sounding_columns_orig = rap_model_utils.MAIN_SOUNDING_COLUMNS_ORIG

    elif model_name == NARR_MODEL_NAME:
        pressure_levels_mb = narr_utils.PRESSURE_LEVELS_MB[
            narr_utils.PRESSURE_LEVELS_MB >= minimum_pressure_mb]
        main_sounding_columns = narr_utils.MAIN_SOUNDING_COLUMNS
        main_sounding_columns_orig = narr_utils.MAIN_SOUNDING_COLUMNS_ORIG

    num_pressure_levels = len(pressure_levels_mb)
    num_main_sounding_columns = len(main_sounding_columns_orig)
    sounding_columns = []
    sounding_columns_orig = []

    for j in range(num_main_sounding_columns):
        for k in range(num_pressure_levels):
            sounding_columns.append('{0:s}_{1:d}mb'.format(
                main_sounding_columns[j], int(pressure_levels_mb[k])))
            sounding_columns_orig.append('{0:s}:{1:d} mb'.format(
                main_sounding_columns_orig[j], int(pressure_levels_mb[k])))

        if model_name == RAP_MODEL_NAME:
            if main_sounding_columns[
                    j] == nwp_model_utils.MAIN_TEMPERATURE_COLUMN:
                sounding_columns.append(
                    rap_model_utils.LOWEST_TEMPERATURE_COLUMN)
                sounding_columns_orig.append(
                    rap_model_utils.LOWEST_TEMPERATURE_COLUMN_ORIG)

            elif main_sounding_columns[j] == nwp_model_utils.MAIN_RH_COLUMN:
                sounding_columns.append(rap_model_utils.LOWEST_RH_COLUMN)
                sounding_columns_orig.append(
                    rap_model_utils.LOWEST_RH_COLUMN_ORIG)

            elif main_sounding_columns[j] == nwp_model_utils.MAIN_GPH_COLUMN:
                sounding_columns.append(rap_model_utils.LOWEST_GPH_COLUMN)
                sounding_columns_orig.append(
                    rap_model_utils.LOWEST_GPH_COLUMN_ORIG)

            elif main_sounding_columns[j] == nwp_model_utils.MAIN_U_WIND_COLUMN:
                sounding_columns.append(rap_model_utils.LOWEST_U_WIND_COLUMN)
                sounding_columns_orig.append(
                    rap_model_utils.LOWEST_U_WIND_COLUMN_ORIG)

            elif main_sounding_columns[j] == nwp_model_utils.MAIN_V_WIND_COLUMN:
                sounding_columns.append(rap_model_utils.LOWEST_V_WIND_COLUMN)
                sounding_columns_orig.append(
                    rap_model_utils.LOWEST_V_WIND_COLUMN_ORIG)

    return sounding_columns, sounding_columns_orig


def _remove_subsurface_sounding_data(sounding_table, surface_row=None,
                                     remove_subsurface_rows=None):
    """Removes sounding data from levels below the surface.

    :param sounding_table: See documentation for get_sounding_indices.
    :param surface_row: Ordinal number, indicating which row in sounding_table
        represents the surface.  If surface_row = i, the [i]th row in
        sounding_table should contain surface data.
    :param remove_subsurface_rows: Boolean flag.  If True, will remove rows with
        subsurface data (except the 1000-mb level, where temperature, dewpoint,
        u-wind, and v-wind will be set to sentinel value).  If False, will set
        temperature, dewpoint, u-wind, and v-wind in subsurface rows to sentinel
        value.
    :return: sounding_table: Same as input, with 2 exceptions:
        [2] Some rows may have been removed.
        [3] Some temperature/dewpoint/u-wind/v-wind values may have been set to
            sentinel value.
    """

    surface_height_m_asl = (
        sounding_table[HEIGHT_COLUMN_FOR_SOUNDING_INDICES].values[surface_row])
    subsurface_flags = (
        sounding_table[HEIGHT_COLUMN_FOR_SOUNDING_INDICES].values <
        surface_height_m_asl)
    subsurface_rows = numpy.where(subsurface_flags)[0]

    if remove_subsurface_rows:
        pressure_1000mb_flags = (
            sounding_table[PRESSURE_COLUMN_FOR_SOUNDING_INDICES].values == 1000)
        bad_flags = numpy.logical_and(
            subsurface_flags, numpy.invert(pressure_1000mb_flags))

        bad_rows = numpy.where(bad_flags)[0]
        sounding_table = sounding_table.drop(
            sounding_table.index[bad_rows], axis=0, inplace=False).reset_index(
                drop=True)

        subsurface_flags = (
            sounding_table[HEIGHT_COLUMN_FOR_SOUNDING_INDICES].values <
            surface_height_m_asl)
        subsurface_rows = numpy.where(subsurface_flags)[0]

    sounding_table[TEMPERATURE_COLUMN_FOR_SOUNDING_INDICES].values[
        subsurface_rows] = SENTINEL_VALUE_FOR_SOUNDING_INDICES
    sounding_table[DEWPOINT_COLUMN_FOR_SOUNDING_INDICES].values[
        subsurface_rows] = SENTINEL_VALUE_FOR_SOUNDING_INDICES
    sounding_table[U_WIND_COLUMN_FOR_SOUNDING_INDICES].values[
        subsurface_rows] = SENTINEL_VALUE_FOR_SOUNDING_INDICES
    sounding_table[V_WIND_COLUMN_FOR_SOUNDING_INDICES].values[
        subsurface_rows] = SENTINEL_VALUE_FOR_SOUNDING_INDICES

    return sounding_table


def _sort_sounding_table_by_height(sounding_table):
    """Sorts rows in sounding table by increasing height.

    :param sounding_table: See documentation for get_sounding_indices.
    :return: sounding_table: Same as input, except sorted by increasing height.
    """

    return sounding_table.sort_values(HEIGHT_COLUMN_FOR_SOUNDING_INDICES,
                                      axis=0, ascending=True, inplace=False)


def _remove_redundant_sounding_data(sorted_sounding_table, surface_row):
    """If surface is very close to first level above, removes surface data.

    :param sorted_sounding_table: See documentation for sounding_table in
        `get_sounding_indices`.  The only difference is that this table must be
        sorted by increasing height.
    :param surface_row: Ordinal number, indicating which row in sounding_table
        represents the surface.  If surface_row = i, the [i]th row in
        sounding_table should contain surface data.
    :return: sorted_sounding_table: Same as input, except that surface
        temperature/dewpoint/u-wind/v-wind may have been set to sentinel value.
    """

    height_diff_metres = (
        sorted_sounding_table[HEIGHT_COLUMN_FOR_SOUNDING_INDICES].values[
            surface_row + 1] -
        sorted_sounding_table[HEIGHT_COLUMN_FOR_SOUNDING_INDICES].values[
            surface_row])
    pressure_diff_mb = numpy.absolute(
        sorted_sounding_table[PRESSURE_COLUMN_FOR_SOUNDING_INDICES].values[
            surface_row + 1] -
        sorted_sounding_table[PRESSURE_COLUMN_FOR_SOUNDING_INDICES].values[
            surface_row])

    if (height_diff_metres >= REDUNDANT_HEIGHT_TOLERANCE_METRES and
            pressure_diff_mb >= REDUNDANT_PRESSURE_TOLERANCE_MB):
        return sorted_sounding_table

    sorted_sounding_table[TEMPERATURE_COLUMN_FOR_SOUNDING_INDICES].values[
        surface_row] = SENTINEL_VALUE_FOR_SOUNDING_INDICES
    sorted_sounding_table[DEWPOINT_COLUMN_FOR_SOUNDING_INDICES].values[
        surface_row] = SENTINEL_VALUE_FOR_SOUNDING_INDICES
    sorted_sounding_table[U_WIND_COLUMN_FOR_SOUNDING_INDICES].values[
        surface_row] = SENTINEL_VALUE_FOR_SOUNDING_INDICES
    sorted_sounding_table[V_WIND_COLUMN_FOR_SOUNDING_INDICES].values[
        surface_row] = SENTINEL_VALUE_FOR_SOUNDING_INDICES
    return sorted_sounding_table


def get_sounding_indices(sounding_table, surface_row=None,
                         eastward_motion_kt=None, northward_motion_kt=None):
    """Computes indices for one sounding.

    :param sounding_table: pandas DataFrame with the following columns.
    sounding_table.pressure_mb: Pressure (millibars).
    sounding_table.height_m_asl: Height (metres above sea level).
    sounding_table.temperature_deg_c: Temperature (degrees Celsius).
    sounding_table.dewpoint_deg_c: Dewpoint (degrees Celsius).
    sounding_table.u_wind_kt: u-component of wind (knots).
    sounding_table.v_wind_kt: v-component of wind (knots).

    :param surface_row: See documentation for _remove_subsurface_sounding_data.
    :param eastward_motion_kt: u-component of storm motion (knots).
    :param northward_motion_kt: v-component of storm motion (knots).
    :return: sounding_index_dict: Dictionary, where each key is the name of a
        sounding index.
    """

    num_rows = len(sounding_table.index)

    error_checking.assert_is_integer(surface_row)
    error_checking.assert_is_geq(surface_row, 0)
    error_checking.assert_is_leq(surface_row, num_rows - 1)

    error_checking.assert_is_not_nan(eastward_motion_kt)
    error_checking.assert_is_not_nan(northward_motion_kt)

    sounding_table = _remove_subsurface_sounding_data(
        sounding_table, surface_row=surface_row, remove_subsurface_rows=False)
    sounding_table = _sort_sounding_table_by_height(sounding_table)
    sounding_table = _remove_redundant_sounding_data(
        sounding_table, surface_row=surface_row)

    try:
        profile_object = sharppy_profile.create_profile(
            profile='convective',
            pres=sounding_table[PRESSURE_COLUMN_FOR_SOUNDING_INDICES].values,
            hght=sounding_table[HEIGHT_COLUMN_FOR_SOUNDING_INDICES].values,
            tmpc=sounding_table[TEMPERATURE_COLUMN_FOR_SOUNDING_INDICES].values,
            dwpc=sounding_table[DEWPOINT_COLUMN_FOR_SOUNDING_INDICES].values,
            u=sounding_table[U_WIND_COLUMN_FOR_SOUNDING_INDICES].values,
            v=sounding_table[V_WIND_COLUMN_FOR_SOUNDING_INDICES].values)

    except:  # Need to catch specific exception?
        sounding_table = _remove_subsurface_sounding_data(
            sounding_table, surface_row=surface_row,
            remove_subsurface_rows=True)

        profile_object = sharppy_profile.create_profile(
            profile='convective',
            pres=sounding_table[PRESSURE_COLUMN_FOR_SOUNDING_INDICES].values,
            hght=sounding_table[HEIGHT_COLUMN_FOR_SOUNDING_INDICES].values,
            tmpc=sounding_table[TEMPERATURE_COLUMN_FOR_SOUNDING_INDICES].values,
            dwpc=sounding_table[DEWPOINT_COLUMN_FOR_SOUNDING_INDICES].values,
            u=sounding_table[U_WIND_COLUMN_FOR_SOUNDING_INDICES].values,
            v=sounding_table[V_WIND_COLUMN_FOR_SOUNDING_INDICES].values)

    # Henceforth, recalculating all indices that depend on storm motion.
    # At this point I should extract/rename sounding indices from the profile
    # object.  Then I don't have to worry about original field names more than
    # once.

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
    equilibrium_level_mb = sharppy_interp.pres(
        profile_object,
        sharppy_interp.to_msl(
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
        profile_object, pbot=profile_object.ebottom, ptop=equilibrium_level_mb,
        stu=eastward_motion_kt, stv=northward_motion_kt)
    profile_object.srw_eff = sharppy_winds.sr_wind(
        profile_object, pbot=profile_object.ebottom, ptop=profile_object.etop,
        stu=eastward_motion_kt, stv=northward_motion_kt)
    profile_object.srw_lcl_el = sharppy_winds.sr_wind(
        profile_object, pbot=profile_object.mupcl.lclpres,
        ptop=profile_object.mupcl.elpres, stu=eastward_motion_kt,
        stv=northward_motion_kt)

    profile_object.critical_angle = sharppy_winds.critical_angle(
        profile_object, stu=eastward_motion_kt, stv=northward_motion_kt)
    profile_object.srh1km = sharppy_winds.helicity(
        profile_object, 0., 1000., stu=eastward_motion_kt,
        stv=northward_motion_kt)
    profile_object.srh3km = sharppy_winds.helicity(
        profile_object, 0., 3000., stu=eastward_motion_kt,
        stv=northward_motion_kt)
    profile_object.right_esrh = sharppy_winds.helicity(
        profile_object, profile_object.ebotm, profile_object.etopm,
        stu=profile_object.srwind[0], stv=profile_object.srwind[1])

    this_shear_magnitude_m_s01 = KT_TO_METRES_PER_SECOND * numpy.sqrt(
        profile_object.sfc_6km_shear[0] ** 2 +
        profile_object.sfc_6km_shear[1] ** 2)

    profile_object.stp_fixed = sharppy_params.stp_fixed(
        profile_object.sfcpcl.bplus, profile_object.sfcpcl.lclhght,
        profile_object.srh1km[0], this_shear_magnitude_m_s01)
    profile_object.stp_cin = sharppy_params.stp_cin(
        profile_object.mlpcl.bplus, profile_object.right_esrh[0],
        profile_object.ebwspd * KT_TO_METRES_PER_SECOND,
        profile_object.mlpcl.lclhght, profile_object.mlpcl.bminus)
    profile_object.right_scp = sharppy_params.scp(
        profile_object.mupcl.bplus, profile_object.right_esrh[0],
        profile_object.ebwspd * KT_TO_METRES_PER_SECOND)

    profile_object.get_traj()  # Recalculates updraft tilt.

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

    profile_object.ehi1km = sharppy_params.ehi(
        profile_object, profile_object.mupcl, 0., 1000., stu=eastward_motion_kt,
        stv=northward_motion_kt)
    profile_object.ehi3km = sharppy_params.ehi(
        profile_object, profile_object.mupcl, 0., 3000., stu=eastward_motion_kt,
        stv=northward_motion_kt)
    profile_object.right_ehi = sharppy_params.ehi(
        profile_object, profile_object.mupcl, profile_object.ebotm,
        profile_object.etopm, stu=profile_object.srwind[0],
        stv=profile_object.srwind[1])
    profile_object.left_ehi = sharppy_params.ehi(
        profile_object, profile_object.mupcl, profile_object.ebotm,
        profile_object.etopm, stu=profile_object.srwind[2],
        stv=profile_object.srwind[3])

    boundary_layer_top_mb = sharppy_params.pbl_top(profile_object)
    boundary_layer_top_m_asl = sharppy_interp.hght(
        profile_object, boundary_layer_top_mb)
    profile_object.pbl_depth = sharppy_interp.to_agl(
        profile_object, boundary_layer_top_m_asl)

    profile_object.edepthm = profile_object.etopm - profile_object.ebotm
