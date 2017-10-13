"""Methods for creating soundings from an NWP model.

--- DEFINITIONS ---

NWP = numerical weather prediction
"""

import numpy
import pandas
from sharppy.sharptab import params as sharppy_params
from sharppy.sharptab import winds as sharppy_winds
from sharppy.sharptab import interp as sharppy_interp
from sharppy.sharptab import profile as sharppy_profile
from sharppy.sharptab import utils as sharppy_utils
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

CELSIUS_TO_KELVINS_ADDEND = 273.15
FAHRENHEIT_TO_CELSIUS_ADDEND = -32.
FAHRENHEIT_TO_CELSIUS_RATIO = 5. / 9.
KT_TO_METRES_PER_SECOND = 1.852 / 3.6
STANDARD_HEIGHTS_M_ASL = numpy.array(
    [1000., 2000., 3000., 4000., 5000., 6000., 8000., 9000., 11000.])

PRESSURE_COLUMN_FOR_SOUNDING_INDICES = 'pressure_mb'
HEIGHT_COLUMN_FOR_SOUNDING_INDICES = 'height_m_asl'
TEMPERATURE_COLUMN_FOR_SOUNDING_INDICES = 'temperature_deg_c'
DEWPOINT_COLUMN_FOR_SOUNDING_INDICES = 'dewpoint_deg_c'
U_WIND_COLUMN_FOR_SOUNDING_INDICES = 'u_wind_kt'
V_WIND_COLUMN_FOR_SOUNDING_INDICES = 'v_wind_kt'

SOUNDING_INDEX_NAME_COLUMN = 'name'
SHARPPY_NAME_COLUMN = 'sharppy_name'
CONVERSION_FACTOR_COLUMN = 'conversion_factor'
IS_VECTOR_COLUMN = 'is_vector'
IN_MUPCL_OBJECT_COLUMN = 'in_mupcl_object'

SOUNDING_INDEX_METADATA_COLUMNS = [
    SOUNDING_INDEX_NAME_COLUMN, SHARPPY_NAME_COLUMN, CONVERSION_FACTOR_COLUMN,
    IS_VECTOR_COLUMN, IN_MUPCL_OBJECT_COLUMN]

METADATA_COLUMN_TYPE_DICT = {
    SOUNDING_INDEX_NAME_COLUMN: str, SHARPPY_NAME_COLUMN: str,
    CONVERSION_FACTOR_COLUMN: numpy.float64, IS_VECTOR_COLUMN: bool,
    IN_MUPCL_OBJECT_COLUMN: bool}

CONVECTIVE_TEMPERATURE_NAME = 'convective_temperature_kelvins'
SHARPPY_NAMES_FOR_HELICITY = ['srh1km', 'srh3km', 'left_esrh', 'right_esrh']
SHARPPY_NAMES_FOR_MASKED_WIND_ARRAYS = [
    'mean_1km', 'mean_3km', 'mean_6km', 'mean_8km', 'mean_lcl_el']

X_COMPONENT_SUFFIX = 'x'
Y_COMPONENT_SUFFIX = 'y'
MAGNITUDE_SUFFIX = 'magnitude'
COSINE_SUFFIX = 'cos'
SINE_SUFFIX = 'sin'


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


def _fahrenheit_to_kelvins(temperatures_deg_f):
    """Converts temperatures from deg F to K.

    :param temperatures_deg_f: numpy array of temperatures in deg F.
    :return: temperatures_kelvins: Same as input, except temperatures are now in
        civilized units.
    """

    return ((temperatures_deg_f + FAHRENHEIT_TO_CELSIUS_ADDEND) *
            FAHRENHEIT_TO_CELSIUS_RATIO) + CELSIUS_TO_KELVINS_ADDEND


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


def _get_srw_correct_storm_motion(profile_object, eastward_motion_kt=None,
                                  northward_motion_kt=None):
    """Recomputes storm-relative winds, using correct storm motion.

    This method operates on a single sounding.

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

    return profile_object


def _get_sounding_indices_correct_storm_motion(profile_object,
                                               eastward_motion_kt=None,
                                               northward_motion_kt=None):
    """Recomputes sounding indices affected by storm motion.

    This method operates on a single sounding.

    :param profile_object: Instance of `sharppy.sharptab.Profile`.
    :param eastward_motion_kt: u-component of storm motion (knots).
    :param northward_motion_kt: v-component of storm motion (knots).
    :return: profile_object: Same as input, but with different storm-relative
        winds.
    """

    profile_object = _get_srw_correct_storm_motion(
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

    profile_object.get_traj()  # Recomputes updraft tilt.

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
    return profile_object


def _fix_vector(sharppy_vector):
    """Converts vector from SHARPpy format to new format.

    SHARPpy format = masked array with speed and direction.  As far as I can
        tell, only 5 vectors are stored as masked arrays: mean wind from 0-1 km
        AGL, 0-3 km AGL, 0-6 km AGL, 0-8 km AGL, and lifting condensation level
        - equilibrium level.  Why, I have no idea.
    New format = simple numpy array with u- and v-components.

    :param sharppy_vector: Vector in SHARPpy format.
    :return: new_vector: Vector in new format.
    """

    if not sharppy_vector[0] or not sharppy_vector[1]:
        return numpy.full(2, numpy.nan)

    return numpy.asarray(sharppy_utils.vec2comp(
        sharppy_vector[0], sharppy_vector[1]))


def _fix_wind_vectors(profile_object):
    """Converts the following wind vectors from SHARPpy format to new format:

    - mean wind from 0-1 km AGL
    - mean wind from 0-3 km AGL
    - mean wind from 0-6 km AGL
    - mean wind from 0-8 km AGL
    - mean wind from lifting condensation level to equilibrium level

    For more on "SHARPpy format" and "new format," see documentation for
    _fix_vector.

    :param profile_object: Instance of `sharppy.sharptab.Profile`.
    :return: profile_object: Same as input, but with wind vectors fixed.
    """

    for this_sharppy_name in SHARPPY_NAMES_FOR_MASKED_WIND_ARRAYS:
        setattr(profile_object, this_sharppy_name,
                _fix_vector(getattr(profile_object, this_sharppy_name)))

    return profile_object


def _profile_object_to_dataframe(profile_object, sounding_index_metadata_table):
    """Converts instance of `sharppy.sharptab.Profile` to pandas DataFrame.

    :param profile_object: Instance of `sharppy.sharptab.Profile`.
    :param sounding_index_metadata_table: pandas DataFrame created by
        read_metadata_for_sounding_indices.
    :param sounding_index_table_sharppy: pandas DataFrame, where column names
        are SHARPpy names for sounding indices.  Values are in SHARPpy units.
    """

    profile_object.mupcl.brndenom = profile_object.mupcl.brnshear
    profile_object.mupcl.brnshear = numpy.array(
        [profile_object.mupcl.brnu, profile_object.mupcl.brnv])

    is_vector_flags = sounding_index_metadata_table[IS_VECTOR_COLUMN].values
    vector_rows = numpy.where(is_vector_flags)[0]
    scalar_rows = numpy.where(numpy.invert(is_vector_flags))[0]

    nan_array = numpy.array([numpy.nan])
    sounding_index_names_sharppy = sounding_index_metadata_table[
        SHARPPY_NAME_COLUMN].values

    sounding_index_dict_sharppy = {}
    for this_row in scalar_rows:
        sounding_index_dict_sharppy.update(
            {sounding_index_names_sharppy[this_row]: nan_array})

    sounding_index_table_sharppy = pandas.DataFrame.from_dict(
        sounding_index_dict_sharppy)

    this_scalar_index_name = list(sounding_index_table_sharppy)[0]
    nested_array = sounding_index_table_sharppy[[
        this_scalar_index_name, this_scalar_index_name]].values.tolist()

    argument_dict = {}
    for this_row in vector_rows:
        argument_dict.update(
            {sounding_index_names_sharppy[this_row]: nested_array})

    sounding_index_table_sharppy = sounding_index_table_sharppy.assign(
        **argument_dict)

    num_sounding_indices = len(sounding_index_names_sharppy)
    for j in range(num_sounding_indices):
        if sounding_index_names_sharppy[j] in SHARPPY_NAMES_FOR_HELICITY:
            this_vector = getattr(
                profile_object, sounding_index_names_sharppy[j])
            sounding_index_table_sharppy[
                sounding_index_names_sharppy[j]].values[0] = this_vector[1]

        elif sounding_index_metadata_table[IN_MUPCL_OBJECT_COLUMN].values[j]:
            sounding_index_table_sharppy[
                sounding_index_names_sharppy[j]].values[0] = getattr(
                    profile_object.mupcl, sounding_index_names_sharppy[j])
        else:
            sounding_index_table_sharppy[
                sounding_index_names_sharppy[j]].values[0] = getattr(
                    profile_object, sounding_index_names_sharppy[j])

    return sounding_index_table_sharppy


def _split_vector_column(input_table):
    """Splits column of 2-D vectors into 5 columns, listed below:

    - x-component
    - y-component
    - magnitude
    - sine of direction
    - cosine of direction

    N = number of vectors

    :param input_table: pandas DataFrame with one column, where each row is a
        2-D vector.
    :return: output_dict: Dictionary with 5 keys, where each value is a length-N
        numpy array.  If the original column (the single column of input_table)
        is "foo", the keys will be as follows.
    output_dict["foo_x"]: x-component
    output_dict["foo_y"]: y-component
    output_dict["foo_magnitude"]: magnitude
    output_dict["foo_cos"]: cosine
    output_dict["foo_sin"]: sine
    """

    input_column = list(input_table)[0]
    num_vectors = len(input_table.index)
    x_components = numpy.full(num_vectors, numpy.nan)
    y_components = numpy.full(num_vectors, numpy.nan)

    for i in range(num_vectors):
        x_components[i] = input_table[input_column].values[i][0]
        y_components[i] = input_table[input_column].values[i][1]

    magnitudes = numpy.sqrt(x_components ** 2 + y_components ** 2)
    cosines = x_components / magnitudes
    sines = y_components / magnitudes

    return {
        '{0:s}_{1:s}'.format(input_column, X_COMPONENT_SUFFIX): x_components,
        '{0:s}_{1:s}'.format(input_column, Y_COMPONENT_SUFFIX): y_components,
        '{0:s}_{1:s}'.format(input_column, MAGNITUDE_SUFFIX): magnitudes,
        '{0:s}_{1:s}'.format(input_column, COSINE_SUFFIX): cosines,
        '{0:s}_{1:s}'.format(input_column, SINE_SUFFIX): sines}


def read_metadata_for_sounding_indices(csv_file_name):
    """Reads metadata for sounding indices from CSV file.

    :param csv_file_name: Path to input file.
    :return: sounding_index_metadata_table: pandas DataFrame, indexed by name of
        sounding index, with the following columns.
    sounding_index_metadata_table.sharppy_name: Name of sounding index in
        SHARPpy.
    sounding_index_metadata_table.conversion_factor: Conversion factor
        (multiplier) from SHARPpy units to GewitterGefahr units.
    sounding_index_metadata_table.is_vector: Boolean flag.  If True, the
        sounding index is a 2-D vector.  If False, it is a scalar.
    sounding_index_metadata_table.in_mupcl_object: Boolean flag.  If True, the
        sounding index is found profile_object.mupcl rather than profile_object,
        where profile_object is an instance of `sharppy.sharptab.Profile`.
    """

    error_checking.assert_file_exists(csv_file_name)
    return pandas.read_csv(
        csv_file_name, header=0, usecols=SOUNDING_INDEX_METADATA_COLUMNS,
        dtype=METADATA_COLUMN_TYPE_DICT)


def get_sounding_indices(sounding_table, surface_row=None,
                         eastward_motion_kt=None, northward_motion_kt=None,
                         sounding_index_metadata_table=None):
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
    :param sounding_index_metadata_table: pandas DataFrame created by
        read_metadata_for_sounding_indices.
    :return: sounding_index_table_sharppy: pandas DataFrame, where column names
        are SHARPpy names for sounding indices.  Values are in SHARPpy units.
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

    except:  # TODO(thunderhoser): specify exception type(s).
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

    profile_object = _get_sounding_indices_correct_storm_motion(
        profile_object, eastward_motion_kt=eastward_motion_kt,
        northward_motion_kt=northward_motion_kt)
    profile_object = _fix_wind_vectors(profile_object)
    return _profile_object_to_dataframe(profile_object,
                                        sounding_index_metadata_table)


def convert_sounding_indices(sounding_index_table_sharppy,
                             sounding_index_metadata_table):
    """Converts names and units of sounding indices from SHARPpy format to new.

    :param sounding_index_table_sharppy: pandas DataFrame with columns generated
        by get_sounding_indices.  Each row is one sounding.  This method may
        work on many soundings at once.
    :param sounding_index_metadata_table: pandas DataFrame created by
        read_metadata_for_sounding_indices.
    :return: sounding_index_table: pandas DataFrame, where column names are
        GewitterGefahr names for sounding indices.  Vectors are split into
        components (i.e., each column is for either a scalar sounding index or
        one component of a vector index).
    """

    orig_column_names = list(sounding_index_table_sharppy)
    sounding_index_table = None

    for this_orig_name in orig_column_names:
        match_flags = [
            s == this_orig_name for s in sounding_index_metadata_table[
                SHARPPY_NAME_COLUMN].values]
        match_index = numpy.where(match_flags)[0][0]

        this_new_name = sounding_index_metadata_table[
            SOUNDING_INDEX_NAME_COLUMN].values[match_index]
        this_conversion_factor = sounding_index_metadata_table[
            CONVERSION_FACTOR_COLUMN].values[match_index]
        this_vector_flag = sounding_index_metadata_table[
            IS_VECTOR_COLUMN].values[match_index]

        if this_vector_flag:
            this_column_as_table = sounding_index_table_sharppy[[
                this_orig_name]]
            this_column_as_table.rename(
                columns={this_orig_name: this_new_name}, inplace=True)

            this_column_as_table[this_new_name] *= this_conversion_factor
            argument_dict = _split_vector_column(this_column_as_table)
        else:
            argument_dict = {this_new_name: this_conversion_factor *
                                            sounding_index_table_sharppy[
                                                this_orig_name].values}

        if sounding_index_table is None:
            sounding_index_table = pandas.DataFrame.from_dict(argument_dict)
        else:
            sounding_index_table = sounding_index_table.assign(**argument_dict)

    temperatures_kelvins = _fahrenheit_to_kelvins(
        sounding_index_table[CONVECTIVE_TEMPERATURE_NAME].values)
    argument_dict = {CONVECTIVE_TEMPERATURE_NAME: temperatures_kelvins}
    return sounding_index_table.assign(**argument_dict)
