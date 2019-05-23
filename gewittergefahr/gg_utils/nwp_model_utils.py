"""Helpful methods for NWP (numerical weather prediction) output."""

import numpy
import pandas
from gewittergefahr.gg_utils import grids
from gewittergefahr.gg_utils import projections
from gewittergefahr.gg_utils import number_rounding as rounder
from gewittergefahr.gg_utils import longitude_conversion as lng_conversion
from gewittergefahr.gg_utils import error_checking

HOURS_TO_SECONDS = 3600
DEGREES_TO_RADIANS = numpy.pi / 180.
TIME_FORMAT = '%Y%m%d-%H%M%S'

RAP_MODEL_NAME = 'rap'
RUC_MODEL_NAME = 'ruc'
NARR_MODEL_NAME = 'narr'
VALID_MODEL_NAMES = [RAP_MODEL_NAME, RUC_MODEL_NAME, NARR_MODEL_NAME]

NAME_OF_130GRID = '130'
NAME_OF_252GRID = '252'
NAME_OF_236GRID = '236'
NAME_OF_221GRID = '221'
NAME_OF_EXTENDED_221GRID = 'x221'

RAP_GRID_NAMES = [NAME_OF_130GRID, NAME_OF_252GRID]
RUC_GRID_NAMES = [NAME_OF_130GRID, NAME_OF_252GRID, NAME_OF_236GRID]
NARR_GRID_NAMES = [NAME_OF_221GRID, NAME_OF_EXTENDED_221GRID]

GRIB1_FILE_TYPE_STRING = 'grib1'
GRIB2_FILE_TYPE_STRING = 'grib2'
SENTINEL_VALUE = 9.999e20

MIN_GRID_POINT_X_METRES = 0.
MIN_GRID_POINT_Y_METRES = 0.

TEMPERATURE_COLUMN_FOR_SOUNDINGS = 'temperature_kelvins'
RH_COLUMN_FOR_SOUNDINGS = 'relative_humidity_percent'
SPFH_COLUMN_FOR_SOUNDINGS = 'specific_humidity'
HEIGHT_COLUMN_FOR_SOUNDINGS = 'geopotential_height_metres'
U_WIND_COLUMN_FOR_SOUNDINGS = 'u_wind_m_s01'
V_WIND_COLUMN_FOR_SOUNDINGS = 'v_wind_m_s01'

TEMPERATURE_COLUMN_FOR_SOUNDINGS_GRIB1 = 'TMP'
RH_COLUMN_FOR_SOUNDINGS_GRIB1 = 'RH'
SPFH_COLUMN_FOR_SOUNDINGS_GRIB1 = 'SPFH'
HEIGHT_COLUMN_FOR_SOUNDINGS_GRIB1 = 'HGT'
U_WIND_COLUMN_FOR_SOUNDINGS_GRIB1 = 'UGRD'
V_WIND_COLUMN_FOR_SOUNDINGS_GRIB1 = 'VGRD'

MIN_QUERY_TIME_COLUMN = 'min_query_time_unix_sec'
MAX_QUERY_TIME_COLUMN = 'max_query_time_unix_sec'
MODEL_TIMES_COLUMN = 'model_times_unix_sec'
MODEL_TIMES_NEEDED_COLUMN = 'model_time_needed_flags'

PREVIOUS_INTERP_METHOD = 'previous'
NEXT_INTERP_METHOD = 'next'
SUPERLINEAR_INTERP_METHODS = ['quadratic', 'cubic']


def _check_wind_rotation_inputs(
        u_winds_m_s01, v_winds_m_s01, rotation_angle_cosines,
        rotation_angle_sines):
    """Error-checks inputs for wind-rotation method.

    The equation is as follows, where alpha is the rotation angle.

    u_Earth = u_grid * cos(alpha) + v_grid * sin(alpha)
    v_Earth = v_grid * cos(alpha) - u_grid * sin(alpha)

    :param u_winds_m_s01: If winds are to be rotated from grid-relative to
        Earth-relative, this is a numpy array of x-components (metres per
        second).  Otherwise, this is a numpy array of eastward components (still
        metres per second).
    :param v_winds_m_s01: Same as above, except y-components or northward
        components.
    :param rotation_angle_cosines: equivalent-shape numpy array with cosines of
        rotation angles.
    :param rotation_angle_sines: equivalent-shape numpy array with sines of
        rotation angles.
    """

    error_checking.assert_is_real_numpy_array(u_winds_m_s01)
    expected_dim = numpy.array(u_winds_m_s01.shape, dtype=int)

    error_checking.assert_is_real_numpy_array(v_winds_m_s01)
    error_checking.assert_is_numpy_array(
        v_winds_m_s01, exact_dimensions=expected_dim)

    error_checking.assert_is_geq_numpy_array(rotation_angle_cosines, -1)
    error_checking.assert_is_leq_numpy_array(rotation_angle_cosines, 1)
    error_checking.assert_is_numpy_array(
        rotation_angle_cosines, exact_dimensions=expected_dim)

    error_checking.assert_is_geq_numpy_array(rotation_angle_sines, -1)
    error_checking.assert_is_leq_numpy_array(rotation_angle_sines, 1)
    error_checking.assert_is_numpy_array(
        rotation_angle_sines, exact_dimensions=expected_dim)


def check_model_name(model_name):
    """Error-checks model name.

    :param model_name: Model name.
    :raises: ValueError: if `model_name not in VALID_MODEL_NAMES`.
    """

    error_checking.assert_is_string(model_name)

    if model_name not in VALID_MODEL_NAMES:
        error_string = (
            '\n{0:s}\nValid model names (listed above) do not include "{1:s}".'
        ).format(str(VALID_MODEL_NAMES), model_name)

        raise ValueError(error_string)


def check_grid_name(model_name, grid_name=None):
    """Error-checks grid name.

    :param model_name: Model name (must be accepted by `check_model_name`).
    :param grid_name: Grid name.  If model is NARR and you leave
        `grid_name = None`, this will default to the 221 grid.
    :raises: ValueError: if grid name is not valid for the given model.
    :return: grid_name: Same as input -- unless input was None, in which case it
        has been replaced with the default value.
    """

    check_model_name(model_name)
    if model_name == NARR_MODEL_NAME and grid_name is None:
        grid_name = NAME_OF_221GRID

    if model_name == RAP_MODEL_NAME:
        valid_grid_names = RAP_GRID_NAMES
    elif model_name == RUC_MODEL_NAME:
        valid_grid_names = RUC_GRID_NAMES
    else:
        valid_grid_names = NARR_GRID_NAMES

    if grid_name not in valid_grid_names:
        error_string = (
            '\n{0:s}\nValid grids for {1:s} model are listed above and do not '
            'include "{2:s}".'
        ).format(str(valid_grid_names), model_name.upper(), grid_name)

        raise ValueError(error_string)

    return grid_name


def get_xy_grid_spacing(model_name, grid_name=None):
    """Returns grid spacing in both x- and y-directions.

    :param model_name: See doc for `check_grid_name`.
    :param grid_name: Same.
    :return: x_spacing_metres: Spacing between adjacent grid points in x-
        direction.
    :return: y_spacing_metres: Spacing between adjacent grid points in y-
        direction.
    """

    grid_name = check_grid_name(model_name=model_name, grid_name=grid_name)

    if model_name == NARR_MODEL_NAME:
        return 32463., 32463.

    if grid_name == NAME_OF_130GRID:
        return 13545., 13545.

    if grid_name == NAME_OF_252GRID:
        return 20317.625, 20317.625

    return 40635., 40635.


def get_grid_dimensions(model_name, grid_name=None):
    """Returns grid dimensions (number of rows and columns).

    :param model_name: See doc for `check_grid_name`.
    :param grid_name: Same.
    :return: num_rows: Number of rows (unique y-coordinates of grid points).
    :return: num_columns: Number of columns (unique x-coordinates of grid
        points).
    """

    grid_name = check_grid_name(model_name=model_name, grid_name=grid_name)

    if grid_name == NAME_OF_221GRID:
        return 277, 349

    if grid_name == NAME_OF_EXTENDED_221GRID:
        return 477, 549

    if grid_name == NAME_OF_130GRID:
        return 337, 451

    if grid_name == NAME_OF_252GRID:
        return 225, 301

    return 113, 151


def dimensions_to_grid(num_rows, num_columns):
    """Determines grid from dimensions.

    :param num_rows: Number of rows (unique y-coordinates of grid points).
    :param num_columns: Number of columns (unique x-coordinates of grid points).
    :return: grid_name: Grid name.
    :raises: ValueError: if dimensions do not match a known grid.
    """

    error_checking.assert_is_integer(num_rows)
    error_checking.assert_is_integer(num_columns)
    grid_dimensions = numpy.array([num_rows, num_columns], dtype=int)

    for this_grid_name in NARR_GRID_NAMES:
        these_dimensions = numpy.array(
            get_grid_dimensions(model_name=NARR_MODEL_NAME,
                                grid_name=this_grid_name),
            dtype=int
        )

        if numpy.array_equal(these_dimensions, grid_dimensions):
            return this_grid_name

    for this_grid_name in RUC_GRID_NAMES:
        these_dimensions = numpy.array(
            get_grid_dimensions(model_name=RUC_MODEL_NAME,
                                grid_name=this_grid_name),
            dtype=int
        )

        if numpy.array_equal(these_dimensions, grid_dimensions):
            return this_grid_name

    error_string = 'Cannot find grid with {0:d} rows and {1:d} columns.'.format(
        num_rows, num_columns)
    raise ValueError(error_string)


def get_time_steps(model_name):
    """Returns forecast and initialization time steps for model.

    :param model_name: Model name (must be accepted by `check_model_name`).
    :return: forecast_time_step_hours: Time between subsequent forecasts from
        the same model run.  If the model is a reanalysis (produces only 0-hour
        "forecasts"), this will be -1.
    :return: init_time_step_hours: Time between subsequent model runs
        (initializations).
    """

    check_model_name(model_name)

    if model_name == NARR_MODEL_NAME:
        return -1, 3

    return 1, 1


def get_false_easting_and_northing(model_name, grid_name=None):
    """Returns false easting and northing for model grid.

    :param model_name: See doc for `check_grid_name`.
    :param grid_name: Same.
    :return: false_easting_metres: False easting (added to x-coordinates after
        projection from lat-long).
    :return: false_northing_metres: False northing (added to y-coordinates after
        projection from lat-long).
    """

    grid_name = check_grid_name(model_name=model_name, grid_name=grid_name)

    if grid_name == NAME_OF_221GRID:
        return 5632564.44008, -1636427.98024

    if grid_name == NAME_OF_EXTENDED_221GRID:
        x_spacing_metres, y_spacing_metres = get_xy_grid_spacing(
            model_name=model_name, grid_name=grid_name)

        false_easting_metres = 5632564.44008 + 100 * x_spacing_metres
        false_northing_metres = -1636427.98024 + 100 * y_spacing_metres
        return false_easting_metres, false_northing_metres

    if grid_name == NAME_OF_130GRID:
        return 3332090.43552, -2279020.17888

    if grid_name == NAME_OF_252GRID:
        return 3332109.11198, -2279005.97252

    return 3332091.40408, -2279020.92053


def model_to_grib_types(model_name):
    """Returns grib types used by model (grib1 and/or grib2).

    :param model_name: Model name (must be accepted by `check_model_name`).
    :return: grib_file_type_strings: 1-D list of file types.
    """

    check_model_name(model_name)

    if model_name == NARR_MODEL_NAME:
        return [GRIB1_FILE_TYPE_STRING]

    if model_name == RAP_MODEL_NAME:
        return [GRIB2_FILE_TYPE_STRING]

    return [GRIB1_FILE_TYPE_STRING, GRIB2_FILE_TYPE_STRING]


def get_online_directories(model_name, grid_name=None):
    """Returns online directories with grib files for the given model/grid.

    :param model_name: See doc for `check_grid_name`.
    :param grid_name: Same.
    :return: top_online_dir_names: 1-D list of directory names (which are also
        URLs).
    """

    grid_name = check_grid_name(model_name=model_name, grid_name=grid_name)

    if model_name == NARR_MODEL_NAME:
        return ['https://www.ncei.noaa.gov/thredds/fileServer/narr-a-files']

    if model_name == RAP_MODEL_NAME:
        if grid_name == NAME_OF_130GRID:
            return ['https://www.ncei.noaa.gov/thredds/fileServer/rap130anl']

        return ['https://www.ncei.noaa.gov/thredds/fileServer/rap252anl']

    if model_name == RUC_MODEL_NAME:
        if grid_name == NAME_OF_130GRID:
            return ['https://www.ncei.noaa.gov/thredds/fileServer/ruc130anl']
        if grid_name == NAME_OF_252GRID:
            return ['https://www.ncei.noaa.gov/thredds/fileServer/ruc252anl']

    return None


def get_pressure_levels(model_name, grid_name=None):
    """Returns pressure levels for the given model/grid.

    :param model_name: See doc for `check_grid_name`.
    :param grid_name: Same.
    :return: pressure_levels_mb: 1-D numpy array of pressure levels (millibars).
    """

    grid_name = check_grid_name(model_name=model_name, grid_name=grid_name)

    if model_name == NARR_MODEL_NAME:
        return numpy.concatenate((
            numpy.linspace(100, 300, num=9, dtype=int),
            numpy.linspace(350, 700, num=8, dtype=int),
            numpy.linspace(725, 1000, num=12, dtype=int)
        ))

    if model_name == RAP_MODEL_NAME:
        return numpy.linspace(100, 1000, num=37, dtype=int)

    if grid_name == NAME_OF_236GRID:
        return numpy.linspace(100, 1000, num=19, dtype=int)

    return numpy.linspace(100, 1000, num=37, dtype=int)


def is_wind_earth_relative(model_name):
    """Indicates whether model has Earth-relative or grid-relative winds.

    :param model_name: Model name (must be accepted by `check_model_name`).
    :return: earth_relative_flag: Boolean flag.
    """

    check_model_name(model_name)
    return model_name == NARR_MODEL_NAME


def get_projection_params(model_name):
    """Returns projection parameters for the given model.

    :param model_name: Model name (must be accepted by `check_model_name`).
    :return: standard_latitudes_deg: length-2 numpy array of standard latitudes
        (deg N).
    :return: central_longitude_deg: Central longitude (deg E).
    """

    check_model_name(model_name)

    if model_name == NARR_MODEL_NAME:
        standard_latitudes_deg = numpy.full(2, 50.)
        return standard_latitudes_deg, 253.

    standard_latitudes_deg = numpy.full(2, 25.)
    return standard_latitudes_deg, 265.


def init_projection(model_name):
    """Initializes projection used by model.

    :param model_name: Model name (must be accepted by `check_model_name`).
    :return: projection_object: Instance of `pyproj.Proj`, specifying the
        projection.
    """

    standard_latitudes_deg, central_longitude_deg = get_projection_params(
        model_name)

    return projections.init_lcc_projection(
        standard_latitudes_deg=standard_latitudes_deg,
        central_longitude_deg=central_longitude_deg)


def get_columns_in_sounding_table(model_name):
    """Returns columns needed for model sounding.

    K = number of columns

    :param model_name: Model name (must be accepted by `check_model_name`).
    :return: sounding_columns: length-K list of column names in GewitterGefahr
        (nice) format.
    :return: sounding_columns_grib1: length-K list of column names in grib1
        (ugly) format.
    """

    sounding_columns = [
        TEMPERATURE_COLUMN_FOR_SOUNDINGS, HEIGHT_COLUMN_FOR_SOUNDINGS,
        U_WIND_COLUMN_FOR_SOUNDINGS, V_WIND_COLUMN_FOR_SOUNDINGS
    ]

    sounding_columns_grib1 = [
        TEMPERATURE_COLUMN_FOR_SOUNDINGS_GRIB1,
        HEIGHT_COLUMN_FOR_SOUNDINGS_GRIB1,
        U_WIND_COLUMN_FOR_SOUNDINGS_GRIB1, V_WIND_COLUMN_FOR_SOUNDINGS_GRIB1
    ]

    check_model_name(model_name)

    if model_name == NARR_MODEL_NAME:
        sounding_columns.append(SPFH_COLUMN_FOR_SOUNDINGS)
        sounding_columns_grib1.append(SPFH_COLUMN_FOR_SOUNDINGS_GRIB1)
    else:
        sounding_columns.append(RH_COLUMN_FOR_SOUNDINGS)
        sounding_columns_grib1.append(RH_COLUMN_FOR_SOUNDINGS_GRIB1)

    return sounding_columns, sounding_columns_grib1


def get_lowest_pressure_name(model_name):
    """Returns name of lowest (nearest-to-surface) pressure field for model.

    :param model_name: Model name (must be accepted by `check_model_name`).
    :return: field_name: Name of lowest pressure field in GewitterGefahr (nice)
        format.
    :return: field_name_grib1: Name of lowest pressure field in grib1 (ugly)
        format.
    """

    check_model_name(model_name)

    if model_name == NARR_MODEL_NAME:
        return 'pressure_pascals_hybrid_level1', 'PRES:hybrid lev 1'

    return 'pressure_pascals_surface', 'PRES:sfc'


def get_lowest_temperature_name(model_name):
    """Returns name of lowest (nearest-to-surface) temperature field for model.

    :param model_name: Model name (must be accepted by `check_model_name`).
    :return: field_name: Name of lowest temperature field in GewitterGefahr
        (nice) format.
    :return: field_name_grib1: Name of lowest temperature field in grib1 (ugly)
        format.
    """

    check_model_name(model_name)

    if model_name == NARR_MODEL_NAME:
        return 'temperature_kelvins_hybrid_level1', 'TMP:hybrid lev 1'

    return 'temperature_kelvins_2m_agl', 'TMP:2 m above gnd'


def get_lowest_humidity_name(model_name):
    """Returns name of lowest (nearest-to-surface) humidity field for model.

    :param model_name: Model name (must be accepted by `check_model_name`).
    :return: field_name: Name of lowest humidity field in GewitterGefahr (nice)
        format.
    :return: field_name_grib1: Name of lowest humidity field in grib1 (ugly)
        format.
    """

    check_model_name(model_name)

    if model_name == NARR_MODEL_NAME:
        return 'specific_humidity_hybrid_level1', 'SPFH:hybrid lev 1'

    return 'relative_humidity_2m_agl', 'RH:2 m above gnd'


def get_lowest_height_name(model_name):
    """Returns name of lowest (nearest-to-surface) height field for model.

    :param model_name: Model name (must be accepted by `check_model_name`).
    :return: field_name: Name of lowest height field in GewitterGefahr (nice)
        format.
    :return: field_name_grib1: Name of lowest height field in grib1 (ugly)
        format.
    """

    check_model_name(model_name)

    if model_name == NARR_MODEL_NAME:
        return 'geopotential_height_metres_hybrid_level1', 'HGT:hybrid lev 1'

    return 'geopotential_height_metres_surface', 'HGT:sfc'


def get_lowest_u_wind_name(model_name):
    """Returns name of lowest (nearest-to-surface) u-wind field for model.

    :param model_name: Model name (must be accepted by `check_model_name`).
    :return: field_name: Name of lowest u-wind field in GewitterGefahr (nice)
        format.
    :return: field_name_grib1: Name of lowest u-wind field in grib1 (ugly)
        format.
    """

    check_model_name(model_name)

    if model_name == NARR_MODEL_NAME:
        return 'u_wind_m_s01_hybrid_level1', 'UGRD:hybrid lev 1'

    return 'u_wind_m_s01_10m_agl', 'UGRD:10 m above gnd'


def get_lowest_v_wind_name(model_name):
    """Returns name of lowest (nearest-to-surface) v-wind field for model.

    :param model_name: Model name (must be accepted by `check_model_name`).
    :return: field_name: Name of lowest v-wind field in GewitterGefahr (nice)
        format.
    :return: field_name_grib1: Name of lowest v-wind field in grib1 (ugly)
        format.
    """

    check_model_name(model_name)

    if model_name == NARR_MODEL_NAME:
        return 'v_wind_m_s01_hybrid_level1', 'VGRD:hybrid lev 1'

    return 'v_wind_m_s01_10m_agl', 'VGRD:10 m above gnd'


def project_latlng_to_xy(latitudes_deg, longitudes_deg, projection_object=None,
                         model_name=None, grid_name=None):
    """Converts points from lat-long to x-y (model) coordinates.

    P = number of points

    :param latitudes_deg: length-P numpy array of latitudes (deg N).
    :param longitudes_deg: length-P numpy array of longitudes (deg E).
    :param projection_object: Instance of `pyproj.Proj`, specifying the
        projection.  If None, will create projection object on the fly.
    :param model_name: See doc for `check_grid_name`.
    :param grid_name: Same.
    :return: x_coords_metres: length-P numpy array of x-coordinates.
    :return: y_coords_metres: length-P numpy array of x-coordinates.
    """

    false_easting_metres, false_northing_metres = (
        get_false_easting_and_northing(
            model_name=model_name, grid_name=grid_name)
    )

    if projection_object is None:
        projection_object = init_projection(model_name)

    return projections.project_latlng_to_xy(
        latitudes_deg=latitudes_deg, longitudes_deg=longitudes_deg,
        projection_object=projection_object,
        false_easting_metres=false_easting_metres,
        false_northing_metres=false_northing_metres)


def project_xy_to_latlng(
        x_coords_metres, y_coords_metres, projection_object=None,
        model_name=None, grid_name=None):
    """Converts points from x-y (model) to lat-long coordinates.

    P = number of points

    :param x_coords_metres: length-P numpy array of x-coordinates.
    :param y_coords_metres: length-P numpy array of x-coordinates.
    :param projection_object: Instance of `pyproj.Proj`, specifying the
        projection.  If None, will create projection object on the fly.
    :param model_name: See doc for `check_grid_name`.
    :param grid_name: Same.
    :return: latitudes_deg: length-P numpy array of latitudes (deg N).
    :return: longitudes_deg: length-P numpy array of longitudes (deg E).
    """

    false_easting_metres, false_northing_metres = (
        get_false_easting_and_northing(
            model_name=model_name, grid_name=grid_name)
    )

    if projection_object is None:
        projection_object = init_projection(model_name)

    return projections.project_xy_to_latlng(
        x_coords_metres=x_coords_metres, y_coords_metres=y_coords_metres,
        projection_object=projection_object,
        false_easting_metres=false_easting_metres,
        false_northing_metres=false_northing_metres)


def get_xy_grid_points(model_name, grid_name=None):
    """Returns unique x- and y-coordinates of grid points.

    M = number of rows (unique y-coordinates of grid points)
    N = number of columns (unique x-coordinates of grid points)

    :param model_name: See doc for `check_grid_name`.
    :param grid_name: Same.
    :return: grid_points_x_metres: length-N numpy array of x-coordinates.
    :return: grid_points_y_metres: length-M numpy array of y-coordinates.
    """

    x_spacing_metres, y_spacing_metres = get_xy_grid_spacing(
        model_name=model_name, grid_name=grid_name)

    num_rows, num_columns = get_grid_dimensions(
        model_name=model_name, grid_name=grid_name)

    return grids.get_xy_grid_points(
        x_min_metres=MIN_GRID_POINT_X_METRES,
        y_min_metres=MIN_GRID_POINT_Y_METRES,
        x_spacing_metres=x_spacing_metres, y_spacing_metres=y_spacing_metres,
        num_rows=num_rows, num_columns=num_columns)


def get_xy_grid_cell_edges(model_name, grid_name=None):
    """Returns unique x- and y-coordinates of grid-cell edges.

    M = number of rows (unique y-coordinates of grid points)
    N = number of columns (unique x-coordinates of grid points)

    :param model_name: See doc for `check_grid_name`.
    :param grid_name: Same.
    :return: grid_cell_edges_x_metres: length-(N + 1) numpy array of
        x-coordinates.
    :return: grid_cell_edges_y_metres: length-(M + 1) numpy array of
        y-coordinates.
    """

    x_spacing_metres, y_spacing_metres = get_xy_grid_spacing(
        model_name=model_name, grid_name=grid_name)

    num_rows, num_columns = get_grid_dimensions(
        model_name=model_name, grid_name=grid_name)

    return grids.get_xy_grid_cell_edges(
        x_min_metres=MIN_GRID_POINT_X_METRES,
        y_min_metres=MIN_GRID_POINT_Y_METRES,
        x_spacing_metres=x_spacing_metres, y_spacing_metres=y_spacing_metres,
        num_rows=num_rows, num_columns=num_columns)


def get_xy_grid_point_matrices(model_name, grid_name=None):
    """Returns x- and y-coordinates of all grid points.

    M = number of rows (unique y-coordinates of grid points)
    N = number of columns (unique x-coordinates of grid points)

    :param model_name: See doc for `check_grid_name`.
    :param grid_name: Same.
    :return: grid_point_x_matrix_metres: M-by-N numpy array of x-coordinates.
    :return: grid_point_y_matrix_metres: M-by-N numpy array of y-coordinates.
    """

    grid_points_x_metres, grid_points_y_metres = get_xy_grid_points(
        model_name=model_name, grid_name=grid_name)

    return grids.xy_vectors_to_matrices(x_unique_metres=grid_points_x_metres,
                                        y_unique_metres=grid_points_y_metres)


def get_xy_grid_cell_edge_matrices(model_name, grid_name=None):
    """Returns x- and y-coordinates of all grid-cell edges.

    M = number of rows (unique y-coordinates of grid points)
    N = number of columns (unique x-coordinates of grid points)

    :param model_name: See doc for `check_grid_name`.
    :param grid_name: Same.
    :return: grid_cell_edge_x_matrix_metres: (M + 1)-by-(N + 1) numpy array of
        x-coordinates.
    :return: grid_cell_edge_y_matrix_metres: (M + 1)-by-(N + 1) numpy array of
        y-coordinates.
    """

    grid_cell_edges_x_metres, grid_cell_edges_y_metres = get_xy_grid_cell_edges(
        model_name=model_name, grid_name=grid_name)

    return grids.xy_vectors_to_matrices(
        x_unique_metres=grid_cell_edges_x_metres,
        y_unique_metres=grid_cell_edges_y_metres)


def get_latlng_grid_point_matrices(model_name, grid_name=None,
                                   projection_object=None):
    """Returns lat-long coordinates of all grid points.

    M = number of rows (unique y-coordinates of grid points)
    N = number of columns (unique x-coordinates of grid points)

    :param model_name: See doc for `check_grid_name`.
    :param grid_name: Same.
    :param projection_object: Instance of `pyproj.Proj`, specifying the
        projection.  If None, will create projection object on the fly.
    :return: grid_point_lat_matrix_deg: M-by-N numpy array of latitudes (deg N).
    :return: grid_point_lng_matrix_deg: M-by-N numpy array of longitudes
        (deg E).
    """

    grid_point_x_matrix_metres, grid_point_y_matrix_metres = (
        get_xy_grid_point_matrices(model_name=model_name, grid_name=grid_name)
    )

    return project_xy_to_latlng(
        x_coords_metres=grid_point_x_matrix_metres,
        y_coords_metres=grid_point_y_matrix_metres,
        projection_object=projection_object,
        model_name=model_name, grid_name=grid_name)


def get_latlng_grid_cell_edge_matrices(model_name, grid_name=None,
                                       projection_object=None):
    """Returns lat-long coordinates of all grid-cell edges.

    M = number of rows (unique y-coordinates of grid points)
    N = number of columns (unique x-coordinates of grid points)

    :param model_name: See doc for `check_grid_name`.
    :param grid_name: Same.
    :param projection_object: Instance of `pyproj.Proj`, specifying the
        projection.  If None, will create projection object on the fly.
    :return: grid_cell_edge_lat_matrix_deg: (M + 1)-by-(N + 1) numpy array of
        latitudes (deg N).
    :return: grid_cell_edge_lng_matrix_deg: (M + 1)-by-(N + 1) numpy array of
        longitudes (deg E).
    """

    grid_cell_edge_x_matrix_metres, grid_cell_edge_y_matrix_metres = (
        get_xy_grid_cell_edge_matrices(
            model_name=model_name, grid_name=grid_name)
    )

    return project_xy_to_latlng(
        x_coords_metres=grid_cell_edge_x_matrix_metres,
        y_coords_metres=grid_cell_edge_y_matrix_metres,
        projection_object=projection_object,
        model_name=model_name, grid_name=grid_name)


def get_wind_rotation_angles(latitudes_deg, longitudes_deg, model_name):
    """Computes wind-rotation angle for each point.

    These angles may be used in `rotate_winds_to_earth_relative` or
    `rotate_winds_to_grid_relative`.

    :param latitudes_deg: numpy array of latitudes (deg N).
    :param longitudes_deg: numpy array of longitudes (deg E) with same shape as
        `latitudes_deg`.
    :param model_name: Model name (must be accepted by `check_model_name`).
    :return: rotation_angle_cosines: numpy array of cosines with same shape as
        `latitudes_deg`.
    :return: rotation_angle_sines: numpy array of sines with same shape as
        `latitudes_deg`.
    """

    error_checking.assert_is_valid_lat_numpy_array(
        latitudes_deg=latitudes_deg, allow_nan=False)

    longitudes_deg = lng_conversion.convert_lng_positive_in_west(
        longitudes_deg=longitudes_deg, allow_nan=False)

    these_expected_dim = numpy.array(latitudes_deg.shape, dtype=int)
    error_checking.assert_is_numpy_array(
        longitudes_deg, exact_dimensions=these_expected_dim)

    standard_latitudes_deg, central_longitude_deg = get_projection_params(
        model_name)

    rotation_angles = (
        numpy.sin(standard_latitudes_deg[0] * DEGREES_TO_RADIANS) *
        (longitudes_deg - central_longitude_deg) * DEGREES_TO_RADIANS
    )

    return numpy.cos(rotation_angles), numpy.sin(rotation_angles)


def rotate_winds_to_earth_relative(
        u_winds_grid_relative_m_s01, v_winds_grid_relative_m_s01,
        rotation_angle_cosines, rotation_angle_sines):
    """Rotates wind vectors from grid-relative to Earth-relative.

    The equation is as follows, where alpha = rotation angle.

    u_Earth = u_grid * cos(alpha) + v_grid * sin(alpha)
    v_Earth = v_grid * cos(alpha) - u_grid * sin(alpha)

    :param u_winds_grid_relative_m_s01: See doc for
        `_check_wind_rotation_inputs`.
    :param v_winds_grid_relative_m_s01: Same.
    :param rotation_angle_cosines: Same.
    :param rotation_angle_sines: Same.
    :return: u_winds_earth_relative_m_s01: numpy array of u-components (metres
        per second) with same shape as input.
    :return: v_winds_earth_relative_m_s01: Same but v-components.
    """

    _check_wind_rotation_inputs(
        u_winds_m_s01=u_winds_grid_relative_m_s01,
        v_winds_m_s01=v_winds_grid_relative_m_s01,
        rotation_angle_cosines=rotation_angle_cosines,
        rotation_angle_sines=rotation_angle_sines)

    u_winds_earth_relative_m_s01 = (
        rotation_angle_cosines * u_winds_grid_relative_m_s01 +
        rotation_angle_sines * v_winds_grid_relative_m_s01
    )

    v_winds_earth_relative_m_s01 = (
        rotation_angle_cosines * v_winds_grid_relative_m_s01 -
        rotation_angle_sines * u_winds_grid_relative_m_s01
    )

    return u_winds_earth_relative_m_s01, v_winds_earth_relative_m_s01


def rotate_winds_to_grid_relative(
        u_winds_earth_relative_m_s01, v_winds_earth_relative_m_s01,
        rotation_angle_cosines, rotation_angle_sines):
    """Rotates wind vectors from Earth-relative to grid-relative.

    The equation is as follows, where alpha = rotation angle.

    u_grid = u_Earth * cos(alpha) - v_Earth * sin(alpha)
    v_grid = v_Earth * cos(alpha) + u_Earth * sin(alpha)

    :param u_winds_earth_relative_m_s01: See doc for
        `_check_wind_rotation_inputs`.
    :param v_winds_earth_relative_m_s01: Same.
    :param rotation_angle_cosines: Same.
    :param rotation_angle_sines: Same.
    :return: u_winds_grid_relative_m_s01: numpy array of u-components (metres
        per second) with same shape as input.
    :return: v_winds_grid_relative_m_s01: Same but v-components.
    """

    _check_wind_rotation_inputs(
        u_winds_m_s01=u_winds_earth_relative_m_s01,
        v_winds_m_s01=v_winds_earth_relative_m_s01,
        rotation_angle_cosines=rotation_angle_cosines,
        rotation_angle_sines=rotation_angle_sines)

    u_winds_grid_relative_m_s01 = (
        rotation_angle_cosines * u_winds_earth_relative_m_s01 -
        rotation_angle_sines * v_winds_earth_relative_m_s01
    )

    v_winds_grid_relative_m_s01 = (
        rotation_angle_cosines * v_winds_earth_relative_m_s01 +
        rotation_angle_sines * u_winds_earth_relative_m_s01
    )

    return u_winds_grid_relative_m_s01, v_winds_grid_relative_m_s01


def get_times_needed_for_interp(query_times_unix_sec, model_time_step_hours,
                                method_string):
    """Finds model times needed for interp to each query time.

    Q = number of query times
    M = number of model times needed

    :param query_times_unix_sec: length-Q numpy array of query times.
    :param model_time_step_hours: Model time step.  If interpolating between
        forecast times from the same run, this should be time between successive
        forecasts.  If interpolating between model runs, this should be time
        between successive model runs.
    :param method_string: Interpolation method.
    :return: model_times_unix_sec: length-M numpy array of model times needed.
    :return: query_to_model_times_table: pandas DataFrame with the following
        columns.  Each row corresponds to a range of query times.
    query_to_model_times_table.min_query_time_unix_sec: Earliest query time in
        range.
    query_to_model_times_table.max_query_time_unix_sec: Latest query time in
        range.
    query_to_model_times_table.these_model_times_unix_sec: 1-D numpy array of
        model times needed for this range.
    query_to_model_times_table.model_time_needed_flags: length-M numpy array of
        Boolean flags.  If model_time_needed_flags[j] = True,
        model_times_unix_sec[j] -- or the [j]th element in the first output --
        is needed for this range of query times.
    """

    error_checking.assert_is_integer_numpy_array(query_times_unix_sec)
    error_checking.assert_is_numpy_array(query_times_unix_sec, num_dimensions=1)
    error_checking.assert_is_string(method_string)

    model_time_step_hours = int(numpy.round(model_time_step_hours))
    model_time_step_sec = model_time_step_hours * HOURS_TO_SECONDS

    min_min_query_time_unix_sec = rounder.floor_to_nearest(
        float(numpy.min(query_times_unix_sec)), model_time_step_sec
    )
    max_max_query_time_unix_sec = rounder.ceiling_to_nearest(
        float(numpy.max(query_times_unix_sec)), model_time_step_sec
    )
    if max_max_query_time_unix_sec == min_min_query_time_unix_sec:
        max_max_query_time_unix_sec += model_time_step_sec

    num_ranges = int(numpy.round(
        (max_max_query_time_unix_sec - min_min_query_time_unix_sec) /
        model_time_step_sec
    ))

    min_query_times_unix_sec = numpy.linspace(
        min_min_query_time_unix_sec,
        max_max_query_time_unix_sec - model_time_step_sec,
        num=num_ranges, dtype=int)

    max_query_times_unix_sec = numpy.linspace(
        min_min_query_time_unix_sec + model_time_step_sec,
        max_max_query_time_unix_sec,
        num=num_ranges, dtype=int)

    if method_string == PREVIOUS_INTERP_METHOD:
        min_model_time_unix_sec = min_min_query_time_unix_sec + 0
        max_model_time_unix_sec = (
            max_max_query_time_unix_sec - model_time_step_sec
        )

    elif method_string == NEXT_INTERP_METHOD:
        min_model_time_unix_sec = (
            min_min_query_time_unix_sec + model_time_step_sec
        )
        max_model_time_unix_sec = max_max_query_time_unix_sec + 0

    elif method_string in SUPERLINEAR_INTERP_METHODS:
        min_model_time_unix_sec = (
            min_min_query_time_unix_sec - model_time_step_sec
        )
        max_model_time_unix_sec = (
            max_max_query_time_unix_sec + model_time_step_sec
        )

    else:
        min_model_time_unix_sec = min_min_query_time_unix_sec + 0
        max_model_time_unix_sec = max_max_query_time_unix_sec + 0

    num_model_times = 1 + int(numpy.round(
        (max_model_time_unix_sec - min_model_time_unix_sec) /
        model_time_step_sec
    ))

    model_times_unix_sec = numpy.linspace(
        min_model_time_unix_sec, max_model_time_unix_sec, num=num_model_times,
        dtype=int)

    query_to_model_times_dict = {
        MIN_QUERY_TIME_COLUMN: min_query_times_unix_sec,
        MAX_QUERY_TIME_COLUMN: max_query_times_unix_sec
    }
    query_to_model_times_table = pandas.DataFrame.from_dict(
        query_to_model_times_dict)

    nested_array = query_to_model_times_table[[
        MIN_QUERY_TIME_COLUMN, MIN_QUERY_TIME_COLUMN
    ]].values.tolist()

    query_to_model_times_table = query_to_model_times_table.assign(**{
        MODEL_TIMES_COLUMN: nested_array,
        MODEL_TIMES_NEEDED_COLUMN: nested_array
    })

    for i in range(num_ranges):
        if method_string == PREVIOUS_INTERP_METHOD:
            these_model_times_unix_sec = min_query_times_unix_sec[[i]]
        elif method_string == NEXT_INTERP_METHOD:
            these_model_times_unix_sec = max_query_times_unix_sec[[i]]
        elif method_string in SUPERLINEAR_INTERP_METHODS:
            these_model_times_unix_sec = numpy.array([
                min_query_times_unix_sec[i] - model_time_step_sec,
                min_query_times_unix_sec[i],
                max_query_times_unix_sec[i],
                max_query_times_unix_sec[i] + model_time_step_sec
            ], dtype=int)

        else:
            these_model_times_unix_sec = numpy.array([
                min_query_times_unix_sec[i], max_query_times_unix_sec[i]
            ], dtype=int)

        query_to_model_times_table[MODEL_TIMES_COLUMN].values[i] = (
            these_model_times_unix_sec
        )

        query_to_model_times_table[MODEL_TIMES_NEEDED_COLUMN].values[i] = (
            numpy.array([
                t in these_model_times_unix_sec for t in model_times_unix_sec
            ], dtype=bool)
        )

    return model_times_unix_sec, query_to_model_times_table
