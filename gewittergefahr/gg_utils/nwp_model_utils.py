"""Processing methods for NWP (numerical weather prediction) data."""

import copy
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
MODEL_NAMES = [RAP_MODEL_NAME, RUC_MODEL_NAME, NARR_MODEL_NAME]

ID_FOR_221GRID = '221'
ID_FOR_130GRID = '130'
ID_FOR_252GRID = '252'
ID_FOR_236GRID = '236'
RAP_GRID_IDS = [ID_FOR_130GRID, ID_FOR_252GRID]
RUC_GRID_IDS = [ID_FOR_130GRID, ID_FOR_252GRID, ID_FOR_236GRID]

GRIB1_FILE_TYPE = 'grib1'
GRIB2_FILE_TYPE = 'grib2'
SENTINEL_VALUE = 9.999e20

MIN_GRID_POINT_X_METRES = 0.
MIN_GRID_POINT_Y_METRES = 0.

TEMPERATURE_COLUMN_FOR_SOUNDING_TABLES = 'temperature_kelvins'
RH_COLUMN_FOR_SOUNDING_TABLES = 'relative_humidity_percent'
SPFH_COLUMN_FOR_SOUNDING_TABLES = 'specific_humidity'
HEIGHT_COLUMN_FOR_SOUNDING_TABLES = 'geopotential_height_metres'
U_WIND_COLUMN_FOR_SOUNDING_TABLES = 'u_wind_m_s01'
V_WIND_COLUMN_FOR_SOUNDING_TABLES = 'v_wind_m_s01'

TEMPERATURE_COLUMN_FOR_SOUNDING_TABLES_GRIB1 = 'TMP'
RH_COLUMN_FOR_SOUNDING_TABLES_GRIB1 = 'RH'
SPFH_COLUMN_FOR_SOUNDING_TABLES_GRIB1 = 'SPFH'
HEIGHT_COLUMN_FOR_SOUNDING_TABLES_GRIB1 = 'HGT'
U_WIND_COLUMN_FOR_SOUNDING_TABLES_GRIB1 = 'UGRD'
V_WIND_COLUMN_FOR_SOUNDING_TABLES_GRIB1 = 'VGRD'

MIN_QUERY_TIME_COLUMN = 'min_query_time_unix_sec'
MAX_QUERY_TIME_COLUMN = 'max_query_time_unix_sec'
MODEL_TIMES_COLUMN = 'model_times_unix_sec'
MODEL_TIMES_NEEDED_COLUMN = 'model_time_needed_flags'

PREVIOUS_INTERP_METHOD = 'previous'
NEXT_INTERP_METHOD = 'next'
SUPERLINEAR_INTERP_METHODS = ['quadratic', 'cubic']


def check_model_name(model_name):
    """Ensures that model name is valid.

    :param model_name: Name of model.
    :raises: ValueError: if model name is not recognized.
    """

    error_checking.assert_is_string(model_name)
    if model_name not in MODEL_NAMES:
        error_string = (
            '\n\n' + str(MODEL_NAMES) +
            '\n\nValid model names (listed above) do not include "' +
            model_name + '".')
        raise ValueError(error_string)


def check_grid_id(model_name, grid_id=None):
    """Ensures that grid ID is valid for the given model.

    :param model_name: Name of model.
    :param grid_id: String ID for grid.  If model_name = "narr", this will not
        be used (so you can leave it as None).
    :raises: ValueError: if grid ID is not recognized for the given model.
    """

    check_model_name(model_name)
    if model_name == NARR_MODEL_NAME:
        return

    if model_name == RAP_MODEL_NAME:
        valid_grid_ids = copy.deepcopy(RAP_GRID_IDS)
    else:
        valid_grid_ids = copy.deepcopy(RUC_GRID_IDS)

    if grid_id not in valid_grid_ids:
        error_string = (
            '\n\n' + str(valid_grid_ids) + '\n\nValid grid IDs for ' +
            model_name.upper() + ' model (listed above) do not include the ' +
            'following: "' + grid_id + '"')
        raise ValueError(error_string)


def get_xy_grid_spacing(model_name, grid_id=None):
    """Returns grid spacing in both x- and y-directions.

    :param model_name: Name of model.
    :param grid_id: ID for model grid.
    :return: x_spacing_metres: Spacing between adjacent grid points in x-
        direction.
    :return: y_spacing_metres: Spacing between adjacent grid points in y-
        direction.
    """

    check_grid_id(model_name, grid_id=grid_id)
    if model_name == NARR_MODEL_NAME:
        return 32463., 32463.

    if grid_id == ID_FOR_130GRID:
        return 13545., 13545.
    if grid_id == ID_FOR_252GRID:
        return 20317.625, 20317.625

    return 40635., 40635.


def get_grid_dimensions(model_name, grid_id=None):
    """Returns dimensions of grid.

    :param model_name: Name of model.
    :param grid_id: ID for model grid.
    :return: num_grid_rows: Number of rows.  If x-y grid, this is the number of
        unique grid-point y-coordinates.  If lat-long grid, this is the number
        of unique grid-point latitudes.
    :return: num_grid_columns: Number of columns.  If x-y grid, this is the
        number of unique grid-point x-coordinates.  If lat-long grid, this is
        the number of unique grid-point longitudes.
    """

    check_grid_id(model_name, grid_id=grid_id)
    if model_name == NARR_MODEL_NAME:
        return 277, 349

    if grid_id == ID_FOR_130GRID:
        return 337, 451
    if grid_id == ID_FOR_252GRID:
        return 225, 301

    return 113, 151


def dimensions_to_grid_id(grid_dimensions):
    """Determines grid from dimensions.

    :param grid_dimensions: 1-D numpy array with [num_rows, num_columns].
    :return: grid_id: String ID for grid.
    :raises: ValueError: if dimensions do not match a known grid.
    """

    error_checking.assert_is_numpy_array(
        grid_dimensions, exact_dimensions=numpy.array([2]))
    error_checking.assert_is_integer_numpy_array(grid_dimensions)
    error_checking.assert_is_greater_numpy_array(grid_dimensions, 1)

    these_dimensions = get_grid_dimensions(NARR_MODEL_NAME)
    if numpy.array_equal(these_dimensions, grid_dimensions):
        return ID_FOR_221GRID

    for this_grid_id in RUC_GRID_IDS:
        these_dimensions = get_grid_dimensions(RUC_MODEL_NAME, this_grid_id)
        if numpy.array_equal(these_dimensions, grid_dimensions):
            return this_grid_id

    raise ValueError(
        'Dimensions (' + str(grid_dimensions[0]) + ' rows x ' +
        str(grid_dimensions[1]) + ' columns) do not match a known grid.')


def get_time_steps(model_name):
    """Returns forecast and initialization time steps for model.

    :param model_name: Name of model.
    :return: forecast_time_step_hours: Time step between subsequent forecasts
        from a single model run (initialization).
    :return: init_time_step_hours: Time step between subsequent model runs
        (initializations).
    """

    check_model_name(model_name)
    if model_name == NARR_MODEL_NAME:
        return 3, 6

    return 1, 1


def get_false_easting_and_northing(model_name, grid_id=None):
    """Returns false easting and northing for model grid.

    :param model_name: Name of model.
    :param grid_id: ID for model grid.
    :return: false_easting_metres: False easting (added to x-coordinates after
        projecting from lat-long).
    :return: false_northing_metres: False northing (added to y-coordinates after
        projecting from lat-long).
    """

    check_grid_id(model_name, grid_id=grid_id)
    if model_name == NARR_MODEL_NAME:
        return 5632564.44008, -1636427.98024

    if grid_id == ID_FOR_130GRID:
        return 3332090.43552, -2279020.17888
    if grid_id == ID_FOR_252GRID:
        return 3332109.11198, -2279005.97252

    return 3332091.40408, -2279020.92053


def get_grib_types(model_name):
    """Returns types of grib files used by model ("grib1" and/or "grib2").

    :param model_name: Name of model.
    :return: grib_types: 1-D list with types of grib files used by model.
    """

    check_model_name(model_name)
    if model_name == NARR_MODEL_NAME:
        return [GRIB1_FILE_TYPE]
    if model_name == RAP_MODEL_NAME:
        return [GRIB2_FILE_TYPE]

    return [GRIB1_FILE_TYPE, GRIB2_FILE_TYPE]


def get_top_online_directories(model_name, grid_id=None):
    """Returns top-level online directories for given model/grid.

    :param model_name: Name of model.
    :param grid_id: String ID for grid.
    :return: top_online_dir_names: 1-D list with names (URLs) of top-level
        online directories.
    """

    check_grid_id(model_name, grid_id)
    if model_name == NARR_MODEL_NAME:
        return ['https://nomads.ncdc.noaa.gov/data/narr']

    if model_name == RAP_MODEL_NAME:
        if grid_id == ID_FOR_130GRID:
            return ['https://www.ncei.noaa.gov/thredds/fileServer/rap130anl']

        return ['https://www.ncei.noaa.gov/thredds/fileServer/rap252anl']

    if model_name == RUC_MODEL_NAME:
        if grid_id == ID_FOR_130GRID:
            return ['https://www.ncei.noaa.gov/thredds/fileServer/ruc130anl']
        if grid_id == ID_FOR_252GRID:
            return ['https://www.ncei.noaa.gov/thredds/fileServer/ruc252anl']


def get_pressure_levels(model_name, grid_id=None):
    """Returns pressure levels used by model.

    :param model_name: Name of model.
    :param grid_id: String ID for grid.
    :return: pressure_levels_mb: 1-D numpy array of pressure levels (millibars).
    """

    check_grid_id(model_name, grid_id)
    if model_name == NARR_MODEL_NAME:
        return numpy.concatenate((
            numpy.linspace(100, 300, num=9, dtype=int),
            numpy.linspace(350, 700, num=8, dtype=int),
            numpy.linspace(725, 1000, num=12, dtype=int)))

    if model_name == RAP_MODEL_NAME:
        return numpy.linspace(100, 1000, num=37, dtype=int)

    if grid_id == ID_FOR_236GRID:
        return numpy.linspace(100, 1000, num=19, dtype=int)

    return numpy.linspace(100, 1000, num=37, dtype=int)


def is_wind_earth_relative(model_name):
    """Indicates whether model winds are Earth-relative or grid-relative.

    :param model_name: Name of model.
    :return: earth_relative_flag: Boolean.  If True, wind is Earth-relative.  If
        False, wind is grid-relative.
    """

    check_model_name(model_name)
    if model_name == NARR_MODEL_NAME:
        return True

    return False


def get_projection_params(model_name):
    """Returns projection parameters for model.

    :param model_name: Name of model.
    :return: standard_latitudes_deg: length-2 numpy array of standard latitudes
        (deg N).
    :return: central_longitude_deg: Central longitude (deg E).
    """

    check_model_name(model_name)
    if model_name == NARR_MODEL_NAME:
        return numpy.array([50., 50.]), 253.

    return numpy.array([25., 25.]), 265.


def init_model_projection(model_name):
    """Initializes projection used by model.

    :param model_name: Name of model.
    :return: projection_object: object created by `pyproj.Proj`, specifying the
        Lambert conformal projection.
    """

    standard_latitudes_deg, central_longitude_deg = get_projection_params(
        model_name)
    return projections.init_lambert_conformal_projection(
        standard_latitudes_deg, central_longitude_deg)


def get_columns_in_sounding_table(model_name):
    """Returns columns required to create sounding from given model.

    N = number of columns

    :param model_name: Name of model.
    :return: sounding_column_names: length-N list of column names in
        GewitterGefahr format (lower-case with underscores and units).
    :return: sounding_column_names_grib1: length-N list of variable names in
        grib1 format (used to read sounding data from grib files).
    """

    sounding_column_names = [TEMPERATURE_COLUMN_FOR_SOUNDING_TABLES,
                             HEIGHT_COLUMN_FOR_SOUNDING_TABLES,
                             U_WIND_COLUMN_FOR_SOUNDING_TABLES,
                             V_WIND_COLUMN_FOR_SOUNDING_TABLES]

    sounding_column_names_grib1 = [TEMPERATURE_COLUMN_FOR_SOUNDING_TABLES_GRIB1,
                                   HEIGHT_COLUMN_FOR_SOUNDING_TABLES_GRIB1,
                                   U_WIND_COLUMN_FOR_SOUNDING_TABLES_GRIB1,
                                   V_WIND_COLUMN_FOR_SOUNDING_TABLES_GRIB1]

    check_model_name(model_name)
    if model_name == NARR_MODEL_NAME:
        sounding_column_names.append(SPFH_COLUMN_FOR_SOUNDING_TABLES)
        sounding_column_names_grib1.append(
            SPFH_COLUMN_FOR_SOUNDING_TABLES_GRIB1)
    else:
        sounding_column_names.append(RH_COLUMN_FOR_SOUNDING_TABLES)
        sounding_column_names_grib1.append(RH_COLUMN_FOR_SOUNDING_TABLES_GRIB1)

    return sounding_column_names, sounding_column_names_grib1


def get_lowest_pressure_name(model_name):
    """Returns name of lowest (nearest to surface) pressure field.

    :param model_name: Name of model.
    :return: lowest_pressure_name: Name of lowest pressure field in
        GewitterGefahr format (lower-case with underscores and units).
    :return: lowest_pressure_name_grib1: Name of lowest pressure field in grib1
        format.
    """

    check_model_name(model_name)
    if model_name == NARR_MODEL_NAME:
        return 'pressure_pascals_hybrid_level1', 'PRES:hybrid lev 1'

    return 'pressure_pascals_surface', 'PRES:sfc'


def get_lowest_temperature_name(model_name):
    """Returns name of lowest (nearest to surface) temperature field.

    :param model_name: Name of model.
    :return: lowest_temperature_name: Name of lowest temperature field in
        GewitterGefahr format (lower-case with underscores and units).
    :return: lowest_temperature_name_grib1: Name of lowest temperature field in
        grib1 format.
    """

    check_model_name(model_name)
    if model_name == NARR_MODEL_NAME:
        return 'temperature_kelvins_hybrid_level1', 'TMP:hybrid lev 1'

    return 'temperature_kelvins_2m_agl', 'TMP:2 m above gnd'


def get_lowest_humidity_name(model_name):
    """Returns name of lowest (nearest to surface) humidity field.

    :param model_name: Name of model.
    :return: lowest_humidity_name: Name of lowest humidity field in
        GewitterGefahr format (lower-case with underscores and units).
    :return: lowest_humidity_name_grib1: Name of lowest humidity field in grib1
        format.
    """

    check_model_name(model_name)
    if model_name == NARR_MODEL_NAME:
        return 'specific_humidity_hybrid_level1', 'SPFH:hybrid lev 1'

    return 'relative_humidity_2m_agl', 'RH:2 m above gnd'


def get_lowest_height_name(model_name):
    """Returns name of lowest (nearest to surface) geopotential-height field.

    :param model_name: Name of model.
    :return: lowest_height_name: Name of lowest geopotential-height field in
        GewitterGefahr format (lower-case with underscores and units).
    :return: lowest_height_name_grib1: Name of lowest geopotential-height field
        in grib1 format.
    """

    check_model_name(model_name)
    if model_name == NARR_MODEL_NAME:
        return 'geopotential_height_metres_hybrid_level1', 'HGT:hybrid lev 1'

    return 'geopotential_height_metres_surface', 'HGT:sfc'


def get_lowest_u_wind_name(model_name):
    """Returns name of lowest (nearest to surface) u-wind field.

    :param model_name: Name of model.
    :return: lowest_u_wind_name: Name of lowest u-wind field in GewitterGefahr
        format (lower-case with underscores and units).
    :return: lowest_u_wind_name_grib1: Name of lowest u-wind field in grib1
        format.
    """

    check_model_name(model_name)
    if model_name == NARR_MODEL_NAME:
        return 'u_wind_m_s01_hybrid_level1', 'UGRD:hybrid lev 1'

    return 'u_wind_m_s01_10m_agl', 'UGRD:10 m above gnd'


def get_lowest_v_wind_name(model_name):
    """Returns name of lowest (nearest to surface) v-wind field.

    :param model_name: Name of model.
    :return: lowest_v_wind_name: Name of lowest v-wind field in GewitterGefahr
        format (lower-case with underscores and units).
    :return: lowest_v_wind_name_grib1: Name of lowest v-wind field in grib1
        format.
    """

    check_model_name(model_name)
    if model_name == NARR_MODEL_NAME:
        return 'v_wind_m_s01_hybrid_level1', 'VGRD:hybrid lev 1'

    return 'v_wind_m_s01_10m_agl', 'VGRD:10 m above gnd'


def project_latlng_to_xy(latitudes_deg, longitudes_deg, projection_object=None,
                         model_name=None, grid_id=None):
    """Converts from lat-long to x-y coordinates (under model projection).

    P = number of points to convert

    :param latitudes_deg: length-P numpy array of latitudes (deg N).
    :param longitudes_deg: length-P numpy array of longitudes (deg E).
    :param projection_object: Projection object created by
        init_model_projection.  If projection_object = None, it will be created
        on the fly, based on args `model_name` and `grid_id`.
    :param model_name: Name of model.
    :param grid_id: ID for model grid.
    :return: x_coords_metres: length-P numpy array of x-coordinates.
    :return: y_coords_metres: length-P numpy array of y-coordinates.
    """

    false_easting_metres, false_northing_metres = (
        get_false_easting_and_northing(model_name, grid_id))
    if projection_object is None:
        projection_object = init_model_projection(model_name)

    return projections.project_latlng_to_xy(
        latitudes_deg, longitudes_deg, projection_object=projection_object,
        false_easting_metres=false_easting_metres,
        false_northing_metres=false_northing_metres)


def project_xy_to_latlng(x_coords_metres, y_coords_metres,
                         projection_object=None, model_name=None, grid_id=None):
    """Converts from x-y coordinates (under model projection) to lat-long.

    P = number of points to convert

    :param x_coords_metres: length-P numpy array of x-coordinates.
    :param y_coords_metres: length-P numpy array of y-coordinates.
    Projection object created by
        init_model_projection.  If projection_object = None, it will be created
        on the fly, based on args `model_name` and `grid_id`.
    :param model_name: Name of model.
    :param grid_id: ID for model grid.
    :return: latitudes_deg: length-P numpy array of latitudes (deg N).
    :return: longitudes_deg: length-P numpy array of longitudes (deg E).
    """

    false_easting_metres, false_northing_metres = (
        get_false_easting_and_northing(model_name, grid_id))
    if projection_object is None:
        projection_object = init_model_projection(model_name)

    return projections.project_xy_to_latlng(
        x_coords_metres, y_coords_metres, projection_object=projection_object,
        false_easting_metres=false_easting_metres,
        false_northing_metres=false_northing_metres)


def get_xy_grid_points(model_name, grid_id=None):
    """Returns unique x- and y-coordinates of grid points.

    M = number of rows (unique grid-point y-coordinates)
    N = number of columns (unique grid-point x-coordinates)

    :param model_name: Name of model.
    :param grid_id: ID for model grid.
    :return: grid_point_x_metres: length-N numpy array with x-coordinates of
        grid points.
    :return: grid_point_y_metres: length-M numpy array with y-coordinates of
        grid points.
    """

    x_spacing_metres, y_spacing_metres = get_xy_grid_spacing(
        model_name, grid_id)
    num_grid_rows, num_grid_columns = get_grid_dimensions(model_name, grid_id)

    return grids.get_xy_grid_points(
        x_min_metres=MIN_GRID_POINT_X_METRES,
        y_min_metres=MIN_GRID_POINT_Y_METRES, x_spacing_metres=x_spacing_metres,
        y_spacing_metres=y_spacing_metres, num_rows=num_grid_rows,
        num_columns=num_grid_columns)


def get_xy_grid_cell_edges(model_name, grid_id=None):
    """Returns unique x- and y-coordinates of grid-cell edges.

    M = number of rows (unique grid-point y-coordinates)
    N = number of columns (unique grid-point x-coordinates)

    :param model_name: Name of model.
    :param grid_id: ID for model grid.
    :return: grid_cell_edge_x_metres: length-(N + 1) numpy array with x-
        coordinates of grid-cell edges.
    :return: grid_cell_edge_y_metres: length-(M + 1) numpy array with y-
        coordinates of grid-cell edges.
    """

    x_spacing_metres, y_spacing_metres = get_xy_grid_spacing(
        model_name, grid_id)
    num_grid_rows, num_grid_columns = get_grid_dimensions(model_name, grid_id)

    return grids.get_xy_grid_cell_edges(
        x_min_metres=MIN_GRID_POINT_X_METRES,
        y_min_metres=MIN_GRID_POINT_Y_METRES, x_spacing_metres=x_spacing_metres,
        y_spacing_metres=y_spacing_metres, num_rows=num_grid_rows,
        num_columns=num_grid_columns)


def get_xy_grid_point_matrices(model_name, grid_id=None):
    """Returns matrices with x- and y-coordinates of grid points.

    M = number of rows (unique grid-point y-coordinates)
    N = number of columns (unique grid-point x-coordinates)

    :param model_name: Name of model.
    :param grid_id: ID for model grid.
    :return: grid_point_x_matrix_metres: M-by-N numpy array with x-coordinates
        of grid points (increasing to the right).
    :return: grid_point_y_matrix_metres: M-by-N numpy array with y-coordinates
        of grid points (increasing downward).
    """

    grid_point_x_metres, grid_point_y_metres = get_xy_grid_points(
        model_name, grid_id=grid_id)
    return grids.xy_vectors_to_matrices(grid_point_x_metres,
                                        grid_point_y_metres)


def get_xy_grid_cell_edge_matrices(model_name, grid_id=None):
    """Returns matrices with x- and y-coordinates of grid-cell edges.

    M = number of rows (unique grid-point y-coordinates)
    N = number of columns (unique grid-point x-coordinates)

    :param model_name: Name of model.
    :param grid_id: ID for model grid.
    :return: grid_cell_edge_x_matrix_metres: (M + 1)-by-(N + 1) numpy array with
        x-coordinates of grid-cell edges (increasing to the right).
    :return: grid_cell_edge_y_matrix_metres: (M + 1)-by-(N + 1) numpy array with
        y-coordinates of grid-cell edges (increasing downward).
    """

    grid_cell_edge_x_metres, grid_cell_edge_y_metres = get_xy_grid_cell_edges(
        model_name, grid_id=grid_id)
    return grids.xy_vectors_to_matrices(grid_cell_edge_x_metres,
                                        grid_cell_edge_y_metres)


def get_latlng_grid_point_matrices(model_name, grid_id=None,
                                   projection_object=None):
    """Returns matrices with latitudes and longitudes of grid points.

    M = number of rows (unique grid-point y-coordinates)
    N = number of columns (unique grid-point x-coordinates)

    :param model_name: Name of model.
    :param grid_id: ID for model grid.
    :param projection_object: Projection object created by
        init_model_projection.  If projection_object = None, it will be created
        on the fly, based on args `model_name` and `grid_id`.
    :return: grid_point_lat_matrix_deg: M-by-N numpy array of grid-point
        latitudes (generally increasing downward).
    :return: grid_point_lng_matrix_deg: M-by-N numpy array of grid-point
        longitudes (generally increasing to the right).
    """

    grid_point_x_matrix_metres, grid_point_y_matrix_metres = (
        get_xy_grid_point_matrices(model_name, grid_id=grid_id))
    return project_xy_to_latlng(
        grid_point_x_matrix_metres, grid_point_y_matrix_metres,
        projection_object=projection_object, model_name=model_name,
        grid_id=grid_id)


def get_latlng_grid_cell_edge_matrices(model_name, grid_id=None,
                                       projection_object=None):
    """Returns matrices with latitudes and longitudes of grid-cell edges.

    M = number of rows (unique grid-point y-coordinates)
    N = number of columns (unique grid-point x-coordinates)

    :param model_name: Name of model.
    :param grid_id: ID for model grid.
    :param projection_object: Projection object created by
        init_model_projection.  If projection_object = None, it will be created
        on the fly, based on args `model_name` and `grid_id`.
    :return: grid_cell_edge_lat_matrix_deg: (M + 1)-by-(N + 1) numpy array with
        latitudes of grid-cell edges (generally increasing downward).
    :return: grid_cell_edge_lat_matrix_deg: (M + 1)-by-(N + 1) numpy array with
        longitudes of grid-cell edges (generally increasing to the right).
    """

    grid_cell_edge_x_matrix_metres, grid_cell_edge_y_matrix_metres = (
        get_xy_grid_cell_edge_matrices(model_name, grid_id=grid_id))
    return project_xy_to_latlng(
        grid_cell_edge_x_matrix_metres, grid_cell_edge_y_matrix_metres,
        projection_object=projection_object, model_name=model_name,
        grid_id=grid_id)


def get_times_needed_for_interp(query_times_unix_sec=None,
                                model_time_step_hours=None, method_string=None):
    """Finds model times needed for interpolation to each range of query times.

    Q = number of query times
    M = number of model times needed

    :param query_times_unix_sec: length-Q numpy array of query times (Unix
        format).
    :param model_time_step_hours: Model time step.  If interpolating between
        forecast times (from the same initialization), this should be the
        model's time resolution (hours between successive forecasts).  If
        interpolating between model runs (forecasts for the same valid time but
        from different initializations), this should be the model's refresh time
        (hours between successive model runs).
    :param method_string: Interpolation method.  Valid options are "previous",
        "next", "linear", "nearest", "zero", "slinear", "quadratic", and
        "cubic".  The last 6 methods are described in the documentation for
        `scipy.interpolate.interp1d`.
    :return: model_times_unix_sec: length-M numpy array of model times needed
        (Unix format).
    :return: query_to_model_times_table: pandas DataFrame with the following
        columns.  Each row corresponds to one range of query times.
    query_to_model_times_table.min_query_time_unix_sec: Minimum query time for
        this range.
    query_to_model_times_table.max_query_time_unix_sec: Max query time for this
        range.
    query_to_model_times_table.model_times_unix_sec: 1-D numpy array of model
        times needed for this range.
    query_to_model_times_table.model_time_needed_flags: length-M numpy array of
        Boolean flags.  If model_time_needed_flags[i] = True at row j, this
        means the [i]th model time is needed for interp to the [j]th range of
        query times.
    """

    error_checking.assert_is_integer_numpy_array(query_times_unix_sec)
    error_checking.assert_is_numpy_array_without_nan(query_times_unix_sec)
    error_checking.assert_is_numpy_array(query_times_unix_sec, num_dimensions=1)
    error_checking.assert_is_integer(model_time_step_hours)
    error_checking.assert_is_string(method_string)

    model_time_step_sec = model_time_step_hours * HOURS_TO_SECONDS
    min_min_query_time_unix_sec = rounder.floor_to_nearest(
        float(numpy.min(query_times_unix_sec)), model_time_step_sec)
    max_max_query_time_unix_sec = rounder.ceiling_to_nearest(
        float(numpy.max(query_times_unix_sec)), model_time_step_sec)
    if max_max_query_time_unix_sec == min_min_query_time_unix_sec:
        max_max_query_time_unix_sec += model_time_step_sec

    num_ranges = int((max_max_query_time_unix_sec -
                      min_min_query_time_unix_sec) / model_time_step_sec)
    min_query_times_unix_sec = numpy.linspace(
        min_min_query_time_unix_sec,
        max_max_query_time_unix_sec - model_time_step_sec, num=num_ranges,
        dtype=int)
    max_query_times_unix_sec = numpy.linspace(
        min_min_query_time_unix_sec + model_time_step_sec,
        max_max_query_time_unix_sec, num=num_ranges, dtype=int)

    if method_string == PREVIOUS_INTERP_METHOD:
        min_model_time_unix_sec = copy.deepcopy(min_min_query_time_unix_sec)
        max_model_time_unix_sec = (
            max_max_query_time_unix_sec - model_time_step_sec)
    elif method_string == NEXT_INTERP_METHOD:
        min_model_time_unix_sec = (
            min_min_query_time_unix_sec + model_time_step_sec)
        max_model_time_unix_sec = copy.deepcopy(max_max_query_time_unix_sec)
    elif method_string in SUPERLINEAR_INTERP_METHODS:
        min_model_time_unix_sec = (
            min_min_query_time_unix_sec - model_time_step_sec)
        max_model_time_unix_sec = (
            max_max_query_time_unix_sec + model_time_step_sec)
    else:
        min_model_time_unix_sec = copy.deepcopy(min_min_query_time_unix_sec)
        max_model_time_unix_sec = copy.deepcopy(max_max_query_time_unix_sec)

    num_model_times = int((max_model_time_unix_sec -
                           min_model_time_unix_sec) / model_time_step_sec) + 1
    model_times_unix_sec = numpy.linspace(
        min_model_time_unix_sec, max_model_time_unix_sec, num=num_model_times,
        dtype=int)

    query_to_model_times_dict = {
        MIN_QUERY_TIME_COLUMN: min_query_times_unix_sec,
        MAX_QUERY_TIME_COLUMN: max_query_times_unix_sec}
    query_to_model_times_table = pandas.DataFrame.from_dict(
        query_to_model_times_dict)

    nested_array = query_to_model_times_table[[
        MIN_QUERY_TIME_COLUMN, MIN_QUERY_TIME_COLUMN]].values.tolist()
    argument_dict = {MODEL_TIMES_COLUMN: nested_array,
                     MODEL_TIMES_NEEDED_COLUMN: nested_array}
    query_to_model_times_table = query_to_model_times_table.assign(
        **argument_dict)

    for i in range(num_ranges):
        if method_string == PREVIOUS_INTERP_METHOD:
            these_model_times_unix_sec = numpy.array(
                [min_query_times_unix_sec[i]], dtype=int)
        elif method_string == NEXT_INTERP_METHOD:
            these_model_times_unix_sec = numpy.array(
                [max_query_times_unix_sec[i]], dtype=int)
        elif method_string in SUPERLINEAR_INTERP_METHODS:
            these_model_times_unix_sec = numpy.array(
                [min_query_times_unix_sec[i] - model_time_step_sec,
                 min_query_times_unix_sec[i], max_query_times_unix_sec[i],
                 max_query_times_unix_sec[i] + model_time_step_sec], dtype=int)
        else:
            these_model_times_unix_sec = numpy.array(
                [min_query_times_unix_sec[i], max_query_times_unix_sec[i]],
                dtype=int)

        query_to_model_times_table[
            MODEL_TIMES_COLUMN].values[i] = these_model_times_unix_sec
        query_to_model_times_table[MODEL_TIMES_NEEDED_COLUMN].values[i] = [
            t in these_model_times_unix_sec for t in model_times_unix_sec]

    return model_times_unix_sec, query_to_model_times_table


def get_wind_rotation_angles(latitudes_deg, longitudes_deg, model_name=None):
    """Returns wind-rotation angle for each lat-long point.

    These angles may be used in `rotate_winds`.

    :param latitudes_deg: numpy array of latitudes (deg N).
    :param longitudes_deg: equivalent-shape numpy array of longitudes (deg E).
    :param model_name: Name of model.
    :return: rotation_angle_cosines: equivalent-shape numpy array with cosines
        of rotation angles.
    :return: rotation_angle_sines: equivalent-shape numpy array with sines of
        rotation angles.
    """

    error_checking.assert_is_valid_lat_numpy_array(
        latitudes_deg, allow_nan=False)

    longitudes_deg = lng_conversion.convert_lng_positive_in_west(
        longitudes_deg, allow_nan=False)
    error_checking.assert_is_numpy_array(
        longitudes_deg, exact_dimensions=numpy.asarray(latitudes_deg.shape))

    standard_latitudes_deg, central_longitude_deg = get_projection_params(
        model_name)

    rotation_angles = (
        numpy.sin(standard_latitudes_deg[0] * DEGREES_TO_RADIANS) * (
            longitudes_deg - central_longitude_deg) * DEGREES_TO_RADIANS)
    return numpy.cos(rotation_angles), numpy.sin(rotation_angles)


def rotate_winds(u_winds_grid_relative_m_s01=None,
                 v_winds_grid_relative_m_s01=None, rotation_angle_cosines=None,
                 rotation_angle_sines=None):
    """Rotates wind vectors from grid-relative to Earth-relative.

    The equation is as follows, where alpha is the rotation angle.

    u_Earth = u_grid * cos(alpha) + v_grid * sin(alpha)
    v_Earth = v_grid * cos(alpha) - u_grid * sin(alpha)

    :param u_winds_grid_relative_m_s01: numpy array of grid-relative u-winds
        (towards positive x-direction).
    :param v_winds_grid_relative_m_s01: equivalent-shape numpy array of grid-
        relative v-winds (towards positive y-direction).
    :param rotation_angle_cosines: equivalent-shape numpy array with cosines of
        rotation angles.
    :param rotation_angle_sines: equivalent-shape numpy array with sines of
        rotation angles.
    :return: u_winds_earth_relative_m_s01: equivalent-shape numpy array of
        Earth-relative (northward) u-winds.
    :return: v_winds_earth_relative_m_s01: equivalent-shape numpy array of
        Earth-relative (eastward) v-winds.
    """

    error_checking.assert_is_real_numpy_array(u_winds_grid_relative_m_s01)
    array_dimensions = numpy.asarray(u_winds_grid_relative_m_s01.shape)

    error_checking.assert_is_real_numpy_array(v_winds_grid_relative_m_s01)
    error_checking.assert_is_numpy_array(
        v_winds_grid_relative_m_s01, exact_dimensions=array_dimensions)

    error_checking.assert_is_geq_numpy_array(rotation_angle_cosines, -1)
    error_checking.assert_is_leq_numpy_array(rotation_angle_cosines, 1)
    error_checking.assert_is_numpy_array(
        rotation_angle_cosines, exact_dimensions=array_dimensions)

    error_checking.assert_is_geq_numpy_array(rotation_angle_sines, -1)
    error_checking.assert_is_leq_numpy_array(rotation_angle_sines, 1)
    error_checking.assert_is_numpy_array(
        rotation_angle_sines, exact_dimensions=array_dimensions)

    u_winds_earth_relative_m_s01 = (
        rotation_angle_cosines * u_winds_grid_relative_m_s01 +
        rotation_angle_sines * v_winds_grid_relative_m_s01)
    v_winds_earth_relative_m_s01 = (
        rotation_angle_cosines * v_winds_grid_relative_m_s01 -
        rotation_angle_sines * u_winds_grid_relative_m_s01)
    return u_winds_earth_relative_m_s01, v_winds_earth_relative_m_s01
