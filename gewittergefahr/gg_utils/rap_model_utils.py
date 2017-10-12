"""Processing methods for data from RAP (Rapid Refresh) model.

DEFINITIONS

"Grid point" = center, as opposed to edge, of grid cell.
"""

import numpy
from gewittergefahr.gg_utils import grids
from gewittergefahr.gg_utils import projections
from gewittergefahr.gg_utils import nwp_model_utils
from gewittergefahr.gg_utils import error_checking

X_MIN_METRES = 0.
Y_MIN_METRES = 0.

ID_FOR_130GRID = '130'
NUM_ROWS_130GRID = 337
NUM_COLUMNS_130GRID = 451
X_SPACING_130GRID_METRES = 13545.
Y_SPACING_130GRID_METRES = 13545.
FALSE_EASTING_130GRID_METRES = 3332090.43552
FALSE_NORTHING_130GRID_METRES = -2279020.17888

ID_FOR_252GRID = '252'
NUM_ROWS_252GRID = 225
NUM_COLUMNS_252GRID = 301
X_SPACING_252GRID_METRES = 20317.625
Y_SPACING_252GRID_METRES = 20317.625
FALSE_EASTING_252GRID_METRES = 3332109.11198
FALSE_NORTHING_252GRID_METRES = -2279005.97252

NUM_ROWS_COLUMN = 'num_grid_rows'
NUM_COLUMNS_COLUMN = 'num_grid_columns'
X_SPACING_COLUMN = 'x_spacing_metres'
Y_SPACING_COLUMN = 'y_spacing_metres'
FALSE_EASTING_COLUMN = 'false_easting_metres'
FALSE_NORTHING_COLUMN = 'false_northing_metres'
WIND_ROTATION_FILE_NAME_COLUMN = 'wind_rotation_file_name'

MAIN_TEMPERATURE_COLUMN_ORIG = 'TMP'
MAIN_RH_COLUMN_ORIG = 'RH'
MAIN_GPH_COLUMN_ORIG = 'HGT'
MAIN_U_WIND_COLUMN_ORIG = 'UGRD'
MAIN_V_WIND_COLUMN_ORIG = 'VGRD'

MAIN_SOUNDING_COLUMNS = [
    nwp_model_utils.MAIN_TEMPERATURE_COLUMN, nwp_model_utils.MAIN_RH_COLUMN,
    nwp_model_utils.MAIN_GPH_COLUMN, nwp_model_utils.MAIN_U_WIND_COLUMN,
    nwp_model_utils.MAIN_V_WIND_COLUMN]
MAIN_SOUNDING_COLUMNS_ORIG = [
    MAIN_TEMPERATURE_COLUMN_ORIG, MAIN_RH_COLUMN_ORIG, MAIN_GPH_COLUMN_ORIG,
    MAIN_U_WIND_COLUMN_ORIG, MAIN_V_WIND_COLUMN_ORIG]

LOWEST_TEMPERATURE_COLUMN = 'temperature_kelvins_2m_agl'
LOWEST_RH_COLUMN = 'relative_humidity_2m_agl'
LOWEST_GPH_COLUMN = 'geopotential_height_metres_surface'
LOWEST_U_WIND_COLUMN = 'u_wind_m_s01_10m_agl'
LOWEST_V_WIND_COLUMN = 'v_wind_m_s01_10m_agl'

LOWEST_TEMPERATURE_COLUMN_ORIG = 'TMP:2 m above gnd'
LOWEST_RH_COLUMN_ORIG = 'RH:2 m above gnd'
LOWEST_GPH_COLUMN_ORIG = 'HGT:sfc'
LOWEST_U_WIND_COLUMN_ORIG = 'UGRD:10 m above gnd'
LOWEST_V_WIND_COLUMN_ORIG = 'VGRD:10 m above gnd'

PRESSURE_LEVELS_MB = numpy.linspace(100, 1000, num=37, dtype=int)
IS_WIND_EARTH_RELATIVE = False
WIND_ROTATION_FILE_NAME_130GRID = 'wind_rotation_angles_grid130.data'
WIND_ROTATION_FILE_NAME_252GRID = 'wind_rotation_angles_grid252.data'

# Projection parameters.
STANDARD_LATITUDES_DEG = numpy.array([25., 25.])
CENTRAL_LONGITUDE_DEG = 265.


def _check_grid_id(grid_id):
    """Ensures that grid ID is valid.

    :param grid_id: Grid ID (should be either "130" or "252").
    :raises: ValueError: grid_id is neither "130" nor "252".
    """

    error_checking.assert_is_string(grid_id)
    if grid_id not in [ID_FOR_130GRID, ID_FOR_252GRID]:
        error_string = (
            'grid_id should be either "' + ID_FOR_130GRID + '" or "' +
            ID_FOR_252GRID + '".  Instead, got "' + grid_id + '".')
        raise ValueError(error_string)


def _get_metadata_for_grid(grid_id=ID_FOR_130GRID):
    """Returns metadata for grid.

    :param grid_id: String ID for grid (either "130" or "252").
    :return: metadata_dict: Dictionary with the following keys.
    metadata_dict.num_grid_rows: Number of grid rows.
    metadata_dict.num_grid_columns: Number of grid columns.
    metadata_dict.x_spacing_metres: Spacing between adjacent grid points in x-
        direction.  Alternate interpretation: length of each grid cell in x-
        direction.
    metadata_dict.y_spacing_metres: Spacing between adjacent grid points in y-
        direction.  Alternate interpretation: length of each grid cell in y-
        direction.
    metadata_dict.false_easting_metres: See documentation for
        `projections.project_latlng_to_xy`.
    metadata_dict.false_northing_metres: See documentation for
        `projections.project_latlng_to_xy`.
    metadata_dict.wind_rotation_file_name: Path to file with wind-rotation
        angles.
    """

    _check_grid_id(grid_id)

    if grid_id == ID_FOR_130GRID:
        return {NUM_ROWS_COLUMN: NUM_ROWS_130GRID,
                NUM_COLUMNS_COLUMN: NUM_COLUMNS_130GRID,
                X_SPACING_COLUMN: X_SPACING_130GRID_METRES,
                Y_SPACING_COLUMN: Y_SPACING_130GRID_METRES,
                FALSE_EASTING_COLUMN: FALSE_EASTING_130GRID_METRES,
                FALSE_NORTHING_COLUMN: FALSE_NORTHING_130GRID_METRES,
                WIND_ROTATION_FILE_NAME_COLUMN: WIND_ROTATION_FILE_NAME_130GRID}

    return {NUM_ROWS_COLUMN: NUM_ROWS_252GRID,
            NUM_COLUMNS_COLUMN: NUM_COLUMNS_252GRID,
            X_SPACING_COLUMN: X_SPACING_252GRID_METRES,
            Y_SPACING_COLUMN: Y_SPACING_252GRID_METRES,
            FALSE_EASTING_COLUMN: FALSE_EASTING_252GRID_METRES,
            FALSE_NORTHING_COLUMN: FALSE_NORTHING_252GRID_METRES,
            WIND_ROTATION_FILE_NAME_COLUMN: WIND_ROTATION_FILE_NAME_252GRID}


def init_projection():
    """Initializes projection.

    :return: projection_object: object created by `pyproj.Proj`, specifying the
        projection.
    """

    return projections.init_lambert_conformal_projection(STANDARD_LATITUDES_DEG,
                                                         CENTRAL_LONGITUDE_DEG)


def read_wind_rotation_angles(grid_id=ID_FOR_130GRID):
    """Reads wind-rotation angles (one for each grid point).

    These angles are required to rotate winds from grid-relative (the native
    format) to Earth-relative.  The rotation is done by
    `nwp_model_utils.rotate_winds`.

    M = number of grid rows
    N = number of grid columns

    :param grid_id: String ID for grid (either "130" or "252").
    :return: cosine_matrix: M-by-N numpy array with cosine of rotation angle.
        x-coordinate increases while moving right across a row, and y-coordinate
        increases while moving down a column.
    :return: sine_matrix: Same as above, but with sines rather than cosines.
    """

    grid_metadata_dict = _get_metadata_for_grid(grid_id)
    (cosine_vector, sine_vector) = numpy.loadtxt(
        grid_metadata_dict[WIND_ROTATION_FILE_NAME_COLUMN], unpack=True)

    cosine_matrix = numpy.reshape(
        cosine_vector, (grid_metadata_dict[NUM_ROWS_COLUMN],
                        grid_metadata_dict[NUM_COLUMNS_COLUMN]))
    sine_matrix = numpy.reshape(
        sine_vector, (grid_metadata_dict[NUM_ROWS_COLUMN],
                      grid_metadata_dict[NUM_COLUMNS_COLUMN]))

    return cosine_matrix, sine_matrix


def project_latlng_to_xy(latitudes_deg, longitudes_deg, grid_id=ID_FOR_130GRID,
                         projection_object=None):
    """Converts from lat-long to projection (x-y) coordinates.

    P = number of points

    :param latitudes_deg: length-P numpy array of latitudes (deg N).
    :param longitudes_deg: length-P numpy array of longitudes (deg E).
    :param grid_id: String ID for grid (either "130" or "252").
    :param projection_object: Projection object created by `pyproj.Proj`.  If
        None, this will be created on the fly, using
        `narr_utils.init_projection`.
    :return: x_coords_metres: length-P numpy array of x-coordinates.
    :return: y_coords_metres: length-P numpy array of y-coordinates.
    """

    grid_metadata_dict = _get_metadata_for_grid(grid_id)

    if projection_object is None:
        projection_object = init_projection()

    return projections.project_latlng_to_xy(
        latitudes_deg, longitudes_deg, projection_object=projection_object,
        false_easting_metres=grid_metadata_dict[FALSE_EASTING_COLUMN],
        false_northing_metres=grid_metadata_dict[FALSE_NORTHING_COLUMN])


def project_xy_to_latlng(x_coords_metres, y_coords_metres,
                         grid_id=ID_FOR_130GRID, projection_object=None):
    """Converts from projection (x-y) to lat-long coordinates.

    P = number of points

    :param x_coords_metres: length-P numpy array of x-coordinates.
    :param y_coords_metres: length-P numpy array of y-coordinates.
    :param grid_id: String ID for grid (either "130" or "252").
    :param projection_object: Projection object created by `pyproj.Proj`.  If
        None, this will be created on the fly, using
        `narr_utils.init_projection`.
    :return: latitudes_deg: length-P numpy array of latitudes (deg N).
    :return: longitudes_deg: length-P numpy array of longitudes (deg E).
    """

    grid_metadata_dict = _get_metadata_for_grid(grid_id)

    if projection_object is None:
        projection_object = init_projection()

    return projections.project_xy_to_latlng(
        x_coords_metres, y_coords_metres, projection_object=projection_object,
        false_easting_metres=grid_metadata_dict[FALSE_EASTING_COLUMN],
        false_northing_metres=grid_metadata_dict[FALSE_NORTHING_COLUMN])


def get_xy_grid_points_unique(grid_id=ID_FOR_130GRID):
    """Generates unique x- and y-coordinates of grid points.

    M = number of grid rows
    N = number of grid columns

    :param grid_id: String ID for grid (either "130" or "252").
    :return: grid_point_x_metres: length-N numpy array with x-coordinates of
        grid points.
    :return: grid_point_y_metres: length-M numpy array with y-coordinates of
        grid points.
    """

    grid_metadata_dict = _get_metadata_for_grid(grid_id)

    return grids.get_xy_grid_points(
        x_min_metres=X_MIN_METRES, y_min_metres=Y_MIN_METRES,
        x_spacing_metres=grid_metadata_dict[X_SPACING_COLUMN],
        y_spacing_metres=grid_metadata_dict[Y_SPACING_COLUMN],
        num_rows=grid_metadata_dict[NUM_ROWS_COLUMN],
        num_columns=grid_metadata_dict[NUM_COLUMNS_COLUMN])


def get_xy_grid_cell_edges_unique(grid_id=ID_FOR_130GRID):
    """Generates unique x- and y-coordinates of grid-cell edges.

    M = number of grid rows
    N = number of grid columns

    :param grid_id: String ID for grid (either "130" or "252").
    :return: grid_cell_edge_x_metres: length-(N + 1) numpy array with x-
        coordinates of grid-cell edges.
    :return: grid_cell_edge_y_metres: length-(M + 1) numpy array with y-
        coordinates of grid-cell edges.
    """

    grid_metadata_dict = _get_metadata_for_grid(grid_id)

    return grids.get_xy_grid_cell_edges(
        x_min_metres=X_MIN_METRES, y_min_metres=Y_MIN_METRES,
        x_spacing_metres=grid_metadata_dict[X_SPACING_COLUMN],
        y_spacing_metres=grid_metadata_dict[Y_SPACING_COLUMN],
        num_rows=grid_metadata_dict[NUM_ROWS_COLUMN],
        num_columns=grid_metadata_dict[NUM_COLUMNS_COLUMN])


def get_xy_grid_point_matrices(grid_id=ID_FOR_130GRID):
    """Generates matrices with x- and y-coordinates of grid points.

    M = number of grid rows
    N = number of grid columns

    :param grid_id: String ID for grid (either "130" or "252").
    :return: grid_point_x_matrix_metres: M-by-N numpy array of x-coordinates.
        x increases while traveling to the right along a row, and each row is
        the same.
    :return: grid_point_y_matrix_metres: M-by-N numpy array of y-coordinates.  y
        increases while traveling down a column, and each column is the same.
    """

    (grid_point_x_unique_metres,
     grid_point_y_unique_metres) = get_xy_grid_points_unique(grid_id=grid_id)
    return grids.xy_vectors_to_matrices(grid_point_x_unique_metres,
                                        grid_point_y_unique_metres)


def get_xy_grid_cell_edge_matrices(grid_id=ID_FOR_130GRID):
    """Generates matrices with x- and y-coordinates of grid-cell edges.

    M = number of grid rows
    N = number of grid columns

    :param grid_id: String ID for grid (either "130" or "252").
    :return: grid_cell_edge_x_matrix_metres: (M + 1)-by-(N + 1) numpy array of
        x-coordinates.  x increases while traveling to the right along a row,
        and each row is the same.
    :return: grid_cell_edge_y_matrix_metres: (M + 1)-by-(N + 1) numpy array of
        y-coordinates.  y increases while traveling down a column, and each
        column is the same.
    """

    (grid_cell_edge_x_unique_metres,
     grid_cell_edge_y_unique_metres) = get_xy_grid_cell_edges_unique(
         grid_id=grid_id)
    return grids.xy_vectors_to_matrices(grid_cell_edge_x_unique_metres,
                                        grid_cell_edge_y_unique_metres)


def get_latlng_grid_point_matrices(grid_id=ID_FOR_130GRID,
                                   projection_object=None):
    """Generates matrices with lat and long coordinates of grid points.

    M = number of grid rows
    N = number of grid columns

    :param grid_id: String ID for grid (either "130" or "252").
    :param projection_object: See documentation for project_xy_to_latlng.
    :return: grid_point_lat_matrix_deg: M-by-N numpy array of latitudes (deg N).
        In general, latitude increases while traveling down the columns.
        However, each column is different.
    :return: grid_point_lng_matrix_deg: M-by-N numpy array of longitudes (deg
        E).  In general, longitude increases while traveling right along the
        rows.  However, each row is different.
    """

    (grid_point_x_matrix_metres,
     grid_point_y_matrix_metres) = get_xy_grid_point_matrices(grid_id=grid_id)

    return project_xy_to_latlng(grid_point_x_matrix_metres,
                                grid_point_y_matrix_metres,
                                grid_id=grid_id,
                                projection_object=projection_object)


def get_latlng_grid_cell_edge_matrices(grid_id=ID_FOR_130GRID,
                                       projection_object=None):
    """Generates matrices with lat and long coordinates of grid-cell edges.

    M = number of grid rows
    N = number of grid columns

    :param grid_id: String ID for grid (either "130" or "252").
    :param projection_object: See documentation for project_xy_to_latlng.
    :return: grid_cell_edge_lat_matrix_deg: (M + 1)-by-(N + 1) numpy array of
        latitudes (deg N).  In general, latitude increases while traveling down
        the columns. However, each column is different.
    :return: grid_cell_edge_lng_matrix_deg: (M + 1)-by-(N + 1) numpy array of
        longitudes (deg E).  In general, longitude increases while traveling
        right along the rows.  However, each row is different.
    """

    (grid_cell_edge_x_matrix_metres,
     grid_cell_edge_y_matrix_metres) = get_xy_grid_cell_edge_matrices(
         grid_id=grid_id)

    return project_xy_to_latlng(grid_cell_edge_x_matrix_metres,
                                grid_cell_edge_y_matrix_metres,
                                grid_id=grid_id,
                                projection_object=projection_object)
