"""Processing methods for NARR (North American Regional Reanalysis) data.

DEFINITIONS

"Grid point" = center, as opposed to edge, of grid cell.
"""

import numpy
from gewittergefahr.gg_utils import grids
from gewittergefahr.gg_utils import projections
from gewittergefahr.gg_utils import nwp_model_utils

X_MIN_METRES = 0.
Y_MIN_METRES = 0.
X_SPACING_METRES = 32463.
Y_SPACING_METRES = 32463.
NUM_GRID_ROWS = 277
NUM_GRID_COLUMNS = 349

TOP_PRESSURE_LEVELS_MB = numpy.linspace(100., 300., num=9)
MID_PRESSURE_LEVELS_MB = numpy.linspace(350., 700., num=8)
BOTTOM_PRESSURE_LEVELS_MB = numpy.linspace(725., 1000., num=12)
PRESSURE_LEVELS_MB = numpy.concatenate((
    TOP_PRESSURE_LEVELS_MB, MID_PRESSURE_LEVELS_MB, BOTTOM_PRESSURE_LEVELS_MB))

IS_WIND_EARTH_RELATIVE = True

MAIN_TEMPERATURE_COLUMN_ORIG = 'TMP'
MAIN_SPFH_COLUMN_ORIG = 'SPFH'
MAIN_GPH_COLUMN_ORIG = 'HGT'
MAIN_U_WIND_COLUMN_ORIG = 'UGRD'
MAIN_V_WIND_COLUMN_ORIG = 'VGRD'

MAIN_SOUNDING_COLUMNS = [
    nwp_model_utils.MAIN_TEMPERATURE_COLUMN, nwp_model_utils.MAIN_SPFH_COLUMN,
    nwp_model_utils.MAIN_GPH_COLUMN, nwp_model_utils.MAIN_U_WIND_COLUMN,
    nwp_model_utils.MAIN_V_WIND_COLUMN]
MAIN_SOUNDING_COLUMNS_ORIG = [
    MAIN_TEMPERATURE_COLUMN_ORIG, MAIN_SPFH_COLUMN_ORIG, MAIN_GPH_COLUMN_ORIG,
    MAIN_U_WIND_COLUMN_ORIG, MAIN_V_WIND_COLUMN_ORIG]

LOWEST_TEMPERATURE_COLUMN = 'temperature_kelvins_2m_agl'
LOWEST_SPFH_COLUMN = 'specific_humidity_2m_agl'
LOWEST_U_WIND_COLUMN = 'u_wind_m_s01_10m_agl'
LOWEST_V_WIND_COLUMN = 'v_wind_m_s01_10m_agl'

LOWEST_TEMPERATURE_COLUMN_ORIG = 'TMP:2 m above gnd'
LOWEST_SPFH_COLUMN_ORIG = 'SPFH:2 m above gnd'
LOWEST_U_WIND_COLUMN_ORIG = 'UGRD:10 m above gnd'
LOWEST_V_WIND_COLUMN_ORIG = 'VGRD:10 m above gnd'

# Projection parameters.
STANDARD_LATITUDES_DEG = numpy.array([50., 50.])
CENTRAL_LONGITUDE_DEG = 253.
FALSE_EASTING_METRES = 5632564.44008
FALSE_NORTHING_METRES = -1636427.98024


def init_projection():
    """Initializes projection.

    :return: projection_object: object created by `pyproj.Proj`, specifying the
        projection.
    """

    return projections.init_lambert_conformal_projection(STANDARD_LATITUDES_DEG,
                                                         CENTRAL_LONGITUDE_DEG)


def project_latlng_to_xy(latitudes_deg, longitudes_deg, projection_object=None):
    """Converts from lat-long to projection (x-y) coordinates.

    P = number of points

    :param latitudes_deg: length-P numpy array of latitudes (deg N).
    :param longitudes_deg: length-P numpy array of longitudes (deg E).
    :param projection_object: Projection object created by `pyproj.Proj`.  If
        None, this will be created on the fly, using
        `narr_utils.init_projection`.
    :return: x_coords_metres: length-P numpy array of x-coordinates.
    :return: y_coords_metres: length-P numpy array of y-coordinates.
    """

    if projection_object is None:
        projection_object = init_projection()

    return projections.project_latlng_to_xy(
        latitudes_deg, longitudes_deg, projection_object=projection_object,
        false_easting_metres=FALSE_EASTING_METRES,
        false_northing_metres=FALSE_NORTHING_METRES)


def project_xy_to_latlng(x_coords_metres, y_coords_metres,
                         projection_object=None):
    """Converts from projection (x-y) to lat-long coordinates.

    P = number of points

    :param x_coords_metres: length-P numpy array of x-coordinates.
    :param y_coords_metres: length-P numpy array of y-coordinates.
    :param projection_object: Projection object created by `pyproj.Proj`.  If
        None, this will be created on the fly, using
        `narr_utils.init_projection`.
    :return: latitudes_deg: length-P numpy array of latitudes (deg N).
    :return: longitudes_deg: length-P numpy array of longitudes (deg E).
    """

    if projection_object is None:
        projection_object = init_projection()

    return projections.project_xy_to_latlng(
        x_coords_metres, y_coords_metres, projection_object=projection_object,
        false_easting_metres=FALSE_EASTING_METRES,
        false_northing_metres=FALSE_NORTHING_METRES)


def get_xy_grid_points_unique():
    """Generates unique x- and y-coordinates of grid points.

    :return: grid_point_x_metres: length-349 numpy array with x-coordinates of
        grid points.
    :return: grid_point_y_metres: length-277 numpy array with y-coordinates of
        grid points.
    """

    return grids.get_xy_grid_points(
        x_min_metres=X_MIN_METRES, y_min_metres=Y_MIN_METRES,
        x_spacing_metres=X_SPACING_METRES, y_spacing_metres=Y_SPACING_METRES,
        num_rows=NUM_GRID_ROWS, num_columns=NUM_GRID_COLUMNS)


def get_xy_grid_cell_edges_unique():
    """Generates unique x- and y-coordinates of grid-cell edges.

    :return: grid_cell_edge_x_metres: length-350 numpy array with x-coordinates
        of grid-cell edges.
    :return: grid_cell_edge_y_metres: length-278 numpy array with y-coordinates
        of grid-cell edges.
    """

    return grids.get_xy_grid_cell_edges(
        x_min_metres=X_MIN_METRES, y_min_metres=Y_MIN_METRES,
        x_spacing_metres=X_SPACING_METRES, y_spacing_metres=Y_SPACING_METRES,
        num_rows=NUM_GRID_ROWS, num_columns=NUM_GRID_COLUMNS)


def get_xy_grid_point_matrices():
    """Generates matrices with x- and y-coordinates of grid points.

    :return: grid_point_x_matrix_metres: 277-by-349 numpy array of x-
        coordinates.  x increases while traveling to the right along a row, and
        each row is the same.
    :return: grid_point_y_matrix_metres: 277-by-349 numpy array of y-
        coordinates.  y increases while traveling down a column, and each column
        is the same.
    """

    (grid_point_x_unique_metres,
     grid_point_y_unique_metres) = get_xy_grid_points_unique()
    return grids.xy_vectors_to_matrices(grid_point_x_unique_metres,
                                        grid_point_y_unique_metres)


def get_xy_grid_cell_edge_matrices():
    """Generates matrices with x- and y-coordinates of grid-cell edges.

    :return: grid_cell_edge_x_matrix_metres: 278-by-350 numpy array of x-
        coordinates.  x increases while traveling to the right along a row, and
        each row is the same.
    :return: grid_cell_edge_y_matrix_metres: 278-by-350 numpy array of y-
        coordinates.  y increases while traveling down a column, and each column
        is the same.
    """

    (grid_cell_edge_x_unique_metres,
     grid_cell_edge_y_unique_metres) = get_xy_grid_cell_edges_unique()
    return grids.xy_vectors_to_matrices(grid_cell_edge_x_unique_metres,
                                        grid_cell_edge_y_unique_metres)


def get_latlng_grid_point_matrices(projection_object=None):
    """Generates matrices with lat and long coordinates of grid points.

    :param projection_object: See documentation for project_xy_to_latlng.
    :return: grid_point_lat_matrix_deg: 277-by-349 numpy array of latitudes (deg
        N).  In general, latitude increases while traveling down the columns.
        However, each column is different.
    :return: grid_point_lng_matrix_deg: 277-by-349 numpy array of longitudes
        (deg E).  In general, longitude increases while traveling right along
        the rows.  However, each row is different.
    """

    (grid_point_x_matrix_metres,
     grid_point_y_matrix_metres) = get_xy_grid_point_matrices()

    return project_xy_to_latlng(grid_point_x_matrix_metres,
                                grid_point_y_matrix_metres,
                                projection_object=projection_object)


def get_latlng_grid_cell_edge_matrices(projection_object=None):
    """Generates matrices with lat and long coordinates of grid-cell edges.

    :param projection_object: See documentation for project_xy_to_latlng.
    :return: grid_cell_edge_lat_matrix_deg: 278-by-350 numpy array of latitudes
        (deg N).  In general, latitude increases while traveling down the
        columns. However, each column is different.
    :return: grid_cell_edge_lng_matrix_deg: 278-by-350 numpy array of longitudes
        (deg E).  In general, longitude increases while traveling right along
        the rows.  However, each row is different.
    """

    (grid_cell_edge_x_matrix_metres,
     grid_cell_edge_y_matrix_metres) = get_xy_grid_cell_edge_matrices()

    return project_xy_to_latlng(grid_cell_edge_x_matrix_metres,
                                grid_cell_edge_y_matrix_metres,
                                projection_object=projection_object)
