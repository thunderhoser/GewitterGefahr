"""IO methods for probSevere storm-tracking."""

import json
import numpy
import pandas
from gewittergefahr.gg_io import myrorss_io
from gewittergefahr.gg_io import segmotion_io
from gewittergefahr.gg_utils import polygons
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import error_checking

# TODO(thunderhoser): add file-management code (will include transferring
# raw files from NSSL machine to local machine).
# TODO(thunderhoser): replace main method with named method.

TIME_COLUMN_ORIG = 'validTime'
FEATURES_COLUMN_ORIG = 'features'
GEOMETRY_COLUMN_ORIG = 'geometry'
COORDINATES_COLUMN_ORIG = 'coordinates'
PROPERTIES_COLUMN_ORIG = 'properties'

STORM_ID_COLUMN_ORIG = 'ID'
EAST_VELOCITY_COLUMN_ORIG = 'MOTION_EAST'
NORTH_VELOCITY_COLUMN_ORIG = 'MOTION_SOUTH'
LAT_COLUMN_INDEX_ORIG = 1
LNG_COLUMN_INDEX_ORIG = 0

NON_POLYGON_COLUMNS = [
    segmotion_io.STORM_ID_COLUMN, segmotion_io.EAST_VELOCITY_COLUMN,
    segmotion_io.NORTH_VELOCITY_COLUMN, segmotion_io.TIME_COLUMN]

NW_GRID_POINT_LAT_DEG = 55.
NW_GRID_POINT_LNG_DEG = 230.
GRID_LAT_SPACING_DEG = 0.01
GRID_LNG_SPACING_DEG = 0.01

TIME_FORMAT_ORIG = '%Y%m%d_%H%M%S UTC'

# The following constants are used only in the main method.
MIN_BUFFER_DISTS_METRES = numpy.array([numpy.nan, 0., 5000.])
MAX_BUFFER_DISTS_METRES = numpy.array([0., 5000., 10000.])

RAW_FILE_NAME = (
    '/localdata/ryan.lagerquist/software/matlab/wdssii/raw_files/segmotion/'
    'probSevere/20170629/2017-06-29-122639_probSevere_0050.00.json')
PROCESSED_FILE_NAME = (
    '/localdata/ryan.lagerquist/gewittergefahr_junk/probSevere/'
    'probSevere_2017-06-29-122639.p')


def _remove_rows_with_nan(input_table):
    """Removes any row with NaN from pandas DataFrame.

    :param input_table: pandas DataFrame.
    :return: output_table: Same as input_table, except that some rows may be
        gone.
    """

    return input_table.loc[input_table.notnull().all(axis=1)]


def read_storm_objects_from_raw_file(json_file_name):
    """Reads storm objects from raw file.

    This file should contain all storm objects for one tracking scale and one
    time step.

    P = number of grid points in given storm object
    V = number of vertices in bounding polygon of given storm object

    :param json_file_name: Path to input file.
    :return: storm_object_table: pandas DataFrame with the following columns.
    storm_object_table.storm_id: String ID for storm cell.
    storm_object_table.east_velocity_m_s01: Eastward velocity (m/s).
    storm_object_table.north_velocity_m_s01: Northward velocity (m/s).
    storm_object_table.unix_time_sec: Time in Unix format.
    storm_object_table.centroid_lat_deg: Latitude at centroid of storm object
        (deg N).
    storm_object_table.centroid_lng_deg: Longitude at centroid of storm object
        (deg E).
    storm_object_table.grid_point_latitudes_deg: length-P numpy array with
        latitudes (deg N) of grid points in polygon.
    storm_object_table.grid_point_longitudes_deg: length-P numpy array with
        longitudes (deg E) of grid points in polygon.
    storm_object_table.grid_point_rows: length-P numpy array with row indices
        (integers) of grid points in polygon.
    storm_object_table.grid_point_columns: length-P numpy array with column
        indices (integers) of grid points in polygon.
    storm_object_table.vertex_latitudes_deg: length-V numpy array with latitudes
        (deg N) of vertices in polygon.
    storm_object_table.vertex_longitudes_deg: length-V numpy array with
        longitudes (deg E) of vertices in polygon.
    storm_object_table.vertex_rows: length-V numpy array with row indices (half-
        integers) of vertices in polygon.
    storm_object_table.vertex_columns: length-V numpy array with column indices
        (half-integers) of vertices in polygon.
    """

    error_checking.assert_file_exists(json_file_name)
    with open(json_file_name) as json_file_handle:
        probsevere_dict = json.load(json_file_handle)

    unix_time_sec = time_conversion.string_to_unix_sec(
        probsevere_dict[TIME_COLUMN_ORIG].encode('ascii', 'ignore'),
        TIME_FORMAT_ORIG)
    num_storms = len(probsevere_dict[FEATURES_COLUMN_ORIG])
    unix_times_sec = numpy.full(num_storms, unix_time_sec, dtype=int)

    storm_ids = [None] * num_storms
    east_velocities_m_s01 = numpy.full(num_storms, numpy.nan)
    north_velocities_m_s01 = numpy.full(num_storms, numpy.nan)

    for i in range(num_storms):
        storm_ids[i] = str(
            probsevere_dict[FEATURES_COLUMN_ORIG][i][PROPERTIES_COLUMN_ORIG][
                STORM_ID_COLUMN_ORIG])
        east_velocities_m_s01[i] = float(
            probsevere_dict[FEATURES_COLUMN_ORIG][i][PROPERTIES_COLUMN_ORIG][
                EAST_VELOCITY_COLUMN_ORIG])
        north_velocities_m_s01[i] = -1 * float(
            probsevere_dict[FEATURES_COLUMN_ORIG][i][PROPERTIES_COLUMN_ORIG][
                NORTH_VELOCITY_COLUMN_ORIG])

    storm_object_dict = {
        segmotion_io.STORM_ID_COLUMN: storm_ids,
        segmotion_io.EAST_VELOCITY_COLUMN: east_velocities_m_s01,
        segmotion_io.NORTH_VELOCITY_COLUMN: north_velocities_m_s01,
        segmotion_io.TIME_COLUMN: unix_times_sec}
    storm_object_table = pandas.DataFrame.from_dict(storm_object_dict)
    storm_object_table = _remove_rows_with_nan(storm_object_table)

    simple_array = numpy.full(num_storms, numpy.nan)
    nested_array = storm_object_table[[
        segmotion_io.STORM_ID_COLUMN,
        segmotion_io.STORM_ID_COLUMN]].values.tolist()

    argument_dict = {segmotion_io.CENTROID_LAT_COLUMN: simple_array,
                     segmotion_io.CENTROID_LNG_COLUMN: simple_array,
                     segmotion_io.GRID_POINT_LAT_COLUMN: nested_array,
                     segmotion_io.GRID_POINT_LNG_COLUMN: nested_array,
                     segmotion_io.GRID_POINT_ROW_COLUMN: nested_array,
                     segmotion_io.GRID_POINT_COLUMN_COLUMN: nested_array,
                     segmotion_io.VERTEX_LAT_COLUMN: nested_array,
                     segmotion_io.VERTEX_LNG_COLUMN: nested_array,
                     segmotion_io.VERTEX_ROW_COLUMN: nested_array,
                     segmotion_io.VERTEX_COLUMN_COLUMN: nested_array}
    storm_object_table = storm_object_table.assign(**argument_dict)

    for i in range(num_storms):
        this_vertex_matrix_deg = numpy.asarray(
            probsevere_dict[FEATURES_COLUMN_ORIG][i][GEOMETRY_COLUMN_ORIG][
                COORDINATES_COLUMN_ORIG][0])
        these_vertex_lat_deg = this_vertex_matrix_deg[:, LAT_COLUMN_INDEX_ORIG]
        these_vertex_lng_deg = this_vertex_matrix_deg[:, LNG_COLUMN_INDEX_ORIG]

        (these_vertex_rows, these_vertex_columns) = myrorss_io.latlng_to_rowcol(
            these_vertex_lat_deg, these_vertex_lng_deg,
            nw_grid_point_lat_deg=NW_GRID_POINT_LAT_DEG,
            nw_grid_point_lng_deg=NW_GRID_POINT_LNG_DEG,
            lat_spacing_deg=GRID_LAT_SPACING_DEG,
            lng_spacing_deg=GRID_LNG_SPACING_DEG)

        (storm_object_table[segmotion_io.VERTEX_ROW_COLUMN].values[i],
         storm_object_table[segmotion_io.VERTEX_COLUMN_COLUMN].values[i]) = (
            polygons.fix_probsevere_vertices(these_vertex_rows,
                                             these_vertex_columns))

        (storm_object_table[segmotion_io.VERTEX_LAT_COLUMN].values[i],
         storm_object_table[segmotion_io.VERTEX_LNG_COLUMN].values[i]) = (
            myrorss_io.rowcol_to_latlng(
                storm_object_table[segmotion_io.VERTEX_ROW_COLUMN].values[i],
                storm_object_table[segmotion_io.VERTEX_COLUMN_COLUMN].values[i],
                nw_grid_point_lat_deg=NW_GRID_POINT_LAT_DEG,
                nw_grid_point_lng_deg=NW_GRID_POINT_LNG_DEG,
                lat_spacing_deg=GRID_LAT_SPACING_DEG,
                lng_spacing_deg=GRID_LNG_SPACING_DEG))

        (storm_object_table[
             segmotion_io.GRID_POINT_ROW_COLUMN].values[i],
         storm_object_table[
             segmotion_io.GRID_POINT_COLUMN_COLUMN].values[i]) = (
            polygons.simple_polygon_to_grid_points(
                storm_object_table[segmotion_io.VERTEX_ROW_COLUMN].values[i],
                storm_object_table[
                    segmotion_io.VERTEX_COLUMN_COLUMN].values[i]))

        (storm_object_table[segmotion_io.GRID_POINT_LAT_COLUMN].values[i],
         storm_object_table[segmotion_io.GRID_POINT_LNG_COLUMN].values[i]) = (
            myrorss_io.rowcol_to_latlng(
                storm_object_table[
                    segmotion_io.GRID_POINT_ROW_COLUMN].values[i],
                storm_object_table[
                    segmotion_io.GRID_POINT_COLUMN_COLUMN].values[i],
                nw_grid_point_lat_deg=NW_GRID_POINT_LAT_DEG,
                nw_grid_point_lng_deg=NW_GRID_POINT_LNG_DEG,
                lat_spacing_deg=GRID_LAT_SPACING_DEG,
                lng_spacing_deg=GRID_LNG_SPACING_DEG))

        (storm_object_table[segmotion_io.CENTROID_LAT_COLUMN].values[i],
         storm_object_table[segmotion_io.CENTROID_LNG_COLUMN].values[i]) = (
            polygons.get_latlng_centroid(
                storm_object_table[segmotion_io.VERTEX_LAT_COLUMN].values[i],
                storm_object_table[segmotion_io.VERTEX_LNG_COLUMN].values[i]))

    return storm_object_table


if __name__ == '__main__':
    STORM_OBJECT_TABLE = read_storm_objects_from_raw_file(RAW_FILE_NAME)
    print STORM_OBJECT_TABLE
