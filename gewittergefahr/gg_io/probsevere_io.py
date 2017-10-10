"""IO methods for probSevere storm-tracking."""

import json
import os.path
import numpy
import pandas
from gewittergefahr.gg_io import radar_io
from gewittergefahr.gg_io import storm_tracking_io as tracking_io
from gewittergefahr.gg_utils import polygons
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import error_checking

# TODO(thunderhoser): add file-management code (will include transferring
# raw files from NSSL machine to local machine).
# TODO(thunderhoser): replace main method with named method.

RAW_FILE_PREFIX = 'SSEC_AWIPS_PROBSEVERE'
RAW_FILE_EXTENSION = '.json'
TIME_FORMAT_IN_RAW_FILE_NAMES = '%Y%m%d_%H%M%S'

TIME_FORMAT_DATE = '%Y%m%d'
TIME_FORMAT_IN_RAW_FILES = '%Y%m%d_%H%M%S UTC'

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
    tracking_io.STORM_ID_COLUMN, tracking_io.EAST_VELOCITY_COLUMN,
    tracking_io.NORTH_VELOCITY_COLUMN, tracking_io.TIME_COLUMN]

NW_GRID_POINT_LAT_DEG = 55.
NW_GRID_POINT_LNG_DEG = 230.
GRID_LAT_SPACING_DEG = 0.01
GRID_LNG_SPACING_DEG = 0.01
NUM_GRID_ROWS = 3501
NUM_GRID_COLUMNS = 7001

# The following constants are used only in the main method.
MIN_BUFFER_DISTS_METRES = numpy.array([numpy.nan, 0., 5000.])
MAX_BUFFER_DISTS_METRES = numpy.array([0., 5000., 10000.])

UNIX_TIME_SEC = 1498739199
TRACKING_SCALE_METRES2 = 4e7

RAW_FILE_NAME = (
    '/localdata/ryan.lagerquist/software/matlab/wdssii/raw_files/segmotion/'
    'probSevere/20170629/2017-06-29-122639_probSevere_0050.00.json')
TOP_PROCESSED_DIR_NAME = (
    '/localdata/ryan.lagerquist/gewittergefahr_junk/probSevere')


def _get_pathless_raw_file_name(unix_time_sec):
    """Generates pathless name for raw file.

    This file should contain all storm objects for one tracking scale and one
    time step.

    :param unix_time_sec: Time in Unix format.
    :return: pathless_raw_file_name: Pathless name for raw file.
    """

    return '{0:s}_{1:s}{2:s}'.format(
        RAW_FILE_PREFIX, time_conversion.unix_sec_to_string(
            unix_time_sec, TIME_FORMAT_IN_RAW_FILE_NAMES),
        RAW_FILE_EXTENSION)


def get_raw_file_name_on_ftp(unix_time_sec, top_ftp_directory_name):
    """Generates name of raw file on FTP server.

    :param unix_time_sec: Time in Unix format.
    :param top_ftp_directory_name: Top-level directory with raw files on FTP
        server.  This should be only the directory, not including user name and
        host name.  Example: "/data/storm_tracking/probSevere".
    :return: raw_ftp_file_name: Expected path on FTP server.
    """

    pathless_file_name = _get_pathless_raw_file_name(unix_time_sec)
    return '{0:s}/{1:s}'.format(top_ftp_directory_name, pathless_file_name)


def find_raw_file_on_local_machine(unix_time_sec=None,
                                   top_local_directory_name=None,
                                   raise_error_if_missing=True):
    """Generates name of raw file on local machine.

    :param unix_time_sec: Time in Unix format.
    :param top_local_directory_name: Top-level directory with raw files on local
        machine.
    :param raise_error_if_missing: Boolean flag.  If raise_error_if_missing =
        True and file is missing, will raise error.
    :return: raw_local_file_name: Path on local machine.  If
        raise_error_if_missing = False and file is missing, this will be the
        *expected* path.
    :raises: ValueError: if raise_error_if_missing = True and file is missing.
    """

    pathless_file_name = _get_pathless_raw_file_name(unix_time_sec)
    raw_local_file_name = '{0:s}/{1:s}/{2:s}'.format(
        top_local_directory_name,
        time_conversion.unix_sec_to_string(unix_time_sec, TIME_FORMAT_DATE),
        pathless_file_name)

    if raise_error_if_missing and not os.path.isfile(raw_local_file_name):
        raise ValueError(
            'Cannot find raw file.  Expected at location: ' +
            raw_local_file_name)

    return raw_local_file_name


def read_storm_objects_from_raw_file(json_file_name):
    """Reads storm objects from raw file.

    This file should contain all storm objects for one tracking scale and one
    time step.

    P = number of grid points in given storm object
    V = number of vertices in bounding polygon of given storm object

    :param json_file_name: Path to input file.
    :return: storm_object_table: pandas DataFrame with the following columns.
    storm_object_table.storm_id: String ID for storm cell.
    storm_object_table.unix_time_sec: Time in Unix format.
    storm_object_table.east_velocity_m_s01: Eastward velocity (m/s).
    storm_object_table.north_velocity_m_s01: Northward velocity (m/s).
    storm_object_table.age_sec: Age of storm cell (seconds).
    storm_object_table.centroid_lat_deg: Latitude at centroid of storm object
        (deg N).
    storm_object_table.centroid_lng_deg: Longitude at centroid of storm object
        (deg E).
    storm_object_table.grid_point_latitudes_deg: length-P numpy array with
        latitudes (deg N) of grid points in storm object.
    storm_object_table.grid_point_longitudes_deg: length-P numpy array with
        longitudes (deg E) of grid points in storm object.
    storm_object_table.grid_point_rows: length-P numpy array with row indices
        (integers) of grid points in storm object.
    storm_object_table.grid_point_columns: length-P numpy array with column
        indices (integers) of grid points in storm object.
    storm_object_table.polygon_object_latlng: Instance of
        `shapely.geometry.Polygon` with vertices in lat-long coordinates.
    storm_object_table.polygon_object_rowcol: Instance of
        `shapely.geometry.Polygon` with vertices in row-column coordinates.
    """

    error_checking.assert_file_exists(json_file_name)
    with open(json_file_name) as json_file_handle:
        probsevere_dict = json.load(json_file_handle)

    unix_time_sec = time_conversion.string_to_unix_sec(
        probsevere_dict[TIME_COLUMN_ORIG].encode('ascii', 'ignore'),
        TIME_FORMAT_IN_RAW_FILES)
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
        tracking_io.STORM_ID_COLUMN: storm_ids,
        tracking_io.EAST_VELOCITY_COLUMN: east_velocities_m_s01,
        tracking_io.NORTH_VELOCITY_COLUMN: north_velocities_m_s01,
        tracking_io.TIME_COLUMN: unix_times_sec}
    storm_object_table = pandas.DataFrame.from_dict(storm_object_dict)
    storm_object_table = tracking_io.remove_rows_with_nan(storm_object_table)

    num_storms = len(storm_object_table.index)
    storm_ages_sec = numpy.full(num_storms, numpy.nan)

    simple_array = numpy.full(num_storms, numpy.nan)
    object_array = numpy.full(num_storms, numpy.nan, dtype=object)
    nested_array = storm_object_table[[
        tracking_io.STORM_ID_COLUMN,
        tracking_io.STORM_ID_COLUMN]].values.tolist()

    argument_dict = {tracking_io.AGE_COLUMN: storm_ages_sec,
                     tracking_io.CENTROID_LAT_COLUMN: simple_array,
                     tracking_io.CENTROID_LNG_COLUMN: simple_array,
                     tracking_io.GRID_POINT_LAT_COLUMN: nested_array,
                     tracking_io.GRID_POINT_LNG_COLUMN: nested_array,
                     tracking_io.GRID_POINT_ROW_COLUMN: nested_array,
                     tracking_io.GRID_POINT_COLUMN_COLUMN: nested_array,
                     tracking_io.POLYGON_OBJECT_LATLNG_COLUMN: object_array,
                     tracking_io.POLYGON_OBJECT_ROWCOL_COLUMN: object_array}
    storm_object_table = storm_object_table.assign(**argument_dict)

    for i in range(num_storms):
        this_vertex_matrix_deg = numpy.asarray(
            probsevere_dict[FEATURES_COLUMN_ORIG][i][GEOMETRY_COLUMN_ORIG][
                COORDINATES_COLUMN_ORIG][0])
        these_vertex_lat_deg = this_vertex_matrix_deg[:, LAT_COLUMN_INDEX_ORIG]
        these_vertex_lng_deg = this_vertex_matrix_deg[:, LNG_COLUMN_INDEX_ORIG]

        (these_vertex_rows, these_vertex_columns) = radar_io.latlng_to_rowcol(
            these_vertex_lat_deg, these_vertex_lng_deg,
            nw_grid_point_lat_deg=NW_GRID_POINT_LAT_DEG,
            nw_grid_point_lng_deg=NW_GRID_POINT_LNG_DEG,
            lat_spacing_deg=GRID_LAT_SPACING_DEG,
            lng_spacing_deg=GRID_LNG_SPACING_DEG)

        these_vertex_rows, these_vertex_columns = (
            polygons.fix_probsevere_vertices(
                these_vertex_rows, these_vertex_columns))

        these_vertex_lat_deg, these_vertex_lng_deg = radar_io.rowcol_to_latlng(
            these_vertex_rows, these_vertex_columns,
            nw_grid_point_lat_deg=NW_GRID_POINT_LAT_DEG,
            nw_grid_point_lng_deg=NW_GRID_POINT_LNG_DEG,
            lat_spacing_deg=GRID_LAT_SPACING_DEG,
            lng_spacing_deg=GRID_LNG_SPACING_DEG)

        (storm_object_table[tracking_io.GRID_POINT_ROW_COLUMN].values[i],
         storm_object_table[tracking_io.GRID_POINT_COLUMN_COLUMN].values[i]) = (
             polygons.simple_polygon_to_grid_points(
                 these_vertex_rows, these_vertex_columns))

        (storm_object_table[tracking_io.GRID_POINT_LAT_COLUMN].values[i],
         storm_object_table[tracking_io.GRID_POINT_LNG_COLUMN].values[i]) = (
             radar_io.rowcol_to_latlng(
                 storm_object_table[tracking_io.GRID_POINT_ROW_COLUMN].values[i],
                 storm_object_table[
                     tracking_io.GRID_POINT_COLUMN_COLUMN].values[i],
                 nw_grid_point_lat_deg=NW_GRID_POINT_LAT_DEG,
                 nw_grid_point_lng_deg=NW_GRID_POINT_LNG_DEG,
                 lat_spacing_deg=GRID_LAT_SPACING_DEG,
                 lng_spacing_deg=GRID_LNG_SPACING_DEG))

        (storm_object_table[tracking_io.CENTROID_LAT_COLUMN].values[i],
         storm_object_table[tracking_io.CENTROID_LNG_COLUMN].values[i]) = (
             polygons.get_latlng_centroid(
                 these_vertex_lat_deg, these_vertex_lng_deg))

        storm_object_table[
            tracking_io.POLYGON_OBJECT_ROWCOL_COLUMN].values[i] = (
                polygons.vertex_arrays_to_polygon_object(
                    these_vertex_columns, these_vertex_rows))
        storm_object_table[
            tracking_io.POLYGON_OBJECT_LATLNG_COLUMN].values[i] = (
                polygons.vertex_arrays_to_polygon_object(
                    these_vertex_lng_deg, these_vertex_lat_deg))

    return storm_object_table


if __name__ == '__main__':
    STORM_OBJECT_TABLE = read_storm_objects_from_raw_file(RAW_FILE_NAME)
    print STORM_OBJECT_TABLE

    (CENTRAL_LATITUDE_DEG, CENTRAL_LONGITUDE_DEG) = (
        radar_io.get_center_of_grid(
            nw_grid_point_lat_deg=NW_GRID_POINT_LAT_DEG,
            nw_grid_point_lng_deg=NW_GRID_POINT_LNG_DEG,
            lat_spacing_deg=GRID_LAT_SPACING_DEG,
            lng_spacing_deg=GRID_LNG_SPACING_DEG, num_grid_rows=NUM_GRID_ROWS,
            num_grid_columns=NUM_GRID_COLUMNS))

    STORM_OBJECT_TABLE = tracking_io.make_buffers_around_polygons(
        STORM_OBJECT_TABLE, min_buffer_dists_metres=MIN_BUFFER_DISTS_METRES,
        max_buffer_dists_metres=MAX_BUFFER_DISTS_METRES,
        central_latitude_deg=CENTRAL_LATITUDE_DEG,
        central_longitude_deg=CENTRAL_LONGITUDE_DEG)
    print STORM_OBJECT_TABLE

    PROCESSED_FILE_NAME = tracking_io.find_processed_file(
        unix_time_sec=UNIX_TIME_SEC,
        data_source=tracking_io.PROBSEVERE_SOURCE_ID,
        top_processed_dir_name=TOP_PROCESSED_DIR_NAME,
        tracking_scale_metres2=TRACKING_SCALE_METRES2,
        raise_error_if_missing=False)

    tracking_io.write_processed_file(STORM_OBJECT_TABLE, PROCESSED_FILE_NAME)
