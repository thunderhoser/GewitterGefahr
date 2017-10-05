"""IO methods for storm-tracking output (both polygons and track statistics)."""

import os
import glob
import pickle
import numpy
from gewittergefahr.gg_io import myrorss_io
from gewittergefahr.gg_utils import polygons
from gewittergefahr.gg_utils import projections
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.gg_utils import error_checking

TIME_FORMAT = '%Y-%m-%d-%H%M%S'
DATE_FORMAT = '%Y%m%d'

SEGMOTION_SOURCE_ID = 'segmotion'
PROBSEVERE_SOURCE_ID = 'probSevere'
DATA_SOURCE_IDS = [SEGMOTION_SOURCE_ID, PROBSEVERE_SOURCE_ID]

PROCESSED_FILE_PREFIX = 'storm-tracking'
PROCESSED_FILE_EXTENSION = '.p'

STORM_ID_COLUMN = 'storm_id'
TIME_COLUMN = 'unix_time_sec'
EAST_VELOCITY_COLUMN = 'east_velocity_m_s01'
NORTH_VELOCITY_COLUMN = 'north_velocity_m_s01'
AGE_COLUMN = 'age_sec'

CENTROID_LAT_COLUMN = 'centroid_lat_deg'
CENTROID_LNG_COLUMN = 'centroid_lng_deg'
GRID_POINT_LAT_COLUMN = 'grid_point_latitudes_deg'
GRID_POINT_LNG_COLUMN = 'grid_point_longitudes_deg'
GRID_POINT_ROW_COLUMN = 'grid_point_rows'
GRID_POINT_COLUMN_COLUMN = 'grid_point_columns'
VERTEX_LAT_COLUMN = 'vertex_latitudes_deg'
VERTEX_LNG_COLUMN = 'vertex_longitudes_deg'
VERTEX_ROW_COLUMN = 'vertex_rows'
VERTEX_COLUMN_COLUMN = 'vertex_columns'

MANDATORY_COLUMNS = [
    STORM_ID_COLUMN, TIME_COLUMN, EAST_VELOCITY_COLUMN, NORTH_VELOCITY_COLUMN,
    AGE_COLUMN, CENTROID_LAT_COLUMN, CENTROID_LNG_COLUMN, GRID_POINT_LAT_COLUMN,
    GRID_POINT_LNG_COLUMN, GRID_POINT_ROW_COLUMN, GRID_POINT_COLUMN_COLUMN,
    VERTEX_LAT_COLUMN, VERTEX_LNG_COLUMN, VERTEX_ROW_COLUMN,
    VERTEX_COLUMN_COLUMN]

BUFFER_VERTEX_LAT_COLUMN_PREFIX = 'vertex_latitudes_deg_buffer'
BUFFER_VERTEX_LNG_COLUMN_PREFIX = 'vertex_longitudes_deg_buffer'


def _check_data_source(data_source):
    """Ensures that data source is either "segmotion" or "probSevere".

    :param data_source: Data source (string).
    :raises: ValueError: data source is neither "segmotion" nor "probSevere".
    """

    error_checking.assert_is_string(data_source)
    if data_source not in DATA_SOURCE_IDS:
        error_string = (
            '\n' + str(DATA_SOURCE_IDS) +
            '\nValid data sources (listed above) do not include "' +
            data_source + '".')
        raise ValueError(error_string)


def _distance_buffers_to_column_names(min_buffer_dists_metres,
                                      max_buffer_dists_metres):
    """Generates column name for each distance buffer.

    N = number of distance buffers

    :param min_buffer_dists_metres: length-N numpy array of minimum distances
        (integers).
    :param max_buffer_dists_metres: length-N numpy array of maximum distances
        (integers).
    :return: buffer_lat_column_names: length-N list of column names for vertex
        latitudes.
    :return: buffer_lng_column_names: length-N list of column names for vertex
        longitudes.
    """

    num_buffers = len(min_buffer_dists_metres)
    buffer_lat_column_names = [''] * num_buffers
    buffer_lng_column_names = [''] * num_buffers

    for j in range(num_buffers):
        if numpy.isnan(min_buffer_dists_metres[j]):
            buffer_lat_column_names[j] = '{0:s}_{1:d}m'.format(
                BUFFER_VERTEX_LAT_COLUMN_PREFIX,
                int(max_buffer_dists_metres[j]))
            buffer_lng_column_names[j] = '{0:s}_{1:d}m'.format(
                BUFFER_VERTEX_LNG_COLUMN_PREFIX,
                int(max_buffer_dists_metres[j]))
        else:
            buffer_lat_column_names[j] = '{0:s}_{1:d}_{2:d}m'.format(
                BUFFER_VERTEX_LAT_COLUMN_PREFIX,
                int(min_buffer_dists_metres[j]),
                int(max_buffer_dists_metres[j]))
            buffer_lng_column_names[j] = '{0:s}_{1:d}_{2:d}m'.format(
                BUFFER_VERTEX_LNG_COLUMN_PREFIX,
                int(min_buffer_dists_metres[j]),
                int(max_buffer_dists_metres[j]))

    return buffer_lat_column_names, buffer_lng_column_names


def _get_pathless_processed_file_name(unix_time_sec, data_source):
    """Generates pathless name for processed storm-tracking file.

    This file should contain both polygons and track statistics for one time
    step and one tracking scale.

    :param unix_time_sec: Time in Unix format.
    :param data_source: Data source (either "segmotion" or "probSevere").
    :return: pathless_processed_file_name: Pathless name for processed file.
    """

    return '{0:s}_{1:s}_{2:s}{3:s}'.format(
        PROCESSED_FILE_PREFIX, data_source,
        time_conversion.unix_sec_to_string(unix_time_sec, TIME_FORMAT),
        PROCESSED_FILE_EXTENSION)


def _get_relative_processed_directory(data_source=None, spc_date_unix_sec=None,
                                      unix_time_sec=None,
                                      tracking_scale_metres2=None):
    """Generates relative path for processed storm-tracking files.

    :param data_source: Data source (either "segmotion" or "probSevere").
    :param spc_date_unix_sec: SPC date in Unix format (needed only if
        data_source = "segmotion").
    :param unix_time_sec: Valid time (needed only if data_source =
        "probSevere").
    :param tracking_scale_metres2: Tracking scale.
    :return: relative_processed_dir_name: Relative path for processed storm-
        tracking files.
    """

    if data_source == SEGMOTION_SOURCE_ID:
        date_string = myrorss_io.time_unix_sec_to_spc_date(spc_date_unix_sec)
    else:
        date_string = time_conversion.unix_sec_to_string(unix_time_sec,
                                                         DATE_FORMAT)

    return '{0:s}/scale_{1:d}m2'.format(date_string,
                                        int(tracking_scale_metres2))


def remove_rows_with_nan(input_table):
    """Removes any row with NaN from pandas DataFrame.

    :param input_table: pandas DataFrame.
    :return: output_table: Same as input_table, except that some rows may be
        gone.
    """

    return input_table.loc[input_table.notnull().all(axis=1)]


def make_buffers_around_polygons(storm_object_table,
                                 min_buffer_dists_metres=None,
                                 max_buffer_dists_metres=None,
                                 central_latitude_deg=None,
                                 central_longitude_deg=None):
    """Creates one or more buffers around each storm polygon.

    N = number of buffers
    V = number of vertices in a given polygon

    :param storm_object_table: pandas DataFrame with the following columns.
    storm_object_table.storm_id: String ID for storm cell.
    storm_object_table.vertex_latitudes_deg: length-V numpy array with latitudes
        (deg N) of vertices.
    storm_object_table.vertex_longitudes_deg: length-V numpy array with
        longitudes (deg E) of vertices.
    :param min_buffer_dists_metres: length-N numpy array of minimum buffer
        distances.  If min_buffer_dists_metres[i] is NaN, the [i]th buffer
        includes the original polygon.  If min_buffer_dists_metres[i] is
        defined, the [i]th buffer is a "nested" buffer, not including the
        original polygon.
    :param max_buffer_dists_metres: length-N numpy array of maximum buffer
        distances.  Must be all real numbers (no NaN).
    :param central_latitude_deg: Central latitude (deg N) for azimuthal
        equidistant projection.
    :param central_longitude_deg: Central longitude (deg E) for azimuthal
        equidistant projection.
    :return: storm_object_table: Same as input, but with 2*N extra columns.
    storm_object_table.vertex_latitudes_deg_buffer_<D>m: length-V numpy array
        with latitudes (deg N) of vertices in D-metre buffer around storm.
    storm_object_table.vertex_longitudes_deg_buffer_<D>m: length-V numpy array
        with longitudes (deg E) of vertices in D-metre buffer around storm.
    storm_object_table.vertex_latitudes_deg_buffer_<d>_<D>m: length-V numpy
        array with latitudes (deg N) of vertices inside D-metre buffer, but
        outside d-metre buffer, around storm.
    storm_object_table.vertex_longitudes_deg_buffer_<d>_<D>m: length-V numpy
        array with longitudes (deg E) of vertices inside D-metre buffer, but
        outside d-metre buffer, around storm.
    """

    error_checking.assert_is_geq_numpy_array(
        min_buffer_dists_metres, 0., allow_nan=True)
    error_checking.assert_is_numpy_array(
        min_buffer_dists_metres, num_dimensions=1)

    num_buffers = len(min_buffer_dists_metres)
    error_checking.assert_is_geq_numpy_array(
        max_buffer_dists_metres, 0., allow_nan=False)
    error_checking.assert_is_numpy_array(
        max_buffer_dists_metres, exact_dimensions=numpy.array([num_buffers]))

    for j in range(num_buffers):
        if numpy.isnan(min_buffer_dists_metres[j]):
            continue
        error_checking.assert_is_greater(
            max_buffer_dists_metres[j], min_buffer_dists_metres[j],
            allow_nan=False)

    projection_object = projections.init_azimuthal_equidistant_projection(
        central_latitude_deg, central_longitude_deg)

    (buffer_lat_column_names,
     buffer_lng_column_names) = _distance_buffers_to_column_names(
         min_buffer_dists_metres, max_buffer_dists_metres)

    nested_array = storm_object_table[[
        STORM_ID_COLUMN, STORM_ID_COLUMN]].values.tolist()
    argument_dict = {}
    for j in range(num_buffers):
        argument_dict.update({buffer_lat_column_names[j]: nested_array,
                              buffer_lng_column_names[j]: nested_array})
    storm_object_table = storm_object_table.assign(**argument_dict)

    num_storms = len(storm_object_table.index)
    for i in range(num_storms):
        (orig_vertex_x_metres,
         orig_vertex_y_metres) = projections.project_latlng_to_xy(
             storm_object_table[VERTEX_LAT_COLUMN].values[i],
             storm_object_table[VERTEX_LNG_COLUMN].values[i],
             projection_object=projection_object)

        for j in range(num_buffers):
            (buffer_vertex_x_metres, buffer_vertex_y_metres) = (
                polygons.make_buffer_around_simple_polygon(
                    orig_vertex_x_metres, orig_vertex_y_metres,
                    min_buffer_dist_metres=min_buffer_dists_metres[j],
                    max_buffer_dist_metres=max_buffer_dists_metres[j]))

            (buffer_vertex_lat_deg,
             buffer_vertex_lng_deg) = projections.project_xy_to_latlng(
                 buffer_vertex_x_metres, buffer_vertex_y_metres,
                 projection_object=projection_object)

            storm_object_table[buffer_lat_column_names[j]].values[
                i] = buffer_vertex_lat_deg
            storm_object_table[buffer_lng_column_names[j]].values[
                i] = buffer_vertex_lng_deg

    return storm_object_table


def find_processed_file(unix_time_sec=None, data_source=None,
                        spc_date_unix_sec=None, top_processed_dir_name=None,
                        tracking_scale_metres2=None,
                        raise_error_if_missing=True):
    """Finds processed tracking file on local machine.

    :param unix_time_sec: Time in Unix format.
    :param data_source: Data source (either "segmotion" or "probSevere").
    :param spc_date_unix_sec: SPC date in Unix format (needed only if
        data_source = "segmotion").
    :param top_processed_dir_name: Top-level directory for processed tracking
        files.
    :param tracking_scale_metres2: Tracking scale.
    :param raise_error_if_missing: Boolean flag.  If raise_error_if_missing =
        True and file is missing, will raise an error.
    :return: processed_file_name: Path to processed tracking file.  If
        raise_error_if_missing = False and file is missing, this will be
        *expected* path.
    :raises: ValueError: if raise_error_if_missing = True and file is missing.
    """

    _check_data_source(data_source)
    error_checking.assert_is_string(top_processed_dir_name)
    error_checking.assert_is_boolean(raise_error_if_missing)

    pathless_file_name = _get_pathless_processed_file_name(unix_time_sec,
                                                           data_source)
    relative_directory_name = _get_relative_processed_directory(
        data_source=data_source, spc_date_unix_sec=spc_date_unix_sec,
        unix_time_sec=unix_time_sec,
        tracking_scale_metres2=tracking_scale_metres2)

    processed_file_name = '{0:s}/{1:s}/{2:s}'.format(
        top_processed_dir_name, relative_directory_name, pathless_file_name)

    if raise_error_if_missing and not os.path.isfile(processed_file_name):
        raise ValueError('Cannot find processed file.  Expected at location: ' +
                         processed_file_name)

    return processed_file_name


def find_processed_segmotion_files_from_dates(spc_dates_unix_sec,
                                              top_processed_dir_name=None,
                                              tracking_scale_metres2=None,
                                              raise_error_if_date_missing=True):
    """Finds processed segmotion files from one or more SPC dates.

    N = number of SPC dates

    :param spc_dates_unix_sec: length-N numpy array of SPC dates (Unix format).
    :param top_processed_dir_name: Top-level directory for processed segmotion
        files.
    :param tracking_scale_metres2: Tracking scale.
    :param raise_error_if_date_missing: Boolean flag.  If
        raise_error_if_date_missing = True and there is any SPC date with no
        files, will raise an error.
    :return: processed_file_names: 1-D list of paths to processed files.
    :raises: ValueError: if raise_error_if_date_missing = True and there is any
        SPC date with no files.
    """

    error_checking.assert_is_numpy_array(spc_dates_unix_sec, num_dimensions=1)
    error_checking.assert_is_string(top_processed_dir_name)
    error_checking.assert_is_boolean(raise_error_if_date_missing)

    processed_file_names = []
    num_spc_dates = len(spc_dates_unix_sec)

    for i in range(num_spc_dates):
        this_relative_dir_name = _get_relative_processed_directory(
            data_source=SEGMOTION_SOURCE_ID,
            spc_date_unix_sec=spc_dates_unix_sec[i],
            tracking_scale_metres2=tracking_scale_metres2)
        this_directory_name = '{0:s}/{1:s}'.format(top_processed_dir_name,
                                                   this_relative_dir_name)
        this_file_pattern = '{0:s}/{1:s}_{2:s}*{3:s}'.format(
            this_directory_name, PROCESSED_FILE_PREFIX, SEGMOTION_SOURCE_ID,
            PROCESSED_FILE_EXTENSION)

        these_processed_file_names = glob.glob(this_file_pattern)
        if not these_processed_file_names:
            if not raise_error_if_date_missing:
                continue

            error_string = ('Cannot find any processed files in directory: ' +
                            this_directory_name)
            raise ValueError(error_string)

        processed_file_names += these_processed_file_names

    return processed_file_names


def write_processed_file(storm_object_table, pickle_file_name):
    """Writes tracking data to file.

    This file should contain both polygons and track statistics for one time
    step and one tracking scale.

    P = number of grid points in a given storm object
    V = number of vertices in a given polygon

    :param storm_object_table: pandas DataFrame with at least the following
        columns.
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
    storm_object_table.vertex_latitudes_deg: length-V numpy array with latitudes
        (deg N) of vertices in bounding polygon.
    storm_object_table.vertex_longitudes_deg: length-V numpy array with
        longitudes (deg E) of vertices in bounding polygon.
    storm_object_table.vertex_rows: length-V numpy array with row indices (half-
        integers) of vertices in bounding polygon.
    storm_object_table.vertex_columns: length-V numpy array with column indices
        (half-integers) of vertices in bounding polygon.
    :param pickle_file_name: Path to output file.
    """

    error_checking.assert_columns_in_dataframe(storm_object_table,
                                               MANDATORY_COLUMNS)
    file_system_utils.mkdir_recursive_if_necessary(file_name=pickle_file_name)

    pickle_file_handle = open(pickle_file_name, 'wb')
    pickle.dump(storm_object_table, pickle_file_handle)
    pickle_file_handle.close()


def read_processed_file(pickle_file_name):
    """Reads tracking data from file.

    This file should contain both polygons and track statistics for one time
    step and one tracking scale.

    :param pickle_file_name: Path to input file.
    :return: storm_object_table: See documentation for write_processed_file.
    """

    pickle_file_handle = open(pickle_file_name, 'rb')
    storm_object_table = pickle.load(pickle_file_handle)
    pickle_file_handle.close()

    error_checking.assert_columns_in_dataframe(storm_object_table,
                                               MANDATORY_COLUMNS)
    return storm_object_table
