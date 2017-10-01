"""Methods for linking wind observations to storm cells.

DEFINITIONS

"Storm cell" = a single thunderstorm (standard meteorological definition).  I
will often use S to denote one storm cell.

"Storm object" = one storm cell at one time step.  I will often use s to denote
one storm object.
"""

import glob
import numpy
import pandas
from gewittergefahr.gg_io import segmotion_io
from gewittergefahr.gg_utils import projections
from gewittergefahr.gg_utils import interp
from gewittergefahr.gg_utils import error_checking

# TODO(thunderhoser): This file is still under construction.  Do not attempt to
# use.

STORM_COLUMNS_TO_KEEP = [
    segmotion_io.STORM_ID_COLUMN, segmotion_io.TIME_COLUMN,
    segmotion_io.CENTROID_LAT_COLUMN, segmotion_io.CENTROID_LNG_COLUMN,
    segmotion_io.VERTEX_LAT_COLUMN, segmotion_io.VERTEX_LNG_COLUMN]

LATLNG_COLUMNS = [
    segmotion_io.CENTROID_LAT_COLUMN, segmotion_io.CENTROID_LNG_COLUMN,
    segmotion_io.VERTEX_LAT_COLUMN, segmotion_io.VERTEX_LNG_COLUMN]

CENTROID_X_COLUMN = 'centroid_x_metres'
CENTROID_Y_COLUMN = 'centroid_y_metres'
VERTICES_X_COLUMN = 'vertices_x_metres'
VERTICES_Y_COLUMN = 'vertices_y_metres'
VERTEX_X_COLUMN = 'vertex_x_metres'
VERTEX_Y_COLUMN = 'vertex_y_metres'

START_TIME_COLUMN = 'start_time_unix_sec'
END_TIME_COLUMN = 'end_time_unix_sec'

PROCESSED_SEGMOTION_DIR_NAME = (
    '/localdata/ryan.lagerquist/gewittergefahr_junk/segmotion/processed/'
    '20040811/scale_50000000m2')

QUERY_TIME_UNIX_SEC = 1092227880
MAX_TIME_BEFORE_START_SEC = 1020
MAX_TIME_AFTER_END_SEC = 1020


def _get_latlng_centroid(latitudes_deg, longitudes_deg):
    """Computes centroid of set of lat-long points.

    N = number of points

    :param latitudes_deg: length-N numpy array of latitudes (deg N).
    :param longitudes_deg: length-N numpy array of longitudes (deg E).
    :return: centroid_lat_deg: Latitude at centroid (deg N).
    :return: centroid_lng_deg: Longitude at centroid (deg E).
    """

    # TODO(thunderhoser): Put this method in common file.  Currently defined
    # both here and in segmotion_io.py.

    return (numpy.mean(latitudes_deg[numpy.invert(numpy.isnan(latitudes_deg))]),
            numpy.mean(
                longitudes_deg[numpy.invert(numpy.isnan(longitudes_deg))]))


def _read_storm_objects(processed_segmotion_file_names):
    """Reads storm objects from processed segmotion files.

    Each file should contain storm statistics and polygons for one time step and
    one tracking scale.  See `segmotion_io.write_processed_file`.

    N = number of files
    V = number of vertices in a given storm object

    :param processed_segmotion_file_names: length-N list of paths to input
        files.
    :return: storm_object_table: pandas DataFrame with the following columns.
    storm_object_table.storm_id: String ID for storm cell.
    storm_object_table.unix_time_sec: Time in Unix format.
    storm_object_table.centroid_lat_deg: Latitude of storm centroid (deg N).
    storm_object_table.centroid_lng_deg: Longitude of storm centroid (deg E).
    storm_object_table.vertex_latitudes_deg: length-V numpy array with latitudes
        of vertices (deg N).
    storm_object_table.vertex_longitudes_deg: length-V numpy array with
        longitudes of vertices (deg E).
    """

    list_of_tables = []

    for this_file_name in processed_segmotion_file_names:
        this_storm_object_table = segmotion_io.read_processed_file(
            this_file_name)
        this_storm_object_table = this_storm_object_table[STORM_COLUMNS_TO_KEEP]

        if not list_of_tables:
            list_of_tables.append(this_storm_object_table)
            continue

        this_storm_object_table, _ = this_storm_object_table.align(
            list_of_tables[-1], axis=1)
        list_of_tables.append(this_storm_object_table)

    return pandas.concat(list_of_tables, axis=0, ignore_index=True)


def _project_storms_to_equidistant(storm_object_table):
    """Projects storm locations to azimuthal equidistant coordinate system.

    V = number of vertices in a given storm object

    :param storm_object_table: pandas DataFrame created by _read_storm_objects.
    :return: storm_object_table: Same as input, but with 4 columns removed
        ("centroid_lat_deg", "centroid_lng_deg", "vertex_latitudes_deg",
        "vertex_longitudes_deg") and replaced by the following.
    storm_object_table.centroid_x_metres: x-coordinate of storm centroid.
    storm_object_table.centroid_y_metres: y-coordinate of storm centroid.
    storm_object_table.vertices_x_metres: length-V numpy array with x-
        coordinates of vertices.
    storm_object_table.vertices_y_metres: length-V numpy array with y-
        coordinates of vertices.
    """

    (global_centroid_lat_deg, global_centroid_lng_deg) = _get_latlng_centroid(
        storm_object_table[segmotion_io.CENTROID_LAT_COLUMN].values,
        storm_object_table[segmotion_io.CENTROID_LNG_COLUMN].values)

    projection_object = projections.init_azimuthal_equidistant_projection(
        global_centroid_lat_deg, global_centroid_lng_deg)

    (centroids_x_metres, centroids_y_metres) = projections.project_latlng_to_xy(
        storm_object_table[segmotion_io.CENTROID_LAT_COLUMN].values,
        storm_object_table[segmotion_io.CENTROID_LNG_COLUMN].values,
        projection_object=projection_object)

    nested_array = storm_object_table[[
        segmotion_io.STORM_ID_COLUMN,
        segmotion_io.STORM_ID_COLUMN]].values.tolist()
    argument_dict = {
        CENTROID_X_COLUMN: centroids_x_metres,
        CENTROID_Y_COLUMN: centroids_y_metres, VERTICES_X_COLUMN: nested_array,
        VERTICES_Y_COLUMN: nested_array}
    storm_object_table = storm_object_table.assign(**argument_dict)

    num_storm_objects = len(storm_object_table.index)
    for i in range(num_storm_objects):
        (storm_object_table[VERTICES_X_COLUMN].values[i],
         storm_object_table[VERTICES_Y_COLUMN].values[i]) = (
             projections.project_latlng_to_xy(
                 storm_object_table[segmotion_io.VERTEX_LAT_COLUMN].values[i],
                 storm_object_table[segmotion_io.VERTEX_LNG_COLUMN].values[i],
                 projection_object=projection_object))

    return storm_object_table.drop(LATLNG_COLUMNS, axis=1, inplace=False)


def _storm_objects_to_cells(storm_object_table):
    """Adds temporal information to storm objects.

    Specifically, adds start time and end time to each row in
    storm_object_table.  These are the first and last times, respectively, with
    any storm object of the same ID -- in other words, the beginning and end of
    the storm track.

    :param storm_object_table: pandas DataFrame created by
        _project_storms_to_equidistant.
    :return: storm_object_table: Same as input, except with additional columns.
    storm_object_table.start_time_unix_sec: Start time of corresponding storm
        cell (Unix format).
    storm_object_table.end_time_unix_sec: End time of corresponding storm cell
        (Unix format).
    """

    storm_id_numpy_array = numpy.array(
        storm_object_table[segmotion_io.STORM_ID_COLUMN].values)
    unique_storm_ids, storm_ids_orig_to_unique = numpy.unique(
        storm_id_numpy_array, return_inverse=True)

    num_storm_objects = len(storm_object_table.index)
    start_times_unix_sec = numpy.full(num_storm_objects, 0, dtype=int)
    end_times_unix_sec = numpy.full(num_storm_objects, 0, dtype=int)

    num_storm_cells = len(unique_storm_ids)
    for i in range(num_storm_cells):
        these_storm_object_indices = numpy.where(
            storm_ids_orig_to_unique == i)[0]
        start_times_unix_sec[these_storm_object_indices] = numpy.min(
            storm_object_table[segmotion_io.TIME_COLUMN].values[
                these_storm_object_indices])
        end_times_unix_sec[these_storm_object_indices] = numpy.max(
            storm_object_table[segmotion_io.TIME_COLUMN].values[
                these_storm_object_indices])

    argument_dict = {START_TIME_COLUMN: start_times_unix_sec,
                     END_TIME_COLUMN: end_times_unix_sec}
    return storm_object_table.assign(**argument_dict)


def _filter_storms_by_time(storm_object_table, max_start_time_unix_sec=None,
                           min_end_time_unix_sec=None):
    """Filters storm cells by time.

    For any storm cell S with start time > max_start_time_unix_sec, no storm
    objects will be included in the output.

    For any storm cell S with end time < min_end_time_unix_sec, no storm objects
    will be included in the output.

    For all other storm cells, all storm objects will be included in the output.

    :param storm_object_table: pandas DataFrame created by
        _storm_objects_to_cells.
    :param max_start_time_unix_sec: Max start time (Unix format) of storm cell.
    :param min_end_time_unix_sec: Minimum end time (Unix format) of storm cell.
    :return: filtered_storm_object_table: Same as input, except with fewer rows.
    """

    bad_start_time_flags = (
        storm_object_table[START_TIME_COLUMN].values > max_start_time_unix_sec)
    bad_end_time_flags = (
        storm_object_table[END_TIME_COLUMN].values < min_end_time_unix_sec)

    invalid_flags = numpy.logical_or(bad_start_time_flags, bad_end_time_flags)
    invalid_indices = numpy.where(invalid_flags)[0]
    return storm_object_table.drop(
        storm_object_table.index[invalid_indices], axis=0, inplace=False)


def _interp_one_storm_in_time(storm_object_table_1cell, storm_id=None,
                              query_time_unix_sec=None):
    """Interpolates location of one storm cell in time.

    V = number of vertices in a given storm object

    :param storm_object_table_1cell: pandas DataFrame with the following
        columns.
    storm_object_table_1cell.unix_time_sec: Time in Unix format.
    storm_object_table_1cell.centroid_x_metres: x-coordinate of storm
        centroid.
    storm_object_table_1cell.centroid_y_metres: y-coordinate of storm
        centroid.
    storm_object_table_1cell.vertices_x_metres: length-V numpy array with x-
        coordinates of vertices.
    storm_object_table_1cell.vertices_y_metres: length-V numpy array with y-
        coordinates of vertices.
    :param storm_id: String ID for storm cell.
    :param query_time_unix_sec: Storm location will be interpolated to this time
        (Unix format).
    :return: interp_vertex_table_1object: pandas DataFrame with the following
        columns (each row is one vertex of the interpolated storm object).
    interp_vertex_table_1object.storm_id: String ID for storm cell.
    interp_vertex_table_1object.vertex_x_metres: x-coordinate of vertex.
    interp_vertex_table_1object.vertex_y_metres: y-coordinate of vertex.
    """

    sort_indices = numpy.argsort(
        storm_object_table_1cell[segmotion_io.TIME_COLUMN].values)
    centroid_matrix = numpy.vstack((
        storm_object_table_1cell[CENTROID_X_COLUMN].values[sort_indices],
        storm_object_table_1cell[CENTROID_Y_COLUMN].values[sort_indices]))

    interp_centroid_vector = interp.interp_in_time(
        centroid_matrix, sorted_input_times_unix_sec=storm_object_table_1cell[
            segmotion_io.TIME_COLUMN].values[sort_indices],
        query_times_unix_sec=numpy.array([query_time_unix_sec]),
        allow_extrap=True)

    absolute_time_diffs_sec = numpy.absolute(
        storm_object_table_1cell[segmotion_io.TIME_COLUMN].values -
        query_time_unix_sec)
    nearest_time_index = numpy.argmin(absolute_time_diffs_sec)

    x_diff_metres = interp_centroid_vector[0] - storm_object_table_1cell[
        CENTROID_X_COLUMN].values[nearest_time_index]
    y_diff_metres = interp_centroid_vector[1] - storm_object_table_1cell[
        CENTROID_Y_COLUMN].values[nearest_time_index]

    num_vertices = len(storm_object_table_1cell[VERTICES_X_COLUMN].values[
        nearest_time_index])
    storm_id_list = [storm_id] * num_vertices

    interp_vertex_dict_1object = {
        segmotion_io.STORM_ID_COLUMN: storm_id_list,
        VERTEX_X_COLUMN: storm_object_table_1cell[VERTICES_X_COLUMN].values[
            nearest_time_index] + x_diff_metres,
        VERTEX_Y_COLUMN: storm_object_table_1cell[VERTICES_Y_COLUMN].values[
            nearest_time_index] + y_diff_metres}
    return pandas.DataFrame.from_dict(interp_vertex_dict_1object)


def _interp_storms_in_time(storm_object_table, query_time_unix_sec=None,
                           max_time_before_start_sec=None,
                           max_time_after_end_sec=None):
    """Interpolates storm locations in time.

    Each storm object is advected as a whole -- i.e., all vertices are moved by
    the same amount and in the same direction -- so that storm objects retain
    realistic shapes.

    :param storm_object_table: pandas DataFrame created by
        _storm_objects_to_cells.
    :param query_time_unix_sec: Storm locations will be interpolated to this
        time (Unix format).
    :param max_time_before_start_sec: Max time before beginning of storm cell.
        For each storm cell S, if query time is > max_time_before_start before
        first time in S, S will not be interpolated.
    :param max_time_after_end_sec: Max time after end of storm cell.  For each
        storm cell S, if query time is > max_time_after_end after last time in
        S, S will not be interpolated.
    :return: interp_vertex_table: pandas DataFrame with the following columns
        (each row is one vertex of one storm object).
    interp_vertex_table.storm_id: String ID for storm cell.
    interp_vertex_table.vertex_x_metres: x-coordinate of vertex.
    interp_vertex_table.vertex_y_metres: y-coordinate of vertex.
    """

    error_checking.assert_is_integer(query_time_unix_sec)
    error_checking.assert_is_not_nan(query_time_unix_sec)
    error_checking.assert_is_integer(max_time_before_start_sec)
    error_checking.assert_is_geq(max_time_before_start_sec, 0)
    error_checking.assert_is_integer(max_time_after_end_sec)
    error_checking.assert_is_geq(max_time_after_end_sec, 0)

    max_start_time_unix_sec = query_time_unix_sec + max_time_before_start_sec
    min_end_time_unix_sec = query_time_unix_sec - max_time_after_end_sec

    filtered_storm_object_table = _filter_storms_by_time(
        storm_object_table, max_start_time_unix_sec=max_start_time_unix_sec,
        min_end_time_unix_sec=min_end_time_unix_sec)

    storm_id_numpy_array = numpy.array(
        filtered_storm_object_table[segmotion_io.STORM_ID_COLUMN].values)
    unique_storm_ids = numpy.unique(storm_id_numpy_array)

    list_of_tables = []
    num_storm_cells = len(unique_storm_ids)
    for i in range(num_storm_cells):
        this_storm_cell_table = (
            filtered_storm_object_table.loc[
                filtered_storm_object_table[
                    segmotion_io.STORM_ID_COLUMN] == unique_storm_ids[i]])
        if len(this_storm_cell_table.index) == 1:
            continue

        this_interp_vertex_table = _interp_one_storm_in_time(
            this_storm_cell_table, storm_id=unique_storm_ids[i],
            query_time_unix_sec=query_time_unix_sec)
        if not list_of_tables:
            list_of_tables.append(this_interp_vertex_table)
            continue

        this_interp_vertex_table, _ = this_interp_vertex_table.align(
            list_of_tables[-1], axis=1)
        list_of_tables.append(this_interp_vertex_table)

    return pandas.concat(list_of_tables, axis=0, ignore_index=True)


if __name__ == '__main__':
    PROCESSED_SEGMOTION_FILE_NAMES = glob.glob(PROCESSED_SEGMOTION_DIR_NAME + '/*.p')
    STORM_OBJECT_TABLE = _read_storm_objects(PROCESSED_SEGMOTION_FILE_NAMES)
    STORM_OBJECT_TABLE = _project_storms_to_equidistant(STORM_OBJECT_TABLE)
    STORM_OBJECT_TABLE = _storm_objects_to_cells(STORM_OBJECT_TABLE)
    print STORM_OBJECT_TABLE

    INTERP_VERTEX_TABLE = _interp_storms_in_time(
        STORM_OBJECT_TABLE, query_time_unix_sec=QUERY_TIME_UNIX_SEC,
        max_time_before_start_sec=MAX_TIME_BEFORE_START_SEC,
        max_time_after_end_sec=MAX_TIME_AFTER_END_SEC)
    print INTERP_VERTEX_TABLE
