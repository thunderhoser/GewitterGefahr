"""Methods for linking wind observations to storm cells.

DEFINITIONS

"Storm cell" = a single thunderstorm (standard meteorological definition).  I
will often use S to denote one storm cell.

"Storm object" = one storm cell at one time step.  I will often use s to denote
one storm object.
"""

import copy
import glob
import pickle
import numpy
import pandas
from gewittergefahr.gg_io import storm_tracking_io as tracking_io
from gewittergefahr.gg_io import raw_wind_io
from gewittergefahr.gg_utils import projections
from gewittergefahr.gg_utils import interp
from gewittergefahr.gg_utils import polygons
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.gg_utils import error_checking

# TODO(thunderhoser): turn main method into end-to-end example.
# TODO(thunderhoser): replace main method with named method.
# TODO(thunderhoser): add method to ensure that wind files cover the required
# time period.

STORM_COLUMNS_TO_KEEP = [
    tracking_io.STORM_ID_COLUMN, tracking_io.TIME_COLUMN,
    tracking_io.CENTROID_LAT_COLUMN, tracking_io.CENTROID_LNG_COLUMN,
    tracking_io.POLYGON_OBJECT_LATLNG_COLUMN]

WIND_COLUMNS_TO_KEEP = [
    raw_wind_io.STATION_ID_COLUMN, raw_wind_io.LATITUDE_COLUMN,
    raw_wind_io.LONGITUDE_COLUMN, raw_wind_io.TIME_COLUMN,
    raw_wind_io.U_WIND_COLUMN, raw_wind_io.V_WIND_COLUMN]

FILE_INDEX_COLUMN = 'file_index'
CENTROID_X_COLUMN = 'centroid_x_metres'
CENTROID_Y_COLUMN = 'centroid_y_metres'
VERTICES_X_COLUMN = 'vertices_x_metres'
VERTICES_Y_COLUMN = 'vertices_y_metres'
VERTEX_X_COLUMN = 'vertex_x_metres'
VERTEX_Y_COLUMN = 'vertex_y_metres'
START_TIME_COLUMN = 'start_time_unix_sec'
END_TIME_COLUMN = 'end_time_unix_sec'

WIND_X_COLUMN = 'x_coord_metres'
WIND_Y_COLUMN = 'y_coord_metres'
NEAREST_STORM_ID_COLUMN = 'nearest_storm_id'
LINKAGE_DISTANCE_COLUMN = 'linkage_distance_metres'

STATION_IDS_COLUMN = 'wind_station_ids'
WIND_LATS_COLUMN = 'wind_latitudes_deg'
WIND_LNGS_COLUMN = 'wind_longitudes_deg'
U_WINDS_COLUMN = 'u_winds_m_s01'
V_WINDS_COLUMN = 'v_winds_m_s01'
LINKAGE_DISTANCES_COLUMN = 'linkage_distances_metres'
RELATIVE_TIMES_COLUMN = 'relative_times_sec'

# The following constants are used only in the main method.
PROCESSED_SEGMOTION_DIR_NAME = (
    '/localdata/ryan.lagerquist/gewittergefahr_junk/segmotion/processed/'
    '20040811/scale_50000000m2')

QUERY_TIME_UNIX_SEC = 1092227880
MAX_TIME_BEFORE_START_SEC = 1020
MAX_TIME_AFTER_END_SEC = 1020


def _get_xy_bounding_box_of_storms(storm_object_table, padding_metres=None):
    """Returns bounding box of all storm objects.

    :param storm_object_table: pandas DataFrame created by
        _project_storms_to_equidistant.
    :param padding_metres: This padding will be added to the edge of the
        bounding box.
    :return: x_limits_metres: length-2 numpy array with [min, max] x-coordinates
        of bounding box.
    :return: y_limits_metres: length-2 numpy array with [min, max] y-coordinates
        of bounding box.
    """

    error_checking.assert_is_geq(padding_metres, 0.)

    all_x_coords_metres = numpy.array([])
    all_y_coords_metres = numpy.array([])
    num_storms = len(storm_object_table.index)

    for i in range(num_storms):
        all_x_coords_metres = numpy.concatenate((
            all_x_coords_metres,
            storm_object_table[VERTICES_X_COLUMN].values[i]))
        all_y_coords_metres = numpy.concatenate((
            all_y_coords_metres,
            storm_object_table[VERTICES_Y_COLUMN].values[i]))

    x_limits_metres = numpy.array(
        [numpy.min(all_x_coords_metres) - padding_metres,
         numpy.max(all_x_coords_metres) + padding_metres])
    y_limits_metres = numpy.array(
        [numpy.min(all_y_coords_metres) - padding_metres,
         numpy.max(all_y_coords_metres) + padding_metres])

    return x_limits_metres, y_limits_metres


def _read_storm_objects(processed_segmotion_file_names):
    """Reads storm objects from processed segmotion files.

    Each file should contain storm statistics and polygons for one time step and
    one tracking scale.  See `storm_tracking_io.write_processed_file`.

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
    storm_object_table.file_index: Index of file from which storm object was
        read.  If file_index = j for the [i]th row, this means the [i]th storm
        object was read from the [j]th file.  This info will be used in writing
        output files (see write_linkage_to_csv).
    """

    error_checking.assert_is_string_list(processed_segmotion_file_names)

    list_of_tables = []
    file_indices = numpy.array([])

    for i in range(len(processed_segmotion_file_names)):
        this_storm_object_table = tracking_io.read_processed_file(
            processed_segmotion_file_names[i])
        this_storm_object_table = this_storm_object_table[STORM_COLUMNS_TO_KEEP]

        this_num_storm_objects = len(this_storm_object_table.index)
        file_indices = numpy.concatenate((
            file_indices,
            numpy.linspace(i, i, num=this_num_storm_objects, dtype=int)))

        if not list_of_tables:
            list_of_tables.append(this_storm_object_table)
            continue

        this_storm_object_table, _ = this_storm_object_table.align(
            list_of_tables[-1], axis=1)
        list_of_tables.append(this_storm_object_table)

    storm_object_table = pandas.concat(list_of_tables, axis=0,
                                       ignore_index=True)
    argument_dict = {FILE_INDEX_COLUMN: file_indices}
    return storm_object_table.assign(**argument_dict)


def _read_wind_observations(processed_wind_file_names,
                            time_limits_unix_sec=None):
    """Reads wind observations from processed files.

    Each file should contain wind observations for all datasets (HFMETARs,
    MADIS, Oklahoma Mesonet, and Storm Events) and some time range.  See
    `raw_wind_io.write_winds_to_csv`.

    N = number of files

    :param processed_wind_file_names: length-N list of paths to input files.
    :param time_limits_unix_sec: length-2 numpy array with [min, max] times
        desired.  Times should be in Unix format.
    :return: wind_table: pandas DataFrame with the following columns.
    wind_table.station_id: String ID for station.
    wind_table.latitude_deg: Latitude (deg N).
    wind_table.longitude_deg: Longitude (deg E).
    wind_table.unix_time_sec: Time in Unix format.
    wind_table.u_wind_m_s01: u-component of wind (metres per second).
    wind_table.v_wind_m_s01: v-component of wind (metres per second).
    """

    error_checking.assert_is_string_list(processed_wind_file_names)
    error_checking.assert_is_integer_numpy_array(time_limits_unix_sec)
    error_checking.assert_is_numpy_array(time_limits_unix_sec,
                                         exact_dimensions=numpy.array([2]))
    error_checking.assert_is_greater(time_limits_unix_sec[1],
                                     time_limits_unix_sec[0])

    list_of_tables = []

    for this_file_name in processed_wind_file_names:
        this_wind_table = raw_wind_io.read_processed_file(this_file_name)
        this_wind_table = this_wind_table[STORM_COLUMNS_TO_KEEP]

        invalid_rows = numpy.logical_or(
            this_wind_table[raw_wind_io.TIME_COLUMN].values <
            time_limits_unix_sec[0],
            this_wind_table[raw_wind_io.TIME_COLUMN].values >
            time_limits_unix_sec[1])
        this_wind_table.drop(this_wind_table.index[invalid_rows], axis=0,
                             inplace=True)

        if not list_of_tables:
            list_of_tables.append(this_wind_table)
            continue

        this_wind_table, _ = this_wind_table.align(list_of_tables[-1], axis=1)
        list_of_tables.append(this_wind_table)

    return pandas.concat(list_of_tables, axis=0, ignore_index=True)


def _project_storms_to_equidistant(storm_object_table):
    """Projects storm locations to azimuthal equidistant coordinate system.

    V = number of vertices in a given storm object

    :param storm_object_table: pandas DataFrame created by _read_storm_objects.
    :return: storm_object_table: Same as input, but with the following added
        columns.
    storm_object_table.centroid_x_metres: x-coordinate of storm centroid.
    storm_object_table.centroid_y_metres: y-coordinate of storm centroid.
    storm_object_table.vertices_x_metres: length-V numpy array with x-
        coordinates of vertices.
    storm_object_table.vertices_y_metres: length-V numpy array with y-
        coordinates of vertices.
    :return: projection_object: Object created by
        `projections.init_azimuthal_equidistant_projection`.  Can be used in the
        future to convert between projection (x-y) and lat-long coordinates.
    """

    (global_centroid_lat_deg,
     global_centroid_lng_deg) = polygons.get_latlng_centroid(
         storm_object_table[tracking_io.CENTROID_LAT_COLUMN].values,
         storm_object_table[tracking_io.CENTROID_LNG_COLUMN].values)

    projection_object = projections.init_azimuthal_equidistant_projection(
        global_centroid_lat_deg, global_centroid_lng_deg)

    (centroids_x_metres, centroids_y_metres) = projections.project_latlng_to_xy(
        storm_object_table[tracking_io.CENTROID_LAT_COLUMN].values,
        storm_object_table[tracking_io.CENTROID_LNG_COLUMN].values,
        projection_object=projection_object)

    nested_array = storm_object_table[[
        tracking_io.STORM_ID_COLUMN,
        tracking_io.STORM_ID_COLUMN]].values.tolist()
    argument_dict = {
        CENTROID_X_COLUMN: centroids_x_metres,
        CENTROID_Y_COLUMN: centroids_y_metres, VERTICES_X_COLUMN: nested_array,
        VERTICES_Y_COLUMN: nested_array}
    storm_object_table = storm_object_table.assign(**argument_dict)

    num_storm_objects = len(storm_object_table.index)
    for i in range(num_storm_objects):
        this_vertex_dict_latlng = (
            polygons.polygon_object_to_vertex_arrays(
                storm_object_table[
                    tracking_io.POLYGON_OBJECT_LATLNG_COLUMN].values[i]))

        (storm_object_table[VERTICES_X_COLUMN].values[i],
         storm_object_table[VERTICES_Y_COLUMN].values[i]) = (
             projections.project_latlng_to_xy(
                 this_vertex_dict_latlng[polygons.EXTERIOR_Y_COLUMN],
                 this_vertex_dict_latlng[polygons.EXTERIOR_X_COLUMN],
                 projection_object=projection_object))

    return storm_object_table, projection_object


def _project_winds_to_equidistant(wind_table, projection_object):
    """Projects wind observations to azimuthal equidistant coordinate system.

    :param wind_table: pandas DataFrame created by _read_wind_observations.
    :param projection_object: Object returned by _project_storms_to_equidistant.
    :return: wind_table: Same as input, but with 2 extra columns.
    wind_table.x_coord_metres: x-coordinate.
    wind_table.y_coord_metres: y-coordinate.
    """

    x_coords_metres, y_coords_metres = projections.project_latlng_to_xy(
        wind_table[raw_wind_io.LATITUDE_COLUMN].values,
        wind_table[raw_wind_io.LONGITUDE_COLUMN].values,
        projection_object=projection_object)

    argument_dict = {
        WIND_X_COLUMN: x_coords_metres, WIND_Y_COLUMN: y_coords_metres}
    return wind_table.assign(**argument_dict)


def _filter_winds_by_bounding_box(wind_table, x_limits_metres=None,
                                  y_limits_metres=None):
    """Removes wind observations outside of bounding box.

    :param wind_table: pandas DataFrame created by
        _project_winds_to_equidistant.
    :param x_limits_metres: length-2 numpy array with [min, max] x-coordinates
        of bounding box.
    :param y_limits_metres: length-2 numpy array with [min, max] y-coordinates
        of bounding box.
    :return: wind_table: Same as input, except with fewer rows.
    """

    error_checking.assert_is_numpy_array_without_nan(x_limits_metres)
    error_checking.assert_is_numpy_array(x_limits_metres,
                                         exact_dimensions=numpy.array([2]))
    error_checking.assert_is_greater(x_limits_metres[1], x_limits_metres[0])

    error_checking.assert_is_numpy_array_without_nan(y_limits_metres)
    error_checking.assert_is_numpy_array(y_limits_metres,
                                         exact_dimensions=numpy.array([2]))
    error_checking.assert_is_greater(y_limits_metres[1], y_limits_metres[0])

    x_invalid_flags = numpy.logical_or(
        wind_table[WIND_X_COLUMN].values < x_limits_metres[0],
        wind_table[WIND_X_COLUMN].values > x_limits_metres[1])
    y_invalid_flags = numpy.logical_or(
        wind_table[WIND_Y_COLUMN].values < y_limits_metres[0],
        wind_table[WIND_Y_COLUMN].values > y_limits_metres[1])

    invalid_rows = numpy.where(
        numpy.logical_or(x_invalid_flags, y_invalid_flags))[0]
    return wind_table.drop(wind_table.index[invalid_rows], axis=0,
                           inplace=False)


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
        storm_object_table[tracking_io.STORM_ID_COLUMN].values)
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
            storm_object_table[tracking_io.TIME_COLUMN].values[
                these_storm_object_indices])
        end_times_unix_sec[these_storm_object_indices] = numpy.max(
            storm_object_table[tracking_io.TIME_COLUMN].values[
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

    error_checking.assert_is_integer(max_start_time_unix_sec)
    error_checking.assert_is_not_nan(max_start_time_unix_sec)
    error_checking.assert_is_integer(min_end_time_unix_sec)
    error_checking.assert_is_not_nan(min_end_time_unix_sec)

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
        storm_object_table_1cell[tracking_io.TIME_COLUMN].values)
    centroid_matrix = numpy.vstack((
        storm_object_table_1cell[CENTROID_X_COLUMN].values[sort_indices],
        storm_object_table_1cell[CENTROID_Y_COLUMN].values[sort_indices]))

    interp_centroid_vector = interp.interp_in_time(
        centroid_matrix, sorted_input_times_unix_sec=storm_object_table_1cell[
            tracking_io.TIME_COLUMN].values[sort_indices],
        query_times_unix_sec=numpy.array([query_time_unix_sec]),
        allow_extrap=True)

    absolute_time_diffs_sec = numpy.absolute(
        storm_object_table_1cell[tracking_io.TIME_COLUMN].values -
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
        tracking_io.STORM_ID_COLUMN: storm_id_list,
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
        filtered_storm_object_table[tracking_io.STORM_ID_COLUMN].values)
    unique_storm_ids = numpy.unique(storm_id_numpy_array)

    list_of_tables = []
    num_storm_cells = len(unique_storm_ids)
    for i in range(num_storm_cells):
        this_storm_cell_table = (
            filtered_storm_object_table.loc[
                filtered_storm_object_table[
                    tracking_io.STORM_ID_COLUMN] == unique_storm_ids[i]])
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


def _find_nearest_storms_one_time(interp_vertex_table,
                                  wind_x_coords_metres=None,
                                  wind_y_coords_metres=None,
                                  max_distance_metres=None):
    """At one valid time, finds nearest storm to each wind observation.

    N = number of wind observations
    t = valid time

    :param interp_vertex_table: pandas DataFrame created by
        _interp_storms_in_time.
    :param wind_x_coords_metres: length-N numpy array with x-coordinates of wind
        observations.
    :param wind_y_coords_metres: length-N numpy array with y-coordinates of wind
        observations.
    :param max_distance_metres: Max linkage distance.  For the [k]th wind
        observation, if there is no storm within max_distance_metres,
        nearest_storm_ids[i] will be None.  In other words, the [k]th wind
        observation will not be linked to a storm cell.
    :return: nearest_storm_ids: length-N list with string ID of nearest storm
        cell to each wind observation.
    :return: linkage_distances_metres: length-N numpy array with distance from
        each wind observation to linked storm cell.  If [k]th wind observation
        is not linked to a storm cell, linkage_distances_metres[k] = NaN.
    """

    num_wind_observations = len(wind_x_coords_metres)
    nearest_storm_ids = [None] * num_wind_observations
    linkage_distances_metres = numpy.full(num_wind_observations, numpy.nan)

    for k in range(num_wind_observations):
        these_x_diffs_metres = numpy.absolute(
            wind_x_coords_metres[k] - interp_vertex_table[
                VERTEX_X_COLUMN].values)
        these_y_diffs_metres = numpy.absolute(
            wind_y_coords_metres[k] - interp_vertex_table[
                VERTEX_Y_COLUMN].values)

        these_valid_flags = numpy.logical_and(
            these_x_diffs_metres <= max_distance_metres,
            these_y_diffs_metres <= max_distance_metres)
        if not numpy.any(these_valid_flags):
            continue

        these_valid_indices = numpy.where(these_valid_flags)[0]
        these_distances_metres = numpy.sqrt(
            these_x_diffs_metres[these_valid_indices] ** 2 +
            these_y_diffs_metres[these_valid_indices] ** 2)

        this_min_index = these_valid_indices[
            numpy.argmin(these_distances_metres)]
        nearest_storm_ids[k] = interp_vertex_table[
            tracking_io.STORM_ID_COLUMN].values[this_min_index]

        this_storm_indices = [
            s == nearest_storm_ids[k] for s in interp_vertex_table[
                tracking_io.STORM_ID_COLUMN].values]
        this_polygon_object = polygons.vertex_arrays_to_polygon_object(
            interp_vertex_table[VERTEX_X_COLUMN].values[this_storm_indices],
            interp_vertex_table[VERTEX_Y_COLUMN].values[this_storm_indices])
        this_point_in_polygon_flag = polygons.is_point_in_or_on_polygon(
            this_polygon_object, query_x_coordinate=wind_x_coords_metres[k],
            query_y_coordinate=wind_y_coords_metres[k])

        if this_point_in_polygon_flag:
            linkage_distances_metres[k] = 0.
        else:
            linkage_distances_metres[k] = numpy.min(these_distances_metres)

    return nearest_storm_ids, linkage_distances_metres


def _find_nearest_storms(storm_object_table=None, wind_table=None,
                         max_time_before_storm_start_sec=None,
                         max_time_after_storm_end_sec=None,
                         max_link_dist_metres=None):
    """Finds nearest storm to each wind observation.

    N = number of wind observations

    :param storm_object_table: pandas DataFrame created by
        _storm_objects_to_cells.
    :param wind_table: pandas DataFrame created by
        _project_winds_to_equidistant.
    :param max_time_before_storm_start_sec: Max time before beginning of storm
        cell.  If wind observation W occurs > `max_time_before_storm_start_sec`
        before storm cell S begins, it cannot be linked to S.
    :param max_time_after_storm_end_sec: Max time after end of storm cell.  If
        wind observation W occurs > `max_time_after_storm_end_sec` after storm
        cell S ends, it cannot be linked to S.
    :param max_link_dist_metres: Max distance from storm cell.  If wind
        observation W occurs > `max_link_dist_metres` from the nearest vertex of
        storm cell S, it cannot be linked to S.
    :return: wind_table: Same as input, but with the following added columns.
    wind_table.nearest_storm_id: String ID for nearest storm cell.
    wind_table.linkage_distance_metres: Distance to nearest storm cell.
    """

    error_checking.assert_is_integer(max_time_before_storm_start_sec)
    error_checking.assert_is_geq(max_time_before_storm_start_sec, 0)
    error_checking.assert_is_integer(max_time_after_storm_end_sec)
    error_checking.assert_is_geq(max_time_after_storm_end_sec, 0)
    error_checking.assert_is_geq(max_link_dist_metres, 0)

    (unique_wind_times_unix_sec,
     wind_times_orig_to_unique) = numpy.unique(
         wind_table[raw_wind_io.TIME_COLUMN].values, return_inverse=True)

    num_wind_observations = len(wind_table.index)
    nearest_storm_ids = [None] * num_wind_observations
    linkage_distances_metres = numpy.full(num_wind_observations, numpy.nan)

    num_unique_times = len(unique_wind_times_unix_sec)
    for i in range(num_unique_times):
        these_wind_rows = numpy.where(wind_times_orig_to_unique == i)[0]
        this_interp_vertex_table = _interp_storms_in_time(
            storm_object_table,
            query_time_unix_sec=unique_wind_times_unix_sec[i],
            max_time_before_start_sec=max_time_before_storm_start_sec,
            max_time_after_end_sec=max_time_after_storm_end_sec)

        (these_nearest_storm_ids,
         these_link_distances_metres) = _find_nearest_storms_one_time(
             this_interp_vertex_table,
             wind_x_coords_metres=wind_table[WIND_X_COLUMN].values[
                 these_wind_rows],
             wind_y_coords_metres=wind_table[WIND_Y_COLUMN].values[
                 these_wind_rows],
             max_distance_metres=max_link_dist_metres)

        linkage_distances_metres[these_wind_rows] = these_link_distances_metres
        for j in range(len(these_wind_rows)):
            nearest_storm_ids[these_wind_rows[j]] = these_nearest_storm_ids[j]

    argument_dict = {NEAREST_STORM_ID_COLUMN: nearest_storm_ids,
                     LINKAGE_DISTANCE_COLUMN: linkage_distances_metres}
    return wind_table.assign(**argument_dict)


def _link_winds_and_storms(storm_object_table=None, wind_table=None):
    """Merges wind observations and storm cells.

    Specifically, for each storm cell S, creates a list of wind observations
    linked to S.

    K = number of wind observations linked to a given storm cell

    :param storm_object_table: pandas DataFrame created by
        _storm_objects_to_cells.
    :param wind_table: pandas DataFrame created by _find_nearest_storms.
    :return: linkage_table: Same as input storm_object_table, but
        with the following added columns.
    linkage_table.storm_id: String ID for storm cell.
    linkage_table.unix_time_sec: Time in Unix format.
    linkage_table.wind_station_ids: length-K list with string IDs
        of wind-reporting stations.
    linkage_table.wind_latitudes_deg: length-K numpy array with
        latitudes (deg N) of wind observations linked to storm cell.
    linkage_table.wind_longitudes_deg: length-K numpy array with
        longitudes (deg E) of wind observations linked to storm cell.
    linkage_table.u_winds_m_s01: length-K numpy array with u-
        velocities (m/s) of wind observations linked to storm cell.
    linkage_table.v_winds_m_s01: length-K numpy array with v-
        velocities (m/s) of wind observations linked to storm cell.
    linkage_table.linkage_distances_metres: length-K numpy array
        with distances of wind observations from storm object.
    linkage_table.relative_times_sec: length-K numpy array with
        relative times of wind observations (wind time minus storm-object time).
    """

    nested_array = storm_object_table[[
        tracking_io.STORM_ID_COLUMN,
        tracking_io.STORM_ID_COLUMN]].values.tolist()
    argument_dict = {
        STATION_IDS_COLUMN: nested_array, WIND_LATS_COLUMN: nested_array,
        WIND_LNGS_COLUMN: nested_array, U_WINDS_COLUMN: nested_array,
        V_WINDS_COLUMN: nested_array, LINKAGE_DISTANCES_COLUMN: nested_array,
        RELATIVE_TIMES_COLUMN: nested_array}

    linkage_table = copy.deepcopy(storm_object_table)
    linkage_table = linkage_table.assign(**argument_dict)

    num_storm_objects = len(storm_object_table.index)
    for i in range(num_storm_objects):
        linkage_table[STATION_IDS_COLUMN].values[i] = []
        linkage_table[WIND_LATS_COLUMN].values[i] = []
        linkage_table[WIND_LNGS_COLUMN].values[i] = []
        linkage_table[U_WINDS_COLUMN].values[i] = []
        linkage_table[V_WINDS_COLUMN].values[i] = []
        linkage_table[LINKAGE_DISTANCES_COLUMN].values[i] = []
        linkage_table[RELATIVE_TIMES_COLUMN].values[i] = []

    num_wind_observations = len(wind_table.index)
    for k in range(num_wind_observations):
        this_storm_id = wind_table[NEAREST_STORM_ID_COLUMN].values[k]
        if this_storm_id is None:
            continue

        this_storm_flags = [s == this_storm_id for s in storm_object_table[
            tracking_io.STORM_ID_COLUMN].values]
        this_storm_rows = numpy.where(this_storm_flags)[0]

        for i in range(len(this_storm_rows)):
            j = this_storm_rows[i]
            linkage_table[STATION_IDS_COLUMN].values[j].append(
                wind_table[raw_wind_io.STATION_ID_COLUMN].values[k])
            linkage_table[WIND_LATS_COLUMN].values[j].append(
                wind_table[raw_wind_io.LATITUDE_COLUMN].values[k])
            linkage_table[WIND_LNGS_COLUMN].values[j].append(
                wind_table[raw_wind_io.LONGITUDE_COLUMN].values[k])
            linkage_table[U_WINDS_COLUMN].values[j].append(
                wind_table[raw_wind_io.U_WIND_COLUMN].values[k])
            linkage_table[V_WINDS_COLUMN].values[j].append(
                wind_table[raw_wind_io.V_WIND_COLUMN].values[k])
            linkage_table[LINKAGE_DISTANCES_COLUMN].values[j].append(
                wind_table[LINKAGE_DISTANCE_COLUMN].values[k])

            this_relative_time_sec = (
                wind_table[raw_wind_io.TIME_COLUMN].values[k] -
                linkage_table[tracking_io.TIME_COLUMN].values[j])
            linkage_table[RELATIVE_TIMES_COLUMN].values[j].append(
                this_relative_time_sec)

    for i in range(num_storm_objects):
        linkage_table[WIND_LATS_COLUMN].values[i] = numpy.asarray(
            linkage_table[WIND_LATS_COLUMN].values[i])
        linkage_table[WIND_LNGS_COLUMN].values[i] = numpy.asarray(
            linkage_table[WIND_LNGS_COLUMN].values[i])
        linkage_table[U_WINDS_COLUMN].values[i] = numpy.asarray(
            linkage_table[U_WINDS_COLUMN].values[i])
        linkage_table[V_WINDS_COLUMN].values[i] = numpy.asarray(
            linkage_table[V_WINDS_COLUMN].values[i])
        linkage_table[LINKAGE_DISTANCES_COLUMN].values[i] = numpy.asarray(
            linkage_table[LINKAGE_DISTANCES_COLUMN].values[i])
        linkage_table[RELATIVE_TIMES_COLUMN].values[i] = numpy.asarray(
            linkage_table[RELATIVE_TIMES_COLUMN].values[i])

    return linkage_table


def _write_linkages_to_pickle(linkage_table, out_file_names):
    """Writes linkages (storm-wind associations) to Pickle files.

    The number of output files (length of out_file_names) should = the number of
    input files (see _read_storm_objects).

    N = number of output files

    :param linkage_table: pandas DataFrame created by _link_winds_and_storms.
    :param out_file_names: length-N list of paths to output files.
    """

    error_checking.assert_is_list(out_file_names)

    max_file_index = numpy.max(linkage_table[FILE_INDEX_COLUMN].values)
    num_linkage_files = len(out_file_names)
    error_checking.assert_is_greater(num_linkage_files, max_file_index)

    for i in range(num_linkage_files):
        file_system_utils.mkdir_recursive_if_necessary(
            file_name=out_file_names[i])

        this_linkage_table = linkage_table.loc[
            linkage_table[FILE_INDEX_COLUMN] == i]
        this_file_handle = open(out_file_names[i], 'wb')
        pickle.dump(this_linkage_table, this_file_handle)
        this_file_handle.close()


def read_linkages_from_pickle(input_file_name):
    """Writes linkages (storm-wind associations) from single Pickle file.

    :param input_file_name: Path to input file.
    :return: linkage_table: pandas DataFrame with columns generated by
        _link_winds_and_storms.
    """

    input_file_handle = open(input_file_name, 'rb')
    linkage_table = pickle.load(input_file_handle)
    input_file_handle.close()
    return linkage_table


if __name__ == '__main__':
    PROCESSED_SEGMOTION_FILE_NAMES = glob.glob(
        PROCESSED_SEGMOTION_DIR_NAME + '/*.p')
    STORM_OBJECT_TABLE = _read_storm_objects(PROCESSED_SEGMOTION_FILE_NAMES)
    STORM_OBJECT_TABLE, _ = _project_storms_to_equidistant(STORM_OBJECT_TABLE)
    STORM_OBJECT_TABLE = _storm_objects_to_cells(STORM_OBJECT_TABLE)
    print STORM_OBJECT_TABLE

    INTERP_VERTEX_TABLE = _interp_storms_in_time(
        STORM_OBJECT_TABLE, query_time_unix_sec=QUERY_TIME_UNIX_SEC,
        max_time_before_start_sec=MAX_TIME_BEFORE_START_SEC,
        max_time_after_end_sec=MAX_TIME_AFTER_END_SEC)
    print INTERP_VERTEX_TABLE
