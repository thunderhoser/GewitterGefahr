"""Methods for linking wind observations to storm cells.

DEFINITIONS

"Storm cell" = a single thunderstorm (standard meteorological definition).  I
will often use S to denote one storm cell.

"Storm object" = one storm cell at one time step.  I will often use s to denote
one storm object.
"""

import copy
import pickle
import numpy
import pandas
from gewittergefahr.gg_io import storm_tracking_io as tracking_io
from gewittergefahr.gg_io import raw_wind_io
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import number_rounding as rounder
from gewittergefahr.gg_utils import projections
from gewittergefahr.gg_utils import interp
from gewittergefahr.gg_utils import polygons
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.gg_utils import error_checking

# TODO(thunderhoser): Add high-level method that does everything.

WIND_DATA_SOURCE = raw_wind_io.MERGED_DATA_SOURCE
MAX_TIME_BEFORE_STORM_START_DEFAULT_SEC = 300
MAX_TIME_AFTER_STORM_END_DEFAULT_SEC = 300
PADDING_FOR_STORM_BOUNDING_BOX_DEFAULT_METRES = 10000.
INTERP_TIME_SPACING_DEFAULT_SEC = 10
MAX_LINKAGE_DIST_DEFAULT_METRES = 30000.

STORM_COLUMNS_TO_READ = [
    tracking_io.STORM_ID_COLUMN, tracking_io.TIME_COLUMN,
    tracking_io.TRACKING_START_TIME_COLUMN,
    tracking_io.TRACKING_END_TIME_COLUMN, tracking_io.CENTROID_LAT_COLUMN,
    tracking_io.CENTROID_LNG_COLUMN, tracking_io.POLYGON_OBJECT_LATLNG_COLUMN]

WIND_COLUMNS_TO_READ = [
    raw_wind_io.STATION_ID_COLUMN, raw_wind_io.LATITUDE_COLUMN,
    raw_wind_io.LONGITUDE_COLUMN, raw_wind_io.TIME_COLUMN,
    raw_wind_io.U_WIND_COLUMN, raw_wind_io.V_WIND_COLUMN]

FILE_INDEX_COLUMN = 'file_index'
CENTROID_X_COLUMN = 'centroid_x_metres'
CENTROID_Y_COLUMN = 'centroid_y_metres'
VERTICES_X_COLUMN = 'vertices_x_metres'
VERTICES_Y_COLUMN = 'vertices_y_metres'
START_TIME_COLUMN = 'start_time_unix_sec'
END_TIME_COLUMN = 'end_time_unix_sec'

WIND_X_COLUMN = 'x_coord_metres'
WIND_Y_COLUMN = 'y_coord_metres'
NEAREST_STORM_ID_COLUMN = 'nearest_storm_id'
LINKAGE_DISTANCE_COLUMN = 'linkage_distance_metres'

VERTEX_X_COLUMN = 'vertex_x_metres'
VERTEX_Y_COLUMN = 'vertex_y_metres'

STATION_IDS_COLUMN = 'wind_station_ids'
WIND_LATITUDES_COLUMN = 'wind_latitudes_deg'
WIND_LONGITUDES_COLUMN = 'wind_longitudes_deg'
U_WINDS_COLUMN = 'u_winds_m_s01'
V_WINDS_COLUMN = 'v_winds_m_s01'
LINKAGE_DISTANCES_COLUMN = 'wind_distances_metres'
RELATIVE_TIMES_COLUMN = 'relative_wind_times_sec'

COLUMNS_TO_WRITE = STORM_COLUMNS_TO_READ + [
    STATION_IDS_COLUMN, WIND_LATITUDES_COLUMN, WIND_LONGITUDES_COLUMN,
    U_WINDS_COLUMN, V_WINDS_COLUMN, LINKAGE_DISTANCES_COLUMN,
    RELATIVE_TIMES_COLUMN
]


def _get_xy_bounding_box_of_storms(
        storm_object_table,
        padding_metres=PADDING_FOR_STORM_BOUNDING_BOX_DEFAULT_METRES):
    """Returns bounding box of all storm objects.

    :param storm_object_table: pandas DataFrame created by
        _project_storms_latlng_to_xy.
    :param padding_metres: This padding will be added to the edge of the
        bounding box.
    :return: x_limits_metres: length-2 numpy array with [min, max] x-coordinates
        of bounding box.
    :return: y_limits_metres: length-2 numpy array with [min, max] y-coordinates
        of bounding box.
    """

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


def _init_azimuthal_equidistant_projection(storm_centroid_latitudes_deg,
                                           storm_centroid_longitudes_deg):
    """Initializes azimuthal equidistant projection.

    This projection will be centered at the "global centroid" (centroid of
    centroids) of all storm objects.

    N = number of storm objects

    :param storm_centroid_latitudes_deg: length-N numpy array with latitudes
        (deg N) of storm centroids.
    :param storm_centroid_longitudes_deg: length-N numpy array with longitudes
        (deg E) of storm centroids.
    :return: projection_object: Instance of `pyproj.Proj`, which can be used to
        convert between lat-long and x-y coordinates.
    """

    (global_centroid_lat_deg,
     global_centroid_lng_deg) = polygons.get_latlng_centroid(
         storm_centroid_latitudes_deg, storm_centroid_longitudes_deg)

    return projections.init_azimuthal_equidistant_projection(
        global_centroid_lat_deg, global_centroid_lng_deg)


def _project_storms_latlng_to_xy(storm_object_table, projection_object):
    """Projects storm objects from lat-long to x-y coordinates.

    This method projects both centroids and outlines.

    V = number of vertices in a given storm object

    :param storm_object_table: pandas DataFrame created by _read_storm_objects.
    :param projection_object: Instance of `pyproj.Proj`, created by
        _init_azimuthal_equidistant_projection.
    :return: storm_object_table: Same as input, but with additional columns
        listed below.
    storm_object_table.centroid_x_metres: x-coordinate of centroid.
    storm_object_table.centroid_y_metres: y-coordinate of centroid.
    storm_object_table.vertices_x_metres: length-V numpy array with x-
        coordinates of vertices.
    storm_object_table.vertices_y_metres: length-V numpy array with y-
        coordinates of vertices.
    """

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

    return storm_object_table


def _project_winds_latlng_to_xy(wind_table, projection_object):
    """Projects wind observations from lat-long to x-y coordinates.

    :param wind_table: pandas DataFrame created by _read_wind_observations.
    :param projection_object: Instance of `pyproj.Proj`, created by
        _init_azimuthal_equidistant_projection.
    :return: wind_table: Same as input, but with additional columns listed
        below.
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

    :param wind_table: pandas DataFrame created by _project_winds_latlng_to_xy.
    :param x_limits_metres: length-2 numpy array with [min, max] x-coordinates
        of bounding box.
    :param y_limits_metres: length-2 numpy array with [min, max] y-coordinates
        of bounding box.
    :return: wind_table: Same as input, except with fewer rows.
    """

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
        _project_storms_latlng_to_xy.
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
    :param max_start_time_unix_sec: Latest start time of any storm cell.
    :param min_end_time_unix_sec: Earliest end time of any storm cell.
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

    The storm object nearest to the query time is advected as a whole -- i.e.,
    all vertices are moved by the same distance and in the same direction -- so
    that the interpolated storm object has a realistic shape.

    N = number of storm objects (snapshots of storm cell)
    V = number of vertices in a given storm object

    :param storm_object_table_1cell: N-row pandas DataFrame with the following
        columns.
    storm_object_table_1cell.unix_time_sec: Time of snapshot.
    storm_object_table_1cell.centroid_x_metres: x-coordinate of centroid.
    storm_object_table_1cell.centroid_y_metres: y-coordinate of centroid.
    storm_object_table_1cell.vertices_x_metres: length-V numpy array with x-
        coordinates of vertices.
    storm_object_table_1cell.vertices_y_metres: length-V numpy array with y-
        coordinates of vertices.
    :param storm_id: String ID for storm cell.
    :param query_time_unix_sec: Storm location will be interpolated to this
        time.
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

    :param storm_object_table: pandas DataFrame created by
        _storm_objects_to_cells.
    :param query_time_unix_sec: Storm locations will be interpolated to this
        time.
    :param max_time_before_start_sec: Max time before beginning of storm cell.
        For each storm cell S, if query time is > max_time_before_start_sec
        before first time in S, this method will not bother interpolating S.
    :param max_time_after_end_sec: Max time after end of storm cell.  For each
        storm cell S, if query time is > max_time_after_end_sec after last in S,
        this method will not bother interpolating S.
    :return: interp_vertex_table: pandas DataFrame with the following columns
        (each row is one vertex of an interpolated storm object).
    interp_vertex_table_1object.storm_id: String ID for storm cell.
    interp_vertex_table_1object.vertex_x_metres: x-coordinate of vertex.
    interp_vertex_table_1object.vertex_y_metres: y-coordinate of vertex.
    """

    print time_conversion.unix_sec_to_string(query_time_unix_sec, '%Y-%m-%d-%H%M%S')

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

    if not list_of_tables:
        return pandas.DataFrame(
            columns=[
                tracking_io.STORM_ID_COLUMN, VERTEX_X_COLUMN, VERTEX_Y_COLUMN])

    return pandas.concat(list_of_tables, axis=0, ignore_index=True)


def _find_nearest_storms_at_one_time(interp_vertex_table,
                                     wind_x_coords_metres=None,
                                     wind_y_coords_metres=None,
                                     max_linkage_dist_metres=None):
    """Finds nearest storm cell to each wind observation at one time step.

    N = number of wind observations

    :param interp_vertex_table: pandas DataFrame created by
        _interp_storms_in_time.
    :param wind_x_coords_metres: length-N numpy array with x-coordinates of wind
        observations.
    :param wind_y_coords_metres: length-N numpy array with y-coordinates of wind
        observations.
    :param max_linkage_dist_metres: Max linkage distance.  For the [i]th wind
        observation, if the nearest storm cell is > max_linkage_dist_metres
        away, nearest_storm_ids[i] will be None.  In other words, the [i]th wind
        observation will not be linked to a storm cell.
    :return: nearest_storm_ids: length-N list of strings.  nearest_storm_ids[i]
        is the ID of the nearest storm cell to the [i]th wind observation.
    :return: linkage_distances_metres: length-N numpy array.
        linkage_distances_metres[i] is the distance between the [i]th wind
        observation and the nearest storm cell.  If nearest_storm_ids[i] = None,
        linkage_distances_metres[i] = NaN.
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
            these_x_diffs_metres <= max_linkage_dist_metres,
            these_y_diffs_metres <= max_linkage_dist_metres)
        if not numpy.any(these_valid_flags):
            continue

        these_valid_indices = numpy.where(these_valid_flags)[0]
        these_distances_metres = numpy.sqrt(
            these_x_diffs_metres[these_valid_indices] ** 2 +
            these_y_diffs_metres[these_valid_indices] ** 2)
        if not numpy.any(these_distances_metres <= max_linkage_dist_metres):
            continue

        this_min_index = these_valid_indices[
            numpy.argmin(these_distances_metres)]
        nearest_storm_ids[k] = interp_vertex_table[
            tracking_io.STORM_ID_COLUMN].values[this_min_index]

        this_storm_flags = [
            s == nearest_storm_ids[k] for s in interp_vertex_table[
                tracking_io.STORM_ID_COLUMN].values]
        this_storm_indices = numpy.where(this_storm_flags)[0]

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


def _find_nearest_storms(
        storm_object_table=None, wind_table=None,
        max_time_before_storm_start_sec=None, max_time_after_storm_end_sec=None,
        max_linkage_dist_metres=None,
        interp_time_spacing_sec=INTERP_TIME_SPACING_DEFAULT_SEC):
    """Finds nearest storm cell to each wind observation.

    N = number of wind observations

    :param storm_object_table: pandas DataFrame created by
        _storm_objects_to_cells.
    :param wind_table: pandas DataFrame created by _project_winds_latlng_to_xy.
    :param max_time_before_storm_start_sec: Max time before beginning of storm
        cell.  If wind observation W occurs > max_time_before_storm_start_sec
        before the first time in storm cell S, W cannot be linked to S.
    :param max_time_after_storm_end_sec: Max time after end of storm cell.  If
        wind observation W occurs > max_time_after_storm_end_sec after the last
        time in storm cell S, W cannot be linked to S.
    :param max_linkage_dist_metres: Max linkage distance.  If wind observation W
        is > max_linkage_dist_metres from the nearest storm cell, W will not be
        linked to a storm cell.
    :param interp_time_spacing_sec: Discretization time for interpolation.  If
        interp_time_spacing_sec = 1, storms will be interpolated to each unique
        wind time (since wind times are rounded to the nearest second).  If
        interp_time_spacing_sec = q, storms will be interpolated to each unique
        multiple of q seconds among wind times.
    :return: wind_to_storm_table: Same as input, but with additional columns
        listed below.
    wind_to_storm_table.nearest_storm_id: String ID for nearest storm cell.  If
        the wind observation is not linked to a storm cell, this will be None.
    wind_to_storm_table.linkage_distance_metres: Distance to nearest storm cell.
        If the wind observation is not linked to a storm cell, this will be NaN.
    """

    rounded_times_unix_sec = rounder.round_to_nearest(
        wind_table[raw_wind_io.TIME_COLUMN].values, interp_time_spacing_sec)
    unique_rounded_times_unix_sec, wind_times_orig_to_unique_rounded = (
        numpy.unique(rounded_times_unix_sec, return_inverse=True))

    num_wind_observations = len(wind_table.index)
    nearest_storm_ids = [None] * num_wind_observations
    linkage_distances_metres = numpy.full(num_wind_observations, numpy.nan)

    num_unique_times = len(unique_rounded_times_unix_sec)
    for i in range(num_unique_times):
        these_wind_rows = numpy.where(wind_times_orig_to_unique_rounded == i)[0]
        this_interp_vertex_table = _interp_storms_in_time(
            storm_object_table,
            query_time_unix_sec=unique_rounded_times_unix_sec[i],
            max_time_before_start_sec=max_time_before_storm_start_sec,
            max_time_after_end_sec=max_time_after_storm_end_sec)

        these_nearest_storm_ids, these_link_distances_metres = (
            _find_nearest_storms_at_one_time(
                this_interp_vertex_table,
                wind_x_coords_metres=wind_table[WIND_X_COLUMN].values[
                    these_wind_rows],
                wind_y_coords_metres=wind_table[WIND_Y_COLUMN].values[
                    these_wind_rows],
                max_linkage_dist_metres=max_linkage_dist_metres))

        linkage_distances_metres[these_wind_rows] = these_link_distances_metres
        for j in range(len(these_wind_rows)):
            nearest_storm_ids[these_wind_rows[j]] = these_nearest_storm_ids[j]

    argument_dict = {NEAREST_STORM_ID_COLUMN: nearest_storm_ids,
                     LINKAGE_DISTANCE_COLUMN: linkage_distances_metres}
    return wind_table.assign(**argument_dict)


def _create_storm_to_winds_table(storm_object_table, wind_to_storm_table):
    """For each storm cell S, creates list of wind observations linked to S.

    N = number of storm objects
    K = number of wind observations
    k = number of wind observations linked to a given storm cell

    :param storm_object_table: N-row pandas DataFrame created by
        _storm_objects_to_cells.
    :param wind_to_storm_table: K-row pandas DataFrame created by
        _find_nearest_storms.
    :return: storm_to_winds_table: Same as input, but with additional columns
        listed below.
    storm_to_winds_table.wind_station_ids: length-K list with string IDs of wind
        stations.
    storm_to_winds_table.wind_latitudes_deg: length-K numpy array with latitudes
        (deg N) of wind stations.
    storm_to_winds_table.wind_longitudes_deg: length-K numpy array with
        longitudes (deg E) of wind stations.
    storm_to_winds_table.u_winds_m_s01: length-K numpy array with u-components
        (metres per second) of wind velocities.
    storm_to_winds_table.v_winds_m_s01: length-K numpy array with v-components
        (metres per second) of wind velocities.
    storm_to_winds_table.wind_distances_metres: length-K numpy array with
        distances of wind observations from storm object.
    storm_to_winds_table.relative_wind_times_sec: length-K numpy array with
        relative times of wind observations (wind time minus storm-object time).
    """

    nested_array = storm_object_table[[
        tracking_io.STORM_ID_COLUMN,
        tracking_io.STORM_ID_COLUMN]].values.tolist()
    argument_dict = {
        STATION_IDS_COLUMN: nested_array, WIND_LATITUDES_COLUMN: nested_array,
        WIND_LONGITUDES_COLUMN: nested_array, U_WINDS_COLUMN: nested_array,
        V_WINDS_COLUMN: nested_array, LINKAGE_DISTANCES_COLUMN: nested_array,
        RELATIVE_TIMES_COLUMN: nested_array}

    storm_to_winds_table = copy.deepcopy(storm_object_table)
    storm_to_winds_table = storm_to_winds_table.assign(**argument_dict)
    num_storm_objects = len(storm_to_winds_table.index)

    for i in range(num_storm_objects):
        storm_to_winds_table[STATION_IDS_COLUMN].values[i] = []
        storm_to_winds_table[WIND_LATITUDES_COLUMN].values[i] = []
        storm_to_winds_table[WIND_LONGITUDES_COLUMN].values[i] = []
        storm_to_winds_table[U_WINDS_COLUMN].values[i] = []
        storm_to_winds_table[V_WINDS_COLUMN].values[i] = []
        storm_to_winds_table[LINKAGE_DISTANCES_COLUMN].values[i] = []
        storm_to_winds_table[RELATIVE_TIMES_COLUMN].values[i] = []

    num_wind_observations = len(wind_to_storm_table.index)
    for k in range(num_wind_observations):
        this_storm_id = wind_to_storm_table[NEAREST_STORM_ID_COLUMN].values[k]
        if this_storm_id is None:
            continue

        this_storm_flags = [s == this_storm_id for s in storm_to_winds_table[
            tracking_io.STORM_ID_COLUMN].values]
        this_storm_rows = numpy.where(this_storm_flags)[0]

        for i in range(len(this_storm_rows)):
            j = this_storm_rows[i]
            storm_to_winds_table[STATION_IDS_COLUMN].values[j].append(
                wind_to_storm_table[raw_wind_io.STATION_ID_COLUMN].values[k])
            storm_to_winds_table[WIND_LATITUDES_COLUMN].values[j].append(
                wind_to_storm_table[raw_wind_io.LATITUDE_COLUMN].values[k])
            storm_to_winds_table[WIND_LONGITUDES_COLUMN].values[j].append(
                wind_to_storm_table[raw_wind_io.LONGITUDE_COLUMN].values[k])
            storm_to_winds_table[U_WINDS_COLUMN].values[j].append(
                wind_to_storm_table[raw_wind_io.U_WIND_COLUMN].values[k])
            storm_to_winds_table[V_WINDS_COLUMN].values[j].append(
                wind_to_storm_table[raw_wind_io.V_WIND_COLUMN].values[k])
            storm_to_winds_table[LINKAGE_DISTANCES_COLUMN].values[j].append(
                wind_to_storm_table[LINKAGE_DISTANCE_COLUMN].values[k])

            this_relative_time_sec = (
                wind_to_storm_table[raw_wind_io.TIME_COLUMN].values[k] -
                storm_to_winds_table[tracking_io.TIME_COLUMN].values[j])
            storm_to_winds_table[RELATIVE_TIMES_COLUMN].values[j].append(
                this_relative_time_sec)

    for i in range(num_storm_objects):
        storm_to_winds_table[WIND_LATITUDES_COLUMN].values[i] = numpy.asarray(
            storm_to_winds_table[WIND_LATITUDES_COLUMN].values[i])
        storm_to_winds_table[WIND_LONGITUDES_COLUMN].values[i] = numpy.asarray(
            storm_to_winds_table[WIND_LONGITUDES_COLUMN].values[i])
        storm_to_winds_table[U_WINDS_COLUMN].values[i] = numpy.asarray(
            storm_to_winds_table[U_WINDS_COLUMN].values[i])
        storm_to_winds_table[V_WINDS_COLUMN].values[i] = numpy.asarray(
            storm_to_winds_table[V_WINDS_COLUMN].values[i])
        storm_to_winds_table[LINKAGE_DISTANCES_COLUMN].values[i] = (
            numpy.asarray(
                storm_to_winds_table[LINKAGE_DISTANCES_COLUMN].values[i]))
        storm_to_winds_table[RELATIVE_TIMES_COLUMN].values[i] = numpy.asarray(
            storm_to_winds_table[RELATIVE_TIMES_COLUMN].values[i])

    return storm_to_winds_table


def _read_wind_observations(storm_object_table,
                            max_time_before_storm_start_sec=None,
                            max_time_after_storm_end_sec=None,
                            top_directory_name=None):
    """Reads wind observations to link with storm objects.

    :param storm_object_table: pandas DataFrame created by _read_storm_objects.
    :param max_time_before_storm_start_sec: Max wind time before beginning of
        storm cell.  If wind observation W occurs >
        max_time_before_storm_start_sec before the first time in storm cell S, W
        cannot be linked to S.
    :param max_time_after_storm_end_sec: Max wind time after end of storm cell.
        If wind observation W occurs > max_time_after_storm_end_sec after the
        last time in storm cell S, W cannot be linked to S.
    :param top_directory_name: Name of top-level directory with processed wind
        files (created by `raw_wind_io.write_processed_file`.
    :return: wind_table: pandas DataFrame with columns documented in
        `raw_wind_io.write_processed_file`.
    """

    min_wind_time_unix_sec = numpy.min(
        storm_object_table[tracking_io.TIME_COLUMN].values
    ) - max_time_before_storm_start_sec
    max_wind_time_unix_sec = numpy.max(
        storm_object_table[tracking_io.TIME_COLUMN].values
    ) + max_time_after_storm_end_sec

    wind_file_names, _ = raw_wind_io.find_processed_hourly_files(
        start_time_unix_sec=min_wind_time_unix_sec,
        end_time_unix_sec=max_wind_time_unix_sec,
        primary_source=WIND_DATA_SOURCE, top_directory_name=top_directory_name,
        raise_error_if_missing=True)

    num_wind_files = len(wind_file_names)
    list_of_wind_tables = [None] * num_wind_files

    for i in range(num_wind_files):
        print ('Reading wind file ' + str(i + 1) + '/' + str(num_wind_files) +
               ': "' + wind_file_names[i] + '"...')

        list_of_wind_tables[i] = raw_wind_io.read_processed_file(
            wind_file_names[i])[WIND_COLUMNS_TO_READ]
        if i == 0:
            continue

        list_of_wind_tables[i], _ = list_of_wind_tables[i].align(
            list_of_wind_tables[0], axis=1)

    print '\n'
    return pandas.concat(list_of_wind_tables, axis=0, ignore_index=True)


def _read_storm_objects(processed_file_names):
    """Reads storm objects to link with wind observations.

    N = number of storm objects

    :param processed_file_names: 1-D list of paths to processed files (created
        by `storm_tracking_io.write_processed_file`).
    :return: storm_object_table: N-row pandas DataFrame with the following
        columns.
    storm_object_table.storm_id: String ID for storm cell.
    storm_object_table.unix_time_sec: Valid time.
    storm_object_table.tracking_start_time_unix_sec: Start time for tracking
        period.
    storm_object_table.tracking_end_time_unix_sec: End time for tracking period.
    storm_object_table.centroid_lat_deg: Latitude (deg N) of centroid.
    storm_object_table.centroid_lng_deg: Longitude (deg E) of centroid.
    storm_object_table.polygon_object_latlng: Instance of
        `shapely.geometry.Polygon`, with vertices in lat-long coordinates.
    storm_object_table.file_index: Index of file from which storm object was
        read.  If file_index = j at the [i]th row, this means the [i]th storm
        object was read from the [j]th input file.
    """

    file_indices = numpy.array([])
    num_files = len(processed_file_names)
    list_of_storm_object_tables = [None] * num_files

    for i in range(num_files):
        print ('Reading storm-object file ' + str(i + 1) + '/' +
               str(num_files) + ': "' + processed_file_names[i] + '"...')

        list_of_storm_object_tables[i] = tracking_io.read_processed_file(
            processed_file_names[i])[STORM_COLUMNS_TO_READ]

        this_num_storm_objects = len(list_of_storm_object_tables[i].index)
        file_indices = numpy.concatenate((
            file_indices,
            numpy.linspace(i, i, num=this_num_storm_objects, dtype=int)))

        if i == 0:
            continue

        list_of_storm_object_tables[i], _ = (
            list_of_storm_object_tables[i].align(
                list_of_storm_object_tables[0], axis=1))

    print '\n'
    storm_object_table = pandas.concat(list_of_storm_object_tables, axis=0,
                                       ignore_index=True)
    argument_dict = {FILE_INDEX_COLUMN: file_indices}
    return storm_object_table.assign(**argument_dict)


def link_each_storm_to_winds(
        storm_object_file_names=None, top_wind_directory_name=None,
        max_time_before_storm_start_sec=MAX_TIME_BEFORE_STORM_START_DEFAULT_SEC,
        max_time_after_storm_end_sec=MAX_TIME_AFTER_STORM_END_DEFAULT_SEC,
        padding_for_storm_bounding_box_metres=
        PADDING_FOR_STORM_BOUNDING_BOX_DEFAULT_METRES,
        interp_time_spacing_sec=INTERP_TIME_SPACING_DEFAULT_SEC,
        max_linkage_dist_metres=MAX_LINKAGE_DIST_DEFAULT_METRES):
    """Links each storm cell to zero or more wind observations.

    :param storm_object_file_names: 1-D list of paths to storm-object files
        (readable by `storm_tracking_io.read_processed_file`).
    :param top_wind_directory_name: Name of top-level directory with processed
        wind files (see _read_wind_observations for more).
    :param max_time_before_storm_start_sec: [integer] Max wind time before
        beginning of storm cell.  If wind observation W occurs >
        max_time_before_storm_start_sec before the first time in storm cell S,
        W cannot be linked to S.
    :param max_time_after_storm_end_sec: [integer] Max wind time after end of
        storm cell.  If wind observation W occurs > max_time_after_storm_end_sec
        after the last time in storm cell S, W cannot be linked to S.
    :param padding_for_storm_bounding_box_metres: All wind observations outside
        a certain box will be thrown out.  If
        padding_for_storm_bounding_box_metres = 0, this will be the bounding box
        of all storm objects.  However, if max_time_before_storm_start_sec > 0
        or max_time_after_storm_end_sec > 0, you should make
        padding_for_storm_bounding_box_metres != 0.  This will allow for
        extrapolation of a storm cell outside its actual lifetime.
    :param interp_time_spacing_sec: Discretization time for interpolation.  If
        interp_time_spacing_sec = 1, storms will be interpolated to each unique
        wind time (since wind times are rounded to the nearest second).  If
        interp_time_spacing_sec = q, storms will be interpolated to each unique
        multiple of q seconds among wind times.  Making interp_time_spacing_sec
        > 1 makes the results less accurate but saves computing time.
    :param max_linkage_dist_metres: Max linkage distance.  If the nearest storm
        to wind observation W is > max_linkage_dist_metres away, W will not be
        linked to any storm cells.
    :return: storm_to_winds_table: pandas DataFrame created by
        _create_storm_to_winds_table.
    """

    error_checking.assert_is_string_list(storm_object_file_names)
    error_checking.assert_is_numpy_array(
        numpy.asarray(storm_object_file_names), num_dimensions=1)

    error_checking.assert_is_integer(max_time_before_storm_start_sec)
    error_checking.assert_is_geq(max_time_before_storm_start_sec, 0)
    error_checking.assert_is_integer(max_time_after_storm_end_sec)
    error_checking.assert_is_geq(max_time_after_storm_end_sec, 0)
    error_checking.assert_is_geq(padding_for_storm_bounding_box_metres, 0.)
    error_checking.assert_is_geq(max_linkage_dist_metres, 0.)

    storm_object_table = _read_storm_objects(storm_object_file_names)
    wind_table = _read_wind_observations(
        storm_object_table,
        max_time_before_storm_start_sec=max_time_before_storm_start_sec,
        max_time_after_storm_end_sec=max_time_after_storm_end_sec,
        top_directory_name=top_wind_directory_name)

    projection_object = _init_azimuthal_equidistant_projection(
        storm_object_table[tracking_io.CENTROID_LAT_COLUMN].values,
        storm_object_table[tracking_io.CENTROID_LNG_COLUMN].values)

    storm_object_table = _project_storms_latlng_to_xy(
        storm_object_table, projection_object)
    wind_table = _project_winds_latlng_to_xy(wind_table, projection_object)

    wind_x_limits_metres, wind_y_limits_metres = _get_xy_bounding_box_of_storms(
        storm_object_table,
        padding_metres=padding_for_storm_bounding_box_metres)
    wind_table = _filter_winds_by_bounding_box(
        wind_table, wind_x_limits_metres, wind_y_limits_metres)

    storm_object_table = _storm_objects_to_cells(storm_object_table)
    wind_to_storm_table = _find_nearest_storms(
        storm_object_table=storm_object_table, wind_table=wind_table,
        max_time_before_storm_start_sec=max_time_before_storm_start_sec,
        max_time_after_storm_end_sec=max_time_after_storm_end_sec,
        max_linkage_dist_metres=max_linkage_dist_metres,
        interp_time_spacing_sec=interp_time_spacing_sec)

    return _create_storm_to_winds_table(storm_object_table, wind_to_storm_table)


def write_storm_to_winds_table(storm_to_winds_table, pickle_file_names):
    """Writes linkages (storm-to-wind assocations) to one or more Pickle files.

    N = number of output files (should equal number of input files to
        _read_storm_objects).

    K = number of wind observations linked to a given storm cell

    :param storm_to_winds_table: pandas DataFrame with the following columns,
        where each row is one storm object.
    storm_to_winds_table.storm_id: String ID for storm cell.
    storm_to_winds_table.unix_time_sec: Valid time.
    storm_to_winds_table.tracking_start_time_unix_sec: Start time for tracking
        period.
    storm_to_winds_table.tracking_end_time_unix_sec: End time for tracking
        period.
    storm_to_winds_table.centroid_lat_deg: Latitude (deg N) of storm-object
        centroid.
    storm_to_winds_table.centroid_lng_deg: Longitude (deg E) of storm-object
        centroid.
    storm_to_winds_table.polygon_object_latlng: Instance of
        `shapely.geometry.Polygon`, with vertices of storm object in lat-long
        coordinates.
    storm_to_winds_table.wind_station_ids: length-K list with string IDs of wind
        stations.
    storm_to_winds_table.wind_latitudes_deg: length-K numpy array with latitudes
        (deg N) of wind stations.
    storm_to_winds_table.wind_longitudes_deg: length-K numpy array with
        longitudes (deg E) of wind stations.
    storm_to_winds_table.u_winds_m_s01: length-K numpy array with u-components
        (metres per second) of wind velocities.
    storm_to_winds_table.v_winds_m_s01: length-K numpy array with v-components
        (metres per second) of wind velocities.
    storm_to_winds_table.wind_distances_metres: length-K numpy array with
        distances of wind observations from storm object.
    storm_to_winds_table.relative_wind_times_sec: length-K numpy array with
        relative times of wind observations (wind time minus storm-object time).

    :param pickle_file_names: length-N list of paths to output files.
    """

    error_checking.assert_is_string_list(pickle_file_names)
    error_checking.assert_is_numpy_array(numpy.asarray(pickle_file_names),
                                         num_dimensions=1)

    max_file_index = numpy.max(storm_to_winds_table[FILE_INDEX_COLUMN].values)
    num_files = len(pickle_file_names)
    error_checking.assert_is_greater(num_files, max_file_index)

    for i in range(num_files):
        print ('Writing storm-to-winds file ' + str(i + 1) + '/' +
               str(num_files) + ': "' + pickle_file_names[i] + '"...')

        file_system_utils.mkdir_recursive_if_necessary(
            file_name=pickle_file_names[i])

        this_file_handle = open(pickle_file_names[i], 'wb')
        pickle.dump(
            storm_to_winds_table.loc[
                storm_to_winds_table[FILE_INDEX_COLUMN] == i][COLUMNS_TO_WRITE],
            this_file_handle)
        this_file_handle.close()


def read_storm_to_winds_table(pickle_file_name):
    """Reads linkages (storm-to-wind assocations) from Pickle file.

    :param pickle_file_name: Path to input file.
    :return: storm_to_winds_table: pandas DataFrame with columns documented in
        write_storm_to_winds_table.
    """

    pickle_file_handle = open(pickle_file_name, 'rb')
    storm_to_winds_table = pickle.load(pickle_file_handle)
    pickle_file_handle.close()

    error_checking.assert_columns_in_dataframe(
        storm_to_winds_table, COLUMNS_TO_WRITE)
    return storm_to_winds_table
