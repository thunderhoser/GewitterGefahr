"""Links hazardous events to storm cells.

Currently, the "hazardous events" handled by this module are tornadoes and
damaging straight-line wind.  Others (e.g., hail) may be added.

--- DEFINITIONS ---

"Storm cell" = a single thunderstorm (standard meteorological definition).  I
will often use S to denote a storm cell.

"Storm object" = one storm cell at one time step.  I will often use s to denote
a storm object.
"""

import copy
import os.path
import shutil
import pickle
import warnings
import numpy
import pandas
from gewittergefahr.gg_io import storm_tracking_io as tracking_io
from gewittergefahr.gg_io import raw_wind_io
from gewittergefahr.gg_io import tornado_io
from gewittergefahr.gg_utils import storm_tracking_utils as tracking_utils
from gewittergefahr.gg_utils import number_rounding as rounder
from gewittergefahr.gg_utils import projections
from gewittergefahr.gg_utils import interp
from gewittergefahr.gg_utils import polygons
from gewittergefahr.gg_utils import geodetic_utils
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.gg_utils import error_checking

YEAR_FORMAT = '%Y'
TIME_FORMAT = '%Y-%m-%d-%H%M%S'
SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

WIND_EVENT_TYPE_STRING = 'wind'
TORNADO_EVENT_TYPE_STRING = 'tornado'
VALID_EVENT_TYPE_STRINGS = [WIND_EVENT_TYPE_STRING, TORNADO_EVENT_TYPE_STRING]

WIND_DATA_SOURCE = raw_wind_io.MERGED_DATA_SOURCE

DEFAULT_MAX_TIME_BEFORE_STORM_SEC = 300
DEFAULT_MAX_TIME_AFTER_STORM_SEC = 300
DEFAULT_PADDING_FOR_BOUNDING_BOX_METRES = 1e5
DEFAULT_INTERP_TIME_RESOLUTION_FOR_WIND_SEC = 10
DEFAULT_INTERP_TIME_RESOLUTION_FOR_TORNADOES_SEC = 1
DEFAULT_MAX_LINK_DISTANCE_FOR_WIND_METRES = 30000.
DEFAULT_MAX_LINK_DISTANCE_FOR_TORNADOES_METRES = 30000.

REQUIRED_STORM_COLUMNS = [
    tracking_utils.STORM_ID_COLUMN, tracking_utils.TIME_COLUMN,
    tracking_utils.TRACKING_START_TIME_COLUMN,
    tracking_utils.TRACKING_END_TIME_COLUMN,
    tracking_utils.CELL_START_TIME_COLUMN, tracking_utils.CELL_END_TIME_COLUMN,
    tracking_utils.CENTROID_LAT_COLUMN, tracking_utils.CENTROID_LNG_COLUMN,
    tracking_utils.POLYGON_OBJECT_LATLNG_COLUMN]

REQUIRED_WIND_COLUMNS = [
    raw_wind_io.STATION_ID_COLUMN, raw_wind_io.LATITUDE_COLUMN,
    raw_wind_io.LONGITUDE_COLUMN, raw_wind_io.TIME_COLUMN,
    raw_wind_io.U_WIND_COLUMN, raw_wind_io.V_WIND_COLUMN]

REQUIRED_TORNADO_COLUMNS = [
    tornado_io.START_TIME_COLUMN, tornado_io.START_LAT_COLUMN,
    tornado_io.START_LNG_COLUMN, tornado_io.FUJITA_RATING_COLUMN]

FILE_INDEX_COLUMN = 'file_index'
STORM_CENTROID_X_COLUMN = 'centroid_x_metres'
STORM_CENTROID_Y_COLUMN = 'centroid_y_metres'
STORM_VERTICES_X_COLUMN = 'vertices_x_metres'
STORM_VERTICES_Y_COLUMN = 'vertices_y_metres'

EVENT_TIME_COLUMN = 'unix_time_sec'
EVENT_LATITUDE_COLUMN = 'latitude_deg'
EVENT_LONGITUDE_COLUMN = 'longitude_deg'
EVENT_X_COLUMN = 'x_coord_metres'
EVENT_Y_COLUMN = 'y_coord_metres'
NEAREST_STORM_ID_COLUMN = 'nearest_storm_id'
LINKAGE_DISTANCE_COLUMN = 'linkage_distance_metres'

STORM_VERTEX_X_COLUMN = 'vertex_x_metres'
STORM_VERTEX_Y_COLUMN = 'vertex_y_metres'

LINKAGE_DISTANCES_COLUMN = 'linkage_distances_metres'
RELATIVE_EVENT_TIMES_COLUMN = 'relative_event_times_sec'
EVENT_LATITUDES_COLUMN = 'event_latitudes_deg'
EVENT_LONGITUDES_COLUMN = 'event_longitudes_deg'

FUJITA_RATINGS_COLUMN = 'f_or_ef_scale_ratings'
WIND_STATION_IDS_COLUMN = 'wind_station_ids'
U_WINDS_COLUMN = 'u_winds_m_s01'
V_WINDS_COLUMN = 'v_winds_m_s01'

REQUIRED_STORM_TO_WINDS_COLUMNS = REQUIRED_STORM_COLUMNS + [
    LINKAGE_DISTANCES_COLUMN, RELATIVE_EVENT_TIMES_COLUMN,
    EVENT_LATITUDES_COLUMN, EVENT_LONGITUDES_COLUMN, WIND_STATION_IDS_COLUMN,
    U_WINDS_COLUMN, V_WINDS_COLUMN, tracking_utils.CELL_END_TIME_COLUMN]

REQUIRED_STORM_TO_TORNADOES_COLUMNS = REQUIRED_STORM_COLUMNS + [
    LINKAGE_DISTANCES_COLUMN, RELATIVE_EVENT_TIMES_COLUMN,
    EVENT_LATITUDES_COLUMN, EVENT_LONGITUDES_COLUMN, FUJITA_RATINGS_COLUMN]


def _check_linkage_params(
        tracking_file_names, max_time_before_storm_start_sec,
        max_time_after_storm_end_sec, padding_for_storm_bounding_box_metres,
        interp_time_resolution_sec, max_link_distance_metres):
    """Error-checking for linkage parameters.

    :param tracking_file_names: 1-D list of paths to storm-tracking files
        (created by `storm_tracking_io.read_processed_file`).
    :param max_time_before_storm_start_sec: Max time before beginning of storm
        cell.  If event E occurs > `max_time_before_storm_start_sec` before the
        beginning of storm cell S, E cannot be linked to S.
    :param max_time_after_storm_end_sec: Max time after end of storm cell.  If
        event E occurs > `max_time_after_storm_end_sec` after the end of storm
        cell E, W cannot be linked to S.
    :param padding_for_storm_bounding_box_metres: This padding will be added to
        each edge (north, east, south, and west) of the bounding box around all
        storm objects.  Events outside the bounding box will be thrown out.
    :param interp_time_resolution_sec: Discretization time for interpolation.
        If `interp_time_resolution_sec` = 1, storms will be interpolated to each
        unique event time (since event times are rounded to the nearest second).
        If `interp_time_resolution_sec` = q, storms will be interpolated to each
        multiple of q seconds between the first and last event times.  Setting
        `interp_time_resolution_sec` > 1 decreases accuracy but saves time.
    :param max_link_distance_metres: Max linkage distance.  If event E is >
        `max_link_distance_metres` from the nearest storm, it will not be linked
        to any storm.
    """

    error_checking.assert_is_string_list(tracking_file_names)
    error_checking.assert_is_numpy_array(
        numpy.asarray(tracking_file_names), num_dimensions=1)

    error_checking.assert_is_integer(max_time_before_storm_start_sec)
    error_checking.assert_is_geq(max_time_before_storm_start_sec, 0)
    error_checking.assert_is_integer(max_time_after_storm_end_sec)
    error_checking.assert_is_geq(max_time_after_storm_end_sec, 0)

    error_checking.assert_is_integer(interp_time_resolution_sec)
    error_checking.assert_is_greater(interp_time_resolution_sec, 0)
    error_checking.assert_is_geq(max_link_distance_metres, 0.)
    error_checking.assert_is_geq(
        padding_for_storm_bounding_box_metres, max_link_distance_metres)


def _get_bounding_box_of_storms(
        storm_object_table,
        padding_metres=DEFAULT_PADDING_FOR_BOUNDING_BOX_METRES):
    """Creates bounding box (with some padding) for all storm objects.

    :param storm_object_table: pandas DataFrame created by
        `_project_storms_latlng_to_xy`.
    :param padding_metres: This padding will be added to each edge (north, east,
        south, and west) of the actual bounding box.
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
            storm_object_table[STORM_VERTICES_X_COLUMN].values[i]))
        all_y_coords_metres = numpy.concatenate((
            all_y_coords_metres,
            storm_object_table[STORM_VERTICES_Y_COLUMN].values[i]))

    x_limits_metres = numpy.array(
        [numpy.min(all_x_coords_metres) - padding_metres,
         numpy.max(all_x_coords_metres) + padding_metres])
    y_limits_metres = numpy.array(
        [numpy.min(all_y_coords_metres) - padding_metres,
         numpy.max(all_y_coords_metres) + padding_metres])

    return x_limits_metres, y_limits_metres


def _project_storms_latlng_to_xy(storm_object_table, projection_object):
    """Projects storm objects from lat-long to x-y coordinates.

    This method projects both the centroids and bounding polygons.

    V = number of vertices in a given storm object.

    :param storm_object_table: pandas DataFrame created by `_read_storm_tracks`.
    :param projection_object: Instance of `pyproj.Proj`, defining an equidistant
        projection.
    :return: storm_object_table: Same as input, but with extra columns listed
        below.
    storm_object_table.centroid_x_metres: x-coordinate of centroid.
    storm_object_table.centroid_y_metres: y-coordinate of centroid.
    storm_object_table.vertices_x_metres: length-V numpy array with x-
        coordinates of vertices.
    storm_object_table.vertices_y_metres: length-V numpy array with y-
        coordinates of vertices.
    """

    centroids_x_metres, centroids_y_metres = projections.project_latlng_to_xy(
        storm_object_table[tracking_utils.CENTROID_LAT_COLUMN].values,
        storm_object_table[tracking_utils.CENTROID_LNG_COLUMN].values,
        projection_object=projection_object)

    nested_array = storm_object_table[[
        tracking_utils.STORM_ID_COLUMN,
        tracking_utils.STORM_ID_COLUMN]].values.tolist()
    argument_dict = {
        STORM_CENTROID_X_COLUMN: centroids_x_metres,
        STORM_CENTROID_Y_COLUMN: centroids_y_metres,
        STORM_VERTICES_X_COLUMN: nested_array,
        STORM_VERTICES_Y_COLUMN: nested_array
    }
    storm_object_table = storm_object_table.assign(**argument_dict)

    num_storm_objects = len(storm_object_table.index)
    for i in range(num_storm_objects):
        this_vertex_dict_latlng = (
            polygons.polygon_object_to_vertex_arrays(
                storm_object_table[
                    tracking_utils.POLYGON_OBJECT_LATLNG_COLUMN].values[i]))

        (storm_object_table[STORM_VERTICES_X_COLUMN].values[i],
         storm_object_table[STORM_VERTICES_Y_COLUMN].values[i]) = (
             projections.project_latlng_to_xy(
                 this_vertex_dict_latlng[polygons.EXTERIOR_Y_COLUMN],
                 this_vertex_dict_latlng[polygons.EXTERIOR_X_COLUMN],
                 projection_object=projection_object))

    return storm_object_table


def _project_events_latlng_to_xy(event_table, projection_object):
    """Projects events (e.g., straight-line or tornadoes) from lat-long to x-y.

    :param event_table: pandas DataFrame with at least the following columns.
    event_table.latitude_deg: Latitude (deg N).
    event_table.longitude_deg: Longitude (deg N).

    :param projection_object: Instance of `pyproj.Proj`, defining an equidistant
        projection.
    :return: event_table: Same as input, but with extra columns listed below.
    event_table.x_coord_metres: x-coordinate of event.
    event_table.y_coord_metres: y-coordinate of event.
    """

    x_coords_metres, y_coords_metres = projections.project_latlng_to_xy(
        event_table[EVENT_LATITUDE_COLUMN].values,
        event_table[EVENT_LONGITUDE_COLUMN].values,
        projection_object=projection_object)

    argument_dict = {
        EVENT_X_COLUMN: x_coords_metres, EVENT_Y_COLUMN: y_coords_metres}
    return event_table.assign(**argument_dict)


def _filter_events_by_location(event_table, x_limits_metres, y_limits_metres):
    """Filters events (e.g., straight-line or tornadoes) by location.

    :param event_table: pandas DataFrame with at least the following columns.
    event_table.x_coord_metres: x-coordinate of event.
    event_table.y_coord_metres: y-coordinate of event.

    :param x_limits_metres: length-2 numpy array with [min, max] x-coordinates
        of bounding box.
    :param y_limits_metres: length-2 numpy array with [min, max] y-coordinates
        of bounding box.
    :return: event_table: Same as input, but maybe with fewer rows.
    """

    x_invalid_flags = numpy.invert(numpy.logical_and(
        event_table[EVENT_X_COLUMN].values >= x_limits_metres[0],
        event_table[EVENT_X_COLUMN].values <= x_limits_metres[1]))

    y_invalid_flags = numpy.invert(numpy.logical_and(
        event_table[EVENT_Y_COLUMN].values >= y_limits_metres[0],
        event_table[EVENT_Y_COLUMN].values <= y_limits_metres[1]))

    invalid_rows = numpy.where(
        numpy.logical_or(x_invalid_flags, y_invalid_flags))[0]
    return event_table.drop(
        event_table.index[invalid_rows], axis=0, inplace=False)


def _filter_storms_by_time(
        storm_object_table, max_start_time_unix_sec, min_end_time_unix_sec):
    """Filters storm cells by time.

    For any storm cell S with start time > `max_start_time_unix_sec`, all storm
    objects will be removed.

    For any storm cell S with end time < `min_end_time_unix_sec`, all storm
    objects will be removed.

    :param storm_object_table: pandas DataFrame, where each row is one storm
        object.  Must contain at least the following columns.
    storm_object_table.cell_start_time_unix_sec: Start time of storm cell (first
        time in track).
    storm_object_table.cell_end_time_unix_sec: End time of storm cell (last time
        in track).

    :param max_start_time_unix_sec: Latest allowed start time.
    :param min_end_time_unix_sec: Earliest allowed end time.
    :return: storm_object_table: Same as input, but maybe with fewer rows.
    """

    invalid_flags = numpy.invert(numpy.logical_and(
        storm_object_table[tracking_utils.CELL_START_TIME_COLUMN].values <=
        max_start_time_unix_sec,
        storm_object_table[tracking_utils.CELL_END_TIME_COLUMN].values >=
        min_end_time_unix_sec))

    return storm_object_table.drop(
        storm_object_table.index[numpy.where(invalid_flags)[0]], axis=0,
        inplace=False)


def _interp_one_storm_in_time(
        storm_object_table_1cell, storm_id, target_time_unix_sec):
    """Interpolates one storm cell in time.

    The storm object nearest to the target time is advected (i.e., moved as a
    whole, so that its shape [bounding polygon] does not change).  Thus, the
    interpolated storm object always has a realistic shape.  Also, interpolation
    is usually over a time span <= 2.5 minutes, over which time we assume that
    changes in shape are negligible.

    N = number of storm objects

    :param storm_object_table_1cell: pandas DataFrame with the following
        columns.  Each row is one storm object.
    storm_object_table_1cell.unix_time_sec: Valid time.
    storm_object_table_1cell.centroid_x_metres: x-coordinate of centroid.
    storm_object_table_1cell.centroid_y_metres: y-coordinate of centroid.
    storm_object_table_1cell.vertices_x_metres: length-V numpy array with x-
        coordinates of vertices.
    storm_object_table_1cell.vertices_y_metres: length-V numpy array with y-
        coordinates of vertices.

    :param storm_id: String ID for storm cell (the same for all objects in
        `storm_object_table_1cell`).
    :param target_time_unix_sec: Storm location will be interpolated to this
        time.
    :return: interp_vertex_table_1object: pandas DataFrame with the following
        columns.  Each row is one vertex of the interpolated storm object.
    interp_vertex_table_1object.storm_id: String ID for storm cell (same as
        input `storm_id`).
    interp_vertex_table_1object.vertex_x_metres: x-coordinate of vertex.
    interp_vertex_table_1object.vertex_y_metres: y-coordinate of vertex.
    """

    sort_indices = numpy.argsort(
        storm_object_table_1cell[tracking_utils.TIME_COLUMN].values)
    centroid_matrix = numpy.vstack((
        storm_object_table_1cell[STORM_CENTROID_X_COLUMN].values[sort_indices],
        storm_object_table_1cell[STORM_CENTROID_Y_COLUMN].values[sort_indices]))

    interp_centroid_vector = interp.interp_in_time(
        input_matrix=centroid_matrix,
        sorted_input_times_unix_sec=storm_object_table_1cell[
            tracking_utils.TIME_COLUMN].values[sort_indices],
        query_times_unix_sec=numpy.array([target_time_unix_sec]),
        method_string=interp.LINEAR_METHOD_STRING, extrapolate=True)

    absolute_time_diffs_sec = numpy.absolute(
        storm_object_table_1cell[tracking_utils.TIME_COLUMN].values -
        target_time_unix_sec)
    nearest_time_index = numpy.argmin(absolute_time_diffs_sec)

    x_diff_metres = interp_centroid_vector[0] - storm_object_table_1cell[
        STORM_CENTROID_X_COLUMN].values[nearest_time_index]
    y_diff_metres = interp_centroid_vector[1] - storm_object_table_1cell[
        STORM_CENTROID_Y_COLUMN].values[nearest_time_index]

    num_vertices = len(
        storm_object_table_1cell[STORM_VERTICES_X_COLUMN].values[
            nearest_time_index])
    storm_id_list = [storm_id] * num_vertices

    interp_vertex_dict_1object = {
        tracking_utils.STORM_ID_COLUMN: storm_id_list,
        STORM_VERTEX_X_COLUMN:
            storm_object_table_1cell[STORM_VERTICES_X_COLUMN].values[
                nearest_time_index] + x_diff_metres,
        STORM_VERTEX_Y_COLUMN:
            storm_object_table_1cell[STORM_VERTICES_Y_COLUMN].values[
                nearest_time_index] + y_diff_metres
    }

    return pandas.DataFrame.from_dict(interp_vertex_dict_1object)


def _interp_storms_in_time(
        storm_object_table, target_time_unix_sec, max_time_before_start_sec,
        max_time_after_end_sec):
    """Interpolates location of each storm cell in time.

    :param storm_object_table: pandas DataFrame created by
        `_find_start_end_times_of_storms`.
    :param target_time_unix_sec: Storm locations will be interpolated to this
        time.
    :param max_time_before_start_sec: Max extrapolation time before start of
        storm cell.  For storm cell S, if `target_time_unix_sec` is before the
        start time of S, this method will not interpolate S.
    :param max_time_after_end_sec: Max extrapolation time after end of storm
        cell.  See above.
    :return: interp_vertex_table: pandas DataFrame with the following columns.
        Each row is one vertex of one interpolated storm object.
    interp_vertex_table.storm_id: String ID for storm cell.
    interp_vertex_table.vertex_x_metres: x-coordinate of vertex.
    interp_vertex_table.vertex_y_metres: y-coordinate of vertex.
    """

    max_start_time_unix_sec = target_time_unix_sec + max_time_before_start_sec
    min_end_time_unix_sec = target_time_unix_sec - max_time_after_end_sec

    filtered_storm_object_table = _filter_storms_by_time(
        storm_object_table, max_start_time_unix_sec=max_start_time_unix_sec,
        min_end_time_unix_sec=min_end_time_unix_sec)

    unique_storm_ids = numpy.unique(numpy.array(
        filtered_storm_object_table[tracking_utils.STORM_ID_COLUMN].values))

    list_of_tables = []
    num_storm_cells = len(unique_storm_ids)

    for j in range(num_storm_cells):
        this_storm_object_table = (
            filtered_storm_object_table.loc[
                filtered_storm_object_table[tracking_utils.STORM_ID_COLUMN] ==
                unique_storm_ids[j]])
        if len(this_storm_object_table.index) == 1:
            continue

        this_interp_vertex_table = _interp_one_storm_in_time(
            this_storm_object_table, storm_id=unique_storm_ids[j],
            target_time_unix_sec=target_time_unix_sec)
        if not list_of_tables:
            list_of_tables.append(this_interp_vertex_table)
            continue

        this_interp_vertex_table, _ = this_interp_vertex_table.align(
            list_of_tables[-1], axis=1)
        list_of_tables.append(this_interp_vertex_table)

    if not list_of_tables:
        return pandas.DataFrame(
            columns=[tracking_utils.STORM_ID_COLUMN, STORM_VERTEX_X_COLUMN,
                     STORM_VERTEX_Y_COLUMN])

    return pandas.concat(list_of_tables, axis=0, ignore_index=True)


def _find_nearest_storms_at_one_time(
        interp_vertex_table, event_x_coords_metres, event_y_coords_metres,
        max_link_distance_metres):
    """For each event at one time step, this method finds the nearest storm.

    N = number of events

    :param interp_vertex_table: pandas DataFrame created by
        `_interp_storms_in_time`.
    :param event_x_coords_metres: length-N numpy array with x-coordinates of
        events.
    :param event_y_coords_metres: length-N numpy array with y-coordinates of
        events.
    :param max_link_distance_metres: Max linkage distance.  For event E, if the
        nearest storm is > `max_link_distance_metres` away, E will not be linked
        to any storm.
    :return: nearest_storm_ids: length-N list of strings, where the [i]th
        element is the ID of the nearest storm to the [i]th event.  If the [i]th
        element is None, the [i]th event was not linked to a storm.
    :return: linkage_distances_metres: length-N numpy array of linkage
        distances.  If the [i]th event was not linked to a storm,
        linkage_distances_metres[i] = NaN.
    """

    num_events = len(event_x_coords_metres)
    nearest_storm_ids = [None] * num_events
    linkage_distances_metres = numpy.full(num_events, numpy.nan)

    for k in range(num_events):
        these_x_diffs_metres = numpy.absolute(
            event_x_coords_metres[k] -
            interp_vertex_table[STORM_VERTEX_X_COLUMN].values)
        these_y_diffs_metres = numpy.absolute(
            event_y_coords_metres[k] -
            interp_vertex_table[STORM_VERTEX_Y_COLUMN].values)

        these_valid_flags = numpy.logical_and(
            these_x_diffs_metres <= max_link_distance_metres,
            these_y_diffs_metres <= max_link_distance_metres)
        if not numpy.any(these_valid_flags):
            continue

        these_valid_indices = numpy.where(these_valid_flags)[0]
        these_distances_metres = numpy.sqrt(
            these_x_diffs_metres[these_valid_indices] ** 2 +
            these_y_diffs_metres[these_valid_indices] ** 2)
        if not numpy.any(these_distances_metres <= max_link_distance_metres):
            continue

        this_min_index = these_valid_indices[
            numpy.argmin(these_distances_metres)]
        nearest_storm_ids[k] = interp_vertex_table[
            tracking_utils.STORM_ID_COLUMN].values[this_min_index]

        this_storm_flags = numpy.array([
            s == nearest_storm_ids[k] for s in interp_vertex_table[
                tracking_utils.STORM_ID_COLUMN].values])
        this_storm_indices = numpy.where(this_storm_flags)[0]

        this_polygon_object = polygons.vertex_arrays_to_polygon_object(
            interp_vertex_table[STORM_VERTEX_X_COLUMN].values[
                this_storm_indices],
            interp_vertex_table[STORM_VERTEX_Y_COLUMN].values[
                this_storm_indices])
        this_point_in_polygon_flag = polygons.point_in_or_on_polygon(
            this_polygon_object, query_x_coordinate=event_x_coords_metres[k],
            query_y_coordinate=event_y_coords_metres[k])

        if this_point_in_polygon_flag:
            linkage_distances_metres[k] = 0.
        else:
            linkage_distances_metres[k] = numpy.min(these_distances_metres)

    return nearest_storm_ids, linkage_distances_metres


def _find_nearest_storms(
        storm_object_table, event_table, max_time_before_storm_start_sec,
        max_time_after_storm_end_sec, max_link_distance_metres,
        interp_time_resolution_sec):
    """Finds the nearest storm cell to each event.

    :param storm_object_table: pandas DataFrame created by
        `_find_start_end_times_of_storms`.
    :param event_table: pandas DataFrame with at least the following columns.
        Each row is one event.
    event_table.unix_time_sec: Valid time.
    event_table.x_coord_metres: x-coordinate of event.
    event_table.y_coord_metres: y-coordinate of event.

    :param max_time_before_storm_start_sec: Max time before beginning of storm
        cell.  If event E occurs > `max_time_before_storm_start_sec` before the
        beginning of storm cell S, E cannot be linked to S.
    :param max_time_after_storm_end_sec: Max time after end of storm cell.  If
        event E occurs > `max_time_after_storm_end_sec` after the end of storm
        cell S, E cannot be linked to S.
    :param max_link_distance_metres: Max linkage distance.  If event E is >
        `max_link_distance_metres` from the nearest storm, it will not be linked
        to any storm.
    :param interp_time_resolution_sec: Discretization time for interpolation.
        If `interp_time_resolution_sec` = 1, storms will be interpolated to each
        unique event time (since event times are rounded to the nearest second).
        If `interp_time_resolution_sec` = q, storms will be interpolated to each
        multiple of q seconds between the first and last event times.
    :return: event_to_storm_table: Same as input `event_table`, but with extra
        columns listed below.
    event_to_storm_table.nearest_storm_id: String ID for nearest storm cell.  If
        event was not linked to a storm cell, this is None.
    event_to_storm_table.linkage_distance_metres: Distance to nearest storm
        cell.  If event was not linked to a storm cell, this is NaN.
    """

    rounded_times_unix_sec = rounder.round_to_nearest(
        event_table[EVENT_TIME_COLUMN].values, interp_time_resolution_sec)

    unique_rounded_times_unix_sec, rounded_times_orig_to_unique = (
        numpy.unique(rounded_times_unix_sec, return_inverse=True))
    unique_rounded_time_strings = [
        time_conversion.unix_sec_to_string(t, TIME_FORMAT)
        for t in unique_rounded_times_unix_sec]
    num_unique_rounded_times = len(unique_rounded_time_strings)

    num_events = len(event_table.index)
    nearest_storm_ids = [None] * num_events
    linkage_distances_metres = numpy.full(num_events, numpy.nan)

    for i in range(num_unique_rounded_times):
        print 'Linking events at ~{0:s} to storms...'.format(
            unique_rounded_time_strings[i])

        these_wind_rows = numpy.where(rounded_times_orig_to_unique == i)[0]
        this_interp_vertex_table = _interp_storms_in_time(
            storm_object_table,
            target_time_unix_sec=unique_rounded_times_unix_sec[i],
            max_time_before_start_sec=max_time_before_storm_start_sec,
            max_time_after_end_sec=max_time_after_storm_end_sec)

        these_nearest_storm_ids, these_link_distances_metres = (
            _find_nearest_storms_at_one_time(
                this_interp_vertex_table,
                event_x_coords_metres=event_table[EVENT_X_COLUMN].values[
                    these_wind_rows],
                event_y_coords_metres=event_table[EVENT_Y_COLUMN].values[
                    these_wind_rows],
                max_link_distance_metres=max_link_distance_metres))
        # print these_nearest_storm_ids

        linkage_distances_metres[these_wind_rows] = these_link_distances_metres
        for j in range(len(these_wind_rows)):
            nearest_storm_ids[these_wind_rows[j]] = these_nearest_storm_ids[j]

    argument_dict = {NEAREST_STORM_ID_COLUMN: nearest_storm_ids,
                     LINKAGE_DISTANCE_COLUMN: linkage_distances_metres}
    return event_table.assign(**argument_dict)


def _create_storm_to_winds_table(storm_object_table, wind_to_storm_table):
    """Creates list of wind observations linked to each storm cell.

    K = number of wind observations linked to a given storm cell

    :param storm_object_table: pandas DataFrame created by
        `_find_start_end_times_of_storms`.
    :param wind_to_storm_table: pandas DataFrame with at least the following
        columns.  Each row is one wind observation.
    wind_to_storm_table.station_id: String ID for station.
    wind_to_storm_table.unix_time_sec: Valid time.
    wind_to_storm_table.latitude_deg: Latitude (deg N).
    wind_to_storm_table.longitude_deg: Longitude (deg E).
    wind_to_storm_table.u_wind_m_s01: u-wind (metres per second).
    wind_to_storm_table.v_wind_m_s01: v-wind (metres per second).
    wind_to_storm_table.nearest_storm_id: String ID for nearest storm cell.  If
        event was not linked to a storm cell, this is None.
    wind_to_storm_table.linkage_distance_metres: Distance to nearest storm
        cell.  If event was not linked to a storm cell, this is NaN.

    :return: storm_to_winds_table: Same as input `storm_object_table`, but with
        extra columns listed below.
    storm_to_winds_table.wind_station_ids: length-K list of station IDs
        (strings).
    storm_to_winds_table.event_latitudes_deg: length-K numpy array of latitudes
        (deg N).
    storm_to_winds_table.event_longitudes_deg: length-K numpy array of
        longitudes (deg E).
    storm_to_winds_table.u_winds_m_s01: length-K numpy array of u-winds (metres
        per second).
    storm_to_winds_table.v_winds_m_s01: length-K numpy array of v-winds (metres
        per second).
    storm_to_winds_table.linkage_distances_metres: length-K numpy array of
        linkage distances (from wind observation to nearest storm object from
        the same cell).
    storm_to_winds_table.relative_event_times_sec: length-K numpy array with
        relative times of wind observations (wind time minus storm-object time).
    """

    nested_array = storm_object_table[[
        tracking_utils.STORM_ID_COLUMN,
        tracking_utils.STORM_ID_COLUMN]].values.tolist()
    argument_dict = {
        WIND_STATION_IDS_COLUMN: nested_array,
        EVENT_LATITUDES_COLUMN: nested_array,
        EVENT_LONGITUDES_COLUMN: nested_array, U_WINDS_COLUMN: nested_array,
        V_WINDS_COLUMN: nested_array, LINKAGE_DISTANCES_COLUMN: nested_array,
        RELATIVE_EVENT_TIMES_COLUMN: nested_array
    }

    storm_to_winds_table = copy.deepcopy(storm_object_table)
    storm_to_winds_table = storm_to_winds_table.assign(**argument_dict)
    num_storm_objects = len(storm_to_winds_table.index)

    for i in range(num_storm_objects):
        storm_to_winds_table[WIND_STATION_IDS_COLUMN].values[i] = []
        storm_to_winds_table[EVENT_LATITUDES_COLUMN].values[i] = []
        storm_to_winds_table[EVENT_LONGITUDES_COLUMN].values[i] = []
        storm_to_winds_table[U_WINDS_COLUMN].values[i] = []
        storm_to_winds_table[V_WINDS_COLUMN].values[i] = []
        storm_to_winds_table[LINKAGE_DISTANCES_COLUMN].values[i] = []
        storm_to_winds_table[RELATIVE_EVENT_TIMES_COLUMN].values[i] = []

    num_wind_observations = len(wind_to_storm_table.index)
    for k in range(num_wind_observations):
        this_storm_id = wind_to_storm_table[NEAREST_STORM_ID_COLUMN].values[k]
        if this_storm_id is None:
            continue

        this_storm_flags = numpy.array(
            [s == this_storm_id for s in storm_to_winds_table[
                tracking_utils.STORM_ID_COLUMN].values])
        this_storm_indices = numpy.where(this_storm_flags)[0]

        for j in this_storm_indices:
            storm_to_winds_table[WIND_STATION_IDS_COLUMN].values[j].append(
                wind_to_storm_table[raw_wind_io.STATION_ID_COLUMN].values[k])
            storm_to_winds_table[EVENT_LATITUDES_COLUMN].values[j].append(
                wind_to_storm_table[EVENT_LATITUDE_COLUMN].values[k])
            storm_to_winds_table[EVENT_LONGITUDES_COLUMN].values[j].append(
                wind_to_storm_table[EVENT_LONGITUDE_COLUMN].values[k])
            storm_to_winds_table[U_WINDS_COLUMN].values[j].append(
                wind_to_storm_table[raw_wind_io.U_WIND_COLUMN].values[k])
            storm_to_winds_table[V_WINDS_COLUMN].values[j].append(
                wind_to_storm_table[raw_wind_io.V_WIND_COLUMN].values[k])
            storm_to_winds_table[LINKAGE_DISTANCES_COLUMN].values[j].append(
                wind_to_storm_table[LINKAGE_DISTANCE_COLUMN].values[k])

            this_relative_time_sec = (
                wind_to_storm_table[EVENT_TIME_COLUMN].values[k] -
                storm_to_winds_table[tracking_utils.TIME_COLUMN].values[j])
            storm_to_winds_table[RELATIVE_EVENT_TIMES_COLUMN].values[j].append(
                this_relative_time_sec)

    for i in range(num_storm_objects):
        storm_to_winds_table[EVENT_LATITUDES_COLUMN].values[i] = numpy.array(
            storm_to_winds_table[EVENT_LATITUDES_COLUMN].values[i])
        storm_to_winds_table[EVENT_LONGITUDES_COLUMN].values[i] = numpy.array(
            storm_to_winds_table[EVENT_LONGITUDES_COLUMN].values[i])
        storm_to_winds_table[U_WINDS_COLUMN].values[i] = numpy.array(
            storm_to_winds_table[U_WINDS_COLUMN].values[i])
        storm_to_winds_table[V_WINDS_COLUMN].values[i] = numpy.array(
            storm_to_winds_table[V_WINDS_COLUMN].values[i])
        storm_to_winds_table[LINKAGE_DISTANCES_COLUMN].values[i] = numpy.array(
            storm_to_winds_table[LINKAGE_DISTANCES_COLUMN].values[i])
        storm_to_winds_table[RELATIVE_EVENT_TIMES_COLUMN].values[i] = (
            numpy.array(
                storm_to_winds_table[RELATIVE_EVENT_TIMES_COLUMN].values[i]))

    return storm_to_winds_table


def _create_storm_to_tornadoes_table(
        storm_object_table, tornado_to_storm_table):
    """Creates list of tornadoes linked to each storm cell.

    K = number of tornadoes linked to a given storm cell

    :param storm_object_table: pandas DataFrame created by
        `_find_start_end_times_of_storms`.
    :param tornado_to_storm_table: pandas DataFrame with at least the following
        columns.  Each row is one tornado.
    tornado_to_storm_table.unix_time_sec: Valid time.
    tornado_to_storm_table.latitude_deg: Latitude (deg N).
    tornado_to_storm_table.longitude_deg: Longitude (deg E).
    tornado_to_storm_table.f_or_ef_rating: F- or EF-scale rating (string).
    tornado_to_storm_table.nearest_storm_id: String ID for nearest storm cell.
        If event was not linked to a storm cell, this is None.
    tornado_to_storm_table.linkage_distance_metres: Distance to nearest storm
        cell.  If event was not linked to a storm cell, this is NaN.

    :return: storm_to_tornadoes_table: Same as input `storm_object_table`, but
        with extra columns listed below.
    storm_to_tornadoes_table.event_latitudes_deg: length-K numpy array of
        latitudes (deg N).
    storm_to_tornadoes_table.event_longitudes_deg: length-K numpy array of
        longitudes (deg E).
    storm_to_tornadoes_table.f_or_ef_scale_ratings: length-K list of F- or
        EF-scale ratings (strings).
    storm_to_tornadoes_table.linkage_distances_metres: length-K numpy array of
        linkage distances (from tornado to nearest storm object from the same
        cell).
    storm_to_tornadoes_table.relative_event_times_sec: length-K numpy array with
        relative times of tornadoes (tornado time minus storm-object time).
    """

    nested_array = storm_object_table[[
        tracking_utils.STORM_ID_COLUMN,
        tracking_utils.STORM_ID_COLUMN]].values.tolist()
    argument_dict = {
        EVENT_LATITUDES_COLUMN: nested_array,
        EVENT_LONGITUDES_COLUMN: nested_array,
        FUJITA_RATINGS_COLUMN: nested_array,
        LINKAGE_DISTANCES_COLUMN: nested_array,
        RELATIVE_EVENT_TIMES_COLUMN: nested_array
    }

    storm_to_tornadoes_table = copy.deepcopy(storm_object_table)
    storm_to_tornadoes_table = storm_to_tornadoes_table.assign(**argument_dict)
    num_storm_objects = len(storm_to_tornadoes_table.index)

    for i in range(num_storm_objects):
        storm_to_tornadoes_table[EVENT_LATITUDES_COLUMN].values[i] = []
        storm_to_tornadoes_table[EVENT_LONGITUDES_COLUMN].values[i] = []
        storm_to_tornadoes_table[FUJITA_RATINGS_COLUMN].values[i] = []
        storm_to_tornadoes_table[LINKAGE_DISTANCES_COLUMN].values[i] = []
        storm_to_tornadoes_table[RELATIVE_EVENT_TIMES_COLUMN].values[i] = []

    num_tornadoes = len(tornado_to_storm_table.index)
    for k in range(num_tornadoes):
        this_storm_id = tornado_to_storm_table[
            NEAREST_STORM_ID_COLUMN].values[k]
        if this_storm_id is None:
            continue

        this_storm_flags = numpy.array(
            [s == this_storm_id for s in storm_to_tornadoes_table[
                tracking_utils.STORM_ID_COLUMN].values])
        this_storm_indices = numpy.where(this_storm_flags)[0]

        for j in this_storm_indices:
            storm_to_tornadoes_table[EVENT_LATITUDES_COLUMN].values[j].append(
                tornado_to_storm_table[EVENT_LATITUDE_COLUMN].values[k])
            storm_to_tornadoes_table[EVENT_LONGITUDES_COLUMN].values[j].append(
                tornado_to_storm_table[EVENT_LONGITUDE_COLUMN].values[k])
            storm_to_tornadoes_table[FUJITA_RATINGS_COLUMN].values[j].append(
                tornado_to_storm_table[
                    tornado_io.FUJITA_RATING_COLUMN].values[k])
            storm_to_tornadoes_table[LINKAGE_DISTANCES_COLUMN].values[j].append(
                tornado_to_storm_table[LINKAGE_DISTANCE_COLUMN].values[k])

            this_relative_time_sec = (
                tornado_to_storm_table[EVENT_TIME_COLUMN].values[k] -
                storm_to_tornadoes_table[tracking_utils.TIME_COLUMN].values[j])
            storm_to_tornadoes_table[RELATIVE_EVENT_TIMES_COLUMN].values[
                j].append(this_relative_time_sec)

    for i in range(num_storm_objects):
        storm_to_tornadoes_table[EVENT_LATITUDES_COLUMN].values[
            i] = numpy.array(
                storm_to_tornadoes_table[EVENT_LATITUDES_COLUMN].values[i])
        storm_to_tornadoes_table[EVENT_LONGITUDES_COLUMN].values[
            i] = numpy.array(
                storm_to_tornadoes_table[EVENT_LONGITUDES_COLUMN].values[i])
        storm_to_tornadoes_table[LINKAGE_DISTANCES_COLUMN].values[
            i] = numpy.array(
                storm_to_tornadoes_table[LINKAGE_DISTANCES_COLUMN].values[i])
        storm_to_tornadoes_table[RELATIVE_EVENT_TIMES_COLUMN].values[
            i] = numpy.array(
                storm_to_tornadoes_table[RELATIVE_EVENT_TIMES_COLUMN].values[i])

    return storm_to_tornadoes_table


def _read_storm_objects(tracking_file_names):
    """Reads storm objects from one or more files.

    :param tracking_file_names: 1-D list of paths to input files (created by
        `storm_tracking_io.write_processed_file`).
    :return: storm_object_table: pandas DataFrame with the following columns.
        Each row is one storm object.
    storm_object_table.storm_id: String ID for storm cell.
    storm_object_table.unix_time_sec: Valid time.
    storm_object_table.tracking_start_time_unix_sec: Start time for tracking
        period.
    storm_object_table.tracking_end_time_unix_sec: End time for tracking period.
    storm_object_table.centroid_lat_deg: Latitude (deg N) of centroid.
    storm_object_table.centroid_lng_deg: Longitude (deg E) of centroid.
    storm_object_table.polygon_object_latlng: Instance of
        `shapely.geometry.Polygon`, with vertices in lat-long coordinates.
    storm_object_table.file_index: Array index of file containing storm
        object.  If storm_object_table.file_index.values[i] = j, the [i]th storm
        object came from tracking_file_names[j].
    """

    columns_to_read = None
    file_indices = numpy.array([])

    num_files = len(tracking_file_names)
    list_of_storm_object_tables = [None] * num_files

    for i in range(num_files):
        print 'Reading storm objects from file: "{0:s}"...'.format(
            tracking_file_names[i])

        if i == 0:
            list_of_storm_object_tables[i] = tracking_io.read_processed_file(
                tracking_file_names[i])

            distance_buffer_columns = (
                tracking_utils.get_distance_buffer_columns(
                    list_of_storm_object_tables[i]))
            if distance_buffer_columns is None:
                distance_buffer_columns = []

            columns_to_read = (
                REQUIRED_STORM_COLUMNS + distance_buffer_columns)
            list_of_storm_object_tables[i] = list_of_storm_object_tables[i][
                columns_to_read]
        else:
            list_of_storm_object_tables[i] = tracking_io.read_processed_file(
                tracking_file_names[i])[columns_to_read]

        these_file_indices = numpy.full(
            len(list_of_storm_object_tables[i].index), i, dtype=int)
        file_indices = numpy.concatenate((file_indices, these_file_indices))

        if i == 0:
            continue

        list_of_storm_object_tables[i], _ = (
            list_of_storm_object_tables[i].align(
                list_of_storm_object_tables[0], axis=1))

    storm_object_table = pandas.concat(
        list_of_storm_object_tables, axis=0, ignore_index=True)
    storm_object_table = storm_object_table.assign(
        **{FILE_INDEX_COLUMN: file_indices}
    )

    good_indices = numpy.where(numpy.invert(numpy.logical_or(
        numpy.isnan(
            storm_object_table[tracking_utils.EAST_VELOCITY_COLUMN].values),
        numpy.isnan(
            storm_object_table[tracking_utils.NORTH_VELOCITY_COLUMN].values)
    )))[0]

    return storm_object_table.iloc[good_indices]


def _read_wind_observations(
        top_directory_name, storm_times_unix_sec,
        max_time_before_storm_start_sec, max_time_after_storm_end_sec):
    """Reads wind observations from one or more files.

    N = number of storm objects

    :param top_directory_name: Name of top-level directory with observed wind
        data (files created by `raw_wind_io.write_processed_file`).
    :param storm_times_unix_sec: 1-D numpy array with valid times of storm
        objects.
    :param max_time_before_storm_start_sec: Max time before beginning of storm
        cell.  If wind observation W occurs > `max_time_before_storm_start_sec`
        before the beginning of storm cell S, W cannot be linked to S.  This
        will be used to limit the number of wind files read.
    :param max_time_after_storm_end_sec: Max time after end of storm cell.  See
        above for more explanation.
    :return: wind_table: pandas DataFrame with the following columns.
    wind_table.station_id: String ID for station.
    wind_table.unix_time_sec: Valid time.
    wind_table.latitude_deg: Latitude (deg N).
    wind_table.longitude_deg: Longitude (deg E).
    wind_table.u_wind_m_s01: u-wind (metres per second).
    wind_table.v_wind_m_s01: v-wind (metres per second).
    """

    min_wind_time_unix_sec = numpy.min(
        storm_times_unix_sec) - max_time_before_storm_start_sec
    max_wind_time_unix_sec = numpy.max(
        storm_times_unix_sec) + max_time_after_storm_end_sec

    wind_file_names, _ = raw_wind_io.find_processed_hourly_files(
        start_time_unix_sec=min_wind_time_unix_sec,
        end_time_unix_sec=max_wind_time_unix_sec,
        primary_source=WIND_DATA_SOURCE, top_directory_name=top_directory_name,
        raise_error_if_missing=True)

    num_files = len(wind_file_names)
    list_of_wind_tables = [None] * num_files

    for i in range(num_files):
        print 'Reading wind observations from "{0:s}"...'.format(
            wind_file_names[i])

        list_of_wind_tables[i] = raw_wind_io.read_processed_file(
            wind_file_names[i])[REQUIRED_WIND_COLUMNS]
        if i == 0:
            continue

        list_of_wind_tables[i], _ = list_of_wind_tables[i].align(
            list_of_wind_tables[0], axis=1)

    wind_table = pandas.concat(list_of_wind_tables, axis=0, ignore_index=True)

    wind_speeds_m_s01 = numpy.sqrt(
        wind_table[raw_wind_io.U_WIND_COLUMN].values ** 2 +
        wind_table[raw_wind_io.V_WIND_COLUMN].values ** 2)
    invalid_rows = raw_wind_io.check_wind_speeds(
        wind_speeds_m_s01, one_component=False)
    wind_table.drop(wind_table.index[invalid_rows], axis=0, inplace=True)

    column_dict_old_to_new = {
        raw_wind_io.TIME_COLUMN: EVENT_TIME_COLUMN,
        raw_wind_io.LATITUDE_COLUMN: EVENT_LATITUDE_COLUMN,
        raw_wind_io.LONGITUDE_COLUMN: EVENT_LONGITUDE_COLUMN
    }
    wind_table.rename(columns=column_dict_old_to_new, inplace=True)

    return wind_table


def _read_tornado_reports(
        input_directory_name, storm_times_unix_sec,
        max_time_before_storm_start_sec, max_time_after_storm_end_sec):
    """Reads tornado reports from one file.

    :param input_directory_name: Name of directory with tornado-report files
        (created by `tornado_io.write_processed_file`).
    :param storm_times_unix_sec: See documentation for
        `_read_wind_observations`.
    :param max_time_before_storm_start_sec: See doc for
        `_read_wind_observations`.
    :param max_time_after_storm_end_sec: See doc for `_read_wind_observations`.
    :return: tornado_table: pandas DataFrame with the following columns.
    tornado_table.unix_time_sec: Valid time.
    tornado_table.latitude_deg: Latitude (deg N).
    tornado_table.longitude_deg: Longitude (deg E).
    tornado_table.f_or_ef_rating: F- or EF-scale rating (string).
    """

    min_tornado_time_unix_sec = numpy.min(
        storm_times_unix_sec) - max_time_before_storm_start_sec
    max_tornado_time_unix_sec = numpy.max(
        storm_times_unix_sec) + max_time_after_storm_end_sec

    min_tornado_year = int(time_conversion.unix_sec_to_string(
        min_tornado_time_unix_sec, YEAR_FORMAT))
    max_tornado_year = int(time_conversion.unix_sec_to_string(
        max_tornado_time_unix_sec, YEAR_FORMAT))
    tornado_years = numpy.linspace(
        min_tornado_year, max_tornado_year,
        num=max_tornado_year - min_tornado_year + 1, dtype=int)

    num_years = len(tornado_years)
    list_of_tornado_tables = [None] * num_years

    for i in range(num_years):
        this_file_name = tornado_io.find_processed_file(
            directory_name=input_directory_name, year=tornado_years[i])
        list_of_tornado_tables[i] = tornado_io.read_processed_file(
            this_file_name)[REQUIRED_TORNADO_COLUMNS]

        if i == 0:
            continue

        list_of_tornado_tables[i], _ = list_of_tornado_tables[i].align(
            list_of_tornado_tables[0], axis=1)

    tornado_table = pandas.concat(
        list_of_tornado_tables, axis=0, ignore_index=True)

    column_dict_old_to_new = {
        tornado_io.START_TIME_COLUMN: EVENT_TIME_COLUMN,
        tornado_io.START_LAT_COLUMN: EVENT_LATITUDE_COLUMN,
        tornado_io.START_LNG_COLUMN: EVENT_LONGITUDE_COLUMN
    }
    tornado_table.rename(columns=column_dict_old_to_new, inplace=True)

    bad_time_flags = numpy.invert(numpy.logical_and(
        tornado_table[EVENT_TIME_COLUMN].values >= min_tornado_time_unix_sec,
        tornado_table[EVENT_TIME_COLUMN].values <= max_tornado_time_unix_sec))
    tornado_table.drop(
        tornado_table.index[numpy.where(bad_time_flags)[0]], axis=0,
        inplace=True)

    return tornado_table


def _share_linkages_between_periods(
        early_storm_to_events_table, late_storm_to_events_table,
        event_type_string):
    """Shares linkage info between two periods.

    For the motivation behind this, see documentation for
    `share_linkages_across_spc_dates`.

    :param early_storm_to_events_table: pandas DataFrame created by either
        `_create_storm_to_winds_table` or `_create_storm_to_tornadoes_table`.
    :param late_storm_to_events_table: Same.
    :param event_type_string: Either "wind" or "tornado".
    :return: early_storm_to_events_table: Same as input, except maybe with new
        linkages.
    :return: late_storm_to_events_table: Same as input, except maybe with new
        linkages.
    """

    unique_early_storm_ids = numpy.unique(
        early_storm_to_events_table[tracking_utils.STORM_ID_COLUMN].values
    ).tolist()
    unique_late_storm_ids = numpy.unique(
        late_storm_to_events_table[tracking_utils.STORM_ID_COLUMN].values
    ).tolist()

    shared_storm_ids = [
        s for s in unique_early_storm_ids if s in unique_late_storm_ids]

    for this_storm_id in shared_storm_ids:
        i = numpy.where(
            early_storm_to_events_table[tracking_utils.STORM_ID_COLUMN].values
            == this_storm_id)[0][0]

        these_early_latitudes_deg = early_storm_to_events_table[
            EVENT_LATITUDES_COLUMN].values[i]
        these_early_longitudes_deg = early_storm_to_events_table[
            EVENT_LONGITUDES_COLUMN].values[i]
        these_early_link_dist_metres = early_storm_to_events_table[
            LINKAGE_DISTANCES_COLUMN].values[i]
        these_early_times_unix_sec = (
            early_storm_to_events_table[tracking_utils.TIME_COLUMN].values[i] +
            early_storm_to_events_table[RELATIVE_EVENT_TIMES_COLUMN].values[i])

        if event_type_string == WIND_EVENT_TYPE_STRING:
            these_early_station_ids = early_storm_to_events_table[
                WIND_STATION_IDS_COLUMN].values[i]
            these_early_u_winds_m_s01 = early_storm_to_events_table[
                U_WINDS_COLUMN].values[i]
            these_early_v_winds_m_s01 = early_storm_to_events_table[
                V_WINDS_COLUMN].values[i]
        else:
            these_early_fujita_ratings = early_storm_to_events_table[
                FUJITA_RATINGS_COLUMN].values[i]

        these_late_indices = numpy.where(
            late_storm_to_events_table[tracking_utils.STORM_ID_COLUMN].values
            == this_storm_id)[0]

        for j in these_late_indices:
            late_storm_to_events_table[
                EVENT_LATITUDES_COLUMN
            ].values[j] = numpy.concatenate((
                late_storm_to_events_table[EVENT_LATITUDES_COLUMN].values[j],
                these_early_latitudes_deg))

            late_storm_to_events_table[
                EVENT_LONGITUDES_COLUMN
            ].values[j] = numpy.concatenate((
                late_storm_to_events_table[EVENT_LONGITUDES_COLUMN].values[j],
                these_early_longitudes_deg))

            late_storm_to_events_table[
                LINKAGE_DISTANCES_COLUMN
            ].values[j] = numpy.concatenate((
                late_storm_to_events_table[LINKAGE_DISTANCES_COLUMN].values[j],
                these_early_link_dist_metres))

            these_relative_times_sec = (
                these_early_times_unix_sec - late_storm_to_events_table[
                    tracking_utils.TIME_COLUMN].values[j])
            late_storm_to_events_table[
                RELATIVE_EVENT_TIMES_COLUMN
            ].values[j] = numpy.concatenate((
                late_storm_to_events_table[
                    RELATIVE_EVENT_TIMES_COLUMN].values[j],
                these_relative_times_sec))

            if event_type_string == WIND_EVENT_TYPE_STRING:
                late_storm_to_events_table[
                    WIND_STATION_IDS_COLUMN
                ].values[j] = numpy.concatenate((
                    late_storm_to_events_table[
                        WIND_STATION_IDS_COLUMN].values[j],
                    these_early_station_ids))

                late_storm_to_events_table[
                    U_WINDS_COLUMN
                ].values[j] = numpy.concatenate((
                    late_storm_to_events_table[U_WINDS_COLUMN].values[j],
                    these_early_u_winds_m_s01))

                late_storm_to_events_table[
                    V_WINDS_COLUMN
                ].values[j] = numpy.concatenate((
                    late_storm_to_events_table[V_WINDS_COLUMN].values[j],
                    these_early_v_winds_m_s01))
            else:
                late_storm_to_events_table[
                    FUJITA_RATINGS_COLUMN
                ].values[j] = numpy.concatenate((
                    late_storm_to_events_table[FUJITA_RATINGS_COLUMN].values[j],
                    these_early_fujita_ratings))

        j = numpy.where(
            late_storm_to_events_table[tracking_utils.STORM_ID_COLUMN].values
            == this_storm_id)[0][0]
        event_times_unix_sec = (
            late_storm_to_events_table[tracking_utils.TIME_COLUMN].values[j] +
            late_storm_to_events_table[RELATIVE_EVENT_TIMES_COLUMN].values[j])

        these_early_indices = numpy.where(
            early_storm_to_events_table[tracking_utils.STORM_ID_COLUMN].values
            == this_storm_id)[0]

        for i in these_early_indices:
            early_storm_to_events_table[EVENT_LATITUDES_COLUMN].values[
                i
            ] = late_storm_to_events_table[EVENT_LATITUDES_COLUMN].values[j]
            early_storm_to_events_table[EVENT_LONGITUDES_COLUMN].values[
                i
            ] = late_storm_to_events_table[EVENT_LONGITUDES_COLUMN].values[j]
            early_storm_to_events_table[LINKAGE_DISTANCES_COLUMN].values[
                i
            ] = late_storm_to_events_table[LINKAGE_DISTANCES_COLUMN].values[j]

            early_storm_to_events_table[
                RELATIVE_EVENT_TIMES_COLUMN
            ].values[i] = event_times_unix_sec - early_storm_to_events_table[
                tracking_utils.TIME_COLUMN].values[i]

            if event_type_string == WIND_EVENT_TYPE_STRING:
                early_storm_to_events_table[WIND_STATION_IDS_COLUMN].values[
                    i] = late_storm_to_events_table[
                        WIND_STATION_IDS_COLUMN].values[j]

                early_storm_to_events_table[U_WINDS_COLUMN].values[
                    i] = late_storm_to_events_table[U_WINDS_COLUMN].values[j]

                early_storm_to_events_table[V_WINDS_COLUMN].values[
                    i] = late_storm_to_events_table[V_WINDS_COLUMN].values[j]
            else:
                early_storm_to_events_table[FUJITA_RATINGS_COLUMN].values[
                    i] = late_storm_to_events_table[
                        FUJITA_RATINGS_COLUMN].values[j]

    return early_storm_to_events_table, late_storm_to_events_table


def check_event_type(event_type_string):
    """Ensures that event type is recognized.

    :param event_type_string: Event type.
    :raises: ValueError: if `event_type_string not in VALID_EVENT_TYPE_STRINGS`.
    """

    error_checking.assert_is_string(event_type_string)
    if event_type_string not in VALID_EVENT_TYPE_STRINGS:
        error_string = (
            '\n\n{0:s}Valid event types (listed above) do not include '
            '"{1:s}".').format(VALID_EVENT_TYPE_STRINGS, event_type_string)
        raise ValueError(error_string)


def get_columns_to_write(
        storm_to_winds_table=None, storm_to_tornadoes_table=None):
    """Returns list of columns to write.

    Only one of the two input arguments may be specified.

    :param storm_to_winds_table: pandas DataFrame created by
        `_create_storm_to_winds_table`.
    :param storm_to_tornadoes_table: pandas DataFrame created by
        `_create_storm_to_tornadoes_table`.
    :return: columns_to_write: 1-D list with names of columns to write.
    """

    if storm_to_winds_table is not None:
        distance_buffer_columns = tracking_utils.get_distance_buffer_columns(
            storm_to_winds_table)
    else:
        distance_buffer_columns = tracking_utils.get_distance_buffer_columns(
            storm_to_tornadoes_table)

    if distance_buffer_columns is None:
        distance_buffer_columns = []

    if storm_to_winds_table is not None:
        return REQUIRED_STORM_TO_WINDS_COLUMNS + distance_buffer_columns

    return REQUIRED_STORM_TO_TORNADOES_COLUMNS + distance_buffer_columns


def link_each_storm_to_winds(
        tracking_file_names, top_wind_directory_name,
        max_time_before_storm_start_sec=DEFAULT_MAX_TIME_BEFORE_STORM_SEC,
        max_time_after_storm_end_sec=DEFAULT_MAX_TIME_AFTER_STORM_SEC,
        padding_for_storm_bounding_box_metres=
        DEFAULT_PADDING_FOR_BOUNDING_BOX_METRES,
        interp_time_resolution_sec=DEFAULT_INTERP_TIME_RESOLUTION_FOR_WIND_SEC,
        max_link_distance_metres=DEFAULT_MAX_LINK_DISTANCE_FOR_WIND_METRES):
    """Links each storm cell to zero or more wind observations.

    :param tracking_file_names: See documentation for
        `_check_linkage_params`.
    :param top_wind_directory_name: Name of top-level directory with observed
        wind data (files created by `raw_wind_io.write_processed_file`).
    :param max_time_before_storm_start_sec: See doc for `_check_linkage_params`.
    :param max_time_after_storm_end_sec: See doc for `_check_linkage_params`.
    :param padding_for_storm_bounding_box_metres: See doc for
        `_check_linkage_params`.
    :param interp_time_resolution_sec: See doc for `_check_linkage_params`.
    :param max_link_distance_metres: See doc for `_check_linkage_params`.
    :return: storm_to_winds_table: pandas DataFrame created by
        `_create_storm_to_winds_table`.
    """

    _check_linkage_params(
        tracking_file_names=tracking_file_names,
        max_time_before_storm_start_sec=max_time_before_storm_start_sec,
        max_time_after_storm_end_sec=max_time_after_storm_end_sec,
        padding_for_storm_bounding_box_metres=
        padding_for_storm_bounding_box_metres,
        interp_time_resolution_sec=interp_time_resolution_sec,
        max_link_distance_metres=max_link_distance_metres)

    storm_object_table = _read_storm_objects(tracking_file_names)
    print SEPARATOR_STRING

    # wind_table = _read_wind_observations(
    #     top_directory_name=top_wind_directory_name,
    #     storm_times_unix_sec=storm_object_table[
    #         tracking_utils.TIME_COLUMN].values,
    #     max_time_before_storm_start_sec=max_time_before_storm_start_sec,
    #     max_time_after_storm_end_sec=max_time_after_storm_end_sec)

    wind_table = _read_wind_observations(
        top_directory_name=top_wind_directory_name,
        storm_times_unix_sec=storm_object_table[
            tracking_utils.TIME_COLUMN].values,
        max_time_before_storm_start_sec=0, max_time_after_storm_end_sec=0)
    print SEPARATOR_STRING

    (global_centroid_lat_deg, global_centroid_lng_deg
    ) = geodetic_utils.get_latlng_centroid(
        latitudes_deg=storm_object_table[
            tracking_utils.CENTROID_LAT_COLUMN].values,
        longitudes_deg=storm_object_table[
            tracking_utils.CENTROID_LNG_COLUMN].values)

    projection_object = projections.init_azimuthal_equidistant_projection(
        global_centroid_lat_deg, global_centroid_lng_deg)

    storm_object_table = _project_storms_latlng_to_xy(
        storm_object_table, projection_object)
    wind_table = _project_events_latlng_to_xy(wind_table, projection_object)

    wind_x_limits_metres, wind_y_limits_metres = _get_bounding_box_of_storms(
        storm_object_table,
        padding_metres=padding_for_storm_bounding_box_metres)
    wind_table = _filter_events_by_location(
        event_table=wind_table, x_limits_metres=wind_x_limits_metres,
        y_limits_metres=wind_y_limits_metres)

    wind_to_storm_table = _find_nearest_storms(
        storm_object_table=storm_object_table, event_table=wind_table,
        max_time_before_storm_start_sec=max_time_before_storm_start_sec,
        max_time_after_storm_end_sec=max_time_after_storm_end_sec,
        max_link_distance_metres=max_link_distance_metres,
        interp_time_resolution_sec=interp_time_resolution_sec)
    print SEPARATOR_STRING

    return _create_storm_to_winds_table(storm_object_table, wind_to_storm_table)


def link_each_storm_to_tornadoes(
        tracking_file_names, tornado_directory_name,
        max_time_before_storm_start_sec=DEFAULT_MAX_TIME_BEFORE_STORM_SEC,
        max_time_after_storm_end_sec=DEFAULT_MAX_TIME_AFTER_STORM_SEC,
        padding_for_storm_bounding_box_metres=
        DEFAULT_PADDING_FOR_BOUNDING_BOX_METRES,
        interp_time_resolution_sec=
        DEFAULT_INTERP_TIME_RESOLUTION_FOR_TORNADOES_SEC,
        max_link_distance_metres=
        DEFAULT_MAX_LINK_DISTANCE_FOR_TORNADOES_METRES):
    """Links each storm cell to zero or more tornadoes.

    :param tracking_file_names: See documentation for
        `_check_linkage_params`.
    :param tornado_directory_name: Name of directory with tornado-report files
        (created by `tornado_io.write_processed_file`).
    :param max_time_before_storm_start_sec: See doc for `_check_linkage_params`.
    :param max_time_after_storm_end_sec: See doc for `_check_linkage_params`.
    :param padding_for_storm_bounding_box_metres: See doc for
        `_check_linkage_params`.
    :param interp_time_resolution_sec: See doc for `_check_linkage_params`.
    :param max_link_distance_metres: See doc for `_check_linkage_params`.
    :return: storm_to_tornadoes_table: pandas DataFrame created by
        `_create_storm_to_tornadoes_table`.
    """

    _check_linkage_params(
        tracking_file_names=tracking_file_names,
        max_time_before_storm_start_sec=max_time_before_storm_start_sec,
        max_time_after_storm_end_sec=max_time_after_storm_end_sec,
        padding_for_storm_bounding_box_metres=
        padding_for_storm_bounding_box_metres,
        interp_time_resolution_sec=interp_time_resolution_sec,
        max_link_distance_metres=max_link_distance_metres)

    storm_object_table = _read_storm_objects(tracking_file_names)
    print SEPARATOR_STRING

    # tornado_table = _read_tornado_reports(
    #     input_directory_name=tornado_directory_name,
    #     storm_times_unix_sec=storm_object_table[
    #         tracking_utils.TIME_COLUMN].values,
    #     max_time_before_storm_start_sec=max_time_before_storm_start_sec,
    #     max_time_after_storm_end_sec=max_time_after_storm_end_sec)

    tornado_table = _read_tornado_reports(
        input_directory_name=tornado_directory_name,
        storm_times_unix_sec=storm_object_table[
            tracking_utils.TIME_COLUMN].values,
        max_time_before_storm_start_sec=0, max_time_after_storm_end_sec=0)
    print SEPARATOR_STRING

    (global_centroid_lat_deg, global_centroid_lng_deg
    ) = geodetic_utils.get_latlng_centroid(
        latitudes_deg=storm_object_table[
            tracking_utils.CENTROID_LAT_COLUMN].values,
        longitudes_deg=storm_object_table[
            tracking_utils.CENTROID_LNG_COLUMN].values)

    projection_object = projections.init_azimuthal_equidistant_projection(
        global_centroid_lat_deg, global_centroid_lng_deg)

    storm_object_table = _project_storms_latlng_to_xy(
        storm_object_table, projection_object)
    tornado_table = _project_events_latlng_to_xy(
        tornado_table, projection_object)

    tornado_x_limits_metres, tornado_y_limits_metres = (
        _get_bounding_box_of_storms(
            storm_object_table,
            padding_metres=padding_for_storm_bounding_box_metres))
    tornado_table = _filter_events_by_location(
        event_table=tornado_table, x_limits_metres=tornado_x_limits_metres,
        y_limits_metres=tornado_y_limits_metres)

    tornado_to_storm_table = _find_nearest_storms(
        storm_object_table=storm_object_table, event_table=tornado_table,
        max_time_before_storm_start_sec=max_time_before_storm_start_sec,
        max_time_after_storm_end_sec=max_time_after_storm_end_sec,
        max_link_distance_metres=max_link_distance_metres,
        interp_time_resolution_sec=interp_time_resolution_sec)
    print SEPARATOR_STRING

    return _create_storm_to_tornadoes_table(
        storm_object_table, tornado_to_storm_table)


def share_linkages_across_spc_dates(
        first_spc_date_string, last_spc_date_string, top_input_dir_name,
        top_output_dir_name, event_type_string):
    """Shares linkage info across SPC dates.

    This is important because linkage is usually done for one SPC date at a
    time, using either `link_each_storm_to_winds` or
    `link_each_storm_to_tornadoes`, which have no way of sharing info with the
    next or previous SPC dates.  Thus, if event E is linked to storm cell S on
    date D, the files for dates D - 1 and D + 1 will not contain this linkage,
    even if storm cell S exists on date D - 1 or D + 1.

    N = number of SPC dates considered

    :param first_spc_date_string: First SPC date (format "yyyymmdd").  This
        method will share linkage info among all dates from
        `first_spc_date_string`...`last_spc_date_string`.
    :param last_spc_date_string: Same.
    :param top_input_dir_name: Name of top-level directory with original linkage
        files (before sharing info across SPC dates).
    :param top_output_dir_name: Name of top-level directory for new linkage
        files (after sharing across SPC dates).
    :param event_type_string: Either "wind" or "tornado".
    :return: new_storm_to_events_file_names: length-N list of paths to output
        files.
    """

    # Find input files, desired locations for output files.
    spc_date_strings = time_conversion.get_spc_dates_in_range(
        first_spc_date_string=first_spc_date_string,
        last_spc_date_string=last_spc_date_string)

    num_spc_dates = len(spc_date_strings)
    orig_linkage_file_names = [''] * num_spc_dates
    new_linkage_file_names = [''] * num_spc_dates

    for i in range(num_spc_dates):
        orig_linkage_file_names[i] = find_storm_to_events_file(
            top_directory_name=top_input_dir_name,
            event_type_string=event_type_string, raise_error_if_missing=True,
            spc_date_string=spc_date_strings[i])
        new_linkage_file_names[i] = find_storm_to_events_file(
            top_directory_name=top_output_dir_name,
            event_type_string=event_type_string, raise_error_if_missing=False,
            spc_date_string=spc_date_strings[i])

    if num_spc_dates == 1:
        warning_string = (
            'There is only one SPC date ("{0:s}"), so nothing to do.'
        ).format(spc_date_strings[0])
        warnings.warn(warning_string)

        if orig_linkage_file_names[0] == new_linkage_file_names[0]:
            return new_linkage_file_names

        print 'Copying file from "{0:s}" to "{1:s}"...'.format(
            orig_linkage_file_names[0], new_linkage_file_names[0])

        file_system_utils.mkdir_recursive_if_necessary(
            file_name=new_linkage_file_names[0])
        shutil.copyfile(orig_linkage_file_names[0], new_linkage_file_names[0])
        return new_linkage_file_names

    # Main loop.
    storm_to_events_table_by_date = [pandas.DataFrame()] * num_spc_dates

    for i in range(num_spc_dates + 1):
        if i == num_spc_dates:

            # Write new linkage tables for the last two SPC dates.
            for j in [num_spc_dates - 2, num_spc_dates - 1]:
                print 'Writing new linkages to: "{0:s}"...'.format(
                    new_linkage_file_names[j])

                if event_type_string == WIND_EVENT_TYPE_STRING:
                    write_storm_to_winds_table(
                        storm_to_winds_table=storm_to_events_table_by_date[j],
                        pickle_file_name=new_linkage_file_names[j])
                else:
                    write_storm_to_tornadoes_table(
                        storm_to_tornadoes_table=
                        storm_to_events_table_by_date[j],
                        pickle_file_name=new_linkage_file_names[j])

            break

        # Write and clear new linkage tables for the [i - 2]th SPC date.
        if i >= 2:
            print 'Writing new linkages to: "{0:s}"...'.format(
                new_linkage_file_names[i - 2])

            if event_type_string == WIND_EVENT_TYPE_STRING:
                write_storm_to_winds_table(
                    storm_to_winds_table=storm_to_events_table_by_date[i - 2],
                    pickle_file_name=new_linkage_file_names[i - 2])
            else:
                write_storm_to_tornadoes_table(
                    storm_to_tornadoes_table=
                    storm_to_events_table_by_date[i - 2],
                    pickle_file_name=new_linkage_file_names[i - 2])

            storm_to_events_table_by_date[i - 2] = pandas.DataFrame()

        # Read original linkage tables for SPC dates {i - 1, i, i + 1}.
        for j in [i - 1, i, i + 1]:
            if j < 0 or j >= num_spc_dates:
                continue
            if not storm_to_events_table_by_date[j].empty:
                continue

            print 'Reading original linkages from: "{0:s}"...'.format(
                orig_linkage_file_names[j])

            if event_type_string == WIND_EVENT_TYPE_STRING:
                storm_to_events_table_by_date[j] = read_storm_to_winds_table(
                    orig_linkage_file_names[j])
            else:
                storm_to_events_table_by_date[j] = (
                    read_storm_to_tornadoes_table(orig_linkage_file_names[j]))

        # Share linkage info between [i]th and [i + 1]th SPC dates.
        if i != num_spc_dates - 1:
            (storm_to_events_table_by_date[i],
             storm_to_events_table_by_date[i + 1]
            ) = _share_linkages_between_periods(
                early_storm_to_events_table=storm_to_events_table_by_date[i],
                late_storm_to_events_table=storm_to_events_table_by_date[i + 1],
                event_type_string=event_type_string)

            print SEPARATOR_STRING

    return new_linkage_file_names


def find_storm_to_events_file(
        top_directory_name, event_type_string, raise_error_if_missing=True,
        unix_time_sec=None, spc_date_string=None):
    """Finds linkage file for either one time step or one SPC date.

    In other words, the file should be one written by
    `write_storm_to_winds_table` or `write_storm_to_tornadoes_table`.

    :param top_directory_name: Name of top-level directory with linkage files.
    :param event_type_string: Either "wind" or "tornado".
    :param raise_error_if_missing: Boolean flag.  If file is missing and
        raise_error_if_missing = True, this method will error out.
    :param unix_time_sec: Valid time.
    :param spc_date_string: [used only if unix_time_sec is None]
        SPC date (format "yyyymmdd").
    :return: storm_to_events_file_name: File path.  If file is missing and
        raise_error_if_missing = False, this is the *expected* path.
    :raises: ValueError: if file is missing and raise_error_if_missing = True.
    """

    error_checking.assert_is_string(top_directory_name)
    check_event_type(event_type_string)
    error_checking.assert_is_boolean(raise_error_if_missing)

    if unix_time_sec is None:
        time_conversion.spc_date_string_to_unix_sec(spc_date_string)

        if event_type_string == WIND_EVENT_TYPE_STRING:
            storm_to_events_file_name = (
                '{0:s}/{1:s}/storm_to_winds_{2:s}.p'
            ).format(
                top_directory_name, spc_date_string[:4], spc_date_string)
        else:
            storm_to_events_file_name = (
                '{0:s}/{1:s}/storm_to_tornadoes_{2:s}.p'
            ).format(
                top_directory_name, spc_date_string[:4], spc_date_string)
    else:
        spc_date_string = time_conversion.time_to_spc_date_string(unix_time_sec)

        if event_type_string == WIND_EVENT_TYPE_STRING:
            storm_to_events_file_name = (
                '{0:s}/{1:s}/{2:s}/storm_to_winds_{3:s}.p'
            ).format(
                top_directory_name, spc_date_string[:4], spc_date_string,
                time_conversion.unix_sec_to_string(unix_time_sec, TIME_FORMAT))
        else:
            storm_to_events_file_name = (
                '{0:s}/{1:s}/{2:s}/storm_to_tornadoes_{3:s}.p'
            ).format(
                top_directory_name, spc_date_string[:4], spc_date_string,
                time_conversion.unix_sec_to_string(unix_time_sec, TIME_FORMAT))

    if raise_error_if_missing and not os.path.isfile(storm_to_events_file_name):
        error_string = (
            'Cannot find file with storm-to-event linkages.  Expected at: '
            '{0:s}').format(storm_to_events_file_name)
        raise ValueError(error_string)

    return storm_to_events_file_name


def write_linkages_one_file_per_storm_time(
        top_directory_name, file_times_unix_sec, storm_to_winds_table=None,
        storm_to_tornadoes_table=None):
    """Writes files with storm-to-wind or storm-to-tornado linkages.

    This method writes one file per time step.

    Only one of the last two input arguments may be specified.

    :param top_directory_name: Name of top-level directory for output files (see
        `find_storm_to_events_file`).
    :param file_times_unix_sec: 1-D numpy array of file times.
    :param storm_to_winds_table: pandas DataFrame created by
        `_create_storm_to_winds_table`.
    :param storm_to_tornadoes_table: pandas DataFrame created by
        `_create_storm_to_tornadoes_table`.
    :return: storm_to_events_file_names: 1-D list of paths to output files.
    """

    error_checking.assert_is_numpy_array(file_times_unix_sec, num_dimensions=1)
    if storm_to_winds_table is not None:
        event_type_string = WIND_EVENT_TYPE_STRING
    else:
        event_type_string = TORNADO_EVENT_TYPE_STRING

    num_files = len(file_times_unix_sec)
    storm_to_events_file_names = [''] * num_files
    for i in range(num_files):
        storm_to_events_file_names[i] = find_storm_to_events_file(
            top_directory_name=top_directory_name,
            unix_time_sec=file_times_unix_sec[i],
            event_type_string=event_type_string, raise_error_if_missing=False)

    for i in range(num_files):
        print 'Writing linkages to file: "{0:s}"...'.format(
            storm_to_events_file_names[i])

        if event_type_string == WIND_EVENT_TYPE_STRING:
            write_storm_to_winds_table(
                storm_to_winds_table.loc[
                    storm_to_winds_table[tracking_utils.TIME_COLUMN] ==
                    file_times_unix_sec[i]], storm_to_events_file_names[i])
        else:
            write_storm_to_tornadoes_table(
                storm_to_tornadoes_table.loc[
                    storm_to_tornadoes_table[tracking_utils.TIME_COLUMN] ==
                    file_times_unix_sec[i]], storm_to_events_file_names[i])

    return storm_to_events_file_names


def write_storm_to_winds_table(storm_to_winds_table, pickle_file_name):
    """Writes storm-to-wind linkages to Pickle file.

    :param storm_to_winds_table: pandas DataFrame with the following mandatory
        columns (may also contain distance buffers, for which column names are
        given by `storm_tracking_utils.distance_buffer_to_column_name`).  Each
        row is one storm object.
    storm_to_winds_table.storm_id: See documentation for `_read_storm_objects`.
    storm_to_winds_table.unix_time_sec: See doc for `_read_storm_objects`.
    storm_to_winds_table.tracking_start_time_unix_sec: See doc for
        `_read_storm_objects`.
    storm_to_winds_table.tracking_end_time_unix_sec: See doc for
        `_read_storm_objects`.
    storm_to_winds_table.centroid_lat_deg: See doc for `_read_storm_objects`.
    storm_to_winds_table.centroid_lng_deg: See doc for `_read_storm_objects`.
    storm_to_winds_table.polygon_object_latlng: See doc for
        `_read_storm_objects`.
    storm_to_winds_table.wind_station_ids: See doc for
        `_create_storm_to_winds_table`.
    storm_to_winds_table.event_latitudes_deg: See doc for
        `_create_storm_to_winds_table`.
    storm_to_winds_table.event_longitudes_deg: See doc for
        `_create_storm_to_winds_table`.
    storm_to_winds_table.u_winds_m_s01: See doc for
        `_create_storm_to_winds_table`.
    storm_to_winds_table.v_winds_m_s01: See doc for
        `_create_storm_to_winds_table`.
    storm_to_winds_table.linkage_distances_metres: See doc for
        `_create_storm_to_winds_table`.
    storm_to_winds_table.relative_event_times_sec: See doc for
        `_create_storm_to_winds_table`.

    :param pickle_file_name: Path to output file.
    """

    error_checking.assert_columns_in_dataframe(
        storm_to_winds_table, REQUIRED_STORM_TO_WINDS_COLUMNS)
    columns_to_write = get_columns_to_write(
        storm_to_winds_table=storm_to_winds_table)

    file_system_utils.mkdir_recursive_if_necessary(file_name=pickle_file_name)
    this_file_handle = open(pickle_file_name, 'wb')
    pickle.dump(storm_to_winds_table[columns_to_write], this_file_handle)
    this_file_handle.close()


def read_storm_to_winds_table(pickle_file_name):
    """Reads storm-to-wind linkages from Pickle file.

    :param pickle_file_name: Path to input file.
    :return: storm_to_winds_table: See documentation for
        `write_storm_to_winds_table`.
    """

    pickle_file_handle = open(pickle_file_name, 'rb')
    storm_to_winds_table = pickle.load(pickle_file_handle)
    pickle_file_handle.close()

    # TODO(thunderhoser): This is a HACK to deal with duplicates of the column
    # "cell_end_time_unix_sec".
    storm_to_winds_table = storm_to_winds_table.loc[
        :, ~storm_to_winds_table.columns.duplicated()]

    error_checking.assert_columns_in_dataframe(
        storm_to_winds_table, REQUIRED_STORM_TO_WINDS_COLUMNS)
    return storm_to_winds_table


def write_storm_to_tornadoes_table(storm_to_tornadoes_table, pickle_file_name):
    """Writes storm-to-tornado linkages to Pickle file.

    :param storm_to_tornadoes_table: pandas DataFrame with the following
        mandatory columns (may also contain distance buffers, for which column
        names are given by
        `storm_tracking_utils.distance_buffer_to_column_name`).  Each row is one
        storm object.
    storm_to_winds_table.storm_id: See documentation for `_read_storm_objects`.
    storm_to_winds_table.unix_time_sec: See doc for `_read_storm_objects`.
    storm_to_winds_table.tracking_start_time_unix_sec: See doc for
        `_read_storm_objects`.
    storm_to_winds_table.tracking_end_time_unix_sec: See doc for
        `_read_storm_objects`.
    storm_to_winds_table.centroid_lat_deg: See doc for `_read_storm_objects`.
    storm_to_winds_table.centroid_lng_deg: See doc for `_read_storm_objects`.
    storm_to_winds_table.polygon_object_latlng: See doc for
        `_read_storm_objects`.
    storm_to_winds_table.event_latitudes_deg: See doc for
        `_create_storm_to_tornadoes_table`.
    storm_to_winds_table.event_longitudes_deg: See doc for
        `_create_storm_to_tornadoes_table`.
    storm_to_winds_table.linkage_distances_metres: See doc for
        `_create_storm_to_tornadoes_table`.
    storm_to_winds_table.relative_event_times_sec: See doc for
        `_create_storm_to_tornadoes_table`.
    storm_to_winds_table.f_or_ef_scale_ratings: See doc for
        `_create_storm_to_tornadoes_table`.

    :param pickle_file_name: Path to output file.
    """

    error_checking.assert_columns_in_dataframe(
        storm_to_tornadoes_table, REQUIRED_STORM_TO_TORNADOES_COLUMNS)
    columns_to_write = get_columns_to_write(
        storm_to_tornadoes_table=storm_to_tornadoes_table)

    file_system_utils.mkdir_recursive_if_necessary(file_name=pickle_file_name)
    this_file_handle = open(pickle_file_name, 'wb')
    pickle.dump(storm_to_tornadoes_table[columns_to_write], this_file_handle)
    this_file_handle.close()


def read_storm_to_tornadoes_table(pickle_file_name):
    """Reads storm-to-tornado linkages from Pickle file.

    :param pickle_file_name: Path to input file.
    :return: storm_to_tornadoes_table: See documentation for
        `write_storm_to_tornadoes_table`.
    """

    pickle_file_handle = open(pickle_file_name, 'rb')
    storm_to_tornadoes_table = pickle.load(pickle_file_handle)
    pickle_file_handle.close()

    # TODO(thunderhoser): This is a HACK to deal with duplicates of the column
    # "cell_end_time_unix_sec".
    storm_to_tornadoes_table = storm_to_tornadoes_table.loc[
        :, ~storm_to_tornadoes_table.columns.duplicated()]

    error_checking.assert_columns_in_dataframe(
        storm_to_tornadoes_table, REQUIRED_STORM_TO_TORNADOES_COLUMNS)
    return storm_to_tornadoes_table
