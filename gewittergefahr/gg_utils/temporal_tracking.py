"""Handles the temporal part of storm-tracking ('connecting the dots')."""

import copy
from collections import OrderedDict
import numpy
import pandas
from geopy.distance import vincenty
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import storm_tracking_utils as tracking_utils
from gewittergefahr.gg_utils import error_checking

LARGE_INTEGER = int(1e10)
LOG_MESSAGE_TIME_FORMAT = '%Y-%m-%d-%H%M%S'

MAX_STORMS_IN_SPLIT = 2
MAX_STORMS_IN_MERGER = 2
DEFAULT_VELOCITY_EFOLD_RADIUS_METRES = 100000.
DEFAULT_MIN_VELOCITY_TIME_SEC = 590
DEFAULT_MAX_VELOCITY_TIME_SEC = 910

VALID_TIME_KEY = 'valid_time_unix_sec'
LATITUDES_KEY = 'latitudes_deg'
LONGITUDES_KEY = 'longitudes_deg'
X_COORDS_KEY = 'x_coords_metres'
Y_COORDS_KEY = 'y_coords_metres'
X_VELOCITIES_KEY = 'x_velocities_m_s01'
Y_VELOCITIES_KEY = 'y_velocities_m_s01'

PRIMARY_IDS_KEY = 'primary_storm_ids'
SECONDARY_IDS_KEY = 'secondary_storm_ids'
PREV_SECONDARY_IDS_KEY = 'prev_secondary_ids_listlist'
NEXT_SECONDARY_IDS_KEY = 'next_secondary_ids_listlist'
CURRENT_TO_PREV_MATRIX_KEY = 'current_to_previous_matrix'

CURRENT_LOCAL_MAXIMA_KEY = 'current_local_max_dict'
PREVIOUS_LOCAL_MAXIMA_KEY = 'previous_local_max_dict'
PREVIOUS_PRIMARY_ID_KEY = 'prev_primary_id_numeric'
PREVIOUS_SECONDARY_ID_KEY = 'prev_secondary_id_numeric'
OLD_TO_NEW_PRIMARY_IDS_KEY = 'old_to_new_primary_id_dict'

GRID_POINT_ROWS_KEY = 'grid_point_rows_arraylist'
GRID_POINT_COLUMNS_KEY = 'grid_point_columns_arraylist'
GRID_POINT_LATITUDES_KEY = 'grid_point_lats_arraylist_deg'
GRID_POINT_LONGITUDES_KEY = 'grid_point_lngs_arraylist_deg'
POLYGON_OBJECTS_ROWCOL_KEY = 'polygon_objects_rowcol'
POLYGON_OBJECTS_LATLNG_KEY = 'polygon_objects_latlng_deg'

CENTROID_X_COLUMN = 'centroid_x_metres'
CENTROID_Y_COLUMN = 'centroid_y_metres'
X_VELOCITY_COLUMN = 'x_velocity_m_s01'
Y_VELOCITY_COLUMN = 'y_velocity_m_s01'

PREV_SECONDARY_ID_COLUMNS = [
    tracking_utils.FIRST_PREV_SECONDARY_ID_COLUMN,
    tracking_utils.SECOND_PREV_SECONDARY_ID_COLUMN
]

NEXT_SECONDARY_ID_COLUMNS = [
    tracking_utils.FIRST_NEXT_SECONDARY_ID_COLUMN,
    tracking_utils.SECOND_NEXT_SECONDARY_ID_COLUMN
]

DEFAULT_EAST_VELOCITY_M_S01 = 0.
DEFAULT_NORTH_VELOCITY_M_S01 = 0.

ANY_CHANGE_STRING = 'any'
SPLIT_STRING = 'split'
MERGER_STRING = 'merger'

SECONDARY_ID_CHANGE_TYPE_STRINGS = [
    ANY_CHANGE_STRING, SPLIT_STRING, MERGER_STRING
]


def __find_predecessors(
        storm_object_table, target_row, max_num_sec_id_changes,
        change_type_string, return_all_on_path):
    """Finds all predecessors of one storm object.

    :param storm_object_table: See doc for public method `find_predecessors`.
    :param target_row: Same.
    :param max_num_sec_id_changes: Same.
    :param change_type_string: Same.
    :param return_all_on_path: Same.
    :return: predecessor_rows: Same.
    """

    unique_times_unix_sec, orig_to_unique_indices = numpy.unique(
        storm_object_table[tracking_utils.VALID_TIME_COLUMN].values,
        return_inverse=True
    )

    target_time_unix_sec = storm_object_table[
        tracking_utils.VALID_TIME_COLUMN].values[target_row]

    this_time_index = numpy.where(
        unique_times_unix_sec == target_time_unix_sec
    )[0][0]

    predecessor_rows = []
    rows_in_frontier = numpy.array([target_row], dtype=int)
    id_change_counts_in_frontier = numpy.array([0], dtype=int)

    while this_time_index >= 0:
        these_current_rows = numpy.where(
            orig_to_unique_indices == this_time_index
        )[0]

        old_rows_in_frontier = copy.deepcopy(rows_in_frontier)
        old_id_change_counts = copy.deepcopy(id_change_counts_in_frontier)

        rows_in_frontier = []
        id_change_counts_in_frontier = []

        for this_row, this_num_changes in zip(
                old_rows_in_frontier, old_id_change_counts
        ):
            if this_row not in these_current_rows:
                rows_in_frontier.append(this_row)
                id_change_counts_in_frontier.append(this_num_changes)
                continue

            these_previous_rows = __find_immediate_predecessors(
                storm_object_table=storm_object_table, target_row=this_row)

            these_change_flags = (
                storm_object_table[tracking_utils.SECONDARY_ID_COLUMN].values[
                    these_previous_rows] !=
                storm_object_table[tracking_utils.SECONDARY_ID_COLUMN].values[
                    this_row]
            )

            if change_type_string == MERGER_STRING:
                this_merger_flag = (
                    storm_object_table[
                        tracking_utils.SECOND_PREV_SECONDARY_ID_COLUMN
                    ].values[this_row] != ''
                )

                these_change_flags = numpy.logical_and(
                    these_change_flags, this_merger_flag)

            if change_type_string == SPLIT_STRING:
                these_split_flags = (
                    storm_object_table[
                        tracking_utils.SECOND_NEXT_SECONDARY_ID_COLUMN
                    ].values[these_previous_rows] != ''
                )

                these_change_flags = numpy.logical_and(
                    these_change_flags, these_split_flags)

            these_previous_num_changes = (
                this_num_changes + these_change_flags.astype(int)
            )

            these_good_indices = numpy.where(
                these_previous_num_changes <= max_num_sec_id_changes
            )[0]

            these_previous_rows = these_previous_rows[these_good_indices]
            these_previous_num_changes = these_previous_num_changes[
                these_good_indices]

            add_this_row = (
                (len(these_previous_rows) == 0 and this_row != target_row) or
                return_all_on_path
            )

            if add_this_row:
                predecessor_rows.append(this_row)

            if len(these_previous_rows) != 0:
                rows_in_frontier += these_previous_rows.tolist()
                id_change_counts_in_frontier += (
                    these_previous_num_changes.tolist()
                )

        this_time_index -= 1

        rows_in_frontier = numpy.array(rows_in_frontier, dtype=int)
        id_change_counts_in_frontier = numpy.array(
            id_change_counts_in_frontier, dtype=int)

        rows_in_frontier, these_unique_indices = numpy.unique(
            rows_in_frontier, return_index=True)
        id_change_counts_in_frontier = id_change_counts_in_frontier[
            these_unique_indices]

    return numpy.array(predecessor_rows + rows_in_frontier.tolist(), dtype=int)


def __find_successors(
        storm_object_table, target_row, max_num_sec_id_changes,
        change_type_string, return_all_on_path):
    """Finds all successors of one storm object.

    :param storm_object_table: See doc for public method `find_successors`.
    :param target_row: Same.
    :param max_num_sec_id_changes: Same.
    :param change_type_string: Same.
    :param return_all_on_path: Same.
    :return: successor_rows: Same.
    """

    unique_times_unix_sec, orig_to_unique_indices = numpy.unique(
        storm_object_table[tracking_utils.VALID_TIME_COLUMN].values,
        return_inverse=True
    )

    target_time_unix_sec = storm_object_table[
        tracking_utils.VALID_TIME_COLUMN].values[target_row]

    this_time_index = numpy.where(
        unique_times_unix_sec == target_time_unix_sec
    )[0][0]

    successor_rows = []
    rows_in_frontier = numpy.array([target_row], dtype=int)
    id_change_counts_in_frontier = numpy.array([0], dtype=int)

    while this_time_index < len(unique_times_unix_sec):
        these_current_rows = numpy.where(
            orig_to_unique_indices == this_time_index
        )[0]

        old_rows_in_frontier = copy.deepcopy(rows_in_frontier)
        old_id_change_counts = copy.deepcopy(id_change_counts_in_frontier)

        rows_in_frontier = []
        id_change_counts_in_frontier = []

        for this_row, this_num_changes in zip(
                old_rows_in_frontier, old_id_change_counts
        ):
            if this_row not in these_current_rows:
                rows_in_frontier.append(this_row)
                id_change_counts_in_frontier.append(this_num_changes)
                continue

            these_next_rows = __find_immediate_successors(
                storm_object_table=storm_object_table, target_row=this_row)

            these_change_flags = (
                storm_object_table[tracking_utils.SECONDARY_ID_COLUMN].values[
                    these_next_rows] !=
                storm_object_table[tracking_utils.SECONDARY_ID_COLUMN].values[
                    this_row]
            )

            if change_type_string == MERGER_STRING:
                these_merger_flags = (
                    storm_object_table[
                        tracking_utils.SECOND_PREV_SECONDARY_ID_COLUMN
                    ].values[these_next_rows] != ''
                )

                these_change_flags = numpy.logical_and(
                    these_change_flags, these_merger_flags)

            if change_type_string == SPLIT_STRING:
                this_split_flag = (
                    storm_object_table[
                        tracking_utils.SECOND_NEXT_SECONDARY_ID_COLUMN
                    ].values[this_row] != ''
                )

                these_change_flags = numpy.logical_and(
                    these_change_flags, this_split_flag)

            these_next_num_changes = (
                this_num_changes + these_change_flags.astype(int)
            )

            these_good_indices = numpy.where(
                these_next_num_changes <= max_num_sec_id_changes
            )[0]

            these_next_rows = these_next_rows[these_good_indices]
            these_next_num_changes = these_next_num_changes[these_good_indices]

            add_this_row = (
                (len(these_next_rows) == 0 and this_row != target_row) or
                return_all_on_path
            )

            if add_this_row:
                successor_rows.append(this_row)

            if len(these_next_rows) != 0:
                rows_in_frontier += these_next_rows.tolist()
                id_change_counts_in_frontier += these_next_num_changes.tolist()

        this_time_index += 1

        rows_in_frontier = numpy.array(rows_in_frontier, dtype=int)
        id_change_counts_in_frontier = numpy.array(
            id_change_counts_in_frontier, dtype=int)

        rows_in_frontier, these_unique_indices = numpy.unique(
            rows_in_frontier, return_index=True)
        id_change_counts_in_frontier = id_change_counts_in_frontier[
            these_unique_indices]

    return numpy.array(successor_rows + rows_in_frontier.tolist(), dtype=int)


def __find_immediate_predecessors(storm_object_table, target_row):
    """Finds immediate predecessors of one storm object.

    :param storm_object_table: See doc for public method
        `find_immediate_predecessors`.
    :param target_row: Same.
    :return: predecessor_rows: Same.
    """

    error_checking.assert_is_integer(target_row)
    error_checking.assert_is_geq(target_row, 0)
    error_checking.assert_is_less_than(
        target_row, len(storm_object_table.index)
    )

    predecessor_sec_id_strings = [
        storm_object_table[c].values[target_row]
        for c in PREV_SECONDARY_ID_COLUMNS
        if storm_object_table[c].values[target_row] != ''
    ]

    num_predecessors = len(predecessor_sec_id_strings)
    if num_predecessors == 0:
        return numpy.array([], dtype=int)

    target_time_unix_sec = storm_object_table[
        tracking_utils.VALID_TIME_COLUMN].values[target_row]

    predecessor_rows = numpy.full(num_predecessors, -1, dtype=int)

    for i in range(num_predecessors):
        these_rows = numpy.where(numpy.logical_and(
            storm_object_table[tracking_utils.SECONDARY_ID_COLUMN].values ==
            predecessor_sec_id_strings[i],
            storm_object_table[tracking_utils.VALID_TIME_COLUMN].values <
            target_time_unix_sec
        ))[0]

        if len(these_rows) == 0:
            continue

        this_subrow = numpy.argmax(
            storm_object_table[tracking_utils.VALID_TIME_COLUMN].values[
                these_rows]
        )

        predecessor_rows[i] = these_rows[this_subrow]

    return predecessor_rows[predecessor_rows >= 0]


def __find_immediate_successors(storm_object_table, target_row):
    """Finds immediate successors of one storm object.

    :param storm_object_table: See doc for public method
        `find_immediate_successors`.
    :param target_row: Same.
    :return: successor_rows: Same.
    """

    error_checking.assert_is_integer(target_row)
    error_checking.assert_is_geq(target_row, 0)
    error_checking.assert_is_less_than(
        target_row, len(storm_object_table.index)
    )

    successor_sec_id_strings = [
        storm_object_table[c].values[target_row]
        for c in NEXT_SECONDARY_ID_COLUMNS
        if storm_object_table[c].values[target_row] != ''
    ]

    num_successors = len(successor_sec_id_strings)
    if num_successors == 0:
        return numpy.array([], dtype=int)

    target_time_unix_sec = storm_object_table[
        tracking_utils.VALID_TIME_COLUMN].values[target_row]

    successor_rows = numpy.full(num_successors, -1, dtype=int)

    for i in range(num_successors):
        these_rows = numpy.where(numpy.logical_and(
            storm_object_table[tracking_utils.SECONDARY_ID_COLUMN].values ==
            successor_sec_id_strings[i],
            storm_object_table[tracking_utils.VALID_TIME_COLUMN].values >
            target_time_unix_sec
        ))[0]

        if len(these_rows) == 0:
            continue

        this_subrow = numpy.argmin(
            storm_object_table[tracking_utils.VALID_TIME_COLUMN].values[
                these_rows]
        )

        successor_rows[i] = these_rows[this_subrow]

    return successor_rows[successor_rows >= 0]


def _estimate_velocity_by_neigh(
        x_coords_metres, y_coords_metres, x_velocities_m_s01,
        y_velocities_m_s01, e_folding_radius_metres):
    """Estimates missing velocities based on non-missing velocities in neigh.

    Specifically, this method replaces each missing velocity with an
    exponentially weighted average of neighbouring non-missing velocities.

    If `e_folding_radius_metres` is NaN, this method will use an unweighted
    average.

    N = number of storm objects

    :param x_coords_metres: length-N numpy array of x-coordinates.
    :param y_coords_metres: length-N numpy array of y-coordinates.
    :param x_velocities_m_s01: length-N numpy array of x-velocities (metres per
        second in positive x-direction).  Some of these may be NaN.
    :param y_velocities_m_s01: Same but for y-direction.
    :param e_folding_radius_metres: e-folding radius for exponentially weighted
        average.
    :return: x_velocities_m_s01: Same as input but without NaN.
    :return: y_velocities_m_s01: Same as input but without NaN.
    """

    if numpy.isnan(e_folding_radius_metres):
        neigh_radius_metres = numpy.inf
    else:
        neigh_radius_metres = 3 * e_folding_radius_metres

    orig_x_velocities_m_s01 = x_velocities_m_s01 + 0.
    orig_y_velocities_m_s01 = y_velocities_m_s01 + 0.

    nan_flags = numpy.logical_or(
        numpy.isnan(orig_x_velocities_m_s01),
        numpy.isnan(orig_y_velocities_m_s01)
    )
    nan_indices = numpy.where(nan_flags)[0]

    for this_index in nan_indices:
        if numpy.isnan(e_folding_radius_metres):
            these_neighbour_indices = numpy.where(numpy.invert(nan_flags))[0]
            if len(these_neighbour_indices) == 0:
                continue

            x_velocities_m_s01[this_index] = numpy.mean(
                orig_x_velocities_m_s01[these_neighbour_indices]
            )

            y_velocities_m_s01[this_index] = numpy.mean(
                orig_y_velocities_m_s01[these_neighbour_indices]
            )

            continue

        these_x_diffs_metres = numpy.absolute(
            x_coords_metres[this_index] - x_coords_metres)
        these_y_diffs_metres = numpy.absolute(
            y_coords_metres[this_index] - y_coords_metres)

        these_neighbour_flags = numpy.logical_and(
            these_x_diffs_metres <= neigh_radius_metres,
            these_y_diffs_metres <= neigh_radius_metres)

        these_neighbour_flags = numpy.logical_and(
            these_neighbour_flags, numpy.invert(nan_flags)
        )

        these_neighbour_indices = numpy.where(these_neighbour_flags)[0]
        if len(these_neighbour_indices) == 0:
            continue

        these_neighbour_dist_metres = numpy.sqrt(
            these_x_diffs_metres[these_neighbour_indices] ** 2 +
            these_y_diffs_metres[these_neighbour_indices] ** 2
        )

        these_neighbour_subindices = numpy.where(
            these_neighbour_dist_metres <= neigh_radius_metres
        )[0]
        if len(these_neighbour_subindices) == 0:
            continue

        these_neighbour_indices = these_neighbour_indices[
            these_neighbour_subindices]
        these_neighbour_dist_metres = these_neighbour_dist_metres[
            these_neighbour_subindices]

        these_weights = numpy.exp(
            -these_neighbour_dist_metres / e_folding_radius_metres
        )
        these_weights = these_weights / numpy.sum(these_weights)

        x_velocities_m_s01[this_index] = numpy.sum(
            these_weights * orig_x_velocities_m_s01[these_neighbour_indices]
        )

        y_velocities_m_s01[this_index] = numpy.sum(
            these_weights * orig_y_velocities_m_s01[these_neighbour_indices]
        )

    return x_velocities_m_s01, y_velocities_m_s01


def _link_local_maxima_by_velocity(
        current_local_max_dict, previous_local_max_dict,
        max_velocity_diff_m_s01):
    """Does velocity-matching for local maxima at successive times.

    N_c = number of maxima at current time
    N_p = number of maxima at previous time

    :param current_local_max_dict: See doc for `link_local_maxima_in_time`.
    :param previous_local_max_dict: Same.
    :param max_velocity_diff_m_s01: Same.
    :return: velocity_diff_matrix_m_s01: numpy array (N_c x N_p) of velocity
        errors.  velocity_diff_matrix_m_s01[i, j] is the velocity error (metres
        per second) between the [i]th current max and [j]th previous max.
    :return: current_to_previous_matrix: See doc for
        `link_local_maxima_in_time`.
    """

    num_current_maxima = len(current_local_max_dict[X_COORDS_KEY])
    num_previous_maxima = len(previous_local_max_dict[X_COORDS_KEY])

    time_diff_seconds = (
        current_local_max_dict[VALID_TIME_KEY] -
        previous_local_max_dict[VALID_TIME_KEY]
    )

    extrap_x_coords_metres = (
        previous_local_max_dict[X_COORDS_KEY] +
        previous_local_max_dict[X_VELOCITIES_KEY] * time_diff_seconds
    )

    extrap_y_coords_metres = (
        previous_local_max_dict[Y_COORDS_KEY] +
        previous_local_max_dict[Y_VELOCITIES_KEY] * time_diff_seconds
    )

    velocity_diff_matrix_m_s01 = numpy.full(
        (num_current_maxima, num_previous_maxima), numpy.nan)
    current_to_previous_matrix = numpy.full(
        (num_current_maxima, num_previous_maxima), False, dtype=bool)

    for i in range(num_current_maxima):
        these_distances_metres = numpy.sqrt(
            (extrap_x_coords_metres - current_local_max_dict[X_COORDS_KEY][i])
            ** 2 +
            (extrap_y_coords_metres - current_local_max_dict[Y_COORDS_KEY][i])
            ** 2
        )

        these_velocity_diffs_m_s01 = these_distances_metres / time_diff_seconds
        these_velocity_diffs_m_s01[
            numpy.isnan(these_velocity_diffs_m_s01)
        ] = numpy.inf

        sort_indices = numpy.argsort(these_velocity_diffs_m_s01)

        for j in sort_indices:
            if these_velocity_diffs_m_s01[j] > max_velocity_diff_m_s01:
                break

            if (numpy.sum(current_to_previous_matrix[i, :]) >=
                    MAX_STORMS_IN_MERGER):
                break

            current_to_previous_matrix[i, j] = True

        velocity_diff_matrix_m_s01[i, :] = these_velocity_diffs_m_s01

    return velocity_diff_matrix_m_s01, current_to_previous_matrix


def _link_local_maxima_by_distance(
        current_local_max_dict, previous_local_max_dict,
        max_link_distance_m_s01, current_to_previous_matrix):
    """Does distance-matching for local maxima at successive times.

    N_c = number of maxima at current time
    N_p = number of maxima at previous time

    :param current_local_max_dict: See doc for `link_local_maxima_in_time`.
    :param previous_local_max_dict: Same.
    :param max_link_distance_m_s01: Same.
    :param current_to_previous_matrix: numpy array created by
        `_link_local_maxima_by_velocity`.
    :return: distance_matrix_m_s01: numpy array (N_c x N_p) of distances.
        distance_matrix_m_s01[i, j] is the distance (metres per second) between
        the [i]th current max and [j]th previous max.
    :return: current_to_previous_matrix: Same as input, except that some
        elements might have been flipped from False to True.
    """

    num_current_maxima = len(current_local_max_dict[X_COORDS_KEY])
    num_previous_maxima = len(previous_local_max_dict[X_COORDS_KEY])

    time_diff_seconds = (
        current_local_max_dict[VALID_TIME_KEY] -
        previous_local_max_dict[VALID_TIME_KEY]
    )

    distance_matrix_m_s01 = numpy.full(
        (num_current_maxima, num_previous_maxima), numpy.nan)

    for i in range(num_current_maxima):
        if numpy.sum(current_to_previous_matrix[i, :]) >= MAX_STORMS_IN_MERGER:
            continue

        these_distances_metres = numpy.sqrt(
            (previous_local_max_dict[X_COORDS_KEY] -
             current_local_max_dict[X_COORDS_KEY][i]) ** 2
            +
            (previous_local_max_dict[Y_COORDS_KEY] -
             current_local_max_dict[Y_COORDS_KEY][i]) ** 2
        )

        these_distances_m_s01 = these_distances_metres / time_diff_seconds
        # these_distances_m_s01[
        #     numpy.invert(numpy.isnan(previous_local_max_dict[X_VELOCITIES_KEY]))
        # ] = numpy.inf

        sort_indices = numpy.argsort(these_distances_m_s01)

        for j in sort_indices:
            if these_distances_m_s01[j] > max_link_distance_m_s01:
                break

            if (numpy.sum(current_to_previous_matrix[i, :]) >=
                    MAX_STORMS_IN_MERGER):
                break

            current_to_previous_matrix[i, j] = True

        distance_matrix_m_s01[i, :] = these_distances_m_s01

    return distance_matrix_m_s01, current_to_previous_matrix


def _prune_connections(velocity_diff_matrix_m_s01, distance_matrix_m_s01,
                       current_to_previous_matrix):
    """Prunes connections between local maxima at successive times.

    :param velocity_diff_matrix_m_s01: numpy array created by
        `_link_local_maxima_by_velocity`.
    :param distance_matrix_m_s01: numpy array created by
        `_link_local_maxima_by_distance`.
    :param current_to_previous_matrix: Same.
    :return: current_to_previous_matrix: Same as input, except that some
        elements might have been flipped from True to False.
    """

    num_current_by_previous = numpy.sum(current_to_previous_matrix, axis=0)
    previous_indices = numpy.argsort(-1 * num_current_by_previous)

    for j in previous_indices:
        this_worst_current_index = -1

        while this_worst_current_index is not None:
            these_current_indices = numpy.where(
                current_to_previous_matrix[:, j]
            )[0]

            this_worst_current_index = None

            # If [j]th previous local max is involved in a split:
            if len(these_current_indices) > 1:
                this_num_previous_by_current = numpy.array([
                    numpy.sum(current_to_previous_matrix[i, :])
                    for i in these_current_indices
                ], dtype=int)

                # Current local max cannot be involved in both a merger and a
                # split.
                if numpy.max(this_num_previous_by_current) > 1:
                    this_worst_current_index = these_current_indices[
                        numpy.argmax(this_num_previous_by_current)
                    ]

            if this_worst_current_index is not None:
                current_to_previous_matrix[this_worst_current_index, j] = False
                continue

            if len(these_current_indices) <= MAX_STORMS_IN_SPLIT:
                continue

            this_max_velocity_diff_m_s01 = numpy.max(
                velocity_diff_matrix_m_s01[these_current_indices, j]
            )

            if numpy.isinf(this_max_velocity_diff_m_s01):
                this_worst_current_index = numpy.argmax(
                    distance_matrix_m_s01[these_current_indices, j]
                )
            else:
                this_worst_current_index = numpy.argmax(
                    velocity_diff_matrix_m_s01[these_current_indices, j]
                )

            this_worst_current_index = these_current_indices[
                this_worst_current_index]

            current_to_previous_matrix[this_worst_current_index, j] = False

    return current_to_previous_matrix


def _create_partial_storm_id(previous_numeric_id):
    """Creates partial (primary or secondary) storm ID.

    :param previous_numeric_id: Integer ID for previous storm.
    :return: string_id: String ID for new storm.
    :return: numeric_id: Integer ID for new storm.
    """

    numeric_id = previous_numeric_id + 1
    return '{0:d}'.format(numeric_id), numeric_id


def _local_maxima_to_tracks_mergers(
        current_local_max_dict, previous_local_max_dict,
        current_to_previous_matrix, prev_primary_id_numeric,
        prev_secondary_id_numeric):
    """Handles mergers for `local_maxima_to_storm_tracks`.

    N_c = number of maxima at current time
    N_p = number of maxima at previous time

    :param current_local_max_dict: Dictionary with the following keys.
    current_local_max_dict['valid_time_unix_sec']: Valid time.
    current_local_max_dict['latitudes_deg']: numpy array (length N_c) with
        latitudes (deg N) of local maxima.
    current_local_max_dict['longitudes_deg']: numpy array (length N_c) with
        longitudes (deg E) of local maxima.
    current_local_max_dict['x_coords_metres']: numpy array (length N_c) with
        x-coordinates of local maxima.
    current_local_max_dict['y_coords_metres']: numpy array (length N_c) with
        y-coordinates of local maxima.
    current_local_max_dict['primary_storm_ids']: List (length N_c) of primary
        storm IDs (strings).
    current_local_max_dict['secondary_storm_ids']: List (length N_c) of
        secondary storm IDs (strings).
    current_local_max_dict['prev_secondary_ids_listlist']: List (length N_c),
        where prev_secondary_ids_listlist[i] is a 1-D list with secondary IDs of
        previous maxima to which the [i]th current max is linked.
    current_local_max_dict['next_secondary_ids_listlist']: List (length N_c),
        where next_secondary_ids_listlist[i] is a 1-D list with secondary IDs of
        next maxima to which the [i]th current max is linked.

    :param previous_local_max_dict: Same.
    :param current_to_previous_matrix: numpy array (N_c x N_p) of Boolean
        flags.  If current_to_previous_matrix[i, j] = True, the [i]th local max
        at the current time is linked to the [j]th local max at the previous
        time.
    :param prev_primary_id_numeric: Previous primary storm ID used.
    :param prev_secondary_id_numeric: Previous secondary storm ID used.
    :return: intermediate_track_dict: Dictionary with the following keys.
    intermediate_track_dict['current_local_max_dict']: Same as input but maybe
        with different IDs.
    intermediate_track_dict['previous_local_max_dict']: Same as input but maybe
        with different IDs.
    intermediate_track_dict['current_to_previous_matrix']: Same as input but
        with some elements (those involved in mergers) flipped from True to
        False.
    intermediate_track_dict['prev_primary_id_numeric']: Same as input but
        possibly incremented.
    intermediate_track_dict['prev_secondary_id_numeric']: Same as input but
        possibly incremented.
    intermediate_track_dict['old_to_new_primary_id_dict']: Dictionary, where
        each key is an old primary ID (string) and each value is the new primary
        ID (string) to replace it with.
    """

    old_to_new_primary_id_dict = OrderedDict({})

    num_previous_by_current = numpy.sum(current_to_previous_matrix, axis=1)
    current_indices_in_merger = numpy.where(num_previous_by_current > 1)[0]

    for i in current_indices_in_merger:
        these_previous_indices = numpy.where(
            current_to_previous_matrix[i, :]
        )[0]
        current_to_previous_matrix[i, :] = False

        # Create new secondary ID for [i]th local max at current time (because
        # this local max results from merger of two previous maxima).
        (current_local_max_dict[SECONDARY_IDS_KEY][i], prev_secondary_id_numeric
        ) = _create_partial_storm_id(prev_secondary_id_numeric)

        # Create new primary ID for [i]th local max at current time.  This will
        # be shared with the two previous local maxima involved in the merger.
        this_primary_id_string, prev_primary_id_numeric = (
            _create_partial_storm_id(prev_primary_id_numeric)
        )

        current_local_max_dict[PRIMARY_IDS_KEY][i] = this_primary_id_string

        for j in these_previous_indices:
            current_local_max_dict[PREV_SECONDARY_IDS_KEY][i].append(
                previous_local_max_dict[SECONDARY_IDS_KEY][j]
            )

            previous_local_max_dict[NEXT_SECONDARY_IDS_KEY][j].append(
                current_local_max_dict[SECONDARY_IDS_KEY][i]
            )

            this_old_id_string = previous_local_max_dict[PRIMARY_IDS_KEY][j]
            old_to_new_primary_id_dict[
                this_old_id_string
            ] = this_primary_id_string

    return {
        CURRENT_LOCAL_MAXIMA_KEY: current_local_max_dict,
        PREVIOUS_LOCAL_MAXIMA_KEY: previous_local_max_dict,
        CURRENT_TO_PREV_MATRIX_KEY: current_to_previous_matrix,
        PREVIOUS_PRIMARY_ID_KEY: prev_primary_id_numeric,
        PREVIOUS_SECONDARY_ID_KEY: prev_secondary_id_numeric,
        OLD_TO_NEW_PRIMARY_IDS_KEY: old_to_new_primary_id_dict
    }


def _local_maxima_to_tracks_splits(
        current_local_max_dict, previous_local_max_dict,
        current_to_previous_matrix, prev_secondary_id_numeric):
    """Handles splits for `local_maxima_to_storm_tracks`.

    N_c = number of maxima at current time
    N_p = number of maxima at previous time

    :param current_local_max_dict: See doc for
        `_local_maxima_to_tracks_mergers`.
    :param previous_local_max_dict: Same.
    :param current_to_previous_matrix: numpy array (N_c x N_p) of Boolean
        flags.  If current_to_previous_matrix[i, j] = True, the [i]th local max
        at the current time is linked to the [j]th local max at the previous
        time.
    :param prev_secondary_id_numeric: Previous secondary storm ID used.
    :return: intermediate_track_dict: Dictionary with the following keys.
    intermediate_track_dict['current_local_max_dict']: Same as input but maybe
        with different IDs.
    intermediate_track_dict['previous_local_max_dict']: Same as input but maybe
        with different IDs.
    intermediate_track_dict['current_to_previous_matrix']: Same as input but
        with some elements (those involved in mergers) flipped from True to
        False.
    intermediate_track_dict['prev_secondary_id_numeric']: Same as input but
        possibly incremented.
    """

    num_current_by_previous = numpy.sum(current_to_previous_matrix, axis=0)
    previous_indices_in_split = numpy.where(num_current_by_previous > 1)[0]

    for j in previous_indices_in_split:
        these_current_indices = numpy.where(current_to_previous_matrix[:, j])[0]
        current_to_previous_matrix[:, j] = False

        # Find primary ID of [j]th local max at previous time (the one that
        # split).  Share this primary ID with the two current local maxima
        # involved in the split.
        this_primary_id_string = previous_local_max_dict[PRIMARY_IDS_KEY][j]

        for i in these_current_indices:
            current_local_max_dict[PRIMARY_IDS_KEY][i] = this_primary_id_string

            # Create new secondary ID for [i]th local max at current time.
            (current_local_max_dict[SECONDARY_IDS_KEY][i],
             prev_secondary_id_numeric
            ) = _create_partial_storm_id(prev_secondary_id_numeric)

            current_local_max_dict[PREV_SECONDARY_IDS_KEY][i].append(
                previous_local_max_dict[SECONDARY_IDS_KEY][j]
            )

            previous_local_max_dict[NEXT_SECONDARY_IDS_KEY][j].append(
                current_local_max_dict[SECONDARY_IDS_KEY][i]
            )

    return {
        CURRENT_LOCAL_MAXIMA_KEY: current_local_max_dict,
        PREVIOUS_LOCAL_MAXIMA_KEY: previous_local_max_dict,
        CURRENT_TO_PREV_MATRIX_KEY: current_to_previous_matrix,
        PREVIOUS_SECONDARY_ID_KEY: prev_secondary_id_numeric
    }


def _local_maxima_to_tracks_simple(
        current_local_max_dict, previous_local_max_dict,
        current_to_previous_matrix, prev_primary_id_numeric,
        prev_secondary_id_numeric):
    """Handles simple connections for `local_maxima_to_storm_tracks`.

    "Simple connections" are those other than splits and mergers.

    N_c = number of maxima at current time
    N_p = number of maxima at previous time

    :param current_local_max_dict: See doc for
        `_local_maxima_to_tracks_mergers`.
    :param previous_local_max_dict: Same.
    :param current_to_previous_matrix: numpy array (N_c x N_p) of Boolean
        flags.  If current_to_previous_matrix[i, j] = True, the [i]th local max
        at the current time is linked to the [j]th local max at the previous
        time.
    :param prev_primary_id_numeric: Previous primary storm ID used.
    :param prev_secondary_id_numeric: Previous secondary storm ID used.
    :return: intermediate_track_dict: Dictionary with the following keys.
    intermediate_track_dict['current_local_max_dict']: Same as input but maybe
        with different IDs.
    intermediate_track_dict['previous_local_max_dict']: Same as input but maybe
        with different IDs.
    intermediate_track_dict['prev_primary_id_numeric']: Same as input but
        possibly incremented.
    intermediate_track_dict['prev_secondary_id_numeric']: Same as input but
        possibly incremented.
    """

    num_storm_objects = len(current_local_max_dict[X_COORDS_KEY])

    for i in range(num_storm_objects):

        # If [i]th local max at current time already has primary ID, there is
        # nothing to do.
        if current_local_max_dict[PRIMARY_IDS_KEY][i]:
            continue

        these_previous_indices = numpy.where(
            current_to_previous_matrix[i, :]
        )[0]

        # If [i]th local max at current time is beginning of a new track:
        if len(these_previous_indices) == 0:
            (current_local_max_dict[PRIMARY_IDS_KEY][i], prev_primary_id_numeric
            ) = _create_partial_storm_id(prev_primary_id_numeric)

            (current_local_max_dict[SECONDARY_IDS_KEY][i],
             prev_secondary_id_numeric
            ) = _create_partial_storm_id(prev_secondary_id_numeric)

            continue

        # If [i]th local max at current time is linked to [j]th local max at
        # previous time:
        j = these_previous_indices[0]

        current_local_max_dict[PRIMARY_IDS_KEY][i] = previous_local_max_dict[
            PRIMARY_IDS_KEY][j]
        current_local_max_dict[SECONDARY_IDS_KEY][i] = previous_local_max_dict[
            SECONDARY_IDS_KEY][j]

        current_local_max_dict[PREV_SECONDARY_IDS_KEY][i].append(
            previous_local_max_dict[SECONDARY_IDS_KEY][j]
        )

        previous_local_max_dict[NEXT_SECONDARY_IDS_KEY][j].append(
            current_local_max_dict[SECONDARY_IDS_KEY][i]
        )

    return {
        CURRENT_LOCAL_MAXIMA_KEY: current_local_max_dict,
        PREVIOUS_LOCAL_MAXIMA_KEY: previous_local_max_dict,
        PREVIOUS_PRIMARY_ID_KEY: prev_primary_id_numeric,
        PREVIOUS_SECONDARY_ID_KEY: prev_secondary_id_numeric
    }


def _get_storm_velocities_missing(
        storm_object_table,
        e_folding_radius_metres=DEFAULT_VELOCITY_EFOLD_RADIUS_METRES):
    """Fills velocities that could not be estimated by `get_storm_velocities`.

    :param storm_object_table: See input doc for `get_storm_velocities`.
    :param e_folding_radius_metres: See doc for `_estimate_velocity_by_neigh`.
    :return: storm_object_table: See output doc for `get_storm_velocities`.
    """

    east_velocities_m_s01 = storm_object_table[
        tracking_utils.EAST_VELOCITY_COLUMN].values

    north_velocities_m_s01 = storm_object_table[
        tracking_utils.NORTH_VELOCITY_COLUMN].values

    if not numpy.any(numpy.isnan(east_velocities_m_s01)):
        return storm_object_table

    unique_times_unix_sec, orig_to_unique_indices = numpy.unique(
        storm_object_table[tracking_utils.VALID_TIME_COLUMN].values,
        return_inverse=True)

    num_times = len(unique_times_unix_sec)

    # Use neighbouring storms at same time to estimate missing velocities.
    for j in range(num_times):
        these_indices = numpy.where(orig_to_unique_indices == j)[0]
        if not numpy.any(numpy.isnan(east_velocities_m_s01[these_indices])):
            continue

        (east_velocities_m_s01[these_indices],
         north_velocities_m_s01[these_indices]
        ) = _estimate_velocity_by_neigh(
            x_coords_metres=storm_object_table[
                CENTROID_X_COLUMN].values[these_indices],
            y_coords_metres=storm_object_table[
                CENTROID_Y_COLUMN].values[these_indices],
            x_velocities_m_s01=east_velocities_m_s01[these_indices],
            y_velocities_m_s01=north_velocities_m_s01[these_indices],
            e_folding_radius_metres=e_folding_radius_metres)

    if not numpy.any(numpy.isnan(east_velocities_m_s01)):
        return storm_object_table.assign(**{
            tracking_utils.EAST_VELOCITY_COLUMN: east_velocities_m_s01,
            tracking_utils.NORTH_VELOCITY_COLUMN: north_velocities_m_s01
        })

    # Use all storms at same time to estimate missing velocities.
    for j in range(num_times):
        these_indices = numpy.where(orig_to_unique_indices == j)[0]
        if not numpy.any(numpy.isnan(east_velocities_m_s01[these_indices])):
            continue

        (east_velocities_m_s01[these_indices],
         north_velocities_m_s01[these_indices]
        ) = _estimate_velocity_by_neigh(
            x_coords_metres=storm_object_table[
                CENTROID_X_COLUMN].values[these_indices],
            y_coords_metres=storm_object_table[
                CENTROID_Y_COLUMN].values[these_indices],
            x_velocities_m_s01=east_velocities_m_s01[these_indices],
            y_velocities_m_s01=north_velocities_m_s01[these_indices],
            e_folding_radius_metres=numpy.nan)

    if not numpy.any(numpy.isnan(east_velocities_m_s01)):
        return storm_object_table.assign(**{
            tracking_utils.EAST_VELOCITY_COLUMN: east_velocities_m_s01,
            tracking_utils.NORTH_VELOCITY_COLUMN: north_velocities_m_s01
        })

    # Use neighbouring storms at all times to estimate missing velocities.
    for j in range(num_times):
        these_indices = numpy.where(orig_to_unique_indices == j)[0]
        if not numpy.any(numpy.isnan(east_velocities_m_s01[these_indices])):
            continue

        these_east_velocities_m_s01, these_north_velocities_m_s01 = (
            _estimate_velocity_by_neigh(
                x_coords_metres=storm_object_table[CENTROID_X_COLUMN].values,
                y_coords_metres=storm_object_table[CENTROID_Y_COLUMN].values,
                x_velocities_m_s01=east_velocities_m_s01 + 0.,
                y_velocities_m_s01=north_velocities_m_s01 + 0.,
                e_folding_radius_metres=e_folding_radius_metres)
        )

        east_velocities_m_s01[these_indices] = these_east_velocities_m_s01[
            these_indices]
        north_velocities_m_s01[these_indices] = these_north_velocities_m_s01[
            these_indices]

    if not numpy.any(numpy.isnan(east_velocities_m_s01)):
        return storm_object_table.assign(**{
            tracking_utils.EAST_VELOCITY_COLUMN: east_velocities_m_s01,
            tracking_utils.NORTH_VELOCITY_COLUMN: north_velocities_m_s01
        })

    # Use all storms at all times to estimate missing velocities.
    for j in range(num_times):
        these_indices = numpy.where(orig_to_unique_indices == j)[0]
        if not numpy.any(numpy.isnan(east_velocities_m_s01[these_indices])):
            continue

        these_east_velocities_m_s01, these_north_velocities_m_s01 = (
            _estimate_velocity_by_neigh(
                x_coords_metres=storm_object_table[CENTROID_X_COLUMN].values,
                y_coords_metres=storm_object_table[CENTROID_Y_COLUMN].values,
                x_velocities_m_s01=east_velocities_m_s01 + 0.,
                y_velocities_m_s01=north_velocities_m_s01 + 0.,
                e_folding_radius_metres=numpy.nan)
        )

        east_velocities_m_s01[these_indices] = these_east_velocities_m_s01[
            these_indices]
        north_velocities_m_s01[these_indices] = these_north_velocities_m_s01[
            these_indices]

    if not numpy.any(numpy.isnan(east_velocities_m_s01)):
        return storm_object_table.assign(**{
            tracking_utils.EAST_VELOCITY_COLUMN: east_velocities_m_s01,
            tracking_utils.NORTH_VELOCITY_COLUMN: north_velocities_m_s01
        })

    # Replace missing velocities with defaults.
    nan_indices = numpy.where(numpy.isnan(east_velocities_m_s01))[0]
    east_velocities_m_s01[nan_indices] = DEFAULT_EAST_VELOCITY_M_S01
    north_velocities_m_s01[nan_indices] = DEFAULT_NORTH_VELOCITY_M_S01

    return storm_object_table.assign(**{
        tracking_utils.EAST_VELOCITY_COLUMN: east_velocities_m_s01,
        tracking_utils.NORTH_VELOCITY_COLUMN: north_velocities_m_s01
    })


def _check_secondary_id_change_type(change_type_string):
    """Error-checks type of secondary-ID change.

    :param change_type_string: Type of change.
    :raises: ValueError: if
        `change_type_string not in SECONDARY_ID_CHANGE_TYPE_STRINGS`.
    """

    error_checking.assert_is_string(change_type_string)

    if change_type_string not in SECONDARY_ID_CHANGE_TYPE_STRINGS:
        error_string = (
            '\n{0:s}\nValid secondary-ID-change types (listed above) do not '
            'include "{1:s}".'
        ).format(str(SECONDARY_ID_CHANGE_TYPE_STRINGS), change_type_string)

        raise ValueError(error_string)


def partial_to_full_ids(primary_id_strings, secondary_id_strings):
    """For each storm object, converts primary and secondary IDs to full ID.

    N = number of storm objects

    :param primary_id_strings: length-N list of primary IDs.
    :param secondary_id_strings: length-N list of secondary IDs.
    :return: full_id_strings: length-N list of full IDs.
    """

    error_checking.assert_is_string_list(primary_id_strings)
    error_checking.assert_is_numpy_array(
        numpy.array(primary_id_strings), num_dimensions=1
    )

    num_storm_objects = len(primary_id_strings)
    expected_dimensions = numpy.array([num_storm_objects], dtype=int)

    error_checking.assert_is_string_list(secondary_id_strings)
    error_checking.assert_is_numpy_array(
        numpy.array(secondary_id_strings), exact_dimensions=expected_dimensions
    )

    assert all('_' not in p for p in primary_id_strings)
    assert all('_' not in s for s in secondary_id_strings)

    return [
        '{0:s}_{1:s}'.format(p, s)
        for p, s in zip(primary_id_strings, secondary_id_strings)
    ]


def full_to_partial_ids(full_id_strings):
    """For each storm object, converts full ID to primary and secondary.

    N = number of storm objects

    :param full_id_strings: length-N list of full IDs.
    :return: primary_id_strings: length-N list of primary IDs.
    :return: secondary_id_strings: length-N list of secondary IDs.
    """

    error_checking.assert_is_string_list(full_id_strings)
    error_checking.assert_is_numpy_array(
        numpy.array(full_id_strings), num_dimensions=1
    )

    primary_id_strings = ['_'.join(f.split('_')[:-1]) for f in full_id_strings]
    secondary_id_strings = ['_'.join(f.split('_')[1:]) for f in full_id_strings]

    assert all('_' not in p for p in primary_id_strings)
    assert all('_' not in s for s in secondary_id_strings)

    return primary_id_strings, secondary_id_strings


def link_local_maxima_in_time(
        current_local_max_dict, previous_local_max_dict, max_link_time_seconds,
        max_velocity_diff_m_s01, max_link_distance_m_s01):
    """Links local maxima between current and previous time steps.

    N_c = number of maxima at current time
    N_p = number of maxima at previous time

    :param current_local_max_dict: Dictionary with the following keys.
    current_local_max_dict['valid_time_unix_sec']: Valid time.
    current_local_max_dict['x_coords_metres']: numpy array (length N_c) with
        x-coordinates of local maxima.
    current_local_max_dict['y_coords_metres']: numpy array (length N_c) with
        y-coordinates of local maxima.

    :param previous_local_max_dict: Dictionary created by
        `get_intermediate_velocities`.  If `current_local_max_dict` represents
        the first time, `previous_local_max_dict` may be None.
    :param max_link_time_seconds: Max difference between current and previous
        time steps.
    :param max_velocity_diff_m_s01: Max difference between expected and actual
        current locations.  Expected current locations are determined by
        extrapolating local maxima from previous time.
    :param max_link_distance_m_s01: Max difference between current and previous
        locations.  This criterion will be used only for previous maxima with no
        velocity estimate.
    :return: current_to_previous_matrix: numpy array (N_c x N_p) of Boolean
        flags.  If current_to_previous_matrix[i, j] = True, the [i]th local max
        at the current time is linked to the [j]th local max at the previous
        time.
    """

    error_checking.assert_is_integer(max_link_time_seconds)
    error_checking.assert_is_greater(max_link_time_seconds, 0)
    error_checking.assert_is_greater(max_velocity_diff_m_s01, 0.)
    # error_checking.assert_is_greater(max_link_distance_m_s01, 0.)

    num_current_maxima = len(current_local_max_dict[X_COORDS_KEY])
    if previous_local_max_dict is None:
        num_previous_maxima = 0
    else:
        num_previous_maxima = len(previous_local_max_dict[X_COORDS_KEY])

    current_to_previous_matrix = numpy.full(
        (num_current_maxima, num_previous_maxima), False, dtype=bool)

    if num_current_maxima == 0 or num_previous_maxima == 0:
        return current_to_previous_matrix

    time_diff_seconds = (
        current_local_max_dict[VALID_TIME_KEY] -
        previous_local_max_dict[VALID_TIME_KEY]
    )

    if time_diff_seconds > max_link_time_seconds:
        return current_to_previous_matrix

    velocity_diff_matrix_m_s01, current_to_previous_matrix = (
        _link_local_maxima_by_velocity(
            current_local_max_dict=current_local_max_dict,
            previous_local_max_dict=previous_local_max_dict,
            max_velocity_diff_m_s01=max_velocity_diff_m_s01)
    )

    distance_matrix_m_s01, current_to_previous_matrix = (
        _link_local_maxima_by_distance(
            current_local_max_dict=current_local_max_dict,
            previous_local_max_dict=previous_local_max_dict,
            max_link_distance_m_s01=max_link_distance_m_s01,
            current_to_previous_matrix=current_to_previous_matrix)
    )

    return _prune_connections(
        velocity_diff_matrix_m_s01=velocity_diff_matrix_m_s01,
        distance_matrix_m_s01=distance_matrix_m_s01,
        current_to_previous_matrix=current_to_previous_matrix)


def get_intermediate_velocities(
        current_local_max_dict, previous_local_max_dict,
        e_folding_radius_metres=DEFAULT_VELOCITY_EFOLD_RADIUS_METRES):
    """Returns intermediate velocity estimate for each storm at current time.

    N_c = number of maxima at current time
    N_p = number of maxima at previous time

    :param current_local_max_dict: Dictionary with the following keys.  If
        `previous_local_max_dict is None`, the key "current_to_previous_matrix"
        is not required.
    current_local_max_dict['valid_time_unix_sec']: Valid time.
    current_local_max_dict['x_coords_metres']: numpy array (length N_c) with
        x-coordinates of local maxima.
    current_local_max_dict['y_coords_metres']: numpy array (length N_c) with
        y-coordinates of local maxima.
    current_local_max_dict['current_to_previous_matrix']: numpy array
        (N_c x N_p) of Boolean flags.  If current_to_previous_matrix[i, j]
        = True, the [i]th local max at the current time is linked to the [j]th
        local max at the previous time.

    :param previous_local_max_dict: Same as `current_local_max_dict` but for
        previous time.  If `current_local_max_dict` represents the first time,
        `previous_local_max_dict` may be None.
    :param e_folding_radius_metres: See doc for `_estimate_velocity_by_neigh`.

    :return: current_local_max_dict: Same as input but with the following
        additional columns.
    current_local_max_dict['x_velocities_m_s01']: numpy array (length N_c) of
        x-velocities (metres per second in positive direction).
    current_local_max_dict['y_velocities_m_s01']: Same but for y-direction.
    """

    error_checking.assert_is_greater(e_folding_radius_metres, 0.)

    num_current_maxima = len(current_local_max_dict[X_COORDS_KEY])
    if previous_local_max_dict is None:
        num_previous_maxima = 0
    else:
        num_previous_maxima = len(previous_local_max_dict[X_COORDS_KEY])

    if num_current_maxima == 0 or num_previous_maxima == 0:
        x_velocities_m_s01 = numpy.full(num_current_maxima, numpy.nan)
        y_velocities_m_s01 = numpy.full(num_current_maxima, numpy.nan)

        current_local_max_dict.update({
            X_VELOCITIES_KEY: x_velocities_m_s01,
            Y_VELOCITIES_KEY: y_velocities_m_s01
        })

        return current_local_max_dict

    first_previous_indices = numpy.full(num_current_maxima, -1, dtype=int)
    second_previous_indices = numpy.full(num_current_maxima, -1, dtype=int)

    for i in range(num_current_maxima):
        these_previous_indices = numpy.where(
            current_local_max_dict[CURRENT_TO_PREV_MATRIX_KEY][i, ...]
        )[0]

        if len(these_previous_indices) > 0:
            first_previous_indices[i] = these_previous_indices[0]
        if len(these_previous_indices) > 1:
            second_previous_indices[i] = these_previous_indices[1]

    time_diff_seconds = (
        current_local_max_dict[VALID_TIME_KEY] -
        previous_local_max_dict[VALID_TIME_KEY]
    )

    first_prev_x_coords_metres = numpy.array([
        numpy.nan if k == -1 else previous_local_max_dict[X_COORDS_KEY][k]
        for k in first_previous_indices
    ])

    first_prev_y_coords_metres = numpy.array([
        numpy.nan if k == -1 else previous_local_max_dict[Y_COORDS_KEY][k]
        for k in first_previous_indices
    ])

    first_x_velocities_m_s01 = (
        current_local_max_dict[X_COORDS_KEY] - first_prev_x_coords_metres
    ) / time_diff_seconds

    first_y_velocities_m_s01 = (
        current_local_max_dict[Y_COORDS_KEY] - first_prev_y_coords_metres
    ) / time_diff_seconds

    second_prev_x_coords_metres = numpy.array([
        numpy.nan if k == -1 else previous_local_max_dict[X_COORDS_KEY][k]
        for k in second_previous_indices
    ])

    second_prev_y_coords_metres = numpy.array([
        numpy.nan if k == -1 else previous_local_max_dict[Y_COORDS_KEY][k]
        for k in second_previous_indices
    ])

    second_x_velocities_m_s01 = (
        current_local_max_dict[X_COORDS_KEY] - second_prev_x_coords_metres
    ) / time_diff_seconds

    second_y_velocities_m_s01 = (
        current_local_max_dict[Y_COORDS_KEY] - second_prev_y_coords_metres
    ) / time_diff_seconds

    x_velocities_m_s01 = numpy.nanmean(
        numpy.array([first_x_velocities_m_s01, second_x_velocities_m_s01]),
        axis=0
    )

    y_velocities_m_s01 = numpy.nanmean(
        numpy.array([first_y_velocities_m_s01, second_y_velocities_m_s01]),
        axis=0
    )

    x_velocities_m_s01, y_velocities_m_s01 = _estimate_velocity_by_neigh(
        x_coords_metres=current_local_max_dict[X_COORDS_KEY],
        y_coords_metres=current_local_max_dict[Y_COORDS_KEY],
        x_velocities_m_s01=x_velocities_m_s01,
        y_velocities_m_s01=y_velocities_m_s01,
        e_folding_radius_metres=e_folding_radius_metres)

    current_local_max_dict.update({
        X_VELOCITIES_KEY: x_velocities_m_s01,
        Y_VELOCITIES_KEY: y_velocities_m_s01
    })

    return current_local_max_dict


def local_maxima_to_storm_tracks(local_max_dict_by_time, first_numeric_id):
    """Converts time series of local maxima to set of storm tracks.

    T = number of time steps
    P = number of local maxima at a given time

    :param local_max_dict_by_time: length-T list of dictionaries, each with the
        following keys.
    "valid_time_unix_sec": Valid time.
    "latitudes_deg": length-P numpy array with latitudes (deg N) of local
        maxima.
    "longitudes_deg": length-P numpy array with longitudes (deg E) of local
        maxima.
    "x_coords_metres": length-P numpy array with x-coordinates of local maxima.
    "y_coords_metres": length-P numpy array with y-coordinates of local maxima.
    "current_to_previous_matrix": See doc for `link_local_maxima_in_time`.

    :param first_numeric_id: First numeric storm ID.  This will be used for
        assigning both primary and secondary IDs.

    :return: storm_object_table: pandas DataFrame with the following columns
        (most of which are explained in `storm_tracking_io.write_file`).  Each
        row is one storm object.
    storm_object_table.full_id_string
    storm_object_table.primary_id_string
    storm_object_table.secondary_id_string
    storm_object_table.first_prev_secondary_id_string
    storm_object_table.second_prev_secondary_id_string
    storm_object_table.first_next_secondary_id_string
    storm_object_table.second_next_secondary_id_string
    storm_object_table.valid_time_unix_sec
    storm_object_table.spc_date_string
    storm_object_table.centroid_latitude_deg
    storm_object_table.centroid_longitude_deg
    storm_object_table.centroid_x_metres: x-coordinate of centroid.
    storm_object_table.centroid_y_metres: y-coordinate of centroid.

    If `local_max_dict_by_time` includes polygons, `storm_object_table` will
    have the additional columns listed below.

    storm_object_table.grid_point_latitudes_deg
    storm_object_table.grid_point_longitudes_deg
    storm_object_table.grid_point_rows
    storm_object_table.grid_point_columns
    storm_object_table.polygon_object_latlng_deg
    storm_object_table.polygon_object_rowcol
    """

    prev_primary_id_numeric = first_numeric_id - 1
    prev_secondary_id_numeric = first_numeric_id - 1
    num_times = len(local_max_dict_by_time)

    for i in range(num_times):
        this_num_storm_objects = len(local_max_dict_by_time[i][LATITUDES_KEY])
        this_empty_2d_list = [
            ['' for _ in range(0)] for _ in range(this_num_storm_objects)
        ]

        local_max_dict_by_time[i].update({
            PRIMARY_IDS_KEY: [''] * this_num_storm_objects,
            SECONDARY_IDS_KEY: [''] * this_num_storm_objects,
            PREV_SECONDARY_IDS_KEY: this_empty_2d_list,
            NEXT_SECONDARY_IDS_KEY: copy.deepcopy(this_empty_2d_list)
        })

        # If first time step, just create storm IDs and don't worry about
        # linking.
        if i == 0:
            for j in range(this_num_storm_objects):
                (local_max_dict_by_time[i][PRIMARY_IDS_KEY][j],
                 prev_primary_id_numeric
                ) = _create_partial_storm_id(prev_primary_id_numeric)

                (local_max_dict_by_time[i][SECONDARY_IDS_KEY][j],
                 prev_secondary_id_numeric
                ) = _create_partial_storm_id(prev_secondary_id_numeric)

            continue

        # Handle mergers between [i]th and [i - 1]th time steps.  This
        # occurs when two storms at [i - 1]th time merge into one storm at
        # [i]th time.
        this_current_to_prev_matrix = copy.deepcopy(
            local_max_dict_by_time[i][CURRENT_TO_PREV_MATRIX_KEY]
        )

        this_dict = _local_maxima_to_tracks_mergers(
            current_local_max_dict=local_max_dict_by_time[i],
            previous_local_max_dict=local_max_dict_by_time[i - 1],
            current_to_previous_matrix=this_current_to_prev_matrix,
            prev_primary_id_numeric=prev_primary_id_numeric,
            prev_secondary_id_numeric=prev_secondary_id_numeric)

        local_max_dict_by_time[i] = this_dict[CURRENT_LOCAL_MAXIMA_KEY]
        local_max_dict_by_time[i - 1] = this_dict[PREVIOUS_LOCAL_MAXIMA_KEY]
        this_current_to_prev_matrix = this_dict[CURRENT_TO_PREV_MATRIX_KEY]
        prev_primary_id_numeric = this_dict[PREVIOUS_PRIMARY_ID_KEY]
        prev_secondary_id_numeric = this_dict[PREVIOUS_SECONDARY_ID_KEY]

        this_old_to_new_dict = this_dict[OLD_TO_NEW_PRIMARY_IDS_KEY]

        for j in range(i + 1):
            local_max_dict_by_time[j][PRIMARY_IDS_KEY] = [
                p if p not in this_old_to_new_dict else
                this_old_to_new_dict[p]
                for p in local_max_dict_by_time[j][PRIMARY_IDS_KEY]
            ]

        # Handle splits between [i]th and [i - 1]th time steps.  This
        # occurs when one storm at [i - 1]th time splits into two storms at
        # [i]th time.
        this_dict = _local_maxima_to_tracks_splits(
            current_local_max_dict=local_max_dict_by_time[i],
            previous_local_max_dict=local_max_dict_by_time[i - 1],
            current_to_previous_matrix=this_current_to_prev_matrix,
            prev_secondary_id_numeric=prev_secondary_id_numeric)

        local_max_dict_by_time[i] = this_dict[CURRENT_LOCAL_MAXIMA_KEY]
        local_max_dict_by_time[i - 1] = this_dict[PREVIOUS_LOCAL_MAXIMA_KEY]
        this_current_to_prev_matrix = this_dict[CURRENT_TO_PREV_MATRIX_KEY]
        prev_secondary_id_numeric = this_dict[PREVIOUS_SECONDARY_ID_KEY]

        # Handle simple connections between [i]th and [i - 1]th time steps.
        # This occurs when one storm at [i - 1]th time has either zero or
        # one successors at [i]th time.  In other words, there is either a
        # one-to-one connection or no connection.
        this_dict = _local_maxima_to_tracks_simple(
            current_local_max_dict=local_max_dict_by_time[i],
            previous_local_max_dict=local_max_dict_by_time[i - 1],
            current_to_previous_matrix=this_current_to_prev_matrix,
            prev_primary_id_numeric=prev_primary_id_numeric,
            prev_secondary_id_numeric=prev_secondary_id_numeric)

        local_max_dict_by_time[i] = this_dict[CURRENT_LOCAL_MAXIMA_KEY]
        local_max_dict_by_time[i - 1] = this_dict[PREVIOUS_LOCAL_MAXIMA_KEY]
        prev_primary_id_numeric = this_dict[PREVIOUS_PRIMARY_ID_KEY]
        prev_secondary_id_numeric = this_dict[PREVIOUS_SECONDARY_ID_KEY]

    all_primary_id_strings = []
    all_secondary_id_strings = []
    all_first_prev_sec_id_strings = []
    all_second_prev_sec_id_strings = []
    all_first_next_sec_id_strings = []
    all_second_next_sec_id_strings = []

    all_times_unix_sec = numpy.array([], dtype=int)
    all_spc_date_strings = []
    all_centroid_latitudes_deg = numpy.array([])
    all_centroid_longitudes_deg = numpy.array([])
    all_centroid_x_metres = numpy.array([])
    all_centroid_y_metres = numpy.array([])

    include_polygons = POLYGON_OBJECTS_LATLNG_KEY in local_max_dict_by_time[0]

    all_polygon_rows_arraylist = []
    all_polygon_columns_arraylist = []
    all_polygon_lat_arraylist_deg = []
    all_polygon_lng_arraylist_deg = []
    all_polygon_objects_latlng = numpy.array([], dtype=object)
    all_polygon_objects_rowcol = numpy.array([], dtype=object)

    for i in range(num_times):
        this_num_storm_objects = len(
            local_max_dict_by_time[i][LATITUDES_KEY]
        )

        all_primary_id_strings += local_max_dict_by_time[i][PRIMARY_IDS_KEY]
        all_secondary_id_strings += local_max_dict_by_time[i][SECONDARY_IDS_KEY]

        all_first_prev_sec_id_strings += [
            x[0] if len(x) > 0 else ''
            for x in local_max_dict_by_time[i][PREV_SECONDARY_IDS_KEY]
        ]

        all_second_prev_sec_id_strings += [
            x[1] if len(x) > 1 else ''
            for x in local_max_dict_by_time[i][PREV_SECONDARY_IDS_KEY]
        ]

        all_first_next_sec_id_strings += [
            x[0] if len(x) > 0 else ''
            for x in local_max_dict_by_time[i][NEXT_SECONDARY_IDS_KEY]
        ]

        all_second_next_sec_id_strings += [
            x[1] if len(x) > 1 else ''
            for x in local_max_dict_by_time[i][NEXT_SECONDARY_IDS_KEY]
        ]

        these_times_unix_sec = numpy.full(
            this_num_storm_objects,
            local_max_dict_by_time[i][VALID_TIME_KEY], dtype=int
        )

        if len(these_times_unix_sec) > 0:
            these_spc_date_strings = this_num_storm_objects * [
                time_conversion.time_to_spc_date_string(these_times_unix_sec[0])
            ]
        else:
            these_spc_date_strings = []

        all_times_unix_sec = numpy.concatenate((
            all_times_unix_sec, these_times_unix_sec
        ))
        all_spc_date_strings += these_spc_date_strings

        all_centroid_latitudes_deg = numpy.concatenate((
            all_centroid_latitudes_deg,
            local_max_dict_by_time[i][LATITUDES_KEY]
        ))
        all_centroid_longitudes_deg = numpy.concatenate((
            all_centroid_longitudes_deg,
            local_max_dict_by_time[i][LONGITUDES_KEY]
        ))
        all_centroid_x_metres = numpy.concatenate((
            all_centroid_x_metres, local_max_dict_by_time[i][X_COORDS_KEY]
        ))
        all_centroid_y_metres = numpy.concatenate((
            all_centroid_y_metres, local_max_dict_by_time[i][Y_COORDS_KEY]
        ))

        if not include_polygons:
            continue

        all_polygon_rows_arraylist += local_max_dict_by_time[i][
            GRID_POINT_ROWS_KEY]
        all_polygon_columns_arraylist += local_max_dict_by_time[i][
            GRID_POINT_COLUMNS_KEY]
        all_polygon_lat_arraylist_deg += local_max_dict_by_time[i][
            GRID_POINT_LATITUDES_KEY]
        all_polygon_lng_arraylist_deg += local_max_dict_by_time[i][
            GRID_POINT_LONGITUDES_KEY]

        all_polygon_objects_latlng = numpy.concatenate((
            all_polygon_objects_latlng,
            local_max_dict_by_time[i][POLYGON_OBJECTS_LATLNG_KEY]
        ))
        all_polygon_objects_rowcol = numpy.concatenate((
            all_polygon_objects_rowcol,
            local_max_dict_by_time[i][POLYGON_OBJECTS_ROWCOL_KEY]
        ))

    storm_object_dict = {
        tracking_utils.PRIMARY_ID_COLUMN: all_primary_id_strings,
        tracking_utils.SECONDARY_ID_COLUMN: all_secondary_id_strings,
        tracking_utils.FIRST_PREV_SECONDARY_ID_COLUMN:
            all_first_prev_sec_id_strings,
        tracking_utils.SECOND_PREV_SECONDARY_ID_COLUMN:
            all_second_prev_sec_id_strings,
        tracking_utils.FIRST_NEXT_SECONDARY_ID_COLUMN:
            all_first_next_sec_id_strings,
        tracking_utils.SECOND_NEXT_SECONDARY_ID_COLUMN:
            all_second_next_sec_id_strings,
        tracking_utils.VALID_TIME_COLUMN: all_times_unix_sec,
        tracking_utils.SPC_DATE_COLUMN: all_spc_date_strings,
        tracking_utils.CENTROID_LATITUDE_COLUMN: all_centroid_latitudes_deg,
        tracking_utils.CENTROID_LONGITUDE_COLUMN: all_centroid_longitudes_deg,
        CENTROID_X_COLUMN: all_centroid_x_metres,
        CENTROID_Y_COLUMN: all_centroid_y_metres
    }

    if include_polygons:
        storm_object_dict.update({
            tracking_utils.ROWS_IN_STORM_COLUMN: all_polygon_rows_arraylist,
            tracking_utils.COLUMNS_IN_STORM_COLUMN:
                all_polygon_columns_arraylist,
            tracking_utils.LATITUDES_IN_STORM_COLUMN:
                all_polygon_lat_arraylist_deg,
            tracking_utils.LONGITUDES_IN_STORM_COLUMN:
                all_polygon_lng_arraylist_deg,
            tracking_utils.LATLNG_POLYGON_COLUMN: all_polygon_objects_latlng,
            tracking_utils.ROWCOL_POLYGON_COLUMN: all_polygon_objects_rowcol
        })

    storm_object_table = pandas.DataFrame.from_dict(storm_object_dict)

    # Create full IDs (primary + secondary).
    full_id_strings = partial_to_full_ids(
        primary_id_strings=storm_object_table[
            tracking_utils.PRIMARY_ID_COLUMN].values.tolist(),
        secondary_id_strings=storm_object_table[
            tracking_utils.SECONDARY_ID_COLUMN].values.tolist()
    )

    argument_dict = {tracking_utils.FULL_ID_COLUMN: full_id_strings}
    return storm_object_table.assign(**argument_dict)


def remove_short_lived_storms(storm_object_table, min_duration_seconds):
    """Removes short-lived storms.

    :param storm_object_table: pandas DataFrame created by
        `local_maxima_to_storm_tracks`.
    :param min_duration_seconds: Minimum duration.
    :return: storm_object_table: Same as input but maybe with fewer rows.
    """

    error_checking.assert_is_integer(min_duration_seconds)
    error_checking.assert_is_geq(min_duration_seconds, 0)

    id_string_by_track, object_to_track_indices = numpy.unique(
        storm_object_table[tracking_utils.PRIMARY_ID_COLUMN].values,
        return_inverse=True)

    num_storm_tracks = len(id_string_by_track)
    object_indices_to_remove = numpy.array([], dtype=int)

    for i in range(num_storm_tracks):
        these_object_indices = numpy.where(object_to_track_indices == i)[0]
        these_times_unix_sec = storm_object_table[
            tracking_utils.VALID_TIME_COLUMN
        ].values[these_object_indices]

        this_duration_seconds = (
            numpy.max(these_times_unix_sec) - numpy.min(these_times_unix_sec)
        )
        if this_duration_seconds >= min_duration_seconds:
            continue

        object_indices_to_remove = numpy.concatenate((
            object_indices_to_remove, these_object_indices))

    return storm_object_table.drop(
        storm_object_table.index[object_indices_to_remove], axis=0,
        inplace=False)


def check_tracking_periods(tracking_start_times_unix_sec,
                           tracking_end_times_unix_sec):
    """Ensures that list of tracking periods is valid.

    T = number of tracking periods

    :param tracking_start_times_unix_sec: length-T numpy array with start times
        of tracking periods.
    :param tracking_end_times_unix_sec: length-T numpy array with end times of
        tracking periods.
    :return: tracking_start_times_unix_sec: Same as input but sorted.
    :return: tracking_end_times_unix_sec: Same as input but sorted.
    """

    error_checking.assert_is_integer_numpy_array(tracking_start_times_unix_sec)
    error_checking.assert_is_numpy_array(
        tracking_start_times_unix_sec, num_dimensions=1)

    num_tracking_periods = len(tracking_start_times_unix_sec)
    these_expected_dim = numpy.array([num_tracking_periods], dtype=int)

    error_checking.assert_is_integer_numpy_array(tracking_end_times_unix_sec)
    error_checking.assert_is_numpy_array(
        tracking_end_times_unix_sec, exact_dimensions=these_expected_dim)

    error_checking.assert_is_geq_numpy_array(
        tracking_end_times_unix_sec - tracking_start_times_unix_sec, 0)

    start_time_sort_indices = numpy.argsort(tracking_start_times_unix_sec)
    end_time_sort_indices = numpy.argsort(tracking_end_times_unix_sec)

    if not numpy.array_equal(start_time_sort_indices, end_time_sort_indices):
        tracking_start_time_strings = [
            time_conversion.unix_sec_to_string(t, LOG_MESSAGE_TIME_FORMAT)
            for t in tracking_start_times_unix_sec
        ]

        tracking_end_time_strings = [
            time_conversion.unix_sec_to_string(t, LOG_MESSAGE_TIME_FORMAT)
            for t in tracking_end_times_unix_sec
        ]

        print('\n')
        for k in range(len(tracking_start_time_strings)):
            print('{0:d}th tracking period = {1:s} to {2:s}'.format(
                k + 1, tracking_start_time_strings[k],
                tracking_end_time_strings[k]
            ))

        print('\n')
        raise ValueError(
            'As shown above, start/end times of tracking periods are not sorted'
            ' in the same order.'
        )

    tracking_start_times_unix_sec = tracking_start_times_unix_sec[
        start_time_sort_indices
    ]
    tracking_end_times_unix_sec = tracking_end_times_unix_sec[
        start_time_sort_indices
    ]

    error_checking.assert_is_geq_numpy_array(
        tracking_end_times_unix_sec - tracking_start_times_unix_sec, 0
    )
    error_checking.assert_is_greater_numpy_array(
        tracking_start_times_unix_sec[1:] - tracking_end_times_unix_sec[:-1], 0
    )

    return tracking_start_times_unix_sec, tracking_end_times_unix_sec


def get_storm_ages(storm_object_table, tracking_start_times_unix_sec,
                   tracking_end_times_unix_sec, max_link_time_seconds):
    """Computes age of each storm cell at each time step.

    T = number of tracking periods

    :param storm_object_table: pandas DataFrame with at least the following
        columns.  Each row is one storm object (one cell at one time step).
    storm_object_table.primary_id_string: Primary storm ID.
    storm_object_table.valid_time_unix_sec: Valid time.

    :param tracking_start_times_unix_sec: length-T numpy array with start times
        of tracking periods.
    :param tracking_end_times_unix_sec: length-T numpy array with end times of
        tracking periods.
    :param max_link_time_seconds: See doc for `link_local_maxima_in_time`.

    :return: storm_object_table: Same as input but with the following extra
        columns.
    storm_object_table.age_seconds: Storm age.
    storm_object_table.tracking_start_time_unix_sec: Start of tracking period
        for the given storm object.
    storm_object_table.tracking_end_time_unix_sec: End of tracking period for
        the given storm object.
    storm_object_table.cell_start_time_unix_sec: Start time of storm cell.
    storm_object_table.cell_end_time_unix_sec: End time of storm cell.

    :raises: ValueError: if `tracking_start_times_unix_sec` and
        `tracking_start_times_unix_sec` are not sorted in the same order.
    """

    tracking_start_times_unix_sec, tracking_end_times_unix_sec = (
        check_tracking_periods(
            tracking_start_times_unix_sec=tracking_start_times_unix_sec,
            tracking_end_times_unix_sec=tracking_end_times_unix_sec)
    )

    error_checking.assert_is_integer(max_link_time_seconds)
    error_checking.assert_is_greater(max_link_time_seconds, 0)

    track_id_strings, object_to_track_indices = numpy.unique(
        storm_object_table[tracking_utils.PRIMARY_ID_COLUMN].values,
        return_inverse=True)

    num_storm_objects = len(storm_object_table.index)
    this_array = numpy.full(num_storm_objects, -1, dtype=int)

    storm_object_table = storm_object_table.assign(**{
        tracking_utils.AGE_COLUMN: this_array,
        tracking_utils.CELL_START_TIME_COLUMN: this_array,
        tracking_utils.CELL_END_TIME_COLUMN: this_array,
        tracking_utils.TRACKING_START_TIME_COLUMN: this_array,
        tracking_utils.TRACKING_END_TIME_COLUMN: this_array
    })

    num_storm_tracks = len(track_id_strings)

    for j in range(num_storm_tracks):
        these_object_indices = numpy.where(object_to_track_indices == j)[0]
        these_object_times_unix_sec = storm_object_table[
            tracking_utils.VALID_TIME_COLUMN
        ].values[these_object_indices]

        this_cell_start_time_unix_sec = numpy.min(these_object_times_unix_sec)

        storm_object_table[tracking_utils.CELL_START_TIME_COLUMN].values[
            these_object_indices
        ] = this_cell_start_time_unix_sec

        storm_object_table[tracking_utils.CELL_END_TIME_COLUMN].values[
            these_object_indices
        ] = numpy.max(these_object_times_unix_sec)

        this_index = numpy.where(
            this_cell_start_time_unix_sec >= tracking_start_times_unix_sec
        )[0][-1]

        storm_object_table[tracking_utils.TRACKING_START_TIME_COLUMN].values[
            these_object_indices
        ] = tracking_start_times_unix_sec[this_index]

        storm_object_table[tracking_utils.TRACKING_END_TIME_COLUMN].values[
            these_object_indices
        ] = tracking_end_times_unix_sec[this_index]

        this_tracking_time_elapsed_sec = (
            this_cell_start_time_unix_sec -
            tracking_start_times_unix_sec[this_index]
        )

        if this_tracking_time_elapsed_sec < max_link_time_seconds:
            continue

        storm_object_table[tracking_utils.AGE_COLUMN].values[
            these_object_indices
        ] = these_object_times_unix_sec - this_cell_start_time_unix_sec

    return storm_object_table


def find_predecessors(
        storm_object_table, target_row, num_seconds_back,
        max_num_sec_id_changes=LARGE_INTEGER,
        change_type_string=ANY_CHANGE_STRING, return_all_on_path=False):
    """Finds all predecessors of one storm object.

    :param storm_object_table: pandas DataFrame with at least the following
        columns.  Each row is one storm object.
    storm_object_table.valid_time_unix_sec: Valid time.
    storm_object_table.secondary_id_string: Secondary storm ID.
    storm_object_table.first_prev_secondary_id_string: Secondary ID of first
        immediate predecessor.
    storm_object_table.second_prev_secondary_id_string: Secondary ID of second
        immediate predecessor.

    :param target_row: Will find predecessors for object in [k]th row of
        `storm_object_table`, where k = `target_row`.
    :param num_seconds_back: Max time difference between target object and a
        given predecesssor.
    :param max_num_sec_id_changes: Max number of secondary-ID changes between
        target object and a given predecesssor.
    :param change_type_string: Type of secondary-ID change for
        `max_num_sec_id_changes`.  Must be accepted by
        `_check_secondary_id_change_type`.
    :param return_all_on_path: Boolean flag.  If True, will return all
        predecessors on path to earliest predecessors.  If False, will return
        only earliest predecessors.
    :return: predecessor_rows: 1-D numpy array with rows of predecessors.  These
        are rows in `storm_object_table`.
    """

    error_checking.assert_is_integer(target_row)
    error_checking.assert_is_geq(target_row, 0)
    error_checking.assert_is_less_than(
        target_row, len(storm_object_table.index)
    )

    error_checking.assert_is_integer(num_seconds_back)
    error_checking.assert_is_greater(num_seconds_back, 0)
    error_checking.assert_is_integer(max_num_sec_id_changes)
    error_checking.assert_is_geq(max_num_sec_id_changes, 0)
    error_checking.assert_is_boolean(return_all_on_path)

    _check_secondary_id_change_type(change_type_string)

    last_time_unix_sec = (
        storm_object_table[tracking_utils.VALID_TIME_COLUMN].values[target_row]
    )
    first_time_unix_sec = (
        storm_object_table[tracking_utils.VALID_TIME_COLUMN].values[target_row]
        - num_seconds_back
    )

    recent_rows = numpy.where(numpy.logical_and(
        storm_object_table[tracking_utils.VALID_TIME_COLUMN].values >=
        first_time_unix_sec,
        storm_object_table[tracking_utils.VALID_TIME_COLUMN].values <=
        last_time_unix_sec
    ))[0]

    target_subrow = numpy.where(recent_rows == target_row)[0][0]

    predecessor_subrows = __find_predecessors(
        storm_object_table=storm_object_table.iloc[recent_rows],
        target_row=target_subrow, max_num_sec_id_changes=max_num_sec_id_changes,
        change_type_string=change_type_string,
        return_all_on_path=return_all_on_path)

    return recent_rows[predecessor_subrows]


def find_immediate_predecessors(
        storm_object_table, target_row, max_time_diff_seconds=1200):
    """Finds immediate predecessors of one storm object.

    :param storm_object_table: See doc for `find_predecessors`.
    :param target_row: Same.
    :param max_time_diff_seconds: Max time difference.  Will not look more than
        `max_time_diff_seconds` before target time.
    :return: predecessor_rows: Same.
    """

    error_checking.assert_is_integer(target_row)
    error_checking.assert_is_geq(target_row, 0)
    error_checking.assert_is_less_than(
        target_row, len(storm_object_table.index)
    )

    error_checking.assert_is_integer(max_time_diff_seconds)
    error_checking.assert_is_greater(max_time_diff_seconds, 0)

    last_time_unix_sec = (
        storm_object_table[tracking_utils.VALID_TIME_COLUMN].values[target_row]
    )
    first_time_unix_sec = (
        storm_object_table[tracking_utils.VALID_TIME_COLUMN].values[target_row]
        - max_time_diff_seconds
    )

    recent_rows = numpy.where(numpy.logical_and(
        storm_object_table[tracking_utils.VALID_TIME_COLUMN].values >=
        first_time_unix_sec,
        storm_object_table[tracking_utils.VALID_TIME_COLUMN].values <=
        last_time_unix_sec
    ))[0]

    target_subrow = numpy.where(recent_rows == target_row)[0][0]

    predecessor_subrows = __find_immediate_predecessors(
        storm_object_table=storm_object_table.iloc[recent_rows],
        target_row=target_subrow)

    return recent_rows[predecessor_subrows]


def find_successors(
        storm_object_table, target_row, num_seconds_forward,
        max_num_sec_id_changes=LARGE_INTEGER,
        change_type_string=ANY_CHANGE_STRING, return_all_on_path=False):
    """Finds all successors of one storm object.

    :param storm_object_table: pandas DataFrame with at least the following
        columns.  Each row is one storm object.
    storm_object_table.valid_time_unix_sec: Valid time.
    storm_object_table.secondary_id_string: Secondary storm ID.
    storm_object_table.first_next_secondary_id_string: Secondary ID of first
        immediate successor.
    storm_object_table.second_next_secondary_id_string: Secondary ID of second
        immediate successor.

    :param target_row: Will find successors for object in [k]th row of
        `storm_object_table`, where k = `target_row`.
    :param num_seconds_forward: Max time difference between target object and a
        given successor.
    :param max_num_sec_id_changes: Max number of secondary-ID changes between
        target object and a given successor.
    :param change_type_string: Type of secondary-ID change for
        `max_num_sec_id_changes`.  Must be accepted by
        `_check_secondary_id_change_type`.
    :param return_all_on_path: Boolean flag.  If True, will return all
        successors on path to earliest successors.  If False, will return
        only earliest successors.
    :return: successor_rows: 1-D numpy array with rows of successors.  These
        are rows in `storm_object_table`.
    """

    error_checking.assert_is_integer(target_row)
    error_checking.assert_is_geq(target_row, 0)
    error_checking.assert_is_less_than(
        target_row, len(storm_object_table.index)
    )

    error_checking.assert_is_integer(num_seconds_forward)
    error_checking.assert_is_greater(num_seconds_forward, 0)
    error_checking.assert_is_integer(max_num_sec_id_changes)
    error_checking.assert_is_geq(max_num_sec_id_changes, 0)
    error_checking.assert_is_boolean(return_all_on_path)

    _check_secondary_id_change_type(change_type_string)

    first_time_unix_sec = (
        storm_object_table[tracking_utils.VALID_TIME_COLUMN].values[target_row]
    )
    last_time_unix_sec = (
        storm_object_table[tracking_utils.VALID_TIME_COLUMN].values[target_row]
        + num_seconds_forward
    )

    soon_rows = numpy.where(numpy.logical_and(
        storm_object_table[tracking_utils.VALID_TIME_COLUMN].values >=
        first_time_unix_sec,
        storm_object_table[tracking_utils.VALID_TIME_COLUMN].values <=
        last_time_unix_sec
    ))[0]

    target_subrow = numpy.where(soon_rows == target_row)[0][0]

    successor_subrows = __find_successors(
        storm_object_table=storm_object_table.iloc[soon_rows],
        target_row=target_subrow, max_num_sec_id_changes=max_num_sec_id_changes,
        change_type_string=change_type_string,
        return_all_on_path=return_all_on_path)

    return soon_rows[successor_subrows]


def find_immediate_successors(
        storm_object_table, target_row, max_time_diff_seconds=1200):
    """Finds immediate successors of one storm object.

    :param storm_object_table: pandas DataFrame with at least the following
        columns.  Each row is one storm object.
    storm_object_table.valid_time_unix_sec: Valid time.
    storm_object_table.secondary_storm_id: Secondary ID (string).
    storm_object_table.first_next_secondary_id: Secondary ID of first
        successor ("" if no successors).
    storm_object_table.second_next_secondary_id: Secondary ID of second
        successor ("" if only one successor).

    :param target_row: Will find successor for object in [k]th row of
        `storm_object_table`, where k = `target_row`.
    :param max_time_diff_seconds: Max time difference.  Will not look more than
        `max_time_diff_seconds` after target time.
    :return: successor_rows: 1-D numpy array with rows of successors.  These are
        rows in `storm_object_table`.
    """

    error_checking.assert_is_integer(target_row)
    error_checking.assert_is_geq(target_row, 0)
    error_checking.assert_is_less_than(
        target_row, len(storm_object_table.index)
    )

    error_checking.assert_is_integer(max_time_diff_seconds)
    error_checking.assert_is_greater(max_time_diff_seconds, 0)

    first_time_unix_sec = (
        storm_object_table[tracking_utils.VALID_TIME_COLUMN].values[target_row]
    )
    last_time_unix_sec = (
        storm_object_table[tracking_utils.VALID_TIME_COLUMN].values[target_row]
        + max_time_diff_seconds
    )

    soon_rows = numpy.where(numpy.logical_and(
        storm_object_table[tracking_utils.VALID_TIME_COLUMN].values >=
        first_time_unix_sec,
        storm_object_table[tracking_utils.VALID_TIME_COLUMN].values <=
        last_time_unix_sec
    ))[0]

    target_subrow = numpy.where(soon_rows == target_row)[0][0]

    successor_subrows = __find_immediate_successors(
        storm_object_table=storm_object_table.iloc[soon_rows],
        target_row=target_subrow)

    return soon_rows[successor_subrows]


def get_storm_velocities(
        storm_object_table,
        min_time_difference_sec=DEFAULT_MIN_VELOCITY_TIME_SEC,
        max_time_difference_sec=DEFAULT_MAX_VELOCITY_TIME_SEC,
        test_mode=False):
    """Estimates instantaneous velocity for each storm object.

    :param storm_object_table: pandas DataFrame with at least the following
        columns.  Each row is one storm object.
    storm_object_table.centroid_lat_deg: Latitude (deg N) of storm centroid.
    storm_object_table.centroid_lng_deg: Longitude (deg E) of storm centroid.
    storm_object_table.centroid_x_metres: x-coordinate of storm centroid.
    storm_object_table.centroid_y_metres: y-coordinate of storm centroid.
    storm_object_table.valid_time_unix_sec: Valid time.
    storm_object_table.secondary_storm_id: Secondary ID (string).
    storm_object_table.first_prev_secondary_id: Secondary ID of first
        predecessor ("" if no predecessors).
    storm_object_table.second_next_secondary_id: Secondary ID of second
        predecessor ("" if only one predecessor).

    :param min_time_difference_sec: Minimum time difference for backwards
        differencing.
    :param max_time_difference_sec: Max time difference for backwards
        differencing.
    :param test_mode: Never mind.  Just leave this empty.
    :return: storm_object_table: Same as input but with the following extra
        columns.
    storm_object_table.east_velocity_m_s01: Eastward velocity (metres per
        second).
    storm_object_table.north_velocity_m_s01: Northward velocity (metres per
        second).
    """

    error_checking.assert_is_integer(min_time_difference_sec)
    error_checking.assert_is_integer(max_time_difference_sec)
    error_checking.assert_is_geq(
        max_time_difference_sec, min_time_difference_sec)

    error_checking.assert_is_boolean(test_mode)

    num_storm_objects = len(storm_object_table.index)
    east_velocities_m_s01 = numpy.full(num_storm_objects, numpy.nan)
    north_velocities_m_s01 = numpy.full(num_storm_objects, numpy.nan)

    for i in range(num_storm_objects):
        if numpy.mod(i, 100) == 0:
            print('Found velocity for {0:d} of {1:d} storm objects...'.format(
                i, num_storm_objects))

        these_predecessor_rows = find_predecessors(
            storm_object_table=storm_object_table, target_row=i,
            num_seconds_back=max_time_difference_sec)

        if len(these_predecessor_rows) == 0:
            continue

        these_time_diffs_seconds = (
            storm_object_table[tracking_utils.VALID_TIME_COLUMN].values[i] -
            storm_object_table[tracking_utils.VALID_TIME_COLUMN].values[
                these_predecessor_rows]
        )

        these_subrows = numpy.where(
            these_time_diffs_seconds >= min_time_difference_sec
        )[0]

        if len(these_subrows) == 0:
            continue

        these_time_diffs_seconds = these_time_diffs_seconds[these_subrows]
        these_predecessor_rows = these_predecessor_rows[these_subrows]
        this_num_predecessors = len(these_predecessor_rows)

        this_end_latitude_deg = storm_object_table[
            tracking_utils.CENTROID_LATITUDE_COLUMN].values[i]
        this_end_longitude_deg = storm_object_table[
            tracking_utils.CENTROID_LONGITUDE_COLUMN].values[i]

        these_east_displacements_metres = numpy.full(
            this_num_predecessors, numpy.nan)
        these_north_displacements_metres = numpy.full(
            this_num_predecessors, numpy.nan)

        for j in range(this_num_predecessors):
            this_start_latitude_deg = storm_object_table[
                tracking_utils.CENTROID_LATITUDE_COLUMN
            ].values[these_predecessor_rows[j]]

            this_start_longitude_deg = storm_object_table[
                tracking_utils.CENTROID_LONGITUDE_COLUMN
            ].values[these_predecessor_rows[j]]

            if test_mode:
                these_east_displacements_metres[j] = (
                    this_end_longitude_deg - this_start_longitude_deg
                )
                these_north_displacements_metres[j] = (
                    this_end_latitude_deg - this_start_latitude_deg
                )
            else:
                these_east_displacements_metres[j] = vincenty(
                    (this_start_latitude_deg, this_start_longitude_deg),
                    (this_start_latitude_deg, this_end_longitude_deg)
                ).meters

                these_north_displacements_metres[j] = vincenty(
                    (this_start_latitude_deg, this_start_longitude_deg),
                    (this_end_latitude_deg, this_start_longitude_deg)
                ).meters

        east_velocities_m_s01[i] = numpy.mean(
            these_east_displacements_metres / these_time_diffs_seconds
        )
        north_velocities_m_s01[i] = numpy.mean(
            these_north_displacements_metres / these_time_diffs_seconds
        )

    storm_object_table = storm_object_table.assign(**{
        tracking_utils.EAST_VELOCITY_COLUMN: east_velocities_m_s01,
        tracking_utils.NORTH_VELOCITY_COLUMN: north_velocities_m_s01
    })

    return _get_storm_velocities_missing(storm_object_table=storm_object_table)


def finish_segmotion_or_probsevere_ids(storm_object_table):
    """Finishes storm IDs created by segmotion or probSevere.

    :param storm_object_table: pandas DataFrame with the following columns,
        where each row is one storm object.  This table should ideally be
        created by `join_stats_and_polygons`.
    storm_object_table.primary_id_string: Primary storm ID.

    :return: storm_object_table: Same as input but with the following extra
        columns.
    storm_object_table.full_id_string: See doc for
        `storm_tracking_io.write_file`.
    storm_object_table.secondary_id_string: Same.
    storm_object_table.first_prev_secondary_id_string: Same.
    storm_object_table.second_prev_secondary_id_string: Same.
    storm_object_table.first_next_secondary_id_string: Same.
    storm_object_table.second_next_secondary_id_string: Same.
    """

    primary_id_strings = storm_object_table[
        tracking_utils.PRIMARY_ID_COLUMN].values

    unique_primary_id_strings, orig_to_unique_indices = numpy.unique(
        numpy.array(primary_id_strings), return_inverse=True
    )

    secondary_id_strings = ['{0:d}'.format(i) for i in orig_to_unique_indices]
    storm_object_table = storm_object_table.assign(**{
        tracking_utils.SECONDARY_ID_COLUMN: secondary_id_strings
    })

    full_id_strings = partial_to_full_ids(
        primary_id_strings=storm_object_table[
            tracking_utils.PRIMARY_ID_COLUMN].values.tolist(),
        secondary_id_strings=storm_object_table[
            tracking_utils.SECONDARY_ID_COLUMN].values.tolist()
    )

    num_storm_objects = len(storm_object_table.index)

    first_prev_secondary_id_strings = numpy.full(
        num_storm_objects, '', dtype=object)
    second_prev_secondary_id_strings = numpy.full(
        num_storm_objects, '', dtype=object)
    first_next_secondary_id_strings = numpy.full(
        num_storm_objects, '', dtype=object)
    second_next_secondary_id_strings = numpy.full(
        num_storm_objects, '', dtype=object)

    num_storm_cells = len(unique_primary_id_strings)

    for j in range(num_storm_cells):
        these_object_indices = numpy.where(orig_to_unique_indices == j)[0]

        first_prev_secondary_id_strings[these_object_indices[1:]] = (
            secondary_id_strings[these_object_indices[0]]
        )
        first_next_secondary_id_strings[these_object_indices[:-1]] = (
            secondary_id_strings[these_object_indices[0]]
        )

    return storm_object_table.assign(**{
        tracking_utils.PRIMARY_ID_COLUMN: primary_id_strings,
        tracking_utils.FULL_ID_COLUMN: full_id_strings,
        tracking_utils.FIRST_PREV_SECONDARY_ID_COLUMN:
            first_prev_secondary_id_strings,
        tracking_utils.SECOND_PREV_SECONDARY_ID_COLUMN:
            second_prev_secondary_id_strings,
        tracking_utils.FIRST_NEXT_SECONDARY_ID_COLUMN:
            first_next_secondary_id_strings,
        tracking_utils.SECOND_NEXT_SECONDARY_ID_COLUMN:
            second_next_secondary_id_strings
    })
