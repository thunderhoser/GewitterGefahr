"""Handles the temporal part of storm-tracking ('connecting the dots')."""

import copy
from collections import OrderedDict
import numpy
import pandas
from geopy.distance import vincenty
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import storm_tracking_utils as tracking_utils
from gewittergefahr.gg_utils import error_checking

MAX_STORMS_IN_SPLIT = 2
MAX_STORMS_IN_MERGER = 2
DEFAULT_VELOCITY_EFOLD_RADIUS_METRES = 100000.
DEFAULT_VELOCITY_WINDOW_SECONDS = 915

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
PREVIOUS_SPC_DATE_KEY = 'prev_spc_date_string'
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


def _estimate_velocity_by_neigh(
        x_coords_metres, y_coords_metres, x_velocities_m_s01,
        y_velocities_m_s01, e_folding_radius_metres):
    """Estimates missing velocities based on non-missing velocities in neigh.

    Specifically, this method replaces each missing velocity with an
    exponentially weighted average of neighbouring non-missing velocities.

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

    neigh_radius_metres = 3 * e_folding_radius_metres
    orig_x_velocities_m_s01 = x_velocities_m_s01 + 0.
    orig_y_velocities_m_s01 = y_velocities_m_s01 + 0.

    nan_flags = numpy.logical_and(
        numpy.isnan(orig_x_velocities_m_s01),
        numpy.isnan(orig_y_velocities_m_s01)
    )
    nan_indices = numpy.where(nan_flags)[0]

    for this_index in nan_indices:
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
        these_distances_m_s01[
            numpy.invert(numpy.isnan(previous_local_max_dict[X_VELOCITIES_KEY]))
        ] = numpy.inf

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

    num_previous_maxima = current_to_previous_matrix.shape[1]

    for j in range(num_previous_maxima):
        this_worst_current_index = -1

        while this_worst_current_index is not None:
            these_current_indices = numpy.where(
                current_to_previous_matrix[:, j]
            )[0]

            this_worst_current_index = None

            if len(these_current_indices) > 1:
                # print 'FOUND {0:d}-WAY SPLIT'.format(len(these_current_indices))

                this_num_previous_by_current = numpy.array([
                    numpy.sum(current_to_previous_matrix[i, :])
                    for i in these_current_indices
                ], dtype=int)

                if numpy.max(this_num_previous_by_current) > 1:
                    this_worst_current_index = these_current_indices[
                        numpy.argmax(this_num_previous_by_current)
                    ]

            if this_worst_current_index is None:
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


def _create_primary_storm_id(storm_start_time_unix_sec, previous_numeric_id,
                             previous_spc_date_string):
    """Creates primary storm ID.

    :param storm_start_time_unix_sec: Start time of storm for which ID is being
        created.
    :param previous_numeric_id: Numeric ID (integer) of previous storm.
    :param previous_spc_date_string: SPC date (format "yyyymmdd") of previous
        storm.
    :return: string_id: String ID for new storm.
    :return: numeric_id: Numeric ID (integer) for new storm.
    :return: spc_date_string: SPC date (format "yyyymmdd") for new storm.
    """

    spc_date_string = time_conversion.time_to_spc_date_string(
        storm_start_time_unix_sec)

    if spc_date_string == previous_spc_date_string:
        numeric_id = previous_numeric_id + 1
    else:
        numeric_id = 0

    string_id = '{0:06d}_{1:s}'.format(numeric_id, spc_date_string)

    return string_id, numeric_id, spc_date_string


def _create_secondary_storm_id(previous_numeric_id):
    """Creates secondary storm ID.

    :param previous_numeric_id: Numeric ID (integer) of previous storm.
    :return: string_id: String ID for new storm.
    :return: numeric_id: Numeric ID (integer) for new storm.
    """

    numeric_id = previous_numeric_id + 1
    return '{0:06d}'.format(numeric_id), numeric_id


def _local_maxima_to_tracks_mergers(
        current_local_max_dict, previous_local_max_dict,
        current_to_previous_matrix, prev_primary_id_numeric,
        prev_spc_date_string, prev_secondary_id_numeric):
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
    :param prev_spc_date_string: Previous SPC date (format "yyyymmdd") used in a
        primary storm ID.
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
    intermediate_track_dict['prev_spc_date_string']: Same as input but possibly
        incremented.
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

        (current_local_max_dict[SECONDARY_IDS_KEY][i], prev_secondary_id_numeric
        ) = _create_secondary_storm_id(prev_secondary_id_numeric)

        (this_primary_id_string, prev_primary_id_numeric,
         prev_spc_date_string
        ) = _create_primary_storm_id(
            storm_start_time_unix_sec=current_local_max_dict[VALID_TIME_KEY],
            previous_numeric_id=prev_primary_id_numeric,
            previous_spc_date_string=prev_spc_date_string)

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
        PREVIOUS_SPC_DATE_KEY: prev_spc_date_string,
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
        print 'FOUND SPLIT'

        these_current_indices = numpy.where(current_to_previous_matrix[:, j])[0]
        current_to_previous_matrix[:, j] = False

        this_primary_id_string = previous_local_max_dict[PRIMARY_IDS_KEY][j]

        for i in these_current_indices:
            current_local_max_dict[PRIMARY_IDS_KEY][i] = this_primary_id_string

            (current_local_max_dict[SECONDARY_IDS_KEY][i],
             prev_secondary_id_numeric
            ) = _create_secondary_storm_id(prev_secondary_id_numeric)

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
        prev_spc_date_string, prev_secondary_id_numeric):
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
    :param prev_spc_date_string: Previous SPC date (format "yyyymmdd") used in a
        primary storm ID.
    :param prev_secondary_id_numeric: Previous secondary storm ID used.
    :return: intermediate_track_dict: Dictionary with the following keys.
    intermediate_track_dict['current_local_max_dict']: Same as input but maybe
        with different IDs.
    intermediate_track_dict['previous_local_max_dict']: Same as input but maybe
        with different IDs.
    intermediate_track_dict['prev_primary_id_numeric']: Same as input but
        possibly incremented.
    intermediate_track_dict['prev_spc_date_string']: Same as input but possibly
        incremented.
    intermediate_track_dict['prev_secondary_id_numeric']: Same as input but
        possibly incremented.
    """

    num_storm_objects = len(current_local_max_dict[X_COORDS_KEY])

    for i in range(num_storm_objects):
        if current_local_max_dict[PRIMARY_IDS_KEY][i]:
            continue

        these_previous_indices = numpy.where(
            current_to_previous_matrix[i, :]
        )[0]

        if len(these_previous_indices) == 0:
            (current_local_max_dict[PRIMARY_IDS_KEY][i],
             prev_primary_id_numeric, prev_spc_date_string
            ) = _create_primary_storm_id(
                storm_start_time_unix_sec=current_local_max_dict[
                    VALID_TIME_KEY],
                previous_numeric_id=prev_primary_id_numeric,
                previous_spc_date_string=prev_spc_date_string)

            (current_local_max_dict[SECONDARY_IDS_KEY][i],
             prev_secondary_id_numeric
            ) = _create_secondary_storm_id(prev_secondary_id_numeric)

            continue

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
        PREVIOUS_SPC_DATE_KEY: prev_spc_date_string,
        PREVIOUS_SECONDARY_ID_KEY: prev_secondary_id_numeric
    }


def create_full_storm_id(primary_id_string, secondary_id_string):
    """Creates full storm ID from primary and secondary IDs.

    :param primary_id_string: Primary ID.
    :param secondary_id_string: Secondary ID.
    :return: full_id_string: Full ID.
    """

    return '{0:s}_{1:s}'.format(primary_id_string, secondary_id_string)


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


def local_maxima_to_storm_tracks(local_max_dict_by_time):
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

    all_primary_ids = []
    all_secondary_ids = []
    all_first_prev_secondary_ids = []
    all_second_prev_secondary_ids = []
    all_first_next_secondary_ids = []
    all_second_next_secondary_ids = []

    all_times_unix_sec = numpy.array([], dtype=int)
    all_spc_date_strings = []
    all_centroid_latitudes_deg = numpy.array([])
    all_centroid_longitudes_deg = numpy.array([])
    all_centroid_x_metres = numpy.array([])
    all_centroid_y_metres = numpy.array([])

    include_polygons = POLYGON_OBJECTS_LATLNG_KEY in local_max_dict_by_time[0]

    all_polygon_rows_arraylist = []
    all_polygon_columns_arraylist = []
    all_polygon_latitudes_arraylist_deg = []
    all_polygon_longitudes_arraylist_deg = []
    all_polygon_objects_latlng = numpy.array([], dtype=object)
    all_polygon_objects_rowcol = numpy.array([], dtype=object)

    prev_primary_id_numeric = -1
    prev_secondary_id_numeric = -1
    prev_spc_date_string = '00000101'

    old_to_new_primary_id_dict = {}
    num_times = len(local_max_dict_by_time)

    for i in range(num_times + 1):
        if i == num_times:
            this_num_storm_objects = 0
        else:
            this_num_storm_objects = len(
                local_max_dict_by_time[i][LATITUDES_KEY]
            )

            this_empty_2d_list = [
                ['' for _ in range(0)] for _ in range(this_num_storm_objects)
            ]

            local_max_dict_by_time[i].update({
                PRIMARY_IDS_KEY: [''] * this_num_storm_objects,
                SECONDARY_IDS_KEY: [''] * this_num_storm_objects,
                PREV_SECONDARY_IDS_KEY: this_empty_2d_list,
                NEXT_SECONDARY_IDS_KEY: copy.deepcopy(this_empty_2d_list)
            })

        if i == 0:
            for j in range(this_num_storm_objects):
                (local_max_dict_by_time[i][PRIMARY_IDS_KEY][j],
                 prev_primary_id_numeric, prev_spc_date_string
                ) = _create_primary_storm_id(
                    storm_start_time_unix_sec=local_max_dict_by_time[i][
                        VALID_TIME_KEY],
                    previous_numeric_id=prev_primary_id_numeric,
                    previous_spc_date_string=prev_spc_date_string)

                (local_max_dict_by_time[i][SECONDARY_IDS_KEY][j],
                 prev_secondary_id_numeric
                ) = _create_secondary_storm_id(prev_secondary_id_numeric)

            continue

        if i != num_times:
            this_current_to_prev_matrix = copy.deepcopy(
                local_max_dict_by_time[i][CURRENT_TO_PREV_MATRIX_KEY]
            )

            this_dict = _local_maxima_to_tracks_mergers(
                current_local_max_dict=local_max_dict_by_time[i],
                previous_local_max_dict=local_max_dict_by_time[i - 1],
                current_to_previous_matrix=this_current_to_prev_matrix,
                prev_primary_id_numeric=prev_primary_id_numeric,
                prev_spc_date_string=prev_spc_date_string,
                prev_secondary_id_numeric=prev_secondary_id_numeric)

            local_max_dict_by_time[i] = this_dict[CURRENT_LOCAL_MAXIMA_KEY]
            local_max_dict_by_time[i - 1] = this_dict[PREVIOUS_LOCAL_MAXIMA_KEY]
            this_current_to_prev_matrix = this_dict[CURRENT_TO_PREV_MATRIX_KEY]
            prev_primary_id_numeric = this_dict[PREVIOUS_PRIMARY_ID_KEY]
            prev_spc_date_string = this_dict[PREVIOUS_SPC_DATE_KEY]
            prev_secondary_id_numeric = this_dict[PREVIOUS_SECONDARY_ID_KEY]
            old_to_new_primary_id_dict.update(
                this_dict[OLD_TO_NEW_PRIMARY_IDS_KEY])

            this_dict = _local_maxima_to_tracks_splits(
                current_local_max_dict=local_max_dict_by_time[i],
                previous_local_max_dict=local_max_dict_by_time[i - 1],
                current_to_previous_matrix=this_current_to_prev_matrix,
                prev_secondary_id_numeric=prev_secondary_id_numeric)

            local_max_dict_by_time[i] = this_dict[CURRENT_LOCAL_MAXIMA_KEY]
            local_max_dict_by_time[i - 1] = this_dict[PREVIOUS_LOCAL_MAXIMA_KEY]
            this_current_to_prev_matrix = this_dict[CURRENT_TO_PREV_MATRIX_KEY]
            prev_secondary_id_numeric = this_dict[PREVIOUS_SECONDARY_ID_KEY]

            this_dict = _local_maxima_to_tracks_simple(
                current_local_max_dict=local_max_dict_by_time[i],
                previous_local_max_dict=local_max_dict_by_time[i - 1],
                current_to_previous_matrix=this_current_to_prev_matrix,
                prev_primary_id_numeric=prev_primary_id_numeric,
                prev_spc_date_string=prev_spc_date_string,
                prev_secondary_id_numeric=prev_secondary_id_numeric)

            local_max_dict_by_time[i] = this_dict[CURRENT_LOCAL_MAXIMA_KEY]
            local_max_dict_by_time[i - 1] = this_dict[PREVIOUS_LOCAL_MAXIMA_KEY]
            prev_primary_id_numeric = this_dict[PREVIOUS_PRIMARY_ID_KEY]
            prev_spc_date_string = this_dict[PREVIOUS_SPC_DATE_KEY]
            prev_secondary_id_numeric = this_dict[PREVIOUS_SECONDARY_ID_KEY]

        prev_num_storm_objects = len(
            local_max_dict_by_time[i - 1][LATITUDES_KEY]
        )

        all_primary_ids += local_max_dict_by_time[i - 1][PRIMARY_IDS_KEY]
        all_secondary_ids += local_max_dict_by_time[i - 1][SECONDARY_IDS_KEY]

        all_first_prev_secondary_ids += [
            x[0] if len(x) > 0 else ''
            for x in local_max_dict_by_time[i - 1][PREV_SECONDARY_IDS_KEY]
        ]

        all_second_prev_secondary_ids += [
            x[1] if len(x) > 1 else ''
            for x in local_max_dict_by_time[i - 1][PREV_SECONDARY_IDS_KEY]
        ]

        all_first_next_secondary_ids += [
            x[0] if len(x) > 0 else ''
            for x in local_max_dict_by_time[i - 1][NEXT_SECONDARY_IDS_KEY]
        ]

        all_second_next_secondary_ids += [
            x[1] if len(x) > 1 else ''
            for x in local_max_dict_by_time[i - 1][NEXT_SECONDARY_IDS_KEY]
        ]

        these_times_unix_sec = numpy.full(
            prev_num_storm_objects,
            local_max_dict_by_time[i - 1][VALID_TIME_KEY], dtype=int)

        these_spc_date_strings = prev_num_storm_objects * [
            time_conversion.time_to_spc_date_string(these_times_unix_sec[0])
        ]

        all_times_unix_sec = numpy.concatenate((
            all_times_unix_sec, these_times_unix_sec
        ))
        all_spc_date_strings += these_spc_date_strings

        all_centroid_latitudes_deg = numpy.concatenate((
            all_centroid_latitudes_deg,
            local_max_dict_by_time[i - 1][LATITUDES_KEY]
        ))
        all_centroid_longitudes_deg = numpy.concatenate((
            all_centroid_longitudes_deg,
            local_max_dict_by_time[i - 1][LONGITUDES_KEY]
        ))
        all_centroid_x_metres = numpy.concatenate((
            all_centroid_x_metres, local_max_dict_by_time[i - 1][X_COORDS_KEY]
        ))
        all_centroid_y_metres = numpy.concatenate((
            all_centroid_y_metres, local_max_dict_by_time[i - 1][Y_COORDS_KEY]
        ))

        if include_polygons:
            all_polygon_rows_arraylist += local_max_dict_by_time[i - 1][
                GRID_POINT_ROWS_KEY]
            all_polygon_columns_arraylist += local_max_dict_by_time[i - 1][
                GRID_POINT_COLUMNS_KEY]
            all_polygon_latitudes_arraylist_deg += local_max_dict_by_time[
                i - 1][GRID_POINT_LATITUDES_KEY]
            all_polygon_longitudes_arraylist_deg += local_max_dict_by_time[
                i - 1][GRID_POINT_LONGITUDES_KEY]

            all_polygon_objects_latlng = numpy.concatenate((
                all_polygon_objects_latlng,
                local_max_dict_by_time[i - 1][POLYGON_OBJECTS_LATLNG_KEY]
            ))
            all_polygon_objects_rowcol = numpy.concatenate((
                all_polygon_objects_rowcol,
                local_max_dict_by_time[i - 1][POLYGON_OBJECTS_ROWCOL_KEY]
            ))

    storm_object_dict = {
        tracking_utils.PRIMARY_ID_COLUMN: all_primary_ids,
        tracking_utils.SECONDARY_ID_COLUMN: all_secondary_ids,
        tracking_utils.FIRST_PREV_SECONDARY_ID_COLUMN:
            all_first_prev_secondary_ids,
        tracking_utils.SECOND_PREV_SECONDARY_ID_COLUMN:
            all_second_prev_secondary_ids,
        tracking_utils.FIRST_NEXT_SECONDARY_ID_COLUMN:
            all_first_next_secondary_ids,
        tracking_utils.SECOND_NEXT_SECONDARY_ID_COLUMN:
            all_second_next_secondary_ids,
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
                all_polygon_latitudes_arraylist_deg,
            tracking_utils.LONGITUDES_IN_STORM_COLUMN:
                all_polygon_longitudes_arraylist_deg,
            tracking_utils.LATLNG_POLYGON_COLUMN:
                all_polygon_objects_latlng,
            tracking_utils.ROWCOL_POLYGON_COLUMN:
                all_polygon_objects_rowcol
        })

    storm_object_table = pandas.DataFrame.from_dict(storm_object_dict)

    for this_key in old_to_new_primary_id_dict:
        storm_object_table[[tracking_utils.PRIMARY_ID_COLUMN]] = (
            storm_object_table[[tracking_utils.PRIMARY_ID_COLUMN]].replace(
                to_replace=this_key, value=old_to_new_primary_id_dict[this_key],
                inplace=False)
        )

    full_id_strings = [
        create_full_storm_id(primary_id_string=p, secondary_id_string=s)
        for p, s in zip(
            storm_object_table[tracking_utils.PRIMARY_ID_COLUMN].values,
            storm_object_table[tracking_utils.SECONDARY_ID_COLUMN].values
        )
    ]

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


def get_storm_ages(storm_object_table, tracking_start_time_unix_sec,
                   tracking_end_time_unix_sec, max_link_time_seconds,
                   max_join_time_seconds=0):
    """Computes age of each storm cell at each time step.

    :param storm_object_table: pandas DataFrame with at least the following
        columns.
    storm_object_table.primary_storm_id: Primary storm ID.
    storm_object_table.unix_time_sec: Valid time.

    :param tracking_start_time_unix_sec: Start of tracking period.
    :param tracking_end_time_unix_sec: End of tracking period.
    :param max_link_time_seconds: See doc for `link_local_maxima_in_time`.
    :param max_join_time_seconds: Max join time for reanalysis.  If tracks in
        `storm_object_table` have not been reanalyzed, leave this alone.
    :return: storm_object_table: Same as input, but with extra columns listed
        below.
    storm_object_table.age_sec: Age of storm cell.
    storm_object_table.tracking_start_time_unix_sec: Start of tracking period
        (same for all storm objects).
    storm_object_table.tracking_end_time_unix_sec: End of tracking period (same
        for all storm objects).
    storm_object_table.cell_start_time_unix_sec: Start time of storm cell.
    storm_object_table.cell_end_time_unix_sec: End time of storm cell.
    """

    error_checking.assert_is_integer(max_link_time_seconds)
    error_checking.assert_is_greater(max_link_time_seconds, 0)
    error_checking.assert_is_integer(max_join_time_seconds)
    error_checking.assert_is_geq(max_join_time_seconds, 0)
    error_checking.assert_is_integer(tracking_start_time_unix_sec)
    error_checking.assert_is_integer(tracking_end_time_unix_sec)
    error_checking.assert_is_greater(
        tracking_end_time_unix_sec, tracking_start_time_unix_sec)

    num_storm_objects = len(storm_object_table.index)
    ages_seconds = numpy.full(num_storm_objects, -1, dtype=int)
    cell_start_times_unix_sec = numpy.full(num_storm_objects, -1, dtype=int)
    cell_end_times_unix_sec = numpy.full(num_storm_objects, -1, dtype=int)

    id_string_by_track, object_to_track_indices = numpy.unique(
        storm_object_table[tracking_utils.PRIMARY_ID_COLUMN].values,
        return_inverse=True)

    num_storm_tracks = len(id_string_by_track)
    age_invalid_before_unix_sec = (
        tracking_start_time_unix_sec +
        max([max_link_time_seconds, max_join_time_seconds])
    )

    for i in range(num_storm_tracks):
        these_object_indices = numpy.where(object_to_track_indices == i)[0]
        these_times_unix_sec = storm_object_table[
            tracking_utils.VALID_TIME_COLUMN
        ].values[these_object_indices]

        this_start_time_unix_sec = numpy.min(these_times_unix_sec)

        cell_start_times_unix_sec[
            these_object_indices] = this_start_time_unix_sec
        cell_end_times_unix_sec[
            these_object_indices] = numpy.max(these_times_unix_sec)

        if this_start_time_unix_sec < age_invalid_before_unix_sec:
            continue

        ages_seconds[these_object_indices] = (
            these_times_unix_sec - this_start_time_unix_sec
        )

    argument_dict = {
        tracking_utils.TRACKING_START_TIME_COLUMN: numpy.full(
            num_storm_objects, tracking_start_time_unix_sec, dtype=int
        ),
        tracking_utils.TRACKING_END_TIME_COLUMN: numpy.full(
            num_storm_objects, tracking_end_time_unix_sec, dtype=int
        ),
        tracking_utils.AGE_COLUMN: ages_seconds,
        tracking_utils.CELL_START_TIME_COLUMN: cell_start_times_unix_sec,
        tracking_utils.CELL_END_TIME_COLUMN: cell_end_times_unix_sec
    }

    return storm_object_table.assign(**argument_dict)


def find_predecessors(storm_object_table, target_row, num_seconds_back):
    """Finds predecessors of one storm object.

    :param storm_object_table: pandas DataFrame with at least the following
        columns.  Each row is one storm object.
    storm_object_table.unix_time_sec: Valid time.
    storm_object_table.secondary_storm_id: Secondary ID (string).
    storm_object_table.first_prev_secondary_id: Secondary ID of first
        predecessor ("" if no predecessors).
    storm_object_table.second_next_secondary_id: Secondary ID of second
        predecessor ("" if only one predecessor).

    :param target_row: This method will find predecessors for the storm object
        in the [k]th row of the table, where k = `target_row`.
    :param num_seconds_back: This method will find predecessors at the earliest
        time <= N seconds back, where N = `num_seconds_back`.
    :return: predecessor_rows: 1-D numpy array with rows of predecessors.
    """

    error_checking.assert_is_integer(target_row)
    error_checking.assert_is_geq(target_row, 0)
    error_checking.assert_is_less_than(
        target_row, len(storm_object_table.index)
    )

    error_checking.assert_is_integer(num_seconds_back)
    error_checking.assert_is_greater(num_seconds_back, 0)

    # TODO(thunderhoser): This algorithm might be very inefficient for large
    # tables.  I don't know yet.

    unique_times_unix_sec, orig_to_unique_indices = numpy.unique(
        storm_object_table[tracking_utils.VALID_TIME_COLUMN].values,
        return_inverse=True
    )

    target_time_unix_sec = storm_object_table[
        tracking_utils.VALID_TIME_COLUMN].values[target_row]

    earliest_time_index = numpy.where(
        unique_times_unix_sec >= target_time_unix_sec - num_seconds_back
    )[0][0]

    if unique_times_unix_sec[earliest_time_index] == target_time_unix_sec:
        return numpy.array([], dtype=int)

    this_time_index = numpy.where(
        unique_times_unix_sec == target_time_unix_sec
    )[0][0]

    early_rows = []
    secondary_ids_in_frontier = storm_object_table[
        tracking_utils.SECONDARY_ID_COLUMN].values[target_row]

    while this_time_index > earliest_time_index:
        this_time_rows = numpy.where(
            orig_to_unique_indices == this_time_index
        )[0]

        rows_in_frontier = numpy.array([
            k for k in this_time_rows if
            storm_object_table[tracking_utils.SECONDARY_ID_COLUMN].values[k] in
            secondary_ids_in_frontier
        ], dtype=int)

        secondary_ids_in_frontier = []

        for i in rows_in_frontier:
            these_secondary_ids = [
                storm_object_table[c].values[i]
                for c in PREV_SECONDARY_ID_COLUMNS
                if storm_object_table[c].values[i] != ''
            ]

            if len(these_secondary_ids) == 0:
                if i != target_row:
                    early_rows.append(i)
            else:
                secondary_ids_in_frontier += these_secondary_ids

        this_time_index -= 1

    new_early_rows = numpy.where(
        orig_to_unique_indices == earliest_time_index
    )[0]

    new_early_rows = [
        k for k in new_early_rows
        if storm_object_table[tracking_utils.SECONDARY_ID_COLUMN].values[k] in
        secondary_ids_in_frontier
    ]

    return numpy.array(early_rows + new_early_rows, dtype=int)


def get_storm_velocities(
        storm_object_table, num_seconds_back=DEFAULT_VELOCITY_WINDOW_SECONDS,
        test_mode=False):
    """Estimates instantaneous velocity for each storm object.

    :param storm_object_table: pandas DataFrame with at least the following
        columns.  Each row is one storm object, and all storm objects should be
        from the same track (same primary ID).
    storm_object_table.centroid_lat_deg: Latitude (deg N) of storm centroid.
    storm_object_table.centroid_lng_deg: Longitude (deg E) of storm centroid.
    storm_object_table.centroid_x_metres: x-coordinate of storm centroid.
    storm_object_table.centroid_y_metres: y-coordinate of storm centroid.
    storm_object_table.unix_time_sec: Valid time.
    storm_object_table.secondary_storm_id: Secondary ID (string).
    storm_object_table.first_prev_secondary_id: Secondary ID of first
        predecessor ("" if no predecessors).
    storm_object_table.second_next_secondary_id: Secondary ID of second
        predecessor ("" if only one predecessor).

    :param num_seconds_back: Number of seconds to use in each estimate
        (backwards differencing).
    :param test_mode: Never mind.  Just leave this empty.
    :return: storm_object_table: Same as input but with the following extra
        columns.
    storm_object_table.east_velocity_m_s01: Eastward velocity (metres per
        second).
    storm_object_table.north_velocity_m_s01: Northward velocity (metres per
        second).
    """

    error_checking.assert_is_boolean(test_mode)
    num_storm_objects = len(storm_object_table.index)
    east_velocities_m_s01 = numpy.full(num_storm_objects, numpy.nan)
    north_velocities_m_s01 = numpy.full(num_storm_objects, numpy.nan)

    for i in range(num_storm_objects):
        these_predecessor_rows = find_predecessors(
            storm_object_table=storm_object_table, target_row=i,
            num_seconds_back=num_seconds_back)

        this_num_predecessors = len(these_predecessor_rows)
        if this_num_predecessors == 0:
            continue

        this_end_latitude_deg = storm_object_table[
            tracking_utils.CENTROID_LATITUDE_COLUMN].values[i]
        this_end_longitude_deg = storm_object_table[
            tracking_utils.CENTROID_LONGITUDE_COLUMN].values[i]
        these_time_diffs_seconds = (
            storm_object_table[tracking_utils.VALID_TIME_COLUMN].values[i] -
            storm_object_table[tracking_utils.VALID_TIME_COLUMN].values[
                these_predecessor_rows]
        )

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

    east_velocities_m_s01, north_velocities_m_s01 = _estimate_velocity_by_neigh(
        x_coords_metres=storm_object_table[CENTROID_X_COLUMN].values,
        y_coords_metres=storm_object_table[CENTROID_Y_COLUMN].values,
        x_velocities_m_s01=east_velocities_m_s01,
        y_velocities_m_s01=north_velocities_m_s01,
        e_folding_radius_metres=DEFAULT_VELOCITY_EFOLD_RADIUS_METRES)

    argument_dict = {
        tracking_utils.EAST_VELOCITY_COLUMN: east_velocities_m_s01,
        tracking_utils.NORTH_VELOCITY_COLUMN: north_velocities_m_s01
    }

    return storm_object_table.assign(**argument_dict)
