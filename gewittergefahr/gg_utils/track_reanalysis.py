"""Reanalyzes storm tracks.

Reanalysis entails two operations:

[1] Joins tracks across gaps between original tracking periods.  The original
    tracking is usually done for one SPC date (24 hours) at a time, leading to
    false truncations at 1200 UTC of every day.  This module allows tracks to be
    joined across the gap (typically 1155 UTC of SPC date k and 1200 UTC of SPC
    date k + 1).

[2] Joins nearly collinear tracks with a small difference between the end time
    of the first and start time of the second.
"""

import copy
import numpy
from gewittergefahr.gg_utils import temporal_tracking
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import storm_tracking_utils as tracking_utils

LOG_MESSAGE_TIME_FORMAT = '%Y-%m-%d-%H%M%S'

RADIANS_TO_DEGREES = 180. / numpy.pi
CENTRAL_PROJ_LATITUDE_DEG = 35.
CENTRAL_PROJ_LONGITUDE_DEG = 265.

STORM_OBJECT_TABLE_KEY = 'storm_object_table'
LATE_TO_EARLY_KEY = 'late_to_early_matrix'
ID_TO_FIRST_ROW_KEY = 'primary_id_to_first_row_dict'
ID_TO_LAST_ROW_KEY = 'primary_id_to_last_row_dict'


def _create_local_max_dict(storm_object_table, row_indices, include_velocity):
    """Converts pandas DataFrame to dictionary.

    :param storm_object_table: See doc for `_join_collinear_tracks`.
    :param row_indices: 1-D numpy array of rows (in `storm_object_table`) to
        include in the dictionary.  These rows must all have the same valid
        time.  However, this method does not enforce the same-valid-time
        constraint.
    :param include_velocity: Boolean flag.  If True, will include velocities in
        the dictionary.
    :return: local_max_dict: See input doc for
        `temporal_tracking.link_local_maxima_in_time`.
    """

    local_max_dict = {
        temporal_tracking.VALID_TIME_KEY:
            storm_object_table[tracking_utils.VALID_TIME_COLUMN].values[
                row_indices[0]
            ],
        temporal_tracking.X_COORDS_KEY:
            storm_object_table[temporal_tracking.CENTROID_X_COLUMN].values[
                row_indices
            ],
        temporal_tracking.Y_COORDS_KEY:
            storm_object_table[temporal_tracking.CENTROID_Y_COLUMN].values[
                row_indices
            ]
    }

    if not include_velocity:
        return local_max_dict

    local_max_dict.update({
        temporal_tracking.X_VELOCITIES_KEY: storm_object_table[
            temporal_tracking.X_VELOCITY_COLUMN].values[row_indices],
        temporal_tracking.Y_VELOCITIES_KEY: storm_object_table[
            temporal_tracking.Y_VELOCITY_COLUMN].values[row_indices]
    })

    return local_max_dict


def _handle_collinear_splits(
        storm_object_table, early_rows, late_rows, late_to_early_matrix,
        primary_id_to_first_row_dict, primary_id_to_last_row_dict):
    """Handles splits caused by joining collinear tracks.

    E = number of early storms
    L = number of late storms
    N = number of unique primary IDs

    :param storm_object_table: See doc for `_join_collinear_tracks`.
    :param early_rows: length-E numpy with rows (in `storm_object_table`) of
        early storm objects.
    :param late_rows: length-L numpy with rows (in `storm_object_table`) of
        late storm objects.
    :param late_to_early_matrix: L-by-E numpy array of Boolean flags.  If
        late_to_early_matrix[i, j] = True, the [i]th late storm has been linked
        to the [j]th early storm.
    :param primary_id_to_first_row_dict: Dictionary, where each key is a primary
        ID (string) and each value is the first row (in `storm_object_table`)
        with the given ID.
    :param primary_id_to_last_row_dict: Same but for last rows.
    :return: result_dict: Dictionary with the following keys.
    result_dict['storm_object_table']: Same as input but maybe with new primary
        IDs.
    result_dict['late_to_early_matrix']: Same as input but maybe with some
        elements flipped from True to False.
    result_dict['primary_id_to_first_row_dict']: Same as input but maybe with
        some values changed.
    result_dict['primary_id_to_last_row_dict']: Same as input but maybe with
        some values changed.
    """

    num_late_by_early = numpy.sum(late_to_early_matrix, axis=0)
    early_indices_in_split = numpy.where(num_late_by_early > 1)[0]
    early_rows_in_split = early_rows[early_indices_in_split]

    for j in range(len(early_indices_in_split)):
        this_new_id_string = storm_object_table[
            tracking_utils.PRIMARY_ID_COLUMN
        ].values[early_rows_in_split[j]]

        this_early_secondary_id_string = storm_object_table[
            tracking_utils.SECONDARY_ID_COLUMN
        ].values[early_rows_in_split[j]]

        storm_object_table[
            tracking_utils.FIRST_NEXT_SECONDARY_ID_COLUMN
        ].values[early_rows_in_split[j]] = ''

        storm_object_table[
            tracking_utils.SECOND_NEXT_SECONDARY_ID_COLUMN
        ].values[early_rows_in_split[j]] = ''

        these_late_indices = numpy.where(
            late_to_early_matrix[:, early_indices_in_split[j]]
        )[0]

        these_late_rows = late_rows[these_late_indices]
        late_to_early_matrix[:, early_indices_in_split[j]] = False

        for i in range(len(these_late_indices)):
            this_old_id_string = storm_object_table[
                tracking_utils.PRIMARY_ID_COLUMN
            ].values[these_late_rows[i]]

            storm_object_table[[tracking_utils.PRIMARY_ID_COLUMN]] = (
                storm_object_table[
                    [tracking_utils.PRIMARY_ID_COLUMN]
                ].replace(
                    to_replace=this_old_id_string, value=this_new_id_string,
                    inplace=False)
            )

            primary_id_to_first_row_dict[this_old_id_string] = -1
            primary_id_to_last_row_dict[this_old_id_string] = -1

            these_new_id_rows = numpy.where(
                storm_object_table[tracking_utils.PRIMARY_ID_COLUMN].values
                == this_new_id_string
            )[0]

            primary_id_to_first_row_dict[
                this_new_id_string
            ] = these_new_id_rows[0]

            primary_id_to_last_row_dict[
                this_new_id_string
            ] = these_new_id_rows[-1]

            storm_object_table[
                tracking_utils.FIRST_PREV_SECONDARY_ID_COLUMN
            ].values[these_late_rows[i]] = this_early_secondary_id_string

            storm_object_table[
                tracking_utils.SECOND_PREV_SECONDARY_ID_COLUMN
            ].values[these_late_rows[i]] = ''

            this_secondary_id_string = storm_object_table[
                tracking_utils.SECONDARY_ID_COLUMN
            ].values[these_late_rows[i]]

            if storm_object_table[
                tracking_utils.FIRST_NEXT_SECONDARY_ID_COLUMN
            ].values[early_rows_in_split[j]] == '':
                storm_object_table[
                    tracking_utils.FIRST_NEXT_SECONDARY_ID_COLUMN
                ].values[early_rows_in_split[j]] = this_secondary_id_string
            else:
                storm_object_table[
                    tracking_utils.SECOND_NEXT_SECONDARY_ID_COLUMN
                ].values[early_rows_in_split[j]] = this_secondary_id_string

    return {
        STORM_OBJECT_TABLE_KEY: storm_object_table,
        LATE_TO_EARLY_KEY: late_to_early_matrix,
        ID_TO_FIRST_ROW_KEY: primary_id_to_first_row_dict,
        ID_TO_LAST_ROW_KEY: primary_id_to_last_row_dict
    }


def _handle_collinear_mergers(
        storm_object_table, early_rows, late_rows, late_to_early_matrix,
        primary_id_to_first_row_dict, primary_id_to_last_row_dict):
    """Handles mergers caused by joining collinear tracks.

    :param storm_object_table: See doc for `_handle_collinear_splits`.
    :param early_rows: Same.
    :param late_rows: Same.
    :param late_to_early_matrix: Same.
    :return: result_dict: Same.
    """

    num_early_by_late = numpy.sum(late_to_early_matrix, axis=1)
    late_indices_in_merger = numpy.where(num_early_by_late > 1)[0]
    late_rows_in_merger = late_rows[late_indices_in_merger]

    for i in range(len(late_indices_in_merger)):
        this_new_id_string = storm_object_table[
            tracking_utils.PRIMARY_ID_COLUMN
        ].values[late_rows_in_merger[i]]

        this_late_secondary_id_string = storm_object_table[
            tracking_utils.SECONDARY_ID_COLUMN
        ].values[late_rows_in_merger[i]]

        storm_object_table[
            tracking_utils.FIRST_PREV_SECONDARY_ID_COLUMN
        ].values[late_rows_in_merger[i]] = ''

        storm_object_table[
            tracking_utils.SECOND_PREV_SECONDARY_ID_COLUMN
        ].values[late_rows_in_merger[i]] = ''

        these_early_indices = numpy.where(
            late_to_early_matrix[late_indices_in_merger[i], :]
        )[0]

        these_early_rows = early_rows[these_early_indices]
        late_to_early_matrix[late_indices_in_merger[i], :] = False

        for j in range(len(these_early_indices)):
            this_old_id_string = storm_object_table[
                tracking_utils.PRIMARY_ID_COLUMN
            ].values[these_early_rows[j]]

            storm_object_table[[tracking_utils.PRIMARY_ID_COLUMN]] = (
                storm_object_table[
                    [tracking_utils.PRIMARY_ID_COLUMN]
                ].replace(
                    to_replace=this_old_id_string, value=this_new_id_string,
                    inplace=False)
            )

            primary_id_to_first_row_dict[this_old_id_string] = -1
            primary_id_to_last_row_dict[this_old_id_string] = -1

            these_new_id_rows = numpy.where(
                storm_object_table[tracking_utils.PRIMARY_ID_COLUMN].values
                == this_new_id_string
            )[0]

            primary_id_to_first_row_dict[
                this_new_id_string
            ] = these_new_id_rows[0]

            primary_id_to_last_row_dict[
                this_new_id_string
            ] = these_new_id_rows[-1]

            storm_object_table[
                tracking_utils.FIRST_NEXT_SECONDARY_ID_COLUMN
            ].values[these_early_rows[j]] = this_late_secondary_id_string

            storm_object_table[
                tracking_utils.SECOND_NEXT_SECONDARY_ID_COLUMN
            ].values[these_early_rows[j]] = ''

            this_secondary_id_string = storm_object_table[
                tracking_utils.SECONDARY_ID_COLUMN
            ].values[these_early_rows[j]]

            if storm_object_table[
                    tracking_utils.FIRST_PREV_SECONDARY_ID_COLUMN
            ].values[late_rows_in_merger[i]] == '':
                storm_object_table[
                    tracking_utils.FIRST_PREV_SECONDARY_ID_COLUMN
                ].values[late_rows_in_merger[i]] = this_secondary_id_string
            else:
                storm_object_table[
                    tracking_utils.SECOND_PREV_SECONDARY_ID_COLUMN
                ].values[late_rows_in_merger[i]] = this_secondary_id_string

    return {
        STORM_OBJECT_TABLE_KEY: storm_object_table,
        LATE_TO_EARLY_KEY: late_to_early_matrix,
        ID_TO_FIRST_ROW_KEY: primary_id_to_first_row_dict,
        ID_TO_LAST_ROW_KEY: primary_id_to_last_row_dict
    }


def _handle_collinear_1to1_joins(
        storm_object_table, early_rows, late_rows, late_to_early_matrix,
        primary_id_to_first_row_dict, primary_id_to_last_row_dict):
    """Handles one-to-one joins caused by joining collinear tracks.

    :param storm_object_table: See doc for `_handle_collinear_splits`.
    :param early_rows: Same.
    :param late_rows: Same.
    :param late_to_early_matrix: Same.
    :return: result_dict: Dictionary with the following keys.
    result_dict['storm_object_table']: Same as input but maybe with new primary
        and secondary IDs.
    result_dict['primary_id_to_first_row_dict']: Same as input but maybe with
        some values changed.
    result_dict['primary_id_to_last_row_dict']: Same as input but maybe with
        some values changed.
    """

    prev_and_next_columns = [
        tracking_utils.FIRST_PREV_SECONDARY_ID_COLUMN,
        tracking_utils.SECOND_PREV_SECONDARY_ID_COLUMN,
        tracking_utils.FIRST_NEXT_SECONDARY_ID_COLUMN,
        tracking_utils.SECOND_NEXT_SECONDARY_ID_COLUMN
    ]

    late_indices_in_join, early_indices_in_join = numpy.where(
        late_to_early_matrix)

    late_rows_in_join = late_rows[late_indices_in_join]
    early_rows_in_join = early_rows[early_indices_in_join]

    for i in range(len(late_rows_in_join)):

        # Deal with primary IDs.
        this_old_id_string = storm_object_table[
            tracking_utils.PRIMARY_ID_COLUMN
        ].values[early_rows_in_join[i]]

        this_new_id_string = storm_object_table[
            tracking_utils.PRIMARY_ID_COLUMN
        ].values[late_rows_in_join[i]]

        storm_object_table[[tracking_utils.PRIMARY_ID_COLUMN]] = (
            storm_object_table[
                [tracking_utils.PRIMARY_ID_COLUMN]
            ].replace(
                to_replace=this_old_id_string, value=this_new_id_string,
                inplace=False)
        )

        primary_id_to_first_row_dict[this_old_id_string] = -1
        primary_id_to_last_row_dict[this_old_id_string] = -1

        these_new_id_rows = numpy.where(
            storm_object_table[tracking_utils.PRIMARY_ID_COLUMN].values
            == this_new_id_string
        )[0]

        primary_id_to_first_row_dict[
            this_new_id_string
        ] = these_new_id_rows[0]

        primary_id_to_last_row_dict[
            this_new_id_string
        ] = these_new_id_rows[-1]

        # Deal with secondary IDs.
        this_old_id_string = storm_object_table[
            tracking_utils.SECONDARY_ID_COLUMN
        ].values[early_rows_in_join[i]]

        this_new_id_string = storm_object_table[
            tracking_utils.SECONDARY_ID_COLUMN
        ].values[late_rows_in_join[i]]

        storm_object_table[[tracking_utils.SECONDARY_ID_COLUMN]] = (
            storm_object_table[
                [tracking_utils.SECONDARY_ID_COLUMN]
            ].replace(
                to_replace=this_old_id_string, value=this_new_id_string,
                inplace=False)
        )

        storm_object_table[
            tracking_utils.FIRST_NEXT_SECONDARY_ID_COLUMN
        ].values[early_rows_in_join[i]] = this_new_id_string

        storm_object_table[
            tracking_utils.SECOND_NEXT_SECONDARY_ID_COLUMN
        ].values[early_rows_in_join[i]] = ''

        storm_object_table[
            tracking_utils.FIRST_PREV_SECONDARY_ID_COLUMN
        ].values[late_rows_in_join[i]] = this_new_id_string

        storm_object_table[
            tracking_utils.SECOND_PREV_SECONDARY_ID_COLUMN
        ].values[late_rows_in_join[i]] = ''

        storm_object_table[prev_and_next_columns] = (
            storm_object_table[prev_and_next_columns].replace(
                to_replace=this_old_id_string, value=this_new_id_string,
                inplace=False)
        )

    return {
        STORM_OBJECT_TABLE_KEY: storm_object_table,
        ID_TO_FIRST_ROW_KEY: primary_id_to_first_row_dict,
        ID_TO_LAST_ROW_KEY: primary_id_to_last_row_dict
    }


def _update_velocities(storm_object_table, early_rows, late_rows,
                       late_to_early_matrix):
    """Updates velocity estimate for each storm object at late time.

    N_l = number of storm objects at late time
    N_e =  number of storm objects at early time

    :param storm_object_table: See doc for
        `temporal_tracking.join_collinear_tracks`.
    :param early_rows: 1-D numpy array of row indices for early storm objects.
        These are indices into `storm_object_table`.
    :param late_rows: Same but for late storm objects.
    :param late_to_early_matrix: numpy array (N_l x N_e) of Boolean flags.  If
        late_to_early_matrix[i, j] = True, the [i]th late storm object is linked
        to the [j]th early storm object.
    :return: storm_object_table: Same but with different values for
        "x_velocity_m_s01" and "y_velocity_m_s01".
    """

    # TODO(thunderhoser): May not want to update all the velocities here.

    early_local_max_dict = _create_local_max_dict(
        storm_object_table=storm_object_table,
        row_indices=early_rows, include_velocity=False)

    late_local_max_dict = _create_local_max_dict(
        storm_object_table=storm_object_table,
        row_indices=late_rows, include_velocity=False)

    late_local_max_dict[
        temporal_tracking.CURRENT_TO_PREV_MATRIX_KEY] = late_to_early_matrix

    late_local_max_dict = temporal_tracking.get_intermediate_velocities(
        current_local_max_dict=late_local_max_dict,
        previous_local_max_dict=early_local_max_dict)

    storm_object_table[temporal_tracking.X_VELOCITY_COLUMN].values[
        late_rows] = late_local_max_dict[temporal_tracking.X_VELOCITIES_KEY]

    storm_object_table[temporal_tracking.Y_VELOCITY_COLUMN].values[
        late_rows] = late_local_max_dict[temporal_tracking.Y_VELOCITIES_KEY]

    return storm_object_table


def _get_intermediate_velocities_old(storm_object_table, early_rows, late_rows):
    """Returns intermediate velocity estimate for each storm object.

    Input args `early_rows` and `late_rows` will be used to find relevant
    primary storm IDs, and only storm objects with these primary IDs will have
    their velocities updated.

    :param storm_object_table: See doc for
        `temporal_tracking.get_storm_velocities`.
    :param early_rows: 1-D numpy array of row indices.
    :param late_rows: 1-D numpy array of row indices.
    :return: storm_object_table: See doc for
        `temporal_tracking.get_storm_velocities`.
    """

    early_primary_id_strings = storm_object_table[
        tracking_utils.PRIMARY_ID_COLUMN].values[early_rows].tolist()

    late_primary_id_strings = storm_object_table[
        tracking_utils.PRIMARY_ID_COLUMN].values[late_rows].tolist()

    primary_id_strings = early_primary_id_strings + late_primary_id_strings

    update_flags = storm_object_table[
        tracking_utils.PRIMARY_ID_COLUMN].isin(primary_id_strings).values

    update_rows = numpy.where(update_flags)[0]

    storm_object_table.iloc[update_rows] = (
        temporal_tracking.get_storm_velocities(
            storm_object_table=storm_object_table.iloc[update_rows]
        )
    )

    return storm_object_table


def _update_full_ids(storm_object_table):
    """Updates full storm IDs in table.

    :param storm_object_table: See doc for `storm_tracking_io.write_file`.
    :return: storm_object_table: Same as input but maybe with different values
        in "storm_id" column.
    """

    full_id_strings = [
        temporal_tracking.create_full_storm_id(
            primary_id_string=p, secondary_id_string=s
        )
        for p, s in zip(
            storm_object_table[tracking_utils.PRIMARY_ID_COLUMN].values,
            storm_object_table[tracking_utils.SECONDARY_ID_COLUMN].values
        )
    ]

    argument_dict = {tracking_utils.FULL_ID_COLUMN: full_id_strings}
    return storm_object_table.assign(**argument_dict)


def join_collinear_tracks(
        storm_object_table, first_late_time_unix_sec, last_late_time_unix_sec,
        max_join_time_seconds, max_join_error_m_s01):
    """Joins collinear storm tracks.

    This method calls `temporal_tracking.link_local_maxima_in_time`.  For each
    call, there is one dictionary with storm tracks ending at the "early time"
    and another with tracks ending at the "late time".  The goal is to find
    collinear pairs of tracks (one from the early time, one from the late time).
    However, this method also handles splits and mergers.

    :param storm_object_table: pandas DataFrame, where each row is one storm
        object.  Should contain columns listed in
        `storm_tracking_io.write_file`, as well as those listed below.
    storm_object_table.centroid_x_metres: x-coordinate of storm-object centroid.
    storm_object_table.centroid_y_metres: y-coordinate of storm-object centroid.
    storm_object_table.x_velocity_m_s01: Velocity in +x-direction (metres per
        second).
    storm_object_table.y_velocity_m_s01: Velocity in +y-direction (metres per
        second).

    :param first_late_time_unix_sec: First late time to try.
    :param last_late_time_unix_sec: Last late time to try.
    :param max_join_time_seconds: Max difference between early time and late
        time.
    :param max_join_error_m_s01: Max error incurred by extrapolating early track
        to beginning of late track.
    :return: storm_object_table: Same as input but maybe with new primary and/or
        secondary IDs.
    """

    # TODO(thunderhoser): Allow different end times in a merger, different start
    # times in a split.

    # TODO(thunderhoser): May want to allow normal join in this method.

    unique_times_unix_sec, orig_to_unique_time_indices = numpy.unique(
        storm_object_table[tracking_utils.VALID_TIME_COLUMN].values,
        return_inverse=True
    )

    unique_time_strings = [
        time_conversion.unix_sec_to_string(t, LOG_MESSAGE_TIME_FORMAT)
        for t in unique_times_unix_sec
    ]

    unique_primary_id_strings, these_first_rows = numpy.unique(
        storm_object_table[tracking_utils.PRIMARY_ID_COLUMN].values,
        return_index=True)

    primary_id_to_first_row_dict = dict(zip(
        unique_primary_id_strings, these_first_rows
    ))

    _, these_last_rows = numpy.unique(
        storm_object_table[tracking_utils.PRIMARY_ID_COLUMN].values[::-1],
        return_index=True)

    these_last_rows = len(storm_object_table.index) - 1 - these_last_rows
    primary_id_to_last_row_dict = dict(zip(
        unique_primary_id_strings, these_last_rows
    ))

    first_late_time_index = numpy.where(
        unique_times_unix_sec >= first_late_time_unix_sec
    )[0][0]
    last_late_time_index = numpy.where(
        unique_times_unix_sec <= last_late_time_unix_sec
    )[0][-1]

    for j in range(first_late_time_index, last_late_time_index + 1):
        if j <= 1:
            continue

        these_late_rows = numpy.where(orig_to_unique_time_indices == j)[0]
        these_late_rows = numpy.array([
            k for k in these_late_rows
            if k in primary_id_to_first_row_dict.values()
        ], dtype=int)

        if len(these_late_rows) == 0:
            continue

        this_late_local_max_dict = _create_local_max_dict(
            storm_object_table=storm_object_table,
            row_indices=these_late_rows, include_velocity=False)

        these_time_diffs_sec = (
            unique_times_unix_sec[j] - unique_times_unix_sec[:(j - 1)]
        )
        these_early_indices = numpy.where(
            these_time_diffs_sec <= max_join_time_seconds
        )[0][::-1]

        for i in these_early_indices:
            these_early_rows = numpy.where(orig_to_unique_time_indices == i)[0]
            these_early_rows = numpy.array([
                k for k in these_early_rows
                if k in primary_id_to_last_row_dict.values()
            ], dtype=int)

            if len(these_early_rows) == 0:
                continue

            if this_late_local_max_dict is None:
                these_late_rows = numpy.where(
                    orig_to_unique_time_indices == j
                )[0]

                these_late_rows = numpy.array([
                    k for k in these_late_rows
                    if k in primary_id_to_first_row_dict.values()
                ], dtype=int)

                if len(these_late_rows) == 0:
                    break

                this_late_local_max_dict = _create_local_max_dict(
                    storm_object_table=storm_object_table,
                    row_indices=these_late_rows, include_velocity=False)

            this_early_local_max_dict = _create_local_max_dict(
                storm_object_table=storm_object_table,
                row_indices=these_early_rows, include_velocity=True)

            print (
                'Attempting to join collinear tracks ({0:d} early tracks at '
                '{1:s}, {2:d} late tracks at {3:s})...'
            ).format(
                len(these_early_rows), unique_time_strings[i],
                len(these_late_rows), unique_time_strings[j]
            )

            this_late_to_early_matrix = (
                temporal_tracking.link_local_maxima_in_time(
                    current_local_max_dict=this_late_local_max_dict,
                    previous_local_max_dict=this_early_local_max_dict,
                    max_link_time_seconds=max_join_time_seconds,
                    max_velocity_diff_m_s01=max_join_error_m_s01,
                    max_link_distance_m_s01=-1.)
            )

            orig_late_to_early_matrix = copy.deepcopy(this_late_to_early_matrix)

            print (
                'Found {0:d} connections between early and late tracks.\n'
            ).format(numpy.sum(this_late_to_early_matrix))

            if not numpy.any(this_late_to_early_matrix):
                continue

            this_dict = _handle_collinear_splits(
                storm_object_table=storm_object_table,
                early_rows=these_early_rows, late_rows=these_late_rows,
                late_to_early_matrix=this_late_to_early_matrix,
                primary_id_to_first_row_dict=primary_id_to_first_row_dict,
                primary_id_to_last_row_dict=primary_id_to_last_row_dict)

            storm_object_table = this_dict[STORM_OBJECT_TABLE_KEY]
            this_late_to_early_matrix = this_dict[LATE_TO_EARLY_KEY]
            primary_id_to_first_row_dict = this_dict[ID_TO_FIRST_ROW_KEY]
            primary_id_to_last_row_dict = this_dict[ID_TO_LAST_ROW_KEY]

            if not numpy.any(this_late_to_early_matrix):
                # storm_object_table = _update_velocities(
                #     storm_object_table=storm_object_table,
                #     early_rows=these_early_rows, late_rows=these_late_rows,
                #     late_to_early_matrix=orig_late_to_early_matrix)

                this_late_local_max_dict = None
                continue

            this_dict = _handle_collinear_mergers(
                storm_object_table=storm_object_table,
                early_rows=these_early_rows, late_rows=these_late_rows,
                late_to_early_matrix=this_late_to_early_matrix,
                primary_id_to_first_row_dict=primary_id_to_first_row_dict,
                primary_id_to_last_row_dict=primary_id_to_last_row_dict)

            storm_object_table = this_dict[STORM_OBJECT_TABLE_KEY]
            this_late_to_early_matrix = this_dict[LATE_TO_EARLY_KEY]
            primary_id_to_first_row_dict = this_dict[ID_TO_FIRST_ROW_KEY]
            primary_id_to_last_row_dict = this_dict[ID_TO_LAST_ROW_KEY]

            if not numpy.any(this_late_to_early_matrix):
                # storm_object_table = _update_velocities(
                #     storm_object_table=storm_object_table,
                #     early_rows=these_early_rows, late_rows=these_late_rows,
                #     late_to_early_matrix=orig_late_to_early_matrix)

                this_late_local_max_dict = None
                continue

            this_dict = _handle_collinear_1to1_joins(
                storm_object_table=storm_object_table,
                early_rows=these_early_rows, late_rows=these_late_rows,
                late_to_early_matrix=this_late_to_early_matrix,
                primary_id_to_first_row_dict=primary_id_to_first_row_dict,
                primary_id_to_last_row_dict=primary_id_to_last_row_dict)

            storm_object_table = this_dict[STORM_OBJECT_TABLE_KEY]
            primary_id_to_first_row_dict = this_dict[ID_TO_FIRST_ROW_KEY]
            primary_id_to_last_row_dict = this_dict[ID_TO_LAST_ROW_KEY]

            # storm_object_table = _update_velocities(
            #     storm_object_table=storm_object_table,
            #     early_rows=these_early_rows, late_rows=these_late_rows,
            #     late_to_early_matrix=orig_late_to_early_matrix)

            this_late_local_max_dict = None

    return _update_full_ids(storm_object_table)
