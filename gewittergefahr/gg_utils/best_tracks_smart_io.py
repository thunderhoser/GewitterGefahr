"""This module runs the w2besttrack algorithm with smart IO.

For more on w2besttrack, see best_tracks.py.

"Smart IO" means that, regardless of the time period to be processed, this
method holds only 3 SPC dates in memory at once.  Specifically, while working on
SPC date k, this method holds dates (k - 1)...(k + 1) in memory.  Holding
adjacent dates in memory prevents arbitrary cutoffs, where all tracks appear to
end when an SPC date ends.  Since no thunderstorm lasts > 24 hours (one day),
holding only dates (k - 1)...(k + 1) is sufficient.  If thunderstorms lasted up
to 48 hours, for example, we would need to hold dates (k - 2)...(k + 2).
"""

import os
import copy
import pickle
import tempfile
import numpy
import pandas
from gewittergefahr.gg_io import storm_tracking_io as tracking_io
from gewittergefahr.gg_utils import best_tracks
from gewittergefahr.gg_utils import projections
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.gg_utils import error_checking

DAYS_TO_SECONDS = 86400
TIME_FORMAT_FOR_MESSAGES = '%Y-%m-%d-%H%M%S'
PROJECTION_OBJECT = projections.init_azimuthal_equidistant_projection(35., 265.)

SPC_DATES_KEY = 'spc_dates_unix_sec'
TEMP_FILE_NAMES_KEY = 'temp_file_names_key'
INPUT_FILE_NAMES_KEY = 'input_file_names_by_spc_date'
OUTPUT_FILE_NAMES_KEY = 'output_file_names_by_spc_date'

INTERMEDIATE_COLUMNS = [
    tracking_io.STORM_ID_COLUMN, tracking_io.ORIG_STORM_ID_COLUMN,
    tracking_io.TIME_COLUMN, tracking_io.SPC_DATE_COLUMN,
    best_tracks.CENTROID_X_COLUMN, best_tracks.CENTROID_Y_COLUMN,
    best_tracks.FILE_INDEX_COLUMN]


def _read_intermediate_results(temp_file_name):
    """Reads intermediate best-track results for a subset of storm objects.

    :param temp_file_name: Path to intermediate file.
    :return: storm_object_table: See documentation for
        _write_intermediate_results.
    """

    pickle_file_handle = open(temp_file_name, 'rb')
    storm_object_table = pickle.load(pickle_file_handle)
    pickle_file_handle.close()

    error_checking.assert_columns_in_dataframe(
        storm_object_table, INTERMEDIATE_COLUMNS)
    return storm_object_table


def _write_intermediate_results(storm_object_table, temp_file_name):
    """Writes intermediate best-track results for a subset of storm objects.

    :param: storm_object_table: pandas DataFrame with the following columns.
        Each row is one storm object.
    storm_object_table.storm_id: String ID for storm track.
    storm_object_table.original_storm_id: Original string ID for storm track.
    storm_object_table.unix_time_sec: Valid time of storm object.
    storm_object_table.centroid_x_metres: x-coordinate of storm-object centroid.
    storm_object_table.centroid_y_metres: y-coordinate of storm-object centroid.
    storm_object_table.spc_date_unix_sec: Valid SPC date for storm object.
    storm_object_table.file_index: Index of file from which storm object was
        read.  This is an index into the file-name array for the given SPC date.

    :param temp_file_name: Path to intermediate file.
    """

    file_system_utils.mkdir_recursive_if_necessary(file_name=temp_file_name)

    pickle_file_handle = open(temp_file_name, 'wb')
    pickle.dump(storm_object_table[INTERMEDIATE_COLUMNS], pickle_file_handle)
    pickle_file_handle.close()


def _shuffle_data_with_smart_io(
        storm_object_table=None, file_dict=None, working_spc_date_unix_sec=None,
        read_from_intermediate=None):
    """Shuffles data with smart IO.

    Specifically, this method ensures that only SPC dates (k - 1)...(k + 1) are
    in memory, where k is the date currently being worked on.

    :param storm_object_table: pandas DataFrame with columns documented in
        _write_intermediate_results.
    :param file_dict: See documentation for find_files_for_smart_io..
    :param working_spc_date_unix_sec: Next SPC date to work on.
    :param read_from_intermediate: Boolean flag.  If True, will read from
        intermediate files.  If False, will read from input files.
    :return: storm_object_table: pandas DataFrame with columns documented in
        _write_intermediate_results.
    """

    working_spc_date_index = numpy.where(
        file_dict[SPC_DATES_KEY] == working_spc_date_unix_sec)[0][0]
    num_spc_dates = len(file_dict[SPC_DATES_KEY])

    if working_spc_date_index == 0:
        read_spc_date_indices = numpy.array([0, 1], dtype=int)
        write_spc_date_indices = numpy.array(
            [num_spc_dates - 2, num_spc_dates - 1], dtype=int)
        clear_table = True

    elif working_spc_date_index == num_spc_dates - 1:
        read_spc_date_indices = numpy.array([], dtype=int)
        write_spc_date_indices = numpy.array([num_spc_dates - 3], dtype=int)
        clear_table = False

    else:
        read_spc_date_indices = numpy.array(
            [working_spc_date_index + 1], dtype=int)
        write_spc_date_indices = numpy.array(
            [working_spc_date_index - 2], dtype=int)
        clear_table = False

    read_spc_date_indices = read_spc_date_indices[read_spc_date_indices >= 0]
    read_spc_date_indices = read_spc_date_indices[
        read_spc_date_indices < num_spc_dates]
    write_spc_date_indices = write_spc_date_indices[write_spc_date_indices >= 0]
    write_spc_date_indices = write_spc_date_indices[
        write_spc_date_indices < num_spc_dates]

    if storm_object_table is not None:
        for this_index in write_spc_date_indices:
            this_spc_date_unix_sec = file_dict[SPC_DATES_KEY][this_index]
            this_spc_date_string = time_conversion.time_to_spc_date_string(
                this_spc_date_unix_sec)

            this_spc_date_indices = numpy.where(
                storm_object_table[tracking_io.SPC_DATE_COLUMN].values ==
                this_spc_date_unix_sec)[0]

            this_temp_file_name = file_dict[TEMP_FILE_NAMES_KEY][this_index]
            print ('Writing intermediate data for ' + this_spc_date_string +
                   ': ' + this_temp_file_name + '...')

            _write_intermediate_results(
                storm_object_table.iloc[this_spc_date_indices],
                this_temp_file_name)
            storm_object_table.drop(
                storm_object_table.index[this_spc_date_indices], axis=0,
                inplace=True)

    if clear_table:
        storm_object_table = None

    for this_index in read_spc_date_indices:
        this_spc_date_unix_sec = file_dict[SPC_DATES_KEY][this_index]
        this_spc_date_string = time_conversion.time_to_spc_date_string(
            this_spc_date_unix_sec)

        if read_from_intermediate:
            this_temp_file_name = file_dict[TEMP_FILE_NAMES_KEY][this_index]
            print ('Reading intermediate data for ' + this_spc_date_string +
                   ': ' + this_temp_file_name + '...')

            this_storm_object_table = _read_intermediate_results(
                this_temp_file_name)

        else:
            this_storm_object_table = best_tracks.read_input_storm_objects(
                file_dict[INPUT_FILE_NAMES_KEY][this_index], keep_spc_date=True)

            these_centroid_x_metres, these_centroid_y_metres = (
                projections.project_latlng_to_xy(
                    this_storm_object_table[
                        tracking_io.CENTROID_LAT_COLUMN].values,
                    this_storm_object_table[
                        tracking_io.CENTROID_LNG_COLUMN].values,
                    projection_object=PROJECTION_OBJECT,
                    false_easting_metres=0., false_northing_metres=0.))

            argument_dict = {
                best_tracks.CENTROID_X_COLUMN: these_centroid_x_metres,
                best_tracks.CENTROID_Y_COLUMN: these_centroid_y_metres}
            this_storm_object_table = this_storm_object_table.assign(
                **argument_dict)
            this_storm_object_table.drop(
                [tracking_io.CENTROID_LAT_COLUMN,
                 tracking_io.CENTROID_LNG_COLUMN], axis=1, inplace=True)

        if storm_object_table is None:
            storm_object_table = copy.deepcopy(this_storm_object_table)
        else:
            this_storm_object_table, _ = this_storm_object_table.align(
                storm_object_table, axis=1)
            storm_object_table = pandas.concat(
                [storm_object_table, this_storm_object_table], axis=0,
                ignore_index=True)

    return storm_object_table


def _find_tracks_with_spc_date(storm_object_table, storm_track_table,
                               spc_date_unix_sec=None):
    """Finds tracks with at least one storm object in given SPC date.

    :param storm_object_table: pandas DataFrame with at least the following
        columns.
    storm_object_table.spc_date_unix_sec: Valid SPC date of storm object.

    :param storm_track_table: pandas DataFrame with columns documented in
        _storm_objects_to_tracks.
    :param spc_date_unix_sec: SPC date.
    :return: rows_with_spc_date: 1-D numpy array of relevant rows in
        storm_track_table.
    """

    rows_with_spc_date = []
    num_storm_tracks = len(storm_track_table)

    for i in range(num_storm_tracks):
        these_object_indices = numpy.array(
            storm_track_table[
                best_tracks.OBJECT_INDICES_COLUMN_FOR_TRACK].values[i])
        these_spc_dates_unix_sec = numpy.array(
            storm_object_table[tracking_io.SPC_DATE_COLUMN].values[
                these_object_indices])
        if not numpy.any(these_spc_dates_unix_sec == spc_date_unix_sec):
            continue

        rows_with_spc_date.append(i)

    return numpy.array(rows_with_spc_date)


def find_files_for_smart_io(
        start_time_unix_sec=None, start_spc_date_string=None,
        end_time_unix_sec=None, end_spc_date_string=None, data_source=None,
        tracking_scale_metres2=None, top_input_dir_name=None,
        top_output_dir_name=None):
    """Finds input, output, and temporary working files for smart IO.

    N = number of SPC dates in period
    T_i = number of time steps in the [i]th SPC date

    :param start_time_unix_sec: Beginning of time period.
    :param start_spc_date_string: SPC date at beginning of time period (format
        "yyyymmdd").
    :param end_time_unix_sec: End of time period.
    :param end_spc_date_string: SPC date at end of time period (format
        "yyyymmdd").
    :param data_source: Source for input data (examples: "segmotion",
        "probSevere").
    :param tracking_scale_metres2: Tracking scale.
    :param top_input_dir_name: Name of top-level directory for input files.
    :param top_output_dir_name: Name of top-level directory for output files.
    :return: file_dict: Dictionary with the following keys.
    file_dict.spc_dates_unix_sec: length-N numpy array of SPC dates.
    file_dict.temp_file_names: 1-D list of paths to temp files (will be used for
        intermediate IO).
    file_dict.input_file_names_by_spc_date: length-N list, where the [i]th
        element is a 1-D list (length T_i) of paths to input files.
    file_dict.output_file_names_by_spc_date: Same but for output files.

    :raises: ValueError: if start_time_unix_sec is not part of the first SPC
        date (determined by start_spc_date_unix_sec).
    :raises: ValueError: if end_time_unix_sec is not part of the last SPC date
        (determined by end_spc_date_unix_sec).
    """

    if not time_conversion.is_time_in_spc_date(
            start_time_unix_sec, start_spc_date_string):
        start_time_string = time_conversion.unix_sec_to_string(
            start_time_unix_sec, TIME_FORMAT_FOR_MESSAGES)
        raise ValueError(
            'Start time (' + start_time_string + ') is not in first SPC date ('
            + start_spc_date_string + ').')

    if not time_conversion.is_time_in_spc_date(
            end_time_unix_sec, end_spc_date_string):
        end_time_string = time_conversion.unix_sec_to_string(
            end_time_unix_sec, TIME_FORMAT_FOR_MESSAGES)
        raise ValueError(
            'End time (' + end_time_string + ') is not in last SPC date (' +
            end_spc_date_string + ').')

    error_checking.assert_is_greater(
        end_time_unix_sec, start_time_unix_sec)

    start_spc_date_unix_sec = time_conversion.spc_date_string_to_unix_sec(
        start_spc_date_string)
    end_spc_date_unix_sec = time_conversion.spc_date_string_to_unix_sec(
        end_spc_date_string)

    num_spc_dates = int(
        1 + (end_spc_date_unix_sec - start_spc_date_unix_sec) / DAYS_TO_SECONDS)
    spc_dates_unix_sec = numpy.linspace(
        start_spc_date_unix_sec, end_spc_date_unix_sec, num=num_spc_dates,
        dtype=int)

    temp_file_names = [''] * num_spc_dates
    input_file_names_by_spc_date = [['']] * num_spc_dates
    output_file_names_by_spc_date = [['']] * num_spc_dates

    for i in range(num_spc_dates):
        spc_dates_unix_sec[i] = time_conversion.time_to_spc_date_unix_sec(
            spc_dates_unix_sec[i])
        temp_file_names[i] = tempfile.NamedTemporaryFile(delete=False).name

        input_file_names_by_spc_date[i] = (
            tracking_io.find_processed_files_one_spc_date(
                spc_dates_unix_sec[i], data_source=data_source,
                top_processed_dir_name=top_input_dir_name,
                tracking_scale_metres2=tracking_scale_metres2,
                raise_error_if_missing=True))

        this_num_files = len(input_file_names_by_spc_date[i])
        these_times_unix_sec = numpy.full(this_num_files, -1, dtype=int)
        output_file_names_by_spc_date[i] = [''] * this_num_files

        for j in range(this_num_files):
            these_times_unix_sec[j] = tracking_io.processed_file_name_to_time(
                input_file_names_by_spc_date[i][j])
            output_file_names_by_spc_date[i][j] = (
                tracking_io.find_processed_file(
                    unix_time_sec=these_times_unix_sec[j],
                    data_source=data_source,
                    spc_date_unix_sec=spc_dates_unix_sec[i],
                    top_processed_dir_name=top_output_dir_name,
                    tracking_scale_metres2=tracking_scale_metres2,
                    raise_error_if_missing=False))

        if i == 0:
            keep_time_indices = numpy.where(
                these_times_unix_sec >= start_time_unix_sec)[0]

            these_times_unix_sec = these_times_unix_sec[keep_time_indices]
            input_file_names_by_spc_date[i] = [
                input_file_names_by_spc_date[i][j] for j in keep_time_indices]
            output_file_names_by_spc_date[i] = [
                output_file_names_by_spc_date[i][j] for j in keep_time_indices]

        if i == num_spc_dates - 1:
            keep_time_indices = numpy.where(
                these_times_unix_sec <= end_time_unix_sec)[0]
            input_file_names_by_spc_date[i] = [
                input_file_names_by_spc_date[i][j] for j in keep_time_indices]
            output_file_names_by_spc_date[i] = [
                output_file_names_by_spc_date[i][j] for j in keep_time_indices]

    return {SPC_DATES_KEY: spc_dates_unix_sec,
            TEMP_FILE_NAMES_KEY: temp_file_names,
            INPUT_FILE_NAMES_KEY: input_file_names_by_spc_date,
            OUTPUT_FILE_NAMES_KEY: output_file_names_by_spc_date}


def run_best_track(
        smart_file_dict=None,
        max_extrap_time_for_breakup_sec=best_tracks.DEFAULT_MAX_EXTRAP_TIME_SEC,
        max_prediction_error_for_breakup_metres=
        best_tracks.DEFAULT_MAX_PREDICTION_ERROR_METRES,
        max_join_time_sec=best_tracks.DEFAULT_MAX_JOIN_TIME_SEC,
        max_join_distance_metres=best_tracks.DEFAULT_MAX_JOIN_DISTANCE_METRES,
        max_mean_join_error_metres=
        best_tracks.DEFAULT_MAX_MEAN_JOIN_ERROR_METRES,
        max_velocity_diff_for_join_m_s01=None,
        num_main_iters=best_tracks.DEFAULT_NUM_MAIN_ITERS,
        num_breakup_iters=best_tracks.DEFAULT_NUM_BREAKUP_ITERS,
        min_objects_in_track=best_tracks.DEFAULT_MIN_OBJECTS_IN_TRACK):
    """Runs the full w2besttrack algorithm with smart IO.

    :param smart_file_dict: Dictionary created by find_files_for_smart_io.
    :param max_extrap_time_for_breakup_sec: See documentation for
        `best_tracks.run_best_track`.
    :param max_prediction_error_for_breakup_metres: See doc for
        `best_tracks.run_best_track`.
    :param max_join_time_sec: See doc for `best_tracks.run_best_track`.
    :param max_join_distance_metres: See doc for `best_tracks.run_best_track`.
    :param max_mean_join_error_metres: See doc for `best_tracks.run_best_track`.
    :param max_velocity_diff_for_join_m_s01: See doc for `best_tracks.run_best_track`.
    :param num_main_iters: See doc for `best_tracks.run_best_track`.
    :param num_breakup_iters: See doc for `best_tracks.run_best_track`.
    :param min_objects_in_track: See doc for
        `best_tracks.run_best_track`.
    """

    best_tracks.check_best_track_params(
        max_extrap_time_for_breakup_sec=max_extrap_time_for_breakup_sec,
        max_prediction_error_for_breakup_metres=
        max_prediction_error_for_breakup_metres,
        max_join_time_sec=max_join_time_sec,
        max_join_distance_metres=max_join_distance_metres,
        max_mean_join_error_metres=max_mean_join_error_metres,
        max_velocity_diff_for_join_m_s01=max_velocity_diff_for_join_m_s01,
        num_main_iters=num_main_iters, num_breakup_iters=num_breakup_iters,
        min_objects_in_track=min_objects_in_track)

    spc_dates_unix_sec = smart_file_dict[SPC_DATES_KEY]
    num_spc_dates = len(spc_dates_unix_sec)
    storm_object_table = None

    for i in range(num_main_iters):
        print ('Starting main iteration ' + str(i + 1) + '/' +
               str(num_main_iters) + '...\n\n')

        for j in range(num_breakup_iters):
            print ('Starting break-up iteration ' + str(j + 1) + '/' +
                   str(num_breakup_iters) + '...\n\n')

            for k in range(num_spc_dates):
                storm_object_table = _shuffle_data_with_smart_io(
                    storm_object_table=storm_object_table,
                    file_dict=smart_file_dict,
                    working_spc_date_unix_sec=spc_dates_unix_sec[k],
                    read_from_intermediate=i > 0 or j > 0)

                if k == 0:
                    best_track_start_time_unix_sec = numpy.min(
                        storm_object_table[tracking_io.TIME_COLUMN].values)
                if k == num_spc_dates - 1:
                    best_track_end_time_unix_sec = numpy.max(
                        storm_object_table[tracking_io.TIME_COLUMN].values)

                storm_track_table = best_tracks.storm_objects_to_tracks(
                    storm_object_table)
                storm_track_table = best_tracks.theil_sen_fit_for_each_track(
                    storm_track_table)
                these_working_indices = numpy.where(
                    storm_object_table[tracking_io.SPC_DATE_COLUMN].values ==
                    spc_dates_unix_sec[k])[0]

                storm_object_table, storm_track_table = (
                    best_tracks.break_storm_tracks(
                        storm_object_table=storm_object_table,
                        storm_track_table=storm_track_table,
                        working_object_indices=these_working_indices,
                        max_extrapolation_time_sec=
                        max_extrap_time_for_breakup_sec,
                        max_prediction_error_metres=
                        max_prediction_error_for_breakup_metres))

        for k in range(num_spc_dates):
            storm_object_table = _shuffle_data_with_smart_io(
                storm_object_table=storm_object_table,
                file_dict=smart_file_dict,
                working_spc_date_unix_sec=spc_dates_unix_sec[k],
                read_from_intermediate=True)

            storm_track_table = best_tracks.storm_objects_to_tracks(
                storm_object_table)
            storm_track_table = best_tracks.theil_sen_fit_for_each_track(
                storm_track_table)
            these_working_indices = _find_tracks_with_spc_date(
                storm_object_table, storm_track_table,
                spc_date_unix_sec=spc_dates_unix_sec[k])

            storm_object_table, storm_track_table = (
                best_tracks.merge_storm_tracks(
                    storm_object_table=storm_object_table,
                    storm_track_table=storm_track_table,
                    working_track_indices=these_working_indices,
                    max_join_time_sec=max_join_time_sec,
                    max_join_distance_metres=max_join_distance_metres,
                    max_mean_prediction_error_metres=max_mean_join_error_metres,
                    max_velocity_diff_m_s01=max_velocity_diff_for_join_m_s01))

        for k in range(num_spc_dates):
            storm_object_table = _shuffle_data_with_smart_io(
                storm_object_table=storm_object_table,
                file_dict=smart_file_dict,
                working_spc_date_unix_sec=spc_dates_unix_sec[k],
                read_from_intermediate=True)

            storm_track_table = best_tracks.storm_objects_to_tracks(
                storm_object_table)
            storm_track_table = best_tracks.theil_sen_fit_for_each_track(
                storm_track_table)
            these_working_indices = _find_tracks_with_spc_date(
                storm_object_table, storm_track_table,
                spc_date_unix_sec=spc_dates_unix_sec[k])

            storm_object_table, storm_track_table = (
                best_tracks.break_ties_among_storm_objects(
                    storm_object_table, storm_track_table,
                    working_track_indices=these_working_indices))

    for k in range(num_spc_dates):
        storm_object_table = _shuffle_data_with_smart_io(
            storm_object_table=storm_object_table,
            file_dict=smart_file_dict,
            working_spc_date_unix_sec=spc_dates_unix_sec[k],
            read_from_intermediate=True)

        print ('Removing storm tracks with < ' + str(min_objects_in_track) +
               ' objects...')
        storm_object_table = best_tracks.remove_short_tracks(
            storm_object_table, min_objects_in_track=min_objects_in_track)

        print 'Recomputing storm attributes...'
        storm_object_table = best_tracks.recompute_attributes(
            storm_object_table,
            best_track_start_time_unix_sec=best_track_start_time_unix_sec,
            best_track_end_time_unix_sec=best_track_end_time_unix_sec)

        this_spc_date_indices = numpy.where(
            storm_object_table[tracking_io.SPC_DATE_COLUMN].values ==
            spc_dates_unix_sec[k])[0]
        best_tracks.write_output_storm_objects(
            storm_object_table.iloc[this_spc_date_indices],
            input_file_names=smart_file_dict[INPUT_FILE_NAMES_KEY][k],
            output_file_names=smart_file_dict[OUTPUT_FILE_NAMES_KEY][k])

    for k in range(num_spc_dates):
        print 'Deleting temp files...'
        os.remove(smart_file_dict[TEMP_FILE_NAMES_KEY][k])
