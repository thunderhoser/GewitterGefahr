"""Methods for objective evaluation of storm tracks.

--- REFERENCES ---

Lakshmanan, V., and T. Smith, 2010: "An objective method of evaluating and
    devising storm-tracking algorithms". Weather and Forecasting, 25 (2),
    701-709.
"""

import pickle
import numpy
from gewittergefahr.gg_utils import best_tracks
from gewittergefahr.gg_utils import radar_utils
from gewittergefahr.gg_utils import radar_statistics as radar_stats
from gewittergefahr.gg_utils import file_system_utils

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

MAX_LINEARITY_ERROR_METRES = 1e5
DURATION_PERCENTILE_LEVELS = numpy.array(
    [5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 99, 100], dtype=int)

DURATIONS_KEY = 'track_durations_sec'
LINEARITY_ERRORS_KEY = 'track_linearity_errors_metres'
MISMATCH_ERRORS_KEY = 'track_mismatch_errors'
MEAN_LINEARITY_ERROR_KEY = 'mean_linearity_error_metres'
MEAN_MISMATCH_ERROR_KEY = 'mean_mismatch_error'
RADAR_FIELD_KEY = 'radar_field_name'

REQUIRED_KEYS = [
    DURATIONS_KEY, LINEARITY_ERRORS_KEY, MISMATCH_ERRORS_KEY,
    MEAN_LINEARITY_ERROR_KEY, MEAN_MISMATCH_ERROR_KEY, RADAR_FIELD_KEY
]


def _check_dictionary(evaluation_dict):
    """Error-checks dictionary (ensures that required keys are present).

    :param evaluation_dict: Dictionary created by `evaluate_tracks`.
    :raises: ValueError: if any required key is missing.
    """

    missing_keys = list(set(REQUIRED_KEYS) - set(evaluation_dict.keys()))
    if len(missing_keys) == 0:
        return

    error_string = (
        '\n{0:s}\nKeys listed above were expected, but not found, in '
        'dictionary.'
    ).format(str(missing_keys))

    raise ValueError(error_string)


def evaluate_tracks(storm_object_table, top_myrorss_dir_name, radar_field_name):
    """Evaluates a set of storm tracks, following Lakshmanan and Smith (2010).

    T = number of storm tracks

    :param storm_object_table: pandas DataFrame with the set of storm tracks.
        Should contain columns listed in
        `storm_tracking_io.write_processed_file`.
    :param top_myrorss_dir_name: Name of top-level directory with MYRORSS data.
        Files therein will be found by `myrorss_and_mrms_io.find_raw_file` and
        read by `myrorss_and_mrms_io.read_data_from_sparse_grid_file`.
    :param radar_field_name: Name of radar field to use in computing mismatch
        error.  Must be accepted by `radar_utils.check_field_name`.
    :return: evaluation_dict: Dictionary with the following keys.
    evaluation_dict['track_durations_sec']: length-T numpy array of storm
        durations.
    evaluation_dict['track_linearity_errors_metres']: length-T numpy array of
        linearity errors.  The "linearity error" for one track is the RMSE
        (root mean square error) of Theil-Sen estimates over all time steps.
    evaluation_dict['track_mismatch_errors']: length-T numpy array of mismatch
        errors.  The "mismatch error" is the standard deviation of X over all
        time steps, where X = median value of `radar_field_name` inside the
        storm.
    evaluation_dict['mean_linearity_error_metres']: Mean linearity error.  This
        is the mean of trackwise linearity errors for tracks with duration >=
        median duration.
    evaluation_dict['mean_mismatch_error']: Mean mismatch error.  This is the
        mean of trackwise mismatch errors for tracks with duration >= median
        duration.
    evaluation_dict['radar_field_name']: Same as input (metadata).
    """

    print 'Projecting storm centroids from lat-long to x-y coords...'
    storm_object_table = best_tracks.project_centroids_latlng_to_xy(
        storm_object_table)

    print 'Finding storm tracks in list of storm objects...'
    storm_track_table = best_tracks.storm_objects_to_tracks(storm_object_table)

    print SEPARATOR_STRING
    storm_track_table = best_tracks.theil_sen_fit_many_tracks(
        storm_track_table=storm_track_table, verbose=True)
    print SEPARATOR_STRING

    # Compute storm durations.
    track_durations_sec = (
        storm_track_table[best_tracks.TRACK_END_TIME_COLUMN].values -
        storm_track_table[best_tracks.TRACK_START_TIME_COLUMN].values
    ).astype(float)

    for this_percentile_level in DURATION_PERCENTILE_LEVELS:
        this_percentile_sec = numpy.percentile(
            track_durations_sec, float(this_percentile_level)
        )

        print '{0:d}th percentile of track durations = {1:.1f} seconds'.format(
            this_percentile_level, this_percentile_sec)

    median_duration_sec = numpy.median(
        track_durations_sec[track_durations_sec != 0]
    )

    long_track_flags = track_durations_sec >= median_duration_sec
    long_track_indices = numpy.where(long_track_flags)[0]

    # Compute linearity error for each track.
    num_tracks = len(storm_track_table.index)
    track_linearity_errors_metres = numpy.full(num_tracks, numpy.nan)

    print SEPARATOR_STRING

    for i in range(num_tracks):
        if numpy.mod(i, 50) == 0:
            print (
                'Have computed linearity error for {0:d} of {1:d} tracks...'
            ).format(i, num_tracks)

        track_linearity_errors_metres[i] = (
            best_tracks.get_theil_sen_error_one_track(
                storm_track_table=storm_track_table, storm_track_index=i)
        )

    print 'Have computed linearity error for all {0:d} tracks!'.format(
        num_tracks)

    good_indices = numpy.where(numpy.logical_and(
        long_track_flags,
        track_linearity_errors_metres <= MAX_LINEARITY_ERROR_METRES
    ))[0]

    mean_linearity_error_metres = numpy.mean(
        track_linearity_errors_metres[good_indices]
    )

    print 'Mean linearity error = {0:.1f} metres'.format(
        mean_linearity_error_metres)
    print SEPARATOR_STRING

    # Compute mismatch error for each track.
    radar_statistic_table = (
        radar_stats.get_storm_based_radar_stats_myrorss_or_mrms(
            storm_object_table=storm_object_table,
            top_radar_dir_name=top_myrorss_dir_name,
            radar_metadata_dict_for_tracking=None,
            statistic_names=[], percentile_levels=numpy.array([50.]),
            radar_field_names=[radar_field_name],
            radar_source=radar_utils.MYRORSS_SOURCE_ID)
    )

    print SEPARATOR_STRING

    radar_height_m_asl = radar_utils.get_valid_heights(
        data_source=radar_utils.MYRORSS_SOURCE_ID, field_name=radar_field_name
    )[0]

    median_column_name = radar_stats.radar_field_and_percentile_to_column_name(
        radar_field_name=radar_field_name,
        radar_height_m_asl=radar_height_m_asl, percentile_level=50.)

    median_by_storm_object = radar_statistic_table[median_column_name]
    track_mismatch_errors = numpy.full(num_tracks, numpy.nan)

    for i in range(num_tracks):
        these_object_indices = storm_track_table[
            best_tracks.OBJECT_INDICES_COLUMN_FOR_TRACK].values[i]

        if len(these_object_indices) < 2:
            continue

        track_mismatch_errors[i] = numpy.std(
            median_by_storm_object[these_object_indices], ddof=1)

    mean_mismatch_error = numpy.nanmean(
        track_mismatch_errors[long_track_indices]
    )

    print 'Mean mismatch error = {0:.4e} units of "{1:s}"'.format(
        mean_mismatch_error, radar_field_name)

    return {
        DURATIONS_KEY: track_durations_sec.astype(int),
        LINEARITY_ERRORS_KEY: track_linearity_errors_metres,
        MISMATCH_ERRORS_KEY: track_mismatch_errors,
        MEAN_LINEARITY_ERROR_KEY: mean_linearity_error_metres,
        MEAN_MISMATCH_ERROR_KEY: mean_mismatch_error,
        RADAR_FIELD_KEY: radar_field_name
    }


def write_file(evaluation_dict, pickle_file_name):
    """Writes evaluation results to file.

    :param evaluation_dict: Dictionary created by `evaluate_tracks`.
    :param pickle_file_name: Path to output file.
    """

    _check_dictionary(evaluation_dict)
    file_system_utils.mkdir_recursive_if_necessary(file_name=pickle_file_name)

    pickle_file_handle = open(pickle_file_name, 'wb')
    pickle.dump(evaluation_dict, pickle_file_handle)
    pickle_file_handle.close()


def read_file(pickle_file_name):
    """Reads evaluation results from file.

    :param pickle_file_name: Path to input file.
    :return: evaluation_dict: Dictionary created by `evaluate_tracks`.
    """

    pickle_file_handle = open(pickle_file_name, 'rb')
    evaluation_dict = pickle.load(pickle_file_handle)
    pickle_file_handle.close()

    _check_dictionary(evaluation_dict)
    return evaluation_dict
