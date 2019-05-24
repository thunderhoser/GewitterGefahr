"""Methods for objective evaluation of storm tracks.

--- REFERENCES ---

Lakshmanan, V., and T. Smith, 2010: "An objective method of evaluating and
    devising storm-tracking algorithms". Weather and Forecasting, 25 (2),
    701-709.
"""

import pickle
import numpy
from sklearn.linear_model import TheilSenRegressor
from gewittergefahr.gg_utils import projections
from gewittergefahr.gg_utils import echo_top_tracking
from gewittergefahr.gg_utils import radar_utils
from gewittergefahr.gg_utils import radar_statistics as radar_stats
from gewittergefahr.gg_utils import storm_tracking_utils as tracking_utils
from gewittergefahr.gg_utils import file_system_utils

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

X_INTERCEPT_KEY = 'x_intercept_metres'
X_VELOCITY_KEY = 'x_velocity_m_s01'
Y_INTERCEPT_KEY = 'y_intercept_metres'
Y_VELOCITY_KEY = 'y_velocity_m_s01'

X_INTERCEPT_COLUMN = 'x_intercept_metres'
X_VELOCITY_COLUMN = 'x_velocity_m_s01'
Y_INTERCEPT_COLUMN = 'y_intercept_metres'
Y_VELOCITY_COLUMN = 'y_velocity_m_s01'

MAX_LINEARITY_ERROR_METRES = 1e5
DURATION_PERCENTILE_LEVELS = numpy.array(
    [5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 99, 100], dtype=float
)

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


def _fit_theil_sen_one_track(x_coords_metres, y_coords_metres,
                             valid_times_unix_sec):
    """Fits Theil-Sen model for one storm track.

    P = number of points in track

    :param x_coords_metres: length-P numpy array of x-coordinates.
    :param y_coords_metres: length-P numpy array of y-coordinates.
    :param valid_times_unix_sec: length-P numpy array of times.
    :return: theil_sen_dict: Dictionary with the following keys.
    theil_sen_dict['x_intercept_metres']: x-intercept.
    theil_sen_dict['x_velocity_m_s01']: x-velocity (metres per second).
    theil_sen_dict['y_intercept_metres']: y-intercept.
    theil_sen_dict['y_velocity_m_s01']: y-velocity (metres per second).
    """

    num_points = len(x_coords_metres)
    valid_times_unix_sec = numpy.reshape(valid_times_unix_sec, (num_points, 1))

    model_object_for_x = TheilSenRegressor(fit_intercept=True)
    model_object_for_x.fit(valid_times_unix_sec, x_coords_metres)
    model_object_for_y = TheilSenRegressor(fit_intercept=True)
    model_object_for_y.fit(valid_times_unix_sec, y_coords_metres)

    return {
        X_INTERCEPT_KEY: model_object_for_x.intercept_,
        X_VELOCITY_KEY: model_object_for_x.coef_,
        Y_INTERCEPT_KEY: model_object_for_y.intercept_,
        Y_VELOCITY_KEY: model_object_for_y.coef_
    }


def _fit_theil_sen_many_tracks(storm_track_table):
    """Fits Theil-Sen model for each storm track.

    :param storm_track_table: pandas DataFrame created by
        `storm_tracking_utils.storm_objects_to_tracks`.
    :return: storm_track_table: Same as input but with extra columns listed
        below.
    storm_track_table['theil_sen_x_intercept_metres']: x-intercept.
    storm_track_table['theil_sen_x_velocity_m_s01']: x-velocity (metres per
        second).
    storm_track_table['theil_sen_y_intercept_metres']: y-intercept.
    storm_track_table['theil_sen_y_velocity_m_s01']: y-velocity (metres per
        second).
    """

    num_tracks = len(storm_track_table.index)

    x_intercepts_metres = numpy.full(num_tracks, numpy.nan)
    x_velocities_m_s01 = numpy.full(num_tracks, numpy.nan)
    y_intercepts_metres = numpy.full(num_tracks, numpy.nan)
    y_velocities_m_s01 = numpy.full(num_tracks, numpy.nan)

    for i in range(num_tracks):
        if numpy.mod(i, 100) == 0:
            print((
                'Have fit Theil-Sen model for {0:d} of {1:d} storm tracks...'
            ).format(i, num_tracks))

        this_dict = _fit_theil_sen_one_track(
            x_coords_metres=storm_track_table[
                tracking_utils.TRACK_X_COORDS_COLUMN].values[i],
            y_coords_metres=storm_track_table[
                tracking_utils.TRACK_Y_COORDS_COLUMN].values[i],
            valid_times_unix_sec=storm_track_table[
                tracking_utils.TRACK_TIMES_COLUMN].values[i]
        )

        x_intercepts_metres[i] = this_dict[X_INTERCEPT_KEY]
        x_velocities_m_s01[i] = this_dict[X_VELOCITY_KEY]
        y_intercepts_metres[i] = this_dict[Y_INTERCEPT_KEY]
        y_velocities_m_s01[i] = this_dict[Y_VELOCITY_KEY]

    print('Have fit Theil-Sen model for all {0:d} storm tracks!'.format(
        num_tracks))

    return storm_track_table.assign(**{
        X_INTERCEPT_COLUMN: x_intercepts_metres,
        X_VELOCITY_COLUMN: x_velocities_m_s01,
        Y_INTERCEPT_COLUMN: y_intercepts_metres,
        Y_VELOCITY_COLUMN: y_velocities_m_s01
    })


def _apply_theil_sen_one_track(storm_track_table, row_index):
    """Applies Theil-Sen model to each time in one storm track.

    P = number of points in track

    :param storm_track_table: pandas DataFrame created by
        `_fit_theil_sen_many_tracks`.
    :param row_index: Will apply to [k]th track in table, where k = `row_index`.
    :return: predicted_x_coords_metres: length-P numpy array of x-coordinates
        predicted by Theil-Sen model.
    :return: predicted_y_coords_metres: Same but for y.
    """

    predicted_x_coords_metres = (
        storm_track_table[X_INTERCEPT_COLUMN].values[row_index] +
        storm_track_table[X_VELOCITY_COLUMN].values[row_index] *
        storm_track_table[tracking_utils.TRACK_TIMES_COLUMN].values[row_index]
    )

    predicted_y_coords_metres = (
        storm_track_table[Y_INTERCEPT_COLUMN].values[row_index] +
        storm_track_table[Y_VELOCITY_COLUMN].values[row_index] *
        storm_track_table[tracking_utils.TRACK_TIMES_COLUMN].values[row_index]
    )

    return predicted_x_coords_metres, predicted_y_coords_metres


def _get_mean_ts_error_one_track(storm_track_table, row_index):
    """Computes mean Theil-Sen error for one storm track.

    :param storm_track_table: pandas DataFrame created by
        `_fit_theil_sen_many_tracks`.
    :param row_index: Will compute error for [k]th row in table, where k =
        `row_index`.
    :return: rmse_metres: Root mean squared error of storm locations predicted
        by Theil-Sen model.
    """

    predicted_x_coords_metres, predicted_y_coords_metres = (
        _apply_theil_sen_one_track(
            storm_track_table=storm_track_table, row_index=row_index)
    )

    x_errors_metres = (
        predicted_x_coords_metres -
        storm_track_table[tracking_utils.TRACK_X_COORDS_COLUMN].values[
            row_index]
    )

    y_errors_metres = (
        predicted_y_coords_metres -
        storm_track_table[tracking_utils.TRACK_Y_COORDS_COLUMN].values[
            row_index]
    )

    return numpy.sqrt(numpy.mean(x_errors_metres ** 2 + y_errors_metres ** 2))


def evaluate_tracks(storm_object_table, top_myrorss_dir_name, radar_field_name):
    """Evaluates a set of storm tracks, following Lakshmanan and Smith (2010).

    T = number of storm tracks

    :param storm_object_table: pandas DataFrame with storm objects.  Should
        contain columns listed in `storm_tracking_io.write_file`.
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

    # Add x-y coordinates.
    projection_object = projections.init_azimuthal_equidistant_projection(
        central_latitude_deg=echo_top_tracking.CENTRAL_PROJ_LATITUDE_DEG,
        central_longitude_deg=echo_top_tracking.CENTRAL_PROJ_LONGITUDE_DEG)

    x_coords_metres, y_coords_metres = projections.project_latlng_to_xy(
        latitudes_deg=storm_object_table[
            tracking_utils.CENTROID_LATITUDE_COLUMN].values,
        longitudes_deg=storm_object_table[
            tracking_utils.CENTROID_LONGITUDE_COLUMN].values,
        projection_object=projection_object,
        false_easting_metres=0., false_northing_metres=0.)

    storm_object_table = storm_object_table.assign(**{
        tracking_utils.CENTROID_X_COLUMN: x_coords_metres,
        tracking_utils.CENTROID_Y_COLUMN: y_coords_metres
    })

    # Convert list of storm objects to list of tracks.
    storm_track_table = tracking_utils.storm_objects_to_tracks(
        storm_object_table)

    # Fit Theil-Sen model to each track.
    print(SEPARATOR_STRING)
    storm_track_table = _fit_theil_sen_many_tracks(storm_track_table)
    print(SEPARATOR_STRING)

    # Compute storm durations.
    num_tracks = len(storm_track_table.index)
    track_durations_sec = numpy.full(num_tracks, -1, dtype=int)

    for i in range(num_tracks):
        this_start_time_unix_sec = numpy.min(
            storm_track_table[tracking_utils.TRACK_TIMES_COLUMN].values[i]
        )
        this_end_time_unix_sec = numpy.max(
            storm_track_table[tracking_utils.TRACK_TIMES_COLUMN].values[i]
        )
        track_durations_sec[i] = (
            this_end_time_unix_sec - this_start_time_unix_sec
        )

    for this_percentile_level in DURATION_PERCENTILE_LEVELS:
        this_percentile_seconds = numpy.percentile(
            track_durations_sec, this_percentile_level)

        print('{0:d}th percentile of track durations = {1:.1f} seconds'.format(
            int(numpy.round(this_percentile_level)), this_percentile_seconds
        ))

    print('\n')

    for this_percentile_level in DURATION_PERCENTILE_LEVELS:
        this_percentile_seconds = numpy.percentile(
            track_durations_sec[track_durations_sec != 0], this_percentile_level)

        print((
            '{0:d}th percentile of non-zero track durations = {1:.1f} seconds'
        ).format(
            int(numpy.round(this_percentile_level)), this_percentile_seconds
        ))

    print(SEPARATOR_STRING)

    median_duration_sec = numpy.median(
        track_durations_sec[track_durations_sec != 0]
    )
    long_track_flags = track_durations_sec >= median_duration_sec
    long_track_indices = numpy.where(long_track_flags)[0]

    # Compute linearity error for each track.
    track_linearity_errors_metres = numpy.full(num_tracks, numpy.nan)

    for i in range(num_tracks):
        if numpy.mod(i, 50) == 0:
            print((
                'Have computed linearity error for {0:d} of {1:d} tracks...'
            ).format(i, num_tracks))

        track_linearity_errors_metres[i] = _get_mean_ts_error_one_track(
            storm_track_table=storm_track_table, row_index=i)

    print('Have computed linearity error for all {0:d} tracks!'.format(
        num_tracks))

    good_indices = numpy.where(numpy.logical_and(
        long_track_flags,
        track_linearity_errors_metres <= MAX_LINEARITY_ERROR_METRES
    ))[0]

    mean_linearity_error_metres = numpy.mean(
        track_linearity_errors_metres[good_indices]
    )

    print('Mean linearity error = {0:.1f} metres'.format(
        mean_linearity_error_metres))
    print(SEPARATOR_STRING)

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

    print(SEPARATOR_STRING)

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
            tracking_utils.OBJECT_INDICES_COLUMN
        ].values[i]

        if len(these_object_indices) < 2:
            continue

        track_mismatch_errors[i] = numpy.std(
            median_by_storm_object[these_object_indices], ddof=1
        )

    mean_mismatch_error = numpy.nanmean(
        track_mismatch_errors[long_track_indices]
    )

    print('Mean mismatch error for "{0:s}" = {1:.4e}'.format(
        radar_field_name, mean_mismatch_error))

    return {
        DURATIONS_KEY: track_durations_sec,
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
