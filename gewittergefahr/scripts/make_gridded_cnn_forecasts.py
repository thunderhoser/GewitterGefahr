"""Projects CNN forecasts onto the RAP grid."""

import argparse
import numpy
from gewittergefahr.gg_io import storm_tracking_io as tracking_io
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import target_val_utils
from gewittergefahr.gg_utils import gridded_forecasts
from gewittergefahr.gg_utils import echo_top_tracking
from gewittergefahr.gg_utils import storm_tracking_utils as tracking_utils
from gewittergefahr.deep_learning import prediction_io

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

INPUT_FILE_ARG_NAME = 'input_prediction_file_name'
TRACKING_DIR_ARG_NAME = 'input_tracking_dir_name'
TRACKING_SCALE_ARG_NAME = 'tracking_scale_metres2'
X_SPACING_ARG_NAME = 'x_spacing_metres'
Y_SPACING_ARG_NAME = 'y_spacing_metres'
EFFECTIVE_RADIUS_ARG_NAME = 'effective_radius_metres'
SMOOTHING_METHOD_ARG_NAME = 'smoothing_method_name'
CUTOFF_RADIUS_ARG_NAME = 'smoothing_cutoff_radius_metres'
EFOLD_RADIUS_ARG_NAME = 'smoothing_efold_radius_metres'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

INPUT_FILE_HELP_STRING = (
    'Path to file with storm-based predictions (which will be gridded by this '
    'script).  File will be read by `prediction_io.read_file`.')

TRACKING_DIR_HELP_STRING = (
    'Name of top-level directory with tracking data.  Files therein will be '
    'found by `storm_tracking_io.find_processed_files_one_spc_date` and read by'
    ' `storm_tracking_io.read_processed_file`.')

TRACKING_SCALE_HELP_STRING = (
    'Tracking scale (used only to find files in `{0:s}`).'
).format(TRACKING_DIR_ARG_NAME)

X_SPACING_HELP_STRING = 'Spacing between adjacent columns in forecast grid.'

Y_SPACING_HELP_STRING = 'Spacing between adjacent rows in forecast grid.'

EFFECTIVE_RADIUS_HELP_STRING = (
    'Effective radius for gridded probabilities.  The forecast at each grid '
    'point will be "probability of event within K metres", where K = `{0:s}`.'
).format(EFFECTIVE_RADIUS_ARG_NAME)

SMOOTHING_METHOD_HELP_STRING = (
    'Smoothing method for gridded probabilities.  If you do not want to smooth,'
    ' leave this alone.  Otherwise, must be in the following list.\n{0:s}'
).format(str(gridded_forecasts.VALID_SMOOTHING_METHODS))

CUTOFF_RADIUS_HELP_STRING = 'Cutoff radius for smoothing method.'

EFOLD_RADIUS_HELP_STRING = (
    'e-folding radius for Gaussian smoother.  If smoothing method is not '
    'Gaussian, leave this alone.')

OUTPUT_DIR_HELP_STRING = (
    'Name of output directory.  Gridded predictions will be written here by '
    '`prediction_io.write_gridded_predictions`, to an exact location determined'
    ' by `prediction_io.find_file`.')

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + INPUT_FILE_ARG_NAME, type=str, required=True,
    help=INPUT_FILE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + TRACKING_DIR_ARG_NAME, type=str, required=True,
    help=TRACKING_DIR_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + TRACKING_SCALE_ARG_NAME, type=int, required=False,
    default=echo_top_tracking.DUMMY_TRACKING_SCALE_METRES2,
    help=TRACKING_SCALE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + X_SPACING_ARG_NAME, type=float, required=False,
    default=gridded_forecasts.DEFAULT_GRID_SPACING_METRES,
    help=X_SPACING_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + Y_SPACING_ARG_NAME, type=float, required=False,
    default=gridded_forecasts.DEFAULT_GRID_SPACING_METRES,
    help=Y_SPACING_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + EFFECTIVE_RADIUS_ARG_NAME, type=float, required=False,
    default=gridded_forecasts.DEFAULT_PROB_RADIUS_FOR_GRID_METRES,
    help=EFFECTIVE_RADIUS_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + SMOOTHING_METHOD_ARG_NAME, type=str, required=False,
    default='', help=SMOOTHING_METHOD_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + CUTOFF_RADIUS_ARG_NAME, type=float, required=False,
    default=gridded_forecasts.DEFAULT_SMOOTHING_CUTOFF_RADIUS_METRES,
    help=CUTOFF_RADIUS_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + EFOLD_RADIUS_ARG_NAME, type=float, required=False,
    default=gridded_forecasts.DEFAULT_SMOOTHING_E_FOLDING_RADIUS_METRES,
    help=EFOLD_RADIUS_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING)


def _run(input_prediction_file_name, top_tracking_dir_name,
         tracking_scale_metres2, x_spacing_metres, y_spacing_metres,
         effective_radius_metres, smoothing_method_name,
         smoothing_cutoff_radius_metres, smoothing_efold_radius_metres,
         top_output_dir_name):
    """Projects CNN forecasts onto the RAP grid.

    This is effectively the same method.

    :param input_prediction_file_name: See documentation at top of file.
    :param top_tracking_dir_name: Same.
    :param tracking_scale_metres2: Same.
    :param x_spacing_metres: Same.
    :param y_spacing_metres: Same.
    :param effective_radius_metres: Same.
    :param smoothing_method_name: Same.
    :param smoothing_cutoff_radius_metres: Same.
    :param smoothing_efold_radius_metres: Same.
    :param top_output_dir_name: Same.
    """

    print 'Reading data from: "{0:s}"...'.format(input_prediction_file_name)
    ungridded_forecast_dict = prediction_io.read_ungridded_predictions(
        input_prediction_file_name)

    target_param_dict = target_val_utils.target_name_to_params(
        ungridded_forecast_dict[prediction_io.TARGET_NAME_KEY]
    )

    min_buffer_dist_metres = target_param_dict[
        target_val_utils.MIN_LINKAGE_DISTANCE_KEY]

    max_buffer_dist_metres = target_param_dict[
        target_val_utils.MAX_LINKAGE_DISTANCE_KEY]

    min_lead_time_seconds = target_param_dict[
        target_val_utils.MIN_LEAD_TIME_KEY]

    max_lead_time_seconds = target_param_dict[
        target_val_utils.MAX_LEAD_TIME_KEY]

    forecast_column_name = gridded_forecasts._distance_buffer_to_column_name(
        min_buffer_dist_metres=min_buffer_dist_metres,
        max_buffer_dist_metres=max_buffer_dist_metres,
        column_type=gridded_forecasts.FORECAST_COLUMN_TYPE)

    init_times_unix_sec = numpy.unique(
        ungridded_forecast_dict[prediction_io.STORM_TIMES_KEY]
    )

    tracking_file_names = []

    for this_time_unix_sec in init_times_unix_sec:
        this_tracking_file_name = tracking_io.find_processed_file(
            top_processed_dir_name=top_tracking_dir_name,
            tracking_scale_metres2=tracking_scale_metres2,
            data_source=tracking_utils.SEGMOTION_SOURCE_ID,
            unix_time_sec=this_time_unix_sec,
            spc_date_string=time_conversion.time_to_spc_date_string(
                this_time_unix_sec),
            raise_error_if_missing=True
        )

        tracking_file_names.append(this_tracking_file_name)

    storm_object_table = tracking_io.read_many_processed_files(
        tracking_file_names)
    print SEPARATOR_STRING

    tracking_utils.find_storm_objects(
        all_storm_ids=ungridded_forecast_dict[
            prediction_io.STORM_IDS_KEY],
        all_times_unix_sec=ungridded_forecast_dict[
            prediction_io.STORM_TIMES_KEY],
        storm_ids_to_keep=storm_object_table[
            tracking_utils.STORM_ID_COLUMN].values.tolist(),
        times_to_keep_unix_sec=storm_object_table[
            tracking_utils.TIME_COLUMN].values,
        allow_missing=False
    )

    sort_indices = tracking_utils.find_storm_objects(
        all_storm_ids=storm_object_table[
            tracking_utils.STORM_ID_COLUMN].values.tolist(),
        all_times_unix_sec=storm_object_table[
            tracking_utils.TIME_COLUMN].values,
        storm_ids_to_keep=ungridded_forecast_dict[
            prediction_io.STORM_IDS_KEY],
        times_to_keep_unix_sec=ungridded_forecast_dict[
            prediction_io.STORM_TIMES_KEY],
        allow_missing=False
    )

    forecast_probabilities = ungridded_forecast_dict[
        prediction_io.PROBABILITY_MATRIX_KEY
    ][sort_indices, 1]

    storm_object_table = storm_object_table.assign(**{
        forecast_column_name: forecast_probabilities
    })

    gridded_forecast_dict = gridded_forecasts.create_forecast_grids(
        storm_object_table=storm_object_table,
        min_lead_time_sec=min_lead_time_seconds,
        max_lead_time_sec=max_lead_time_seconds,
        lead_time_resolution_sec=
        gridded_forecasts.DEFAULT_LEAD_TIME_RES_SECONDS,
        grid_spacing_x_metres=x_spacing_metres,
        grid_spacing_y_metres=y_spacing_metres,
        interp_to_latlng_grid=False,
        prob_radius_for_grid_metres=effective_radius_metres,
        smoothing_method=smoothing_method_name,
        smoothing_e_folding_radius_metres=smoothing_efold_radius_metres,
        smoothing_cutoff_radius_metres=smoothing_cutoff_radius_metres)

    print SEPARATOR_STRING

    output_file_name = prediction_io.find_file(
        top_prediction_dir_name=top_output_dir_name,
        first_init_time_unix_sec=numpy.min(
            storm_object_table[tracking_utils.TIME_COLUMN].values),
        last_init_time_unix_sec=numpy.max(
            storm_object_table[tracking_utils.TIME_COLUMN].values),
        gridded=True, raise_error_if_missing=False
    )

    print (
        'Writing results (forecast grids for {0:d} initial times) to: '
        '"{1:s}"...'
    ).format(
        len(gridded_forecast_dict[prediction_io.INIT_TIMES_KEY]),
        output_file_name
    )

    prediction_io.write_gridded_predictions(
        gridded_forecast_dict=gridded_forecast_dict,
        pickle_file_name=output_file_name)


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        input_prediction_file_name=getattr(
            INPUT_ARG_OBJECT, INPUT_FILE_ARG_NAME),
        top_tracking_dir_name=getattr(INPUT_ARG_OBJECT, TRACKING_DIR_ARG_NAME),
        tracking_scale_metres2=getattr(
            INPUT_ARG_OBJECT, TRACKING_SCALE_ARG_NAME),
        x_spacing_metres=getattr(INPUT_ARG_OBJECT, X_SPACING_ARG_NAME),
        y_spacing_metres=getattr(INPUT_ARG_OBJECT, Y_SPACING_ARG_NAME),
        effective_radius_metres=getattr(
            INPUT_ARG_OBJECT, EFFECTIVE_RADIUS_ARG_NAME),
        smoothing_method_name=getattr(
            INPUT_ARG_OBJECT, SMOOTHING_METHOD_ARG_NAME),
        smoothing_cutoff_radius_metres=getattr(
            INPUT_ARG_OBJECT, CUTOFF_RADIUS_ARG_NAME),
        smoothing_efold_radius_metres=getattr(
            INPUT_ARG_OBJECT, EFOLD_RADIUS_ARG_NAME),
        top_output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
