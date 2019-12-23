"""Plots spatial distribution of false alarms."""

import argparse
import numpy
import pandas
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot
from gewittergefahr.gg_io import storm_tracking_io as tracking_io
from gewittergefahr.gg_utils import grids
from gewittergefahr.gg_utils import projections
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import storm_tracking_utils as tracking_utils
from gewittergefahr.gg_utils import echo_top_tracking
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.gg_utils import error_checking
from gewittergefahr.deep_learning import prediction_io
from gewittergefahr.scripts import plot_spatially_subset_eval as plotter

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'
DUMMY_TRACKING_SCALE_METRES2 = echo_top_tracking.DUMMY_TRACKING_SCALE_METRES2

MIN_LATITUDE_DEG = 25.
MAX_LATITUDE_DEG = 50.
MIN_LONGITUDE_DEG = 230.
MAX_LONGITUDE_DEG = 300.

CMAP_OBJECT_FOR_COUNTS = pyplot.get_cmap('viridis')
CMAP_OBJECT_FOR_FAR = pyplot.get_cmap('plasma')
MAX_COUNT_PERCENTILE_TO_PLOT = 90.
MAX_FAR_PERCENTILE_TO_PLOT = 99.

FIGURE_RESOLUTION_DPI = 600

PREDICTION_FILE_ARG_NAME = 'input_prediction_file_name'
TRACKING_DIR_ARG_NAME = 'input_tracking_dir_name'
PROB_THRESHOLD_ARG_NAME = 'prob_threshold'
GRID_SPACING_ARG_NAME = 'grid_spacing_metres'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

PREDICTION_FILE_HELP_STRING = (
    'Path to prediction file.  Will be read by '
    '`prediction_io.read_ungridded_predictions`.'
)

TRACKING_DIR_HELP_STRING = (
    'Name of top-level directory with tracking data.  Files therein will be '
    'found by `storm_tracking_io.find_file` and read by'
    ' `storm_tracking_io.read_file`.'
)

PROB_THRESHOLD_HELP_STRING = (
    'Probability threshold.  Any negative example with forecast >= `{0:s}` will'
    ' be considered a false alarm.'
).format(PROB_THRESHOLD_ARG_NAME)

GRID_SPACING_HELP_STRING = 'Grid spacing.'

OUTPUT_DIR_HELP_STRING = (
    'Name of output directory (figures will be saved here).'
)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + PREDICTION_FILE_ARG_NAME, type=str, required=True,
    help=PREDICTION_FILE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + TRACKING_DIR_ARG_NAME, type=str, required=True,
    help=TRACKING_DIR_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + PROB_THRESHOLD_ARG_NAME, type=float, required=True,
    help=PROB_THRESHOLD_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + GRID_SPACING_ARG_NAME, type=float, required=False, default=1e5,
    help=GRID_SPACING_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING
)


def _run(prediction_file_name, top_tracking_dir_name, prob_threshold,
         grid_spacing_metres, output_dir_name):
    """Plots spatial distribution of false alarms.

    This is effectively the main method.

    :param prediction_file_name: See documentation at top of file.
    :param top_tracking_dir_name: Same.
    :param prob_threshold: Same.
    :param grid_spacing_metres: Same.
    :param output_dir_name: Same.
    """

    # Process input args.
    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=output_dir_name
    )
    error_checking.assert_is_greater(prob_threshold, 0.)
    error_checking.assert_is_less_than(prob_threshold, 1.)

    grid_metadata_dict = grids.create_equidistant_grid(
        min_latitude_deg=MIN_LATITUDE_DEG, max_latitude_deg=MAX_LATITUDE_DEG,
        min_longitude_deg=MIN_LONGITUDE_DEG,
        max_longitude_deg=MAX_LONGITUDE_DEG,
        x_spacing_metres=grid_spacing_metres,
        y_spacing_metres=grid_spacing_metres, azimuthal=False)

    # Read predictions and find positive forecasts and false alarms.
    print('Reading predictions from: "{0:s}"...'.format(prediction_file_name))
    prediction_dict = prediction_io.read_ungridded_predictions(
        prediction_file_name
    )

    observed_labels = prediction_dict[prediction_io.OBSERVED_LABELS_KEY]
    forecast_labels = (
        prediction_dict[prediction_io.PROBABILITY_MATRIX_KEY][:, -1] >=
        prob_threshold
    ).astype(int)

    pos_forecast_indices = numpy.where(forecast_labels == 1)[0]
    false_alarm_indices = numpy.where(numpy.logical_and(
        observed_labels == 0, forecast_labels == 1
    ))[0]

    num_examples = len(observed_labels)
    num_positive_forecasts = len(pos_forecast_indices)
    num_false_alarms = len(false_alarm_indices)

    print((
        'Probability threshold = {0:.3f} ... number of examples, positive '
        'forecasts, false alarms = {1:d}, {2:d}, {3:d}'
    ).format(
        prob_threshold, num_examples, num_positive_forecasts, num_false_alarms
    ))

    # Find and read tracking files.
    pos_forecast_id_strings = [
        prediction_dict[prediction_io.STORM_IDS_KEY][k]
        for k in pos_forecast_indices
    ]
    pos_forecast_times_unix_sec = (
        prediction_dict[prediction_io.STORM_TIMES_KEY][pos_forecast_indices]
    )

    file_times_unix_sec = numpy.unique(pos_forecast_times_unix_sec)
    num_files = len(file_times_unix_sec)
    storm_object_tables = [None] * num_files

    print(SEPARATOR_STRING)

    for i in range(num_files):
        this_tracking_file_name = tracking_io.find_file(
            top_tracking_dir_name=top_tracking_dir_name,
            tracking_scale_metres2=DUMMY_TRACKING_SCALE_METRES2,
            source_name=tracking_utils.SEGMOTION_NAME,
            valid_time_unix_sec=file_times_unix_sec[i],
            spc_date_string=time_conversion.time_to_spc_date_string(
                file_times_unix_sec[i]
            ),
            raise_error_if_missing=True
        )

        print('Reading data from: "{0:s}"...'.format(this_tracking_file_name))
        this_table = tracking_io.read_file(this_tracking_file_name)
        storm_object_tables[i] = this_table.loc[
            this_table[tracking_utils.FULL_ID_COLUMN].isin(
                pos_forecast_id_strings
            )
        ]

        if i == 0:
            continue

        storm_object_tables[i] = storm_object_tables[i].align(
            storm_object_tables[0], axis=1
        )[0]

    storm_object_table = pandas.concat(
        storm_object_tables, axis=0, ignore_index=True
    )
    print(SEPARATOR_STRING)

    # Find latitudes and longitudes of false alarms.
    all_id_strings = (
        storm_object_table[tracking_utils.FULL_ID_COLUMN].values.tolist()
    )
    all_times_unix_sec = (
        storm_object_table[tracking_utils.VALID_TIME_COLUMN].values
    )
    good_indices = tracking_utils.find_storm_objects(
        all_id_strings=all_id_strings, all_times_unix_sec=all_times_unix_sec,
        id_strings_to_keep=pos_forecast_id_strings,
        times_to_keep_unix_sec=pos_forecast_times_unix_sec,
        allow_missing=False
    )

    pos_forecast_latitudes_deg = storm_object_table[
        tracking_utils.CENTROID_LATITUDE_COLUMN
    ].values[good_indices]

    pos_forecast_longitudes_deg = storm_object_table[
        tracking_utils.CENTROID_LONGITUDE_COLUMN
    ].values[good_indices]

    false_alarm_id_strings = [
        prediction_dict[prediction_io.STORM_IDS_KEY][k]
        for k in false_alarm_indices
    ]
    false_alarm_times_unix_sec = (
        prediction_dict[prediction_io.STORM_TIMES_KEY][false_alarm_indices]
    )
    good_indices = tracking_utils.find_storm_objects(
        all_id_strings=all_id_strings, all_times_unix_sec=all_times_unix_sec,
        id_strings_to_keep=false_alarm_id_strings,
        times_to_keep_unix_sec=false_alarm_times_unix_sec,
        allow_missing=False
    )

    false_alarm_latitudes_deg = storm_object_table[
        tracking_utils.CENTROID_LATITUDE_COLUMN
    ].values[good_indices]

    false_alarm_longitudes_deg = storm_object_table[
        tracking_utils.CENTROID_LONGITUDE_COLUMN
    ].values[good_indices]

    pos_forecast_x_coords_metres, pos_forecast_y_coords_metres = (
        projections.project_latlng_to_xy(
            latitudes_deg=pos_forecast_latitudes_deg,
            longitudes_deg=pos_forecast_longitudes_deg,
            projection_object=grid_metadata_dict[grids.PROJECTION_KEY]
        )
    )

    num_pos_forecasts_matrix = grids.count_events_on_equidistant_grid(
        event_x_coords_metres=pos_forecast_x_coords_metres,
        event_y_coords_metres=pos_forecast_y_coords_metres,
        grid_point_x_coords_metres=grid_metadata_dict[grids.X_COORDS_KEY],
        grid_point_y_coords_metres=grid_metadata_dict[grids.Y_COORDS_KEY]
    )[0]
    print(SEPARATOR_STRING)

    false_alarm_x_coords_metres, false_alarm_y_coords_metres = (
        projections.project_latlng_to_xy(
            latitudes_deg=false_alarm_latitudes_deg,
            longitudes_deg=false_alarm_longitudes_deg,
            projection_object=grid_metadata_dict[grids.PROJECTION_KEY]
        )
    )

    num_false_alarms_matrix = grids.count_events_on_equidistant_grid(
        event_x_coords_metres=false_alarm_x_coords_metres,
        event_y_coords_metres=false_alarm_y_coords_metres,
        grid_point_x_coords_metres=grid_metadata_dict[grids.X_COORDS_KEY],
        grid_point_y_coords_metres=grid_metadata_dict[grids.Y_COORDS_KEY]
    )[0]
    print(SEPARATOR_STRING)

    num_pos_forecasts_matrix = num_pos_forecasts_matrix.astype(float)
    num_pos_forecasts_matrix[num_pos_forecasts_matrix == 0] = numpy.nan
    num_false_alarms_matrix = num_false_alarms_matrix.astype(float)
    num_false_alarms_matrix[num_false_alarms_matrix == 0] = numpy.nan
    far_matrix = num_false_alarms_matrix / num_pos_forecasts_matrix

    this_max_value = numpy.nanpercentile(
        num_false_alarms_matrix, MAX_COUNT_PERCENTILE_TO_PLOT
    )
    if this_max_value < 10:
        this_max_value = numpy.nanmax(num_false_alarms_matrix)

    figure_object = plotter._plot_one_value(
        data_matrix=num_false_alarms_matrix,
        grid_metadata_dict=grid_metadata_dict,
        colour_map_object=CMAP_OBJECT_FOR_COUNTS,
        min_colour_value=0, max_colour_value=this_max_value,
        plot_cbar_min_arrow=False, plot_cbar_max_arrow=True
    )[0]

    num_false_alarms_file_name = '{0:s}/num_false_alarms.jpg'.format(
        output_dir_name
    )

    print('Saving figure to: "{0:s}"...'.format(num_false_alarms_file_name))
    figure_object.savefig(
        num_false_alarms_file_name, dpi=FIGURE_RESOLUTION_DPI,
        pad_inches=0, bbox_inches='tight'
    )
    pyplot.close(figure_object)

    this_max_value = numpy.nanpercentile(
        num_pos_forecasts_matrix, MAX_COUNT_PERCENTILE_TO_PLOT
    )
    if this_max_value < 10:
        this_max_value = numpy.nanmax(num_pos_forecasts_matrix)

    figure_object = plotter._plot_one_value(
        data_matrix=num_pos_forecasts_matrix,
        grid_metadata_dict=grid_metadata_dict,
        colour_map_object=CMAP_OBJECT_FOR_COUNTS,
        min_colour_value=0, max_colour_value=this_max_value,
        plot_cbar_min_arrow=False, plot_cbar_max_arrow=True
    )[0]

    num_pos_forecasts_file_name = '{0:s}/num_positive_forecasts.jpg'.format(
        output_dir_name
    )

    print('Saving figure to: "{0:s}"...'.format(num_pos_forecasts_file_name))
    figure_object.savefig(
        num_pos_forecasts_file_name, dpi=FIGURE_RESOLUTION_DPI,
        pad_inches=0, bbox_inches='tight'
    )
    pyplot.close(figure_object)

    this_max_value = numpy.nanpercentile(far_matrix, MAX_FAR_PERCENTILE_TO_PLOT)
    this_min_value = numpy.nanpercentile(
        far_matrix, 100. - MAX_FAR_PERCENTILE_TO_PLOT
    )

    figure_object = plotter._plot_one_value(
        data_matrix=far_matrix, grid_metadata_dict=grid_metadata_dict,
        colour_map_object=CMAP_OBJECT_FOR_FAR,
        min_colour_value=this_min_value, max_colour_value=this_max_value,
        plot_cbar_min_arrow=this_min_value > 0.,
        plot_cbar_max_arrow=this_max_value < 1.
    )[0]

    far_file_name = '{0:s}/false_alarm_ratio.jpg'.format(output_dir_name)

    print('Saving figure to: "{0:s}"...'.format(far_file_name))
    figure_object.savefig(
        far_file_name, dpi=FIGURE_RESOLUTION_DPI,
        pad_inches=0, bbox_inches='tight'
    )
    pyplot.close(figure_object)


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        prediction_file_name=getattr(
            INPUT_ARG_OBJECT, PREDICTION_FILE_ARG_NAME
        ),
        top_tracking_dir_name=getattr(INPUT_ARG_OBJECT, TRACKING_DIR_ARG_NAME),
        prob_threshold=getattr(INPUT_ARG_OBJECT, PROB_THRESHOLD_ARG_NAME),
        grid_spacing_metres=getattr(INPUT_ARG_OBJECT, GRID_SPACING_ARG_NAME),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
