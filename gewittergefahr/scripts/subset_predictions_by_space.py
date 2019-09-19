"""Subsets ungridded predictions by space.

Specifically, this script groups predictions into cells on an equidistant grid
and writes one prediction file per grid cell.
"""

import argparse
import numpy
from gewittergefahr.gg_io import storm_tracking_io as tracking_io
from gewittergefahr.gg_utils import grids
from gewittergefahr.gg_utils import projections
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import echo_top_tracking
from gewittergefahr.gg_utils import storm_tracking_utils as tracking_utils
from gewittergefahr.deep_learning import prediction_io

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'
DUMMY_TRACKING_SCALE_METRES2 = echo_top_tracking.DUMMY_TRACKING_SCALE_METRES2

INPUT_FILE_ARG_NAME = 'input_file_name'
TRACKING_DIR_ARG_NAME = 'input_tracking_dir_name'
MIN_LATITUDE_ARG_NAME = 'min_latitude_deg'
MAX_LATITUDE_ARG_NAME = 'max_latitude_deg'
MIN_LONGITUDE_ARG_NAME = 'min_longitude_deg'
MAX_LONGITUDE_ARG_NAME = 'max_longitude_deg'
GRID_SPACING_ARG_NAME = 'grid_spacing_metres'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

INPUT_FILE_HELP_STRING = (
    'Path to main input file (with predictions to be subset).  Will be read by '
    '`prediction_io.read_ungridded_predictions`.')

TRACKING_DIR_HELP_STRING = (
    'Name of top-level tracking directory (will be used to find storm '
    'locations).  Files therein will be found by `storm_tracking_io.find_file` '
    'and read by `storm_tracking_io.read_file`.')

MIN_LATITUDE_HELP_STRING = 'Minimum latitude (deg N) in equidistant grid.'
MAX_LATITUDE_HELP_STRING = 'Max latitude (deg N) in equidistant grid.'
MIN_LONGITUDE_HELP_STRING = 'Minimum longitude (deg E) in equidistant grid.'
MAX_LONGITUDE_HELP_STRING = 'Max longitude (deg E) in equidistant grid.'
GRID_SPACING_HELP_SPACING = 'Spacing for equidistant grid.'

OUTPUT_DIR_HELP_STRING = (
    'Name of output directory.  Spatially subset predictions (one file per '
    'equidistant grid cell) will be written here by '
    '`prediction_io.write_ungridded_predictions`, to exact locations determined'
    ' by `prediction_io.find_ungridded_file`.')

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + INPUT_FILE_ARG_NAME, type=str, required=True,
    help=INPUT_FILE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + TRACKING_DIR_ARG_NAME, type=str, required=True,
    help=TRACKING_DIR_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + MIN_LATITUDE_ARG_NAME, type=float, required=False, default=24.,
    help=MIN_LATITUDE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + MAX_LATITUDE_ARG_NAME, type=float, required=False, default=50.,
    help=MAX_LATITUDE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + MIN_LONGITUDE_ARG_NAME, type=float, required=False, default=234.,
    help=MIN_LONGITUDE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + MAX_LONGITUDE_ARG_NAME, type=float, required=False, default=294.,
    help=MAX_LONGITUDE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + GRID_SPACING_ARG_NAME, type=float, required=False, default=1e5,
    help=GRID_SPACING_HELP_SPACING)

INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING)


def _read_storm_locations_one_time(
        top_tracking_dir_name, valid_time_unix_sec, desired_full_id_strings):
    """Reads storm locations at one time.

    K = number of storm objects desired

    :param top_tracking_dir_name: See documentation at top of file.
    :param valid_time_unix_sec: Valid time.
    :param desired_full_id_strings: length-K list of full storm IDs.  Locations
        will be read for these storms only.
    :return: desired_latitudes_deg: length-K numpy array of latitudes (deg N).
    :return: desired_longitudes_deg: length-K numpy array of longitudes (deg E).
    """

    spc_date_string = time_conversion.time_to_spc_date_string(
        valid_time_unix_sec)
    desired_times_unix_sec = numpy.full(
        len(desired_full_id_strings), valid_time_unix_sec, dtype=int
    )

    tracking_file_name = tracking_io.find_file(
        top_tracking_dir_name=top_tracking_dir_name,
        tracking_scale_metres2=DUMMY_TRACKING_SCALE_METRES2,
        source_name=tracking_utils.SEGMOTION_NAME,
        valid_time_unix_sec=valid_time_unix_sec,
        spc_date_string=spc_date_string, raise_error_if_missing=True)

    print('Reading storm locations from: "{0:s}"...'.format(tracking_file_name))
    storm_object_table = tracking_io.read_file(tracking_file_name)

    desired_indices = tracking_utils.find_storm_objects(
        all_id_strings=storm_object_table[
            tracking_utils.FULL_ID_COLUMN].values.tolist(),
        all_times_unix_sec=storm_object_table[
            tracking_utils.VALID_TIME_COLUMN].values,
        id_strings_to_keep=desired_full_id_strings,
        times_to_keep_unix_sec=desired_times_unix_sec, allow_missing=False)

    desired_latitudes_deg = storm_object_table[
        tracking_utils.CENTROID_LATITUDE_COLUMN].values[desired_indices]
    desired_longitudes_deg = storm_object_table[
        tracking_utils.CENTROID_LONGITUDE_COLUMN].values[desired_indices]

    return desired_latitudes_deg, desired_longitudes_deg


def _run(input_file_name, top_tracking_dir_name, min_latitude_deg,
         max_latitude_deg, min_longitude_deg, max_longitude_deg,
         grid_spacing_metres, output_dir_name):
    """Subsets ungridded predictions by space.

    This is effectively the main method.

    :param input_file_name: See documentation at top of file.
    :param top_tracking_dir_name: Same.
    :param min_latitude_deg: Same.
    :param max_latitude_deg: Same.
    :param min_longitude_deg: Same.
    :param max_longitude_deg: Same.
    :param grid_spacing_metres: Same.
    :param output_dir_name: Same.
    """

    equidistant_grid_dict = grids.create_equidistant_grid(
        min_latitude_deg=min_latitude_deg, max_latitude_deg=max_latitude_deg,
        min_longitude_deg=min_longitude_deg,
        max_longitude_deg=max_longitude_deg,
        x_spacing_metres=grid_spacing_metres,
        y_spacing_metres=grid_spacing_metres, azimuthal=False)

    grid_metafile_name = grids.find_equidistant_metafile(
        directory_name=output_dir_name, raise_error_if_missing=False)

    print('Writing metadata for equidistant grid to: "{0:s}"...'.format(
        grid_metafile_name
    ))

    grids.write_equidistant_metafile(grid_dict=equidistant_grid_dict,
                                     pickle_file_name=grid_metafile_name)

    grid_point_x_coords_metres = equidistant_grid_dict[grids.X_COORDS_KEY]
    grid_point_y_coords_metres = equidistant_grid_dict[grids.Y_COORDS_KEY]
    projection_object = equidistant_grid_dict[grids.PROJECTION_KEY]

    grid_edge_x_coords_metres = numpy.append(
        grid_point_x_coords_metres - 0.5 * grid_spacing_metres,
        grid_point_x_coords_metres[-1] + 0.5 * grid_spacing_metres
    )
    grid_edge_y_coords_metres = numpy.append(
        grid_point_y_coords_metres - 0.5 * grid_spacing_metres,
        grid_point_y_coords_metres[-1] + 0.5 * grid_spacing_metres
    )

    print('Reading input data from: "{0:s}"...'.format(input_file_name))
    prediction_dict = prediction_io.read_ungridded_predictions(input_file_name)
    print(SEPARATOR_STRING)

    full_id_strings = prediction_dict[prediction_io.STORM_IDS_KEY]
    storm_times_unix_sec = prediction_dict[prediction_io.STORM_TIMES_KEY]
    unique_storm_times_unix_sec = numpy.unique(storm_times_unix_sec)

    num_storm_objects = len(storm_times_unix_sec)
    storm_latitudes_deg = numpy.full(num_storm_objects, numpy.nan)
    storm_longitudes_deg = numpy.full(num_storm_objects, numpy.nan)

    for this_time_unix_sec in unique_storm_times_unix_sec:
        these_indices = numpy.where(
            storm_times_unix_sec == this_time_unix_sec
        )[0]
        these_full_id_strings = [full_id_strings[k] for k in these_indices]

        (storm_latitudes_deg[these_indices],
         storm_longitudes_deg[these_indices]
        ) = _read_storm_locations_one_time(
            top_tracking_dir_name=top_tracking_dir_name,
            valid_time_unix_sec=this_time_unix_sec,
            desired_full_id_strings=these_full_id_strings)

    print(SEPARATOR_STRING)

    storm_x_coords_metres, storm_y_coords_metres = (
        projections.project_latlng_to_xy(
            latitudes_deg=storm_latitudes_deg,
            longitudes_deg=storm_longitudes_deg,
            projection_object=projection_object)
    )

    num_grid_rows = len(grid_point_x_coords_metres)
    num_grid_columns = len(grid_point_y_coords_metres)

    for i in range(num_grid_rows):
        for j in range(num_grid_columns):
            these_indices = grids.find_events_in_grid_cell(
                event_x_coords_metres=storm_x_coords_metres,
                event_y_coords_metres=storm_y_coords_metres,
                grid_edge_x_coords_metres=grid_edge_x_coords_metres,
                grid_edge_y_coords_metres=grid_edge_y_coords_metres,
                row_index=i, column_index=j, verbose=True)

            this_prediction_dict = prediction_io.subset_ungridded_predictions(
                prediction_dict=prediction_dict,
                desired_storm_indices=these_indices)

            this_output_file_name = prediction_io.find_ungridded_file(
                directory_name=output_dir_name, grid_row=i, grid_column=j,
                raise_error_if_missing=False)

            print('Writing subset to: "{0:s}"...'.format(this_output_file_name))

            prediction_io.write_ungridded_predictions(
                netcdf_file_name=this_output_file_name,
                class_probability_matrix=this_prediction_dict[
                    prediction_io.PROBABILITY_MATRIX_KEY],
                storm_ids=this_prediction_dict[prediction_io.STORM_IDS_KEY],
                storm_times_unix_sec=this_prediction_dict[
                    prediction_io.STORM_TIMES_KEY],
                target_name=this_prediction_dict[prediction_io.TARGET_NAME_KEY],
                observed_labels=this_prediction_dict[
                    prediction_io.OBSERVED_LABELS_KEY]
            )

            print('\n')


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        input_file_name=getattr(INPUT_ARG_OBJECT, INPUT_FILE_ARG_NAME),
        top_tracking_dir_name=getattr(INPUT_ARG_OBJECT, TRACKING_DIR_ARG_NAME),
        min_latitude_deg=getattr(INPUT_ARG_OBJECT, MIN_LATITUDE_ARG_NAME),
        max_latitude_deg=getattr(INPUT_ARG_OBJECT, MAX_LATITUDE_ARG_NAME),
        min_longitude_deg=getattr(INPUT_ARG_OBJECT, MIN_LONGITUDE_ARG_NAME),
        max_longitude_deg=getattr(INPUT_ARG_OBJECT, MAX_LONGITUDE_ARG_NAME),
        grid_spacing_metres=getattr(INPUT_ARG_OBJECT, GRID_SPACING_ARG_NAME),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
