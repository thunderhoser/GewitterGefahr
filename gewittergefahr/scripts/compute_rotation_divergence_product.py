"""Computes rotation-divergence product for each storm object.

The "rotation-divergence product," with units of s^-2, is defined in Sandmael et
al. (2018).

--- REFERENCES ---

Sandmael, T., C.R. Homeyer, K.M. Bedka, J.M. Apke, J.R. Mecikalski, and
K. Khlopenkov, 2018: Using remotely sensed updraft characteristics to
discriminate between tornadic and non-tornadic storms. American Meteorological
Society (I don't know which journal), under review.
"""

import pickle
import argparse
import numpy
from gewittergefahr.gg_io import storm_tracking_io as tracking_io
from gewittergefahr.gg_io import gridrad_io
from gewittergefahr.gg_utils import storm_tracking_utils as tracking_utils
from gewittergefahr.gg_utils import radar_utils
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import file_system_utils

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'
TIME_FORMAT_FOR_LOG_MESSAGES = '%Y-%m-%d-%H%M%S'

DUMMY_TRACKING_SCALE_METRES2 = 314159265
MIN_VORTICITY_HEIGHT_M_ASL = 4000

STORM_IDS_KEY = 'storm_ids'
STORM_TIMES_KEY = 'storm_times_unix_sec'
ROTATION_DIVERGENCE_PRODUCTS_KEY = 'rotation_divergence_products_s02'

TRACKING_DIR_ARG_NAME = 'input_tracking_dir_name'
GRIDRAD_DIR_ARG_NAME = 'input_gridrad_dir_name'
SPC_DATE_ARG_NAME = 'spc_date_string'
OUTPUT_FILE_ARG_NAME = 'output_pickle_file_name'

TRACKING_DIR_HELP_STRING = (
    'Name of top-level directory with storm tracks (files will be found by '
    '`storm_tracking_io.find_processed_file`).  The rotation-divergence product'
    ' will be computed for each storm object on SPC date `{0:s}`.'
).format(SPC_DATE_ARG_NAME)
GRIDRAD_DIR_HELP_STRING = (
    'Name of top-level directory with native GridRad data (files will be found '
    'by `gridrad_io.find_file`).')
SPC_DATE_HELP_STRING = (
    'SPC date (format "yyyymmdd").  The rotation-divergence product will be '
    'computed for each storm object on this date.')
OUTPUT_FILE_HELP_STRING = (
    'Path to output file (with valid time, storm ID, and rotation-divergence '
    'product for each storm object).')

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + TRACKING_DIR_ARG_NAME, type=str, required=True,
    help=TRACKING_DIR_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + GRIDRAD_DIR_ARG_NAME, type=str, required=True,
    help=GRIDRAD_DIR_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + SPC_DATE_ARG_NAME, type=str, required=True,
    help=SPC_DATE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_FILE_ARG_NAME, type=str, required=True,
    help=OUTPUT_FILE_HELP_STRING)


def _compute_rdp_for_each_storm_object(
        top_tracking_dir_name, top_gridrad_dir_name, spc_date_string,
        output_pickle_file_name):
    """Computes rotation-divergence product for each storm object.

    :param top_tracking_dir_name: See documentation at top of file.
    :param top_gridrad_dir_name: Same.
    :param spc_date_string: Same.
    :param output_pickle_file_name: Same.
    """

    file_system_utils.mkdir_recursive_if_necessary(
        file_name=output_pickle_file_name)

    tracking_file_names, _ = tracking_io.find_processed_files_one_spc_date(
        top_processed_dir_name=top_tracking_dir_name,
        tracking_scale_metres2=DUMMY_TRACKING_SCALE_METRES2,
        data_source=tracking_utils.SEGMOTION_SOURCE_ID,
        spc_date_string=spc_date_string)

    storm_object_table = tracking_io.read_many_processed_files(
        tracking_file_names)
    print SEPARATOR_STRING

    num_storm_objects = len(storm_object_table.index)
    rdp_by_storm_object_s02 = numpy.full(num_storm_objects, numpy.nan)
    unique_storm_times_unix_sec = numpy.unique(
        storm_object_table[tracking_utils.TIME_COLUMN].values)

    for this_time_unix_sec in unique_storm_times_unix_sec:
        this_gridrad_file_name = gridrad_io.find_file(
            top_directory_name=top_gridrad_dir_name,
            unix_time_sec=this_time_unix_sec)
        this_metadata_dict = gridrad_io.read_metadata_from_full_grid_file(
            netcdf_file_name=this_gridrad_file_name)

        print 'Reading "{0:s}" field from: "{1:s}"...'.format(
            radar_utils.VORTICITY_NAME, this_gridrad_file_name)
        (this_vorticity_matrix_s01, these_grid_point_heights_m_asl, _, _
        ) = gridrad_io.read_field_from_full_grid_file(
            netcdf_file_name=this_gridrad_file_name,
            field_name=radar_utils.VORTICITY_NAME,
            metadata_dict=this_metadata_dict)

        print 'Removing heights < {0:d} m ASL from "{1:s}" data...'.format(
            MIN_VORTICITY_HEIGHT_M_ASL, radar_utils.VORTICITY_NAME)
        these_valid_height_indices = numpy.where(
            these_grid_point_heights_m_asl >= MIN_VORTICITY_HEIGHT_M_ASL)[0]
        this_vorticity_matrix_s01 = this_vorticity_matrix_s01[
            these_valid_height_indices, ...]
        del these_grid_point_heights_m_asl

        print 'Reading "{0:s}" field from: "{1:s}"...'.format(
            radar_utils.DIVERGENCE_NAME, this_gridrad_file_name)
        (this_divgerence_matrix_s01, _, _, _
        ) = gridrad_io.read_field_from_full_grid_file(
            netcdf_file_name=this_gridrad_file_name,
            field_name=radar_utils.DIVERGENCE_NAME,
            metadata_dict=this_metadata_dict)

        this_time_string = time_conversion.unix_sec_to_string(
            this_time_unix_sec, TIME_FORMAT_FOR_LOG_MESSAGES)
        these_storm_object_indices = numpy.where(
            storm_object_table[tracking_utils.TIME_COLUMN] ==
            this_time_unix_sec)[0]

        print (
            'Computing rotation-divergence product for all storm objects at '
            '{0:s}...\n'
        ).format(this_time_string)

        for this_storm_object_index in these_storm_object_indices:
            these_grid_point_rows = storm_object_table[
                tracking_utils.GRID_POINT_ROW_COLUMN
            ].values[this_storm_object_index]
            these_grid_point_columns = storm_object_table[
                tracking_utils.GRID_POINT_COLUMN_COLUMN
            ].values[this_storm_object_index]

            this_vorticity_s01 = numpy.nanmax(
                this_vorticity_matrix_s01[
                    :, these_grid_point_rows, these_grid_point_columns])
            this_divergence_s01 = numpy.nanmax(
                this_divgerence_matrix_s01[
                    :, these_grid_point_rows, these_grid_point_columns])
            rdp_by_storm_object_s02[this_storm_object_index] = (
                this_vorticity_s01 * this_divergence_s01)

    rdp_dict = {
        STORM_IDS_KEY:
            storm_object_table[tracking_utils.STORM_ID_COLUMN].values.tolist(),
        STORM_TIMES_KEY: storm_object_table[tracking_utils.TIME_COLUMN],
        ROTATION_DIVERGENCE_PRODUCTS_KEY: rdp_by_storm_object_s02
    }

    print (
        'Writing results (storm IDs, times, and RDP values) to: "{0:s}"...'
    ).format(output_pickle_file_name)
    pickle_file_handle = open(output_pickle_file_name, 'wb')
    pickle.dump(rdp_dict, pickle_file_handle)
    pickle_file_handle.close()


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _compute_rdp_for_each_storm_object(
        top_tracking_dir_name=getattr(INPUT_ARG_OBJECT, TRACKING_DIR_ARG_NAME),
        top_gridrad_dir_name=getattr(INPUT_ARG_OBJECT, GRIDRAD_DIR_ARG_NAME),
        spc_date_string=getattr(INPUT_ARG_OBJECT, SPC_DATE_ARG_NAME),
        output_pickle_file_name=getattr(INPUT_ARG_OBJECT, OUTPUT_FILE_ARG_NAME))
