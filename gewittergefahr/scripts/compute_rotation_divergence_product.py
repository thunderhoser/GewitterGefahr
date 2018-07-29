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
from gewittergefahr.gg_io import storm_tracking_io as tracking_io
from gewittergefahr.gg_utils import storm_tracking_utils as tracking_utils
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.deep_learning import storm_images

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

MIN_VORTICITY_HEIGHT_FOR_RDP_M_ASL = 4000
DUMMY_TRACKING_SCALE_METRES2 = 314159265

STORM_IDS_KEY = 'storm_ids'
STORM_TIMES_KEY = 'storm_times_unix_sec'
ROTATION_DIVERGENCE_PRODUCTS_KEY = 'rotation_divergence_products_s02'

TRACKING_DIR_ARG_NAME = 'input_tracking_dir_name'
GRIDRAD_DIR_ARG_NAME = 'input_gridrad_dir_name'
SPC_DATE_ARG_NAME = 'spc_date_string'
RADIUS_ARG_NAME = 'horizontal_radius_metres'
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
RADIUS_HELP_STRING = (
    'For each storm object, the rotation-divergence product (RDP) will be max '
    'rotation * max divergence within this horizontal radius from the storm '
    'centroid.')
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
    '--' + RADIUS_ARG_NAME, type=int, required=False, default=10000,
    help=RADIUS_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_FILE_ARG_NAME, type=str, required=True,
    help=OUTPUT_FILE_HELP_STRING)


def _run(
        top_tracking_dir_name, top_gridrad_dir_name, spc_date_string,
        horizontal_radius_metres, output_pickle_file_name):
    """Computes rotation-divergence product for each storm object.

    This is effectively the main method.

    :param top_tracking_dir_name: See documentation at top of file.
    :param top_gridrad_dir_name: Same.
    :param spc_date_string: Same.
    :param horizontal_radius_metres: Same.
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

    rotation_divergence_products_s02 = (
        storm_images.get_max_rdp_for_each_storm_object(
            storm_object_table=storm_object_table,
            top_gridrad_dir_name=top_gridrad_dir_name,
            horizontal_radius_metres=horizontal_radius_metres,
            min_vorticity_height_m_asl=MIN_VORTICITY_HEIGHT_FOR_RDP_M_ASL))
    print SEPARATOR_STRING

    rdp_dict = {
        STORM_IDS_KEY:
            storm_object_table[tracking_utils.STORM_ID_COLUMN].values.tolist(),
        STORM_TIMES_KEY: storm_object_table[tracking_utils.TIME_COLUMN].values,
        ROTATION_DIVERGENCE_PRODUCTS_KEY: rotation_divergence_products_s02
    }

    print (
        'Writing results (storm IDs, times, and RDP values) to: "{0:s}"...'
    ).format(output_pickle_file_name)
    pickle_file_handle = open(output_pickle_file_name, 'wb')
    pickle.dump(rdp_dict, pickle_file_handle)
    pickle_file_handle.close()


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        top_tracking_dir_name=getattr(INPUT_ARG_OBJECT, TRACKING_DIR_ARG_NAME),
        top_gridrad_dir_name=getattr(INPUT_ARG_OBJECT, GRIDRAD_DIR_ARG_NAME),
        spc_date_string=getattr(INPUT_ARG_OBJECT, SPC_DATE_ARG_NAME),
        horizontal_radius_metres=getattr(INPUT_ARG_OBJECT, RADIUS_ARG_NAME),
        output_pickle_file_name=getattr(INPUT_ARG_OBJECT, OUTPUT_FILE_ARG_NAME))
