"""Runs human novelty detection on class-activation maps."""

import argparse
import numpy
from gewittergefahr.gg_utils import human_polygons
from gewittergefahr.gg_utils import storm_tracking_utils as tracking_utils
from gewittergefahr.deep_learning import gradcam

INPUT_GRADCAM_FILE_ARG_NAME = 'input_gradcam_file_name'
HUMAN_FILE_ARG_NAME = 'input_human_file_name'
OUTPUT_GRADCAM_FILE_ARG_NAME = 'output_gradcam_file_name'

INPUT_GRADCAM_FILE_HELP_STRING = (
    'Path to file with Grad-CAM output from machine (will be read by '
    '`gradcam.read_standard_file` or `gradcam.read_pmm_file`).  Class-'
    'activation maps therein will be compared with human points of interest in '
    '`{0:s}`.'
).format(HUMAN_FILE_ARG_NAME)

HUMAN_FILE_HELP_STRING = (
    'Path to file with human points of interest (will be read by '
    '`human_polygons.read_points`).')

OUTPUT_GRADCAM_FILE_HELP_STRING = (
    'Path to output file (will be written by `gradcam.read_standard_file` or '
    '`gradcam.read_pmm_file`).  Will be same as input file, except that only '
    'novel regions of interest (those in which the human clicked) will be '
    'retained.')

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + INPUT_GRADCAM_FILE_ARG_NAME, type=str, required=True,
    help=INPUT_GRADCAM_FILE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + HUMAN_FILE_ARG_NAME, type=str, required=True,
    help=HUMAN_FILE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_GRADCAM_FILE_ARG_NAME, type=str, required=True,
    help=OUTPUT_GRADCAM_FILE_HELP_STRING)


def _run(input_gradcam_file_name, input_human_file_name,
         output_gradcam_file_name):
    """Runs human novelty detection on class-activation maps.
    
    This is effectively the main method.
    
    :param input_gradcam_file_name: See documentation at top of file.
    :param input_human_file_name: Same.
    :param output_gradcam_file_name: Same.
    :raises: TypeError: if `input_gradcam_file_name` was created by a net that
        does both 2-D and 3-D convolution.
    :raises: TypeError: if class-activation maps are not 2-D.
    """

    print('Reading data from: "{0:s}"...'.format(input_human_file_name))
    human_point_dict = human_polygons.read_points(input_human_file_name)
    
    # TODO(thunderhoser): Deal with no human points.
    human_grid_rows = human_point_dict[human_polygons.GRID_ROW_BY_POINT_KEY]
    human_grid_columns = human_point_dict[
        human_polygons.GRID_COLUMN_BY_POINT_KEY]
    human_panel_rows = human_point_dict[human_polygons.PANEL_ROW_BY_POINT_KEY]
    human_panel_columns = human_point_dict[
        human_polygons.PANEL_COLUMN_BY_POINT_KEY]

    full_storm_id_string = human_point_dict[human_polygons.STORM_ID_KEY]
    storm_time_unix_sec = human_point_dict[human_polygons.STORM_TIME_KEY]
    pmm_flag = full_storm_id_string is None and storm_time_unix_sec is None

    print('Reading data from: "{0:s}"...'.format(input_gradcam_file_name))

    if pmm_flag:
        gradcam_dict = gradcam.read_pmm_file(input_gradcam_file_name)
    else:
        gradcam_dict = gradcam.read_standard_file(input_gradcam_file_name)

    machine_region_dict = gradcam_dict[gradcam.REGION_DICT_KEY]
    list_of_mask_matrices = machine_region_dict[gradcam.MASK_MATRICES_KEY]

    if len(list_of_mask_matrices) == 3:
        raise TypeError('This script cannot handle nets that do both 2-D and '
                        '3-D convolution.')

    machine_mask_matrix = list_of_mask_matrices[0]

    if pmm_flag:
        storm_object_index = -1
    else:
        storm_object_index = tracking_utils.find_storm_objects(
            all_id_strings=gradcam_dict[gradcam.FULL_IDS_KEY],
            all_times_unix_sec=gradcam_dict[gradcam.STORM_TIMES_KEY],
            id_strings_to_keep=[full_storm_id_string],
            times_to_keep_unix_sec=numpy.array(
                [storm_time_unix_sec], dtype=int
            ),
            allow_missing=False
        )[0]

        machine_mask_matrix = machine_mask_matrix[storm_object_index, ...]

    num_spatial_dimensions = len(machine_mask_matrix.shape) - 1
    if num_spatial_dimensions != 2:
        raise TypeError('This script can compare only with 2-D class-activation'
                        ' maps.')

    if pmm_flag:
        machine_polygon_objects = machine_region_dict[
            gradcam.POLYGON_OBJECTS_KEY][0][0]
    else:
        machine_polygon_objects = machine_region_dict[
            gradcam.POLYGON_OBJECTS_KEY][storm_object_index][0]

    # TODO(thunderhoser): Compare with human points of interest for each
    #  channel.  This will take the form of a PIP test for each channel.


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        input_gradcam_file_name=getattr(
            INPUT_ARG_OBJECT, INPUT_GRADCAM_FILE_ARG_NAME),
        input_human_file_name=getattr(INPUT_ARG_OBJECT, HUMAN_FILE_ARG_NAME),
        output_gradcam_file_name=getattr(
            INPUT_ARG_OBJECT, OUTPUT_GRADCAM_FILE_ARG_NAME)
    )
