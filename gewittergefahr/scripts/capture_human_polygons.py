"""Captures polygons drawn by a human over each pre-existing image.

NOTE: This script is interactive and requires an interactive display.  You
cannot run it on a supercomputer without X-forwarding or whatever it's called.
"""

import glob
import os.path
import warnings
import argparse
from gewittergefahr.gg_utils import human_polygons
from gewittergefahr.gg_utils import error_checking

FILE_FORMAT_STRING = (
    '"storm=foo_time=bar", where "foo" is an ID with no underscores '
    'and "time" is the Unix time in seconds')

IMAGE_PATH_ARG_NAME = 'image_dir_or_file_name'
POS_NEG_ARG_NAME = 'positive_and_negative'
NUM_GRID_ROWS_ARG_NAME = 'num_grid_rows'
NUM_GRID_COLUMNS_ARG_NAME = 'num_grid_columns'
NUM_PANEL_ROWS_ARG_NAME = 'num_panel_rows'
NUM_PANEL_COLUMNS_ARG_NAME = 'num_panel_columns'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

IMAGE_PATH_HELP_STRING = (
    'Path to input file or directory.  This script will allow you to draw '
    'polygons over each image.  Each file name must be in the format {0:s}.'
).format(FILE_FORMAT_STRING)

POS_NEG_HELP_STRING = (
    'Boolean flag.  If 1, will capture positive and negative polygons (regions '
    'where the quantity of interest is strongly positive and negative, '
    'respectively).  If 0, will capture only positive polygons.')

NUM_GRID_ROWS_HELP_STRING = (
    'Number of rows in grid.  This method assumes that the image contains one '
    'or more panels with gridded data.')

NUM_GRID_COLUMNS_HELP_STRING = (
    'Number of columns in grid.  This method assumes that the image contains '
    'one or more panels with gridded data.')

NUM_PANEL_ROWS_HELP_STRING = (
    'Number of panel rows in image.  Each panel may contain a different '
    'variable, but they must all contain the same grid, with the same aspect '
    'ratio and no whitespace border (between the panels or around the outside '
    'of the image).')

NUM_PANEL_COLUMNS_HELP_STRING = 'Number of panel columns in image.'

OUTPUT_DIR_HELP_STRING = (
    'Name of output directory.  Files will be saved here by '
    '`human_polygons.write_polygons`.')

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '-i', '--' + IMAGE_PATH_ARG_NAME, type=str, required=True,
    help=IMAGE_PATH_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '-posandneg', '--' + POS_NEG_ARG_NAME, type=int, required=False, default=0,
    help=POS_NEG_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '-ngridrows', '--' + NUM_GRID_ROWS_ARG_NAME, type=int, required=False,
    default=32, help=NUM_GRID_ROWS_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '-ngridcols', '--' + NUM_GRID_COLUMNS_ARG_NAME, type=int, required=False,
    default=32, help=NUM_GRID_COLUMNS_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '-npanelrows', '--' + NUM_PANEL_ROWS_ARG_NAME, type=int, required=False,
    default=1, help=NUM_PANEL_ROWS_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '-npanelcols', '--' + NUM_PANEL_COLUMNS_ARG_NAME, type=int, required=False,
    default=2, help=NUM_PANEL_COLUMNS_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '-o', '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING)


def _capture_polygons_one_image(
        image_file_name, positive_and_negative, num_grid_rows, num_grid_columns,
        num_panel_rows, num_panel_columns, output_file_name):
    """Captures human polygons for one image.

    :param image_file_name: Path to image file.
    :param positive_and_negative: See documentation at top of file.
    :param num_grid_rows: Same.
    :param num_grid_columns: Same.
    :param num_panel_rows: Same.
    :param num_panel_columns: Same.
    :param output_file_name: Same.
    """

    # instruction_string = (
    #     'Outline POSITIVE regions of interest.  Left-click for new vertex, '
    #     'right-click to close polygon.')

    instruction_string = (
        'Outline regions that most support tornado production in the next hour.'
        '  LEFT-CLICK for new vertex; RIGHT-CLICK to close polygon.')

    positive_objects_pixel_coords, num_pixel_rows, num_pixel_columns = (
        human_polygons.capture_polygons(
            image_file_name=image_file_name,
            instruction_string=instruction_string)
    )

    (positive_objects_grid_coords, positive_panel_row_by_polygon,
     positive_panel_column_by_polygon
    ) = human_polygons.polygons_from_pixel_to_grid_coords(
        polygon_objects_pixel_coords=positive_objects_pixel_coords,
        num_grid_rows=num_grid_rows, num_grid_columns=num_grid_columns,
        num_pixel_rows=num_pixel_rows, num_pixel_columns=num_pixel_columns,
        num_panel_rows=num_panel_rows, num_panel_columns=num_panel_columns)

    positive_mask_matrix = human_polygons.polygons_to_mask(
        polygon_objects_grid_coords=positive_objects_grid_coords,
        num_grid_rows=num_grid_rows, num_grid_columns=num_grid_columns,
        num_panel_rows=num_panel_rows, num_panel_columns=num_panel_columns,
        panel_row_by_polygon=positive_panel_row_by_polygon,
        panel_column_by_polygon=positive_panel_column_by_polygon)

    if positive_and_negative:
        instruction_string = (
            'Outline NEGATIVE regions of interest.  Left-click for new vertex, '
            'right-click to close polygon.')

        negative_objects_pixel_coords, num_pixel_rows, num_pixel_columns = (
            human_polygons.capture_polygons(
                image_file_name=image_file_name,
                instruction_string=instruction_string)
        )

        (negative_objects_grid_coords, negative_panel_row_by_polygon,
         negative_panel_column_by_polygon
        ) = human_polygons.polygons_from_pixel_to_grid_coords(
            polygon_objects_pixel_coords=negative_objects_pixel_coords,
            num_grid_rows=num_grid_rows, num_grid_columns=num_grid_columns,
            num_pixel_rows=num_pixel_rows, num_pixel_columns=num_pixel_columns,
            num_panel_rows=num_panel_rows, num_panel_columns=num_panel_columns)

        negative_mask_matrix = human_polygons.polygons_to_mask(
            polygon_objects_grid_coords=negative_objects_grid_coords,
            num_grid_rows=num_grid_rows, num_grid_columns=num_grid_columns,
            num_panel_rows=num_panel_rows, num_panel_columns=num_panel_columns,
            panel_row_by_polygon=negative_panel_row_by_polygon,
            panel_column_by_polygon=negative_panel_column_by_polygon)
    else:
        negative_objects_grid_coords = None
        negative_panel_row_by_polygon = None
        negative_panel_column_by_polygon = None
        negative_mask_matrix = None

    print('Writing polygons and masks to: "{0:s}"...'.format(output_file_name))
    full_storm_id_string, storm_time_unix_sec = check_image_file_name(
        image_file_name)

    human_polygons.write_polygons(
        output_file_name=output_file_name,
        full_storm_id_string=full_storm_id_string,
        storm_time_unix_sec=storm_time_unix_sec,
        positive_objects_grid_coords=positive_objects_grid_coords,
        positive_panel_row_by_polygon=positive_panel_row_by_polygon,
        positive_panel_column_by_polygon=positive_panel_column_by_polygon,
        positive_mask_matrix=positive_mask_matrix,
        negative_objects_grid_coords=negative_objects_grid_coords,
        negative_panel_row_by_polygon=negative_panel_row_by_polygon,
        negative_panel_column_by_polygon=negative_panel_column_by_polygon,
        negative_mask_matrix=negative_mask_matrix)


def _run(image_dir_or_file_name, positive_and_negative, num_grid_rows,
         num_grid_columns, num_panel_rows, num_panel_columns, output_dir_name):
    """Captures polygons drawn by a human over each pre-existing image.

    This is effectively the main method.

    :param image_dir_or_file_name: See documentation at top of file.
    :param positive_and_negative: Same.
    :param num_grid_rows: Same.
    :param num_grid_columns: Same.
    :param num_panel_rows: Same.
    :param num_panel_columns: Same.
    :param output_dir_name: Same.
    """

    image_file_names = get_image_files(image_dir_or_file_name)

    for this_image_file_name in image_file_names:
        this_pathless_file_name = os.path.split(this_image_file_name)[-1]
        this_bare_file_name = os.path.splitext(this_pathless_file_name)[0]

        this_output_file_name = '{0:s}/{1:s}_human.nc'.format(
            output_dir_name, this_bare_file_name)

        _capture_polygons_one_image(
            image_file_name=this_image_file_name,
            positive_and_negative=positive_and_negative,
            num_grid_rows=num_grid_rows, num_grid_columns=num_grid_columns,
            num_panel_rows=num_panel_rows, num_panel_columns=num_panel_columns,
            output_file_name=this_output_file_name)


def check_image_file_name(image_file_name):
    """Error-checks name of image file.

    If the image file contains a composite rather than one storm, both output
    variables will be None.

    :param image_file_name: Path to image file.
    :return: full_storm_id_string: Full storm ID.
    :return: storm_time_unix_sec: Valid time.
    :raises: ValueError: if file name is not in the desired format.
    """

    # TODO(thunderhoser): Put this method somewhere more general.

    pathless_image_file_name = os.path.split(image_file_name)[-1]
    bare_image_file_name = os.path.splitext(pathless_image_file_name)[0]
    bare_file_name_parts = bare_image_file_name.split('_')

    error_string = (
        'File name ("{0:s}") is not in desired format ({1:s}).'
    ).format(image_file_name, FILE_FORMAT_STRING)

    if len(bare_file_name_parts) != 2:
        raise ValueError(error_string)

    storm_id_part = bare_file_name_parts[0]
    time_part = bare_file_name_parts[1]

    if not storm_id_part.startswith('storm='):
        raise ValueError(error_string)
    if not time_part.startswith('time='):
        raise ValueError(error_string)

    full_storm_id_string = storm_id_part.replace('storm=', '')
    if full_storm_id_string == 'pmm':
        return None, None

    storm_time_unix_sec = int(time_part.replace('time=', ''))

    return full_storm_id_string, storm_time_unix_sec


def get_image_files(image_dir_or_file_name):
    """Finds image files.

    :param image_dir_or_file_name: See documentation at top of file.
    :return: image_file_names: 1-D list of paths to image files.
    :raises: ValueError: if `image_dir_or_file_name` is a directory and contains
        no files that match the desired format.
    """

    error_checking.assert_is_string(image_dir_or_file_name)
    image_file_names = []

    if os.path.isdir(image_dir_or_file_name):
        file_pattern = '{0:s}/storm=*time=*'.format(image_dir_or_file_name)
        all_file_names = glob.glob(file_pattern)
        all_file_names.sort()

        for this_file_name in all_file_names:
            try:
                check_image_file_name(this_file_name)
                image_file_names.append(this_file_name)
            except:
                warning_string = (
                    'Directory "{0:s}" contains a file ("{1:s}") that does not '
                    'match the desired format ({2:s}).'
                ).format(
                    image_dir_or_file_name, os.path.split(this_file_name)[-1],
                    FILE_FORMAT_STRING
                )

                warnings.warn(warning_string)

        if len(image_file_names) == 0:
            error_string = (
                'Directory "{0:s}" contains no files that match the desired '
                'format ({1:s}).'
            ).format(image_dir_or_file_name, FILE_FORMAT_STRING)

            raise ValueError(error_string)
    else:
        check_image_file_name(image_dir_or_file_name)
        image_file_names = [image_dir_or_file_name]

    return image_file_names


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        image_dir_or_file_name=getattr(INPUT_ARG_OBJECT, IMAGE_PATH_ARG_NAME),
        positive_and_negative=bool(getattr(
            INPUT_ARG_OBJECT, POS_NEG_ARG_NAME
        )),
        num_grid_rows=getattr(INPUT_ARG_OBJECT, NUM_GRID_ROWS_ARG_NAME),
        num_grid_columns=getattr(INPUT_ARG_OBJECT, NUM_GRID_COLUMNS_ARG_NAME),
        num_panel_rows=getattr(INPUT_ARG_OBJECT, NUM_PANEL_ROWS_ARG_NAME),
        num_panel_columns=getattr(INPUT_ARG_OBJECT, NUM_PANEL_COLUMNS_ARG_NAME),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
