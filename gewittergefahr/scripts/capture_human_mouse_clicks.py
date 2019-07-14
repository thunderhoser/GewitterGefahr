"""Captures mouse clicks from a human over each pre-existing image.

NOTE: This script is interactive and requires an interactive display.  You
cannot run it on a supercomputer without X-forwarding or whatever it's called.
"""

import os.path
import argparse
import numpy
from gewittergefahr.gg_utils import human_polygons
from gewittergefahr.scripts import capture_human_polygons

IMAGE_PATH_ARG_NAME = 'image_dir_or_file_name'
NUM_GRID_ROWS_ARG_NAME = 'num_grid_rows'
NUM_GRID_COLUMNS_ARG_NAME = 'num_grid_columns'
NUM_PANEL_ROWS_ARG_NAME = 'num_panel_rows'
NUM_PANEL_COLUMNS_ARG_NAME = 'num_panel_columns'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

IMAGE_PATH_HELP_STRING = (
    'Path to input file or directory.  This script will allow you to record '
    'mouse clicks over each image.  Each file name must be in the format {0:s}.'
).format(capture_human_polygons.FILE_FORMAT_STRING)

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


def _capture_clicks_one_image(
        image_file_name, num_grid_rows, num_grid_columns, num_panel_rows,
        num_panel_columns, output_file_name):
    """Captures mouse clicks over one image.

    :param image_file_name: Path to image file.
    :param num_grid_rows: See documentation at top of this file.
    :param num_grid_columns: Same.
    :param num_panel_rows: Same.
    :param num_panel_columns: Same.
    :param output_file_name: Same.
    """

    instruction_string = (
        'Polygons = regions that most support tornado production in the next '
        'hour.  Click in regions that you did NOT expect.')

    point_objects_pixel_coords, num_pixel_rows, num_pixel_columns = (
        human_polygons.capture_mouse_clicks(
            image_file_name=image_file_name,
            instruction_string=instruction_string)
    )

    x_coords_px = numpy.array([p.x for p in point_objects_pixel_coords])
    y_coords_px = numpy.array([p.y for p in point_objects_pixel_coords])

    grid_row_by_point, panel_row_by_point = (
        human_polygons.pixel_rows_to_grid_rows(
            pixel_row_by_vertex=y_coords_px, num_pixel_rows=num_pixel_rows,
            num_panel_rows=num_panel_rows, num_grid_rows=num_grid_rows,
            assert_same_panel=False)
    )

    grid_column_by_point, panel_column_by_point = (
        human_polygons.pixel_columns_to_grid_columns(
            pixel_column_by_vertex=x_coords_px,
            num_pixel_columns=num_pixel_columns,
            num_panel_columns=num_panel_columns,
            num_grid_columns=num_grid_columns, assert_same_panel=False)
    )

    print('Writing points of interest to: "{0:s}"...'.format(output_file_name))
    full_storm_id_string, storm_time_unix_sec = (
        capture_human_polygons.check_image_file_name(image_file_name)
    )

    human_polygons.write_points(
        output_file_name=output_file_name,
        full_storm_id_string=full_storm_id_string,
        storm_time_unix_sec=storm_time_unix_sec,
        grid_row_by_point=grid_row_by_point,
        grid_column_by_point=grid_column_by_point,
        panel_row_by_point=panel_row_by_point,
        panel_column_by_point=panel_column_by_point)


def _run(image_dir_or_file_name, num_grid_rows, num_grid_columns,
         num_panel_rows, num_panel_columns, output_dir_name):
    """Captures mouse clicks from a human over each pre-existing image.

    This is effectively the main method.

    :param image_dir_or_file_name: See documentation at top of file.
    :param num_grid_rows: Same.
    :param num_grid_columns: Same.
    :param num_panel_rows: Same.
    :param num_panel_columns: Same.
    :param output_dir_name: Same.
    :raises: ValueError: if `image_dir_or_file_name` is a directory and contains
        no files that match the desired format.
    """

    image_file_names = capture_human_polygons.get_image_files(
        image_dir_or_file_name)

    for this_image_file_name in image_file_names:
        this_pathless_file_name = os.path.split(this_image_file_name)[-1]
        this_bare_file_name = os.path.splitext(this_pathless_file_name)[0]

        this_output_file_name = '{0:s}/{1:s}_human.nc'.format(
            output_dir_name, this_bare_file_name)

        _capture_clicks_one_image(
            image_file_name=this_image_file_name,
            num_grid_rows=num_grid_rows, num_grid_columns=num_grid_columns,
            num_panel_rows=num_panel_rows, num_panel_columns=num_panel_columns,
            output_file_name=this_output_file_name)


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        image_dir_or_file_name=getattr(INPUT_ARG_OBJECT, IMAGE_PATH_ARG_NAME),
        num_grid_rows=getattr(INPUT_ARG_OBJECT, NUM_GRID_ROWS_ARG_NAME),
        num_grid_columns=getattr(INPUT_ARG_OBJECT, NUM_GRID_COLUMNS_ARG_NAME),
        num_panel_rows=getattr(INPUT_ARG_OBJECT, NUM_PANEL_ROWS_ARG_NAME),
        num_panel_columns=getattr(INPUT_ARG_OBJECT, NUM_PANEL_COLUMNS_ARG_NAME),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
