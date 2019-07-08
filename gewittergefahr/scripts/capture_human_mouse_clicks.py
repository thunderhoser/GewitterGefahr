"""Captures mouse clicks from a human over a pre-existing image.

NOTE: This script is interactive and requires an interactive display.  You
cannot run it on a supercomputer without X-forwarding or whatever it's called.
"""

import argparse
import numpy
from gewittergefahr.gg_utils import human_polygons

IMAGE_FILE_ARG_NAME = 'input_image_file_name'
NUM_GRID_ROWS_ARG_NAME = 'num_grid_rows'
NUM_GRID_COLUMNS_ARG_NAME = 'num_grid_columns'
NUM_PANEL_ROWS_ARG_NAME = 'num_panel_rows'
NUM_PANEL_COLUMNS_ARG_NAME = 'num_panel_columns'
OUTPUT_FILE_ARG_NAME = 'output_file_name'

IMAGE_FILE_HELP_STRING = (
    'Path to input file.  Mouse clicks will be recorded over this image.')

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

OUTPUT_FILE_HELP_STRING = (
    'Path to output file.  Will be written by `human_polygons.write_polygons`.')

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + IMAGE_FILE_ARG_NAME, type=str, required=True,
    help=IMAGE_FILE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + NUM_GRID_ROWS_ARG_NAME, type=int, required=True,
    help=NUM_GRID_ROWS_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + NUM_GRID_COLUMNS_ARG_NAME, type=int, required=True,
    help=NUM_GRID_COLUMNS_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + NUM_PANEL_ROWS_ARG_NAME, type=int, required=True,
    help=NUM_PANEL_ROWS_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + NUM_PANEL_COLUMNS_ARG_NAME, type=int, required=True,
    help=NUM_PANEL_COLUMNS_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_FILE_ARG_NAME, type=str, required=True,
    help=OUTPUT_FILE_HELP_STRING)


def _run(input_image_file_name, num_grid_rows, num_grid_columns, num_panel_rows,
         num_panel_columns, output_file_name):
    """Captures mouse clicks from a human over a pre-existing image.

    This is effectively the main method.

    :param input_image_file_name: See documentation at top of file.
    :param num_grid_rows: Same.
    :param num_grid_columns: Same.
    :param num_panel_rows: Same.
    :param num_panel_columns: Same.
    :param output_file_name: Same.
    """

    instruction_string = (
        'Click in polygons (areas of interest) that you did not expect.  Close '
        'when you are done.')

    point_objects_pixel_coords, num_pixel_rows, num_pixel_columns = (
        human_polygons.capture_mouse_clicks(
            image_file_name=input_image_file_name,
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

    human_polygons.write_points(
        output_file_name=output_file_name,
        orig_image_file_name=input_image_file_name,
        grid_row_by_point=grid_row_by_point,
        grid_column_by_point=grid_column_by_point,
        panel_row_by_point=panel_row_by_point,
        panel_column_by_point=panel_column_by_point)


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        input_image_file_name=getattr(INPUT_ARG_OBJECT, IMAGE_FILE_ARG_NAME),
        num_grid_rows=getattr(INPUT_ARG_OBJECT, NUM_GRID_ROWS_ARG_NAME),
        num_grid_columns=getattr(INPUT_ARG_OBJECT, NUM_GRID_COLUMNS_ARG_NAME),
        num_panel_rows=getattr(INPUT_ARG_OBJECT, NUM_PANEL_ROWS_ARG_NAME),
        num_panel_columns=getattr(INPUT_ARG_OBJECT, NUM_PANEL_COLUMNS_ARG_NAME),
        output_file_name=getattr(INPUT_ARG_OBJECT, OUTPUT_FILE_ARG_NAME)
    )
