"""Captures polygons drawn by a human over a pre-existing image.

NOTE: This script is interactive and requires an interactive display.  You
cannot run it on a supercomputer without X-forwarding or whatever it's called.
"""

import argparse
import numpy
from gewittergefahr.gg_utils import human_polygons

IMAGE_FILE_ARG_NAME = 'input_image_file_name'
POS_NEG_ARG_NAME = 'positive_and_negative'
NUM_ROWS_ARG_NAME = 'num_grid_rows'
NUM_COLUMNS_ARG_NAME = 'num_grid_columns'
OUTPUT_FILE_ARG_NAME = 'output_file_name'

IMAGE_FILE_HELP_STRING = (
    'Path to input file.  Polygons will be drawn over this image.')

POS_NEG_HELP_STRING = (
    'Boolean flag.  If 1, will capture positive and negative polygons (regions '
    'where the quantity of interest is strongly positive and negative, '
    'respectively).  If 0, will capture only positive polygons.')

NUM_ROWS_HELP_STRING = (
    'Number of rows in grid.  This script assumes that `{0:s}` contains gridded'
    ' data with no padding around the grid.  Number of rows may be != number of'
    ' pixel rows, so this argument helps convert between pixel and grid coords.'
    '  For a complete list of assumptions, see doc for '
    '`human_polygons.polygons_from_pixel_to_grid_coords`.'
).format(IMAGE_FILE_ARG_NAME)

NUM_COLUMNS_HELP_STRING = 'Number of columns in grid.'

OUTPUT_FILE_HELP_STRING = (
    'Path to output file.  Will be written by `human_polygons.write_polygons`.')

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + IMAGE_FILE_ARG_NAME, type=str, required=True,
    help=IMAGE_FILE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + POS_NEG_ARG_NAME, type=int, required=True, help=POS_NEG_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + NUM_ROWS_ARG_NAME, type=int, required=True,
    help=NUM_ROWS_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + NUM_COLUMNS_ARG_NAME, type=int, required=True,
    help=NUM_COLUMNS_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_FILE_ARG_NAME, type=str, required=True,
    help=OUTPUT_FILE_HELP_STRING)


def _run(input_image_file_name, positive_and_negative, num_grid_rows,
         num_grid_columns, output_file_name):
    """Captures polygons drawn by a human over a pre-existing image.

    This is effectively the main method.

    :param input_image_file_name: See documentation at top of file.
    :param positive_and_negative: Same.
    :param num_grid_rows: Same.
    :param num_grid_columns: Same.
    :param output_file_name: Same.
    """

    positive_polygon_objects_xy, num_pixel_rows, num_pixel_columns = (
        human_polygons.capture_polygons(
            image_file_name=input_image_file_name,
            instruction_string='Outline POSITIVE regions of interest.')
    )

    positive_polygon_objects_rowcol = (
        human_polygons.polygons_from_pixel_to_grid_coords(
            list_of_polygon_objects_xy=positive_polygon_objects_xy,
            num_grid_rows=num_grid_rows, num_grid_columns=num_grid_columns,
            num_pixel_rows=num_pixel_rows, num_pixel_columns=num_pixel_columns)
    )

    positive_mask_matrix = human_polygons.polygons_to_mask(
        list_of_polygon_objects_rowcol=positive_polygon_objects_rowcol,
        num_grid_rows=num_grid_rows, num_grid_columns=num_grid_columns)

    if positive_and_negative:
        negative_polygon_objects_xy, num_pixel_rows, num_pixel_columns = (
            human_polygons.capture_polygons(
                image_file_name=input_image_file_name,
                instruction_string='Outline NEGATIVE regions of interest.')
        )

        negative_polygon_objects_rowcol = (
            human_polygons.polygons_from_pixel_to_grid_coords(
                list_of_polygon_objects_xy=negative_polygon_objects_xy,
                num_grid_rows=num_grid_rows, num_grid_columns=num_grid_columns,
                num_pixel_rows=num_pixel_rows,
                num_pixel_columns=num_pixel_columns)
        )

        negative_mask_matrix = human_polygons.polygons_to_mask(
            list_of_polygon_objects_rowcol=negative_polygon_objects_rowcol,
            num_grid_rows=num_grid_rows, num_grid_columns=num_grid_columns)
    else:
        negative_polygon_objects_rowcol = []
        negative_mask_matrix = (
            numpy.zeros(positive_mask_matrix.shape).astype(bool)
        )

    print('Writing polygons and masks to: "{0:s}"...'.format(output_file_name))

    human_polygons.write_polygons(
        output_file_name=output_file_name,
        orig_image_file_name=input_image_file_name,
        positive_polygon_objects_rowcol=positive_polygon_objects_rowcol,
        positive_mask_matrix=positive_mask_matrix,
        negative_polygon_objects_rowcol=negative_polygon_objects_rowcol,
        negative_mask_matrix=negative_mask_matrix)


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        input_image_file_name=getattr(INPUT_ARG_OBJECT, IMAGE_FILE_ARG_NAME),
        positive_and_negative=bool(getattr(
            INPUT_ARG_OBJECT, POS_NEG_ARG_NAME
        )),
        num_grid_rows=getattr(INPUT_ARG_OBJECT, NUM_ROWS_ARG_NAME),
        num_grid_columns=getattr(INPUT_ARG_OBJECT, NUM_COLUMNS_ARG_NAME),
        output_file_name=getattr(INPUT_ARG_OBJECT, OUTPUT_FILE_ARG_NAME)
    )
