"""Wrapper methods for ImageMagick."""

import os
import numpy
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.gg_utils import error_checking

ERROR_STRING = (
    '\nUnix command failed (log messages shown above should explain why).')

DEFAULT_CONVERT_EXE_NAME = '/usr/bin/convert'
DEFAULT_MONTAGE_EXE_NAME = '/usr/bin/montage'


def trim_whitespace(
        input_file_name, output_file_name, border_width_pixels=10,
        convert_exe_name=DEFAULT_CONVERT_EXE_NAME):
    """Trims whitespace around edge of image.

    :param input_file_name: Path to input file (may be in any format handled by
        ImageMagick).
    :param output_file_name: Path to output file.
    :param border_width_pixels: Desired border width (whitespace).
    :param convert_exe_name: Path to executable file for ImageMagick's "convert"
        function.  If you installed ImageMagick with root access, this should be
        the default.  Regardless, the pathless file name should be just
        "convert".
    :raises: ValueError: if ImageMagick command (which is ultimately a Unix
        command) fails.
    """

    error_checking.assert_file_exists(input_file_name)
    file_system_utils.mkdir_recursive_if_necessary(file_name=output_file_name)
    error_checking.assert_is_integer(border_width_pixels)
    error_checking.assert_is_geq(border_width_pixels, 0)
    error_checking.assert_file_exists(convert_exe_name)

    command_string = (
        '"{0:s}" "{1:s}" -trim -bordercolor White -border {2:d} "{3:s}"'
    ).format(
        convert_exe_name, input_file_name, border_width_pixels, output_file_name
    )

    exit_code = os.system(command_string)
    if exit_code == 0:
        return
    raise ValueError(ERROR_STRING)


def concatenate_images(
        input_file_names, output_file_name, num_panel_rows, num_panel_columns,
        border_width_pixels=50, montage_exe_name=DEFAULT_MONTAGE_EXE_NAME):
    """Concatenates many images into one paneled image.

    :param input_file_names: 1-D list of paths to input files (may be in any
        format handled by ImageMagick).
    :param output_file_name: Path to output file.
    :param num_panel_rows: Number of rows in paneled image.
    :param num_panel_columns: Number of columns in paneled image.
    :param border_width_pixels: Border width (whitespace) around each pixel.
    :param montage_exe_name: Path to executable file for ImageMagick's "montage"
        function.  If you installed ImageMagick with root access, this should be
        the default.  Regardless, the pathless file name should be just
        "montage".
    :raises: ValueError: if ImageMagick command (which is ultimately a Unix
        command) fails.
    """

    error_checking.assert_is_numpy_array(
        numpy.array(input_file_names), num_dimensions=1)
    for this_file_name in input_file_names:
        error_checking.assert_file_exists(this_file_name)

    file_system_utils.mkdir_recursive_if_necessary(file_name=output_file_name)
    error_checking.assert_is_integer(num_panel_rows)
    error_checking.assert_is_integer(num_panel_columns)
    error_checking.assert_is_integer(border_width_pixels)
    error_checking.assert_is_geq(border_width_pixels, 0)
    error_checking.assert_file_exists(montage_exe_name)

    num_panels = num_panel_rows * num_panel_columns
    error_checking.assert_is_geq(num_panels, len(input_file_names))

    command_string = '"{0:s}" -mode concatenate -tile {1:d}x{2:d}'.format(
        montage_exe_name, num_panel_columns, num_panel_rows)
    for this_file_name in input_file_names:
        command_string += ' "{0:s}"'.format(this_file_name)

    command_string += ' -trim -bordercolor White -border {0:d} "{1:s}"'.format(
        border_width_pixels, output_file_name)

    exit_code = os.system(command_string)
    if exit_code == 0:
        return
    raise ValueError(ERROR_STRING)


def resize_image(input_file_name, output_file_name, output_size_pixels,
                 convert_exe_name=DEFAULT_CONVERT_EXE_NAME):
    """Resizes image.

    :param input_file_name: Path to input file (may be in any format handled by
        ImageMagick).
    :param output_file_name: Path to output file.
    :param output_size_pixels: Output size.
    :param convert_exe_name: See doc for `trim_whitespace`.
    :raises: ValueError: if ImageMagick command (which is ultimately a Unix
        command) fails.
    """

    error_checking.assert_file_exists(input_file_name)
    file_system_utils.mkdir_recursive_if_necessary(file_name=output_file_name)
    error_checking.assert_is_integer(output_size_pixels)
    error_checking.assert_is_greater(output_size_pixels, 0)
    error_checking.assert_file_exists(convert_exe_name)

    command_string = '"{0:s}" "{1:s}" -resize {2:d}@ "{3:s}"'.format(
        convert_exe_name, input_file_name, output_size_pixels, output_file_name)

    exit_code = os.system(command_string)
    if exit_code == 0:
        return
    raise ValueError(ERROR_STRING)
