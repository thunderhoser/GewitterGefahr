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
        border_width_pixels=50, montage_exe_name=DEFAULT_MONTAGE_EXE_NAME,
        extra_args_string=None):
    """Concatenates many images into one paneled image.

    :param input_file_names: 1-D list of paths to input files (may be in any
        format handled by ImageMagick).
    :param output_file_name: Path to output file.
    :param num_panel_rows: Number of rows in paneled image.
    :param num_panel_columns: Number of columns in paneled image.
    :param border_width_pixels: Border width (whitespace) around each pixel.
    :param montage_exe_name: Path to executable file for ImageMagick's `montage`
        function.  If you installed ImageMagick with root access, this should be
        the default.  Regardless, the pathless file name should be just
        "montage".
    :param extra_args_string: String with extra args for ImageMagick's `montage`
        function.  This string will be inserted into the command after
        "montage -mode concatenate".  An example is "-gravity south", in which
        case the beginning of the command is
        "montage -mode concatenate -gravity south".
    :raises: ValueError: if ImageMagick command (which is ultimately a Unix
        command) fails.
    """

    if extra_args_string is None:
        extra_args_string = ''
    else:
        error_checking.assert_is_string(extra_args_string)

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

    command_string = '"{0:s}" -mode concatenate {1:s} -tile {2:d}x{3:d}'.format(
        montage_exe_name, extra_args_string, num_panel_columns, num_panel_rows)

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


def create_gif(input_file_names, output_file_name, num_seconds_per_frame,
               resize_factor=0.5, convert_exe_name=DEFAULT_CONVERT_EXE_NAME):
    """Creates GIF from static images.

    :param input_file_names: 1-D list of paths to input files (static images).
    :param output_file_name: Path to output file (GIF).
    :param num_seconds_per_frame: Number of seconds per frame.
    :param resize_factor: Resize factor.  When creating GIF, each static image
        (frame) will be resized to q times its original size, where q =
        `resize_factor`.  This will affect only the GIF.  The images themselves,
        at locations specified in `input_file_names`, will not be changed.
    :param convert_exe_name: See doc for `trim_whitespace`.
    :raises: ValueError: if ImageMagick command (which is ultimately a Unix
        command) fails.
    """

    error_checking.assert_is_string_list(input_file_names)
    error_checking.assert_is_numpy_array(
        numpy.array(input_file_names), num_dimensions=1
    )

    file_system_utils.mkdir_recursive_if_necessary(file_name=output_file_name)
    error_checking.assert_file_exists(convert_exe_name)

    error_checking.assert_is_greater(num_seconds_per_frame, 0.)
    error_checking.assert_is_leq(num_seconds_per_frame, 10.)
    error_checking.assert_is_geq(resize_factor, 0.2)
    error_checking.assert_is_leq(resize_factor, 1.)

    num_centiseconds_per_frame = int(numpy.round(100 * num_seconds_per_frame))
    num_centiseconds_per_frame = max([num_centiseconds_per_frame, 1])
    resize_percentage = int(numpy.round(100 * resize_factor))
    resize_percentage = max([resize_percentage, 1])

    command_string = '"{0:s}" -delay {1:d} '.format(
        convert_exe_name, num_centiseconds_per_frame)

    command_string += ' '.join(['"{0:s}"'.format(f) for f in input_file_names])

    command_string += ' -resize {0:d}% "{1:s}"'.format(
        resize_percentage, output_file_name)

    exit_code = os.system(command_string)
    if exit_code == 0:
        return

    raise ValueError(ERROR_STRING)
