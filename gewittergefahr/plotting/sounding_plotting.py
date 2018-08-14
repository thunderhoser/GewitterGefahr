"""Plotting methods for atmospheric soundings."""

import os
import tempfile
import numpy
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as pyplot
from skewt import SkewT
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.gg_utils import error_checking

# Paths to ImageMagick executables.
CONVERT_EXE_NAME = '/usr/bin/convert'
MONTAGE_EXE_NAME = '/usr/bin/montage'

DEFAULT_LINE_WIDTH = 2
DEFAULT_LINE_COLOUR = numpy.array([228., 26., 28.]) / 255

FONT_SIZE = 13
DOTS_PER_INCH = 300
NUM_PIXELS_FOR_UNPANELED_IMAGE = int(1e6)
BORDER_WIDTH_FOR_UNPANELED_IMAGE_PX = 10
BORDER_WIDTH_FOR_PANELED_IMAGE_PX = 50

pyplot.rc('font', size=FONT_SIZE)
pyplot.rc('axes', titlesize=FONT_SIZE)
pyplot.rc('axes', labelsize=FONT_SIZE)
pyplot.rc('xtick', labelsize=FONT_SIZE)
pyplot.rc('ytick', labelsize=FONT_SIZE)
pyplot.rc('legend', fontsize=FONT_SIZE)
pyplot.rc('figure', titlesize=FONT_SIZE)


def plot_sounding(
        sounding_dict_for_skewt, title_string, line_colour=DEFAULT_LINE_COLOUR,
        line_width=DEFAULT_LINE_WIDTH):
    """Plots atmospheric sounding.

    H = number of vertical levels in sounding

    :param sounding_dict_for_skewt: Dictionary with the following keys.
    sounding_dict_for_skewt['pres']: length-H numpy array of pressures
        (Pascals).
    sounding_dict_for_skewt['temp']: length-H numpy array of temperatures
        (deg C).
    sounding_dict_for_skewt['dwpt']: length-H numpy array of dewpoints (deg C).
    sounding_dict_for_skewt['sknt']: length-H numpy array of wind speeds
        (nautical miles per hour, or "knots").
    sounding_dict_for_skewt['drct']: length-H numpy array of wind directions
        (degrees of origin, as per meteorological convention).

    :param title_string: Title.
    :param line_colour: Colour for temperature line, dewpoint line, and wind
        barbs (in any format accepted by `matplotlib.colors`).
    :param line_width: Width for temperature line, dewpoint line, and wind
        barbs.
    """

    sounding_object = SkewT.Sounding(soundingdata=sounding_dict_for_skewt)
    sounding_object.plot_skewt(
        color=line_colour, lw=line_width, parcel_type=None, title=title_string,
        tmin=-30., tmax=40.)


def plot_many_soundings(
        list_of_skewt_dictionaries, title_strings, num_panel_rows,
        output_file_name, temp_directory_name=None,
        line_colour=DEFAULT_LINE_COLOUR, line_width=DEFAULT_LINE_WIDTH):
    """Creates paneled figure with many soundings.

    N = number of soundings to plot

    :param list_of_skewt_dictionaries: length-N list of dictionaries.  Each
        dictionary must satisfy the input format for `sounding_dict_for_skewt`
        in `plot_sounding`.
    :param title_strings: length-N list of titles.
    :param num_panel_rows: Number of rows in paneled figure.
    :param output_file_name: Path to output (image) file.
    :param temp_directory_name: Name of temporary directory.  Each panel will be
        stored here, then deleted after the panels have been concatenated into
        the final image.  If `temp_directory_name is None`, will use the default
        temp directory on the local machine.
    :param line_colour: See doc for `plot_sounding`.
    :param line_width: Same.
    """

    error_checking.assert_is_numpy_array(
        numpy.array(title_strings), num_dimensions=1)
    num_soundings = len(title_strings)

    error_checking.assert_is_list(list_of_skewt_dictionaries)
    error_checking.assert_is_geq(len(list_of_skewt_dictionaries), num_soundings)
    error_checking.assert_is_leq(len(list_of_skewt_dictionaries), num_soundings)

    error_checking.assert_is_integer(num_panel_rows)
    error_checking.assert_is_geq(num_panel_rows, 1)
    error_checking.assert_is_leq(num_panel_rows, num_soundings)

    file_system_utils.mkdir_recursive_if_necessary(file_name=output_file_name)
    if temp_directory_name is not None:
        file_system_utils.mkdir_recursive_if_necessary(
            directory_name=temp_directory_name)

    temp_file_names = [None] * num_soundings
    num_panel_columns = int(numpy.ceil(float(num_soundings) / num_panel_rows))
    main_command_string = '"{0:s}" -mode concatenate -tile {1:d}x{2:d}'.format(
        MONTAGE_EXE_NAME, num_panel_columns, num_panel_rows)

    for i in range(num_panel_rows):
        for j in range(num_panel_columns):
            this_sounding_index = i * num_panel_columns + j
            if this_sounding_index >= num_soundings:
                break

            plot_sounding(
                sounding_dict_for_skewt=list_of_skewt_dictionaries[
                    this_sounding_index],
                title_string=title_strings[this_sounding_index],
                line_colour=line_colour, line_width=line_width)

            temp_file_names[this_sounding_index] = '{0:s}.jpg'.format(
                tempfile.NamedTemporaryFile(
                    dir=temp_directory_name, delete=False).name
            )
            main_command_string += ' "{0:s}"'.format(
                temp_file_names[this_sounding_index])

            print 'Saving sounding to: "{0:s}"...'.format(
                temp_file_names[this_sounding_index])
            pyplot.savefig(
                temp_file_names[this_sounding_index], dpi=DOTS_PER_INCH)
            pyplot.close()

            command_string = (
                '"{0:s}" "{1:s}" -resize {2:d}@ -trim -bordercolor White '
                '-border {3:d} "{1:s}"'
            ).format(CONVERT_EXE_NAME, temp_file_names[this_sounding_index],
                     NUM_PIXELS_FOR_UNPANELED_IMAGE,
                     BORDER_WIDTH_FOR_UNPANELED_IMAGE_PX)

            exit_code = os.system(command_string)
            if exit_code != 0:
                raise ValueError('\nUnix command failed (log messages shown '
                                 'above should explain why).')

    print 'Concatenating panels into one figure: "{0:s}"...'.format(
        output_file_name)

    main_command_string += (
        ' -trim -bordercolor White -border {0:d} "{1:s}"'
    ).format(BORDER_WIDTH_FOR_UNPANELED_IMAGE_PX, output_file_name)

    exit_code = os.system(main_command_string)
    if exit_code != 0:
        raise ValueError('\nUnix command failed (log messages shown above '
                         'should explain why).')

    for i in range(num_soundings):
        os.remove(temp_file_names[i])
