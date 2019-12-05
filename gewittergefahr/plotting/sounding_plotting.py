"""Plotting methods for atmospheric soundings."""

import os
import tempfile
import numpy
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as pyplot
import metpy.plots
import metpy.units
from gewittergefahr.gg_utils import soundings
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.gg_utils import error_checking
from gewittergefahr.plotting import plotting_utils
from gewittergefahr.plotting import imagemagick_utils

MAIN_LINE_COLOUR_KEY = 'main_line_colour'
MAIN_LINE_WIDTH_KEY = 'main_line_width'
DRY_ADIABAT_COLOUR_KEY = 'dry_adiabat_colour'
MOIST_ADIABAT_COLOUR_KEY = 'moist_adiabat_colour'
ISOHUME_COLOUR_KEY = 'isohume_colour'
CONTOUR_LINE_WIDTH_KEY = 'contour_line_width'
GRID_LINE_COLOUR_KEY = 'grid_line_colour'
GRID_LINE_WIDTH_KEY = 'grid_line_width'
FIGURE_WIDTH_KEY = 'figure_width_inches'
FIGURE_HEIGHT_KEY = 'figure_height_inches'

DEFAULT_OPTION_DICT = {
    MAIN_LINE_COLOUR_KEY: numpy.array([0, 0, 0], dtype=float),
    MAIN_LINE_WIDTH_KEY: 3,
    DRY_ADIABAT_COLOUR_KEY: numpy.array([217, 95, 2], dtype=float) / 255,
    MOIST_ADIABAT_COLOUR_KEY: numpy.array([117, 112, 179], dtype=float) / 255,
    ISOHUME_COLOUR_KEY: numpy.array([27, 158, 119], dtype=float) / 255,
    CONTOUR_LINE_WIDTH_KEY: 0.5,
    GRID_LINE_COLOUR_KEY: numpy.array([152, 152, 152], dtype=float) / 255,
    GRID_LINE_WIDTH_KEY: 1.5,
    FIGURE_WIDTH_KEY: 15,
    FIGURE_HEIGHT_KEY: 15
}

DEFAULT_FONT_SIZE = 30
TITLE_FONT_SIZE = 25

pyplot.rc('font', size=DEFAULT_FONT_SIZE)
pyplot.rc('axes', titlesize=DEFAULT_FONT_SIZE)
pyplot.rc('axes', labelsize=DEFAULT_FONT_SIZE)
pyplot.rc('xtick', labelsize=DEFAULT_FONT_SIZE)
pyplot.rc('ytick', labelsize=DEFAULT_FONT_SIZE)
pyplot.rc('legend', fontsize=DEFAULT_FONT_SIZE)
pyplot.rc('figure', titlesize=DEFAULT_FONT_SIZE)

# Paths to ImageMagick executables.
CONVERT_EXE_NAME = '/usr/bin/convert'
MONTAGE_EXE_NAME = '/usr/bin/montage'

DOTS_PER_INCH = 300
SINGLE_IMAGE_SIZE_PX = int(1e6)
SINGLE_IMAGE_BORDER_WIDTH_PX = 10
PANELED_IMAGE_BORDER_WIDTH_PX = 50


def plot_sounding(
        sounding_dict_for_metpy, font_size=DEFAULT_FONT_SIZE, title_string=None,
        option_dict=None):
    """Plots atmospheric sounding.

    H = number of vertical levels in sounding

    :param sounding_dict_for_metpy: Dictionary with the following keys.
    sounding_dict_for_metpy['pressures_mb']: length-H numpy array of pressures
        (millibars).
    sounding_dict_for_metpy['temperatures_deg_c']: length-H numpy array of
        temperatures.
    sounding_dict_for_metpy['dewpoints_deg_c']: length-H numpy array of
        dewpoints.
    sounding_dict_for_metpy['u_winds_kt']: length-H numpy array of eastward wind
        components (nautical miles per hour, or "knots").
    sounding_dict_for_metpy['v_winds_kt']: length-H numpy array of northward
        wind components.

    :param font_size: Font size.
    :param title_string: Title.
    :param option_dict: Dictionary with the following keys.
    option_dict['main_line_colour']: Colour for temperature and dewpoint lines
        (in any format accepted by matplotlib).
    option_dict['main_line_width']: Width for temperature and dewpoint lines.
    option_dict['dry_adiabat_colour']: Colour for dry adiabats.
    option_dict['moist_adiabat_colour']: Colour for moist adiabats.
    option_dict['isohume_colour']: Colour for isohumes (lines of constant mixing
        ratio).
    option_dict['contour_line_width']: Width for adiabats and isohumes.
    option_dict['grid_line_colour']: Colour for grid lines (temperature and
        pressure contours).
    option_dict['grid_line_width']: Width for grid lines.
    option_dict['figure_width_inches']: Figure width.
    option_dict['figure_height_inches']: Figure height.

    :return: figure_object: Figure handle (instance of
        `matplotlib.figure.Figure`).
    :return: axes_object: Axes handle (instance of
        `matplotlib.axes._subplots.AxesSubplot`).
    """

    if title_string is not None:
        error_checking.assert_is_string(title_string)

    if option_dict is None:
        orig_option_dict = {}
    else:
        orig_option_dict = option_dict.copy()

    option_dict = DEFAULT_OPTION_DICT.copy()
    option_dict.update(orig_option_dict)

    main_line_colour = option_dict[MAIN_LINE_COLOUR_KEY]
    main_line_width = option_dict[MAIN_LINE_WIDTH_KEY]
    dry_adiabat_colour = option_dict[DRY_ADIABAT_COLOUR_KEY]
    moist_adiabat_colour = option_dict[MOIST_ADIABAT_COLOUR_KEY]
    isohume_colour = option_dict[ISOHUME_COLOUR_KEY]
    contour_line_width = option_dict[CONTOUR_LINE_WIDTH_KEY]
    grid_line_colour = option_dict[GRID_LINE_COLOUR_KEY]
    grid_line_width = option_dict[GRID_LINE_WIDTH_KEY]
    figure_width_inches = option_dict[FIGURE_WIDTH_KEY]
    figure_height_inches = option_dict[FIGURE_HEIGHT_KEY]

    figure_object = pyplot.figure(
        figsize=(figure_width_inches, figure_height_inches)
    )
    skewt_object = metpy.plots.SkewT(figure_object, rotation=45)

    pressures_mb = sounding_dict_for_metpy[
        soundings.PRESSURE_COLUMN_METPY] * metpy.units.units.hPa
    temperatures_deg_c = sounding_dict_for_metpy[
        soundings.TEMPERATURE_COLUMN_METPY] * metpy.units.units.degC
    dewpoints_deg_c = sounding_dict_for_metpy[
        soundings.DEWPOINT_COLUMN_METPY] * metpy.units.units.degC

    skewt_object.plot(
        pressures_mb, temperatures_deg_c,
        color=plotting_utils.colour_from_numpy_to_tuple(main_line_colour),
        linewidth=main_line_width, linestyle='solid'
    )

    num_points = len(pressures_mb)
    middle_index = int(numpy.floor(
        float(num_points) / 2
    ))

    skewt_object.ax.text(
        pressures_mb[middle_index], temperatures_deg_c[middle_index],
        'Air\ntemperature', fontsize=font_size,
        color=plotting_utils.colour_from_numpy_to_tuple(main_line_colour),
        horizontalalignment='left', verticalalignment='middle'
    )

    skewt_object.plot(
        pressures_mb, dewpoints_deg_c,
        color=plotting_utils.colour_from_numpy_to_tuple(main_line_colour),
        linewidth=main_line_width, linestyle='dashed'
    )

    skewt_object.ax.text(
        pressures_mb[middle_index], dewpoints_deg_c[middle_index],
        'Dewpoint\ntemperature', fontsize=font_size,
        color=plotting_utils.colour_from_numpy_to_tuple(main_line_colour),
        horizontalalignment='right', verticalalignment='middle'
    )

    try:
        u_winds_kt = sounding_dict_for_metpy[
            soundings.U_WIND_COLUMN_METPY] * metpy.units.units.knots
        v_winds_kt = sounding_dict_for_metpy[
            soundings.V_WIND_COLUMN_METPY] * metpy.units.units.knots
        plot_wind = True
    except KeyError:
        plot_wind = False

    if plot_wind:
        skewt_object.plot_barbs(pressures_mb, u_winds_kt, v_winds_kt)

    axes_object = skewt_object.ax
    axes_object.grid(
        color=plotting_utils.colour_from_numpy_to_tuple(grid_line_colour),
        linewidth=grid_line_width, linestyle='dashed'
    )

    skewt_object.plot_dry_adiabats(
        color=plotting_utils.colour_from_numpy_to_tuple(dry_adiabat_colour),
        linewidth=contour_line_width, linestyle='solid', alpha=1.
    )
    skewt_object.plot_moist_adiabats(
        color=plotting_utils.colour_from_numpy_to_tuple(moist_adiabat_colour),
        linewidth=contour_line_width, linestyle='solid', alpha=1.
    )
    skewt_object.plot_mixing_lines(
        color=plotting_utils.colour_from_numpy_to_tuple(isohume_colour),
        linewidth=contour_line_width, linestyle='solid', alpha=1.
    )

    axes_object.set_ylim(1000, 100)
    axes_object.set_xlim(-40, 50)
    axes_object.set_xlabel('')
    axes_object.set_ylabel('')

    # TODO(thunderhoser): Shouldn't need this hack.
    tick_values_deg_c = numpy.linspace(-40, 50, num=10)
    axes_object.set_xticks(tick_values_deg_c)

    x_tick_labels = [
        '{0:d}'.format(int(numpy.round(x))) for x in axes_object.get_xticks()
    ]
    axes_object.set_xticklabels(x_tick_labels, fontsize=font_size)

    y_tick_labels = [
        '{0:d}'.format(int(numpy.round(y))) for y in axes_object.get_yticks()
    ]
    axes_object.set_yticklabels(y_tick_labels, fontsize=font_size)

    # TODO(thunderhoser): Shouldn't need this hack.
    axes_object.set_xlim(-40, 50)

    if title_string is not None:
        pyplot.title(title_string, fontsize=font_size)

    return figure_object, axes_object


def plot_many_soundings(
        list_of_metpy_dictionaries, title_strings, num_panel_rows,
        output_file_name, temp_directory_name=None, option_dict=None):
    """Creates paneled figure with many soundings.

    N = number of soundings to plot

    :param list_of_metpy_dictionaries: length-N list of dictionaries.  Each
        dictionary must satisfy the input format for `sounding_dict_for_metpy`
        in `plot_sounding`.
    :param title_strings: length-N list of titles.
    :param num_panel_rows: Number of rows in paneled figure.
    :param output_file_name: Path to output (image) file.
    :param temp_directory_name: Name of temporary directory.  Each panel will be
        stored here, then deleted after the panels have been concatenated into
        the final image.  If `temp_directory_name is None`, will use the default
        temp directory on the local machine.
    :param option_dict: See doc for `plot_sounding`.
    """

    error_checking.assert_is_numpy_array(
        numpy.array(title_strings), num_dimensions=1)
    num_soundings = len(title_strings)

    error_checking.assert_is_list(list_of_metpy_dictionaries)
    error_checking.assert_is_geq(len(list_of_metpy_dictionaries), num_soundings)
    error_checking.assert_is_leq(len(list_of_metpy_dictionaries), num_soundings)

    error_checking.assert_is_integer(num_panel_rows)
    error_checking.assert_is_geq(num_panel_rows, 1)
    error_checking.assert_is_leq(num_panel_rows, num_soundings)

    file_system_utils.mkdir_recursive_if_necessary(file_name=output_file_name)
    if temp_directory_name is not None:
        file_system_utils.mkdir_recursive_if_necessary(
            directory_name=temp_directory_name)

    temp_file_names = [None] * num_soundings
    num_panel_columns = int(numpy.ceil(float(num_soundings) / num_panel_rows))

    for i in range(num_panel_rows):
        for j in range(num_panel_columns):
            this_sounding_index = i * num_panel_columns + j
            if this_sounding_index >= num_soundings:
                break

            plot_sounding(
                sounding_dict_for_metpy=list_of_metpy_dictionaries[
                    this_sounding_index],
                title_string=title_strings[this_sounding_index],
                option_dict=option_dict)

            temp_file_names[this_sounding_index] = '{0:s}.jpg'.format(
                tempfile.NamedTemporaryFile(
                    dir=temp_directory_name, delete=False).name
            )

            print('Saving sounding to: "{0:s}"...'.format(
                temp_file_names[this_sounding_index]
            ))

            pyplot.savefig(
                temp_file_names[this_sounding_index], dpi=DOTS_PER_INCH)
            pyplot.close()

            imagemagick_utils.trim_whitespace(
                input_file_name=temp_file_names[this_sounding_index],
                output_file_name=temp_file_names[this_sounding_index],
                border_width_pixels=SINGLE_IMAGE_BORDER_WIDTH_PX)

            imagemagick_utils.resize_image(
                input_file_name=temp_file_names[this_sounding_index],
                output_file_name=temp_file_names[this_sounding_index],
                output_size_pixels=SINGLE_IMAGE_SIZE_PX)

    print('Concatenating panels into one figure: "{0:s}"...'.format(
        output_file_name))

    imagemagick_utils.concatenate_images(
        input_file_names=temp_file_names, output_file_name=output_file_name,
        num_panel_rows=num_panel_rows, num_panel_columns=num_panel_columns,
        border_width_pixels=PANELED_IMAGE_BORDER_WIDTH_PX)

    for i in range(num_soundings):
        os.remove(temp_file_names[i])
