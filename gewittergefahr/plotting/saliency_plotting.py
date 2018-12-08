"""Plotting methods for saliency maps.

--- NOTATION ---

The following letters will be used throughout this module.

E = number of storm objects for which saliency maps were created
M = number of grid rows per image
N = number of grid columns per image
H = number of grid heights per image (only for 3-D images)
F = number of radar fields per image (only for 3-D images)
C = number of radar channels (field/height pairs) per image (only for 2-D)
"""

import os
import tempfile
import numpy
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as pyplot
from gewittergefahr.gg_utils import soundings
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.gg_utils import error_checking
from gewittergefahr.deep_learning import saliency_maps
from gewittergefahr.deep_learning import deep_learning_utils as dl_utils
from gewittergefahr.plotting import plotting_utils
from gewittergefahr.plotting import sounding_plotting
from gewittergefahr.plotting import imagemagick_utils

TIME_FORMAT = '%Y-%m-%d-%H%M%S'

DEFAULT_MIN_FONT_SIZE = 10.
DEFAULT_MAX_FONT_SIZE = 25.

COLOUR_MAP_KEY = 'colour_map_object'
MAX_COLOUR_VALUE_KEY = 'max_colour_value'
MIN_FONT_SIZE_RADAR_KEY = 'min_font_size_for_radar_points'
MAX_FONT_SIZE_RADAR_KEY = 'max_font_size_for_radar_points'
MIN_FONT_SIZE_SOUNDING_KEY = 'min_font_size_for_sounding_points'
MAX_FONT_SIZE_SOUNDING_KEY = 'max_font_size_for_sounding_points'

DEFAULT_OPTION_DICT = {
    COLOUR_MAP_KEY: pyplot.cm.gist_yarg,
    MAX_COLOUR_VALUE_KEY: None,
    MIN_FONT_SIZE_RADAR_KEY: 8.,
    MAX_FONT_SIZE_RADAR_KEY: 20.,
    MIN_FONT_SIZE_SOUNDING_KEY: 24.,
    MAX_FONT_SIZE_SOUNDING_KEY: 60.
}

WIND_NAME = 'wind_m_s01'
WIND_COMPONENT_NAMES = [soundings.U_WIND_NAME, soundings.V_WIND_NAME]

WIND_BARB_LENGTH = 10.
EMPTY_WIND_BARB_RADIUS = 0.2
SALIENCY_MULTIPLIER_FOR_WIND_BARBS = 10.

SOUNDING_FIELD_NAME_TO_ABBREV_DICT = {
    soundings.SPECIFIC_HUMIDITY_NAME: r'$q_{v}$',
    soundings.VIRTUAL_POTENTIAL_TEMPERATURE_NAME: r'$\theta_{v}$',
    soundings.TEMPERATURE_NAME: r'$T$',
    soundings.RELATIVE_HUMIDITY_NAME: 'RH',
    soundings.U_WIND_NAME: r'$u$',
    soundings.V_WIND_NAME: r'$v$',
    soundings.PRESSURE_NAME: r'$p$',
    WIND_NAME: 'Wind'
}

METRES_TO_KM = 1e-3
DEFAULT_FIG_WIDTH_INCHES = 15.
DEFAULT_FIG_HEIGHT_INCHES = 15.
TITLE_FONT_SIZE = 20

# Paths to ImageMagick executables.
CONVERT_EXE_NAME = '/usr/bin/convert'
MONTAGE_EXE_NAME = '/usr/bin/montage'

DOTS_PER_INCH = 300
SINGLE_IMAGE_SIZE_PIXELS = int(1e6)
SINGLE_IMAGE_BORDER_WIDTH_PIXELS = 10
PANELED_IMAGE_BORDER_WIDTH_PIXELS = 10


def _saliency_to_colour_and_size(
        saliency_matrix, colour_map_object, max_colour_value,
        min_font_size_points, max_font_size_points):
    """Returns font size and colour for each saliency value.

    :param saliency_matrix: M-by-N numpy array of saliency values.
    :param colour_map_object: See doc for `plot_saliency_for_radar`.
    :param max_colour_value: Same.
    :param min_font_size_points: Same.
    :param max_font_size_points: Same.
    :return: rgb_matrix: M-by-N-by-3 numpy array of colours.
    :return: font_size_matrix_points: M-by-N numpy array of font sizes.
    """

    error_checking.assert_is_greater(max_colour_value, 0.)
    error_checking.assert_is_greater(min_font_size_points, 0.)
    error_checking.assert_is_greater(max_font_size_points, min_font_size_points)

    colour_norm_object = pyplot.Normalize(vmin=0., vmax=max_colour_value)
    rgba_matrix = colour_map_object(colour_norm_object(
        numpy.absolute(saliency_matrix)))
    rgb_matrix = rgba_matrix[..., :-1]

    norm_abs_saliency_matrix = (
        numpy.absolute(saliency_matrix) / max_colour_value)
    norm_abs_saliency_matrix[norm_abs_saliency_matrix > 1.] = 1.
    font_size_matrix_points = (
        min_font_size_points + norm_abs_saliency_matrix *
        (max_font_size_points - min_font_size_points)
    )

    return rgb_matrix, font_size_matrix_points


def plot_saliency_for_sounding(
        saliency_matrix, sounding_field_names, pressure_levels_mb, axes_object,
        title_string='', option_dict=None):
    """Plots saliency for each sounding field.

    f = number of sounding fields
    p = number of pressure levels

    :param saliency_matrix: p-by-f numpy array of saliency values.
    :param sounding_field_names: length-f list with names of sounding fields.
    :param pressure_levels_mb: length-p numpy array of pressure levels
        (millibars).
    :param axes_object: Instance of `matplotlib.axes._subplots.AxesSubplot`.
    :param title_string: Figure title.
    :param option_dict: Dictionary with the following keys.
    option_dict['colour_map_object']: Instance of
        `matplotlib.colors.ListedColormap`.
    option_dict['max_colour_value']: Max saliency in colour map.  Minimum
        saliency in colour map will be -1 * `max_colour_value`.
    option_dict['min_font_size_points']: Font size for saliency = 0.
    option_dict['max_font_size_points']: Font size for saliency =
        `max_colour_value`.
    """

    if option_dict is None:
        orig_option_dict = {}
    else:
        orig_option_dict = option_dict.copy()

    option_dict = DEFAULT_OPTION_DICT.copy()
    option_dict.update(orig_option_dict)

    error_checking.assert_is_greater_numpy_array(pressure_levels_mb, 0.)
    error_checking.assert_is_numpy_array(pressure_levels_mb, num_dimensions=1)
    num_pressure_levels = len(pressure_levels_mb)

    error_checking.assert_is_list(sounding_field_names)
    error_checking.assert_is_numpy_array(
        numpy.array(sounding_field_names), num_dimensions=1)
    num_sounding_fields = len(sounding_field_names)

    error_checking.assert_is_numpy_array_without_nan(saliency_matrix)
    error_checking.assert_is_numpy_array(
        saliency_matrix,
        exact_dimensions=numpy.array(
            [num_pressure_levels, num_sounding_fields])
    )

    try:
        u_wind_index = sounding_field_names.index(soundings.U_WIND_NAME)
        v_wind_index = sounding_field_names.index(soundings.V_WIND_NAME)
        plot_wind_barbs = True
    except ValueError:
        plot_wind_barbs = False

    if plot_wind_barbs:
        u_wind_saliency_values = saliency_matrix[:, u_wind_index]
        v_wind_saliency_values = saliency_matrix[:, v_wind_index]
        wind_saliency_magnitudes = numpy.sqrt(
            u_wind_saliency_values ** 2 + v_wind_saliency_values ** 2)

        colour_norm_object = pyplot.Normalize(
            vmin=0., vmax=option_dict[MAX_COLOUR_VALUE_KEY])
        colour_map_object = option_dict[COLOUR_MAP_KEY]
        rgb_matrix_for_wind = colour_map_object(colour_norm_object(
            wind_saliency_magnitudes))
        rgb_matrix_for_wind = rgb_matrix_for_wind[..., :-1]

        non_wind_flags = numpy.array(
            [f not in WIND_COMPONENT_NAMES for f in sounding_field_names],
            dtype=bool)
        non_wind_indices = numpy.where(non_wind_flags)[0]
        sounding_field_names = [
            sounding_field_names[k] for k in non_wind_indices]
        saliency_matrix = saliency_matrix[:, non_wind_indices]

        sounding_field_names.append(WIND_NAME)
        num_sounding_fields = len(sounding_field_names)

    rgb_matrix, font_size_matrix_points = _saliency_to_colour_and_size(
        saliency_matrix=saliency_matrix,
        colour_map_object=option_dict[COLOUR_MAP_KEY],
        max_colour_value=option_dict[MAX_COLOUR_VALUE_KEY],
        min_font_size_points=option_dict[MIN_FONT_SIZE_SOUNDING_KEY],
        max_font_size_points=option_dict[MAX_FONT_SIZE_SOUNDING_KEY])

    for j in range(num_sounding_fields):
        if sounding_field_names[j] == WIND_NAME:
            for i in range(num_pressure_levels):
                this_vector = numpy.array(
                    [u_wind_saliency_values[i], v_wind_saliency_values[i]])
                this_vector = (
                    50 * this_vector / numpy.linalg.norm(this_vector, ord=2))

                axes_object.barbs(
                    j, pressure_levels_mb[i], this_vector[0], this_vector[1],
                    length=WIND_BARB_LENGTH, fill_empty=True, rounding=False,
                    sizes={'emptybarb': EMPTY_WIND_BARB_RADIUS},
                    color=rgb_matrix_for_wind[i, ...])

            continue

        for i in range(num_pressure_levels):
            if saliency_matrix[i, j] >= 0:
                axes_object.text(
                    j, pressure_levels_mb[i], '+',
                    fontsize=font_size_matrix_points[i, j],
                    color=rgb_matrix[i, j, ...], horizontalalignment='center',
                    verticalalignment='center')
            else:
                axes_object.text(
                    j, pressure_levels_mb[i], '_',
                    fontsize=font_size_matrix_points[i, j],
                    color=rgb_matrix[i, j, ...], horizontalalignment='center',
                    verticalalignment='bottom')

    axes_object.set_xlim(-0.5, num_sounding_fields - 0.5)
    axes_object.set_ylim(100, 1000)
    axes_object.invert_yaxis()
    pyplot.yscale('log')
    pyplot.minorticks_off()

    y_tick_locations = numpy.linspace(100, 1000, num=10, dtype=int)
    y_tick_labels = ['{0:d}'.format(p) for p in y_tick_locations]
    pyplot.yticks(y_tick_locations, y_tick_labels)

    x_tick_locations = numpy.linspace(
        0, num_sounding_fields - 1, num=num_sounding_fields, dtype=float)
    x_tick_labels = [
        SOUNDING_FIELD_NAME_TO_ABBREV_DICT[f] for f in sounding_field_names]
    pyplot.xticks(x_tick_locations, x_tick_labels)

    pyplot.title(title_string)


def plot_saliency_with_sounding(
        sounding_matrix, saliency_matrix, sounding_field_names,
        output_file_name, sounding_title_string='', saliency_title_string='',
        pressure_levels_pascals=None, saliency_option_dict=None,
        sounding_option_dict=None, temp_directory_name=None):
    """Plots saliency for each sounding field, along with the sounding itself.

    The sounding itself will be in the left panel; saliency map will be in the
    right panel.

    f = number of sounding fields
    p = number of pressure levels

    :param sounding_matrix: p-by-f numpy array of sounding measurements.
    :param saliency_matrix: p-by-f numpy array of corresponding saliency values.
    :param sounding_field_names: See doc for `plot_saliency_for_sounding`.
    :param output_file_name: Path to output (image) file.
    :param sounding_title_string: Title for sounding itself.
    :param saliency_title_string: Title for saliency map.
    :param pressure_levels_pascals: length-p numpy array of pressure levels.  If
        `sounding_matrix` already includes pressures, this can be `None`.
    :param saliency_option_dict: See doc for `plot_saliency_for_sounding`.
    :param sounding_option_dict: See doc for `sounding_plotting.plot_sounding`.
    :param temp_directory_name: Name of temporary directory.  Each panel will be
        stored here, then deleted after the panels have been concatenated into
        the final image.  If `temp_directory_name is None`, will use the default
        temp directory on the local machine.
    """

    if saliency_option_dict is None:
        orig_saliency_option_dict = {}
    else:
        orig_saliency_option_dict = saliency_option_dict.copy()

    if sounding_option_dict is None:
        orig_sounding_option_dict = {}
    else:
        orig_sounding_option_dict = sounding_option_dict.copy()

    saliency_option_dict = DEFAULT_OPTION_DICT.copy()
    saliency_option_dict.update(orig_saliency_option_dict)
    sounding_option_dict = sounding_plotting.DEFAULT_OPTION_DICT.copy()
    sounding_option_dict.update(orig_sounding_option_dict)

    file_system_utils.mkdir_recursive_if_necessary(file_name=output_file_name)
    if temp_directory_name is not None:
        file_system_utils.mkdir_recursive_if_necessary(
            directory_name=temp_directory_name)

    # Plot sounding.
    try:
        list_of_metpy_dictionaries = dl_utils.soundings_to_metpy_dictionaries(
            sounding_matrix=numpy.expand_dims(sounding_matrix, axis=0),
            field_names=sounding_field_names)
    except:
        these_pressure_levels_pa = numpy.reshape(
            pressure_levels_pascals, (pressure_levels_pascals.size, 1))
        sounding_matrix = numpy.hstack((
            sounding_matrix, these_pressure_levels_pa))

        list_of_metpy_dictionaries = dl_utils.soundings_to_metpy_dictionaries(
            sounding_matrix=numpy.expand_dims(sounding_matrix, axis=0),
            field_names=sounding_field_names + [soundings.PRESSURE_NAME])

    sounding_plotting.plot_sounding(
        sounding_dict_for_metpy=list_of_metpy_dictionaries[0],
        title_string=sounding_title_string, option_dict=sounding_option_dict)

    temp_sounding_file_name = '{0:s}.jpg'.format(
        tempfile.NamedTemporaryFile(dir=temp_directory_name, delete=False).name)
    pyplot.savefig(temp_sounding_file_name, dpi=DOTS_PER_INCH)
    pyplot.close()

    imagemagick_utils.trim_whitespace(
        input_file_name=temp_sounding_file_name,
        output_file_name=temp_sounding_file_name,
        border_width_pixels=SINGLE_IMAGE_BORDER_WIDTH_PIXELS,
        output_size_pixels=SINGLE_IMAGE_SIZE_PIXELS)

    # Plot saliency map.
    _, axes_object = pyplot.subplots(
        1, 1,
        figsize=(sounding_option_dict[sounding_plotting.FIGURE_WIDTH_KEY],
                 sounding_option_dict[sounding_plotting.FIGURE_HEIGHT_KEY])
    )

    plot_saliency_for_sounding(
        saliency_matrix=saliency_matrix,
        sounding_field_names=sounding_field_names,
        pressure_levels_mb=list_of_metpy_dictionaries[0][
            soundings.PRESSURE_COLUMN_METPY],
        axes_object=axes_object, title_string=saliency_title_string,
        option_dict=saliency_option_dict)

    plotting_utils.add_linear_colour_bar(
        axes_object_or_list=axes_object, values_to_colour=saliency_matrix,
        colour_map=saliency_option_dict[COLOUR_MAP_KEY], colour_min=0.,
        colour_max=saliency_option_dict[MAX_COLOUR_VALUE_KEY],
        orientation='vertical', extend_min=True, extend_max=True)

    temp_saliency_file_name = '{0:s}.jpg'.format(
        tempfile.NamedTemporaryFile(dir=temp_directory_name, delete=False).name)
    pyplot.savefig(temp_saliency_file_name, dpi=DOTS_PER_INCH)
    pyplot.close()

    imagemagick_utils.trim_whitespace(
        input_file_name=temp_saliency_file_name,
        output_file_name=temp_saliency_file_name,
        border_width_pixels=SINGLE_IMAGE_BORDER_WIDTH_PIXELS,
        output_size_pixels=SINGLE_IMAGE_SIZE_PIXELS)

    # Concatenate the sounding and saliency map.
    imagemagick_utils.concatenate_images(
        input_file_names=[temp_sounding_file_name, temp_saliency_file_name],
        output_file_name=output_file_name, num_panel_rows=1,
        num_panel_columns=2,
        border_width_pixels=PANELED_IMAGE_BORDER_WIDTH_PIXELS)

    os.remove(temp_sounding_file_name)
    os.remove(temp_saliency_file_name)


def plot_saliency_with_soundings(
        sounding_matrix, saliency_matrix, saliency_metadata_dict,
        sounding_field_names, output_dir_name, saliency_option_dict=None,
        temp_directory_name=None):
    """For each storm object, plots sounding along with saliency values.

    f = number of sounding fields
    p = number of pressure levels

    This method creates one figure per storm object.

    :param sounding_matrix: E-by-p-by-f numpy array of sounding measurements.
    :param saliency_matrix: E-by-p-by-f numpy array of corresponding saliency
        values.
    :param saliency_metadata_dict: Dictionary returned by
        `saliency_maps.read_file`.
    :param sounding_field_names: See doc for `plot_saliency_for_sounding`.
    :param output_dir_name: Name of output directory (figures will be saved
        here).
    :param saliency_option_dict: See doc for `plot_saliency_with_sounding`.
    :param temp_directory_name: Same.
    """

    error_checking.assert_is_numpy_array(sounding_matrix)
    error_checking.assert_is_numpy_array(
        saliency_matrix, exact_dimensions=numpy.array(sounding_matrix.shape))
    num_storm_objects = sounding_matrix.shape[0]

    sounding_option_dict = {
        sounding_plotting.FIGURE_WIDTH_KEY: DEFAULT_FIG_WIDTH_INCHES,
        sounding_plotting.FIGURE_HEIGHT_KEY: DEFAULT_FIG_HEIGHT_INCHES
    }

    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=output_dir_name)

    for i in range(num_storm_objects):
        this_storm_id = saliency_metadata_dict[
            saliency_maps.STORM_IDS_KEY][i]
        this_storm_time_string = time_conversion.unix_sec_to_string(
            saliency_metadata_dict[saliency_maps.STORM_TIMES_KEY][i],
            TIME_FORMAT)

        if saliency_metadata_dict[saliency_maps.SOUNDING_PRESSURES_KEY] is None:
            these_pressure_levels_pascals = None
        else:
            these_pressure_levels_pascals = saliency_metadata_dict[
                saliency_maps.SOUNDING_PRESSURES_KEY][i, ...]

        this_sounding_title_string = 'Storm "{0:s}" at {1:s}'.format(
            this_storm_id, this_storm_time_string)
        this_saliency_title_string = 'Saliency'

        this_figure_file_name = (
            '{0:s}/saliency_{1:s}_{2:s}_soundings.jpg'
        ).format(output_dir_name, this_storm_id.replace('_', '-'),
                 this_storm_time_string)

        print 'Saving figure to file: "{0:s}"...'.format(this_figure_file_name)
        plot_saliency_with_sounding(
            sounding_matrix=sounding_matrix[i, ...],
            saliency_matrix=saliency_matrix[i, ...],
            sounding_field_names=sounding_field_names,
            output_file_name=this_figure_file_name,
            sounding_title_string=this_sounding_title_string,
            saliency_title_string=this_saliency_title_string,
            pressure_levels_pascals=these_pressure_levels_pascals,
            saliency_option_dict=saliency_option_dict,
            sounding_option_dict=sounding_option_dict,
            temp_directory_name=temp_directory_name)


def plot_2d_grid(saliency_matrix_2d, axes_object, colour_map_object,
                 max_absolute_colour_value, min_font_size=DEFAULT_MIN_FONT_SIZE,
                 max_font_size=DEFAULT_MAX_FONT_SIZE):
    """Plots 2-D saliency map with "+" and "-" signs.

    M = number of rows in grid
    N = number of columns in grid

    :param saliency_matrix_2d: M-by-N numpy array of saliency values.
    :param axes_object: Instance of `matplotlib.axes._subplots.AxesSubplot`.
    :param colour_map_object: Instance of `matplotlib.pyplot.cm`.  This colour
        map will be applied to absolute values, rather than signed values.  In
        other words, this colour map will be duplicated and flipped, to create a
        diverging colour map.  Thus, `colour_map_object` itself should be a
        sequential colour map, not a diverging one.  However, this is not
        enforced by the code, so do whatever you want.
    :param max_absolute_colour_value: Max absolute saliency in colour scheme.
        The min and max values, respectively, will be
        `-1 * max_absolute_colour_value` and `max_absolute_colour_value`.
    :param min_font_size: Minimum font size (for zero saliency).
    :param max_font_size: Max font size (for `max_absolute_colour_value`).
    """

    error_checking.assert_is_numpy_array_without_nan(saliency_matrix_2d)
    error_checking.assert_is_numpy_array(saliency_matrix_2d, num_dimensions=2)
    # error_checking.assert_is_greater(max_absolute_colour_value, 0.)
    # error_checking.assert_is_greater(min_font_size, 0.)
    # error_checking.assert_is_greater(max_font_size, min_font_size)

    rgb_matrix, font_size_matrix = _saliency_to_colour_and_size(
        saliency_matrix=saliency_matrix_2d,
        colour_map_object=colour_map_object,
        max_colour_value=max_absolute_colour_value,
        min_font_size_points=min_font_size,
        max_font_size_points=max_font_size)

    num_grid_rows = saliency_matrix_2d.shape[0]
    num_grid_columns = saliency_matrix_2d.shape[1]
    y_coord_vector = numpy.linspace(
        0, num_grid_rows - 1, num=num_grid_rows, dtype=float)
    x_coord_vector = numpy.linspace(
        0, num_grid_columns - 1, num=num_grid_columns, dtype=float)

    x_coord_matrix, y_coord_matrix = numpy.meshgrid(
        x_coord_vector, y_coord_vector)
    x_coord_matrix += 0.5
    y_coord_matrix += 0.5

    for i in range(num_grid_rows):
        for j in range(num_grid_columns):
            if saliency_matrix_2d[i, j] >= 0:
                axes_object.text(
                    x_coord_matrix[i, j], y_coord_matrix[i, j], '+',
                    fontsize=font_size_matrix[i, j],
                    color=rgb_matrix[i, j, ...], horizontalalignment='center',
                    verticalalignment='center', transform=axes_object.transAxes)
            else:
                axes_object.text(
                    x_coord_matrix[i, j], y_coord_matrix[i, j], '_',
                    fontsize=font_size_matrix[i, j],
                    color=rgb_matrix[i, j, ...], horizontalalignment='center',
                    verticalalignment='bottom', transform=axes_object.transAxes)


def plot_many_2d_grids(
        saliency_matrix_3d, axes_objects_2d_list, colour_map_object,
        max_absolute_colour_value, min_font_size=DEFAULT_MIN_FONT_SIZE,
        max_font_size=DEFAULT_MAX_FONT_SIZE):
    """Plots many 2-D saliency maps with "+" and "-" signs.

    The saliency map for each field will be one panel in a paneled figure.

    M = number of spatial rows
    N = number of spatial columns
    C = number of channels (predictors)

    :param saliency_matrix_3d: M-by-N-by-P numpy array of saliency values.
    :param axes_objects_2d_list: 2-D list, where axes_objects_2d_list[i][j] is
        the handle (instance of `matplotlib.axes._subplots.AxesSubplot`) for the
        [i]th row and [j]th column.
    :param colour_map_object: See doc for `plot_2d_grid`.
    :param max_absolute_colour_value: Same.
    :param min_font_size: Same.
    :param max_font_size: Same.
    """

    error_checking.assert_is_numpy_array_without_nan(saliency_matrix_3d)
    error_checking.assert_is_numpy_array(saliency_matrix_3d, num_dimensions=3)
    num_predictors = saliency_matrix_3d.shape[-1]

    num_panel_rows = len(axes_objects_2d_list)
    num_panel_columns = len(axes_objects_2d_list[0])

    for k in range(num_predictors):
        this_panel_row, this_panel_column = numpy.unravel_index(
            k, (num_panel_rows, num_panel_columns)
        )

        plot_2d_grid(
            saliency_matrix_2d=saliency_matrix_3d[..., k],
            axes_object=axes_objects_2d_list[this_panel_row][this_panel_column],
            colour_map_object=colour_map_object,
            max_absolute_colour_value=max_absolute_colour_value,
            min_font_size=min_font_size, max_font_size=max_font_size)
