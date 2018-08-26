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
from gewittergefahr.gg_utils import radar_utils
from gewittergefahr.gg_utils import soundings_only
from gewittergefahr.gg_utils import grids
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.gg_utils import error_checking
from gewittergefahr.deep_learning import saliency_maps
from gewittergefahr.deep_learning import deep_learning_utils as dl_utils
from gewittergefahr.plotting import plotting_utils
from gewittergefahr.plotting import radar_plotting
from gewittergefahr.plotting import sounding_plotting
from gewittergefahr.plotting import imagemagick_utils

TIME_FORMAT = '%Y-%m-%d-%H%M%S'

COLOUR_MAP_KEY = 'colour_map_object'
MIN_COLOUR_VALUE_KEY = 'min_colour_value'
MAX_COLOUR_VALUE_KEY = 'max_colour_value'
MIN_FONT_SIZE_KEY = 'min_font_size_points'
MAX_FONT_SIZE_KEY = 'max_font_size_points'

DEFAULT_OPTION_DICT = {
    COLOUR_MAP_KEY: pyplot.cm.gist_yarg,
    MIN_COLOUR_VALUE_KEY: None,
    MAX_COLOUR_VALUE_KEY: None,
    MIN_FONT_SIZE_KEY: 8.,
    MAX_FONT_SIZE_KEY: 20.
}

SOUNDING_FIELD_NAME_TO_ABBREV_DICT = {
    soundings_only.SPECIFIC_HUMIDITY_NAME: r'$q_{v}$',
    soundings_only.VIRTUAL_POTENTIAL_TEMPERATURE_NAME: r'$\theta_{v}$',
    soundings_only.TEMPERATURE_NAME: r'$T$',
    soundings_only.RELATIVE_HUMIDITY_NAME: 'RH',
    soundings_only.U_WIND_NAME: r'$u$',
    soundings_only.V_WIND_NAME: r'$v$',
    soundings_only.GEOPOTENTIAL_HEIGHT_NAME: r'$Z$'
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

    rgb_matrix, font_size_matrix_points = _saliency_to_colour_and_size(
        saliency_matrix=saliency_matrix,
        colour_map_object=option_dict[COLOUR_MAP_KEY],
        max_colour_value=option_dict[MAX_COLOUR_VALUE_KEY],
        min_font_size_points=option_dict[MIN_FONT_SIZE_KEY],
        max_font_size_points=option_dict[MAX_FONT_SIZE_KEY])

    error_checking.assert_is_integer_numpy_array(pressure_levels_mb)
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

    for i in range(num_pressure_levels):
        for j in range(num_sounding_fields):
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

    y_tick_locations = numpy.linspace(100, 1000, num=10, dtype=int)
    y_tick_labels = ['{0:d}'.format(p) for p in y_tick_locations]
    pyplot.yticks([], [])
    pyplot.yticks(y_tick_locations, y_tick_labels)

    x_tick_locations = numpy.linspace(
        0, num_sounding_fields - 1, num=num_sounding_fields, dtype=float)
    x_tick_labels = [
        SOUNDING_FIELD_NAME_TO_ABBREV_DICT[f] for f in sounding_field_names]
    pyplot.xticks(x_tick_locations, x_tick_labels)

    pyplot.title(title_string)


def plot_saliency_with_sounding(
        sounding_matrix, saliency_matrix, sounding_field_names,
        pressure_levels_mb, output_file_name, sounding_title_string='',
        saliency_title_string='', saliency_option_dict=None,
        sounding_option_dict=None, temp_directory_name=None):
    """Plots saliency for each sounding field, along with the sounding itself.

    The sounding itself will be in the left panel; saliency map will be in the
    right panel.

    f = number of sounding fields
    p = number of pressure levels

    :param sounding_matrix: p-by-f numpy array of sounding measurements.
    :param saliency_matrix: p-by-f numpy array of corresponding saliency values.
    :param sounding_field_names: See doc for `plot_saliency_for_sounding`.
    :param pressure_levels_mb: Same.
    :param output_file_name: Path to output (image) file.
    :param sounding_title_string: Title for sounding itself.
    :param saliency_title_string: Title for saliency map.
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
    list_of_metpy_dictionaries = dl_utils.soundings_to_metpy_dictionaries(
        sounding_matrix=numpy.expand_dims(sounding_matrix, axis=0),
        pressure_levels_mb=pressure_levels_mb,
        pressureless_field_names=sounding_field_names)

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
        pressure_levels_mb=pressure_levels_mb, axes_object=axes_object,
        title_string=saliency_title_string, option_dict=saliency_option_dict)

    plotting_utils.add_linear_colour_bar(
        axes_object_or_list=axes_object, values_to_colour=saliency_matrix,
        colour_map=saliency_option_dict[COLOUR_MAP_KEY],
        colour_min=saliency_option_dict[MIN_COLOUR_VALUE_KEY],
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
        sounding_field_names, pressure_levels_mb, output_dir_name,
        saliency_option_dict=None, temp_directory_name=None):
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
    :param pressure_levels_mb: Same.
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

        this_sounding_title_string = 'Storm "{0:s}" at {1:s}'.format(
            this_storm_id, this_storm_time_string)
        this_saliency_title_string = 'Saliency'

        this_figure_file_name = (
            '{0:s}/sounding-saliency_{1:s}_{2:s}.jpg'
        ).format(output_dir_name, this_storm_id.replace('_', '-'),
                 this_storm_time_string)

        print 'Saving figure to file: "{0:s}"...'.format(this_figure_file_name)
        plot_saliency_with_sounding(
            sounding_matrix=sounding_matrix[i, ...],
            saliency_matrix=saliency_matrix[i, ...],
            sounding_field_names=sounding_field_names,
            pressure_levels_mb=pressure_levels_mb,
            output_file_name=this_figure_file_name,
            sounding_title_string=this_sounding_title_string,
            saliency_title_string=this_saliency_title_string,
            saliency_option_dict=saliency_option_dict,
            sounding_option_dict=sounding_option_dict,
            temp_directory_name=temp_directory_name)


def plot_saliency_for_radar(saliency_matrix, axes_object, option_dict=None):
    """Plots saliency map for a 2-D radar field.

    :param saliency_matrix: M-by-N numpy array of saliency values.
    :param axes_object: Instance of `matplotlib.axes._subplots.AxesSubplot`.
    :param option_dict: Dictionary with the following keys.
    option_dict['colour_map_object']: Instance of
        `matplotlib.colors.ListedColormap`.
    option_dict['max_colour_value']: Max saliency in colour map.  Minimum
        saliency in colour map will be -1 * `max_colour_value`.
    option_dict['min_font_size_points']: Font size for saliency = 0.
    option_dict['max_font_size_points']: Font size for saliency =
        `max_colour_value`.
    """

    error_checking.assert_is_numpy_array_without_nan(saliency_matrix)
    error_checking.assert_is_numpy_array(saliency_matrix, num_dimensions=2)

    if option_dict is None:
        orig_option_dict = {}
    else:
        orig_option_dict = option_dict.copy()

    option_dict = DEFAULT_OPTION_DICT.copy()
    option_dict.update(orig_option_dict)

    rgb_matrix, font_size_matrix_points = _saliency_to_colour_and_size(
        saliency_matrix=saliency_matrix,
        colour_map_object=option_dict[COLOUR_MAP_KEY],
        max_colour_value=option_dict[MAX_COLOUR_VALUE_KEY],
        min_font_size_points=option_dict[MIN_FONT_SIZE_KEY],
        max_font_size_points=option_dict[MAX_FONT_SIZE_KEY])

    num_grid_rows = saliency_matrix.shape[0]
    num_grid_columns = saliency_matrix.shape[1]
    x_coord_spacing = num_grid_columns ** -1
    y_coord_spacing = num_grid_rows ** -1
    x_coords, y_coords = grids.get_xy_grid_points(
        x_min_metres=x_coord_spacing / 2, y_min_metres=y_coord_spacing / 2,
        x_spacing_metres=x_coord_spacing, y_spacing_metres=y_coord_spacing,
        num_rows=num_grid_rows, num_columns=num_grid_columns)

    for i in range(num_grid_rows):
        for j in range(num_grid_columns):
            if saliency_matrix[i, j] >= 0:
                axes_object.text(
                    x_coords[j], y_coords[i], '+',
                    fontsize=font_size_matrix_points[i, j],
                    color=rgb_matrix[i, j, ...], horizontalalignment='center',
                    verticalalignment='center', transform=axes_object.transAxes)
            else:
                axes_object.text(
                    x_coords[j], y_coords[i], '_',
                    fontsize=font_size_matrix_points[i, j],
                    color=rgb_matrix[i, j, ...], horizontalalignment='center',
                    verticalalignment='bottom', transform=axes_object.transAxes)


def plot_saliency_with_radar_2d_fields(
        radar_matrix, saliency_matrix, saliency_metadata_dict,
        field_name_by_pair, height_by_pair_m_asl, one_fig_per_storm_object,
        num_panel_rows, output_dir_name, saliency_option_dict=None,
        figure_width_inches=DEFAULT_FIG_WIDTH_INCHES,
        figure_height_inches=DEFAULT_FIG_HEIGHT_INCHES):
    """Plots many 2-D saliency fields with the underlying radar fields.

    :param radar_matrix: E-by-M-by-N-by-C numpy array of radar values.
    :param saliency_matrix: E-by-M-by-N-by-C numpy array of corresponding
        saliency values.
    :param saliency_metadata_dict: Dictionary returned by
        `saliency_maps.read_file`.
    :param field_name_by_pair: length-C list of field names (each must be
        accepted by `radar_utils.check_field_name`).
    :param height_by_pair_m_asl: length-C integer numpy array of radar heights
        (metres above sea level).
    :param one_fig_per_storm_object: Boolean flag.  If True, this method will
        created one paneled figure for each storm object, where each panel
        contains the saliency map for a different radar field/height.  If False,
        will create one paneled figure for each radar field/height, where each
        panel contains the saliency map for a different storm object.
    :param num_panel_rows: Number of panel rows in each figure.
    :param output_dir_name: Name of output directory (figures will be saved
        here).
    :param saliency_option_dict: See doc for `plot_saliency_for_radar`.
    :param figure_width_inches: Width of each figure.
    :param figure_height_inches: Height of each figure.
    """

    error_checking.assert_is_numpy_array(radar_matrix, num_dimensions=4)
    error_checking.assert_is_numpy_array(
        saliency_matrix, exact_dimensions=numpy.array(radar_matrix.shape))

    num_storm_objects = radar_matrix.shape[0]
    num_field_height_pairs = radar_matrix.shape[-1]
    error_checking.assert_is_numpy_array(
        numpy.array(field_name_by_pair),
        exact_dimensions=numpy.array([num_field_height_pairs]))

    error_checking.assert_is_integer_numpy_array(height_by_pair_m_asl)
    error_checking.assert_is_geq_numpy_array(height_by_pair_m_asl, 0)
    error_checking.assert_is_numpy_array(
        height_by_pair_m_asl,
        exact_dimensions=numpy.array([num_field_height_pairs]))

    error_checking.assert_is_boolean(one_fig_per_storm_object)
    error_checking.assert_is_integer(num_panel_rows)
    error_checking.assert_is_geq(num_panel_rows, 1)

    if one_fig_per_storm_object:
        error_checking.assert_is_leq(num_panel_rows, num_field_height_pairs)
        num_panel_columns = int(
            numpy.ceil(float(num_field_height_pairs) / num_panel_rows))
    else:
        error_checking.assert_is_leq(num_panel_rows, num_storm_objects)
        num_panel_columns = int(
            numpy.ceil(float(num_storm_objects) / num_panel_rows))

    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=output_dir_name)

    if one_fig_per_storm_object:
        for i in range(num_storm_objects):
            _, axes_objects_2d_list = plotting_utils.init_panels(
                num_panel_rows=num_panel_rows,
                num_panel_columns=num_panel_columns,
                figure_width_inches=figure_width_inches,
                figure_height_inches=figure_height_inches)

            for j in range(num_panel_rows):
                for k in range(num_panel_columns):
                    this_fh_pair_index = j * num_panel_columns + k
                    if this_fh_pair_index >= num_field_height_pairs:
                        continue

                    this_annotation_string = '{0:s}'.format(
                        field_name_by_pair[this_fh_pair_index])

                    if (field_name_by_pair[this_fh_pair_index] ==
                            radar_utils.REFL_NAME):
                        this_annotation_string += '\nat {0:.1f} km'.format(
                            height_by_pair_m_asl[this_fh_pair_index] *
                            METRES_TO_KM)

                    radar_plotting.plot_2d_grid_without_coords(
                        field_matrix=numpy.flipud(
                            radar_matrix[i, ..., this_fh_pair_index]),
                        field_name=field_name_by_pair[this_fh_pair_index],
                        axes_object=axes_objects_2d_list[j][k],
                        annotation_string=this_annotation_string)

                    plot_saliency_for_radar(
                        saliency_matrix=numpy.flipud(
                            saliency_matrix[i, ..., this_fh_pair_index]),
                        axes_object=axes_objects_2d_list[j][k],
                        option_dict=saliency_option_dict)

            this_storm_id = saliency_metadata_dict[
                saliency_maps.STORM_IDS_KEY][i]
            this_storm_time_string = time_conversion.unix_sec_to_string(
                saliency_metadata_dict[saliency_maps.STORM_TIMES_KEY][i],
                TIME_FORMAT)

            this_title_string = 'Saliency for storm "{0:s}" at {1:s}'.format(
                this_storm_id, this_storm_time_string)
            pyplot.suptitle(this_title_string, fontsize=TITLE_FONT_SIZE)

            this_figure_file_name = (
                '{0:s}/radar-saliency_{1:s}_{2:s}.jpg'
            ).format(output_dir_name, this_storm_id.replace('_', '-'),
                     this_storm_time_string)

            print 'Saving figure to file: "{0:s}"...'.format(
                this_figure_file_name)
            pyplot.savefig(this_figure_file_name, dpi=DOTS_PER_INCH)
            pyplot.close()
    else:
        for i in range(num_field_height_pairs):
            _, axes_objects_2d_list = plotting_utils.init_panels(
                num_panel_rows=num_panel_rows,
                num_panel_columns=num_panel_columns,
                figure_width_inches=figure_width_inches,
                figure_height_inches=figure_height_inches)

            for j in range(num_panel_rows):
                for k in range(num_panel_columns):
                    this_storm_object_index = j * num_panel_columns + k
                    if this_storm_object_index >= num_storm_objects:
                        continue

                    this_annotation_string = '"{0:s}"\nat {1:s}'.format(
                        saliency_metadata_dict[saliency_maps.STORM_IDS_KEY][
                            this_storm_object_index],
                        time_conversion.unix_sec_to_string(
                            saliency_metadata_dict[
                                saliency_maps.STORM_TIMES_KEY
                            ][this_storm_object_index],
                            TIME_FORMAT)
                    )

                    radar_plotting.plot_2d_grid_without_coords(
                        field_matrix=numpy.flipud(
                            radar_matrix[this_storm_object_index, ..., i]),
                        field_name=field_name_by_pair[i],
                        axes_object=axes_objects_2d_list[j][k],
                        annotation_string=this_annotation_string)

                    plot_saliency_for_radar(
                        saliency_matrix=numpy.flipud(
                            saliency_matrix[
                                this_storm_object_index, ..., i]),
                        axes_object=axes_objects_2d_list[j][k],
                        option_dict=saliency_option_dict)

            (this_colour_map_object, this_colour_norm_object, _
            ) = radar_plotting.get_default_colour_scheme(
                field_name_by_pair[i])

            plotting_utils.add_colour_bar(
                axes_object_or_list=axes_objects_2d_list,
                values_to_colour=radar_matrix[..., i],
                colour_map=this_colour_map_object,
                colour_norm_object=this_colour_norm_object,
                orientation='horizontal', extend_min=True, extend_max=True)

            this_title_string = '{0:s} at {1:.1f} km ASL'.format(
                field_name_by_pair[i], height_by_pair_m_asl[i] * METRES_TO_KM)
            pyplot.suptitle(this_title_string, fontsize=TITLE_FONT_SIZE)

            this_figure_file_name = (
                '{0:s}/saliency_{1:s}_{2:05d}metres.jpg'
            ).format(output_dir_name, field_name_by_pair[i].replace('_', '-'),
                     height_by_pair_m_asl[i])

            print 'Saving figure to file: "{0:s}"...'.format(
                this_figure_file_name)
            pyplot.savefig(this_figure_file_name, dpi=DOTS_PER_INCH)
            pyplot.close()


def plot_saliency_with_radar_3d_fields(
        radar_matrix, saliency_matrix, saliency_metadata_dict,
        radar_field_names, radar_heights_m_asl, one_fig_per_storm_object,
        num_panel_rows, output_dir_name, saliency_option_dict=None,
        figure_width_inches=DEFAULT_FIG_WIDTH_INCHES,
        figure_height_inches=DEFAULT_FIG_HEIGHT_INCHES):
    """Plots many 3-D saliency fields (with the underlying radar fields).

    :param radar_matrix: E-by-M-by-N-by-H-by-F numpy array of radar values.
    :param saliency_matrix: E-by-M-by-N-by-H-by-F numpy array of corresponding
        saliency values.
    :param saliency_metadata_dict: Dictionary returned by
        `saliency_maps.read_file`.
    :param radar_field_names: length-F list of field names (each must be
        accepted by `radar_utils.check_field_name`).
    :param radar_heights_m_asl: length-H integer numpy array of radar heights
        (metres above sea level).
    :param one_fig_per_storm_object: See doc for
        `plot_saliency_with_radar_2d_fields`.
    :param num_panel_rows: Same.
    :param output_dir_name: Same.
    :param saliency_option_dict: See doc for `plot_saliency_for_radar`.
    :param figure_width_inches: Same.
    :param figure_height_inches: Same.
    """

    error_checking.assert_is_numpy_array(radar_matrix, num_dimensions=5)
    error_checking.assert_is_numpy_array(
        saliency_matrix, exact_dimensions=numpy.array(radar_matrix.shape))

    num_storm_objects = radar_matrix.shape[0]
    num_fields = radar_matrix.shape[-1]
    num_heights = radar_matrix.shape[-2]
    error_checking.assert_is_numpy_array(
        numpy.array(radar_field_names),
        exact_dimensions=numpy.array([num_fields]))

    error_checking.assert_is_integer_numpy_array(radar_heights_m_asl)
    error_checking.assert_is_geq_numpy_array(radar_heights_m_asl, 0)
    error_checking.assert_is_numpy_array(
        radar_heights_m_asl, exact_dimensions=numpy.array([num_heights]))

    error_checking.assert_is_boolean(one_fig_per_storm_object)
    error_checking.assert_is_integer(num_panel_rows)
    error_checking.assert_is_geq(num_panel_rows, 1)

    if one_fig_per_storm_object:
        error_checking.assert_is_leq(num_panel_rows, num_heights)
        num_panel_columns = int(
            numpy.ceil(float(num_heights) / num_panel_rows))
    else:
        error_checking.assert_is_leq(num_panel_rows, num_storm_objects)
        num_panel_columns = int(
            numpy.ceil(float(num_storm_objects) / num_panel_rows))

    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=output_dir_name)

    if one_fig_per_storm_object:
        for i in range(num_storm_objects):
            for m in range(num_fields):
                _, axes_objects_2d_list = plotting_utils.init_panels(
                    num_panel_rows=num_panel_rows,
                    num_panel_columns=num_panel_columns,
                    figure_width_inches=figure_width_inches,
                    figure_height_inches=figure_height_inches)

                for j in range(num_panel_rows):
                    for k in range(num_panel_columns):
                        this_height_index = j * num_panel_columns + k
                        if this_height_index >= num_heights:
                            continue

                        this_annotation_string = '{1:.1f} km ASL'.format(
                            radar_field_names[m],
                            radar_heights_m_asl[this_height_index] *
                            METRES_TO_KM)

                        radar_plotting.plot_2d_grid_without_coords(
                            field_matrix=numpy.flipud(
                                radar_matrix[
                                    i, ..., this_height_index, m]),
                            field_name=radar_field_names[m],
                            axes_object=axes_objects_2d_list[j][k],
                            annotation_string=this_annotation_string)

                        plot_saliency_for_radar(
                            saliency_matrix=numpy.flipud(
                                saliency_matrix[i, ..., this_height_index, m]),
                            axes_object=axes_objects_2d_list[j][k],
                            option_dict=saliency_option_dict)

                (this_colour_map_object, this_colour_norm_object, _
                ) = radar_plotting.get_default_colour_scheme(
                    radar_field_names[m])

                plotting_utils.add_colour_bar(
                    axes_object_or_list=axes_objects_2d_list,
                    values_to_colour=radar_matrix[i, ..., m],
                    colour_map=this_colour_map_object,
                    colour_norm_object=this_colour_norm_object,
                    orientation='horizontal', extend_min=True, extend_max=True)

                this_storm_id = saliency_metadata_dict[
                    saliency_maps.STORM_IDS_KEY][i]
                this_storm_time_string = time_conversion.unix_sec_to_string(
                    saliency_metadata_dict[saliency_maps.STORM_TIMES_KEY][i],
                    TIME_FORMAT)

                this_title_string = (
                    'Saliency for storm "{0:s}" at {1:s}; {2:s}'
                ).format(this_storm_id, this_storm_time_string,
                         radar_field_names[m])
                pyplot.suptitle(this_title_string, fontsize=TITLE_FONT_SIZE)

                this_figure_file_name = (
                    '{0:s}/saliency_{1:s}_{2:s}_{3:s}.jpg'
                ).format(output_dir_name, this_storm_id.replace('_', '-'),
                         this_storm_time_string,
                         radar_field_names[m].replace('_', '-'))

                print 'Saving figure to file: "{0:s}"...'.format(
                    this_figure_file_name)
                pyplot.savefig(this_figure_file_name, dpi=DOTS_PER_INCH)
                pyplot.close()
    else:
        for i in range(num_fields):
            for m in range(num_heights):
                _, axes_objects_2d_list = plotting_utils.init_panels(
                    num_panel_rows=num_panel_rows,
                    num_panel_columns=num_panel_columns,
                    figure_width_inches=figure_width_inches,
                    figure_height_inches=figure_height_inches)

                for j in range(num_panel_rows):
                    for k in range(num_panel_columns):
                        this_storm_object_index = j * num_panel_columns + k
                        if this_storm_object_index >= num_storm_objects:
                            continue

                        this_annotation_string = '"{0:s}"\nat {1:s}'.format(
                            saliency_metadata_dict[saliency_maps.STORM_IDS_KEY][
                                this_storm_object_index],
                            time_conversion.unix_sec_to_string(
                                saliency_metadata_dict[
                                    saliency_maps.STORM_TIMES_KEY
                                ][this_storm_object_index],
                                TIME_FORMAT)
                        )

                        radar_plotting.plot_2d_grid_without_coords(
                            field_matrix=numpy.flipud(
                                radar_matrix[
                                    this_storm_object_index, ..., m, i]),
                            field_name=radar_field_names[i],
                            axes_object=axes_objects_2d_list[j][k],
                            annotation_string=this_annotation_string)

                        plot_saliency_for_radar(
                            saliency_matrix=numpy.flipud(
                                saliency_matrix[
                                    this_storm_object_index, ..., m, i]),
                            axes_object=axes_objects_2d_list[j][k],
                            option_dict=saliency_option_dict)

                (this_colour_map_object, this_colour_norm_object, _
                ) = radar_plotting.get_default_colour_scheme(
                    radar_field_names[i])

                plotting_utils.add_colour_bar(
                    axes_object_or_list=axes_objects_2d_list,
                    values_to_colour=radar_matrix[..., m, i],
                    colour_map=this_colour_map_object,
                    colour_norm_object=this_colour_norm_object,
                    orientation='horizontal', extend_min=True, extend_max=True)

                this_title_string = '{0:s} at {1:.1f} km ASL'.format(
                    radar_field_names[i], radar_heights_m_asl[m] * METRES_TO_KM)
                pyplot.suptitle(this_title_string, fontsize=TITLE_FONT_SIZE)

                this_figure_file_name = (
                    '{0:s}/saliency_{1:s}_{2:05d}metres.jpg'
                ).format(output_dir_name,
                         radar_field_names[i].replace('_', '-'),
                         radar_heights_m_asl[m])

                print 'Saving figure to file: "{0:s}"...'.format(
                    this_figure_file_name)
                pyplot.savefig(this_figure_file_name, dpi=DOTS_PER_INCH)
                pyplot.close()
