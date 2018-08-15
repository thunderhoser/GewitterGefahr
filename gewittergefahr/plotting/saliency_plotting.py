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

import numpy
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as pyplot
from gewittergefahr.gg_utils import radar_utils
from gewittergefahr.gg_utils import number_rounding
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.gg_utils import error_checking
from gewittergefahr.deep_learning import saliency_maps
from gewittergefahr.plotting import plotting_utils
from gewittergefahr.plotting import radar_plotting

TIME_FORMAT = '%Y-%m-%d-%H%M%S'

POSITIVE_LINE_STYLE = 'solid'
NEGATIVE_LINE_STYLE = 'dotted'
PIXEL_PADDING_FOR_CONTOUR_LABELS = 10
STRING_FORMAT_FOR_POSITIVE_LABELS = '%.3f'
STRING_FORMAT_FOR_NEGATIVE_LABELS = '-%.3f'
FONT_SIZE_FOR_CONTOUR_LABELS = 20

MAX_CONTOUR_VALUE_KEY = 'max_contour_value'
COLOUR_MAP_KEY = 'colour_map_object'
LABEL_CONTOURS_KEY = 'label_contours'
LINE_WIDTH_KEY = 'line_width'
NUM_CONTOUR_LEVELS_KEY = 'num_contour_levels'
SALIENCY_OPTION_KEYS = [
    MAX_CONTOUR_VALUE_KEY, COLOUR_MAP_KEY, LABEL_CONTOURS_KEY, LINE_WIDTH_KEY,
    NUM_CONTOUR_LEVELS_KEY
]

DEFAULT_SALIENCY_OPTION_DICT = {
    MAX_CONTOUR_VALUE_KEY: None,
    COLOUR_MAP_KEY: pyplot.cm.gist_yarg,
    LABEL_CONTOURS_KEY: False,
    LINE_WIDTH_KEY: 3,
    NUM_CONTOUR_LEVELS_KEY: 12
}

METRES_TO_KM = 1e-3
DEFAULT_FIG_WIDTH_INCHES = 15.
DEFAULT_FIG_HEIGHT_INCHES = 15.
TITLE_FONT_SIZE = 20
DOTS_PER_INCH = 300


def plot_saliency_field_2d(saliency_matrix, axes_object, option_dict):
    """Plots 2-D saliency field with unfilled, coloured contours.

    M = number of rows in spatial grid
    N = number of columns in spatial grid

    :param saliency_matrix: M-by-N numpy array of saliency values.
    :param axes_object: Instance of `matplotlib.axes._subplots.AxesSubplot`.
    :param option_dict: Dictionary with the following keys.
    option_dict['max_contour_value']: Max saliency value with a contour assigned
        to it.  Minimum saliency value will be -1 * max_contour_value.  Positive
        values will be shown with solid contours, and negative values with
        dashed contours.
    option_dict['colour_map_object']: Instance of
        `matplotlib.colors.ListedColormap`.
    option_dict['label_contours']: Boolean flag.  If True, each contour will be
        labeled with the corresponding value.
    option_dict['line_width']: Width of contour lines (scalar).
    option_dict['num_contour_levels']: Number of contour levels (i.e., number of
        saliency values corresponding to a contour).
    """

    error_checking.assert_is_numpy_array_without_nan(saliency_matrix)
    error_checking.assert_is_numpy_array(saliency_matrix, num_dimensions=2)

    try:
        max_contour_value = option_dict[MAX_CONTOUR_VALUE_KEY]
    except KeyError:
        max_contour_value = DEFAULT_SALIENCY_OPTION_DICT[MAX_CONTOUR_VALUE_KEY]

    try:
        colour_map_object = option_dict[COLOUR_MAP_KEY]
    except KeyError:
        colour_map_object = DEFAULT_SALIENCY_OPTION_DICT[COLOUR_MAP_KEY]

    try:
        label_contours = option_dict[LABEL_CONTOURS_KEY]
    except KeyError:
        label_contours = DEFAULT_SALIENCY_OPTION_DICT[LABEL_CONTOURS_KEY]

    try:
        line_width = option_dict[LINE_WIDTH_KEY]
    except KeyError:
        line_width = DEFAULT_SALIENCY_OPTION_DICT[LINE_WIDTH_KEY]

    try:
        num_contour_levels = option_dict[NUM_CONTOUR_LEVELS_KEY]
    except KeyError:
        num_contour_levels = DEFAULT_SALIENCY_OPTION_DICT[
            NUM_CONTOUR_LEVELS_KEY]

    error_checking.assert_is_greater(max_contour_value, 0.)
    error_checking.assert_is_boolean(label_contours)
    error_checking.assert_is_integer(num_contour_levels)
    error_checking.assert_is_greater(num_contour_levels, 0)
    num_contour_levels = int(
        number_rounding.ceiling_to_nearest(num_contour_levels, 2))

    positive_contour_levels = numpy.linspace(
        0., max_contour_value, num=num_contour_levels / 2 + 1)
    positive_contour_levels = positive_contour_levels[1:]
    positive_contour_object = axes_object.contour(
        saliency_matrix, levels=positive_contour_levels, cmap=colour_map_object,
        vmin=0., vmax=max_contour_value, linewidths=line_width,
        linestyles=POSITIVE_LINE_STYLE)

    if label_contours:
        pyplot.clabel(
            positive_contour_object, inline=True,
            inline_spacing=PIXEL_PADDING_FOR_CONTOUR_LABELS,
            fmt=STRING_FORMAT_FOR_POSITIVE_LABELS,
            fontsize=FONT_SIZE_FOR_CONTOUR_LABELS)

    negative_contour_object = axes_object.contour(
        -1 * saliency_matrix, levels=positive_contour_levels,
        cmap=colour_map_object, vmin=0., vmax=max_contour_value,
        linewidths=line_width, linestyles=NEGATIVE_LINE_STYLE)

    if label_contours:
        pyplot.clabel(
            negative_contour_object, inline=True,
            inline_spacing=PIXEL_PADDING_FOR_CONTOUR_LABELS,
            fmt=STRING_FORMAT_FOR_NEGATIVE_LABELS,
            fontsize=FONT_SIZE_FOR_CONTOUR_LABELS)


def plot_many_saliency_fields_2d(
        radar_field_matrix, saliency_field_matrix, saliency_metadata_dict,
        field_name_by_pair, height_by_pair_m_asl, one_fig_per_storm_object,
        num_panel_rows, output_dir_name, saliency_option_dict,
        figure_width_inches=DEFAULT_FIG_WIDTH_INCHES,
        figure_height_inches=DEFAULT_FIG_HEIGHT_INCHES):
    """Plots many 2-D saliency fields (with the underlying radar fields).

    :param radar_field_matrix: E-by-M-by-N-by-C numpy array of radar values.
    :param saliency_field_matrix: E-by-M-by-N-by-C numpy array of corresponding
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
    :param saliency_option_dict: See doc for `plot_saliency_field_2d`.
    :param figure_width_inches: Width of each figure.
    :param figure_height_inches: Height of each figure.
    """

    error_checking.assert_is_numpy_array(radar_field_matrix, num_dimensions=4)
    error_checking.assert_is_numpy_array(
        saliency_field_matrix,
        exact_dimensions=numpy.array(radar_field_matrix.shape))

    num_storm_objects = radar_field_matrix.shape[0]
    num_field_height_pairs = radar_field_matrix.shape[-1]
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
                        field_matrix=radar_field_matrix[
                            i, ..., this_fh_pair_index],
                        field_name=field_name_by_pair[this_fh_pair_index],
                        axes_object=axes_objects_2d_list[j][k],
                        annotation_string=this_annotation_string)

                    plot_saliency_field_2d(
                        saliency_matrix=saliency_field_matrix[
                            i, ..., this_fh_pair_index],
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

            this_figure_file_name = '{0:s}/saliency_{1:s}_{2:s}.jpg'.format(
                output_dir_name, this_storm_id.replace('_', '-'),
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
                        field_matrix=radar_field_matrix[
                            this_storm_object_index, ..., i],
                        field_name=field_name_by_pair[i],
                        axes_object=axes_objects_2d_list[j][k],
                        annotation_string=this_annotation_string)

                    plot_saliency_field_2d(
                        saliency_matrix=saliency_field_matrix[
                            this_storm_object_index, ..., i],
                        axes_object=axes_objects_2d_list[j][k],
                        option_dict=saliency_option_dict)

            (this_colour_map_object, this_colour_norm_object, _
            ) = radar_plotting.get_default_colour_scheme(
                field_name_by_pair[i])

            plotting_utils.add_colour_bar(
                axes_object_or_list=axes_objects_2d_list,
                values_to_colour=radar_field_matrix[..., i],
                colour_map=this_colour_map_object,
                colour_norm_object=this_colour_norm_object,
                orientation='vertical', extend_min=True, extend_max=True)

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


def plot_many_saliency_fields_3d(
        radar_field_matrix, saliency_field_matrix, saliency_metadata_dict,
        radar_field_names, radar_heights_m_asl, one_fig_per_storm_object,
        num_panel_rows, output_dir_name, saliency_option_dict,
        figure_width_inches=DEFAULT_FIG_WIDTH_INCHES,
        figure_height_inches=DEFAULT_FIG_HEIGHT_INCHES):
    """Plots many 3-D saliency fields (with the underlying radar fields).

    :param radar_field_matrix: E-by-M-by-N-by-H-by-F numpy array of radar
        values.
    :param saliency_field_matrix: E-by-M-by-N-by-H-by-F numpy array of
        corresponding saliency values.
    :param saliency_metadata_dict: Dictionary returned by
        `saliency_maps.read_file`.
    :param radar_field_names: length-F list of field names (each must be
        accepted by `radar_utils.check_field_name`).
    :param radar_heights_m_asl: length-H integer numpy array of radar heights
        (metres above sea level).
    :param one_fig_per_storm_object: See doc for `plot_many_saliency_fields_2d`.
    :param num_panel_rows: Same.
    :param output_dir_name: Same.
    :param saliency_option_dict: See doc for `plot_saliency_field_2d`.
    :param figure_width_inches: Same.
    :param figure_height_inches: Same.
    """

    error_checking.assert_is_numpy_array(radar_field_matrix, num_dimensions=5)
    error_checking.assert_is_numpy_array(
        saliency_field_matrix,
        exact_dimensions=numpy.array(radar_field_matrix.shape))

    num_storm_objects = radar_field_matrix.shape[0]
    num_fields = radar_field_matrix.shape[-1]
    num_heights = radar_field_matrix.shape[-2]
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
                            field_matrix=radar_field_matrix[
                                i, ..., this_height_index, m],
                            field_name=radar_field_names[m],
                            axes_object=axes_objects_2d_list[j][k],
                            annotation_string=this_annotation_string)

                        plot_saliency_field_2d(
                            saliency_matrix=saliency_field_matrix[
                                i, ..., this_height_index, m],
                            axes_object=axes_objects_2d_list[j][k],
                            option_dict=saliency_option_dict)

                (this_colour_map_object, this_colour_norm_object, _
                ) = radar_plotting.get_default_colour_scheme(
                    radar_field_names[m])

                plotting_utils.add_colour_bar(
                    axes_object_or_list=axes_objects_2d_list,
                    values_to_colour=radar_field_matrix[i, ..., m],
                    colour_map=this_colour_map_object,
                    colour_norm_object=this_colour_norm_object,
                    orientation='vertical', extend_min=True, extend_max=True)

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
                            field_matrix=radar_field_matrix[
                                this_storm_object_index, ..., m, i],
                            field_name=radar_field_names[i],
                            axes_object=axes_objects_2d_list[j][k],
                            annotation_string=this_annotation_string)

                        plot_saliency_field_2d(
                            saliency_matrix=saliency_field_matrix[
                                this_storm_object_index, ..., m, i],
                            axes_object=axes_objects_2d_list[j][k],
                            option_dict=saliency_option_dict)

                (this_colour_map_object, this_colour_norm_object, _
                ) = radar_plotting.get_default_colour_scheme(
                    radar_field_names[i])

                plotting_utils.add_colour_bar(
                    axes_object_or_list=axes_objects_2d_list,
                    values_to_colour=radar_field_matrix[..., m, i],
                    colour_map=this_colour_map_object,
                    colour_norm_object=this_colour_norm_object,
                    orientation='vertical', extend_min=True, extend_max=True)

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
