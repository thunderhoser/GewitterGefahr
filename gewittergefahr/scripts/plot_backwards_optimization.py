"""Plots results of backwards optimization."""

import copy
import os.path
import argparse
import numpy
import matplotlib
matplotlib.use('agg')
import matplotlib.colors
import matplotlib.pyplot as pyplot
from gewittergefahr.gg_utils import radar_utils
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.gg_utils import error_checking
from gewittergefahr.deep_learning import cnn
from gewittergefahr.deep_learning import deep_learning_utils as dl_utils
from gewittergefahr.deep_learning import training_validation_io as trainval_io
from gewittergefahr.deep_learning import backwards_optimization as backwards_opt
from gewittergefahr.plotting import plotting_utils
from gewittergefahr.plotting import radar_plotting
from gewittergefahr.plotting import sounding_plotting

# TODO(thunderhoser): This file contains a lot of duplicated code for
# determining output paths and titles.

TIME_FORMAT = '%Y-%m-%d-%H%M%S'
SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

TITLE_FONT_SIZE = 20
FONT_SIZE_WITH_COLOUR_BARS = 16
FONT_SIZE_SANS_COLOUR_BARS = 20
FIGURE_RESOLUTION_DPI = 300

INPUT_FILE_ARG_NAME = 'input_file_name'
COLOUR_MAP_ARG_NAME = 'diff_colour_map_name'
MAX_PERCENTILE_ARG_NAME = 'max_colour_percentile_for_diff'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

INPUT_FILE_HELP_STRING = (
    'Path to input file.  Will be read by `backwards_opt.read_standard_file` or'
    ' `backwards_opt.read_pmm_file`.')

COLOUR_MAP_HELP_STRING = (
    'Name of colour map.  Differences for each predictor will be plotted with '
    'the same colour map.  For example, if name is "Greys", the colour map used'
    ' will be `pyplot.cm.Greys`.  This argument supports only pyplot colour '
    'maps.')

MAX_PERCENTILE_HELP_STRING = (
    'Determines max absolute value for each difference map.  The max absolute '
    'value for example e and predictor p will be the [q]th percentile of all '
    'differences for example, where q = `{0:s}`.'
).format(MAX_PERCENTILE_ARG_NAME)

OUTPUT_DIR_HELP_STRING = (
    'Name of output directory.  Figures will be saved here.')

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + INPUT_FILE_ARG_NAME, type=str, required=True,
    help=INPUT_FILE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + COLOUR_MAP_ARG_NAME, type=str, required=False, default='bwr',
    help=COLOUR_MAP_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + MAX_PERCENTILE_ARG_NAME, type=float, required=False,
    default=99., help=MAX_PERCENTILE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING)


def _plot_bwo_for_2d3d_radar(
        list_of_optimized_matrices, training_option_dict,
        diff_colour_map_object, max_colour_percentile_for_diff, pmm_flag,
        bwo_metadata_dict, top_output_dir_name, list_of_input_matrices=None):
    """Plots BWO results for 2-D azimuthal-shear and 3-D reflectivity fields.

    E = number of examples (storm objects)
    T = number of input tensors to the model

    :param list_of_optimized_matrices: length-T list of numpy arrays, where the
        [i]th array is the optimized version of the [i]th input matrix to the
        model.
    :param training_option_dict: See doc for `cnn.read_model_metadata`.
    :param diff_colour_map_object: See documentation at top of file.
    :param max_colour_percentile_for_diff: Same.
    :param pmm_flag: Boolean flag.  If True, `list_of_predictor_matrices`
        contains probability-matched means.
    :param bwo_metadata_dict: Dictionary with metadata for backwards
        optimization (returned by `backwards_optimization.read_standard_file`).
    :param top_output_dir_name: Path to top-level output directory (figures will
        be saved here).
    :param list_of_input_matrices: Same as `list_of_optimized_matrices` but with
        non-optimized input matrices.
    """

    before_optimization_dir_name = '{0:s}/before_optimization'.format(
        top_output_dir_name)
    after_optimization_dir_name = '{0:s}/after_optimization'.format(
        top_output_dir_name)
    difference_dir_name = '{0:s}/after_minus_before_optimization'.format(
        top_output_dir_name)

    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=before_optimization_dir_name)
    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=after_optimization_dir_name)
    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=difference_dir_name)

    full_id_strings = bwo_metadata_dict[backwards_opt.FULL_IDS_KEY]
    storm_times_unix_sec = bwo_metadata_dict[backwards_opt.STORM_TIMES_KEY]

    if pmm_flag:
        have_storm_ids = False

        initial_activations = numpy.array([
            bwo_metadata_dict[backwards_opt.MEAN_INITIAL_ACTIVATION_KEY]
        ])
        final_activations = numpy.array([
            bwo_metadata_dict[backwards_opt.MEAN_FINAL_ACTIVATION_KEY]
        ])
    else:
        have_storm_ids = not (
            full_id_strings is None or storm_times_unix_sec is None
        )

        initial_activations = bwo_metadata_dict[
            backwards_opt.INITIAL_ACTIVATIONS_KEY]
        final_activations = bwo_metadata_dict[
            backwards_opt.FINAL_ACTIVATIONS_KEY]

    az_shear_field_names = training_option_dict[trainval_io.RADAR_FIELDS_KEY]
    num_az_shear_fields = len(az_shear_field_names)
    plot_colour_bar_flags = numpy.full(num_az_shear_fields, False, dtype=bool)

    num_storms = list_of_optimized_matrices[0].shape[0]

    for i in range(num_storms):
        print('\n')

        if pmm_flag:
            this_base_title_string = 'Probability-matched mean'
            this_base_pathless_file_name = 'pmm'
        else:
            if have_storm_ids:
                this_storm_time_string = time_conversion.unix_sec_to_string(
                    storm_times_unix_sec[i], TIME_FORMAT)

                this_base_title_string = 'Storm "{0:s}" at {1:s}'.format(
                    full_id_strings[i], this_storm_time_string)

                this_base_pathless_file_name = '{0:s}_{1:s}'.format(
                    full_id_strings[i].replace('_', '-'),
                    this_storm_time_string)

            else:
                this_base_title_string = 'Example {0:d}'.format(i + 1)
                this_base_pathless_file_name = 'example{0:06d}'.format(i)

        this_reflectivity_matrix_dbz = numpy.flip(
            list_of_optimized_matrices[0][i, ..., 0], axis=0)

        this_num_heights = this_reflectivity_matrix_dbz.shape[-1]
        this_num_panel_rows = int(numpy.floor(
            numpy.sqrt(this_num_heights)
        ))

        _, this_axes_object_matrix = radar_plotting.plot_3d_grid_without_coords(
            field_matrix=this_reflectivity_matrix_dbz,
            field_name=radar_utils.REFL_NAME,
            grid_point_heights_metres=training_option_dict[
                trainval_io.RADAR_HEIGHTS_KEY],
            ground_relative=True, num_panel_rows=this_num_panel_rows,
            font_size=FONT_SIZE_SANS_COLOUR_BARS)

        this_colour_map_object, this_colour_norm_object = (
            radar_plotting.get_default_colour_scheme(radar_utils.REFL_NAME)
        )

        plotting_utils.plot_colour_bar(
            axes_object_or_matrix=this_axes_object_matrix,
            data_matrix=this_reflectivity_matrix_dbz,
            colour_map_object=this_colour_map_object,
            colour_norm_object=this_colour_norm_object,
            orientation_string='horizontal', extend_min=True, extend_max=True)

        this_title_string = '{0:s} (AFTER; {1:.2e})'.format(
            this_base_title_string, final_activations[i]
        )

        this_file_name = (
            '{0:s}/{1:s}_after-optimization_reflectivity.jpg'
        ).format(after_optimization_dir_name, this_base_pathless_file_name)

        pyplot.suptitle(this_title_string, fontsize=TITLE_FONT_SIZE)

        print('Saving figure to: "{0:s}"...'.format(this_file_name))
        pyplot.savefig(this_file_name, dpi=FIGURE_RESOLUTION_DPI)
        pyplot.close()

        this_az_shear_matrix_s01 = numpy.flip(
            list_of_optimized_matrices[1][i, ..., 0], axis=0)

        _, this_axes_object_matrix = (
            radar_plotting.plot_many_2d_grids_without_coords(
                field_matrix=this_az_shear_matrix_s01,
                field_name_by_panel=az_shear_field_names, num_panel_rows=1,
                panel_names=az_shear_field_names,
                plot_colour_bar_by_panel=plot_colour_bar_flags,
                font_size=FONT_SIZE_SANS_COLOUR_BARS)
        )

        this_colour_map_object, this_colour_norm_object = (
            radar_plotting.get_default_colour_scheme(
                radar_utils.LOW_LEVEL_SHEAR_NAME)
        )

        plotting_utils.plot_colour_bar(
            axes_object_or_matrix=this_axes_object_matrix,
            data_matrix=this_az_shear_matrix_s01,
            colour_map_object=this_colour_map_object,
            colour_norm_object=this_colour_norm_object,
            orientation_string='horizontal', extend_min=True, extend_max=True)

        this_file_name = (
            '{0:s}/{1:s}_after-optimization_azimuthal-shear.jpg'
        ).format(after_optimization_dir_name, this_base_pathless_file_name)

        pyplot.suptitle(this_title_string, fontsize=TITLE_FONT_SIZE)

        print('Saving figure to: "{0:s}"...'.format(this_file_name))
        pyplot.savefig(this_file_name, dpi=FIGURE_RESOLUTION_DPI)
        pyplot.close()

        if list_of_input_matrices is None:
            continue

        this_reflectivity_matrix_dbz = numpy.flip(
            list_of_input_matrices[0][i, ..., 0], axis=0)

        _, this_axes_object_matrix = radar_plotting.plot_3d_grid_without_coords(
            field_matrix=this_reflectivity_matrix_dbz,
            field_name=radar_utils.REFL_NAME,
            grid_point_heights_metres=training_option_dict[
                trainval_io.RADAR_HEIGHTS_KEY],
            ground_relative=True, num_panel_rows=this_num_panel_rows,
            font_size=FONT_SIZE_SANS_COLOUR_BARS)

        this_colour_map_object, this_colour_norm_object = (
            radar_plotting.get_default_colour_scheme(radar_utils.REFL_NAME)
        )

        plotting_utils.plot_colour_bar(
            axes_object_or_matrix=this_axes_object_matrix,
            data_matrix=this_reflectivity_matrix_dbz,
            colour_map_object=this_colour_map_object,
            colour_norm_object=this_colour_norm_object,
            orientation_string='horizontal', extend_min=True, extend_max=True)

        this_title_string = '{0:s} (BEFORE; {1:.2e})'.format(
            this_base_title_string, initial_activations[i]
        )

        this_file_name = (
            '{0:s}/{1:s}_before-optimization_reflectivity.jpg'
        ).format(before_optimization_dir_name, this_base_pathless_file_name)

        pyplot.suptitle(this_title_string, fontsize=TITLE_FONT_SIZE)

        print('Saving figure to: "{0:s}"...'.format(this_file_name))
        pyplot.savefig(this_file_name, dpi=FIGURE_RESOLUTION_DPI)
        pyplot.close()

        this_az_shear_matrix_s01 = numpy.flip(
            list_of_input_matrices[1][i, ..., 0], axis=0)

        _, this_axes_object_matrix = (
            radar_plotting.plot_many_2d_grids_without_coords(
                field_matrix=this_az_shear_matrix_s01,
                field_name_by_panel=az_shear_field_names, num_panel_rows=1,
                panel_names=az_shear_field_names,
                plot_colour_bar_by_panel=plot_colour_bar_flags,
                font_size=FONT_SIZE_SANS_COLOUR_BARS)
        )

        this_colour_map_object, this_colour_norm_object = (
            radar_plotting.get_default_colour_scheme(
                radar_utils.LOW_LEVEL_SHEAR_NAME)
        )

        plotting_utils.plot_colour_bar(
            axes_object_or_matrix=this_axes_object_matrix,
            data_matrix=this_az_shear_matrix_s01,
            colour_map_object=this_colour_map_object,
            colour_norm_object=this_colour_norm_object,
            orientation_string='horizontal', extend_min=True, extend_max=True)

        this_file_name = (
            '{0:s}/{1:s}_before-optimization_azimuthal-shear.jpg'
        ).format(before_optimization_dir_name, this_base_pathless_file_name)

        pyplot.suptitle(this_title_string, fontsize=TITLE_FONT_SIZE)

        print('Saving figure to: "{0:s}"...'.format(this_file_name))
        pyplot.savefig(this_file_name, dpi=FIGURE_RESOLUTION_DPI)
        pyplot.close()

        this_refl_diff_matrix_dbz = (
            list_of_optimized_matrices[0][i, ..., 0] -
            list_of_input_matrices[0][i, ..., 0]
        )
        this_refl_diff_matrix_dbz = numpy.flip(
            this_refl_diff_matrix_dbz, axis=0)

        this_max_value_dbz = numpy.percentile(
            numpy.absolute(this_refl_diff_matrix_dbz),
            max_colour_percentile_for_diff)

        this_colour_norm_object = matplotlib.colors.Normalize(
            vmin=-1 * this_max_value_dbz, vmax=this_max_value_dbz, clip=False)

        _, this_axes_object_matrix = radar_plotting.plot_3d_grid_without_coords(
            field_matrix=this_refl_diff_matrix_dbz,
            field_name=radar_utils.REFL_NAME,
            grid_point_heights_metres=training_option_dict[
                trainval_io.RADAR_HEIGHTS_KEY],
            ground_relative=True, num_panel_rows=this_num_panel_rows,
            font_size=FONT_SIZE_SANS_COLOUR_BARS,
            colour_map_object=diff_colour_map_object,
            colour_norm_object=this_colour_norm_object)

        plotting_utils.plot_colour_bar(
            axes_object_or_matrix=this_axes_object_matrix,
            data_matrix=this_refl_diff_matrix_dbz,
            colour_map_object=this_colour_map_object,
            colour_norm_object=this_colour_norm_object,
            orientation_string='horizontal', extend_min=True, extend_max=True)

        this_title_string = '{0:s} (after minus before)'.format(
            this_base_title_string)

        this_file_name = (
            '{0:s}/{1:s}_optimization-diff_reflectivity.jpg'
        ).format(difference_dir_name, this_base_pathless_file_name)

        pyplot.suptitle(this_title_string, fontsize=TITLE_FONT_SIZE)

        print('Saving figure to: "{0:s}"...'.format(this_file_name))
        pyplot.savefig(this_file_name, dpi=FIGURE_RESOLUTION_DPI)
        pyplot.close()

        this_shear_diff_matrix_s01 = (
            list_of_optimized_matrices[1][i, ..., 0] -
            list_of_input_matrices[1][i, ..., 0]
        )
        this_shear_diff_matrix_s01 = numpy.flip(
            this_shear_diff_matrix_s01, axis=0)

        this_max_value_s01 = numpy.percentile(
            numpy.absolute(this_shear_diff_matrix_s01),
            max_colour_percentile_for_diff)

        this_colour_norm_object = matplotlib.colors.Normalize(
            vmin=-1 * this_max_value_s01, vmax=this_max_value_s01, clip=False)

        _, this_axes_object_matrix = (
            radar_plotting.plot_many_2d_grids_without_coords(
                field_matrix=this_shear_diff_matrix_s01,
                field_name_by_panel=az_shear_field_names, num_panel_rows=1,
                panel_names=az_shear_field_names,
                colour_map_object_by_panel=
                [diff_colour_map_object] * num_az_shear_fields,
                colour_norm_object_by_panel=
                [copy.deepcopy(this_colour_norm_object)] * num_az_shear_fields,
                plot_colour_bar_by_panel=plot_colour_bar_flags,
                font_size=FONT_SIZE_SANS_COLOUR_BARS)
        )

        plotting_utils.plot_colour_bar(
            axes_object_or_matrix=this_axes_object_matrix,
            data_matrix=this_shear_diff_matrix_s01,
            colour_map_object=this_colour_map_object,
            colour_norm_object=this_colour_norm_object,
            orientation_string='horizontal', extend_min=True, extend_max=True)

        this_title_string = '{0:s} (after minus before)'.format(
            this_base_title_string)

        this_file_name = (
            '{0:s}/{1:s}_optimization-diff_azimuthal-shear.jpg'
        ).format(difference_dir_name, this_base_pathless_file_name)

        pyplot.suptitle(this_title_string, fontsize=TITLE_FONT_SIZE)

        print('Saving figure to: "{0:s}"...'.format(this_file_name))
        pyplot.savefig(this_file_name, dpi=FIGURE_RESOLUTION_DPI)
        pyplot.close()


def _plot_bwo_for_3d_radar(
        optimized_radar_matrix, training_option_dict, diff_colour_map_object,
        max_colour_percentile_for_diff, pmm_flag, bwo_metadata_dict,
        top_output_dir_name, input_radar_matrix=None):
    """Plots BWO results for 3-D radar fields.

    E = number of examples (storm objects)
    M = number of rows in spatial grid
    N = number of columns in spatial grid
    H = number of heights in spatial grid
    F = number of fields

    :param optimized_radar_matrix: E-by-M-by-N-by-H-by-F numpy array of radar
        values (predictors).
    :param training_option_dict: See doc for `_plot_bwo_for_2d3d_radar`.
    :param diff_colour_map_object: Same.
    :param max_colour_percentile_for_diff: Same.
    :param pmm_flag: Same.
    :param bwo_metadata_dict: Same.
    :param top_output_dir_name: Same.
    :param input_radar_matrix: Same as `optimized_radar_matrix` but with
        non-optimized input.
    """

    before_optimization_dir_name = '{0:s}/before_optimization'.format(
        top_output_dir_name)
    after_optimization_dir_name = '{0:s}/after_optimization'.format(
        top_output_dir_name)
    difference_dir_name = '{0:s}/after_minus_before_optimization'.format(
        top_output_dir_name)

    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=before_optimization_dir_name)
    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=after_optimization_dir_name)
    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=difference_dir_name)

    full_id_strings = bwo_metadata_dict[backwards_opt.FULL_IDS_KEY]
    storm_times_unix_sec = bwo_metadata_dict[backwards_opt.STORM_TIMES_KEY]

    if pmm_flag:
        have_storm_ids = False

        initial_activations = numpy.array([
            bwo_metadata_dict[backwards_opt.MEAN_INITIAL_ACTIVATION_KEY]
        ])
        final_activations = numpy.array([
            bwo_metadata_dict[backwards_opt.MEAN_FINAL_ACTIVATION_KEY]
        ])
    else:
        have_storm_ids = not (
            full_id_strings is None or storm_times_unix_sec is None
        )

        initial_activations = bwo_metadata_dict[
            backwards_opt.INITIAL_ACTIVATIONS_KEY]
        final_activations = bwo_metadata_dict[
            backwards_opt.FINAL_ACTIVATIONS_KEY]

    radar_field_names = training_option_dict[trainval_io.RADAR_FIELDS_KEY]
    radar_heights_m_agl = training_option_dict[trainval_io.RADAR_HEIGHTS_KEY]

    num_storms = optimized_radar_matrix.shape[0]
    num_heights = optimized_radar_matrix.shape[-2]
    num_panel_rows = int(numpy.floor(
        numpy.sqrt(num_heights)
    ))

    for i in range(num_storms):
        print('\n')

        if pmm_flag:
            this_base_title_string = 'Probability-matched mean'
            this_base_pathless_file_name = 'pmm'
        else:
            if have_storm_ids:
                this_storm_time_string = time_conversion.unix_sec_to_string(
                    storm_times_unix_sec[i], TIME_FORMAT)

                this_base_title_string = 'Storm "{0:s}" at {1:s}'.format(
                    full_id_strings[i], this_storm_time_string)

                this_base_pathless_file_name = '{0:s}_{1:s}'.format(
                    full_id_strings[i].replace('_', '-'),
                    this_storm_time_string)

            else:
                this_base_title_string = 'Example {0:d}'.format(i + 1)
                this_base_pathless_file_name = 'example{0:06d}'.format(i)

        for j in range(len(radar_field_names)):
            _, this_axes_object_matrix = (
                radar_plotting.plot_3d_grid_without_coords(
                    field_matrix=numpy.flip(
                        optimized_radar_matrix[i, ..., j], axis=0),
                    field_name=radar_field_names[j],
                    grid_point_heights_metres=radar_heights_m_agl,
                    ground_relative=True, num_panel_rows=num_panel_rows,
                    font_size=FONT_SIZE_SANS_COLOUR_BARS)
            )

            this_colour_map_object, this_colour_norm_object = (
                radar_plotting.get_default_colour_scheme(
                    radar_field_names[j])
            )

            plotting_utils.plot_colour_bar(
                axes_object_or_matrix=this_axes_object_matrix,
                data_matrix=optimized_radar_matrix[i, ..., j],
                colour_map_object=this_colour_map_object,
                colour_norm_object=this_colour_norm_object,
                orientation_string='horizontal', extend_min=True,
                extend_max=True)

            this_title_string = '{0:s} (AFTER; {1:.2e})'.format(
                this_base_title_string, final_activations[i]
            )

            this_file_name = (
                '{0:s}/{1:s}_after-optimization_{2:s}.jpg'
            ).format(
                after_optimization_dir_name, this_base_pathless_file_name,
                radar_field_names[j].replace('_', '-')
            )

            pyplot.suptitle(this_title_string, fontsize=TITLE_FONT_SIZE)

            print('Saving figure to: "{0:s}"...'.format(this_file_name))
            pyplot.savefig(this_file_name, dpi=FIGURE_RESOLUTION_DPI)
            pyplot.close()

            if input_radar_matrix is None:
                continue

            _, this_axes_object_matrix = (
                radar_plotting.plot_3d_grid_without_coords(
                    field_matrix=numpy.flip(
                        input_radar_matrix[i, ..., j], axis=0),
                    field_name=radar_field_names[j],
                    grid_point_heights_metres=radar_heights_m_agl,
                    ground_relative=True, num_panel_rows=num_panel_rows,
                    font_size=FONT_SIZE_SANS_COLOUR_BARS)
            )

            this_colour_map_object, this_colour_norm_object = (
                radar_plotting.get_default_colour_scheme(
                    radar_field_names[j])
            )

            plotting_utils.plot_colour_bar(
                axes_object_or_matrix=this_axes_object_matrix,
                data_matrix=input_radar_matrix[i, ..., j],
                colour_map_object=this_colour_map_object,
                colour_norm_object=this_colour_norm_object,
                orientation_string='horizontal', extend_min=True,
                extend_max=True)

            this_title_string = '{0:s} (BEFORE; {1:.2e})'.format(
                this_base_title_string, initial_activations[i]
            )

            this_file_name = (
                '{0:s}/{1:s}_before-optimization_{2:s}.jpg'
            ).format(
                before_optimization_dir_name, this_base_pathless_file_name,
                radar_field_names[j].replace('_', '-')
            )

            pyplot.suptitle(this_title_string, fontsize=TITLE_FONT_SIZE)

            print('Saving figure to: "{0:s}"...'.format(this_file_name))
            pyplot.savefig(this_file_name, dpi=FIGURE_RESOLUTION_DPI)
            pyplot.close()

            this_diff_matrix = (
                optimized_radar_matrix[i, ..., j] -
                input_radar_matrix[i, ..., j]
            )

            this_max_value = numpy.percentile(
                numpy.absolute(this_diff_matrix),
                max_colour_percentile_for_diff)

            this_colour_norm_object = matplotlib.colors.Normalize(
                vmin=-1 * this_max_value, vmax=this_max_value, clip=False)

            _, this_axes_object_matrix = (
                radar_plotting.plot_3d_grid_without_coords(
                    field_matrix=numpy.flip(this_diff_matrix, axis=0),
                    field_name=radar_field_names[j],
                    grid_point_heights_metres=radar_heights_m_agl,
                    ground_relative=True, num_panel_rows=num_panel_rows,
                    font_size=FONT_SIZE_SANS_COLOUR_BARS,
                    colour_map_object=diff_colour_map_object,
                    colour_norm_object=this_colour_norm_object)
            )

            plotting_utils.plot_colour_bar(
                axes_object_or_matrix=this_axes_object_matrix,
                data_matrix=this_diff_matrix,
                colour_map_object=this_colour_map_object,
                colour_norm_object=this_colour_norm_object,
                orientation_string='horizontal', extend_min=True,
                extend_max=True)

            this_title_string = '{0:s} (after minus before)'.format(
                this_base_title_string)

            this_file_name = (
                '{0:s}/{1:s}_optimization-diff_{2:s}.jpg'
            ).format(
                difference_dir_name, this_base_pathless_file_name,
                radar_field_names[j].replace('_', '-')
            )

            pyplot.suptitle(this_title_string, fontsize=TITLE_FONT_SIZE)

            print('Saving figure to: "{0:s}"...'.format(this_file_name))
            pyplot.savefig(this_file_name, dpi=FIGURE_RESOLUTION_DPI)
            pyplot.close()


def _plot_bwo_for_2d_radar(
        optimized_radar_matrix, model_metadata_dict, diff_colour_map_object,
        max_colour_percentile_for_diff, pmm_flag, bwo_metadata_dict,
        top_output_dir_name, input_radar_matrix=None):
    """Plots BWO results for 2-D radar fields.

    E = number of examples (storm objects)
    M = number of rows in spatial grid
    N = number of columns in spatial grid
    C = number of channels (field/height pairs)

    :param optimized_radar_matrix: E-by-M-by-N-by-C numpy array of radar values
        (predictors).
    :param model_metadata_dict: Dictionary returned by
        `cnn.read_model_metadata`.
    :param diff_colour_map_object: See doc for `_plot_bwo_for_2d3d_radar`.
    :param max_colour_percentile_for_diff: Same.
    :param pmm_flag: Same.
    :param bwo_metadata_dict: Same.
    :param top_output_dir_name: Same.
    :param input_radar_matrix: Same as `optimized_radar_matrix` but with
        non-optimized input.
    """

    before_optimization_dir_name = '{0:s}/before_optimization'.format(
        top_output_dir_name)
    after_optimization_dir_name = '{0:s}/after_optimization'.format(
        top_output_dir_name)
    difference_dir_name = '{0:s}/after_minus_before_optimization'.format(
        top_output_dir_name)

    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=before_optimization_dir_name)
    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=after_optimization_dir_name)
    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=difference_dir_name)

    training_option_dict = model_metadata_dict[cnn.TRAINING_OPTION_DICT_KEY]
    list_of_layer_operation_dicts = model_metadata_dict[
        cnn.LAYER_OPERATIONS_KEY]

    full_id_strings = bwo_metadata_dict[backwards_opt.FULL_IDS_KEY]
    storm_times_unix_sec = bwo_metadata_dict[backwards_opt.STORM_TIMES_KEY]

    if pmm_flag:
        have_storm_ids = False

        initial_activations = numpy.array([
            bwo_metadata_dict[backwards_opt.MEAN_INITIAL_ACTIVATION_KEY]
        ])
        final_activations = numpy.array([
            bwo_metadata_dict[backwards_opt.MEAN_FINAL_ACTIVATION_KEY]
        ])
    else:
        have_storm_ids = not (
            full_id_strings is None or storm_times_unix_sec is None
        )

        initial_activations = bwo_metadata_dict[
            backwards_opt.INITIAL_ACTIVATIONS_KEY]
        final_activations = bwo_metadata_dict[
            backwards_opt.FINAL_ACTIVATIONS_KEY]

    if list_of_layer_operation_dicts is None:
        field_name_by_panel = training_option_dict[
            trainval_io.RADAR_FIELDS_KEY]

        panel_names = (
            radar_plotting.radar_fields_and_heights_to_panel_names(
                field_names=field_name_by_panel,
                heights_m_agl=training_option_dict[
                    trainval_io.RADAR_HEIGHTS_KEY]
            )
        )

        plot_colour_bar_by_panel = numpy.full(
            len(panel_names), True, dtype=bool)

    else:
        field_name_by_panel, panel_names = (
            radar_plotting.layer_ops_to_field_and_panel_names(
                list_of_layer_operation_dicts)
        )

        plot_colour_bar_by_panel = numpy.full(
            len(panel_names), False, dtype=bool)
        plot_colour_bar_by_panel[2::3] = True

    num_panels = len(panel_names)
    num_storms = optimized_radar_matrix.shape[0]
    num_channels = optimized_radar_matrix.shape[-1]
    num_panel_rows = int(numpy.floor(
        numpy.sqrt(num_channels)
    ))

    for i in range(num_storms):
        print('\n')

        if pmm_flag:
            this_base_title_string = 'Probability-matched mean'
            this_base_pathless_file_name = 'pmm'
        else:
            if have_storm_ids:
                this_storm_time_string = time_conversion.unix_sec_to_string(
                    storm_times_unix_sec[i], TIME_FORMAT)

                this_base_title_string = 'Storm "{0:s}" at {1:s}'.format(
                    full_id_strings[i], this_storm_time_string)

                this_base_pathless_file_name = '{0:s}_{1:s}'.format(
                    full_id_strings[i].replace('_', '-'),
                    this_storm_time_string)

            else:
                this_base_title_string = 'Example {0:d}'.format(i + 1)
                this_base_pathless_file_name = 'example{0:06d}'.format(i)

        radar_plotting.plot_many_2d_grids_without_coords(
            field_matrix=numpy.flip(optimized_radar_matrix[i, ...], axis=0),
            field_name_by_panel=field_name_by_panel,
            num_panel_rows=num_panel_rows, panel_names=panel_names,
            plot_colour_bar_by_panel=plot_colour_bar_by_panel,
            font_size=FONT_SIZE_WITH_COLOUR_BARS, row_major=False)

        this_title_string = '{0:s} (AFTER; activation = {1:.2e})'.format(
            this_base_title_string, final_activations[i]
        )

        this_file_name = '{0:s}/{1:s}_after-optimization_radar.jpg'.format(
            after_optimization_dir_name, this_base_pathless_file_name)

        pyplot.suptitle(this_title_string, fontsize=TITLE_FONT_SIZE)

        print('Saving figure to: "{0:s}"...'.format(this_file_name))
        pyplot.savefig(this_file_name, dpi=FIGURE_RESOLUTION_DPI)
        pyplot.close()

        if input_radar_matrix is None:
            continue

        radar_plotting.plot_many_2d_grids_without_coords(
            field_matrix=numpy.flip(input_radar_matrix[i, ...], axis=0),
            field_name_by_panel=field_name_by_panel,
            num_panel_rows=num_panel_rows, panel_names=panel_names,
            plot_colour_bar_by_panel=plot_colour_bar_by_panel,
            font_size=FONT_SIZE_WITH_COLOUR_BARS, row_major=False)

        this_title_string = '{0:s} (BEFORE; activation = {1:.2e})'.format(
            this_base_title_string, initial_activations[i]
        )

        this_file_name = '{0:s}/{1:s}_before-optimization_radar.jpg'.format(
            before_optimization_dir_name, this_base_pathless_file_name)

        pyplot.suptitle(this_title_string, fontsize=TITLE_FONT_SIZE)

        print('Saving figure to: "{0:s}"...'.format(this_file_name))
        pyplot.savefig(this_file_name, dpi=FIGURE_RESOLUTION_DPI)
        pyplot.close()

        this_cmap_object_by_panel = [diff_colour_map_object] * num_panels
        this_cnorm_object_by_panel = [None] * num_panels

        if list_of_layer_operation_dicts is None:
            for j in range(num_panels):
                this_diff_matrix = (
                    optimized_radar_matrix[i, ..., j] -
                    input_radar_matrix[i, ..., j]
                )

                this_max_value = numpy.percentile(
                    numpy.absolute(this_diff_matrix),
                    max_colour_percentile_for_diff)

                this_cnorm_object_by_panel[j] = matplotlib.colors.Normalize(
                    vmin=-1 * this_max_value, vmax=this_max_value, clip=False)

        else:
            unique_field_names = numpy.unique(numpy.array(field_name_by_panel))

            for this_field_name in unique_field_names:
                these_panel_indices = numpy.where(
                    numpy.array(field_name_by_panel) == this_field_name
                )[0]

                this_diff_matrix = (
                    optimized_radar_matrix[i, ..., these_panel_indices] -
                    input_radar_matrix[i, ..., these_panel_indices]
                )

                this_max_value = numpy.percentile(
                    numpy.absolute(this_diff_matrix),
                    max_colour_percentile_for_diff)

                for this_index in these_panel_indices:
                    this_cnorm_object_by_panel[this_index] = (
                        matplotlib.colors.Normalize(
                            vmin=-1 * this_max_value, vmax=this_max_value,
                            clip=False)
                    )

        this_diff_matrix = (
            optimized_radar_matrix[i, ...] - input_radar_matrix[i, ...]
        )

        radar_plotting.plot_many_2d_grids_without_coords(
            field_matrix=numpy.flip(this_diff_matrix, axis=0),
            field_name_by_panel=field_name_by_panel,
            num_panel_rows=num_panel_rows, panel_names=panel_names,
            colour_map_object_by_panel=this_cmap_object_by_panel,
            colour_norm_object_by_panel=this_cnorm_object_by_panel,
            plot_colour_bar_by_panel=plot_colour_bar_by_panel,
            font_size=FONT_SIZE_WITH_COLOUR_BARS, row_major=False)

        this_title_string = '{0:s} (after minus before)'.format(
            this_base_title_string)
        this_file_name = '{0:s}/{1:s}_optimization-diff_radar.jpg'.format(
            difference_dir_name, this_base_pathless_file_name)

        pyplot.suptitle(this_title_string, fontsize=TITLE_FONT_SIZE)

        print('Saving figure to: "{0:s}"...'.format(this_file_name))
        pyplot.savefig(this_file_name, dpi=FIGURE_RESOLUTION_DPI)
        pyplot.close()


def _plot_bwo_for_soundings(
        optimized_sounding_matrix, training_option_dict, pmm_flag,
        bwo_metadata_dict, top_output_dir_name, input_sounding_matrix=None):
    """Plots BWO results for soundings.

    E = number of examples (storm objects)
    H = number of sounding heights
    F = number of sounding fields

    :param optimized_sounding_matrix: E-by-H-by-F numpy array of sounding values
        (predictors).
    :param training_option_dict: See doc for `_plot_bwo_for_2d3d_radar`.
    :param pmm_flag: Same.
    :param bwo_metadata_dict: Same.
    :param top_output_dir_name: Same.
    :param input_sounding_matrix: Same as `optimized_sounding_matrix` but with
        non-optimized input.
    """

    before_optimization_dir_name = '{0:s}/before_optimization'.format(
        top_output_dir_name)
    after_optimization_dir_name = '{0:s}/after_optimization'.format(
        top_output_dir_name)

    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=before_optimization_dir_name)
    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=after_optimization_dir_name)

    full_id_strings = bwo_metadata_dict[backwards_opt.FULL_IDS_KEY]
    storm_times_unix_sec = bwo_metadata_dict[backwards_opt.STORM_TIMES_KEY]

    if pmm_flag:
        have_storm_ids = False

        initial_activations = numpy.array([
            bwo_metadata_dict[backwards_opt.MEAN_INITIAL_ACTIVATION_KEY]
        ])
        final_activations = numpy.array([
            bwo_metadata_dict[backwards_opt.MEAN_FINAL_ACTIVATION_KEY]
        ])
    else:
        have_storm_ids = not (
            full_id_strings is None or storm_times_unix_sec is None
        )

        initial_activations = bwo_metadata_dict[
            backwards_opt.INITIAL_ACTIVATIONS_KEY]
        final_activations = bwo_metadata_dict[
            backwards_opt.FINAL_ACTIVATIONS_KEY]

    num_examples = optimized_sounding_matrix.shape[0]

    list_of_optimized_metpy_dicts = dl_utils.soundings_to_metpy_dictionaries(
        sounding_matrix=optimized_sounding_matrix,
        field_names=training_option_dict[trainval_io.SOUNDING_FIELDS_KEY],
        height_levels_m_agl=training_option_dict[
            trainval_io.SOUNDING_HEIGHTS_KEY],
        storm_elevations_m_asl=numpy.zeros(num_examples)
    )

    if input_sounding_matrix is None:
        list_of_input_metpy_dicts = None
    else:
        list_of_input_metpy_dicts = dl_utils.soundings_to_metpy_dictionaries(
            sounding_matrix=input_sounding_matrix,
            field_names=training_option_dict[trainval_io.SOUNDING_FIELDS_KEY],
            height_levels_m_agl=training_option_dict[
                trainval_io.SOUNDING_HEIGHTS_KEY],
            storm_elevations_m_asl=numpy.zeros(num_examples)
        )

    for i in range(num_examples):
        if pmm_flag:
            this_base_title_string = 'Probability-matched mean'
            this_base_pathless_file_name = 'pmm'
        else:
            if have_storm_ids:
                this_storm_time_string = time_conversion.unix_sec_to_string(
                    storm_times_unix_sec[i], TIME_FORMAT)

                this_base_title_string = 'Storm "{0:s}" at {1:s}'.format(
                    full_id_strings[i], this_storm_time_string)

                this_base_pathless_file_name = '{0:s}_{1:s}'.format(
                    full_id_strings[i].replace('_', '-'),
                    this_storm_time_string)
            else:
                this_base_title_string = 'Example {0:d}'.format(i + 1)
                this_base_pathless_file_name = 'example{0:06d}'.format(i)

        this_title_string = '{0:s} (AFTER; activation = {1:.2e})'.format(
            this_base_title_string, final_activations[i]
        )

        this_file_name = '{0:s}/{1:s}_after-optimization_sounding.jpg'.format(
            after_optimization_dir_name, this_base_pathless_file_name)

        sounding_plotting.plot_sounding(
            sounding_dict_for_metpy=list_of_optimized_metpy_dicts[i],
            title_string=this_title_string)

        print('Saving figure to: "{0:s}"...'.format(this_file_name))
        pyplot.savefig(this_file_name, dpi=FIGURE_RESOLUTION_DPI)
        pyplot.close()

        if input_sounding_matrix is None:
            continue

        this_title_string = '{0:s} (BEFORE; activation = {1:.2e})'.format(
            this_base_title_string, initial_activations[i]
        )

        this_file_name = '{0:s}/{1:s}_before-optimization_sounding.jpg'.format(
            before_optimization_dir_name, this_base_pathless_file_name)

        sounding_plotting.plot_sounding(
            sounding_dict_for_metpy=list_of_input_metpy_dicts[i],
            title_string=this_title_string)

        print('Saving figure to: "{0:s}"...'.format(this_file_name))
        pyplot.savefig(this_file_name, dpi=FIGURE_RESOLUTION_DPI)
        pyplot.close()


def _run(input_file_name, diff_colour_map_name, max_colour_percentile_for_diff,
         top_output_dir_name):
    """Plots results of backwards optimization.

    This is effectively the main method.

    :param input_file_name: See documentation at top of file.
    :param diff_colour_map_name: Same.
    :param max_colour_percentile_for_diff: Same.
    :param top_output_dir_name: Same.
    """

    pmm_flag = False

    error_checking.assert_is_geq(max_colour_percentile_for_diff, 0.)
    error_checking.assert_is_leq(max_colour_percentile_for_diff, 100.)
    diff_colour_map_object = pyplot.cm.get_cmap(diff_colour_map_name)

    print('Reading data from: "{0:s}"...'.format(input_file_name))

    try:
        backwards_opt_dict = backwards_opt.read_standard_file(input_file_name)
        list_of_optimized_matrices = backwards_opt_dict.pop(
            backwards_opt.OPTIMIZED_MATRICES_KEY)
        list_of_input_matrices = backwards_opt_dict.pop(
            backwards_opt.INIT_FUNCTION_KEY)

        if not isinstance(list_of_input_matrices, list):
            list_of_input_matrices = None

        bwo_metadata_dict = backwards_opt_dict

    except ValueError:
        pmm_flag = True
        backwards_opt_dict = backwards_opt.read_pmm_file(input_file_name)

        list_of_input_matrices = backwards_opt_dict.pop(
            backwards_opt.MEAN_INPUT_MATRICES_KEY)
        list_of_optimized_matrices = backwards_opt_dict.pop(
            backwards_opt.MEAN_OPTIMIZED_MATRICES_KEY)

        for i in range(len(list_of_input_matrices)):
            list_of_input_matrices[i] = numpy.expand_dims(
                list_of_input_matrices[i], axis=0)
            list_of_optimized_matrices[i] = numpy.expand_dims(
                list_of_optimized_matrices[i], axis=0)

        bwo_metadata_dict = backwards_opt_dict
        bwo_metadata_dict[backwards_opt.FULL_IDS_KEY] = None
        bwo_metadata_dict[backwards_opt.STORM_TIMES_KEY] = None

    model_file_name = bwo_metadata_dict[backwards_opt.MODEL_FILE_KEY]
    model_metafile_name = '{0:s}/model_metadata.p'.format(
        os.path.split(model_file_name)[0]
    )

    print('Reading metadata from: "{0:s}"...'.format(model_metafile_name))
    model_metadata_dict = cnn.read_model_metadata(model_metafile_name)
    training_option_dict = model_metadata_dict[cnn.TRAINING_OPTION_DICT_KEY]
    sounding_field_names = training_option_dict[trainval_io.SOUNDING_FIELDS_KEY]

    print(SEPARATOR_STRING)

    if sounding_field_names is not None:
        if list_of_input_matrices is None:
            this_input_matrix = None
        else:
            this_input_matrix = list_of_input_matrices[-1]

        _plot_bwo_for_soundings(
            input_sounding_matrix=this_input_matrix,
            optimized_sounding_matrix=list_of_optimized_matrices[-1],
            training_option_dict=training_option_dict, pmm_flag=pmm_flag,
            bwo_metadata_dict=bwo_metadata_dict,
            top_output_dir_name=top_output_dir_name)
        print(SEPARATOR_STRING)

    if model_metadata_dict[cnn.USE_2D3D_CONVOLUTION_KEY]:
        _plot_bwo_for_2d3d_radar(
            list_of_optimized_matrices=list_of_optimized_matrices,
            training_option_dict=training_option_dict,
            diff_colour_map_object=diff_colour_map_object,
            max_colour_percentile_for_diff=max_colour_percentile_for_diff,
            pmm_flag=pmm_flag, bwo_metadata_dict=bwo_metadata_dict,
            top_output_dir_name=top_output_dir_name,
            list_of_input_matrices=list_of_input_matrices)
        return

    if list_of_input_matrices is None:
        this_input_matrix = None
    else:
        this_input_matrix = list_of_input_matrices[0]

    num_radar_dimensions = len(list_of_optimized_matrices[0].shape) - 2
    if num_radar_dimensions == 3:
        _plot_bwo_for_3d_radar(
            optimized_radar_matrix=list_of_optimized_matrices[0],
            training_option_dict=training_option_dict,
            diff_colour_map_object=diff_colour_map_object,
            max_colour_percentile_for_diff=max_colour_percentile_for_diff,
            pmm_flag=pmm_flag, bwo_metadata_dict=bwo_metadata_dict,
            top_output_dir_name=top_output_dir_name,
            input_radar_matrix=this_input_matrix)
        return

    _plot_bwo_for_2d_radar(
        optimized_radar_matrix=list_of_optimized_matrices[0],
        model_metadata_dict=model_metadata_dict,
        diff_colour_map_object=diff_colour_map_object,
        max_colour_percentile_for_diff=max_colour_percentile_for_diff,
        pmm_flag=pmm_flag, bwo_metadata_dict=bwo_metadata_dict,
        top_output_dir_name=top_output_dir_name,
        input_radar_matrix=this_input_matrix)


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        input_file_name=getattr(INPUT_ARG_OBJECT, INPUT_FILE_ARG_NAME),
        diff_colour_map_name=getattr(INPUT_ARG_OBJECT, COLOUR_MAP_ARG_NAME),
        max_colour_percentile_for_diff=getattr(
            INPUT_ARG_OBJECT, MAX_PERCENTILE_ARG_NAME),
        top_output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
