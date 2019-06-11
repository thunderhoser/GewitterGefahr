"""Plots many dataset examples (storm objects)."""

import os.path
import argparse
import numpy
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as pyplot
from gewittergefahr.gg_io import storm_tracking_io as tracking_io
from gewittergefahr.gg_utils import radar_utils
from gewittergefahr.gg_utils import soundings
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.deep_learning import cnn
from gewittergefahr.deep_learning import testing_io
from gewittergefahr.deep_learning import model_activation
from gewittergefahr.deep_learning import training_validation_io as trainval_io
from gewittergefahr.plotting import plotting_utils
from gewittergefahr.plotting import radar_plotting
from gewittergefahr.plotting import imagemagick_utils
# from gewittergefahr.plotting import sounding_plotting

TIME_FORMAT = '%Y-%m-%d-%H%M%S'
SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

SOUNDING_FIELD_NAMES = [
    soundings.U_WIND_NAME, soundings.V_WIND_NAME,
    soundings.TEMPERATURE_NAME, soundings.SPECIFIC_HUMIDITY_NAME,
    soundings.PRESSURE_NAME
]
SOUNDING_HEIGHTS_M_AGL = soundings.DEFAULT_HEIGHT_LEVELS_M_AGL

ACTIVATIONS_KEY = 'storm_activations'

TITLE_FONT_SIZE = 20
FONT_SIZE_WITH_COLOUR_BARS = 16
FONT_SIZE_SANS_COLOUR_BARS = 20

FIGURE_WIDTH_INCHES = 15
FIGURE_HEIGHT_INCHES = 15
FIGURE_RESOLUTION_DPI = 300

ACTIVATION_FILE_ARG_NAME = 'input_activation_file_name'
STORM_METAFILE_ARG_NAME = 'input_storm_metafile_name'
SAVE_PANELED_ARG_NAME = 'save_paneled_figs'
NUM_EXAMPLES_ARG_NAME = 'num_examples'
EXAMPLE_DIR_ARG_NAME = 'input_example_dir_name'
RADAR_FIELDS_ARG_NAME = 'radar_field_names'
RADAR_HEIGHTS_ARG_NAME = 'radar_heights_m_agl'
PLOT_SOUNDINGS_ARG_NAME = 'plot_soundings'
NUM_ROWS_ARG_NAME = 'num_radar_rows'
NUM_COLUMNS_ARG_NAME = 'num_radar_columns'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

ACTIVATION_FILE_HELP_STRING = (
    'Path to activation file (will be read by `model_activation.read_file`).  '
    'If this argument is empty, will use `{0:s}`.'
).format(STORM_METAFILE_ARG_NAME)

STORM_METAFILE_HELP_STRING = (
    'Path to Pickle file with storm IDs and times (will be read by '
    '`storm_tracking_io.read_ids_and_times`).  If this argument is empty, will '
    'use `{0:s}`.'
).format(ACTIVATION_FILE_ARG_NAME)

SAVE_PANELED_HELP_STRING = (
    'Boolean flag.  If 1, will save paneled figures, leading to only a few '
    'figures per storm object.  If 0, will save each 2-D or 1-D field as one '
    'figure, leading to many figures per storm object.')

NUM_EXAMPLES_HELP_STRING = (
    'Number of examples (storm objects) to read from `{0:s}` or `{1:s}`.  If '
    'you want to read all examples, make this non-positive.'
).format(ACTIVATION_FILE_ARG_NAME, STORM_METAFILE_ARG_NAME)

EXAMPLE_DIR_HELP_STRING = (
    'Name of top-level directory with input examples.  Files therein will be '
    'found by `input_examples.find_example_file` and read by '
    '`input_examples.read_example_file`.')

RADAR_FIELDS_HELP_STRING = (
    '[used only if `{0:s}` is empty] List of radar fields (used as input to '
    '`input_examples.read_example_file`).  If you want to plot all radar '
    'fields, leave this argument empty.'
).format(ACTIVATION_FILE_ARG_NAME)

RADAR_HEIGHTS_HELP_STRING = (
    '[used only if `{0:s}` is empty] List of radar heights (used as input to '
    '`input_examples.read_example_file`).  If you want to plot all radar '
    'heights, leave this argument empty.'
).format(ACTIVATION_FILE_ARG_NAME)

PLOT_SOUNDINGS_HELP_STRING = (
    'Boolean flag.  If 1, will plot sounding for each example.  If 0, will not '
    'plot soundings.')

NUM_ROWS_HELP_STRING = (
    '[used only if `{0:s}` is empty] Number of rows in each storm-centered '
    'radar grid.  If you want to plot the largest grids available, leave this '
    'argument empty.'
).format(ACTIVATION_FILE_ARG_NAME)

NUM_COLUMNS_HELP_STRING = (
    '[used only if `{0:s}` is empty] Number of columns in each storm-centered '
    'radar grid.  If you want to plot the largest grids available, leave this '
    'argument empty.'
).format(ACTIVATION_FILE_ARG_NAME)

OUTPUT_DIR_HELP_STRING = (
    'Name of output directory (figures will be saved here).')

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + ACTIVATION_FILE_ARG_NAME, type=str, required=False, default='',
    help=ACTIVATION_FILE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + STORM_METAFILE_ARG_NAME, type=str, required=False, default='',
    help=STORM_METAFILE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + SAVE_PANELED_ARG_NAME, type=int, required=False, default=0,
    help=SAVE_PANELED_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + NUM_EXAMPLES_ARG_NAME, type=int, required=False, default=-1,
    help=NUM_EXAMPLES_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + EXAMPLE_DIR_ARG_NAME, type=str, required=True,
    help=EXAMPLE_DIR_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + RADAR_FIELDS_ARG_NAME, type=str, nargs='+', required=False,
    default=[''], help=RADAR_FIELDS_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + RADAR_HEIGHTS_ARG_NAME, type=int, nargs='+', required=False,
    default=[-1], help=RADAR_HEIGHTS_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + PLOT_SOUNDINGS_ARG_NAME, type=int, required=False, default=1,
    help=PLOT_SOUNDINGS_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + NUM_ROWS_ARG_NAME, type=int, required=False, default=-1,
    help=NUM_ROWS_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + NUM_COLUMNS_ARG_NAME, type=int, required=False, default=-1,
    help=NUM_COLUMNS_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING)


def _plot_3d_examples(
        list_of_predictor_matrices, full_id_strings, storm_time_strings,
        model_metadata_dict, save_paneled_figs, output_dir_name,
        storm_activations=None):
    """Plots examples with 3-D radar data.

    E = number of examples

    :param list_of_predictor_matrices: List created by
        `testing_io.read_specific_examples`, containing data to be plotted.
    :param full_id_strings: length-E list of full storm IDs.
    :param storm_time_strings: length-E list of valid times.
    :param model_metadata_dict: Dictionary returned by
        `cnn.read_model_metadata`.
    :param save_paneled_figs: See documentation at top of file.
    :param output_dir_name: Name of output directory.
    :param storm_activations: length-E numpy array of storm activations (will be
        included in figure titles).  If this is None, it will just be skipped.
    """

    training_option_dict = model_metadata_dict[cnn.TRAINING_OPTION_DICT_KEY]

    radar_matrix = list_of_predictor_matrices[0]
    radar_field_names = training_option_dict[trainval_io.RADAR_FIELDS_KEY]
    radar_heights_m_agl = training_option_dict[trainval_io.RADAR_HEIGHTS_KEY]

    num_storm_objects = len(full_id_strings)
    num_radar_fields = len(radar_field_names)
    num_radar_heights = len(radar_heights_m_agl)

    if save_paneled_figs:
        k_max = 1
        num_panel_rows = int(numpy.floor(
            numpy.sqrt(num_radar_heights)
        ))
    else:
        k_max = num_radar_heights
        num_panel_rows = None

    # sounding_field_names = training_option_dict[trainval_io.SOUNDING_FIELDS_KEY]
    # plot_soundings = sounding_field_names is not None
    #
    # if plot_soundings:
    #     list_of_metpy_dictionaries = dl_utils.soundings_to_metpy_dictionaries(
    #         sounding_matrix=list_of_predictor_matrices[-1],
    #         field_names=sounding_field_names)
    # else:
    #     list_of_metpy_dictionaries = None

    for i in range(num_storm_objects):
        this_base_title_string = 'Storm "{0:s}" at {1:s}'.format(
            full_id_strings[i], storm_time_strings[i]
        )

        if storm_activations is not None:
            this_base_title_string += ' (activation = {0:.3f})'.format(
                storm_activations[i]
            )

        this_base_file_name = '{0:s}/storm={1:s}_{2:s}'.format(
            output_dir_name, full_id_strings[i].replace('_', '-'),
            storm_time_strings[i]
        )

        # if plot_soundings:
        #     sounding_plotting.plot_sounding(
        #         sounding_dict_for_metpy=list_of_metpy_dictionaries[i],
        #         title_string=this_base_title_string)
        #
        #     this_file_name = '{0:s}_sounding.jpg'.format(this_base_file_name)
        #
        #     print('Saving figure to: "{0:s}"...'.format(this_file_name))
        #     pyplot.savefig(this_file_name, dpi=FIGURE_RESOLUTION_DPI)
        #     pyplot.close()

        for j in range(num_radar_fields):
            for k in range(k_max):
                if save_paneled_figs:
                    this_radar_matrix = numpy.flip(
                        radar_matrix[i, ..., j], axis=0
                    )

                    _, this_axes_object_matrix = (
                        radar_plotting.plot_3d_grid_without_coords(
                            field_matrix=this_radar_matrix,
                            field_name=radar_field_names[j],
                            grid_point_heights_metres=radar_heights_m_agl,
                            ground_relative=True, num_panel_rows=num_panel_rows,
                            font_size=FONT_SIZE_SANS_COLOUR_BARS)
                    )

                    this_colour_map_object, this_colour_norm_object = (
                        radar_plotting.get_default_colour_scheme(
                            radar_field_names[j]
                        )
                    )

                    plotting_utils.plot_colour_bar(
                        axes_object_or_matrix=this_axes_object_matrix,
                        data_matrix=this_radar_matrix,
                        colour_map_object=this_colour_map_object,
                        colour_norm_object=this_colour_norm_object,
                        orientation_string='horizontal', extend_min=True,
                        extend_max=True)
                else:
                    this_radar_matrix = numpy.flip(
                        radar_matrix[i, ..., k, j], axis=0
                    )

                    _, this_axes_object = pyplot.subplots(
                        nrows=1, ncols=1,
                        figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
                    )

                    radar_plotting.plot_2d_grid_without_coords(
                        field_matrix=this_radar_matrix,
                        field_name=radar_field_names[j],
                        axes_object=this_axes_object)

                this_title_string = '{0:s}; {1:s}'.format(
                    this_base_title_string, radar_field_names[j]
                )
                this_file_name = '{0:s}_{1:s}.jpg'.format(
                    this_base_file_name, radar_field_names[j].replace('_', '-')
                )

                if not save_paneled_figs:
                    this_height_m_agl = int(numpy.round(radar_heights_m_agl[k]))

                    this_title_string += ' at {0:d} m AGL'.format(
                        this_height_m_agl)

                    this_file_name = '{0:s}_{1:05d}m-agl.jpg'.format(
                        this_file_name[:-4], this_height_m_agl
                    )

                pyplot.suptitle(this_title_string, fontsize=TITLE_FONT_SIZE)
                print('Saving figure to: "{0:s}"...'.format(this_file_name))
                pyplot.savefig(this_file_name, dpi=FIGURE_RESOLUTION_DPI)
                pyplot.close()


def _plot_2d3d_examples(
        list_of_predictor_matrices, full_id_strings, storm_time_strings,
        model_metadata_dict, save_paneled_figs, output_dir_name,
        storm_activations=None):
    """Plots examples with 2-D and 3-D radar data.

    :param list_of_predictor_matrices: See doc for `_plot_3d_examples`.
    :param full_id_strings: Same.
    :param storm_time_strings: Same.
    :param model_metadata_dict: Same.
    :param save_paneled_figs: Same.
    :param output_dir_name: Same.
    :param storm_activations: Same.
    """

    training_option_dict = model_metadata_dict[cnn.TRAINING_OPTION_DICT_KEY]

    reflectivity_matrix_dbz = list_of_predictor_matrices[0]
    azimuthal_shear_matrix_s01 = list_of_predictor_matrices[0]

    azimuthal_shear_field_names = training_option_dict[
        trainval_io.RADAR_FIELDS_KEY]
    reflectivity_heights_m_agl = training_option_dict[
        trainval_io.RADAR_HEIGHTS_KEY]

    num_storm_objects = len(full_id_strings)
    num_azimuthal_shear_fields = len(azimuthal_shear_field_names)
    num_reflectivity_heights = len(reflectivity_heights_m_agl)

    for i in range(num_storm_objects):
        this_base_title_string = 'Storm "{0:s}" at {1:s}'.format(
            full_id_strings[i], storm_time_strings[i]
        )

        if storm_activations is not None:
            this_base_title_string += ' (activation = {0:.3f})'.format(
                storm_activations[i]
            )

        this_base_file_name = '{0:s}/storm={1:s}_{2:s}'.format(
            output_dir_name, full_id_strings[i].replace('_', '-'),
            storm_time_strings[i]
        )

        if save_paneled_figs:
            this_reflectivity_matrix_dbz = numpy.flip(
                reflectivity_matrix_dbz[i, ..., 0], axis=0
            )

            this_az_shear_matrix_s01 = numpy.flip(
                azimuthal_shear_matrix_s01[i, ...], axis=0
            )

            this_num_panel_rows = int(numpy.floor(
                numpy.sqrt(num_reflectivity_heights)
            ))

            _, this_axes_object_matrix = (
                radar_plotting.plot_3d_grid_without_coords(
                    field_matrix=this_reflectivity_matrix_dbz,
                    field_name=radar_utils.REFL_NAME,
                    grid_point_heights_metres=reflectivity_heights_m_agl,
                    ground_relative=True, num_panel_rows=this_num_panel_rows,
                    font_size=FONT_SIZE_SANS_COLOUR_BARS)
            )

            this_colour_map_object, this_colour_norm_object = (
                radar_plotting.get_default_colour_scheme(radar_utils.REFL_NAME)
            )

            plotting_utils.plot_colour_bar(
                axes_object_or_matrix=this_axes_object_matrix,
                data_matrix=this_reflectivity_matrix_dbz,
                colour_map_object=this_colour_map_object,
                colour_norm_object=this_colour_norm_object,
                orientation_string='horizontal', extend_min=True,
                extend_max=True)

            this_title_string = '{0:s}; {1:s}'.format(
                this_base_title_string, radar_utils.REFL_NAME)
            this_file_name = '{0:s}_reflectivity.jpg'.format(
                this_base_file_name)

            pyplot.suptitle(this_title_string, fontsize=TITLE_FONT_SIZE)
            print('Saving figure to: "{0:s}"...'.format(this_file_name))
            pyplot.savefig(this_file_name, dpi=FIGURE_RESOLUTION_DPI)
            pyplot.close()

            _, this_axes_object_matrix = (
                radar_plotting.plot_many_2d_grids_without_coords(
                    field_matrix=this_az_shear_matrix_s01,
                    field_name_by_panel=azimuthal_shear_field_names,
                    panel_names=azimuthal_shear_field_names, num_panel_rows=1,
                    plot_colour_bar_by_panel=numpy.full(
                        num_azimuthal_shear_fields, False, dtype=bool
                    ),
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
                orientation_string='horizontal', extend_min=True,
                extend_max=True)

            pyplot.suptitle(this_base_title_string, fontsize=TITLE_FONT_SIZE)

            this_file_name = '{0:s}_shear.jpg'.format(this_base_file_name)
            print('Saving figure to: "{0:s}"...'.format(this_file_name))
            pyplot.savefig(this_file_name, dpi=FIGURE_RESOLUTION_DPI)
            pyplot.close()

            continue

        for k in range(num_reflectivity_heights):
            this_reflectivity_matrix_dbz = numpy.flip(
                reflectivity_matrix_dbz[i, ..., k, 0], axis=0
            )

            _, this_axes_object = pyplot.subplots(
                nrows=1, ncols=1,
                figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
            )

            radar_plotting.plot_2d_grid_without_coords(
                field_matrix=this_reflectivity_matrix_dbz,
                field_name=radar_utils.REFL_NAME, axes_object=this_axes_object)

            this_file_name = '{0:s}_reflectivity_{1:05d}m-agl.jpg'.format(
                this_base_file_name,
                int(numpy.round(reflectivity_heights_m_agl[k]))
            )

            print('Saving figure to: "{0:s}"...'.format(this_file_name))
            pyplot.savefig(this_file_name, dpi=FIGURE_RESOLUTION_DPI)
            pyplot.close()

            imagemagick_utils.trim_whitespace(
                input_file_name=this_file_name, output_file_name=this_file_name,
                border_width_pixels=0)

        for j in range(num_azimuthal_shear_fields):
            this_az_shear_matrix_s01 = numpy.flip(
                azimuthal_shear_matrix_s01[i, ..., j], axis=0
            )

            _, this_axes_object = pyplot.subplots(
                nrows=1, ncols=1,
                figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
            )

            radar_plotting.plot_2d_grid_without_coords(
                field_matrix=this_az_shear_matrix_s01,
                field_name=azimuthal_shear_field_names[j],
                axes_object=this_axes_object)

            this_file_name = '{0:s}_{1:s}.jpg'.format(
                this_base_file_name,
                azimuthal_shear_field_names[j].replace('_', '-')
            )

            print('Saving figure to: "{0:s}"...'.format(this_file_name))
            pyplot.savefig(this_file_name, dpi=FIGURE_RESOLUTION_DPI)
            pyplot.close()

            imagemagick_utils.trim_whitespace(
                input_file_name=this_file_name, output_file_name=this_file_name,
                border_width_pixels=0)


def _plot_2d_examples(
        list_of_predictor_matrices, full_id_strings, storm_time_strings,
        model_metadata_dict, save_paneled_figs, output_dir_name,
        storm_activations=None):
    """Plots examples with 2-D radar data.

    :param list_of_predictor_matrices: See doc for `_plot_3d_examples`.
    :param full_id_strings: Same.
    :param storm_time_strings: Same.
    :param model_metadata_dict: Same.
    :param save_paneled_figs: Same.
    :param output_dir_name: Same.
    :param storm_activations: Same.
    """

    training_option_dict = model_metadata_dict[cnn.TRAINING_OPTION_DICT_KEY]
    list_of_layer_operation_dicts = model_metadata_dict[
        cnn.LAYER_OPERATIONS_KEY]

    if list_of_layer_operation_dicts is None:
        field_name_by_panel = training_option_dict[trainval_io.RADAR_FIELDS_KEY]

        panel_names_with_units = (
            radar_plotting.radar_fields_and_heights_to_panel_names(
                field_names=field_name_by_panel,
                heights_m_agl=training_option_dict[
                    trainval_io.RADAR_HEIGHTS_KEY],
                include_units=True
            )
        )

        panel_names_sans_units = (
            radar_plotting.radar_fields_and_heights_to_panel_names(
                field_names=field_name_by_panel,
                heights_m_agl=training_option_dict[
                    trainval_io.RADAR_HEIGHTS_KEY],
                include_units=False
            )
        )

        plot_colour_bar_by_panel = numpy.full(
            len(field_name_by_panel), True, dtype=bool
        )
    else:
        field_name_by_panel, panel_names_with_units = (
            radar_plotting.layer_ops_to_field_and_panel_names(
                list_of_layer_operation_dicts=list_of_layer_operation_dicts,
                include_units=True
            )
        )

        _, panel_names_sans_units = (
            radar_plotting.layer_ops_to_field_and_panel_names(
                list_of_layer_operation_dicts=list_of_layer_operation_dicts,
                include_units=False
            )
        )

        plot_colour_bar_by_panel = numpy.full(
            len(field_name_by_panel), False, dtype=bool
        )
        plot_colour_bar_by_panel[2::3] = True

    for j in range(len(panel_names_sans_units)):
        panel_names_sans_units[j] = panel_names_sans_units[
            j
        ].replace('\n', '').replace(' ', '_').replace('_', '-')

    radar_matrix = list_of_predictor_matrices[0]
    num_storm_objects = len(full_id_strings)
    num_panels = len(field_name_by_panel)

    if save_paneled_figs:
        j_max = 1
        num_panel_rows = int(numpy.floor(
            numpy.sqrt(num_panels)
        ))
    else:
        j_max = num_panels
        num_panel_rows = None

    for i in range(num_storm_objects):
        this_title_string = 'Storm "{0:s}" at {1:s}'.format(
            full_id_strings[i], storm_time_strings[i]
        )

        if storm_activations is not None:
            this_title_string += ' (activation = {0:.3f})'.format(
                storm_activations[i]
            )

        this_base_file_name = '{0:s}/storm={1:s}_{2:s}'.format(
            output_dir_name, full_id_strings[i].replace('_', '-'),
            storm_time_strings[i]
        )

        for j in range(j_max):
            if save_paneled_figs:
                this_radar_matrix = numpy.flip(radar_matrix[i, ...], axis=0)

                radar_plotting.plot_many_2d_grids_without_coords(
                    field_matrix=this_radar_matrix,
                    field_name_by_panel=field_name_by_panel,
                    panel_names=panel_names_with_units,
                    num_panel_rows=num_panel_rows,
                    plot_colour_bar_by_panel=plot_colour_bar_by_panel,
                    font_size=FONT_SIZE_WITH_COLOUR_BARS, row_major=False)

                pyplot.suptitle(this_title_string, fontsize=TITLE_FONT_SIZE)
                this_file_name = '{0:s}.jpg'.format(this_base_file_name)
            else:
                this_radar_matrix = numpy.flip(radar_matrix[i, ..., j], axis=0)

                _, this_axes_object = pyplot.subplots(
                    nrows=1, ncols=1,
                    figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
                )

                radar_plotting.plot_2d_grid_without_coords(
                    field_matrix=this_radar_matrix,
                    field_name=field_name_by_panel[j],
                    axes_object=this_axes_object)

                this_file_name = '{0:s}_{1:s}.jpg'.format(
                    this_base_file_name, panel_names_sans_units[j]
                )

            print('Saving figure to: "{0:s}"...'.format(this_file_name))
            pyplot.savefig(this_file_name, dpi=FIGURE_RESOLUTION_DPI)
            pyplot.close()


def plot_examples(
        list_of_predictor_matrices, full_id_strings, storm_time_strings,
        model_metadata_dict, save_paneled_figs, output_dir_name,
        storm_activations=None):
    """Plots examples.

    :param list_of_predictor_matrices: See doc for `_plot_3d_examples`.
    :param full_id_strings: Same.
    :param storm_time_strings: Same.
    :param model_metadata_dict: Same.
    :param save_paneled_figs: Same.
    :param output_dir_name: Same.
    :param storm_activations: Same.
    """

    if len(list_of_predictor_matrices) == 3:
        _plot_2d3d_examples(
            list_of_predictor_matrices=list_of_predictor_matrices,
            full_id_strings=full_id_strings,
            storm_time_strings=storm_time_strings,
            model_metadata_dict=model_metadata_dict,
            save_paneled_figs=save_paneled_figs,
            output_dir_name=output_dir_name,
            storm_activations=storm_activations)

        return

    num_radar_dimensions = len(list_of_predictor_matrices[0].shape) - 2

    if num_radar_dimensions == 3:
        _plot_3d_examples(
            list_of_predictor_matrices=list_of_predictor_matrices,
            full_id_strings=full_id_strings,
            storm_time_strings=storm_time_strings,
            model_metadata_dict=model_metadata_dict,
            save_paneled_figs=save_paneled_figs,
            output_dir_name=output_dir_name,
            storm_activations=storm_activations)

        return

    _plot_2d_examples(
        list_of_predictor_matrices=list_of_predictor_matrices,
        full_id_strings=full_id_strings,
        storm_time_strings=storm_time_strings,
        model_metadata_dict=model_metadata_dict,
        save_paneled_figs=save_paneled_figs,
        output_dir_name=output_dir_name,
        storm_activations=storm_activations)


def _run(activation_file_name, storm_metafile_name, save_paneled_figs,
         num_examples, top_example_dir_name, radar_field_names,
         radar_heights_m_agl, plot_soundings, num_radar_rows, num_radar_columns,
         output_dir_name):
    """Plots many dataset examples (storm objects).

    This is effectively the main method.

    :param activation_file_name: See documentation at top of file.
    :param storm_metafile_name: Same.
    :param save_paneled_figs: Same.
    :param num_examples: Same.
    :param top_example_dir_name: Same.
    :param radar_field_names: Same.
    :param radar_heights_m_agl: Same.
    :param plot_soundings: Same.
    :param num_radar_rows: Same.
    :param num_radar_columns: Same.
    :param output_dir_name: Same.
    :raises: TypeError: if activation file contains activations for more than
        one model component.
    """

    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=output_dir_name)

    storm_activations = None
    if activation_file_name in ['', 'None']:
        activation_file_name = None

    if activation_file_name is None:
        print('Reading data from: "{0:s}"...'.format(storm_metafile_name))
        full_id_strings, storm_times_unix_sec = tracking_io.read_ids_and_times(
            storm_metafile_name)

        training_option_dict = dict()
        training_option_dict[trainval_io.RADAR_FIELDS_KEY] = radar_field_names
        training_option_dict[
            trainval_io.RADAR_HEIGHTS_KEY] = radar_heights_m_agl

        if plot_soundings:
            training_option_dict[
                trainval_io.SOUNDING_FIELDS_KEY] = SOUNDING_FIELD_NAMES
            training_option_dict[
                trainval_io.SOUNDING_HEIGHTS_KEY] = SOUNDING_HEIGHTS_M_AGL
        else:
            training_option_dict[trainval_io.SOUNDING_FIELDS_KEY] = None
            training_option_dict[trainval_io.SOUNDING_HEIGHTS_KEY] = None

        training_option_dict[trainval_io.NUM_ROWS_KEY] = num_radar_rows
        training_option_dict[trainval_io.NUM_COLUMNS_KEY] = num_radar_columns
        training_option_dict[trainval_io.NORMALIZATION_TYPE_KEY] = None
        training_option_dict[trainval_io.BINARIZE_TARGET_KEY] = False
        training_option_dict[trainval_io.SAMPLING_FRACTIONS_KEY] = None
        training_option_dict[trainval_io.REFLECTIVITY_MASK_KEY] = None

        model_metadata_dict = {
            cnn.TRAINING_OPTION_DICT_KEY: training_option_dict,
            cnn.LAYER_OPERATIONS_KEY: None,
        }

    else:
        print('Reading data from: "{0:s}"...'.format(activation_file_name))
        activation_matrix, activation_metadata_dict = (
            model_activation.read_file(activation_file_name))

        num_model_components = activation_matrix.shape[1]
        if num_model_components > 1:
            error_string = (
                'The file should contain activations for only one model '
                'component, not {0:d}.'
            ).format(num_model_components)

            raise TypeError(error_string)

        full_id_strings = activation_metadata_dict[
            model_activation.FULL_IDS_KEY]
        storm_times_unix_sec = activation_metadata_dict[
            model_activation.STORM_TIMES_KEY]
        storm_activations = activation_matrix[:, 0]

        model_file_name = activation_metadata_dict[
            model_activation.MODEL_FILE_NAME_KEY]
        model_metafile_name = '{0:s}/model_metadata.p'.format(
            os.path.split(model_file_name)[0]
        )

        print('Reading metadata from: "{0:s}"...'.format(model_metafile_name))
        model_metadata_dict = cnn.read_model_metadata(model_metafile_name)

        training_option_dict = model_metadata_dict[cnn.TRAINING_OPTION_DICT_KEY]
        training_option_dict[trainval_io.NORMALIZATION_TYPE_KEY] = None
        training_option_dict[trainval_io.SAMPLING_FRACTIONS_KEY] = None
        training_option_dict[trainval_io.REFLECTIVITY_MASK_KEY] = None

        if plot_soundings:
            training_option_dict[
                trainval_io.SOUNDING_FIELDS_KEY] = SOUNDING_FIELD_NAMES
        else:
            training_option_dict[trainval_io.SOUNDING_FIELDS_KEY] = None

        model_metadata_dict[cnn.TRAINING_OPTION_DICT_KEY] = training_option_dict

    if 0 < num_examples < len(full_id_strings):
        full_id_strings = full_id_strings[:num_examples]
        storm_times_unix_sec = storm_times_unix_sec[:num_examples]

        if storm_activations is not None:
            storm_activations = storm_activations[:num_examples]

    print(SEPARATOR_STRING)
    list_of_predictor_matrices = testing_io.read_specific_examples(
        desired_full_id_strings=full_id_strings,
        desired_times_unix_sec=storm_times_unix_sec,
        option_dict=training_option_dict,
        top_example_dir_name=top_example_dir_name,
        list_of_layer_operation_dicts=model_metadata_dict[
            cnn.LAYER_OPERATIONS_KEY]
    )[0]
    print(SEPARATOR_STRING)

    storm_time_strings = [
        time_conversion.unix_sec_to_string(t, TIME_FORMAT)
        for t in storm_times_unix_sec
    ]

    plot_examples(
        list_of_predictor_matrices=list_of_predictor_matrices,
        full_id_strings=full_id_strings,
        storm_time_strings=storm_time_strings,
        model_metadata_dict=model_metadata_dict,
        save_paneled_figs=save_paneled_figs, output_dir_name=output_dir_name,
        storm_activations=storm_activations)


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        activation_file_name=getattr(
            INPUT_ARG_OBJECT, ACTIVATION_FILE_ARG_NAME),
        storm_metafile_name=getattr(INPUT_ARG_OBJECT, STORM_METAFILE_ARG_NAME),
        save_paneled_figs=bool(getattr(
            INPUT_ARG_OBJECT, SAVE_PANELED_ARG_NAME
        )),
        num_examples=getattr(INPUT_ARG_OBJECT, NUM_EXAMPLES_ARG_NAME),
        top_example_dir_name=getattr(INPUT_ARG_OBJECT, EXAMPLE_DIR_ARG_NAME),
        radar_field_names=getattr(INPUT_ARG_OBJECT, RADAR_FIELDS_ARG_NAME),
        radar_heights_m_agl=numpy.array(
            getattr(INPUT_ARG_OBJECT, RADAR_HEIGHTS_ARG_NAME), dtype=int),
        plot_soundings=bool(getattr(INPUT_ARG_OBJECT, PLOT_SOUNDINGS_ARG_NAME)),
        num_radar_rows=getattr(INPUT_ARG_OBJECT, NUM_ROWS_ARG_NAME),
        num_radar_columns=getattr(INPUT_ARG_OBJECT, NUM_COLUMNS_ARG_NAME),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
