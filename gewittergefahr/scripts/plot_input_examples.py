"""Plots one or more examples (storm objects)."""

import copy
import os.path
import argparse
import numpy
import matplotlib
matplotlib.use('agg')
import matplotlib.colors
from matplotlib import pyplot
from gewittergefahr.gg_io import storm_tracking_io as tracking_io
from gewittergefahr.gg_utils import radar_utils
from gewittergefahr.gg_utils import soundings
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.gg_utils import error_checking
from gewittergefahr.deep_learning import cnn
from gewittergefahr.deep_learning import testing_io
from gewittergefahr.deep_learning import input_examples
from gewittergefahr.deep_learning import model_activation
from gewittergefahr.deep_learning import deep_learning_utils as dl_utils
from gewittergefahr.deep_learning import training_validation_io as trainval_io
from gewittergefahr.plotting import plotting_utils
from gewittergefahr.plotting import radar_plotting
from gewittergefahr.plotting import sounding_plotting

# TODO(thunderhoser): This is a HACK.
DUMMY_TARGET_NAME = 'tornado_lead-time=0000-3600sec_distance=00000-10000m'

TIME_FORMAT = '%Y-%m-%d-%H%M%S'
SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

SOUNDING_FIELD_NAMES = [
    soundings.U_WIND_NAME, soundings.V_WIND_NAME,
    soundings.TEMPERATURE_NAME, soundings.SPECIFIC_HUMIDITY_NAME,
    soundings.PRESSURE_NAME
]
SOUNDING_HEIGHTS_M_AGL = soundings.DEFAULT_HEIGHT_LEVELS_M_AGL

SOUNDING_FIGURE_KEY = 'sounding_figure_object'
SOUNDING_AXES_KEY = 'sounding_axes_object'
RADAR_FIGURES_KEY = 'radar_figure_objects'
RADAR_AXES_KEY = 'radar_axes_object_matrices'

IS_SOUNDING_KEY = 'is_sounding'
PMM_FLAG_KEY = 'pmm_flag'
FULL_STORM_ID_KEY = 'full_storm_id_string'
STORM_TIME_KEY = 'storm_time_unix_sec'
RADAR_FIELD_KEY = 'radar_field_name'
RADAR_HEIGHT_KEY = 'radar_height_m_agl'
LAYER_OPERATION_KEY = 'layer_operation_dict'

COLOUR_BAR_PADDING = 0.01
SHEAR_VORT_DIV_NAMES = radar_plotting.SHEAR_VORT_DIV_NAMES

DEFAULT_PANEL_NAME_FONT_SIZE = 25
DEFAULT_TITLE_FONT_SIZE = 30
DEFAULT_CBAR_FONT_SIZE = 30
DEFAULT_SOUNDING_FONT_SIZE = 30
DEFAULT_CBAR_LENGTH = 0.8
DEFAULT_REFL_OPACITY = 1.

FIGURE_WIDTH_INCHES = 15
FIGURE_HEIGHT_INCHES = 15
DEFAULT_RESOLUTION_DPI = 300

ACTIVATION_FILE_ARG_NAME = 'input_activation_file_name'
STORM_METAFILE_ARG_NAME = 'input_storm_metafile_name'
NUM_EXAMPLES_ARG_NAME = 'num_examples'
EXAMPLE_DIR_ARG_NAME = 'input_example_dir_name'
RADAR_FIELDS_ARG_NAME = 'radar_field_names'
RADAR_HEIGHTS_ARG_NAME = 'radar_heights_m_agl'
NUM_ROWS_ARG_NAME = 'num_radar_rows'
NUM_COLUMNS_ARG_NAME = 'num_radar_columns'
PLOT_SOUNDINGS_ARG_NAME = 'plot_soundings'
ALLOW_WHITESPACE_ARG_NAME = 'allow_whitespace'
PLOT_PANEL_NAMES_ARG_NAME = 'plot_panel_names'
ADD_TITLES_ARG_NAME = 'add_titles'
LABEL_CBARS_ARG_NAME = 'label_colour_bars'
CBAR_LENGTH_ARG_NAME = 'colour_bar_length'
RESOLUTION_ARG_NAME = 'figure_resolution_dpi'
REFL_OPACITY_ARG_NAME = 'refl_opacity'
PLOT_GRID_LINES_ARG_NAME = 'plot_grid_lines'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

ACTIVATION_FILE_HELP_STRING = (
    'Path to activation file (will be read by `model_activation.read_file`).  '
    'If empty, will use `{0:s}` instead.'
).format(STORM_METAFILE_ARG_NAME)

STORM_METAFILE_HELP_STRING = (
    'Path to file with storm IDs and times (will be read by `storm_tracking_io.'
    'read_ids_and_times`).  If empty, will use `{0:s}` instead.'
).format(ACTIVATION_FILE_ARG_NAME)

NUM_EXAMPLES_HELP_STRING = (
    'Number of examples to read from `{0:s}` or `{1:s}`.  To read all examples,'
    ' leave this alone.'
).format(ACTIVATION_FILE_ARG_NAME, STORM_METAFILE_ARG_NAME)

EXAMPLE_DIR_HELP_STRING = (
    'Name of top-level directory with examples.  Files therein will be found by'
    ' `input_examples.find_example_file` and read by'
    ' `input_examples.read_example_file`.')

RADAR_FIELDS_HELP_STRING = (
    '[used only if `{0:s}` is empty] List of radar fields (used as input to '
    '`input_examples.read_example_file`).  To plot all fields, leave this '
    'alone.'
).format(ACTIVATION_FILE_ARG_NAME)

RADAR_HEIGHTS_HELP_STRING = (
    '[used only if `{0:s}` is empty] List of radar heights (used as input to '
    '`input_examples.read_example_file`).  To plot all heights, leave this '
    'alone.'
).format(ACTIVATION_FILE_ARG_NAME)

NUM_ROWS_HELP_STRING = (
    '[used only if `{0:s}` is empty] Number of rows in storm-centered grid.  To'
    ' plot all rows available, leave this alone.'
).format(ACTIVATION_FILE_ARG_NAME)

NUM_COLUMNS_HELP_STRING = (
    '[used only if `{0:s}` is empty] Number of columns in storm-centered grid.'
    '  To plot all columns available, leave this alone.'
).format(ACTIVATION_FILE_ARG_NAME)

PLOT_SOUNDINGS_HELP_STRING = (
    'Boolean flag.  If 1 (0), will (not) plot sounding for each example.')

ALLOW_WHITESPACE_HELP_STRING = (
    'Boolean flag.  If 1, will not allow whitespace between panels or around '
    'outside of image.')

PLOT_PANEL_NAMES_HELP_STRING = (
    'Boolean flag.  If 1, will plot title at bottom of each panel.')

ADD_TITLES_HELP_STRING = (
    'Boolean flag.  If 1, will plot title above each figure.')

LABEL_CBARS_HELP_STRING = 'Boolean flag.  If 1, will label colour bars.'
CBAR_LENGTH_HELP_STRING = 'Length of colour bars (as fraction of axis length).'
RESOLUTION_HELP_STRING = 'Resolution of saved images (dots per inch).'
REFL_OPACITY_HELP_STRING = (
    'Opacity of colour scheme for reflectivity (in range 0...1).')

PLOT_GRID_LINES_HELP_STRING = (
    'Boolean flag.  If 1, will plot grid lines on radar images.')

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
    '--' + NUM_ROWS_ARG_NAME, type=int, required=False, default=-1,
    help=NUM_ROWS_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + NUM_COLUMNS_ARG_NAME, type=int, required=False, default=-1,
    help=NUM_COLUMNS_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + PLOT_SOUNDINGS_ARG_NAME, type=int, required=False, default=1,
    help=PLOT_SOUNDINGS_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + ALLOW_WHITESPACE_ARG_NAME, type=int, required=False, default=1,
    help=ALLOW_WHITESPACE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + PLOT_PANEL_NAMES_ARG_NAME, type=int, required=False, default=1,
    help=PLOT_PANEL_NAMES_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + ADD_TITLES_ARG_NAME, type=int, required=False, default=1,
    help=ADD_TITLES_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + LABEL_CBARS_ARG_NAME, type=int, required=False, default=0,
    help=LABEL_CBARS_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + CBAR_LENGTH_ARG_NAME, type=float, required=False,
    default=DEFAULT_CBAR_LENGTH, help=CBAR_LENGTH_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + RESOLUTION_ARG_NAME, type=int, required=False,
    default=DEFAULT_RESOLUTION_DPI, help=RESOLUTION_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + REFL_OPACITY_ARG_NAME, type=float, required=False,
    default=DEFAULT_REFL_OPACITY, help=REFL_OPACITY_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + PLOT_GRID_LINES_ARG_NAME, type=int, required=False,
    default=1, help=PLOT_GRID_LINES_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING)


def _plot_sounding(
        list_of_predictor_matrices, model_metadata_dict, pressures_pascals=None,
        font_size=DEFAULT_SOUNDING_FONT_SIZE):
    """Plots sounding for one example.

    H = number of heights

    :param list_of_predictor_matrices: See doc for `_plot_3d_radar_scan`.  In
        the sounding matrix (the last one), the second axis should have length
        H.
    :param model_metadata_dict: Same.
    :param pressures_pascals:
        [used only if `list_of_predictor_matrices` does not contain pressure]
        length-H numpy array of pressures.
    :param font_size: Font size.
    :return: figure_object: Figure handle (instance of
        `matplotlib.figure.Figure`).
    :return: axes_object: Axes handle (instance of
        `matplotlib.axes._subplots.AxesSubplot`).
    """

    training_option_dict = model_metadata_dict[cnn.TRAINING_OPTION_DICT_KEY]

    sounding_matrix = numpy.expand_dims(list_of_predictor_matrices[-1], axis=0)
    sounding_field_names = copy.deepcopy(
        training_option_dict[trainval_io.SOUNDING_FIELDS_KEY]
    )

    if soundings.PRESSURE_NAME not in sounding_field_names:
        num_heights = len(pressures_pascals)
        pressures_matrix_pascals = numpy.reshape(
            pressures_pascals, (1, num_heights, 1)
        )

        sounding_matrix = numpy.concatenate(
            (sounding_matrix, pressures_matrix_pascals), axis=-1
        )
        sounding_field_names.append(soundings.PRESSURE_NAME)

    metpy_dict = dl_utils.soundings_to_metpy_dictionaries(
        sounding_matrix=sounding_matrix, field_names=sounding_field_names
    )[0]

    figure_object, axes_object = sounding_plotting.plot_sounding(
        sounding_dict_for_metpy=metpy_dict, font_size=font_size,
        title_string=None)

    axes_object.set_xlabel(r'Temperature ($^{\circ}$C)')
    axes_object.set_ylabel('Pressure (mb)')
    return figure_object, axes_object


def _plot_3d_radar_scan(
        list_of_predictor_matrices, model_metadata_dict, allow_whitespace,
        plot_panel_names, panel_name_font_size, add_title, title_font_size,
        label_colour_bar, colour_bar_length, colour_bar_font_size,
        refl_opacity, plot_grid_lines, plot_differences, num_panel_rows=None,
        diff_colour_map_object=None, max_diff_percentile=None):
    """Plots 3-D radar images for one example.

    Specifically, this method plots one figure per field, with one panel per
    height.

    J = number of panel rows
    K = number of panel columns
    F = number of radar fields

    :param list_of_predictor_matrices: List created by
        `testing_io.read_predictors_specific_examples`, except that the first
        axis (example dimension) is missing.
    :param model_metadata_dict: Dictionary returned by
        `cnn.read_model_metadata`.
    :param allow_whitespace: Boolean flag.
    :param plot_panel_names: Boolean flag.  If True, will plot height (example:
        "3 km AGL") at bottom of each panel.
    :param panel_name_font_size: Font size for panel names.
    :param add_title: Boolean flag.  If True, title will be added with name of
        radar field.
    :param title_font_size: Font size for title.
    :param label_colour_bar: Boolean flag.  If True, colour bar will be labeled
        with name of radar field.
    :param colour_bar_length: Length of colour bars (as fraction of axis
        length).
    :param colour_bar_font_size: Font size for colour bar.
    :param refl_opacity: Opacity of colour scheme for reflectivity (in range
        0...1).
    :param plot_grid_lines: Boolean flag.  If True, will plot grid lines on
        radar images.
    :param plot_differences: Boolean flag.  If True, this method will plot
        differences rather than actual values.
    :param num_panel_rows: Number of panel rows in each figure.  If None, will
        use default.
    :param diff_colour_map_object: [used only if `plot_differences == True`]
        Colour scheme (instance of `matplotlib.pyplot.cm` or similar).
    :param max_diff_percentile: [used only if `plot_differences == True`]
        Used to set max value for colour scheme.
    :return: figure_objects: length-F list of figure handles (instances of
        `matplotlib.figure.Figure`).
    :return: axes_object_matrices: length-F list.  Each element is a J-by-K
        numpy array of axes handles (instances of
        `matplotlib.axes._subplots.AxesSubplot`).
    """

    training_option_dict = model_metadata_dict[cnn.TRAINING_OPTION_DICT_KEY]
    radar_field_names = training_option_dict[trainval_io.RADAR_FIELDS_KEY]
    radar_heights_m_agl = training_option_dict[trainval_io.RADAR_HEIGHTS_KEY]

    radar_field_names_verbose = [
        radar_plotting.field_name_to_verbose(field_name=f, include_units=True)
        for f in radar_field_names
    ]

    num_radar_fields = len(radar_field_names)
    num_radar_heights = len(radar_heights_m_agl)

    if num_panel_rows is None:
        num_panel_rows = int(numpy.floor(
            numpy.sqrt(num_radar_heights)
        ))

    num_panel_columns = int(numpy.ceil(
        float(num_radar_heights) / num_panel_rows
    ))

    figure_objects = [None] * num_radar_fields
    axes_object_matrices = [None] * num_radar_fields
    radar_matrix = list_of_predictor_matrices[0]

    for k in range(num_radar_fields):
        this_radar_matrix = numpy.flip(radar_matrix[..., k], axis=0)

        if allow_whitespace:
            figure_objects[k], axes_object_matrices[k] = (
                plotting_utils.create_paneled_figure(
                    num_rows=num_panel_rows, num_columns=num_panel_columns,
                    shared_x_axis=False, shared_y_axis=False,
                    keep_aspect_ratio=True)
            )
        else:
            figure_objects[k], axes_object_matrices[k] = (
                plotting_utils.create_paneled_figure(
                    num_rows=num_panel_rows, num_columns=num_panel_columns,
                    horizontal_spacing=0., vertical_spacing=0.,
                    shared_x_axis=False, shared_y_axis=False,
                    keep_aspect_ratio=True)
            )

        these_axes_objects = numpy.ravel(axes_object_matrices[k], order='C')

        if plot_differences:
            this_max_colour_value = numpy.percentile(
                numpy.absolute(this_radar_matrix), max_diff_percentile
            )

            this_colour_map_object = diff_colour_map_object
            this_colour_norm_object = matplotlib.colors.Normalize(
                vmin=-1 * this_max_colour_value, vmax=this_max_colour_value,
                clip=False)

            this_refl_opacity = 1.
        else:
            this_colour_map_object = None
            this_colour_norm_object = None
            this_refl_opacity = refl_opacity

        radar_plotting.plot_3d_grid(
            data_matrix=this_radar_matrix,
            axes_objects=these_axes_objects[:num_radar_heights],
            field_name=radar_field_names[k], heights_metres=radar_heights_m_agl,
            ground_relative=True, plot_panel_names=plot_panel_names,
            panel_name_font_size=panel_name_font_size,
            plot_grid_lines=plot_grid_lines,
            colour_map_object=this_colour_map_object,
            colour_norm_object=this_colour_norm_object,
            refl_opacity=this_refl_opacity)

        for j in range(num_radar_heights, len(these_axes_objects)):
            these_axes_objects[j].axis('off')

        if not allow_whitespace:
            continue

        extend_min = (
            plot_differences or radar_field_names[k] in SHEAR_VORT_DIV_NAMES
        )

        if this_colour_map_object is None:
            this_colour_map_object, this_colour_norm_object = (
                radar_plotting.get_default_colour_scheme(
                    field_name=radar_field_names[k], opacity=this_refl_opacity)
            )

        this_colour_bar_object = plotting_utils.plot_colour_bar(
            axes_object_or_matrix=axes_object_matrices[k],
            data_matrix=this_radar_matrix,
            colour_map_object=this_colour_map_object,
            colour_norm_object=this_colour_norm_object,
            orientation_string='horizontal', padding=COLOUR_BAR_PADDING,
            font_size=colour_bar_font_size,
            fraction_of_axis_length=colour_bar_length,
            extend_min=extend_min, extend_max=True)

        if label_colour_bar:
            this_colour_bar_object.set_label(
                radar_field_names_verbose[k], fontsize=colour_bar_font_size
            )

        if add_title:
            figure_objects[k].suptitle(
                radar_field_names_verbose[k], fontsize=title_font_size
            )

    return figure_objects, axes_object_matrices


def _plot_2d3d_radar_scan(
        list_of_predictor_matrices, model_metadata_dict, allow_whitespace,
        plot_panel_names, panel_name_font_size, add_titles, title_font_size,
        label_colour_bars, colour_bar_length, colour_bar_font_size,
        refl_opacity, plot_grid_lines, plot_differences, num_panel_rows=None,
        diff_colour_map_object=None, max_diff_percentile=None):
    """Plots 2-D azimuthal shear and 3-D reflectivity for one example.

    Specifically, this method plots one figure with reflectivity (one panel per
    height) and one figure with az shear (one panel per layer).

    :param list_of_predictor_matrices: See doc for `_plot_3d_radar_scan`.
    :param model_metadata_dict: Same.
    :param allow_whitespace: Same.
    :param plot_panel_names: Boolean flag.  If True, will plot height (example:
        "3 km AGL") at bottom of each reflectivity panel and field name
        [example: "Low-level shear (ks^-1)"] at bottom of each az-shear panel.
    :param panel_name_font_size: See doc for `_plot_3d_radar_scan`.
    :param add_titles: Same.
    :param title_font_size: Same.
    :param label_colour_bars: Same.
    :param colour_bar_length: Same.
    :param colour_bar_font_size: Same.
    :param refl_opacity: Same.
    :param plot_grid_lines: Same.
    :param plot_differences: Same.
    :param num_panel_rows: Same.
    :param diff_colour_map_object: Same.
    :param max_diff_percentile: Same.
    :return: figure_objects: length-2 list of figure handles (instances of
        `matplotlib.figure.Figure`).  The first is for reflectivity; the second
        is for azimuthal shear.
    :return: axes_object_matrices: length-2 list (the first is for reflectivity;
        the second is for azimuthal shear).  Each element is a 2-D numpy
        array of axes handles (instances of
        `matplotlib.axes._subplots.AxesSubplot`).
    """

    # Housekeeping.
    training_option_dict = model_metadata_dict[cnn.TRAINING_OPTION_DICT_KEY]
    shear_field_names = training_option_dict[trainval_io.RADAR_FIELDS_KEY]
    shear_field_names_verbose = [
        radar_plotting.field_name_to_verbose(field_name=f, include_units=True)
        for f in shear_field_names
    ]

    refl_heights_m_agl = training_option_dict[trainval_io.RADAR_HEIGHTS_KEY]
    refl_name_verbose = radar_plotting.field_name_to_verbose(
        field_name=radar_utils.REFL_NAME, include_units=True)

    num_refl_heights = len(refl_heights_m_agl)
    if num_panel_rows is None:
        num_panel_rows = int(numpy.floor(
            numpy.sqrt(num_refl_heights)
        ))

    num_panel_columns = int(numpy.ceil(
        float(num_refl_heights) / num_panel_rows
    ))

    # Plot reflectivity.
    if allow_whitespace:
        refl_figure_object, refl_axes_object_matrix = (
            plotting_utils.create_paneled_figure(
                num_rows=num_panel_rows, num_columns=num_panel_columns,
                shared_x_axis=False, shared_y_axis=False,
                keep_aspect_ratio=True)
        )
    else:
        refl_figure_object, refl_axes_object_matrix = (
            plotting_utils.create_paneled_figure(
                num_rows=num_panel_rows, num_columns=num_panel_columns,
                horizontal_spacing=0., vertical_spacing=0.,
                shared_x_axis=False, shared_y_axis=False,
                keep_aspect_ratio=True)
        )

    reflectivity_matrix_dbz = list_of_predictor_matrices[0][..., 0]
    these_axes_objects = numpy.ravel(refl_axes_object_matrix, order='C')

    if plot_differences:
        max_colour_value = numpy.percentile(
            numpy.absolute(reflectivity_matrix_dbz), max_diff_percentile
        )

        colour_map_object = diff_colour_map_object
        colour_norm_object = matplotlib.colors.Normalize(
            vmin=-1 * max_colour_value, vmax=max_colour_value, clip=False)

        this_refl_opacity = 1.
    else:
        colour_map_object = None
        colour_norm_object = None
        this_refl_opacity = refl_opacity

    radar_plotting.plot_3d_grid(
        data_matrix=numpy.flip(reflectivity_matrix_dbz, axis=0),
        axes_objects=these_axes_objects[:num_refl_heights],
        field_name=radar_utils.REFL_NAME, heights_metres=refl_heights_m_agl,
        ground_relative=True, plot_panel_names=plot_panel_names,
        panel_name_font_size=panel_name_font_size,
        plot_grid_lines=plot_grid_lines,
        colour_map_object=colour_map_object,
        colour_norm_object=colour_norm_object, refl_opacity=this_refl_opacity)

    for j in range(num_refl_heights, len(these_axes_objects)):
        these_axes_objects[j].axis('off')

    if allow_whitespace:
        if colour_map_object is None:
            colour_map_object, colour_norm_object = (
                radar_plotting.get_default_colour_scheme(
                    field_name=radar_utils.REFL_NAME, opacity=this_refl_opacity)
            )

        colour_bar_object = plotting_utils.plot_colour_bar(
            axes_object_or_matrix=refl_axes_object_matrix,
            data_matrix=reflectivity_matrix_dbz,
            colour_map_object=colour_map_object,
            colour_norm_object=colour_norm_object,
            orientation_string='horizontal', padding=COLOUR_BAR_PADDING,
            font_size=colour_bar_font_size,
            fraction_of_axis_length=colour_bar_length,
            extend_min=plot_differences, extend_max=True)

        if label_colour_bars:
            colour_bar_object.set_label(
                refl_name_verbose, fontsize=colour_bar_font_size)

        if add_titles:
            refl_figure_object.suptitle(
                refl_name_verbose, fontsize=title_font_size)

    # Plot azimuthal shear.
    num_shear_fields = len(shear_field_names)

    if allow_whitespace:
        shear_figure_object, shear_axes_object_matrix = (
            plotting_utils.create_paneled_figure(
                num_columns=1, num_rows=num_shear_fields,
                shared_x_axis=False, shared_y_axis=False,
                keep_aspect_ratio=True)
        )
    else:
        shear_figure_object, shear_axes_object_matrix = (
            plotting_utils.create_paneled_figure(
                num_columns=1, num_rows=num_shear_fields,
                horizontal_spacing=0., vertical_spacing=0.,
                shared_x_axis=False, shared_y_axis=False,
                keep_aspect_ratio=True)
        )

    shear_matrix_s01 = list_of_predictor_matrices[1]
    these_axes_objects = numpy.ravel(shear_axes_object_matrix, order='C')

    if plot_differences:
        max_colour_value = numpy.percentile(
            numpy.absolute(shear_matrix_s01), max_diff_percentile
        )

        colour_map_object = diff_colour_map_object
        colour_norm_object = matplotlib.colors.Normalize(
            vmin=-1 * max_colour_value, vmax=max_colour_value, clip=False)
    else:
        colour_map_object = None
        colour_norm_object = None

    radar_plotting.plot_many_2d_grids(
        data_matrix=numpy.flip(shear_matrix_s01, axis=0),
        field_names=shear_field_names, axes_objects=these_axes_objects,
        panel_names=shear_field_names_verbose if plot_panel_names else None,
        panel_name_font_size=panel_name_font_size,
        plot_grid_lines=plot_grid_lines,
        colour_map_objects=[colour_map_object] * num_shear_fields,
        colour_norm_objects=[colour_norm_object] * num_shear_fields
    )

    if allow_whitespace:
        if colour_map_object is None:
            colour_map_object, colour_norm_object = (
                radar_plotting.get_default_colour_scheme(
                    radar_utils.LOW_LEVEL_SHEAR_NAME)
            )

        colour_bar_object = plotting_utils.plot_colour_bar(
            axes_object_or_matrix=shear_axes_object_matrix,
            data_matrix=shear_matrix_s01,
            colour_map_object=colour_map_object,
            colour_norm_object=colour_norm_object,
            orientation_string='horizontal', padding=COLOUR_BAR_PADDING,
            font_size=colour_bar_font_size,
            fraction_of_axis_length=colour_bar_length / 2,
            extend_min=True, extend_max=True)

        this_label_string = r'Azimuthal shear (ks$^{-1}$)'

        if label_colour_bars:
            colour_bar_object.set_label(
                this_label_string, fontsize=colour_bar_font_size)

        if add_titles:
            shear_figure_object.suptitle(
                this_label_string, fontsize=title_font_size)

    figure_objects = [refl_figure_object, shear_figure_object]
    axes_object_matrices = [refl_axes_object_matrix, shear_axes_object_matrix]
    return figure_objects, axes_object_matrices


def _get_colour_norms_for_layer_op_diffs(
        data_matrix, list_of_layer_operation_dicts, max_colour_percentile):
    """Creates colour-normalizers for differences between layer operations.

    F = number of layer operations

    :param data_matrix: numpy array of data values (created by layer
        operations), where the last axis has length F.
    :param list_of_layer_operation_dicts: length-F list of dictionaries (each in
        format specified by `input_examples._check_layer_operation`).
    :param max_colour_percentile: Used to set max value in each colour scheme.
    :return: colour_norm_objects: length-F list of colour-normalizers (instances
        of `matplotlib.colors.BoundaryNorm` or similar).
    """

    num_operations = len(list_of_layer_operation_dicts)
    radar_field_names = [
        d[input_examples.RADAR_FIELD_KEY] for d in list_of_layer_operation_dicts
    ]

    min_heights_m_agl = numpy.array([
        d[input_examples.MIN_HEIGHT_KEY]
        for d in list_of_layer_operation_dicts
    ], dtype=int)

    max_heights_m_agl = numpy.array([
        d[input_examples.MAX_HEIGHT_KEY]
        for d in list_of_layer_operation_dicts
    ], dtype=int)

    field_layer_strings = [
        '{0:s}_{1:d}_{2:d}'.format(f, z1, z2) for f, z1, z2 in
        zip(radar_field_names, min_heights_m_agl, max_heights_m_agl)
    ]

    _, orig_to_unique_indices = numpy.unique(
        numpy.array(field_layer_strings), return_inverse=True
    )

    num_unique_schemes = int(numpy.round(
        1 + numpy.max(orig_to_unique_indices)
    ))

    colour_norm_objects = [None] * num_operations

    for i in range(num_unique_schemes):
        these_indices = numpy.where(orig_to_unique_indices == i)[0]
        this_max_value = numpy.percentile(
            numpy.absolute(data_matrix[..., these_indices]),
            max_colour_percentile
        )

        this_colour_norm_object = matplotlib.colors.Normalize(
            vmin=-1 * this_max_value, vmax=this_max_value, clip=False)

        for k in these_indices:
            colour_norm_objects[k] = this_colour_norm_object

    return colour_norm_objects


def _plot_2d_radar_scan(
        list_of_predictor_matrices, model_metadata_dict, allow_whitespace,
        plot_panel_names, panel_name_font_size, label_colour_bars,
        colour_bar_length, colour_bar_font_size, refl_opacity, plot_grid_lines,
        plot_differences, num_panel_rows=None, diff_colour_map_object=None,
        max_diff_percentile=None):
    """Plots 2-D radar scan for one example.

    Specifically, this method plots one figure, with each panel containing a
    different field-height pair.

    J = number of panel rows
    K = number of panel columns

    :param list_of_predictor_matrices: See doc for `_plot_3d_radar_scan`.
    :param model_metadata_dict: Same.
    :param allow_whitespace: Same.
    :param plot_panel_names: Boolean flag.  If True, will plot field and height
        (example: "Reflectivity at 3.0 km AGL") at bottom of each panel.
    :param panel_name_font_size: See doc for `_plot_3d_radar_scan`.
    :param label_colour_bars: Same.
    :param colour_bar_length: Same.
    :param colour_bar_font_size: Same.
    :param refl_opacity: Same.
    :param plot_grid_lines: Same.
    :param plot_differences: Same.
    :param num_panel_rows: Same.
    :param diff_colour_map_object: Same.
    :param max_diff_percentile: Same.
    :return: figure_objects: length-1 list of figure handles (instances of
        `matplotlib.figure.Figure`).
    :return: axes_object_matrices: length-1 list.  Each element is a J-by-K
        numpy array of axes handles (instances of
        `matplotlib.axes._subplots.AxesSubplot`).
    """

    training_option_dict = model_metadata_dict[cnn.TRAINING_OPTION_DICT_KEY]
    list_of_layer_operation_dicts = model_metadata_dict[
        cnn.LAYER_OPERATIONS_KEY]

    if list_of_layer_operation_dicts is None:
        radar_field_names = training_option_dict[trainval_io.RADAR_FIELDS_KEY]

        panel_names = radar_plotting.fields_and_heights_to_names(
            field_names=radar_field_names,
            heights_m_agl=training_option_dict[trainval_io.RADAR_HEIGHTS_KEY]
        )
    else:
        radar_field_names, panel_names = (
            radar_plotting.layer_operations_to_names(
                list_of_layer_operation_dicts=list_of_layer_operation_dicts
            )
        )

    num_radar_fields = len(radar_field_names)
    if num_panel_rows is None:
        num_panel_rows = int(numpy.floor(
            numpy.sqrt(num_radar_fields)
        ))

    num_panel_columns = int(numpy.ceil(
        float(num_radar_fields) / num_panel_rows
    ))

    this_flag = (
        list_of_layer_operation_dicts is not None and num_radar_fields == 12
    )

    if this_flag:
        plot_colour_bar_flags = numpy.full(num_radar_fields, 0, dtype=bool)
        plot_colour_bar_flags[2::3] = True
    else:
        plot_colour_bar_flags = numpy.full(num_radar_fields, 1, dtype=bool)

    if allow_whitespace:
        figure_object, axes_object_matrix = (
            plotting_utils.create_paneled_figure(
                num_rows=num_panel_rows, num_columns=num_panel_columns,
                shared_x_axis=False, shared_y_axis=False,
                keep_aspect_ratio=True)
        )
    else:
        figure_object, axes_object_matrix = (
            plotting_utils.create_paneled_figure(
                num_rows=num_panel_rows, num_columns=num_panel_columns,
                horizontal_spacing=0., vertical_spacing=0.,
                shared_x_axis=False, shared_y_axis=False,
                keep_aspect_ratio=True)
        )

    radar_matrix = list_of_predictor_matrices[0]
    these_axes_objects = numpy.ravel(axes_object_matrix, order='F')

    if plot_differences:
        colour_map_objects = [diff_colour_map_object] * num_radar_fields

        if list_of_layer_operation_dicts is None:
            max_colour_values = numpy.percentile(
                numpy.absolute(radar_matrix), max_diff_percentile, axis=-1
            )

            colour_norm_objects = [
                matplotlib.colors.Normalize(vmin=-1 * v, vmax=v, clip=False)
                for v in max_colour_values
            ]
        else:
            colour_norm_objects = _get_colour_norms_for_layer_op_diffs(
                data_matrix=radar_matrix,
                list_of_layer_operation_dicts=list_of_layer_operation_dicts,
                max_colour_percentile=max_diff_percentile)

        this_refl_opacity = 1.
    else:
        colour_map_objects = None
        colour_norm_objects = None
        this_refl_opacity = refl_opacity

    colour_bar_objects = radar_plotting.plot_many_2d_grids(
        data_matrix=numpy.flip(radar_matrix, axis=0),
        field_names=radar_field_names,
        axes_objects=these_axes_objects[:num_radar_fields],
        panel_names=panel_names if plot_panel_names else None,
        panel_name_font_size=panel_name_font_size,
        plot_grid_lines=plot_grid_lines,
        colour_map_objects=colour_map_objects,
        colour_norm_objects=colour_norm_objects, refl_opacity=this_refl_opacity,
        plot_colour_bar_flags=plot_colour_bar_flags,
        colour_bar_font_size=colour_bar_font_size,
        colour_bar_length=colour_bar_length)

    for k in range(num_radar_fields, len(these_axes_objects)):
        these_axes_objects[k].axis('off')

    if not label_colour_bars:
        return [figure_object], [axes_object_matrix]

    for k in range(num_radar_fields):
        this_label_string = panel_names[k].replace('Low-level', 'Azimuthal')
        this_label_string = this_label_string.replace('Mid-level', 'Azimuthal')

        colour_bar_objects[k].set_label(
            this_label_string, fontsize=colour_bar_font_size)

    return [figure_object], [axes_object_matrix]


def _append_activation_to_title(figure_object, activation, title_font_size):
    """Appends activation to figure title.

    :param figure_object: Figure handle (instance of
        `matplotlib.figure.Figure`).
    :param activation: Activation (real number).
    :param title_font_size: Font size.
    """

    title_string = '{0:s}; activation = {1:.3e}'.format(
        figure_object._suptitle.get_text(), activation
    )

    figure_object.suptitle(title_string, fontsize=title_font_size)


def metadata_to_file_name(
        output_dir_name, is_sounding, pmm_flag=False, full_storm_id_string=None,
        storm_time_unix_sec=None, radar_field_name=None,
        radar_height_m_agl=None, layer_operation_dict=None):
    """Creates name for image file.

    If `is_sounding == False and radar_field_name is None and
        layer_operation_dict is None`,
    will assume that figure contains all radar data for one example.

    :param output_dir_name: Name of output directory.
    :param is_sounding: Boolean flag, indicating whether or not figure contains
        sounding.
    :param pmm_flag: Boolean flag.  If True, output file contains a PMM
        (probability-matched mean) over many examples.  If False, contains one
        example.
    :param full_storm_id_string: [used only if `pmm_flag == False`]
        Full storm ID.
    :param storm_time_unix_sec: [used only if `pmm_flag == False`]
        Storm time.
    :param radar_field_name: Name of radar field.  May be None.
    :param radar_height_m_agl: Radar height (metres above ground level).  May be
        None.
    :param layer_operation_dict: See doc for
        `input_examples._check_layer_operation`.  May be None.
    :return: output_file_name: Path to output file.
    """

    error_checking.assert_is_string(output_dir_name)
    error_checking.assert_is_boolean(is_sounding)
    error_checking.assert_is_boolean(pmm_flag)

    if pmm_flag:
        output_file_name = '{0:s}/storm=pmm_time=0'.format(output_dir_name)
    else:
        storm_time_string = time_conversion.unix_sec_to_string(
            storm_time_unix_sec, TIME_FORMAT)

        output_file_name = '{0:s}/storm={1:s}_time={2:s}'.format(
            output_dir_name, full_storm_id_string.replace('_', '-'),
            storm_time_string
        )

    if layer_operation_dict is not None:
        radar_field_name = layer_operation_dict[input_examples.RADAR_FIELD_KEY]
        operation_name = layer_operation_dict[input_examples.OPERATION_NAME_KEY]
        min_height_m_agl = layer_operation_dict[input_examples.MIN_HEIGHT_KEY]
        max_height_m_agl = layer_operation_dict[input_examples.MAX_HEIGHT_KEY]

        return '{0:s}_{1:s}_{2:s}-{3:05d}-{4:05d}metres.jpg'.format(
            output_file_name, radar_field_name.replace('_', '-'),
            operation_name, int(numpy.round(min_height_m_agl)),
            int(numpy.round(max_height_m_agl))
        )

    if is_sounding:
        return '{0:s}_sounding.jpg'.format(output_file_name)

    if radar_field_name is None and layer_operation_dict is None:
        return '{0:s}_radar.jpg'.format(output_file_name)

    output_file_name += '_{0:s}'.format(
        radar_field_name.replace('_', '-')
    )

    if radar_height_m_agl is None:
        output_file_name += '.jpg'
    else:
        output_file_name += '_{0:05d}metres.jpg'.format(
            int(numpy.round(radar_height_m_agl))
        )

    return output_file_name


def file_name_to_metadata(figure_file_name):
    """Inverse of `metadata_to_file_name`.

    :param figure_file_name: Path to figure file with radar data.
    :return: metadata_dict: Dictionary with the following keys.
    metadata_dict['is_sounding']: See doc for `metadata_to_file_name`.
    metadata_dict['pmm_flag']: Same.
    metadata_dict['full_storm_id_string']: Same.
    metadata_dict['storm_time_unix_sec']: Same.
    metadata_dict['radar_field_name']: Same.
    metadata_dict['radar_height_m_agl']: Same.
    metadata_dict['layer_operation_dict']: Same.
    """

    pathless_file_name = os.path.split(figure_file_name)[-1]
    extensionless_file_name = os.path.splitext(pathless_file_name)[0]

    full_storm_id_string = extensionless_file_name.split('_')[0]
    full_storm_id_string = (
        full_storm_id_string.replace('storm=', '').replace('-', '_')
    )

    pmm_flag = full_storm_id_string == 'pmm'

    if pmm_flag:
        full_storm_id_string = None
        storm_time_unix_sec = None
    else:
        storm_time_string = extensionless_file_name.split('_')[1]
        storm_time_string = storm_time_string.replace('time=', '')
        storm_time_unix_sec = time_conversion.string_to_unix_sec(
            storm_time_string, TIME_FORMAT)

    metadata_dict = {
        IS_SOUNDING_KEY: False,
        PMM_FLAG_KEY: pmm_flag,
        FULL_STORM_ID_KEY: full_storm_id_string,
        STORM_TIME_KEY: storm_time_unix_sec,
        RADAR_FIELD_KEY: None,
        RADAR_HEIGHT_KEY: None,
        LAYER_OPERATION_KEY: None
    }

    field_name = extensionless_file_name.split('_')[2]
    if field_name == 'radar':
        return metadata_dict

    if field_name == 'sounding':
        metadata_dict[IS_SOUNDING_KEY] = True
        return metadata_dict

    field_name = field_name.replace('-', '_')
    metadata_dict[RADAR_FIELD_KEY] = field_name

    try:
        height_string = extensionless_file_name.split('_')[3]
    except IndexError:
        return metadata_dict

    height_string = height_string.replace('metres', '')
    height_string_parts = height_string.split('-')

    if len(height_string_parts) == 1:
        metadata_dict[RADAR_HEIGHT_KEY] = int(height_string)
        return metadata_dict

    metadata_dict[RADAR_FIELD_KEY] = None
    metadata_dict[LAYER_OPERATION_KEY] = {
        input_examples.RADAR_FIELD_KEY: field_name,
        input_examples.OPERATION_NAME_KEY: height_string_parts[0],
        input_examples.MIN_HEIGHT_KEY: int(height_string_parts[1]),
        input_examples.MAX_HEIGHT_KEY: int(height_string_parts[2])
    }

    return metadata_dict


def plot_one_example(
        list_of_predictor_matrices, model_metadata_dict, pmm_flag,
        example_index=None, plot_sounding=True, sounding_pressures_pascals=None,
        allow_whitespace=True, plot_panel_names=True,
        panel_name_font_size=DEFAULT_PANEL_NAME_FONT_SIZE,
        add_titles=True, title_font_size=DEFAULT_TITLE_FONT_SIZE,
        label_colour_bars=False, colour_bar_length=DEFAULT_CBAR_LENGTH,
        colour_bar_font_size=DEFAULT_CBAR_FONT_SIZE,
        refl_opacity=DEFAULT_REFL_OPACITY, plot_grid_lines=True,
        sounding_font_size=DEFAULT_SOUNDING_FONT_SIZE, num_panel_rows=None,
        plot_radar_diffs=False, diff_colour_map_object=None,
        max_diff_percentile=None):
    """Plots predictors for one example.

    R = number of radar figures

    :param list_of_predictor_matrices: List created by
        `testing_io.read_predictors_specific_examples`.
    :param model_metadata_dict: Dictionary returned by
        `cnn.read_model_metadata`.
    :param pmm_flag: Boolean flag.  If True, plotting probability-matched means
        (PMM) rather than single example.
    :param example_index: [used only if `pmm_flag == False`]
        Will plot the [i]th example, where i = `example_index`.
    :param plot_sounding: Boolean flag.  If True, will plot sounding (if
        available) for the given example.
    :param sounding_pressures_pascals: See doc for `_plot_soundings`.
    :param allow_whitespace: Boolean flag.  If True, will allow whitespace
        between figure panels.
    :param plot_panel_names: Boolean flag.  If True, will plot label at bottom
        of each panel.
    :param panel_name_font_size: Font size for panel names.
    :param add_titles: Boolean flag.  If True, will plot title at top of each
        figure.
    :param title_font_size: Font size for titles.
    :param label_colour_bars: Boolean flag.  If True, will label each colour
        bar.
    :param colour_bar_length: Length of colour bars (as fraction of axis
        length).
    :param colour_bar_font_size: Font size for colour bars.
    :param refl_opacity: Opacity of colour scheme for reflectivity (in range
        0...1).
    :param plot_grid_lines: Boolean flag.  If True, will plot grid lines on
        radar images.
    :param sounding_font_size: Font size for sounding.
    :param plot_radar_diffs: Boolean flag.  If True, radar matrices contain
        differences rather than actual values.
    :param num_panel_rows: Number of panel rows in each figure.  If None, will
        use default.
    :param diff_colour_map_object: [used only if `plot_radar_diffs == True`]
        Colour scheme (instance of `matplotlib.pyplot.cm` or similar).
    :param max_diff_percentile: [used only if `plot_radar_diffs == True`]
        Used to set max value for colour scheme.

    :return: handle_dict: Dictionary with the following keys.
    handle_dict['sounding_figure_object']: One figure handle (instance of
        `matplotlib.figure.Figure`).  If sounding was not plotted, this is None.
    handle_dict['sounding_axes_object']: One axes handle (instance of
        `matplotlib.axes._subplots.AxesSubplot`).  If sounding was not plotted,
        this is None.
    handle_dict['radar_figure_objects']: length-R list of figure handles
        (instances of `matplotlib.figure.Figure`).
    handle_dict['radar_axes_object_matrices']: length-R list.  Each element is a
        2-D numpy array of axes handles (instances of
        `matplotlib.axes._subplots.AxesSubplot`).
    """

    error_checking.assert_is_boolean(pmm_flag)
    error_checking.assert_is_boolean(plot_sounding)
    error_checking.assert_is_boolean(allow_whitespace)
    error_checking.assert_is_boolean(plot_panel_names)
    error_checking.assert_is_boolean(add_titles)
    error_checking.assert_is_boolean(label_colour_bars)
    error_checking.assert_is_boolean(plot_radar_diffs)

    error_checking.assert_is_greater(panel_name_font_size, 0.)
    error_checking.assert_is_greater(title_font_size, 0.)
    error_checking.assert_is_greater(colour_bar_font_size, 0.)

    if plot_radar_diffs:
        error_checking.assert_is_geq(max_diff_percentile, 90.)
        error_checking.assert_is_leq(max_diff_percentile, 100.)

    if pmm_flag:
        if list_of_predictor_matrices[0].shape[0] == 1:
            predictor_matrices_to_plot = [
                a[0, ...] for a in list_of_predictor_matrices
            ]
        else:
            predictor_matrices_to_plot = list_of_predictor_matrices
    else:
        error_checking.assert_is_integer(example_index)
        predictor_matrices_to_plot = [
            a[example_index, ...] for a in list_of_predictor_matrices
        ]

    training_option_dict = model_metadata_dict[cnn.TRAINING_OPTION_DICT_KEY]
    has_sounding = (
        training_option_dict[trainval_io.SOUNDING_FIELDS_KEY] is not None
    )

    if plot_sounding and has_sounding:
        sounding_figure_object, sounding_axes_object = _plot_sounding(
            list_of_predictor_matrices=predictor_matrices_to_plot,
            pressures_pascals=sounding_pressures_pascals,
            model_metadata_dict=model_metadata_dict,
            font_size=sounding_font_size)
    else:
        sounding_figure_object = None
        sounding_axes_object = None

    num_radar_matrices = len(predictor_matrices_to_plot) - int(has_sounding)
    num_radar_dimensions = len(predictor_matrices_to_plot[0].shape) - 1

    if num_radar_matrices == 2:
        radar_figure_objects, radar_axes_object_matrices = (
            _plot_2d3d_radar_scan(
                list_of_predictor_matrices=predictor_matrices_to_plot,
                model_metadata_dict=model_metadata_dict,
                allow_whitespace=allow_whitespace,
                plot_panel_names=plot_panel_names,
                panel_name_font_size=panel_name_font_size,
                add_titles=add_titles, title_font_size=title_font_size,
                label_colour_bars=label_colour_bars,
                colour_bar_length=colour_bar_length,
                colour_bar_font_size=colour_bar_font_size,
                refl_opacity=refl_opacity, plot_grid_lines=plot_grid_lines,
                num_panel_rows=num_panel_rows,
                plot_differences=plot_radar_diffs,
                diff_colour_map_object=diff_colour_map_object,
                max_diff_percentile=max_diff_percentile)
        )
    elif num_radar_dimensions == 3:
        radar_figure_objects, radar_axes_object_matrices = _plot_3d_radar_scan(
            list_of_predictor_matrices=predictor_matrices_to_plot,
            model_metadata_dict=model_metadata_dict,
            allow_whitespace=allow_whitespace,
            plot_panel_names=plot_panel_names,
            panel_name_font_size=panel_name_font_size,
            add_title=add_titles, title_font_size=title_font_size,
            label_colour_bar=label_colour_bars,
            colour_bar_length=colour_bar_length,
            colour_bar_font_size=colour_bar_font_size,
            refl_opacity=refl_opacity, plot_grid_lines=plot_grid_lines,
            num_panel_rows=num_panel_rows, plot_differences=plot_radar_diffs,
            diff_colour_map_object=diff_colour_map_object,
            max_diff_percentile=max_diff_percentile)
    else:
        radar_figure_objects, radar_axes_object_matrices = _plot_2d_radar_scan(
            list_of_predictor_matrices=predictor_matrices_to_plot,
            model_metadata_dict=model_metadata_dict,
            allow_whitespace=allow_whitespace,
            plot_panel_names=plot_panel_names,
            panel_name_font_size=panel_name_font_size,
            label_colour_bars=label_colour_bars,
            colour_bar_length=colour_bar_length,
            colour_bar_font_size=colour_bar_font_size,
            refl_opacity=refl_opacity, plot_grid_lines=plot_grid_lines,
            num_panel_rows=num_panel_rows, plot_differences=plot_radar_diffs,
            diff_colour_map_object=diff_colour_map_object,
            max_diff_percentile=max_diff_percentile)

    return {
        SOUNDING_FIGURE_KEY: sounding_figure_object,
        SOUNDING_AXES_KEY: sounding_axes_object,
        RADAR_FIGURES_KEY: radar_figure_objects,
        RADAR_AXES_KEY: radar_axes_object_matrices
    }


def plot_examples(
        list_of_predictor_matrices, model_metadata_dict, pmm_flag,
        output_dir_name, plot_soundings=True,
        sounding_pressure_matrix_pascals=None, allow_whitespace=True,
        plot_panel_names=True,
        panel_name_font_size=DEFAULT_PANEL_NAME_FONT_SIZE,
        add_titles=True, title_font_size=DEFAULT_TITLE_FONT_SIZE,
        label_colour_bars=False, colour_bar_length=DEFAULT_CBAR_LENGTH,
        colour_bar_font_size=DEFAULT_CBAR_FONT_SIZE,
        figure_resolution_dpi=DEFAULT_RESOLUTION_DPI,
        refl_opacity=DEFAULT_REFL_OPACITY, plot_grid_lines=True,
        sounding_font_size=DEFAULT_SOUNDING_FONT_SIZE, num_panel_rows=None,
        plot_radar_diffs=False, diff_colour_map_object=None,
        max_diff_percentile=None, full_storm_id_strings=None,
        storm_times_unix_sec=None, storm_activations=None):
    """Plots predictors for each example.

    E = number of examples (storm objects)
    H_s = number of sounding heights

    :param list_of_predictor_matrices: See doc for `plot_one_example`.
    :param model_metadata_dict: Same.
    :param pmm_flag: Same.
    :param output_dir_name: Path to output directory.  Figures will be saved
        here.
    :param plot_soundings: See doc for `plot_one_example`.
    :param sounding_pressure_matrix_pascals:
        [used only if `plot_soundings == True` and `list_of_predictor_matrices`
        does not contain sounding pressure]
        numpy array (E x H_s) of pressures.
    :param allow_whitespace: See doc for `plot_one_example`.
    :param plot_panel_names: Same.
    :param panel_name_font_size: Same.
    :param add_titles: Same.
    :param title_font_size: Same.
    :param label_colour_bars: Same.
    :param colour_bar_length: Same.
    :param colour_bar_font_size: Same.
    :param figure_resolution_dpi: Resolution of saved images (dots per inch).
    :param refl_opacity: See doc for `plot_one_example`.
    :param plot_grid_lines: Same.
    :param sounding_font_size: Same.
    :param num_panel_rows: Same.
    :param plot_radar_diffs: Same.
    :param diff_colour_map_object: Same.
    :param max_diff_percentile: Same.
    :param full_storm_id_strings: [used only if `pmm_flag == False`]
        length-E list of storm IDs.
    :param storm_times_unix_sec: [used only if `pmm_flag == False`]
        length-E numpy array of valid times.
    :param storm_activations: [used only if `pmm_flag == False`]
        length-E numpy array of model activations.  This may be None.  If not
        None, will be used in titles.
    :return: figure_file_names: Paths to file saved by this method.
    """

    error_checking.assert_is_boolean(pmm_flag)
    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=output_dir_name)

    if pmm_flag:
        num_examples = 1
        full_storm_id_strings = [None]
        storm_times_unix_sec = [None]
    else:
        num_examples = list_of_predictor_matrices[0].shape[0]
        these_expected_dim = numpy.array([num_examples], dtype=int)

        error_checking.assert_is_string_list(full_storm_id_strings)
        error_checking.assert_is_numpy_array(
            numpy.array(full_storm_id_strings),
            exact_dimensions=these_expected_dim
        )

        error_checking.assert_is_integer_numpy_array(storm_times_unix_sec)
        error_checking.assert_is_numpy_array(
            storm_times_unix_sec, exact_dimensions=these_expected_dim)

    if storm_activations is not None:
        these_expected_dim = numpy.array([num_examples], dtype=int)
        error_checking.assert_is_numpy_array_without_nan(storm_activations)
        error_checking.assert_is_numpy_array(
            storm_activations, exact_dimensions=these_expected_dim)

    training_option_dict = model_metadata_dict[cnn.TRAINING_OPTION_DICT_KEY]
    has_soundings = (
        training_option_dict[trainval_io.SOUNDING_FIELDS_KEY] is not None
    )

    num_radar_matrices = len(list_of_predictor_matrices) - int(has_soundings)
    num_radar_dimensions = len(list_of_predictor_matrices[0].shape) - 2

    figure_file_names = []

    for i in range(num_examples):
        if sounding_pressure_matrix_pascals is None:
            these_pressures_pascals = None
        else:
            these_pressures_pascals = sounding_pressure_matrix_pascals[i, ...]

        this_handle_dict = plot_one_example(
            list_of_predictor_matrices=list_of_predictor_matrices,
            model_metadata_dict=model_metadata_dict, pmm_flag=pmm_flag,
            example_index=i, plot_sounding=plot_soundings,
            sounding_pressures_pascals=these_pressures_pascals,
            allow_whitespace=allow_whitespace,
            plot_panel_names=plot_panel_names,
            panel_name_font_size=panel_name_font_size,
            add_titles=add_titles, title_font_size=title_font_size,
            label_colour_bars=label_colour_bars,
            colour_bar_length=colour_bar_length,
            colour_bar_font_size=colour_bar_font_size,
            refl_opacity=refl_opacity, plot_grid_lines=plot_grid_lines,
            sounding_font_size=sounding_font_size,
            num_panel_rows=num_panel_rows, plot_radar_diffs=plot_radar_diffs,
            diff_colour_map_object=diff_colour_map_object,
            max_diff_percentile=max_diff_percentile)

        this_sounding_figure_object = this_handle_dict[SOUNDING_FIGURE_KEY]

        if this_sounding_figure_object is not None:
            if add_titles and storm_activations is not None:
                this_title_string = 'Activation = {0:.3e}'.format(
                    storm_activations[i]
                )

                this_handle_dict[SOUNDING_AXES_KEY].set_title(
                    this_title_string, fontsize=sounding_font_size)

            this_file_name = metadata_to_file_name(
                output_dir_name=output_dir_name, is_sounding=True,
                pmm_flag=pmm_flag,
                full_storm_id_string=full_storm_id_strings[i],
                storm_time_unix_sec=storm_times_unix_sec[i]
            )

            figure_file_names.append(this_file_name)

            print('Saving figure to: "{0:s}"...'.format(this_file_name))
            this_sounding_figure_object.savefig(
                figure_file_names[-1], dpi=figure_resolution_dpi, pad_inches=0,
                bbox_inches='tight'
            )
            pyplot.close(this_sounding_figure_object)

        these_radar_figure_objects = this_handle_dict[RADAR_FIGURES_KEY]

        if num_radar_matrices == 2:
            if add_titles and storm_activations is not None:
                _append_activation_to_title(
                    figure_object=these_radar_figure_objects[0],
                    activation=storm_activations[i],
                    title_font_size=title_font_size
                )

            this_file_name = metadata_to_file_name(
                output_dir_name=output_dir_name, is_sounding=False,
                pmm_flag=pmm_flag,
                full_storm_id_string=full_storm_id_strings[i],
                storm_time_unix_sec=storm_times_unix_sec[i],
                radar_field_name='reflectivity')

            figure_file_names.append(this_file_name)

            print('Saving figure to: "{0:s}"...'.format(this_file_name))
            these_radar_figure_objects[0].savefig(
                figure_file_names[-1], dpi=figure_resolution_dpi, pad_inches=0,
                bbox_inches='tight'
            )
            pyplot.close(these_radar_figure_objects[0])

            if add_titles and storm_activations is not None:
                _append_activation_to_title(
                    figure_object=these_radar_figure_objects[1],
                    activation=storm_activations[i],
                    title_font_size=title_font_size
                )

            this_file_name = metadata_to_file_name(
                output_dir_name=output_dir_name, is_sounding=False,
                pmm_flag=pmm_flag,
                full_storm_id_string=full_storm_id_strings[i],
                storm_time_unix_sec=storm_times_unix_sec[i],
                radar_field_name='shear')

            figure_file_names.append(this_file_name)

            print('Saving figure to: "{0:s}"...'.format(this_file_name))
            these_radar_figure_objects[1].savefig(
                figure_file_names[-1], dpi=figure_resolution_dpi, pad_inches=0,
                bbox_inches='tight'
            )
            pyplot.close(these_radar_figure_objects[1])

            continue

        if num_radar_dimensions == 3:
            radar_field_names = training_option_dict[
                trainval_io.RADAR_FIELDS_KEY]

            for j in range(len(radar_field_names)):
                if add_titles and storm_activations is not None:
                    _append_activation_to_title(
                        figure_object=these_radar_figure_objects[j],
                        activation=storm_activations[i],
                        title_font_size=title_font_size
                    )

                this_file_name = metadata_to_file_name(
                    output_dir_name=output_dir_name, is_sounding=False,
                    pmm_flag=pmm_flag,
                    full_storm_id_string=full_storm_id_strings[i],
                    storm_time_unix_sec=storm_times_unix_sec[i],
                    radar_field_name=radar_field_names[j],
                    radar_height_m_agl=None)

                figure_file_names.append(this_file_name)

                print('Saving figure to: "{0:s}"...'.format(this_file_name))
                these_radar_figure_objects[j].savefig(
                    figure_file_names[-1], dpi=figure_resolution_dpi,
                    pad_inches=0, bbox_inches='tight'
                )
                pyplot.close(these_radar_figure_objects[j])

            continue

        if add_titles and storm_activations is not None:
            _append_activation_to_title(
                figure_object=these_radar_figure_objects[0],
                activation=storm_activations[i],
                title_font_size=title_font_size
            )

        this_file_name = metadata_to_file_name(
            output_dir_name=output_dir_name, is_sounding=False,
            pmm_flag=pmm_flag,
            full_storm_id_string=full_storm_id_strings[i],
            storm_time_unix_sec=storm_times_unix_sec[i]
        )

        figure_file_names.append(this_file_name)

        print('Saving figure to: "{0:s}"...'.format(this_file_name))
        these_radar_figure_objects[0].savefig(
            figure_file_names[-1], dpi=figure_resolution_dpi, pad_inches=0,
            bbox_inches='tight'
        )
        pyplot.close(these_radar_figure_objects[0])

    return figure_file_names


def _run(activation_file_name, storm_metafile_name, num_examples,
         top_example_dir_name, radar_field_names, radar_heights_m_agl,
         num_radar_rows, num_radar_columns, plot_soundings, allow_whitespace,
         plot_panel_names, add_titles, label_colour_bars, colour_bar_length,
         figure_resolution_dpi, refl_opacity, plot_grid_lines, output_dir_name):
    """Plots one or more examples (storm objects).

    This is effectively the main method.

    :param activation_file_name: See documentation at top of file.
    :param storm_metafile_name: Same.
    :param num_examples: Same.
    :param top_example_dir_name: Same.
    :param radar_field_names: Same.
    :param radar_heights_m_agl: Same.
    :param num_radar_rows: Same.
    :param num_radar_columns: Same.
    :param plot_soundings: Same.
    :param allow_whitespace: Same.
    :param plot_panel_names: Same.
    :param add_titles: Same.
    :param label_colour_bars: Same.
    :param colour_bar_length: Same.
    :param figure_resolution_dpi: Same.
    :param refl_opacity: Same.
    :param plot_grid_lines: Same.
    :param output_dir_name: Same.
    :raises: ValueError: if activation file contains activations for more than
        one model component.
    """

    if num_radar_rows <= 0:
        num_radar_rows = None
    if num_radar_columns <= 0:
        num_radar_columns = None

    storm_activations = None
    if activation_file_name in ['', 'None']:
        activation_file_name = None

    if activation_file_name is None:
        print('Reading data from: "{0:s}"...'.format(storm_metafile_name))
        full_storm_id_strings, storm_times_unix_sec = (
            tracking_io.read_ids_and_times(storm_metafile_name)
        )

        training_option_dict = dict()
        training_option_dict[trainval_io.RADAR_FIELDS_KEY] = radar_field_names
        training_option_dict[
            trainval_io.RADAR_HEIGHTS_KEY] = radar_heights_m_agl
        training_option_dict[
            trainval_io.SOUNDING_FIELDS_KEY] = SOUNDING_FIELD_NAMES
        training_option_dict[
            trainval_io.SOUNDING_HEIGHTS_KEY] = SOUNDING_HEIGHTS_M_AGL

        training_option_dict[trainval_io.NUM_ROWS_KEY] = num_radar_rows
        training_option_dict[trainval_io.NUM_COLUMNS_KEY] = num_radar_columns
        training_option_dict[trainval_io.NORMALIZATION_TYPE_KEY] = None
        training_option_dict[trainval_io.TARGET_NAME_KEY] = DUMMY_TARGET_NAME
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
            model_activation.read_file(activation_file_name)
        )

        num_model_components = activation_matrix.shape[1]
        if num_model_components > 1:
            error_string = (
                'The file should contain activations for only one model '
                'component, not {0:d}.'
            ).format(num_model_components)

            raise TypeError(error_string)

        full_storm_id_strings = activation_metadata_dict[
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

        model_metadata_dict[cnn.TRAINING_OPTION_DICT_KEY] = training_option_dict

    model_metadata_dict[cnn.TRAINING_OPTION_DICT_KEY][
        trainval_io.UPSAMPLE_REFLECTIVITY_KEY
    ] = False

    if 0 < num_examples < len(full_storm_id_strings):
        full_storm_id_strings = full_storm_id_strings[:num_examples]
        storm_times_unix_sec = storm_times_unix_sec[:num_examples]
        if storm_activations is not None:
            storm_activations = storm_activations[:num_examples]

    print(SEPARATOR_STRING)
    example_dict = testing_io.read_predictors_specific_examples(
        top_example_dir_name=top_example_dir_name,
        desired_full_id_strings=full_storm_id_strings,
        desired_times_unix_sec=storm_times_unix_sec,
        option_dict=model_metadata_dict[cnn.TRAINING_OPTION_DICT_KEY],
        layer_operation_dicts=model_metadata_dict[cnn.LAYER_OPERATIONS_KEY]
    )
    print(SEPARATOR_STRING)

    predictor_matrices = example_dict[testing_io.INPUT_MATRICES_KEY]
    sounding_pressure_matrix_pa = example_dict[
        testing_io.SOUNDING_PRESSURES_KEY]

    plot_examples(
        list_of_predictor_matrices=predictor_matrices,
        model_metadata_dict=model_metadata_dict, pmm_flag=False,
        output_dir_name=output_dir_name, plot_soundings=plot_soundings,
        sounding_pressure_matrix_pascals=sounding_pressure_matrix_pa,
        allow_whitespace=allow_whitespace, plot_panel_names=plot_panel_names,
        add_titles=add_titles, label_colour_bars=label_colour_bars,
        colour_bar_length=colour_bar_length,
        figure_resolution_dpi=figure_resolution_dpi,
        refl_opacity=refl_opacity, plot_grid_lines=plot_grid_lines,
        full_storm_id_strings=full_storm_id_strings,
        storm_times_unix_sec=storm_times_unix_sec,
        storm_activations=storm_activations)


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        activation_file_name=getattr(
            INPUT_ARG_OBJECT, ACTIVATION_FILE_ARG_NAME),
        storm_metafile_name=getattr(INPUT_ARG_OBJECT, STORM_METAFILE_ARG_NAME),
        num_examples=getattr(INPUT_ARG_OBJECT, NUM_EXAMPLES_ARG_NAME),
        top_example_dir_name=getattr(INPUT_ARG_OBJECT, EXAMPLE_DIR_ARG_NAME),
        radar_field_names=getattr(INPUT_ARG_OBJECT, RADAR_FIELDS_ARG_NAME),
        radar_heights_m_agl=numpy.array(
            getattr(INPUT_ARG_OBJECT, RADAR_HEIGHTS_ARG_NAME), dtype=int
        ),
        num_radar_rows=getattr(INPUT_ARG_OBJECT, NUM_ROWS_ARG_NAME),
        num_radar_columns=getattr(INPUT_ARG_OBJECT, NUM_COLUMNS_ARG_NAME),
        plot_soundings=bool(getattr(INPUT_ARG_OBJECT, PLOT_SOUNDINGS_ARG_NAME)),
        allow_whitespace=bool(getattr(
            INPUT_ARG_OBJECT, ALLOW_WHITESPACE_ARG_NAME
        )),
        plot_panel_names=bool(getattr(
            INPUT_ARG_OBJECT, PLOT_PANEL_NAMES_ARG_NAME
        )),
        add_titles=bool(getattr(INPUT_ARG_OBJECT, ADD_TITLES_ARG_NAME)),
        label_colour_bars=bool(getattr(
            INPUT_ARG_OBJECT, LABEL_CBARS_ARG_NAME
        )),
        colour_bar_length=getattr(INPUT_ARG_OBJECT, CBAR_LENGTH_ARG_NAME),
        figure_resolution_dpi=getattr(INPUT_ARG_OBJECT, RESOLUTION_ARG_NAME),
        refl_opacity=getattr(INPUT_ARG_OBJECT, REFL_OPACITY_ARG_NAME),
        plot_grid_lines=bool(getattr(
            INPUT_ARG_OBJECT, PLOT_GRID_LINES_ARG_NAME
        )),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
