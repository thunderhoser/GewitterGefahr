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
LAYER_OP_INDICES_TO_KEEP = numpy.array([2, 3, 7, 10], dtype=int)
DUMMY_TARGET_NAME = 'tornado_lead-time=0000-3600sec_distance=00000-10000m'

TIME_FORMAT = '%Y-%m-%d-%H%M%S'
SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

SOUNDING_FIELD_NAMES = [
    soundings.U_WIND_NAME, soundings.V_WIND_NAME,
    soundings.TEMPERATURE_NAME, soundings.SPECIFIC_HUMIDITY_NAME,
    soundings.PRESSURE_NAME
]
SOUNDING_HEIGHTS_M_AGL = soundings.DEFAULT_HEIGHT_LEVELS_M_AGL

ACTIVATIONS_KEY = 'storm_activations'

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

TITLE_FONT_SIZE = 16
FONT_SIZE_WITH_COLOUR_BARS = 16
FONT_SIZE_SANS_COLOUR_BARS = 20

FIGURE_WIDTH_INCHES = 15
FIGURE_HEIGHT_INCHES = 15
FIGURE_RESOLUTION_DPI = 300

ACTIVATION_FILE_ARG_NAME = 'input_activation_file_name'
STORM_METAFILE_ARG_NAME = 'input_storm_metafile_name'
NUM_EXAMPLES_ARG_NAME = 'num_examples'
ALLOW_WHITESPACE_ARG_NAME = 'allow_whitespace'
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

NUM_EXAMPLES_HELP_STRING = (
    'Number of examples (storm objects) to read from `{0:s}` or `{1:s}`.  If '
    'you want to read all examples, make this non-positive.'
).format(ACTIVATION_FILE_ARG_NAME, STORM_METAFILE_ARG_NAME)

ALLOW_WHITESPACE_HELP_STRING = (
    'Boolean flag.  If 0, will plot with no whitespace between panels or around'
    ' outside of image.')

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
    '--' + NUM_EXAMPLES_ARG_NAME, type=int, required=False, default=-1,
    help=NUM_EXAMPLES_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + ALLOW_WHITESPACE_ARG_NAME, type=int, required=False, default=1,
    help=ALLOW_WHITESPACE_HELP_STRING)

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


def _plot_sounding(
        list_of_predictor_matrices, model_metadata_dict, allow_whitespace,
        title_string=None):
    """Plots sounding for one example.

    :param list_of_predictor_matrices: See doc for `_plot_3d_radar_scan`.
    :param model_metadata_dict: Same.
    :param allow_whitespace: Same.
    :param title_string: Same.
    :return: figure_object: Figure handle (instance of
        `matplotlib.figure.Figure`).
    :return: axes_object: Axes handle (instance of
        `matplotlib.axes._subplots.AxesSubplot`).
    """

    training_option_dict = model_metadata_dict[cnn.TRAINING_OPTION_DICT_KEY]
    sounding_field_names = training_option_dict[trainval_io.SOUNDING_FIELDS_KEY]
    sounding_matrix = numpy.expand_dims(list_of_predictor_matrices[-1], axis=0)

    metpy_dict = dl_utils.soundings_to_metpy_dictionaries(
        sounding_matrix=sounding_matrix, field_names=sounding_field_names
    )[0]

    return sounding_plotting.plot_sounding(
        sounding_dict_for_metpy=metpy_dict,
        title_string=title_string if allow_whitespace else None
    )


def _plot_3d_radar_scan(
        list_of_predictor_matrices, model_metadata_dict, allow_whitespace,
        title_string=None):
    """Plots 3-D radar scan for one example.

    J = number of panel rows in image
    K = number of panel columns in image
    F = number of radar fields

    :param list_of_predictor_matrices: List created by
        `testing_io.read_specific_examples`, except that the first axis (example
        dimension) is removed.
    :param model_metadata_dict: Dictionary returned by
        `cnn.read_model_metadata`.
    :param allow_whitespace: See documentation at top of file.
    :param title_string: Title (may be None).

    :return: figure_objects: length-F list of figure handles (instances of
        `matplotlib.figure.Figure`).
    :return: axes_object_matrices: length-F list.  Each element is a J-by-K
        numpy array of axes handles (instances of
        `matplotlib.axes._subplots.AxesSubplot`).
    """

    training_option_dict = model_metadata_dict[cnn.TRAINING_OPTION_DICT_KEY]
    radar_field_names = training_option_dict[trainval_io.RADAR_FIELDS_KEY]
    radar_heights_m_agl = training_option_dict[trainval_io.RADAR_HEIGHTS_KEY]

    num_radar_fields = len(radar_field_names)
    num_radar_heights = len(radar_heights_m_agl)

    num_panel_rows = int(numpy.floor(
        numpy.sqrt(num_radar_heights)
    ))
    num_panel_columns = int(numpy.ceil(
        float(num_radar_heights) / num_panel_rows
    ))

    figure_objects = [None] * num_radar_fields
    axes_object_matrices = [None] * num_radar_fields
    radar_matrix = list_of_predictor_matrices[0]

    for j in range(num_radar_fields):
        this_radar_matrix = numpy.flip(radar_matrix[..., j], axis=0)

        if not allow_whitespace:
            figure_objects[j], axes_object_matrices[j] = (
                plotting_utils.create_paneled_figure(
                    num_rows=num_panel_rows, num_columns=num_panel_columns,
                    horizontal_spacing=0., vertical_spacing=0.,
                    shared_x_axis=False, shared_y_axis=False,
                    keep_aspect_ratio=True)
            )

        figure_objects[j], axes_object_matrices[j] = (
            radar_plotting.plot_3d_grid_without_coords(
                field_matrix=this_radar_matrix,
                field_name=radar_field_names[j],
                grid_point_heights_metres=radar_heights_m_agl,
                ground_relative=True, num_panel_rows=num_panel_rows,
                figure_object=figure_objects[j],
                axes_object_matrix=axes_object_matrices[j],
                font_size=FONT_SIZE_SANS_COLOUR_BARS)
        )

        if allow_whitespace:
            this_colour_map_object, this_colour_norm_object = (
                radar_plotting.get_default_colour_scheme(
                    radar_field_names[j]
                )
            )

            plotting_utils.plot_colour_bar(
                axes_object_or_matrix=axes_object_matrices[j],
                data_matrix=this_radar_matrix,
                colour_map_object=this_colour_map_object,
                colour_norm_object=this_colour_norm_object,
                orientation_string='horizontal', extend_min=True,
                extend_max=True)

            if title_string is not None:
                this_title_string = '{0:s}; {1:s}'.format(
                    title_string, radar_field_names[j]
                )
                pyplot.suptitle(this_title_string, fontsize=TITLE_FONT_SIZE)

    return figure_objects, axes_object_matrices


def _plot_2d3d_radar_scan(
        list_of_predictor_matrices, model_metadata_dict, allow_whitespace,
        title_string=None):
    """Plots 3-D reflectivity and 2-D azimuthal shear for one example.

    :param list_of_predictor_matrices: See doc for `_plot_3d_radar_scan`.
    :param model_metadata_dict: Same.
    :param allow_whitespace: Same.
    :param title_string: Same.
    :return: figure_objects: length-2 list of figure handles (instances of
        `matplotlib.figure.Figure`).  The first is for reflectivity; the second
        is for azimuthal shear.
    :return: axes_object_matrices: length-2 list (the first is for reflectivity;
        the second is for azimuthal shear).  Each element is a 2-D numpy
        array of axes handles (instances of
        `matplotlib.axes._subplots.AxesSubplot`).
    """

    training_option_dict = model_metadata_dict[cnn.TRAINING_OPTION_DICT_KEY]
    az_shear_field_names = training_option_dict[trainval_io.RADAR_FIELDS_KEY]
    refl_heights_m_agl = training_option_dict[trainval_io.RADAR_HEIGHTS_KEY]

    num_az_shear_fields = len(az_shear_field_names)
    num_refl_heights = len(refl_heights_m_agl)

    this_num_panel_rows = int(numpy.floor(
        numpy.sqrt(num_refl_heights)
    ))
    this_num_panel_columns = int(numpy.ceil(
        float(num_refl_heights) / this_num_panel_rows
    ))

    if allow_whitespace:
        refl_figure_object = None
        refl_axes_object_matrix = None
    else:
        refl_figure_object, refl_axes_object_matrix = (
            plotting_utils.create_paneled_figure(
                num_rows=this_num_panel_rows,
                num_columns=this_num_panel_columns, horizontal_spacing=0.,
                vertical_spacing=0., shared_x_axis=False,
                shared_y_axis=False, keep_aspect_ratio=True)
        )

    refl_figure_object, refl_axes_object_matrix = (
        radar_plotting.plot_3d_grid_without_coords(
            field_matrix=numpy.flip(
                list_of_predictor_matrices[0][..., 0], axis=0
            ),
            field_name=radar_utils.REFL_NAME,
            grid_point_heights_metres=refl_heights_m_agl,
            ground_relative=True, num_panel_rows=this_num_panel_rows,
            figure_object=refl_figure_object,
            axes_object_matrix=refl_axes_object_matrix,
            font_size=FONT_SIZE_SANS_COLOUR_BARS)
    )

    if allow_whitespace:
        this_colour_map_object, this_colour_norm_object = (
            radar_plotting.get_default_colour_scheme(radar_utils.REFL_NAME)
        )

        plotting_utils.plot_colour_bar(
            axes_object_or_matrix=refl_axes_object_matrix,
            data_matrix=list_of_predictor_matrices[0],
            colour_map_object=this_colour_map_object,
            colour_norm_object=this_colour_norm_object,
            orientation_string='horizontal', extend_min=True,
            extend_max=True)

        if title_string is not None:
            this_title_string = '{0:s}; {1:s}'.format(
                title_string, radar_utils.REFL_NAME)
            pyplot.suptitle(this_title_string, fontsize=TITLE_FONT_SIZE)

    if allow_whitespace:
        shear_figure_object = None
        shear_axes_object_matrix = None
    else:
        shear_figure_object, shear_axes_object_matrix = (
            plotting_utils.create_paneled_figure(
                num_rows=1, num_columns=num_az_shear_fields,
                horizontal_spacing=0., vertical_spacing=0.,
                shared_x_axis=False, shared_y_axis=False,
                keep_aspect_ratio=True)
        )

    shear_figure_object, shear_axes_object_matrix = (
        radar_plotting.plot_many_2d_grids_without_coords(
            field_matrix=numpy.flip(list_of_predictor_matrices[1], axis=0),
            field_name_by_panel=az_shear_field_names,
            panel_names=az_shear_field_names, num_panel_rows=1,
            figure_object=shear_figure_object,
            axes_object_matrix=shear_axes_object_matrix,
            plot_colour_bar_by_panel=numpy.full(
                num_az_shear_fields, False, dtype=bool
            ),
            font_size=FONT_SIZE_SANS_COLOUR_BARS)
    )

    if allow_whitespace:
        this_colour_map_object, this_colour_norm_object = (
            radar_plotting.get_default_colour_scheme(
                radar_utils.LOW_LEVEL_SHEAR_NAME)
        )

        plotting_utils.plot_colour_bar(
            axes_object_or_matrix=shear_axes_object_matrix,
            data_matrix=list_of_predictor_matrices[1],
            colour_map_object=this_colour_map_object,
            colour_norm_object=this_colour_norm_object,
            orientation_string='horizontal', extend_min=True,
            extend_max=True)

        if title_string is not None:
            pyplot.suptitle(title_string, fontsize=TITLE_FONT_SIZE)

    figure_objects = [refl_figure_object, shear_figure_object]
    axes_object_matrices = [refl_axes_object_matrix, shear_axes_object_matrix]
    return figure_objects, axes_object_matrices


def _plot_2d_radar_scan(
        list_of_predictor_matrices, model_metadata_dict, allow_whitespace,
        title_string=None):
    """Plots 2-D radar scan for one example.

    J = number of panel rows in image
    K = number of panel columns in image

    :param list_of_predictor_matrices: See doc for `_plot_3d_radar_scan`.
    :param model_metadata_dict: Same.
    :param allow_whitespace: Same.
    :param title_string: Same.
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
        field_name_by_panel = training_option_dict[trainval_io.RADAR_FIELDS_KEY]
        num_panels = len(field_name_by_panel)

        panel_names = radar_plotting.radar_fields_and_heights_to_panel_names(
            field_names=field_name_by_panel,
            heights_m_agl=training_option_dict[trainval_io.RADAR_HEIGHTS_KEY]
        )

        plot_cbar_by_panel = numpy.full(num_panels, True, dtype=bool)
    else:
        list_of_layer_operation_dicts = [
            list_of_layer_operation_dicts[k] for k in LAYER_OP_INDICES_TO_KEEP
        ]

        list_of_predictor_matrices[0] = list_of_predictor_matrices[0][
            ..., LAYER_OP_INDICES_TO_KEEP]

        field_name_by_panel, panel_names = (
            radar_plotting.layer_ops_to_field_and_panel_names(
                list_of_layer_operation_dicts=list_of_layer_operation_dicts
            )
        )

        num_panels = len(field_name_by_panel)
        plot_cbar_by_panel = numpy.full(num_panels, True, dtype=bool)

        # if allow_whitespace:
        #     if len(field_name_by_panel) == 12:
        #         plot_cbar_by_panel[2::3] = True
        #     else:
        #         plot_cbar_by_panel[:] = True

    num_panel_rows = int(numpy.floor(
        numpy.sqrt(num_panels)
    ))
    num_panel_columns = int(numpy.ceil(
        float(num_panels) / num_panel_rows
    ))

    if allow_whitespace:
        figure_object = None
        axes_object_matrix = None
    else:
        figure_object, axes_object_matrix = (
            plotting_utils.create_paneled_figure(
                num_rows=num_panel_rows, num_columns=num_panel_columns,
                horizontal_spacing=0., vertical_spacing=0.,
                shared_x_axis=False, shared_y_axis=False,
                keep_aspect_ratio=True)
        )

    figure_object, axes_object_matrix = (
        radar_plotting.plot_many_2d_grids_without_coords(
            field_matrix=numpy.flip(list_of_predictor_matrices[0], axis=0),
            field_name_by_panel=field_name_by_panel, panel_names=panel_names,
            num_panel_rows=num_panel_rows, figure_object=figure_object,
            axes_object_matrix=axes_object_matrix,
            plot_colour_bar_by_panel=plot_cbar_by_panel,
            font_size=FONT_SIZE_WITH_COLOUR_BARS, row_major=False)
    )

    if allow_whitespace and title_string is not None:
        pyplot.suptitle(title_string, fontsize=TITLE_FONT_SIZE)

    return [figure_object], [axes_object_matrix]


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
        list_of_predictor_matrices, model_metadata_dict, plot_sounding=True,
        allow_whitespace=True, pmm_flag=False, example_index=None,
        full_storm_id_string=None, storm_time_unix_sec=None,
        storm_activation=None):
    """Plots predictors for one example.

    R = number of radar figures

    :param list_of_predictor_matrices: List created by
        `testing_io.read_specific_examples`.
    :param model_metadata_dict: Dictionary returned by
        `cnn.read_model_metadata`.
    :param plot_sounding: See documentation at top of file.
    :param allow_whitespace: Same.
    :param pmm_flag: Boolean flag.  If True, will plot PMM (probability-matched
        mean) composite of many examples (storm objects).  If False, will plot
        one example.
    :param example_index: [used only if `pmm_flag == False`]
        Will plot the [i]th example, where i = `example_index`.
    :param full_storm_id_string: [used only if `pmm_flag == False`]
        Full storm ID.
    :param storm_time_unix_sec: [used only if `pmm_flag == False`]
        Storm time.
    :param storm_activation: [used only if `pmm_flag == False`]
        Model activation for this example.  Even if `pmm_flag == True`, this may
        be None.
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

    error_checking.assert_is_boolean(plot_sounding)
    error_checking.assert_is_boolean(allow_whitespace)
    error_checking.assert_is_boolean(pmm_flag)

    if pmm_flag:
        title_string = 'PMM composite'

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

        error_checking.assert_is_string(full_storm_id_string)
        storm_time_string = time_conversion.unix_sec_to_string(
            storm_time_unix_sec, TIME_FORMAT)

        if storm_activation is not None:
            error_checking.assert_is_not_nan(storm_activation)

        title_string = 'Storm "{0:s}" at {1:s}'.format(
            full_storm_id_string, storm_time_string)

    training_option_dict = model_metadata_dict[cnn.TRAINING_OPTION_DICT_KEY]
    has_sounding = (
        training_option_dict[trainval_io.SOUNDING_FIELDS_KEY] is not None
    )

    if plot_sounding and has_sounding:
        sounding_figure_object, sounding_axes_object = _plot_sounding(
            list_of_predictor_matrices=list_of_predictor_matrices,
            model_metadata_dict=model_metadata_dict,
            allow_whitespace=allow_whitespace,
            title_string=title_string)
    else:
        sounding_figure_object = None
        sounding_axes_object = None

    num_radar_matrices = len(list_of_predictor_matrices) - int(has_sounding)
    num_radar_dimensions = len(predictor_matrices_to_plot[0].shape) - 1

    if num_radar_matrices == 2:
        radar_figure_objects, radar_axes_object_matrices = (
            _plot_2d3d_radar_scan(
                list_of_predictor_matrices=predictor_matrices_to_plot,
                model_metadata_dict=model_metadata_dict,
                allow_whitespace=allow_whitespace, title_string=title_string)
        )
    elif num_radar_dimensions == 3:
        radar_figure_objects, radar_axes_object_matrices = _plot_3d_radar_scan(
            list_of_predictor_matrices=predictor_matrices_to_plot,
            model_metadata_dict=model_metadata_dict,
            allow_whitespace=allow_whitespace, title_string=title_string)
    else:
        radar_figure_objects, radar_axes_object_matrices = _plot_2d_radar_scan(
            list_of_predictor_matrices=predictor_matrices_to_plot,
            model_metadata_dict=model_metadata_dict,
            allow_whitespace=allow_whitespace, title_string=title_string)

    return {
        SOUNDING_FIGURE_KEY: sounding_figure_object,
        SOUNDING_AXES_KEY: sounding_axes_object,
        RADAR_FIGURES_KEY: radar_figure_objects,
        RADAR_AXES_KEY: radar_axes_object_matrices
    }


def plot_examples(
        list_of_predictor_matrices, model_metadata_dict, output_dir_name,
        plot_soundings=True, allow_whitespace=True, pmm_flag=False,
        full_storm_id_strings=None, storm_times_unix_sec=None,
        storm_activations=None):
    """Plots predictors for each example.

    E = number of examples

    :param list_of_predictor_matrices: See doc for `plot_one_example`.
    :param model_metadata_dict: Same.
    :param output_dir_name: Path to output directory.  Figures will be saved
        here (one or more figures per example).
    :param plot_soundings: See doc for `plot_one_example`.
    :param allow_whitespace: Same.
    :param pmm_flag: Same.
    :param full_storm_id_strings: [used only if `pmm_flag == False`]
        length-E list of full storm IDs.
    :param storm_times_unix_sec: [used only if `pmm_flag == False`]
        length-E numpy array of storm times.
    :param storm_activations: [used only if `pmm_flag == False`]
        length-E numpy array of model activations.  Even if `pmm_flag == True`,
        this may be None.
    """

    error_checking.assert_is_boolean(pmm_flag)

    if pmm_flag:
        num_examples = 1
        full_storm_id_strings = [None]
        storm_times_unix_sec = [None]
    else:
        num_examples = list_of_predictor_matrices[0].shape[0]

    if storm_activations is None:
        storm_activations = [None] * num_examples

    training_option_dict = model_metadata_dict[cnn.TRAINING_OPTION_DICT_KEY]
    has_soundings = (
        training_option_dict[trainval_io.SOUNDING_FIELDS_KEY] is not None
    )

    num_radar_matrices = len(list_of_predictor_matrices) - int(has_soundings)
    num_radar_dimensions = len(list_of_predictor_matrices[0].shape) - 2

    for i in range(num_examples):
        this_handle_dict = plot_one_example(
            list_of_predictor_matrices=list_of_predictor_matrices,
            model_metadata_dict=model_metadata_dict,
            plot_sounding=plot_soundings, allow_whitespace=allow_whitespace,
            pmm_flag=pmm_flag, example_index=i,
            full_storm_id_string=full_storm_id_strings[i],
            storm_time_unix_sec=storm_times_unix_sec[i],
            storm_activation=storm_activations[i]
        )

        this_sounding_figure_object = this_handle_dict[SOUNDING_FIGURE_KEY]

        if this_sounding_figure_object is not None:
            this_file_name = metadata_to_file_name(
                output_dir_name=output_dir_name, is_sounding=True,
                pmm_flag=pmm_flag,
                full_storm_id_string=full_storm_id_strings[i],
                storm_time_unix_sec=storm_times_unix_sec[i]
            )

            print('Saving figure to: "{0:s}"...'.format(this_file_name))
            this_sounding_figure_object.savefig(
                this_file_name, dpi=FIGURE_RESOLUTION_DPI, pad_inches=0,
                bbox_inches='tight'
            )
            pyplot.close(this_sounding_figure_object)

        these_radar_figure_objects = this_handle_dict[RADAR_FIGURES_KEY]

        if num_radar_matrices == 2:
            this_file_name = metadata_to_file_name(
                output_dir_name=output_dir_name, is_sounding=False,
                pmm_flag=pmm_flag,
                full_storm_id_string=full_storm_id_strings[i],
                storm_time_unix_sec=storm_times_unix_sec[i],
                radar_field_name='reflectivity')

            print('Saving figure to: "{0:s}"...'.format(this_file_name))
            these_radar_figure_objects[0].savefig(
                this_file_name, dpi=FIGURE_RESOLUTION_DPI, pad_inches=0,
                bbox_inches='tight'
            )
            pyplot.close(these_radar_figure_objects[0])

            this_file_name = metadata_to_file_name(
                output_dir_name=output_dir_name, is_sounding=False,
                pmm_flag=pmm_flag,
                full_storm_id_string=full_storm_id_strings[i],
                storm_time_unix_sec=storm_times_unix_sec[i],
                radar_field_name='shear')

            print('Saving figure to: "{0:s}"...'.format(this_file_name))
            these_radar_figure_objects[1].savefig(
                this_file_name, dpi=FIGURE_RESOLUTION_DPI, pad_inches=0,
                bbox_inches='tight'
            )
            pyplot.close(these_radar_figure_objects[1])

            continue

        if num_radar_dimensions == 3:
            radar_field_names = training_option_dict[
                trainval_io.RADAR_FIELDS_KEY]

            for j in range(len(radar_field_names)):
                this_file_name = metadata_to_file_name(
                    output_dir_name=output_dir_name, is_sounding=False,
                    pmm_flag=pmm_flag,
                    full_storm_id_string=full_storm_id_strings[i],
                    storm_time_unix_sec=storm_times_unix_sec[i],
                    radar_field_name=radar_field_names[j],
                    radar_height_m_agl=None)

                print('Saving figure to: "{0:s}"...'.format(this_file_name))
                these_radar_figure_objects[j].savefig(
                    this_file_name, dpi=FIGURE_RESOLUTION_DPI, pad_inches=0,
                    bbox_inches='tight'
                )
                pyplot.close(these_radar_figure_objects[j])

            continue

        this_file_name = metadata_to_file_name(
            output_dir_name=output_dir_name, is_sounding=False,
            pmm_flag=pmm_flag,
            full_storm_id_string=full_storm_id_strings[i],
            storm_time_unix_sec=storm_times_unix_sec[i]
        )

        print('Saving figure to: "{0:s}"...'.format(this_file_name))
        these_radar_figure_objects[0].savefig(
            this_file_name, dpi=FIGURE_RESOLUTION_DPI, pad_inches=0,
            bbox_inches='tight'
        )
        pyplot.close(these_radar_figure_objects[0])


def _run(activation_file_name, storm_metafile_name, num_examples,
         allow_whitespace, top_example_dir_name, radar_field_names,
         radar_heights_m_agl, plot_soundings, num_radar_rows, num_radar_columns,
         output_dir_name):
    """Plots many dataset examples (storm objects).

    This is effectively the main method.

    :param activation_file_name: See documentation at top of file.
    :param storm_metafile_name: Same.
    :param num_examples: Same.
    :param allow_whitespace: Same.
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
    list_of_predictor_matrices = testing_io.read_specific_examples(
        desired_full_id_strings=full_storm_id_strings,
        desired_times_unix_sec=storm_times_unix_sec,
        option_dict=model_metadata_dict[cnn.TRAINING_OPTION_DICT_KEY],
        top_example_dir_name=top_example_dir_name,
        list_of_layer_operation_dicts=model_metadata_dict[
            cnn.LAYER_OPERATIONS_KEY]
    )[0]
    print(SEPARATOR_STRING)

    plot_examples(
        list_of_predictor_matrices=list_of_predictor_matrices,
        model_metadata_dict=model_metadata_dict,
        output_dir_name=output_dir_name, plot_soundings=plot_soundings,
        allow_whitespace=allow_whitespace, pmm_flag=False,
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
        allow_whitespace=bool(getattr(
            INPUT_ARG_OBJECT, ALLOW_WHITESPACE_ARG_NAME
        )),
        top_example_dir_name=getattr(INPUT_ARG_OBJECT, EXAMPLE_DIR_ARG_NAME),
        radar_field_names=getattr(INPUT_ARG_OBJECT, RADAR_FIELDS_ARG_NAME),
        radar_heights_m_agl=numpy.array(
            getattr(INPUT_ARG_OBJECT, RADAR_HEIGHTS_ARG_NAME), dtype=int),
        plot_soundings=bool(getattr(INPUT_ARG_OBJECT, PLOT_SOUNDINGS_ARG_NAME)),
        num_radar_rows=getattr(INPUT_ARG_OBJECT, NUM_ROWS_ARG_NAME),
        num_radar_columns=getattr(INPUT_ARG_OBJECT, NUM_COLUMNS_ARG_NAME),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
