"""Plots saliency maps for a CNN (convolutional neural network)."""

import os
import argparse
import numpy
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as pyplot
from gewittergefahr.gg_utils import radar_utils
from gewittergefahr.gg_utils import soundings
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.gg_utils import error_checking
from gewittergefahr.deep_learning import cnn
from gewittergefahr.deep_learning import saliency_maps
from gewittergefahr.deep_learning import deep_learning_utils as dl_utils
from gewittergefahr.deep_learning import training_validation_io as trainval_io
from gewittergefahr.plotting import plotting_utils
from gewittergefahr.plotting import radar_plotting
from gewittergefahr.plotting import sounding_plotting
from gewittergefahr.plotting import saliency_plotting
from gewittergefahr.plotting import imagemagick_utils

# TODO(thunderhoser): Use threshold counts at some point.

TIME_FORMAT = '%Y-%m-%d-%H%M%S'
SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

HALF_NUM_CONTOURS = 10

TITLE_FONT_SIZE = 20
FONT_SIZE_WITH_COLOUR_BARS = 16
FONT_SIZE_SANS_COLOUR_BARS = 20
FIGURE_RESOLUTION_DPI = 300
SOUNDING_IMAGE_SIZE_PX = int(5e6)

INPUT_FILE_ARG_NAME = 'input_file_name'
SALIENCY_CMAP_ARG_NAME = 'saliency_colour_map_name'
MAX_SALIENCY_PRCTILE_ARG_NAME = 'max_colour_prctile_for_saliency'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

INPUT_FILE_HELP_STRING = (
    'Path to input file.  Will be read by `saliency_maps.read_standard_file` or'
    ' `saliency_maps.read_pmm_file`.')

SALIENCY_CMAP_HELP_STRING = (
    'Name of colour map.  Saliency for each predictor will be plotted with the '
    'same colour map.  For example, if name is "Greys", the colour map used '
    'will be `pyplot.cm.Greys`.  This argument supports only pyplot colour '
    'maps.')

MAX_SALIENCY_PRCTILE_HELP_STRING = (
    'Used to set max absolute value for each saliency map.  The max absolute '
    'value for example e and predictor p will be the [q]th percentile of all '
    'saliency values for example e, where q = `{0:s}`.'
).format(MAX_SALIENCY_PRCTILE_ARG_NAME)

OUTPUT_DIR_HELP_STRING = (
    'Path to output directory.  Figures will be saved here.')

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + INPUT_FILE_ARG_NAME, type=str, required=True,
    help=INPUT_FILE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + SALIENCY_CMAP_ARG_NAME, type=str, required=False, default='Greys',
    help=SALIENCY_CMAP_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + MAX_SALIENCY_PRCTILE_ARG_NAME, type=float, required=False,
    default=99., help=MAX_SALIENCY_PRCTILE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING)


def _plot_saliency_for_2d3d_radar(
        list_of_input_matrices, list_of_saliency_matrices,
        training_option_dict, saliency_colour_map_object,
        max_colour_value_by_example, output_dir_name, storm_ids=None,
        storm_times_unix_sec=None):
    """Plots saliency for 2-D azimuthal-shear and 3-D reflectivity fields.

    E = number of examples (storm objects)

    If `storm_ids is None` and `storm_times_unix_sec is None`, will assume that
    the input matrices contain probability-matched means.

    :param list_of_input_matrices: See doc for
        `saliency_maps.read_standard_file`.
    :param list_of_saliency_matrices: Same.
    :param training_option_dict: Dictionary returned by
        `cnn.read_model_metadata`.
    :param saliency_colour_map_object: See documentation at top of file.
    :param max_colour_value_by_example: length-E numpy array with max value in
        colour scheme for each example.  Minimum value for [i]th example will be
        -1 * max_colour_value_by_example[i], since the colour scheme is
        zero-centered and divergent.
    :param output_dir_name: Name of output directory (figures will be saved
        here).
    :param storm_ids: length-E list of storm IDs (strings).
    :param storm_times_unix_sec: length-E numpy array of storm times.
    """

    pmm_flag = storm_ids is None and storm_times_unix_sec is None

    reflectivity_matrix_dbz = list_of_input_matrices[0]
    reflectivity_saliency_matrix = list_of_saliency_matrices[0]
    az_shear_matrix_s01 = list_of_input_matrices[1]
    az_shear_saliency_matrix = list_of_saliency_matrices[1]

    num_examples = reflectivity_matrix_dbz.shape[0]
    num_reflectivity_heights = len(
        training_option_dict[trainval_io.RADAR_HEIGHTS_KEY]
    )
    num_panel_rows_for_reflectivity = int(numpy.floor(
        numpy.sqrt(num_reflectivity_heights)
    ))

    for i in range(num_examples):
        _, these_axes_objects = radar_plotting.plot_3d_grid_without_coords(
            field_matrix=numpy.flip(reflectivity_matrix_dbz[i, ..., 0], axis=0),
            field_name=radar_utils.REFL_NAME,
            grid_point_heights_metres=training_option_dict[
                trainval_io.RADAR_HEIGHTS_KEY],
            ground_relative=True,
            num_panel_rows=num_panel_rows_for_reflectivity,
            font_size=FONT_SIZE_SANS_COLOUR_BARS)

        saliency_plotting.plot_many_2d_grids_with_pm_signs(
            saliency_matrix_3d=numpy.flip(
                reflectivity_saliency_matrix[i, ..., 0], axis=0),
            axes_objects_2d_list=these_axes_objects,
            colour_map_object=saliency_colour_map_object,
            max_absolute_colour_value=max_colour_value_by_example[i])

        this_colour_map_object, this_colour_norm_object = (
            radar_plotting.get_default_colour_scheme(radar_utils.REFL_NAME)
        )

        plotting_utils.add_colour_bar(
            axes_object_or_list=these_axes_objects,
            values_to_colour=reflectivity_matrix_dbz[i, ..., 0],
            colour_map=this_colour_map_object,
            colour_norm_object=this_colour_norm_object,
            orientation='horizontal', extend_min=True, extend_max=True)

        if pmm_flag:
            this_title_string = 'Probability-matched mean'
            this_file_name = '{0:s}/saliency_pmm_reflectivity.jpg'.format(
                output_dir_name)
        else:
            this_storm_time_string = time_conversion.unix_sec_to_string(
                storm_times_unix_sec[i], TIME_FORMAT)

            this_title_string = 'Storm "{0:s}" at {1:s}'.format(
                storm_ids[i], this_storm_time_string)

            this_file_name = (
                '{0:s}/saliency_{1:s}_{2:s}_reflectivity.jpg'
            ).format(
                output_dir_name, storm_ids[i].replace('_', '-'),
                this_storm_time_string
            )

        this_title_string += ' (max absolute saliency = {0:.3f})'.format(
            max_colour_value_by_example[i])
        pyplot.suptitle(this_title_string, fontsize=TITLE_FONT_SIZE)

        print 'Saving figure to file: "{0:s}"...'.format(this_file_name)
        pyplot.savefig(this_file_name, dpi=FIGURE_RESOLUTION_DPI)
        pyplot.close()

        _, these_axes_objects = (
            radar_plotting.plot_many_2d_grids_without_coords(
                field_matrix=numpy.flip(az_shear_matrix_s01[i, ...], axis=0),
                field_name_by_panel=training_option_dict[
                    trainval_io.RADAR_FIELDS_KEY],
                num_panel_rows=1,
                panel_names=training_option_dict[trainval_io.RADAR_FIELDS_KEY],
                font_size=FONT_SIZE_SANS_COLOUR_BARS, plot_colour_bars=False)
        )

        saliency_plotting.plot_many_2d_grids_with_pm_signs(
            saliency_matrix_3d=numpy.flip(
                az_shear_saliency_matrix[i, ...], axis=0),
            axes_objects_2d_list=these_axes_objects,
            colour_map_object=saliency_colour_map_object,
            max_absolute_colour_value=max_colour_value_by_example[i])

        this_colour_map_object, this_colour_norm_object = (
            radar_plotting.get_default_colour_scheme(
                radar_utils.LOW_LEVEL_SHEAR_NAME)
        )

        plotting_utils.add_colour_bar(
            axes_object_or_list=these_axes_objects,
            values_to_colour=az_shear_saliency_matrix[i, ...],
            colour_map=this_colour_map_object,
            colour_norm_object=this_colour_norm_object,
            orientation='horizontal', extend_min=True, extend_max=True)

        pyplot.suptitle(this_title_string, fontsize=TITLE_FONT_SIZE)
        this_file_name = this_file_name.replace(
            '_reflectivity.jpg', '_azimuthal-shear.jpg')

        print 'Saving figure to file: "{0:s}"...'.format(this_file_name)
        pyplot.savefig(this_file_name, dpi=FIGURE_RESOLUTION_DPI)
        pyplot.close()


def _plot_saliency_for_2d_radar(
        radar_matrix, radar_saliency_matrix, model_metadata_dict,
        saliency_colour_map_object, max_colour_value_by_example,
        output_dir_name, storm_ids=None, storm_times_unix_sec=None):
    """Plots saliency for 2-D radar fields.

    E = number of examples
    M = number of rows in spatial grid
    N = number of columns in spatial grid
    C = number of channels (field/height pairs)

    If `storm_ids is None` and `storm_times_unix_sec is None`, will assume that
    the input matrices contain probability-matched means.

    :param radar_matrix: E-by-M-by-N-by-C numpy array of radar values
        (predictors).
    :param radar_saliency_matrix: E-by-M-by-N-by-C numpy array of saliency
        values.
    :param model_metadata_dict: See doc for `cnn.read_model_metadata`.
    :param saliency_colour_map_object: See doc for
        `_plot_saliency_for_2d3d_radar`.
    :param max_colour_value_by_example: Same.
    :param output_dir_name: Same.
    :param storm_ids: Same.
    :param storm_times_unix_sec: Same.
    """

    pmm_flag = storm_ids is None and storm_times_unix_sec is None
    training_option_dict = model_metadata_dict[cnn.TRAINING_OPTION_DICT_KEY]
    list_of_layer_operation_dicts = model_metadata_dict[
        cnn.LAYER_OPERATIONS_KEY]

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
    else:
        field_name_by_panel, panel_names = (
            radar_plotting.layer_ops_to_field_and_panel_names(
                list_of_layer_operation_dicts)
        )

    num_examples = radar_matrix.shape[0]
    num_panels = len(field_name_by_panel)
    num_panel_rows = int(numpy.floor(numpy.sqrt(num_panels)))

    for i in range(num_examples):
        _, these_axes_objects = (
            radar_plotting.plot_many_2d_grids_without_coords(
                field_matrix=numpy.flip(radar_matrix[i, ...], axis=0),
                field_name_by_panel=field_name_by_panel,
                num_panel_rows=num_panel_rows, panel_names=panel_names,
                font_size=FONT_SIZE_WITH_COLOUR_BARS, plot_colour_bars=True,
                row_major=False)
        )

        this_contour_interval = (
            max_colour_value_by_example[i] / HALF_NUM_CONTOURS
        )

        saliency_plotting.plot_many_2d_grids_with_contours(
            saliency_matrix_3d=numpy.flip(
                radar_saliency_matrix[i, ...], axis=0),
            axes_objects_2d_list=these_axes_objects,
            colour_map_object=saliency_colour_map_object,
            max_absolute_contour_level=max_colour_value_by_example[i],
            contour_interval=this_contour_interval, row_major=False)

        if pmm_flag:
            this_title_string = 'Probability-matched mean'
            this_file_name = '{0:s}/saliency_pmm_radar.jpg'.format(
                output_dir_name)
        else:
            this_storm_time_string = time_conversion.unix_sec_to_string(
                storm_times_unix_sec[i], TIME_FORMAT)

            this_title_string = 'Storm "{0:s}" at {1:s}'.format(
                storm_ids[i], this_storm_time_string)

            this_file_name = '{0:s}/saliency_{1:s}_{2:s}_radar.jpg'.format(
                output_dir_name, storm_ids[i].replace('_', '-'),
                this_storm_time_string)

        this_title_string += ' (max absolute saliency = {0:.3f})'.format(
            max_colour_value_by_example[i])
        pyplot.suptitle(this_title_string, fontsize=TITLE_FONT_SIZE)

        print 'Saving figure to file: "{0:s}"...'.format(this_file_name)
        pyplot.savefig(this_file_name, dpi=FIGURE_RESOLUTION_DPI)
        pyplot.close()


def _plot_saliency_for_3d_radar(
        radar_matrix, radar_saliency_matrix, model_metadata_dict,
        saliency_colour_map_object, max_colour_value_by_example,
        output_dir_name, storm_ids=None, storm_times_unix_sec=None):
    """Plots saliency for 3-D radar fields.

    E = number of examples
    M = number of rows in spatial grid
    N = number of columns in spatial grid
    H = number of heights in spatial grid
    F = number of fields

    If `storm_ids is None` and `storm_times_unix_sec is None`, will assume that
    the input matrices contain probability-matched means.

    :param radar_matrix: E-by-M-by-N-by-H-by-F numpy array of radar values
        (predictors).
    :param radar_saliency_matrix: E-by-M-by-N-by-H-by-F numpy array of saliency
        values.
    :param model_metadata_dict: See doc for `cnn.read_model_metadata`.
    :param saliency_colour_map_object: See doc for
        `_plot_saliency_for_2d3d_radar`.
    :param max_colour_value_by_example: Same.
    :param output_dir_name: Same.
    :param storm_ids: Same.
    :param storm_times_unix_sec: Same.
    """

    pmm_flag = storm_ids is None and storm_times_unix_sec is None
    training_option_dict = model_metadata_dict[cnn.TRAINING_OPTION_DICT_KEY]

    num_examples = radar_matrix.shape[0]
    num_fields = radar_matrix.shape[-1]
    num_heights = len(training_option_dict[trainval_io.RADAR_HEIGHTS_KEY])
    num_panel_rows = int(numpy.floor(numpy.sqrt(num_heights)))

    for i in range(num_examples):
        for k in range(num_fields):
            this_field_name = training_option_dict[
                trainval_io.RADAR_FIELDS_KEY][k]

            _, these_axes_objects = (
                radar_plotting.plot_3d_grid_without_coords(
                    field_matrix=numpy.flip(radar_matrix[i, ..., k], axis=0),
                    field_name=this_field_name,
                    grid_point_heights_metres=training_option_dict[
                        trainval_io.RADAR_HEIGHTS_KEY],
                    ground_relative=True, num_panel_rows=num_panel_rows,
                    font_size=FONT_SIZE_SANS_COLOUR_BARS)
            )

            saliency_plotting.plot_many_2d_grids_with_pm_signs(
                saliency_matrix_3d=numpy.flip(
                    radar_saliency_matrix[i, ..., k], axis=0),
                axes_objects_2d_list=these_axes_objects,
                colour_map_object=saliency_colour_map_object,
                max_absolute_colour_value=max_colour_value_by_example[i])

            this_colour_map_object, this_colour_norm_object = (
                radar_plotting.get_default_colour_scheme(this_field_name)
            )

            plotting_utils.add_colour_bar(
                axes_object_or_list=these_axes_objects,
                values_to_colour=radar_matrix[i, ..., k],
                colour_map=this_colour_map_object,
                colour_norm_object=this_colour_norm_object,
                orientation='horizontal', extend_min=True, extend_max=True)

            if pmm_flag:
                this_title_string = 'Probability-matched mean'
                this_file_name = '{0:s}/saliency_pmm_{1:s}.jpg'.format(
                    output_dir_name, this_field_name.replace('_', '-')
                )
            else:
                this_storm_time_string = time_conversion.unix_sec_to_string(
                    storm_times_unix_sec[i], TIME_FORMAT)

                this_title_string = 'Storm "{0:s}" at {1:s}'.format(
                    storm_ids[i], this_storm_time_string)

                this_file_name = '{0:s}/saliency_{1:s}_{2:s}_{3:s}.jpg'.format(
                    output_dir_name, storm_ids[i].replace('_', '-'),
                    this_storm_time_string, this_field_name.replace('_', '-')
                )

            this_title_string += ' (max absolute saliency = {0:.3f})'.format(
                max_colour_value_by_example[i])
            pyplot.suptitle(this_title_string, fontsize=TITLE_FONT_SIZE)

            print 'Saving figure to file: "{0:s}"...'.format(this_file_name)
            pyplot.savefig(this_file_name, dpi=FIGURE_RESOLUTION_DPI)
            pyplot.close()


def _plot_sounding_saliency(
        sounding_matrix, sounding_saliency_matrix, sounding_field_names,
        saliency_metadata_dict, colour_map_object, max_colour_value_by_example,
        output_dir_name):
    """Plots soundings along with their saliency maps.

    E = number of examples
    H = number of sounding heights
    F = number of sounding fields

    :param sounding_matrix: E-by-H-by-F numpy array of sounding values.
    :param sounding_saliency_matrix: E-by-H-by-F numpy array of corresponding
        saliency values.
    :param sounding_field_names: length-F list of field names.
    :param saliency_metadata_dict: Dictionary returned by
        `saliency_maps.read_standard_file`.
    :param colour_map_object: See doc for `_plot_2d3d_radar_saliency`.
    :param max_colour_value_by_example: Same.
    :param output_dir_name: Name of output directory (figures will be saved
        here).
    """

    # TODO(thunderhoser): Generalize this method to deal with
    # probability-matched means.

    num_examples = sounding_matrix.shape[0]

    try:
        metpy_dict_by_example = dl_utils.soundings_to_metpy_dictionaries(
            sounding_matrix=sounding_matrix, field_names=sounding_field_names)
    except:
        sounding_pressure_matrix_pa = numpy.expand_dims(
            saliency_metadata_dict[saliency_maps.SOUNDING_PRESSURES_KEY],
            axis=-1)

        this_sounding_matrix = numpy.concatenate(
            (sounding_matrix, sounding_pressure_matrix_pa), axis=-1)

        metpy_dict_by_example = dl_utils.soundings_to_metpy_dictionaries(
            sounding_matrix=this_sounding_matrix,
            field_names=sounding_field_names + [soundings.PRESSURE_NAME])

    for i in range(num_examples):
        this_storm_id = saliency_metadata_dict[
            saliency_maps.STORM_IDS_KEY][i]
        this_storm_time_string = time_conversion.unix_sec_to_string(
            saliency_metadata_dict[saliency_maps.STORM_TIMES_KEY][i],
            TIME_FORMAT)

        this_title_string = 'Storm "{0:s}" at {1:s}'.format(
            this_storm_id, this_storm_time_string)
        sounding_plotting.plot_sounding(
            sounding_dict_for_metpy=metpy_dict_by_example[i],
            title_string=this_title_string)

        this_left_file_name = (
            '{0:s}/{1:s}_{2:s}_sounding-actual.jpg'
        ).format(
            output_dir_name, this_storm_id.replace('_', '-'),
            this_storm_time_string
        )

        print 'Saving figure to file: "{0:s}"...'.format(
            this_left_file_name)
        pyplot.savefig(this_left_file_name, dpi=FIGURE_RESOLUTION_DPI)
        pyplot.close()

        imagemagick_utils.trim_whitespace(
            input_file_name=this_left_file_name,
            output_file_name=this_left_file_name)

        saliency_plotting.plot_saliency_for_sounding(
            saliency_matrix=sounding_saliency_matrix[i, ...],
            sounding_field_names=sounding_field_names,
            pressure_levels_mb=metpy_dict_by_example[i][
                soundings.PRESSURE_COLUMN_METPY],
            colour_map_object=colour_map_object,
            max_absolute_colour_value=max_colour_value_by_example[i])

        this_right_file_name = (
            '{0:s}/{1:s}_{2:s}_sounding-saliency.jpg'
        ).format(
            output_dir_name, this_storm_id.replace('_', '-'),
            this_storm_time_string
        )

        print 'Saving figure to file: "{0:s}"...'.format(
            this_right_file_name)
        pyplot.savefig(this_right_file_name, dpi=FIGURE_RESOLUTION_DPI)
        pyplot.close()

        imagemagick_utils.trim_whitespace(
            input_file_name=this_right_file_name,
            output_file_name=this_right_file_name)

        this_file_name = (
            '{0:s}/saliency_{1:s}_{2:s}_sounding.jpg'
        ).format(
            output_dir_name, this_storm_id.replace('_', '-'),
            this_storm_time_string
        )

        print 'Concatenating panels into file: "{0:s}"...\n'.format(
            this_file_name)

        imagemagick_utils.concatenate_images(
            input_file_names=[this_left_file_name, this_right_file_name],
            output_file_name=this_file_name, num_panel_rows=1,
            num_panel_columns=2)

        imagemagick_utils.resize_image(
            input_file_name=this_file_name,
            output_file_name=this_file_name,
            output_size_pixels=SOUNDING_IMAGE_SIZE_PX)

        os.remove(this_left_file_name)
        os.remove(this_right_file_name)


def _run(input_file_name, saliency_colour_map_name,
         max_colour_prctile_for_saliency, output_dir_name):
    """Plots saliency maps for a CNN (convolutional neural network).

    This is effectively the main method.

    :param input_file_name: See documentation at top of file.
    :param saliency_colour_map_name: Same.
    :param max_colour_prctile_for_saliency: Same.
    :param output_dir_name: Same.
    """

    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=output_dir_name)

    error_checking.assert_is_geq(max_colour_prctile_for_saliency, 0.)
    error_checking.assert_is_leq(max_colour_prctile_for_saliency, 100.)
    saliency_colour_map_object = pyplot.cm.get_cmap(saliency_colour_map_name)

    print 'Reading data from: "{0:s}"...'.format(input_file_name)

    try:
        saliency_dict = saliency_maps.read_standard_file(input_file_name)
        list_of_input_matrices = saliency_dict.pop(
            saliency_maps.INPUT_MATRICES_KEY)
        list_of_saliency_matrices = saliency_dict.pop(
            saliency_maps.SALIENCY_MATRICES_KEY)

        saliency_metadata_dict = saliency_dict
        storm_ids = saliency_metadata_dict[saliency_maps.STORM_IDS_KEY]
        storm_times_unix_sec = saliency_metadata_dict[
            saliency_maps.STORM_TIMES_KEY]

    except ValueError:
        saliency_dict = saliency_maps.read_pmm_file(input_file_name)
        list_of_input_matrices = saliency_dict[
            saliency_maps.MEAN_INPUT_MATRICES_KEY]
        list_of_saliency_matrices = saliency_dict[
            saliency_maps.MEAN_SALIENCY_MATRICES_KEY]

        for i in range(len(list_of_input_matrices)):
            list_of_input_matrices[i] = numpy.expand_dims(
                list_of_input_matrices[i], axis=0)
            list_of_saliency_matrices[i] = numpy.expand_dims(
                list_of_saliency_matrices[i], axis=0)

        orig_saliency_file_name = saliency_dict[
            saliency_maps.STANDARD_FILE_NAME_KEY]

        print 'Reading metadata from: "{0:s}"...'.format(
            orig_saliency_file_name)
        orig_saliency_dict = saliency_maps.read_standard_file(
            orig_saliency_file_name)

        orig_saliency_dict.pop(saliency_maps.INPUT_MATRICES_KEY)
        orig_saliency_dict.pop(saliency_maps.SALIENCY_MATRICES_KEY)
        saliency_metadata_dict = orig_saliency_dict

        storm_ids = None
        storm_times_unix_sec = None

    num_examples = list_of_input_matrices[0].shape[0]
    max_colour_value_by_example = numpy.full(num_examples, numpy.nan)

    for i in range(num_examples):
        these_saliency_values = numpy.concatenate(
            [numpy.ravel(s[i, ...]) for s in list_of_saliency_matrices]
        )
        max_colour_value_by_example[i] = numpy.percentile(
            numpy.absolute(these_saliency_values),
            max_colour_prctile_for_saliency)

    model_file_name = saliency_metadata_dict[saliency_maps.MODEL_FILE_NAME_KEY]
    model_metafile_name = '{0:s}/model_metadata.p'.format(
        os.path.split(model_file_name)[0]
    )

    print 'Reading metadata from: "{0:s}"...'.format(model_metafile_name)
    model_metadata_dict = cnn.read_model_metadata(model_metafile_name)
    training_option_dict = model_metadata_dict[cnn.TRAINING_OPTION_DICT_KEY]
    sounding_field_names = training_option_dict[trainval_io.SOUNDING_FIELDS_KEY]

    print SEPARATOR_STRING

    # if sounding_field_names is not None:
    #     _plot_sounding_saliency(
    #         sounding_matrix=list_of_input_matrices[-1],
    #         sounding_saliency_matrix=list_of_saliency_matrices[-1],
    #         sounding_field_names=sounding_field_names,
    #         saliency_metadata_dict=saliency_metadata_dict,
    #         colour_map_object=saliency_colour_map_object,
    #         max_colour_value_by_example=max_colour_value_by_example,
    #         output_dir_name=output_dir_name)
    #     print SEPARATOR_STRING

    if model_metadata_dict[cnn.USE_2D3D_CONVOLUTION_KEY]:
        _plot_saliency_for_2d3d_radar(
            list_of_input_matrices=list_of_input_matrices,
            list_of_saliency_matrices=list_of_saliency_matrices,
            training_option_dict=training_option_dict,
            saliency_colour_map_object=saliency_colour_map_object,
            max_colour_value_by_example=max_colour_value_by_example,
            output_dir_name=output_dir_name, storm_ids=storm_ids,
            storm_times_unix_sec=storm_times_unix_sec)
        return

    num_radar_dimensions = len(list_of_input_matrices[0].shape) - 2
    if num_radar_dimensions == 3:
        _plot_saliency_for_3d_radar(
            radar_matrix=list_of_input_matrices[0],
            radar_saliency_matrix=list_of_saliency_matrices[0],
            model_metadata_dict=model_metadata_dict,
            saliency_colour_map_object=saliency_colour_map_object,
            max_colour_value_by_example=max_colour_value_by_example,
            output_dir_name=output_dir_name, storm_ids=storm_ids,
            storm_times_unix_sec=storm_times_unix_sec)
        return

    _plot_saliency_for_2d_radar(
        radar_matrix=list_of_input_matrices[0],
        radar_saliency_matrix=list_of_saliency_matrices[0],
        model_metadata_dict=model_metadata_dict,
        saliency_colour_map_object=saliency_colour_map_object,
        max_colour_value_by_example=max_colour_value_by_example,
        output_dir_name=output_dir_name, storm_ids=storm_ids,
        storm_times_unix_sec=storm_times_unix_sec)


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        input_file_name=getattr(INPUT_ARG_OBJECT, INPUT_FILE_ARG_NAME),
        saliency_colour_map_name=getattr(
            INPUT_ARG_OBJECT, SALIENCY_CMAP_ARG_NAME),
        max_colour_prctile_for_saliency=getattr(
            INPUT_ARG_OBJECT, MAX_SALIENCY_PRCTILE_ARG_NAME),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
