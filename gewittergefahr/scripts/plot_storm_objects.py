"""Plots radar and sounding data for each storm object."""

import os.path
import argparse
import numpy
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as pyplot
from gewittergefahr.gg_io import storm_tracking_io as tracking_io
from gewittergefahr.gg_utils import storm_tracking_utils as tracking_utils
from gewittergefahr.gg_utils import labels
from gewittergefahr.gg_utils import link_events_to_storms as events2storms
from gewittergefahr.gg_utils import radar_utils
from gewittergefahr.gg_utils import soundings
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.deep_learning import storm_images
from gewittergefahr.deep_learning import deep_learning_utils as dl_utils
from gewittergefahr.deep_learning import deployment_io
from gewittergefahr.deep_learning import training_validation_io as trainval_io
from gewittergefahr.deep_learning import model_activation
from gewittergefahr.deep_learning import cnn
from gewittergefahr.plotting import plotting_utils
from gewittergefahr.plotting import sounding_plotting
from gewittergefahr.plotting import radar_plotting

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'
MINOR_SEPARATOR_STRING = '\n\n' + '-' * 50 + '\n\n'

SOUNDING_FIELD_NAMES = [
    soundings.U_WIND_NAME, soundings.V_WIND_NAME,
    soundings.TEMPERATURE_NAME, soundings.SPECIFIC_HUMIDITY_NAME,
    soundings.PRESSURE_NAME
]

FIELD_NAMES_2D_KEY = 'field_name_by_pair'
HEIGHTS_2D_KEY = 'height_by_pair_m_agl'
FIELD_NAMES_3D_KEY = 'radar_field_names'
HEIGHTS_3D_KEY = 'radar_heights_m_agl'
STORM_IDS_KEY = 'storm_ids'
STORM_TIMES_KEY = 'storm_times_unix_sec'
RADAR_IMAGE_MATRIX_KEY = 'radar_image_matrix'
SOUNDING_MATRIX_KEY = 'sounding_matrix'
STORM_ACTIVATIONS_KEY = 'storm_activations'

LARGE_INTEGER = int(1e10)
TIME_FORMAT = '%Y-%m-%d-%H%M%S'

NUM_PANEL_ROWS = 3
TITLE_FONT_SIZE = 20
DOTS_PER_INCH = 300

ACTIVATION_FILE_ARG_NAME = 'input_activation_file_name'
STORM_METAFILE_ARG_NAME = 'input_storm_metafile_name'
RADAR_IMAGE_DIR_ARG_NAME = 'input_radar_image_dir_name'
RADAR_SOURCE_ARG_NAME = 'radar_source'
RADAR_FIELDS_ARG_NAME = 'radar_field_names'
RADAR_HEIGHTS_ARG_NAME = 'radar_heights_m_agl'
REFL_HEIGHTS_ARG_NAME = 'refl_heights_m_agl'
NUM_ROWS_TO_KEEP_ARG_NAME = 'num_rows_to_keep'
NUM_COLUMNS_TO_KEEP_ARG_NAME = 'num_columns_to_keep'
SOUNDING_DIR_ARG_NAME = 'input_sounding_dir_name'
SOUNDING_LAG_TIME_ARG_NAME = 'sounding_lag_time_sec'
SOUNDING_LEAD_TIME_ARG_NAME = 'sounding_lead_time_sec'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

ACTIVATION_FILE_HELP_STRING = (
    'Path to activation file.  This will be read by `model_activation.'
    'read_file`, and all storm objects therein will be plotted.  If this '
    'argument is empty, `{0:s}` will be used, instead.'
).format(STORM_METAFILE_ARG_NAME)

STORM_METAFILE_HELP_STRING = (
    'Path to metafile (with ID-time for each storm object).  This will be read '
    'by `storm_tracking_io.read_storm_ids_and_times`, and all storm objects '
    'therein will be plotted.  If this argument is empty, `{0:s}` will be used,'
    ' instead.'
).format(ACTIVATION_FILE_ARG_NAME)

RADAR_IMAGE_DIR_HELP_STRING = (
    'Name of top-level directory with storm-centered radar images.  Files '
    'therein will be found by `storm_images.find_storm_image_file` and read by '
    '`storm_images.read_storm_images`.')

RADAR_SOURCE_HELP_STRING = (
    '[used only if `{0:s}` is empty] Radar source.  Must be in the following '
    'list.\n{1:s}'
).format(ACTIVATION_FILE_ARG_NAME, str(radar_utils.DATA_SOURCE_IDS))

RADAR_FIELDS_HELP_STRING = (
    '[used only if `{0:s}` is empty] List of radar fields to plot.  Each must '
    'be in the following list.\n{1:s}'
).format(ACTIVATION_FILE_ARG_NAME, str(radar_utils.RADAR_FIELD_NAMES))

RADAR_HEIGHTS_HELP_STRING = (
    '[used only if {0:s} is empty and {1:s} = "{2:s}"] List of radar heights.  '
    'Each field in `{3:s}` will be plotted at each height.'
).format(ACTIVATION_FILE_ARG_NAME, RADAR_SOURCE_ARG_NAME,
         radar_utils.GRIDRAD_SOURCE_ID, RADAR_FIELDS_ARG_NAME)

REFL_HEIGHTS_HELP_STRING = (
    '[used only if {0:s} is empty and {1:s} = "{2:s}"] List of reflectivity '
    'heights.  Field "{3:s}" will be plotted at each height.'
).format(ACTIVATION_FILE_ARG_NAME, RADAR_SOURCE_ARG_NAME,
         radar_utils.MYRORSS_SOURCE_ID, radar_utils.REFL_NAME)

NUM_ROWS_TO_KEEP_HELP_STRING = (
    '[used only if {0:s} is empty] Number of rows to keep in each storm-'
    'centered radar image.  To use the full images, leave this alone.'
).format(ACTIVATION_FILE_ARG_NAME)

NUM_COLUMNS_TO_KEEP_HELP_STRING = (
    '[used only if {0:s} is empty] Number of columns to keep in each storm-'
    'centered radar image.  To use the full images, leave this alone.'
).format(ACTIVATION_FILE_ARG_NAME)

SOUNDING_DIR_HELP_STRING = (
    'Name of top-level directory with storm-centered soundings.  Files therein '
    'will be found by `soundings.find_sounding_file` and read by '
    '`soundings.read_soundings`.  To plot only radar (no soundings), leave '
    'this alone.')

SOUNDING_LAG_TIME_HELP_STRING = (
    '[used only if {0:s} is empty and {1:s} is non-empty] Lag time (used to '
    'find sounding files).'
).format(ACTIVATION_FILE_ARG_NAME, SOUNDING_DIR_ARG_NAME)

SOUNDING_LEAD_TIME_HELP_STRING = (
    '[used only if {0:s} is empty and {1:s} is non-empty] Lead time (used to '
    'find sounding files).'
).format(ACTIVATION_FILE_ARG_NAME, SOUNDING_DIR_ARG_NAME)

OUTPUT_DIR_HELP_STRING = (
    'Name of output directory.  Figures will be saved here.')

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + ACTIVATION_FILE_ARG_NAME, type=str, required=False, default='',
    help=ACTIVATION_FILE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + STORM_METAFILE_ARG_NAME, type=str, required=False, default='',
    help=STORM_METAFILE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + RADAR_IMAGE_DIR_ARG_NAME, type=str, required=True,
    help=RADAR_IMAGE_DIR_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + RADAR_SOURCE_ARG_NAME, type=str, required=False, default='',
    help=RADAR_SOURCE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + RADAR_FIELDS_ARG_NAME, type=str, nargs='+', required=False,
    default=[''], help=RADAR_FIELDS_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + RADAR_HEIGHTS_ARG_NAME, type=int, nargs='+', required=False,
    default=[-1], help=RADAR_HEIGHTS_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + REFL_HEIGHTS_ARG_NAME, type=int, nargs='+', required=False,
    default=[-1], help=REFL_HEIGHTS_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + NUM_ROWS_TO_KEEP_ARG_NAME, type=int, required=False,
    default=-1, help=NUM_ROWS_TO_KEEP_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + NUM_COLUMNS_TO_KEEP_ARG_NAME, type=int, required=False,
    default=-1, help=NUM_COLUMNS_TO_KEEP_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + SOUNDING_DIR_ARG_NAME, type=str, required=False, default='',
    help=SOUNDING_DIR_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + SOUNDING_LAG_TIME_ARG_NAME, type=int, required=False, default=-1,
    help=SOUNDING_LAG_TIME_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + SOUNDING_LEAD_TIME_ARG_NAME, type=int, required=False, default=0,
    help=SOUNDING_LEAD_TIME_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING)


def _read_inputs(
        storm_ids, storm_times_unix_sec, top_radar_image_dir_name, radar_source,
        radar_field_names, radar_heights_m_agl, refl_heights_m_agl,
        num_rows_to_keep, num_columns_to_keep, top_sounding_dir_name,
        sounding_lag_time_sec, sounding_lead_time_sec):
    """Reads radar and sounding data for each storm object.

    E = number of examples (storm objects)
    M = number of rows in each storm-centered grid
    N = number of columns in each storm-centered grid
    F = number of radar fields
    H = number of radar heights
    C = number of field/height pairs

    :param storm_ids: See documentation at top of file.
    :param storm_times_unix_sec: Same.
    :param top_radar_image_dir_name: Same.
    :param radar_source: Same.
    :param radar_field_names: Same.
    :param radar_heights_m_agl: Same.
    :param refl_heights_m_agl: Same.
    :param num_rows_to_keep: Same.
    :param num_columns_to_keep: Same.
    :param top_sounding_dir_name: Same.
    :param sounding_lag_time_sec: Same.
    :param sounding_lead_time_sec: Same.
    :return: storm_object_dict: Dictionary with the following keys.
    storm_object_dict['radar_field_names']: length-F list with names of radar
        fields.  If radar source is MYRORSS, this is `None`.
    storm_object_dict['radar_heights_m_agl']: length-H numpy array of radar
        heights (metres above ground level).  If radar source is MYRORSS, this
        is `None`.
    storm_object_dict['field_name_by_pair']: length-C list with names of radar
        fields.  If radar source is GridRad, this is `None`.
    storm_object_dict['height_by_pair_m_agl']: length-C numpy array of radar
        heights (metres above ground level).  If radar source is GridRad, this
        is `None`.
    storm_object_dict['storm_ids']: length-E list of storm IDs.
    storm_object_dict['storm_times_unix_sec']: length-E numpy array of storm
        times.
    storm_object_dict['radar_image_matrix']: numpy array
        (either E x M x N x C or E x M x N x H x F) of radar values.
    storm_object_dict['sounding_matrix']: 3-D numpy array of sounding values.
    """

    dummy_target_name = labels.get_column_name_for_classification_label(
        min_lead_time_sec=sounding_lead_time_sec,
        max_lead_time_sec=sounding_lead_time_sec, min_link_distance_metres=0,
        max_link_distance_metres=1,
        event_type_string=events2storms.TORNADO_EVENT_TYPE_STRING)

    storm_spc_date_strings_numpy = numpy.array(
        [time_conversion.time_to_spc_date_string(t)
         for t in storm_times_unix_sec],
        dtype=object)
    unique_spc_date_strings_numpy = numpy.unique(storm_spc_date_strings_numpy)
    num_spc_dates = len(unique_spc_date_strings_numpy)

    radar_image_matrix = None
    sounding_matrix = None
    field_name_by_pair = None
    height_by_pair_m_agl = None
    sort_indices_for_storm_id = numpy.array([], dtype=int)

    read_soundings = top_sounding_dir_name != ''
    if read_soundings:
        sounding_field_names = SOUNDING_FIELD_NAMES + []
    else:
        sounding_field_names = None

    for i in range(num_spc_dates):
        if radar_source == radar_utils.GRIDRAD_SOURCE_ID:
            this_radar_file_name_matrix = trainval_io.find_radar_files_3d(
                top_directory_name=top_radar_image_dir_name,
                radar_source=radar_source, radar_field_names=radar_field_names,
                radar_heights_m_agl=radar_heights_m_agl,
                first_file_time_unix_sec=
                time_conversion.spc_date_string_to_unix_sec(
                    unique_spc_date_strings_numpy[i]),
                last_file_time_unix_sec=
                time_conversion.spc_date_string_to_unix_sec(
                    unique_spc_date_strings_numpy[i]),
                one_file_per_time_step=False, shuffle_times=False)[0]


            print MINOR_SEPARATOR_STRING

            this_storm_object_dict = deployment_io.create_storm_images_3d({
                deployment_io.RADAR_FILE_NAMES_KEY: this_radar_file_name_matrix,
                deployment_io.NUM_EXAMPLES_PER_FILE_KEY: LARGE_INTEGER,
                deployment_io.NUM_ROWS_TO_KEEP_KEY: num_rows_to_keep,
                deployment_io.NUM_COLUMNS_TO_KEEP_KEY: num_columns_to_keep,
                deployment_io.NORMALIZATION_TYPE_KEY: None,
                deployment_io.RETURN_TARGET_KEY: False,
                deployment_io.TARGET_NAME_KEY: dummy_target_name,
                deployment_io.SOUNDING_FIELDS_KEY: sounding_field_names,
                deployment_io.SOUNDING_DIRECTORY_KEY: top_sounding_dir_name,
                deployment_io.SOUNDING_LAG_TIME_KEY: sounding_lag_time_sec,
                deployment_io.REFLECTIVITY_MASK_KEY: None
            })
        else:
            this_radar_file_name_matrix = trainval_io.find_radar_files_2d(
                top_directory_name=top_radar_image_dir_name,
                radar_source=radar_source, radar_field_names=radar_field_names,
                first_file_time_unix_sec=
                time_conversion.spc_date_string_to_unix_sec(
                    unique_spc_date_strings_numpy[i]),
                last_file_time_unix_sec=
                time_conversion.spc_date_string_to_unix_sec(
                    unique_spc_date_strings_numpy[i]),
                one_file_per_time_step=False, shuffle_times=False,
                reflectivity_heights_m_agl=refl_heights_m_agl)[0]
            print MINOR_SEPARATOR_STRING

            if i == 0:
                field_name_by_pair = [
                    storm_images.image_file_name_to_field(f) for f in
                    this_radar_file_name_matrix[0, :]
                ]
                height_by_pair_m_agl = numpy.array([
                    storm_images.image_file_name_to_height(f) for f in
                    this_radar_file_name_matrix[0, :]
                ], dtype=int)

            this_storm_object_dict = deployment_io.create_storm_images_2d({
                deployment_io.RADAR_FILE_NAMES_KEY: this_radar_file_name_matrix,
                deployment_io.NUM_EXAMPLES_PER_FILE_KEY: LARGE_INTEGER,
                deployment_io.NUM_ROWS_TO_KEEP_KEY: num_rows_to_keep,
                deployment_io.NUM_COLUMNS_TO_KEEP_KEY: num_columns_to_keep,
                deployment_io.NORMALIZATION_TYPE_KEY: None,
                deployment_io.RETURN_TARGET_KEY: False,
                deployment_io.TARGET_NAME_KEY: dummy_target_name,
                deployment_io.SOUNDING_FIELDS_KEY: sounding_field_names,
                deployment_io.SOUNDING_DIRECTORY_KEY: top_sounding_dir_name,
                deployment_io.SOUNDING_LAG_TIME_KEY: sounding_lag_time_sec
            })

        these_indices = numpy.where(
            storm_spc_date_strings_numpy == unique_spc_date_strings_numpy[i])[0]
        sort_indices_for_storm_id = numpy.concatenate((
            sort_indices_for_storm_id, these_indices))

        # TODO(thunderhoser): Handle possibility of missing storm objects.
        these_indices = tracking_utils.find_storm_objects(
            all_storm_ids=this_storm_object_dict[deployment_io.STORM_IDS_KEY],
            all_times_unix_sec=this_storm_object_dict[
                deployment_io.STORM_TIMES_KEY],
            storm_ids_to_keep=[storm_ids[k] for k in these_indices],
            times_to_keep_unix_sec=storm_times_unix_sec[these_indices])

        if radar_image_matrix is None:
            radar_image_matrix = this_storm_object_dict[
                deployment_io.RADAR_IMAGE_MATRIX_KEY][these_indices, ...] + 0.

            if read_soundings:
                sounding_matrix = this_storm_object_dict[
                    deployment_io.SOUNDING_MATRIX_KEY][these_indices, ...] + 0.
        else:
            radar_image_matrix = numpy.concatenate(
                (radar_image_matrix,
                 this_storm_object_dict[deployment_io.RADAR_IMAGE_MATRIX_KEY][
                     these_indices, ...]),
                axis=0)

            if read_soundings:
                sounding_matrix = numpy.concatenate(
                    (sounding_matrix,
                     this_storm_object_dict[deployment_io.SOUNDING_MATRIX_KEY][
                         these_indices, ...]),
                    axis=0)

        if i != num_spc_dates - 1:
            print MINOR_SEPARATOR_STRING

    return {
        FIELD_NAMES_2D_KEY: field_name_by_pair,
        HEIGHTS_2D_KEY: height_by_pair_m_agl,
        FIELD_NAMES_3D_KEY: radar_field_names,
        HEIGHTS_3D_KEY: radar_heights_m_agl,
        STORM_IDS_KEY: [storm_ids[k] for k in sort_indices_for_storm_id],
        STORM_TIMES_KEY: storm_times_unix_sec[sort_indices_for_storm_id],
        RADAR_IMAGE_MATRIX_KEY: radar_image_matrix,
        SOUNDING_MATRIX_KEY: sounding_matrix
    }


def _plot_storm_objects(storm_object_dict, output_dir_name):
    """Plots radar and sounding data for each storm object.

    :param storm_object_dict: Dictionary created by `_read_inputs`, but with one
        additional key.
    storm_object_dict['storm_activations']: length-E numpy array of model
        activations, where E = number of storm objects.  This may also be None,
        in which case activations will not be included in figure titles.
    :param output_dir_name: Name of output directory.  Figures will be saved
        here.
    """

    field_name_by_pair = storm_object_dict[FIELD_NAMES_2D_KEY]
    height_by_pair_m_agl = storm_object_dict[HEIGHTS_2D_KEY]
    radar_field_names = storm_object_dict[FIELD_NAMES_3D_KEY]
    radar_heights_m_agl = storm_object_dict[HEIGHTS_3D_KEY]
    storm_ids = storm_object_dict[STORM_IDS_KEY]
    storm_times_unix_sec = storm_object_dict[STORM_TIMES_KEY]
    radar_image_matrix = storm_object_dict[RADAR_IMAGE_MATRIX_KEY]
    sounding_matrix = storm_object_dict[SOUNDING_MATRIX_KEY]
    storm_activations = storm_object_dict[STORM_ACTIVATIONS_KEY]

    plot_soundings = sounding_matrix is not None
    if plot_soundings:
        list_of_metpy_dictionaries = dl_utils.soundings_to_metpy_dictionaries(
            sounding_matrix=sounding_matrix, field_names=SOUNDING_FIELD_NAMES)

    num_radar_dimensions = 2 + int(radar_field_names is not None)
    num_storm_objects = len(storm_ids)

    for i in range(num_storm_objects):
        this_time_string = time_conversion.unix_sec_to_string(
            storm_times_unix_sec[i], TIME_FORMAT)
        this_base_title_string = 'Storm "{0:s}" at {1:s}'.format(
            storm_ids[i], this_time_string)
        if storm_activations is not None:
            this_base_title_string += ' (activation = {0:.3f})'.format(
                storm_activations[i])

        this_base_file_name = '{0:s}/storm={1:s}_{2:s}'.format(
            output_dir_name, storm_ids[i].replace('_', '-'), this_time_string)

        if num_radar_dimensions == 2:
            j_max = 1
        else:
            j_max = len(radar_field_names)

        for j in range(j_max):
            if num_radar_dimensions == 2:
                radar_plotting.plot_many_2d_grids_without_coords(
                    field_matrix=numpy.flip(radar_image_matrix[i, ...], axis=0),
                    field_name_by_pair=field_name_by_pair,
                    height_by_pair_metres=height_by_pair_m_agl,
                    ground_relative=True, num_panel_rows=NUM_PANEL_ROWS)

                this_title_string = this_base_title_string + ''
                this_figure_file_name = '{0:s}.jpg'.format(this_base_file_name)
            else:
                (_, these_axes_objects_2d_list
                ) = radar_plotting.plot_3d_grid_without_coords(
                    field_matrix=numpy.flip(
                        radar_image_matrix[i, ..., j], axis=0),
                    field_name=radar_field_names[j],
                    grid_point_heights_metres=radar_heights_m_agl,
                    ground_relative=True, num_panel_rows=NUM_PANEL_ROWS)

                (this_colour_map_object, this_colour_norm_object, _
                ) = radar_plotting.get_default_colour_scheme(
                    radar_field_names[j])

                plotting_utils.add_colour_bar(
                    axes_object_or_list=these_axes_objects_2d_list,
                    values_to_colour=radar_image_matrix[i, ..., j],
                    colour_map=this_colour_map_object,
                    colour_norm_object=this_colour_norm_object,
                    orientation='horizontal', extend_min=True, extend_max=True)

                this_title_string = '{0:s}; {1:s}'.format(
                    this_base_title_string, radar_field_names[j])
                this_figure_file_name = '{0:s}_{1:s}.jpg'.format(
                    this_base_file_name, radar_field_names[j].replace('_', '-'))

            pyplot.suptitle(this_title_string, fontsize=TITLE_FONT_SIZE)
            print 'Saving figure to: "{0:s}"...'.format(this_figure_file_name)
            pyplot.savefig(this_figure_file_name, dpi=DOTS_PER_INCH)
            pyplot.close()

        if not plot_soundings:
            continue

        sounding_plotting.plot_sounding(
            sounding_dict_for_metpy=list_of_metpy_dictionaries[i],
            title_string=this_base_title_string)

        this_figure_file_name = '{0:s}_sounding.jpg'.format(this_base_file_name)
        print 'Saving figure to: "{0:s}"...'.format(this_figure_file_name)
        pyplot.savefig(this_figure_file_name, dpi=DOTS_PER_INCH)
        pyplot.close()


def _run(
        input_activation_file_name, input_storm_metafile_name,
        top_radar_image_dir_name, radar_source, radar_field_names,
        radar_heights_m_agl, refl_heights_m_agl, num_rows_to_keep,
        num_columns_to_keep, top_sounding_dir_name, sounding_lag_time_sec,
        sounding_lead_time_sec, output_dir_name):
    """Plots radar and sounding data for each storm object.

    This is effectively the main method.

    :param input_activation_file_name: See documentation at top of file.
    :param input_storm_metafile_name: Same.
    :param top_radar_image_dir_name: Same.
    :param radar_source: Same.
    :param radar_field_names: Same.
    :param radar_heights_m_agl: Same.
    :param refl_heights_m_agl: Same.
    :param num_rows_to_keep: Same.
    :param num_columns_to_keep: Same.
    :param top_sounding_dir_name: Same.
    :param sounding_lag_time_sec: Same.
    :param sounding_lead_time_sec: Same.
    :param output_dir_name: Same.
    :raises: ValueError: if the activation file contains activations for more
        than one model component.
    """

    if input_activation_file_name != '':

        # Read activation file.
        print 'Reading data from: "{0:s}"...'.format(input_activation_file_name)
        (activation_matrix, activation_metadata_dict
        ) = model_activation.read_file(input_activation_file_name)

        num_model_components = activation_matrix.shape[1]
        if num_model_components > 1:
            error_string = (
                'The file should contain activations for only one model '
                'component, not {0:d}.'
            ).format(num_model_components)
            raise ValueError(error_string)

        storm_ids = activation_metadata_dict[model_activation.STORM_IDS_KEY]
        storm_times_unix_sec = activation_metadata_dict[
            model_activation.STORM_TIMES_KEY]
        storm_activations = activation_matrix[:, 0]

        # Read metadata for model that generated the activations.
        model_file_name = activation_metadata_dict[
            model_activation.MODEL_FILE_NAME_KEY]
        model_metafile_name = '{0:s}/model_metadata.p'.format(
            os.path.split(model_file_name)[0])

        print 'Reading metadata from: "{0:s}"...'.format(model_metafile_name)
        model_metadata_dict = cnn.read_model_metadata(model_metafile_name)
        training_option_dict = model_metadata_dict[cnn.TRAINING_OPTION_DICT_KEY]

        radar_source = model_metadata_dict[cnn.RADAR_SOURCE_KEY]
        radar_field_names = model_metadata_dict[cnn.RADAR_FIELDS_KEY]
        radar_heights_m_agl = model_metadata_dict[cnn.RADAR_HEIGHTS_KEY]
        refl_heights_m_agl = model_metadata_dict[cnn.REFLECTIVITY_HEIGHTS_KEY]
        num_rows_to_keep = training_option_dict[
            trainval_io.NUM_ROWS_TO_KEEP_KEY]
        num_columns_to_keep = training_option_dict[
            trainval_io.NUM_COLUMNS_TO_KEEP_KEY]
        sounding_lag_time_sec = training_option_dict[
            trainval_io.SOUNDING_LAG_TIME_KEY]

        target_param_dict = labels.column_name_to_label_params(
            training_option_dict[trainval_io.TARGET_NAME_KEY])
        min_lead_time_sec = target_param_dict[labels.MIN_LEAD_TIME_KEY]
        max_lead_time_sec = target_param_dict[labels.MAX_LEAD_TIME_KEY]
        sounding_lead_time_sec = int(numpy.round(
            numpy.mean([min_lead_time_sec, max_lead_time_sec])))
    else:

        # Read storm IDs and times.
        print 'Reading data from: "{0:s}"...'.format(input_storm_metafile_name)
        storm_ids, storm_times_unix_sec = tracking_io.read_storm_ids_and_times(
            input_storm_metafile_name)
        storm_activations = None

    if num_rows_to_keep <= 0:
        num_rows_to_keep = None
    if num_columns_to_keep <= 0:
        num_columns_to_keep = None
    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=output_dir_name)

    storm_object_dict = _read_inputs(
        storm_ids=storm_ids,
        storm_times_unix_sec=storm_times_unix_sec,
        top_radar_image_dir_name=top_radar_image_dir_name,
        radar_source=radar_source, radar_field_names=radar_field_names,
        radar_heights_m_agl=radar_heights_m_agl,
        refl_heights_m_agl=refl_heights_m_agl,
        num_rows_to_keep=num_rows_to_keep,
        num_columns_to_keep=num_columns_to_keep,
        top_sounding_dir_name=top_sounding_dir_name,
        sounding_lag_time_sec=sounding_lag_time_sec,
        sounding_lead_time_sec=sounding_lead_time_sec)
    print SEPARATOR_STRING

    if storm_activations is not None:
        these_indices = tracking_utils.find_storm_objects(
            all_storm_ids=storm_object_dict[STORM_IDS_KEY],
            all_times_unix_sec=storm_object_dict[STORM_TIMES_KEY],
            storm_ids_to_keep=storm_ids,
            times_to_keep_unix_sec=storm_times_unix_sec)
        storm_activations = storm_activations[these_indices]

    storm_object_dict.update({STORM_ACTIVATIONS_KEY: storm_activations})
    _plot_storm_objects(
        storm_object_dict=storm_object_dict, output_dir_name=output_dir_name)


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        input_activation_file_name=getattr(
            INPUT_ARG_OBJECT, ACTIVATION_FILE_ARG_NAME),
        input_storm_metafile_name=getattr(
            INPUT_ARG_OBJECT, STORM_METAFILE_ARG_NAME),
        top_radar_image_dir_name=getattr(
            INPUT_ARG_OBJECT, RADAR_IMAGE_DIR_ARG_NAME),
        radar_source=getattr(INPUT_ARG_OBJECT, RADAR_SOURCE_ARG_NAME),
        radar_field_names=getattr(INPUT_ARG_OBJECT, RADAR_FIELDS_ARG_NAME),
        radar_heights_m_agl=numpy.array(
            getattr(INPUT_ARG_OBJECT, RADAR_HEIGHTS_ARG_NAME), dtype=int),
        refl_heights_m_agl=numpy.array(
            getattr(INPUT_ARG_OBJECT, REFL_HEIGHTS_ARG_NAME), dtype=int),
        num_rows_to_keep=getattr(INPUT_ARG_OBJECT, NUM_ROWS_TO_KEEP_ARG_NAME),
        num_columns_to_keep=getattr(
            INPUT_ARG_OBJECT, NUM_COLUMNS_TO_KEEP_ARG_NAME),
        top_sounding_dir_name=getattr(INPUT_ARG_OBJECT, SOUNDING_DIR_ARG_NAME),
        sounding_lag_time_sec=getattr(
            INPUT_ARG_OBJECT, SOUNDING_LAG_TIME_ARG_NAME),
        sounding_lead_time_sec=getattr(
            INPUT_ARG_OBJECT, SOUNDING_LEAD_TIME_ARG_NAME),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME))
