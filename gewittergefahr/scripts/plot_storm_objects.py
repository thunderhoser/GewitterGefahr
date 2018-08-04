"""Plots radar and sounding data for each storm object."""

import argparse
import numpy
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as pyplot
from gewittergefahr.gg_utils import labels
from gewittergefahr.gg_utils import link_events_to_storms as events2storms
from gewittergefahr.gg_utils import radar_utils
from gewittergefahr.gg_utils import soundings_only
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.deep_learning import storm_images
from gewittergefahr.deep_learning import deployment_io
from gewittergefahr.deep_learning import training_validation_io as trainval_io
from gewittergefahr.plotting import plotting_utils
from gewittergefahr.plotting import radar_plotting

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'
MINOR_SEPARATOR_STRING = '\n\n' + '-' * 50 + '\n\n'

# SOUNDING_FIELD_NAMES = [
#     soundings_only.U_WIND_NAME, soundings_only.V_WIND_NAME,
#     soundings_only.TEMPERATURE_NAME, soundings_only.SPECIFIC_HUMIDITY_NAME
# ]

# TODO(thunderhoser): Fix this hack.
SOUNDING_FIELD_NAMES = None

FIELD_NAMES_2D_KEY = 'field_name_by_pair'
HEIGHTS_2D_KEY = 'height_by_pair_m_asl'
FIELD_NAMES_3D_KEY = 'radar_field_names'
HEIGHTS_3D_KEY = 'radar_heights_m_asl'
STORM_IDS_KEY = 'storm_ids'
STORM_TIMES_KEY = 'storm_times_unix_sec'
RADAR_IMAGE_MATRIX_KEY = 'radar_image_matrix'
SOUNDING_MATRIX_KEY = 'sounding_matrix'

LARGE_INTEGER = int(1e10)
TIME_FORMAT = '%Y-%m-%d-%H%M%S'

NUM_PANEL_ROWS = 3
TITLE_FONT_SIZE = 20
DOTS_PER_INCH = 300

STORM_IDS_ARG_NAME = 'storm_ids'
STORM_TIMES_ARG_NAME = 'storm_times_unix_sec'
RADAR_IMAGE_DIR_ARG_NAME = 'input_radar_image_dir_name'
RADAR_SOURCE_ARG_NAME = 'radar_source'
RADAR_FIELDS_ARG_NAME = 'radar_field_names'
RADAR_HEIGHTS_ARG_NAME = 'radar_heights_m_asl'
REFL_HEIGHTS_ARG_NAME = 'refl_heights_m_asl'
SOUNDING_DIR_ARG_NAME = 'input_sounding_dir_name'
SOUNDING_LAG_TIME_ARG_NAME = 'sounding_lag_time_sec'
SOUNDING_LEAD_TIME_ARG_NAME = 'sounding_lead_time_sec'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

STORM_IDS_HELP_STRING = 'List of storm IDs (one per storm object).'
STORM_TIMES_HELP_STRING = (
    'List of storm times (one per storm object).  This list must have the same '
    'length as `{0:s}`.'
).format(STORM_IDS_ARG_NAME)
RADAR_IMAGE_DIR_HELP_STRING = (
    'Name of top-level directory with storm-centered radar images.  Files '
    'therein will be found by `storm_images.find_storm_image_file` and read by '
    '`storm_images.read_storm_images`.')
RADAR_SOURCE_HELP_STRING = (
    'Radar source.  Must be one of the following list.\n{0:s}'
).format(str(radar_utils.DATA_SOURCE_IDS))
RADAR_FIELDS_HELP_STRING = (
    'List of radar fields to plot.  Each must be accepted by `radar_utils.'
    'check_field_name`.')
RADAR_HEIGHTS_HELP_STRING = (
    '[used only if {0:s} = "{1:s}"] List of radar heights.  Each field in '
    '`{2:s}` will be plotted at each height.'
).format(RADAR_SOURCE_ARG_NAME, radar_utils.GRIDRAD_SOURCE_ID,
         RADAR_FIELDS_ARG_NAME)
REFL_HEIGHTS_HELP_STRING = (
    '[used only if {0:s} = "{1:s}"] List of reflectivity heights.  "{2:s}" will'
    ' be plotted at each of these heights.'
).format(RADAR_SOURCE_ARG_NAME, radar_utils.MYRORSS_SOURCE_ID,
         radar_utils.REFL_NAME)
SOUNDING_DIR_HELP_STRING = (
    'Name of top-level directory with storm-centered soundings.  Files therein '
    'will be found by `soundings_only.find_sounding_file` and read by '
    '`soundings_only.read_soundings`.')
SOUNDING_LAG_TIME_HELP_STRING = 'Lag time (used to find sounding files).'
SOUNDING_LEAD_TIME_HELP_STRING = 'Lead time (used to find sounding files).'
OUTPUT_DIR_HELP_STRING = (
    'Name of output directory.  Figures will be saved here.')

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + STORM_IDS_ARG_NAME, type=str, nargs='+', required=True,
    help=STORM_IDS_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + STORM_TIMES_ARG_NAME, type=int, nargs='+', required=True,
    help=STORM_TIMES_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + RADAR_IMAGE_DIR_ARG_NAME, type=str, required=True,
    help=RADAR_IMAGE_DIR_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + RADAR_SOURCE_ARG_NAME, type=str, required=True,
    help=RADAR_SOURCE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + RADAR_FIELDS_ARG_NAME, type=str, nargs='+', required=True,
    help=RADAR_FIELDS_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + RADAR_HEIGHTS_ARG_NAME, type=int, nargs='+', required=False,
    default=[-1], help=RADAR_HEIGHTS_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + REFL_HEIGHTS_ARG_NAME, type=int, nargs='+', required=False,
    default=[-1], help=REFL_HEIGHTS_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + SOUNDING_DIR_ARG_NAME, type=str, required=True,
    help=SOUNDING_DIR_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + SOUNDING_LAG_TIME_ARG_NAME, type=int, required=False, default=1800,
    help=SOUNDING_LAG_TIME_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + SOUNDING_LEAD_TIME_ARG_NAME, type=int, required=True,
    help=SOUNDING_LEAD_TIME_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING)


def _read_inputs(
        storm_ids, storm_times_unix_sec, top_radar_image_dir_name, radar_source,
        radar_field_names, radar_heights_m_asl, refl_heights_m_asl,
        top_sounding_dir_name, sounding_lag_time_sec, sounding_lead_time_sec):
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
    :param radar_heights_m_asl: Same.
    :param refl_heights_m_asl: Same.
    :param top_sounding_dir_name: Same.
    :param sounding_lag_time_sec: Same.
    :param sounding_lead_time_sec: Same.
    :return: storm_object_dict: Dictionary with the following keys.
    storm_object_dict['radar_field_names']: length-F list with names of radar
        fields.  If radar source is MYRORSS, this is `None`.
    storm_object_dict['radar_heights_m_asl']: length-H numpy array of radar
        heights (metres above sea level).  If radar source is MYRORSS, this is
        `None`.
    storm_object_dict['field_name_by_pair']: length-C list with names of radar
        fields.  If radar source is GridRad, this is `None`.
    storm_object_dict['height_by_pair_m_asl']: length-C numpy array of radar
        heights (metres above sea level).  If radar source is GridRad, this is
        `None`.
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
    height_by_pair_m_asl = None
    sort_indices_for_storm_id = numpy.array([], dtype=int)

    for i in range(num_spc_dates):
        if radar_source == radar_utils.GRIDRAD_SOURCE_ID:
            this_radar_file_name_matrix = trainval_io.find_radar_files_3d(
                top_directory_name=top_radar_image_dir_name,
                radar_source=radar_source, radar_field_names=radar_field_names,
                radar_heights_m_asl=radar_heights_m_asl,
                first_file_time_unix_sec=
                time_conversion.spc_date_string_to_unix_sec(
                    unique_spc_date_strings_numpy[i]),
                last_file_time_unix_sec=
                time_conversion.spc_date_string_to_unix_sec(
                    unique_spc_date_strings_numpy[i]),
                one_file_per_time_step=False, shuffle_times=False)[0]
            print MINOR_SEPARATOR_STRING

            this_storm_object_dict = deployment_io.create_storm_images_3d(
                radar_file_name_matrix=this_radar_file_name_matrix,
                num_examples_per_file_time=LARGE_INTEGER, return_target=False,
                target_name=dummy_target_name, radar_normalization_dict=None,
                refl_masking_threshold_dbz=None,
                return_rotation_divergence_product=False,
                sounding_field_names=SOUNDING_FIELD_NAMES,
                top_sounding_dir_name=top_sounding_dir_name,
                sounding_lag_time_for_convective_contamination_sec=
                sounding_lag_time_sec,
                sounding_normalization_dict=None)
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
                reflectivity_heights_m_asl=refl_heights_m_asl)[0]
            print MINOR_SEPARATOR_STRING

            if i == 0:
                field_name_by_pair = [
                    storm_images.image_file_name_to_field(f) for f in
                    this_radar_file_name_matrix[0, :]
                ]
                height_by_pair_m_asl = numpy.array([
                    storm_images.image_file_name_to_height(f) for f in
                    this_radar_file_name_matrix[0, :]
                ], dtype=int)

            this_storm_object_dict = deployment_io.create_storm_images_2d(
                radar_file_name_matrix=this_radar_file_name_matrix,
                num_examples_per_file_time=LARGE_INTEGER, return_target=False,
                target_name=dummy_target_name, radar_normalization_dict=None,
                sounding_field_names=SOUNDING_FIELD_NAMES,
                top_sounding_dir_name=top_sounding_dir_name,
                sounding_lag_time_for_convective_contamination_sec=
                sounding_lag_time_sec,
                sounding_normalization_dict=None)

        these_indices = numpy.where(
            storm_spc_date_strings_numpy == unique_spc_date_strings_numpy[i])[0]
        sort_indices_for_storm_id = numpy.concatenate((
            sort_indices_for_storm_id, these_indices))

        # TODO(thunderhoser): Handle possibility of missing storm objects.
        these_indices = storm_images.find_storm_objects(
            all_storm_ids=this_storm_object_dict[deployment_io.STORM_IDS_KEY],
            all_valid_times_unix_sec=this_storm_object_dict[
                deployment_io.STORM_TIMES_KEY],
            storm_ids_to_keep=[storm_ids[k] for k in these_indices],
            valid_times_to_keep_unix_sec=storm_times_unix_sec[these_indices])

        if radar_image_matrix is None:
            radar_image_matrix = this_storm_object_dict[
                deployment_io.RADAR_IMAGE_MATRIX_KEY][these_indices, ...] + 0.

            if SOUNDING_FIELD_NAMES is not None:
                sounding_matrix = this_storm_object_dict[
                    deployment_io.SOUNDING_MATRIX_KEY][these_indices, ...] + 0.
        else:
            radar_image_matrix = numpy.concatenate(
                (radar_image_matrix,
                 this_storm_object_dict[deployment_io.RADAR_IMAGE_MATRIX_KEY][
                     these_indices, ...]),
                axis=0)

            if SOUNDING_FIELD_NAMES is not None:
                sounding_matrix = numpy.concatenate(
                    (sounding_matrix,
                     this_storm_object_dict[deployment_io.SOUNDING_MATRIX_KEY][
                         these_indices, ...]),
                    axis=0)

        if i != num_spc_dates - 1:
            print MINOR_SEPARATOR_STRING

    return {
        FIELD_NAMES_2D_KEY: field_name_by_pair,
        HEIGHTS_2D_KEY: height_by_pair_m_asl,
        FIELD_NAMES_3D_KEY: radar_field_names,
        HEIGHTS_3D_KEY: radar_heights_m_asl,
        STORM_IDS_KEY: [storm_ids[k] for k in sort_indices_for_storm_id],
        STORM_TIMES_KEY: storm_times_unix_sec[sort_indices_for_storm_id],
        RADAR_IMAGE_MATRIX_KEY: radar_image_matrix,
        SOUNDING_MATRIX_KEY: sounding_matrix
    }


def _plot_storm_objects(storm_object_dict, output_dir_name):
    """Plots radar and sounding data for each storm object.

    :param storm_object_dict: Dictionary created by `_read_inputs`.
    :param output_dir_name: Name of output directory.  Figures will be saved
        here.
    """

    field_name_by_pair = storm_object_dict[FIELD_NAMES_2D_KEY]
    height_by_pair_m_asl = storm_object_dict[HEIGHTS_2D_KEY]
    radar_field_names = storm_object_dict[FIELD_NAMES_3D_KEY]
    radar_heights_m_asl = storm_object_dict[HEIGHTS_3D_KEY]
    storm_ids = storm_object_dict[STORM_IDS_KEY]
    storm_times_unix_sec = storm_object_dict[STORM_TIMES_KEY]
    radar_image_matrix = storm_object_dict[RADAR_IMAGE_MATRIX_KEY]

    num_radar_dimensions = 2 + int(radar_field_names is not None)
    num_storm_objects = len(storm_ids)

    for i in range(num_storm_objects):
        this_time_string = time_conversion.unix_sec_to_string(
            storm_times_unix_sec[i], TIME_FORMAT)
        this_title_string = 'Storm "{0:s}" at {1:s}'.format(
            storm_ids[i], this_time_string)
        this_base_file_name = '{0:s}/storm={1:s}_{2:s}'.format(
            output_dir_name, storm_ids[i].replace('_', '-'), this_time_string)

        if num_radar_dimensions == 2:
            j_max = 1
        else:
            j_max = len(radar_field_names)

        for j in range(j_max):
            if num_radar_dimensions == 2:
                radar_plotting.plot_many_2d_grids_without_coords(
                    field_matrix=radar_image_matrix[i, ...],
                    field_name_by_pair=field_name_by_pair,
                    height_by_pair_m_asl=height_by_pair_m_asl,
                    num_panel_rows=NUM_PANEL_ROWS)

                this_figure_file_name = '{0:s}.jpg'.format(this_base_file_name)
            else:
                (_, these_axes_objects_2d_list
                ) = radar_plotting.plot_3d_grid_without_coords(
                    field_matrix=radar_image_matrix[i, ..., j],
                    field_name=radar_field_names[j],
                    grid_point_heights_m_asl=radar_heights_m_asl,
                    num_panel_rows=NUM_PANEL_ROWS)

                (this_colour_map_object, this_colour_norm_object, _
                ) = radar_plotting.get_default_colour_scheme(
                    radar_field_names[j])

                plotting_utils.add_colour_bar(
                    axes_object=these_axes_objects_2d_list,
                    values_to_colour=radar_image_matrix[i, ..., j],
                    colour_map=this_colour_map_object,
                    colour_norm_object=this_colour_norm_object,
                    orientation='vertical', extend_min=True, extend_max=True)

                this_figure_file_name = '{0:s}_{1:s}.jpg'.format(
                    this_base_file_name, radar_field_names[j].replace('_', '-'))

            pyplot.suptitle(this_title_string, fontsize=TITLE_FONT_SIZE)
            print 'Saving figure to: "{0:s}"...'.format(this_figure_file_name)
            pyplot.savefig(this_figure_file_name, dpi=DOTS_PER_INCH)
            pyplot.close()


def _run(
        storm_ids, storm_times_unix_sec, top_radar_image_dir_name, radar_source,
        radar_field_names, radar_heights_m_asl, refl_heights_m_asl,
        top_sounding_dir_name, sounding_lag_time_sec, sounding_lead_time_sec,
        output_dir_name):
    """Plots radar and sounding data for each storm object.

    This is effectively the main method.

    :param storm_ids: See documentation at top of file.
    :param storm_times_unix_sec: Same.
    :param top_radar_image_dir_name: Same.
    :param radar_source: Same.
    :param radar_field_names: Same.
    :param radar_heights_m_asl: Same.
    :param refl_heights_m_asl: Same.
    :param top_sounding_dir_name: Same.
    :param sounding_lag_time_sec: Same.
    :param sounding_lead_time_sec: Same.
    :param output_dir_name: Same.
    """

    storm_object_dict = _read_inputs(
        storm_ids=storm_ids, storm_times_unix_sec=storm_times_unix_sec,
        top_radar_image_dir_name=top_radar_image_dir_name,
        radar_source=radar_source, radar_field_names=radar_field_names,
        radar_heights_m_asl=radar_heights_m_asl,
        refl_heights_m_asl=refl_heights_m_asl,
        top_sounding_dir_name=top_sounding_dir_name,
        sounding_lag_time_sec=sounding_lag_time_sec,
        sounding_lead_time_sec=sounding_lead_time_sec)
    print SEPARATOR_STRING

    _plot_storm_objects(
        storm_object_dict=storm_object_dict, output_dir_name=output_dir_name)


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        storm_ids=getattr(INPUT_ARG_OBJECT, STORM_IDS_ARG_NAME),
        storm_times_unix_sec=numpy.array(
            getattr(INPUT_ARG_OBJECT, STORM_TIMES_ARG_NAME), dtype=int),
        top_radar_image_dir_name=getattr(
            INPUT_ARG_OBJECT, RADAR_IMAGE_DIR_ARG_NAME),
        radar_source=getattr(INPUT_ARG_OBJECT, RADAR_SOURCE_ARG_NAME),
        radar_field_names=getattr(INPUT_ARG_OBJECT, RADAR_FIELDS_ARG_NAME),
        radar_heights_m_asl=numpy.array(
            getattr(INPUT_ARG_OBJECT, RADAR_HEIGHTS_ARG_NAME), dtype=int),
        refl_heights_m_asl=numpy.array(
            getattr(INPUT_ARG_OBJECT, REFL_HEIGHTS_ARG_NAME), dtype=int),
        top_sounding_dir_name=getattr(INPUT_ARG_OBJECT, SOUNDING_DIR_ARG_NAME),
        sounding_lag_time_sec=getattr(
            INPUT_ARG_OBJECT, SOUNDING_LAG_TIME_ARG_NAME),
        sounding_lead_time_sec=getattr(
            INPUT_ARG_OBJECT, SOUNDING_LEAD_TIME_ARG_NAME),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME))
