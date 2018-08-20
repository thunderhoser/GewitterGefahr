"""Plots examples (storm objects) with extremely low or high CNN activations.

CNN = convolutional neural network
"""

import os.path
import argparse
import numpy
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as pyplot
from gewittergefahr.gg_utils import labels
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import nwp_model_utils
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.gg_utils import error_checking
from gewittergefahr.deep_learning import cnn
from gewittergefahr.deep_learning import storm_images
from gewittergefahr.deep_learning import deep_learning_utils as dl_utils
from gewittergefahr.deep_learning import model_activation
from gewittergefahr.deep_learning import deployment_io
from gewittergefahr.deep_learning import training_validation_io as trainval_io
from gewittergefahr.plotting import plotting_utils
from gewittergefahr.plotting import radar_plotting
from gewittergefahr.plotting import sounding_plotting

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'
MINOR_SEPARATOR_STRING = '\n\n' + '-' * 50 + '\n\n'

SOUNDING_PRESSURE_LEVELS_MB = nwp_model_utils.get_pressure_levels(
    model_name=nwp_model_utils.RAP_MODEL_NAME,
    grid_id=nwp_model_utils.ID_FOR_130GRID)

LARGE_INTEGER = int(1e10)
TIME_FORMAT = '%Y-%m-%d-%H%M%S'

NUM_PANEL_ROWS = 3
TITLE_FONT_SIZE = 20
DOTS_PER_INCH = 300

FIELD_NAMES_2D_KEY = 'field_name_by_pair'
HEIGHTS_2D_KEY = 'height_by_pair_m_asl'
FIELD_NAMES_3D_KEY = 'radar_field_names'
HEIGHTS_3D_KEY = 'radar_heights_m_asl'
STORM_IDS_KEY = 'storm_ids'
STORM_TIMES_KEY = 'storm_times_unix_sec'
TARGET_VALUES_KEY = 'storm_target_values'
STORM_ACTIVATIONS_KEY = 'storm_activations'
RADAR_IMAGE_MATRIX_KEY = 'radar_image_matrix'
SOUNDING_FIELD_NAMES_KEY = 'sounding_field_names'
SOUNDING_MATRIX_KEY = 'sounding_matrix'

ACTIVATION_FILE_ARG_NAME = 'input_activation_file_name'
NUM_LOW_ARG_NAME = 'num_low_activation_examples'
NUM_HIGH_ARG_NAME = 'num_high_activation_examples'
NUM_HITS_ARG_NAME = 'num_hits'
NUM_MISSES_ARG_NAME = 'num_misses'
NUM_FALSE_ALARMS_ARG_NAME = 'num_false_alarms'
NUM_CORRECT_NULLS_ARG_NAME = 'num_correct_nulls'
RADAR_IMAGE_DIR_ARG_NAME = 'input_radar_image_dir_name'
SOUNDING_DIR_ARG_NAME = 'input_sounding_dir_name'
TARGET_DIR_ARG_NAME = 'input_target_dir_name'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

ACTIVATION_FILE_HELP_STRING = (
    'Path to input file, containing activation of a single model component for '
    'each example.  This file will be read by `model_activation.read_file`.  If'
    ' the file contains activations for more than one model component, this '
    'script will error out.')
NUM_LOW_HELP_STRING = (
    'Number of low-activation examples (storm objects) to plot.  The examples '
    'with the `{0:s}` lowest activations will be plotted.'
).format(NUM_LOW_ARG_NAME)
NUM_HIGH_HELP_STRING = (
    'Number of high-activation examples (storm objects) to plot.  The examples '
    'with the `{0:s}` highest activations will be plotted.'
).format(NUM_HIGH_ARG_NAME)
NUM_HITS_HELP_STRING = (
    'Number of "hits" to plot.  Specifically, the `{0:s}` positive examples '
    '(storm objects with target class = 1) with the highest activations will be'
    ' plotted.'
).format(NUM_HITS_ARG_NAME)
NUM_MISSES_HELP_STRING = (
    'Number of "misses" to plot.  Specifically, the `{0:s}` positive examples '
    '(storm objects with target class = 1) with the lowest activations will be '
    'plotted.'
).format(NUM_MISSES_ARG_NAME)
NUM_FALSE_ALARMS_HELP_STRING = (
    'Number of false alarms to plot.  Specifically, the `{0:s}` negative '
    'examples (storm objects with target class = 0) with the highest '
    'activations will be plotted.'
).format(NUM_FALSE_ALARMS_ARG_NAME)
NUM_CORRECT_NULLS_HELP_STRING = (
    'Number of correct nulls to plot.  Specifically, the `{0:s}` negative '
    'examples (storm objects with target class = 0) with the lowest activations'
    ' will be plotted.'
).format(NUM_CORRECT_NULLS_ARG_NAME)
RADAR_IMAGE_DIR_HELP_STRING = (
    'Name of top-level directory with storm-centered radar images.  Files '
    'therein will be found by `storm_images.find_storm_image_file` and read by '
    '`storm_images.read_storm_images`.')
SOUNDING_DIR_HELP_STRING = (
    'Name of top-level directory with storm-centered soundings.  Files therein '
    'will be found by `soundings_only.find_sounding_file` and read by '
    '`soundings_only.read_soundings`.')
TARGET_DIR_HELP_STRING = (
    '[used only if {0:s} + {1:s} + {2:s} + {3:s} > 0] Name of top-level '
    'directory with target values (storm-hazard labels).  Files therein will be'
    ' found by `labels.find_label_file` and read by '
    '`labels.read_labels_from_netcdf`.'
).format(NUM_HITS_ARG_NAME, NUM_MISSES_ARG_NAME, NUM_FALSE_ALARMS_ARG_NAME,
         NUM_CORRECT_NULLS_ARG_NAME)
OUTPUT_DIR_HELP_STRING = (
    'Name of output directory.  Figures will be saved here.')

DEFAULT_TOP_RADAR_IMAGE_DIR_NAME = (
    '/condo/swatcommon/common/gridrad_final/myrorss_format/tracks/reanalyzed/'
    'storm_images_with_rdp')
DEFAULT_TOP_SOUNDING_DIR_NAME = (
    '/condo/swatcommon/common/gridrad_final/myrorss_format/tracks/reanalyzed/'
    'soundings')
DEFAULT_TOP_TARGET_DIR_NAME = (
    '/condo/swatcommon/common/gridrad_final/myrorss_format/tracks/reanalyzed/'
    'tornado_linkages/reanalyzed/labels')

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + ACTIVATION_FILE_ARG_NAME, type=str, required=True,
    help=ACTIVATION_FILE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + NUM_LOW_ARG_NAME, type=int, required=False, default=100,
    help=NUM_LOW_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + NUM_HIGH_ARG_NAME, type=int, required=False, default=100,
    help=NUM_HIGH_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + NUM_HITS_ARG_NAME, type=int, required=False, default=0,
    help=NUM_HITS_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + NUM_MISSES_ARG_NAME, type=int, required=False, default=0,
    help=NUM_MISSES_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + NUM_FALSE_ALARMS_ARG_NAME, type=int, required=False, default=0,
    help=NUM_FALSE_ALARMS_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + NUM_CORRECT_NULLS_ARG_NAME, type=int, required=False, default=0,
    help=NUM_CORRECT_NULLS_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + RADAR_IMAGE_DIR_ARG_NAME, type=str, required=False,
    default=DEFAULT_TOP_RADAR_IMAGE_DIR_NAME,
    help=RADAR_IMAGE_DIR_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + SOUNDING_DIR_ARG_NAME, type=str, required=False,
    default=DEFAULT_TOP_SOUNDING_DIR_NAME, help=SOUNDING_DIR_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + TARGET_DIR_ARG_NAME, type=str, required=False,
    default=DEFAULT_TOP_TARGET_DIR_NAME, help=TARGET_DIR_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING)


def _read_target_values(
        top_target_dir_name, storm_activations, activation_metadata_dict):
    """Reads target value for each storm object.

    E = number of examples (storm objects)

    :param top_target_dir_name: See documentation at top of file.
    :param storm_activations: length-E numpy array of activations.
    :param activation_metadata_dict: Dictionary returned by
        `model_activation.read_file`.
    :return: predictor_dict: Dictionary with the following keys.
    predictor_dict['storm_ids']: length-E list of storm IDs.
    predictor_dict['storm_times_unix_sec']: length-E numpy array of storm times.
    predictor_dict['storm_activations']: length-E numpy array of model
        activations.
    predictor_dict['storm_target_values']: length-E numpy array of target
        values.

    :raises: ValueError: if the target is multiclass and not binarized.
    """

    # Convert input args.
    storm_ids = activation_metadata_dict[model_activation.STORM_IDS_KEY]
    storm_times_unix_sec = activation_metadata_dict[
        model_activation.STORM_TIMES_KEY]

    storm_spc_date_strings_numpy = numpy.array(
        [time_conversion.time_to_spc_date_string(t)
         for t in storm_times_unix_sec],
        dtype=object)
    unique_spc_date_strings_numpy = numpy.unique(storm_spc_date_strings_numpy)

    # Read metadata for machine-learning model.
    model_file_name = activation_metadata_dict[
        model_activation.MODEL_FILE_NAME_KEY]
    model_metadata_file_name = '{0:s}/model_metadata.p'.format(
        os.path.split(model_file_name)[0])

    print 'Reading metadata from: "{0:s}"...'.format(model_metadata_file_name)
    model_metadata_dict = cnn.read_model_metadata(model_metadata_file_name)

    target_name = model_metadata_dict[cnn.TARGET_NAME_KEY]
    num_classes = labels.column_name_to_num_classes(target_name)
    binarize_target = (
        model_metadata_dict[cnn.BINARIZE_TARGET_KEY] and num_classes > 2)
    if num_classes > 2 and not binarize_target:
        error_string = (
            'The target variable ("{0:s}") is multiclass, which this script '
            'cannot handle.'
        ).format(target_name)
        raise ValueError(error_string)

    event_type_string = labels.column_name_to_label_params(
        target_name)[labels.EVENT_TYPE_KEY]

    # Read target values.
    storm_target_values = numpy.array([], dtype=int)
    sort_indices_for_storm_id = numpy.array([], dtype=int)
    num_spc_dates = len(unique_spc_date_strings_numpy)

    for i in range(num_spc_dates):
        this_target_file_name = labels.find_label_file(
            top_directory_name=top_target_dir_name,
            event_type_string=event_type_string, file_extension='.nc',
            spc_date_string=unique_spc_date_strings_numpy[i],
            raise_error_if_missing=True)

        print 'Reading data from: "{0:s}"...'.format(this_target_file_name)
        this_target_value_dict = labels.read_labels_from_netcdf(
            netcdf_file_name=this_target_file_name, label_name=target_name)

        these_indices = numpy.where(
            storm_spc_date_strings_numpy == unique_spc_date_strings_numpy[i])[0]
        sort_indices_for_storm_id = numpy.concatenate((
            sort_indices_for_storm_id, these_indices))

        these_indices = storm_images.find_storm_objects(
            all_storm_ids=this_target_value_dict[labels.STORM_IDS_KEY],
            all_valid_times_unix_sec=this_target_value_dict[
                labels.VALID_TIMES_KEY],
            storm_ids_to_keep=[storm_ids[k] for k in these_indices],
            valid_times_to_keep_unix_sec=storm_times_unix_sec[these_indices])
        storm_target_values = numpy.concatenate((
            storm_target_values,
            this_target_value_dict[labels.LABEL_VALUES_KEY][these_indices]))

    good_indices = numpy.where(
        storm_target_values != labels.INVALID_STORM_INTEGER)[0]
    storm_target_values = storm_target_values[good_indices]
    sort_indices_for_storm_id = sort_indices_for_storm_id[good_indices]

    if binarize_target:
        storm_target_values = (
            storm_target_values == num_classes - 1).astype(int)

    return {
        STORM_IDS_KEY: [storm_ids[k] for k in sort_indices_for_storm_id],
        STORM_TIMES_KEY: storm_times_unix_sec[sort_indices_for_storm_id],
        STORM_ACTIVATIONS_KEY: storm_activations[sort_indices_for_storm_id],
        TARGET_VALUES_KEY: storm_target_values
    }


def _read_predictors(
        top_radar_image_dir_name, top_sounding_dir_name,
        activation_metadata_dict, storm_ids, storm_times_unix_sec,
        storm_activations):
    """Reads radar and sounding data for each storm object.

    E = number of examples (storm objects)
    M = number of rows in each storm-centered grid
    N = number of columns in each storm-centered grid
    F = number of radar fields
    H = number of radar heights
    C = number of field/height pairs

    :param top_radar_image_dir_name: See documentation at top of file.
    :param top_sounding_dir_name: Same.
    :param activation_metadata_dict: Dictionary returned by
        `model_activation.read_file`.
    :param storm_ids: length-E list of storm IDs.
    :param storm_times_unix_sec: length-E numpy array of storm times.
    :param storm_activations: length-E numpy array of storm activations.
    :return: predictor_dict: Dictionary with the following keys.
    predictor_dict['radar_field_names']: length-F list with names of radar
        fields.  If radar images are 2-D, this is `None`.
    predictor_dict['radar_heights_m_asl']: length-H numpy array of radar heights
        (metres above sea level).  If radar images are 2-D, this is `None`.
    predictor_dict['field_name_by_pair']: length-C list with names of radar
        fields.  If radar images are 3-D, this is `None`.
    predictor_dict['height_by_pair_m_asl']: length-C numpy array of radar
        heights (metres above sea level).  If radar images are 3-D, this is
        `None`.
    predictor_dict['storm_ids']: length-E list of storm IDs.
    predictor_dict['storm_times_unix_sec']: length-E numpy array of storm times.
    predictor_dict['storm_activations']: length-E numpy array of model
        activations.
    predictor_dict['radar_image_matrix']: numpy array
        (either E x M x N x C or E x M x N x H x F) of radar values.
    predictor_dict['sounding_matrix']: 3-D numpy array of sounding values.
    """

    # Convert input args.
    storm_spc_date_strings_numpy = numpy.array(
        [time_conversion.time_to_spc_date_string(t)
         for t in storm_times_unix_sec],
        dtype=object)
    unique_spc_date_strings_numpy = numpy.unique(storm_spc_date_strings_numpy)

    # Read metadata for machine-learning model.
    model_file_name = activation_metadata_dict[
        model_activation.MODEL_FILE_NAME_KEY]
    model_metadata_file_name = '{0:s}/model_metadata.p'.format(
        os.path.split(model_file_name)[0])

    print 'Reading metadata from: "{0:s}"...'.format(model_metadata_file_name)
    model_metadata_dict = cnn.read_model_metadata(model_metadata_file_name)

    # Find files with storm-centered radar images.
    num_radar_dimensions = len(
        model_metadata_dict[cnn.TRAINING_FILE_NAMES_KEY].shape)
    num_spc_dates = len(unique_spc_date_strings_numpy)

    radar_image_matrix = None
    sounding_matrix = None
    radar_field_names = None
    radar_heights_m_asl = None
    field_name_by_pair = None
    height_by_pair_m_asl = None
    sort_indices_for_storm_id = numpy.array([], dtype=int)

    read_soundings = (
        model_metadata_dict[cnn.SOUNDING_FIELD_NAMES_KEY] is not None)

    for i in range(num_spc_dates):
        if num_radar_dimensions == 2:
            this_radar_file_name_matrix = trainval_io.find_radar_files_2d(
                top_directory_name=top_radar_image_dir_name,
                radar_source=model_metadata_dict[cnn.RADAR_SOURCE_KEY],
                radar_field_names=model_metadata_dict[
                    cnn.RADAR_FIELD_NAMES_KEY],
                first_file_time_unix_sec=
                time_conversion.spc_date_string_to_unix_sec(
                    unique_spc_date_strings_numpy[i]),
                last_file_time_unix_sec=
                time_conversion.spc_date_string_to_unix_sec(
                    unique_spc_date_strings_numpy[i]),
                one_file_per_time_step=False, shuffle_times=False,
                radar_heights_m_asl=model_metadata_dict[cnn.RADAR_HEIGHTS_KEY],
                reflectivity_heights_m_asl=model_metadata_dict[
                    cnn.REFLECTIVITY_HEIGHTS_KEY])[0]
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

            example_dict = deployment_io.create_storm_images_2d(
                radar_file_name_matrix=this_radar_file_name_matrix,
                num_examples_per_file=LARGE_INTEGER,
                normalization_type_string=None, return_target=False,
                target_name=model_metadata_dict[cnn.TARGET_NAME_KEY],
                num_rows_to_keep=model_metadata_dict[cnn.NUM_ROWS_TO_KEEP_KEY],
                num_columns_to_keep=model_metadata_dict[
                    cnn.NUM_COLUMNS_TO_KEEP_KEY],
                sounding_field_names=model_metadata_dict[
                    cnn.SOUNDING_FIELD_NAMES_KEY],
                top_sounding_dir_name=top_sounding_dir_name,
                sounding_lag_time_for_convective_contamination_sec=
                model_metadata_dict[cnn.SOUNDING_LAG_TIME_KEY])
        else:
            this_radar_file_name_matrix = trainval_io.find_radar_files_3d(
                top_directory_name=top_radar_image_dir_name,
                radar_source=model_metadata_dict[cnn.RADAR_SOURCE_KEY],
                radar_field_names=model_metadata_dict[
                    cnn.RADAR_FIELD_NAMES_KEY],
                radar_heights_m_asl=model_metadata_dict[cnn.RADAR_HEIGHTS_KEY],
                first_file_time_unix_sec=
                time_conversion.spc_date_string_to_unix_sec(
                    unique_spc_date_strings_numpy[i]),
                last_file_time_unix_sec=
                time_conversion.spc_date_string_to_unix_sec(
                    unique_spc_date_strings_numpy[i]),
                one_file_per_time_step=False, shuffle_times=False)[0]
            print MINOR_SEPARATOR_STRING

            if i == 0:
                radar_field_names = [
                    storm_images.image_file_name_to_field(f) for f in
                    this_radar_file_name_matrix[0, :, 0]
                ]
                radar_heights_m_asl = numpy.array([
                    storm_images.image_file_name_to_height(f) for f in
                    this_radar_file_name_matrix[0, 0, :]
                ], dtype=int)

            example_dict = deployment_io.create_storm_images_3d(
                radar_file_name_matrix=this_radar_file_name_matrix,
                num_examples_per_file=LARGE_INTEGER,
                normalization_type_string=None, return_target=False,
                target_name=model_metadata_dict[cnn.TARGET_NAME_KEY],
                num_rows_to_keep=model_metadata_dict[cnn.NUM_ROWS_TO_KEEP_KEY],
                num_columns_to_keep=model_metadata_dict[
                    cnn.NUM_COLUMNS_TO_KEEP_KEY],
                refl_masking_threshold_dbz=None,
                return_rotation_divergence_product=False,
                sounding_field_names=model_metadata_dict[
                    cnn.SOUNDING_FIELD_NAMES_KEY],
                top_sounding_dir_name=top_sounding_dir_name,
                sounding_lag_time_for_convective_contamination_sec=
                model_metadata_dict[cnn.SOUNDING_LAG_TIME_KEY])

        these_indices = numpy.where(
            storm_spc_date_strings_numpy == unique_spc_date_strings_numpy[i])[0]
        sort_indices_for_storm_id = numpy.concatenate((
            sort_indices_for_storm_id, these_indices))

        these_indices = storm_images.find_storm_objects(
            all_storm_ids=example_dict[deployment_io.STORM_IDS_KEY],
            all_valid_times_unix_sec=example_dict[
                deployment_io.STORM_TIMES_KEY],
            storm_ids_to_keep=[storm_ids[k] for k in these_indices],
            valid_times_to_keep_unix_sec=storm_times_unix_sec[these_indices])

        if radar_image_matrix is None:
            radar_image_matrix = example_dict[
                deployment_io.RADAR_IMAGE_MATRIX_KEY][these_indices, ...] + 0.
            if read_soundings:
                sounding_matrix = example_dict[
                    deployment_io.SOUNDING_MATRIX_KEY][these_indices, ...] + 0.
        else:
            radar_image_matrix = numpy.concatenate(
                (radar_image_matrix,
                 example_dict[deployment_io.RADAR_IMAGE_MATRIX_KEY][
                     these_indices, ...]),
                axis=0)
            if read_soundings:
                sounding_matrix = numpy.concatenate(
                    (sounding_matrix,
                     example_dict[deployment_io.SOUNDING_MATRIX_KEY][
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
        STORM_ACTIVATIONS_KEY: storm_activations[sort_indices_for_storm_id],
        RADAR_IMAGE_MATRIX_KEY: radar_image_matrix,
        SOUNDING_FIELD_NAMES_KEY: model_metadata_dict[
            cnn.SOUNDING_FIELD_NAMES_KEY],
        SOUNDING_MATRIX_KEY: sounding_matrix
    }


def _plot_storm_objects(predictor_dict, output_dir_name):
    """Plots radar and sounding data for each storm object.

    :param predictor_dict: Dictionary created by `_read_predictors`.
    :param output_dir_name: Name of output directory.  Figures will be saved
        here.
    """

    field_name_by_pair = predictor_dict[FIELD_NAMES_2D_KEY]
    height_by_pair_m_asl = predictor_dict[HEIGHTS_2D_KEY]
    radar_field_names = predictor_dict[FIELD_NAMES_3D_KEY]
    radar_heights_m_asl = predictor_dict[HEIGHTS_3D_KEY]
    storm_ids = predictor_dict[STORM_IDS_KEY]
    storm_times_unix_sec = predictor_dict[STORM_TIMES_KEY]
    storm_activations = predictor_dict[STORM_ACTIVATIONS_KEY]
    radar_image_matrix = predictor_dict[RADAR_IMAGE_MATRIX_KEY]
    sounding_field_names = predictor_dict[SOUNDING_FIELD_NAMES_KEY]
    sounding_matrix = predictor_dict[SOUNDING_MATRIX_KEY]

    plot_soundings = sounding_matrix is not None
    if plot_soundings:
        list_of_metpy_dictionaries = dl_utils.soundings_to_metpy_dictionaries(
            sounding_matrix=sounding_matrix,
            pressure_levels_mb=SOUNDING_PRESSURE_LEVELS_MB,
            pressureless_field_names=sounding_field_names)

    num_radar_dimensions = 2 + int(radar_field_names is not None)
    num_storm_objects = len(storm_ids)

    for i in range(num_storm_objects):
        this_time_string = time_conversion.unix_sec_to_string(
            storm_times_unix_sec[i], TIME_FORMAT)
        this_base_title_string = (
            'Storm "{0:s}" at {1:s} (activation = {2:.3f})'
        ).format(storm_ids[i], this_time_string, storm_activations[i])
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
                    height_by_pair_m_asl=height_by_pair_m_asl,
                    num_panel_rows=NUM_PANEL_ROWS)

                this_title_string = this_base_title_string + ''
                this_figure_file_name = '{0:s}.jpg'.format(this_base_file_name)
            else:
                (_, these_axes_objects_2d_list
                ) = radar_plotting.plot_3d_grid_without_coords(
                    field_matrix=numpy.flip(
                        radar_image_matrix[i, ..., j], axis=0),
                    field_name=radar_field_names[j],
                    grid_point_heights_m_asl=radar_heights_m_asl,
                    num_panel_rows=NUM_PANEL_ROWS)

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
        input_activation_file_name, num_low_activation_examples,
        num_high_activation_examples, num_hits, num_misses, num_false_alarms,
        num_correct_nulls, top_radar_image_dir_name, top_sounding_dir_name,
        top_target_dir_name, output_dir_name):
    """Plots examples (storm objects) with extremely low/high CNN activations.

    This is effectively the main method.

    :param input_activation_file_name: See documentation at top of file.
    :param num_low_activation_examples: Same.
    :param num_high_activation_examples: Same.
    :param num_hits: Same.
    :param num_misses: Same.
    :param num_false_alarms: Same.
    :param num_correct_nulls: Same.
    :param top_radar_image_dir_name: Same.
    :param top_sounding_dir_name: Same.
    :param top_target_dir_name: Same.
    :param output_dir_name: Same.
    :raises: ValueError: if the activation file contains activations for more
        than one model component.
    """

    # Check input args.
    example_counts = numpy.array(
        [num_low_activation_examples, num_high_activation_examples, num_hits,
         num_misses, num_false_alarms, num_correct_nulls], dtype=int)
    error_checking.assert_is_geq_numpy_array(example_counts, 0)
    error_checking.assert_is_greater(numpy.sum(example_counts), 0)

    # Read activations.
    print 'Reading activations from: "{0:s}"...'.format(
        input_activation_file_name)
    activation_matrix, activation_metadata_dict = model_activation.read_file(
        input_activation_file_name)

    num_model_components = activation_matrix.shape[1]
    if num_model_components > 1:
        error_string = (
            'The file should contain activations for only one model component, '
            'not {0:d}.'
        ).format(num_model_components)
        raise ValueError(error_string)

    storm_activations = activation_matrix[:, 0]
    storm_ids = activation_metadata_dict[model_activation.STORM_IDS_KEY]
    storm_times_unix_sec = activation_metadata_dict[
        model_activation.STORM_TIMES_KEY]

    num_storm_objects = len(storm_ids)
    error_checking.assert_is_leq(numpy.sum(example_counts), num_storm_objects)

    # Plot high-activation examples.
    high_indices, low_indices = model_activation.get_hilo_activation_examples(
        storm_activations=storm_activations,
        num_low_activation_examples=num_low_activation_examples,
        num_high_activation_examples=num_high_activation_examples)

    if len(high_indices) > 0:
        print SEPARATOR_STRING
        this_predictor_dict = _read_predictors(
            top_radar_image_dir_name=top_radar_image_dir_name,
            top_sounding_dir_name=top_sounding_dir_name,
            activation_metadata_dict=activation_metadata_dict,
            storm_ids=[storm_ids[k] for k in high_indices],
            storm_times_unix_sec=storm_times_unix_sec[high_indices],
            storm_activations=storm_activations[high_indices])
        print SEPARATOR_STRING

        this_directory_name = '{0:s}/high_activations'.format(output_dir_name)
        file_system_utils.mkdir_recursive_if_necessary(
            directory_name=this_directory_name)
        _plot_storm_objects(
            predictor_dict=this_predictor_dict,
            output_dir_name=this_directory_name)

    # Plot low-activation examples.
    if len(low_indices) > 0:
        print SEPARATOR_STRING
        this_predictor_dict = _read_predictors(
            top_radar_image_dir_name=top_radar_image_dir_name,
            top_sounding_dir_name=top_sounding_dir_name,
            activation_metadata_dict=activation_metadata_dict,
            storm_ids=[storm_ids[k] for k in low_indices],
            storm_times_unix_sec=storm_times_unix_sec[low_indices],
            storm_activations=storm_activations[low_indices])
        print SEPARATOR_STRING

        this_directory_name = '{0:s}/low_activations'.format(output_dir_name)
        file_system_utils.mkdir_recursive_if_necessary(
            directory_name=this_directory_name)
        _plot_storm_objects(
            predictor_dict=this_predictor_dict,
            output_dir_name=this_directory_name)

    if num_hits + num_misses + num_false_alarms + num_correct_nulls == 0:
        return

    # Find contingency-table extremes.
    print SEPARATOR_STRING
    target_value_dict = _read_target_values(
        top_target_dir_name=top_target_dir_name,
        storm_activations=storm_activations,
        activation_metadata_dict=activation_metadata_dict)

    storm_ids = target_value_dict[STORM_IDS_KEY]
    storm_times_unix_sec = target_value_dict[STORM_TIMES_KEY]
    storm_activations = target_value_dict[STORM_ACTIVATIONS_KEY]
    storm_target_values = target_value_dict[TARGET_VALUES_KEY]

    ct_extreme_dict = model_activation.get_contingency_table_extremes(
        storm_activations=storm_activations,
        storm_target_values=storm_target_values, num_hits=num_hits,
        num_misses=num_misses, num_false_alarms=num_false_alarms,
        num_correct_nulls=num_correct_nulls)

    hit_indices = ct_extreme_dict[model_activation.HIT_INDICES_KEY]
    miss_indices = ct_extreme_dict[model_activation.MISS_INDICES_KEY]
    false_alarm_indices = ct_extreme_dict[
        model_activation.FALSE_ALARM_INDICES_KEY]
    correct_null_indices = ct_extreme_dict[
        model_activation.CORRECT_NULL_INDICES_KEY]

    # Plot best hits (true positives).
    if len(hit_indices) > 0:
        print SEPARATOR_STRING
        this_predictor_dict = _read_predictors(
            top_radar_image_dir_name=top_radar_image_dir_name,
            top_sounding_dir_name=top_sounding_dir_name,
            activation_metadata_dict=activation_metadata_dict,
            storm_ids=[storm_ids[k] for k in hit_indices],
            storm_times_unix_sec=storm_times_unix_sec[hit_indices],
            storm_activations=storm_activations[hit_indices])
        print SEPARATOR_STRING

        this_directory_name = '{0:s}/hits'.format(output_dir_name)
        file_system_utils.mkdir_recursive_if_necessary(
            directory_name=this_directory_name)
        _plot_storm_objects(
            predictor_dict=this_predictor_dict,
            output_dir_name=this_directory_name)

    # Plot worst misses (false negatives).
    if len(miss_indices) > 0:
        print SEPARATOR_STRING
        this_predictor_dict = _read_predictors(
            top_radar_image_dir_name=top_radar_image_dir_name,
            top_sounding_dir_name=top_sounding_dir_name,
            activation_metadata_dict=activation_metadata_dict,
            storm_ids=[storm_ids[k] for k in miss_indices],
            storm_times_unix_sec=storm_times_unix_sec[miss_indices],
            storm_activations=storm_activations[miss_indices])
        print SEPARATOR_STRING

        this_directory_name = '{0:s}/misses'.format(output_dir_name)
        file_system_utils.mkdir_recursive_if_necessary(
            directory_name=this_directory_name)
        _plot_storm_objects(
            predictor_dict=this_predictor_dict,
            output_dir_name=this_directory_name)

    # Plot worst false alarms (false positives).
    if len(false_alarm_indices) > 0:
        print SEPARATOR_STRING
        this_predictor_dict = _read_predictors(
            top_radar_image_dir_name=top_radar_image_dir_name,
            top_sounding_dir_name=top_sounding_dir_name,
            activation_metadata_dict=activation_metadata_dict,
            storm_ids=[storm_ids[k] for k in false_alarm_indices],
            storm_times_unix_sec=storm_times_unix_sec[false_alarm_indices],
            storm_activations=storm_activations[false_alarm_indices])
        print SEPARATOR_STRING

        this_directory_name = '{0:s}/false_alarms'.format(output_dir_name)
        file_system_utils.mkdir_recursive_if_necessary(
            directory_name=this_directory_name)
        _plot_storm_objects(
            predictor_dict=this_predictor_dict,
            output_dir_name=this_directory_name)

    # Plot best correct nulls (true negatives).
    if len(correct_null_indices) > 0:
        print SEPARATOR_STRING
        this_predictor_dict = _read_predictors(
            top_radar_image_dir_name=top_radar_image_dir_name,
            top_sounding_dir_name=top_sounding_dir_name,
            activation_metadata_dict=activation_metadata_dict,
            storm_ids=[storm_ids[k] for k in correct_null_indices],
            storm_times_unix_sec=storm_times_unix_sec[correct_null_indices],
            storm_activations=storm_activations[correct_null_indices])
        print SEPARATOR_STRING

        this_directory_name = '{0:s}/correct_nulls'.format(output_dir_name)
        file_system_utils.mkdir_recursive_if_necessary(
            directory_name=this_directory_name)
        _plot_storm_objects(
            predictor_dict=this_predictor_dict,
            output_dir_name=this_directory_name)


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        input_activation_file_name=getattr(
            INPUT_ARG_OBJECT, ACTIVATION_FILE_ARG_NAME),
        num_low_activation_examples=getattr(INPUT_ARG_OBJECT, NUM_LOW_ARG_NAME),
        num_high_activation_examples=getattr(
            INPUT_ARG_OBJECT, NUM_HIGH_ARG_NAME),
        num_hits=getattr(INPUT_ARG_OBJECT, NUM_HITS_ARG_NAME),
        num_misses=getattr(INPUT_ARG_OBJECT, NUM_MISSES_ARG_NAME),
        num_false_alarms=getattr(INPUT_ARG_OBJECT, NUM_FALSE_ALARMS_ARG_NAME),
        num_correct_nulls=getattr(INPUT_ARG_OBJECT, NUM_CORRECT_NULLS_ARG_NAME),
        top_radar_image_dir_name=getattr(
            INPUT_ARG_OBJECT, RADAR_IMAGE_DIR_ARG_NAME),
        top_sounding_dir_name=getattr(INPUT_ARG_OBJECT, SOUNDING_DIR_ARG_NAME),
        top_target_dir_name=getattr(INPUT_ARG_OBJECT, TARGET_DIR_ARG_NAME),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME))
