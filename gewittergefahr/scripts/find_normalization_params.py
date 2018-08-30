"""Finds normalization parameters for either GridRad or MYRORSS data.

"GridRad data" or "MYRORSS data," in this case, includes both storm-centered
radar images and storm-centered soundings.
"""

import copy
import argparse
import numpy
import pandas
from scipy.interpolate import interp1d
from gewittergefahr.gg_utils import radar_utils
from gewittergefahr.gg_utils import soundings
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import number_rounding
from gewittergefahr.gg_utils import error_checking
from gewittergefahr.deep_learning import storm_images
from gewittergefahr.deep_learning import deep_learning_utils as dl_utils
from gewittergefahr.deep_learning import training_validation_io as trainval_io

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'
MINOR_SEPARATOR_STRING = '\n\n' + '-' * 50 + '\n\n'

NUM_VALUES_KEY = 'num_values'
MEAN_VALUE_KEY = 'mean_value'
MEAN_OF_SQUARES_KEY = 'mean_of_squares'

RADAR_INTERVAL_DICT = {
    radar_utils.ECHO_TOP_18DBZ_NAME: 0.01,  # km
    radar_utils.ECHO_TOP_40DBZ_NAME: 0.01,  # km
    radar_utils.ECHO_TOP_50DBZ_NAME: 0.01,  # km
    radar_utils.LOW_LEVEL_SHEAR_NAME: 1e-5,  # s^-1
    radar_utils.MID_LEVEL_SHEAR_NAME: 1e-5,  # s^-1
    radar_utils.MESH_NAME: 0.1,  # mm
    radar_utils.REFL_NAME: 0.1,  # dBZ
    radar_utils.REFL_COLUMN_MAX_NAME: 0.1,  # dBZ
    radar_utils.REFL_0CELSIUS_NAME: 0.1,  # dBZ
    radar_utils.REFL_M10CELSIUS_NAME: 0.1,  # dBZ
    radar_utils.REFL_M20CELSIUS_NAME: 0.1,  # dBZ
    radar_utils.REFL_LOWEST_ALTITUDE_NAME: 0.1,  # dBZ
    radar_utils.SHI_NAME: 0.1,  # unitless
    radar_utils.VIL_NAME: 0.1,  # mm
    radar_utils.DIFFERENTIAL_REFL_NAME: 1e-3,  # dB
    radar_utils.SPEC_DIFF_PHASE_NAME: 1e-3,  # deg km^-1
    radar_utils.CORRELATION_COEFF_NAME: 1e-3,  # unitless
    radar_utils.SPECTRUM_WIDTH_NAME: 0.01,  # m s^-1
    radar_utils.VORTICITY_NAME: 1e-5,  # s^-1
    radar_utils.DIVERGENCE_NAME: 1e-5  # s^-1
}

SOUNDING_INTERVAL_DICT = {
    soundings.PRESSURE_NAME: 1.,  # Pascals
    soundings.TEMPERATURE_NAME: 0.01,  # Kelvins
    soundings.VIRTUAL_POTENTIAL_TEMPERATURE_NAME: 0.01,  # Kelvins
    soundings.U_WIND_NAME: 0.01,  # m s^-1
    soundings.V_WIND_NAME: 0.01,  # m s^-1
    soundings.SPECIFIC_HUMIDITY_NAME: 1e-6,  # kg kg^-1
    soundings.RELATIVE_HUMIDITY_NAME: 1e-4  # unitless
}

RADAR_HEIGHTS_M_AGL = numpy.linspace(1000, 12000, num=12, dtype=int)
REFLECTIVITY_HEIGHTS_M_AGL = RADAR_HEIGHTS_M_AGL + 0
SOUNDING_HEIGHTS_M_AGL = soundings.DEFAULT_HEIGHT_LEVELS_M_AGL + 0

SOUNDING_FIELD_NAMES = [
    soundings.PRESSURE_NAME, soundings.TEMPERATURE_NAME,
    soundings.VIRTUAL_POTENTIAL_TEMPERATURE_NAME,
    soundings.U_WIND_NAME, soundings.V_WIND_NAME,
    soundings.SPECIFIC_HUMIDITY_NAME, soundings.RELATIVE_HUMIDITY_NAME
]

DUMMY_TARGET_NAME = 'tornado_lead-time=0000-3600sec_distance=00000-10000m'
LAG_TIME_FOR_CONVECTIVE_CONTAMINATION_SEC = 1800

RADAR_IMAGE_DIR_ARG_NAME = 'input_radar_image_dir_name'
RADAR_SOURCE_ARG_NAME = 'radar_source'
MIN_PERCENTILE_ARG_NAME = 'min_percentile_level'
MAX_PERCENTILE_ARG_NAME = 'max_percentile_level'
SOUNDING_DIR_ARG_NAME = 'input_sounding_dir_name'
FIRST_SPC_DATE_ARG_NAME = 'first_spc_date_string'
LAST_SPC_DATE_ARG_NAME = 'last_spc_date_string'
RADAR_FIELD_NAMES_ARG_NAME = 'radar_field_names'
OUTPUT_FILE_ARG_NAME = 'output_file_name'

DEFAULT_MIN_PERCENTILE_LEVEL = 0.1
DEFAULT_MAX_PERCENTILE_LEVEL = 99.9
DEFAULT_TOP_GRIDRAD_RADAR_DIR_NAME = (
    '/condo/swatcommon/common/gridrad_final/myrorss_format/tracks/reanalyzed/'
    'storm_images_rotated')
DEFAULT_TOP_GRIDRAD_SOUNDING_DIR_NAME = (
    '/condo/swatcommon/common/gridrad_final/myrorss_format/tracks/reanalyzed/'
    'soundings')
DEFAULT_TOP_MYRORSS_RADAR_DIR_NAME = (
    '/condo/swatcommon/common/myrorss_40dbz_echo_tops/final_tracks/reanalyzed/'
    'storm_images')
DEFAULT_TOP_MYRORSS_SOUNDING_DIR_NAME = (
    '/condo/swatwork/ralager/myrorss_40dbz_echo_tops/final_tracks/reanalyzed/'
    'soundings')

DEFAULT_GRIDRAD_FIELD_NAMES = [
    radar_utils.REFL_NAME, radar_utils.SPECTRUM_WIDTH_NAME,
    radar_utils.DIVERGENCE_NAME, radar_utils.VORTICITY_NAME,
    radar_utils.DIFFERENTIAL_REFL_NAME, radar_utils.CORRELATION_COEFF_NAME,
    radar_utils.SPEC_DIFF_PHASE_NAME
]

DEFAULT_MYRORSS_FIELD_NAMES = [
    radar_utils.ECHO_TOP_18DBZ_NAME, radar_utils.ECHO_TOP_50DBZ_NAME,
    radar_utils.LOW_LEVEL_SHEAR_NAME, radar_utils.MID_LEVEL_SHEAR_NAME,
    radar_utils.REFL_NAME, radar_utils.REFL_COLUMN_MAX_NAME,
    radar_utils.REFL_0CELSIUS_NAME, radar_utils.REFL_M10CELSIUS_NAME,
    radar_utils.REFL_M20CELSIUS_NAME, radar_utils.REFL_LOWEST_ALTITUDE_NAME,
    radar_utils.MESH_NAME, radar_utils.SHI_NAME, radar_utils.VIL_NAME
]

RADAR_IMAGE_DIR_HELP_STRING = (
    'Name of top-level directory with storm-centered radar images.  Files '
    'therein will be found by `storm_images.find_storm_image_file` and read by '
    '`storm_images.read_storm_images`.  Default is "{0:s}" for MYRORSS data, '
    '"{1:s}" for GridRad.'
).format(DEFAULT_TOP_MYRORSS_RADAR_DIR_NAME, DEFAULT_TOP_GRIDRAD_RADAR_DIR_NAME)

RADAR_SOURCE_HELP_STRING = (
    'Source of radar data.  Must be in the following list.\n{0:s}'
).format(str(radar_utils.DATA_SOURCE_IDS))

MIN_PERCENTILE_HELP_STRING = (
    'Minimum percentile level.  The "minimum value" for each field will '
    'actually be the [q]th percentile, where q = `{0:s}`.'
).format(MIN_PERCENTILE_ARG_NAME)

MAX_PERCENTILE_HELP_STRING = (
    'Max percentile level.  The "max value" for each field will actually be the'
    ' [q]th percentile, where q = `{0:s}`.'
).format(MAX_PERCENTILE_ARG_NAME)

SOUNDING_DIR_HELP_STRING = (
    'Name of top-level directory with storm-centered soundings.  Files therein '
    'will be found by `soundings.find_sounding_file` and read by '
    '`soundings.read_soundings`.  Default is "{0:s}" for MYRORSS data, '
    '"{1:s}" for GridRad.'
).format(DEFAULT_TOP_MYRORSS_SOUNDING_DIR_NAME,
         DEFAULT_TOP_GRIDRAD_SOUNDING_DIR_NAME)

SPC_DATE_HELP_STRING = (
    'SPC (Storm Prediction Center) date in format "yyyymmdd".  Normalization '
    'params will be based on all data from `{0:s}`...`{1:s}`.'
).format(FIRST_SPC_DATE_ARG_NAME, LAST_SPC_DATE_ARG_NAME)

RADAR_FIELD_NAMES_HELP_STRING = (
    'List of radar fields (each must be accepted by `radar_utils.'
    'check_field_name`).  Normalization params will be computed for each of '
    'these fields, once over all heights and once at each height (metres above '
    'ground level) in the following list.\n{0:s}\nDefault fields for MYRORSS:'
    '\n{1:s}\nDefault fields for GridRad:\n{2:s}'
).format(str(RADAR_HEIGHTS_M_AGL), str(DEFAULT_MYRORSS_FIELD_NAMES),
         str(DEFAULT_GRIDRAD_FIELD_NAMES))

OUTPUT_FILE_HELP_STRING = (
    'Path to output file (will be written by `deep_learning_utils.'
    'write_normalization_params_to_file`).')

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + RADAR_IMAGE_DIR_ARG_NAME, type=str, required=False, default='',
    help=RADAR_IMAGE_DIR_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + RADAR_SOURCE_ARG_NAME, type=str, required=True,
    help=RADAR_SOURCE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + MIN_PERCENTILE_ARG_NAME, type=float, required=False,
    default=DEFAULT_MIN_PERCENTILE_LEVEL, help=MIN_PERCENTILE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + MAX_PERCENTILE_ARG_NAME, type=float, required=False,
    default=DEFAULT_MAX_PERCENTILE_LEVEL, help=MAX_PERCENTILE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + SOUNDING_DIR_ARG_NAME, type=str, required=False, default='',
    help=SOUNDING_DIR_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + FIRST_SPC_DATE_ARG_NAME, type=str, required=True,
    help=SPC_DATE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + LAST_SPC_DATE_ARG_NAME, type=str, required=True,
    help=SPC_DATE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + RADAR_FIELD_NAMES_ARG_NAME, type=str, nargs='+', required=False,
    default=[''], help=RADAR_FIELD_NAMES_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_FILE_ARG_NAME, type=str, required=True,
    help=OUTPUT_FILE_HELP_STRING)


def _update_z_score_params(z_score_param_dict, new_data_matrix):
    """Updates z-score parameters.

    :param z_score_param_dict: Dictionary with the following keys.
    z_score_param_dict['num_values']: Number of values on which current
        estimates are based.
    z_score_param_dict['mean_value']: Current mean.
    z_score_param_dict['mean_of_squares']: Current mean of squared values.

    :param new_data_matrix: numpy array with new values.  Will be used to
        update estimates in `z_score_param_dict`.
    :return: z_score_param_dict: Same as input, but with new estimates.
    """

    mean_values = numpy.array(
        [z_score_param_dict[MEAN_VALUE_KEY], numpy.mean(new_data_matrix)])
    weights = numpy.array(
        [z_score_param_dict[NUM_VALUES_KEY], new_data_matrix.size])
    z_score_param_dict[MEAN_VALUE_KEY] = numpy.average(
        mean_values, weights=weights)

    mean_values = numpy.array([
        z_score_param_dict[MEAN_OF_SQUARES_KEY],
        numpy.mean(new_data_matrix ** 2)])
    weights = numpy.array(
        [z_score_param_dict[NUM_VALUES_KEY], new_data_matrix.size])
    z_score_param_dict[MEAN_OF_SQUARES_KEY] = numpy.average(
        mean_values, weights=weights)

    z_score_param_dict[NUM_VALUES_KEY] += new_data_matrix.size
    return z_score_param_dict


def _update_frequency_dict(frequency_dict, new_data_matrix, rounding_base):
    """Updates measurement frequencies.

    :param frequency_dict: Dictionary, where each key is a measurement and the
        corresponding value is the number of times said measurement occurs.
    :param new_data_matrix: numpy array with new values.  Will be used to
        update frequencies in `frequency_dict`.
    :param rounding_base: Each value in `new_data_matrix` will be rounded to the
        nearest multiple of this base.
    :return: frequency_dict: Same as input, but with new frequencies.
    """

    new_unique_values, new_counts = numpy.unique(
        number_rounding.round_to_nearest(new_data_matrix, rounding_base),
        return_counts=True)

    for i in range(len(new_unique_values)):
        if new_unique_values[i] in frequency_dict:
            frequency_dict[new_unique_values[i]] += new_counts[i]
        else:
            frequency_dict[new_unique_values[i]] = new_counts[i]

    return frequency_dict


def _get_standard_deviation(z_score_param_dict):
    """Computes standard deviation.

    :param z_score_param_dict: See doc for `_update_z_score_params`.
    :return: standard_deviation: Standard deviation.
    """

    multiplier = float(
        z_score_param_dict[NUM_VALUES_KEY]
    ) / (z_score_param_dict[NUM_VALUES_KEY] - 1)

    return numpy.sqrt(multiplier * (
        z_score_param_dict[MEAN_OF_SQUARES_KEY] -
        z_score_param_dict[MEAN_VALUE_KEY] ** 2))


def _get_percentile(frequency_dict, percentile_level):
    """Computes percentile.

    :param frequency_dict: See doc for `_update_frequency_dict`.
    :param percentile_level: Percentile level.  Will take the [q]th percentile,
        where q = `percentile_level`.
    :return: percentile: [q]th percentile.
    """

    unique_values, counts = zip(*frequency_dict.iteritems())
    unique_values = numpy.array(unique_values)
    counts = numpy.array(counts, dtype=int)

    sort_indices = numpy.argsort(unique_values)
    unique_values = unique_values[sort_indices]
    counts = counts[sort_indices]

    cumulative_frequencies = (
        numpy.cumsum(counts).astype(float) / numpy.sum(counts))
    percentile_levels = 100 * (
        (cumulative_frequencies * numpy.sum(counts) - 1) /
        (numpy.sum(counts) - 1)
    )

    if percentile_level > percentile_levels[-1]:
        return unique_values[-1]
    if percentile_level < percentile_levels[0]:
        return unique_values[0]

    interp_object = interp1d(
        x=percentile_levels, y=unique_values, kind='linear', bounds_error=True,
        assume_sorted=True)
    return interp_object(percentile_level)


def _convert_normalization_params(
        z_score_dict_dict, frequency_dict_dict=None,
        min_percentile_level=None, max_percentile_level=None):
    """Converts normalization params from nested dicts to pandas DataFrame.

    :param z_score_dict_dict: Dictionary of dictionaries, where each inner
        dictionary follows the input format for `_update_z_score_params`.
    :param frequency_dict_dict: Dictionary of dictionaries, where each inner
        dictionary follows the input format for `_update_frequency_dict`.  Must
        have the same outer keys as `z_score_dict_dict`.
    :param min_percentile_level: [used iff `frequency_dict_dict is not None`]
        Minimum percentile level (used to create "min_value" column in output
        table).
    :param max_percentile_level: [used iff `frequency_dict_dict is not None`]
        Max percentile level (used to create "max_value" column in output
        table).
    :return: normalization_table: pandas DataFrame, where the indices are outer
        keys in `z_score_dict_dict`.  For example, if `z_score_dict_dict`
        contains 80 inner dictionaries, this table will have 80 rows.  Columns
        are as follows.
    normalization_table.mean_value: Mean value.
    normalization_table.standard_deviation: Standard deviation.

    If `frequency_dict_dict is not None`, will also contain the following.

    normalization_table.min_value: Minimum value.
    normalization_table.max_value: Max value.
    """

    normalization_dict = {}

    for this_key in z_score_dict_dict:
        this_inner_dict = z_score_dict_dict[this_key]
        this_standard_deviation = _get_standard_deviation(this_inner_dict)
        normalization_dict[this_key] = [
            this_inner_dict[MEAN_VALUE_KEY], this_standard_deviation]

        if frequency_dict_dict is not None:
            this_inner_dict = frequency_dict_dict[this_key]
            this_min_value = _get_percentile(
                frequency_dict=this_inner_dict,
                percentile_level=min_percentile_level)
            this_max_value = _get_percentile(
                frequency_dict=this_inner_dict,
                percentile_level=max_percentile_level)

            normalization_dict[this_key].append(this_min_value)
            normalization_dict[this_key].append(this_max_value)

        normalization_dict[this_key] = numpy.array(normalization_dict[this_key])

    normalization_table = pandas.DataFrame.from_dict(
        normalization_dict, orient='index')

    column_dict_old_to_new = {
        0: dl_utils.MEAN_VALUE_COLUMN,
        1: dl_utils.STANDARD_DEVIATION_COLUMN
    }
    if frequency_dict_dict is not None:
        column_dict_old_to_new.update({
            2: dl_utils.MIN_VALUE_COLUMN,
            3: dl_utils.MAX_VALUE_COLUMN
        })

    return normalization_table.rename(
        columns=column_dict_old_to_new, inplace=False)


def _run(
        top_radar_image_dir_name, radar_source, min_percentile_level,
        max_percentile_level, top_sounding_dir_name, first_spc_date_string,
        last_spc_date_string, radar_field_names, output_file_name):
    """Finds normalization parameters for GridRad data.

    This is effectively the main method.

    :param top_radar_image_dir_name: See documentation at top of file.
    :param radar_source: Same.
    :param min_percentile_level: Same.
    :param max_percentile_level: Same.
    :param top_sounding_dir_name: Same.
    :param first_spc_date_string: Same.
    :param last_spc_date_string: Same.
    :param radar_field_names: Same.
    :param output_file_name: Same.
    """

    error_checking.assert_is_greater(max_percentile_level, min_percentile_level)

    # Find radar files.
    if radar_source == radar_utils.MYRORSS_SOURCE_ID:
        if radar_field_names == ['']:
            radar_field_names = DEFAULT_MYRORSS_FIELD_NAMES + []
        if top_radar_image_dir_name == '':
            top_radar_image_dir_name = DEFAULT_TOP_MYRORSS_RADAR_DIR_NAME + ''
        if top_sounding_dir_name == '':
            top_sounding_dir_name = DEFAULT_TOP_MYRORSS_SOUNDING_DIR_NAME + ''

        radar_file_name_matrix = trainval_io.find_radar_files_2d(
            top_directory_name=top_radar_image_dir_name,
            radar_source=radar_source, radar_field_names=radar_field_names,
            reflectivity_heights_m_agl=REFLECTIVITY_HEIGHTS_M_AGL,
            first_file_time_unix_sec=
            time_conversion.spc_date_string_to_unix_sec(first_spc_date_string),
            last_file_time_unix_sec=
            time_conversion.spc_date_string_to_unix_sec(last_spc_date_string),
            one_file_per_time_step=False, shuffle_times=False)[0]

    else:
        if radar_field_names == ['']:
            radar_field_names = DEFAULT_GRIDRAD_FIELD_NAMES + []
        if top_radar_image_dir_name == '':
            top_radar_image_dir_name = DEFAULT_TOP_GRIDRAD_RADAR_DIR_NAME + ''
        if top_sounding_dir_name == '':
            top_sounding_dir_name = DEFAULT_TOP_GRIDRAD_SOUNDING_DIR_NAME + ''

        radar_file_name_matrix = trainval_io.find_radar_files_2d(
            top_directory_name=top_radar_image_dir_name,
            radar_source=radar_source, radar_field_names=radar_field_names,
            radar_heights_m_agl=RADAR_HEIGHTS_M_AGL,
            first_file_time_unix_sec=
            time_conversion.spc_date_string_to_unix_sec(first_spc_date_string),
            last_file_time_unix_sec=
            time_conversion.spc_date_string_to_unix_sec(last_spc_date_string),
            one_file_per_time_step=False, shuffle_times=False)[0]

    print SEPARATOR_STRING

    field_name_by_pair = [
        storm_images.image_file_name_to_field(f) for f in
        radar_file_name_matrix[0, :]
    ]
    height_by_pair_m_agl = numpy.array(
        [storm_images.image_file_name_to_height(f)
         for f in radar_file_name_matrix[0, :]
        ], dtype=int)

    # Find sounding files.
    sounding_file_names = trainval_io.find_sounding_files(
        top_sounding_dir_name=top_sounding_dir_name,
        radar_file_name_matrix=radar_file_name_matrix,
        target_name=DUMMY_TARGET_NAME,
        lag_time_for_convective_contamination_sec=
        LAG_TIME_FOR_CONVECTIVE_CONTAMINATION_SEC)
    print SEPARATOR_STRING

    # Initialize normalization params.
    orig_parameter_dict = {
        NUM_VALUES_KEY: 0, MEAN_VALUE_KEY: 0., MEAN_OF_SQUARES_KEY: 0.
    }

    radar_z_score_dict_no_height = {}
    radar_z_score_dict_with_height = {}
    radar_freq_dict_no_height = {}
    num_unique_fields = len(radar_field_names)
    num_field_height_pairs = len(field_name_by_pair)

    for j in range(num_unique_fields):
        radar_z_score_dict_no_height[radar_field_names[j]] = copy.deepcopy(
            orig_parameter_dict)
        radar_freq_dict_no_height[radar_field_names[j]] = {}

    for k in range(num_field_height_pairs):
        radar_z_score_dict_with_height[
            field_name_by_pair[k], height_by_pair_m_agl[k]
        ] = copy.deepcopy(orig_parameter_dict)

    sounding_z_score_dict_no_height = {}
    sounding_z_score_dict_with_height = {}
    sounding_freq_dict_no_height = {}
    num_sounding_fields = len(SOUNDING_FIELD_NAMES)
    num_sounding_heights = len(SOUNDING_HEIGHTS_M_AGL)

    for j in range(num_sounding_fields):
        sounding_z_score_dict_no_height[
            SOUNDING_FIELD_NAMES[j]
        ] = copy.deepcopy(orig_parameter_dict)
        sounding_freq_dict_no_height[SOUNDING_FIELD_NAMES[j]] = {}

        for k in range(num_sounding_heights):
            sounding_z_score_dict_with_height[
                SOUNDING_FIELD_NAMES[j], SOUNDING_HEIGHTS_M_AGL[k]
            ] = copy.deepcopy(orig_parameter_dict)

    # Update normalization params.
    num_spc_dates = len(sounding_file_names)
    for i in range(num_spc_dates):
        print 'Reading data from: "{0:s}"...'.format(
            radar_file_name_matrix[i, 0])
        this_radar_image_dict = storm_images.read_storm_images(
            netcdf_file_name=radar_file_name_matrix[i, 0], return_images=True)

        these_storm_ids = this_radar_image_dict[storm_images.STORM_IDS_KEY]
        these_storm_times_unix_sec = this_radar_image_dict[
            storm_images.VALID_TIMES_KEY]
        this_matrix_by_field_dict = {}

        for k in range(num_field_height_pairs):
            if not len(these_storm_ids):
                continue

            if k != 0:
                print 'Reading data from: "{0:s}"...'.format(
                    radar_file_name_matrix[i, k])
                this_radar_image_dict = storm_images.read_storm_images(
                    netcdf_file_name=radar_file_name_matrix[i, k],
                    return_images=True, storm_ids_to_keep=these_storm_ids,
                    valid_times_to_keep_unix_sec=these_storm_times_unix_sec)

            this_field_height_matrix = this_radar_image_dict[
                storm_images.STORM_IMAGE_MATRIX_KEY]

            if field_name_by_pair[k] in this_matrix_by_field_dict:
                this_matrix_by_field_dict[
                    field_name_by_pair[k]
                ] = numpy.concatenate(
                    (this_matrix_by_field_dict[field_name_by_pair[k]],
                     numpy.expand_dims(this_field_height_matrix, axis=-1)),
                    axis=-1)
            else:
                this_matrix_by_field_dict[
                    field_name_by_pair[k]
                ] = numpy.expand_dims(
                    this_field_height_matrix, axis=-1)

            print (
                'Updating normalization params for "{0:s}" at {1:d} metres '
                'AGL...'
            ).format(field_name_by_pair[k], height_by_pair_m_agl[k])

            radar_z_score_dict_with_height[
                field_name_by_pair[k], height_by_pair_m_agl[k]
            ] = _update_z_score_params(
                z_score_param_dict=radar_z_score_dict_with_height[
                    field_name_by_pair[k], height_by_pair_m_agl[k]],
                new_data_matrix=this_field_height_matrix)

        print MINOR_SEPARATOR_STRING

        for j in range(num_unique_fields):
            if not len(these_storm_ids):
                continue

            print 'Updating normalization params for "{0:s}"...'.format(
                radar_field_names[j])

            radar_z_score_dict_no_height[
                radar_field_names[j]
            ] = _update_z_score_params(
                z_score_param_dict=radar_z_score_dict_no_height[
                    radar_field_names[j]],
                new_data_matrix=this_matrix_by_field_dict[radar_field_names[j]])

            radar_freq_dict_no_height[
                radar_field_names[j]
            ] = _update_frequency_dict(
                frequency_dict=radar_freq_dict_no_height[radar_field_names[j]],
                new_data_matrix=this_matrix_by_field_dict[radar_field_names[j]],
                rounding_base=RADAR_INTERVAL_DICT[radar_field_names[j]])

        print MINOR_SEPARATOR_STRING

        print 'Reading data from: "{0:s}"...'.format(sounding_file_names[i])
        this_sounding_dict = soundings.read_soundings(
            netcdf_file_name=sounding_file_names[i],
            field_names_to_keep=SOUNDING_FIELD_NAMES,
            storm_ids_to_keep=these_storm_ids,
            init_times_to_keep_unix_sec=these_storm_times_unix_sec)[0]

        this_num_storm_objects = len(
            this_sounding_dict[soundings.STORM_IDS_KEY])

        for j in range(num_sounding_fields):
            if this_num_storm_objects == 0:
                continue

            this_field_index = this_sounding_dict[
                soundings.FIELD_NAMES_KEY].index(SOUNDING_FIELD_NAMES[j])

            print 'Updating normalization params for "{0:s}"...'.format(
                SOUNDING_FIELD_NAMES[j])

            sounding_z_score_dict_no_height[
                SOUNDING_FIELD_NAMES[j]
            ] = _update_z_score_params(
                z_score_param_dict=sounding_z_score_dict_no_height[
                    SOUNDING_FIELD_NAMES[j]],
                new_data_matrix=this_sounding_dict[
                    soundings.SOUNDING_MATRIX_KEY][..., this_field_index])

            sounding_freq_dict_no_height[
                SOUNDING_FIELD_NAMES[j]
            ] = _update_frequency_dict(
                frequency_dict=sounding_freq_dict_no_height[
                    SOUNDING_FIELD_NAMES[j]],
                new_data_matrix=this_sounding_dict[
                    soundings.SOUNDING_MATRIX_KEY][..., this_field_index],
                rounding_base=SOUNDING_INTERVAL_DICT[SOUNDING_FIELD_NAMES[j]])

            for k in range(num_sounding_heights):
                this_height_index = numpy.where(
                    this_sounding_dict[soundings.HEIGHT_LEVELS_KEY] ==
                    SOUNDING_HEIGHTS_M_AGL[k]
                )[0]

                print (
                    'Updating normalization params for "{0:s}" at {1:d} mb...'
                ).format(SOUNDING_FIELD_NAMES[j], SOUNDING_HEIGHTS_M_AGL[k])

                sounding_z_score_dict_with_height[
                    SOUNDING_FIELD_NAMES[j], SOUNDING_HEIGHTS_M_AGL[k]
                ] = _update_z_score_params(
                    z_score_param_dict=sounding_z_score_dict_with_height[
                        SOUNDING_FIELD_NAMES[j], SOUNDING_HEIGHTS_M_AGL[k]
                    ],
                    new_data_matrix=this_sounding_dict[
                        soundings.SOUNDING_MATRIX_KEY][
                            ..., this_height_index, this_field_index])

        if i == num_spc_dates - 1:
            print SEPARATOR_STRING
        else:
            print MINOR_SEPARATOR_STRING

    # Convert dictionaries to pandas DataFrames.
    radar_table_no_height = _convert_normalization_params(
        z_score_dict_dict=radar_z_score_dict_no_height,
        frequency_dict_dict=radar_freq_dict_no_height,
        min_percentile_level=min_percentile_level,
        max_percentile_level=max_percentile_level)
    print 'Normalization params for each radar field:\n{0:s}\n\n'.format(
        str(radar_table_no_height))

    radar_table_with_height = _convert_normalization_params(
        z_score_dict_dict=radar_z_score_dict_with_height)
    print (
        'Normalization params for each radar field/height pair:\n{0:s}\n\n'
    ).format(str(radar_table_with_height))

    sounding_table_no_height = _convert_normalization_params(
        z_score_dict_dict=sounding_z_score_dict_no_height,
        frequency_dict_dict=sounding_freq_dict_no_height,
        min_percentile_level=min_percentile_level,
        max_percentile_level=max_percentile_level)
    print 'Normalization params for each sounding field:\n{0:s}\n\n'.format(
        str(sounding_table_no_height))

    sounding_table_with_height = _convert_normalization_params(
        z_score_dict_dict=sounding_z_score_dict_with_height)
    print (
        'Normalization params for each sounding field/height pair:\n{0:s}\n\n'
    ).format(str(sounding_table_with_height))

    print 'Writing normalization params to file: "{0:s}"...'.format(
        output_file_name)
    dl_utils.write_normalization_params_to_file(
        pickle_file_name=output_file_name,
        radar_table_no_height=radar_table_no_height,
        radar_table_with_height=radar_table_with_height,
        sounding_table_no_height=sounding_table_no_height,
        sounding_table_with_height=sounding_table_with_height)


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        top_radar_image_dir_name=getattr(
            INPUT_ARG_OBJECT, RADAR_IMAGE_DIR_ARG_NAME),
        radar_source=getattr(INPUT_ARG_OBJECT, RADAR_SOURCE_ARG_NAME),
        min_percentile_level=getattr(INPUT_ARG_OBJECT, MIN_PERCENTILE_ARG_NAME),
        max_percentile_level=getattr(INPUT_ARG_OBJECT, MAX_PERCENTILE_ARG_NAME),
        top_sounding_dir_name=getattr(INPUT_ARG_OBJECT, SOUNDING_DIR_ARG_NAME),
        first_spc_date_string=getattr(
            INPUT_ARG_OBJECT, FIRST_SPC_DATE_ARG_NAME),
        last_spc_date_string=getattr(INPUT_ARG_OBJECT, LAST_SPC_DATE_ARG_NAME),
        radar_field_names=getattr(INPUT_ARG_OBJECT, RADAR_FIELD_NAMES_ARG_NAME),
        output_file_name=getattr(INPUT_ARG_OBJECT, OUTPUT_FILE_ARG_NAME))
