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
from gewittergefahr.gg_utils import number_rounding
from gewittergefahr.deep_learning import input_examples
from gewittergefahr.deep_learning import deep_learning_utils as dl_utils

LARGE_INTEGER = int(1e12)
SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

# TODO(thunderhoser): Make these input args.
NUM_RADAR_ROWS = 24
NUM_RADAR_COLUMNS = 24

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

EXAMPLE_DIR_ARG_NAME = 'input_example_dir_name'
MIN_PERCENTILE_ARG_NAME = 'min_percentile_level'
MAX_PERCENTILE_ARG_NAME = 'max_percentile_level'
OUTPUT_FILE_ARG_NAME = 'output_file_name'

EXAMPLE_DIR_HELP_STRING = (
    'Name of top-level directory with input examples.  Shuffled files therein '
    'will be found by `input_examples.find_many_example_files` and read by '
    '`input_examples.read_example_file`.')

MIN_PERCENTILE_HELP_STRING = (
    'Minimum percentile level.  The "minimum value" for each field will '
    'actually be this percentile.')

MAX_PERCENTILE_HELP_STRING = (
    'Max percentile level.  The "max value" for each field will actually be '
    'this percentile.')

OUTPUT_FILE_HELP_STRING = (
    'Path to output file (will be written by `deep_learning_utils.'
    'write_normalization_params`).')

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + EXAMPLE_DIR_ARG_NAME, type=str, required=True,
    help=EXAMPLE_DIR_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + MIN_PERCENTILE_ARG_NAME, type=float, required=False, default=0.1,
    help=MIN_PERCENTILE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + MAX_PERCENTILE_ARG_NAME, type=float, required=False, default=99.9,
    help=MAX_PERCENTILE_HELP_STRING)

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


def _run(top_example_dir_name, min_percentile_level, max_percentile_level,
         output_file_name):
    """Finds normalization parameters for GridRad data.

    This is effectively the main method.

    :param top_example_dir_name: See documentation at top of file.
    :param min_percentile_level: Same.
    :param max_percentile_level: Same.
    :param output_file_name: Same.
    """

    example_file_names = input_examples.find_many_example_files(
        top_directory_name=top_example_dir_name, shuffled=True,
        first_batch_number=0, last_batch_number=LARGE_INTEGER,
        raise_error_if_any_missing=False)

    this_example_dict = input_examples.read_example_file(example_file_names[0])
    sounding_field_names = this_example_dict[input_examples.SOUNDING_FIELDS_KEY]
    sounding_heights_m_agl = this_example_dict[
        input_examples.SOUNDING_HEIGHTS_KEY]

    if input_examples.REFL_IMAGE_MATRIX_KEY in this_example_dict:
        num_radar_dimensions = -1
    else:
        num_radar_dimensions = (
            len(this_example_dict[input_examples.RADAR_IMAGE_MATRIX_KEY].shape)
            - 2
        )

    # TODO(thunderhoser): Put this in separate method.
    if num_radar_dimensions == 3:
        radar_field_names = this_example_dict[input_examples.RADAR_FIELDS_KEY]
        radar_heights_m_agl = this_example_dict[
            input_examples.RADAR_HEIGHTS_KEY]

        radar_field_name_by_pair = []
        radar_height_by_pair_m_agl = numpy.array([], dtype=int)

        for this_field_name in radar_field_names:
            radar_field_name_by_pair += (
                [this_field_name] * len(radar_heights_m_agl)
            )
            radar_height_by_pair_m_agl = numpy.concatenate((
                radar_height_by_pair_m_agl, radar_heights_m_agl))

    elif num_radar_dimensions == 2:
        radar_field_name_by_pair = this_example_dict[
            input_examples.RADAR_FIELDS_KEY]
        radar_height_by_pair_m_agl = this_example_dict[
            input_examples.RADAR_HEIGHTS_KEY]

        radar_field_names = list(set(radar_field_name_by_pair))
        radar_field_names.sort()

    else:
        az_shear_field_names = this_example_dict[
            input_examples.RADAR_FIELDS_KEY]
        radar_field_names = [radar_utils.REFL_NAME] + az_shear_field_names

        refl_heights_m_agl = this_example_dict[input_examples.RADAR_HEIGHTS_KEY]
        radar_field_name_by_pair = (
            [radar_utils.REFL_NAME] * len(refl_heights_m_agl) +
            az_shear_field_names
        )

        az_shear_heights_m_agl = numpy.full(
            len(az_shear_field_names), radar_utils.SHEAR_NAMES)
        radar_height_by_pair_m_agl = numpy.concatenate((
            refl_heights_m_agl, az_shear_heights_m_agl
        )).astype(int)

    # Initialize parameters.
    orig_parameter_dict = {
        NUM_VALUES_KEY: 0, MEAN_VALUE_KEY: 0., MEAN_OF_SQUARES_KEY: 0.
    }

    radar_z_score_dict_no_height = {}
    radar_z_score_dict_with_height = {}
    radar_freq_dict_no_height = {}
    num_radar_fields = len(radar_field_names)
    num_radar_field_height_pairs = len(radar_field_name_by_pair)

    for j in range(num_radar_fields):
        radar_z_score_dict_no_height[radar_field_names[j]] = copy.deepcopy(
            orig_parameter_dict)
        radar_freq_dict_no_height[radar_field_names[j]] = {}

    for k in range(num_radar_field_height_pairs):
        radar_z_score_dict_with_height[
            radar_field_name_by_pair[k], radar_height_by_pair_m_agl[k]
        ] = copy.deepcopy(orig_parameter_dict)

    sounding_z_score_dict_no_height = {}
    sounding_z_score_dict_with_height = {}
    sounding_freq_dict_no_height = {}
    num_sounding_fields = len(sounding_field_names)
    num_sounding_heights = len(sounding_heights_m_agl)

    for j in range(num_sounding_fields):
        sounding_z_score_dict_no_height[sounding_field_names[j]] = (
            copy.deepcopy(orig_parameter_dict))
        sounding_freq_dict_no_height[sounding_field_names[j]] = {}

        for k in range(num_sounding_heights):
            sounding_z_score_dict_with_height[
                sounding_field_names[j], sounding_heights_m_agl[k]
            ] = copy.deepcopy(orig_parameter_dict)

    for this_example_file_name in example_file_names:
        print 'Reading data from: "{0:s}"...'.format(this_example_file_name)
        this_example_dict = input_examples.read_example_file(
            netcdf_file_name=this_example_file_name,
            num_rows_to_keep=NUM_RADAR_ROWS,
            num_columns_to_keep=NUM_RADAR_COLUMNS)

        for j in range(num_radar_fields):
            print 'Updating normalization params for "{0:s}"...'.format(
                radar_field_names[j])

            if num_radar_dimensions == 3:
                this_field_index = this_example_dict[
                    input_examples.RADAR_FIELDS_KEY
                ].index(radar_field_names[j])

                this_radar_matrix = this_example_dict[
                    input_examples.RADAR_IMAGE_MATRIX_KEY
                ][..., this_field_index]

            elif num_radar_dimensions == 2:
                all_field_names = numpy.array(
                    this_example_dict[input_examples.RADAR_FIELDS_KEY])

                these_field_indices = numpy.where(
                    all_field_names == radar_field_names[j]
                )[0]

                this_radar_matrix = this_example_dict[
                    input_examples.RADAR_IMAGE_MATRIX_KEY
                ][..., these_field_indices]

            else:
                if radar_field_names[j] == radar_utils.REFL_NAME:
                    this_radar_matrix = this_example_dict[
                        input_examples.REFL_IMAGE_MATRIX_KEY][..., 0]
                else:
                    this_field_index = this_example_dict[
                        input_examples.RADAR_FIELDS_KEY
                    ].index(radar_field_names[j])

                    this_radar_matrix = this_example_dict[
                        input_examples.AZ_SHEAR_IMAGE_MATRIX_KEY
                    ][..., this_field_index]

            radar_z_score_dict_no_height[radar_field_names[j]] = (
                _update_z_score_params(
                    z_score_param_dict=radar_z_score_dict_no_height[
                        radar_field_names[j]],
                    new_data_matrix=this_radar_matrix)
            )

            radar_freq_dict_no_height[radar_field_names[j]] = (
                _update_frequency_dict(
                    frequency_dict=radar_freq_dict_no_height[
                        radar_field_names[j]],
                    new_data_matrix=this_radar_matrix,
                    rounding_base=RADAR_INTERVAL_DICT[radar_field_names[j]])
            )

        for k in range(num_radar_field_height_pairs):
            print (
                'Updating normalization params for "{0:s}" at {1:d} metres '
                'AGL...'
            ).format(radar_field_name_by_pair[k], radar_height_by_pair_m_agl[k])

            if num_radar_dimensions == 3:
                this_field_index = this_example_dict[
                    input_examples.RADAR_FIELDS_KEY
                ].index(radar_field_name_by_pair[k])

                this_height_index = numpy.where(
                    this_example_dict[input_examples.RADAR_HEIGHTS_KEY] ==
                    radar_height_by_pair_m_agl[k]
                )[0][0]

                this_radar_matrix = this_example_dict[
                    input_examples.RADAR_IMAGE_MATRIX_KEY
                ][..., this_height_index, this_field_index]

            elif num_radar_dimensions == 2:
                all_field_names = numpy.array(
                    this_example_dict[input_examples.RADAR_FIELDS_KEY])
                all_heights_m_agl = this_example_dict[
                    input_examples.RADAR_HEIGHTS_KEY]

                this_index = numpy.where(numpy.logical_and(
                    all_field_names == radar_field_name_by_pair[k],
                    all_heights_m_agl == radar_height_by_pair_m_agl[k]
                ))[0][0]

                this_radar_matrix = this_example_dict[
                    input_examples.RADAR_IMAGE_MATRIX_KEY][..., this_index]

            else:
                if radar_field_name_by_pair[k] == radar_utils.REFL_NAME:
                    this_height_index = numpy.where(
                        this_example_dict[input_examples.RADAR_HEIGHTS_KEY] ==
                        radar_height_by_pair_m_agl[k]
                    )[0][0]

                    this_radar_matrix = this_example_dict[
                        input_examples.REFL_IMAGE_MATRIX_KEY
                    ][..., this_height_index, 0]
                else:
                    this_field_index = this_example_dict[
                        input_examples.RADAR_FIELDS_KEY
                    ].index(radar_field_name_by_pair[k])

                    this_radar_matrix = this_example_dict[
                        input_examples.AZ_SHEAR_IMAGE_MATRIX_KEY
                    ][..., this_field_index]

            radar_z_score_dict_with_height[
                radar_field_name_by_pair[k], radar_height_by_pair_m_agl[k]
            ] = _update_z_score_params(
                z_score_param_dict=radar_z_score_dict_with_height[
                    radar_field_name_by_pair[k], radar_height_by_pair_m_agl[k]],
                new_data_matrix=this_radar_matrix)

        for j in range(num_sounding_fields):
            print 'Updating normalization params for "{0:s}"...'.format(
                sounding_field_names[j])

            this_field_index = this_example_dict[
                input_examples.SOUNDING_FIELDS_KEY
            ].index(sounding_field_names[j])

            this_sounding_matrix = this_example_dict[
                input_examples.SOUNDING_MATRIX_KEY][..., this_field_index]

            sounding_z_score_dict_no_height[sounding_field_names[j]] = (
                _update_z_score_params(
                    z_score_param_dict=sounding_z_score_dict_no_height[
                        sounding_field_names[j]],
                    new_data_matrix=this_sounding_matrix)
            )

            sounding_freq_dict_no_height[sounding_field_names[j]] = (
                _update_frequency_dict(
                    frequency_dict=sounding_freq_dict_no_height[
                        sounding_field_names[j]],
                    new_data_matrix=this_sounding_matrix,
                    rounding_base=SOUNDING_INTERVAL_DICT[
                        sounding_field_names[j]]
                )
            )

            for k in range(num_sounding_heights):
                this_height_index = numpy.where(
                    this_example_dict[input_examples.SOUNDING_HEIGHTS_KEY] ==
                    sounding_heights_m_agl[k]
                )[0][0]

                this_sounding_matrix = this_example_dict[
                    input_examples.SOUNDING_MATRIX_KEY
                ][..., this_height_index, this_field_index]

                print (
                    'Updating normalization params for "{0:s}" at {1:d} m '
                    'AGL...'
                ).format(sounding_field_names[j], sounding_heights_m_agl[k])

                sounding_z_score_dict_with_height[
                    sounding_field_names[j], sounding_heights_m_agl[k]
                ] = _update_z_score_params(
                    z_score_param_dict=sounding_z_score_dict_with_height[
                        sounding_field_names[j], sounding_heights_m_agl[k]
                    ],
                    new_data_matrix=this_sounding_matrix)

        print SEPARATOR_STRING

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
    dl_utils.write_normalization_params(
        pickle_file_name=output_file_name,
        radar_table_no_height=radar_table_no_height,
        radar_table_with_height=radar_table_with_height,
        sounding_table_no_height=sounding_table_no_height,
        sounding_table_with_height=sounding_table_with_height)


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        top_example_dir_name=getattr(INPUT_ARG_OBJECT, EXAMPLE_DIR_ARG_NAME),
        min_percentile_level=getattr(INPUT_ARG_OBJECT, MIN_PERCENTILE_ARG_NAME),
        max_percentile_level=getattr(INPUT_ARG_OBJECT, MAX_PERCENTILE_ARG_NAME),
        output_file_name=getattr(INPUT_ARG_OBJECT, OUTPUT_FILE_ARG_NAME)
    )
