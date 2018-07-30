"""Finds climatological averages for GridRad training data.

GridRad training data consist of storm-centered radar images and storm-centered
NWP soundings.
"""

import argparse
import numpy
from gewittergefahr.gg_utils import radar_utils
from gewittergefahr.gg_utils import soundings_only
from gewittergefahr.gg_utils import nwp_model_utils
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.deep_learning import storm_images
from gewittergefahr.deep_learning import deep_learning_utils as dl_utils
from gewittergefahr.deep_learning import training_validation_io as trainval_io

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

RADAR_HEIGHTS_M_ASL = numpy.linspace(1000, 12000, num=12, dtype=int)
SOUNDING_FIELD_NAMES = [
    soundings_only.TEMPERATURE_NAME,
    soundings_only.VIRTUAL_POTENTIAL_TEMPERATURE_NAME,
    soundings_only.U_WIND_NAME, soundings_only.V_WIND_NAME,
    soundings_only.SPECIFIC_HUMIDITY_NAME, soundings_only.RELATIVE_HUMIDITY_NAME
]
SOUNDING_PRESSURE_LEVELS_MB = nwp_model_utils.get_pressure_levels(
    model_name=nwp_model_utils.RAP_MODEL_NAME,
    grid_id=nwp_model_utils.ID_FOR_130GRID)

DUMMY_TARGET_NAME = 'tornado_lead-time=0000-3600sec_distance=00000-10000m'
LAG_TIME_FOR_CONVECTIVE_CONTAMINATION_SEC = 1800

RADAR_IMAGE_DIR_ARG_NAME = 'input_radar_image_dir_name'
SOUNDING_DIR_ARG_NAME = 'input_sounding_dir_name'
FIRST_SPC_DATE_ARG_NAME = 'first_spc_date_string'
LAST_SPC_DATE_ARG_NAME = 'last_spc_date_string'
RADAR_FIELD_NAMES_ARG_NAME = 'radar_field_names'
OUTPUT_FILE_ARG_NAME = 'output_file_name'

RADAR_IMAGE_DIR_HELP_STRING = (
    'Name of top-level directory with storm-centered radar images.  Files '
    'therein will be found by `storm_images.find_storm_image_file` and read by '
    '`storm_images.read_storm_images`.')
SOUNDING_DIR_HELP_STRING = (
    'Name of top-level directory with storm-centered soundings.  Files therein '
    'will be found by `soundings_only.find_sounding_file` and read by '
    '`soundings_only.read_soundings`.')
SPC_DATE_HELP_STRING = (
    'SPC (Storm Prediction Center) date in format "yyyymmdd".  Climatological '
    'averages will be based on all data from `{0:s}`...`{1:s}`.'
).format(FIRST_SPC_DATE_ARG_NAME, LAST_SPC_DATE_ARG_NAME)
RADAR_FIELD_NAMES_HELP_STRING = (
    'List of radar fields (each must be accepted by '
    '`radar_utils.check_field_name`).  The climatological average will be '
    'computed for each of these fields at each height (metres above sea level) '
    'in the following list.\n{0:s}'
).format(str(RADAR_HEIGHTS_M_ASL))
OUTPUT_FILE_HELP_STRING = (
    'Path to output file.  This will be a Pickle file with the climatological '
    'mean for each radar field/height and each sounding field/pressure.')

DEFAULT_TOP_RADAR_IMAGE_DIR_NAME = (
    '/condo/swatcommon/common/gridrad_final/myrorss_format/tracks/reanalyzed/'
    'storm_images_with_rdp')
DEFAULT_TOP_SOUNDING_DIR_NAME = (
    '/condo/swatcommon/common/gridrad_final/myrorss_format/tracks/reanalyzed/'
    'soundings')
DEFAULT_RADAR_FIELD_NAMES = [
    radar_utils.REFL_NAME, radar_utils.SPECTRUM_WIDTH_NAME,
    radar_utils.DIVERGENCE_NAME, radar_utils.VORTICITY_NAME,
    radar_utils.DIFFERENTIAL_REFL_NAME, radar_utils.CORRELATION_COEFF_NAME,
    radar_utils.SPEC_DIFF_PHASE_NAME
]

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + RADAR_IMAGE_DIR_ARG_NAME, type=str, required=False,
    default=DEFAULT_TOP_RADAR_IMAGE_DIR_NAME,
    help=RADAR_IMAGE_DIR_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + SOUNDING_DIR_ARG_NAME, type=str, required=False,
    default=DEFAULT_TOP_SOUNDING_DIR_NAME, help=SOUNDING_DIR_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + FIRST_SPC_DATE_ARG_NAME, type=str, required=True,
    help=SPC_DATE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + LAST_SPC_DATE_ARG_NAME, type=str, required=True,
    help=SPC_DATE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + RADAR_FIELD_NAMES_ARG_NAME, type=str, nargs='+', required=False,
    default=DEFAULT_RADAR_FIELD_NAMES, help=RADAR_FIELD_NAMES_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_FILE_ARG_NAME, type=str, required=True,
    help=OUTPUT_FILE_HELP_STRING)


def _get_weighted_average(input_values, input_weights):
    """Computes weighted average.

    N = number of values to average

    :param input_values: length-N numpy array of values to average.
    :param input_weights: length-N numpy array of weights.
    :return: weighted_average: Weighted average.
    """

    return numpy.average(input_values, weights=input_weights)


def _run(
        top_radar_image_dir_name, top_sounding_dir_name, first_spc_date_string,
        last_spc_date_string, radar_field_names, output_file_name):
    """Finds climatological averages for GridRad training data.

    This is effectively the main method.

    :param top_radar_image_dir_name: See documentation at top of file.
    :param top_sounding_dir_name: Same.
    :param first_spc_date_string: Same.
    :param last_spc_date_string: Same.
    :param radar_field_names: Same.
    :param output_file_name: Same.
    """

    radar_file_name_matrix = trainval_io.find_radar_files_3d(
        top_directory_name=top_radar_image_dir_name,
        radar_source=radar_utils.GRIDRAD_SOURCE_ID,
        radar_field_names=radar_field_names,
        radar_heights_m_asl=RADAR_HEIGHTS_M_ASL,
        first_file_time_unix_sec=time_conversion.spc_date_string_to_unix_sec(
            first_spc_date_string),
        last_file_time_unix_sec=time_conversion.spc_date_string_to_unix_sec(
            last_spc_date_string),
        one_file_per_time_step=False, shuffle_times=False)[0]
    print SEPARATOR_STRING

    sounding_file_names = trainval_io.find_sounding_files(
        top_sounding_dir_name=top_sounding_dir_name,
        radar_file_name_matrix=radar_file_name_matrix,
        target_name=DUMMY_TARGET_NAME,
        lag_time_for_convective_contamination_sec=
        LAG_TIME_FOR_CONVECTIVE_CONTAMINATION_SEC)
    print SEPARATOR_STRING

    num_spc_dates = len(sounding_file_names)
    num_radar_fields = len(radar_field_names)
    num_radar_heights = len(RADAR_HEIGHTS_M_ASL)
    num_sounding_fields = len(SOUNDING_FIELD_NAMES)
    num_sounding_pressures = len(SOUNDING_PRESSURE_LEVELS_MB)

    mean_radar_value_dict = {}
    for j in range(num_radar_fields):
        for k in range(num_radar_heights):
            mean_radar_value_dict[
                radar_field_names[j], RADAR_HEIGHTS_M_ASL[k]] = 0.

    mean_sounding_value_dict = {}
    for j in range(num_sounding_fields):
        for k in range(num_sounding_pressures):
            mean_sounding_value_dict[
                SOUNDING_FIELD_NAMES[j], SOUNDING_PRESSURE_LEVELS_MB[k]] = 0.

    num_storm_objects_with_radar_images = 0
    num_storm_objects_with_soundings = 0

    for i in range(num_spc_dates):
        print 'Reading data from: "{0:s}"...'.format(
            radar_file_name_matrix[i, 0, 0])
        this_radar_image_dict = storm_images.read_storm_images(
            netcdf_file_name=radar_file_name_matrix[i, 0, 0],
            return_images=True)

        these_storm_ids = this_radar_image_dict[storm_images.STORM_IDS_KEY]
        these_storm_times_unix_sec = this_radar_image_dict[
            storm_images.VALID_TIMES_KEY]
        this_num_storm_objects = len(these_storm_ids)

        for j in range(num_radar_fields):
            for k in range(num_radar_heights):
                if not j == k == 0:
                    print 'Reading data from: "{0:s}"...'.format(
                        radar_file_name_matrix[i, j, k])
                    this_radar_image_dict = storm_images.read_storm_images(
                        netcdf_file_name=radar_file_name_matrix[i, j, k],
                        return_images=True, storm_ids_to_keep=these_storm_ids,
                        valid_times_to_keep_unix_sec=these_storm_times_unix_sec)

                this_mean_value = numpy.mean(
                    this_radar_image_dict[
                        storm_images.STORM_IMAGE_MATRIX_KEY][:, j, k])
                these_values = numpy.array([
                    mean_radar_value_dict[
                        radar_field_names[j], RADAR_HEIGHTS_M_ASL[k]],
                    this_mean_value
                ])

                mean_radar_value_dict[
                    radar_field_names[j], RADAR_HEIGHTS_M_ASL[k]
                ] = _get_weighted_average(
                    input_values=these_values,
                    input_weights=numpy.array([
                        num_storm_objects_with_radar_images,
                        this_num_storm_objects])
                )

        num_storm_objects_with_radar_images += this_num_storm_objects

        print 'Reading data from: "{0:s}"...'.format(sounding_file_names[i])
        this_sounding_dict = soundings_only.read_soundings(
            netcdf_file_name=sounding_file_names[i],
            pressureless_field_names_to_keep=SOUNDING_FIELD_NAMES,
            storm_ids_to_keep=these_storm_ids,
            init_times_to_keep_unix_sec=these_storm_times_unix_sec)[0]

        this_num_storm_objects = len(
            this_sounding_dict[soundings_only.STORM_IDS_KEY])

        for j in range(num_sounding_fields):
            if this_num_storm_objects == 0:
                continue

            this_field_index = this_sounding_dict[
                soundings_only.PRESSURELESS_FIELD_NAMES_KEY
            ].index(SOUNDING_FIELD_NAMES[j])

            for k in range(num_sounding_pressures):
                this_pressure_index = numpy.where(
                    this_sounding_dict[soundings_only.VERTICAL_LEVELS_KEY] ==
                    SOUNDING_PRESSURE_LEVELS_MB[k]
                )[0]

                this_mean_value = numpy.mean(
                    this_sounding_dict[soundings_only.SOUNDING_MATRIX_KEY][
                        :, this_pressure_index, this_field_index])
                these_values = numpy.array([
                    mean_sounding_value_dict[
                        SOUNDING_FIELD_NAMES[j], SOUNDING_PRESSURE_LEVELS_MB[k]
                    ],
                    this_mean_value
                ])

                mean_sounding_value_dict[
                    SOUNDING_FIELD_NAMES[j], SOUNDING_PRESSURE_LEVELS_MB[k]
                ] = _get_weighted_average(
                    input_values=these_values,
                    input_weights=numpy.array([
                        num_storm_objects_with_soundings,
                        this_num_storm_objects
                    ])
                )

        num_storm_objects_with_soundings += this_num_storm_objects
        print SEPARATOR_STRING

    for j in range(num_radar_fields):
        for k in range(num_radar_heights):
            print 'Mean "{0:s}" at {1:d} metres ASL = {2:.2e}'.format(
                radar_field_names[j], RADAR_HEIGHTS_M_ASL[k],
                mean_radar_value_dict[
                    radar_field_names[j], RADAR_HEIGHTS_M_ASL[k]])

    print SEPARATOR_STRING

    for j in range(num_sounding_fields):
        for k in range(num_sounding_pressures):
            print 'Mean "{0:s}" at {1:d} mb = {2:.2e}'.format(
                SOUNDING_FIELD_NAMES[j], SOUNDING_PRESSURE_LEVELS_MB[k],
                mean_sounding_value_dict[
                    SOUNDING_FIELD_NAMES[j], SOUNDING_PRESSURE_LEVELS_MB[k]])

    print SEPARATOR_STRING
    print 'Writing climatological averages to file: "{0:s}"...'.format(
        output_file_name)
    dl_utils.write_climo_averages_to_file(
        pickle_file_name=output_file_name,
        mean_radar_value_dict=mean_radar_value_dict,
        mean_sounding_value_dict=mean_sounding_value_dict)


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        top_radar_image_dir_name=getattr(
            INPUT_ARG_OBJECT, RADAR_IMAGE_DIR_ARG_NAME),
        top_sounding_dir_name=getattr(INPUT_ARG_OBJECT, SOUNDING_DIR_ARG_NAME),
        first_spc_date_string=getattr(
            INPUT_ARG_OBJECT, FIRST_SPC_DATE_ARG_NAME),
        last_spc_date_string=getattr(INPUT_ARG_OBJECT, LAST_SPC_DATE_ARG_NAME),
        radar_field_names=getattr(INPUT_ARG_OBJECT, RADAR_FIELD_NAMES_ARG_NAME),
        output_file_name=getattr(INPUT_ARG_OBJECT, OUTPUT_FILE_ARG_NAME))
