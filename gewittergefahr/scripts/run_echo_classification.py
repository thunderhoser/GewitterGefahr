"""Runs echo classification (convective vs. non-convective)."""

import os.path
import argparse
import numpy
from gewittergefahr.gg_io import gridrad_io
from gewittergefahr.gg_io import myrorss_io
from gewittergefahr.gg_io import myrorss_and_mrms_io
from gewittergefahr.gg_utils import radar_utils
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import time_periods
from gewittergefahr.gg_utils import unzipping
from gewittergefahr.gg_utils import radar_sparse_to_full as radar_s2f
from gewittergefahr.gg_utils import echo_classification as echo_classifn

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

TIME_INTERVAL_SEC = 300
RADAR_HEIGHTS_M_ASL = numpy.linspace(1000, 12000, num=12, dtype=int)

RADAR_SOURCE_ARG_NAME = 'radar_source_name'
SPC_DATE_ARG_NAME = 'spc_date_string'
TARRED_RADAR_DIR_ARG_NAME = 'input_radar_dir_name_tarred'
RADAR_DIR_ARG_NAME = 'input_radar_dir_name'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'
PEAKEDNESS_NEIGH_ARG_NAME = 'peakedness_neigh_metres'
MAX_PEAKEDNESS_HEIGHT_ARG_NAME = 'max_peakedness_height_m_asl'
MIN_ECHO_TOP_ARG_NAME = 'min_echo_top_m_asl'
ECHO_TOP_LEVEL_ARG_NAME = 'echo_top_level_dbz'
MIN_COMPOSITE_REFL_CRITERION1_ARG_NAME = 'min_composite_refl_criterion1_dbz'
MIN_COMPOSITE_REFL_CRITERION5_ARG_NAME = 'min_composite_refl_criterion5_dbz'
MIN_COMPOSITE_REFL_AML_ARG_NAME = 'min_composite_refl_aml_dbz'

RADAR_SOURCE_HELP_STRING = (
    'Source of radar data (must be accepted by '
    '`radar_utils.check_data_source`).')

SPC_DATE_HELP_STRING = (
    'SPC date (format "yyyymmdd").  Echo classification will be done for all '
    'time steps on this date.')

TARRED_RADAR_DIR_HELP_STRING = (
    '[used only if {0:s} = "{1:s}"] Name of top-level directory with tarred '
    'MYRORSS files.  These files will be untarred before processing, to the '
    'directory `{2:s}`, and the untarred files will be deleted after '
    'processing.'
).format(RADAR_SOURCE_ARG_NAME, radar_utils.MYRORSS_SOURCE_ID,
         RADAR_DIR_ARG_NAME)

RADAR_DIR_HELP_STRING = (
    'Name of top-level radar directory.  Files therein will be found by either '
    '`myrorss_and_mrms_io.find_raw_file` or `gridrad_io.find_file`.')

OUTPUT_DIR_HELP_STRING = (
    'Name of top-level output directory.  Echo classifications will be written '
    'by `echo_classification.write_classifications`, to locations therein '
    'determined by `echo_classification.find_classification_file`.')

PEAKEDNESS_NEIGH_HELP_STRING = (
    'Neighbourhood radius for peakedness calculations.')

MAX_PEAKEDNESS_HEIGHT_HELP_STRING = (
    'Max height (metres above sea level) for peakedness calculations.')

MIN_ECHO_TOP_HELP_STRING = (
    'Minimum echo top (metres above sea level), used for criterion 3.')

ECHO_TOP_LEVEL_HELP_STRING = (
    'Critical reflectivity (used to compute echo top for criterion 3).')

MIN_COMPOSITE_REFL_CRITERION1_HELP_STRING = (
    'Minimum composite (column-max) reflectivity for criterion 1.  To exclude '
    'this criterion, make the value negative.')

MIN_COMPOSITE_REFL_CRITERION5_HELP_STRING = (
    'Minimum composite reflectivity for criterion 5.')

MIN_COMPOSITE_REFL_AML_HELP_STRING = (
    'Minimum composite reflectivity above melting level, used for criterion 2.')

DEFAULT_TARRED_RADAR_DIR_NAME = '/condo/swatcommon/common/myrorss'

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + RADAR_SOURCE_ARG_NAME, type=str, required=True,
    help=RADAR_SOURCE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + SPC_DATE_ARG_NAME, type=str, required=True,
    help=SPC_DATE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + TARRED_RADAR_DIR_ARG_NAME, type=str, required=False,
    default=DEFAULT_TARRED_RADAR_DIR_NAME, help=TARRED_RADAR_DIR_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + RADAR_DIR_ARG_NAME, type=str, required=True,
    help=RADAR_DIR_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + PEAKEDNESS_NEIGH_ARG_NAME, type=float, required=False,
    default=echo_classifn.DEFAULT_OPTION_DICT[
        echo_classifn.PEAKEDNESS_NEIGH_KEY],
    help=PEAKEDNESS_NEIGH_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + MAX_PEAKEDNESS_HEIGHT_ARG_NAME, type=float, required=False,
    default=echo_classifn.DEFAULT_OPTION_DICT[
        echo_classifn.MAX_PEAKEDNESS_HEIGHT_KEY],
    help=MAX_PEAKEDNESS_HEIGHT_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + MIN_ECHO_TOP_ARG_NAME, type=int, required=False,
    default=echo_classifn.DEFAULT_OPTION_DICT[echo_classifn.MIN_ECHO_TOP_KEY],
    help=MIN_ECHO_TOP_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + ECHO_TOP_LEVEL_ARG_NAME, type=float, required=False,
    default=echo_classifn.DEFAULT_OPTION_DICT[echo_classifn.ECHO_TOP_LEVEL_KEY],
    help=ECHO_TOP_LEVEL_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + MIN_COMPOSITE_REFL_CRITERION1_ARG_NAME, type=float, required=False,
    default=echo_classifn.DEFAULT_OPTION_DICT[
        echo_classifn.MIN_COMPOSITE_REFL_CRITERION1_KEY],
    help=MIN_COMPOSITE_REFL_CRITERION1_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + MIN_COMPOSITE_REFL_CRITERION5_ARG_NAME, type=float, required=False,
    default=echo_classifn.DEFAULT_OPTION_DICT[
        echo_classifn.MIN_COMPOSITE_REFL_CRITERION5_KEY],
    help=MIN_COMPOSITE_REFL_CRITERION5_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + MIN_COMPOSITE_REFL_AML_ARG_NAME, type=float, required=False,
    default=echo_classifn.DEFAULT_OPTION_DICT[
        echo_classifn.MIN_COMPOSITE_REFL_AML_KEY],
    help=MIN_COMPOSITE_REFL_AML_HELP_STRING)


def _run_for_gridrad(
        spc_date_string, top_radar_dir_name, top_output_dir_name, option_dict):
    """Runs echo classification for GridRad data.

    :param spc_date_string: See documentation at top of file.
    :param top_radar_dir_name: Same.
    :param top_output_dir_name: Same.
    :param option_dict: See doc for
        `echo_classification.find_convective_pixels`.
    """

    valid_times_unix_sec = time_periods.range_and_interval_to_list(
        start_time_unix_sec=time_conversion.get_start_of_spc_date(
            spc_date_string),
        end_time_unix_sec=time_conversion.get_end_of_spc_date(
            spc_date_string),
        time_interval_sec=TIME_INTERVAL_SEC, include_endpoint=True)

    num_times = len(valid_times_unix_sec)
    radar_file_names = [''] * num_times
    indices_to_keep = []

    for i in range(num_times):
        radar_file_names[i] = gridrad_io.find_file(
            top_directory_name=top_radar_dir_name,
            unix_time_sec=valid_times_unix_sec[i],
            raise_error_if_missing=False)

        if os.path.isfile(radar_file_names[i]):
            indices_to_keep.append(i)

    indices_to_keep = numpy.array(indices_to_keep, dtype=int)
    valid_times_unix_sec = valid_times_unix_sec[indices_to_keep]
    radar_file_names = [radar_file_names[k] for k in indices_to_keep]
    num_times = len(valid_times_unix_sec)

    for i in range(num_times):
        print 'Reading data from: "{0:s}"...\n'.format(radar_file_names[i])
        radar_metadata_dict = gridrad_io.read_metadata_from_full_grid_file(
            netcdf_file_name=radar_file_names[i])

        (reflectivity_matrix_dbz, all_heights_m_asl, grid_point_latitudes_deg,
         grid_point_longitudes_deg
        ) = gridrad_io.read_field_from_full_grid_file(
            netcdf_file_name=radar_file_names[i],
            field_name=radar_utils.REFL_NAME, metadata_dict=radar_metadata_dict)

        reflectivity_matrix_dbz = numpy.rollaxis(
            reflectivity_matrix_dbz, axis=0, start=3)

        height_indices = numpy.array(
            [all_heights_m_asl.tolist().index(h) for h in RADAR_HEIGHTS_M_ASL],
            dtype=int)
        reflectivity_matrix_dbz = reflectivity_matrix_dbz[..., height_indices]

        grid_metadata_dict = {
            echo_classifn.MIN_LATITUDE_KEY: numpy.min(grid_point_latitudes_deg),
            echo_classifn.LATITUDE_SPACING_KEY:
                grid_point_latitudes_deg[1] - grid_point_latitudes_deg[0],
            echo_classifn.MIN_LONGITUDE_KEY: numpy.min(grid_point_longitudes_deg),
            echo_classifn.LONGITUDE_SPACING_KEY:
                grid_point_longitudes_deg[1] - grid_point_longitudes_deg[0],
            echo_classifn.HEIGHTS_KEY: RADAR_HEIGHTS_M_ASL
        }

        convective_flag_matrix = echo_classifn.find_convective_pixels(
            reflectivity_matrix_dbz=reflectivity_matrix_dbz,
            grid_metadata_dict=grid_metadata_dict,
            valid_time_unix_sec=valid_times_unix_sec[i],
            option_dict=option_dict)

        print 'Number of convective pixels = {0:d}\n'.format(
            numpy.sum(convective_flag_matrix))

        this_output_file_name = echo_classifn.find_classification_file(
            top_directory_name=top_output_dir_name,
            valid_time_unix_sec=valid_times_unix_sec[i],
            raise_error_if_missing=False)

        print 'Writing echo classifications to: "{0:s}"...'.format(
            this_output_file_name)

        echo_classifn.write_classifications(
            convective_flag_matrix=convective_flag_matrix,
            grid_metadata_dict=grid_metadata_dict,
            valid_time_unix_sec=valid_times_unix_sec[i],
            option_dict=option_dict, netcdf_file_name=this_output_file_name)

        print SEPARATOR_STRING


def _run_for_myrorss(
        spc_date_string, top_radar_dir_name_tarred, top_radar_dir_name_untarred,
        top_output_dir_name, option_dict):
    """Runs echo classification for MYRORSS data.

    :param spc_date_string: See documentation at top of file.
    :param top_radar_dir_name_tarred: Same.
    :param top_radar_dir_name_untarred: Same.
    :param top_output_dir_name: Same.
    :param option_dict: See doc for
        `echo_classification.find_convective_pixels`.
    """

    tar_file_name = '{0:s}/{1:s}/{2:s}.tar'.format(
        top_radar_dir_name_tarred, spc_date_string[:4], spc_date_string)

    myrorss_io.unzip_1day_tar_file(
        tar_file_name=tar_file_name, field_names=[radar_utils.REFL_NAME],
        spc_date_string=spc_date_string,
        top_target_directory_name=top_radar_dir_name_untarred,
        refl_heights_m_asl=RADAR_HEIGHTS_M_ASL)
    print SEPARATOR_STRING

    desired_times_unix_sec = time_periods.range_and_interval_to_list(
        start_time_unix_sec=time_conversion.get_start_of_spc_date(
            spc_date_string),
        end_time_unix_sec=time_conversion.get_end_of_spc_date(
            spc_date_string),
        time_interval_sec=TIME_INTERVAL_SEC, include_endpoint=True)

    radar_file_dict = myrorss_and_mrms_io.find_many_raw_files(
        desired_times_unix_sec=desired_times_unix_sec,
        spc_date_strings=[spc_date_string] * len(desired_times_unix_sec),
        data_source=radar_utils.MYRORSS_SOURCE_ID,
        field_names=[radar_utils.REFL_NAME],
        top_directory_name=top_radar_dir_name_untarred,
        reflectivity_heights_m_asl=RADAR_HEIGHTS_M_ASL)

    radar_file_name_matrix = radar_file_dict[
        myrorss_and_mrms_io.RADAR_FILE_NAMES_KEY]
    valid_times_unix_sec = radar_file_dict[
        myrorss_and_mrms_io.UNIQUE_TIMES_KEY]

    num_times = len(valid_times_unix_sec)
    num_heights = len(RADAR_HEIGHTS_M_ASL)

    for i in range(num_times):
        reflectivity_matrix_dbz = None
        fine_grid_point_latitudes_deg = None
        fine_grid_point_longitudes_deg = None

        for j in range(num_heights):
            print 'Reading data from: "{0:s}"...'.format(
                radar_file_name_matrix[i, j])

            this_metadata_dict = (
                myrorss_and_mrms_io.read_metadata_from_raw_file(
                    netcdf_file_name=radar_file_name_matrix[i, j],
                    data_source=radar_utils.MYRORSS_SOURCE_ID)
            )

            this_sparse_grid_table = (
                myrorss_and_mrms_io.read_data_from_sparse_grid_file(
                    netcdf_file_name=radar_file_name_matrix[i, j],
                    field_name_orig=this_metadata_dict[
                        myrorss_and_mrms_io.FIELD_NAME_COLUMN_ORIG],
                    data_source=radar_utils.MYRORSS_SOURCE_ID,
                    sentinel_values=this_metadata_dict[
                        radar_utils.SENTINEL_VALUE_COLUMN]
                )
            )

            (this_refl_matrix_dbz, fine_grid_point_latitudes_deg,
             fine_grid_point_longitudes_deg
            ) = radar_s2f.sparse_to_full_grid(
                sparse_grid_table=this_sparse_grid_table,
                metadata_dict=this_metadata_dict)

            this_refl_matrix_dbz = numpy.expand_dims(
                this_refl_matrix_dbz[::2, ::2], axis=-1)

            if reflectivity_matrix_dbz is None:
                reflectivity_matrix_dbz = this_refl_matrix_dbz + 0.
            else:
                reflectivity_matrix_dbz = numpy.concatenate(
                    (reflectivity_matrix_dbz, this_refl_matrix_dbz), axis=-1)

        print '\n'

        reflectivity_matrix_dbz = numpy.flip(reflectivity_matrix_dbz, axis=0)
        fine_grid_point_latitudes_deg = fine_grid_point_latitudes_deg[::-1]
        coarse_grid_point_latitudes_deg = fine_grid_point_latitudes_deg[::2]
        coarse_grid_point_longitudes_deg = fine_grid_point_longitudes_deg[::2]

        coarse_grid_metadata_dict = {
            echo_classifn.MIN_LATITUDE_KEY:
                numpy.min(coarse_grid_point_latitudes_deg),
            echo_classifn.LATITUDE_SPACING_KEY:
                (coarse_grid_point_latitudes_deg[1] -
                 coarse_grid_point_latitudes_deg[0]),
            echo_classifn.MIN_LONGITUDE_KEY:
                numpy.min(coarse_grid_point_longitudes_deg),
            echo_classifn.LONGITUDE_SPACING_KEY:
                (coarse_grid_point_longitudes_deg[1] -
                 coarse_grid_point_longitudes_deg[0]),
            echo_classifn.HEIGHTS_KEY: RADAR_HEIGHTS_M_ASL
        }

        fine_grid_metadata_dict = {
            echo_classifn.MIN_LATITUDE_KEY:
                numpy.min(fine_grid_point_latitudes_deg),
            echo_classifn.LATITUDE_SPACING_KEY:
                (fine_grid_point_latitudes_deg[1] -
                 fine_grid_point_latitudes_deg[0]),
            echo_classifn.MIN_LONGITUDE_KEY:
                numpy.min(fine_grid_point_longitudes_deg),
            echo_classifn.LONGITUDE_SPACING_KEY:
                (fine_grid_point_longitudes_deg[1] -
                 fine_grid_point_longitudes_deg[0]),
            echo_classifn.HEIGHTS_KEY: RADAR_HEIGHTS_M_ASL
        }

        convective_flag_matrix = echo_classifn.find_convective_pixels(
            reflectivity_matrix_dbz=reflectivity_matrix_dbz,
            grid_metadata_dict=coarse_grid_metadata_dict,
            valid_time_unix_sec=valid_times_unix_sec[i],
            option_dict=option_dict)

        print 'Number of convective pixels = {0:d}\n'.format(
            numpy.sum(convective_flag_matrix))

        convective_flag_matrix = echo_classifn._double_class_resolution(
            coarse_convective_flag_matrix=convective_flag_matrix,
            coarse_grid_point_latitudes_deg=coarse_grid_point_latitudes_deg,
            coarse_grid_point_longitudes_deg=coarse_grid_point_longitudes_deg,
            fine_grid_point_latitudes_deg=fine_grid_point_latitudes_deg,
            fine_grid_point_longitudes_deg=fine_grid_point_longitudes_deg)

        this_output_file_name = echo_classifn.find_classification_file(
            top_directory_name=top_output_dir_name,
            valid_time_unix_sec=valid_times_unix_sec[i],
            raise_error_if_missing=False)

        print 'Writing echo classifications to: "{0:s}"...'.format(
            this_output_file_name)

        echo_classifn.write_classifications(
            convective_flag_matrix=convective_flag_matrix,
            grid_metadata_dict=fine_grid_metadata_dict,
            valid_time_unix_sec=valid_times_unix_sec[i],
            option_dict=option_dict, netcdf_file_name=this_output_file_name)

        unzipping.gzip_file(input_file_name=this_output_file_name,
                            delete_input_file=True)

        print SEPARATOR_STRING

    myrorss_io.remove_unzipped_data_1day(
        spc_date_string=spc_date_string,
        top_directory_name=top_radar_dir_name_untarred,
        field_names=[radar_utils.REFL_NAME],
        refl_heights_m_asl=RADAR_HEIGHTS_M_ASL)
    print SEPARATOR_STRING


def _run(radar_source_name, spc_date_string, top_radar_dir_name_tarred,
         top_radar_dir_name, top_output_dir_name, peakedness_neigh_metres,
         max_peakedness_height_m_asl, min_echo_top_m_asl, echo_top_level_dbz,
         min_composite_refl_criterion1_dbz, min_composite_refl_criterion5_dbz,
         min_composite_refl_aml_dbz):
    """Runs echo classification (convective vs. non-convective).

    This is effectively the main method.

    :param radar_source_name: See documentation at top of file.
    :param spc_date_string: Same.
    :param top_radar_dir_name_tarred: Same.
    :param top_radar_dir_name: Same.
    :param top_output_dir_name: Same.
    :param peakedness_neigh_metres: Same.
    :param max_peakedness_height_m_asl: Same.
    :param min_echo_top_m_asl: Same.
    :param echo_top_level_dbz: Same.
    :param min_composite_refl_criterion1_dbz: Same.
    :param min_composite_refl_criterion5_dbz: Same.
    :param min_composite_refl_aml_dbz: Same.
    """

    if min_composite_refl_criterion1_dbz <= 0:
        min_composite_refl_criterion1_dbz = None

    option_dict = {
        echo_classifn.PEAKEDNESS_NEIGH_KEY: peakedness_neigh_metres,
        echo_classifn.MAX_PEAKEDNESS_HEIGHT_KEY: max_peakedness_height_m_asl,
        echo_classifn.HALVE_RESOLUTION_KEY: False,
        # radar_source_name != radar_utils.GRIDRAD_SOURCE_ID,
        echo_classifn.MIN_ECHO_TOP_KEY: min_echo_top_m_asl,
        echo_classifn.ECHO_TOP_LEVEL_KEY: echo_top_level_dbz,
        echo_classifn.MIN_COMPOSITE_REFL_CRITERION1_KEY:
            min_composite_refl_criterion1_dbz,
        echo_classifn.MIN_COMPOSITE_REFL_CRITERION5_KEY:
            min_composite_refl_criterion5_dbz,
        echo_classifn.MIN_COMPOSITE_REFL_AML_KEY: min_composite_refl_aml_dbz
    }

    if radar_source_name == radar_utils.GRIDRAD_SOURCE_ID:
        _run_for_gridrad(
            spc_date_string=spc_date_string,
            top_radar_dir_name=top_radar_dir_name,
            top_output_dir_name=top_output_dir_name, option_dict=option_dict)
    else:
        _run_for_myrorss(
            spc_date_string=spc_date_string,
            top_radar_dir_name_tarred=top_radar_dir_name_tarred,
            top_radar_dir_name_untarred=top_radar_dir_name,
            top_output_dir_name=top_output_dir_name, option_dict=option_dict)


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        radar_source_name=getattr(INPUT_ARG_OBJECT, RADAR_SOURCE_ARG_NAME),
        spc_date_string=getattr(INPUT_ARG_OBJECT, SPC_DATE_ARG_NAME),
        top_radar_dir_name_tarred=getattr(
            INPUT_ARG_OBJECT, TARRED_RADAR_DIR_ARG_NAME),
        top_radar_dir_name=getattr(INPUT_ARG_OBJECT, RADAR_DIR_ARG_NAME),
        top_output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME),
        peakedness_neigh_metres=getattr(
            INPUT_ARG_OBJECT, PEAKEDNESS_NEIGH_ARG_NAME),
        max_peakedness_height_m_asl=getattr(
            INPUT_ARG_OBJECT, MAX_PEAKEDNESS_HEIGHT_ARG_NAME),
        min_echo_top_m_asl=getattr(INPUT_ARG_OBJECT, MIN_ECHO_TOP_ARG_NAME),
        echo_top_level_dbz=getattr(INPUT_ARG_OBJECT, ECHO_TOP_LEVEL_ARG_NAME),
        min_composite_refl_criterion1_dbz=getattr(
            INPUT_ARG_OBJECT, MIN_COMPOSITE_REFL_CRITERION1_ARG_NAME),
        min_composite_refl_criterion5_dbz=getattr(
            INPUT_ARG_OBJECT, MIN_COMPOSITE_REFL_CRITERION5_ARG_NAME),
        min_composite_refl_aml_dbz=getattr(
            INPUT_ARG_OBJECT, MIN_COMPOSITE_REFL_AML_ARG_NAME)
    )
