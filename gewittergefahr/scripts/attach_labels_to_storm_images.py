"""Attaches labels (target variables) to storm-centered radar images."""

import argparse
import numpy
from gewittergefahr.gg_utils import radar_utils
from gewittergefahr.gg_utils import gridrad_utils
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import link_events_to_storms as events2storms
from gewittergefahr.gg_utils import number_rounding as rounder
from gewittergefahr.gg_utils import labels
from gewittergefahr.deep_learning import storm_images

EMPTY_STRING = 'None'

SPC_DATE_ARG_NAME = 'spc_date_string'
STORM_IMAGE_DIR_ARG_NAME = 'inout_storm_image_dir_name'
WIND_LABEL_DIR_ARG_NAME = 'input_wind_label_dir_name'
TORNADO_LABEL_DIR_ARG_NAME = 'input_tornado_label_dir_name'
RADAR_SOURCE_ARG_NAME = 'radar_source'

SPC_DATE_HELP_STRING = (
    'SPC (Storm Prediction Center) date in format "yyyymmdd".  Labels will be '
    'added to the file for each time step, radar field, and radar height on '
    'this date.  Fields and heights are determined by defaults in '
    'storm_images.py.')
STORM_IMAGE_DIR_HELP_STRING = (
    'Name of top-level directory with storm-image files (one per time step, '
    'radar field, and radar height; readable by'
    ' `storm_images.read_storm_images`).')
WIND_LABEL_DIR_HELP_STRING = (
    'Name of top-level directory with files containing wind-speed labels (one '
    'per SPC date, readable by `labels.read_wind_speed_labels`).  If empty, '
    'this script will attach only tornado labels.')
TORNADO_LABEL_DIR_HELP_STRING = (
    'Name of top-level directory with files containing tornado labels (one per '
    'SPC date, readable by `labels.read_tornado_labels`).  If empty, this '
    'script will attach only wind-speed labels.')
RADAR_SOURCE_HELP_STRING = (
    'Radar source.  Must be a string in the following list:\n{0:s}'
).format(radar_utils.DATA_SOURCE_IDS)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + SPC_DATE_ARG_NAME, type=str, required=True,
    help=SPC_DATE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + STORM_IMAGE_DIR_ARG_NAME, type=str, required=True,
    help=STORM_IMAGE_DIR_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + WIND_LABEL_DIR_ARG_NAME, type=str, required=False,
    default=EMPTY_STRING, help=WIND_LABEL_DIR_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + TORNADO_LABEL_DIR_ARG_NAME, type=str, required=False,
    default=EMPTY_STRING, help=TORNADO_LABEL_DIR_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + RADAR_SOURCE_ARG_NAME, type=str, required=True,
    help=RADAR_SOURCE_HELP_STRING)


def _run_attach_labels(
        spc_date_string, top_storm_image_dir_name, top_wind_label_dir_name,
        top_tornado_label_dir_name, radar_source):
    """Attaches labels (target variables) to storm-centered radar images.

    :param spc_date_string: See documentation at top of file.
    :param top_storm_image_dir_name: Same.
    :param top_wind_label_dir_name: Same.
    :param top_tornado_label_dir_name: Same.
    :param radar_source: Same.
    """

    radar_utils.check_data_source(radar_source)
    start_time_unix_sec = time_conversion.get_start_of_spc_date(spc_date_string)
    end_time_unix_sec = time_conversion.get_end_of_spc_date(spc_date_string)
    end_time_unix_sec = int(rounder.floor_to_nearest(
        end_time_unix_sec, storm_images.GRIDRAD_TIME_INTERVAL_SEC))

    if radar_source == radar_utils.GRIDRAD_SOURCE_ID:
        image_file_name_matrix, _ = storm_images.find_many_files_gridrad(
            top_directory_name=top_storm_image_dir_name,
            start_time_unix_sec=start_time_unix_sec,
            end_time_unix_sec=end_time_unix_sec,
            radar_field_names=storm_images.DEFAULT_GRIDRAD_FIELD_NAMES,
            radar_heights_m_asl=storm_images.DEFAULT_GRIDRAD_HEIGHTS_M_ASL,
            raise_error_if_missing=True)

        field_name_by_predictor, _ = (
            gridrad_utils.fields_and_refl_heights_to_pairs(
                field_names=storm_images.DEFAULT_GRIDRAD_FIELD_NAMES,
                heights_m_asl=storm_images.DEFAULT_GRIDRAD_HEIGHTS_M_ASL))

        num_times = image_file_name_matrix.shape[0]
        num_predictors = len(field_name_by_predictor)
        image_file_name_matrix = numpy.reshape(
            image_file_name_matrix, (num_times, num_predictors))

    else:
        image_file_name_matrix, _, _, _ = (
            storm_images.find_many_files_myrorss_or_mrms(
                top_directory_name=top_storm_image_dir_name,
                start_time_unix_sec=start_time_unix_sec,
                end_time_unix_sec=end_time_unix_sec, radar_source=radar_source,
                radar_field_names=storm_images.DEFAULT_MYRORSS_MRMS_FIELD_NAMES,
                raise_error_if_missing=True))

    if top_wind_label_dir_name == EMPTY_STRING:
        storm_to_winds_table = None
    else:
        wind_label_file_name = labels.find_label_file(
            top_directory_name=top_wind_label_dir_name,
            event_type_string=events2storms.WIND_EVENT_TYPE_STRING,
            spc_date_string=spc_date_string, raise_error_if_missing=True)

        print 'Reading labels from: "{0:s}"...'.format(wind_label_file_name)
        storm_to_winds_table = labels.read_wind_speed_labels(
            wind_label_file_name)

    if top_tornado_label_dir_name == EMPTY_STRING:
        storm_to_tornadoes_table = None
    else:
        tornado_label_file_name = labels.find_label_file(
            top_directory_name=top_tornado_label_dir_name,
            event_type_string=events2storms.TORNADO_EVENT_TYPE_STRING,
            spc_date_string=spc_date_string, raise_error_if_missing=True)

        print 'Reading labels from: "{0:s}"...'.format(tornado_label_file_name)
        storm_to_tornadoes_table = labels.read_tornado_labels(
            tornado_label_file_name)

    storm_images.attach_labels_to_storm_images(
        image_file_name_matrix, storm_to_winds_table=storm_to_winds_table,
        storm_to_tornadoes_table=storm_to_tornadoes_table)


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run_attach_labels(
        spc_date_string=getattr(INPUT_ARG_OBJECT, SPC_DATE_ARG_NAME),
        top_storm_image_dir_name=getattr(
            INPUT_ARG_OBJECT, STORM_IMAGE_DIR_ARG_NAME),
        top_wind_label_dir_name=getattr(
            INPUT_ARG_OBJECT, WIND_LABEL_DIR_ARG_NAME),
        top_tornado_label_dir_name=getattr(
            INPUT_ARG_OBJECT, TORNADO_LABEL_DIR_ARG_NAME),
        radar_source=getattr(INPUT_ARG_OBJECT, RADAR_SOURCE_ARG_NAME))
