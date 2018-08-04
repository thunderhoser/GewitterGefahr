"""Computes area of each storm object."""

import pickle
import argparse
import numpy
from gewittergefahr.gg_io import storm_tracking_io as tracking_io
from gewittergefahr.gg_utils import polygons
from gewittergefahr.gg_utils import storm_tracking_utils as tracking_utils
from gewittergefahr.gg_utils import file_system_utils

DUMMY_TRACKING_SCALE_METRES2 = 314159265
SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

STORM_IDS_KEY = 'storm_ids'
STORM_TIMES_KEY = 'storm_times_unix_sec'
STORM_AREAS_KEY = 'storm_areas_metres2'

TRACKING_DIR_ARG_NAME = 'input_tracking_dir_name'
SPC_DATE_ARG_NAME = 'spc_date_string'
OUTPUT_FILE_ARG_NAME = 'output_pickle_file_name'

TRACKING_DIR_HELP_STRING = (
    'Name of top-level directory with storm tracks.  Files therein will be '
    'found by `storm_tracking_io.find_processed_file` and read by '
    '`storm_tracking_io.read_processed_file`.')
SPC_DATE_HELP_STRING = (
    'SPC (Storm Prediction Center) date in format "yyyymmdd".  Area will be '
    'computed for each storm object on this date.')
OUTPUT_FILE_HELP_STRING = (
    'Path to output file.  Will contain storm ID, valid time, and area for each'
    ' storm object.')

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + TRACKING_DIR_ARG_NAME, type=str, required=True,
    help=TRACKING_DIR_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + SPC_DATE_ARG_NAME, type=str, required=True,
    help=SPC_DATE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_FILE_ARG_NAME, type=str, required=True,
    help=OUTPUT_FILE_HELP_STRING)


def _run(top_tracking_dir_name, spc_date_string, output_pickle_file_name):
    """Computes area of each storm object.

    This is effectively the main method.

    :param top_tracking_dir_name: See documentation at top of file.
    :param spc_date_string: Same.
    :param output_pickle_file_name: Same.
    """

    file_system_utils.mkdir_recursive_if_necessary(
        file_name=output_pickle_file_name)

    tracking_file_names, _ = tracking_io.find_processed_files_one_spc_date(
        top_processed_dir_name=top_tracking_dir_name,
        tracking_scale_metres2=DUMMY_TRACKING_SCALE_METRES2,
        data_source=tracking_utils.SEGMOTION_SOURCE_ID,
        spc_date_string=spc_date_string)

    storm_object_table = tracking_io.read_many_processed_files(
        tracking_file_names)
    print SEPARATOR_STRING

    num_storm_objects = len(storm_object_table.index)
    storm_areas_metres2 = numpy.full(num_storm_objects, numpy.nan)

    for i in range(num_storm_objects):
        if numpy.mod(i, 100) == 0:
            print (
                'Have computed area for {0:d} of {1:d} storm objects...'
            ).format(i, num_storm_objects)

        this_polygon_object_latlng = storm_object_table[
            tracking_utils.POLYGON_OBJECT_LATLNG_COLUMN].values[i]
        storm_areas_metres2[i] = polygons.project_latlng_to_xy(
            polygon_object_latlng=this_polygon_object_latlng)[0].area

    print 'Have computed area for all {0:d} storm objects!'.format(
        num_storm_objects)
    print SEPARATOR_STRING

    storm_areas_km2 = storm_areas_metres2 * 1e-6
    print (
        'Minimum area = {0:.1f} km^2 ... max = {1:.1f} km^2 ... '
        'mean = {2:.1f} km^2 ... median = {3:.1f} km^2'
    ).format(numpy.min(storm_areas_km2), numpy.max(storm_areas_km2),
             numpy.mean(storm_areas_km2), numpy.median(storm_areas_km2))

    storm_area_dict = {
        STORM_IDS_KEY:
            storm_object_table[tracking_utils.STORM_ID_COLUMN].values.tolist(),
        STORM_TIMES_KEY: storm_object_table[tracking_utils.TIME_COLUMN].values,
        STORM_AREAS_KEY: storm_areas_metres2
    }

    print 'Writing storm areas to: "{0:s}"...'.format(output_pickle_file_name)
    pickle_file_handle = open(output_pickle_file_name, 'wb')
    pickle.dump(storm_area_dict, pickle_file_handle)
    pickle_file_handle.close()


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        top_tracking_dir_name=getattr(INPUT_ARG_OBJECT, TRACKING_DIR_ARG_NAME),
        spc_date_string=getattr(INPUT_ARG_OBJECT, SPC_DATE_ARG_NAME),
        output_pickle_file_name=getattr(INPUT_ARG_OBJECT, OUTPUT_FILE_ARG_NAME))
