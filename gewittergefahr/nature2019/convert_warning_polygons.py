"""Converts tornado-warning polygons to nicer file format."""

import pickle
import argparse
import warnings
import shapefile
import numpy
import pandas
from gewittergefahr.gg_utils import polygons
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import longitude_conversion as lng_conversion
from gewittergefahr.gg_utils import file_system_utils

INPUT_TIME_FORMAT = '%Y-%m-%d-%H%M'
SHAPEFILE_TIME_FORMAT = '%Y%m%d%H%M'

TIME_ISSUED_INDEX = 1
TIME_EXPIRED_INDEX = 2
EVENT_TYPE_INDEX = 5
COUNTY_OR_WARNING_INDEX = 6

TORNADO_TYPE_STRING = 'TO'
WARNING_TYPE_STRING = 'P'

AREA_TYPE_KEY = 'type'
COORDINATES_KEY = 'coordinates'

POLYGON_TYPE_STRING = 'Polygon'
MULTI_POLYGON_TYPE_STRING = 'MultiPolygon'
VALID_AREA_TYPE_STRINGS = [POLYGON_TYPE_STRING, MULTI_POLYGON_TYPE_STRING]

START_TIME_COLUMN = 'start_time_unix_sec'
END_TIME_COLUMN = 'end_time_unix_sec'
POLYGON_COLUMN = 'polygon_object_latlng'

INPUT_FILE_ARG_NAME = 'input_shapefile_name'
FIRST_TIME_ARG_NAME = 'first_time_string'
LAST_TIME_ARG_NAME = 'last_time_string'
OUTPUT_FILE_ARG_NAME = 'output_pickle_file_name'

INPUT_FILE_HELP_STRING = (
    'Path to input file.  This should be a shapefile with warning polygons, one'
    ' of the "pre-generated zip files" here: '
    'https://mesonet.agron.iastate.edu/request/gis/watchwarn.phtml')

TIME_HELP_STRING = (
    'Time (format "yyyy-mm-dd-HHMM").  This script will extract tornado '
    'warnings in the period `{0:s}`...`{1:s}`.'
).format(FIRST_TIME_ARG_NAME, LAST_TIME_ARG_NAME)

OUTPUT_FILE_HELP_STRING = (
    'Path to output file.  Warning polygons will be saved here in a pandas '
    'DataFrame.')

DEFAULT_INPUT_FILE_NAME = (
    '/localdata/ryan.lagerquist/nature_ml_paper/warning_polygons/raw/'
    'wwa_201801010000_201812312359.shp')

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + INPUT_FILE_ARG_NAME, type=str, required=False,
    default=DEFAULT_INPUT_FILE_NAME, help=INPUT_FILE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + FIRST_TIME_ARG_NAME, type=str, required=True, help=TIME_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + LAST_TIME_ARG_NAME, type=str, required=True, help=TIME_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_FILE_ARG_NAME, type=str, required=True,
    help=OUTPUT_FILE_HELP_STRING)


def _run(input_shapefile_name, first_time_string, last_time_string,
         output_pickle_file_name):
    """Converts tornado-warning polygons to nicer file format.

    This is effectively the main method.

    :param input_shapefile_name: See documentation at top of file.
    :param first_time_string: Same.
    :param last_time_string: Same.
    :param output_pickle_file_name: Same.
    """

    first_time_unix_sec = time_conversion.string_to_unix_sec(
        first_time_string, INPUT_TIME_FORMAT)
    last_time_unix_sec = time_conversion.string_to_unix_sec(
        last_time_string, INPUT_TIME_FORMAT)

    print 'Reading data from: "{0:s}"...'.format(input_shapefile_name)
    shapefile_handle = shapefile.Reader(input_shapefile_name)

    list_of_polygon_objects_latlng = []
    start_times_unix_sec = []
    end_times_unix_sec = []

    for this_record_object in shapefile_handle.iterShapeRecords():
        if this_record_object.record[EVENT_TYPE_INDEX] != TORNADO_TYPE_STRING:
            continue

        if this_record_object.record[
                COUNTY_OR_WARNING_INDEX] != WARNING_TYPE_STRING:
            continue

        this_start_time_unix_sec = time_conversion.string_to_unix_sec(
            this_record_object.record[TIME_ISSUED_INDEX], SHAPEFILE_TIME_FORMAT
        )
        if this_start_time_unix_sec > last_time_unix_sec:
            continue

        this_end_time_unix_sec = time_conversion.string_to_unix_sec(
            this_record_object.record[TIME_EXPIRED_INDEX], SHAPEFILE_TIME_FORMAT
        )
        if this_end_time_unix_sec < first_time_unix_sec:
            continue

        print this_record_object.record

        this_area_dict = this_record_object.shape.__geo_interface__

        if this_area_dict[AREA_TYPE_KEY] not in VALID_AREA_TYPE_STRINGS:
            warning_string = (
                '\n{0:s}\nValid area types (listed above) do not include '
                '"{1:s}".'
            ).format(
                str(VALID_AREA_TYPE_STRINGS), this_area_dict[AREA_TYPE_KEY]
            )

            warnings.warn(warning_string)
            continue

        this_coords_object = this_area_dict[COORDINATES_KEY]
        if isinstance(this_coords_object, tuple):
            these_latlng_tuples = [this_coords_object[0]]
        else:
            these_latlng_tuples = this_coords_object

        for this_latlng_tuple in these_latlng_tuples:
            if isinstance(this_latlng_tuple, list):
                this_latlng_tuple = this_latlng_tuple[0]

            this_num_vertices = len(this_latlng_tuple)
            these_latitudes_deg = numpy.array(
                [this_latlng_tuple[k][1] for k in range(this_num_vertices)]
            )
            these_longitudes_deg = numpy.array(
                [this_latlng_tuple[k][0] for k in range(this_num_vertices)]
            )

            these_longitudes_deg = lng_conversion.convert_lng_positive_in_west(
                longitudes_deg=these_longitudes_deg, allow_nan=False)

            this_polygon_object_latlng = (
                polygons.vertex_arrays_to_polygon_object(
                    exterior_x_coords=these_longitudes_deg,
                    exterior_y_coords=these_latitudes_deg)
            )

            start_times_unix_sec.append(this_start_time_unix_sec)
            end_times_unix_sec.append(this_end_time_unix_sec)
            list_of_polygon_objects_latlng.append(this_polygon_object_latlng)

    warning_dict = {
        START_TIME_COLUMN: start_times_unix_sec,
        END_TIME_COLUMN: end_times_unix_sec,
        POLYGON_COLUMN: list_of_polygon_objects_latlng
    }

    warning_table = pandas.DataFrame.from_dict(warning_dict)
    # print warning_table

    print 'Writing warnings to file: "{0:s}"...'.format(output_pickle_file_name)
    file_system_utils.mkdir_recursive_if_necessary(
        file_name=output_pickle_file_name)

    pickle_file_handle = open(output_pickle_file_name, 'wb')
    pickle.dump(warning_table, pickle_file_handle)
    pickle_file_handle.close()


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        input_shapefile_name=getattr(INPUT_ARG_OBJECT, INPUT_FILE_ARG_NAME),
        first_time_string=getattr(INPUT_ARG_OBJECT, FIRST_TIME_ARG_NAME),
        last_time_string=getattr(INPUT_ARG_OBJECT, LAST_TIME_ARG_NAME),
        output_pickle_file_name=getattr(INPUT_ARG_OBJECT, OUTPUT_FILE_ARG_NAME)
    )
