"""Converts SPC convective outlook to nicer file format."""

import pickle
import argparse
import shapefile
import numpy
import pandas
from gewittergefahr.gg_utils import polygons
from gewittergefahr.gg_utils import projections
from gewittergefahr.gg_utils import file_system_utils

RISK_TYPE_INDEX = 0
MARGINAL_RISK_ENUM = 3
SLIGHT_RISK_ENUM = 4
ENHANCED_RISK_ENUM = 5
MODERATE_RISK_ENUM = 6
HIGH_RISK_ENUM = 7

MARGINAL_RISK_STRING = 'marginal'
SLIGHT_RISK_STRING = 'slight'
ENHANCED_RISK_STRING = 'enhanced'
MODERATE_RISK_STRING = 'moderate'
HIGH_RISK_STRING = 'high'

RISK_TYPE_ENUM_TO_STRING = {
    MARGINAL_RISK_ENUM: MARGINAL_RISK_STRING,
    SLIGHT_RISK_ENUM: SLIGHT_RISK_STRING,
    ENHANCED_RISK_ENUM: ENHANCED_RISK_STRING,
    MODERATE_RISK_ENUM: MODERATE_RISK_STRING,
    HIGH_RISK_ENUM: HIGH_RISK_STRING
}

STANDARD_LATITUDES_DEG = numpy.array([33, 45], dtype=float)
CENTRAL_LONGITUDE_DEG = 0.
ELLIPSOID_NAME = projections.WGS84_NAME
FALSE_EASTING_METRES = 0.
FALSE_NORTHING_METRES = 0.

RISK_TYPE_COLUMN = 'risk_type_string'
POLYGON_COLUMN = 'polygon_object_latlng'

INPUT_FILE_ARG_NAME = 'input_shapefile_name'
OUTPUT_FILE_ARG_NAME = 'output_pickle_file_name'

INPUT_FILE_HELP_STRING = (
    'Path to input file.  This should be a shapefile with one convective '
    'outlook, such as those found here: '
    'https://www.spc.noaa.gov/cgi-bin-spc/getacrange.pl?date0=20180403&'
    'date1=20180403')

OUTPUT_FILE_HELP_STRING = (
    'Path to output file.  Risk polygons (slight, enhanced, moderate, etc.) '
    'will be saved here in a pandas DataFrame.')

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + INPUT_FILE_ARG_NAME, type=str, required=True,
    help=INPUT_FILE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_FILE_ARG_NAME, type=str, required=True,
    help=OUTPUT_FILE_HELP_STRING)


def _run(input_shapefile_name, output_pickle_file_name):
    """Converts SPC convective outlook to nicer file format.

    This is effectively the main method.

    :param input_shapefile_name: See documentation at top of file.
    :param output_pickle_file_name: Same.
    """

    projection_object = projections.init_lcc_projection(
        standard_latitudes_deg=STANDARD_LATITUDES_DEG,
        central_longitude_deg=CENTRAL_LONGITUDE_DEG,
        ellipsoid_name=ELLIPSOID_NAME)

    print 'Reading data from: "{0:s}"...'.format(input_shapefile_name)
    shapefile_handle = shapefile.Reader(input_shapefile_name)

    list_of_polygon_objects_latlng = []
    risk_type_strings = []

    for this_record_object in shapefile_handle.iterShapeRecords():
        # print this_record_object.record
        this_risk_type_enum = this_record_object.record[RISK_TYPE_INDEX]

        try:
            this_risk_type_string = RISK_TYPE_ENUM_TO_STRING[
                this_risk_type_enum]
        except KeyError:
            continue

        these_xy_tuples = this_record_object.shape.points
        this_num_vertices = len(these_xy_tuples)

        these_x_coords_metres = numpy.array([
            these_xy_tuples[k][0] for k in range(this_num_vertices)
        ])

        these_y_coords_metres = numpy.array([
            these_xy_tuples[k][1] for k in range(this_num_vertices)
        ])

        these_latitudes_deg, these_longitudes_deg = (
            projections.project_xy_to_latlng(
                x_coords_metres=these_x_coords_metres,
                y_coords_metres=these_y_coords_metres,
                projection_object=projection_object,
                false_easting_metres=FALSE_EASTING_METRES,
                false_northing_metres=FALSE_NORTHING_METRES)
        )

        this_polygon_object_latlng = (
            polygons.vertex_arrays_to_polygon_object(
                exterior_x_coords=these_longitudes_deg,
                exterior_y_coords=these_latitudes_deg)
        )

        risk_type_strings.append(this_risk_type_string)
        list_of_polygon_objects_latlng.append(this_polygon_object_latlng)

    outlook_dict = {
        RISK_TYPE_COLUMN: risk_type_strings,
        POLYGON_COLUMN: list_of_polygon_objects_latlng
    }

    outlook_table = pandas.DataFrame.from_dict(outlook_dict)
    print outlook_table

    print 'Writing outlook polygons to file: "{0:s}"...'.format(
        output_pickle_file_name)
    file_system_utils.mkdir_recursive_if_necessary(
        file_name=output_pickle_file_name)

    pickle_file_handle = open(output_pickle_file_name, 'wb')
    pickle.dump(outlook_table, pickle_file_handle)
    pickle_file_handle.close()


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        input_shapefile_name=getattr(INPUT_ARG_OBJECT, INPUT_FILE_ARG_NAME),
        output_pickle_file_name=getattr(INPUT_ARG_OBJECT, OUTPUT_FILE_ARG_NAME)
    )
