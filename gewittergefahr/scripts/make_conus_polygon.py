"""Creates simple polygon that outlines all land in the continental U.S.

How to use this script:

[1] Download shapefile from this webpage:
    https://github.com/matplotlib/basemap/blob/master/examples/st99_d00.shp

[2] Download the corresponding .dbf and .shx files and put them in the same
    directory.

[3] Call this script from the command line:

python make_conus_polygon.py \
--input_shapefile_name="blahblahblah" \
--output_file_name="blahblah"
"""

import os
import errno
import argparse
import shapefile
import numpy
import netCDF4
import shapely.geometry
from shapely.ops import cascaded_union

SHAPEFILE_NAME_KEY = 'NAME'
SHAPEFILE_PERIMETER_KEY = 'PERIMETER'

NETCDF_VERTEX_DIMENSION_KEY = 'vertex'
NETCDF_LATITUDES_KEY = 'latitudes_deg'
NETCDF_LONGITUDES_KEY = 'longitudes_deg'

STATES_TO_EXCLUDE = ['Alaska', 'Hawaii', 'Puerto Rico']

INPUT_FILE_ARG_NAME = 'input_shapefile_name'
OUTPUT_FILE_ARG_NAME = 'output_file_name'

INPUT_FILE_HELP_STRING = (
    'Path to input file on local machine.  This should be the same as '
    'https://github.com/matplotlib/basemap/blob/master/examples/st99_d00.shp, '
    'and the corresponding .dbf and .shx files should be in the same directory.'
)

OUTPUT_FILE_HELP_STRING = (
    'Path to output file (also on local machine).  The polygon will be saved '
    'here in NetCDF format.'
)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + INPUT_FILE_ARG_NAME, type=str, required=True,
    help=INPUT_FILE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_FILE_ARG_NAME, type=str, required=True,
    help=OUTPUT_FILE_HELP_STRING)


def _run(shapefile_name, output_file_name):
    """Creates simple polygon that outlines all land in the continental U.S.

    This is effectively the main method.

    :param shapefile_name: See documentation at top of file.
    :param output_file_name: Same.
    """

    print('Reading data from: "{0:s}"...'.format(shapefile_name))
    shapefile_handle = shapefile.Reader(shapefile_name)

    state_names = []
    perimeters = []
    polygon_objects_latlng_deg = []

    for this_record_object in shapefile_handle.iterShapeRecords():
        this_record_dict = this_record_object.record.as_dict()
        this_state_name = this_record_dict[SHAPEFILE_NAME_KEY]

        # Skip states outside the continental U.S.
        if this_state_name in STATES_TO_EXCLUDE:
            continue

        # Find perimeter of this region.  Each region is a simple polygon, so
        # some states, like those with islands, are defined by multiple regions.
        # However, I don't want to keep the islands, because I want only the
        # continental U.S.
        #
        # (Okay, a few islands are in the Great Lakes instead of the ocean, so I
        # should technically keep them, but they're small enough that I don't
        # care).
        #
        # In general, for states with multiple polygons, the one with the
        # largest perimeter is the mainland part.  The only exception is
        # Michigan, which ruins everything by having the UP.  So for Michigan I
        # keep the two largest regions, but for every other state I keep the one
        # largest region.  For Michigan the two largest regions are those with a
        # perimeter > 5 (I don't know what the units are, but it doesn't really
        # matter).

        this_perimeter = this_record_dict[SHAPEFILE_PERIMETER_KEY]

        if this_state_name in state_names:
            i = state_names.index(this_state_name)

            # If this region is the largest for the given state, keep it.
            keep_region = this_perimeter > perimeters[i]

            # Deal with Michigan (which ruins everything).
            in_mainland_michigan = (
                this_state_name == 'Michigan' and this_perimeter > 5
            )
            keep_region = keep_region or in_mainland_michigan

            if not keep_region:
                continue

            # If the state is not Michigan, delete previous largest region for
            # state.
            delete_old_region = (
                not in_mainland_michigan or
                (in_mainland_michigan and perimeters[i] < 5)
            )

            if delete_old_region:
                del state_names[i]
                del perimeters[i]
                del polygon_objects_latlng_deg[i]

        state_names.append(this_state_name)
        perimeters.append(this_perimeter)
        polygon_objects_latlng_deg.append(
            shapely.geometry.Polygon(shell=this_record_object.shape.points)
        )

    # Merge polygons into one.
    main_polygon_object_latlng_deg = cascaded_union(polygon_objects_latlng_deg)
    main_latitudes_deg = numpy.array(
        main_polygon_object_latlng_deg.exterior.xy[1]
    )
    main_longitudes_deg = numpy.array(
        main_polygon_object_latlng_deg.exterior.xy[0]
    )
    main_longitudes_deg[main_longitudes_deg < 0] += 360.

    # Write the one polygon to NetCDF.
    print('Writing the one polygon to: "{0:s}"...'.format(output_file_name))
    output_dir_name = os.path.dirname(output_file_name)

    try:
        os.makedirs(output_dir_name)
    except OSError as this_error:
        if this_error.errno == errno.EEXIST and os.path.isdir(output_dir_name):
            pass
        else:
            raise

    dataset_object = netCDF4.Dataset(
        output_file_name, 'w', format='NETCDF3_64BIT_OFFSET'
    )

    num_points = len(main_latitudes_deg)
    dataset_object.createDimension(NETCDF_VERTEX_DIMENSION_KEY, num_points)

    dataset_object.createVariable(
        NETCDF_LATITUDES_KEY, datatype=numpy.float32,
        dimensions=NETCDF_VERTEX_DIMENSION_KEY
    )
    dataset_object.variables[NETCDF_LATITUDES_KEY][:] = main_latitudes_deg

    dataset_object.createVariable(
        NETCDF_LONGITUDES_KEY, datatype=numpy.float32,
        dimensions=NETCDF_VERTEX_DIMENSION_KEY
    )
    dataset_object.variables[NETCDF_LONGITUDES_KEY][:] = main_longitudes_deg

    dataset_object.close()


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        shapefile_name=getattr(INPUT_ARG_OBJECT, INPUT_FILE_ARG_NAME),
        output_file_name=getattr(INPUT_ARG_OBJECT, OUTPUT_FILE_ARG_NAME)
    )
