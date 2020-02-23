"""Links each NWS tornado warning to nearest storm."""

import copy
import pickle
import argparse
import numpy
import shapely.geometry
from gewittergefahr.gg_io import storm_tracking_io as tracking_io
from gewittergefahr.gg_utils import storm_tracking_utils as tracking_utils
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import time_periods
from gewittergefahr.gg_utils import echo_top_tracking
from gewittergefahr.gg_utils import polygons
from gewittergefahr.gg_utils import geodetic_utils
from gewittergefahr.gg_utils import projections
from gewittergefahr.gg_utils import linkage
from gewittergefahr.gg_utils import error_checking
from gewittergefahr.nature2019 import convert_warning_polygons

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

NUM_SECONDS_PER_DAY = 86400
DUMMY_TRACKING_SCALE_METRES2 = echo_top_tracking.DUMMY_TRACKING_SCALE_METRES2

WARNING_START_TIME_KEY = convert_warning_polygons.START_TIME_COLUMN
WARNING_END_TIME_KEY = convert_warning_polygons.END_TIME_COLUMN
WARNING_LATLNG_POLYGON_KEY = convert_warning_polygons.POLYGON_COLUMN
WARNING_XY_POLYGON_KEY = 'polygon_object_xy'

# TODO(thunderhoser): Still need output file.

WARNING_FILE_ARG_NAME = 'input_warning_file_name'
TRACKING_DIR_ARG_NAME = 'input_tracking_dir_name'
SPC_DATE_ARG_NAME = 'spc_date_string'
MAX_DISTANCE_ARG_NAME = 'max_distance_metres'
MIN_LIFETIME_FRACTION_ARG_NAME = 'min_lifetime_fraction'

WARNING_FILE_HELP_STRING = (
    'Path to Pickle file with tornado warnings (created by '
    'convert_warning_polygons.py).'
)
TRACKING_DIR_HELP_STRING = (
    'Name of top-level tracking directory.  Files therein will be found by '
    '`storm_tracking_io.find_processed_files_one_spc_date` and read by '
    '`storm_tracking_io.read_processed_file`.'
)
SPC_DATE_HELP_STRING = (
    'SPC date (format "yyyymmdd").  This script will link only warnings that '
    '*begin* on the given SPC date.'
)
MAX_DISTANCE_HELP_STRING = (
    'Max linkage distance.  Will link each warning to the nearest storm, as '
    'long as the storm''s mean distance outside the warning polygon does not '
    'exceed this value.'
)
MIN_LIFETIME_FRACTION_HELP_STRING = (
    'Minimum lifetime fraction.  Will link each warning to the nearest storm, '
    'as long as the storm is in existence for at least this fraction (range '
    '0...1) of the warning period.'
)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + WARNING_FILE_ARG_NAME, type=str, required=True,
    help=WARNING_FILE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + TRACKING_DIR_ARG_NAME, type=str, required=True,
    help=TRACKING_DIR_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + SPC_DATE_ARG_NAME, type=str, required=True,
    help=SPC_DATE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + MAX_DISTANCE_ARG_NAME, type=float, required=False, default=5000.,
    help=MAX_DISTANCE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + MIN_LIFETIME_FRACTION_ARG_NAME, type=float, required=False,
    default=0.5, help=MIN_LIFETIME_FRACTION_HELP_STRING
)


def _remove_far_away_storms(warning_polygon_object_latlng, storm_object_table):
    """Removes storms that are far away from a warning polygon.

    :param warning_polygon_object_latlng: See doc for `_link_one_warning`.
    :param storm_object_table: Same.
    :return: storm_object_table: Same as input but with fewer rows.
    """

    this_vertex_dict = polygons.polygon_object_to_vertex_arrays(
        warning_polygon_object_latlng
    )
    warning_latitudes_deg = this_vertex_dict[polygons.EXTERIOR_Y_COLUMN]
    warning_longitudes_deg = this_vertex_dict[polygons.EXTERIOR_X_COLUMN]

    unique_primary_id_strings = numpy.unique(
        storm_object_table[tracking_utils.PRIMARY_ID_COLUMN].values
    )
    good_indices = []

    for i in range(len(unique_primary_id_strings)):
        these_rows = numpy.where(
            storm_object_table[tracking_utils.PRIMARY_ID_COLUMN].values ==
            unique_primary_id_strings[i]
        )[0]

        these_latitudes_deg = storm_object_table[
            tracking_utils.CENTROID_LATITUDE_COLUMN
        ].values[these_rows]

        these_longitudes_deg = storm_object_table[
            tracking_utils.CENTROID_LONGITUDE_COLUMN
        ].values[these_rows]

        these_latitude_flags = numpy.logical_and(
            these_latitudes_deg >= numpy.min(warning_latitudes_deg) - 1.,
            these_latitudes_deg <= numpy.max(warning_latitudes_deg) + 1.
        )
        these_longitude_flags = numpy.logical_and(
            these_longitudes_deg >= numpy.min(warning_longitudes_deg) - 1.,
            these_longitudes_deg <= numpy.max(warning_longitudes_deg) + 1.
        )
        these_coord_flags = numpy.logical_and(
            these_latitude_flags, these_longitude_flags
        )

        if not numpy.any(these_coord_flags):
            continue

        good_indices.append(i)

    unique_primary_id_strings = [
        unique_primary_id_strings[k] for k in good_indices
    ]

    return storm_object_table.loc[
        storm_object_table[tracking_utils.PRIMARY_ID_COLUMN].isin(
            unique_primary_id_strings
        )
    ]


def _find_one_distance(storm_x_vertices_metres, storm_y_vertices_metres,
                       warning_polygon_object_xy):
    """Finds distance between one storm object and one warning.

    V = number of vertices in storm outline

    :param storm_x_vertices_metres: length-V numpy array of x-coordinates.
    :param storm_y_vertices_metres: length-V numpy array of y-coordinates.
    :param warning_polygon_object_xy: Polygon (instance of
        `shapely.geometry.Polygon`) with x-y coordinates of warning boundary.
    :return: distance_metres: Distance between storm object and warning (minimum
        distance to polygon interior over all storm vertices).
    """

    num_vertices = len(storm_x_vertices_metres)
    distance_metres = numpy.inf

    for k in range(num_vertices):
        this_flag = polygons.point_in_or_on_polygon(
            polygon_object=warning_polygon_object_xy,
            query_x_coordinate=storm_x_vertices_metres[k],
            query_y_coordinate=storm_y_vertices_metres[k]
        )

        if this_flag:
            return 0.

        this_point_object = shapely.geometry.Point(
            storm_x_vertices_metres[k], storm_y_vertices_metres[k]
        )

        distance_metres = numpy.minimum(
            distance_metres,
            warning_polygon_object_xy.exterior.distance(this_point_object)
        )

    return distance_metres


def _link_one_warning(warning_table, storm_object_table, max_distance_metres,
                      min_lifetime_fraction):
    """Links one warning to nearest storm.

    :param warning_table: pandas DataFrame with one row and the following
        columns.
    warning_table.start_time_unix_sec: Start time.
    warning_table.end_time_unix_sec: End time.
    warning_table.polygon_object_latlng: Polygon (instance of
        `shapely.geometry.Polygon`) with lat-long coordinates of warning
        boundary.
    warning_table.polygon_object_xy: Polygon (instance of
        `shapely.geometry.Polygon`) with x-y coordinates of warning boundary.

    :param storm_object_table: pandas DataFrame returned by
        `storm_tracking_io.read_file`.
    :param max_distance_metres: See documentation at top of file.
    :param min_lifetime_fraction: Same.
    :return: secondary_id_strings: 1-D list of secondary IDs for storms to which
        warning is linked.  If warning is not linked to a storm, this is empty.
    """

    warning_start_time_unix_sec = (
        warning_table[WARNING_START_TIME_KEY].values[0]
    )
    warning_end_time_unix_sec = warning_table[WARNING_END_TIME_KEY].values[0]
    warning_polygon_object_xy = warning_table[WARNING_XY_POLYGON_KEY].values[0]

    orig_num_storm_objects = len(storm_object_table.index)
    storm_object_table = linkage._filter_storms_by_time(
        storm_object_table=storm_object_table,
        max_start_time_unix_sec=warning_end_time_unix_sec + 720,
        min_end_time_unix_sec=warning_start_time_unix_sec - 720
    )
    num_storm_objects = len(storm_object_table.index)

    print('Filtering by time removed {0:d} of {1:d} storm objects.'.format(
        orig_num_storm_objects - num_storm_objects, orig_num_storm_objects
    ))

    orig_num_storm_objects = num_storm_objects + 0
    storm_object_table = _remove_far_away_storms(
        warning_polygon_object_latlng=
        warning_table[WARNING_LATLNG_POLYGON_KEY].values[0],
        storm_object_table=storm_object_table
    )
    num_storm_objects = len(storm_object_table.index)

    print('Filtering by distance removed {0:d} of {1:d} storm objects.'.format(
        orig_num_storm_objects - num_storm_objects, orig_num_storm_objects
    ))

    warning_times_unix_sec = time_periods.range_and_interval_to_list(
        start_time_unix_sec=warning_start_time_unix_sec,
        end_time_unix_sec=warning_end_time_unix_sec,
        time_interval_sec=60, include_endpoint=True
    )

    unique_sec_id_strings = numpy.unique(
        storm_object_table[tracking_utils.SECONDARY_ID_COLUMN].values
    )

    num_sec_id_strings = len(unique_sec_id_strings)
    num_warning_times = len(warning_times_unix_sec)
    distance_matrix_metres = numpy.full(
        (num_sec_id_strings, num_warning_times), numpy.nan
    )

    for j in range(num_warning_times):
        this_interp_vertex_table = linkage._interp_storms_in_time(
            storm_object_table=storm_object_table,
            target_time_unix_sec=warning_times_unix_sec[j],
            max_time_before_start_sec=180, max_time_after_end_sec=180
        )

        for i in range(num_sec_id_strings):
            these_indices = numpy.where(
                this_interp_vertex_table[
                    tracking_utils.SECONDARY_ID_COLUMN].values
                == unique_sec_id_strings[i]
            )[0]

            if len(these_indices) == 0:
                continue

            distance_matrix_metres[i, j] = _find_one_distance(
                storm_x_vertices_metres=
                this_interp_vertex_table[linkage.STORM_VERTEX_X_COLUMN].values,
                storm_y_vertices_metres=
                this_interp_vertex_table[linkage.STORM_VERTEX_Y_COLUMN].values,
                warning_polygon_object_xy=warning_polygon_object_xy
            )

    lifetime_fractions = (
        1. - numpy.mean(numpy.isnan(distance_matrix_metres), axis=1)
    )
    bad_indices = numpy.where(lifetime_fractions < min_lifetime_fraction)[0]
    distance_matrix_metres[bad_indices, ...] = numpy.inf

    mean_distances_metres = numpy.nanmean(distance_matrix_metres, axis=1)
    good_indices = numpy.where(mean_distances_metres <= max_distance_metres)[0]

    print((
        'Linked warning to {0:d} storms (distances in metres printed below):'
        '\n{1:s}'
    ).format(
        len(good_indices), str(mean_distances_metres[good_indices])
    ))

    return [unique_sec_id_strings[k] for k in good_indices]


def _run(warning_file_name, top_tracking_dir_name, spc_date_string,
         max_distance_metres, min_lifetime_fraction):
    """Links each NWS tornado warning to nearest storm.

    This is effectively the main method.

    :param warning_file_name: See documentation at top of file.
    :param top_tracking_dir_name: Same.
    :param spc_date_string: Same.
    :param max_distance_metres: Same.
    :param min_lifetime_fraction: Same.
    """

    error_checking.assert_is_greater(max_distance_metres, 0.)
    error_checking.assert_is_greater(min_lifetime_fraction, 0.)
    error_checking.assert_is_leq(min_lifetime_fraction, 1.)

    print('Reading warnings from: "{0:s}"...'.format(warning_file_name))
    this_file_handle = open(warning_file_name, 'rb')
    warning_table = pickle.load(this_file_handle)
    this_file_handle.close()

    date_start_time_unix_sec = (
        time_conversion.get_start_of_spc_date(spc_date_string)
    )
    date_end_time_unix_sec = (
        time_conversion.get_end_of_spc_date(spc_date_string)
    )
    warning_table = warning_table.loc[
        (warning_table[WARNING_START_TIME_KEY] >= date_start_time_unix_sec) &
        (warning_table[WARNING_END_TIME_KEY] >= date_end_time_unix_sec)
    ]
    num_warnings = len(warning_table.index)

    print('Number of warnings beginning on SPC date "{0:s}" = {1:d}'.format(
        spc_date_string, num_warnings
    ))

    tracking_file_names = []

    for i in [-1, 0, 1]:
        this_spc_date_string = time_conversion.time_to_spc_date_string(
            date_start_time_unix_sec + i * NUM_SECONDS_PER_DAY
        )

        tracking_file_names += tracking_io.find_files_one_spc_date(
            top_tracking_dir_name=top_tracking_dir_name,
            tracking_scale_metres2=DUMMY_TRACKING_SCALE_METRES2,
            source_name=tracking_utils.SEGMOTION_NAME,
            spc_date_string=this_spc_date_string,
            raise_error_if_missing=i == 0
        )[0]

    print(SEPARATOR_STRING)
    storm_object_table = tracking_io.read_many_files(tracking_file_names)
    print(SEPARATOR_STRING)

    global_centroid_lat_deg, global_centroid_lng_deg = (
        geodetic_utils.get_latlng_centroid(
            latitudes_deg=
            storm_object_table[tracking_utils.CENTROID_LATITUDE_COLUMN].values,
            longitudes_deg=
            storm_object_table[tracking_utils.CENTROID_LONGITUDE_COLUMN].values
        )
    )

    projection_object = projections.init_azimuthal_equidistant_projection(
        central_latitude_deg=global_centroid_lat_deg,
        central_longitude_deg=global_centroid_lng_deg
    )

    storm_object_table = linkage._project_storms_latlng_to_xy(
        storm_object_table=storm_object_table,
        projection_object=projection_object
    )

    warning_polygon_objects_xy = [None] * num_warnings

    for k in range(num_warnings):
        warning_polygon_objects_xy[k] = polygons.project_latlng_to_xy(
            polygon_object_latlng=
            warning_table[WARNING_LATLNG_POLYGON_KEY].values[k],
            projection_object=projection_object
        )[0]

    warning_table = warning_table.assign(**{
        WARNING_XY_POLYGON_KEY: warning_polygon_objects_xy
    })

    for k in range(num_warnings):
        these_sec_id_strings = _link_one_warning(
            warning_table=warning_table.iloc[[k]],
            storm_object_table=copy.deepcopy(storm_object_table),
            max_distance_metres=max_distance_metres,
            min_lifetime_fraction=min_lifetime_fraction
        )

        print('\n')


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        warning_file_name=getattr(INPUT_ARG_OBJECT, WARNING_FILE_ARG_NAME),
        top_tracking_dir_name=getattr(INPUT_ARG_OBJECT, TRACKING_DIR_ARG_NAME),
        spc_date_string=getattr(INPUT_ARG_OBJECT, SPC_DATE_ARG_NAME),
        max_distance_metres=getattr(INPUT_ARG_OBJECT, MAX_DISTANCE_ARG_NAME),
        min_lifetime_fraction=getattr(
            INPUT_ARG_OBJECT, MIN_LIFETIME_FRACTION_ARG_NAME
        )
    )
