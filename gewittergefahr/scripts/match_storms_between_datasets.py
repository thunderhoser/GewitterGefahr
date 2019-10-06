"""Matches storm objects between MYRORSS and GridRad datasets."""

import argparse
import numpy
from sklearn.metrics.pairwise import euclidean_distances
from gewittergefahr.gg_io import storm_tracking_io as tracking_io
from gewittergefahr.gg_utils import radar_utils
from gewittergefahr.gg_utils import projections
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import echo_top_tracking
from gewittergefahr.gg_utils import storm_tracking_utils as tracking_utils

TIME_FORMAT = '%Y-%m-%d-%H%M%S'
SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

TRACKING_SCALE_METRES2 = echo_top_tracking.DUMMY_TRACKING_SCALE_METRES2

MAX_TIME_DIFF_SECONDS = 180
VALID_DATASET_NAMES = [
    radar_utils.MYRORSS_SOURCE_ID, radar_utils.GRIDRAD_SOURCE_ID
]

MYRORSS_DIR_ARG_NAME = 'myrorss_tracking_dir_name'
GRIDRAD_DIR_ARG_NAME = 'gridrad_tracking_dir_name'
MAX_DISTANCE_ARG_NAME = 'max_distance_metres'
SOURCE_DATASET_ARG_NAME = 'source_dataset_name'
FIRST_DATE_ARG_NAME = 'first_spc_date_string'
LAST_DATE_ARG_NAME = 'last_spc_date_string'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

MYRORSS_DIR_HELP_STRING = (
    'Name of top-level directory with MYRORSS tracks.  Files therein will be '
    'found by `storm_tracking_io.find_file` and read by '
    '`storm_tracking_io.read_file`.')

GRIDRAD_DIR_HELP_STRING = (
    'Same as `{0:s}` but for GridRad.'
).format(MYRORSS_DIR_ARG_NAME)

MAX_DISTANCE_HELP_STRING = (
    'Max distance between two storm centers.  Objects at a greater distance '
    'cannot be matched.')

SOURCE_DATASET_HELP_STRING = (
    'Name of source dataset.  Storm objects in this dataset will be matched to '
    'those in the other (target) dataset.  Must be in the following list:'
    '\n{0:s}'
).format(str(VALID_DATASET_NAMES))

DATE_HELP_STRING = (
    'SPC date (format "yyyymmdd").  Storm objects will be matched for all dates'
    ' in the period `{0:s}`...`{1:s}`.'
).format(FIRST_DATE_ARG_NAME, LAST_DATE_ARG_NAME)

# TODO(thunderhoser): Update this help string.
OUTPUT_DIR_HELP_STRING = (
    'Name of output directory.  Files will be written here by ``, to exact '
    'locations determined by ``.')

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + MYRORSS_DIR_ARG_NAME, type=str, required=True,
    help=MYRORSS_DIR_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + GRIDRAD_DIR_ARG_NAME, type=str, required=True,
    help=GRIDRAD_DIR_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + MAX_DISTANCE_ARG_NAME, type=float, required=False, default=1e4,
    help=MAX_DISTANCE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + SOURCE_DATASET_ARG_NAME, type=str, required=True,
    help=SOURCE_DATASET_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + FIRST_DATE_ARG_NAME, type=str, required=True, help=DATE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + LAST_DATE_ARG_NAME, type=str, required=True, help=DATE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING)


def _match_one_time(source_time_unix_sec, target_times_unix_sec,
                    max_diff_seconds):
    """Matches one source time to nearest target time.

    :param source_time_unix_sec: Source time.
    :param target_times_unix_sec: 1-D numpy array of target times.
    :param max_diff_seconds: Max difference.
    :return: nearest_index: Index of nearest target time.  If
        `nearest_index == i`, the nearest target time is
        target_times_unix_sec[i].
    :raises: ValueError: if no target time is found within `max_diff_seconds` of
        `source_time_unix_sec`.
    """

    diffs_seconds = numpy.absolute(source_time_unix_sec - target_times_unix_sec)
    min_diff_seconds = numpy.min(diffs_seconds)
    nearest_index = numpy.argmin(diffs_seconds)

    if min_diff_seconds <= max_diff_seconds:
        return nearest_index

    source_time_string = time_conversion.unix_sec_to_string(
        source_time_unix_sec, TIME_FORMAT)
    nearest_target_time_string = time_conversion.unix_sec_to_string(
        target_times_unix_sec[nearest_index], TIME_FORMAT
    )

    error_string = (
        'Cannot find target time within {0:d} seconds of source time ({1:s}).  '
        'Nearest target time is {2:s}.'
    ).format(max_diff_seconds, source_time_string, nearest_target_time_string)

    raise ValueError(error_string)


def _match_all_times(source_times_unix_sec, target_times_unix_sec,
                     max_diff_seconds):
    """Matches each source time to nearest target time.

    S = number of source times

    :param source_times_unix_sec: length-S numpy array of source times.
    :param target_times_unix_sec: 1-D numpy array of target times.
    :param max_diff_seconds: Max difference between any source time and its
        nearest target time.
    :return: source_to_target_indices: length-S numpy array of indices.
        If source_to_target_indices[i] = j, the nearest target time to
        source_times_unix_sec[i] is target_times_unix_sec[j].
    """

    num_source_times = len(source_times_unix_sec)
    source_to_target_indices = numpy.full(num_source_times, -1, dtype=int)

    for i in range(num_source_times):
        if numpy.mod(i, 50) == 0:
            print((
                'Have matched {0:d} of {1:d} source times to nearest target '
                'time...'
            ).format(
                i, num_source_times
            ))

        source_to_target_indices[i] = _match_one_time(
            source_time_unix_sec=source_times_unix_sec[i],
            target_times_unix_sec=target_times_unix_sec,
            max_diff_seconds=max_diff_seconds)

    print('Have matched all {0:d} source times to nearest target time!'.format(
        num_source_times))

    return source_to_target_indices


def _match_locations_one_time(source_object_table, target_object_table,
                              max_distance_metres):
    """Matches storm locations at one time.

    :param source_object_table: pandas DataFrame, where each row is a storm
        object in the source dataset.  See `storm_tracking_io.write_file` for a
        list of expected columns.
    :param target_object_table: Same but for target dataset.
    :param max_distance_metres: Max distance for matching.
    :return: source_to_target_dict: Dictionary, where each key is a tuple with
        (source ID, source time) and each value is a list with [target ID,
        target time].  The IDs are strings, and the times are Unix seconds
        (integers).  For source objects with no match in the target dataset, the
        corresponding value is None (rather than a list).
    """

    # TODO(thunderhoser): Maybe use polygons here?

    num_source_objects = len(source_object_table.index)
    source_to_target_dict = dict()

    if num_source_objects == 0:
        return source_to_target_dict

    for i in range(num_source_objects):
        this_key = (
            source_object_table[tracking_utils.FULL_ID_COLUMN].values[i],
            source_object_table[tracking_utils.VALID_TIME_COLUMN].values[i]
        )

        source_to_target_dict[this_key] = None

    num_target_objects = len(target_object_table.index)
    if num_target_objects == 0:
        return source_to_target_dict

    # Create equidistant projection.
    all_latitudes_deg = numpy.concatenate((
        source_object_table[tracking_utils.CENTROID_LATITUDE_COLUMN].values,
        target_object_table[tracking_utils.CENTROID_LATITUDE_COLUMN].values
    ))
    all_longitudes_deg = numpy.concatenate((
        source_object_table[tracking_utils.CENTROID_LONGITUDE_COLUMN].values,
        target_object_table[tracking_utils.CENTROID_LONGITUDE_COLUMN].values
    ))
    projection_object = projections.init_azimuthal_equidistant_projection(
        central_latitude_deg=numpy.mean(all_latitudes_deg),
        central_longitude_deg=numpy.mean(all_longitudes_deg)
    )

    # Project storm centers from lat-long to x-y.
    source_x_coords_metres, source_y_coords_metres = (
        projections.project_latlng_to_xy(
            latitudes_deg=source_object_table[
                tracking_utils.CENTROID_LATITUDE_COLUMN].values,
            longitudes_deg=source_object_table[
                tracking_utils.CENTROID_LONGITUDE_COLUMN].values,
            projection_object=projection_object
        )
    )

    target_x_coords_metres, target_y_coords_metres = (
        projections.project_latlng_to_xy(
            latitudes_deg=target_object_table[
                tracking_utils.CENTROID_LATITUDE_COLUMN].values,
            longitudes_deg=target_object_table[
                tracking_utils.CENTROID_LONGITUDE_COLUMN].values,
            projection_object=projection_object
        )
    )

    # Find nearest target object to each source object.
    source_coord_matrix = numpy.transpose(numpy.vstack((
        source_x_coords_metres, source_y_coords_metres
    )))
    target_coord_matrix = numpy.transpose(numpy.vstack((
        target_x_coords_metres, target_y_coords_metres
    )))

    distance_matrix_metres2 = euclidean_distances(
        X=source_coord_matrix, Y=target_coord_matrix, squared=True)

    nearest_target_indices = numpy.argmin(distance_matrix_metres2, axis=1)
    source_indices = numpy.linspace(
        0, num_source_objects - 1, num=num_source_objects, dtype=int)

    min_distances_metres2 = distance_matrix_metres2[
        source_indices, nearest_target_indices
    ]
    bad_subindices = numpy.where(
        min_distances_metres2 > max_distance_metres ** 2
    )[0]
    nearest_target_indices[bad_subindices] = -1

    # Print results to command window.
    num_matched_source_objects = numpy.sum(nearest_target_indices >= 0)
    source_time_string = time_conversion.unix_sec_to_string(
        source_object_table[tracking_utils.VALID_TIME_COLUMN].values[0],
        TIME_FORMAT
    )

    print('Matched {0:d} of {1:d} source objects at {2:s}.'.format(
        num_matched_source_objects, num_source_objects, source_time_string
    ))

    # Fill dictionary.
    for i in range(num_source_objects):
        this_key = (
            source_object_table[tracking_utils.FULL_ID_COLUMN].values[i],
            source_object_table[tracking_utils.VALID_TIME_COLUMN].values[i]
        )

        j = nearest_target_indices[i]
        if j == -1:
            continue

        source_to_target_dict[this_key] = [
            target_object_table[tracking_utils.FULL_ID_COLUMN].values[j],
            target_object_table[tracking_utils.VALID_TIME_COLUMN].values[j]
        ]

    return source_to_target_dict


def _run(myrorss_tracking_dir_name, gridrad_tracking_dir_name,
         max_distance_metres, source_dataset_name, first_spc_date_string,
         last_spc_date_string, output_dir_name):
    """Matches storm objects between MYRORSS and GridRad datasets.

    This is effectively the main method.

    :param myrorss_tracking_dir_name: See documentation at end of file.
    :param gridrad_tracking_dir_name: Same.
    :param max_distance_metres: Same.
    :param source_dataset_name: Same.
    :param first_spc_date_string: Same.
    :param last_spc_date_string: Same.
    :param output_dir_name: Same.
    :raises: ValueError: if `source_dataset_name not in VALID_DATASET_NAMES`.
    """

    if source_dataset_name not in VALID_DATASET_NAMES:
        error_string = (
            '\n{0:s}\nValid datasets (listed above) do not include "{1:s}".'
        ).format(str(VALID_DATASET_NAMES), source_dataset_name)

        raise ValueError(error_string)

    spc_date_strings = time_conversion.get_spc_dates_in_range(
        first_spc_date_string=first_spc_date_string,
        last_spc_date_string=last_spc_date_string)

    if source_dataset_name == radar_utils.MYRORSS_SOURCE_ID:
        source_tracking_dir_name = myrorss_tracking_dir_name
        target_tracking_dir_name = gridrad_tracking_dir_name
        target_dataset_name = radar_utils.GRIDRAD_SOURCE_ID
    else:
        source_tracking_dir_name = gridrad_tracking_dir_name
        target_tracking_dir_name = myrorss_tracking_dir_name
        target_dataset_name = radar_utils.MYRORSS_SOURCE_ID

    source_tracking_file_names = []
    target_tracking_file_names = []

    for this_spc_date_string in spc_date_strings:
        source_tracking_file_names += tracking_io.find_files_one_spc_date(
            top_tracking_dir_name=source_tracking_dir_name,
            tracking_scale_metres2=TRACKING_SCALE_METRES2,
            source_name=tracking_utils.SEGMOTION_NAME,
            spc_date_string=this_spc_date_string,
            raise_error_if_missing=True
        )[0]

        target_tracking_file_names += tracking_io.find_files_one_spc_date(
            top_tracking_dir_name=target_tracking_dir_name,
            tracking_scale_metres2=TRACKING_SCALE_METRES2,
            source_name=tracking_utils.SEGMOTION_NAME,
            spc_date_string=this_spc_date_string,
            raise_error_if_missing=True
        )[0]

    source_times_unix_sec = numpy.array([
        tracking_io.file_name_to_time(f) for f in source_tracking_file_names
    ], dtype=int)

    target_times_unix_sec = numpy.array([
        tracking_io.file_name_to_time(f) for f in target_tracking_file_names
    ], dtype=int)

    source_to_target_indices = _match_all_times(
        source_times_unix_sec=source_times_unix_sec,
        target_times_unix_sec=target_times_unix_sec,
        max_diff_seconds=MAX_TIME_DIFF_SECONDS)
    print(SEPARATOR_STRING)

    del target_times_unix_sec
    target_tracking_file_names = [
        target_tracking_file_names[k] for k in source_to_target_indices
    ]

    num_source_times = len(source_times_unix_sec)

    for i in range(num_source_times):
        print('Reading data from: "{0:s}"...'.format(
            source_tracking_file_names[i]
        ))
        this_source_object_table = tracking_io.read_file(
            source_tracking_file_names[i]
        )

        print('Reading data from: "{0:s}"...'.format(
            target_tracking_file_names[i]
        ))
        this_target_object_table = tracking_io.read_file(
            target_tracking_file_names[i]
        )

        this_source_to_target_dict = _match_locations_one_time(
            source_object_table=this_source_object_table,
            target_object_table=this_target_object_table,
            max_distance_metres=max_distance_metres)

        this_match_file_name = tracking_io.find_match_file(
            top_directory_name=output_dir_name,
            valid_time_unix_sec=source_times_unix_sec[i],
            raise_error_if_missing=False)

        print('Writing results to: "{0:s}"...\n'.format(this_match_file_name))
        tracking_io.write_matches(
            pickle_file_name=this_match_file_name,
            source_to_target_dict=this_source_to_target_dict,
            max_time_diff_seconds=MAX_TIME_DIFF_SECONDS,
            max_distance_metres=max_distance_metres,
            source_dataset_name=source_dataset_name,
            source_tracking_dir_name=source_tracking_dir_name,
            target_dataset_name=target_dataset_name,
            target_tracking_dir_name=target_tracking_dir_name)


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        myrorss_tracking_dir_name=getattr(
            INPUT_ARG_OBJECT, MYRORSS_DIR_ARG_NAME),
        gridrad_tracking_dir_name=getattr(
            INPUT_ARG_OBJECT, GRIDRAD_DIR_ARG_NAME),
        max_distance_metres=getattr(INPUT_ARG_OBJECT, MAX_DISTANCE_ARG_NAME),
        source_dataset_name=getattr(INPUT_ARG_OBJECT, SOURCE_DATASET_ARG_NAME),
        first_spc_date_string=getattr(INPUT_ARG_OBJECT, FIRST_DATE_ARG_NAME),
        last_spc_date_string=getattr(INPUT_ARG_OBJECT, LAST_DATE_ARG_NAME),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
