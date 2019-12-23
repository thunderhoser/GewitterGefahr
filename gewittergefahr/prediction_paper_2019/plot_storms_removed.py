"""Plots storms that were removed by remove_storms_outside_conus.py."""

import argparse
import numpy
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot
from gewittergefahr.gg_io import storm_tracking_io as tracking_io
from gewittergefahr.gg_utils import storm_tracking_utils as tracking_utils
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import echo_top_tracking
from gewittergefahr.gg_utils import conus_boundary
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.plotting import plotting_utils

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'
DUMMY_TRACKING_SCALE_METRES2 = echo_top_tracking.DUMMY_TRACKING_SCALE_METRES2

EROSION_DISTANCE_METRES = 3e4

NUM_PARALLELS = 8
NUM_MERIDIANS = 8
BORDER_COLOUR = numpy.full(3, 0.)

LINE_WIDTH = 2
LINE_COLOUR = numpy.array([117, 112, 179], dtype=float) / 255

MARKER_TYPE = 'o'
MARKER_SIZE = 4
MARKER_COLOUR = numpy.array([217, 95, 2], dtype=float) / 255

FIGURE_RESOLUTION_DPI = 600

ORIG_TRACKING_DIR_ARG_NAME = 'orig_tracking_dir_name'
NEW_TRACKING_DIR_ARG_NAME = 'new_tracking_dir_name'
FIRST_DATE_ARG_NAME = 'first_spc_date_string'
LAST_DATE_ARG_NAME = 'last_spc_date_string'
OUTPUT_FILE_ARG_NAME = 'output_file_name'

ORIG_TRACKING_DIR_HELP_STRING = (
    'Name of top-level directory with original tracking data (before removing '
    'storms outside CONUS).  Files therein will be found by '
    '`storm_tracking_io.find_file` and read by `storm_tracking_io.read_file`.'
)

NEW_TRACKING_DIR_HELP_STRING = (
    'Same as `{0:s}` but for new tracking data (after removing storms outside '
    'CONUS).'
).format(ORIG_TRACKING_DIR_ARG_NAME)

SPC_DATE_HELP_STRING = (
    'SPC date (format "yyyymmdd").  This script will plot storms removed over '
    'the period `{0:s}`...`{1:s}`.'
).format(FIRST_DATE_ARG_NAME, LAST_DATE_ARG_NAME)

OUTPUT_FILE_HELP_STRING = 'Path to output file (figure will be saved here).'

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + ORIG_TRACKING_DIR_ARG_NAME, type=str, required=True,
    help=ORIG_TRACKING_DIR_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + NEW_TRACKING_DIR_ARG_NAME, type=str, required=True,
    help=NEW_TRACKING_DIR_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + FIRST_DATE_ARG_NAME, type=str, required=True,
    help=SPC_DATE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + LAST_DATE_ARG_NAME, type=str, required=True,
    help=SPC_DATE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_FILE_ARG_NAME, type=str, required=True,
    help=OUTPUT_FILE_HELP_STRING
)


def _run(top_orig_tracking_dir_name, top_new_tracking_dir_name,
         first_spc_date_string, last_spc_date_string, output_file_name):
    """Plots storms that were removed by remove_storms_outside_conus.py.

    This is effectively the main method.

    :param top_orig_tracking_dir_name: See documentation at top of file.
    :param top_new_tracking_dir_name: Same.
    :param first_spc_date_string: Same.
    :param last_spc_date_string: Same.
    :param output_file_name: Same.
    """

    file_system_utils.mkdir_recursive_if_necessary(file_name=output_file_name)

    spc_date_strings = time_conversion.get_spc_dates_in_range(
        first_spc_date_string=first_spc_date_string,
        last_spc_date_string=last_spc_date_string)

    orig_tracking_file_names = []

    for d in spc_date_strings:
        orig_tracking_file_names += tracking_io.find_files_one_spc_date(
            top_tracking_dir_name=top_orig_tracking_dir_name,
            tracking_scale_metres2=DUMMY_TRACKING_SCALE_METRES2,
            source_name=tracking_utils.SEGMOTION_NAME, spc_date_string=d,
            raise_error_if_missing=False
        )[0]

    valid_times_unix_sec = numpy.array([
        tracking_io.file_name_to_time(f) for f in orig_tracking_file_names
    ], dtype=int)

    new_tracking_file_names = [
        tracking_io.find_file(
            top_tracking_dir_name=top_new_tracking_dir_name,
            tracking_scale_metres2=DUMMY_TRACKING_SCALE_METRES2,
            source_name=tracking_utils.SEGMOTION_NAME,
            valid_time_unix_sec=t,
            spc_date_string=time_conversion.time_to_spc_date_string(t),
            raise_error_if_missing=True
        )
        for t in valid_times_unix_sec
    ]

    orig_storm_object_table = tracking_io.read_many_files(
        orig_tracking_file_names
    )
    print(SEPARATOR_STRING)

    new_storm_object_table = tracking_io.read_many_files(
        new_tracking_file_names
    )
    print(SEPARATOR_STRING)

    orig_storm_id_strings = (
        orig_storm_object_table[tracking_utils.FULL_ID_COLUMN].values.tolist()
    )
    orig_storm_times_unix_sec = (
        orig_storm_object_table[tracking_utils.VALID_TIME_COLUMN].values
    )
    new_storm_id_strings = (
        new_storm_object_table[tracking_utils.FULL_ID_COLUMN].values.tolist()
    )
    new_storm_times_unix_sec = (
        new_storm_object_table[tracking_utils.VALID_TIME_COLUMN].values
    )

    num_orig_storm_objects = len(orig_storm_object_table.index)
    orig_kept_flags = numpy.full(num_orig_storm_objects, 0, dtype=bool)

    these_indices = tracking_utils.find_storm_objects(
        all_id_strings=orig_storm_id_strings,
        all_times_unix_sec=orig_storm_times_unix_sec,
        id_strings_to_keep=new_storm_id_strings,
        times_to_keep_unix_sec=new_storm_times_unix_sec,
        allow_missing=False
    )

    orig_kept_flags[these_indices] = True
    orig_removed_indices = numpy.where(numpy.invert(orig_kept_flags))[0]
    removed_storm_object_table = orig_storm_object_table.iloc[
        orig_removed_indices
    ]

    removed_latitudes_deg = removed_storm_object_table[
        tracking_utils.CENTROID_LATITUDE_COLUMN
    ].values

    removed_longitudes_deg = removed_storm_object_table[
        tracking_utils.CENTROID_LONGITUDE_COLUMN
    ].values

    figure_object, axes_object, basemap_object = (
        plotting_utils.create_equidist_cylindrical_map(
            min_latitude_deg=numpy.min(removed_latitudes_deg) - 1.,
            max_latitude_deg=numpy.max(removed_latitudes_deg) + 1.,
            min_longitude_deg=numpy.min(removed_longitudes_deg) - 1.,
            max_longitude_deg=numpy.max(removed_longitudes_deg) + 1.,
            resolution_string='i'
        )
    )

    plotting_utils.plot_coastlines(
        basemap_object=basemap_object, axes_object=axes_object,
        line_colour=BORDER_COLOUR
    )
    plotting_utils.plot_countries(
        basemap_object=basemap_object, axes_object=axes_object,
        line_colour=BORDER_COLOUR
    )
    plotting_utils.plot_states_and_provinces(
        basemap_object=basemap_object, axes_object=axes_object,
        line_colour=BORDER_COLOUR
    )
    plotting_utils.plot_parallels(
        basemap_object=basemap_object, axes_object=axes_object,
        num_parallels=NUM_PARALLELS
    )
    plotting_utils.plot_meridians(
        basemap_object=basemap_object, axes_object=axes_object,
        num_meridians=NUM_MERIDIANS
    )

    conus_latitudes_deg, conus_longitudes_deg = (
        conus_boundary.read_from_netcdf()
    )
    conus_latitudes_deg, conus_longitudes_deg = conus_boundary.erode_boundary(
        latitudes_deg=conus_latitudes_deg, longitudes_deg=conus_longitudes_deg,
        erosion_distance_metres=EROSION_DISTANCE_METRES
    )

    axes_object.plot(
        conus_longitudes_deg, conus_latitudes_deg,
        color=LINE_COLOUR, linestyle='solid', linewidth=LINE_WIDTH
    )
    axes_object.plot(
        removed_longitudes_deg, removed_latitudes_deg, linestyle='None',
        marker=MARKER_TYPE, markersize=MARKER_SIZE, markeredgewidth=0,
        markerfacecolor=MARKER_COLOUR, markeredgecolor=MARKER_COLOUR
    )

    print('Saving figure to: "{0:s}"...'.format(output_file_name))
    figure_object.savefig(
        output_file_name, dpi=FIGURE_RESOLUTION_DPI,
        pad_inches=0, bbox_inches='tight'
    )
    pyplot.close(figure_object)


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        top_orig_tracking_dir_name=getattr(
            INPUT_ARG_OBJECT, ORIG_TRACKING_DIR_ARG_NAME
        ),
        top_new_tracking_dir_name=getattr(
            INPUT_ARG_OBJECT, NEW_TRACKING_DIR_ARG_NAME
        ),
        first_spc_date_string=getattr(INPUT_ARG_OBJECT, FIRST_DATE_ARG_NAME),
        last_spc_date_string=getattr(INPUT_ARG_OBJECT, LAST_DATE_ARG_NAME),
        output_file_name=getattr(INPUT_ARG_OBJECT, OUTPUT_FILE_ARG_NAME)
    )
