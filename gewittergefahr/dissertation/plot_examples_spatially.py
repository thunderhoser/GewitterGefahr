"""Plots spatial distribution of examples (storm objects) in file.

This script plots the examples as points, colour-coded by time of day.
"""

import argparse
import numpy
import pandas
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot
from gewittergefahr.gg_io import storm_tracking_io as tracking_io
from gewittergefahr.gg_utils import storm_tracking_utils as tracking_utils
from gewittergefahr.gg_utils import temporal_tracking
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import echo_top_tracking
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.plotting import plotting_utils
from gewittergefahr.plotting import storm_plotting

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'
DUMMY_TRACKING_SCALE_METRES2 = echo_top_tracking.DUMMY_TRACKING_SCALE_METRES2

NUM_HOURS_IN_DAY = 24
NUM_SECONDS_IN_DAY = 86400
COLOUR_BAR_TIME_FORMAT = '%H%M UTC'

NUM_PARALLELS = 8
NUM_MERIDIANS = 8
LATLNG_BUFFER_DEG = 0.25
BORDER_WIDTH = 0.5
BORDER_COLOUR = numpy.full(3, 0.)

TRACK_LINE_WIDTH = 4
BACKGROUND_COLOUR = numpy.array([31, 120, 180], dtype=float) / 255

try:
    COLOUR_MAP_OBJECT = pyplot.get_cmap(
        name='twilight_shifted', lut=NUM_HOURS_IN_DAY
    )
except:
    COLOUR_MAP_OBJECT = pyplot.get_cmap(name='hsv', lut=NUM_HOURS_IN_DAY)

FIGURE_RESOLUTION_DPI = 300

STORM_METAFILE_ARG_NAME = 'input_storm_metafile_name'
TRACKING_DIR_ARG_NAME = 'input_tracking_dir_name'
LEAD_TIME_ARG_NAME = 'lead_time_seconds'
OUTPUT_FILE_ARG_NAME = 'output_file_name'

STORM_METAFILE_HELP_STRING = (
    'Path to file with metadata (IDs and valid times) for desired examples.  '
    'Will be read by `storm_tracking_io.read_ids_and_times`.'
)
TRACKING_DIR_HELP_STRING = (
    'Name of top-level directory with tracking data.  Files therein will be '
    'found by `storm_tracking_io.find_file` and read by '
    '`storm_tracking_io.read_file`.'
)
LEAD_TIME_HELP_STRING = (
    'Lead time.  For each example in `{0:s}`, will plot successors over the '
    'next `{1:s}` seconds.'
).format(STORM_METAFILE_ARG_NAME, LEAD_TIME_ARG_NAME)

OUTPUT_FILE_HELP_STRING = 'Path to output file (figure will be saved here).'

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + STORM_METAFILE_ARG_NAME, type=str, required=True,
    help=STORM_METAFILE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + TRACKING_DIR_ARG_NAME, type=str, required=True,
    help=TRACKING_DIR_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + LEAD_TIME_ARG_NAME, type=int, required=False, default=3600,
    help=LEAD_TIME_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_FILE_ARG_NAME, type=str, required=True,
    help=OUTPUT_FILE_HELP_STRING
)


def _run(storm_metafile_name, top_tracking_dir_name, lead_time_seconds,
         output_file_name):
    """Plots spatial distribution of examples (storm objects) in file.

    This is effectively the main method.

    :param storm_metafile_name: See documentation at top of file.
    :param top_tracking_dir_name: Same.
    :param lead_time_seconds: Same.
    :param output_file_name: Same.
    """

    file_system_utils.mkdir_recursive_if_necessary(file_name=output_file_name)

    # Read storm metadata.
    print('Reading storm metadata from: "{0:s}"...'.format(storm_metafile_name))
    orig_full_id_strings, orig_times_unix_sec = (
        tracking_io.read_ids_and_times(storm_metafile_name)
    )
    orig_primary_id_strings = temporal_tracking.full_to_partial_ids(
        orig_full_id_strings
    )[0]

    # Find relevant tracking files.
    spc_date_strings = [
        time_conversion.time_to_spc_date_string(t) for t in orig_times_unix_sec
    ]
    spc_date_strings += [
        time_conversion.time_to_spc_date_string(t + lead_time_seconds)
        for t in orig_times_unix_sec
    ]
    spc_date_strings = list(set(spc_date_strings))

    tracking_file_names = []

    for this_spc_date_string in spc_date_strings:
        tracking_file_names += tracking_io.find_files_one_spc_date(
            top_tracking_dir_name=top_tracking_dir_name,
            tracking_scale_metres2=DUMMY_TRACKING_SCALE_METRES2,
            source_name=tracking_utils.SEGMOTION_NAME,
            spc_date_string=this_spc_date_string, raise_error_if_missing=False
        )[0]

    file_times_unix_sec = numpy.array(
        [tracking_io.file_name_to_time(f) for f in tracking_file_names],
        dtype=int
    )

    num_orig_storm_objects = len(orig_full_id_strings)
    num_files = len(file_times_unix_sec)
    keep_file_flags = numpy.full(num_files, 0, dtype=bool)

    for i in range(num_orig_storm_objects):
        these_flags = numpy.logical_and(
            file_times_unix_sec >= orig_times_unix_sec[i],
            file_times_unix_sec <= orig_times_unix_sec[i] + lead_time_seconds
        )
        keep_file_flags = numpy.logical_or(keep_file_flags, these_flags)

    del file_times_unix_sec
    keep_file_indices = numpy.where(keep_file_flags)[0]
    tracking_file_names = [tracking_file_names[k] for k in keep_file_indices]

    # Read relevant tracking files.
    num_files = len(tracking_file_names)
    storm_object_tables = [None] * num_files
    print(SEPARATOR_STRING)

    for i in range(num_files):
        print('Reading data from: "{0:s}"...'.format(tracking_file_names[i]))
        this_table = tracking_io.read_file(tracking_file_names[i])

        storm_object_tables[i] = this_table.loc[
            this_table[tracking_utils.PRIMARY_ID_COLUMN].isin(
                numpy.array(orig_primary_id_strings)
            )
        ]

        if i == 0:
            continue

        storm_object_tables[i] = storm_object_tables[i].align(
            storm_object_tables[0], axis=1
        )[0]

    storm_object_table = pandas.concat(
        storm_object_tables, axis=0, ignore_index=True
    )
    print(SEPARATOR_STRING)

    # Find relevant storm objects.
    orig_object_rows = tracking_utils.find_storm_objects(
        all_id_strings=
        storm_object_table[tracking_utils.FULL_ID_COLUMN].values.tolist(),
        all_times_unix_sec=
        storm_object_table[tracking_utils.VALID_TIME_COLUMN].values,
        id_strings_to_keep=orig_full_id_strings,
        times_to_keep_unix_sec=orig_times_unix_sec
    )

    good_object_rows = numpy.array([], dtype=int)

    for i in range(num_orig_storm_objects):
        # Non-merging successors only!

        first_rows = temporal_tracking.find_successors(
            storm_object_table=storm_object_table,
            target_row=orig_object_rows[i],
            num_seconds_forward=lead_time_seconds,
            max_num_sec_id_changes=1,
            change_type_string=temporal_tracking.SPLIT_STRING,
            return_all_on_path=True)

        second_rows = temporal_tracking.find_successors(
            storm_object_table=storm_object_table,
            target_row=orig_object_rows[i],
            num_seconds_forward=lead_time_seconds,
            max_num_sec_id_changes=0,
            change_type_string=temporal_tracking.MERGER_STRING,
            return_all_on_path=True)

        first_rows = first_rows.tolist()
        second_rows = second_rows.tolist()
        these_rows = set(first_rows) & set(second_rows)
        these_rows = numpy.array(list(these_rows), dtype=int)

        good_object_rows = numpy.concatenate((good_object_rows, these_rows))

    good_object_rows = numpy.unique(good_object_rows)
    storm_object_table = storm_object_table.iloc[good_object_rows]

    times_of_day_sec = numpy.mod(
        storm_object_table[tracking_utils.VALID_TIME_COLUMN].values,
        NUM_SECONDS_IN_DAY
    )
    storm_object_table = storm_object_table.assign(**{
        tracking_utils.VALID_TIME_COLUMN: times_of_day_sec
    })

    min_plot_latitude_deg = -LATLNG_BUFFER_DEG + numpy.min(
        storm_object_table[tracking_utils.CENTROID_LATITUDE_COLUMN].values
    )
    max_plot_latitude_deg = LATLNG_BUFFER_DEG + numpy.max(
        storm_object_table[tracking_utils.CENTROID_LATITUDE_COLUMN].values
    )
    min_plot_longitude_deg = -LATLNG_BUFFER_DEG + numpy.min(
        storm_object_table[tracking_utils.CENTROID_LONGITUDE_COLUMN].values
    )
    max_plot_longitude_deg = LATLNG_BUFFER_DEG + numpy.max(
        storm_object_table[tracking_utils.CENTROID_LONGITUDE_COLUMN].values
    )

    _, axes_object, basemap_object = (
        plotting_utils.create_equidist_cylindrical_map(
            min_latitude_deg=min_plot_latitude_deg,
            max_latitude_deg=max_plot_latitude_deg,
            min_longitude_deg=min_plot_longitude_deg,
            max_longitude_deg=max_plot_longitude_deg,
            resolution_string='i')
    )

    plotting_utils.plot_coastlines(
        basemap_object=basemap_object, axes_object=axes_object,
        line_colour=BORDER_COLOUR, line_width=BORDER_WIDTH * 2
    )
    plotting_utils.plot_countries(
        basemap_object=basemap_object, axes_object=axes_object,
        line_colour=BORDER_COLOUR, line_width=BORDER_WIDTH
    )
    plotting_utils.plot_states_and_provinces(
        basemap_object=basemap_object, axes_object=axes_object,
        line_colour=BORDER_COLOUR, line_width=BORDER_WIDTH
    )
    plotting_utils.plot_parallels(
        basemap_object=basemap_object, axes_object=axes_object,
        num_parallels=NUM_PARALLELS, line_width=BORDER_WIDTH
    )
    plotting_utils.plot_meridians(
        basemap_object=basemap_object, axes_object=axes_object,
        num_meridians=NUM_MERIDIANS, line_width=BORDER_WIDTH
    )

    # colour_bar_object = storm_plotting.plot_storm_tracks(
    #     storm_object_table=storm_object_table, axes_object=axes_object,
    #     basemap_object=basemap_object, colour_map_object=COLOUR_MAP_OBJECT,
    #     colour_min_unix_sec=0, colour_max_unix_sec=NUM_SECONDS_IN_DAY - 1,
    #     line_width=TRACK_LINE_WIDTH,
    #     start_marker_type=None, end_marker_type=None
    # )

    colour_bar_object = storm_plotting.plot_storm_centroids(
        storm_object_table=storm_object_table,
        axes_object=axes_object, basemap_object=basemap_object,
        colour_map_object=COLOUR_MAP_OBJECT,
        colour_min_unix_sec=0, colour_max_unix_sec=NUM_SECONDS_IN_DAY - 1
    )

    tick_times_unix_sec = numpy.linspace(
        0, NUM_SECONDS_IN_DAY, num=NUM_HOURS_IN_DAY + 1, dtype=int
    )
    tick_times_unix_sec = tick_times_unix_sec[:-1]
    tick_times_unix_sec = tick_times_unix_sec[::2]

    tick_time_strings = [
        time_conversion.unix_sec_to_string(t, COLOUR_BAR_TIME_FORMAT)
        for t in tick_times_unix_sec
    ]

    colour_bar_object.set_ticks(tick_times_unix_sec)
    colour_bar_object.set_ticklabels(tick_time_strings)

    print('Saving figure to: "{0:s}"...'.format(output_file_name))
    pyplot.savefig(
        output_file_name, dpi=FIGURE_RESOLUTION_DPI,
        pad_inches=0, bbox_inches='tight'
    )
    pyplot.close()


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        storm_metafile_name=getattr(INPUT_ARG_OBJECT, STORM_METAFILE_ARG_NAME),
        top_tracking_dir_name=getattr(INPUT_ARG_OBJECT, TRACKING_DIR_ARG_NAME),
        lead_time_seconds=getattr(INPUT_ARG_OBJECT, LEAD_TIME_ARG_NAME),
        output_file_name=getattr(INPUT_ARG_OBJECT, OUTPUT_FILE_ARG_NAME)
    )
