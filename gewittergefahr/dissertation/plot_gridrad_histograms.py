"""Plots histograms for GridRad dataset.

Specifically, this script plots two histograms:

- number of convective days per month
- number of tornado reports in each convective day
"""

import os.path
import argparse
import numpy
import pandas
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot
from gewittergefahr.gg_io import tornado_io
from gewittergefahr.gg_io import gridrad_io
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import time_periods
from gewittergefahr.gg_utils import file_system_utils

LARGE_INTEGER = int(1e6)
SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

TIME_INTERVAL_SEC = 300
NUM_MONTHS_IN_YEAR = 12

FACE_COLOUR = numpy.array([27, 158, 119], dtype=float) / 255
EDGE_COLOUR = numpy.full(3, 0.)
EDGE_WIDTH = 1.5

FIGURE_RESOLUTION_DPI = 300
FIGURE_WIDTH_INCHES = FIGURE_HEIGHT_INCHES = 15

FONT_SIZE = 30
pyplot.rc('font', size=FONT_SIZE)
pyplot.rc('axes', titlesize=FONT_SIZE)
pyplot.rc('axes', labelsize=FONT_SIZE)
pyplot.rc('xtick', labelsize=FONT_SIZE)
pyplot.rc('ytick', labelsize=FONT_SIZE)
pyplot.rc('legend', fontsize=FONT_SIZE)
pyplot.rc('figure', titlesize=FONT_SIZE)

TORNADO_DIR_ARG_NAME = 'input_tornado_dir_name'
GRIDRAD_DIR_ARG_NAME = 'input_gridrad_dir_name'
FIRST_DATE_ARG_NAME = 'first_spc_date_string'
LAST_DATE_ARG_NAME = 'last_spc_date_string'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

TORNADO_DIR_HELP_STRING = (
    'Name of directory with tornado reports.  Files therein will be found by '
    '`tornado_io.find_processed_file` and read by '
    '`tornado_io.read_processed_file`.')

GRIDRAD_DIR_HELP_STRING = (
    'Name of top-level GridRad directory, used to determine which convective '
    'days are covered.  Files therein will be found by `gridrad_io.find_file`.')

SPC_DATE_HELP_STRING = (
    'SPC date or convective day (format "yyyymmdd").  This script will look for'
    ' GridRad files in the period `{0:s}`...`{1:s}`.'
).format(FIRST_DATE_ARG_NAME, LAST_DATE_ARG_NAME)

OUTPUT_DIR_HELP_STRING = (
    'Name of output directory.  Figures will be saved here.')

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + TORNADO_DIR_ARG_NAME, type=str, required=True,
    help=TORNADO_DIR_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + GRIDRAD_DIR_ARG_NAME, type=str, required=True,
    help=GRIDRAD_DIR_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + FIRST_DATE_ARG_NAME, type=str, required=True,
    help=SPC_DATE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + LAST_DATE_ARG_NAME, type=str, required=True,
    help=SPC_DATE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING)


def _find_gridrad_file_for_date(top_gridrad_dir_name, spc_date_string):
    """Tries to find one GridRad file for given SPC date.

    :param top_gridrad_dir_name: See documentation at top of file.
    :param spc_date_string: SPC date or convective day (format "yyyymmdd").
    :return: gridrad_file_name: Path to GridRad file.  If no files were found
        for the given SPC date, returns None.
    """

    first_time_unix_sec = time_conversion.get_start_of_spc_date(spc_date_string)
    last_time_unix_sec = time_conversion.get_end_of_spc_date(spc_date_string)
    all_times_unix_sec = time_periods.range_and_interval_to_list(
        start_time_unix_sec=first_time_unix_sec,
        end_time_unix_sec=last_time_unix_sec,
        time_interval_sec=TIME_INTERVAL_SEC, include_endpoint=True)

    for this_time_unix_sec in all_times_unix_sec:
        this_gridrad_file_name = gridrad_io.find_file(
            unix_time_sec=this_time_unix_sec,
            top_directory_name=top_gridrad_dir_name,
            raise_error_if_missing=False)

        if os.path.isfile(this_gridrad_file_name):
            return this_gridrad_file_name

    return None


def _spc_dates_to_years(spc_date_strings):
    """Finds first and last years in set of SPC dates.

    :param spc_date_strings: 1-D list of SPC dates (format "yyyymmdd").
    :return: first_year: First year.
    :return: last_year: Last year.
    """

    start_times_unix_sec = numpy.array(
        [time_conversion.get_start_of_spc_date(d) for d in spc_date_strings],
        dtype=int
    )
    end_times_unix_sec = numpy.array(
        [time_conversion.get_end_of_spc_date(d) for d in spc_date_strings],
        dtype=int
    )

    start_years = numpy.array([
        int(time_conversion.unix_sec_to_string(t, '%Y'))
        for t in start_times_unix_sec
    ], dtype=int)

    end_years = numpy.array([
        int(time_conversion.unix_sec_to_string(t, '%Y'))
        for t in end_times_unix_sec
    ], dtype=int)

    all_years = numpy.concatenate((start_years, end_years))

    return numpy.min(all_years), numpy.max(all_years)


def _read_tornado_reports(tornado_dir_name, first_year, last_year):
    """Reads tornado reports from the given years.

    :param tornado_dir_name: See documentation at top of file.
    :param first_year: First year.
    :param last_year: Last year.
    :return: tornado_table: pandas DataFrame in format returned by
        `tornado_io.read_processed_file`.
    """

    list_of_tornado_tables = []

    for this_year in range(first_year, last_year + 1):
        this_file_name = tornado_io.find_processed_file(
            directory_name=tornado_dir_name, year=this_year)

        print('Reading data from: "{0:s}"...'.format(this_file_name))
        list_of_tornado_tables.append(
            tornado_io.read_processed_file(this_file_name)
        )

        if len(list_of_tornado_tables) == 1:
            continue

        list_of_tornado_tables[-1] = list_of_tornado_tables[-1].align(
            list_of_tornado_tables[0], axis=1
        )[0]

    return pandas.concat(list_of_tornado_tables, axis=0, ignore_index=True)


def _get_num_tornadoes_in_day(tornado_table, spc_date_string):
    """Returns number of tornado reports for given SPC date (convective day).

    :param tornado_table: pandas DataFrame in format returned by
        `tornado_io.read_processed_file`.
    :param spc_date_string: SPC date (format "yyyymmdd").
    :return: num_tornadoes: Number of tornado reports.
    """

    first_time_unix_sec = time_conversion.get_start_of_spc_date(spc_date_string)
    last_time_unix_sec = time_conversion.get_end_of_spc_date(spc_date_string)

    good_start_point_flags = numpy.logical_and(
        tornado_table[tornado_io.START_TIME_COLUMN].values >=
        first_time_unix_sec,
        tornado_table[tornado_io.START_TIME_COLUMN].values <= last_time_unix_sec
    )

    good_end_point_flags = numpy.logical_and(
        tornado_table[tornado_io.END_TIME_COLUMN].values >= first_time_unix_sec,
        tornado_table[tornado_io.END_TIME_COLUMN].values <= last_time_unix_sec
    )

    return numpy.sum(numpy.logical_or(
        good_start_point_flags, good_end_point_flags
    ))


def _plot_tornado_histogram(num_tornadoes_by_day, output_file_name):
    """Plots histogram for daily number of tornado reports.

    D = number of SPC dates with GridRad data

    :param num_tornadoes_by_day: length-D numpy array with number of tornado
        reports by day.
    :param output_file_name: Path to output file (figure will be saved here).
    """

    lower_bin_edges = numpy.concatenate((
        numpy.linspace(0, 6, num=7, dtype=int),
        numpy.linspace(11, 101, num=10, dtype=int)
    ))

    upper_bin_edges = numpy.concatenate((
        numpy.linspace(0, 5, num=6, dtype=int),
        numpy.linspace(10, 100, num=10, dtype=int),
        numpy.array([LARGE_INTEGER], dtype=int)
    ))

    num_bins = len(lower_bin_edges)
    num_days_by_bin = numpy.full(num_bins, -1, dtype=int)
    x_tick_labels = [''] * num_bins

    for k in range(num_bins):
        num_days_by_bin[k] = numpy.sum(numpy.logical_and(
            num_tornadoes_by_day >= lower_bin_edges[k],
            num_tornadoes_by_day <= upper_bin_edges[k]
        ))

        if lower_bin_edges[k] == upper_bin_edges[k]:
            x_tick_labels[k] = '{0:d}'.format(lower_bin_edges[k])
        elif k == num_bins - 1:
            x_tick_labels[k] = '{0:d}+'.format(lower_bin_edges[k])
        else:
            x_tick_labels[k] = '{0:d}-{1:d}'.format(
                lower_bin_edges[k], upper_bin_edges[k]
            )

    figure_object, axes_object = pyplot.subplots(
        1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
    )

    x_tick_coords = 0.5 + numpy.linspace(
        0, num_bins - 1, num=num_bins, dtype=float
    )

    axes_object.bar(
        x=x_tick_coords, height=num_days_by_bin, width=1.,
        color=FACE_COLOUR, edgecolor=EDGE_COLOUR, linewidth=EDGE_WIDTH)

    axes_object.set_xlim([
        x_tick_coords[0] - 0.5, x_tick_coords[-1] + 0.5
    ])
    axes_object.set_xticks(x_tick_coords)
    axes_object.set_xticklabels(x_tick_labels, rotation=90.)

    axes_object.set_title(
        'Histogram of daily tornado reports'
    )
    axes_object.set_ylabel('Number of convective days')
    axes_object.set_xlabel('Number of tornado reports')

    print('Saving figure to: "{0:s}"...'.format(output_file_name))
    figure_object.savefig(
        output_file_name, dpi=FIGURE_RESOLUTION_DPI, pad_inches=0,
        bbox_inches='tight'
    )
    pyplot.close(figure_object)


def _plot_month_histogram(spc_date_strings, output_file_name):
    """Plots histogram of months.

    :param spc_date_strings: 1-D list of SPC dates (format "yyyymmdd") with
        GridRad data.
    :param output_file_name: Path to output file (figure will be saved here).
    """

    start_times_unix_sec = numpy.array(
        [time_conversion.get_start_of_spc_date(d) for d in spc_date_strings],
        dtype=int
    )
    end_times_unix_sec = numpy.array(
        [time_conversion.get_end_of_spc_date(d) for d in spc_date_strings],
        dtype=int
    )

    start_month_by_date = numpy.array([
        int(time_conversion.unix_sec_to_string(t, '%m'))
        for t in start_times_unix_sec
    ], dtype=int)

    end_month_by_date = numpy.array([
        int(time_conversion.unix_sec_to_string(t, '%m'))
        for t in end_times_unix_sec
    ], dtype=int)

    num_days_by_month = numpy.full(NUM_MONTHS_IN_YEAR, numpy.nan)

    for k in range(NUM_MONTHS_IN_YEAR):
        num_days_by_month[k] = 0.5 * (
            numpy.sum(start_month_by_date == k + 1) +
            numpy.sum(end_month_by_date == k + 1)
        )

    figure_object, axes_object = pyplot.subplots(
        1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
    )

    x_tick_coords = 0.5 + numpy.linspace(
        0, NUM_MONTHS_IN_YEAR - 1, num=NUM_MONTHS_IN_YEAR, dtype=float
    )
    x_tick_labels = [
        'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct',
        'Nov', 'Dec'
    ]

    axes_object.bar(
        x=x_tick_coords, height=num_days_by_month, width=1.,
        color=FACE_COLOUR, edgecolor=EDGE_COLOUR, linewidth=EDGE_WIDTH)

    axes_object.set_xlim([
        x_tick_coords[0] - 0.5, x_tick_coords[-1] + 0.5
    ])
    axes_object.set_xticks(x_tick_coords)
    axes_object.set_xticklabels(x_tick_labels, rotation=90.)

    axes_object.set_title('Histogram of months')
    axes_object.set_ylabel('Number of convective days')
    axes_object.set_xlabel('Month')

    print('Saving figure to: "{0:s}"...'.format(output_file_name))
    figure_object.savefig(
        output_file_name, dpi=FIGURE_RESOLUTION_DPI, pad_inches=0,
        bbox_inches='tight'
    )
    pyplot.close(figure_object)


def _run(tornado_dir_name, top_gridrad_dir_name, first_spc_date_string,
         last_spc_date_string, output_dir_name):
    """Plots histograms for GridRad dataset.

    This is effectively the main method.

    :param tornado_dir_name: See documentation at top of file.
    :param top_gridrad_dir_name: Same.
    :param first_spc_date_string: Same.
    :param last_spc_date_string: Same.
    :param output_dir_name: Same.
    """

    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=output_dir_name)

    all_spc_date_strings = time_conversion.get_spc_dates_in_range(
        first_spc_date_string=first_spc_date_string,
        last_spc_date_string=last_spc_date_string)

    spc_date_strings = []

    for this_spc_date_string in all_spc_date_strings:
        this_gridrad_file_name = _find_gridrad_file_for_date(
            top_gridrad_dir_name=top_gridrad_dir_name,
            spc_date_string=this_spc_date_string)

        if this_gridrad_file_name is None:
            continue

        spc_date_strings.append(this_spc_date_string)

    first_year, last_year = _spc_dates_to_years(spc_date_strings)
    tornado_table = _read_tornado_reports(
        tornado_dir_name=tornado_dir_name, first_year=first_year,
        last_year=last_year
    )
    print(SEPARATOR_STRING)

    num_days = len(spc_date_strings)
    num_tornadoes_by_day = numpy.full(num_days, -1, dtype=int)

    for i in range(num_days):
        num_tornadoes_by_day[i] = _get_num_tornadoes_in_day(
            tornado_table=tornado_table, spc_date_string=spc_date_strings[i]
        )

        print('Number of tornadoes on SPC date "{0:s}" = {1:d}'.format(
            spc_date_strings[i], num_tornadoes_by_day[i]
        ))

    print(SEPARATOR_STRING)

    _plot_tornado_histogram(
        num_tornadoes_by_day=num_tornadoes_by_day,
        output_file_name='{0:s}/tornado_histogram.jpg'.format(output_dir_name)
    )

    _plot_month_histogram(
        spc_date_strings=spc_date_strings,
        output_file_name='{0:s}/month_histogram.jpg'.format(output_dir_name)
    )


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        tornado_dir_name=getattr(INPUT_ARG_OBJECT, TORNADO_DIR_ARG_NAME),
        top_gridrad_dir_name=getattr(INPUT_ARG_OBJECT, GRIDRAD_DIR_ARG_NAME),
        first_spc_date_string=getattr(INPUT_ARG_OBJECT, FIRST_DATE_ARG_NAME),
        last_spc_date_string=getattr(INPUT_ARG_OBJECT, LAST_DATE_ARG_NAME),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
