"""Plots RAP/RUC field centered on each example (storm object)."""

import socket
import os.path
import argparse
import numpy
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot
from gewittergefahr.gg_io import storm_tracking_io as tracking_io
from gewittergefahr.gg_io import nwp_model_io
from gewittergefahr.gg_utils import storm_tracking_utils as tracking_utils
from gewittergefahr.gg_utils import nwp_model_utils
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import number_rounding
from gewittergefahr.gg_utils import echo_top_tracking
from gewittergefahr.gg_utils import soundings
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.gg_utils import error_checking
from gewittergefahr.plotting import plotting_utils
from gewittergefahr.plotting import nwp_plotting

FILE_NAME_TIME_FORMAT = '%Y-%m-%d-%H%M%S'

FIRST_RAP_TIME_STRING = '2012-05-01-00'
FIRST_RAP_TIME_UNIX_SEC = time_conversion.string_to_unix_sec(
    FIRST_RAP_TIME_STRING, '%Y-%m-%d-%H'
)

INIT_TIME_INTERVAL_SEC = 3600
DUMMY_TRACKING_SCALE_METRES2 = echo_top_tracking.DUMMY_TRACKING_SCALE_METRES2

MARKER_COLOUR = numpy.array([31, 120, 180], dtype=float) / 255
ORIGIN_MARKER_TYPE = 'o'
ORIGIN_MARKER_SIZE = 36
ORIGIN_MARKER_EDGE_WIDTH = 0
EXTRAP_MARKER_TYPE = 'x'
EXTRAP_MARKER_SIZE = 36
EXTRAP_MARKER_EDGE_WIDTH = 4

COLOUR_MAP_OBJECT = pyplot.get_cmap('YlOrRd')
MAX_COLOUR_PERCENTILE = 99.

NUM_PARALLELS = 8
NUM_MERIDIANS = 6
BORDER_COLOUR = numpy.full(3, 0.)
FIGURE_RESOLUTION_DPI = 300

STORM_METAFILE_ARG_NAME = 'input_storm_metafile_name'
TRACKING_DIR_ARG_NAME = 'input_tracking_dir_name'
LATITUDE_BUFFER_ARG_NAME = 'latitude_buffer_deg'
LONGITUDE_BUFFER_ARG_NAME = 'longitude_buffer_deg'
RAP_DIRECTORY_ARG_NAME = 'input_rap_directory_name'
RUC_DIRECTORY_ARG_NAME = 'input_ruc_directory_name'
LEAD_TIME_ARG_NAME = 'lead_time_seconds'
LAG_TIME_ARG_NAME = 'lag_time_seconds'
FIELD_ARG_NAME = 'field_name_grib1'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

STORM_METAFILE_HELP_STRING = (
    'Path to file with metadata (IDs and valid times) for storm objects.  Will '
    'be read by `storm_tracking_io.read_ids_and_times`.  This script will plot '
    'one RAP/RUC field for each storm object.'
)
TRACKING_DIR_HELP_STRING = (
    'Name of top-level directory with storm-tracking files.  Files will be '
    'found by `storm_tracking_io.find_file` and read by '
    '`storm_tracking_io.read_file`.'
)
LATITUDE_BUFFER_HELP_STRING = (
    'Latitude buffer for plotting domain.  Will plot this many degrees on '
    'either side of the storm center.'
)
LONGITUDE_BUFFER_HELP_STRING = (
    'Longitude buffer for plotting domain.  Will plot this many degrees on '
    'either side of the storm center.'
)
RAP_DIRECTORY_HELP_STRING = (
    'Name of top-level directory with RAP data.  Files therein will be found by'
    ' `nwp_model_io.find_rap_file_any_grid` and read by '
    '`nwp_model_io.read_field_from_grib_file`.  RAP data will be used only for '
    'times {0:s} and later.'
).format(FIRST_RAP_TIME_STRING)

RUC_DIRECTORY_HELP_STRING = (
    'Same as `{0:s}` but for RUC data.  RUC data will be used only for times '
    'before {1:s}.'
).format(RAP_DIRECTORY_ARG_NAME, FIRST_RAP_TIME_STRING)

LEAD_TIME_HELP_STRING = (
    'Lead time.  In each plot, the storm object will be plotted at its current '
    'location and extrapolated this far into the future.'
)
LAG_TIME_HELP_STRING = (
    'Lag time.  In each plot, letting t_0 be the valid time of the storm and '
    't_lead be the lead time, the RAP/RUC analysis used will be the most recent'
    ' one before (t_0 - t_lead).'
)
FIELD_HELP_STRING = (
    'RAP/RUC field to plot.  This should be in grib1 format, like "HGT:850 mb" '
    'for 850-mb height.'
)
OUTPUT_DIR_HELP_STRING = (
    'Name of output directory.  Figures will be saved here.'
)

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
    '--' + LATITUDE_BUFFER_ARG_NAME, type=float, required=False, default=5.,
    help=LATITUDE_BUFFER_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + LONGITUDE_BUFFER_ARG_NAME, type=float, required=False, default=5.,
    help=LONGITUDE_BUFFER_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + RAP_DIRECTORY_ARG_NAME, type=str, required=True,
    help=RAP_DIRECTORY_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + RUC_DIRECTORY_ARG_NAME, type=str, required=True,
    help=RUC_DIRECTORY_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + LEAD_TIME_ARG_NAME, type=int, required=False, default=1800,
    help=LEAD_TIME_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + LAG_TIME_ARG_NAME, type=int, required=False, default=1800,
    help=LAG_TIME_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + FIELD_ARG_NAME, type=str, required=True, help=FIELD_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING
)


def _plot_rapruc_one_example(
        full_storm_id_string, storm_time_unix_sec, top_tracking_dir_name,
        latitude_buffer_deg, longitude_buffer_deg, lead_time_seconds,
        field_name_grib1, output_dir_name, rap_file_name=None,
        ruc_file_name=None):
    """Plots RAP or RUC field for one example.

    :param full_storm_id_string: Full storm ID.
    :param storm_time_unix_sec: Valid time.
    :param top_tracking_dir_name: See documentation at top of file.
    :param latitude_buffer_deg: Same.
    :param longitude_buffer_deg: Same.
    :param lead_time_seconds: Same.
    :param field_name_grib1: Same.
    :param output_dir_name: Same.
    :param rap_file_name: Path to file with RAP analysis.
    :param ruc_file_name: [used only if `rap_file_name is None`]
        Path to file with RUC analysis.
    """

    tracking_file_name = tracking_io.find_file(
        top_tracking_dir_name=top_tracking_dir_name,
        tracking_scale_metres2=DUMMY_TRACKING_SCALE_METRES2,
        source_name=tracking_utils.SEGMOTION_NAME,
        valid_time_unix_sec=storm_time_unix_sec,
        spc_date_string=
        time_conversion.time_to_spc_date_string(storm_time_unix_sec),
        raise_error_if_missing=True
    )

    print('Reading data from: "{0:s}"...'.format(tracking_file_name))
    storm_object_table = tracking_io.read_file(tracking_file_name)
    storm_object_table = storm_object_table.loc[
        storm_object_table[tracking_utils.FULL_ID_COLUMN] ==
        full_storm_id_string
    ]

    extrap_times_sec = numpy.array([0, lead_time_seconds], dtype=int)
    storm_object_table = soundings._create_target_points_for_interp(
        storm_object_table=storm_object_table,
        lead_times_seconds=extrap_times_sec
    )

    orig_latitude_deg = (
        storm_object_table[tracking_utils.CENTROID_LATITUDE_COLUMN].values[0]
    )
    orig_longitude_deg = (
        storm_object_table[tracking_utils.CENTROID_LONGITUDE_COLUMN].values[0]
    )
    extrap_latitude_deg = (
        storm_object_table[tracking_utils.CENTROID_LATITUDE_COLUMN].values[1]
    )
    extrap_longitude_deg = (
        storm_object_table[tracking_utils.CENTROID_LONGITUDE_COLUMN].values[1]
    )

    if rap_file_name is None:
        grib_file_name = ruc_file_name
        model_name = nwp_model_utils.RUC_MODEL_NAME
    else:
        grib_file_name = rap_file_name
        model_name = nwp_model_utils.RAP_MODEL_NAME

    pathless_grib_file_name = os.path.split(grib_file_name)[-1]
    grid_name = pathless_grib_file_name.split('_')[1]

    host_name = socket.gethostname()

    if 'casper' in host_name:
        wgrib_exe_name = '/glade/work/ryanlage/wgrib/wgrib'
        wgrib2_exe_name = '/glade/work/ryanlage/wgrib2/wgrib2/wgrib2'
    elif 'schooner' in host_name:
        wgrib_exe_name = '/condo/swatwork/ralager/wgrib/wgrib'
        wgrib2_exe_name = '/condo/swatwork/ralager/grib2/wgrib2/wgrib2'
    else:
        wgrib_exe_name = '/usr/bin/wgrib'
        wgrib2_exe_name = '/usr/bin/wgrib2'

    print('Reading data from: "{0:s}"...'.format(grib_file_name))
    field_matrix = nwp_model_io.read_field_from_grib_file(
        grib_file_name=grib_file_name, field_name_grib1=field_name_grib1,
        model_name=model_name, grid_id=grid_name,
        wgrib_exe_name=wgrib_exe_name, wgrib2_exe_name=wgrib2_exe_name
    )

    min_plot_latitude_deg = (
        min([orig_latitude_deg, extrap_latitude_deg]) - latitude_buffer_deg
    )
    max_plot_latitude_deg = (
        max([orig_latitude_deg, extrap_latitude_deg]) + latitude_buffer_deg
    )
    min_plot_longitude_deg = (
        min([orig_longitude_deg, extrap_longitude_deg]) - longitude_buffer_deg
    )
    max_plot_longitude_deg = (
        max([orig_longitude_deg, extrap_longitude_deg]) + longitude_buffer_deg
    )

    row_limits, column_limits = nwp_plotting.latlng_limits_to_rowcol_limits(
        min_latitude_deg=min_plot_latitude_deg,
        max_latitude_deg=max_plot_latitude_deg,
        min_longitude_deg=min_plot_longitude_deg,
        max_longitude_deg=max_plot_longitude_deg,
        model_name=model_name, grid_id=grid_name
    )

    field_matrix = field_matrix[
        row_limits[0]:(row_limits[1] + 1),
        column_limits[0]:(column_limits[1] + 1)
    ]

    _, axes_object, basemap_object = nwp_plotting.init_basemap(
        model_name=model_name, grid_id=grid_name,
        first_row_in_full_grid=row_limits[0],
        last_row_in_full_grid=row_limits[1],
        first_column_in_full_grid=column_limits[0],
        last_column_in_full_grid=column_limits[1]
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

    min_colour_value = numpy.nanpercentile(
        field_matrix, 100. - MAX_COLOUR_PERCENTILE
    )
    max_colour_value = numpy.nanpercentile(field_matrix, MAX_COLOUR_PERCENTILE)

    nwp_plotting.plot_subgrid(
        field_matrix=field_matrix, model_name=model_name, grid_id=grid_name,
        axes_object=axes_object, basemap_object=basemap_object,
        colour_map_object=COLOUR_MAP_OBJECT, min_colour_value=min_colour_value,
        max_colour_value=max_colour_value,
        first_row_in_full_grid=row_limits[0],
        first_column_in_full_grid=column_limits[0]
    )

    orig_x_metres, orig_y_metres = basemap_object(
        orig_longitude_deg, orig_latitude_deg
    )
    axes_object.plot(
        orig_x_metres, orig_y_metres, linestyle='None',
        marker=ORIGIN_MARKER_TYPE, markersize=ORIGIN_MARKER_SIZE,
        markeredgewidth=ORIGIN_MARKER_EDGE_WIDTH,
        markerfacecolor=MARKER_COLOUR, markeredgecolor=MARKER_COLOUR
    )

    extrap_x_metres, extrap_y_metres = basemap_object(
        extrap_longitude_deg, extrap_latitude_deg
    )
    axes_object.plot(
        extrap_x_metres, extrap_y_metres, linestyle='None',
        marker=EXTRAP_MARKER_TYPE, markersize=EXTRAP_MARKER_SIZE,
        markeredgewidth=EXTRAP_MARKER_EDGE_WIDTH,
        markerfacecolor=MARKER_COLOUR, markeredgecolor=MARKER_COLOUR
    )

    output_file_name = '{0:s}/{1:s}_{2:s}.jpg'.format(
        output_dir_name, full_storm_id_string.replace('_', '-'),
        time_conversion.unix_sec_to_string(
            storm_time_unix_sec, FILE_NAME_TIME_FORMAT
        )
    )

    print('Saving figure to: "{0:s}"...'.format(output_file_name))
    pyplot.savefig(
        output_file_name, dpi=FIGURE_RESOLUTION_DPI,
        pad_inches=0, bbox_inches='tight'
    )
    pyplot.close()


def _run(storm_metafile_name, top_tracking_dir_name, latitude_buffer_deg,
         longitude_buffer_deg, rap_directory_name, ruc_directory_name,
         lead_time_seconds, lag_time_seconds, field_name_grib1,
         output_dir_name):
    """Plots RAP/RUC field centered on each example (storm object).

    This is effectively the main method.

    :param storm_metafile_name: See documentation at top of file.
    :param top_tracking_dir_name: Same.
    :param latitude_buffer_deg: Same.
    :param longitude_buffer_deg: Same.
    :param rap_directory_name: Same.
    :param ruc_directory_name: Same.
    :param lead_time_seconds: Same.
    :param lag_time_seconds: Same.
    :param field_name_grib1: Same.
    :param output_dir_name: Same.
    """

    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=output_dir_name
    )

    error_checking.assert_is_geq(latitude_buffer_deg, 1.)
    error_checking.assert_is_geq(longitude_buffer_deg, 1.)
    error_checking.assert_is_greater(lead_time_seconds, 0)
    error_checking.assert_is_greater(lag_time_seconds, 0)

    print('Reading metadata from: "{0:s}"...'.format(storm_metafile_name))
    full_storm_id_strings, storm_times_unix_sec = (
        tracking_io.read_ids_and_times(storm_metafile_name)
    )

    init_times_unix_sec = (
        storm_times_unix_sec + lead_time_seconds - lag_time_seconds
    )
    init_times_unix_sec = number_rounding.floor_to_nearest(
        init_times_unix_sec, INIT_TIME_INTERVAL_SEC
    )
    init_times_unix_sec = init_times_unix_sec.astype(int)

    num_examples = len(full_storm_id_strings)
    rap_file_names = [None] * num_examples
    ruc_file_names = [None] * num_examples

    for i in range(num_examples):
        if init_times_unix_sec[i] >= FIRST_RAP_TIME_UNIX_SEC:
            rap_file_names[i] = nwp_model_io.find_rap_file_any_grid(
                top_directory_name=rap_directory_name,
                init_time_unix_sec=init_times_unix_sec[i],
                lead_time_hours=0, raise_error_if_missing=True
            )

            continue

        ruc_file_names[i] = nwp_model_io.find_rap_file_any_grid(
            top_directory_name=ruc_directory_name,
            init_time_unix_sec=init_times_unix_sec[i],
            lead_time_hours=0, raise_error_if_missing=True
        )

    for i in range(num_examples):
        _plot_rapruc_one_example(
            full_storm_id_string=full_storm_id_strings[i],
            storm_time_unix_sec=storm_times_unix_sec[i],
            top_tracking_dir_name=top_tracking_dir_name,
            latitude_buffer_deg=latitude_buffer_deg,
            longitude_buffer_deg=longitude_buffer_deg,
            lead_time_seconds=lead_time_seconds,
            field_name_grib1=field_name_grib1, output_dir_name=output_dir_name,
            rap_file_name=rap_file_names[i], ruc_file_name=ruc_file_names[i]
        )


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        storm_metafile_name=getattr(INPUT_ARG_OBJECT, STORM_METAFILE_ARG_NAME),
        top_tracking_dir_name=getattr(INPUT_ARG_OBJECT, TRACKING_DIR_ARG_NAME),
        latitude_buffer_deg=getattr(INPUT_ARG_OBJECT, LATITUDE_BUFFER_ARG_NAME),
        longitude_buffer_deg=getattr(
            INPUT_ARG_OBJECT, LONGITUDE_BUFFER_ARG_NAME
        ),
        rap_directory_name=getattr(INPUT_ARG_OBJECT, RAP_DIRECTORY_ARG_NAME),
        ruc_directory_name=getattr(INPUT_ARG_OBJECT, RUC_DIRECTORY_ARG_NAME),
        lead_time_seconds=getattr(INPUT_ARG_OBJECT, LEAD_TIME_ARG_NAME),
        lag_time_seconds=getattr(INPUT_ARG_OBJECT, LAG_TIME_ARG_NAME),
        field_name_grib1=getattr(INPUT_ARG_OBJECT, FIELD_ARG_NAME),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
