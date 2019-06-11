"""Plots examples (storm objects) with surrounding context.

Specifically, for each storm object, this script plots surrounding radar data
and tornado reports for near-past and near-future times.
"""

import copy
import os.path
import argparse
from itertools import chain
import numpy
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as pyplot
from gewittergefahr.gg_io import myrorss_and_mrms_io
from gewittergefahr.gg_io import storm_tracking_io as tracking_io
from gewittergefahr.gg_utils import radar_utils
from gewittergefahr.gg_utils import linkage
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import target_val_utils
from gewittergefahr.gg_utils import echo_top_tracking
from gewittergefahr.gg_utils import temporal_tracking
from gewittergefahr.gg_utils import radar_sparse_to_full as radar_s2f
from gewittergefahr.gg_utils import storm_tracking_utils as tracking_utils
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.deep_learning import cnn
from gewittergefahr.deep_learning import model_activation
from gewittergefahr.deep_learning import model_interpretation
from gewittergefahr.deep_learning import training_validation_io as trainval_io
from gewittergefahr.plotting import plotting_utils
from gewittergefahr.plotting import radar_plotting
from gewittergefahr.plotting import storm_plotting
from gewittergefahr.plotting import imagemagick_utils

# TODO(thunderhoser): Deal with both tornado occurrence and genesis as target
# variables.

TIME_INTERVAL_SECONDS = 300
TIME_FORMAT = '%Y-%m-%d-%H%M%S'
SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

TORNADO_MARKER_TYPE = '^'
TORNADO_MARKER_SIZE = 16
TORNADO_MARKER_EDGE_WIDTH = 1
TORNADO_MARKER_COLOUR = numpy.full(3, 0.)

TITLE_FONT_SIZE = 16
FONT_SIZE = 16

NUM_PARALLELS = 8
NUM_MERIDIANS = 6
BORDER_COLOUR = numpy.full(3, 0.)
FIGURE_RESOLUTION_DPI = 300

ACTIVATION_FILE_ARG_NAME = 'input_activation_file_name'
TORNADO_DIR_ARG_NAME = 'input_tornado_dir_name'
TRACKING_DIR_ARG_NAME = 'input_tracking_dir_name'
MYRORSS_DIR_ARG_NAME = 'input_myrorss_dir_name'
RADAR_FIELD_ARG_NAME = 'radar_field_name'
RADAR_HEIGHT_ARG_NAME = 'radar_height_m_asl'
LATITUDE_BUFFER_ARG_NAME = 'latitude_buffer_deg'
LONGITUDE_BUFFER_ARG_NAME = 'longitude_buffer_deg'
TIME_BUFFER_ARG_NAME = 'time_buffer_seconds'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

ACTIVATION_FILE_HELP_STRING = (
    'Path to activation file, containing model predictions for one or more '
    'examples (storm objects).  Will be read by `model_activation.read_file`.')

TORNADO_DIR_HELP_STRING = (
    'Name of directory with tornado reports.  Files therein will be found by '
    '`tornado_io.find_processed_file` and read by '
    '`tornado_io.read_processed_file`.')

TRACKING_DIR_HELP_STRING = (
    'Name of top-level directory with storm-tracking files.  Files therein will'
    ' be found by `storm_tracking_io.find_file` and read by '
    '`storm_tracking_io.read_file`.')

MYRORSS_DIR_HELP_STRING = (
    'Name of top-level directory with MYRORSS-formatted radar data.  Files '
    'therein will be found by `myrorss_and_mrms_io.find_raw_file` and read by '
    '`myrorss_and_mrms_io.read_data_from_sparse_grid_file`.')

RADAR_FIELD_HELP_STRING = (
    'Will plot this radar field around storm object.  Must be accepted by '
    '`radar_utils.check_field_name`.')

RADAR_HEIGHT_HELP_STRING = (
    'Height of radar field (required only if field is "{0:s}").'
).format(radar_utils.REFL_NAME)

LATITUDE_BUFFER_HELP_STRING = (
    'Latitude buffer (deg N).  Will plot this much latitude around the edge of '
    'the storm object at each time step.')

LONGITUDE_BUFFER_HELP_STRING = (
    'Longitude buffer (deg E).  Will plot this much longitude around the edge '
    'of the storm object at each time step.')

TIME_BUFFER_HELP_STRING = (
    'Time buffer.  Will plot radar fields for this many seconds before and '
    'after storm object.')

OUTPUT_DIR_HELP_STRING = (
    'Name of output directory.  Figures will be saved here.')

DEFAULT_TORNADO_DIR_NAME = (
    '/condo/swatwork/ralager/tornado_observations/processed')
DEFAULT_TRACKING_DIR_NAME = (
    '/condo/swatcommon/common/gridrad_final/myrorss_format/new_tracks/'
    'reanalyzed')
DEFAULT_MYRORSS_DIR_NAME = (
    '/condo/swatcommon/common/gridrad_final/myrorss_format')

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + ACTIVATION_FILE_ARG_NAME, type=str, required=True,
    help=ACTIVATION_FILE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + TORNADO_DIR_ARG_NAME, type=str, required=False,
    default=DEFAULT_TORNADO_DIR_NAME, help=TORNADO_DIR_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + TRACKING_DIR_ARG_NAME, type=str, required=False,
    default=DEFAULT_TRACKING_DIR_NAME, help=TRACKING_DIR_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + MYRORSS_DIR_ARG_NAME, type=str, required=False,
    default=DEFAULT_MYRORSS_DIR_NAME, help=MYRORSS_DIR_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + RADAR_FIELD_ARG_NAME, type=str, required=False,
    default=radar_utils.ECHO_TOP_40DBZ_NAME, help=RADAR_FIELD_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + RADAR_HEIGHT_ARG_NAME, type=str, required=False,
    default=-1, help=RADAR_HEIGHT_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + LATITUDE_BUFFER_ARG_NAME, type=float, required=False,
    default=0.5, help=LATITUDE_BUFFER_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + LONGITUDE_BUFFER_ARG_NAME, type=float, required=False,
    default=0.5, help=LONGITUDE_BUFFER_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + TIME_BUFFER_ARG_NAME, type=int, required=False,
    default=1800, help=TIME_BUFFER_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING)


def _get_plotting_limits(storm_object_table, latitude_buffer_deg,
                         longitude_buffer_deg):
    """Returns plotting limits (lat-long box).

    :param storm_object_table: See doc for `_plot_one_example_one_time`.
    :param latitude_buffer_deg: See documentation at top of file.
    :param longitude_buffer_deg: Same.
    :return: latitude_limits_deg: length-2 numpy array with [min, max]
        latitudes in deg N.
    :return: longitude_limits_deg: length-2 numpy array with [min, max]
        longitudes in deg E.
    """

    vertex_latitudes_deg_2d_list = [
        numpy.array(p.exterior.xy[1]) for p in
        storm_object_table[tracking_utils.LATLNG_POLYGON_COLUMN].values
    ]

    vertex_longitudes_deg_2d_list = [
        numpy.array(p.exterior.xy[0]) for p in
        storm_object_table[tracking_utils.LATLNG_POLYGON_COLUMN].values
    ]

    vertex_latitudes_deg = numpy.concatenate(
        tuple(vertex_latitudes_deg_2d_list)
    )

    vertex_longitudes_deg = numpy.concatenate(
        tuple(vertex_longitudes_deg_2d_list)
    )

    min_plot_latitude_deg = (
        numpy.min(vertex_latitudes_deg) - latitude_buffer_deg
    )
    max_plot_latitude_deg = (
        numpy.max(vertex_latitudes_deg) + latitude_buffer_deg
    )
    min_plot_longitude_deg = (
        numpy.min(vertex_longitudes_deg) - longitude_buffer_deg
    )
    max_plot_longitude_deg = (
        numpy.max(vertex_longitudes_deg) + longitude_buffer_deg
    )

    return (
        numpy.array([min_plot_latitude_deg, max_plot_latitude_deg]),
        numpy.array([min_plot_longitude_deg, max_plot_longitude_deg])
    )


def _plot_one_example_one_time(
        storm_object_table, tornado_table, top_myrorss_dir_name,
        radar_field_name, radar_height_m_asl, latitude_buffer_deg,
        longitude_buffer_deg):
    """Plots one example with surrounding context at one time.

    :param storm_object_table: pandas DataFrame, containing only storm objects
        at one time with the relevant primary ID.  Columns are documented in
        `storm_tracking_io.write_file`.
    :param tornado_table: pandas DataFrame created by
        `linkage._read_input_tornado_reports`.
    :param top_myrorss_dir_name: See documentation at top of file.
    :param radar_field_name: Same.
    :param radar_height_m_asl: Same.
    :param latitude_buffer_deg: Same.
    :param longitude_buffer_deg: Same.
    """

    latitude_limits_deg, longitude_limits_deg = _get_plotting_limits(
        storm_object_table=storm_object_table,
        latitude_buffer_deg=latitude_buffer_deg,
        longitude_buffer_deg=longitude_buffer_deg)

    min_plot_latitude_deg = latitude_limits_deg[0]
    max_plot_latitude_deg = latitude_limits_deg[1]
    min_plot_longitude_deg = longitude_limits_deg[0]
    max_plot_longitude_deg = longitude_limits_deg[1]

    valid_time_unix_sec = storm_object_table[
        tracking_utils.VALID_TIME_COLUMN].values[0]

    # TODO(thunderhoser): The "3600" is a HACK.
    tornado_table = tornado_table.loc[
        (tornado_table[linkage.EVENT_TIME_COLUMN] >= valid_time_unix_sec) &
        (tornado_table[linkage.EVENT_TIME_COLUMN] <= valid_time_unix_sec + 3600)
    ]

    tornado_table = tornado_table.loc[
        (tornado_table[linkage.EVENT_LATITUDE_COLUMN] >= min_plot_latitude_deg)
        &
        (tornado_table[linkage.EVENT_LATITUDE_COLUMN] <= max_plot_latitude_deg)
    ]

    tornado_table = tornado_table.loc[
        (tornado_table[linkage.EVENT_LONGITUDE_COLUMN] >=
         min_plot_longitude_deg)
        &
        (tornado_table[linkage.EVENT_LONGITUDE_COLUMN] <=
         max_plot_longitude_deg)
    ]

    radar_file_name = myrorss_and_mrms_io.find_raw_file(
        top_directory_name=top_myrorss_dir_name,
        spc_date_string=time_conversion.time_to_spc_date_string(
            valid_time_unix_sec),
        unix_time_sec=valid_time_unix_sec,
        data_source=radar_utils.MYRORSS_SOURCE_ID,
        field_name=radar_field_name, height_m_asl=radar_height_m_asl,
        raise_error_if_missing=True)

    print('Reading data from: "{0:s}"...'.format(radar_file_name))

    radar_metadata_dict = myrorss_and_mrms_io.read_metadata_from_raw_file(
        netcdf_file_name=radar_file_name,
        data_source=radar_utils.MYRORSS_SOURCE_ID)

    sparse_grid_table = (
        myrorss_and_mrms_io.read_data_from_sparse_grid_file(
            netcdf_file_name=radar_file_name,
            field_name_orig=radar_metadata_dict[
                myrorss_and_mrms_io.FIELD_NAME_COLUMN_ORIG],
            data_source=radar_utils.MYRORSS_SOURCE_ID,
            sentinel_values=radar_metadata_dict[
                radar_utils.SENTINEL_VALUE_COLUMN]
        )
    )

    radar_matrix, grid_point_latitudes_deg, grid_point_longitudes_deg = (
        radar_s2f.sparse_to_full_grid(
            sparse_grid_table=sparse_grid_table,
            metadata_dict=radar_metadata_dict)
    )

    radar_matrix = numpy.flip(radar_matrix, axis=0)
    grid_point_latitudes_deg = grid_point_latitudes_deg[::-1]

    axes_object, basemap_object = (
        plotting_utils.create_equidist_cylindrical_map(
            min_latitude_deg=min_plot_latitude_deg,
            max_latitude_deg=max_plot_latitude_deg,
            min_longitude_deg=min_plot_longitude_deg,
            max_longitude_deg=max_plot_longitude_deg, resolution_string='i'
        )[1:]
    )

    plotting_utils.plot_coastlines(
        basemap_object=basemap_object, axes_object=axes_object,
        line_colour=BORDER_COLOUR)

    plotting_utils.plot_countries(
        basemap_object=basemap_object, axes_object=axes_object,
        line_colour=BORDER_COLOUR)

    plotting_utils.plot_states_and_provinces(
        basemap_object=basemap_object, axes_object=axes_object,
        line_colour=BORDER_COLOUR)

    plotting_utils.plot_parallels(
        basemap_object=basemap_object, axes_object=axes_object,
        num_parallels=NUM_PARALLELS)

    plotting_utils.plot_meridians(
        basemap_object=basemap_object, axes_object=axes_object,
        num_meridians=NUM_MERIDIANS)

    radar_plotting.plot_latlng_grid(
        field_matrix=radar_matrix, field_name=radar_field_name,
        axes_object=axes_object,
        min_grid_point_latitude_deg=numpy.min(grid_point_latitudes_deg),
        min_grid_point_longitude_deg=numpy.min(grid_point_longitudes_deg),
        latitude_spacing_deg=numpy.diff(grid_point_latitudes_deg[:2])[0],
        longitude_spacing_deg=numpy.diff(grid_point_longitudes_deg[:2])[0]
    )

    storm_plotting.plot_storm_outlines(
        storm_object_table=storm_object_table, axes_object=axes_object,
        basemap_object=basemap_object, line_width=2, line_colour='k')

    tornado_latitudes_deg = tornado_table[linkage.EVENT_LATITUDE_COLUMN].values
    tornado_longitudes_deg = tornado_table[
        linkage.EVENT_LONGITUDE_COLUMN].values

    tornado_times_unix_sec = tornado_table[linkage.EVENT_TIME_COLUMN].values
    tornado_time_strings = [
        time_conversion.unix_sec_to_string(t, TIME_FORMAT)
        for t in tornado_times_unix_sec
    ]

    axes_object.plot(
        tornado_longitudes_deg, tornado_latitudes_deg, linestyle='None',
        marker=TORNADO_MARKER_TYPE, markersize=TORNADO_MARKER_SIZE,
        markeredgewidth=TORNADO_MARKER_EDGE_WIDTH,
        markerfacecolor=plotting_utils.colour_from_numpy_to_tuple(
            TORNADO_MARKER_COLOUR),
        markeredgecolor=plotting_utils.colour_from_numpy_to_tuple(
            TORNADO_MARKER_COLOUR)
    )

    num_tornadoes = len(tornado_latitudes_deg)

    for j in range(num_tornadoes):
        axes_object.text(
            tornado_longitudes_deg[j], tornado_latitudes_deg[j],
            tornado_time_strings[j], fontsize=FONT_SIZE,
            color=TORNADO_MARKER_COLOUR,
            horizontalalignment='left', verticalalignment='top')


def _find_tracking_files_one_example(
        valid_time_unix_sec, top_tracking_dir_name, time_buffer_seconds):
    """Finds tracking files needed to make plots for one example.

    :param valid_time_unix_sec: Valid time for example.
    :param top_tracking_dir_name: See documentation at top of file.
    :param time_buffer_seconds: Same.
    :return: tracking_file_names: 1-D list of paths to tracking files.
    :raises: ValueError: if no tracking files are found.
    """

    first_time_unix_sec = valid_time_unix_sec - time_buffer_seconds
    last_time_unix_sec = valid_time_unix_sec + time_buffer_seconds

    first_spc_date_string = time_conversion.time_to_spc_date_string(
        first_time_unix_sec - TIME_INTERVAL_SECONDS)
    last_spc_date_string = time_conversion.time_to_spc_date_string(
        last_time_unix_sec + TIME_INTERVAL_SECONDS)
    spc_date_strings = time_conversion.get_spc_dates_in_range(
        first_spc_date_string=first_spc_date_string,
        last_spc_date_string=last_spc_date_string)

    tracking_file_names = []

    for this_spc_date_string in spc_date_strings:
        these_file_names = tracking_io.find_files_one_spc_date(
            top_tracking_dir_name=top_tracking_dir_name,
            tracking_scale_metres2=
            echo_top_tracking.DUMMY_TRACKING_SCALE_METRES2,
            source_name=tracking_utils.SEGMOTION_NAME,
            spc_date_string=this_spc_date_string, raise_error_if_missing=False
        )[0]

        tracking_file_names += these_file_names

    if len(tracking_file_names) == 0:
        error_string = (
            'Cannot find any tracking files for SPC dates "{0:s}" to "{1:s}".'
        ).format(first_spc_date_string, last_spc_date_string)

        raise ValueError(error_string)

    tracking_times_unix_sec = numpy.array([
        tracking_io.file_name_to_time(f) for f in tracking_file_names
    ], dtype=int)

    sort_indices = numpy.argsort(tracking_times_unix_sec)
    tracking_times_unix_sec = tracking_times_unix_sec[sort_indices]
    tracking_file_names = [tracking_file_names[k] for k in sort_indices]

    these_indices = numpy.where(
        tracking_times_unix_sec <= first_time_unix_sec
    )[0]

    if len(these_indices) == 0:
        first_index = 0
    else:
        first_index = these_indices[-1]

    these_indices = numpy.where(
        tracking_times_unix_sec >= last_time_unix_sec
    )[0]

    if len(these_indices) == 0:
        last_index = len(tracking_file_names) - 1
    else:
        last_index = these_indices[0]

    return tracking_file_names[first_index:(last_index + 1)]


def _plot_one_example(
        full_id_string, storm_time_unix_sec, forecast_probability,
        tornado_dir_name, top_tracking_dir_name, top_myrorss_dir_name,
        radar_field_name, radar_height_m_asl, latitude_buffer_deg,
        longitude_buffer_deg, time_buffer_seconds, top_output_dir_name):
    """Plots one example with surrounding context at several times.

    :param full_id_string: Full storm ID.
    :param storm_time_unix_sec: Storm time.
    :param forecast_probability: Forecast tornado probability for this example.
    :param tornado_dir_name: See documentation at top of file.
    :param top_tracking_dir_name: Same.
    :param top_myrorss_dir_name: Same.
    :param radar_field_name: Same.
    :param radar_height_m_asl: Same.
    :param latitude_buffer_deg: Same.
    :param longitude_buffer_deg: Same.
    :param time_buffer_seconds: Same.
    :param top_output_dir_name: Same.
    """

    storm_time_string = time_conversion.unix_sec_to_string(
        storm_time_unix_sec, TIME_FORMAT)

    primary_id_string = temporal_tracking.full_to_partial_ids(
        [full_id_string]
    )[0][0]

    output_dir_name = '{0:s}/{1:s}'.format(top_output_dir_name, full_id_string)
    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=output_dir_name)

    tracking_file_names = _find_tracking_files_one_example(
        valid_time_unix_sec=storm_time_unix_sec,
        top_tracking_dir_name=top_tracking_dir_name,
        time_buffer_seconds=time_buffer_seconds)

    tracking_times_unix_sec = numpy.array([
        tracking_io.file_name_to_time(f) for f in tracking_file_names
    ], dtype=int)

    # TODO(thunderhoser): Get rid of the "3600" hack.
    tornado_table = linkage._read_input_tornado_reports(
        input_directory_name=tornado_dir_name,
        storm_times_unix_sec=tracking_times_unix_sec,
        max_time_before_storm_start_sec=0, max_time_after_storm_end_sec=3600,
        genesis_only=True)

    for this_tracking_file_name in tracking_file_names:
        print('Reading data from: "{0:s}"...'.format(this_tracking_file_name))
        this_storm_object_table = tracking_io.read_file(this_tracking_file_name)

        this_storm_object_table = this_storm_object_table.loc[
            this_storm_object_table[tracking_utils.PRIMARY_ID_COLUMN] ==
            primary_id_string
        ]

        _plot_one_example_one_time(
            storm_object_table=this_storm_object_table,
            tornado_table=copy.deepcopy(tornado_table),
            top_myrorss_dir_name=top_myrorss_dir_name,
            radar_field_name=radar_field_name,
            radar_height_m_asl=radar_height_m_asl,
            latitude_buffer_deg=latitude_buffer_deg,
            longitude_buffer_deg=longitude_buffer_deg)

        this_title_string = (
            'Storm object "{0:s}" at {1:s} ... forecast prob = {2:.3f}'
        ).format(full_id_string, storm_time_string, forecast_probability)
        pyplot.title(this_title_string)

        this_file_name = '{0:s}/{1:s}.jpg'.format(
            output_dir_name, storm_time_string)

        print('Saving figure to file: "{0:s}"...\n'.format(this_file_name))
        pyplot.savefig(this_file_name, dpi=FIGURE_RESOLUTION_DPI)
        pyplot.close()

        imagemagick_utils.trim_whitespace(
            input_file_name=this_file_name, output_file_name=this_file_name)


def _run(activation_file_name, tornado_dir_name, top_tracking_dir_name,
         top_myrorss_dir_name, radar_field_name, radar_height_m_asl,
         latitude_buffer_deg, longitude_buffer_deg, time_buffer_seconds,
         top_output_dir_name):
    """Plots examples (storm objects) with surrounding context.

    This is effectively the main method.

    :param activation_file_name: See documentation at top of file.
    :param tornado_dir_name: Same.
    :param top_tracking_dir_name: Same.
    :param top_myrorss_dir_name: Same.
    :param radar_field_name: Same.
    :param radar_height_m_asl: Same.
    :param latitude_buffer_deg: Same.
    :param longitude_buffer_deg: Same.
    :param time_buffer_seconds: Same.
    :param top_output_dir_name: Same.
    :raises: ValueError: if activation file contains activations of some
        intermediate model component, rather than final predictions.
    :raises: ValueError: if target variable is not related to tornadogenesis.
    """

    print('Reading data from: "{0:s}"...'.format(activation_file_name))
    activation_matrix, activation_dict = model_activation.read_file(
        activation_file_name)

    component_type_string = activation_dict[model_activation.COMPONENT_TYPE_KEY]

    if (component_type_string !=
            model_interpretation.CLASS_COMPONENT_TYPE_STRING):
        error_string = (
            'Activation file should contain final predictions (component type '
            '"{0:s}").  Instead, component type is "{1:s}".'
        ).format(
            model_interpretation.CLASS_COMPONENT_TYPE_STRING,
            component_type_string
        )

        raise ValueError(error_string)

    forecast_probabilities = numpy.squeeze(activation_matrix)
    num_storm_objects = len(forecast_probabilities)

    model_file_name = activation_dict[model_activation.MODEL_FILE_NAME_KEY]
    model_metafile_name = '{0:s}/model_metadata.p'.format(
        os.path.split(model_file_name)[0]
    )

    print('Reading model metadata from: "{0:s}"...'.format(model_metafile_name))
    model_metadata_dict = cnn.read_model_metadata(model_metafile_name)

    training_option_dict = model_metadata_dict[cnn.TRAINING_OPTION_DICT_KEY]
    target_name = training_option_dict[trainval_io.TARGET_NAME_KEY]
    target_param_dict = target_val_utils.target_name_to_params(target_name)
    event_type_string = target_param_dict[target_val_utils.EVENT_TYPE_KEY]

    if event_type_string != linkage.TORNADO_EVENT_STRING:
        error_string = (
            'Target variable should be related to tornadogenesis.  Instead, got'
            ' "{0:s}".'
        ).format(target_name)

        raise ValueError(error_string)

    print(SEPARATOR_STRING)

    for i in range(num_storm_objects):
        _plot_one_example(
            full_id_string=activation_dict[model_activation.FULL_IDS_KEY][i],
            storm_time_unix_sec=activation_dict[
                model_activation.STORM_TIMES_KEY][i],
            forecast_probability=forecast_probabilities[i],
            tornado_dir_name=tornado_dir_name,
            top_tracking_dir_name=top_tracking_dir_name,
            top_myrorss_dir_name=top_myrorss_dir_name,
            radar_field_name=radar_field_name,
            radar_height_m_asl=radar_height_m_asl,
            latitude_buffer_deg=latitude_buffer_deg,
            longitude_buffer_deg=longitude_buffer_deg,
            time_buffer_seconds=time_buffer_seconds,
            top_output_dir_name=top_output_dir_name)

        if i != num_storm_objects - 1:
            print(SEPARATOR_STRING)


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        activation_file_name=getattr(
            INPUT_ARG_OBJECT, ACTIVATION_FILE_ARG_NAME),
        tornado_dir_name=getattr(INPUT_ARG_OBJECT, TORNADO_DIR_ARG_NAME),
        top_tracking_dir_name=getattr(INPUT_ARG_OBJECT, TRACKING_DIR_ARG_NAME),
        top_myrorss_dir_name=getattr(INPUT_ARG_OBJECT, MYRORSS_DIR_ARG_NAME),
        radar_field_name=getattr(INPUT_ARG_OBJECT, RADAR_FIELD_ARG_NAME),
        radar_height_m_asl=getattr(INPUT_ARG_OBJECT, RADAR_HEIGHT_ARG_NAME),
        latitude_buffer_deg=getattr(INPUT_ARG_OBJECT, LATITUDE_BUFFER_ARG_NAME),
        longitude_buffer_deg=getattr(
            INPUT_ARG_OBJECT, LONGITUDE_BUFFER_ARG_NAME),
        time_buffer_seconds=getattr(INPUT_ARG_OBJECT, TIME_BUFFER_ARG_NAME),
        top_output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
