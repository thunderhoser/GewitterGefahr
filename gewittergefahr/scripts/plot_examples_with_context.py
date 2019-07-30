"""Plots examples (storm objects) with surrounding context.

Specifically, for each storm object, this script plots surrounding radar data
and tornado reports for near-past and near-future times.
"""

import copy
import os.path
import argparse
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

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

TIME_INTERVAL_SECONDS = 300
TIME_FORMAT = '%Y-%m-%d-%H%M%S'
FORECAST_PROBABILITY_COLUMN = 'forecast_probability'

PROBABILITY_BACKGROUND_COLOUR = numpy.array([117, 112, 179], dtype=float) / 255
PROBABILITY_BACKGROUND_OPACITY = 0.75
PROBABILITY_FONT_COLOUR = numpy.full(3, 0.)
# PROBABILITY_FONT_COLOUR = numpy.array([217, 95, 2], dtype=float) / 255

FONT_SIZE = 20
FONT_COLOUR = numpy.full(3, 0.)

TORNADO_TIME_FORMAT = '%H%MZ'
TORNADO_MARKER_TYPE = 'D'
TORNADO_MARKER_SIZE = 16
TORNADO_MARKER_EDGE_WIDTH = 1
TORNADO_MARKER_COLOUR = numpy.full(3, 0.)
# TORNADO_MARKER_COLOUR = numpy.array([228, 26, 28], dtype=float) / 255

NUM_PARALLELS = 8
NUM_MERIDIANS = 5
TITLE_FONT_SIZE = 16
BORDER_COLOUR = numpy.full(3, 0.)
FIGURE_RESOLUTION_DPI = 300

MAIN_ACTIVATION_FILE_ARG_NAME = 'input_main_activn_file_name'
AUX_ACTIVATION_FILE_ARG_NAME = 'input_aux_activn_file_name'
TORNADO_DIR_ARG_NAME = 'input_tornado_dir_name'
TRACKING_DIR_ARG_NAME = 'input_tracking_dir_name'
MYRORSS_DIR_ARG_NAME = 'input_myrorss_dir_name'
RADAR_FIELD_ARG_NAME = 'radar_field_name'
RADAR_HEIGHT_ARG_NAME = 'radar_height_m_asl'
LATITUDE_BUFFER_ARG_NAME = 'latitude_buffer_deg'
LONGITUDE_BUFFER_ARG_NAME = 'longitude_buffer_deg'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

MAIN_ACTIVATION_FILE_HELP_STRING = (
    'Path to main activation file (to be read by `model_activation.read_file`),'
    ' containing model predictions for one or more examples (storm objects).  '
    'Each storm object in this file will be plotted.')

AUX_ACTIVATION_FILE_HELP_STRING = (
    'Path to auxiliary activation file.  Must contain all examples in `{0:s}` '
    'and more.  Will be used to plot forecast probability next to each example.'
    '  If you do not want to plot forecast prob next to each example, leave '
    'this argument alone.'
).format(MAIN_ACTIVATION_FILE_ARG_NAME)

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
    '--' + MAIN_ACTIVATION_FILE_ARG_NAME, type=str, required=True,
    help=MAIN_ACTIVATION_FILE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + AUX_ACTIVATION_FILE_ARG_NAME, type=str, required=False, default='',
    help=AUX_ACTIVATION_FILE_HELP_STRING)

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
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING)


def _plot_one_example_one_time(
        storm_object_table, full_id_string, valid_time_unix_sec,
        tornado_table, top_myrorss_dir_name, radar_field_name,
        radar_height_m_asl, latitude_limits_deg, longitude_limits_deg):
    """Plots one example with surrounding context at one time.

    :param storm_object_table: pandas DataFrame, containing only storm objects
        at one time with the relevant primary ID.  Columns are documented in
        `storm_tracking_io.write_file`.
    :param full_id_string: Full ID of storm of interest.
    :param valid_time_unix_sec: Valid time.
    :param tornado_table: pandas DataFrame created by
        `linkage._read_input_tornado_reports`.
    :param top_myrorss_dir_name: See documentation at top of file.
    :param radar_field_name: Same.
    :param radar_height_m_asl: Same.
    :param latitude_limits_deg: See doc for `_get_plotting_limits`.
    :param longitude_limits_deg: Same.
    """

    min_plot_latitude_deg = latitude_limits_deg[0]
    max_plot_latitude_deg = latitude_limits_deg[1]
    min_plot_longitude_deg = longitude_limits_deg[0]
    max_plot_longitude_deg = longitude_limits_deg[1]

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
            max_longitude_deg=max_plot_longitude_deg, resolution_string='h'
        )[1:]
    )

    plotting_utils.plot_coastlines(
        basemap_object=basemap_object, axes_object=axes_object,
        line_colour=plotting_utils.DEFAULT_COUNTY_COLOUR)

    plotting_utils.plot_countries(
        basemap_object=basemap_object, axes_object=axes_object)

    plotting_utils.plot_states_and_provinces(
        basemap_object=basemap_object, axes_object=axes_object)

    # plotting_utils.plot_counties(
    #     basemap_object=basemap_object, axes_object=axes_object)

    plotting_utils.plot_parallels(
        basemap_object=basemap_object, axes_object=axes_object,
        num_parallels=NUM_PARALLELS, line_width=0)

    plotting_utils.plot_meridians(
        basemap_object=basemap_object, axes_object=axes_object,
        num_meridians=NUM_MERIDIANS, line_width=0)

    radar_plotting.plot_latlng_grid(
        field_matrix=radar_matrix, field_name=radar_field_name,
        axes_object=axes_object,
        min_grid_point_latitude_deg=numpy.min(grid_point_latitudes_deg),
        min_grid_point_longitude_deg=numpy.min(grid_point_longitudes_deg),
        latitude_spacing_deg=numpy.diff(grid_point_latitudes_deg[:2])[0],
        longitude_spacing_deg=numpy.diff(grid_point_longitudes_deg[:2])[0]
    )

    colour_map_object, colour_norm_object = (
        radar_plotting.get_default_colour_scheme(radar_field_name)
    )

    plotting_utils.plot_colour_bar(
        axes_object_or_matrix=axes_object, data_matrix=radar_matrix,
        colour_map_object=colour_map_object,
        colour_norm_object=colour_norm_object, orientation_string='horizontal',
        extend_min=False, extend_max=True, fraction_of_axis_length=0.8)

    first_list, second_list = temporal_tracking.full_to_partial_ids(
        [full_id_string]
    )
    primary_id_string = first_list[0]
    secondary_id_string = second_list[0]

    # Plot outlines of unrelated storms (with different primary IDs).
    this_storm_object_table = storm_object_table.loc[
        storm_object_table[tracking_utils.PRIMARY_ID_COLUMN] !=
        primary_id_string
    ]

    storm_plotting.plot_storm_outlines(
        storm_object_table=this_storm_object_table, axes_object=axes_object,
        basemap_object=basemap_object, line_width=2, line_colour='k',
        line_style='dashed')

    # Plot outlines of related storms (with the same primary ID).
    this_storm_object_table = storm_object_table.loc[
        (storm_object_table[tracking_utils.PRIMARY_ID_COLUMN] ==
         primary_id_string) &
        (storm_object_table[tracking_utils.SECONDARY_ID_COLUMN] !=
         secondary_id_string)
    ]

    this_num_storm_objects = len(this_storm_object_table.index)

    if this_num_storm_objects > 0:
        storm_plotting.plot_storm_outlines(
            storm_object_table=this_storm_object_table, axes_object=axes_object,
            basemap_object=basemap_object, line_width=2, line_colour='k',
            line_style='solid')

        for j in range(len(this_storm_object_table)):
            axes_object.text(
                this_storm_object_table[
                    tracking_utils.CENTROID_LONGITUDE_COLUMN].values[j],
                this_storm_object_table[
                    tracking_utils.CENTROID_LATITUDE_COLUMN].values[j],
                'P', fontsize=FONT_SIZE, color=FONT_COLOUR, fontweight='bold',
                horizontalalignment='center', verticalalignment='center')

    # Plot outline of storm of interest (same secondary ID).
    this_storm_object_table = storm_object_table.loc[
        storm_object_table[tracking_utils.SECONDARY_ID_COLUMN] ==
        secondary_id_string
    ]

    storm_plotting.plot_storm_outlines(
        storm_object_table=this_storm_object_table, axes_object=axes_object,
        basemap_object=basemap_object, line_width=4, line_colour='k',
        line_style='solid')

    this_num_storm_objects = len(this_storm_object_table.index)

    plot_forecast = (
        this_num_storm_objects > 0 and
        FORECAST_PROBABILITY_COLUMN in list(this_storm_object_table)
    )

    if plot_forecast:
        this_polygon_object_latlng = this_storm_object_table[
            tracking_utils.LATLNG_POLYGON_COLUMN].values[0]

        this_latitude_deg = numpy.min(
            numpy.array(this_polygon_object_latlng.exterior.xy[1])
        )

        this_longitude_deg = this_storm_object_table[
            tracking_utils.CENTROID_LONGITUDE_COLUMN].values[0]

        label_string = 'Prob = {0:.3f}\nat {1:s}'.format(
            this_storm_object_table[FORECAST_PROBABILITY_COLUMN].values[0],
            time_conversion.unix_sec_to_string(
                valid_time_unix_sec, TORNADO_TIME_FORMAT)
        )

        bounding_box_dict = {
            'facecolor': plotting_utils.colour_from_numpy_to_tuple(
                PROBABILITY_BACKGROUND_COLOUR),
            'alpha': PROBABILITY_BACKGROUND_OPACITY,
            'edgecolor': 'k',
            'linewidth': 1
        }

        axes_object.text(
            this_longitude_deg, this_latitude_deg, label_string,
            fontsize=FONT_SIZE,
            color=plotting_utils.colour_from_numpy_to_tuple(
                PROBABILITY_FONT_COLOUR),
            fontweight='bold', bbox=bounding_box_dict,
            horizontalalignment='center', verticalalignment='top', zorder=1e10)

    tornado_latitudes_deg = tornado_table[linkage.EVENT_LATITUDE_COLUMN].values
    tornado_longitudes_deg = tornado_table[
        linkage.EVENT_LONGITUDE_COLUMN].values

    tornado_times_unix_sec = tornado_table[linkage.EVENT_TIME_COLUMN].values
    tornado_time_strings = [
        time_conversion.unix_sec_to_string(t, TORNADO_TIME_FORMAT)
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
            tornado_longitudes_deg[j] + 0.02, tornado_latitudes_deg[j] - 0.02,
            tornado_time_strings[j], fontsize=FONT_SIZE,
            color=FONT_COLOUR, fontweight='bold',
            horizontalalignment='left', verticalalignment='top')


def _find_tracking_files_one_example(
        top_tracking_dir_name, valid_time_unix_sec, target_name):
    """Finds tracking files needed to make plots for one example.

    :param top_tracking_dir_name: See documentation at top of file.
    :param valid_time_unix_sec: Valid time for example.
    :param target_name: Name of target variable.
    :return: tracking_file_names: 1-D list of paths to tracking files.
    :raises: ValueError: if no tracking files are found.
    """

    target_param_dict = target_val_utils.target_name_to_params(target_name)
    min_lead_time_seconds = target_param_dict[
        target_val_utils.MIN_LEAD_TIME_KEY]
    max_lead_time_seconds = target_param_dict[
        target_val_utils.MAX_LEAD_TIME_KEY]

    first_time_unix_sec = valid_time_unix_sec + min_lead_time_seconds
    last_time_unix_sec = valid_time_unix_sec + max_lead_time_seconds

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


def _plot_one_example(
        full_id_string, storm_time_unix_sec, target_name, forecast_probability,
        tornado_dir_name, top_tracking_dir_name, top_myrorss_dir_name,
        radar_field_name, radar_height_m_asl, latitude_buffer_deg,
        longitude_buffer_deg, top_output_dir_name,
        aux_forecast_probabilities=None, aux_activation_dict=None):
    """Plots one example with surrounding context at several times.

    N = number of storm objects read from auxiliary activation file

    :param full_id_string: Full storm ID.
    :param storm_time_unix_sec: Storm time.
    :param target_name: Name of target variable.
    :param forecast_probability: Forecast tornado probability for this example.
    :param tornado_dir_name: See documentation at top of file.
    :param top_tracking_dir_name: Same.
    :param top_myrorss_dir_name: Same.
    :param radar_field_name: Same.
    :param radar_height_m_asl: Same.
    :param latitude_buffer_deg: Same.
    :param longitude_buffer_deg: Same.
    :param top_output_dir_name: Same.
    :param aux_forecast_probabilities: length-N numpy array of forecast
        probabilities.  If this is None, will not plot forecast probs in maps.
    :param aux_activation_dict: Dictionary returned by
        `model_activation.read_file` from auxiliary file.  If this is None, will
        not plot forecast probs in maps.
    """

    storm_time_string = time_conversion.unix_sec_to_string(
        storm_time_unix_sec, TIME_FORMAT)

    # Create output directory for this example.
    output_dir_name = '{0:s}/{1:s}_{2:s}'.format(
        top_output_dir_name, full_id_string, storm_time_string)
    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=output_dir_name)

    # Find tracking files.
    tracking_file_names = _find_tracking_files_one_example(
        valid_time_unix_sec=storm_time_unix_sec,
        top_tracking_dir_name=top_tracking_dir_name, target_name=target_name)

    tracking_times_unix_sec = numpy.array([
        tracking_io.file_name_to_time(f) for f in tracking_file_names
    ], dtype=int)

    tracking_time_strings = [
        time_conversion.unix_sec_to_string(t, TIME_FORMAT)
        for t in tracking_times_unix_sec
    ]

    # Read tracking files.
    storm_object_table = tracking_io.read_many_files(tracking_file_names)
    print('\n')

    if aux_activation_dict is not None:
        these_indices = tracking_utils.find_storm_objects(
            all_id_strings=aux_activation_dict[model_activation.FULL_IDS_KEY],
            all_times_unix_sec=aux_activation_dict[
                model_activation.STORM_TIMES_KEY],
            id_strings_to_keep=storm_object_table[
                tracking_utils.FULL_ID_COLUMN].values.tolist(),
            times_to_keep_unix_sec=storm_object_table[
                tracking_utils.VALID_TIME_COLUMN].values,
            allow_missing=True
        )

        storm_object_probs = numpy.array([
            aux_forecast_probabilities[k] if k >= 0 else numpy.nan
            for k in these_indices
        ])

        storm_object_table = storm_object_table.assign(**{
            FORECAST_PROBABILITY_COLUMN: storm_object_probs
        })

    primary_id_string = temporal_tracking.full_to_partial_ids(
        [full_id_string]
    )[0][0]

    this_storm_object_table = storm_object_table.loc[
        storm_object_table[tracking_utils.PRIMARY_ID_COLUMN] ==
        primary_id_string
    ]

    latitude_limits_deg, longitude_limits_deg = _get_plotting_limits(
        storm_object_table=this_storm_object_table,
        latitude_buffer_deg=latitude_buffer_deg,
        longitude_buffer_deg=longitude_buffer_deg)

    storm_min_latitudes_deg = numpy.array([
        numpy.min(numpy.array(p.exterior.xy[1])) for p in
        storm_object_table[tracking_utils.LATLNG_POLYGON_COLUMN].values
    ])

    storm_max_latitudes_deg = numpy.array([
        numpy.max(numpy.array(p.exterior.xy[1])) for p in
        storm_object_table[tracking_utils.LATLNG_POLYGON_COLUMN].values
    ])

    storm_min_longitudes_deg = numpy.array([
        numpy.min(numpy.array(p.exterior.xy[0])) for p in
        storm_object_table[tracking_utils.LATLNG_POLYGON_COLUMN].values
    ])

    storm_max_longitudes_deg = numpy.array([
        numpy.max(numpy.array(p.exterior.xy[0])) for p in
        storm_object_table[tracking_utils.LATLNG_POLYGON_COLUMN].values
    ])

    min_latitude_flags = numpy.logical_and(
        storm_min_latitudes_deg >= latitude_limits_deg[0],
        storm_min_latitudes_deg <= latitude_limits_deg[1]
    )

    max_latitude_flags = numpy.logical_and(
        storm_max_latitudes_deg >= latitude_limits_deg[0],
        storm_max_latitudes_deg <= latitude_limits_deg[1]
    )

    latitude_flags = numpy.logical_or(min_latitude_flags, max_latitude_flags)

    min_longitude_flags = numpy.logical_and(
        storm_min_longitudes_deg >= longitude_limits_deg[0],
        storm_min_longitudes_deg <= longitude_limits_deg[1]
    )

    max_longitude_flags = numpy.logical_and(
        storm_max_longitudes_deg >= longitude_limits_deg[0],
        storm_max_longitudes_deg <= longitude_limits_deg[1]
    )

    longitude_flags = numpy.logical_or(min_longitude_flags, max_longitude_flags)
    good_indices = numpy.where(
        numpy.logical_and(latitude_flags, longitude_flags)
    )[0]

    storm_object_table = storm_object_table.iloc[good_indices]

    # Read tornado reports.
    target_param_dict = target_val_utils.target_name_to_params(target_name)
    min_lead_time_seconds = target_param_dict[
        target_val_utils.MIN_LEAD_TIME_KEY]
    max_lead_time_seconds = target_param_dict[
        target_val_utils.MAX_LEAD_TIME_KEY]

    tornado_table = linkage._read_input_tornado_reports(
        input_directory_name=tornado_dir_name,
        storm_times_unix_sec=numpy.array([storm_time_unix_sec], dtype=int),
        max_time_before_storm_start_sec=-1 * min_lead_time_seconds,
        max_time_after_storm_end_sec=max_lead_time_seconds,
        genesis_only=True)

    tornado_table = tornado_table.loc[
        (tornado_table[linkage.EVENT_LATITUDE_COLUMN] >= latitude_limits_deg[0])
        &
        (tornado_table[linkage.EVENT_LATITUDE_COLUMN] <= latitude_limits_deg[1])
    ]

    tornado_table = tornado_table.loc[
        (tornado_table[linkage.EVENT_LONGITUDE_COLUMN] >=
         longitude_limits_deg[0])
        &
        (tornado_table[linkage.EVENT_LONGITUDE_COLUMN] <=
         longitude_limits_deg[1])
    ]

    for i in range(len(tracking_file_names)):
        this_storm_object_table = storm_object_table.loc[
            storm_object_table[tracking_utils.VALID_TIME_COLUMN] ==
            tracking_times_unix_sec[i]
        ]

        _plot_one_example_one_time(
            storm_object_table=this_storm_object_table,
            full_id_string=full_id_string,
            valid_time_unix_sec=tracking_times_unix_sec[i],
            tornado_table=copy.deepcopy(tornado_table),
            top_myrorss_dir_name=top_myrorss_dir_name,
            radar_field_name=radar_field_name,
            radar_height_m_asl=radar_height_m_asl,
            latitude_limits_deg=latitude_limits_deg,
            longitude_limits_deg=longitude_limits_deg)

        if aux_activation_dict is None:
            this_title_string = (
                'Valid time = {0:s} ... forecast prob at {1:s} = {2:.3f}'
            ).format(
                tracking_time_strings[i], storm_time_string,
                forecast_probability
            )

            pyplot.title(this_title_string, fontsize=TITLE_FONT_SIZE)

        this_file_name = '{0:s}/{1:s}.jpg'.format(
            output_dir_name, tracking_time_strings[i]
        )

        print('Saving figure to file: "{0:s}"...\n'.format(this_file_name))
        pyplot.savefig(this_file_name, dpi=FIGURE_RESOLUTION_DPI)
        pyplot.close()

        imagemagick_utils.trim_whitespace(
            input_file_name=this_file_name, output_file_name=this_file_name)


def _run(main_activation_file_name, aux_activation_file_name, tornado_dir_name,
         top_tracking_dir_name, top_myrorss_dir_name, radar_field_name,
         radar_height_m_asl, latitude_buffer_deg, longitude_buffer_deg,
         top_output_dir_name):
    """Plots examples (storm objects) with surrounding context.

    This is effectively the main method.

    :param main_activation_file_name: See documentation at top of file.
    :param aux_activation_file_name: Same.
    :param tornado_dir_name: Same.
    :param top_tracking_dir_name: Same.
    :param top_myrorss_dir_name: Same.
    :param radar_field_name: Same.
    :param radar_height_m_asl: Same.
    :param latitude_buffer_deg: Same.
    :param longitude_buffer_deg: Same.
    :param top_output_dir_name: Same.
    :raises: ValueError: if activation file contains activations of some
        intermediate model component, rather than final predictions.
    :raises: ValueError: if target variable is not related to tornadogenesis.
    """

    if aux_activation_file_name in ['', 'None']:
        aux_activation_file_name = None

    print('Reading data from: "{0:s}"...'.format(main_activation_file_name))
    activation_matrix, activation_dict = model_activation.read_file(
        main_activation_file_name)

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

    if aux_activation_file_name is None:
        aux_forecast_probabilities = None
        aux_activation_dict = None
    else:
        print('Reading data from: "{0:s}"...'.format(aux_activation_file_name))
        this_matrix, aux_activation_dict = model_activation.read_file(
            aux_activation_file_name)

        aux_forecast_probabilities = numpy.squeeze(this_matrix)

    print(SEPARATOR_STRING)

    for i in range(num_storm_objects):
        _plot_one_example(
            full_id_string=activation_dict[model_activation.FULL_IDS_KEY][i],
            storm_time_unix_sec=activation_dict[
                model_activation.STORM_TIMES_KEY][i],
            target_name=target_name,
            forecast_probability=forecast_probabilities[i],
            tornado_dir_name=tornado_dir_name,
            top_tracking_dir_name=top_tracking_dir_name,
            top_myrorss_dir_name=top_myrorss_dir_name,
            radar_field_name=radar_field_name,
            radar_height_m_asl=radar_height_m_asl,
            latitude_buffer_deg=latitude_buffer_deg,
            longitude_buffer_deg=longitude_buffer_deg,
            top_output_dir_name=top_output_dir_name,
            aux_forecast_probabilities=aux_forecast_probabilities,
            aux_activation_dict=aux_activation_dict)

        if i != num_storm_objects - 1:
            print(SEPARATOR_STRING)


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        main_activation_file_name=getattr(
            INPUT_ARG_OBJECT, MAIN_ACTIVATION_FILE_ARG_NAME),
        aux_activation_file_name=getattr(
            INPUT_ARG_OBJECT, AUX_ACTIVATION_FILE_ARG_NAME),
        tornado_dir_name=getattr(INPUT_ARG_OBJECT, TORNADO_DIR_ARG_NAME),
        top_tracking_dir_name=getattr(INPUT_ARG_OBJECT, TRACKING_DIR_ARG_NAME),
        top_myrorss_dir_name=getattr(INPUT_ARG_OBJECT, MYRORSS_DIR_ARG_NAME),
        radar_field_name=getattr(INPUT_ARG_OBJECT, RADAR_FIELD_ARG_NAME),
        radar_height_m_asl=getattr(INPUT_ARG_OBJECT, RADAR_HEIGHT_ARG_NAME),
        latitude_buffer_deg=getattr(INPUT_ARG_OBJECT, LATITUDE_BUFFER_ARG_NAME),
        longitude_buffer_deg=getattr(
            INPUT_ARG_OBJECT, LONGITUDE_BUFFER_ARG_NAME),
        top_output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
