"""Methods for computing radar statistics.

These are usually spatial statistics based on values inside a storm object.
"""

import os.path
import warnings
import pickle
import numpy
import pandas
import scipy.stats
from gewittergefahr.gg_io import radar_io
from gewittergefahr.gg_io import storm_tracking_io as tracking_io
from gewittergefahr.gg_utils import radar_sparse_to_full as radar_s2f
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import dilation
from gewittergefahr.gg_utils import number_rounding as rounder
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.gg_utils import error_checking

TOLERANCE = 1e-6
DEFAULT_DILATION_PERCENTILE_LEVEL = 90.

TIME_FORMAT_FOR_LOG_MESSAGES = '%Y-%m-%d-%H%M%S'
STORM_COLUMNS_TO_KEEP = [tracking_io.STORM_ID_COLUMN, tracking_io.TIME_COLUMN]

RADAR_FIELD_NAME_KEY = 'radar_field_name'
RADAR_HEIGHT_KEY = 'radar_height_m_agl'
STATISTIC_NAME_KEY = 'statistic_name'
PERCENTILE_LEVEL_KEY = 'percentile_level'

GRID_METADATA_KEYS_TO_COMPARE = [
    radar_io.NW_GRID_POINT_LAT_COLUMN, radar_io.NW_GRID_POINT_LNG_COLUMN,
    radar_io.LAT_SPACING_COLUMN, radar_io.LNG_SPACING_COLUMN,
    radar_io.NUM_LAT_COLUMN, radar_io.NUM_LNG_COLUMN]

STORM_OBJECT_TO_GRID_PTS_COLUMNS = [
    tracking_io.STORM_ID_COLUMN, tracking_io.GRID_POINT_ROW_COLUMN,
    tracking_io.GRID_POINT_COLUMN_COLUMN]
GRID_POINT_LATLNG_COLUMNS = [tracking_io.GRID_POINT_LAT_COLUMN,
                             tracking_io.GRID_POINT_LNG_COLUMN]

# TODO(thunderhoser): Currently statistic names cannot have underscores (this
# will ruin _column_name_to_statistic_params).  I should change that.
AVERAGE_NAME = 'mean'
STANDARD_DEVIATION_NAME = 'stdev'
SKEWNESS_NAME = 'skewness'
KURTOSIS_NAME = 'kurtosis'
STATISTIC_NAMES = [
    AVERAGE_NAME, STANDARD_DEVIATION_NAME, SKEWNESS_NAME, KURTOSIS_NAME]
DEFAULT_STATISTIC_NAMES = [
    AVERAGE_NAME, STANDARD_DEVIATION_NAME, SKEWNESS_NAME, KURTOSIS_NAME]

DEFAULT_PERCENTILE_LEVELS = numpy.array([0., 5., 25., 50., 75., 95., 100.])
PERCENTILE_LEVEL_PRECISION = 0.1

DEFAULT_RADAR_FIELD_NAMES = [
    radar_io.ECHO_TOP_18DBZ_NAME, radar_io.ECHO_TOP_50DBZ_NAME,
    radar_io.LOW_LEVEL_SHEAR_NAME, radar_io.MID_LEVEL_SHEAR_NAME,
    radar_io.REFL_COLUMN_MAX_NAME, radar_io.MESH_NAME,
    radar_io.REFL_0CELSIUS_NAME, radar_io.REFL_M10CELSIUS_NAME,
    radar_io.REFL_M20CELSIUS_NAME, radar_io.REFL_LOWEST_ALTITUDE_NAME,
    radar_io.SHI_NAME, radar_io.VIL_NAME]

IGNORABLE_FIELD_NAMES = [
    radar_io.LOW_LEVEL_SHEAR_NAME, radar_io.MID_LEVEL_SHEAR_NAME]
AZIMUTHAL_SHEAR_FIELD_NAMES = [
    radar_io.LOW_LEVEL_SHEAR_NAME, radar_io.MID_LEVEL_SHEAR_NAME]


def _radar_field_and_statistic_to_column_name(
        radar_field_name=None, radar_height_m_agl=None, statistic_name=None):
    """Generates column name for radar field and statistic.

    :param radar_field_name: Name of radar field.
    :param radar_height_m_agl: Radar height (metres above ground level).
    :param statistic_name: Name of statistic.
    :return: column_name: Name of column.
    """

    if radar_field_name == radar_io.REFL_NAME:
        return '{0:s}_{1:d}m_{2:s}'.format(
            radar_field_name, int(radar_height_m_agl), statistic_name)

    return '{0:s}_{1:s}'.format(
        radar_field_name, statistic_name)


def _radar_field_and_percentile_to_column_name(
        radar_field_name=None, radar_height_m_agl=None, percentile_level=None):
    """Generates column name for radar field and percentile level.

    :param radar_field_name: Name of radar field.
    :param radar_height_m_agl: Radar height (metres above ground level).
    :param percentile_level: Percentile level.
    :return: column_name: Name of column.
    """

    if radar_field_name == radar_io.REFL_NAME:
        return '{0:s}_{1:d}m_percentile{2:05.1f}'.format(
            radar_field_name, int(radar_height_m_agl), percentile_level)

    return '{0:s}_percentile{1:05.1f}'.format(
        radar_field_name, percentile_level)


def _column_name_to_statistic_params(column_name):
    """Determines parameters of statistic from column name.

    If column name does not correspond to a statistic, this method will return
    None.

    :param column_name: Name of column.
    :return: parameter_dict: Dictionary with the following keys.
    parameter_dict['radar_field_name']: Name of radar field on which statistic
        is based.
    parameter_dict['radar_height_m_agl']: Radar height (metres above ground
        level).  If radar field is not single-elevation reflectivity, this will
        be None.
    parameter_dict['statistic_name']: Name of statistic.  If statistic is a
        percentile, this will be None.
    parameter_dict['percentile_level']: Percentile level.  If statistic is non-
        percentile, this will be None.
    """

    column_name_parts = column_name.split('_')
    if len(column_name_parts) < 2:
        return None

    # Determine statistic name or percentile level.
    if column_name_parts[-1] in STATISTIC_NAMES:
        statistic_name = column_name_parts[-1]
        percentile_level = None
    else:
        statistic_name = None
        if not column_name_parts[-1].startswith('percentile'):
            return None

        percentile_part = column_name_parts[-1].replace('percentile', '')
        try:
            percentile_level = float(percentile_part)
        except ValueError:
            return None

    # Determine radar field.
    radar_field_name = '_'.join(column_name_parts[:-1])
    radar_height_part = None
    try:
        radar_io.check_field_name(radar_field_name)
    except ValueError:
        radar_field_name = '_'.join(column_name_parts[:-2])
        radar_height_part = column_name_parts[-2]

    try:
        radar_io.check_field_name(radar_field_name)
    except ValueError:
        return None

    # Determine radar height.
    if radar_height_part is None:
        radar_height_m_agl = None
    else:
        if not radar_height_part.endswith('m'):
            return None

        radar_height_part = radar_height_part.replace('m', '')
        try:
            radar_height_m_agl = int(radar_height_part)
        except ValueError:
            return None

    return {RADAR_FIELD_NAME_KEY: radar_field_name,
            RADAR_HEIGHT_KEY: radar_height_m_agl,
            STATISTIC_NAME_KEY: statistic_name,
            PERCENTILE_LEVEL_KEY: percentile_level}


def _are_grids_equal(metadata_dict_orig, metadata_dict_new):
    """Indicates whether or not two grids are equal.

    :param metadata_dict_orig: Dictionary (with keys specified by
        `radar_io.read_metadata_from_raw_file`) describing original radar grid.
    :param metadata_dict_new: Dictionary (with keys specified by
        `radar_io.read_metadata_from_raw_file`) describing new radar grid.
    :return: are_grids_equal_flag: Boolean flag.
    """

    metadata_dict_orig = {
        k: metadata_dict_orig[k] for k in GRID_METADATA_KEYS_TO_COMPARE}
    metadata_dict_new = {
        k: metadata_dict_new[k] for k in GRID_METADATA_KEYS_TO_COMPARE}

    return metadata_dict_orig == metadata_dict_new


def _check_statistic_params(statistic_names, percentile_levels):
    """Ensures that parameters of statistic are valid.

    :param statistic_names: 1-D list with names of non-percentile-based
        statistics.
    :param percentile_levels: 1-D numpy array of percentile levels.
    :return: percentile_levels: Same as input, but rounded to the nearest 0.1%.
    :raises: ValueError: if any element of `statistic_names` is not in
        `STATISTIC_NAMES`.
    """

    error_checking.assert_is_string_list(statistic_names)
    error_checking.assert_is_numpy_array(
        numpy.array(statistic_names), num_dimensions=1)

    error_checking.assert_is_numpy_array(percentile_levels, num_dimensions=1)
    error_checking.assert_is_geq_numpy_array(percentile_levels, 0.)
    error_checking.assert_is_leq_numpy_array(percentile_levels, 100.)

    for this_name in statistic_names:
        if this_name in STATISTIC_NAMES:
            continue

        error_string = (
            '\n\n' + str(STATISTIC_NAMES) + '\n\nValid statistic names ' +
            '(listed above) do not include the following: "' + this_name + '"')
        raise ValueError(error_string)

    return numpy.unique(
        rounder.round_to_nearest(percentile_levels, PERCENTILE_LEVEL_PRECISION))


def get_statistic_columns(statistic_table):
    """Returns names of columns with radar statistics.

    :param statistic_table: pandas DataFrame.
    :return: statistic_column_names: 1-D list containing names of columns with
        radar statistics.  If there are no columns with radar stats, this is
        None.
    """

    column_names = list(statistic_table)
    statistic_column_names = None

    for this_column_name in column_names:
        this_parameter_dict = _column_name_to_statistic_params(this_column_name)
        if this_parameter_dict is None:
            continue

        if statistic_column_names is None:
            statistic_column_names = [this_column_name]
        else:
            statistic_column_names.append(this_column_name)

    return statistic_column_names


def check_statistic_table(statistic_table, require_storm_objects=True):
    """Ensures that pandas DataFrame contains radar statistics.

    :param statistic_table: pandas DataFrame.
    :param require_storm_objects: Boolean flag.  If True, statistic_table must
        contain columns "storm_id" and "unix_time_sec".  If False,
        statistic_table does not need these columns.
    :return: statistic_column_names: 1-D list containing names of columns with
        radar statistics.
    :raises: ValueError: if statistic_table does not contain any columns with
        radar statistics.
    """

    statistic_column_names = get_statistic_columns(statistic_table)
    if statistic_column_names is None:
        raise ValueError(
            'statistic_table does not contain any column with radar '
            'statistics.')

    if require_storm_objects:
        error_checking.assert_columns_in_dataframe(
            statistic_table, STORM_COLUMNS_TO_KEEP)

    return statistic_column_names


def extract_radar_grid_points(field_matrix, row_indices=None,
                              column_indices=None):
    """Extracts grid points from radar field.

    M = number of rows (unique grid-point latitudes)
    N = number of columns (unique grid-point longitudes)
    P = number of points to extract

    :param field_matrix: M-by-N numpy array with values of a single radar field.
    :param row_indices: length-P numpy array with row indices of points to
        extract.
    :param column_indices: length-P numpy array with column indices of points to
        extract.
    :return: extracted_values: length-P numpy array of values extracted from
        field_matrix.
    """

    error_checking.assert_is_real_numpy_array(field_matrix)
    error_checking.assert_is_numpy_array(field_matrix, num_dimensions=2)
    num_grid_rows = field_matrix.shape[0]
    num_grid_columns = field_matrix.shape[1]

    error_checking.assert_is_integer_numpy_array(row_indices)
    error_checking.assert_is_geq_numpy_array(row_indices, 0)
    error_checking.assert_is_less_than_numpy_array(row_indices, num_grid_rows)

    error_checking.assert_is_integer_numpy_array(column_indices)
    error_checking.assert_is_geq_numpy_array(column_indices, 0)
    error_checking.assert_is_less_than_numpy_array(column_indices,
                                                   num_grid_columns)

    return field_matrix[row_indices, column_indices]


def get_grid_points_in_storm_objects(storm_object_table,
                                     metadata_dict_for_storm_objects,
                                     new_metadata_dict):
    """Finds grid points inside each storm object.

    :param storm_object_table: pandas DataFrame with columns specified by
        `storm_tracking_io.write_processed_file`.
    :param metadata_dict_for_storm_objects: Dictionary (with keys specified by
        `radar_io.read_metadata_from_raw_file`) describing grid used to create
        storm objects.
    :param new_metadata_dict: Dictionary (with keys specified by
        `radar_io.read_metadata_from_raw_file`) describing new grid, for which
        points in each storm object will be found.
    :return: storm_object_to_grid_points_table: pandas DataFrame with the
        following columns.  Each row is one storm object.
    storm_object_to_grid_points_table.storm_id: String ID for storm cell.
    storm_object_to_grid_points_table.grid_point_rows: 1-D numpy array with row
        indices (integers) of grid points in storm object.
    storm_object_to_grid_points_table.grid_point_columns: 1-D numpy array with
        column indices (integers) of grid points in storm object.
    """

    if _are_grids_equal(metadata_dict_for_storm_objects, new_metadata_dict):
        return storm_object_table[STORM_OBJECT_TO_GRID_PTS_COLUMNS]

    storm_object_to_grid_points_table = storm_object_table[
        STORM_OBJECT_TO_GRID_PTS_COLUMNS + GRID_POINT_LATLNG_COLUMNS]
    num_storm_objects = len(storm_object_to_grid_points_table.index)

    for i in range(num_storm_objects):
        (storm_object_to_grid_points_table[
            tracking_io.GRID_POINT_ROW_COLUMN].values[i],
         storm_object_to_grid_points_table[
             tracking_io.GRID_POINT_COLUMN_COLUMN].values[i]) = (
                 radar_io.latlng_to_rowcol(
                     storm_object_to_grid_points_table[
                         tracking_io.GRID_POINT_LAT_COLUMN].values[i],
                     storm_object_to_grid_points_table[
                         tracking_io.GRID_POINT_LNG_COLUMN].values[i],
                     nw_grid_point_lat_deg=
                     new_metadata_dict[radar_io.NW_GRID_POINT_LAT_COLUMN],
                     nw_grid_point_lng_deg=
                     new_metadata_dict[radar_io.NW_GRID_POINT_LNG_COLUMN],
                     lat_spacing_deg=
                     new_metadata_dict[radar_io.LAT_SPACING_COLUMN],
                     lng_spacing_deg=
                     new_metadata_dict[radar_io.LNG_SPACING_COLUMN]))

    return storm_object_to_grid_points_table[STORM_OBJECT_TO_GRID_PTS_COLUMNS]


def get_spatial_statistics(radar_field, statistic_names=DEFAULT_STATISTIC_NAMES,
                           percentile_levels=DEFAULT_PERCENTILE_LEVELS):
    """Computes spatial statistics for a single radar field.

    "Single field" = one variable at one elevation, one time step, many spatial
    locations.

    Radar field may have any number of dimensions (1-D, 2-D, etc.).

    N = number of non-percentile-based statistics
    P = number of percentile levels

    :param radar_field: numpy array.  Each position in the array should be a
        different spatial location.
    :param statistic_names: length-N list of non-percentile-based statistics.
    :param percentile_levels: length-P numpy array of percentile levels.
    :return: statistic_values: length-N numpy with values of non-percentile-
        based statistics.
    :return: percentile_values: length-P numpy array of percentiles.
    """

    error_checking.assert_is_real_numpy_array(radar_field)
    percentile_levels = _check_statistic_params(
        statistic_names, percentile_levels)

    num_statistics = len(statistic_names)
    statistic_values = numpy.full(num_statistics, numpy.nan)
    for i in range(num_statistics):
        if statistic_names[i] == AVERAGE_NAME:
            statistic_values[i] = numpy.nanmean(radar_field)
        elif statistic_names[i] == STANDARD_DEVIATION_NAME:
            statistic_values[i] = numpy.nanstd(radar_field, ddof=1)
        elif statistic_names[i] == SKEWNESS_NAME:
            statistic_values[i] = scipy.stats.skew(
                radar_field, bias=False, nan_policy='omit', axis=None)
        elif statistic_names[i] == KURTOSIS_NAME:
            statistic_values[i] = scipy.stats.kurtosis(
                radar_field, fisher=True, bias=False, nan_policy='omit',
                axis=None)

    percentile_values = numpy.nanpercentile(
        radar_field, percentile_levels, interpolation='linear')
    return statistic_values, percentile_values


def get_stats_for_storm_objects(
        storm_object_table, metadata_dict_for_storm_objects,
        statistic_names=DEFAULT_STATISTIC_NAMES,
        percentile_levels=DEFAULT_PERCENTILE_LEVELS,
        radar_field_names=DEFAULT_RADAR_FIELD_NAMES,
        reflectivity_heights_m_agl=None,
        radar_data_source=radar_io.MYRORSS_SOURCE_ID,
        top_radar_directory_name=None, dilate_azimuthal_shear=False,
        dilation_half_width_in_pixels=dilation.DEFAULT_HALF_WIDTH,
        dilation_percentile_level=DEFAULT_DILATION_PERCENTILE_LEVEL):
    """Computes radar statistics for one or more storm objects.

    F = number of radar field-height pairs
    K = number of statistics (percentile- and non-percentile-based)

    :param storm_object_table: pandas DataFrame with columns documented in
        `storm_tracking_io.write_processed_file`.  May contain additional
        columns.
    :param metadata_dict_for_storm_objects: Dictionary (with keys specified by
        `radar_io.read_metadata_from_raw_file`) describing grid used to create
        storm objects.
    :param statistic_names: 1-D list of non-percentile-based statistics.
    :param percentile_levels: 1-D numpy array of percentile levels.
    :param radar_field_names: 1-D list with names of radar fields.
    :param reflectivity_heights_m_agl: 1-D numpy array of heights (metres above
        ground level) for radar field "reflectivity_dbz".  If "reflectivity_dbz"
        is not in `radar_field_names`, this can be left as None.
    :param radar_data_source: Data source for radar field.
    :param top_radar_directory_name: Name of top-level directory with radar
        files from given source.
    :param dilate_azimuthal_shear: Boolean flag.  If False, azimuthal-shear
        stats will be based only on values inside the storm object.  If True,
        azimuthal-shear fields will be dilated, so azimuthal-shear stats will be
        based on values inside and near the storm object.  This is useful
        because sometimes large az-shear values occur in regions of low
        reflectivity, which may not be included in the storm object.
    :param dilation_half_width_in_pixels: See documentation for
        `dilation.dilate_2d_matrix`.
    :param dilation_percentile_level: See documentation for
        `dilation.dilate_2d_matrix`.
    :return: storm_radar_statistic_table: pandas DataFrame with 2 + K * F
        columns, where the last K * F columns are one for each statistic-field
        pair.  Names of these columns are determined by
        _radar_field_and_statistic_to_column_name and
        _radar_field_and_percentile_to_column_name.  The first 2 columns are
        listed below.
    storm_radar_statistic_table.storm_id: String ID for storm cell.  Same as
        input column `storm_object_table.storm_id`.
    storm_radar_statistic_table.unix_time_sec: Valid time.  Same as input column
        `storm_object_table.unix_time_sec`.
    """

    percentile_levels = _check_statistic_params(
        statistic_names, percentile_levels)
    error_checking.assert_is_boolean(dilate_azimuthal_shear)

    radar_field_name_by_pair, radar_height_by_pair_m_agl = (
        radar_io.unique_fields_and_heights_to_pairs(
            radar_field_names, refl_heights_m_agl=reflectivity_heights_m_agl,
            data_source=radar_data_source))
    num_radar_fields = len(radar_field_name_by_pair)

    storm_object_time_matrix = storm_object_table.as_matrix(
        columns=[tracking_io.TIME_COLUMN, tracking_io.SPC_DATE_COLUMN])
    unique_time_matrix = numpy.vstack(
        {tuple(this_row) for this_row in storm_object_time_matrix}).astype(int)
    unique_storm_times_unix_sec = unique_time_matrix[:, 0]
    unique_spc_dates_unix_sec = unique_time_matrix[:, 1]

    num_unique_storm_times = len(unique_storm_times_unix_sec)
    radar_file_name_matrix = numpy.full(
        (num_unique_storm_times, num_radar_fields), '', dtype=object)

    for i in range(num_unique_storm_times):
        for j in range(num_radar_fields):
            radar_file_name_matrix[i, j] = radar_io.find_raw_file(
                unix_time_sec=unique_storm_times_unix_sec[i],
                spc_date_unix_sec=unique_spc_dates_unix_sec[i],
                field_name=radar_field_name_by_pair[j],
                height_m_agl=radar_height_by_pair_m_agl[j],
                data_source=radar_data_source,
                top_directory_name=top_radar_directory_name,
                raise_error_if_missing=
                radar_field_name_by_pair[j] not in IGNORABLE_FIELD_NAMES)

            if not os.path.isfile(radar_file_name_matrix[i, j]):
                radar_file_name_matrix[i, j] = None

                this_time_string = time_conversion.unix_sec_to_string(
                    unique_storm_times_unix_sec[i],
                    TIME_FORMAT_FOR_LOG_MESSAGES)
                warning_string = (
                    'Cannot find file for "{0:s}" at {1:d} metres AGL and '
                    '{2:s}.  File expected at: "{3:s}"').format(
                        radar_field_name_by_pair[j],
                        radar_height_by_pair_m_agl[j], this_time_string,
                        radar_file_name_matrix[i, j])
                warnings.warn(warning_string)

    num_statistics = len(statistic_names)
    num_percentiles = len(percentile_levels)
    num_storms = len(storm_object_table.index)
    statistic_matrix = numpy.full(
        (num_storms, num_radar_fields, num_statistics), numpy.nan)
    percentile_matrix = numpy.full(
        (num_storms, num_radar_fields, num_percentiles), numpy.nan)

    for j in range(num_radar_fields):
        metadata_dict_for_this_field = None

        for i in range(num_unique_storm_times):
            if radar_file_name_matrix[i, j] is None:
                continue

            this_time_string = time_conversion.unix_sec_to_string(
                unique_storm_times_unix_sec[i], TIME_FORMAT_FOR_LOG_MESSAGES)
            print ('Computing stats for "' + str(radar_field_name_by_pair[j]) +
                   '" at ' + str(radar_height_by_pair_m_agl[j]) +
                   ' m AGL and ' + this_time_string + '...')

            if metadata_dict_for_this_field is None:
                metadata_dict_for_this_field = (
                    radar_io.read_metadata_from_raw_file(
                        radar_file_name_matrix[i, j],
                        data_source=radar_data_source))
                storm_object_to_grid_pts_table_this_field = (
                    get_grid_points_in_storm_objects(
                        storm_object_table, metadata_dict_for_storm_objects,
                        metadata_dict_for_this_field))

            sparse_grid_table_this_field = (
                radar_io.read_data_from_sparse_grid_file(
                    radar_file_name_matrix[i, j],
                    field_name_orig=metadata_dict_for_this_field[
                        radar_io.FIELD_NAME_COLUMN_ORIG],
                    data_source=radar_data_source,
                    sentinel_values=metadata_dict_for_this_field[
                        radar_io.SENTINEL_VALUE_COLUMN]))

            radar_matrix_this_field, _, _ = radar_s2f.sparse_to_full_grid(
                sparse_grid_table_this_field, metadata_dict_for_this_field)

            if (dilate_azimuthal_shear and radar_field_name_by_pair[j] in
                    AZIMUTHAL_SHEAR_FIELD_NAMES):
                print 'Dilating azimuthal-shear field...'
                radar_matrix_this_field = dilation.dilate_2d_matrix(
                    radar_matrix_this_field,
                    percentile_level=dilation_percentile_level,
                    half_width_in_pixels=dilation_half_width_in_pixels,
                    take_largest_absolute_value=True)

            radar_matrix_this_field[numpy.isnan(radar_matrix_this_field)] = 0.

            these_storm_flags = numpy.logical_and(
                storm_object_table[tracking_io.TIME_COLUMN].values ==
                unique_storm_times_unix_sec[i],
                storm_object_table[tracking_io.SPC_DATE_COLUMN].values ==
                unique_spc_dates_unix_sec[i])
            these_storm_indices = numpy.where(these_storm_flags)[0]

            for this_storm_index in these_storm_indices:
                radar_values_this_storm = extract_radar_grid_points(
                    radar_matrix_this_field,
                    row_indices=storm_object_to_grid_pts_table_this_field[
                        tracking_io.GRID_POINT_ROW_COLUMN].values[
                            this_storm_index].astype(int),
                    column_indices=storm_object_to_grid_pts_table_this_field[
                        tracking_io.GRID_POINT_COLUMN_COLUMN].values[
                            this_storm_index].astype(int))

                (statistic_matrix[this_storm_index, j, :],
                 percentile_matrix[this_storm_index, j, :]) = (
                     get_spatial_statistics(
                         radar_values_this_storm,
                         statistic_names=statistic_names,
                         percentile_levels=percentile_levels))

    storm_radar_statistic_dict = {}
    for j in range(num_radar_fields):
        for k in range(num_statistics):
            this_column_name = _radar_field_and_statistic_to_column_name(
                radar_field_name=radar_field_name_by_pair[j],
                radar_height_m_agl=radar_height_by_pair_m_agl[j],
                statistic_name=statistic_names[k])

            storm_radar_statistic_dict.update(
                {this_column_name: statistic_matrix[:, j, k]})

        for k in range(num_percentiles):
            this_column_name = _radar_field_and_percentile_to_column_name(
                radar_field_name=radar_field_name_by_pair[j],
                radar_height_m_agl=radar_height_by_pair_m_agl[j],
                percentile_level=percentile_levels[k])

            storm_radar_statistic_dict.update(
                {this_column_name: percentile_matrix[:, j, k]})

    storm_radar_statistic_table = pandas.DataFrame.from_dict(
        storm_radar_statistic_dict)
    return pandas.concat(
        [storm_object_table[STORM_COLUMNS_TO_KEEP],
         storm_radar_statistic_table], axis=1)


def write_stats_for_storm_objects(storm_radar_statistic_table,
                                  pickle_file_name):
    """Writes radar statistics for storm objects to a Pickle file.

    :param storm_radar_statistic_table: pandas DataFrame created by
        get_stats_for_storm_objects.
    :param pickle_file_name: Path to output file.
    """

    statistic_column_names = check_statistic_table(
        storm_radar_statistic_table, require_storm_objects=True)
    columns_to_write = STORM_COLUMNS_TO_KEEP + statistic_column_names

    file_system_utils.mkdir_recursive_if_necessary(file_name=pickle_file_name)
    pickle_file_handle = open(pickle_file_name, 'wb')
    pickle.dump(storm_radar_statistic_table[columns_to_write],
                pickle_file_handle)
    pickle_file_handle.close()


def read_stats_for_storm_objects(pickle_file_name):
    """Reads radar statistics for storm objects from a Pickle file.

    :param pickle_file_name: Path to input file.
    :return: storm_radar_statistic_table: pandas DataFrame with columns
        documented in get_stats_for_storm_objects.
    """

    pickle_file_handle = open(pickle_file_name, 'rb')
    storm_radar_statistic_table = pickle.load(pickle_file_handle)
    pickle_file_handle.close()

    check_statistic_table(
        storm_radar_statistic_table, require_storm_objects=True)
    return storm_radar_statistic_table
