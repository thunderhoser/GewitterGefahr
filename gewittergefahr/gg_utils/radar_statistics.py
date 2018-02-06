"""Methods for computing radar statistics.

These are usually spatial statistics based on values inside a storm object.
"""

import pickle
import numpy
import pandas
import scipy.stats
from gewittergefahr.gg_io import myrorss_and_mrms_io
from gewittergefahr.gg_io import gridrad_io
from gewittergefahr.gg_utils import storm_tracking_utils as tracking_utils
from gewittergefahr.gg_utils import radar_utils
from gewittergefahr.gg_utils import gridrad_utils
from gewittergefahr.gg_utils import radar_sparse_to_full as radar_s2f
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import dilation
from gewittergefahr.gg_utils import number_rounding as rounder
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.gg_utils import error_checking

TOLERANCE = 1e-6
DEFAULT_DILATION_PERCENTILE_LEVEL = 100.

TIME_FORMAT_FOR_LOG_MESSAGES = '%Y-%m-%d-%H%M%S'
STORM_COLUMNS_TO_KEEP = [
    tracking_utils.STORM_ID_COLUMN, tracking_utils.TIME_COLUMN]

RADAR_FIELD_NAME_KEY = 'radar_field_name'
RADAR_HEIGHT_KEY = 'radar_height_m_asl'
STATISTIC_NAME_KEY = 'statistic_name'
PERCENTILE_LEVEL_KEY = 'percentile_level'

GRID_METADATA_KEYS_TO_COMPARE = [
    radar_utils.NW_GRID_POINT_LAT_COLUMN, radar_utils.NW_GRID_POINT_LNG_COLUMN,
    radar_utils.LAT_SPACING_COLUMN, radar_utils.LNG_SPACING_COLUMN,
    radar_utils.NUM_LAT_COLUMN, radar_utils.NUM_LNG_COLUMN]

STORM_OBJECT_TO_GRID_PTS_COLUMNS = [
    tracking_utils.STORM_ID_COLUMN, tracking_utils.GRID_POINT_ROW_COLUMN,
    tracking_utils.GRID_POINT_COLUMN_COLUMN]
GRID_POINT_LATLNG_COLUMNS = [
    tracking_utils.GRID_POINT_LAT_COLUMN, tracking_utils.GRID_POINT_LNG_COLUMN]

# TODO(thunderhoser): Currently statistic names cannot have underscores (this
# will ruin _column_name_to_statistic_params).  This should be fixed.
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

DEFAULT_FIELDS_FOR_MYRORSS_AND_MRMS = [
    radar_utils.ECHO_TOP_18DBZ_NAME, radar_utils.ECHO_TOP_50DBZ_NAME,
    radar_utils.LOW_LEVEL_SHEAR_NAME, radar_utils.MID_LEVEL_SHEAR_NAME,
    radar_utils.REFL_COLUMN_MAX_NAME, radar_utils.MESH_NAME,
    radar_utils.REFL_0CELSIUS_NAME, radar_utils.REFL_M10CELSIUS_NAME,
    radar_utils.REFL_M20CELSIUS_NAME, radar_utils.REFL_LOWEST_ALTITUDE_NAME,
    radar_utils.SHI_NAME, radar_utils.VIL_NAME]

AZIMUTHAL_SHEAR_FIELD_NAMES = [
    radar_utils.LOW_LEVEL_SHEAR_NAME, radar_utils.MID_LEVEL_SHEAR_NAME]

# TODO(thunderhoser): Deal with dual-pol variables in GridRad and the fact that
# they might be missing.
DEFAULT_FIELDS_FOR_GRIDRAD = [
    radar_utils.REFL_NAME, radar_utils.SPECTRUM_WIDTH_NAME,
    radar_utils.VORTICITY_NAME, radar_utils.DIVERGENCE_NAME]

DEFAULT_HEIGHTS_FOR_GRIDRAD_M_ASL = numpy.array(
    [1000, 2000, 3000, 4000, 5000, 8000, 10000, 12000], dtype=int)


def _column_name_to_statistic_params(column_name):
    """Determines parameters of statistic from column name.

    If column name does not correspond to a statistic, this method will return
    None.

    :param column_name: Name of column.
    :return: parameter_dict: Dictionary with the following keys.
    parameter_dict['radar_field_name']: Name of radar field on which statistic
        is based.
    parameter_dict['radar_height_m_asl']: Radar height (metres above sea level).
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
    radar_field_name = '_'.join(column_name_parts[:-2])
    try:
        radar_utils.check_field_name(radar_field_name)
    except ValueError:
        return None

    # Determine radar height.
    radar_height_part = column_name_parts[-2]
    if not radar_height_part.endswith('metres'):
        return None

    radar_height_part = radar_height_part.replace('metres', '')
    try:
        radar_height_m_asl = int(radar_height_part)
    except ValueError:
        return None

    return {RADAR_FIELD_NAME_KEY: radar_field_name,
            RADAR_HEIGHT_KEY: radar_height_m_asl,
            STATISTIC_NAME_KEY: statistic_name,
            PERCENTILE_LEVEL_KEY: percentile_level}


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


def are_grids_equal(orig_metadata_dict, new_metadata_dict):
    """Indicates whether or not two grids are equal.

    :param orig_metadata_dict: Dictionary with metadata for original grid.  Keys
        are listed in documentation of `get_grid_points_in_storm_objects`.
    :param new_metadata_dict: Dictionary with metadata for new grid.  Keys are
        listed in documentation of `get_grid_points_in_storm_objects`.
    :return: are_grids_equal_flag: Boolean flag.
    """

    # TODO(thunderhoser): Put this method somewhere else.

    for this_key in GRID_METADATA_KEYS_TO_COMPARE:
        this_absolute_diff = numpy.absolute(
            orig_metadata_dict[this_key] - new_metadata_dict[this_key])
        if this_absolute_diff > TOLERANCE:
            return False

    return True


def radar_field_and_statistic_to_column_name(
        radar_field_name, radar_height_m_asl, statistic_name):
    """Generates column name for radar field and statistic.

    :param radar_field_name: Name of radar field.
    :param radar_height_m_asl: Radar height (metres above sea level).
    :param statistic_name: Name of statistic.
    :return: column_name: Name of column.
    """

    error_checking.assert_is_string(radar_field_name)
    error_checking.assert_is_not_nan(radar_height_m_asl)
    error_checking.assert_is_string(statistic_name)

    return '{0:s}_{1:d}metres_{2:s}'.format(
        radar_field_name, int(numpy.round(radar_height_m_asl)), statistic_name)


def radar_field_and_percentile_to_column_name(
        radar_field_name, radar_height_m_asl, percentile_level):
    """Generates column name for radar field and percentile level.

    :param radar_field_name: Name of radar field.
    :param radar_height_m_asl: Radar height (metres above sea level).
    :param percentile_level: Percentile level.
    :return: column_name: Name of column.
    """

    error_checking.assert_is_string(radar_field_name)
    error_checking.assert_is_not_nan(radar_height_m_asl)
    error_checking.assert_is_not_nan(percentile_level)

    return '{0:s}_{1:d}metres_percentile{2:05.1f}'.format(
        radar_field_name, int(numpy.round(radar_height_m_asl)),
        percentile_level)


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


def extract_radar_grid_points(field_matrix, row_indices, column_indices):
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


def get_grid_points_in_storm_objects(
        storm_object_table, orig_grid_metadata_dict, new_grid_metadata_dict):
    """Finds grid points inside each storm object.

    :param storm_object_table: pandas DataFrame with columns specified by
        `storm_tracking_io.write_processed_file`.
    :param orig_grid_metadata_dict: Dictionary with the following keys,
        describing radar grid used to create storm objects.
    orig_grid_metadata_dict['nw_grid_point_lat_deg']: Latitude (deg N) of
        northwesternmost grid point.
    orig_grid_metadata_dict['nw_grid_point_lng_deg']: Longitude (deg E) of
        northwesternmost grid point.
    orig_grid_metadata_dict['lat_spacing_deg']: Spacing (deg N) between adjacent
        rows.
    orig_grid_metadata_dict['lng_spacing_deg']: Spacing (deg E) between adjacent
        columns.
    orig_grid_metadata_dict['num_lat_in_grid']: Number of rows (unique grid-
        point latitudes).
    orig_grid_metadata_dict['num_lng_in_grid']: Number of columns (unique grid-
        point longitudes).

    :param new_grid_metadata_dict: Same as `orig_grid_metadata_dict`, except for
        new radar grid.  We want to know grid points inside each storm object
        for the new grid.
    :return: storm_object_to_grid_points_table: pandas DataFrame with the
        following columns.  Each row is one storm object.
    storm_object_to_grid_points_table.storm_id: String ID for storm cell.
    storm_object_to_grid_points_table.grid_point_rows: 1-D numpy array with row
        indices (integers) of grid points in storm object.
    storm_object_to_grid_points_table.grid_point_columns: 1-D numpy array with
        column indices (integers) of grid points in storm object.
    """

    if are_grids_equal(orig_grid_metadata_dict, new_grid_metadata_dict):
        return storm_object_table[STORM_OBJECT_TO_GRID_PTS_COLUMNS]

    storm_object_to_grid_points_table = storm_object_table[
        STORM_OBJECT_TO_GRID_PTS_COLUMNS + GRID_POINT_LATLNG_COLUMNS]
    num_storm_objects = len(storm_object_to_grid_points_table.index)

    for i in range(num_storm_objects):
        (storm_object_to_grid_points_table[
            tracking_utils.GRID_POINT_ROW_COLUMN].values[i],
         storm_object_to_grid_points_table[
             tracking_utils.GRID_POINT_COLUMN_COLUMN].values[i]) = (
                 radar_utils.latlng_to_rowcol(
                     storm_object_to_grid_points_table[
                         tracking_utils.GRID_POINT_LAT_COLUMN].values[i],
                     storm_object_to_grid_points_table[
                         tracking_utils.GRID_POINT_LNG_COLUMN].values[i],
                     nw_grid_point_lat_deg=new_grid_metadata_dict[
                         radar_utils.NW_GRID_POINT_LAT_COLUMN],
                     nw_grid_point_lng_deg=new_grid_metadata_dict[
                         radar_utils.NW_GRID_POINT_LNG_COLUMN],
                     lat_spacing_deg=
                     new_grid_metadata_dict[radar_utils.LAT_SPACING_COLUMN],
                     lng_spacing_deg=
                     new_grid_metadata_dict[radar_utils.LNG_SPACING_COLUMN]))

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


def get_storm_based_radar_stats_myrorss_or_mrms(
        storm_object_table, top_radar_dir_name, metadata_dict_for_storm_objects,
        statistic_names=DEFAULT_STATISTIC_NAMES,
        percentile_levels=DEFAULT_PERCENTILE_LEVELS,
        radar_field_names=DEFAULT_FIELDS_FOR_MYRORSS_AND_MRMS,
        reflectivity_heights_m_asl=None,
        radar_source=radar_utils.MYRORSS_SOURCE_ID,
        dilate_azimuthal_shear=False,
        dilation_half_width_in_pixels=dilation.DEFAULT_HALF_WIDTH,
        dilation_percentile_level=DEFAULT_DILATION_PERCENTILE_LEVEL):
    """Computes radar statistics for each storm object.

    In this case, radar data must be from MYRORSS or MRMS.

    N = number of storm objects
    P = number of field/height pairs
    S = number of statistics (percentile- and non-percentile-based)

    :param storm_object_table: See documentation for
        `get_storm_based_radar_stats_gridrad`.
    :param top_radar_dir_name: See doc for
        `get_storm_based_radar_stats_gridrad`.
    :param metadata_dict_for_storm_objects: Dictionary created by
        `myrorss_and_mrms_io.read_metadata_from_raw_file`, describing radar grid
        used to create storm objects.
    :param statistic_names: 1-D list of non-percentile-based statistics.
    :param percentile_levels: 1-D numpy array of percentile levels.
    :param radar_field_names: 1-D list of radar fields for which stats will be
        computed.
    :param reflectivity_heights_m_asl: 1-D numpy array of heights (metres above
        sea level) for the field "reflectivity_dbz".  If "reflectivity_dbz" is
        not in `radar_field_names`, you can leave this as None.
    :param radar_source: Source of radar data (either "myrorss" or "mrms").
    :param dilate_azimuthal_shear: Boolean flag.  If False, azimuthal-shear
        stats will be based only on values inside the storm object.  If True,
        azimuthal-shear fields will be dilated, so azimuthal-shear stats will be
        based on values inside and near the storm object.  This is useful
        because sometimes large az-shear values occur just outside the storm
        object.
    :param dilation_half_width_in_pixels: See documentation for
        `dilation.dilate_2d_matrix`.
    :param dilation_percentile_level: See documentation for
        `dilation.dilate_2d_matrix`.
    :return: storm_object_statistic_table: pandas DataFrame with 2 + S * P
        columns.  The last S * P columns are one for each statistic-field-height
        tuple.  Names of these columns are determined by
        `radar_field_and_statistic_to_column_name` and
        `radar_field_and_percentile_to_column_name`.  The first 2 columns are
        listed below.
    storm_object_statistic_table.storm_id: Storm ID (string) (taken from input
        table).
    storm_object_statistic_table.unix_time_sec: Valid time (taken from input
        table).
    """

    # Error-checking.
    percentile_levels = _check_statistic_params(
        statistic_names, percentile_levels)
    error_checking.assert_is_boolean(dilate_azimuthal_shear)

    # Find radar files.
    spc_date_strings = [
        time_conversion.time_to_spc_date_string(t)
        for t in storm_object_table[tracking_utils.SPC_DATE_COLUMN].values]

    file_dictionary = myrorss_and_mrms_io.find_many_raw_files(
        desired_times_unix_sec=
        storm_object_table[tracking_utils.TIME_COLUMN].values,
        spc_date_strings=spc_date_strings, data_source=radar_source,
        field_names=radar_field_names, top_directory_name=top_radar_dir_name,
        reflectivity_heights_m_asl=reflectivity_heights_m_asl)

    radar_file_names_2d_list = file_dictionary[
        myrorss_and_mrms_io.RADAR_FILE_NAME_LIST_KEY]
    valid_times_unix_sec = file_dictionary[myrorss_and_mrms_io.UNIQUE_TIMES_KEY]
    valid_spc_dates_unix_sec = file_dictionary[
        myrorss_and_mrms_io.SPC_DATES_AT_UNIQUE_TIMES_KEY]
    radar_field_name_by_pair = file_dictionary[
        myrorss_and_mrms_io.FIELD_NAME_BY_PAIR_KEY]
    radar_height_by_pair_m_asl = file_dictionary[
        myrorss_and_mrms_io.HEIGHT_BY_PAIR_KEY]

    # Initialize output.
    num_field_height_pairs = len(radar_field_name_by_pair)
    num_valid_times = len(valid_times_unix_sec)
    num_statistics = len(statistic_names)
    num_percentiles = len(percentile_levels)
    num_storm_objects = len(storm_object_table.index)

    statistic_matrix = numpy.full(
        (num_storm_objects, num_field_height_pairs, num_statistics), numpy.nan)
    percentile_matrix = numpy.full(
        (num_storm_objects, num_field_height_pairs, num_percentiles), numpy.nan)

    valid_time_strings = [
        time_conversion.unix_sec_to_string(t, TIME_FORMAT_FOR_LOG_MESSAGES)
        for t in valid_times_unix_sec]

    for j in range(num_field_height_pairs):
        metadata_dict_this_field_height = None

        for i in range(num_valid_times):
            if radar_file_names_2d_list[i][j] is None:
                continue

            print (
                'Computing stats for "{0:s}" at {1:d} metres ASL and '
                '{2:s}...').format(
                    radar_field_name_by_pair[j],
                    int(numpy.round(radar_height_by_pair_m_asl[j])),
                    valid_time_strings[i])

            if metadata_dict_this_field_height is None:

                # Find grid points in each storm object for the [j]th
                # field/height pair.
                metadata_dict_this_field_height = (
                    myrorss_and_mrms_io.read_metadata_from_raw_file(
                        radar_file_names_2d_list[i][j],
                        data_source=radar_source))

                storm_to_grid_pts_table_this_field_height = (
                    get_grid_points_in_storm_objects(
                        storm_object_table=storm_object_table,
                        orig_grid_metadata_dict=metadata_dict_for_storm_objects,
                        new_grid_metadata_dict=metadata_dict_this_field_height))

            # Read data for [j]th field/height pair at [i]th time step.
            sparse_grid_table_this_field_height = (
                myrorss_and_mrms_io.read_data_from_sparse_grid_file(
                    radar_file_names_2d_list[i][j],
                    field_name_orig=metadata_dict_this_field_height[
                        myrorss_and_mrms_io.FIELD_NAME_COLUMN_ORIG],
                    data_source=radar_source,
                    sentinel_values=metadata_dict_this_field_height[
                        radar_utils.SENTINEL_VALUE_COLUMN]))

            radar_matrix_this_field_height, _, _ = (
                radar_s2f.sparse_to_full_grid(
                    sparse_grid_table_this_field_height,
                    metadata_dict_this_field_height))

            if (dilate_azimuthal_shear and radar_field_name_by_pair[j] in
                    AZIMUTHAL_SHEAR_FIELD_NAMES):
                print 'Dilating azimuthal-shear field...'
                radar_matrix_this_field_height = dilation.dilate_2d_matrix(
                    radar_matrix_this_field_height,
                    percentile_level=dilation_percentile_level,
                    half_width_in_pixels=dilation_half_width_in_pixels,
                    take_largest_absolute_value=True)

            radar_matrix_this_field_height[
                numpy.isnan(radar_matrix_this_field_height)] = 0.

            # Find storm objects at [i]th valid time.
            these_storm_flags = numpy.logical_and(
                storm_object_table[tracking_utils.TIME_COLUMN].values ==
                valid_times_unix_sec[i],
                storm_object_table[tracking_utils.SPC_DATE_COLUMN].values ==
                valid_spc_dates_unix_sec[i])
            these_storm_indices = numpy.where(these_storm_flags)[0]

            # Extract storm-based radar stats for [j]th field/height pair at
            # [i]th time step.
            for this_storm_index in these_storm_indices:
                radar_values_this_storm = extract_radar_grid_points(
                    radar_matrix_this_field_height,
                    row_indices=storm_to_grid_pts_table_this_field_height[
                        tracking_utils.GRID_POINT_ROW_COLUMN].values[
                            this_storm_index].astype(int),
                    column_indices=storm_to_grid_pts_table_this_field_height[
                        tracking_utils.GRID_POINT_COLUMN_COLUMN].values[
                            this_storm_index].astype(int))

                (statistic_matrix[this_storm_index, j, :],
                 percentile_matrix[this_storm_index, j, :]) = (
                     get_spatial_statistics(
                         radar_values_this_storm,
                         statistic_names=statistic_names,
                         percentile_levels=percentile_levels))

    # Create pandas DataFrame.
    storm_object_statistic_dict = {}
    for j in range(num_field_height_pairs):
        for k in range(num_statistics):
            this_column_name = radar_field_and_statistic_to_column_name(
                radar_field_name=radar_field_name_by_pair[j],
                radar_height_m_asl=radar_height_by_pair_m_asl[j],
                statistic_name=statistic_names[k])

            storm_object_statistic_dict.update(
                {this_column_name: statistic_matrix[:, j, k]})

        for k in range(num_percentiles):
            this_column_name = radar_field_and_percentile_to_column_name(
                radar_field_name=radar_field_name_by_pair[j],
                radar_height_m_asl=radar_height_by_pair_m_asl[j],
                percentile_level=percentile_levels[k])

            storm_object_statistic_dict.update(
                {this_column_name: percentile_matrix[:, j, k]})

    storm_object_statistic_table = pandas.DataFrame.from_dict(
        storm_object_statistic_dict)
    return pandas.concat(
        [storm_object_table[STORM_COLUMNS_TO_KEEP],
         storm_object_statistic_table], axis=1)


def get_storm_based_radar_stats_gridrad(
        storm_object_table, top_radar_dir_name,
        statistic_names=DEFAULT_STATISTIC_NAMES,
        percentile_levels=DEFAULT_PERCENTILE_LEVELS,
        radar_field_names=DEFAULT_FIELDS_FOR_GRIDRAD,
        radar_heights_m_asl=DEFAULT_HEIGHTS_FOR_GRIDRAD_M_ASL):
    """Computes radar statistics for each storm object.

    In this case, radar data must be from GridRad.

    N = number of storm objects
    F = number of radar fields
    H = number of radar heights
    S = number of statistics (percentile- and non-percentile-based)

    :param storm_object_table: N-row pandas DataFrame with columns listed in
        `storm_tracking_io.write_processed_file`.  Each row is one storm object.
    :param top_radar_dir_name: [input] Name of top-level directory with radar
        data from the given source.
    :param statistic_names: 1-D list of non-percentile-based statistics.
    :param percentile_levels: 1-D numpy array of percentile levels.
    :param radar_field_names: length-F list of radar fields for which stats will
        be computed.
    :param radar_heights_m_asl: length-H numpy array of radar heights (metres
        above sea level).
    :return: storm_object_statistic_table: pandas DataFrame with 2 + S * F * H
        columns.  The last S * F * H columns are one for each statistic-field-
        height tuple.  Names of these columns are determined by
        `radar_field_and_statistic_to_column_name` and
        `radar_field_and_percentile_to_column_name`.  The first 2 columns are
        listed below.
    storm_object_statistic_table.storm_id: Storm ID (string) (taken from input
        table).
    storm_object_statistic_table.unix_time_sec: Valid time (taken from input
        table).
    """

    # Error-checking.
    percentile_levels = _check_statistic_params(
        statistic_names, percentile_levels)

    _, _ = gridrad_utils.fields_and_refl_heights_to_pairs(
        field_names=radar_field_names, heights_m_asl=radar_heights_m_asl)
    radar_heights_m_asl = numpy.sort(
        numpy.round(radar_heights_m_asl).astype(int))

    # Find radar files.
    radar_times_unix_sec = numpy.unique(
        storm_object_table[tracking_utils.TIME_COLUMN].values)
    radar_time_strings = [
        time_conversion.unix_sec_to_string(t, TIME_FORMAT_FOR_LOG_MESSAGES)
        for t in radar_times_unix_sec]

    num_radar_times = len(radar_times_unix_sec)
    radar_file_names = [None] * num_radar_times
    for i in range(num_radar_times):
        radar_file_names[i] = gridrad_io.find_file(
            unix_time_sec=radar_times_unix_sec[i],
            top_directory_name=top_radar_dir_name, raise_error_if_missing=True)

    # Initialize output.
    num_radar_fields = len(radar_field_names)
    num_radar_heights = len(radar_heights_m_asl)
    num_statistics = len(statistic_names)
    num_percentiles = len(percentile_levels)
    num_storm_objects = len(storm_object_table.index)

    statistic_matrix = numpy.full(
        (num_storm_objects, num_radar_fields, num_radar_heights,
         num_statistics),
        numpy.nan)
    percentile_matrix = numpy.full(
        (num_storm_objects, num_radar_fields, num_radar_heights,
         num_percentiles),
        numpy.nan)

    for i in range(num_radar_times):

        # Read metadata for [i]th valid time and find storm objects at [i]th
        # valid time.
        this_metadata_dict = gridrad_io.read_metadata_from_full_grid_file(
            radar_file_names[i])
        these_storm_indices = numpy.where(
            storm_object_table[tracking_utils.TIME_COLUMN].values ==
            radar_times_unix_sec[i])[0]

        for j in range(num_radar_fields):

            # Read data for [j]th field at [i]th valid time.
            print 'Reading "{0:s}" from file "{1:s}"...'.format(
                radar_field_names[j], radar_time_strings[i])

            radar_matrix_this_field, these_grid_point_heights_m_asl, _, _ = (
                gridrad_io.read_field_from_full_grid_file(
                    radar_file_names[i], field_name=radar_field_names[j],
                    metadata_dict=this_metadata_dict))

            these_grid_point_heights_m_asl = numpy.round(
                these_grid_point_heights_m_asl).astype(int)
            these_height_indices_to_keep = numpy.array(
                [these_grid_point_heights_m_asl.tolist().index(h)
                 for h in radar_heights_m_asl], dtype=int)
            del these_grid_point_heights_m_asl

            radar_matrix_this_field = (
                radar_matrix_this_field[these_height_indices_to_keep, :, :])
            radar_matrix_this_field[numpy.isnan(radar_matrix_this_field)] = 0.

            for k in range(num_radar_heights):

                # Compute radar stats for [j]th field at [k]th height and [i]th
                # valid time.
                print (
                    'Computing stats for "{0:s}" at {1:d} metres ASL and '
                    '{2:s}...').format(
                        radar_field_names[j], radar_heights_m_asl[k],
                        radar_time_strings[i])

                for this_storm_index in these_storm_indices:
                    these_grid_point_rows = storm_object_table[
                        tracking_utils.GRID_POINT_ROW_COLUMN].values[
                            this_storm_index].astype(int)
                    these_grid_point_columns = storm_object_table[
                        tracking_utils.GRID_POINT_COLUMN_COLUMN].values[
                            this_storm_index].astype(int)

                    radar_values_this_storm = extract_radar_grid_points(
                        field_matrix=numpy.flipud(
                            radar_matrix_this_field[k, :, :]),
                        row_indices=these_grid_point_rows,
                        column_indices=these_grid_point_columns)

                    (statistic_matrix[this_storm_index, j, k, :],
                     percentile_matrix[this_storm_index, j, k, :]) = (
                         get_spatial_statistics(
                             radar_values_this_storm,
                             statistic_names=statistic_names,
                             percentile_levels=percentile_levels))

            print '\n'

    # Create pandas DataFrame.
    storm_object_statistic_dict = {}
    for j in range(num_radar_fields):
        for k in range(num_radar_heights):
            for m in range(num_statistics):
                this_column_name = radar_field_and_statistic_to_column_name(
                    radar_field_name=radar_field_names[j],
                    radar_height_m_asl=radar_heights_m_asl[k],
                    statistic_name=statistic_names[m])

                storm_object_statistic_dict.update(
                    {this_column_name: statistic_matrix[:, j, k, m]})

            for m in range(num_percentiles):
                this_column_name = radar_field_and_percentile_to_column_name(
                    radar_field_name=radar_field_names[j],
                    radar_height_m_asl=radar_heights_m_asl[k],
                    percentile_level=percentile_levels[m])

                storm_object_statistic_dict.update(
                    {this_column_name: percentile_matrix[:, j, k, m]})

    storm_object_statistic_table = pandas.DataFrame.from_dict(
        storm_object_statistic_dict)
    return pandas.concat(
        [storm_object_table[STORM_COLUMNS_TO_KEEP],
         storm_object_statistic_table], axis=1)


def write_stats_for_storm_objects(storm_object_statistic_table,
                                  pickle_file_name):
    """Writes radar statistics for storm objects to a Pickle file.

    :param storm_object_statistic_table: pandas DataFrame created by
        get_stats_for_storm_objects.
    :param pickle_file_name: Path to output file.
    """

    statistic_column_names = check_statistic_table(
        storm_object_statistic_table, require_storm_objects=True)
    columns_to_write = STORM_COLUMNS_TO_KEEP + statistic_column_names

    file_system_utils.mkdir_recursive_if_necessary(file_name=pickle_file_name)
    pickle_file_handle = open(pickle_file_name, 'wb')
    pickle.dump(storm_object_statistic_table[columns_to_write],
                pickle_file_handle)
    pickle_file_handle.close()


def read_stats_for_storm_objects(pickle_file_name):
    """Reads radar statistics for storm objects from a Pickle file.

    :param pickle_file_name: Path to input file.
    :return: storm_object_statistic_table: pandas DataFrame with columns
        documented in get_stats_for_storm_objects.
    """

    pickle_file_handle = open(pickle_file_name, 'rb')
    storm_object_statistic_table = pickle.load(pickle_file_handle)
    pickle_file_handle.close()

    check_statistic_table(
        storm_object_statistic_table, require_storm_objects=True)
    return storm_object_statistic_table
