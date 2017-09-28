"""IO methods for raw wind data.  Raw wind data come from 4 different sources:

- MADIS (Meteorological Assimilation Data Ingest System)
- HFMETARs (high-frequency [1-minute and 5-minute] meteorological aerodrome
  reports)
- Oklahoma Mesonet stations
- LSRs (local storm reports)
"""

import numpy
import pandas
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.gg_utils import error_checking

# TODO(thunderhoser): find better way to deal with missing wind directions.
# Currently changing them all to 0 degrees, but this masks the difference
# between actual 0 degrees and NaN.

TOLERANCE = 1e-6
WIND_DIR_DEFAULT_DEG = 0.

DEGREES_TO_RADIANS = numpy.pi / 180
RADIANS_TO_DEGREES = 180. / numpy.pi

MIN_WIND_DIRECTION_DEG = 0.
MAX_WIND_DIRECTION_DEG = 360. - TOLERANCE
MIN_WIND_SPEED_M_S01 = 0.
MAX_WIND_SPEED_M_S01 = 100.
MIN_ELEVATION_M_ASL = -418.  # Lowest point on land (shore of Dead Sea).
MAX_ELEVATION_M_ASL = 8848.  # Highest point on land (Mount Everest).
MIN_LATITUDE_DEG = -90.
MAX_LATITUDE_DEG = 90.
MIN_LONGITUDE_DEG = -180.
MAX_LONGITUDE_DEG = 360.
MIN_LNG_NEGATIVE_IN_WEST_DEG = -180.
MAX_LNG_NEGATIVE_IN_WEST_DEG = 180.
MIN_LNG_POSITIVE_IN_WEST_DEG = 0.
MAX_LNG_POSITIVE_IN_WEST_DEG = 360.

STATION_ID_COLUMN = 'station_id'
STATION_NAME_COLUMN = 'station_name'
LATITUDE_COLUMN = 'latitude_deg'
LONGITUDE_COLUMN = 'longitude_deg'
ELEVATION_COLUMN = 'elevation_m_asl'
WIND_SPEED_COLUMN = 'wind_speed_m_s01'
WIND_DIR_COLUMN = 'wind_direction_deg'
WIND_GUST_SPEED_COLUMN = 'wind_gust_speed_m_s01'
WIND_GUST_DIR_COLUMN = 'wind_gust_direction_deg'
U_WIND_COLUMN = 'u_wind_m_s01'
V_WIND_COLUMN = 'v_wind_m_s01'
TIME_COLUMN = 'unix_time_sec'


def append_source_to_station_id(station_id, data_source):
    """Appends data source to station ID.

    There are 4 possible data sources: "madis", "hfmetar", "ok_mesonet", and
    "lsr".  For the non-abbreviated versions, see the docstring for this file.

    The data source will be append with an underscore.  For example, if the
    station ID is "CYEG" and data source is "madis", the new ID will be
    "CYEG_madis".  Stations from different sources might have the same IDs, so
    this ensures the uniqueness of IDs.

    :param station_id: String ID for station.
    :param data_source: String ID for data source.
    :return: station_id: Same as input, but with underscore and data source
        appended.
    """

    error_checking.assert_is_string(station_id)
    error_checking.assert_is_string(data_source)
    return '{0:s}_{1:s}'.format(station_id, data_source)


def check_elevations(elevations_m_asl):
    """Finds invalid surface elevations.

    N = number of elevations

    :param elevations_m_asl: length-N numpy array of elevations (metres above
        sea level).
    :return: invalid_indices: 1-D numpy array with indices of invalid surface
        elevations.  For example, if 5th and 12th elevations are invalid, this
        array will contain 4 and 11.
    """

    error_checking.assert_is_real_numpy_array(elevations_m_asl)
    error_checking.assert_is_numpy_array(elevations_m_asl, num_dimensions=1)

    valid_flags = numpy.logical_and(elevations_m_asl >= MIN_ELEVATION_M_ASL,
                                    elevations_m_asl <= MAX_ELEVATION_M_ASL)
    return numpy.where(numpy.invert(valid_flags))[0]


def check_latitudes(latitudes_deg):
    """Finds invalid latitudes.

    N = number of latitudes.

    :param latitudes_deg: length-N numpy array of latitudes (deg N).
    :return: invalid_indices: 1-D numpy array with indices of invalid latitudes.
    """

    error_checking.assert_is_real_numpy_array(latitudes_deg)
    error_checking.assert_is_numpy_array(latitudes_deg, num_dimensions=1)

    valid_flags = numpy.logical_and(latitudes_deg >= MIN_LATITUDE_DEG,
                                    latitudes_deg <= MAX_LATITUDE_DEG)
    return numpy.where(numpy.invert(valid_flags))[0]


def check_longitudes(longitudes_deg):
    """Finds invalid longitudes.

    N = number of longitudes.

    :param longitudes_deg: length-N numpy array of longitudes (deg E).
    :return: invalid_indices: 1-D numpy array with indices of invalid
        longitudes.
    """

    error_checking.assert_is_real_numpy_array(longitudes_deg)
    error_checking.assert_is_numpy_array(longitudes_deg, num_dimensions=1)

    valid_flags = numpy.logical_and(longitudes_deg >= MIN_LONGITUDE_DEG,
                                    longitudes_deg <= MAX_LONGITUDE_DEG)
    return numpy.where(numpy.invert(valid_flags))[0]


def check_longitudes_negative_in_west(longitudes_deg):
    """Finds invalid longitudes.

    N = number of longitudes.

    :param longitudes_deg: length-N numpy array of longitudes (deg E), where
        values in western hemisphere are negative.
    :return: invalid_indices: 1-D numpy array with indices of invalid
        longitudes.
    """

    error_checking.assert_is_real_numpy_array(longitudes_deg)
    error_checking.assert_is_numpy_array(longitudes_deg, num_dimensions=1)

    valid_flags = numpy.logical_and(
        longitudes_deg >= MIN_LNG_NEGATIVE_IN_WEST_DEG,
        longitudes_deg <= MAX_LNG_NEGATIVE_IN_WEST_DEG)
    return numpy.where(numpy.invert(valid_flags))[0]


def check_longitudes_positive_in_west(longitudes_deg):
    """Finds invalid longitudes.

    N = number of longitudes.

    :param longitudes_deg: length-N numpy array of longitudes (deg E), where
        values in western hemisphere are positive.
    :return: invalid_indices: 1-D numpy array with indices of invalid
        longitudes.
    """

    error_checking.assert_is_real_numpy_array(longitudes_deg)
    error_checking.assert_is_numpy_array(longitudes_deg, num_dimensions=1)

    valid_flags = numpy.logical_and(
        longitudes_deg >= MIN_LNG_POSITIVE_IN_WEST_DEG,
        longitudes_deg <= MAX_LNG_POSITIVE_IN_WEST_DEG)
    return numpy.where(numpy.invert(valid_flags))[0]


def check_wind_speeds(wind_speeds_m_s01):
    """Finds invalid wind speeds.

    N = number of observations.

    :param wind_speeds_m_s01: length-N numpy array of wind speeds (m/s).
    :return: invalid_indices: 1-D numpy array with indices of invalid speeds.
    """

    error_checking.assert_is_real_numpy_array(wind_speeds_m_s01)
    error_checking.assert_is_numpy_array(wind_speeds_m_s01, num_dimensions=1)

    valid_flags = numpy.logical_and(
        wind_speeds_m_s01 >= MIN_WIND_SPEED_M_S01,
        wind_speeds_m_s01 <= MAX_WIND_SPEED_M_S01)
    return numpy.where(numpy.invert(valid_flags))[0]


def check_wind_directions(wind_directions_deg):
    """Finds invalid wind directions.

    N = number of observations

    :param wind_directions_deg: length-N numpy array of wind directions (degrees
        of origin).
    :return: invalid_indices: 1-D numpy array with indices of invalid
        directions.
    """

    error_checking.assert_is_real_numpy_array(wind_directions_deg)
    error_checking.assert_is_numpy_array(wind_directions_deg, num_dimensions=1)

    valid_flags = numpy.logical_and(
        wind_directions_deg >= MIN_WIND_DIRECTION_DEG,
        wind_directions_deg <= MAX_WIND_DIRECTION_DEG)
    return numpy.where(numpy.invert(valid_flags))[0]


def get_max_of_sustained_and_gust(wind_speeds_m_s01, wind_gust_speeds_m_s01,
                                  wind_directions_deg,
                                  wind_gust_directions_deg):
    """Converts wind data from 4 variables to 2.

    Original variables:

    - speed of sustained wind
    - direction of sustained wind
    - speed of wind gust
    - direction of wind gust

    New variables:

    - speed of max wind (whichever is higher between sustained and gust)
    - direction of max wind (whichever is higher between sustained and gust)

    "Whichever is higher between sustained and gust" sounds like a stupid
    phrase, because gust speed should always be >= sustained speed.  However,
    due to quality issues, this is not always the case.

    N = number of wind observations.

    :param wind_speeds_m_s01: length-N numpy array of sustained speeds (m/s).
    :param wind_gust_speeds_m_s01: length-N numpy array of gust speeds (m/s).
    :param wind_directions_deg: length-N numpy array of sustained directions
        (degrees of origin).
    :param wind_gust_directions_deg: length-N numpy array of gust directions
        (degrees of origin).
    :return: max_wind_speeds_m_s01: length-N numpy array of max wind speeds
        (m/s).
    :return: max_wind_directions_deg: length-N numpy array with directions of
        max wind (degrees of origin).
    """

    error_checking.assert_is_real_numpy_array(wind_speeds_m_s01)
    error_checking.assert_is_numpy_array(wind_speeds_m_s01, num_dimensions=1)
    num_observations = len(wind_speeds_m_s01)

    error_checking.assert_is_real_numpy_array(wind_gust_speeds_m_s01)
    error_checking.assert_is_numpy_array(
        wind_gust_speeds_m_s01,
        exact_dimensions=numpy.array([num_observations]))

    error_checking.assert_is_real_numpy_array(wind_directions_deg)
    error_checking.assert_is_numpy_array(
        wind_directions_deg,
        exact_dimensions=numpy.array([num_observations]))

    error_checking.assert_is_real_numpy_array(wind_gust_directions_deg)
    error_checking.assert_is_numpy_array(
        wind_gust_directions_deg,
        exact_dimensions=numpy.array([num_observations]))

    wind_speed_matrix_m_s01 = numpy.vstack(
        (wind_speeds_m_s01, wind_gust_speeds_m_s01))
    wind_direction_matrix_deg = numpy.vstack(
        (wind_directions_deg, wind_gust_directions_deg))

    max_wind_speeds_m_s01 = numpy.nanmax(wind_speed_matrix_m_s01,
                                         axis=0).astype(numpy.float64)
    row_indices = numpy.nanargmax(wind_speed_matrix_m_s01, axis=0)

    num_observations = len(max_wind_speeds_m_s01)
    column_indices = numpy.linspace(0, num_observations - 1,
                                    num=num_observations, dtype=int)

    linear_indices = numpy.ravel_multi_index((row_indices, column_indices),
                                             (2, num_observations))
    all_wind_directions_deg = numpy.reshape(wind_direction_matrix_deg,
                                            2 * num_observations)
    max_wind_directions_deg = all_wind_directions_deg[linear_indices]

    return max_wind_speeds_m_s01, max_wind_directions_deg


def speed_and_direction_to_uv(wind_speeds_m_s01, wind_directions_deg):
    """Converts wind vectors from speed and direction to u- and v-components.

    N = number of wind vectors

    :param wind_speeds_m_s01: length-N numpy array of wind speeds (m/s).
    :param wind_directions_deg: length-N numpy array of wind directions (degrees
        of origin).
    :return: u_winds_m_s01: length-N numpy array of u-components (m/s).
    :return: v_winds_m_s01: length-N numpy array of v-components (m/s).
    """

    error_checking.assert_is_real_numpy_array(wind_speeds_m_s01)
    error_checking.assert_is_numpy_array(wind_speeds_m_s01, num_dimensions=1)
    num_observations = len(wind_speeds_m_s01)

    error_checking.assert_is_real_numpy_array(wind_directions_deg)
    error_checking.assert_is_numpy_array(
        wind_directions_deg,
        exact_dimensions=numpy.array([num_observations]))

    u_winds_m_s01 = -1 * wind_speeds_m_s01 * numpy.sin(
        wind_directions_deg * DEGREES_TO_RADIANS)
    v_winds_m_s01 = -1 * wind_speeds_m_s01 * numpy.cos(
        wind_directions_deg * DEGREES_TO_RADIANS)
    return u_winds_m_s01, v_winds_m_s01


def uv_to_speed_and_direction(u_winds_m_s01, v_winds_m_s01):
    """Converts wind vectors from u- and v-components to speed and direction.

    N = number of wind vectors

    :param u_winds_m_s01: length-N numpy array of u-components (m/s).
    :param v_winds_m_s01: length-N numpy array of v-components (m/s).
    :return: wind_speeds_m_s01: length-N numpy array of wind speeds (m/s).
    :return: wind_directions_deg: length-N numpy array of wind directions
        (degrees of origin).
    """

    error_checking.assert_is_real_numpy_array(u_winds_m_s01)
    error_checking.assert_is_numpy_array(u_winds_m_s01, num_dimensions=1)
    num_observations = len(u_winds_m_s01)

    error_checking.assert_is_real_numpy_array(v_winds_m_s01)
    error_checking.assert_is_numpy_array(
        v_winds_m_s01, exact_dimensions=numpy.array([num_observations]))

    wind_directions_deg = RADIANS_TO_DEGREES * numpy.arctan2(-u_winds_m_s01,
                                                             -v_winds_m_s01)
    wind_directions_deg = numpy.mod(wind_directions_deg + 360., 360)

    wind_speeds_m_s01 = numpy.sqrt(u_winds_m_s01 ** 2 + v_winds_m_s01 ** 2)
    return wind_speeds_m_s01, wind_directions_deg


def sustained_and_gust_to_uv_max(wind_table):
    """Converts each wind observation from 4 variables to 2.

    Original variables:

    - speed of sustained wind
    - direction of sustained wind
    - speed of wind gust
    - direction of wind gust

    New variables:

    - speed of max wind (whichever is higher between sustained and gust)
    - direction of max wind (whichever is higher between sustained and gust)

    "Whichever is higher between sustained and gust" sounds like a stupid
    phrase, because gust speed should always be >= sustained speed.  However,
    due to quality issues, this is not always the case.

    :param wind_table: pandas DataFrame with at least 4 columns.
    wind_table.wind_speed_m_s01: Speed of sustained wind (m/s).
    wind_table.wind_direction_deg: Direction of sustained wind (degrees of
        origin -- i.e., direction that the wind is coming from -- as per
        meteorological convention).
    wind_table.wind_gust_speed_m_s01: Speed of wind gust (m/s).
    wind_table.wind_gust_direction_deg: Direction of wind gust (degrees of
        origin).
    :return: wind_table: Same as input, but with 4 columns ("wind_speed_m_s01",
        "wind_direction_deg", "wind_gust_speed_m_s01",
        "wind_gust_direction_deg") removed and the following columns added.
    wind_table.u_wind_m_s01: Northward component (m/s) of wind.
    wind_table.v_wind_m_s01: Eastward component (m/s) of wind.
    """

    (max_wind_speeds_m_s01,
     max_wind_directions_deg) = get_max_of_sustained_and_gust(
         wind_table[WIND_SPEED_COLUMN].values,
         wind_table[WIND_GUST_SPEED_COLUMN].values,
         wind_table[WIND_DIR_COLUMN].values,
         wind_table[WIND_GUST_DIR_COLUMN].values)

    columns_to_drop = [WIND_SPEED_COLUMN, WIND_DIR_COLUMN,
                       WIND_GUST_SPEED_COLUMN, WIND_GUST_DIR_COLUMN]
    wind_table.drop(columns_to_drop, axis=1, inplace=True)

    u_winds_m_s01, v_winds_m_s01 = speed_and_direction_to_uv(
        max_wind_speeds_m_s01, max_wind_directions_deg)

    argument_dict = {U_WIND_COLUMN: u_winds_m_s01, V_WIND_COLUMN: v_winds_m_s01}
    return wind_table.assign(**argument_dict)


def write_station_metadata_to_csv(station_metadata_table, csv_file_name):
    """Writes metadata for weather stations* to CSV file.

    * These may be either HFMETAR or Oklahoma Mesonet stations.

    :param station_metadata_table: pandas DataFrame with the following columns.
    station_metadata_table.station_id: String ID for station.
    station_metadata_table.station_name: Verbose name for station.
    station_metadata_table.latitude_deg: Latitude (deg N).
    station_metadata_table.longitude_deg: Longitude (deg E).
    station_metadata_table.elevation_m_asl: Elevation (metres above sea level).
    station_metadata_table.utc_offset_hours: [HFMETAR only] Difference between
        local station time and UTC (local minus UTC).
    :param csv_file_name: Path to output file.
    """

    file_system_utils.mkdir_recursive_if_necessary(file_name=csv_file_name)
    station_metadata_table.to_csv(csv_file_name, header=True, sep=',',
                                  index=False)


def read_station_metadata_from_csv(csv_file_name):
    """Reads metadata for weather stations* from CSV file.

    * These may be either HFMETAR or Oklahoma Mesonet stations.

    :param csv_file_name: Path to input file.
    :return: station_metadata_table: pandas DataFrame with columns specified in
        write_station_metadata_to_csv
    """

    error_checking.assert_file_exists(csv_file_name)
    return pandas.read_csv(csv_file_name, header=0, sep=',')


def write_winds_to_csv(wind_table, csv_file_name):
    """Writes wind data to CSV file.

    :param wind_table: pandas DataFrame with the following columns.
    wind_table.station_id: String ID for station.
    wind_table.station_name: Verbose name for station.
    wind_table.latitude_deg: Latitude (deg N).
    wind_table.longitude_deg: Longitude (deg E).
    wind_table.elevation_m_asl: Elevation (metres above sea level).
    wind_table.unit_time_sec: Observation time (seconds since 0000 UTC 1 Jan
        1970).
    wind_table.u_wind_m_s01: Northward component (m/s) of wind.
    wind_table.v_wind_m_s01: Eastward component (m/s) of wind.
    :param csv_file_name: Path to output file.
    """

    file_system_utils.mkdir_recursive_if_necessary(file_name=csv_file_name)
    wind_table.to_csv(csv_file_name, header=True, sep=',', index=False)


def read_winds_from_csv(csv_file_name):
    """Reads wind data from CSV file.

    :param csv_file_name: Path to input file.
    :return: wind_table: pandas DataFrame with columns produced by
        sustained_and_gust_to_uv_max.
    """

    error_checking.assert_file_exists(csv_file_name)
    return pandas.read_csv(csv_file_name, header=0, sep=',')
