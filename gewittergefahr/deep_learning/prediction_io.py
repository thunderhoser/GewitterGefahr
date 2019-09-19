"""IO methods for probabilistic predictions."""

import pickle
import os.path
import numpy
import netCDF4
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import target_val_utils
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.gg_utils import error_checking

FILE_NAME_TIME_FORMAT = '%Y-%m-%d-%H%M%S'
UNGRIDDED_FILE_NAME_PREFIX = 'ungridded_predictions'
GRIDDED_FILE_NAME_PREFIX = 'gridded_predictions'

EXAMPLE_DIMENSION_KEY = 'storm_object'
CLASS_DIMENSION_KEY = 'class'
STORM_ID_CHAR_DIM_KEY = 'storm_id_character'

TARGET_NAME_KEY = 'target_name'
STORM_IDS_KEY = 'storm_ids'
STORM_TIMES_KEY = 'storm_times_unix_sec'
PROBABILITY_MATRIX_KEY = 'class_probability_matrix'
OBSERVED_LABELS_KEY = 'observed_labels'

REQUIRED_KEYS_FOR_UNGRIDDED = [
    TARGET_NAME_KEY, STORM_IDS_KEY, STORM_TIMES_KEY, PROBABILITY_MATRIX_KEY,
    OBSERVED_LABELS_KEY
]

INIT_TIMES_KEY = 'init_times_unix_sec'
MIN_LEAD_TIME_KEY = 'min_lead_time_seconds'
MAX_LEAD_TIME_KEY = 'max_lead_time_seconds'
GRID_X_COORDS_KEY = 'grid_point_x_coords_metres'
GRID_Y_COORDS_KEY = 'grid_point_y_coords_metres'
GRID_LATITUDES_KEY = 'grid_point_latitudes_deg'
GRID_LONGITUDES_KEY = 'grid_point_longitudes_deg'
XY_PROBABILITIES_KEY = 'sparse_prob_matrices_xy'
LATLNG_PROBABILITIES_KEY = 'sparse_prob_matrices_latlng'
PROJECTION_KEY = 'projection_object'

REQUIRED_KEYS_FOR_GRIDDED = [
    INIT_TIMES_KEY, MIN_LEAD_TIME_KEY, MAX_LEAD_TIME_KEY, GRID_X_COORDS_KEY,
    GRID_Y_COORDS_KEY, XY_PROBABILITIES_KEY, PROJECTION_KEY
]

LATLNG_KEYS = [
    GRID_LATITUDES_KEY, GRID_LONGITUDES_KEY, LATLNG_PROBABILITIES_KEY
]


def subset_ungridded_predictions(prediction_dict, desired_storm_indices):
    """Subsets ungridded predictions.

    :param prediction_dict: See doc for `read_ungridded_predictions`.
    :param desired_storm_indices: 1-D numpy array with indices of desired storm
        objects.
    :return: small_prediction_dict: Same as input but maybe with fewer storm
        objects.
    :raises: ValueError: if dictionary is missing any expected keys.
    """

    missing_keys = list(
        set(REQUIRED_KEYS_FOR_UNGRIDDED) - set(prediction_dict.keys())
    )

    if len(missing_keys) > 0:
        error_string = (
            '\n{0:s}\nKeys listed above were expected, but not found, in '
            'dictionary.'
        ).format(str(missing_keys))

        raise ValueError(error_string)

    small_prediction_dict = dict()

    for this_key in prediction_dict:
        if isinstance(prediction_dict[this_key], list):
            small_prediction_dict[this_key] = [
                prediction_dict[this_key][k] for k in desired_storm_indices
            ]
        elif isinstance(prediction_dict[this_key], numpy.ndarray):
            small_prediction_dict[this_key] = prediction_dict[this_key][
                desired_storm_indices, ...]
        else:
            small_prediction_dict[this_key] = prediction_dict[this_key]

    return small_prediction_dict


def find_ungridded_file(
        directory_name, raise_error_if_missing=True, months_in_subset=None,
        hours_in_subset=None, grid_row=None, grid_column=None):
    """Finds file with ungridded predictions.

    If file is a temporal subset, `months_in_subset` or `hours_in_subset` must
    be specified.

    If file is a spatial subset, `grid_row` and `grid_column` must be specified.

    :param directory_name: Directory name.
    :param raise_error_if_missing: Boolean flag.  If file is missing and
        `raise_error_if_missing = True`, this method will error out.
    :param months_in_subset: 1-D numpy array of months in subset (range 1...12).
    :param hours_in_subset: 1-D numpy array of hours in subset (range 0...23).
    :param grid_row: Grid row in subset (integer).
    :param grid_column: Grid column in subset (integer).
    :return: prediction_file_name: Path to prediction file.  If file is missing
        and `raise_error_if_missing = False`, this will be the expected path.
    :raises: ValueError: if file is missing and `raise_error_if_missing = True`.
    """

    is_temporal_subset = False

    if months_in_subset is not None:
        is_temporal_subset = True
        hours_in_subset = None
        grid_row = None
        grid_column = None

        error_checking.assert_is_integer_numpy_array(months_in_subset)
        error_checking.assert_is_numpy_array(months_in_subset, num_dimensions=1)
        error_checking.assert_is_geq_numpy_array(months_in_subset, 1)
        error_checking.assert_is_leq_numpy_array(months_in_subset, 12)

    if hours_in_subset is not None:
        is_temporal_subset = True
        grid_row = None
        grid_column = None

        error_checking.assert_is_integer_numpy_array(hours_in_subset)
        error_checking.assert_is_numpy_array(hours_in_subset, num_dimensions=1)
        error_checking.assert_is_geq_numpy_array(hours_in_subset, 0)
        error_checking.assert_is_leq_numpy_array(hours_in_subset, 23)

    is_spatial_subset = (
        not is_temporal_subset
        and grid_row is not None and grid_column is not None
    )

    if is_spatial_subset:
        error_checking.assert_is_integer(grid_row)
        error_checking.assert_is_geq(grid_row, 0)
        error_checking.assert_is_integer(grid_column)
        error_checking.assert_is_geq(grid_column, 0)

    prediction_file_name = '{0:s}/{1:s}'.format(
        directory_name, UNGRIDDED_FILE_NAME_PREFIX)

    if months_in_subset is not None:
        month_array_string = '-'.join([
            '{0:02d}'.format(m) for m in months_in_subset
        ])
        prediction_file_name += '_months={0:s}'.format(month_array_string)

    if hours_in_subset is not None:
        hour_array_string = '-'.join([
            '{0:02d}'.format(h) for h in hours_in_subset
        ])
        prediction_file_name += '_hours={0:s}'.format(hour_array_string)

    if is_spatial_subset:
        prediction_file_name += '_grid-row={0:04d}_grid-column={1:04d}'.format(
            grid_row, grid_column)

    prediction_file_name += '.nc'

    if raise_error_if_missing and not os.path.isfile(prediction_file_name):
        error_string = 'Cannot find file.  Expected at: "{0:s}"'.format(
            prediction_file_name)
        raise ValueError(error_string)

    return prediction_file_name


def find_gridded_file(
        directory_name, first_init_time_unix_sec, last_init_time_unix_sec,
        raise_error_if_missing=False):
    """Finds file with gridded predictions.

    :param directory_name: Directory name.
    :param first_init_time_unix_sec: First initial time (forecast-initialization
        time, not valid time) in file.
    :param last_init_time_unix_sec: Last initial time in file.
    :param raise_error_if_missing: Boolean flag.  If file is missing and
        `raise_error_if_missing = True`, this method will error out.
    :return: prediction_file_name: Path to prediction file.  If file is missing
        and `raise_error_if_missing = False`, this will be the expected path.
    :raises: ValueError: if file is missing and `raise_error_if_missing = True`.
    """

    error_checking.assert_is_string(directory_name)
    error_checking.assert_is_integer(first_init_time_unix_sec)
    error_checking.assert_is_integer(last_init_time_unix_sec)
    error_checking.assert_is_geq(
        last_init_time_unix_sec, first_init_time_unix_sec)
    error_checking.assert_is_boolean(raise_error_if_missing)

    prediction_file_name = '{0:s}/{1:s}_{2:s}_{3:s}.p'.format(
        directory_name, GRIDDED_FILE_NAME_PREFIX,
        time_conversion.unix_sec_to_string(
            first_init_time_unix_sec, FILE_NAME_TIME_FORMAT),
        time_conversion.unix_sec_to_string(
            last_init_time_unix_sec, FILE_NAME_TIME_FORMAT)
    )

    if raise_error_if_missing and not os.path.isfile(prediction_file_name):
        error_string = 'Cannot find file.  Expected at: "{0:s}"'.format(
            prediction_file_name)
        raise ValueError(error_string)

    return prediction_file_name


def write_ungridded_predictions(
        netcdf_file_name, class_probability_matrix, storm_ids,
        storm_times_unix_sec, target_name, observed_labels=None):
    """Writes predictions to NetCDF file.

    K = number of classes
    E = number of examples (storm objects)

    :param netcdf_file_name: Path to output file.
    :param class_probability_matrix: E-by-K numpy array of forecast
        probabilities.
    :param storm_ids: length-E list of storm IDs (strings).
    :param storm_times_unix_sec: length-E numpy array of valid times.
    :param target_name: Name of target variable.
    :param observed_labels: [this may be None]
        length-E numpy array of observed labels (integers in 0...[K - 1]).
    """

    # Check input args.
    error_checking.assert_is_numpy_array(
        class_probability_matrix, num_dimensions=2)
    error_checking.assert_is_geq_numpy_array(class_probability_matrix, 0.)
    error_checking.assert_is_leq_numpy_array(class_probability_matrix, 1.)

    num_examples = class_probability_matrix.shape[0]
    these_expected_dim = numpy.array([num_examples], dtype=int)

    error_checking.assert_is_string_list(storm_ids)
    error_checking.assert_is_numpy_array(
        numpy.array(storm_ids), exact_dimensions=these_expected_dim)

    error_checking.assert_is_integer_numpy_array(storm_times_unix_sec)
    error_checking.assert_is_numpy_array(
        storm_times_unix_sec, exact_dimensions=these_expected_dim)

    target_val_utils.target_name_to_params(target_name)

    if observed_labels is not None:
        error_checking.assert_is_integer_numpy_array(observed_labels)
        error_checking.assert_is_numpy_array(
            observed_labels, exact_dimensions=these_expected_dim)

    # Write to NetCDF file.
    file_system_utils.mkdir_recursive_if_necessary(file_name=netcdf_file_name)
    dataset_object = netCDF4.Dataset(
        netcdf_file_name, 'w', format='NETCDF3_64BIT_OFFSET')

    dataset_object.setncattr(TARGET_NAME_KEY, target_name)
    dataset_object.createDimension(
        EXAMPLE_DIMENSION_KEY, class_probability_matrix.shape[0]
    )
    dataset_object.createDimension(
        CLASS_DIMENSION_KEY, class_probability_matrix.shape[1]
    )

    if num_examples == 0:
        num_id_characters = 1
    else:
        num_id_characters = 1 + numpy.max(numpy.array([
            len(s) for s in storm_ids
        ]))

    dataset_object.createDimension(STORM_ID_CHAR_DIM_KEY, num_id_characters)

    # Add storm IDs.
    this_string_format = 'S{0:d}'.format(num_id_characters)
    storm_ids_char_array = netCDF4.stringtochar(numpy.array(
        storm_ids, dtype=this_string_format
    ))

    dataset_object.createVariable(
        STORM_IDS_KEY, datatype='S1',
        dimensions=(EXAMPLE_DIMENSION_KEY, STORM_ID_CHAR_DIM_KEY)
    )
    dataset_object.variables[STORM_IDS_KEY][:] = numpy.array(
        storm_ids_char_array)

    # Add storm times.
    dataset_object.createVariable(
        STORM_TIMES_KEY, datatype=numpy.int32, dimensions=EXAMPLE_DIMENSION_KEY
    )
    dataset_object.variables[STORM_TIMES_KEY][:] = storm_times_unix_sec

    # Add probabilities.
    dataset_object.createVariable(
        PROBABILITY_MATRIX_KEY, datatype=numpy.float32,
        dimensions=(EXAMPLE_DIMENSION_KEY, CLASS_DIMENSION_KEY)
    )
    dataset_object.variables[PROBABILITY_MATRIX_KEY][:] = (
        class_probability_matrix
    )

    if observed_labels is not None:
        dataset_object.createVariable(
            OBSERVED_LABELS_KEY, datatype=numpy.int32,
            dimensions=EXAMPLE_DIMENSION_KEY
        )
        dataset_object.variables[OBSERVED_LABELS_KEY][:] = observed_labels

    dataset_object.close()


def read_ungridded_predictions(netcdf_file_name):
    """Reads predictions from NetCDF file made by `write_ungridded_predictions`.

    :param netcdf_file_name: Path to input file.
    :return: prediction_dict: Dictionary with the following keys.
    prediction_dict['target_name']: See doc for `write_ungridded_predictions`.
    prediction_dict['storm_ids']: Same.
    prediction_dict['storm_times_unix_sec']: Same.
    prediction_dict['class_probability_matrix']: Same.
    prediction_dict['observed_labels']: See doc for
        `write_ungridded_predictions`.  This may be None.
    """

    dataset_object = netCDF4.Dataset(netcdf_file_name)
    num_examples = dataset_object.variables[STORM_IDS_KEY].shape[0]

    if num_examples == 0:
        num_classes = dataset_object.variables[PROBABILITY_MATRIX_KEY].shape[1]

        prediction_dict = {
            TARGET_NAME_KEY: str(getattr(dataset_object, TARGET_NAME_KEY)),
            STORM_IDS_KEY: [],
            STORM_TIMES_KEY: numpy.array([], dtype=int),
            PROBABILITY_MATRIX_KEY: numpy.full((0, num_classes), numpy.nan),
            OBSERVED_LABELS_KEY: None
        }

        if OBSERVED_LABELS_KEY in dataset_object.variables:
            prediction_dict[OBSERVED_LABELS_KEY] = numpy.array([], dtype=int)

        dataset_object.close()
        return prediction_dict

    prediction_dict = {
        TARGET_NAME_KEY: str(getattr(dataset_object, TARGET_NAME_KEY)),
        STORM_IDS_KEY: [
            str(s) for s in
            netCDF4.chartostring(dataset_object.variables[STORM_IDS_KEY][:])
        ],
        STORM_TIMES_KEY: numpy.array(
            dataset_object.variables[STORM_TIMES_KEY][:], dtype=int
        ),
        PROBABILITY_MATRIX_KEY:
            dataset_object.variables[PROBABILITY_MATRIX_KEY][:],
        OBSERVED_LABELS_KEY: None
    }

    if OBSERVED_LABELS_KEY in dataset_object.variables:
        prediction_dict[OBSERVED_LABELS_KEY] = (
            dataset_object.variables[OBSERVED_LABELS_KEY][:].astype(int)
        )

    dataset_object.close()
    return prediction_dict


def write_gridded_predictions(gridded_forecast_dict, pickle_file_name):
    """Writes gridded predictions to Pickle file.

    T = number of forecast grids = number of initial times
    J = number of rows in x-y grid
    K = number of columns in x-y grid
    M = number of rows in lat-long grid
    N = number of columns in lat-long grid

    The dictionary may or may not include lat-long grids.  If it doesn't, the
    following keys are not expected and will not be used:

    - "grid_point_latitudes_deg"
    - "grid_point_longitudes_deg"
    - "sparse_prob_matrices_latlng"

    :param gridded_forecast_dict: Dictionary with the following keys.
    gridded_forecast_dict['init_times_unix_sec']: length-T numpy array of
        initial times.
    gridded_forecast_dict['min_lead_time_seconds']: Minimum lead time in
        forecast window (same for each initial time).
    gridded_forecast_dict['max_lead_time_seconds']: Max lead time in forecast
        window (same for each initial time).
    gridded_forecast_dict['grid_point_x_coords_metres']: length-K numpy array of
        x-coordinates.
    gridded_forecast_dict['grid_point_y_coords_metres']: length-J numpy array of
        y-coordinates.
    gridded_forecast_dict['grid_point_latitudes_deg']: length-M numpy array of
        latitudes (deg N).
    gridded_forecast_dict['grid_point_longitudes_deg']: length-N numpy array of
        longitudes (deg E).
    gridded_forecast_dict['sparse_prob_matrices_xy']: length-T list of
        probability matrices.  Each matrix has dimensions of J x K, but the
        matrices are sparse (instances of `scipy.sparse.csr_matrix`).
    gridded_forecast_dict['sparse_prob_matrices_latlng']: Same, except that the
        matrices have dimensions M x N.
    gridded_forecast_dict['projection_object']: Instance of `pyproj.Proj`,
        relating x-y coordinates to lat-long.

    :param pickle_file_name: Path to output file.
    :raises: ValueError: if any of the required keys are not in the dictionary.
    """

    # TODO(thunderhoser): Need more memory-efficient output format (probably
    # MYRORSS format -- or NetCDF, anyways).

    if any([k in gridded_forecast_dict for k in LATLNG_KEYS]):
        these_required_keys = REQUIRED_KEYS_FOR_GRIDDED + LATLNG_KEYS
    else:
        these_required_keys = REQUIRED_KEYS_FOR_GRIDDED

    missing_keys = list(
        set(these_required_keys) - set(gridded_forecast_dict.keys())
    )

    if len(missing_keys) > 0:
        error_string = (
            '\n{0:s}\nKeys listed above were expected, but not found, in file '
            '"{1:s}".'
        ).format(str(missing_keys), pickle_file_name)

        raise ValueError(error_string)

    file_system_utils.mkdir_recursive_if_necessary(file_name=pickle_file_name)

    pickle_file_handle = open(pickle_file_name, 'wb')
    pickle.dump(gridded_forecast_dict, pickle_file_handle)
    pickle_file_handle.close()


def read_gridded_predictions(pickle_file_name):
    """Reads gridded predictions from Pickle file.

    :param pickle_file_name: Path to input file.
    :return: gridded_forecast_dict: See doc for `write_gridded_predictions`.
    """

    pickle_file_handle = open(pickle_file_name, 'rb')
    gridded_forecast_dict = pickle.load(pickle_file_handle)
    pickle_file_handle.close()

    if any([k in gridded_forecast_dict for k in LATLNG_KEYS]):
        these_required_keys = REQUIRED_KEYS_FOR_GRIDDED + LATLNG_KEYS
    else:
        these_required_keys = REQUIRED_KEYS_FOR_GRIDDED

    missing_keys = list(
        set(these_required_keys) - set(gridded_forecast_dict.keys())
    )

    if len(missing_keys) == 0:
        return gridded_forecast_dict

    error_string = (
        '\n{0:s}\nKeys listed above were expected, but not found, in file '
        '"{1:s}".'
    ).format(str(missing_keys), pickle_file_name)

    raise ValueError(error_string)
