"""Methods for error-checking.

These methods are designed mainly to check for errors in input arguments.
"""

import copy
import numbers
import os.path
import numpy

# TODO(thunderhoser): Fix hack in `REAL_NUMBER_TYPES` (where I added a bunch of
# numpy types).  There must be a better way to deal with all numpy float and int
# types.

BOOLEAN_TYPES = (bool, numpy.bool_)
REAL_NUMBER_TYPES = (float, numpy.float16, numpy.float32, numpy.float64,
                     numbers.Integral)
TREE_TYPES = (tuple, list)
ARRAY_TYPES = (tuple, list, numpy.ndarray)

MIN_LATITUDE_DEG = -90.
MAX_LATITUDE_DEG = 90.
MIN_LNG_NEGATIVE_IN_WEST_DEG = -180.
MAX_LNG_NEGATIVE_IN_WEST_DEG = 180.
MIN_LNG_POSITIVE_IN_WEST_DEG = 0.
MAX_LNG_POSITIVE_IN_WEST_DEG = 360.
MIN_LONGITUDE_DEG = -180.
MAX_LONGITUDE_DEG = 360.


def _traverse_array(input_array):
    """Traverses array.

    :param input_array: numpy array, list with any kind of nesting, or tuple
        with any kind of nesting.
    :return: array_generator: Generator (iterates through array).
    """

    if isinstance(input_array, TREE_TYPES):
        for this_value in input_array:
            for this_subvalue in _traverse_array(this_value):
                yield this_subvalue

    elif isinstance(input_array, numpy.ndarray):
        for _, this_value in numpy.ndenumerate(input_array):
            yield this_value

    else:
        yield input_array


def assert_is_array(input_variable):
    """Input variable must be tuple, list, or numpy array.

    :param input_variable: Input variable.
    :raises: TypeError: if input variable is *not* tuple, list, or numpy array.
    """

    if not isinstance(input_variable, ARRAY_TYPES):
        error_string = ('\n' + str(input_variable) +
                        '\nInput variable (shown above) is not an array.')
        raise TypeError(error_string)


def assert_is_list(input_variable):
    """Input variable must be list.

    :param input_variable: Input variable.
    :return: TypeError: if input variable is not a list.
    """

    if not isinstance(input_variable, list):
        error_string = ('\n' + str(input_variable) +
                        '\nInput variable (shown above) is not a list.')
        raise TypeError(error_string)


def assert_is_tuple(input_variable):
    """Input variable must be tuple.

    :param input_variable: Input variable.
    :return: TypeError: if input variable is not a tuple.
    """

    if not isinstance(input_variable, tuple):
        error_string = ('\n' + str(input_variable) +
                        '\nInput variable (shown above) is not a tuple.')
        raise TypeError(error_string)


def assert_is_numpy_array(input_variable, num_dimensions=None,
                          exact_dimensions=None):
    """Input variable must be numpy array.

    This method may also check for certain dimensions.

    N = number of dimensions

    :param input_variable: Input variable.
    :param num_dimensions: If None, will not check number of dimensions.  If
        defined, must be positive integer.
    :param exact_dimensions: If None, will not check exact dimensions.  If
        defined, must be length-N numpy array of positive integers.
    :return: TypeError: `input_variable` is not numpy array; `num_dimensions` is
        not positive integer; `exact_dimensions` is not 1-D numpy array of
        positive integers; or `input_variable` does not have given dimensions or
        number of dimensions.
    """

    if not isinstance(input_variable, numpy.ndarray):
        error_string = ('\n' + str(input_variable) + '\n' +
                        'Input variable (shown above) is not numpy array.')
        raise TypeError(error_string)

    if exact_dimensions is not None:
        assert_is_integer_numpy_array(exact_dimensions)
        assert_is_greater_numpy_array(exact_dimensions, 0)
        assert_is_numpy_array(exact_dimensions, num_dimensions=1)

        if not numpy.array_equal(input_variable.shape, exact_dimensions):
            error_string = (
                '\nExpected dimensions: ' + str(exact_dimensions) +
                '\nActual dimensions: ' + str(input_variable.shape) +
                '\nAs shown above, input array has unexpected dimensions.')
            raise TypeError(error_string)

    elif num_dimensions is not None:
        assert_is_integer(num_dimensions)
        assert_is_greater(num_dimensions, 0)

        if input_variable.ndim != num_dimensions:
            error_string = (
                'Input array should have ' + str(num_dimensions) +
                ' dimensions.  Got ' + str(input_variable.ndim) +
                ' dimensions.')
            raise TypeError(error_string)


def assert_is_non_array(input_variable):
    """Input variable must *not* be tuple, list, or numpy array.

    :param input_variable: Input variable.
    :raises: TypeError: if input variable is tuple, list, or numpy array.
    """

    if isinstance(input_variable, ARRAY_TYPES):
        error_string = ('\n' + str(input_variable) +
                        '\nInput variable (shown above) is an array.')
        raise TypeError(error_string)


def assert_is_string(input_variable):
    """Input variable must be string.

    :param input_variable: Input variable.
    :raises: TypeError: if input variable is not string.
    """

    if not isinstance(input_variable, str):
        error_string = ('\n' + str(input_variable) +
                        '\nInput variable (shown above) is not a string.')
        raise TypeError(error_string)


def assert_is_string_list(input_variable):
    """Input variable must be list of strings.

    :param input_variable: Input variable.
    :raises: TypeError: if input variable is not string list.
    """

    assert_is_list(input_variable)
    for this_value in _traverse_array(input_variable):
        assert_is_string(this_value)


def assert_file_exists(file_name):
    """Input variable must be name of existent file.

    :param file_name: File name.
    :raises: ValueError: if file does not exist.
    """

    assert_is_string(file_name)
    if not os.path.isfile(file_name):
        raise ValueError('File ("' + file_name + '") does not exist.')


def assert_directory_exists(directory_name):
    """Input variable must be name of existent directory.

    :param directory_name: Directory name.
    :raises: ValueError: if directory does not exist.
    """

    assert_is_string(directory_name)
    if not os.path.isdir(directory_name):
        raise ValueError('Directory ("' + directory_name + '") does not exist.')


def assert_is_integer(input_variable):
    """Input variable must be integer.

    :param input_variable: Input variable.
    :raises: TypeError: if input variable is not integer.
    """

    if (isinstance(input_variable, BOOLEAN_TYPES) or not isinstance(
            input_variable, numbers.Integral)):
        error_string = ('\n' + str(input_variable) +
                        '\nInput variable (shown above) is not integer.')
        raise TypeError(error_string)


def assert_is_integer_numpy_array(input_variable):
    """Input variable must be numpy array of integers.

    :param input_variable: Input variable.
    :raises: TypeError: if input variable is not numpy array of integers.
    """

    assert_is_numpy_array(input_variable)
    if not numpy.issubdtype(input_variable.dtype, int):
        error_string = (
            '\n' + str(input_variable) +
            '\nInput array (shown above) has type "' +
            str(input_variable.dtype) + '", which is not integer.')
        raise TypeError(error_string)


def assert_is_boolean(input_variable):
    """Input variable must be Boolean.

    :param input_variable: Input variable.
    :raises: TypeError: if input variable is not Boolean.
    """

    if not isinstance(input_variable, BOOLEAN_TYPES):
        error_string = ('\n' + str(input_variable) +
                        '\nInput variable (shown above) is not Boolean.')
        raise TypeError(error_string)


def assert_is_boolean_numpy_array(input_variable):
    """Input variable must be numpy array of Booleans.

    :param input_variable: Input variable.
    :raises: TypeError: if input variable is not numpy array of Booleans.
    """

    assert_is_numpy_array(input_variable)
    if input_variable.dtype != numpy.bool_:
        error_string = (
            '\n' + str(input_variable) +
            '\nInput array (shown above) has type "' +
            str(input_variable.dtype) + '", which is not Boolean.')
        raise TypeError(error_string)


def assert_is_float(input_variable):
    """Input variable must be float.

    :param input_variable: Input variable.
    :raises: TypeError: if input variable is not float.
    """

    if not isinstance(input_variable, float):
        error_string = ('\n' + str(input_variable) +
                        '\nInput variable (shown above) is not float.')
        raise TypeError(error_string)


def assert_is_float_numpy_array(input_variable):
    """Input variable must be numpy array of floats.

    :param input_variable: Input variable.
    :raises: TypeError: if input variable is not numpy array of floats.
    """

    assert_is_numpy_array(input_variable)
    if not numpy.issubdtype(input_variable.dtype, float):
        error_string = (
            '\n' + str(input_variable) +
            '\nInput array (shown above) has type "' +
            str(input_variable.dtype) + '", which is not float.')
        raise TypeError(error_string)


def assert_is_real_number(input_variable):
    """Input variable must be real number.

    :param input_variable: Input variable.
    :raises: TypeError: if input variable is not real number.
    """

    if (isinstance(input_variable, BOOLEAN_TYPES) or not isinstance(
            input_variable, REAL_NUMBER_TYPES)):
        print type(input_variable)
        error_string = ('\n' + str(input_variable) +
                        '\nInput variable (shown above) is not real number.')
        raise TypeError(error_string)


def assert_is_real_numpy_array(input_variable):
    """Input variable must be numpy array of real numbers.

    :param input_variable: Input variable.
    :raises: TypeError: if input variable is not numpy array of real numbers.
    """

    assert_is_numpy_array(input_variable)
    if not (numpy.issubdtype(input_variable.dtype, int) or numpy.issubdtype(
            input_variable.dtype, float)):
        error_string = (
            '\n' + str(input_variable) +
            '\nInput array (shown above) has type "' +
            str(input_variable.dtype) + '", which is not a real number.')
        raise TypeError(error_string)


def assert_is_not_nan(input_variable):
    """Input variable must be real number and not NaN.

    :param input_variable: Input variable.
    :raises: ValueError: if input variable is NaN.
    """

    assert_is_real_number(input_variable)
    if numpy.isnan(input_variable):
        raise ValueError('Input variable is NaN.')


def assert_is_numpy_array_without_nan(input_variable):
    """Input variable must be numpy array of real numbers without NaN.

    :param input_variable: Input variable.
    :return: ValueError: if input array contains one or more NaN's.
    """

    assert_is_real_numpy_array(input_variable)
    if numpy.any(numpy.isnan(input_variable)):
        raise ValueError("Input array contains one or more NaN's.")


def assert_is_greater(input_variable, base_value, allow_nan=False):
    """Input variable must be real number > some value.

    :param input_variable: Input variable.
    :param base_value: Input variable must be > this number.
    :param allow_nan: Boolean flag.  If True, input variable is allowed to be
        NaN.
    :raises: ValueError: if input variable is not > base_value.
    """

    assert_is_boolean(allow_nan)
    assert_is_not_nan(base_value)
    if allow_nan:
        assert_is_real_number(input_variable)
    else:
        assert_is_not_nan(input_variable)

    if input_variable <= base_value:
        error_string = (
            'Input value (' + str(input_variable) + ') should be > base value ('
            + str(base_value) + ').')
        raise ValueError(error_string)


def assert_is_greater_numpy_array(input_variable, base_value, allow_nan=False):
    """Input variable must be numpy array with all elements > some value.

    :param input_variable: Input variable.
    :param base_value: All elements must be > this number.
    :param allow_nan: Boolean flag.  If True, array elements are allowed to be
        NaN.
    :return: ValueError: if any element is not > base_value.
    """

    assert_is_boolean(allow_nan)
    assert_is_not_nan(base_value)
    if allow_nan:
        assert_is_real_numpy_array(input_variable)
    else:
        assert_is_numpy_array_without_nan(input_variable)

    if numpy.any(input_variable <= base_value):
        error_string = (
            '\n' + str(input_variable) + '\nInput array (shown above) has ' +
            'some elements not > base value (' + str(base_value) + ').')
        raise ValueError(error_string)


def assert_is_less_than(input_variable, base_value, allow_nan=False):
    """Input variable must be real number < some value.

    :param input_variable: Input variable.
    :param base_value: Input variable must be < this number.
    :param allow_nan: Boolean flag.  If True, input variable is allowed to be
        NaN.
    :raises: ValueError: if input variable is not < base_value.
    """

    assert_is_boolean(allow_nan)
    assert_is_not_nan(base_value)
    if allow_nan:
        assert_is_real_number(input_variable)
    else:
        assert_is_not_nan(input_variable)

    if input_variable >= base_value:
        error_string = (
            'Input value (' + str(input_variable) + ') should be < base value ('
            + str(base_value) + ').')
        raise ValueError(error_string)


def assert_is_less_than_numpy_array(input_variable, base_value,
                                    allow_nan=False):
    """Input variable must be numpy array with all elements < some value.

    :param input_variable: Input variable.
    :param base_value: All elements must be < this number.
    :param allow_nan: Boolean flag.  If True, array elements are allowed to be
        NaN.
    :return: ValueError: if any element is not < base_value.
    """

    assert_is_boolean(allow_nan)
    assert_is_not_nan(base_value)
    if allow_nan:
        assert_is_real_numpy_array(input_variable)
    else:
        assert_is_numpy_array_without_nan(input_variable)

    if numpy.any(input_variable >= base_value):
        error_string = (
            '\n' + str(input_variable) + '\nInput array (shown above) has ' +
            'some elements not < base value (' + str(base_value) + ').')
        raise ValueError(error_string)


def assert_is_geq(input_variable, base_value, allow_nan=False):
    """Input variable must be real number >= some value.

    :param input_variable: Input variable.
    :param base_value: Input variable must be >= this number.
    :param allow_nan: Boolean flag.  If True, input variable is allowed to be
        NaN.
    :raises: ValueError: if input variable is not >= base_value.
    """

    assert_is_boolean(allow_nan)
    assert_is_not_nan(base_value)
    if allow_nan:
        assert_is_real_number(input_variable)
    else:
        assert_is_not_nan(input_variable)

    if input_variable < base_value:
        error_string = (
            'Input value (' + str(input_variable) +
            ') should be >= base value (' + str(base_value) + ').')
        raise ValueError(error_string)


def assert_is_geq_numpy_array(input_variable, base_value, allow_nan=False):
    """Input variable must be numpy array with all elements >= some value.

    :param input_variable: Input variable.
    :param base_value: All elements must be >= this number.
    :param allow_nan: Boolean flag.  If True, array elements are allowed to be
        NaN.
    :return: ValueError: if any element is not >= base_value.
    """

    assert_is_boolean(allow_nan)
    assert_is_not_nan(base_value)
    if allow_nan:
        assert_is_real_numpy_array(input_variable)
    else:
        assert_is_numpy_array_without_nan(input_variable)

    if numpy.any(input_variable < base_value):
        error_string = (
            '\n' + str(input_variable) + '\nInput array (shown above) has ' +
            'some elements not >= base value (' + str(base_value) + ').')
        raise ValueError(error_string)


def assert_is_leq(input_variable, base_value, allow_nan=False):
    """Input variable must be real number <= some value.

    :param input_variable: Input variable.
    :param base_value: Input variable must be <= this number.
    :param allow_nan: Boolean flag.  If True, input variable is allowed to be
        NaN.
    :raises: ValueError: if input variable is not <= base_value.
    """

    assert_is_boolean(allow_nan)
    assert_is_not_nan(base_value)
    if allow_nan:
        assert_is_real_number(input_variable)
    else:
        assert_is_not_nan(input_variable)

    if input_variable > base_value:
        error_string = (
            'Input value (' + str(input_variable) +
            ') should be <= base value (' + str(base_value) + ').')
        raise ValueError(error_string)


def assert_is_leq_numpy_array(input_variable, base_value, allow_nan=False):
    """Input variable must be numpy array with all elements <= some value.

    :param input_variable: Input variable.
    :param base_value: All elements must be <= this number.
    :param allow_nan: Boolean flag.  If True, array elements are allowed to be
        NaN.
    :return: ValueError: if any element is not <= base_value.
    """

    assert_is_boolean(allow_nan)
    assert_is_not_nan(base_value)
    if allow_nan:
        assert_is_real_numpy_array(input_variable)
    else:
        assert_is_numpy_array_without_nan(input_variable)

    if numpy.any(input_variable > base_value):
        error_string = (
            '\n' + str(input_variable) + '\nInput array (shown above) has ' +
            'some elements not <= base value (' + str(base_value) + ').')
        raise ValueError(error_string)


def assert_is_valid_latitude(latitude_deg, allow_nan=False):
    """Input variable must be valid latitude (in -90...90 deg N).

    :param latitude_deg: Latitude (deg N).
    :param allow_nan: Boolean flag.  If True, input variable is allowed to be
        NaN.
    :raises: ValueError: if input variable is not valid latitude.
    """

    assert_is_boolean(allow_nan)
    if allow_nan:
        assert_is_real_number(latitude_deg)
    else:
        assert_is_not_nan(latitude_deg)

    if latitude_deg < MIN_LATITUDE_DEG or latitude_deg > MAX_LATITUDE_DEG:
        error_string = (
            'Latitude (' + str(latitude_deg) + ' deg N) is not in range [' +
            str(MIN_LATITUDE_DEG) + ', ' + str(MAX_LATITUDE_DEG) + '] deg N.')
        raise ValueError(error_string)


def assert_is_valid_longitude(longitude_deg, positive_in_west_flag=False,
                              negative_in_west_flag=False, allow_nan=False):
    """Input variable must be valid longitude.

    If positive_in_west_flag = True, longitude must be in [0, 360] deg E.
    If negative_in_west_flag = True, longitude must be in [-180, 180] deg E.
    If both flags are False, longitude must be in [-180, 360] deg E.

    :param longitude_deg: Longitude (deg E).
    :param positive_in_west_flag: Boolean flag.  If True, all longitudes in
        western hemisphere must be positive.
    :param negative_in_west_flag: Boolean flag.  If True, all longitudes in
        western hemisphere must be negative.
    :param allow_nan: Boolean flag.  If True, input variable is allowed to be
        NaN.
    :raises: ValueError: if input variable is not valid longitude.
    """

    assert_is_boolean(positive_in_west_flag)
    assert_is_boolean(negative_in_west_flag)
    assert_is_boolean(allow_nan)
    if allow_nan:
        assert_is_real_number(longitude_deg)
    else:
        assert_is_not_nan(longitude_deg)

    if positive_in_west_flag:
        this_min_longitude_deg = copy.deepcopy(MIN_LNG_POSITIVE_IN_WEST_DEG)
        this_max_longitude_deg = copy.deepcopy(MAX_LNG_POSITIVE_IN_WEST_DEG)
    elif negative_in_west_flag:
        this_min_longitude_deg = copy.deepcopy(MIN_LNG_NEGATIVE_IN_WEST_DEG)
        this_max_longitude_deg = copy.deepcopy(MAX_LNG_NEGATIVE_IN_WEST_DEG)
    else:
        this_min_longitude_deg = copy.deepcopy(MIN_LONGITUDE_DEG)
        this_max_longitude_deg = copy.deepcopy(MAX_LONGITUDE_DEG)

    if (longitude_deg < this_min_longitude_deg or
            longitude_deg > this_max_longitude_deg):
        error_string = (
            'Longitude (' + str(longitude_deg) + ' deg E) is not in range [' +
            str(this_min_longitude_deg) + ', ' + str(this_max_longitude_deg) +
            '] deg E.')
        raise ValueError(error_string)


def assert_is_valid_lat_numpy_array(latitudes_deg, allow_nan=False):
    """Input variable must be numpy array of valid lat (in -90...90 deg N).

    :param latitudes_deg: numpy array of latitudes (deg N).
    :param allow_nan: Boolean flag.  If True, array elements are allowed to be
        NaN.
    :raises: ValueError: if any element is not valid latitude.
    """

    assert_is_boolean(allow_nan)
    if allow_nan:
        assert_is_real_numpy_array(latitudes_deg)
    else:
        assert_is_numpy_array_without_nan(latitudes_deg)

    invalid_flags = numpy.logical_or(latitudes_deg < MIN_LATITUDE_DEG,
                                     latitudes_deg > MAX_LATITUDE_DEG)
    if numpy.any(invalid_flags):
        error_string = (
            '\n' + str(latitudes_deg) + '\nInput array (shown above) has ' +
            'some elements outside of [' + str(MIN_LATITUDE_DEG) + ', ' +
            str(MAX_LATITUDE_DEG) + '] deg N.')
        raise ValueError(error_string)


def assert_is_valid_lng_numpy_array(longitudes_deg, positive_in_west_flag=False,
                                    negative_in_west_flag=False,
                                    allow_nan=False):
    """Input variable must be numpy array of valid longitudes.

    If positive_in_west_flag = True, longitude must be in [0, 360] deg E.
    If negative_in_west_flag = True, longitude must be in [-180, 180] deg E.
    If both flags are False, longitude must be in [-180, 360] deg E.

    :param longitudes_deg: numpy array of longitudes (deg E).
    :param positive_in_west_flag: Boolean flag.  If True, all longitudes in
        western hemisphere must be positive.
    :param negative_in_west_flag: Boolean flag.  If True, all longitudes in
        western hemisphere must be negative.
    :param allow_nan: Boolean flag.  If True, array elements are allowed to be
        NaN.
    :raises: ValueError: if any element is not valid longitude.
    """

    assert_is_boolean(positive_in_west_flag)
    assert_is_boolean(negative_in_west_flag)
    assert_is_boolean(allow_nan)
    if allow_nan:
        assert_is_real_numpy_array(longitudes_deg)
    else:
        assert_is_numpy_array_without_nan(longitudes_deg)

    if positive_in_west_flag:
        this_min_longitude_deg = copy.deepcopy(MIN_LNG_POSITIVE_IN_WEST_DEG)
        this_max_longitude_deg = copy.deepcopy(MAX_LNG_POSITIVE_IN_WEST_DEG)
    elif negative_in_west_flag:
        this_min_longitude_deg = copy.deepcopy(MIN_LNG_NEGATIVE_IN_WEST_DEG)
        this_max_longitude_deg = copy.deepcopy(MAX_LNG_NEGATIVE_IN_WEST_DEG)
    else:
        this_min_longitude_deg = copy.deepcopy(MIN_LONGITUDE_DEG)
        this_max_longitude_deg = copy.deepcopy(MAX_LONGITUDE_DEG)

    invalid_flags = numpy.logical_or(longitudes_deg < this_min_longitude_deg,
                                     longitudes_deg > this_max_longitude_deg)
    if numpy.any(invalid_flags):
        error_string = (
            '\n' + str(longitudes_deg) + '\nInput array (shown above) has ' +
            'some elements outside of [' + str(this_min_longitude_deg) + ', ' +
            str(this_max_longitude_deg) + '] deg E.')
        raise ValueError(error_string)
