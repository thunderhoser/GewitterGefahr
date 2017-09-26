"""Methods for error-checking.

These methods are designed mainly to check for errors in input arguments.
"""

import numbers
import os.path
import numpy

BOOLEAN_TYPES = (bool, numpy.bool_)
REAL_NUMBER_TYPES = (float, numbers.Integral)

MIN_LATITUDE_DEG = -90.
MAX_LATITUDE_DEG = 90.
MIN_LNG_NEGATIVE_IN_WEST_DEG = -180.
MAX_LNG_NEGATIVE_IN_WEST_DEG = 180.
MIN_LNG_POSITIVE_IN_WEST_DEG = 0.
MAX_LNG_POSITIVE_IN_WEST_DEG = 360.
MIN_LONGITUDE_DEG = -180.
MAX_LONGITUDE_DEG = 360.


def assert_is_non_array(input_variable):
    """Raises error if input variable is tuple, list, or numpy array.

    :param input_variable: Input variable (should *not* be tuple, list, or numpy
        array).
    :raises: TypeError: if input variable is tuple, list, or numpy array.
    """

    if isinstance(input_variable, (tuple, list, numpy.ndarray)):
        error_string = ('\n' + str(input_variable) + '\n' +
                        'Input variable (displayed above) is tuple, list, or '
                        'numpy array.')
        raise TypeError(error_string)


def assert_is_array(input_variable):
    """Raises error if input variable is not tuple, list, or numpy array.

    :param input_variable: Input variable (should be tuple, list, or numpy
        array).
    :raises: TypeError: if input variable is not tuple, list, or numpy array.
    """

    if not isinstance(input_variable, (tuple, list, numpy.ndarray)):
        error_string = ('\n' + str(input_variable) + '\n' +
                        'Input variable (displayed above) is *not* tuple, list,'
                        ' or numpy array.')
        raise TypeError(error_string)


def assert_is_numpy_array(input_variable, num_dimensions=None,
                          exact_dimensions=None):
    """Raises error if input variable is not numpy array.

    May also enforce a certain number of dimensions or the dimensions
    themselves.

    N = number of dimensions

    :param input_variable: Input variable (should be numpy array).
    :param num_dimensions: If None, number of dimensions will not be checked.
        If defined, must be positive integer.
    :param exact_dimensions: If None, exact dimensions will not be checked.  If
        defined, must be length-N numpy array of positive integers.
    :raises: TypeError: `input_variable` is not numpy array; `num_dimensions` is
        not positive integer; `input_variable` does not have expected number of
        dimensions; `exact_dimensions` is not 1-D numpy array of positive
        integers with length `num_dimensions`; `input_variable` does not have
        expected exact dimensions.
    """

    if not isinstance(input_variable, numpy.ndarray):
        error_string = ('\n' + str(input_variable) + '\n' +
                        'Input variable (displayed above) is not numpy array.')
        raise TypeError(error_string)

    if num_dimensions is not None:
        assert_is_integer(num_dimensions)
        assert_is_positive(num_dimensions)

        if input_variable.ndim != num_dimensions:
            error_string = (
                'Input variable should be ' + str(num_dimensions) +
                '-D numpy array.  Got ' + str(input_variable.ndim) +
                '-D array.')
            raise TypeError(error_string)

    if exact_dimensions is not None:
        assert_is_integer_array(exact_dimensions)
        assert_is_positive_array(exact_dimensions)
        assert_is_numpy_array(exact_dimensions, num_dimensions=1)

        if (num_dimensions is not None and len(
                exact_dimensions) != num_dimensions):
            error_string = (
                'exact_dimensions should be length-' + str(num_dimensions) +
                ' numpy array.  Got length-' + str(len(exact_dimensions)) +
                ' array.')
            raise TypeError(error_string)

        if not numpy.array_equal(input_variable.shape, exact_dimensions):
            error_string = (
                '\nExpected dimensions: ' + str(exact_dimensions) +
                '\nActual dimensions: ' + str(input_variable.shape) +
                '\nAs shown above, input variable has unexpected dimensions.')
            raise TypeError(error_string)


def assert_is_string(input_variable):
    """Raises error if input variable is not string.

    :param input_variable: Input variable (should be string).
    :raises: TypeError: if input variable is not string.
    """

    if not isinstance(input_variable, str):
        error_string = ('\n' + str(input_variable) + '\n' +
                        'Input variable (displayed above) is not string.')
        raise TypeError(error_string)


def assert_is_string_array(input_variable):
    """Raises error if input variable is not string array.

    :param input_variable: Input variable (should be string array).
    """

    assert_is_array(input_variable)
    for this_element in input_variable:
        assert_is_string(this_element)


def assert_file_exists(file_name):
    """Raises error if file does not exist.

    :param file_name: Path to local file.
    :raises: ValueError: if file does not exist.
    """

    assert_is_string(file_name)
    if not os.path.isfile(file_name):
        raise ValueError('File ("' + file_name + '") does not exist.')


def assert_directory_exists(directory_name):
    """Raises error if directory does not exist.

    :param directory_name: Path to local directory.
    :raises: ValueError: if directory does not exist.
    """

    assert_is_string(directory_name)
    if not os.path.isdir(directory_name):
        raise ValueError('Directory ("' + directory_name + '") does not exist.')


def assert_is_integer(input_variable):
    """Raises error if input variable is not integer.

    :param input_variable: Input variable (should be integer).
    :raises: TypeError: if input variable is not integer.
    """

    if (isinstance(input_variable, BOOLEAN_TYPES) or not isinstance(
            input_variable, numbers.Integral)):
        error_string = ('\n' + str(input_variable) + '\n' +
                        'Input variable (displayed above) is not integer.')
        raise TypeError(error_string)


def assert_is_integer_array(input_variable):
    """Raises error if input variable is not integer array.

    :param input_variable: Input variable (should be integer array).
    """

    assert_is_array(input_variable)
    for this_element in input_variable:
        assert_is_integer(this_element)


def assert_is_boolean(input_variable):
    """Raises error if input variable is not Boolean.

    :param input_variable: Input variable (should be Boolean).
    :raises: TypeError: if input variable is not Boolean.
    """

    if not isinstance(input_variable, BOOLEAN_TYPES):
        error_string = ('\n' + str(input_variable) + '\n' +
                        'Input variable (displayed above) is not Boolean.')
        raise TypeError(error_string)


def assert_is_boolean_array(input_variable):
    """Raises error if input variable is not Boolean array.

    :param input_variable: Input variable (should be Boolean array).
    """

    assert_is_array(input_variable)
    for _, x in numpy.ndenumerate(numpy.asarray(input_variable)):
        assert_is_boolean(x)


def assert_is_float(input_variable):
    """Raises error if input variable is not float.

    :param input_variable: Input variable (should be float).
    :raises: TypeError: if input variable is not float.
    """

    if not isinstance(input_variable, float):
        error_string = ('\n' + str(input_variable) + '\n' +
                        'Input variable (displayed above) is not float.')
        raise TypeError(error_string)


def assert_is_float_array(input_variable):
    """Raises error if input variable is not array of floats.

    :param input_variable: Input variable (should be array of floats).
    """

    assert_is_array(input_variable)
    for this_element in input_variable:
        assert_is_float(this_element)


def assert_is_real_number(input_variable):
    """Raises error if input variable is not real number.

    :param input_variable: Input variable (should be real number).
    :raises: TypeError: if input variable is not real number.
    """

    if (isinstance(input_variable, BOOLEAN_TYPES) or not isinstance(
            input_variable, REAL_NUMBER_TYPES)):
        error_string = ('\n' + str(input_variable) + '\n' +
                        'Input variable (displayed above) is not real number.')
        raise TypeError(error_string)


def assert_is_real_number_array(input_variable):
    """Raises error if input variable is not array of real numbers.

    :param input_variable: Input variable (should be array of real numbers).
    """

    assert_is_array(input_variable)
    for this_element in input_variable:
        assert_is_real_number(this_element)


def assert_is_not_nan(input_variable):
    """Raises error if input variable is NaN.

    :param input_variable: Input variable (should not be NaN).
    :raises: ValueError: if input variable is NaN.
    """

    assert_is_real_number(input_variable)
    if numpy.isnan(input_variable):
        raise ValueError('Input variable should not be NaN.')


def assert_is_not_nan_array(input_variable):
    """Input variable must be array of numbers that are not NaN.

    :param input_variable: Input variable (should be array of numbers that are
        not NaN).
    """

    assert_is_array(input_variable)
    for this_element in input_variable:
        assert_is_not_nan(this_element)


def assert_is_positive(input_variable):
    """Raises error if input variable is not positive.

    :param input_variable: Input variable (should be positive).
    :raises: ValueError: if input variable is not positive.
    """

    assert_is_real_number(input_variable)
    if not input_variable > 0:
        raise ValueError(
            'Input variable (' + str(input_variable) + ') is not positive.')


def assert_is_positive_array(input_variable):
    """Raises error if input variable is not array of positive numbers.

    :param input_variable: Input variable (should be array of positive numbers).
    """

    assert_is_real_number_array(input_variable)
    for this_element in input_variable:
        assert_is_positive(this_element)


def assert_is_non_negative(input_variable):
    """Raises error if input variable is negative.

    :param input_variable: Input variable (should be non-negative).
    :raises: ValueError: if input variable is negative.
    """

    assert_is_real_number(input_variable)
    if not input_variable >= 0:
        raise ValueError(
            'Input variable (' + str(input_variable) + ') is negative.')


def assert_is_non_negative_array(input_variable):
    """Raises error if input variable is not array of non-negative numbers.

    :param input_variable: Input variable (should be array of non-negative
        numbers).
    """

    assert_is_real_number_array(input_variable)
    for this_element in input_variable:
        assert_is_non_negative(this_element)


def assert_is_negative(input_variable):
    """Raises error if input variable is not negative.

    :param input_variable: Input variable (should be negative).
    :raises: ValueError: if input variable is not negative.
    """

    assert_is_real_number(input_variable)
    if not input_variable < 0:
        raise ValueError(
            'Input variable (' + str(input_variable) + ') is not negative.')


def assert_is_negative_array(input_variable):
    """Raises error if input variable is not array of negative numbers.

    :param input_variable: Input variable (should be array of negative numbers).
    """

    assert_is_real_number_array(input_variable)
    for this_element in input_variable:
        assert_is_negative(this_element)


def assert_is_non_positive(input_variable):
    """Raises error if input variable is positive.

    :param input_variable: Input variable (should be non-positive).
    :raises: ValueError: if input variable is positive.
    """

    assert_is_real_number(input_variable)
    if not input_variable <= 0:
        raise ValueError(
            'Input variable (' + str(input_variable) + ') is positive.')


def assert_is_non_positive_array(input_variable):
    """Raises error if input variable is not array of non-positive numbers.

    :param input_variable: Input variable (should be array of non-positive
        numbers).
    """

    assert_is_real_number_array(input_variable)
    for this_element in input_variable:
        assert_is_non_positive(this_element)


def assert_is_valid_latitude(latitude_deg):
    """Raises error if input variable is not valid latitude.

    :param latitude_deg: Input variable (should be latitude in deg N).
    :raises: ValueError: if input variable is not valid latitude.
    """

    assert_is_real_number(latitude_deg)
    if not MIN_LATITUDE_DEG <= latitude_deg <= MAX_LATITUDE_DEG:
        raise ValueError(
            'Latitude should be in range [' + str(MIN_LATITUDE_DEG) + ', ' +
            str(MAX_LATITUDE_DEG) + '] deg N.  Instead, got ' +
            str(latitude_deg) + ' deg N.')


def assert_is_valid_longitude(longitude_deg):
    """Raises error if input variable is not valid longitude.

    :param longitude_deg: Input variable (should be longitude in deg E).
    :raises: ValueError: if input variable is not valid longitude.
    """

    assert_is_real_number(longitude_deg)

    if not MIN_LONGITUDE_DEG <= longitude_deg <= MAX_LONGITUDE_DEG:
        raise ValueError(
            'Longitude should be in range [' + str(MIN_LONGITUDE_DEG) + ', ' +
            str(MAX_LONGITUDE_DEG) + '] deg E.  Instead, got ' +
            str(longitude_deg) + ' deg E.')


def assert_is_valid_lng_positive_in_west(longitude_deg):
    """Raises error if input variable is not valid longitude.

    :param longitude_deg: Input variable (should be longitude in deg E, with
        values > 180 deg E in western hemisphere).
    :raises: ValueError: if input variable is not valid longitude.
    """

    assert_is_real_number(longitude_deg)

    if not (MIN_LNG_POSITIVE_IN_WEST_DEG <= longitude_deg <=
            MAX_LNG_POSITIVE_IN_WEST_DEG):
        raise ValueError(
            'Longitude should be in range [' + str(MIN_LNG_POSITIVE_IN_WEST_DEG)
            + ', ' + str(MAX_LNG_POSITIVE_IN_WEST_DEG) +
            '] deg E.  Instead, got ' + str(longitude_deg) + ' deg E.')


def assert_is_valid_lng_negative_in_west(longitude_deg):
    """Raises error if input variable is not valid longitude.

    :param longitude_deg: Input variable (should be longitude in deg E, with
        values < 0 deg E in western hemisphere).
    :raises: ValueError: if input variable is not valid longitude.
    """

    assert_is_real_number(longitude_deg)

    if not (MIN_LNG_NEGATIVE_IN_WEST_DEG <= longitude_deg <=
            MAX_LNG_NEGATIVE_IN_WEST_DEG):
        raise ValueError(
            'Longitude should be in range [' + str(MIN_LNG_NEGATIVE_IN_WEST_DEG)
            + ', ' + str(MAX_LNG_NEGATIVE_IN_WEST_DEG) +
            '] deg E.  Instead, got ' + str(longitude_deg) + ' deg E.')
