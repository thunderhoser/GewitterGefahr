"""Unit tests for error_checking.py."""

import unittest
import numpy
import os.path
from gewittergefahr.gg_utils import error_checking

GRID_POINT_FILE_NAME = 'grid_point_latlng_grid130.data'
GRID_POINT_DIR_NAME = os.path.dirname(os.path.realpath(GRID_POINT_FILE_NAME))
FAKE_FILE_NAME = GRID_POINT_FILE_NAME + '-_=+'
FAKE_DIR_NAME = GRID_POINT_DIR_NAME + '-_=+'

SINGLE_INTEGER = 9
INTEGER_LIST = [9]
INTEGER_TUPLE = (9,)
INTEGER_NUMPY_ARRAY = numpy.array([9])

SINGLE_STRING = 'foo'
STRING_LIST = ['foo']
SINGLE_BOOLEAN = True
BOOLEAN_NUMPY_ARRAY = numpy.array([1], dtype=bool)
SINGLE_FLOAT = 9.
FLOAT_NUMPY_ARRAY = numpy.array([9.])

COMPLEX_NUMBER = complex(1., 1.)

SINGLE_ZERO = 0.
SINGLE_NEGATIVE = -9.
SINGLE_POSITIVE = 9.

POSITIVE_NUMPY_ARRAY = numpy.array([8., 4., 2.5, 17.])
NON_NEGATIVE_NUMPY_ARRAY = numpy.array([8., 0., 4., 2.5, 17.])
NEGATIVE_NUMPY_ARRAY = numpy.array([-8., -4., -2.5, -17.])
NON_POSITIVE_NUMPY_ARRAY = numpy.array([-8., 0., -4., -2.5, -17.])
MIXED_SIGN_NUMPY_ARRAY = numpy.array([8., 0., -4., 2.5, -17.])

NAN_NUMPY_ARRAY = numpy.array([numpy.nan, numpy.nan, numpy.nan, numpy.nan])
MIXED_NAN_NUMPY_ARRAY = numpy.array([numpy.nan, 0., numpy.nan, 3.])
NUMPY_ARRAY_WITHOUT_NANS = numpy.array([-8., 0., 16., 3.])


class ErrorCheckingTests(unittest.TestCase):
    """Each method is a unit test for error_checking.py."""

    def test_assert_is_non_array_tuple(self):
        """Checks assert_is_non_array when input is tuple."""

        with self.assertRaises(TypeError):
            error_checking.assert_is_non_array(INTEGER_TUPLE)

    def test_assert_is_non_array_list(self):
        """Checks assert_is_non_array when input is list."""

        with self.assertRaises(TypeError):
            error_checking.assert_is_non_array(INTEGER_LIST)

    def test_assert_is_non_array_numpy_array(self):
        """Checks assert_is_non_array when input is numpy array."""

        with self.assertRaises(TypeError):
            error_checking.assert_is_non_array(INTEGER_NUMPY_ARRAY)

    def test_assert_is_non_array_true(self):
        """Checks assert_is_non_array when input is non-array."""

        error_checking.assert_is_non_array(SINGLE_INTEGER)

    def test_assert_is_array_false(self):
        """Checks assert_is_array when input is non-array."""

        with self.assertRaises(TypeError):
            error_checking.assert_is_array(SINGLE_INTEGER)

    def test_assert_is_array_tuple(self):
        """Checks assert_is_array when input is tuple."""

        error_checking.assert_is_array(INTEGER_TUPLE)

    def test_assert_is_array_list(self):
        """Checks assert_is_array when input is list."""

        error_checking.assert_is_array(INTEGER_LIST)

    def test_assert_is_array_numpy_array(self):
        """Checks assert_is_array when input is numpy array."""

        error_checking.assert_is_array(INTEGER_NUMPY_ARRAY)

    def test_assert_is_numpy_array_non_array(self):
        """Checks assert_is_numpy_array when input is non-array."""

        with self.assertRaises(TypeError):
            error_checking.assert_is_numpy_array(SINGLE_INTEGER)

    def test_assert_is_numpy_array_tuple(self):
        """Checks assert_is_numpy_array when input is tuple."""

        with self.assertRaises(TypeError):
            error_checking.assert_is_numpy_array(INTEGER_TUPLE)

    def test_assert_is_numpy_array_list(self):
        """Checks assert_is_numpy_array when input is list."""

        with self.assertRaises(TypeError):
            error_checking.assert_is_numpy_array(INTEGER_LIST)

    def test_assert_is_numpy_array_true(self):
        """Checks assert_is_numpy_array when input is numpy array."""

        error_checking.assert_is_numpy_array(INTEGER_NUMPY_ARRAY)

    def test_assert_is_numpy_array_num_dim_not_integer(self):
        """Checks assert_is_numpy_array when `num_dimensions` is not integer."""

        with self.assertRaises(TypeError):
            error_checking.assert_is_numpy_array(
                INTEGER_NUMPY_ARRAY,
                num_dimensions=float(INTEGER_NUMPY_ARRAY.ndim))

    def test_assert_is_numpy_array_num_dim_not_positive(self):
        """Checks assert_is_numpy_array when `num_dimensions` is <= 0."""

        with self.assertRaises(ValueError):
            error_checking.assert_is_numpy_array(INTEGER_NUMPY_ARRAY,
                                                 num_dimensions=0)

    def test_assert_is_numpy_array_num_dim_unexpected(self):
        """Checks assert_is_numpy_array with unexpected number of dimensions."""

        with self.assertRaises(TypeError):
            error_checking.assert_is_numpy_array(
                INTEGER_NUMPY_ARRAY,
                num_dimensions=INTEGER_NUMPY_ARRAY.ndim + 1)

    def test_assert_is_numpy_array_num_dim_correct(self):
        """Checks assert_is_numpy_array with expected number of dimensions."""

        error_checking.assert_is_numpy_array(
            INTEGER_NUMPY_ARRAY, num_dimensions=INTEGER_NUMPY_ARRAY.ndim)

    def test_assert_is_numpy_array_exact_dim_not_array(self):
        """Checks assert_is_numpy_array when `exact_dimensions` is not array."""

        with self.assertRaises(TypeError):
            error_checking.assert_is_numpy_array(
                INTEGER_NUMPY_ARRAY, num_dimensions=INTEGER_NUMPY_ARRAY.ndim,
                exact_dimensions=INTEGER_NUMPY_ARRAY.shape[0])

    def test_assert_is_numpy_array_exact_dim_not_integers(self):
        """Checks assert_is_numpy_array; `exact_dimensions` has non-integers."""

        with self.assertRaises(TypeError):
            error_checking.assert_is_numpy_array(
                INTEGER_NUMPY_ARRAY, num_dimensions=INTEGER_NUMPY_ARRAY.ndim,
                exact_dimensions=numpy.asarray(INTEGER_NUMPY_ARRAY.shape,
                                               dtype=numpy.float64))

    def test_assert_is_numpy_array_exact_dim_not_positive(self):
        """Checks assert_is_numpy_array; `exact_dimensions` has non-positive."""

        with self.assertRaises(ValueError):
            error_checking.assert_is_numpy_array(
                INTEGER_NUMPY_ARRAY, num_dimensions=INTEGER_NUMPY_ARRAY.ndim,
                exact_dimensions=numpy.array([0]))

    def test_assert_is_numpy_array_exact_dim_too_long(self):
        """Checks assert_is_numpy_array; `exact_dimensions` is too long."""

        these_exact_dimensions = numpy.concatenate((
            numpy.asarray(INTEGER_NUMPY_ARRAY.shape, dtype=numpy.int64),
            numpy.array([1])))

        with self.assertRaises(TypeError):
            error_checking.assert_is_numpy_array(
                INTEGER_NUMPY_ARRAY, num_dimensions=INTEGER_NUMPY_ARRAY.ndim,
                exact_dimensions=these_exact_dimensions)

    def test_assert_is_numpy_array_exact_dim_not_numpy(self):
        """Checks assert_is_numpy_array; `exact_dimensions` is not numpy."""

        with self.assertRaises(TypeError):
            error_checking.assert_is_numpy_array(
                INTEGER_NUMPY_ARRAY, num_dimensions=INTEGER_NUMPY_ARRAY.ndim,
                exact_dimensions=INTEGER_NUMPY_ARRAY.shape)

    def test_assert_is_numpy_array_exact_dim_unexpected(self):
        """Checks assert_is_numpy_array with unexpected exact dimensions."""

        these_exact_dimensions = (
            numpy.asarray(INTEGER_NUMPY_ARRAY.shape, dtype=numpy.int64) + 1)

        with self.assertRaises(TypeError):
            error_checking.assert_is_numpy_array(
                INTEGER_NUMPY_ARRAY, num_dimensions=INTEGER_NUMPY_ARRAY.ndim,
                exact_dimensions=these_exact_dimensions)

    def test_assert_is_numpy_array_exact_dim_correct(self):
        """Checks assert_is_numpy_array with expected exact dimensions."""

        error_checking.assert_is_numpy_array(
            INTEGER_NUMPY_ARRAY, num_dimensions=INTEGER_NUMPY_ARRAY.ndim,
            exact_dimensions=numpy.asarray(INTEGER_NUMPY_ARRAY.shape,
                                           dtype=numpy.int64))

    def test_assert_is_string_number(self):
        """Checks assert_is_string when input is number."""

        with self.assertRaises(TypeError):
            error_checking.assert_is_string(SINGLE_INTEGER)

    def test_assert_is_string_nan(self):
        """Checks assert_is_string when input is NaN."""

        with self.assertRaises(TypeError):
            error_checking.assert_is_string(numpy.nan)

    def test_assert_is_string_none(self):
        """Checks assert_is_string when input is None."""

        with self.assertRaises(TypeError):
            error_checking.assert_is_string(None)

    def test_assert_is_string_true(self):
        """Checks assert_is_string when input is string."""

        error_checking.assert_is_string(SINGLE_STRING)

    def test_assert_is_string_array_true(self):
        """Checks assert_is_string_array when input is list of strings."""

        error_checking.assert_is_string_array(STRING_LIST)

    def test_assert_file_exists_directory(self):
        """Checks assert_file_exists when input is directory."""

        with self.assertRaises(ValueError):
            error_checking.assert_file_exists(GRID_POINT_DIR_NAME)

    def test_assert_file_exists_fake(self):
        """Checks assert_file_exists when input is fake file."""

        with self.assertRaises(ValueError):
            error_checking.assert_file_exists(FAKE_FILE_NAME)

    def test_assert_file_exists_true(self):
        """Checks assert_file_exists when input is existent file."""

        error_checking.assert_file_exists(GRID_POINT_FILE_NAME)

    def test_assert_directory_exists_file(self):
        """Checks assert_directory_exists when input is file."""

        with self.assertRaises(ValueError):
            error_checking.assert_directory_exists(GRID_POINT_FILE_NAME)

    def test_assert_directory_exists_fake(self):
        """Checks assert_directory_exists when input is fake directory."""

        with self.assertRaises(ValueError):
            error_checking.assert_directory_exists(FAKE_DIR_NAME)

    def test_assert_directory_exists_true(self):
        """Checks assert_directory_exists when input is existent directory."""

        error_checking.assert_directory_exists(GRID_POINT_DIR_NAME)

    def test_assert_is_integer_too_many_inputs(self):
        """Checks assert_is_integer when input is array of integers."""

        with self.assertRaises(TypeError):
            error_checking.assert_is_integer(INTEGER_NUMPY_ARRAY)

    def test_assert_is_integer_float(self):
        """Checks assert_is_integer when input is float."""

        with self.assertRaises(TypeError):
            error_checking.assert_is_integer(SINGLE_FLOAT)

    def test_assert_is_integer_boolean(self):
        """Checks assert_is_integer when input is Boolean."""

        with self.assertRaises(TypeError):
            error_checking.assert_is_integer(SINGLE_BOOLEAN)

    def test_assert_is_integer_nan(self):
        """Checks assert_is_integer when input is NaN."""

        with self.assertRaises(TypeError):
            error_checking.assert_is_integer(numpy.nan)

    def test_assert_is_integer_none(self):
        """Checks assert_is_integer when input is None."""

        with self.assertRaises(TypeError):
            error_checking.assert_is_integer(None)

    def test_assert_is_integer_true(self):
        """Checks assert_is_integer when input is integer."""

        error_checking.assert_is_integer(SINGLE_INTEGER)

    def test_assert_is_integer_array_true(self):
        """Checks assert_is_integer_array when input is integer array."""

        error_checking.assert_is_integer_array(INTEGER_NUMPY_ARRAY)

    def test_assert_is_boolean_too_many_inputs(self):
        """Checks assert_is_boolean when input is array of Booleans."""

        with self.assertRaises(TypeError):
            error_checking.assert_is_boolean(BOOLEAN_NUMPY_ARRAY)

    def test_assert_is_boolean_integer(self):
        """Checks assert_is_boolean when input is integer."""

        with self.assertRaises(TypeError):
            error_checking.assert_is_boolean(SINGLE_INTEGER)

    def test_assert_is_boolean_float(self):
        """Checks assert_is_boolean when input is float."""

        with self.assertRaises(TypeError):
            error_checking.assert_is_boolean(SINGLE_FLOAT)

    def test_assert_is_boolean_nan(self):
        """Checks assert_is_boolean when input is NaN."""

        with self.assertRaises(TypeError):
            error_checking.assert_is_boolean(numpy.nan)

    def test_assert_is_boolean_none(self):
        """Checks assert_is_boolean when input is None."""

        with self.assertRaises(TypeError):
            error_checking.assert_is_boolean(None)

    def test_assert_is_boolean_true(self):
        """Checks assert_is_boolean when input is Boolean."""

        error_checking.assert_is_boolean(SINGLE_BOOLEAN)

    def test_assert_is_boolean_array_true(self):
        """Checks assert_is_boolean_array when input is Boolean array."""

        error_checking.assert_is_boolean_array(BOOLEAN_NUMPY_ARRAY)

    def test_assert_is_float_too_many_inputs(self):
        """Checks assert_is_float when input is array of floats."""

        with self.assertRaises(TypeError):
            error_checking.assert_is_float(FLOAT_NUMPY_ARRAY)

    def test_assert_is_float_boolean(self):
        """Checks assert_is_float when input is Boolean."""

        with self.assertRaises(TypeError):
            error_checking.assert_is_float(SINGLE_BOOLEAN)

    def test_assert_is_float_integer(self):
        """Checks assert_is_float when input is integer."""

        with self.assertRaises(TypeError):
            error_checking.assert_is_float(SINGLE_INTEGER)

    def test_assert_is_float_none(self):
        """Checks assert_is_float when input is None."""

        with self.assertRaises(TypeError):
            error_checking.assert_is_float(None)

    def test_assert_is_float_true(self):
        """Checks assert_is_float when input is float."""

        error_checking.assert_is_float(SINGLE_FLOAT)

    def test_assert_is_float_array_true(self):
        """Checks assert_is_float_array when input is float array."""

        error_checking.assert_is_float_array(FLOAT_NUMPY_ARRAY)

    def test_assert_is_real_number_too_many_inputs(self):
        """Checks assert_is_real_number when input is array of real numbers."""

        with self.assertRaises(TypeError):
            error_checking.assert_is_real_number(FLOAT_NUMPY_ARRAY)

    def test_assert_is_real_number_boolean(self):
        """Checks assert_is_real_number when input is Boolean."""

        with self.assertRaises(TypeError):
            error_checking.assert_is_real_number(SINGLE_BOOLEAN)

    def test_assert_is_real_number_none(self):
        """Checks assert_is_real_number when input is None."""

        with self.assertRaises(TypeError):
            error_checking.assert_is_real_number(None)

    def test_assert_is_real_number_complex(self):
        """Checks assert_is_real_number when input is complex number."""

        with self.assertRaises(TypeError):
            error_checking.assert_is_real_number(COMPLEX_NUMBER)

    def test_assert_is_real_number_integer(self):
        """Checks assert_is_real_number when input is integer."""

        error_checking.assert_is_real_number(SINGLE_INTEGER)

    def test_assert_is_real_number_float(self):
        """Checks assert_is_real_number when input is float."""

        error_checking.assert_is_real_number(SINGLE_FLOAT)

    def test_assert_is_real_number_array_true(self):
        """Checks assert_is_real_number_array; input is real-number array."""

        error_checking.assert_is_real_number_array(FLOAT_NUMPY_ARRAY)

    def test_assert_is_not_nan_false(self):
        """Checks assert_is_not_nan when input is NaN."""

        with self.assertRaises(ValueError):
            error_checking.assert_is_not_nan(numpy.nan)

    def test_assert_is_not_nan_true(self):
        """Checks assert_is_not_nan when input is not NaN."""

        error_checking.assert_is_not_nan(SINGLE_FLOAT)

    def test_assert_is_not_nan_array_all_nan(self):
        """Checks assert_is_not_nan_array when input is array of NaN's."""

        with self.assertRaises(ValueError):
            error_checking.assert_is_not_nan_array(NAN_NUMPY_ARRAY)

    def test_assert_is_not_nan_array_mixed(self):
        """Checks assert_is_not_nan_array when input contains NaN's."""

        with self.assertRaises(ValueError):
            error_checking.assert_is_not_nan_array(MIXED_NAN_NUMPY_ARRAY)

    def test_assert_is_not_nan_array_true(self):
        """Checks assert_is_not_nan_array when input is array without NaN."""

        error_checking.assert_is_not_nan_array(NUMPY_ARRAY_WITHOUT_NANS)

    def test_assert_is_positive_negative(self):
        """Checks assert_is_positive when input is negative."""

        with self.assertRaises(ValueError):
            error_checking.assert_is_positive(SINGLE_NEGATIVE)

    def test_assert_is_positive_zero(self):
        """Checks assert_is_positive when input is zero."""

        with self.assertRaises(ValueError):
            error_checking.assert_is_positive(SINGLE_ZERO)

    def test_assert_is_positive_true(self):
        """Checks assert_is_positive when input is positive."""

        error_checking.assert_is_positive(SINGLE_POSITIVE)

    def test_assert_is_positive_array_negative(self):
        """Checks assert_is_positive_array when input is array of negatives."""

        with self.assertRaises(ValueError):
            error_checking.assert_is_positive_array(NEGATIVE_NUMPY_ARRAY)

    def test_assert_is_positive_array_mixed_sign(self):
        """Checks assert_is_positive_array when input is array of mixed sign."""

        with self.assertRaises(ValueError):
            error_checking.assert_is_positive_array(MIXED_SIGN_NUMPY_ARRAY)

    def test_assert_is_positive_array_true(self):
        """Checks assert_is_positive_array when input is array of positives."""

        error_checking.assert_is_positive_array(POSITIVE_NUMPY_ARRAY)

    def test_assert_is_non_negative_false(self):
        """Checks assert_is_non_negative when input is negative."""

        with self.assertRaises(ValueError):
            error_checking.assert_is_non_negative(SINGLE_NEGATIVE)

    def test_assert_is_non_negative_zero(self):
        """Checks assert_is_non_negative when input is zero."""

        error_checking.assert_is_non_negative(SINGLE_ZERO)

    def test_assert_is_non_negative_positive(self):
        """Checks assert_is_non_negative when input is positive."""

        error_checking.assert_is_non_negative(SINGLE_POSITIVE)

    def test_assert_is_non_negative_array_negative(self):
        """Checks assert_is_non_negative_array; input is array of negatives."""

        with self.assertRaises(ValueError):
            error_checking.assert_is_non_negative_array(NEGATIVE_NUMPY_ARRAY)

    def test_assert_is_non_negative_array_mixed_sign(self):
        """Checks assert_is_non_negative_array; input is array of mixed sign."""

        with self.assertRaises(ValueError):
            error_checking.assert_is_non_negative_array(
                MIXED_SIGN_NUMPY_ARRAY)

    def test_assert_is_non_negative_array_true(self):
        """Checks assert_is_non_negative_array; input is array of positives."""

        error_checking.assert_is_non_negative_array(NON_NEGATIVE_NUMPY_ARRAY)

    def test_assert_is_negative_positive(self):
        """Checks assert_is_negative when input is positive."""

        with self.assertRaises(ValueError):
            error_checking.assert_is_negative(SINGLE_POSITIVE)

    def test_assert_is_negative_zero(self):
        """Checks assert_is_negative when input is zero."""

        with self.assertRaises(ValueError):
            error_checking.assert_is_negative(SINGLE_ZERO)

    def test_assert_is_negative_true(self):
        """Checks assert_is_negative when input is negative."""

        error_checking.assert_is_negative(SINGLE_NEGATIVE)

    def test_assert_is_negative_array_positive(self):
        """Checks assert_is_negative_array when input is array of positives."""

        with self.assertRaises(ValueError):
            error_checking.assert_is_negative_array(POSITIVE_NUMPY_ARRAY)

    def test_assert_is_negative_array_mixed_sign(self):
        """Checks assert_is_negative_array when input is array of mixed sign."""

        with self.assertRaises(ValueError):
            error_checking.assert_is_negative_array(MIXED_SIGN_NUMPY_ARRAY)

    def test_assert_is_negative_array_true(self):
        """Checks assert_is_negative_array when input is array of negatives."""

        error_checking.assert_is_negative_array(NEGATIVE_NUMPY_ARRAY)

    def test_assert_is_non_positive_false(self):
        """Checks assert_is_non_positive when input is positive."""

        with self.assertRaises(ValueError):
            error_checking.assert_is_non_positive(SINGLE_POSITIVE)

    def test_assert_is_non_positive_zero(self):
        """Checks assert_is_non_positive when input is zero."""

        error_checking.assert_is_non_positive(SINGLE_ZERO)

    def test_assert_is_non_positive_negative(self):
        """Checks assert_is_non_positive when input is negative."""

        error_checking.assert_is_non_positive(SINGLE_NEGATIVE)

    def test_assert_is_non_positive_array_negative(self):
        """Checks assert_is_non_positive_array; input is array of positive."""

        with self.assertRaises(ValueError):
            error_checking.assert_is_non_positive_array(POSITIVE_NUMPY_ARRAY)

    def test_assert_is_non_positive_array_mixed_sign(self):
        """Checks assert_is_non_positive_array; input is array of mixed sign."""

        with self.assertRaises(ValueError):
            error_checking.assert_is_non_positive_array(
                MIXED_SIGN_NUMPY_ARRAY)

    def test_assert_is_non_positive_array_true(self):
        """Checks assert_is_non_positive_array; input is all non-positives."""

        error_checking.assert_is_non_positive_array(NON_POSITIVE_NUMPY_ARRAY)


if __name__ == '__main__':
    unittest.main()
