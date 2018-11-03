"""Unit tests for error_checking.py."""

import unittest
import os.path
import numpy
import pandas
from gewittergefahr.gg_utils import error_checking

COLUMNS_IN_DATAFRAME = ['foo', 'bar']
FAKE_COLUMNS_IN_DATAFRAME = ['foo', 'bar', 'moo']
DATAFRAME = pandas.DataFrame.from_dict(
    {'foo': numpy.array([]), 'bar': numpy.array([])})

THIS_FILE_NAME = __file__
THIS_DIRECTORY_NAME = os.path.split(THIS_FILE_NAME)[0]
FAKE_FILE_NAME = THIS_FILE_NAME + '-_=+'
FAKE_DIRECTORY_NAME = THIS_DIRECTORY_NAME + '-_=+'

SINGLE_INTEGER = 1959
SINGLE_FLOAT = 1959.
SINGLE_BOOLEAN = True
SINGLE_COMPLEX_NUMBER = complex(1., 1.)
SINGLE_STRING = '1959'
STRING_LIST = [['do', 're'],
               [['mi', 'fa']],
               [[['so'], 'la']],
               [['ti', 'do'], [[[' ']]]],
               '']

REAL_NUMBER_LIST = [[211., 215],
                    [[214, 199.]],
                    [[[226.], 205.]],
                    [[221, 211], [[[32]]]],
                    0.]
REAL_NUMBER_TUPLE = ((211., 215),
                     ((214, 199.),),
                     (((226.,), 205.),),
                     ((221, 211), (((32,),),)),
                     0.)

REAL_NUMPY_ARRAY = numpy.array([[211., 215],
                                [214, 199.],
                                [226., 205.],
                                [221, 211],
                                [32, 0.]])
BOOLEAN_NUMPY_ARRAY = numpy.array([[False, True],
                                   [True, False],
                                   [True, False],
                                   [True, False],
                                   [False, False]])
FLOAT_NUMPY_ARRAY = numpy.array([[211., 215.],
                                 [214., 199.],
                                 [226., 205.],
                                 [221., 211.],
                                 [32., 0.]])
INTEGER_NUMPY_ARRAY = numpy.array([[211, 215],
                                   [214, 199],
                                   [226, 205],
                                   [221, 211],
                                   [32, 0]])
NAN_NUMPY_ARRAY = numpy.array([[numpy.nan, numpy.nan],
                               [numpy.nan, numpy.nan],
                               [numpy.nan, numpy.nan],
                               [numpy.nan, numpy.nan],
                               [numpy.nan, numpy.nan]])

SINGLE_ZERO = 0.
SINGLE_NEGATIVE = -2.2
SINGLE_POSITIVE = 4.3

POSITIVE_NUMPY_ARRAY = numpy.array([[211., 215],
                                    [214, 199.],
                                    [226., 205.],
                                    [221, 211],
                                    [32, 1.]])
NON_NEGATIVE_NUMPY_ARRAY = numpy.array([[211., 215],
                                        [214, 199.],
                                        [226., 205.],
                                        [221, 211],
                                        [32, 0.]])
NEGATIVE_NUMPY_ARRAY = numpy.array([[-211., -215],
                                    [-214, -199.],
                                    [-226., -205.],
                                    [-221, -211],
                                    [-32, -1.]])
NON_POSITIVE_NUMPY_ARRAY = numpy.array([[-211., -215],
                                        [-214, -199.],
                                        [-226., -205.],
                                        [-221, -211],
                                        [-32, 0.]])
MIXED_SIGN_NUMPY_ARRAY = numpy.array([[-211., 215],
                                      [-214, -199.],
                                      [-226., 205.],
                                      [221, 211],
                                      [-32, 0.]])

POSITIVE_NUMPY_ARRAY_WITH_NANS = numpy.array([[numpy.nan, 215],
                                              [214, 199.],
                                              [226., numpy.nan],
                                              [221, 211],
                                              [32, 1.]])
NON_NEGATIVE_NUMPY_ARRAY_WITH_NANS = numpy.array([[numpy.nan, 215],
                                                  [214, 199.],
                                                  [226., numpy.nan],
                                                  [221, 211],
                                                  [32, 0.]])
NEGATIVE_NUMPY_ARRAY_WITH_NANS = numpy.array([[numpy.nan, -215],
                                              [-214, -199.],
                                              [-226., numpy.nan],
                                              [-221, -211],
                                              [-32, -1.]])
NON_POSITIVE_NUMPY_ARRAY_WITH_NANS = numpy.array([[numpy.nan, -215],
                                                  [-214, -199.],
                                                  [-226., numpy.nan],
                                                  [-221, -211],
                                                  [-32, 0.]])

SINGLE_LATITUDE_DEG = 45.
SINGLE_LAT_INVALID_DEG = -500.
LAT_NUMPY_ARRAY_DEG = numpy.array([[42., -35.],
                                   [35., -61.],
                                   [33., 30.],
                                   [-44., 39.]])
LAT_NUMPY_ARRAY_INVALID_DEG = numpy.array([[420., -350.],
                                           [350., -610.],
                                           [330., 300.],
                                           [-440., 390.]])
LAT_NUMPY_ARRAY_SOME_INVALID_DEG = numpy.array([[42., -350.],
                                                [35., -61.],
                                                [330., 30.],
                                                [-440., 39.]])
LAT_NUMPY_ARRAY_WITH_NANS_DEG = numpy.array([[42., -35.],
                                             [numpy.nan, -61.],
                                             [33., 30.],
                                             [-44., numpy.nan]])

SINGLE_LONGITUDE_DEG = 45.
SINGLE_LNG_INVALID_DEG = 7000.
SINGLE_LNG_POSITIVE_IN_WEST_DEG = 270.
SINGLE_LNG_NEGATIVE_IN_WEST_DEG = -90.
LNG_NUMPY_ARRAY_DEG = numpy.array([[-73., 254.],
                                   [101., -149.],
                                   [84., 263.],
                                   [243., 76.]])
LNG_NUMPY_ARRAY_INVALID_DEG = numpy.array([[-730., 2540.],
                                           [1010., -1490.],
                                           [840., 2630.],
                                           [2430., 760.]])
LNG_NUMPY_ARRAY_SOME_INVALID_DEG = numpy.array([[-73., 2540.],
                                                [101., -1490.],
                                                [840., 263.],
                                                [243., 76.]])
LNG_NUMPY_ARRAY_POSITIVE_IN_WEST_DEG = numpy.array([[287., 254.],
                                                    [101., 211.],
                                                    [84., 263.],
                                                    [243., 76.]])
LNG_NUMPY_ARRAY_NEGATIVE_IN_WEST_DEG = numpy.array([[-73., -106.],
                                                    [101., -149.],
                                                    [84., -97.],
                                                    [-117., 76.]])


class ErrorCheckingTests(unittest.TestCase):
    """Each method is a unit test for error_checking.py."""

    def test_assert_columns_in_dataframe_list(self):
        """Checks assert_columns_in_dataframe when input is list."""

        with self.assertRaises(TypeError):
            error_checking.assert_columns_in_dataframe(
                REAL_NUMBER_LIST, FAKE_COLUMNS_IN_DATAFRAME)

    def test_assert_columns_in_dataframe_tuple(self):
        """Checks assert_columns_in_dataframe when input is tuple."""

        with self.assertRaises(TypeError):
            error_checking.assert_columns_in_dataframe(
                REAL_NUMBER_TUPLE, FAKE_COLUMNS_IN_DATAFRAME)

    def test_assert_columns_in_dataframe_numpy_array(self):
        """Checks assert_columns_in_dataframe when input is numpy array."""

        with self.assertRaises(TypeError):
            error_checking.assert_columns_in_dataframe(
                REAL_NUMPY_ARRAY, FAKE_COLUMNS_IN_DATAFRAME)

    def test_assert_columns_in_dataframe_missing_columns(self):
        """Checks assert_columns_in_dataframe.

        In this case, input is pandas DataFrame but is missing one of the
        desired columns.
        """

        with self.assertRaises(KeyError):
            error_checking.assert_columns_in_dataframe(
                DATAFRAME, FAKE_COLUMNS_IN_DATAFRAME)

    def test_assert_columns_in_dataframe_true(self):
        """Checks assert_columns_in_dataframe.

        In this case, input is pandas DataFrame with all desired columns.
        """

        error_checking.assert_columns_in_dataframe(DATAFRAME,
                                                   COLUMNS_IN_DATAFRAME)

    def test_assert_is_array_scalar(self):
        """Checks assert_is_array when input is scalar."""

        with self.assertRaises(TypeError):
            error_checking.assert_is_array(SINGLE_INTEGER)

    def test_assert_is_array_list(self):
        """Checks assert_is_array when input is list."""

        error_checking.assert_is_array(REAL_NUMBER_LIST)

    def test_assert_is_array_tuple(self):
        """Checks assert_is_array when input is tuple."""

        error_checking.assert_is_array(REAL_NUMBER_TUPLE)

    def test_assert_is_array_numpy_array(self):
        """Checks assert_is_array when input is numpy array."""

        error_checking.assert_is_array(REAL_NUMPY_ARRAY)

    def test_assert_is_list_scalar(self):
        """Checks assert_is_list when input is scalar."""

        with self.assertRaises(TypeError):
            error_checking.assert_is_list(SINGLE_INTEGER)

    def test_assert_is_list_true(self):
        """Checks assert_is_list when input is list."""

        error_checking.assert_is_list(REAL_NUMBER_LIST)

    def test_assert_is_list_tuple(self):
        """Checks assert_is_list when input is tuple."""

        with self.assertRaises(TypeError):
            error_checking.assert_is_list(REAL_NUMBER_TUPLE)

    def test_assert_is_list_numpy_array(self):
        """Checks assert_is_list when input is numpy array."""

        with self.assertRaises(TypeError):
            error_checking.assert_is_list(REAL_NUMPY_ARRAY)

    def test_assert_is_tuple_scalar(self):
        """Checks assert_is_tuple when input is scalar."""

        with self.assertRaises(TypeError):
            error_checking.assert_is_tuple(SINGLE_INTEGER)

    def test_assert_is_tuple_list(self):
        """Checks assert_is_tuple when input is list."""

        with self.assertRaises(TypeError):
            error_checking.assert_is_tuple(REAL_NUMBER_LIST)

    def test_assert_is_tuple_true(self):
        """Checks assert_is_tuple when input is tuple."""

        error_checking.assert_is_tuple(REAL_NUMBER_TUPLE)

    def test_assert_is_tuple_numpy_array(self):
        """Checks assert_is_tuple when input is numpy array."""

        with self.assertRaises(TypeError):
            error_checking.assert_is_tuple(REAL_NUMPY_ARRAY)

    def test_assert_is_numpy_array_scalar(self):
        """Checks assert_is_numpy_array when input is scalar."""

        with self.assertRaises(TypeError):
            error_checking.assert_is_numpy_array(SINGLE_INTEGER)

    def test_assert_is_numpy_array_list(self):
        """Checks assert_is_numpy_array when input is list."""

        with self.assertRaises(TypeError):
            error_checking.assert_is_numpy_array(REAL_NUMBER_LIST)

    def test_assert_is_numpy_array_tuple(self):
        """Checks assert_is_numpy_array when input is tuple."""

        with self.assertRaises(TypeError):
            error_checking.assert_is_numpy_array(REAL_NUMBER_TUPLE)

    def test_assert_is_numpy_array_true(self):
        """Checks assert_is_numpy_array when input is numpy array."""

        error_checking.assert_is_numpy_array(REAL_NUMPY_ARRAY)

    def test_assert_is_numpy_array_num_dim_not_integer(self):
        """Checks assert_is_numpy_array when `num_dimensions` is not integer."""

        with self.assertRaises(TypeError):
            error_checking.assert_is_numpy_array(
                REAL_NUMPY_ARRAY, num_dimensions=float(REAL_NUMPY_ARRAY.ndim))

    def test_assert_is_numpy_array_num_dim_negative(self):
        """Checks assert_is_numpy_array when `num_dimensions` is negative."""

        with self.assertRaises(ValueError):
            error_checking.assert_is_numpy_array(REAL_NUMPY_ARRAY,
                                                 num_dimensions=-1)

    def test_assert_is_numpy_array_num_dim_unexpected(self):
        """Checks assert_is_numpy_array when `num_dimensions` is unexpected."""

        with self.assertRaises(TypeError):
            error_checking.assert_is_numpy_array(
                REAL_NUMPY_ARRAY, num_dimensions=REAL_NUMPY_ARRAY.ndim + 1)

    def test_assert_is_numpy_array_num_dim_correct(self):
        """Checks assert_is_numpy_array when `num_dimensions` is correct."""

        error_checking.assert_is_numpy_array(
            REAL_NUMPY_ARRAY, num_dimensions=REAL_NUMPY_ARRAY.ndim)

    def test_assert_is_numpy_array_exact_dim_scalar(self):
        """Checks assert_is_numpy_array when `exact_dimensions` is scalar."""

        with self.assertRaises(TypeError):
            error_checking.assert_is_numpy_array(
                REAL_NUMPY_ARRAY, num_dimensions=REAL_NUMPY_ARRAY.ndim,
                exact_dimensions=REAL_NUMPY_ARRAY.shape[0])

    def test_assert_is_numpy_array_exact_dim_list(self):
        """Checks assert_is_numpy_array when `exact_dimensions` is list."""

        with self.assertRaises(TypeError):
            error_checking.assert_is_numpy_array(
                REAL_NUMPY_ARRAY, num_dimensions=REAL_NUMPY_ARRAY.ndim,
                exact_dimensions=REAL_NUMPY_ARRAY.shape)

    def test_assert_is_numpy_array_exact_dim_not_integers(self):
        """Checks assert_is_numpy_array when `exact_dimensions` is not int."""

        with self.assertRaises(TypeError):
            error_checking.assert_is_numpy_array(
                REAL_NUMPY_ARRAY, num_dimensions=REAL_NUMPY_ARRAY.ndim,
                exact_dimensions=numpy.asarray(REAL_NUMPY_ARRAY.shape,
                                               dtype=numpy.float64))

    def test_assert_is_numpy_array_exact_dim_negative(self):
        """Checks assert_is_numpy_array when `exact_dimensions` has negative."""

        these_dimensions = -1 * numpy.asarray(REAL_NUMPY_ARRAY.shape,
                                              dtype=numpy.int64)

        with self.assertRaises(ValueError):
            error_checking.assert_is_numpy_array(
                REAL_NUMPY_ARRAY, num_dimensions=REAL_NUMPY_ARRAY.ndim,
                exact_dimensions=these_dimensions)

    def test_assert_is_numpy_array_exact_dim_too_long(self):
        """Checks assert_is_numpy_array when `exact_dimensions` is too long."""

        these_dimensions = numpy.concatenate((
            numpy.asarray(REAL_NUMPY_ARRAY.shape, dtype=numpy.int64),
            numpy.array([1])))

        with self.assertRaises(TypeError):
            error_checking.assert_is_numpy_array(
                REAL_NUMPY_ARRAY, num_dimensions=REAL_NUMPY_ARRAY.ndim,
                exact_dimensions=these_dimensions)

    def test_assert_is_numpy_array_exact_dim_unexpected(self):
        """Checks assert_is_numpy_array when `exact_dimensions` is wrong."""

        these_dimensions = 1 + numpy.asarray(REAL_NUMPY_ARRAY.shape,
                                             dtype=numpy.int64)

        with self.assertRaises(TypeError):
            error_checking.assert_is_numpy_array(
                REAL_NUMPY_ARRAY, num_dimensions=REAL_NUMPY_ARRAY.ndim,
                exact_dimensions=these_dimensions)

    def test_assert_is_numpy_array_exact_dim_correct(self):
        """Checks assert_is_numpy_array when `exact_dimensions` is correct."""

        error_checking.assert_is_numpy_array(
            REAL_NUMPY_ARRAY, num_dimensions=REAL_NUMPY_ARRAY.ndim,
            exact_dimensions=numpy.asarray(REAL_NUMPY_ARRAY.shape,
                                           dtype=numpy.int64))

    def test_assert_is_non_array_true(self):
        """Checks assert_is_non_array when input is scalar."""

        error_checking.assert_is_non_array(SINGLE_INTEGER)

    def test_assert_is_non_array_list(self):
        """Checks assert_is_non_array when input is list."""

        with self.assertRaises(TypeError):
            error_checking.assert_is_non_array(REAL_NUMBER_LIST)

    def test_assert_is_non_array_tuple(self):
        """Checks assert_is_non_array when input is tuple."""

        with self.assertRaises(TypeError):
            error_checking.assert_is_non_array(REAL_NUMBER_TUPLE)

    def test_assert_is_non_array_numpy_array(self):
        """Checks assert_is_non_array when input is numpy array."""

        with self.assertRaises(TypeError):
            error_checking.assert_is_non_array(REAL_NUMPY_ARRAY)

    def test_assert_is_string_number(self):
        """Checks assert_is_string when input is number."""

        with self.assertRaises(TypeError):
            error_checking.assert_is_string(SINGLE_INTEGER)

    def test_assert_is_string_none(self):
        """Checks assert_is_string when input is None."""

        with self.assertRaises(TypeError):
            error_checking.assert_is_string(None)

    def test_assert_is_string_true(self):
        """Checks assert_is_string when input is string."""

        error_checking.assert_is_string(SINGLE_STRING)

    def test_assert_is_string_list_true(self):
        """Checks assert_is_string_list when input is string list."""

        error_checking.assert_is_string_list(STRING_LIST)

    def test_assert_file_exists_directory(self):
        """Checks assert_file_exists when input is directory."""

        with self.assertRaises(ValueError):
            error_checking.assert_file_exists(THIS_DIRECTORY_NAME)

    def test_assert_file_exists_fake(self):
        """Checks assert_file_exists when input is fake file."""

        with self.assertRaises(ValueError):
            error_checking.assert_file_exists(FAKE_FILE_NAME)

    def test_assert_file_exists_true(self):
        """Checks assert_file_exists when input is existent file."""

        error_checking.assert_file_exists(THIS_FILE_NAME)

    def test_assert_directory_exists_file(self):
        """Checks assert_directory_exists when input is file."""

        with self.assertRaises(ValueError):
            error_checking.assert_directory_exists(THIS_FILE_NAME)

    def test_assert_directory_exists_fake(self):
        """Checks assert_directory_exists when input is fake directory."""

        with self.assertRaises(ValueError):
            error_checking.assert_directory_exists(FAKE_DIRECTORY_NAME)

    def test_assert_directory_exists_true(self):
        """Checks assert_directory_exists when input is existent directory."""

        error_checking.assert_directory_exists(THIS_DIRECTORY_NAME)

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

    def test_assert_is_integer_complex(self):
        """Checks assert_is_integer when input is complex."""

        with self.assertRaises(TypeError):
            error_checking.assert_is_integer(SINGLE_COMPLEX_NUMBER)

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

    def test_assert_is_integer_numpy_array_true(self):
        """Checks assert_is_integer_numpy_array when condition is true."""

        error_checking.assert_is_integer_numpy_array(INTEGER_NUMPY_ARRAY)

    def test_assert_is_boolean_too_many_inputs(self):
        """Checks assert_is_boolean when input is array of Booleans."""

        with self.assertRaises(TypeError):
            error_checking.assert_is_boolean(BOOLEAN_NUMPY_ARRAY)

    def test_assert_is_boolean_float(self):
        """Checks assert_is_boolean when input is float."""

        with self.assertRaises(TypeError):
            error_checking.assert_is_boolean(SINGLE_FLOAT)

    def test_assert_is_boolean_true(self):
        """Checks assert_is_boolean when input is Boolean."""

        error_checking.assert_is_boolean(SINGLE_BOOLEAN)

    def test_assert_is_boolean_complex(self):
        """Checks assert_is_boolean when input is complex."""

        with self.assertRaises(TypeError):
            error_checking.assert_is_boolean(SINGLE_COMPLEX_NUMBER)

    def test_assert_is_boolean_nan(self):
        """Checks assert_is_boolean when input is NaN."""

        with self.assertRaises(TypeError):
            error_checking.assert_is_boolean(numpy.nan)

    def test_assert_is_boolean_none(self):
        """Checks assert_is_boolean when input is None."""

        with self.assertRaises(TypeError):
            error_checking.assert_is_boolean(None)

    def test_assert_is_boolean_integer(self):
        """Checks assert_is_boolean when input is integer."""

        with self.assertRaises(TypeError):
            error_checking.assert_is_boolean(SINGLE_INTEGER)

    def test_assert_is_boolean_numpy_array_true(self):
        """Checks assert_is_boolean_numpy_array when condition is true."""

        error_checking.assert_is_boolean_numpy_array(BOOLEAN_NUMPY_ARRAY)

    def test_assert_is_float_too_many_inputs(self):
        """Checks assert_is_float when input is array of floats."""

        with self.assertRaises(TypeError):
            error_checking.assert_is_float(FLOAT_NUMPY_ARRAY)

    def test_assert_is_float_true(self):
        """Checks assert_is_float when input is float."""

        error_checking.assert_is_float(SINGLE_FLOAT)

    def test_assert_is_float_boolean(self):
        """Checks assert_is_float when input is Boolean."""

        with self.assertRaises(TypeError):
            error_checking.assert_is_float(SINGLE_BOOLEAN)

    def test_assert_is_float_complex(self):
        """Checks assert_is_float when input is complex."""

        with self.assertRaises(TypeError):
            error_checking.assert_is_float(SINGLE_COMPLEX_NUMBER)

    def test_assert_is_float_nan(self):
        """Checks assert_is_float when input is NaN."""

        error_checking.assert_is_float(numpy.nan)

    def test_assert_is_float_none(self):
        """Checks assert_is_float when input is None."""

        with self.assertRaises(TypeError):
            error_checking.assert_is_float(None)

    def test_assert_is_float_integer(self):
        """Checks assert_is_float when input is integer."""

        with self.assertRaises(TypeError):
            error_checking.assert_is_float(SINGLE_INTEGER)

    def test_assert_is_float_numpy_array_true(self):
        """Checks assert_is_float_numpy_array when condition is true."""

        error_checking.assert_is_float_numpy_array(FLOAT_NUMPY_ARRAY)

    def test_assert_is_real_number_too_many_inputs(self):
        """Checks assert_is_real_number when input is array of real numbers."""

        with self.assertRaises(TypeError):
            error_checking.assert_is_real_number(FLOAT_NUMPY_ARRAY)

    def test_assert_is_real_number_float(self):
        """Checks assert_is_real_number when input is float."""

        error_checking.assert_is_real_number(SINGLE_FLOAT)

    def test_assert_is_real_number_boolean(self):
        """Checks assert_is_real_number when input is Boolean."""

        with self.assertRaises(TypeError):
            error_checking.assert_is_real_number(SINGLE_BOOLEAN)

    def test_assert_is_real_number_complex(self):
        """Checks assert_is_real_number when input is complex."""

        with self.assertRaises(TypeError):
            error_checking.assert_is_real_number(SINGLE_COMPLEX_NUMBER)

    def test_assert_is_real_number_nan(self):
        """Checks assert_is_real_number when input is NaN."""

        error_checking.assert_is_real_number(numpy.nan)

    def test_assert_is_real_number_none(self):
        """Checks assert_is_real_number when input is None."""

        with self.assertRaises(TypeError):
            error_checking.assert_is_real_number(None)

    def test_assert_is_real_number_integer(self):
        """Checks assert_is_real_number when input is integer."""

        error_checking.assert_is_real_number(SINGLE_INTEGER)

    def test_assert_is_real_numpy_array_true(self):
        """Checks assert_is_real_numpy_array when condition is true."""

        error_checking.assert_is_real_numpy_array(FLOAT_NUMPY_ARRAY)

    def test_assert_is_not_nan_too_many_inputs(self):
        """Checks assert_is_not_nan when input is array of floats."""

        with self.assertRaises(TypeError):
            error_checking.assert_is_not_nan(FLOAT_NUMPY_ARRAY)

    def test_assert_is_not_nan_float(self):
        """Checks assert_is_not_nan when input is float."""

        error_checking.assert_is_not_nan(SINGLE_FLOAT)

    def test_assert_is_not_nan_boolean(self):
        """Checks assert_is_not_nan when input is Boolean."""

        with self.assertRaises(TypeError):
            error_checking.assert_is_not_nan(SINGLE_BOOLEAN)

    def test_assert_is_not_nan_complex(self):
        """Checks assert_is_not_nan when input is complex."""

        with self.assertRaises(TypeError):
            error_checking.assert_is_not_nan(SINGLE_COMPLEX_NUMBER)

    def test_assert_is_not_nan_nan(self):
        """Checks assert_is_not_nan when input is NaN."""

        with self.assertRaises(ValueError):
            error_checking.assert_is_not_nan(numpy.nan)

    def test_assert_is_not_nan_none(self):
        """Checks assert_is_not_nan when input is None."""

        with self.assertRaises(TypeError):
            error_checking.assert_is_not_nan(None)

    def test_assert_is_not_nan_integer(self):
        """Checks assert_is_not_nan when input is integer."""

        error_checking.assert_is_not_nan(SINGLE_INTEGER)

    def test_assert_is_numpy_array_without_nan_all_nan(self):
        """Checks assert_is_numpy_array_without_nan; input is all NaN's."""

        with self.assertRaises(ValueError):
            error_checking.assert_is_numpy_array_without_nan(NAN_NUMPY_ARRAY)

    def test_assert_is_numpy_array_without_nan_mixed(self):
        """Checks assert_is_numpy_array_without_nan; input has some NaN's."""

        with self.assertRaises(ValueError):
            error_checking.assert_is_numpy_array_without_nan(
                POSITIVE_NUMPY_ARRAY_WITH_NANS)

    def test_assert_is_numpy_array_without_nan_true(self):
        """Checks assert_is_numpy_array_without_nan; input has no NaN's."""

        error_checking.assert_is_numpy_array_without_nan(POSITIVE_NUMPY_ARRAY)

    def test_assert_is_positive_negative(self):
        """Checks assert_is_greater with base_value = 0, input_variable < 0."""

        with self.assertRaises(ValueError):
            error_checking.assert_is_greater(SINGLE_NEGATIVE, 0)

    def test_assert_is_positive_zero(self):
        """Checks assert_is_greater with base_value = 0, input_variable = 0."""

        with self.assertRaises(ValueError):
            error_checking.assert_is_greater(SINGLE_ZERO, 0)

    def test_assert_is_positive_true(self):
        """Checks assert_is_greater with base_value = 0, input_variable > 0."""

        error_checking.assert_is_greater(SINGLE_POSITIVE, 0)

    def test_assert_is_positive_nan_allowed(self):
        """Checks assert_is_greater; input_variable = NaN, allow_nan = True."""

        error_checking.assert_is_greater(numpy.nan, 0, allow_nan=True)

    def test_assert_is_positive_nan_banned(self):
        """Checks assert_is_greater; input_variable = NaN, allow_nan = False."""

        with self.assertRaises(ValueError):
            error_checking.assert_is_greater(numpy.nan, 0, allow_nan=False)

    def test_assert_is_positive_numpy_array_true(self):
        """Checks assert_is_greater_numpy_array; base_value = 0, inputs > 0."""

        error_checking.assert_is_greater_numpy_array(POSITIVE_NUMPY_ARRAY, 0)

    def test_assert_is_positive_numpy_array_true_with_nan_allowed(self):
        """Checks assert_is_greater_numpy_array; base_value = 0, inputs > 0.

        In this case, input array contains NaN's and allow_nan = True.
        """

        error_checking.assert_is_greater_numpy_array(
            POSITIVE_NUMPY_ARRAY_WITH_NANS, 0, allow_nan=True)

    def test_assert_is_positive_numpy_array_true_with_nan_banned(self):
        """Checks assert_is_greater_numpy_array; base_value = 0, inputs > 0.

        In this case, input array contains NaN's and allow_nan = False.
        """

        with self.assertRaises(ValueError):
            error_checking.assert_is_greater_numpy_array(
                POSITIVE_NUMPY_ARRAY_WITH_NANS, 0, allow_nan=False)

    def test_assert_is_positive_numpy_array_non_negative(self):
        """Checks assert_is_greater_numpy_array; base_value = 0, inputs >= 0."""

        with self.assertRaises(ValueError):
            error_checking.assert_is_greater_numpy_array(
                NON_NEGATIVE_NUMPY_ARRAY, 0)

    def test_assert_is_positive_numpy_array_negative(self):
        """Checks assert_is_greater_numpy_array; base_value = 0, inputs < 0."""

        with self.assertRaises(ValueError):
            error_checking.assert_is_greater_numpy_array(
                NEGATIVE_NUMPY_ARRAY, 0)

    def test_assert_is_positive_numpy_array_non_positive(self):
        """Checks assert_is_greater_numpy_array; base_value = 0, inputs <= 0."""

        with self.assertRaises(ValueError):
            error_checking.assert_is_greater_numpy_array(
                NON_POSITIVE_NUMPY_ARRAY, 0)

    def test_assert_is_positive_numpy_array_mixed_sign(self):
        """assert_is_greater_numpy_array; base_value = 0, inputs mixed sign."""

        with self.assertRaises(ValueError):
            error_checking.assert_is_greater_numpy_array(
                MIXED_SIGN_NUMPY_ARRAY, 0)

    def test_assert_is_non_negative_false(self):
        """Checks assert_is_geq with base_value = 0, input_variable < 0."""

        with self.assertRaises(ValueError):
            error_checking.assert_is_geq(SINGLE_NEGATIVE, 0)

    def test_assert_is_non_negative_zero(self):
        """Checks assert_is_geq with base_value = 0, input_variable = 0."""

        error_checking.assert_is_geq(SINGLE_ZERO, 0)

    def test_assert_is_non_negative_positive(self):
        """Checks assert_is_geq with base_value = 0, input_variable > 0."""

        error_checking.assert_is_geq(SINGLE_POSITIVE, 0)

    def test_assert_is_non_negative_numpy_array_positive(self):
        """Checks assert_is_geq_numpy_array; base_value = 0, inputs > 0."""

        error_checking.assert_is_geq_numpy_array(POSITIVE_NUMPY_ARRAY, 0)

    def test_assert_is_non_negative_numpy_array_positive_with_nan_allowed(self):
        """Checks assert_is_geq_numpy_array; base_value = 0, inputs > 0.

        In this case, input array contains NaN's and allow_nan = True.
        """

        error_checking.assert_is_geq_numpy_array(
            POSITIVE_NUMPY_ARRAY_WITH_NANS, 0, allow_nan=True)

    def test_assert_is_non_negative_numpy_array_positive_with_nan_banned(self):
        """Checks assert_is_geq_numpy_array; base_value = 0, inputs > 0.

        In this case, input array contains NaN's and allow_nan = False.
        """

        with self.assertRaises(ValueError):
            error_checking.assert_is_geq_numpy_array(
                POSITIVE_NUMPY_ARRAY_WITH_NANS, 0, allow_nan=False)

    def test_assert_is_non_negative_numpy_array_non_negative_with_nan_allowed(
            self):
        """Checks assert_is_geq_numpy_array; base_value = 0, inputs >= 0.

        In this case, input array contains NaN's and allow_nan = True.
        """

        error_checking.assert_is_geq_numpy_array(
            NON_NEGATIVE_NUMPY_ARRAY_WITH_NANS, 0, allow_nan=True)

    def test_assert_is_non_negative_numpy_array_non_negative_with_nan_banned(
            self):
        """Checks assert_is_geq_numpy_array; base_value = 0, inputs >= 0.

        In this case, input array contains NaN's and allow_nan = False.
        """

        with self.assertRaises(ValueError):
            error_checking.assert_is_geq_numpy_array(
                NON_NEGATIVE_NUMPY_ARRAY_WITH_NANS, 0, allow_nan=False)

    def test_assert_is_non_negative_numpy_array_negative(self):
        """Checks assert_is_geq_numpy_array; base_value = 0, inputs < 0."""

        with self.assertRaises(ValueError):
            error_checking.assert_is_geq_numpy_array(NEGATIVE_NUMPY_ARRAY, 0)

    def test_assert_is_non_negative_numpy_array_non_positive(self):
        """Checks assert_is_geq_numpy_array; base_value = 0, inputs <= 0."""

        with self.assertRaises(ValueError):
            error_checking.assert_is_geq_numpy_array(
                NON_POSITIVE_NUMPY_ARRAY, 0)

    def test_assert_is_non_negative_numpy_array_mixed_sign(self):
        """assert_is_geq_numpy_array; base_value = 0, inputs mixed sign."""

        with self.assertRaises(ValueError):
            error_checking.assert_is_geq_numpy_array(MIXED_SIGN_NUMPY_ARRAY, 0)

    def test_assert_is_negative_true(self):
        """Checks assert_is_less_than; base_value = 0, input_variable < 0."""

        error_checking.assert_is_less_than(SINGLE_NEGATIVE, 0)

    def test_assert_is_negative_zero(self):
        """Checks assert_is_less_than; base_value = 0, input_variable = 0."""

        with self.assertRaises(ValueError):
            error_checking.assert_is_less_than(SINGLE_ZERO, 0)

    def test_assert_is_negative_positive(self):
        """Checks assert_is_less_than; base_value = 0, input_variable > 0."""

        with self.assertRaises(ValueError):
            error_checking.assert_is_less_than(SINGLE_POSITIVE, 0)

    def test_assert_is_negative_numpy_array_positive(self):
        """assert_is_less_than_numpy_array; base_value = 0, inputs > 0."""

        with self.assertRaises(ValueError):
            error_checking.assert_is_less_than_numpy_array(
                POSITIVE_NUMPY_ARRAY, 0)

    def test_assert_is_negative_numpy_array_non_negative(self):
        """assert_is_less_than_numpy_array; base_value = 0, inputs >= 0."""

        with self.assertRaises(ValueError):
            error_checking.assert_is_less_than_numpy_array(
                NON_NEGATIVE_NUMPY_ARRAY, 0)

    def test_assert_is_negative_numpy_array_true(self):
        """assert_is_less_than_numpy_array; base_value = 0, inputs < 0."""

        error_checking.assert_is_less_than_numpy_array(NEGATIVE_NUMPY_ARRAY, 0)

    def test_assert_is_negative_numpy_array_true_with_nan_allowed(self):
        """Checks assert_is_less_than_numpy_array; base_value = 0, inputs < 0.

        In this case, input array contains NaN's and allow_nan = True.
        """

        error_checking.assert_is_less_than_numpy_array(
            NEGATIVE_NUMPY_ARRAY_WITH_NANS, 0, allow_nan=True)

    def test_assert_is_negative_numpy_array_true_with_nan_banned(self):
        """Checks assert_is_less_than_numpy_array; base_value = 0, inputs < 0.

        In this case, input array contains NaN's and allow_nan = False.
        """

        with self.assertRaises(ValueError):
            error_checking.assert_is_less_than_numpy_array(
                NEGATIVE_NUMPY_ARRAY_WITH_NANS, 0, allow_nan=False)

    def test_assert_is_negative_numpy_array_non_positive(self):
        """assert_is_less_than_numpy_array; base_value = 0, inputs <= 0."""

        with self.assertRaises(ValueError):
            error_checking.assert_is_less_than_numpy_array(
                NON_POSITIVE_NUMPY_ARRAY, 0)

    def test_assert_is_negative_numpy_array_mixed_sign(self):
        """assert_is_less_than_numpy_array; base_value = 0, inputs mixed."""

        with self.assertRaises(ValueError):
            error_checking.assert_is_less_than_numpy_array(
                MIXED_SIGN_NUMPY_ARRAY, 0)

    def test_assert_is_non_positive_negative(self):
        """Checks assert_is_leq with base_value = 0, input_variable < 0."""

        error_checking.assert_is_leq(SINGLE_NEGATIVE, 0)

    def test_assert_is_non_positive_zero(self):
        """Checks assert_is_leq with base_value = 0, input_variable = 0."""

        error_checking.assert_is_leq(SINGLE_ZERO, 0)

    def test_assert_is_non_positive_false(self):
        """Checks assert_is_leq with base_value = 0, input_variable > 0."""

        with self.assertRaises(ValueError):
            error_checking.assert_is_leq(SINGLE_POSITIVE, 0)

    def test_assert_is_non_positive_numpy_array_positive(self):
        """Checks assert_is_leq_numpy_array; base_value = 0, inputs > 0."""

        with self.assertRaises(ValueError):
            error_checking.assert_is_leq_numpy_array(POSITIVE_NUMPY_ARRAY, 0)

    def test_assert_is_non_positive_numpy_array_non_negative(self):
        """Checks assert_is_leq_numpy_array; base_value = 0, inputs >= 0."""

        with self.assertRaises(ValueError):
            error_checking.assert_is_leq_numpy_array(
                NON_NEGATIVE_NUMPY_ARRAY, 0)

    def test_assert_is_non_positive_numpy_array_negative(self):
        """Checks assert_is_leq_numpy_array; base_value = 0, inputs < 0."""

        error_checking.assert_is_leq_numpy_array(NEGATIVE_NUMPY_ARRAY, 0)

    def test_assert_is_non_positive_numpy_array_negative_with_nan_allowed(self):
        """Checks assert_is_leq_numpy_array; base_value = 0, inputs < 0.

        In this case, input array contains NaN's and allow_nan = True.
        """

        error_checking.assert_is_leq_numpy_array(
            NEGATIVE_NUMPY_ARRAY_WITH_NANS, 0, allow_nan=True)

    def test_assert_is_non_positive_numpy_array_negative_with_nan_banned(self):
        """Checks assert_is_leq_numpy_array; base_value = 0, inputs < 0.

        In this case, input array contains NaN's and allow_nan = False.
        """

        with self.assertRaises(ValueError):
            error_checking.assert_is_leq_numpy_array(
                NEGATIVE_NUMPY_ARRAY_WITH_NANS, 0, allow_nan=False)

    def test_assert_is_non_positive_numpy_array_non_positive(self):
        """Checks assert_is_leq_numpy_array; base_value = 0, inputs <= 0."""

        error_checking.assert_is_leq_numpy_array(NON_POSITIVE_NUMPY_ARRAY, 0)

    def test_assert_is_non_positive_numpy_array_non_positive_with_nan_allowed(
            self):
        """Checks assert_is_leq_numpy_array; base_value = 0, inputs <= 0.

        In this case, input array contains NaN's and allow_nan = True.
        """

        error_checking.assert_is_leq_numpy_array(
            NON_POSITIVE_NUMPY_ARRAY_WITH_NANS, 0, allow_nan=True)

    def test_assert_is_non_positive_numpy_array_non_positive_with_nan_banned(
            self):
        """Checks assert_is_leq_numpy_array; base_value = 0, inputs <= 0.

        In this case, input array contains NaN's and allow_nan = False.
        """

        with self.assertRaises(ValueError):
            error_checking.assert_is_leq_numpy_array(
                NON_POSITIVE_NUMPY_ARRAY_WITH_NANS, 0, allow_nan=False)

    def test_assert_is_non_positive_numpy_array_mixed_sign(self):
        """assert_is_leq_numpy_array; base_value = 0, inputs mixed sign."""

        with self.assertRaises(ValueError):
            error_checking.assert_is_leq_numpy_array(MIXED_SIGN_NUMPY_ARRAY, 0)

    def test_assert_is_valid_latitude_false(self):
        """Checks assert_is_valid_latitude when latitude is invalid."""

        with self.assertRaises(ValueError):
            error_checking.assert_is_valid_latitude(SINGLE_LAT_INVALID_DEG)

    def test_assert_is_valid_latitude_true(self):
        """Checks assert_is_valid_latitude when latitude is valid."""

        error_checking.assert_is_valid_latitude(SINGLE_LATITUDE_DEG)

    def test_assert_is_valid_latitude_nan_allowed(self):
        """Checks assert_is_valid_latitude; input = NaN, allow_nan = True."""

        error_checking.assert_is_valid_latitude(numpy.nan, allow_nan=True)

    def test_assert_is_valid_latitude_nan_not_allowed(self):
        """Checks assert_is_valid_latitude; input = NaN, allow_nan = False."""

        with self.assertRaises(ValueError):
            error_checking.assert_is_valid_latitude(numpy.nan, allow_nan=False)

    def test_assert_is_valid_lat_numpy_array_all_invalid(self):
        """Checks assert_is_valid_lat_numpy_array; all latitudes invalid."""

        with self.assertRaises(ValueError):
            error_checking.assert_is_valid_lat_numpy_array(
                LAT_NUMPY_ARRAY_INVALID_DEG)

    def test_assert_is_valid_lat_numpy_array_some_invalid(self):
        """Checks assert_is_valid_lat_numpy_array; some latitudes invalid."""

        with self.assertRaises(ValueError):
            error_checking.assert_is_valid_lat_numpy_array(
                LAT_NUMPY_ARRAY_SOME_INVALID_DEG)

    def test_assert_is_valid_lat_numpy_array_true(self):
        """Checks assert_is_valid_lat_numpy_array; all latitudes valid."""

        error_checking.assert_is_valid_lat_numpy_array(LAT_NUMPY_ARRAY_DEG)

    def test_assert_is_valid_lat_numpy_array_true_with_nan_allowed(self):
        """Checks assert_is_valid_lat_numpy_array; all latitudes valid or NaN.

        In this case, allow_nan = True."""

        error_checking.assert_is_valid_lat_numpy_array(
            LAT_NUMPY_ARRAY_WITH_NANS_DEG, allow_nan=True)

    def test_assert_is_valid_lat_numpy_array_true_with_nan_banned(self):
        """Checks assert_is_valid_lat_numpy_array; all latitudes valid or NaN.

        In this case, allow_nan = False."""

        with self.assertRaises(ValueError):
            error_checking.assert_is_valid_lat_numpy_array(
                LAT_NUMPY_ARRAY_WITH_NANS_DEG, allow_nan=False)

    def test_assert_is_valid_longitude_false(self):
        """Checks assert_is_valid_longitude when longitude is invalid."""

        with self.assertRaises(ValueError):
            error_checking.assert_is_valid_longitude(SINGLE_LNG_INVALID_DEG)

    def test_assert_is_valid_longitude_true(self):
        """Checks assert_is_valid_longitude when longitude is valid."""

        error_checking.assert_is_valid_longitude(SINGLE_LONGITUDE_DEG)

    def test_assert_is_valid_longitude_positive_in_west_false(self):
        """Checks assert_is_valid_longitude with positive_in_west_flag = True.

        In this case, longitude is negative in western hemisphere.
        """

        with self.assertRaises(ValueError):
            error_checking.assert_is_valid_longitude(
                SINGLE_LNG_NEGATIVE_IN_WEST_DEG, positive_in_west_flag=True)

    def test_assert_is_valid_longitude_positive_in_west_true(self):
        """Checks assert_is_valid_longitude with positive_in_west_flag = True.

        In this case, longitude is positive in western hemisphere.
        """

        error_checking.assert_is_valid_longitude(
            SINGLE_LNG_POSITIVE_IN_WEST_DEG, positive_in_west_flag=True)

    def test_assert_is_valid_longitude_negative_in_west_false(self):
        """Checks assert_is_valid_longitude with negative_in_west_flag = True.

        In this case, longitude is positive in western hemisphere.
        """

        with self.assertRaises(ValueError):
            error_checking.assert_is_valid_longitude(
                SINGLE_LNG_POSITIVE_IN_WEST_DEG, negative_in_west_flag=True)

    def test_assert_is_valid_longitude_negative_in_west_true(self):
        """Checks assert_is_valid_longitude with negative_in_west_flag = True.

        In this case, longitude is negative in western hemisphere.
        """

        error_checking.assert_is_valid_longitude(
            SINGLE_LNG_NEGATIVE_IN_WEST_DEG, negative_in_west_flag=True)

    def test_assert_is_valid_lng_numpy_array_all_invalid(self):
        """Checks assert_is_valid_lng_numpy_array; all longitudes invalid."""

        with self.assertRaises(ValueError):
            error_checking.assert_is_valid_lng_numpy_array(
                LNG_NUMPY_ARRAY_INVALID_DEG)

    def test_assert_is_valid_lng_numpy_array_some_invalid(self):
        """Checks assert_is_valid_lng_numpy_array; some longitudes invalid."""

        with self.assertRaises(ValueError):
            error_checking.assert_is_valid_lng_numpy_array(
                LNG_NUMPY_ARRAY_SOME_INVALID_DEG)

    def test_assert_is_valid_lng_numpy_array_true(self):
        """Checks assert_is_valid_lng_numpy_array; all longitudes valid."""

        error_checking.assert_is_valid_lng_numpy_array(LNG_NUMPY_ARRAY_DEG)

    def test_assert_is_valid_lng_numpy_array_positive_in_west_false(self):
        """Checks assert_is_valid_lng_numpy_array; positive_in_west_flag = True.

        In this case, longitudes in western hemisphere are negative.
        """

        with self.assertRaises(ValueError):
            error_checking.assert_is_valid_lng_numpy_array(
                LNG_NUMPY_ARRAY_NEGATIVE_IN_WEST_DEG,
                positive_in_west_flag=True)

    def test_assert_is_valid_lng_numpy_array_positive_in_west_true(self):
        """Checks assert_is_valid_lng_numpy_array; positive_in_west_flag = True.

        In this case, longitudes in western hemisphere are positive.
        """

        error_checking.assert_is_valid_lng_numpy_array(
            LNG_NUMPY_ARRAY_POSITIVE_IN_WEST_DEG, positive_in_west_flag=True)

    def test_assert_is_valid_lng_numpy_array_negative_in_west_false(self):
        """Checks assert_is_valid_lng_numpy_array; negative_in_west_flag = True.

        In this case, longitudes in western hemisphere are positive.
        """

        with self.assertRaises(ValueError):
            error_checking.assert_is_valid_lng_numpy_array(
                LNG_NUMPY_ARRAY_POSITIVE_IN_WEST_DEG,
                negative_in_west_flag=True)

    def test_assert_is_valid_lng_numpy_array_negative_in_west_true(self):
        """Checks assert_is_valid_lng_numpy_array; negative_in_west_flag = True.

        In this case, longitudes in western hemisphere are negative.
        """

        error_checking.assert_is_valid_lng_numpy_array(
            LNG_NUMPY_ARRAY_NEGATIVE_IN_WEST_DEG, negative_in_west_flag=True)


if __name__ == '__main__':
    unittest.main()
