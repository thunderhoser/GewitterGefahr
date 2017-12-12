"""Methods for processing colours."""

import copy
import numpy
import skimage.color
from gewittergefahr.gg_utils import error_checking

MIN_L_FOR_LAB_SPACE = 0.
MAX_L_FOR_LAB_SPACE = 100.
MIN_A_FOR_LAB_SPACE = -128.
MAX_A_FOR_LAB_SPACE = 127.
MIN_B_FOR_LAB_SPACE = -128.
MAX_B_FOR_LAB_SPACE = 127.

NUM_H_FOR_HSV_SPACE = 256
NUM_S_FOR_HSV_SPACE = 256
NUM_V_FOR_HSV_SPACE = 256
DEFAULT_MIN_HSV_DISTANCE_FROM_COLOUR = 0.5


def get_uniform_colours_in_lab_space(num_colours):
    """Returns array of uniformly spaced colours in LAB space.

    N = number of colours

    :param num_colours: Number of colours.
    :return: rgb_matrix: N-by-3 numpy array, where each row is an RGB colour.
        All values range from 0...1.
    """

    num_l_values = int(numpy.round(MAX_L_FOR_LAB_SPACE - MIN_L_FOR_LAB_SPACE))
    num_a_values = int(numpy.round(MAX_A_FOR_LAB_SPACE - MIN_A_FOR_LAB_SPACE))
    num_b_values = int(numpy.round(MAX_B_FOR_LAB_SPACE - MIN_B_FOR_LAB_SPACE))
    num_lab_values = num_l_values * num_a_values * num_b_values

    linear_indices = numpy.linspace(
        0., float(num_lab_values - 1), num=num_colours)
    linear_indices = numpy.round(linear_indices).astype(int)

    lab_matrix = numpy.full((num_colours, 3), numpy.nan)
    lab_matrix[:, 0], lab_matrix[:, 1], lab_matrix[:, 2] = numpy.unravel_index(
        linear_indices, dims=(num_l_values, num_a_values, num_b_values))

    rgb_matrix = skimage.color.lab2rgb(
        numpy.reshape(lab_matrix, (1, num_colours, 3)))
    return numpy.reshape(rgb_matrix, (num_colours, 3))


def get_uniform_colours_in_hsv_space(
        num_colours, colour_to_exclude_rgb=None,
        min_hsv_distance_from_colour=DEFAULT_MIN_HSV_DISTANCE_FROM_COLOUR):
    """Returns array of uniformly spaced colours in HSV space.

    N = number of colours

    :param num_colours: Number of colours.
    :param colour_to_exclude_rgb: This colour (and similar colours) will not be
        returned.  If None, no colours will be excluded.
    :param min_hsv_distance_from_colour: No colour within this Euclidean HSV
        distance of `colour_to_exclude_rgb` will be returned.
    :return: rgb_matrix: N-by-3 numpy array, where each row is an RGB colour.
        All values range from 0...1.
    """

    if colour_to_exclude_rgb is not None:
        error_checking.assert_is_numpy_array(
            colour_to_exclude_rgb, exact_dimensions=numpy.array([3]))
        error_checking.assert_is_geq_numpy_array(colour_to_exclude_rgb, 0.)
        error_checking.assert_is_leq_numpy_array(colour_to_exclude_rgb, 1.)

        error_checking.assert_is_greater(min_hsv_distance_from_colour, 0.)
        error_checking.assert_is_leq(min_hsv_distance_from_colour, 1.)

        colour_to_exclude_hsv = skimage.color.rgb2hsv(
            numpy.reshape(colour_to_exclude_rgb, (1, 1, 3)))
        colour_to_exclude_hsv = numpy.reshape(colour_to_exclude_hsv, 3)

        orig_num_colours = copy.deepcopy(num_colours)
        num_colours = 10 * num_colours

    num_hsv_values = (
        NUM_H_FOR_HSV_SPACE * NUM_S_FOR_HSV_SPACE * NUM_V_FOR_HSV_SPACE)
    linear_indices = numpy.linspace(
        0., float(num_hsv_values - 1), num=num_colours)
    linear_indices = numpy.round(linear_indices).astype(int)

    hsv_matrix = numpy.full((num_colours, 3), numpy.nan)
    hsv_matrix[:, 0], hsv_matrix[:, 1], hsv_matrix[:, 2] = numpy.unravel_index(
        linear_indices,
        dims=(NUM_H_FOR_HSV_SPACE, NUM_S_FOR_HSV_SPACE, NUM_V_FOR_HSV_SPACE))
    hsv_matrix = hsv_matrix / 255

    if colour_to_exclude_rgb is not None:
        good_indices = []
        all_indices = range(num_colours)
        numpy.random.shuffle(all_indices)

        for i in all_indices:
            this_distance = numpy.linalg.norm(
                hsv_matrix[i, :] - colour_to_exclude_hsv)
            if this_distance < min_hsv_distance_from_colour:
                continue

            good_indices.append(i)
            if len(good_indices) == orig_num_colours:
                break

        good_indices = numpy.array(good_indices, dtype=int)
        hsv_matrix = hsv_matrix[good_indices, :]
        num_colours = copy.deepcopy(orig_num_colours)

    rgb_matrix = skimage.color.hsv2rgb(
        numpy.reshape(hsv_matrix, (1, num_colours, 3)))
    return numpy.reshape(rgb_matrix, (num_colours, 3))
