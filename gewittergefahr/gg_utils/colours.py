"""Methods for processing colours."""

import copy
import numpy
import skimage.color
from sklearn.metrics.pairwise import euclidean_distances
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
DEFAULT_MIN_RGB_DISTANCE = 0.25


def get_random_colours(num_colours, colour_to_exclude_rgb=None,
                       min_rgb_distance=DEFAULT_MIN_RGB_DISTANCE):
    """Returns list of random colours.

    N = number of colours

    :param num_colours: Number of colours desired.
    :param colour_to_exclude_rgb: Colour to exclude (length-3 numpy array with
        values in 0...1).
    :param min_rgb_distance: All colours returned will be at least this far away
        from `colour_to_exclude_rgb`.  Distance is Euclidean.
    :return: rgb_matrix: N-by-3 numpy array with values in 0...1.  Each row is
        one colour.
    """

    orig_num_colours = num_colours + 0

    if colour_to_exclude_rgb is not None:
        error_checking.assert_is_numpy_array(
            colour_to_exclude_rgb, exact_dimensions=numpy.array([3], dtype=int)
        )

        error_checking.assert_is_geq_numpy_array(colour_to_exclude_rgb, 0.)
        error_checking.assert_is_leq_numpy_array(colour_to_exclude_rgb, 1.)
        error_checking.assert_is_greater(min_rgb_distance, 0.)
        error_checking.assert_is_leq(min_rgb_distance, 1.)

        num_colours = 10 * num_colours

    rgb_matrix = numpy.random.uniform(low=0., high=1., size=(num_colours, 3))

    if colour_to_exclude_rgb is not None:
        colour_to_exclude_rgb = numpy.reshape(colour_to_exclude_rgb, (1, 3))

        squared_distances = euclidean_distances(
            X=rgb_matrix, Y=numpy.reshape(colour_to_exclude_rgb, (1, 3)),
            squared=True
        )

        good_indices = numpy.where(
            squared_distances >= min_rgb_distance ** 2
        )[0]

        rgb_matrix = rgb_matrix[good_indices, ...]

    num_colours = min([
        orig_num_colours, rgb_matrix.shape[0]
    ])

    return rgb_matrix[:num_colours, ...]


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
        0., float(num_lab_values - 1), num=num_colours
    )
    linear_indices = numpy.round(linear_indices).astype(int)

    lab_matrix = numpy.full((num_colours, 3), numpy.nan)
    lab_matrix[:, 0], lab_matrix[:, 1], lab_matrix[:, 2] = numpy.unravel_index(
        linear_indices, dims=(num_l_values, num_a_values, num_b_values)
    )

    rgb_matrix = skimage.color.lab2rgb(
        numpy.reshape(lab_matrix, (1, num_colours, 3))
    )
    return numpy.reshape(rgb_matrix, (num_colours, 3))


def get_uniform_colours_in_hsv_space(
        num_colours, colour_to_exclude_rgb=None,
        min_rgb_distance_from_colour=DEFAULT_MIN_RGB_DISTANCE):
    """Returns array of uniformly spaced colours in HSV space.

    N = number of colours

    :param num_colours: Number of colours.
    :param colour_to_exclude_rgb: This colour (and similar colours) will not be
        returned.  If None, no colours will be excluded.
    :param min_rgb_distance_from_colour: No colour within this Euclidean RGB
        distance of `colour_to_exclude_rgb` will be returned.
    :return: rgb_matrix: N-by-3 numpy array, where each row is an RGB colour.
        All values range from 0...1.
    """

    if colour_to_exclude_rgb is not None:
        error_checking.assert_is_numpy_array(
            colour_to_exclude_rgb, exact_dimensions=numpy.array([3], dtype=int)
        )

        error_checking.assert_is_geq_numpy_array(colour_to_exclude_rgb, 0.)
        error_checking.assert_is_leq_numpy_array(colour_to_exclude_rgb, 1.)
        error_checking.assert_is_greater(min_rgb_distance_from_colour, 0.)
        error_checking.assert_is_leq(min_rgb_distance_from_colour, 1.)

        orig_num_colours = num_colours + 0
        num_colours = 10 * num_colours

    num_hsv_values = (
        NUM_H_FOR_HSV_SPACE * NUM_S_FOR_HSV_SPACE * NUM_V_FOR_HSV_SPACE
    )

    linear_indices = numpy.linspace(
        0., float(num_hsv_values - 1), num=num_colours
    )
    linear_indices = numpy.round(linear_indices).astype(int)

    hsv_matrix = numpy.full((num_colours, 3), numpy.nan)
    hsv_matrix[:, 0], hsv_matrix[:, 1], hsv_matrix[:, 2] = numpy.unravel_index(
        linear_indices,
        dims=(NUM_H_FOR_HSV_SPACE, NUM_S_FOR_HSV_SPACE, NUM_V_FOR_HSV_SPACE)
    )

    hsv_matrix = hsv_matrix / 255

    rgb_matrix = skimage.color.hsv2rgb(
        numpy.reshape(hsv_matrix, (1, num_colours, 3))
    )
    rgb_matrix = numpy.reshape(rgb_matrix, (num_colours, 3))

    if colour_to_exclude_rgb is not None:
        all_indices = numpy.linspace(
            0, num_colours - 1, num=num_colours, dtype=int)

        numpy.random.shuffle(all_indices)
        good_indices = []

        for i in all_indices:
            this_distance = numpy.linalg.norm(
                rgb_matrix[i, :] - colour_to_exclude_rgb
            )
            if not this_distance >= min_rgb_distance_from_colour:
                continue

            good_indices.append(i)
            if len(good_indices) == orig_num_colours:
                break

        good_indices = numpy.array(good_indices, dtype=int)
        rgb_matrix = rgb_matrix[good_indices, :]
        num_colours = copy.deepcopy(orig_num_colours)

    return rgb_matrix[0:(num_colours + 1), :]
