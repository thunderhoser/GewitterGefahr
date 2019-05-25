"""Very fancy downsampling."""

import numpy
from gewittergefahr.gg_utils import target_val_utils
from gewittergefahr.gg_utils import error_checking
from gewittergefahr.deep_learning import deep_learning_utils as dl_utils

LARGE_INTEGER = int(1e12)
FRACTION_UNINTERESTING_TIMES_TO_OMIT = 0.8


def _report_class_fractions(target_values):
    """Reports fraction of examples in each class.

    :param target_values: 1-D numpy array of target values (integer class
        labels).
    """

    unique_target_values, unique_counts = numpy.unique(
        target_values, return_counts=True)

    print('\n')

    for k in range(len(unique_target_values)):
        print('{0:d} examples in class = {1:d}'.format(
            unique_counts[k], unique_target_values[k]
        ))

    print('\n')


def _find_storm_cells(object_id_strings, desired_cell_id_strings):
    """Finds storm IDs from set 2 in set 1.

    N = number of storm objects
    n = number of desired storm cells

    :param object_id_strings: length-N list with all storm IDs.
    :param desired_cell_id_strings: length-n list with desired storm IDs.
    :return: relevant_indices: 1-D numpy array of indices, such that
        `object_id_strings[relevant_indices]` yields all IDs in
        `object_id_strings` that are in `desired_cell_id_strings`, including
        duplicates.
    :raises: ValueError: if not all desired ID were found in
        `object_id_strings`.
    """

    desired_cell_id_strings = numpy.unique(numpy.array(desired_cell_id_strings))

    relevant_flags = numpy.in1d(
        numpy.array(object_id_strings), desired_cell_id_strings,
        assume_unique=False)
    relevant_indices = numpy.where(relevant_flags)[0]

    cell_id_strings = numpy.unique(
        numpy.array(object_id_strings)[relevant_indices]
    )
    if numpy.array_equal(cell_id_strings, desired_cell_id_strings):
        return relevant_indices

    error_string = (
        '\nDesired storm IDs:\n{0:s}\nFound storm IDs:\n{1:s}\nNot all desired '
        'storm IDs were found, as shown above.'
    ).format(
        str(desired_cell_id_strings), str(cell_id_strings)
    )

    raise ValueError(error_string)


def _find_uncovered_times(all_times_unix_sec, covered_times_unix_sec):
    """Finds times in set 1 that are not in set 2.

    :param all_times_unix_sec: 1-D numpy array with all times.
    :param covered_times_unix_sec: 1-D numpy array of covered times.
    :return: uncovered_indices: 1-D numpy array of indices, such that
        `all_times_unix_sec[uncovered_indices]` yields all times in
        `all_times_unix_sec` that are not in `covered_times_unix_sec`, including
        duplicates.
    :raises: ValueError: if not all covered times were found in
        `all_times_unix_sec`.
    """

    covered_times_unix_sec = numpy.unique(covered_times_unix_sec)

    covered_flags = numpy.in1d(
        all_times_unix_sec, covered_times_unix_sec, assume_unique=False)
    covered_indices = numpy.where(covered_flags)[0]

    found_times_unix_sec = numpy.unique(all_times_unix_sec[covered_indices])
    if numpy.array_equal(found_times_unix_sec, covered_times_unix_sec):
        return numpy.where(numpy.invert(covered_flags))[0]

    error_string = (
        '\nCovered times:\n{0:s}\nCovered times found in all_times_unix_sec:\n'
        '{1:s}\nNot all covered times were found, as shown above.'
    ).format(str(covered_times_unix_sec), str(found_times_unix_sec))

    raise ValueError(error_string)


def _downsampling_base(
        primary_id_strings, storm_times_unix_sec, target_values, target_name,
        class_fraction_dict, test_mode=False):
    """Base for `downsample_for_training` and `downsample_for_non_training`.

    The procedure is described below.

    [1] Find all storm objects in the highest class (e.g., tornadic).  Call this
        set {s_highest}.
    [2] Find all storm cells with at least one object in {s_highest}.  Call this
        set {S_highest}.
    [3] Find all time steps with at least one storm cell in {S_highest}.  Call
        this set {t_highest}.
    [4] Randomly remove a large fraction of time steps NOT in {t_highest}.
    [5] Downsample remaining storm objects, leaving a prescribed fraction in
        each class (according to `class_fraction_dict`).

    N = number of storm objects before downsampling

    :param primary_id_strings: length-N list of primary storm IDs.
    :param storm_times_unix_sec: length-N numpy array of corresponding times.
    :param target_values: length-N numpy array of corresponding target values
        (integer class labels).
    :param target_name: Name of target variable (must be accepted by
        `target_val_utils.target_name_to_params`).
    :param class_fraction_dict: Dictionary, where each key is an integer class
        label (-2 for "dead storm") and the corresponding value is the
        sampling fraction.
    :param test_mode: Never mind.  Just leave this alone.
    :return: indices_to_keep: 1-D numpy array of indices to keep.
    """

    _report_class_fractions(target_values)
    error_checking.assert_is_boolean(test_mode)

    num_storm_objects = len(primary_id_strings)
    num_classes = target_val_utils.target_name_to_num_classes(
        target_name=target_name, include_dead_storms=False)

    # Step 1.
    print((
        'Finding storm objects in class {0:d} (the highest class), yielding set'
        ' {{s_highest}}...'
    ).format(num_classes - 1))

    highest_class_indices = numpy.where(target_values == num_classes - 1)[0]

    print('{{s_highest}} contains {0:d} of {1:d} storm objects.'.format(
        len(highest_class_indices), num_storm_objects
    ))

    # Step 2.
    print ('Finding storm cells with at least one object in {s_highest}, '
           'yielding set {S_highest}...')

    highest_class_indices = _find_storm_cells(
        object_id_strings=primary_id_strings,
        desired_cell_id_strings=
        [primary_id_strings[k] for k in highest_class_indices]
    )

    print('{{S_highest}} contains {0:d} of {1:d} storm objects.'.format(
        len(highest_class_indices), num_storm_objects)
    )

    # Step 3.
    print ('Finding all time steps with at least one storm cell in '
           '{S_highest}, yielding set {t_highest}...')

    lower_class_times_unix_sec = (
        set(storm_times_unix_sec.tolist()) -
        set(storm_times_unix_sec[highest_class_indices].tolist())
    )
    lower_class_times_unix_sec = numpy.array(
        list(lower_class_times_unix_sec), dtype=int)

    # Step 4.
    print('Randomly removing {0:.1f}% of times not in {{t_highest}}...'.format(
        FRACTION_UNINTERESTING_TIMES_TO_OMIT * 100))

    this_num_times = int(numpy.round(
        FRACTION_UNINTERESTING_TIMES_TO_OMIT * len(lower_class_times_unix_sec)
    ))

    if test_mode:
        times_to_remove_unix_sec = lower_class_times_unix_sec[:this_num_times]
    else:
        times_to_remove_unix_sec = numpy.random.choice(
            lower_class_times_unix_sec, size=this_num_times, replace=False)

    indices_to_keep = _find_uncovered_times(
        all_times_unix_sec=storm_times_unix_sec,
        covered_times_unix_sec=times_to_remove_unix_sec)

    _report_class_fractions(target_values[indices_to_keep])

    # Step 5.
    print('Downsampling storm objects from remaining times...')
    subindices_to_keep = dl_utils.sample_by_class(
        sampling_fraction_by_class_dict=class_fraction_dict,
        target_name=target_name, target_values=target_values[indices_to_keep],
        num_examples_total=LARGE_INTEGER, test_mode=test_mode)

    return indices_to_keep[subindices_to_keep]


def downsample_for_non_training(
        primary_id_strings, storm_times_unix_sec, target_values, target_name,
        class_fraction_dict, test_mode=False):
    """Fancy downsampling to create validation or testing data.

    The procedure is described in `_downsampling_base`.

    N = number of storm objects before downsampling
    n = number of storm objects after final downsampling

    :param primary_id_strings: See doc for `_downsampling_base`.
    :param storm_times_unix_sec: Same.
    :param target_values: Same.
    :param target_name: Same.
    :param class_fraction_dict: Same.
    :param test_mode: Same.
    :return: indices_to_keep: Same.
    """

    indices_to_keep = _downsampling_base(
        primary_id_strings=primary_id_strings,
        storm_times_unix_sec=storm_times_unix_sec,
        target_values=target_values, target_name=target_name,
        class_fraction_dict=class_fraction_dict, test_mode=test_mode)

    _report_class_fractions(target_values[indices_to_keep])

    return indices_to_keep


def downsample_for_training(
        primary_id_strings, storm_times_unix_sec, target_values, target_name,
        class_fraction_dict, test_mode=False):
    """Fancy downsampling to create training data.

    The procedure is described below.

    [1-5] Run `downsample_for_non_training`.
    [6] Find remaining storm objects in the highest class (e.g., tornadic).
        Call this set {s_highest}.
    [7] Find remaining storm cells with at least one object in {s_highest}.
        Call this set {S_highest}.  Add storm objects from {S_highest} to the
        selected set.

    :param primary_id_strings: See doc for `_downsampling_base`.
    :param storm_times_unix_sec: Same.
    :param target_values: Same.
    :param target_name: Same.
    :param class_fraction_dict: Same.
    :param test_mode: Same.
    :return: indices_to_keep: Same.
    """

    indices_to_keep = _downsampling_base(
        primary_id_strings=primary_id_strings,
        storm_times_unix_sec=storm_times_unix_sec,
        target_values=target_values, target_name=target_name,
        class_fraction_dict=class_fraction_dict, test_mode=test_mode)

    num_classes = target_val_utils.target_name_to_num_classes(
        target_name=target_name, include_dead_storms=False)

    # Step 6.
    print((
        'Finding storm objects in class {0:d} (the highest class), yielding set'
        ' {{s_highest}}...'
    ).format(num_classes - 1))

    these_subindices = numpy.where(
        target_values[indices_to_keep] == num_classes - 1
    )[0]

    highest_class_indices = indices_to_keep[these_subindices]

    print('{{s_highest}} contains {0:d} of {1:d} storm objects.'.format(
        len(highest_class_indices), len(indices_to_keep)
    ))

    # Step 7.
    print ('Finding storm cells with at least one object in {s_highest}, '
           'yielding set {S_highest}...')

    highest_class_indices = _find_storm_cells(
        object_id_strings=primary_id_strings,
        desired_cell_id_strings=
        [primary_id_strings[k] for k in highest_class_indices]
    )

    print('{{S_highest}} contains {0:d} of {1:d} storm objects.'.format(
        len(highest_class_indices), len(primary_id_strings)
    ))

    indices_to_keep = (
        set(indices_to_keep.tolist()) | set(highest_class_indices.tolist())
    )
    indices_to_keep = numpy.array(list(indices_to_keep), dtype=int)

    _report_class_fractions(target_values[indices_to_keep])
    return indices_to_keep
