"""Very fancy downsampling."""

import numpy
from gewittergefahr.gg_utils import labels
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

    print '\n'
    for k in range(len(unique_target_values)):
        print '{0:d} examples in class = {1:d}'.format(
            unique_counts[k], unique_target_values[k])
    print '\n'


def _find_storm_cells(storm_id_by_object, desired_storm_cell_ids):
    """Finds storm IDs from set 2 in set 1.

    N = number of storm objects
    n = number of desired storm cells

    :param storm_id_by_object: length-N list of storm IDs (strings).
    :param desired_storm_cell_ids: length-n list of storm IDs (strings).
    :return: relevant_indices: 1-D numpy array of indices, such that
        `storm_id_by_object[relevant_indices]` yields all IDs in
        `storm_id_by_object` that are in `desired_storm_cell_ids`, including
        duplicates.
    :raises: ValueError: if not all desired ID were found in
        `storm_id_by_object`.
    """

    desired_storm_cell_ids = numpy.unique(numpy.array(desired_storm_cell_ids))

    relevant_flags = numpy.in1d(
        numpy.array(storm_id_by_object), desired_storm_cell_ids,
        assume_unique=False)
    relevant_indices = numpy.where(relevant_flags)[0]

    found_storm_ids = numpy.unique(
        numpy.array(storm_id_by_object)[relevant_indices])
    if numpy.array_equal(found_storm_ids, desired_storm_cell_ids):
        return relevant_indices

    error_string = (
        '\nDesired storm IDs:\n{0:s}\nFound storm IDs:\n{1:s}\nNot all desired '
        'storm IDs were found, as shown above.'
    ).format(str(desired_storm_cell_ids), str(found_storm_ids))
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


def downsample(storm_ids, storm_times_unix_sec, target_values, target_name,
               class_fraction_dict):
    """Does very fancy downsampling.

    This is effectively the "main method" of this file.  The downsampling
    procedure is summarized below.

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
    n = number of storm objects after downsampling

    :param storm_ids: length-N list of storm IDs (strings).
    :param storm_times_unix_sec: length-N numpy array of corresponding times.
    :param target_values: length-N numpy array of corresponding target values
        (integer class labels).
    :param target_name: Name of target variable (must be accepted by
        `labels.column_name_to_label_params`).
    :param class_fraction_dict: Dictionary, where each key is an integer class
        label (-2 for "dead storm") and the corresponding value is the
        sampling fraction.
    :return: storm_ids: length-n list of storm IDs (strings).
    :return: storm_times_unix_sec: length-n numpy array of corresponding times.
    :return: target_values: length-n numpy array of corresponding target values.
    """

    _report_class_fractions(target_values)

    num_storm_objects = len(storm_ids)
    num_classes = labels.column_name_to_num_classes(
        column_name=target_name, include_dead_storms=False)

    # Step 1.
    print (
        'Finding storm objects in class {0:d} (the highest class), yielding set'
        ' {{s_highest}}...'
    ).format(num_classes - 1)

    highest_class_indices = numpy.where(target_values == num_classes - 1)[0]

    print '{{s_highest}} contains {0:d} of {1:d} storm objects.'.format(
        len(highest_class_indices), num_storm_objects)

    # Step 2.
    print ('Finding storm cells with at least one object in {{s_highest}}, '
           'yielding set {{S_highest}}...')
    highest_class_indices = _find_storm_cells(
        storm_id_by_object=storm_ids,
        desired_storm_cell_ids=[storm_ids[k] for k in highest_class_indices])

    print '{{S_highest}} contains {0:d} of {1:d} storm objects.'.format(
        len(highest_class_indices), num_storm_objects)

    # Step 3.
    print ('Finding all time steps with at least one storm cell in '
           '{{S_highest}}, yielding set {{t_highest}}...')

    lower_class_times_unix_sec = (
        set(storm_times_unix_sec.tolist()) -
        set(storm_times_unix_sec[highest_class_indices].tolist())
    )
    lower_class_times_unix_sec = numpy.array(
        list(lower_class_times_unix_sec), dtype=int)

    # Step 4.
    print 'Randomly removing {0:.1f}% of times not in {{t_highest}}...'.format(
        FRACTION_UNINTERESTING_TIMES_TO_OMIT * 100)

    this_num_times = int(numpy.round(
        FRACTION_UNINTERESTING_TIMES_TO_OMIT * len(lower_class_times_unix_sec)
    ))
    times_to_remove_unix_sec = numpy.random.choice(
        lower_class_times_unix_sec, size=this_num_times, replace=False)

    indices_to_keep = _find_uncovered_times(
        all_times_unix_sec=storm_times_unix_sec,
        covered_times_unix_sec=times_to_remove_unix_sec)

    storm_ids = [storm_ids[k] for k in indices_to_keep]
    storm_times_unix_sec = storm_times_unix_sec[indices_to_keep]
    target_values = target_values[indices_to_keep]

    _report_class_fractions(target_values)

    # Step 5.
    print 'Downsampling storm objects from remaining times...'
    indices_to_keep = dl_utils.sample_by_class(
        sampling_fraction_by_class_dict=class_fraction_dict,
        target_name=target_name, target_values=target_values,
        num_examples_total=LARGE_INTEGER)

    storm_ids = [storm_ids[k] for k in indices_to_keep]
    storm_times_unix_sec = storm_times_unix_sec[indices_to_keep]
    target_values = target_values[indices_to_keep]

    _report_class_fractions(target_values)

    return storm_ids, storm_times_unix_sec, target_values
