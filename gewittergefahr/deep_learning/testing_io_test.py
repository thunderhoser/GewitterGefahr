"""Unit tests for testing_io.py."""

import unittest
import numpy
from gewittergefahr.deep_learning import testing_io

# The following constants are used to test _find_next_batch.
EXAMPLE_TO_FILE_INDICES = numpy.array(
    [0, 1, 1, 2, 2, 2, 3, 3, 3, 3, 5], dtype=int
)
NUM_EXAMPLES_PER_BATCH = 3

NEXT_EXAMPLE_TO_BATCH_INDICES = {
    0: [0],
    1: [1, 2],
    2: [2],
    3: [3, 4, 5],
    4: [4, 5],
    5: [5],
    6: [6, 7, 8],
    7: [7, 8, 9],
    8: [8, 9],
    9: [9],
    10: [10],
    11: None
}

for k in NEXT_EXAMPLE_TO_BATCH_INDICES:
    if NEXT_EXAMPLE_TO_BATCH_INDICES[k] is None:
        continue

    NEXT_EXAMPLE_TO_BATCH_INDICES[k] = numpy.array(
        NEXT_EXAMPLE_TO_BATCH_INDICES[k], dtype=int
    )


class TestingIoTests(unittest.TestCase):
    """Each method is a unit test for testing_io.py."""

    def test_find_next_batch(self):
        """Ensures correct output from _find_next_batch."""

        for this_key in NEXT_EXAMPLE_TO_BATCH_INDICES:
            actual_batch_indices = testing_io._find_next_batch(
                example_to_file_indices=EXAMPLE_TO_FILE_INDICES,
                num_examples_per_batch=NUM_EXAMPLES_PER_BATCH,
                next_example_index=this_key)

            expected_batch_indices = NEXT_EXAMPLE_TO_BATCH_INDICES[this_key]

            self.assertTrue(numpy.array_equal(
                actual_batch_indices, expected_batch_indices
            ))


if __name__ == '__main__':
    unittest.main()
