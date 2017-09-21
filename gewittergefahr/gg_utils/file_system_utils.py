"""Methods for file-system access."""

import os
import os.path


def mkdir_recursive_if_necessary(file_or_directory_name):
    """Creates directory if necessary (i.e., if it doesn't already exist).

    Also creates parent directories if necessary.

    :param file_or_directory_name: Path to local file or directory.  If this is
        a file, the directory part will be extracted.
    """

    directory_name = os.path.dirname(file_or_directory_name)
    if not os.path.isdir(directory_name):
        os.makedirs(directory_name)
