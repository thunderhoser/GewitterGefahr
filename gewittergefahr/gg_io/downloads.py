"""Methods for downloading files."""

import os.path
from ftplib import FTP
from urllib2 import urlopen
from gewittergefahr.gg_utils import file_system_utils

NUM_BYTES_PER_DOWNLOAD_CHUNK = 16384


def download_file_from_url(file_url, local_file_name,
                           raise_error_if_fails=True):
    """Downloads file from URL.

    :param file_url: [string] File will be downloaded from here.
    :param local_file_name: [string] File will be saved to this path on local
        machine.
    :param raise_error_if_fails: [Boolean] If True and download fails, this
        method will raise an error.
    :return: local_file_name: Path to file on local machine.  If download failed
        but raise_error_if_fails = False, this will be None.
    :raises: ValueError: if raise_error_if_fails = True and download failed.
    """

    response_object = urlopen(file_url)
    file_system_utils.mkdir_recursive_if_necessary(file_name=local_file_name)

    with open(local_file_name, 'wb') as local_file_handle:
        while True:
            this_chunk = response_object.read(NUM_BYTES_PER_DOWNLOAD_CHUNK)
            if not this_chunk:
                break
            local_file_handle.write(this_chunk)

    if raise_error_if_fails and not os.path.isfile(local_file_name):
        raise ValueError(
            'Download failed.  Local file expected at: ' + local_file_name)

    if not os.path.isfile(local_file_name):
        local_file_name = None

    return local_file_name


def download_file_from_ftp(server_name=None, user_name=None, password=None,
                           ftp_file_name=None, local_file_name=None,
                           raise_error_if_fails=True):
    """Downloads file from FTP server.

    :param server_name: [string] Name of FTP server.
    :param user_name: [string] Username on FTP server.  If you want to login
        anonymously, leave this as None.
    :param password: [string] Password on FTP server.  If you want to login
        anonymously, leave this as None.
    :param ftp_file_name: [string] File path on FTP server.
    :param local_file_name: [string] File will be saved to this path on local
        machine.
    :param raise_error_if_fails: [Boolean] If True and download fails, this
        method will raise an error.
    :return: local_file_name: Path to file on local machine.  If download failed
        but raise_error_if_fails = False, this will be None.
    :raises: ValueError: if raise_error_if_fails = True and download failed.
    """

    if user_name is None or password is None:
        ftp_object = FTP(server_name)
        ftp_object.login()
    else:
        ftp_object = FTP(server_name, user_name, password)

    file_system_utils.mkdir_recursive_if_necessary(file_name=local_file_name)
    local_file_handle = open(local_file_name, 'wb')
    ftp_object.retrbinary('RETR ' + ftp_file_name, local_file_handle.write,
                          blocksize=NUM_BYTES_PER_DOWNLOAD_CHUNK)

    if raise_error_if_fails and not os.path.isfile(local_file_name):
        raise ValueError(
            'Download failed.  Local file expected at: ' + local_file_name)

    if not os.path.isfile(local_file_name):
        local_file_name = None

    return local_file_name
