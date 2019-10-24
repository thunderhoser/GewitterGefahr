"""Methods for downloading files."""

import os
import warnings
import subprocess
import ftplib
# import urllib.request, urllib.error, urllib.parse
import numpy
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.gg_utils import error_checking

NUM_BYTES_PER_BLOCK = 16384
SSH_ARG_STRING = (
    'ssh -o StrictHostKeyChecking=no -o BatchMode=yes -o ConnectTimeout=5')

HTTP_NOT_FOUND_ERROR_CODE = 404
SERVICE_TEMP_UNAVAILABLE_ERROR_CODE = 503
URL_NOT_FOUND_ERROR_CODE = 550
URL_TIMEOUT_ERROR_CODE = 110
FTP_NOT_FOUND_ERROR_CODE = 550

ACCEPTABLE_URL_ERROR_CODES = [URL_NOT_FOUND_ERROR_CODE, URL_TIMEOUT_ERROR_CODE]
ACCEPTABLE_HTTP_ERROR_CODES = [
    HTTP_NOT_FOUND_ERROR_CODE, SERVICE_TEMP_UNAVAILABLE_ERROR_CODE
]


def download_file_via_passwordless_ssh(host_name=None, user_name=None,
                                       remote_file_name=None,
                                       local_file_name=None,
                                       raise_error_if_fails=True):
    """Downloads file via passwordless SSH.

    For this to work, the remote machine (from which you are downloading) must
    have the RSA key of the local machine.  See the following page for
    instructions on sharing RSA keys: http://www.linuxproblem.org/art_9.html

    :param host_name: Name of remote machine (example: "thunderhoser.ou.edu").
    :param user_name: User name on remote machine (example: "thunderhoser").
    :param remote_file_name: File path on remote machine (where the file will be
        downloaded from).
    :param local_file_name: File path on local machine (where the file will be
        stored).
    :param raise_error_if_fails: Boolean flag.  If raise_error_if_fails = True
        and download fails, will raise an error.
    :return: local_file_name: If raise_error_if_fails = False and download
        failed, this will be None.  Otherwise, this will be the same as input.
    :raises: ValueError: if download failed and raise_error_if_fails = True.
    """

    # TODO(thunderhoser): Handle exceptions more intelligently.  Currently, if
    # the download fails, this method does not know why it failed.  If the
    # download failed because the file does not exist, this is less severe than
    # if it failed because we can't login to the remote machine.

    error_checking.assert_is_string(host_name)
    error_checking.assert_is_string(user_name)
    error_checking.assert_is_string(remote_file_name)
    error_checking.assert_is_string(local_file_name)
    error_checking.assert_is_boolean(raise_error_if_fails)

    file_system_utils.mkdir_recursive_if_necessary(file_name=local_file_name)

    unix_command_string = (
        'LD_LIBRARY_PATH= rsync -rv -e "{0:s}" {1:s}@{2:s}:"{3:s}" "{4:s}"'
    ).format(
        SSH_ARG_STRING, user_name, host_name, remote_file_name, local_file_name
    )

    devnull_handle = open(os.devnull, 'w')
    subprocess.call(unix_command_string, shell=True, stdout=devnull_handle,
                    stderr=devnull_handle)

    if not os.path.isfile(local_file_name):
        info_string = (
            'Download failed.  Local file expected at: ' + local_file_name)
        if raise_error_if_fails:
            raise ValueError(info_string)
        else:
            warnings.warn(info_string)
            local_file_name = None

    return local_file_name


def download_files_via_http(
        online_file_names, local_file_names, user_name=None, password=None,
        host_name=None, raise_error_if_fails=True):
    """Downloads files via HTTP.

    N = number of files to download

    :param online_file_names: length-N list of URLs.  Example:
        "https://nomads.ncdc.noaa.gov/data/narr/201212/20121212/
        narr-a_221_20121212_1200_000.grb"
    :param local_file_names: length-N list of target paths on local machine (to
        which files will be downloaded).
    :param user_name: User name on HTTP server.  To login anonymously, leave
        this as None.
    :param password: Password on HTTP server.  To login anonymously, leave
        this as None.
    :param host_name: Host name (base URL name) for HTTP server.  Example:
        "https://nomads.ncdc.noaa.gov"
    :param raise_error_if_fails: Boolean flag.  If True and download fails, this
        method will raise an error.
    :return: local_file_names: Same as input, except that if download failed for
        the [i]th file, local_file_names[i] = None.
    :raises: ValueError: if download failed and raise_error_if_fails = True.
    :raises: urllib2.HTTPError: if download failed for any reason not in
        `ACCEPTABLE_HTTP_ERROR_CODES` or `ACCEPTABLE_URL_ERROR_CODES`.  This
        error will be raised regardless of the flag `raise_error_if_fails`.
    """

    # if not(user_name is None or password is None):
    #     error_checking.assert_is_string(user_name)
    #     error_checking.assert_is_string(password)
    #     error_checking.assert_is_string(host_name)
    #
    #     manager_object = urllib.request.HTTPPasswordMgrWithDefaultRealm()
    #     manager_object.add_password(
    #         realm=None, uri=host_name, user=user_name, passwd=password)
    #
    #     authentication_handler = urllib.request.HTTPBasicAuthHandler(
    #         manager_object)
    #     opener_object = urllib.request.build_opener(authentication_handler)
    #     urllib.request.install_opener(opener_object)

    error_checking.assert_is_string_list(online_file_names)
    error_checking.assert_is_numpy_array(
        numpy.asarray(online_file_names), num_dimensions=1)
    num_files = len(online_file_names)

    error_checking.assert_is_string_list(local_file_names)
    error_checking.assert_is_numpy_array(
        numpy.asarray(online_file_names),
        exact_dimensions=numpy.array([num_files]))

    error_checking.assert_is_boolean(raise_error_if_fails)

    for i in range(num_files):
        continue
        
        # this_download_succeeded = False
        # this_response_object = None
        #
        # try:
        #     this_response_object = urllib.request.urlopen(online_file_names[i])
        #     this_download_succeeded = True
        #
        # except urllib.error.HTTPError as this_error:
        #     if (raise_error_if_fails or
        #             this_error.code not in ACCEPTABLE_HTTP_ERROR_CODES):
        #         raise
        #
        # except urllib.error.URLError as this_error:
        #     error_words = this_error.reason.split()
        #     acceptable_error_flags = numpy.array(
        #         [w in str(ACCEPTABLE_URL_ERROR_CODES) for w in error_words])
        #     if raise_error_if_fails or not numpy.any(acceptable_error_flags):
        #         raise
        #
        # if not this_download_succeeded:
        #     warnings.warn(
        #         'Could not download file: {0:s}'.format(online_file_names[i])
        #     )
        #
        #     local_file_names[i] = None
        #     continue
        #
        # file_system_utils.mkdir_recursive_if_necessary(
        #     file_name=local_file_names[i])
        # with open(local_file_names[i], 'wb') as this_file_handle:
        #     while True:
        #         this_chunk = this_response_object.read(NUM_BYTES_PER_BLOCK)
        #         if not this_chunk:
        #             break
        #         this_file_handle.write(this_chunk)
        #
        # if not os.path.isfile(local_file_names[i]):
        #     error_string = (
        #         'Could not download file.  Local file expected at: "{0:s}"'
        #     ).format(local_file_names[i])
        #
        #     if raise_error_if_fails:
        #         raise ValueError(error_string)
        #
        #     warnings.warn(error_string)
        #     local_file_names[i] = None

    return local_file_names


def download_file_via_ftp(server_name=None, user_name=None, password=None,
                          ftp_file_name=None, local_file_name=None,
                          raise_error_if_fails=True):
    """Downloads file from FTP server.

    :param server_name: Name of FTP server (for example,
        "madis-data.ncep.noaa.gov").
    :param user_name: User name on FTP server (example: "thunderhoser").  If you
        want to login anonymously, leave this as None.
    :param password: Password on FTP server.  If you want to login anonymously,
        leave this as None.
    :param ftp_file_name: File path on FTP server.
    :param local_file_name: File path on local machine (where the file will be
        stored).
    :param raise_error_if_fails: Boolean flag.  If raise_error_if_fails = True
        and download fails, will raise an error.
    :return: local_file_name: If raise_error_if_fails = False and download
        failed, this will be None.  Otherwise, this will be the same as input.
    :raises: ValueError: if download failed and raise_error_if_fails = True.
    :raises: ftplib.error_perm: if download failed for any reason other than
        file not found.
    """

    error_checking.assert_is_string(server_name)
    error_checking.assert_is_string(ftp_file_name)
    error_checking.assert_is_string(local_file_name)
    error_checking.assert_is_boolean(raise_error_if_fails)

    if user_name is None or password is None:
        ftp_object = ftplib.FTP(server_name)
        ftp_object.login()
    else:
        error_checking.assert_is_string(user_name)
        error_checking.assert_is_string(password)
        ftp_object = ftplib.FTP(server_name, user_name, password)

    file_system_utils.mkdir_recursive_if_necessary(file_name=local_file_name)
    local_file_handle = open(local_file_name, 'wb')

    try:
        ftp_object.retrbinary('RETR ' + ftp_file_name, local_file_handle.write,
                              blocksize=NUM_BYTES_PER_BLOCK)
    except ftplib.error_perm as this_error:
        if (raise_error_if_fails or not this_error.message.startswith(
                str(FTP_NOT_FOUND_ERROR_CODE))):
            raise

        warnings.warn(
            'Cannot find file on FTP server.  Expected at: ' + ftp_file_name)
        return None

    if not os.path.isfile(local_file_name):
        info_string = (
            'Download failed.  Local file expected at: ' + local_file_name)
        if raise_error_if_fails:
            raise ValueError(info_string)
        else:
            warnings.warn(info_string)
            local_file_name = None

    return local_file_name
