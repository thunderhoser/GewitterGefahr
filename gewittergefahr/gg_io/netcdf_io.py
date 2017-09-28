"""IO methods for NetCDF files."""

from netCDF4 import Dataset
from gewittergefahr.gg_utils import error_checking


def open_netcdf(netcdf_file_name, raise_error_if_fails=False):
    """Attempts to open NetCDF file.

    :param netcdf_file_name: Path to input file.
    :param raise_error_if_fails: Boolean flag.  If raise_error_if_fails = True
        and file cannot be opened, this method will throw an error.
    :return: netcdf_dataset: Instance of `NetCDF4.Dataset`, containing all data
        from the file.  If raise_error_if_fails = False and file could not be
        opened, this will be None.
    :raises: IOError: if file could not be opened and raise_error_if_fails =
        True.
    """

    error_checking.assert_file_exists(netcdf_file_name)
    error_checking.assert_is_boolean(raise_error_if_fails)

    try:
        netcdf_dataset = Dataset(netcdf_file_name)
    except IOError:
        if raise_error_if_fails:
            raise

        netcdf_dataset = None

    return netcdf_dataset
