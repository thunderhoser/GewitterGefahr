"""Deals with boundary of continental United States (CONUS)."""

import os.path
import shapefile
import numpy
import netCDF4
import shapely.geometry
from shapely.ops import cascaded_union
from gewittergefahr.gg_utils import longitude_conversion as lng_conversion
from gewittergefahr.gg_utils import polygons
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.gg_utils import error_checking

SHAPEFILE_NAME_KEY = 'NAME'
SHAPEFILE_PERIMETER_KEY = 'PERIMETER'
STATES_TO_EXCLUDE = ['Alaska', 'Hawaii', 'Puerto Rico']

NETCDF_VERTEX_DIMENSION_KEY = 'vertex'
NETCDF_LATITUDES_KEY = 'latitudes_deg'
NETCDF_LONGITUDES_KEY = 'longitudes_deg'

SHORTCUT_BOX_LATITUDES_DEG = numpy.array([34, 41], dtype=float)
SHORTCUT_BOX_LONGITUDES_DEG = numpy.array([245, 276], dtype=float)


def _check_boundary(latitudes_deg, longitudes_deg):
    """Error-checks boundary.

    V = number of vertices in boundary

    :param latitudes_deg: length-V numpy array of latitudes (deg N).
    :param longitudes_deg: length-V numpy array of longitudes (deg E).
    :return: longitudes_deg: Same as input but positive in western hemisphere.
    """

    longitudes_deg = lng_conversion.convert_lng_positive_in_west(
        longitudes_deg=longitudes_deg, allow_nan=False
    )
    error_checking.assert_is_valid_lat_numpy_array(
        latitudes_deg=latitudes_deg, allow_nan=False
    )

    num_vertices = len(latitudes_deg)
    expected_dim = numpy.array([num_vertices], dtype=int)
    error_checking.assert_is_numpy_array(
        longitudes_deg, exact_dimensions=expected_dim
    )

    return longitudes_deg


def read_from_shapefile(shapefile_name):
    """Reads boundary from shapefile.

    :param shapefile_name: Path to input file.
    :return: latitudes_deg: See doc for `_check_boundary`.
    :return: longitudes_deg: Same.
    """

    print('Reading data from: "{0:s}"...'.format(shapefile_name))
    shapefile_handle = shapefile.Reader(shapefile_name)

    state_names = []
    perimeters = []
    polygon_objects_latlng_deg = []

    for this_record_object in shapefile_handle.iterShapeRecords():
        this_record_dict = this_record_object.record.as_dict()
        this_state_name = this_record_dict[SHAPEFILE_NAME_KEY]

        # Skip states outside the continental U.S.
        if this_state_name in STATES_TO_EXCLUDE:
            continue

        # Find perimeter of this region.  Each region is a simple polygon, so
        # some states, like those with islands, are defined by multiple regions.
        # However, I don't want to keep the islands, because I want only the
        # continental U.S.
        #
        # (Okay, a few islands are in the Great Lakes instead of the ocean, so I
        # should technically keep them, but they're small enough that I don't
        # care).
        #
        # In general, for states with multiple polygons, the one with the
        # largest perimeter is the mainland part.  The only exception is
        # Michigan, which ruins everything by having the UP.  So for Michigan I
        # keep the two largest regions, but for every other state I keep the one
        # largest region.  For Michigan the two largest regions are those with a
        # perimeter > 5 (I don't know what the units are, but it doesn't really
        # matter).

        this_perimeter = this_record_dict[SHAPEFILE_PERIMETER_KEY]

        if this_state_name in state_names:
            i = state_names.index(this_state_name)

            # If this region is the largest for the given state, keep it.
            keep_region = this_perimeter > perimeters[i]

            # Deal with Michigan (which ruins everything).
            in_mainland_michigan = (
                this_state_name == 'Michigan' and this_perimeter > 5
            )
            keep_region = keep_region or in_mainland_michigan

            if not keep_region:
                continue

            # If the state is not Michigan, delete previous largest region for
            # state.
            delete_old_region = (
                not in_mainland_michigan or
                (in_mainland_michigan and perimeters[i] < 5)
            )

            if delete_old_region:
                del state_names[i]
                del perimeters[i]
                del polygon_objects_latlng_deg[i]

        state_names.append(this_state_name)
        perimeters.append(this_perimeter)
        polygon_objects_latlng_deg.append(
            shapely.geometry.Polygon(shell=this_record_object.shape.points)
        )

    # Merge polygons into one.
    main_polygon_object_latlng_deg = cascaded_union(polygon_objects_latlng_deg)
    latitudes_deg = numpy.array(
        main_polygon_object_latlng_deg.exterior.xy[1]
    )
    longitudes_deg = numpy.array(
        main_polygon_object_latlng_deg.exterior.xy[0]
    )

    longitudes_deg = _check_boundary(
        latitudes_deg=latitudes_deg, longitudes_deg=longitudes_deg)

    return latitudes_deg, longitudes_deg


def write_to_netcdf(latitudes_deg, longitudes_deg, netcdf_file_name):
    """Writes boundary to NetCDF file.

    :param latitudes_deg: See doc for `_check_boundary`.
    :param longitudes_deg: Same.
    :param netcdf_file_name: Path to output file.
    """

    longitudes_deg = _check_boundary(
        latitudes_deg=latitudes_deg, longitudes_deg=longitudes_deg)

    file_system_utils.mkdir_recursive_if_necessary(file_name=netcdf_file_name)
    dataset_object = netCDF4.Dataset(
        netcdf_file_name, 'w', format='NETCDF3_64BIT_OFFSET'
    )

    num_points = len(latitudes_deg)
    dataset_object.createDimension(NETCDF_VERTEX_DIMENSION_KEY, num_points)

    dataset_object.createVariable(
        NETCDF_LATITUDES_KEY, datatype=numpy.float32,
        dimensions=NETCDF_VERTEX_DIMENSION_KEY
    )
    dataset_object.variables[NETCDF_LATITUDES_KEY][:] = latitudes_deg

    dataset_object.createVariable(
        NETCDF_LONGITUDES_KEY, datatype=numpy.float32,
        dimensions=NETCDF_VERTEX_DIMENSION_KEY
    )
    dataset_object.variables[NETCDF_LONGITUDES_KEY][:] = longitudes_deg

    dataset_object.close()


def read_from_netcdf(netcdf_file_name=None):
    """Reads boundary from NetCDF file.

    :param netcdf_file_name: Path to input file.  If None, will look for file in
        repository.
    :return: latitudes_deg: See doc for `_check_boundary`.
    :return: longitudes_deg: Same.
    """

    if netcdf_file_name is None:
        module_dir_name = os.path.dirname(__file__)
        parent_dir_name = '/'.join(module_dir_name.split('/')[:-1])
        netcdf_file_name = '{0:s}/conus_polygon.nc'.format(parent_dir_name)

    dataset_object = netCDF4.Dataset(netcdf_file_name)
    latitudes_deg = numpy.array(
        dataset_object.variables[NETCDF_LATITUDES_KEY][:]
    )
    longitudes_deg = numpy.array(
        dataset_object.variables[NETCDF_LONGITUDES_KEY][:]
    )
    dataset_object.close()

    longitudes_deg = _check_boundary(
        latitudes_deg=latitudes_deg, longitudes_deg=longitudes_deg
    )
    return latitudes_deg, longitudes_deg


def erode_boundary(latitudes_deg, longitudes_deg, erosion_distance_metres):
    """Erodes boundary.

    Erosion is the same thing as applying a negative buffer distance.  The new
    boundary will be contained inside the old boundary.

    :param latitudes_deg: See doc for `_check_boundary`.
    :param longitudes_deg: Same.
    :param erosion_distance_metres: Erosion distance.
    :return: latitudes_deg: Eroded version of input.
    :return: longitudes_deg: Eroded version of input.
    """

    longitudes_deg = _check_boundary(
        latitudes_deg=latitudes_deg, longitudes_deg=longitudes_deg
    )
    # error_checking.assert_is_greater(erosion_distance_metres, 0.)

    polygon_object_latlng = polygons.vertex_arrays_to_polygon_object(
        exterior_x_coords=longitudes_deg, exterior_y_coords=latitudes_deg
    )
    polygon_object_xy, projection_object = polygons.project_latlng_to_xy(
        polygon_object_latlng=polygon_object_latlng
    )
    polygon_object_xy = polygon_object_xy.buffer(
        -erosion_distance_metres, join_style=shapely.geometry.JOIN_STYLE.round
    )

    if 'MultiPolygon' in str(type(polygon_object_xy)):
        polygon_object_xy = list(polygon_object_xy)[0]

    polygon_object_latlng = polygons.project_xy_to_latlng(
        polygon_object_xy_metres=polygon_object_xy,
        projection_object=projection_object
    )
    polygon_dict_latlng = polygons.polygon_object_to_vertex_arrays(
        polygon_object_latlng
    )

    latitudes_deg = polygon_dict_latlng[polygons.EXTERIOR_Y_COLUMN]
    longitudes_deg = polygon_dict_latlng[polygons.EXTERIOR_X_COLUMN]
    longitudes_deg = _check_boundary(
        latitudes_deg=latitudes_deg, longitudes_deg=longitudes_deg
    )

    return latitudes_deg, longitudes_deg


def find_points_in_conus(
        conus_latitudes_deg, conus_longitudes_deg, query_latitudes_deg,
        query_longitudes_deg, use_shortcuts=True):
    """Finds points in CONUS.

    Q = number of query points

    This method assumes that the domain doesn't wrap around 0 deg E (Greenwich).

    If you set `use_shortcuts = True`, this method will assume that input
    coordinates `conus_latitudes_deg` and `conus_longitudes_deg` have been
    eroded by less than 100 km.

    :param conus_latitudes_deg: See doc for `_check_boundary`.
    :param conus_longitudes_deg: Same.
    :param query_latitudes_deg: length-Q numpy with latitudes (deg N) of query
        points.
    :param query_longitudes_deg: length-Q numpy with longitudes (deg E) of query
        points.
    :param use_shortcuts: Boolean flag.  If True, will use shortcuts to speed up
        calculation.
    :return: in_conus_flags: length-Q numpy array of Boolean flags.
    """

    conus_longitudes_deg = _check_boundary(
        latitudes_deg=conus_latitudes_deg, longitudes_deg=conus_longitudes_deg
    )
    query_longitudes_deg = _check_boundary(
        latitudes_deg=query_latitudes_deg, longitudes_deg=query_longitudes_deg
    )
    error_checking.assert_is_boolean(use_shortcuts)

    num_query_points = len(query_latitudes_deg)
    in_conus_flags = numpy.full(num_query_points, -1, dtype=int)

    if use_shortcuts:

        # Use rectangle.
        latitude_flags = numpy.logical_and(
            query_latitudes_deg >= SHORTCUT_BOX_LATITUDES_DEG[0],
            query_latitudes_deg <= SHORTCUT_BOX_LATITUDES_DEG[1]
        )
        longitude_flags = numpy.logical_and(
            query_longitudes_deg >= SHORTCUT_BOX_LONGITUDES_DEG[0],
            query_longitudes_deg <= SHORTCUT_BOX_LONGITUDES_DEG[1]
        )
        in_conus_flags[numpy.logical_and(latitude_flags, longitude_flags)] = 1

        # Use simplified eroded boundary.
        module_dir_name = os.path.dirname(__file__)
        parent_dir_name = '/'.join(module_dir_name.split('/')[:-1])
        inner_boundary_file_name = (
            '{0:s}/conus_polygon_100-km-eroded.nc'.format(parent_dir_name)
        )

        inner_conus_latitudes_deg, inner_conus_longitudes_deg = (
            read_from_netcdf(inner_boundary_file_name)
        )
        trial_indices = numpy.where(in_conus_flags == -1)[0]

        these_flags = find_points_in_conus(
            conus_latitudes_deg=inner_conus_latitudes_deg,
            conus_longitudes_deg=inner_conus_longitudes_deg,
            query_latitudes_deg=query_latitudes_deg[trial_indices],
            query_longitudes_deg=query_longitudes_deg[trial_indices],
            use_shortcuts=False
        )
        these_indices = trial_indices[numpy.where(these_flags)]
        in_conus_flags[these_indices] = 1

        # Use simplified dilated boundary.
        module_dir_name = os.path.dirname(__file__)
        parent_dir_name = '/'.join(module_dir_name.split('/')[:-1])
        outer_boundary_file_name = (
            '{0:s}/conus_polygon_100-km-dilated.nc'.format(parent_dir_name)
        )

        outer_conus_latitudes_deg, outer_conus_longitudes_deg = (
            read_from_netcdf(outer_boundary_file_name)
        )
        trial_indices = numpy.where(in_conus_flags == -1)[0]

        these_flags = find_points_in_conus(
            conus_latitudes_deg=outer_conus_latitudes_deg,
            conus_longitudes_deg=outer_conus_longitudes_deg,
            query_latitudes_deg=query_latitudes_deg[trial_indices],
            query_longitudes_deg=query_longitudes_deg[trial_indices],
            use_shortcuts=False
        )
        these_indices = trial_indices[numpy.where(numpy.invert(these_flags))]
        in_conus_flags[these_indices] = 0

    conus_polygon_object = polygons.vertex_arrays_to_polygon_object(
        exterior_x_coords=conus_longitudes_deg,
        exterior_y_coords=conus_latitudes_deg)

    for i in range(num_query_points):
        if numpy.mod(i, 1000) == 0:
            print((
                'Have done point-in-CONUS test for {0:d} of {1:d} points...'
            ).format(
                i, num_query_points
            ))

        if in_conus_flags[i] != -1:
            continue

        in_conus_flags[i] = polygons.point_in_or_on_polygon(
            polygon_object=conus_polygon_object,
            query_x_coordinate=query_longitudes_deg[i],
            query_y_coordinate=query_latitudes_deg[i]
        )

    print('Have done point-in-CONUS test for all {0:d} points!'.format(
        num_query_points
    ))

    return in_conus_flags.astype(bool)
