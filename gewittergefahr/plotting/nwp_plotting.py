"""Plots gridded NWP (numerical weather prediction) output.

--- DEFINITIONS ---

"Subgrid" = contiguous rectangular subset of the full model grid.  This need not
be a *strict* subset (in other words, the subgrid could be the full grid).

M = number of rows in subgrid (unique y-coordinates at grid points)
N = number of columns in subgrid (unique x-coordinates at grid points)
"""

import numpy
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as pyplot
import matplotlib.colors
from mpl_toolkits.basemap import Basemap
from gewittergefahr.gg_utils import grids
from gewittergefahr.gg_utils import projections
from gewittergefahr.gg_utils import nwp_model_utils
from gewittergefahr.gg_utils import longitude_conversion as lng_conversion
from gewittergefahr.gg_utils import error_checking
from gewittergefahr.plotting import wind_plotting

DEFAULT_FIGURE_WIDTH_INCHES = 15.
DEFAULT_FIGURE_HEIGHT_INCHES = 15.
DEFAULT_BOUNDARY_RESOLUTION_STRING = 'l'

X_COORD_MATRIX_KEY = 'grid_point_x_matrix_metres'
Y_COORD_MATRIX_KEY = 'grid_point_y_matrix_metres'
LATITUDE_MATRIX_KEY = 'grid_point_lat_matrix_deg'
LONGITUDE_MATRIX_KEY = 'grid_point_lng_matrix_deg'


def _get_grid_point_coords(
        model_name, first_row_in_full_grid, last_row_in_full_grid,
        first_column_in_full_grid, last_column_in_full_grid, grid_id=None,
        basemap_object=None):
    """Returns x-y and lat-long coords for a subgrid of the full model grid.

    This method generates different x-y coordinates than
    `nwp_model_utils.get_xy_grid_point_matrices`, because (like
    `mpl_toolkits.basemap.Basemap`) this method sets false easting = false
    northing = 0 metres.

    :param model_name: Name of NWP model (must be accepted by
        `nwp_model_utils.check_grid_name`).
    :param first_row_in_full_grid: Row 0 in the subgrid is row
        `first_row_in_full_grid` in the full grid.
    :param last_row_in_full_grid: Last row in the subgrid is row
        `last_row_in_full_grid` in the full grid.  If you want last row in the
        subgrid to equal last row in the full grid, make this -1.
    :param first_column_in_full_grid: Column 0 in the subgrid is column
        `first_column_in_full_grid` in the full grid.
    :param last_column_in_full_grid: Last column in the subgrid is column
        `last_column_in_full_grid` in the full grid.  If you want last column in
        the subgrid to equal last column in the full grid, make this -1.
    :param grid_id: Grid for NWP model (must be accepted by
        `nwp_model_utils.check_grid_name`).
    :param basemap_object: Instance of `mpl_toolkits.basemap.Basemap` for the
        given NWP model.  If you don't have one, no big deal -- leave this
        argument empty.
    :return: coordinate_dict: Dictionary with the following keys.
    coordinate_dict['grid_point_x_matrix_metres']: M-by-N numpy array of
        x-coordinates.
    coordinate_dict['grid_point_y_matrix_metres']: M-by-N numpy array of
        y-coordinates.
    coordinate_dict['grid_point_lat_matrix_deg']: M-by-N numpy array of
        latitudes (deg N).
    coordinate_dict['grid_point_lng_matrix_deg']: M-by-N numpy array of
        longitudes (deg E).
    """

    num_rows_in_full_grid, num_columns_in_full_grid = (
        nwp_model_utils.get_grid_dimensions(
            model_name=model_name, grid_name=grid_id)
    )

    error_checking.assert_is_integer(first_row_in_full_grid)
    error_checking.assert_is_geq(first_row_in_full_grid, 0)
    error_checking.assert_is_integer(last_row_in_full_grid)
    if last_row_in_full_grid < 0:
        last_row_in_full_grid += num_rows_in_full_grid

    error_checking.assert_is_greater(
        last_row_in_full_grid, first_row_in_full_grid)
    error_checking.assert_is_less_than(
        last_row_in_full_grid, num_rows_in_full_grid)

    error_checking.assert_is_integer(first_column_in_full_grid)
    error_checking.assert_is_geq(first_column_in_full_grid, 0)
    error_checking.assert_is_integer(last_column_in_full_grid)
    if last_column_in_full_grid < 0:
        last_column_in_full_grid += num_columns_in_full_grid

    error_checking.assert_is_greater(
        last_column_in_full_grid, first_column_in_full_grid)
    error_checking.assert_is_less_than(
        last_column_in_full_grid, num_columns_in_full_grid)

    grid_point_lat_matrix_deg, grid_point_lng_matrix_deg = (
        nwp_model_utils.get_latlng_grid_point_matrices(
            model_name=model_name, grid_name=grid_id)
    )

    grid_point_lat_matrix_deg = grid_point_lat_matrix_deg[
        first_row_in_full_grid:(last_row_in_full_grid + 1),
        first_column_in_full_grid:(last_column_in_full_grid + 1)
    ]

    grid_point_lng_matrix_deg = grid_point_lng_matrix_deg[
        first_row_in_full_grid:(last_row_in_full_grid + 1),
        first_column_in_full_grid:(last_column_in_full_grid + 1)
    ]

    if basemap_object is None:
        standard_latitudes_deg, central_longitude_deg = (
            nwp_model_utils.get_projection_params(model_name)
        )

        projection_object = projections.init_lcc_projection(
            standard_latitudes_deg=standard_latitudes_deg,
            central_longitude_deg=central_longitude_deg)

        grid_point_x_matrix_metres, grid_point_y_matrix_metres = (
            projections.project_latlng_to_xy(
                latitudes_deg=grid_point_lat_matrix_deg,
                longitudes_deg=grid_point_lng_matrix_deg,
                projection_object=projection_object, false_northing_metres=0.,
                false_easting_metres=0.)
        )
    else:
        grid_point_x_matrix_metres, grid_point_y_matrix_metres = basemap_object(
            grid_point_lng_matrix_deg, grid_point_lat_matrix_deg)

    return {
        X_COORD_MATRIX_KEY: grid_point_x_matrix_metres,
        Y_COORD_MATRIX_KEY: grid_point_y_matrix_metres,
        LATITUDE_MATRIX_KEY: grid_point_lat_matrix_deg,
        LONGITUDE_MATRIX_KEY: grid_point_lng_matrix_deg,
    }


def latlng_limits_to_rowcol_limits(
        min_latitude_deg, max_latitude_deg, min_longitude_deg,
        max_longitude_deg, model_name, grid_id=None):
    """Converts lat-long limits to row-column limits in the given model grid.

    :param min_latitude_deg: Minimum latitude (deg N).
    :param max_latitude_deg: Max latitude (deg N).
    :param min_longitude_deg: Minimum longitude (deg E).
    :param max_longitude_deg: Max longitude (deg E).
    :param model_name: Name of NWP model (must be accepted by
        `nwp_model_utils.check_grid_name`).
    :param grid_id: Grid for NWP model (must be accepted by
        `nwp_model_utils.check_grid_name`).
    :return: row_limits: length-2 numpy array, containing min and max rows in
        model grid, respectively.
    :return: column_limits: Same but for columns.
    """

    error_checking.assert_is_valid_latitude(min_latitude_deg)
    error_checking.assert_is_valid_latitude(max_latitude_deg)
    error_checking.assert_is_greater(max_latitude_deg, min_latitude_deg)

    both_longitudes_deg = numpy.array([min_longitude_deg, max_longitude_deg])
    both_longitudes_deg = lng_conversion.convert_lng_positive_in_west(
        both_longitudes_deg)

    min_longitude_deg = both_longitudes_deg[0]
    max_longitude_deg = both_longitudes_deg[1]
    error_checking.assert_is_greater(max_longitude_deg, min_longitude_deg)

    grid_point_lat_matrix_deg, grid_point_lng_matrix_deg = (
        nwp_model_utils.get_latlng_grid_point_matrices(
            model_name=model_name, grid_name=grid_id)
    )

    good_lat_flag_matrix = numpy.logical_and(
        grid_point_lat_matrix_deg >= min_latitude_deg,
        grid_point_lat_matrix_deg <= max_latitude_deg
    )
    good_lng_flag_matrix = numpy.logical_and(
        grid_point_lng_matrix_deg >= min_longitude_deg,
        grid_point_lng_matrix_deg <= max_longitude_deg
    )

    good_row_indices, good_column_indices = numpy.where(
        numpy.logical_and(good_lat_flag_matrix, good_lng_flag_matrix)
    )

    row_limits = numpy.array([
        numpy.min(good_row_indices), numpy.max(good_row_indices)
    ], dtype=int)

    column_limits = numpy.array([
        numpy.min(good_column_indices), numpy.max(good_column_indices)
    ], dtype=int)

    return row_limits, column_limits


def init_basemap(
        model_name, grid_id=None,
        figure_width_inches=DEFAULT_FIGURE_WIDTH_INCHES,
        figure_height_inches=DEFAULT_FIGURE_HEIGHT_INCHES,
        resolution_string=DEFAULT_BOUNDARY_RESOLUTION_STRING,
        first_row_in_full_grid=0, last_row_in_full_grid=-1,
        first_column_in_full_grid=0, last_column_in_full_grid=-1):
    """Initializes basemap with the given model's projection.

    :param model_name: Name of NWP model (must be accepted by
        `nwp_model_utils.check_grid_name`).
    :param grid_id: Grid for NWP model (must be accepted by
        `nwp_model_utils.check_grid_name`).
    :param figure_width_inches: Figure width.
    :param figure_height_inches: Figure height.
    :param resolution_string: Resolution for boundaries (e.g., coastlines and
        political borders).  Options are "c" for crude, "l" for low, "i" for
        intermediate, "h" for high, and "f" for full.  Keep in mind that higher-
        resolution boundaries take much longer to draw.
    :param first_row_in_full_grid: See doc for `_get_grid_point_coords`.
    :param last_row_in_full_grid: Same.
    :param first_column_in_full_grid: Same.
    :param last_column_in_full_grid: Same.
    :return: figure_object: Instance of `matplotlib.figure.Figure`.
    :return: axes_object: Instance of `matplotlib.axes._subplots.AxesSubplot`.
    :return: basemap_object: Instance of `mpl_toolkits.basemap.Basemap`.
    """

    error_checking.assert_is_greater(figure_width_inches, 0.)
    error_checking.assert_is_greater(figure_height_inches, 0.)
    error_checking.assert_is_string(resolution_string)

    coordinate_dict = _get_grid_point_coords(
        model_name=model_name, grid_id=grid_id,
        first_row_in_full_grid=first_row_in_full_grid,
        last_row_in_full_grid=last_row_in_full_grid,
        first_column_in_full_grid=first_column_in_full_grid,
        last_column_in_full_grid=last_column_in_full_grid)

    grid_point_x_matrix_metres = coordinate_dict[X_COORD_MATRIX_KEY]
    grid_point_y_matrix_metres = coordinate_dict[Y_COORD_MATRIX_KEY]

    figure_object, axes_object = pyplot.subplots(
        1, 1, figsize=(figure_width_inches, figure_height_inches))

    standard_latitudes_deg, central_longitude_deg = (
        nwp_model_utils.get_projection_params(model_name)
    )

    basemap_object = Basemap(
        projection='lcc', lat_1=standard_latitudes_deg[0],
        lat_2=standard_latitudes_deg[1], lon_0=central_longitude_deg,
        rsphere=projections.DEFAULT_EARTH_RADIUS_METRES, ellps='sphere',
        resolution=resolution_string,
        llcrnrx=grid_point_x_matrix_metres[0, 0],
        llcrnry=grid_point_y_matrix_metres[0, 0],
        urcrnrx=grid_point_x_matrix_metres[-1, -1],
        urcrnry=grid_point_y_matrix_metres[-1, -1]
    )

    return figure_object, axes_object, basemap_object


def plot_subgrid(
        field_matrix, model_name, axes_object, basemap_object,
        colour_map_object, colour_norm_object=None, min_colour_value=None,
        max_colour_value=None, grid_id=None,
        first_row_in_full_grid=0, first_column_in_full_grid=0, opacity=1.):
    """Plots colour map on subset of the full model grid.

    M = number of rows in subgrid
    N = number of columns in subgrid

    If `colour_norm_object is None`, both `min_colour_value` and
    `max_colour_value` must be specified.

    :param field_matrix: M-by-N numpy array of data values.
    :param model_name: See doc for `_get_grid_point_coords`.
    :param axes_object: Instance of `matplotlib.axes._subplots.AxesSubplot`.
        Will plot on these axes.
    :param basemap_object: Instance of `mpl_toolkits.basemap.Basemap`.  Will be
        used to convert between x-y and lat-long coordinates.
    :param colour_map_object: Instance of `matplotlib.pyplot.cm`.  Determines
        colours in scheme.
    :param colour_norm_object: Instance of `matplotlib.colors.BoundaryNorm`.
        Determines boundaries in colour scheme.
    :param min_colour_value: [used only if `colour_norm_object is None`]
        Minimum data value in colour scheme.
    :param max_colour_value: [used only if `colour_norm_object is None`]
        Max data value in colour scheme.
    :param grid_id: See doc for `_get_grid_point_coords`.
    :param first_row_in_full_grid: Row offset.  field_matrix[0, 0] is at row m
        in the full model grid, where m = `first_row_in_full_grid`.
    :param first_column_in_full_grid: Same but for columns.
    :param opacity: Opacity of colour map (in range 0...1).
    """

    if colour_norm_object is None:
        error_checking.assert_is_greater(max_colour_value, min_colour_value)

        colour_norm_object = matplotlib.colors.Normalize(
            vmin=min_colour_value, vmax=max_colour_value, clip=False)

    error_checking.assert_is_real_numpy_array(field_matrix)
    error_checking.assert_is_numpy_array(field_matrix, num_dimensions=2)

    num_rows_in_subgrid = field_matrix.shape[0]
    num_columns_in_subgrid = field_matrix.shape[1]

    coordinate_dict = _get_grid_point_coords(
        model_name=model_name, grid_id=grid_id,
        first_row_in_full_grid=first_row_in_full_grid,
        last_row_in_full_grid=first_row_in_full_grid + num_rows_in_subgrid - 1,
        first_column_in_full_grid=first_column_in_full_grid,
        last_column_in_full_grid=
        first_column_in_full_grid + num_columns_in_subgrid - 1,
        basemap_object=basemap_object
    )

    grid_point_x_matrix_metres = coordinate_dict[X_COORD_MATRIX_KEY]
    grid_point_y_matrix_metres = coordinate_dict[Y_COORD_MATRIX_KEY]

    x_spacing_metres = (
        (grid_point_x_matrix_metres[0, -1] - grid_point_x_matrix_metres[0, 0]) /
        (num_columns_in_subgrid - 1)
    )

    y_spacing_metres = (
        (grid_point_y_matrix_metres[-1, 0] - grid_point_y_matrix_metres[0, 0]) /
        (num_rows_in_subgrid - 1)
    )

    (field_matrix_at_edges, grid_cell_edges_x_metres, grid_cell_edges_y_metres
    ) = grids.xy_field_grid_points_to_edges(
        field_matrix=field_matrix,
        x_min_metres=grid_point_x_matrix_metres[0, 0],
        y_min_metres=grid_point_y_matrix_metres[0, 0],
        x_spacing_metres=x_spacing_metres, y_spacing_metres=y_spacing_metres)

    field_matrix_at_edges = numpy.ma.masked_where(
        numpy.isnan(field_matrix_at_edges), field_matrix_at_edges
    )

    if hasattr(colour_norm_object, 'boundaries'):
        min_colour_value = colour_norm_object.boundaries[0]
        max_colour_value = colour_norm_object.boundaries[-1]
    else:
        min_colour_value = colour_norm_object.vmin
        max_colour_value = colour_norm_object.vmax

    basemap_object.pcolormesh(
        grid_cell_edges_x_metres, grid_cell_edges_y_metres,
        field_matrix_at_edges, cmap=colour_map_object, norm=colour_norm_object,
        vmin=min_colour_value, vmax=max_colour_value, shading='flat',
        edgecolors='None', axes=axes_object, zorder=-1e9, alpha=opacity)


def plot_wind_barbs_on_subgrid(
        u_wind_matrix_m_s01, v_wind_matrix_m_s01, model_name, axes_object,
        basemap_object, grid_id=None, first_row_in_full_grid=0,
        first_column_in_full_grid=0, plot_every_k_rows=1,
        plot_every_k_columns=1, barb_length=wind_plotting.DEFAULT_BARB_LENGTH,
        empty_barb_radius=wind_plotting.DEFAULT_EMPTY_BARB_RADIUS,
        fill_empty_barb=wind_plotting.FILL_EMPTY_BARB_DEFAULT,
        colour_map=wind_plotting.DEFAULT_COLOUR_MAP,
        colour_minimum_kt=wind_plotting.DEFAULT_COLOUR_MINIMUM_KT,
        colour_maximum_kt=wind_plotting.DEFAULT_COLOUR_MAXIMUM_KT):
    """Plots wind barbs over subgrid.

    :param u_wind_matrix_m_s01: M-by-N numpy array of zonal wind speeds (metres
        per second).
    :param v_wind_matrix_m_s01: M-by-N numpy array of meridional wind speeds
        (metres per second).
    :param model_name: See doc for `plot_subgrid`.
    :param axes_object: Same.
    :param basemap_object: Same.
    :param grid_id: Same.
    :param first_row_in_full_grid: Row 0 in the subgrid (i.e., row 0 in
        `u_wind_matrix_m_s01` and `v_wind_matrix_m_s01` is row
        `first_row_in_full_grid` in the full grid).
    :param first_column_in_full_grid: Column 0 in the subgrid (i.e., column 0 in
        `u_wind_matrix_m_s01` and `v_wind_matrix_m_s01` is column
        `first_column_in_full_grid` in the full grid).
    :param plot_every_k_rows: Wind barbs will be plotted for every [k]th row in
        the subgrid, where k = `plot_every_k_rows`.  For example, if
        `plot_every_k_rows = 2`, wind barbs will be plotted for rows 0, 2, 4,
        etc.
    :param plot_every_k_columns: Same as above, but for columns.
    :param barb_length: See doc for `wind_plotting.plot_wind_barbs`.
    :param empty_barb_radius: Same.
    :param fill_empty_barb: Same.
    :param colour_map: Same.
    :param colour_minimum_kt: Same.
    :param colour_maximum_kt: Same.
    """

    error_checking.assert_is_real_numpy_array(u_wind_matrix_m_s01)
    error_checking.assert_is_numpy_array(u_wind_matrix_m_s01, num_dimensions=2)
    error_checking.assert_is_real_numpy_array(v_wind_matrix_m_s01)

    these_expected_dim = numpy.array(u_wind_matrix_m_s01.shape, dtype=int)
    error_checking.assert_is_numpy_array(
        v_wind_matrix_m_s01, exact_dimensions=these_expected_dim)

    error_checking.assert_is_integer(plot_every_k_rows)
    error_checking.assert_is_geq(plot_every_k_rows, 1)
    error_checking.assert_is_integer(plot_every_k_columns)
    error_checking.assert_is_geq(plot_every_k_columns, 1)

    num_rows_in_subgrid = u_wind_matrix_m_s01.shape[0]
    num_columns_in_subgrid = u_wind_matrix_m_s01.shape[1]

    coordinate_dict = _get_grid_point_coords(
        model_name=model_name, grid_id=grid_id,
        first_row_in_full_grid=first_row_in_full_grid,
        last_row_in_full_grid=first_row_in_full_grid + num_rows_in_subgrid - 1,
        first_column_in_full_grid=first_column_in_full_grid,
        last_column_in_full_grid=
        first_column_in_full_grid + num_columns_in_subgrid - 1,
        basemap_object=basemap_object
    )

    grid_point_lat_matrix_deg = coordinate_dict[LATITUDE_MATRIX_KEY]
    grid_point_lng_matrix_deg = coordinate_dict[LONGITUDE_MATRIX_KEY]

    u_wind_matrix_m_s01 = u_wind_matrix_m_s01[
        ::plot_every_k_rows, ::plot_every_k_columns
    ]
    v_wind_matrix_m_s01 = v_wind_matrix_m_s01[
        ::plot_every_k_rows, ::plot_every_k_columns
    ]
    grid_point_lat_matrix_deg = grid_point_lat_matrix_deg[
        ::plot_every_k_rows, ::plot_every_k_columns
    ]
    grid_point_lng_matrix_deg = grid_point_lng_matrix_deg[
        ::plot_every_k_rows, ::plot_every_k_columns
    ]

    num_wind_barbs = u_wind_matrix_m_s01.size
    u_winds_m_s01 = numpy.reshape(u_wind_matrix_m_s01, num_wind_barbs)
    v_winds_m_s01 = numpy.reshape(v_wind_matrix_m_s01, num_wind_barbs)
    latitudes_deg = numpy.reshape(grid_point_lat_matrix_deg, num_wind_barbs)
    longitudes_deg = numpy.reshape(grid_point_lng_matrix_deg, num_wind_barbs)

    nan_flags = numpy.logical_or(
        numpy.isnan(u_winds_m_s01), numpy.isnan(v_winds_m_s01)
    )
    real_indices = numpy.where(numpy.invert(nan_flags))[0]

    wind_plotting.plot_wind_barbs(
        basemap_object=basemap_object, axes_object=axes_object,
        latitudes_deg=latitudes_deg[real_indices],
        longitudes_deg=longitudes_deg[real_indices],
        u_winds_m_s01=u_winds_m_s01[real_indices],
        v_winds_m_s01=v_winds_m_s01[real_indices],
        barb_length=barb_length, empty_barb_radius=empty_barb_radius,
        fill_empty_barb=fill_empty_barb, colour_map=colour_map,
        colour_minimum_kt=colour_minimum_kt,
        colour_maximum_kt=colour_maximum_kt)
