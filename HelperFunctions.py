import numpy as np
from random import randrange
from sklearn.neighbors import BallTree


def get_nearest(src_points, candidates, k_neighbors=1):
    """Find nearest neighbors for all source points from a set of candidate points"""

    # Create tree from the candidate points
    tree = BallTree(candidates, leaf_size=15, metric='haversine')

    # Find closest points and distances
    _, indices = tree.query(src_points, k=k_neighbors)

    # Transpose to get distances and indices into arrays
    indices = indices.transpose()

    # Get closest indices and distances (i.e. array at index :k_neighbors)
    closest = indices[:k_neighbors]

    # Return indices in matrix with shape [k_neighbors, n_points]
    return closest


def nearest_neighbor(left_gdf, right_gdf, n_neighbors_sampling=1, index=False):
    """
    For each point in left_gdf, find the n_neighbors_sampling closest points in
    right GeoDataFrame and return a uniform random selection of one point for each.

    NOTICE: Assumes that the input Points are in WGS84 projection (lat/lon).
    """

    left_geom_col = left_gdf.geometry.name
    right_geom_col = right_gdf.geometry.name

    # Ensure that index in right gdf is formed of sequential numbers
    right = right_gdf.copy().reset_index(drop=True)

    # Parse coordinates from points and insert them into a numpy array as RADIANS
    left_radians = np.array(left_gdf[left_geom_col].apply(lambda geom: (geom.x * np.pi / 180, geom.y * np.pi / 180)).to_list())
    right_radians = np.array(right[right_geom_col].apply(lambda geom: (geom.x * np.pi / 180, geom.y * np.pi / 180)).to_list())

    # Find the nearest points
    # -----------------------
    # closest ==> index in right_gdf that corresponds to the closest point

    closest = get_nearest(src_points=left_radians, candidates=right_radians, k_neighbors=n_neighbors_sampling)

    if index:
        return closest

    # Return points from right GeoDataFrame that are a random selection from the
    # closest to points in left GeoDataFrame
    closest_points = right.loc[[closest[randrange(n_neighbors_sampling), i] for i in range(closest.shape[1])]]

    # Ensure that the index corresponds the one in left_gdf
    closest_points = closest_points.reset_index(drop=True)

    return closest_points
