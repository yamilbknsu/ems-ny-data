import pickle
import itertools
import numpy as np
import multiprocessing
import geopandas as gpd
from typing import List
from random import randrange
from datetime import datetime
from sklearn.neighbors import BallTree
from shapely.ops import nearest_points
import sys
import os

# Insert the location of the rest of the scripts
sys.path.insert(1, 'C://Users//Yamil//Proyectos//Proyectos en Git//Memoria' +
                   ' Ambulancias//ems-ny-data//')

# Internal imports
import Events                                                                                                               # noqa E402

"""
The following two functions were extracted and the modified from the site:
https://automating-gis-processes.github.io/site/notebooks/L3/nearest-neighbor-faster.html

For a deeper explaination please refer to it.
"""

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


def arrivalShpToArrivalEvents(gdf, street_nodes, simulation_start_date, save_dir, k_neighbors_sample = 1,
    id_column = 'osmid', datetime_format = '%m/%d/%Y %H:%M:%S', date_column='INCIDENT_D', severity_level = 1):
    """
    Expecting a geodataframe with only a datetime column and
    geometry of points on WSG 84 CRS
    """

    gdf['Node'] = nearest_neighbor(gdf, street_nodes, k_neighbors_sample)[id_column]

    events: List['Events.EmergencyArrivalEvent'] = []
    for _, row in gdf.iterrows():
        severity = severity_level
        if severity == 2:
            if np.random.rand() > 0.57555:
                severity = 3
        events.append(Events.EmergencyArrivalEvent(None, 
            (datetime.strptime(row[date_column], datetime_format) - simulation_start_date).total_seconds(), row['Node'], severity))
    
    with open(save_dir, 'wb') as file:
        pickle.dump(events, file)


if __name__ == "__main__":
    DATA_DIR = 'C://Users//Yamil//Proyectos//Proyectos en Git//Memoria Ambulancias//'
    nodes_file = 'ems-ny-data//NYC Graph//NYC_nodes_revised.geojson'


    interval_methods = ['nsnr']
    initial_dataset_setups = ['LS19', 'HS19']
    spatio_temporal_replications = list(range(5))

    # Replication sampling for accounting for the random effect of the random node selected
    sampling_replications = 4
    gdfs = []
    nodes = gpd.read_file(DATA_DIR + nodes_file)

    for method, setup, strep in itertools.product(interval_methods,                                                         # noqa E501
                                                  initial_dataset_setups, spatio_temporal_replications):                                # noqa E501

        arrivals_file = 'Codigos Seba//output_arrivals//{}//stkde_{}_{}.shp'.format(setup, method, strep)
        save_dir = DATA_DIR + 'ems-ny-data//Arrival events//{}//{}//'.format(setup, method)

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        gdf = gpd.read_file(DATA_DIR + arrivals_file)
        gdf = gdf.drop(['arrival', 'interarriv'], axis=1)

        # Check for crs of the data
        if gdf.crs['init'] == 'epsg:3857':
            gdf = gdf.to_crs(epsg=4326)
        
        gdfs.append(gdf)

        for replica in range(sampling_replications):
            save_name = 'strep_{}_rep_{}.pickle'.format(strep, replica)
            print('Setting up process for {},{},{},{}'.format(method, setup, strep, replica))
            severity_level = 1
            if setup == 'LS19':
                severity_level = 2
            arrivalShpToArrivalEvents(gdfs[-1], nodes, datetime(2020, 1, 5), save_dir+save_name, 1, severity_level=severity_level)

    print()