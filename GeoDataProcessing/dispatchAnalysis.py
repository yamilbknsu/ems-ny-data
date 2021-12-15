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

    disposition_codes = [82, 83, 91, 92, 93, 94, 95, 96]
    disposition_prob_HS = np.array([.819188, .018657, .007153, .000004, .126319, .001084, .000075, .027521])/sum([.819188, .018657, .007153, .000004, .126319, .001084, .000075, .027521])
    disposition_prob_LS = np.array([.399125, .000565, .022335, .000010, .106771, .000464, .000124, .035278])/np.sum([.399125, .000565, .022335, .000010, .106771, .000464, .000124, .035278])
    disposition_prob_ULS = np.array([.349892, .000183, .011662, .000002, .052106, .000285, .000076, .021123])/.43532899999999

    gdf['Node'] = nearest_neighbor(gdf, street_nodes, k_neighbors_sample)[id_column]

    events: List['Events.EmergencyArrivalEvent'] = []
    for _, row in gdf.iterrows():
        severity = severity_level
        if severity == 2:
            if np.random.rand() > 0.564672:
                severity = 3
        
        if severity == 1:
            disposition_code = np.random.choice(disposition_codes, p=disposition_prob_HS)
        elif severity == 2:
            disposition_code = np.random.choice(disposition_codes, p=disposition_prob_LS)
        else:
            disposition_code = np.random.choice(disposition_codes, p=disposition_prob_ULS)

        events.append(Events.EmergencyArrivalEvent(None, 
            (datetime.strptime(row[date_column], datetime_format) - simulation_start_date).total_seconds(), row['Node'], severity, disposition_code=disposition_code))
    
    with open(save_dir, 'wb') as file:
        pickle.dump(events, file)


if __name__ == "__main__":
    DATA_DIR = 'C://Users//Yamil//Proyectos//Proyectos en Git//Memoria Ambulancias//'
    demand_nodes_file = 'Old Files//Generated Shapefiles//GeoTools//Uniform600m//Uniform600mDemandNew.geojson'
    DATA_DIR = 'C://Users//Yamil//Proyectos//Proyectos en Git//Memoria Ambulancias//'
    nodes_file = 'ems-ny-data//NYC Graph//NYC_nodes_revised.geojson'

    initial_dataset_setups = ['LS19', 'HS19']
    spatio_temporal_replications = list(range(50))

    gdfs = []
    nodes = gpd.read_file(DATA_DIR + nodes_file)
    demand_nodes = gpd.read_file(DATA_DIR + demand_nodes_file)

    nodes = nearest_neighbor(demand_nodes, nodes, 1)

    for setup, strep in itertools.product(initial_dataset_setups, spatio_temporal_replications):                                # noqa E501

        arrivals_file = 'Codigos Seba//output_arrivals//Friday//{}//stkde_nsnr_{}.shp'.format(setup, strep)
        save_dir = DATA_DIR + 'ems-ny-data//Arrival events//Friday//{}//'.format(setup)

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        gdf = gpd.read_file(DATA_DIR + arrivals_file)
        gdf = gdf.drop(['arrival', 'interarriv'], axis=1)

        # Check for crs of the data
        #if gdf.crs['init'] == 'epsg:3857':
        gdf = gdf.to_crs(epsg=4326)
        
        gdfs.append(gdf)

        save_name = 'strep_{}.pickle'.format(strep)
        print('Setting up process for {},{}'.format(setup, strep))
        severity_level = 1
        if setup == 'LS19':
            severity_level = 2
        arrivalShpToArrivalEvents(gdfs[-1], nodes, datetime(2020, 1, 17), save_dir+save_name, 1, severity_level=severity_level)

    print()