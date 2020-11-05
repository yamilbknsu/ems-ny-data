import pickle
import igraph
import random
import numpy as np
import pandas as pd
import geopandas as gpd
from typing import List

# Internal imports
from dispatchAnalysis import nearest_neighbor

DATA_DIR = 'C://Users//Yamil//Proyectos//Proyectos en Git//Memoria Ambulancias//Old files//'         # noqa E501


def timeBasedRange(candidates, demand, graph, speeds, travel_limit):
    weights = np.array(graph.es['length'])/speeds
    distance_matrix = graph.shortest_paths(candidates, demand, weights)

    # Get the indices of the values that have a travel time below the limit
    valid_indices = np.where(np.array(distance_matrix) <= travel_limit)

    # Transform these indices into a dict
    output = {candidates[c]: [demand[i] for i in valid_indices[1][np.where(valid_indices[0] == c)]]    # noqa E501
              for c in range(len(candidates))}
    return output


def hour_to_interval(hour):
    if hour < 7:
        return 1
    elif hour < 10:
        return 2
    elif hour < 16:
        return 3
    elif hour < 19:
        return 4
    else:
        return 5


def computeDemandValues(_preprocesing_df_dir, 
                        _real_data_df_dir,
                        demand_nodes_df,
                        demand_to_node,
                        save_dir):
    """
    This function computes the mean demand of
    hours of ambulance per hour for each node
    and the mean busytime per event per node.
    """
    with open(_preprocesing_df_dir, 'rb') as f:
        df = pickle.load(f)

    with open(_real_data_df_dir, 'rb') as f:
        df_reduced = pickle.load(f)

    # Some preprocessing
    d_to_n_inverse ={}
    for n in demand_to_node.values():
        if n not in demand_to_node:
            l = []
            for k,v in demand_to_node.items():
                if v == n:
                    l.append(k)
            d_to_n_inverse[n] = l
    
    df.index = df['CAD_INCIDENT_ID']
    df_reduced = df_reduced[df_reduced['year'] == 2019]
    df_reduced['INCIDENT_CLOSE_DATETIME'] = list(df.loc[df_reduced['CAD_INCIDENT_ID'],:]['INCIDENT_CLOSE_DATETIME'])

    zip_codes = demand_nodes_df['ZIPCODE'].unique()

    events = df_reduced[(df_reduced['year'] == 2019)]
    # Low Severity calls
    LSdf = events[events['FINAL_SEVERITY_LEVEL_CODE'] >= 4]
    # High Severity calls
    HSdf = events[events['FINAL_SEVERITY_LEVEL_CODE'] <= 3]

    events_dfs = [HSdf, LSdf]
    names = ['HS', 'LS']

    for i, event_df in enumerate(events_dfs):
        print('Demand points for ', names[i], '...')
        final_events_df = pd.DataFrame([], columns = ['INCIDENT_DATETIME',
            'FIRST_ASSIGNMENT_DATETIME', 'FINAL_SEVERITY_LEVEL_CODE',
            'INCIDENT_CLOSE_DATETIME', 'hour', 'pointid', 'geometry'])

        for code in zip_codes:
            zip_points = demand_nodes_df[demand_nodes_df['ZIPCODE'] == code]
            sub_df = event_df[event_df['ZIPCODE'] == int(code)]

            new_df = sub_df[['INCIDENT_DATETIME', 'FIRST_ASSIGNMENT_DATETIME',
                            'FINAL_SEVERITY_LEVEL_CODE','INCIDENT_CLOSE_DATETIME',
                                                    'hour']]
            samplepoints = zip_points.sample(len(new_df), replace=True)[['geometry', 'id']]
            new_df['geometry'] = list(samplepoints['geometry'])
            new_df['pointid'] = list(samplepoints['id'].astype('int64'))

            final_events_df = final_events_df.append(new_df, ignore_index = True)
        final_events_df = final_events_df.sort_values(by='INCIDENT_DATETIME')
        final_events_df.index = range(len(final_events_df))

        final_events_df['hour_interval'] = [hour_to_interval(row.hour)
                                            for _, row in final_events_df.iterrows()]

        hourly_demand = final_events_df.groupby(['hour_interval', 'pointid']).size()\
            .unstack().fillna(0).mul([1 / 7, 1 / 3, 1 / 6, 1 / 3, 1 / 5], axis=0) / 365

        new_df_final = []
        for k in d_to_n_inverse:
            try:
                new_df_final.append(list(hourly_demand[d_to_n_inverse[k]].sum(axis = 1)))
            except KeyError:
                new_df_final.append([0]*5)
        hourly_demand = pd.DataFrame(np.array(new_df_final).T, columns=list(d_to_n_inverse.keys()), index = range(1,6))

        with open(save_dir + '/hourly_demand_rates_{}.pickle'.format(names[i]), 'wb') as f:
            pickle.dump(hourly_demand, f)
        
        final_events_df['busytime'] = (final_events_df['INCIDENT_CLOSE_DATETIME'] -\
             final_events_df['FIRST_ASSIGNMENT_DATETIME']).dt.seconds

        mean_busytime = final_events_df.groupby(['hour_interval','pointid']).mean().unstack()
        mean_busytime = mean_busytime.fillna(mean_busytime.mean().mean())/3600


        new_df_final = []
        mean = mean_busytime['busytime'].mean().mean()
        for k in d_to_n_inverse:
            try:
                new_df_final.append(list(mean_busytime['busytime'][d_to_n_inverse[k]].sum(axis = 1)))
            except KeyError:
                new_df_final.append([mean]*5)
        mean_busytime = pd.DataFrame(np.array(new_df_final).T, columns=list(d_to_n_inverse.keys()), index = range(1,6))

        with open(save_dir + '/mean_activity_time_{}.pickle'.format(names[i]), 'wb') as f:
            pickle.dump(mean_busytime, f)

def computePreprocessingData(candidates_df, demand_df, hospital_df,
                              base_nodes_w_borough, city_graph, 
                              speeds_df, save_dir,
                              reachable_limits = 8*60,
                              compute_uber = False):
    """
    As you see, a huge function, but don't be afraid, it's just a
    bunch of little processes that compute some statistics and lists.
    (Excep for the neighborhood, that's a big deal)
    """

    # Compute the graph points for the candidates
    print('Computing candidate nodes...')
    candidate_nodes = list(set(nearest_neighbor(candidates_df, base_nodes_w_borough, 1)['osmid']))

    with open(save_dir + "candidate_nodes.pickle", 'wb') as f:
        pickle.dump(candidate_nodes, f)
    
    print('Computing candidate nodes borough...')
    candidate_borough = {b: list(base_nodes_w_borough[(base_nodes_w_borough['boro_code'] == b) & 
                                                      (base_nodes_w_borough['osmid'].isin(candidate_nodes))]['osmid'])
                         for b in range(1,6)}
    
    with open(save_dir + "candidate_borough.pickle", 'wb') as f:
        pickle.dump(candidate_borough, f)

    # Compute the graph points for the hospitals
    print('Computing Hospital nodes...')
    NN = nearest_neighbor(hospital_df, base_nodes_w_borough, 1)
    hospital_nodes = list(set(NN['osmid']))
    hospital_borough = {hospital: int(NN[NN['osmid'] == hospital]['boro_code']) for hospital in hospital_nodes}

    with open(save_dir + "hospital_nodes.pickle", 'wb') as f:
        pickle.dump(hospital_nodes, f)
    with open(save_dir + "hospital_borough.pickle", 'wb') as f:
        pickle.dump(hospital_borough, f)
    
    # Compute the graph points for the demands
    print('Computing demand nodes...')
    demand_nearest = nearest_neighbor(demand_df, base_nodes_w_borough, 1)['osmid']
    demand_to_node = dict(zip(list(demand_points['id']), list(demand_nearest)))

    with open(save_dir + "/demand_nodes.pickle", 'wb') as f:
        pickle.dump(list(set(demand_to_node.values())), f)

    print('Computing demand nodes borough...')
    demand_borough = {b: list(base_nodes_w_borough[(base_nodes_w_borough['boro_code'] == b) & 
                                                   (base_nodes_w_borough['osmid'].isin(list(set(demand_to_node.values()))))]['osmid'])
                         for b in range(1,6)}
    
    with open(save_dir + "demand_borough.pickle", 'wb') as f:
        pickle.dump(demand_borough, f)

    print('Computing graph nodes nearest demand ...')
    nearest_demand = nearest_neighbor(base_nodes_w_borough, demand_df, 1)
    graph_to_demand = dict(zip(base_nodes_w_borough['osmid'], [demand_to_node[n] for n in list(nearest_demand['id'])]))

    with open(save_dir + "graph_to_demand.pickle", 'wb') as f:
        pickle.dump(graph_to_demand, f)
    
    if compute_uber:
        print('Computing uber nodes ...')
        uber_nodes = {}
        for node in city_graph.vs:
            distances = city_graph.shortest_paths(node['name'], city_graph.vs['name'], weights= city_graph.es['length']/speeds_df['p1n'])[0]
            distances = np.array(distances)
            uber_nodes[node['name']] = city_graph.vs[random.choice(np.where((distances > 4.5*60) & (distances < 10 * 60))[0])]['name']

        with open(save_dir + "uber_nodes.pickle", 'wb') as f:
            pickle.dump(uber_nodes, f)

    print('Computing demand values...')
    computeDemandValues(DATA_DIR + '/data/dispatch_data_for_preprocessing.pickle',
                        DATA_DIR + '/data/dispatch_data_reduced.pickle',
                        demand_df, demand_to_node,
                        save_dir)
    
    # Compute reachable demand
    print('Computing reachable demand...')
    reachable_demand: List[dict] = []
    for time_period in range(1,6):
        reachable_demand.append(timeBasedRange(candidate_nodes, list(set(demand_nearest)), city_graph,
                    list(speeds['p' + str(time_period) + 'n']), reachable_limits))
    
    with open(save_dir + '/reachable_demand.pickle', 'wb') as f:
        pickle.dump(reachable_demand, f)
    
    ## Compute the reachable demand inverse (demand to candidate)
    print('Computing reachable inverse...')
    reachable_inverse = []
    for time_period in range(5):
        inverse_dict = {}
        for d in demand_points['id']:
            node = demand_to_node[d]
            d_list = []
            for candidate, r_demand in reachable_demand[time_period].items():
                if node in r_demand:
                    d_list.append(candidate)
            inverse_dict[node] = d_list
        reachable_inverse.append(inverse_dict)

    with open(save_dir + '/reachable_inverse.pickle', 'wb') as f:
        pickle.dump(reachable_inverse, f)
    
    # Compute the neighborhoods
    print('Computing neighborhoods...')
    neighborhood: List[dict] = []
    neighborhood_candidates: List[dict] = []
    neighborhood_k: List[dict] = []

    for time_period in range(5):
        print(time_period)
        neighborhood_dict = {}
        neighborhood_candidates_dict = {}
        neighborhood_k_dict = {}

        for c in candidate_nodes:
            neighbor_demands = reachable_demand[time_period][c]
            neighbor_k: List[str] = []

            for d in reachable_demand[time_period][c]:
                neighbor_k = neighbor_k + reachable_inverse[time_period][d]
                for c1 in reachable_inverse[time_period][d]:
                    neighbor_demands = neighbor_demands + \
                        reachable_demand[time_period][c1]
            neighbor_demands = list(set(neighbor_demands))
            neighbor_k = list(set(neighbor_k))
    
            neighborhood_dict[c] = neighbor_demands
            neighborhood_candidates_dict[c] = neighbor_k
            neighborhood_k_dict[c] = len(neighbor_k)

        neighborhood.append(neighborhood_dict)
        neighborhood_candidates.append(neighborhood_candidates_dict)
        neighborhood_k.append(neighborhood_k_dict)

    
    with open(save_dir + '/neighborhood.pickle', 'wb') as f:
        pickle.dump(neighborhood, f)
    
    with open(save_dir + '/neighborhood_k.pickle', 'wb') as f:
        pickle.dump(neighborhood_k, f)
    
    with open(save_dir + '/neighborhood_candidates.pickle', 'wb') as f:
        pickle.dump(neighborhood_candidates, f)

    ## Compute the candidate demand travel matrix for each time period
    print('Computing time matrices...')
    time_matrix = []
    unique_candidate_nodes = list(set(candidate_nodes))
    unique_demand_nodes = list(set(demand_to_node.values()))
    for time_period in range(1,6):
        weights = np.array(city_graph.es['length'])/list(speeds['p' + str(time_period) + 'n'])
        time_matrix.append(np.array(city_graph.shortest_paths(candidate_nodes, unique_demand_nodes, weights)))

    with open(save_dir + '/candidate_demand_time.pickle', 'wb') as f:
        pickle.dump(time_matrix, f)
    

    # Compute the travel candidate candidate matrix for each time period
    print('Computing time matrices...')
    time_matrix = []
    unique_candidate_nodes = list(set(candidate_nodes))
    for time_period in range(1,6):
        weights = np.array(city_graph.es['length'])/list(speeds['p' + str(time_period) + 'n'])
        time_matrix.append(np.array(city_graph.shortest_paths(unique_candidate_nodes, unique_candidate_nodes, weights)))
    
    with open(save_dir + '/candidate_candidate_time.pickle', 'wb') as f:
        pickle.dump(time_matrix, f)


if __name__ == "__main__":
    # Load the original candidates and nodes
    original_candidates = gpd.read_file('C://Users//Yamil//Proyectos//Proyectos en Git//Memoria Ambulancias//ems-ny-data//NYC Graph/EMScandidatesMixedLRNew.geojson')

    # Load the nodes of the graph with borough
    graph_nodes = gpd.read_file('C://Users//Yamil//Proyectos//Proyectos en Git//Memoria Ambulancias//ems-ny-data//NYC Graph/NYC_nodes_w_borough/NYC_nodes_w_borough.shp')

    # Load the uniform demand points
    demand_points = gpd.read_file(DATA_DIR + 'Generated Shapefiles/GeoTools/Uniform600m/Uniform600mDemandNew.geojson')

    # Load the hospital points
    hospital_df = gpd.read_file(DATA_DIR + 'Generated Shapefiles/NYC_Hospitals/NYC_Hospitals.geojson')

    # Load the igraph
    with open("C://Users//Yamil//Proyectos//Proyectos en Git//Memoria Ambulancias//ems-ny-data//NYC Graph//NYC_graph_revised.pickle", 'rb') as f:
        city_graph = pickle.load(f)

    # Load the speeds df
    speeds = pd.read_csv('C://Users//Yamil//Proyectos//Proyectos en Git//Memoria Ambulancias//ems-ny-data//NYC Graph//edge_speeds_vicky.csv')
    speeds = speeds.drop('Unnamed: 0', axis=1)

    # Sort the df according to city graph order
    speeds.index = speeds['edgeid']
    speeds = speeds.loc[city_graph.es['edgeid'], :]

    computePreprocessingData(original_candidates, demand_points, hospital_df, graph_nodes, city_graph, speeds, 'C://Users//Yamil//Proyectos//Proyectos en Git//Memoria Ambulancias//ems-ny-data//Preprocessing Values//Base//')
