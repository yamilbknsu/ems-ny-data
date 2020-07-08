# Import Statements
import pickle
import igraph
import numpy as np
import pandas as pd
import geopandas as gpd
from typing import List

# Internal Imports
import Models
import Events
import Solvers
import Generators

"""
General TODO for the project
----------------------------

"""

# Graph importing
DATA_DIR = 'C://Users//Yamil//Proyectos//Proyectos en Git//' \
    + 'Memoria Ambulancias//ems-ny-data//'

with open(DATA_DIR + 'NYC Graph//NYC_graph_revised.pickle', 'rb') as file:
    graph: igraph.Graph = pickle.load(file)

# Importing low severity emergencies file
with open(DATA_DIR + 'Arrival Events//LS19//nsnr//strep_0_rep_0.pickle', 'rb') as file:               # noqa E501
    emergencies: List['Events.EmergencyArrivalEvent'] = pickle.load(file)

# Importing high severity emergencies file
with open(DATA_DIR + 'Arrival Events//HS19//nsnr//strep_0_rep_0.pickle', 'rb') as file:               # noqa E501
    emergencies += pickle.load(file)

# Importing parameters
nodes_with_borough = gpd.read_file(DATA_DIR + 'NYC Graph//NYC_nodes_w_borough//NYC_nodes_w_borough.shp')
with open(DATA_DIR + 'Preprocessing Values//Mixed//candidate_nodes.pickle', 'rb') as file:
    candidate_nodes = pickle.load(file)
with open(DATA_DIR + 'Preprocessing Values//Mixed//demand_nodes.pickle', 'rb') as file:
    demand_nodes = pickle.load(file)
with open(DATA_DIR + 'Preprocessing Values//Mixed//hospital_nodes.pickle', 'rb') as file:
    hospital_nodes = pickle.load(file)
    hospital_nodes = {s: hospital_nodes for s in range(1,4)}
with open(DATA_DIR + 'Preprocessing Values//Mixed//hourly_demand_rates_HS.pickle', 'rb') as file:
    demand_rates = [pickle.load(file)]
with open(DATA_DIR + 'Preprocessing Values//Mixed//hourly_demand_rates_LS.pickle', 'rb') as file:
    demand_rates += [pickle.load(file)]
with open(DATA_DIR + 'Preprocessing Values//Mixed//mean_activity_time_HS.pickle', 'rb') as file:
    busy_time = [pickle.load(file)]
with open(DATA_DIR + 'Preprocessing Values//Mixed//mean_activity_time_LS.pickle', 'rb') as file:
    busy_time += [pickle.load(file)]
with open(DATA_DIR + 'Preprocessing Values//Mixed//candidate_candidate_time.pickle', 'rb') as file:
    cand_cand_time = pickle.load(file)
with open(DATA_DIR + 'Preprocessing Values//Mixed//candidate_demand_time.pickle', 'rb') as file:
    cand_demand_time = pickle.load(file)
with open(DATA_DIR + 'Preprocessing Values//Mixed//neighborhood_candidates.pickle', 'rb') as file:
    neighborhood_candidates = pickle.load(file)
with open(DATA_DIR + 'Preprocessing Values//Mixed//neighborhood.pickle', 'rb') as file:
    neighborhood = pickle.load(file)
with open(DATA_DIR + 'Preprocessing Values//Mixed//neighborhood_k.pickle', 'rb') as file:
    neighborhood_k = pickle.load(file)
with open(DATA_DIR + 'Preprocessing Values//Mixed//candidate_borough.pickle', 'rb') as file:
    candidate_borough = pickle.load(file)
with open(DATA_DIR + 'Preprocessing Values//Mixed//demand_borough.pickle', 'rb') as file:
    demand_borough = pickle.load(file)
with open(DATA_DIR + 'Preprocessing Values//Mixed//reachable_demand.pickle', 'rb') as file:
    reachable_demand = pickle.load(file)
with open(DATA_DIR + 'Preprocessing Values//Mixed//reachable_inverse.pickle', 'rb') as file:
    reachable_inverse = pickle.load(file)

 # Load the speeds df
speeds = pd.read_csv(DATA_DIR + 'NYC Graph//edge_speeds_vicky.csv')
speeds = speeds.drop('Unnamed: 0', axis=1)

# Sort the df according to city graph order
speeds.index = speeds['edgeid']
speeds = speeds.loc[graph.es['edgeid'], :]

generator: Generators.ArrivalGenerator = \
           Generators.CustomArrivalsGenerator([e for e in emergencies])

sim_parameters = Models.SimulationParameters(
                simulation_time=1*3600,
                initial_nodes=None,
                speeds_df = speeds,
                candidate_nodes=candidate_nodes,
                hospital_nodes=hospital_nodes,
                nodes_with_borough=nodes_with_borough,
                demand_nodes=demand_nodes,
                demand_rates=demand_rates,
                n_vehicles = [[312, 505]] * 5,
                mean_busytime=busy_time,
                cand_cand_time=cand_cand_time,
                cand_demand_time=cand_demand_time,
                neighborhood=neighborhood,
                neighborhood_candidates=neighborhood_candidates,
                neighbor_k=neighborhood_k,
                candidates_borough=candidate_borough,
                demand_borough=demand_borough,
                reachable_demand=reachable_demand,
                reachable_inverse=reachable_inverse,
                maximum_uber_per_period = 0
)

#simulator: Models.EMSModel = Models.EMSModel(graph, generator, Solvers.PreparednessDispatcher(), Solvers.MaxExpectedSurvivalRelocator(), sim_parameters)
#simulator.run()

#simulator.recorder.saveToJSON(DATA_DIR + 'JSON events//events2.json')