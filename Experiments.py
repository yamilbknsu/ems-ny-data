# Import Statements
import pickle
import igraph
import numpy as np
import pandas as pd
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
with open(DATA_DIR + 'Preprocessing Values//candidate_nodes.pickle', 'rb') as file:
    candidate_nodes = pickle.load(file)
with open(DATA_DIR + 'Preprocessing Values//demand_nodes.pickle', 'rb') as file:
    demand_nodes = pickle.load(file)
with open(DATA_DIR + 'Preprocessing Values//hourly_demand_rates_HS.pickle', 'rb') as file:
    demand_rates = [pickle.load(file)]
with open(DATA_DIR + 'Preprocessing Values//hourly_demand_rates_LS.pickle', 'rb') as file:
    demand_rates += [pickle.load(file)]
with open(DATA_DIR + 'Preprocessing Values//mean_activity_time_HS.pickle', 'rb') as file:
    busy_time = [pickle.load(file)]
with open(DATA_DIR + 'Preprocessing Values//mean_activity_time_LS.pickle', 'rb') as file:
    busy_time += [pickle.load(file)]
with open(DATA_DIR + 'Preprocessing Values//candidate_candidate_time.pickle', 'rb') as file:
    cand_cand_time = pickle.load(file)
with open(DATA_DIR + 'Preprocessing Values//candidate_demand_time.pickle', 'rb') as file:
    cand_demand_time = pickle.load(file)
with open(DATA_DIR + 'Preprocessing Values//neighborhood_candidates.pickle', 'rb') as file:
    neighborhood_candidates = pickle.load(file)
with open(DATA_DIR + 'Preprocessing Values//neighborhood.pickle', 'rb') as file:
    neighborhood = pickle.load(file)
with open(DATA_DIR + 'Preprocessing Values//neighborhood_k.pickle', 'rb') as file:
    neighborhood_k = pickle.load(file)
with open(DATA_DIR + 'Preprocessing Values//reachable_demand.pickle', 'rb') as file:
    reachable_demand = pickle.load(file)
with open(DATA_DIR + 'Preprocessing Values//reachable_inverse.pickle', 'rb') as file:
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
                simulation_time=7*24*3600,
                initial_nodes=None,
                speeds_df = speeds,
                candidate_nodes=candidate_nodes,
                demand_nodes=demand_nodes,
                demand_rates=demand_rates,
                mean_busytime=busy_time,
                cand_cand_time=cand_cand_time,
                cand_demand_time=cand_demand_time,
                neighborhood=neighborhood,
                neighborhood_candidates=neighborhood_candidates,
                neighbor_k=neighborhood_k,
                reachable_demand=reachable_demand,
                reachable_inverse=reachable_inverse
)

simulator: Models.EMSModel = Models.EMSModel(graph, generator, Solvers.PreparednessDispatcher(), sim_parameters)

#solver = Solvers.MaxExpectedSurvivalRelocator()
#solver.relocate(simulator, simulator.parameters)
simulator.run()
