# Import Statements
import pickle
import igraph
import pandas as pd
import geopandas as gpd
from typing import List

# Internal Imports
import Models
import Events
import OnlineSolvers, ROASolver
import Generators

# Graph importing
DATA_DIR = ''

with open(DATA_DIR + 'NYC Graph//NYC_graph_revised.pickle', 'rb') as file:
    graph: igraph.Graph = pickle.load(file)

# Importing parameters
nodes_with_borough = gpd.read_file(DATA_DIR + 'NYC Graph//NYC_nodes_w_borough//NYC_nodes_w_borough.shp')

# Load the speeds df
speeds = pd.read_csv(DATA_DIR + 'NYC Graph//edge_speeds_vicky.csv')
speeds = speeds.drop('Unnamed: 0', axis=1)

# Sort the df according to city graph order
speeds.index = speeds['edgeid']
speeds = speeds.loc[graph.es['edgeid'], :]


# Experiment configuration
experiment = {'day': 'friday', 'dataReplica': 0, 'simTime': 24 * 3600, 'relocatorModel': 'survivalNoExp', 'dispatcher': 'preparedness',
              'relocate': False, 'ambulance_distribution': [355, 802], 'workload_restriction': True, 'workload_limit': .4, 'useUber': True, 'GAP': 0.05,
              'parameters_dir': 'HalfManhattan', 'relocation_period': 600, 'uberRatio': 1}

# Importing the data
with open(DATA_DIR + 'Preprocessing Values//{}//candidate_nodes.pickle'.format(experiment['parameters_dir']), 'rb') as file:
    candidate_nodes = pickle.load(file)
with open(DATA_DIR + 'Preprocessing Values//{}//demand_nodes.pickle'.format(experiment['parameters_dir']), 'rb') as file:
    demand_nodes = pickle.load(file)
with open(DATA_DIR + 'Preprocessing Values//{}//hospital_nodes.pickle'.format(experiment['parameters_dir']), 'rb') as file:
    hospital_nodes = pickle.load(file)
    hospital_nodes = {1: hospital_nodes}
with open(DATA_DIR + 'Preprocessing Values//{}//hospital_borough.pickle'.format(experiment['parameters_dir']), 'rb') as file:
    hospital_borough = pickle.load(file)
with open(DATA_DIR + 'Preprocessing Values//{}//hourly_demand_rates_HS.pickle'.format(experiment['parameters_dir']), 'rb') as file:
    demand_rates = [pickle.load(file)]
with open(DATA_DIR + 'Preprocessing Values//{}//hourly_demand_rates_LS.pickle'.format(experiment['parameters_dir']), 'rb') as file:
    demand_rates += [pickle.load(file)]
with open(DATA_DIR + 'Preprocessing Values//{}//mean_activity_time_HS.pickle'.format(experiment['parameters_dir']), 'rb') as file:
    busy_time = [pickle.load(file)]
with open(DATA_DIR + 'Preprocessing Values//{}//mean_activity_time_LS.pickle'.format(experiment['parameters_dir']), 'rb') as file:
    busy_time += [pickle.load(file)]
with open(DATA_DIR + 'Preprocessing Values//{}//candidate_candidate_time.pickle'.format(experiment['parameters_dir']), 'rb') as file:
    cand_cand_time = pickle.load(file)
with open(DATA_DIR + 'Preprocessing Values//{}//candidate_demand_time.pickle'.format(experiment['parameters_dir']), 'rb') as file:
    cand_demand_time = pickle.load(file)
with open(DATA_DIR + 'Preprocessing Values//{}//candidate_borough.pickle'.format(experiment['parameters_dir']), 'rb') as file:
    candidate_borough = pickle.load(file)
with open(DATA_DIR + 'Preprocessing Values//{}//demand_borough.pickle'.format(experiment['parameters_dir']), 'rb') as file:
    demand_borough = pickle.load(file)
with open(DATA_DIR + 'Preprocessing Values//{}//reachable_demand.pickle'.format(experiment['parameters_dir']), 'rb') as file:
    reachable_demand = pickle.load(file)
with open(DATA_DIR + 'Preprocessing Values//{}//reachable_inverse.pickle'.format(experiment['parameters_dir']), 'rb') as file:
    reachable_inverse = pickle.load(file)
with open(DATA_DIR + 'Preprocessing Values//{}//uber_nodes.pickle'.format(experiment['parameters_dir']), 'rb') as file:
    uber_nodes = pickle.load(file)
with open(DATA_DIR + 'Preprocessing Values//{}//graph_to_demand.pickle'.format(experiment['parameters_dir']), 'rb') as file:
    graph_to_demand = pickle.load(file)

# Importing low severity emergencies file
with open(DATA_DIR + 'Arrival Events//{}//LS//strep_{}.pickle'.format('Friday' if experiment['day'] == 'friday' else 'Thursday', experiment['dataReplica']), 'rb') as file:               # noqa E501
    emergencies: List['Events.EmergencyArrivalEvent'] = pickle.load(file)

# Importing high severity emergencies file
with open(DATA_DIR + 'Arrival Events//{}//HS//strep_{}.pickle'.format('Friday' if experiment['day'] == 'friday' else 'Thursday', experiment['dataReplica']), 'rb') as file:               # noqa E501
    emergencies += pickle.load(file)

# Filter events if HalfManhattan is the scenario
if experiment['parameters_dir'] == 'HalfManhattan':
    nodes = gpd.read_file(DATA_DIR + 'Preprocessing Values//HalfManhattan//NYC_nodes.geojson')
    emergencies = [e for e in emergencies if (e.node in demand_nodes or e.node in nodes.osmid.values.tolist())]

generator: Generators.ArrivalGenerator = Generators.CustomArrivalsGenerator([e for e in emergencies])

sim_parameters = Models.SimulationParameters(simulation_time=24 * 3600,
                                             initial_nodes=None,
                                             speeds_df=speeds,
                                             candidate_nodes=candidate_nodes,
                                             hospital_nodes=hospital_nodes,
                                             hospital_borough=hospital_borough,
                                             nodes_with_borough=nodes_with_borough,
                                             graph_to_demand=graph_to_demand,
                                             demand_nodes=demand_nodes,
                                             demand_rates=demand_rates,
                                             ALS_tours=80, # 355 for 5 boroughs, 80 for half Manhattan
                                             BLS_tours=175, # 802 for 5 boroughs, 175 for half Manhattan
                                             mean_busytime=busy_time,
                                             cand_cand_time=cand_cand_time,
                                             cand_demand_time=cand_demand_time,
                                             candidates_borough=candidate_borough,
                                             demand_borough=demand_borough,
                                             reachable_demand=reachable_demand,
                                             reachable_inverse=reachable_inverse,
                                             uber_nodes=uber_nodes,
                                             maximum_overload_ALS=.7,
                                             maximum_overload_BLS=.7,
                                             uber_seconds=0,
                                             optimization_gap=.05,
                                             max_expected_simultaneous_relocations=8,
                                             dispatching_penalty=1,
                                             max_relocation_time=1200,
                                             max_redeployment_time=800,
                                             force_static=False,
                                             random_seed=0)

optimizer: OnlineSolvers.RelocationModel = OnlineSolvers.SORDARelocatorDispatcher()

simulator: Models.EMSModel = Models.EMSModel(graph, generator, optimizer, sim_parameters, verbose=True)
statistics = simulator.run()
with open('StatisticsResults/test.pickle', 'wb') as f:
    pickle.dump(statistics, f)