# Import Statements
import os.path
import pickle
import igraph
import numpy as np
import pandas as pd
import geopandas as gpd
from typing import List, Dict, Any, Optional

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

# Importing parameters
nodes_with_borough = gpd.read_file(DATA_DIR + 'NYC Graph//NYC_nodes_w_borough//NYC_nodes_w_borough.shp')
with open(DATA_DIR + 'Preprocessing Values//MixedLR//candidate_nodes.pickle', 'rb') as file:
    candidate_nodes = pickle.load(file)
with open(DATA_DIR + 'Preprocessing Values//MixedLR//demand_nodes.pickle', 'rb') as file:
    demand_nodes = pickle.load(file)
with open(DATA_DIR + 'Preprocessing Values//MixedLR//hospital_nodes.pickle', 'rb') as file:
    hospital_nodes = pickle.load(file)
    hospital_nodes = {1: hospital_nodes}
with open(DATA_DIR + 'Preprocessing Values//MixedLR//hourly_demand_rates_HS.pickle', 'rb') as file:
    demand_rates = [pickle.load(file)]
with open(DATA_DIR + 'Preprocessing Values//MixedLR//hourly_demand_rates_LS.pickle', 'rb') as file:
    demand_rates += [pickle.load(file)]
with open(DATA_DIR + 'Preprocessing Values//MixedLR//mean_activity_time_HS.pickle', 'rb') as file:
    busy_time = [pickle.load(file)]
with open(DATA_DIR + 'Preprocessing Values//MixedLR//mean_activity_time_LS.pickle', 'rb') as file:
    busy_time += [pickle.load(file)]
with open(DATA_DIR + 'Preprocessing Values//MixedLR//candidate_candidate_time.pickle', 'rb') as file:
    cand_cand_time = pickle.load(file)
with open(DATA_DIR + 'Preprocessing Values//MixedLR//candidate_demand_time.pickle', 'rb') as file:
    cand_demand_time = pickle.load(file)
with open(DATA_DIR + 'Preprocessing Values//MixedLR//neighborhood_candidates.pickle', 'rb') as file:
    neighborhood_candidates = pickle.load(file)
with open(DATA_DIR + 'Preprocessing Values//MixedLR//neighborhood.pickle', 'rb') as file:
    neighborhood = pickle.load(file)
with open(DATA_DIR + 'Preprocessing Values//MixedLR//neighborhood_k.pickle', 'rb') as file:
    neighborhood_k = pickle.load(file)
with open(DATA_DIR + 'Preprocessing Values//MixedLR//candidate_borough.pickle', 'rb') as file:
    candidate_borough = pickle.load(file)
with open(DATA_DIR + 'Preprocessing Values//MixedLR//demand_borough.pickle', 'rb') as file:
    demand_borough = pickle.load(file)
with open(DATA_DIR + 'Preprocessing Values//MixedLR//reachable_demand.pickle', 'rb') as file:
    reachable_demand = pickle.load(file)
with open(DATA_DIR + 'Preprocessing Values//MixedLR//reachable_inverse.pickle', 'rb') as file:
    reachable_inverse = pickle.load(file)
with open(DATA_DIR + 'Preprocessing Values//MixedLR//uber_nodes.pickle', 'rb') as file:
    uber_nodes = pickle.load(file)

 # Load the speeds df
speeds = pd.read_csv(DATA_DIR + 'NYC Graph//edge_speeds_vicky.csv')
speeds = speeds.drop('Unnamed: 0', axis=1)

# Sort the df according to city graph order
speeds.index = speeds['edgeid']
speeds = speeds.loc[graph.es['edgeid'], :]


EXPERIMENTS: List[Dict[str, Any]] \
            = [{'day': 'friday', 'dataReplica': 0, 'simTime': 24*3600, 'relocatorModel': 'survival', 'dispatcher': 'preparedness',
                'relocate': True, 'workload_restriction': True, 'workload_limit': .4, 'useUber': False},

                {'day': 'friday', 'dataReplica': 0, 'simTime': 24*3600, 'relocatorModel': 'survival', 'dispatcher': 'nearest',
                'relocate': True, 'workload_restriction': True, 'workload_limit': .4, 'useUber': False},

                {'day': 'friday', 'dataReplica': 0, 'simTime': 24*3600, 'relocatorModel': 'survival', 'dispatcher': 'preparedness',
                'relocate': True, 'workload_restriction': False, 'workload_limit': .4, 'useUber': False},

                {'day': 'friday', 'dataReplica': 0, 'simTime': 24*3600, 'relocatorModel': 'survival', 'dispatcher': 'nearest',
                'relocate': True, 'workload_restriction': False, 'workload_limit': .4, 'useUber': False},

                {'day': 'friday', 'dataReplica': 0, 'simTime': 24*3600, 'relocatorModel': 'survival', 'dispatcher': 'preparedness',
                'relocate': False, 'workload_restriction': True, 'workload_limit': .4, 'useUber': False},

                {'day': 'friday', 'dataReplica': 0, 'simTime': 24*3600, 'relocatorModel': 'survival', 'dispatcher': 'nearest',
                'relocate': False, 'workload_restriction': True, 'workload_limit': .4, 'useUber': False},

                {'day': 'friday', 'dataReplica': 0, 'simTime': 24*3600, 'relocatorModel': 'survival', 'dispatcher': 'preparedness',
                'relocate': False, 'workload_restriction': False, 'workload_limit': .4, 'useUber': False},

                {'day': 'friday', 'dataReplica': 0, 'simTime': 24*3600, 'relocatorModel': 'survival', 'dispatcher': 'nearest',
                'relocate': False, 'workload_restriction': False, 'workload_limit': .4, 'useUber': False},

                {'day': 'friday', 'dataReplica': 0, 'simTime': 24*3600, 'relocatorModel': 'survival', 'dispatcher': 'preparedness',
                'relocate': True, 'workload_restriction': True, 'workload_limit': .4, 'useUber': True},

                {'day': 'friday', 'dataReplica': 0, 'simTime': 24*3600, 'relocatorModel': 'coverage', 'dispatcher': 'preparedness',
                'relocate': True, 'workload_restriction': True, 'workload_limit': .4, 'useUber': False}]

for experiment in EXPERIMENTS:

    name = '{}_{}_{}_{}_{}_{}_{}_{}'.format(experiment['day'], experiment['dataReplica'], experiment['relocatorModel'], experiment['dispatcher'],
                                            'Relocate' if experiment['relocate'] else 'NoRelocation', 'Workload' if experiment['workload_restriction'] else 'NoWorkloadRestriction',
                                            experiment['workload_limit'], 'Uber' if experiment['useUber'] else 'NoUber')

    if os.path.exists('StatisticsResults/{}.pickle'.format(name)):
        print('Skipping ' + name)
        continue

    print('Starting ' + name)
    # Importing low severity emergencies file
    with open(DATA_DIR + 'Arrival Events//{}//LS19//strep_{}.pickle'.format('Friday' if experiment['day'] == 'friday' else 'Monday', experiment['dataReplica']), 'rb') as file:               # noqa E501
        emergencies: List['Events.EmergencyArrivalEvent'] = pickle.load(file)

    # Importing high severity emergencies file
    with open(DATA_DIR + 'Arrival Events//{}//HS19//strep_{}.pickle'.format('Friday' if experiment['day'] == 'friday' else 'Monday', experiment['dataReplica']), 'rb') as file:               # noqa E501
        emergencies += pickle.load(file)


    generator: Generators.ArrivalGenerator = \
            Generators.CustomArrivalsGenerator([e for e in emergencies])

    sim_parameters = Models.SimulationParameters(
                    simulation_time=experiment['simTime'],
                    initial_nodes=None,
                    speeds_df = speeds,
                    candidate_nodes=candidate_nodes,
                    hospital_nodes=hospital_nodes,
                    nodes_with_borough=nodes_with_borough,
                    demand_nodes=demand_nodes,
                    demand_rates=demand_rates,
                    n_vehicles = [[312, 505]] * 3,
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
                    uber_nodes=uber_nodes,
                    relocation_optimization = experiment['relocate'],
                    apply_workload_restriction = experiment['workload_restriction'],
                    maximum_overload_ALS = experiment['workload_limit'],
                    maximum_overload_BLS = experiment['workload_limit'],
                    is_uber_available = experiment['useUber']
    )

    dispatcher: Solvers.DispatcherModel = Solvers.DispatcherModel()
    relocator: Solvers.RelocationModel = Solvers.RelocationModel()

    if experiment['dispatcher'] == 'nearest':
        dispatcher = Solvers.NearestDispatcher()
    elif experiment['dispatcher'] == 'preparedness':
        dispatcher = Solvers.PreparednessDispatcher()
    
    if experiment['relocatorModel'] == 'survival':
        relocator = Solvers.MaxExpectedSurvivalRelocator()
    elif experiment['relocatorModel'] == 'coverage':
        relocator = Solvers.MaxSingleCoverRelocator()

    simulator: Models.EMSModel = Models.EMSModel(graph, generator, dispatcher, relocator, sim_parameters, verbose=False)
    statistics = simulator.run()

    with open('StatisticsResults/{}.pickle'.format(name), 'wb') as f:
        pickle.dump(statistics, f)


"""
DUMMY EXPERIMENT
dummy_emergencies: List[Events.EmergencyArrivalEvent] = [Events.EmergencyArrivalEvent(None, 15, '42823598', 1, 82),
                                                         Events.EmergencyArrivalEvent(None, 50, '42902721', 1, 82),
                                                         Events.EmergencyArrivalEvent(None, 120, '42917270', 1, 82),
                                                         Events.EmergencyArrivalEvent(None, 200, '42823718', 1, 82),
                                                         Events.EmergencyArrivalEvent(None, 300, '42503901', 1, 82)]
dummy_generator: Generators.ArrivalGenerator = \
           Generators.CustomArrivalsGenerator([e for e in dummy_emergencies])

dummy_sim_parameters = Models.SimulationParameters(
                       simulation_time=3600,
                       initial_nodes= None,
                       speeds_df=speeds,
                       candidate_nodes=candidate_nodes,
                       hospital_nodes=hospital_nodes,
                       nodes_with_borough=nodes_with_borough,
                       demand_nodes=demand_nodes,
                       demand_rates=demand_rates,
                       n_vehicles=[[312, 505]],
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
                       uber_nodes=uber_nodes,
                       maximum_uber_per_period = 0
)


experiment_name = "Experiment 4"
#simulator: Models.EMSModel = Models.EMSModel(graph, generator, Solvers.PreparednessDispatcher(), Solvers.MaxExpectedSurvivalRelocator(), sim_parameters, verbose=False)
simulator: Models.EMSModel = Models.EMSModel(graph, generator, Solvers.NearestDispatcher(), Solvers.MaxExpectedSurvivalRelocator(), sim_parameters, verbose=False)
statistics = simulator.run()

#with open('StatisticsResults/{}.pickle'.format(experiment_name), 'wb') as f:
#    pickle.dump(statistics, f)

#simulator.recorder.saveToJSON(DATA_DIR + 'JSON events//events2.json')
"""