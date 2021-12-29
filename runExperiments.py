# Import Statements
import os.path
import argparse
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

"""
General TODO for the project
----------------------------

"""

parser = argparse.ArgumentParser()
parser.add_argument('-i', type=int, default=1, help='Index of the experiment to run')
parser.add_argument('-n', type=int, default=1, help='Number of replicas per task')
args = parser.parse_args()

DATA_DIR = ''
RESULTS_DIR = 'StatisticsResults'

# Graph importing
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

with open('experimentsConfig.pickle', 'rb') as f:
    EXPERIMENTS = pickle.load(f)

for i in range((args.i) * args.n, (args.i + 1) * args.n):
    experimentInfo = EXPERIMENTS[i]

    name = experimentInfo[0]
    experiment = experimentInfo[1]

    if os.path.exists('{}/{}.pickle'.format(RESULTS_DIR, name)):
        print('Skipping ' + name)
        continue

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

    print('Starting ' + name)
    # Importing low severity emergencies file
    with open(DATA_DIR + 'Arrival Events//{}//LS//strep_{}.pickle'.format('Friday' if experiment['day'] == 'friday' else 'Thursday', experiment['dataReplica']), 'rb') as file:               # noqa E501
        emergencies: List['Events.EmergencyArrivalEvent'] = pickle.load(file)

    # Importing high severity emergencies file
    with open(DATA_DIR + 'Arrival Events//{}//HS//strep_{}.pickle'.format('Friday' if experiment['day'] == 'friday' else 'Thursday', experiment['dataReplica']), 'rb') as file:               # noqa E501
        emergencies += pickle.load(file)
    
    # If simulation time exceeds one day, select from both thursday
    if experiment['simTime'] == 48 * 3600 and experiment['day'] == 'thursday':
        # Importing low severity emergencies file
        with open(DATA_DIR + 'Arrival Events//{}//LS//strep_{}.pickle'.format('Thursday', experiment['dataReplica']), 'rb') as file:               # noqa E501
            emergencies: List['Events.EmergencyArrivalEvent'] = [e for e in pickle.load(file) if e.time >= 24 * 3600]

        # Importing high severity emergencies file
        with open(DATA_DIR + 'Arrival Events//{}//HS//strep_{}.pickle'.format('Thursday', experiment['dataReplica']), 'rb') as file:               # noqa E501
            emergencies += [e for e in pickle.load(file) if e.time >= 24 * 3600]
        
        # Importing high severity emergencies file
        with open(DATA_DIR + 'Arrival Events//{}//LS//strep_{}.pickle'.format('Friday', experiment['dataReplica']), 'rb') as file:               # noqa E501
            emergencies += [e for e in pickle.load(file) if e.time <= 24 * 3600]

        # Importing high severity emergencies file
        with open(DATA_DIR + 'Arrival Events//{}//HS//strep_{}.pickle'.format('Friday', experiment['dataReplica']), 'rb') as file:               # noqa E501
            emergencies += [e for e in pickle.load(file) if e.time <= 24 * 3600]

    # Filter events if HalfManhattan is the scenario
    if experiment['parameters_dir'] == 'HalfManhattan':
        nodes = gpd.read_file(DATA_DIR + 'Preprocessing Values//HalfManhattan//NYC_nodes.geojson')
        emergencies = [e for e in emergencies if (e.node in demand_nodes or e.node in nodes.osmid.values.tolist())]

    generator: Generators.ArrivalGenerator = Generators.CustomArrivalsGenerator([e for e in emergencies])

    sim_parameters = Models.SimulationParameters(simulation_time=experiment['simTime'],
                                                    initial_nodes=None,
                                                    speeds_df=speeds,
                                                    candidate_nodes=candidate_nodes,
                                                    hospital_nodes=hospital_nodes,
                                                    hospital_borough=hospital_borough,
                                                    nodes_with_borough=nodes_with_borough,
                                                    graph_to_demand=graph_to_demand,
                                                    demand_nodes=demand_nodes,
                                                    demand_rates=demand_rates,
                                                    ALS_tours=experiment['ambulance_distribution'][0],
                                                    BLS_tours=experiment['ambulance_distribution'][1],
                                                    mean_busytime=busy_time,
                                                    cand_cand_time=cand_cand_time,
                                                    cand_demand_time=cand_demand_time,
                                                    candidates_borough=candidate_borough,
                                                    demand_borough=demand_borough,
                                                    reachable_demand=reachable_demand,
                                                    reachable_inverse=reachable_inverse,
                                                    uber_nodes=uber_nodes,
                                                    maximum_overload_ALS=experiment['workload_limit'],
                                                    maximum_overload_BLS=experiment['workload_limit'],
                                                    uber_seconds=experiment['uberHours'] * 3600,
                                                    optimization_gap=experiment['GAP'],
                                                    max_expected_simultaneous_relocations=experiment['relocQty'],
                                                    force_static='Static' in experiment['model'],
                                                    uncovered_penalty=experiment['uncovered'],
                                                    late_response_penalty=experiment['lateResponse'],
                                                    dispatching_penalty=experiment['disaptchingPenalt'],
                                                    travel_distance_penalty=experiment['ttPenalty'],
                                                    target_relocation_time=experiment['targetReloc'],
                                                    relocation_cooldown=experiment['relocCooldown'],
                                                    max_relocation_time=experiment['maxReloc'],
                                                    max_redeployment_time=experiment['maxRedeployment'],
                                                    name=name,
                                                    random_seed=experiment['dataReplica'])

    if 'SBRDA' in experiment['model']:
        optimizer: OnlineSolvers.RelocationModel = OnlineSolvers.SORDARelocatorDispatcher()
    elif experiment['model'] == 'CoverageSORDA':
        optimizer: OnlineSolvers.RelocationModel = OnlineSolvers.CoverageSORDA()
    else:
        optimizer = ROASolver.ROA()

    simulator: Models.EMSModel = Models.EMSModel(graph, generator, parameters=sim_parameters, optimizer=optimizer, verbose=True)
    statistics = simulator.run()

    with open('{}/{}.pickle'.format(RESULTS_DIR, name), 'wb') as f:
        pickle.dump(statistics, f)