import Models
import Events
import OnlineSolvers
import Generators
import SimulatorBasics

import igraph
import pickle
import json
import threading
import pandas as pd
import geopandas as gpd
from fastapi import FastAPI
from typing import List


def secondsToTimestring(seconds: float) -> str:
    days = int(seconds // 86400)
    seconds = seconds % 86400
    hours = int(seconds // 3600)
    seconds = seconds % 3600
    minutes = int(seconds // 60)
    seconds = seconds % 60

    return 'day {} {:02d}:{:02d}:{:0>5}'.format(days, hours, minutes, '{:.2f}'.format(seconds))


class AsyncEMSModel(Models.EMSModel):

    def __init__(self,
                 city_graph: igraph.Graph,
                 generator_object: Generators.ArrivalGenerator,
                 optimizer,
                 parameters: Models.SimulationParameters):
        super().__init__(city_graph, generator_object, optimizer, parameters)

        self.threadLock = threading.Lock()

        with self.threadLock:
            self.recorded_events: List["SimulatorBasics.Event"] = []

    def run(self):
        if self.parameters.simulation_time is not None:
            self.insert(Events.EndSimulationEvent(self,
                                                  self.parameters.simulation_time))
        else:
            self.insert(Events.EndSimulationEvent(self, 3600 * 24))

        while self.events.size() > 0:
            executed_events = self.doOneEvent()

            with self.threadLock:
                for e in executed_events:
                    self.recorded_events.append(e)

    def getAndClearEvents(self):
        with self.threadLock:
            output_events = [{**{key: value if type(value) == list else str(value) for key, value in e.__dict__.items()}, **{'type': type(e).__name__}} for e in self.recorded_events]
            self.recorded_events = []
            return output_events


# loading necesary parameters
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

experiment = {'day': 'monday', 'dataReplica': 0, 'simTime': 24 * 3600, 'relocatorModel': 'survivalNoExp', 'dispatcher': 'preparedness',
              'relocate': False, 'ambulance_distribution': [355, 802], 'workload_restriction': True, 'workload_limit': .4, 'useUber': True, 'GAP': 0.05,
              'parameters_dir': 'Base', 'relocation_period': 600, 'uberRatio': 1}

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
with open(DATA_DIR + 'Preprocessing Values//{}//neighborhood_candidates.pickle'.format(experiment['parameters_dir']), 'rb') as file:
    neighborhood_candidates = pickle.load(file)
with open(DATA_DIR + 'Preprocessing Values//{}//neighborhood.pickle'.format(experiment['parameters_dir']), 'rb') as file:
    neighborhood = pickle.load(file)
with open(DATA_DIR + 'Preprocessing Values//{}//neighborhood_k.pickle'.format(experiment['parameters_dir']), 'rb') as file:
    neighborhood_k = pickle.load(file)
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
with open(DATA_DIR + 'Arrival Events//{}//LS19//strep_{}.pickle'.format('Friday' if experiment['day'] == 'friday' else 'Monday', experiment['dataReplica']), 'rb') as file:               # noqa E501
    emergencies: List['Events.EmergencyArrivalEvent'] = pickle.load(file)

# Importing high severity emergencies file
with open(DATA_DIR + 'Arrival Events//{}//HS19//strep_{}.pickle'.format('Friday' if experiment['day'] == 'friday' else 'Monday', experiment['dataReplica']), 'rb') as file:               # noqa E501
    emergencies += pickle.load(file)


generator: Generators.ArrivalGenerator = Generators.CustomArrivalsGenerator([e for e in emergencies])

sim_parameters = Models.SimulationParameters(simulation_time=.5 * 3600,
                                             initial_nodes=None,
                                             speeds_df=speeds,
                                             candidate_nodes=candidate_nodes,
                                             hospital_nodes=hospital_nodes,
                                             hospital_borough=hospital_borough,
                                             nodes_with_borough=nodes_with_borough,
                                             graph_to_demand=graph_to_demand,
                                             demand_nodes=demand_nodes,
                                             demand_rates=demand_rates,
                                             ALS_tours=355,
                                             BLS_tours=802,
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

# optimizer: OnlineSolvers.RelocationModel = OnlineSolvers.UberRelocatorDispatcher()
# optimizer: OnlineSolvers.RelocationModel = ROASolver.ROA()
optimizer: OnlineSolvers.RelocationModel = OnlineSolvers.AlternativeUberRelocatorDispatcher()

# emsModel: Models.EMSModel = Models.EMSModel(graph, generator, optimizer, sim_parameters, verbose=True)# Initializing the model
emsModel = AsyncEMSModel(graph, generator, optimizer, sim_parameters)
#emsModel.run()
#events = emsModel.getAndClearEvents()
## Serializing json  
#json_object = json.dumps(events, indent = 4)
#with open("Maps and others/30minutesEvents.json", "w") as outfile: 
#    outfile.write(json_object)

# The simulation Thread
simulationThread = threading.Thread(target=emsModel.run, daemon=True)


app = FastAPI()


@app.get('/')
async def root():
    return {'message': 'Backend is up and ok!'}


@app.get('/init')
async def init():
    return {'message': 'Backend is up and ok!', 'alive': True}


@app.get('/alive')
async def alive():
    return {'message': 'Backend is up and ok!', 'alive': True}


@app.get('/startSim')
async def startSim():
    simulationThread.start()
    return {'message': 'Simulation started succesfully!'}


@app.get('/getEvents')
async def getEvents():
    outgoingEvents = emsModel.getAndClearEvents()
    return {'message': 'Sending {} events'.format(len(outgoingEvents)), 'events': outgoingEvents}
