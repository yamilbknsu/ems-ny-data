import Models
import Events
import Solvers
import Generators
import Experiments
import SimulatorBasics

import time
import igraph
import threading
from fastapi import FastAPI
from typing import List


class AsyncEMSModel(Models.EMSModel):

    def __init__(self,
                 city_graph: igraph.Graph,
                 generator_object: Generators.ArrivalGenerator,
                 dispatcher: Solvers.DispatcherModel,
                 repositioner: Solvers.RelocationModel,
                 parameters: Models.SimulationParameters):
        super().__init__(city_graph, generator_object, dispatcher, repositioner, parameters)

        self.threadLock = threading.Lock()

        with self.threadLock:
            self.recorded_events: List["SimulatorBasics.Event"] = []
    
    def run(self):
        if self.parameters.simulation_time is not None:
            self.insert(Events.EndSimulationEvent(self,
                                                  self.parameters.simulation_time))                         
        else:
            self.insert(Events.EndSimulationEvent(self, simulation_time))
        
        while self.events.size() > 0:
            executed_events = self.doOneEvent()

            with self.threadLock:
                for e in executed_events:
                    self.recorded_events.append(e)

    def getAndClearEvents(self):
        with self.threadLock:
            output_events = [{**{key:value if type(value) == list else str(value) for key, value in e.__dict__.items()}, **{'type':type(e).__name__}} for e in self.recorded_events]
            self.recorded_events = []
            return output_events

# Initializing the model
emsModel = AsyncEMSModel(Experiments.graph, Experiments.generator,
                           Solvers.PreparednessDispatcher(),
                           Solvers.MaxExpectedSurvivalRelocator(),
                           Experiments.sim_parameters)

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