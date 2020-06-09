from flask import Flask, jsonify
from threading import Lock, Thread
import pickle
import igraph

# Internal Imports
import Models
import Events
import Solvers
import Generators
import SimulatorBasics as Sim

app = Flask(__name__)
SimThread = None
SimLock = None
simulator = None

# Connection lifecicle routes
@app.route("/")
def hello():
    return "Hello World!"


@app.route("/init")
def init_conn():
    return jsonify(alive=True, message='OK!')


@app.route("/alive")
def alive_request():
    return jsonify(alive=True, message='OK!')

# Simulation related routes
@app.route("/startSim")
def start_request():
    result = startTestThread()
    return jsonify(**result)


@app.route("/updateSimEvents")
def update_request():
    event_list = getExecutedEvents()
    return jsonify(events=event_list)


class BackEndSimulator(Models.EMSModel):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.event_recorder = Sim.EventRecorder()

    def getRecorder(self):
        return self.event_recorder

    def do_all_events(self):
        global SimLock

        while self.events.size() > 0:
            e = self.events.remove_first()
            if self.verbose:
                print(Sim.secondsToTimestring(e.time), e.message)

            self.time = e.time

            # Handling chained events
            chained_event = e.execute(self)
            to_record_events = [e]

            while chained_event is not None:
                to_record_events.append(chained_event)

                if self.verbose:
                    print('{:>17}'.format('### Chained:'), chained_event.message) # noqa E501
                chained_event = chained_event.execute(self)

            # Recording event execution
            if SimLock is not None:
                with SimLock:
                    for event in to_record_events:
                        self.event_recorder.record(event)

        self.recoverMetrics()
        return self.metrics


def startTestThread():
    global SimThread
    global simulator
    global SimLock

    simulator = createBackendSimulator()

    SimThread = Thread(target=startSimulator, args=[simulator])
    SimLock = Lock()
    SimThread.start()

    return {'message': 'Thread Started!'}


def getExecutedEvents():
    global simulator
    global SimLock

    if simulator is not None and SimLock is not None:
        events = []
        with SimLock:
            events = simulator.getRecorder().getEvents()
        return events


def createBackendSimulator():
    DATA_DIR = 'C://Users//Yamil//Proyectos//Proyectos en Git//' \
             + 'Memoria Ambulancias//ems-ny-data//Generated Shapefiles//'

    with open(DATA_DIR + 'NYC_graph.pickle', 'rb') as file:
        graph: igraph.Graph = pickle.load(file)

    # Hard-coding emergency arrivals
    emergencies = [Events.EmergencyArrivalEvent(None, 20, '42860963', 1),
                   Events.EmergencyArrivalEvent(None, 250, '2924892679', 1),
                   Events.EmergencyArrivalEvent(None, 70, '42822341', 1)]
    generator: Generators.ArrivalGenerator = \
        Generators.CustomArrivalsGenerator([e for e in emergencies])

    simulator: Models.EMSModel = BackEndSimulator(graph, generator, Solvers.NearestDispatcher(), 1, # noqa E501
                                ['42862892'], {1: ['42855606', '42889788']})
    return simulator


def startSimulator(simulator):
    simulator.run(4800)


if __name__ == "__main__":
    app.run()
