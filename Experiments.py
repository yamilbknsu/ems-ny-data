# %% Import Statements
import pickle
import igraph

# Internal Imports
import Models
import Events
import Solvers
import Generators

# %% Graph testing
DATA_DIR = 'C://Users//Yamil//Proyectos//Proyectos en Git//' \
    + 'Memoria Ambulancias//ems-ny-data//Generated Shapefiles//'

with open(DATA_DIR + 'NYC_graph.pickle', 'rb') as file:
    graph: igraph.Graph = pickle.load(file)

# %% Basic Debugging

# Hard-coding emergency arrivals
emergencies = [Events.EmergencyArrivalEvent(None, 20, '42860963', 1),
               Events.EmergencyArrivalEvent(None, 250, '2924892679', 1),
               Events.EmergencyArrivalEvent(None, 70, '42822341', 1)]
generator: Generators.ArrivalGenerator = \
           Generators.CustomArrivalsGenerator([e for e in emergencies])
simulator: Models.EMSModel = Models.EMSModel(graph, generator, Solvers.NearestAssigner(), 1, # noqa E501
                            ['42862892'], {1: ['42855606', '42889788']})
simulator.run()
