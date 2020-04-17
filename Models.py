# Import Statements
import igraph
import numpy as np
from typing import List, Optional, Callable, Dict

# Local Imports
import Events
import Solvers
import Generators
import SimulatorBasics as Sim


class Vehicle(Sim.SimulationEntity):
    """
    Vechicle that can move through a graph.


    In principle I could then extend this class to an ambulance class
    but since there won't be any other kind of vehicles in the model,
    I'll just take this as a synonym for ambulance.
    """

    def __init__(self,
                 start_node: str,
                 name: str = None,
                 default_speed:float = 5.0):
        super().__init__(name)
        self.speed:float = default_speed
        self.patient: Optional[Emergency] = None

        self.stopMoving()                   # Though obviously is not
        self.path: List[int] = []
        self.node_path: List[str] = []
        self.expected_arrival: float = 0    # Expected arrival time to next node
        self.teleportToNode(start_node)

    def teleportToNode(self, node: str):
        """
        Use Odin's powers to teleport this vehicle to a
        given node in the graph
        """
        self.pos = node
        self.onArrivalToNode(node)

    def stopMoving(self):
        self.moving: bool = False

    def onArrivalToNode(self, node: str) -> bool:
        """
        Function to be called when the vehicle arrives to a node.

        Returns a bool -> Whether or not this node was the end of
                          the vehicle's journey
        """
        self.pos = node
        self.actual_edge: Optional[int] = None
        self.from_node: str = node

        if len(self.node_path) > 0:
            self.to_node: str = self.node_path.pop(0)
            self.actual_edge = self.path.pop(0)
            return False
        else:
            self.to_node = node
            return True
    
    def onMovingToNextNode(self,
                           expected_arrival: float):
        self.expected_arrival = expected_arrival
    
    def onAssignedToEmergency(self,
                              patient: "Emergency",
                              path: List[int],
                              node_path: List[str]):
        self.patient = patient
        self.onAssignedMovement(path, node_path)
    
    def onAssignedMovement(self,
                           path: List[int],
                           node_path: List[str]):
        self.path = path
        self.node_path = node_path

    def onPathEnded(self) -> int:
        """
        Returns a status:
        0 : Means nothing is expected of the ambulance
            after this, do an assignment
        1 : Means arrived to attend an emergency to the
            location of the incident, schedule a FinishAttending
            event
        2 : Means delivered patient to medical service, remove
            emergency from the system
        """

        self.stopMoving()

        if self.patient is not None:
            if self.patient.status == 1:
                self.patient.onVehicleArrive(self)
                return 1
            elif self.patient.status == 3:
                #self.patient.markStatus(3)
                return 2
        
        return 0
        

class EMSModel(Sim.Simulator):

    def __init__(self,
                 city_graph: igraph.Graph,
                 generator_object: Generators.ArrivalGenerator,
                 assigner: Solvers.AssignmentModel,
                 initial_vehicles: int,
                 initial_nodes: List[str],
                 hospitals: Dict[int, List[str]],
                 severity_timer_function: Callable[[int], float] =
                        lambda s: 1000,
                 random_seed=420):

        super().__init__()

        self.city_graph: igraph.Graph = city_graph
        self.generator_object: Generators.ArrivalGenerator = generator_object
        self.assigner: Solvers.AssignmentModel = assigner
        self.arrival_generator = self.generator_object.generator(self)
        self.hospitals: Dict[int, List[str]] = hospitals
        self.severity_timer_function = severity_timer_function

        # Randomly initialize initial positions for ambulances if needed
        if initial_nodes is None:
            n: int = self.city_graph.vcount()
            initial_nodes = []
            for v in range(initial_vehicles):
                while True:
                    node = self.city_graph.vs[np.random.randint(n)]
                    if node not in initial_nodes:
                        initial_nodes.append(node)
                        break

        # Creating the objects for the ambulances
        self.vehicles: List[Vehicle] = list()
        assert len(initial_nodes) == initial_vehicles
        for v in range(initial_vehicles):
            self.vehicles.append(Vehicle(initial_nodes[v], 'Ambulance ' + str(v)))

        # Lists representing the state of the model
        self.activeEmergencies: List[Emergency] = list()        # All the emergencies in the system
        self.assignedEmergencies: List[Emergency] = list()      # Those which have been assigned

        # Schedule the first arrival
        self.insert(next(self.arrival_generator))

    def run(self,
            simulation_time:int = 3600):
        self.insert(Events.EndSimulationEvent(self, simulation_time))
        self.do_all_events()

    def getAvaliableVehicles(self) -> List[Vehicle]:
        return [v for v in self.vehicles if v.patient is None]
    
    def getHospitalType(self, emergency: "Emergency") -> int:
        return 1

    def getShortestPath(self, from_nodes, to_nodes) -> List[List[int]]:
        """
        Function to obtain the shortest paths based on the speed
        at the time of the simulation
        """
        return self.city_graph.get_shortest_paths(from_nodes,
                to_nodes, weights='length', output='epath')
    
    def getShortestDistances(self, from_nodes, to_node) -> np.array:
        return np.array(self.city_graph.shortest_paths(from_nodes, to_node, weights='length'))


class Emergency:
    """
    An emergency has 4 possible status:
      0: Waiting
      1: Assigned
      2: Being attended
      3: Being Carried

    and the vehicle property represents the corresponding vehicle
    whose nature will depend on the emergency status.    
    """

    N_EMERGENCIES = 0

    def __init__(self,
                 arrival_time: float,
                 node: str,
                 severity: int,
                 time_window: float = 1000):
        self.arrival_time: float = arrival_time
        self.max_time = arrival_time + time_window
        self.node: str = node
        self.severity:int = severity
        self.vechicle: Optional[Vehicle] = None

        self.status: int = 0

        Emergency.N_EMERGENCIES += 1
        self.name = 'Emergency {}'.format(Emergency.N_EMERGENCIES)
    
    def onVehicleArrive(self, vehicle: Vehicle):
        self.vehicle = vehicle
        self.markStatus(2)
        self.attending_time = 200
    
    def markStatus(self, status):
        self.status = status