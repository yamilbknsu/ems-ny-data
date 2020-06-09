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
                 vehicle_type: int,
                 name: str = None,
                 default_speed: float = 5.0):
        super().__init__(name)
        self.speed: float = default_speed
        self.patient: Optional[Emergency] = None
        self.type = vehicle_type

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
                # self.patient.markStatus(3)
                return 2
        
        return 0


class SimulationParameters:

    def __init__(self,
                 demand_rates: List[np.array],
                 mean_busytime: List[np.array],
                 cand_cand_time: List[np.array],
                 cand_demand_time: List[np.array],
                 speeds_df,
                 neighborhood: List[Dict[str, List[str]]],
                 neighborhood_candidates,
                 neighbor_k: List[Dict[str, int]],
                 reachable_demand: List[Dict[str, List[str]]],
                 reachable_inverse: List[Dict[str, List[str]]],
                 simulation_time: float = 3600,
                 time_periods: List[int] = [7, 3, 6, 3, 5],
                 vehicle_types: int = 2,
                 n_vehicles: List[List[int]] = [[312, 505]] * 5,
                 initial_nodes: Optional[List[List[str]]] = [['42862892'] * 312, ['42862892'] * 505],         
                 hospital_nodes: Dict[int, List[str]] = {1: ['42855606', '42889788'],
                                                         2: ['42855606', '42889788'],
                                                         3: ['42855606', '42889788']},
                 candidate_nodes: List[str] = ['42862892'],
                 demand_nodes: List[str] = ['42858015'],
                 random_seed: float = 420):
        """
        Parameters for the simulation
        -----------------------------

        simulation_time float: Simulation time limit
        time_periods [t]: The length of the divisions of one day (in hours)
        vehicle_types int: Number of vehicle types
        n_vehicles [t, v]: The amount of vehicles v at time shift t
        initial_nodes [v, n_vehicles[0,v]]: Initial position for the ambulances
        hospital_nodes Dict[s: [h_s]]: Nodes for the hospitals in the graph

        candidate_nodes List[str]: Nodes where an ambulance can wait
        demand_nodes List[str]: Nodes that approximate a spatial demand
        demand_rates List[Array[demand_node, t]]: List of hourly demand rates (by severity)
        mean_busytime List[Array[demand_node, t]]: List of mean total busy time of an 
                                                ambulance for emergency (by severity)
        cand_cand_time List[Array[candidate, candidate]]: Travel time between candidate nodes (by time period)
        cand_demand_time List[Array[candidate, demand]]: Travel time from candidate to demand nodes (by time period)
        neighborhood List[Dict[str: List[str]]]: Neighborhood of a candidate point (by time period)
        neighbor_k List[Dict[str: int]]: Number of candidate points in a neighborhood (by time period)
        reachable_demand List[Dict[str: List[str]]]: The list of demand points reachable from 
                                            a candidate point in less that a threshold (by time period)
        reachable_inverse List[Dict[str: List[str]]]: The list of candidate points tht reach a 
                                                demand point in less that a threshold (by time period)

        random_seed float: Seed for random generator
        """

        # General parameters
        self.simulation_time = simulation_time
        self.time_periods = time_periods
        self.vehicle_types = vehicle_types
        self.n_vehicles = n_vehicles
        self.initial_nodes = initial_nodes
        self.hospital_nodes = hospital_nodes
        self.random_seed = random_seed

        # Routing parameters
        self.speeds_df = speeds_df
        self.candidate_nodes = candidate_nodes
        self.demand_nodes = demand_nodes
        self.demand_rates = demand_rates
        self.mean_busytime = mean_busytime
        self.cand_cand_time = cand_cand_time
        self.cand_demand_time = cand_demand_time
        self.neighborhood = neighborhood
        self.neighborhood_candidates = neighborhood_candidates
        self.neighbor_k = neighbor_k
        self.reachable_demand = reachable_demand
        self.reachable_inverse = reachable_inverse

        # Set the RNG seed
        np.random.seed(self.random_seed)

        #self.reachable_matrix = np.array([[[1 if c_node in self.reachable_inverse[t][d_node]
        #                                    else 0
        #                                    for c_node in self.candidate_nodes]
        #                                   for d_node in self.demand_nodes]
        #                                  for t in range(len(time_periods))])
        
    def getSpeedList(self, time_period):
        return list(self.speeds_df['p' + str(time_period+1) + 'n'])

class EMSModel(Sim.Simulator):

    def __init__(self,
                 city_graph: igraph.Graph,
                 generator_object: Generators.ArrivalGenerator,
                 dispatcher: Solvers.DispatcherModel,
                 parameters: SimulationParameters):

        super().__init__()

        self.city_graph: igraph.Graph = city_graph
        self.generator_object: Generators.ArrivalGenerator = generator_object
        self.assigner: Solvers.DispatcherModel = dispatcher
        self.arrival_generator = self.generator_object.generator(self)

        self.parameters = parameters
        self.hospitals: Dict[int, List[str]] = parameters.hospital_nodes

        # Randomly initialize initial positions for ambulances if needed
        if parameters.initial_nodes is None:
            n: int = self.city_graph.vcount()
            parameters.initial_nodes = []
            for m in range(parameters.vehicle_types):
                pos_list = []
                for v in range(parameters.n_vehicles[0][m]):
                    while True:
                        node = self.city_graph.vs[np.random.randint(n)]['name']
                        if node not in parameters.initial_nodes:
                            pos_list.append(node)
                            break
                parameters.initial_nodes.append(pos_list)

        # Creating the objects for the ambulances
        self.vehicles: List[Vehicle] = list()
        for v in range(parameters.vehicle_types):
            assert len(parameters.initial_nodes[v]) == parameters.n_vehicles[0][v]

        n = 0
        for v in range(parameters.vehicle_types):
            for m in range(parameters.n_vehicles[0][v]):
                self.insert(Events.AmbulanceArrivalEvent(self, 0, parameters.initial_nodes[v][m],           
                            Vehicle(parameters.initial_nodes[v][m], v, 'Ambulance ' + str(n))))             
                n += 1

        # Lists representing the state of the model
        self.activeEmergencies: List[Emergency] = list()        # All the emergencies in the system         
        self.assignedEmergencies: List[Emergency] = list()      # Those which have been assigned            

        # Schedule the first arrival
        self.insert(next(self.arrival_generator))

    def run(self,
            simulation_time: int = 3600):
        if self.parameters.simulation_time is not None:
            self.insert(Events.EndSimulationEvent(self,
                                                  self.parameters.simulation_time))                         
        else:
            self.insert(Events.EndSimulationEvent(self, simulation_time))
        self.do_all_events()

    def getAvaliableVehicles(self, v_type: Optional[int] = None) -> List[Vehicle]:                          
        if v_type is not None:
            return [v for v in self.vehicles if v.patient is None and v.type == v_type]                     
        else:
            return [v for v in self.vehicles if v.patient is None]

    def getHospitalType(self, emergency: "Emergency") -> int:
        return 1

    def getShortestPath(self, from_nodes, to_nodes) -> List[List[int]]:
        """
        Function to obtain the shortest paths based on the speed
        at the time of the simulation
        """
        return self.city_graph.get_shortest_paths(from_nodes, to_nodes,
                                                  weights='length',
                                                  output='epath')

    def getShortestDistances(self, from_nodes, to_node) -> np.array:
        return np.array(self.city_graph.shortest_paths(from_nodes, to_node, weights='length'))              

    def computeSystemPreparedness(self,
                                  vehicle_positions: Optional[List[List[str]]] = None,
                                  travel_matrix: Optional[np.array] = None):
        """
        Expecting a list of lists, containing a list of nodes for each vehicle type
        Will use simulator.parameters for calculations
        """

        if travel_matrix is None:

            if vehicle_positions is None:
                vehicle_positions = [[ambulance.to_node
                                    for ambulance in self.getAvaliableVehicles(v_type=v)]                     
                                    for v in range(self.parameters.vehicle_types)]

            travel_matrix = [self.city_graph.shortest_paths(vehicle_positions[v],
                                                            self.parameters.demand_nodes,
                                                            self.parameters.getSpeedList(self.timePeriod()))
                                 for v in range(self.parameters.vehicle_types)]

        # Emergencies per hour per demand node
        demand_rates = np.array([self.parameters.demand_rates[0].loc[self.timePeriod()+1, self.parameters.demand_nodes],
                                 self.parameters.demand_rates[1].loc[self.timePeriod()+1, self.parameters.demand_nodes]])
        # Mean time per emergency per demand node
        busytime_means = np.array([self.parameters.mean_busytime[0].loc[self.timePeriod()+1, self.parameters.demand_nodes],
                                   self.parameters.mean_busytime[1].loc[self.timePeriod()+1, self.parameters.demand_nodes]])

        # Ponderation of each type of emergency at each demand node
        demand_ratios = demand_rates/np.where(np.sum(demand_rates, axis = 0) == 0, 1, np.sum(demand_rates, axis = 0))

        # Compute the amount of vehicles available per node per emergency level
        valid_indexes = [np.where(np.array(travel_matrix[0]) < 8*60), np.where(np.array(travel_matrix[1]) < 8*60)]
        number_available = np.array([[len(np.where(valid_indexes[v][0] == i)[0]) if len(np.where(valid_indexes[v][0] == i)[0]) > 0 else 1e-10
                                     for i in range(len(self.parameters.demand_nodes))]
                                     for v in range(self.parameters.vehicle_types)])

        # Compute final preparedness
        preparedness = 1 - np.sum((demand_ratios * demand_rates * busytime_means)/number_available, axis = 0)
        preparedness[preparedness < 0] = 0

        return np.sum(preparedness)


    def _timePeriod(self, seconds, periods):
        seconds = (seconds % 86400) // 3600
        for i, l in enumerate(periods):
            if seconds < l:
                return i
    
    def timePeriod(self):
        return self._timePeriod(self.now(), self.parameters.time_periods)
    
    def __str__(self):
        return 'Simulator'


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
    
    def __str__(self):
        return self.name