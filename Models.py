# Import Statements
import time
import random
import igraph
import numpy as np
import pandas as pd
import geopandas as gpd
from collections import defaultdict
from typing import List, Optional, Callable, Dict, Tuple

# Local Imports
import Events
import Solvers
import Generators
import SimulatorBasics as Sim


class SimulationParameters:
    """
    A SimulationParameters instance is meant to hold all the parameters
    for the simulation, most of them are constants (like the id's of the)
    candidate and demand nodes, but there are others that are computed on
    the fly. For the latter ones, compute functions are provided in this
    class.

    In a normal setting, you have one instance of this class in order
    to represent one configuration or replica of the system.

    The main abstractions of the model that are necessary to understand the
    data structures found here are:

    - Nodes in the graph of NYC are indetified by the osmid attribute, which
      we pass along as a string.
    - Similarly, edges of this graph are identified by the edgeid attribute
      of the spatial object, also referenced here as a string.
    - Emergency events were divided into High and Low severity groups. This
      means that demand related parameters are usually divided in two (usually
      as a list with two elements). High priority is always the index 0.
    - It's assumed that a working day is divided into time periods with fixed
      duration, and that there is a distribution describing the average speed
      of each street for each one. So, all of the parameters that are dependent
      on this speed, are expected to be a list with as much elements as time periods.
    - Simulation time is always in seconds.
    """    

    def __init__(self,
                 demand_rates: List["pd.DataFrame"],
                 mean_busytime: List["pd.DataFrame"],
                 cand_cand_time: List["np.array"],
                 cand_demand_time: List["np.array"],
                 hospital_nodes: Dict[int, List[str]],
                 hospital_borough,
                 candidate_nodes: List[str],
                 demand_nodes: List[str],
                 speeds_df: "pd.DataFrame",
                 ALS_tours: float,
                 BLS_tours: float,
                 neighborhood: List[Dict[str, List[str]]],
                 neighborhood_candidates: List[Dict[str, List[str]]],
                 neighbor_k: List[Dict[str, int]],
                 nodes_with_borough: "gpd.GeoDataFrame",
                 candidates_borough: Dict[int, List[str]],
                 demand_borough: Dict[int, List[str]],
                 reachable_demand: List[Dict[str, List[str]]],
                 reachable_inverse: List[Dict[str, List[str]]],
                 uber_nodes: Dict[str, str],
                 vehicle_shift: Callable = lambda _: 8*3600,
                 vehicle_arrival_deviation: float = 3600,
                 is_uber_available: bool = False,
                 uber_low_severity_ratio: float = .1,
                 simulation_time: float = 3600,
                 relocation_optimization: bool = True,
                 relocation_period: float = 600,
                 optimization_gap: float = .1,
                 apply_workload_restriction: bool = True,
                 time_periods: List[float] = [7, 3, 6, 3, 5],
                 time_shifts: List[float] = [8,8,8],
                 vehicle_types: int = 2,
                 initial_nodes: Optional[List[List[str]]] = None,
                 overload_penalty: float = 100,
                 maximum_overload_ALS = .4,
                 maximum_overload_BLS = .4,
                 random_seed: float = 420):
        """
        A SimulationParameters instance is meant to hold all the parameters
        for the simulation, most of them are constants (like the id's of the)
        candidate and demand nodes, but there are others that are computed on
        the fly. For the latter ones, compute functions are provided in this
        class.

        In a normal setting, you have one instance of this class in order
        to represent one configuration or replica of the system.

        Refer to class documentation for some abstrations of the model useful
        for understanding these data structures.

        In order to facilitate even more the understanding of this parameters,
        the expected size of the matrices is defined in brackets using this
        notation:

        C: Number of candidadte Nodes
        D: Number of demand Nodes
        T: Number of time periods
        V: Number of ambulance types
        S: Number of emergency categories
        B: Number of Boroughs
        H: Number of hospital nodes

        Note: This class also holds the value of Q and P, which are matrices computed
        on the fly and act as parameters for the optimization models. Similarly,
        the ambulance distribution between borough is approximated using an heuristic
        that runs on the initialization function of this class.

        Args:
            demand_rates (List[DataFrame]): [T, S, D] Hourly demand rate for each node
            mean_busytime (List[DataFrame]): [T, S, D] Mean busy time taken from an ambulance
                                            attending an emergency coming for each node. This matrix
                                            is updated during the simulation with simulated values.
            cand_cand_time (List[numpy.Array]): [T, C, C] Travel time between candidate nodes
            cand_demand_time (List[numpy.Array]): [T, C, D] Travel time between candidate and demand nodes
            hospital_nodes (Dict[int, List[str]]): [S: H] Hospitals nodes that can attend emergencies
                                                          of type S
            candidate_nodes (List[str]): [C] List of candidate nodes (by osmid)
            demand_nodes (List[str]): [D] List of demand nodes (by osmid)
            speeds_df (pd.DataFrame): Dataframe with columns: edgeid and p+'x'+n for x in range(1,T) 

            TODO: FIX THIS
            n_vehicles (List[List[int]]): [T, V] Number of ambulances of each type available for
                                                 each time period
            ambulance_distribution (List[List[int]]): The amount of ambulances of each type at each borough. 
                                                      The sum should be equal to the values at n_vehicles.



            neighborhood (List[Dict[str, List[str]]]): [C: ?] List of demand nodes inside the neighborhood
                                                              of each candidate node
            neighborhood_candidates (List[Dict[str, List[str]]]): [C: ?] List of candidate nodes sharing the neighborhood
                                                                         of each candidate node
            neighbor_k (List[Dict[str, int]]): [C: 1] Number of demand points inside the neighborhood of each candidate node
            nodes_with_borough (gpd.GeoDataFrame): [D rows] Dataframe with spatial info of each demand node and a column with
                                                            its corresponding borough (this is the only column used in the model
                                                            but it was the most direct way of include it)
            candidates_borough (Dict[int, List[str]]): [B: ?] List of candidate nodes in each borough
            demand_borough (Dict[int, List[str]]): [B: ?] List of demand nodes in each borough
            reachable_demand (List[Dict[str, List[str]]]): [C: ?] List of reachable demand nodes for each candidate node in less
                                                                  than 8 minutes
            reachable_inverse (List[Dict[str, List[str]]]): [D: ?] List of candidate nodes that are able to reach each demand node
                                                                   in less than 8 minutes
            uber_nodes (Dict[str, str]): A dict with each node on the graph as a key and the node where a uber might appear
                                         if an emergency calls one from the key node as value of the dict.
            is_uber_available (bool, optional): Whether or not ride handling services is available for a percentage of
                                              non life-thretening non urgen calls. Defaults to False.
            uber_low_severity_ratio (float, optional): The percentage of non lifethretening non urgent calls to be
                                                       assigned to an uber trip. Defaults to 0.1.
            simulation_time (float, optional): Time limit for the simulation .Defaults to 3600.
            relocation_optimization (bool, optional): Whether or not to perform online relocation. Defaults to True.
            relocation_period (float, optional): Time in seconds between each relocation process. Defaults to 600.
            optimization_gap (float, optional): MIPGap parameter for gurobi solver. Defaults to .1
            apply_workload_restriction (bool, optional): Whether or not to apply workload restrictions in the
                                                         optimization process. Defaults to True.
            time_periods (List[float], optional): [T] Duration in hours of each time period. Defaults to [7, 3, 6, 3, 5].
            time_shifts (List[float], optional): [T] Duration in hours of each work shift. Defaults to [8, 8, 8].
            vehicle_types (int, optional): Number of vehicle types. Defaults to 2.
            initial_nodes (List[List[str]], optional): [V, ?] Initial nodes for  the ambulances. Defaults to None.
            overload_penalty (float, optional): Penalty parameter for relocation overload exceeding Defaults to 1000.
            maximum_overload_ALS (float, optional): Fraction of the time periods that an ALS vehicle is allowed
                                                    spend on relocation. Defaults to .5.
            maximum_overload_BLS (float, optional): Fraction of the time periods that an BLS vehicle is allowed
                                                    spend on relocation. Defaults to .5.
            random_seed (float, optional): Seed for random generator. Defaults to 420.
        """

        # General parameters
        self.simulation_time = simulation_time
        self.relocation_optimization = relocation_optimization
        self.relocation_period = relocation_period
        self.optimization_gap = optimization_gap
        self.apply_workload_restriction = apply_workload_restriction
        self.time_periods = time_periods
        self.time_shifts = time_shifts
        self.vehicle_types = vehicle_types
        self.ALS_tours = ALS_tours
        self.BLS_tours = BLS_tours
        self.vehicle_shift = vehicle_shift
        self.vehicle_arrival_deviation = vehicle_arrival_deviation
        self.initial_nodes = initial_nodes
        self.hospital_nodes = hospital_nodes
        self.hospital_borough = hospital_borough
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
        self.candidates_borough = candidates_borough
        self.demand_borough = demand_borough
        self.reachable_demand = reachable_demand
        self.reachable_inverse = reachable_inverse
        self.uber_nodes = uber_nodes
        self.is_uber_available = is_uber_available
        self.uber_low_severity_ratio = uber_low_severity_ratio
        self.nodes_with_borough = nodes_with_borough

        # Hyper-Parameters
        self.overload_penalty = overload_penalty
        self.maximum_overload_ALS = maximum_overload_ALS
        self.maximum_overload_BLS = maximum_overload_BLS

        # Set the RNG seed
        np.random.seed(self.random_seed)
        random.seed(self.random_seed)

        # Initialization of some parameters
        # ---------------------------------

        # It was necessary to acknowledge the fact that the borough 0 exists (those points outside the border)
        # So it's set to be those nodes that don't fit inside any other borough
        candidates_with_borough = candidates_borough[1] + candidates_borough[2] + candidates_borough[3] + \
                                  candidates_borough[4] + candidates_borough[5]
        self.candidates_borough[0] = list(set(candidate_nodes) - set(candidates_with_borough))

        # Initialize Q and P matrices
        self.computeQandP(0)

        # Compute ambulance distribution
        ALS_temporal_demand_factor = np.array([np.sum(self.demand_rates[0].loc[t, :]) for t in range(1, 1 + len(time_periods))])
        ALS_temporal_demand_factor /= np.sum(ALS_temporal_demand_factor)
        BLS_temporal_demand_factor = np.array([np.sum(self.demand_rates[1].loc[t, :]) for t in range(1, 1 + len(time_periods))])
        BLS_temporal_demand_factor /= np.sum(BLS_temporal_demand_factor)

        cumulative_time_periods = [sum(self.time_periods[:l+1]) for l in range(len(self.time_periods))]
        cumulative_time_shifts = [sum(self.time_shifts[:l+1]) for l in range(len(self.time_shifts))]

        used_hours = np.zeros(len(self.time_periods))
        ALS_final_temporal_factor = []
        BLS_final_temporal_factor = []
        for ts in cumulative_time_shifts:
            i = 0
            for tp in cumulative_time_periods:
                if tp >= ts:
                    break
                i += 1
            factor = np.array([np.clip((ts - sum(self.time_periods[:j]))/self.time_periods[j], a_min=0, a_max=1) if j <= i else 0 for j in range(len(self.time_periods))]) - used_hours

            ALS_final_temporal_factor.append(sum(ALS_temporal_demand_factor * factor))
            BLS_final_temporal_factor.append(sum(BLS_temporal_demand_factor * factor))

            used_hours += factor

        ALS_temporal_distribution = np.trunc(ALS_tours * np.array(ALS_final_temporal_factor))
        BLS_temporal_distribution = np.trunc(BLS_tours * np.array(BLS_final_temporal_factor))

        borough_demand = np.array([np.sum(self.demand_rates[0][d][1] for d in demand_borough[b]) for b in range(1,6)])
        borough_demand = borough_demand/np.sum(borough_demand)

        # Allocate the ambulances to each borough each time shift
        self.ambulance_distribution = [[np.round(ALS_temporal_distribution[t]*borough_demand), np.round(BLS_temporal_distribution[t]*borough_demand)] for t in range(len(time_shifts))]
        
        #self.ambulance_distribution = [[0, 70, 74, 89, 65, 14],
        #                               [0, 114, 126, 140, 105, 20]]

    def getSpeedList(self, time_period: int):
        """
        Get the list of the speeds for each edge on the graph
        for the corresponding time period, in the requested order.

        Args:
            time_period (int): Index of the time period (0-indexed)

        Returns:
            (list): list of speeds (in meters/second)
        """
        return list(self.speeds_df['p' + str(time_period+1) + 'n'])

    def computeQandP(self, t):
        """
        Compute Q and P matrices for time period t, which are parameters
        for the optimization process.

        They are computed as:

        Q_ij = max(0, 1 - (sum(demand_rate[i]*mean_busy_time[i] for i in demand_nodes)/ k) ** k)

        (The only difference between Q and P is that Q is only considering high severity demand,
         while P considers only low severity demand)
         
        Args:
            t (int): Time period
        """
        neighbor_k = self.neighbor_k
        neighborhood = self.neighborhood
        D_rates = self.demand_rates
        Busy_rates = self.mean_busytime
        C = self.candidate_nodes

        self.Q = np.array([[np.sum(D_rates[0].loc[t+1, neighborhood[t][j]] * 
                                        Busy_rates[0].loc[t+1, neighborhood[t][j]]) for j in C]]).T
        self.Q = self.Q @ np.array([[1/k if k > 0 else 0 for k in range(max(neighbor_k[t].values())+1)]])
        self.Q = 1-np.power(self.Q, [k if k > 0 else 0 for k in range(max(neighbor_k[t].values())+1)])
        self.Q[self.Q < 0] = 0

        self.P = np.array([[np.sum(D_rates[0].loc[t+1, neighborhood[t][j]] * 
                                        Busy_rates[0].loc[t+1, neighborhood[t][j]]) for j in C]]).T
        self.P = self.P @ np.array([[1/k if k > 0 else 0 for k in range(max(neighbor_k[t].values())+1)]])
        self.P = 1-np.power(self.P, [k if k > 0 else 0 for k in range(max(neighbor_k[t].values())+1)])
        self.P[self.P < 0] = 0



class Vehicle(Sim.SimulationEntity):
    """
    Object representing a vechicle that can move through a graph.
    This object holds information about the current state of the vehicle.
    This includes, among other things, the list of edge elements that
    are part of the actual route, a reference to an emergency instance
    if attending any and the actual position of the vehicle.


    In principle we could then extend this class to an ambulance class
    but since there won't be any other kind of vehicles in the model,
    We'll just take this as a synonym for ambulance.

    Properties of the class:
    
    patient (Emergency, optional): Emergency object of the actual patient
    """

    def __init__(self,
                 start_node: str,
                 vehicle_type: int,
                 borough: int,
                 uber: bool = False,
                 name: str = None):
        """
        Args:
            start_node (str): Starting node for the ambulance.
            vehicle_type (int): int describing which type of vehicle this is (ALS: 0, BLS: 1).
            borough (int): Borough to which this vehicle belongs to.
            name (str, optional): [description]. Defaults to None.
        """
        super().__init__(name)
        self.patient: Optional[Emergency] = None
        self.type = vehicle_type
        self.borough = borough
        self.isUber = uber

        self.stopMoving()                   # Though obviously is not
        self.path: List[int] = []
        self.node_path: List[str] = []
        self.expected_arrival: float = 0    # Expected arrival time to next node    
        self.teleportToNode(start_node)
        self.cleaning: bool = False
        self.leaving: bool = False

        # The time spent repositioning
        self.reposition_workload: float = 0
        self.station = start_node
        self.arrival_time = 0

        # Statistics
        self.statistics: Dict[str, Sim.Statistic] = {}

        self.statistics['RelocationTime'] = Sim.StateStatistic('RelcationTime{}'.format(self.name))             # At start moving event and change shift
        self.statistics['MetersDriven'] = Sim.StateStatistic('MetersDriven{}'.format(self.name))
        self.statistics['EmergenciesServed'] = Sim.CounterStatistic('EmergenciesServed{}'.format(self.name))    # At emergency leaving event
        self.statistics['State'] = Sim.StateStatistic('VehicleState{}'.format(self.name))
        self.statistics['TimeInSystem'] = Sim.CounterStatistic('TimeInSystem{}'.format(self.name))

        # State posibilities
        # 0: Idle                   @ Ambulance end trip event and Ambulance end cleaning
        # 1: Repositioning          @ Ambulance start moving event
        # 2: Attending              @ Assigned event
        
        # This is a record of all the movement an ambulance has
        # Each element is a tuple with the form:
        # (time, from_node, to_node, emergency, emergency severity, hospital)
        self.record: List[Tuple] = []

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
            self.stopMoving()
            return True
    
    def onMovingToNextNode(self,
                           expected_arrival: float):
        self.moving = True
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


class EMSModel(Sim.Simulator):

    def __init__(self,
                 city_graph: igraph.Graph,
                 generator_object: Generators.ArrivalGenerator,
                 dispatcher: Solvers.DispatcherModel,
                 repositioner: Solvers.RelocationModel,
                 parameters: SimulationParameters,
                 **kwargs):

        super().__init__(**kwargs)

        self.city_graph: igraph.Graph = city_graph
        self.generator_object: Generators.ArrivalGenerator = generator_object
        self.assigner: Solvers.DispatcherModel = dispatcher
        self.repositioner: Solvers.RelocationModel = repositioner
        self.arrival_generator = self.generator_object.generator(self)

        self.parameters = parameters
        self.hospitals: Dict[int, List[str]] = parameters.hospital_nodes

        
        # State variables
        # ---------------
        # The last reposition status
        self.ambulance_stations: Dict[int, List[List[str]]] = {}

        # The amount of uber calls used
        self.uber_calls = 0

        # Covering state, useful for preparedness
        self.covering_state: Dict[int, Dict[Vehicle, Tuple[str, List[float]]]] = {}

        # A record of the time spent on an emergency on a specific node
        # (time_period, severity, node) = [times]
        self.time_records: Dict[Tuple[int, int, str], List[float]] = defaultdict(list)


        # Initialization processes
        # ------------------------
        # Randomly initialize initial positions for ambulances if needed
        if parameters.initial_nodes is None:
            n: int = self.city_graph.vcount()
            parameters.initial_nodes = []
            for m in range(parameters.vehicle_types):
                pos_list = []
                for v in range(int(sum(parameters.ambulance_distribution[0][m]))):
                    while True:
                        node = self.city_graph.vs[np.random.randint(n)]['name']
                        if node not in parameters.initial_nodes:
                            pos_list.append(node)
                            break
                parameters.initial_nodes.append(pos_list)

        # Creating the objects for the ambulances
        self.ubers: List[Vehicle] = list()
        self.vehicles: List[Vehicle] = list()
        for v in range(parameters.vehicle_types):
            assert len(parameters.initial_nodes[v]) == int(sum(parameters.ambulance_distribution[0][v]))


        self.n_vehicles = 0
        for v in range(parameters.vehicle_types):
            cumulative_distribution = np.cumsum(parameters.ambulance_distribution[0][v])
            b = 1
            i = 0

            for m in range(int(sum(parameters.ambulance_distribution[0][v]))):
                # Compute the corresponding borough
                if i == int(cumulative_distribution[b-1]):
                    b += 1
                    
                self.insert(Events.AmbulanceArrivalEvent(self, 0, parameters.initial_nodes[v][m],           
                            Vehicle(parameters.initial_nodes[v][m], v, b, uber=False, name='Ambulance ' + str(self.n_vehicles)), prior_worked_time=np.random.random()*parameters.vehicle_arrival_deviation/2))             
                self.n_vehicles += 1
                i += 1

        cumu_shifts = np.cumsum(parameters.time_shifts)
        for t, distribution in enumerate(parameters.ambulance_distribution[1:]):
            for v in range(parameters.vehicle_types):
                cumulative_distribution = np.cumsum(distribution[v])
                b = 1
                i = 0

                for m in range(int(sum(distribution[v]))):
                    # Compute the corresponding borough
                    if i == int(cumulative_distribution[b-1]):
                        b += 1

                    arrival_time = cumu_shifts[t]*3600 + np.random.random()*parameters.vehicle_arrival_deviation - parameters.vehicle_arrival_deviation/2
                    hospital_node = random.choice([hospital for hospital in parameters.hospital_borough if parameters.hospital_borough[hospital] == b])
                    self.insert(Events.AmbulanceArrivalEvent(self, np.clip(arrival_time, 0, np.inf), hospital_node,           
                                Vehicle(hospital_node, v, b, uber=False, name='Ambulance ' + str(self.n_vehicles)), prior_worked_time=np.clip(-arrival_time, 0, np.inf)))             
                    self.n_vehicles += 1
                    i += 1
        
        day_offset = 24*3600
        for t, distribution in enumerate(parameters.ambulance_distribution):
            for v in range(parameters.vehicle_types):
                cumulative_distribution = np.cumsum(distribution[v])
                b = 1
                i = 0

                for m in range(int(sum(distribution[v]))):
                    # Compute the corresponding borough
                    if i == int(cumulative_distribution[b-1]):
                        b += 1

                    arrival_time = cumu_shifts[t]*3600 + np.random.random()*parameters.vehicle_arrival_deviation - parameters.vehicle_arrival_deviation/2 + day_offset
                    hospital_node = random.choice([hospital for hospital in parameters.hospital_borough if parameters.hospital_borough[hospital] == b])
                    self.insert(Events.AmbulanceArrivalEvent(self, np.clip(arrival_time, 0, np.inf), hospital_node,           
                                Vehicle(hospital_node, v, b, uber=False, name='Ambulance ' + str(self.n_vehicles)), prior_worked_time=np.clip(-arrival_time, 0, np.inf)))             
                    self.n_vehicles += 1
                    i += 1
        
        # Schedule the ambulance setting event
        self.insert(Events.HospitalSettingEvent(self, 0))

        # Schedule the starting positions event
        self.insert(Events.InitialPositioningEvent(self, 0.1))

        # Lists representing the state of the model
        self.activeEmergencies: List[Emergency] = list()        # All the emergencies in the system         
        self.assignedEmergencies: List[Emergency] = list()      # Those which have been assigned

        self.assignedNotArrived: int = 0          

        # Schedule the first arrival
        self.insert(next(self.arrival_generator))

        # Schedule the initializer of the time period change
        #self.insert(Events.ShiftChangeEvent(self, 0))

        # Schedule the first compute Q and P event
        self.insert(Events.ComputeQandPEvent(self, 3600))

        # Initialize model Statistics
        self.statistics: Dict[str, Sim.Statistic] = {}
        self.vehicle_statistics: Dict[str, Dict] = {}

        for b in range(1, 6):
            self.statistics['OptimizationSizeALS' + str(b)] = Sim.TimedTallyStatistic('OptimizationSizeALS' + str(b))   # @ solver execution
            self.statistics['OptimizationTimeALS' + str(b)] = Sim.TimedTallyStatistic('OptimizationTimeALS' + str(b))   # @ solver execution
            self.statistics['OptimizationSizeBLS' + str(b)] = Sim.TimedTallyStatistic('OptimizationSizeBLS' + str(b))   # @ solver execution
            self.statistics['OptimizationTimeBLS' + str(b)] = Sim.TimedTallyStatistic('OptimizationTimeBLS' + str(b))   # @ solver execution

        self.statistics['AvailableALSVehicles'] = Sim.StateStatistic('AvailableALSVehicles', len(self.getAvaliableVehicles(v_type=0)))
        self.statistics['AvailableBLSVehicles'] = Sim.StateStatistic('AvailableBLSVehicles', len(self.getAvaliableVehicles(v_type=1)))
        self.statistics['ALSVehiclesInSystem'] = Sim.StateStatistic('ALSVehiclesInSystem')
        self.statistics['BLSVehiclesInSystem'] = Sim.StateStatistic('BLSVehiclesInSystem')

        self.statistics['EmergenciesServed'] = Sim.CounterStatistic('TotalEmergenciesServed')           # @ Emergency leaving event
        self.statistics['EmergenciesTimeInSystem'] = Sim.TallyStatistic('EmergenciesTimeInSystem')      # @ Emergency leaving event
        self.statistics['NumberHSemergencies'] = Sim.CounterStatistic('NumberHSemergencies')            # @ Emergency leaving event
        self.statistics['NumberLSemergencies'] = Sim.CounterStatistic('NumberLSemergencies')            # @ Emergency leaving event

        self.statistics['HSresponseTime'] = Sim.TimedTallyStatistic('HSresponseTime')                   # @ Ambulance end trip event
        self.statistics['HSaverageResponseTime'] = Sim.TimedTallyStatistic('HSaverageResponseTime')     # @ Ambulance end trip event

        self.statistics['LSresponseTime'] = Sim.TimedTallyStatistic('LSresponseTime')                   # @ Ambulance end trip event
        self.statistics['LSaverageResponseTime'] = Sim.TimedTallyStatistic('LSaverageResponseTime')     # @ Ambulance end trip event

        self.statistics['GeneralAverageResponseTime'] = Sim.TimedTallyStatistic('GeneralAverageResponseTime')  # @ Ambulance end trip event
        self.statistics['PercentageALSlt10min'] = Sim.TimedTallyStatistic('PercentageALSlt10min')              # @ Ambulance end trip event
        self.statistics['PercentageALSlt8min'] = Sim.TimedTallyStatistic('PercentageALSlt8min')                # @ Ambulance end trip event
        self.statistics['PercentageALSlt7min'] = Sim.TimedTallyStatistic('PercentageALSlt7min')                # @ Ambulance end trip event

        self.statistics['AverageAssignmentTime'] = Sim.TimedTallyStatistic('AverageAssignmentTime')     # @ AssignedEvent

        self.statistics['TravelTime'] = Sim.TimedTallyStatistic('EmergencyTravelTime')                  # @ Ambulance end trip event
        self.statistics['HSAttentionTime'] = Sim.TimedTallyStatistic('HSEmergencyAttentionTime')        # @ AmbulanceStartAttendingEvent
        self.statistics['LSAttentionTime'] = Sim.TimedTallyStatistic('LSEmergencyAttentionTime')        # @ AmbulanceStartAttendingEvent
        self.statistics['ToHospitalTime'] = Sim.TimedTallyStatistic('ToHospitalTime')                   # @ Ambulance end trip event

        self.statistics['UberCalls'] = Sim.TimedTallyStatistic('UberCalls')                             # @ AssignedEvent
        self.statistics['UberResponseTime'] = Sim.TimedTallyStatistic('UberResponseTime')               # @ Ambulance end trip event

        self.statistics['SpatialHS10minCover'] = Sim.SpatialStatistic('SpatialHS10minCover')            # @ Ambulance end trip event
        self.statistics['SpatialHS8minCover'] = Sim.SpatialStatistic('SpatialHS8minCover')              # @ Ambulance end trip event
        self.statistics['SpatialHS7minCover'] = Sim.SpatialStatistic('SpatialHS7minCover')              # @ Ambulance end trip event
    
        self.statistics['SpatialHSAverageResponseTime'] = Sim.SpatialStatistic('SpatialHSAverageResponseTime')            # @ Ambulance end trip event        
        self.statistics['SpatialLSAverageResponseTime'] = Sim.SpatialStatistic('SpatialLSAverageResponseTime')            # @ Ambulance end trip event       
        self.statistics['SpatialGeneralAverageResponseTime'] = Sim.SpatialStatistic('SpatialGeneralAverageResponseTime')  # @ Ambulance end trip event      

        self.statistics['SpatialALSRelocation'] = Sim.SpatialStatistic('SpatialALSRelocation')          # @ Relocation Event
        self.statistics['SpatialBLSRelocation'] = Sim.SpatialStatistic('SpatialBLSRelocation')          # @ Relocation Event

        self.sim_start_time = time.time()
        self.statistics['RunTime'] = Sim.CounterStatistic('RunTime')                                    # @ End Simulation event

        self.statistics['GAPALSPart1'] = Sim.TimedTallyStatistic('GAPALSPart1')                         # @ Solver relocate
        self.statistics['GAPALSPart2'] = Sim.TimedTallyStatistic('GAPALSPart2')                         # @ Solver relocate
        self.statistics['GAPBLSPart1'] = Sim.TimedTallyStatistic('GAPBLSPart1')                         # @ Solver relocate
        self.statistics['GAPBLSPart2'] = Sim.TimedTallyStatistic('GAPBLSPart2')                         # @ Solver relocate

        self.statistics['EmergenciesWaiting'] = Sim.StateStatistic('EmergenciesWaiting')


        # Tuples of the form
        # (emergency name, node, severity, arrival time, attending time, to_hospital_time, disposition code, hospital)
        self.emergencyRecord: List[Tuple] = []

    def run(self,
            simulation_time: int = 3600):
        if self.parameters.simulation_time is not None:
            self.insert(Events.EndSimulationEvent(self,
                                                  self.parameters.simulation_time))                         
        else:
            self.insert(Events.EndSimulationEvent(self, simulation_time))
        self.doAllEvents()

        return self.getStatistics()

    def getAvaliableVehicles(self, v_type: Optional[int] = None, borough: Optional[int] = None) -> List[Vehicle]:
        to_check_vehicles = self.vehicles
        if borough is not None:
            to_check_vehicles = [v for v in self.vehicles if v.borough == borough] 
        if v_type is not None:
            return [v for v in to_check_vehicles if v.patient is None and not v.cleaning and v.type == v_type] 
        else:
            return [v for v in to_check_vehicles if v.patient is None and not v.cleaning]
    
    def newUberVehicle(self, from_node, borough = None):
        if borough is None:
            print("Warning: borough detection not implemented")
        
        self.uber_calls += 1
        uber = Vehicle(from_node, 3, borough, uber=True, name="Uber {}".format(self.uber_calls))
        self.ubers.append(uber)
        
        return uber

    def getHospitalType(self, emergency: "Emergency") -> int:
        if emergency.disposition_code in [82, 94]:
            return 1
        else:
            return 0

    def getShortestPath(self, from_nodes, to_nodes, weights = None) -> List[List[int]]:
        """
        Function to obtain the shortest paths based on the speed
        at the time of the simulation
        """
        if weights is None:
            weights = self.city_graph.es['length'] / np.array(self.parameters.getSpeedList(self.timePeriod()))

        return self.city_graph.get_shortest_paths(from_nodes, to_nodes,
                                                  weights=weights,
                                                  output='epath')

    def getShortestDistances(self, from_nodes, to_node, weights = None) -> np.array:
        if weights is None:
            weights = self.city_graph.es['length'] / np.array(self.parameters.getSpeedList(self.timePeriod()))

        return np.array(self.city_graph.shortest_paths(from_nodes, to_node, weights=weights))              

    def computeSystemPreparedness(self,
                                  borough,
                                  vehicle_positions: Optional[List[List[str]]] = None,
                                  travel_matrix: Optional[np.array] = None):
        """
        Expecting a list of lists, containing a list of nodes for each vehicle type
        Will use simulator.parameters for calculations
        """

        if travel_matrix is None:

            if vehicle_positions is None:
                vehicle_positions = [[ambulance.to_node
                                    for ambulance in self.getAvaliableVehicles(v_type=v, borough=borough)]                     
                                    for v in range(self.parameters.vehicle_types)]

            travel_matrix = [self.getShortestPath(vehicle_positions[v],
                                                  self.parameters.demand_borough[borough])
                                 for v in range(self.parameters.vehicle_types)]

        # Emergencies per hour per demand node
        demand_rates = np.array([self.parameters.demand_rates[0].loc[self.timePeriod()+1, self.parameters.demand_borough[borough]],
                                 self.parameters.demand_rates[1].loc[self.timePeriod()+1, self.parameters.demand_borough[borough]]])
        # Mean time per emergency per demand node
        busytime_means = np.array([self.parameters.mean_busytime[0].loc[self.timePeriod()+1, self.parameters.demand_borough[borough]],
                                   self.parameters.mean_busytime[1].loc[self.timePeriod()+1, self.parameters.demand_borough[borough]]])

        # Ponderation of each type of emergency at each demand node
        demand_ratios = demand_rates/np.where(np.sum(demand_rates, axis = 0) == 0, 1, np.sum(demand_rates, axis = 0))

        # Compute the amount of vehicles available per node per emergency level
        valid_indexes = [np.where(np.array(travel_matrix[0]) < 8*60), np.where(np.array(travel_matrix[1]) < 8*60)]
        number_available = np.array([[len(np.where(valid_indexes[v][0] == i)[0]) if len(np.where(valid_indexes[v][0] == i)[0]) > 0 else 1e-10
                                     for i in range(len(self.parameters.demand_borough[borough]))]
                                     for v in range(self.parameters.vehicle_types)])

        # Compute final preparedness
        preparedness = 1 - np.sum((demand_ratios * demand_rates * busytime_means)/number_available, axis = 0)
        preparedness[preparedness < 0] = 0

        return np.sum(preparedness)


    def clearVehicleMovement(self, vehicle):
        """
        This function will clear the future path for the vehicle, so that once
        it arrives to the next node (if moving) it will stop or follow the new
        assigned path
        """
        events_copy = self.events.elements[:]
        for e in events_copy:
            if isinstance(e, Events.AmbulanceStartMovingEvent):
                if e.vehicle == vehicle:
                    self.events.elements.remove(e)

        vehicle.record.append((self.now(), vehicle.pos, None, vehicle.patient, vehicle.patient.hospital if vehicle.patient is not None else None))
    
    def registerVehicleStationChange(self, vehicle, node):
        if isinstance(self.assigner, Solvers.NearestDispatcher):
            return

        if vehicle.borough not in self.covering_state:
            self.covering_state[vehicle.borough] = {}

        if vehicle.patient is None:
            self.covering_state[vehicle.borough][vehicle] = (node, self.getShortestDistances(node, self.parameters.demand_borough[vehicle.borough])[0])
        else:
            if vehicle in self.covering_state:
                del self.covering_state[vehicle.borough][vehicle]

    def _timePeriod(self, seconds, periods):
        hours = (seconds % 86400) // 3600
        cummulative_periods = [sum(periods[:i+1]) for i in range(len(periods))]
        for i, l in enumerate(cummulative_periods):
            if hours < l:
                return i
    
    def timePeriod(self):
        return self._timePeriod(self.now(), self.parameters.time_periods)
    
    def actualPeriodLength(self):
        return self.parameters.time_periods[self.timePeriod()]*3600

    def timeInsidePeriod(self):
        periods = self.parameters.time_periods
        cummulative_periods = [0] + [sum(periods[:i+1])*3600 for i in range(len(periods))]
        time_period = self.timePeriod()
        return self.now() - cummulative_periods[time_period]

    def getStatistics(self):
        for v in self.vehicles:
            self.vehicle_statistics[v.name] = {'Statistics': v.statistics, 'Record': v.record}

        return [self.statistics, self.vehicle_statistics, self.emergencyRecord]
    
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
                 disposition_code: int):
        self.arrival_time: float = arrival_time
        self.node: str = node
        self.severity:int = severity
        self.disposition_code = disposition_code
        self.vechicle: Optional[Vehicle] = None

        self.status: int = 0
        self.vehicle_assigned_time: float = 0
        self.start_attending_time: float = 0
        self.to_hospital_time: float = 0
        self.hospital: Optional[str] = None

        Emergency.N_EMERGENCIES += 1
        self.name = 'Emergency {}'.format(Emergency.N_EMERGENCIES)
    
    def onVehicleArrive(self, vehicle: Vehicle):
        self.vehicle = vehicle
        self.markStatus(2)

        # If disposition code marks no attending time
        if self.disposition_code in [83, 91, 93, 96]:
            self.attending_time = 0
        else:
            if self.severity == 1:
                self.attending_time = np.random.gamma(6.099280636730107)*271.21712543885883-243.7935927436323
            else:
                self.attending_time = np.random.beta(3.4911624003509623, 229327901002.9558)*82395177367522.72-38.931290935256456
    
    def markStatus(self, status):
        self.status = status
    
    def __str__(self):
        return self.name