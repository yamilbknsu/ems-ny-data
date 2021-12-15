# Import Statements
import time
import random
import igraph
import numpy as np
from collections import defaultdict
from typing import List, Optional, Dict, Tuple

# Local Imports
import Events
import OnlineSolvers
import Generators
import SimulatorBasics as Sim
from SimulationParameters import SimulationParameters


class Vehicle(Sim.SimulationEntity):
    """
    Object representing a vechicle that can move through a graph.
    This object holds information about the current state of the vehicle.
    This includes, among other things, the list of edge elements that
    are part of the actual route, a reference to an emergency instance,
    if attending any, and the actual position of the vehicle on the graph.


    In principle we could then extend this class to an ambulance-like class
    but since there won't be any other kind of vehicles in the model,
    We'll just take this as a synonym for ambulance.

    Properties of the class:

    patient (Emergency, optional): Emergency object of the actual patient
    type (int): Ambulance type (0: ALS, 1: BLS, 3: RHS)
    borough (int): Which of the 5 boroughs this ambulance is assigned to serve
    isUber (bool): Whether or not this vehicle is RHS (TODO: redundant)
    shift (int): Number of shift this ambulance is assigned.
    arrive_in_system (float): Time at which the ambulance arrived to the system
    record: List of all the activities this ambulance has performed, including
            relocation and providing attention.
    cleaning (bool): Whether this ambulance is currently being cleaned
    leaving (bool): Whether this ambulance is leaving the system once it becomes idle.
    """

    def __init__(self,
                 start_node: str,
                 vehicle_type: int,
                 arrive_in_system: float,
                 borough: int,
                 uber: bool = False,
                 shift: int = 0,
                 name: str = None):
        """
        Args:
            start_node (str): Starting node for the ambulance.
            vehicle_type (int): int describing which type of vehicle this is (ALS: 0, BLS: 1).
            arrive_in_system (float): Time at which the ambulance arrived to the system
            borough (int): Borough to which this vehicle belongs to.
            name (str, optional): [description]. Defaults to None.
        """
        super().__init__(name)
        self.patient: Optional[Emergency] = None
        self.type = vehicle_type
        self.borough = borough
        self.isUber = uber
        self.shift = shift
        self.arrive_in_system = arrive_in_system

        # This is a record of all the movement an ambulance has performed
        # Each element is a tuple with the form:
        # (time, from_node, to_node, emergency, emergency severity, hospital)
        self.record: List[Tuple] = []

        self.stopMoving()                   # Though obviously is not
        self.path: List[int] = []
        self.node_path: List[str] = []
        self.expected_arrival: float = 0    # Expected arrival time to next node
        self.teleportToNode(start_node)
        self.cleaning: bool = False
        self.leaving: bool = False
        #self.station_changed: bool = False

        # The time spent repositioning
        self.accumulated_relocation: float = 0
        self.reposition_workload: float = 0
        self.total_busy_time: float = 0
        self.station = start_node
        self.arrival_time = 0
        self.relocating: bool = False
        self.can_relocate: bool = True

        # Statistics definition
        self.statistics: Dict[str, Sim.Statistic] = {}

        self.statistics['RelocationTime'] = Sim.StateStatistic('RelcationTime{}'.format(self.name))
        self.statistics['MetersDriven'] = Sim.StateStatistic('MetersDriven{}'.format(self.name))
        self.statistics['EmergenciesServed'] = Sim.CounterStatistic('EmergenciesServed{}'.format(self.name))
        self.statistics['State'] = Sim.StateStatistic('VehicleState{}'.format(self.name))
        self.statistics['TimeInSystem'] = Sim.CounterStatistic('TimeInSystem{}'.format(self.name))
        self.statistics['ArriveInSystem'] = Sim.CounterStatistic('ArriveInSystem{}'.format(self.name))
        self.statistics['ArriveInSystem'].record(arrive_in_system)
        self.statistics['Shift'] = Sim.CounterStatistic('AmbulanceShift{}'.format(self.name))
        self.statistics['Shift'].record(self.shift)
        self.statistics['Borough'] = Sim.CounterStatistic('AmbulanceBorough{}'.format(self.name))
        self.statistics['Borough'].record(self.borough)
        self.statistics['Type'] = Sim.CounterStatistic('AmbulanceType{}'.format(self.name))
        self.statistics['Type'].record(self.type)
        self.statistics['BusyWorkload'] = Sim.StateStatistic('BusyWorkload{}'.format(self.name))
        self.statistics['AccumulatedWorkload'] = Sim.StateStatistic('AccumulatedWorkload{}'.format(self.name))

        # State posibilities
        # 0: Idle                   @ Ambulance end trip event and Ambulance end cleaning
        # 1: Repositioning          @ Ambulance start moving event
        # 2: Attending              @ Assigned event

    def teleportToNode(self, node: str):
        """
        Use Odin's powers to teleport this vehicle to a
        given node in the graph.

        NOTE:
        This should only be used at the beggining of the simulation as the movement
        will automatically be recorded as happening at time 0.
        """
        self.pos = node
        self.onArrivalToNode(node)

        # Save a record of the trip
        self.record.append((0, self.pos, node, str(self.patient),
                            self.patient.hospital if self.patient is not None else None))

    def stopMoving(self):
        """
        This function is kept in case further action were to take place
        or statistics were to be collected whenever the vehicle stops moving.
        """
        self.moving: bool = False

    def onArrivalToNode(self, node: str) -> bool:
        """
        Function to be called when the vehicle arrives to a node.
        The information about the optimal path stored in the object
        is used to determine where to move next and whether or not
        the trip is over.

        Returns a bool -> Whether or not this node was the end of
                          the vehicle's journey
        """

        # Update logistic state
        self.pos = node
        self.actual_edge: Optional[int] = None
        self.from_node: str = node

        # Check for future movement
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
        """
        This function is kept in case further action were to take place
        or statistics were to be collected whenever the vehicle starts moving.
        """
        self.moving = True
        self.expected_arrival = expected_arrival

    def onAssignedToEmergency(self,
                              patient: "Emergency",
                              path: List[int],
                              node_path: List[str]):
        """
        This function is kept in case further action were to take place
        or statistics were to be collected whenever the vehicle is assigned to an emergency.
        """
        self.patient = patient
        self.onAssignedMovement(path, node_path)

    def onAssignedMovement(self,
                           path: List[int],
                           node_path: List[str]):
        """
        This function is kept in case further action were to take place
        or statistics were to be collected whenever the vehicle is assigned to move.
        """
        self.path = path
        self.node_path = node_path

    def onPathEnded(self) -> int:
        """
        Called when the path has been completed.

        Returns a status:
        0 : Means nothing is expected of the ambulance
            after this, do an assignment step
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
                return 2

        return 0


class EMSModel(Sim.Simulator):
    """
    Main class of the simulation model.
    Here, the main state of the system is stored, as well as
    references to all the vehicles and emergencies in the system and simulation and optimization
    parameters.

    A simulation experiment is excuted by instancing this class with all the required parameters
    (initial setup is performed during this initialization) and calling the run function which then
    starts executing the events in the queue until there are no left or the simulation clock has
    passed the stipulated limit.

    Given the global scope of this class, the current instance being used is passed alongside each event
    during its execution, in order to provide context to all the events.
    """

    def __init__(self,
                 city_graph: igraph.Graph,
                 generator_object: Generators.ArrivalGenerator,
                 optimizer: OnlineSolvers.RelocationModel,
                 parameters: SimulationParameters,
                 **kwargs):
        """
        Args:
            city_graph (Graph): An iGraph  object that represents the road network of the city. This is going
                                to be used to compute the shortest distances and paths when routing the ambulances
            generator_object (ArrivalGenerator): Object extending the ArrivalGenerator class that produces emergencies
                                arrivals to the system.
            optimizer (RelocationModel): Object extending the RelocationModel class that is used to optimize the relocation,
                                redeployment and dispatching of available ambulances.
            parameters (SimulationParameters): Instance of the SimilationParameter class containing all the simulation and
                                optimization parameters needed for the model to run.
        """

        super().__init__(**kwargs)

        # Object definition
        self.city_graph: igraph.Graph = city_graph
        self.generator_object: Generators.ArrivalGenerator = generator_object
        self.optimizer: OnlineSolvers.RelocationModel = optimizer
        self.arrival_generator = self.generator_object.generator(self)
        self.parameters = parameters

        # Hospital nodes
        self.hospitals: Dict[int, List[str]] = parameters.hospital_nodes

        # State variables
        # ---------------
        # The last reposition status TODO: I think is not used
        # self.ambulance_stations: Dict[int, Dict[int, List[str]]] = {b: {s: [] for s in range(0, 1)} for b in range(1, 6)}

        # The amount of uber calls and uber seconds used
        self.uber_calls = 0
        self.uber_seconds = 0

        # A record of the time spent on an emergency on a specific node
        # (time_period, severity, node) = [times]
        self.time_records: Dict[Tuple[int, int, str], List[float]] = defaultdict(list)

        # Initialization processes
        # ------------------------
        # Randomly initialize initial positions for ambulances inside
        # the corresponding borough if needed
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

        # Initialize vehicle arrivals
        self.InitializeVehicles(self.parameters)

        # Schedule the ambulance setting event
        self.insert(Events.HospitalSettingEvent(self, 0))

        # Schedule the starting positions event
        self.insert(Events.InitialPositioningEvent(self, 1e-6))

        # Lists representing the state of the model
        self.activeEmergencies: List[Emergency] = list()        # All the emergencies in the system
        self.assignedEmergencies: List[Emergency] = list()      # Those which have been assigned

        # Number of emergencies in the system that have been
        # assigned an ambulance but this has not yet arrived
        self.assignedNotArrived: int = 0

        # Schedule the first arrival
        self.insert(next(self.arrival_generator))

        # Schedule the first event that uptades the busytime parameters
        self.insert(Events.ComputeParemeters(self, 3600))

        # Initialize model Statistics
        self.statistics: Dict[str, Sim.Statistic] = {}
        self.vehicle_statistics: Dict[str, Dict] = {}

        # The comment next to each definition explains where this statistic is updated in the code
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
        self.statistics['RunTime'] = Sim.CounterStatistic('RunTime')                                    # @ End Simulation event
        self.statistics['EmergenciesWaiting'] = Sim.StateStatistic('EmergenciesWaiting')

        self.sim_start_time = time.time()

        # Tuples of the form
        # (emergency name, node, severity, arrival time, attending time, to_hospital_time, disposition code, hospital, vehicle)
        self.emergencyRecord: List[Tuple] = []

    def run(self,
            simulation_time: int = 3600):
        """
        Starts the execution of the simulation model, which will continue until no events are
        left or the simulation time has exceeded the stipulated limit.
        """

        # Add Enf of simulation event
        if self.parameters.simulation_time is not None:
            self.insert(Events.EndSimulationEvent(self,
                                                  self.parameters.simulation_time))
        else:
            self.insert(Events.EndSimulationEvent(self, simulation_time))
        
        # Start the model
        self.doAllEvents()

        return self.getStatistics()

    def InitializeVehicles(self, parameters):
        """
        Initialize the vehicle objects and schedule their arrival to the system
        based on the input parameters and the ambulance tour distribution computed during parameter
        initialization.

        What is really going on in this code, is that for the first shift of the first day of simulation,
        we use the initial nodes that were passed as an input parameter to the model or were randomly assigned.
        Then, for the rest of the shifts, we randomly select an hospital for each arrival.

        NOTE:
        This NEEDS to be reviewed if you are using a scheme for the shifts other than the 8-hour shift used in our model.
        """
        self.n_vehicles = 0

        # First shift of first day
        for v in range(parameters.vehicle_types):
            cumulative_distribution = np.cumsum(parameters.ambulance_distribution[0][v])
            b = 1
            i = 0

            for m in range(int(sum(parameters.ambulance_distribution[0][v]))):
                # Compute the corresponding borough
                if i == int(cumulative_distribution[b - 1]):
                    b += 1

                self.insert(Events.AmbulanceArrivalEvent(self, 0, parameters.initial_nodes[v][m],
                            Vehicle(parameters.initial_nodes[v][m], v, arrive_in_system=0, borough=b, shift=1, uber=False, name='Ambulance ' + str(self.n_vehicles)), prior_worked_time=np.random.random() * parameters.vehicle_arrival_deviation / 2))
                self.n_vehicles += 1
                i += 1

        # Rest of the shifts of the first day
        cumu_shifts = np.cumsum(parameters.time_shifts)
        for t, distribution in enumerate(parameters.ambulance_distribution[1:]):
            for v in range(parameters.vehicle_types):
                cumulative_distribution = np.cumsum(distribution[v])
                b = 1
                i = 0

                for m in range(int(sum(distribution[v]))):
                    # Compute the corresponding borough
                    if i == int(cumulative_distribution[b - 1]):
                        b += 1

                    arrival_time = cumu_shifts[t] * 3600 + np.random.random() * parameters.vehicle_arrival_deviation - parameters.vehicle_arrival_deviation / 2
                    hospital_node = random.choice([hospital for hospital in parameters.hospital_borough if parameters.hospital_borough[hospital] == b])
                    self.insert(Events.AmbulanceArrivalEvent(self, np.clip(arrival_time, 0, np.inf), hospital_node,
                                Vehicle(hospital_node, v, arrive_in_system=np.clip(arrival_time, 0, np.inf), borough=b, shift=t + 2, uber=False, name='Ambulance ' + str(self.n_vehicles)), prior_worked_time=np.clip(-arrival_time, 0, np.inf)))
                    self.n_vehicles += 1
                    i += 1

        # The rest of the days (Assuming the same amount of ambulance tours)
        n_days = int(np.clip((parameters.simulation_time // (3600 * 24)), 0, np.inf))
        for d in range(1, n_days):
            day_offset = int(24 * 3600 * d)
            cumu_shifts = np.cumsum(parameters.time_shifts) - 8
            for t, distribution in enumerate(parameters.ambulance_distribution):
                for v in range(parameters.vehicle_types):
                    cumulative_distribution = np.cumsum(distribution[v])
                    b = 1
                    i = 0

                    for m in range(int(sum(distribution[v]))):
                        # Compute the corresponding borough
                        if i == int(cumulative_distribution[b - 1]):
                            b += 1

                        arrival_time = cumu_shifts[t] * 3600 + np.random.random() * parameters.vehicle_arrival_deviation - parameters.vehicle_arrival_deviation / 2 + day_offset
                        hospital_node = random.choice([hospital for hospital in parameters.hospital_borough if parameters.hospital_borough[hospital] == b])
                        self.insert(Events.AmbulanceArrivalEvent(self, np.clip(arrival_time, 0, np.inf), hospital_node,
                                    Vehicle(hospital_node, v, arrive_in_system=np.clip(arrival_time, 0, np.inf), borough=b, shift=4, uber=False, name='Ambulance ' + str(self.n_vehicles)), prior_worked_time=np.clip(-arrival_time, 0, np.inf)))
                        self.n_vehicles += 1
                        i += 1

    def getAvaliableVehicles(self, v_type: Optional[int] = None, borough: Optional[int] = None) -> List[Vehicle]:
        """
        Returns a list with the available vehicles, that is, vehicles that have no emergency assigned and are not being cleaned.
        If values of v_type or borough are passed, they are used to filter the list.
        """
        to_check_vehicles = self.vehicles
        if borough is not None:
            to_check_vehicles = [v for v in self.vehicles if v.borough == borough]
        if v_type is not None:
            return [v for v in to_check_vehicles if v.patient is None and not v.cleaning and v.type == v_type]
        else:
            return [v for v in to_check_vehicles if v.patient is None and not v.cleaning]

    def getUnassignedEmergencies(self, severity: int, borough: int):
        """
        Returns the list of emergencies in the systems that have noot been assigned an ambulance.
        """
        return [e for e in self.activeEmergencies if e.severity == severity and e.borough == borough and e not in self.assignedEmergencies]

    def newUberVehicle(self, from_node, borough=None):
        """
        Generates a new RHS vehicle, inserts it into the system and returns the object.
        """
        if borough is None:
            print("Warning: borough detection not implemented")

        self.uber_calls += 1
        uber = Vehicle(from_node, 3, arrive_in_system=self.now(), borough=borough, uber=True, name="Uber {}".format(self.uber_calls))
        self.ubers.append(uber)

        return uber

    def getHospitalType(self, emergency: "Emergency") -> int:
        """
        Auxliary function that implements the mapping of an emergency to the type of hospital
        required.
        In our case, there was only one type of hospital so the function only returns a value indicating
        whether this emergency has to be translated to a hospital or not.
        """
        if emergency.disposition_code in [82, 94]:
            return 1
        else:
            return 0

    def getShortestPath(self, from_nodes, to_nodes, weights=None) -> List[List[int]]:
        """
        Function to obtain the shortest paths based on time-dependent congestion.
        """
        if weights is None:
            weights = self.city_graph.es['length'] / np.array(self.parameters.getSpeedList(self.timePeriod()))

        return self.city_graph.get_shortest_paths(from_nodes, to_nodes,
                                                  weights=weights,
                                                  output='epath')

    def getShortestDistances(self, from_nodes, to_node, weights=None) -> np.array:
        """
        Function to obtain the shortest distances based on time-dependent congestion.
        """
        if weights is None:
            weights = self.city_graph.es['length'] / np.array(self.parameters.getSpeedList(self.timePeriod()))

        return np.array(self.city_graph.shortest_paths(from_nodes, to_node, weights=weights))

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
                    self.events.remove(e)

        vehicle.record.append((self.now(), vehicle.pos, None, vehicle.patient, vehicle.patient.hospital if vehicle.patient is not None else None))

    def _timePeriod(self, seconds, periods):
        """
        Private function to compute the current time period based on simulation time
        """
        hours = (seconds % 86400) // 3600
        cummulative_periods = [sum(periods[:i + 1]) for i in range(len(periods))]
        for i, l in enumerate(cummulative_periods):
            if hours < l:
                return i

    def timePeriod(self):
        """
        Returns the current time period of the simulation
        """
        return self._timePeriod(self.now(), self.parameters.time_periods)

    def actualPeriodLength(self):
        return self.parameters.time_periods[self.timePeriod()] * 3600

    def timeInsidePeriod(self):
        """
        Return how much simulation time has elapsed since the time period started
        """
        periods = self.parameters.time_periods
        cummulative_periods = [0] + [sum(periods[:i + 1]) * 3600 for i in range(len(periods))]
        time_period = self.timePeriod()
        return self.now() - cummulative_periods[time_period]

    def getStatistics(self):
        """
        Returns all the statistics collected at the moment, including all the statistics at each vehicle and the
        emergency record.
        """
        for v in self.vehicles:
            self.vehicle_statistics[v.name] = {'Statistics': v.statistics, 'Record': v.record}

        return [self.statistics, self.vehicle_statistics, self.emergencyRecord]

    def __str__(self):
        return 'EMS Simulator'


class Emergency:
    """
    An emergency has 4 possible status:
      0: Waiting
      1: Assigned
      2: Being attended
      3: Being Carried

    and the vehicle property represents the corresponding vehicle
    whose nature (Vehicle object or None) will depend on the emergency status.
    """

    N_EMERGENCIES = 0

    def __init__(self,
                 arrival_time: float,
                 node: str,
                 severity: int,
                 disposition_code: int):
        self.arrival_time: float = arrival_time
        self.node: str = node
        self.severity: int = severity
        self.disposition_code = disposition_code
        self.vechicle: Optional[Vehicle] = None

        # Emergency state variables
        self.status: int = 0
        self.vehicle_assigned_time: float = 0       # Ambulance assigned
        self.start_attending_time: float = 0        # Service started
        self.to_hospital_time: float = 0            # Transfer to hospital started
        self.hospital: Optional[str] = None         # Hospital assigned

        Emergency.N_EMERGENCIES += 1
        self.name = 'Emergency {}'.format(Emergency.N_EMERGENCIES)

    def onVehicleArrive(self, vehicle: Vehicle):
        """
        Function to be called when a vehicle arrives to attend the emergency.

        TODO:
        Disposition codes and service times are hard-coded, should be parametrized
        """
        self.vehicle = vehicle
        self.markStatus(2)

        # If disposition code marks no attending time
        if self.disposition_code in [83, 91, 93, 96]:
            self.attending_time = 0
        else:
            if self.severity == 1:
                self.attending_time = np.random.beta(2.1509242634375183, 4.592402000128885) * (3500 - 1) + 1
            else:
                self.attending_time = np.random.beta(2.8715309024153197, 4.588531730319283) * (3500 - 2) + 2

    def markStatus(self, status):
        self.status = status

    def assignBorough(self, simulator):
        """
        This function is called to map the arriving emergency to a specific borough.
        This is necessary since some of the nodes are outside the borders of all the boroughs. In the case an emergency arrives to one of them,
        the borough of the vehicle node is used.
        """
        emergency_borough = int(simulator.parameters.nodes_with_borough[simulator.parameters.nodes_with_borough['osmid'] == self.node]['boro_code'])

        # Get the nearest borough
        node = self.node
        if emergency_borough == 0:
            distances = simulator.getShortestDistances([v.to_node for v in simulator.vehicles], node)
            emergency_borough = simulator.vehicles[np.argmin(distances.reshape(-1))].borough
            print("Emergency from node {} mapped to borough {}".format(self.node, emergency_borough))

        self.borough = emergency_borough

    def __str__(self):
        return self.name
