import time
import igraph
import numpy as np
from typing import List, Optional

# Internal imports
import Models
import SimulatorBasics as Sim


class DebugEvent(Sim.Event):
    """
    Event used for debugging purposes. You add a breakpoint at the execute method and use it for debugging.
    """

    def __init__(self,
                 entity: object,
                 time: float,
                 name: str = None):
        super().__init__(time, name)

        self.entity = entity
        self.message: str = 'Debug Event'

    def execute(self, simulator: "Models.EMSModel"):
        pass


class ComputeParemeters(Sim.Event):
    """
    Simulation Event:
    This event is called every hour to update utilization parameters.
    """

    def __init__(self,
                 entity: object,
                 time: float,
                 name: str = None):
        super().__init__(time, name)

        self.entity = entity
        self.message: str = 'Precomputing parameters ...'

    def execute(self, simulator: "Models.EMSModel"):
        # Update the mean values in the simulation parameters
        for key, value in simulator.time_records.items():
            simulator.parameters.mean_busytime[key[1]].at[key[0], key[2]] = np.mean(np.array(value) / 3600)

        simulator.insert(ComputeParemeters(simulator, simulator.now() + 3600))


class RelocationAndDispatchingEvent(Sim.Event):
    """
    Simulation Event:
    Optimized relocation and dispatching at an arbitrary time.
    """

    def __init__(self,
                 entity: object,
                 time: float,
                 severity: int,
                 borough: int,
                 name: str = None):
        super().__init__(time, name)

        self.entity = entity
        self.message: str = 'Relocation Event'
        self.severity = severity
        self.borough = borough

    def execute(self, simulator: "Models.EMSModel"):
        # TODO: I believe this if doesn't do anything
        if self.severity != 3:
            simulator.statistics['AvailableALSVehicles'].record(simulator.now(), len(simulator.getAvaliableVehicles(v_type=0)))
            simulator.statistics['AvailableBLSVehicles'].record(simulator.now(), len(simulator.getAvaliableVehicles(v_type=1)))
            
            # If this is a BLS call or the number of active emergencies is smaller than the number of available ambulances
            if self.severity == 1 or len(simulator.getAvaliableVehicles(0, self.borough)) >= len(simulator.getUnassignedEmergencies(1, self.borough)):
                # Run optimization process with relocation and dispatching
                optimal_positions, reposition_dict, dispatching_dict = simulator.optimizer.optimize(simulator, simulator.parameters, False, severity=self.severity, borough=self.borough)
            else:
                # Only dispatch process
                optimal_positions, reposition_dict = [], {}
                dispatching_dict = simulator.optimizer.dispatch(simulator, simulator.parameters, severity=self.severity, borough=self.borough)

            # Record for logging
            self.optimal_positions = optimal_positions

            # Statistics
            C = simulator.parameters.candidates_borough[self.borough]
            for candidate in C:
                simulator.statistics['Spatial{}Relocation'.format('ALS' if self.severity == 0 else 'BLS')].record(simulator.now(), candidate, 1 if candidate in optimal_positions else 0)

            if simulator.verbose:
                print('Total relocations:', len(reposition_dict), ', Total dispatched vehicles:', len(dispatching_dict),
                      ', Total emergencies:', len(simulator.getUnassignedEmergencies(self.severity, self.borough)))

            # Schedule relocation
            for v in reposition_dict:
                v.station = reposition_dict[v]
                simulator.insert(TripAssignedEvent(simulator, simulator.now(), v, reposition_dict[v]))

            # Schedule dispatching
            for v in dispatching_dict:
                # Schedule the assignment event
                simulator.insert(AssignedEvent(simulator, simulator.now(),
                                               v, dispatching_dict[v]))

                # Mark emergency as assigned
                dispatching_dict[v].markStatus(1)
                simulator.assignedEmergencies.append(dispatching_dict[v])


class InitialPositioningEvent(Sim.Event):
    """
    Simulation Event:
    Initial positioning of the ambulances
    """

    def __init__(self,
                 entity: object,
                 time: float,
                 name: str = None):
        super().__init__(time, name)

        self.entity = entity
        self.message: str = 'Computing starting positions'

    def execute(self, simulator: "Models.EMSModel"):
        # For getting the information on the visualization
        self.new_positions = []

        # If there is an optimizer defined
        if simulator.optimizer is not None:
            # For each of the 5 boroughs
            for b in range(1, 6):
                optimal_positions_ALS, reposition_dict_ALS, _ = simulator.optimizer.optimize(simulator, simulator.parameters, True, severity=0, borough=b)
                optimal_positions_BLS, reposition_dict_BLS, _ = simulator.optimizer.optimize(simulator, simulator.parameters, True, severity=1, borough=b)

                # Move ALS ambulances
                for v in reposition_dict_ALS:
                    v.station = reposition_dict_ALS[v]
                    v.teleportToNode(reposition_dict_ALS[v])

                    self.new_positions.append([v.name, reposition_dict_ALS[v]])

                # Move BLS ambulances
                for v in reposition_dict_BLS:
                    v.station = reposition_dict_BLS[v]
                    v.teleportToNode(reposition_dict_BLS[v])

                    self.new_positions.append([v.name, reposition_dict_BLS[v]])

                # Statistics
                C = simulator.parameters.candidates_borough[b]
                for candidate in C:
                    simulator.statistics['SpatialALSRelocation'].record(simulator.now(), candidate, 1 if candidate in optimal_positions_ALS[0] else 0)
                    simulator.statistics['SpatialBLSRelocation'].record(simulator.now(), candidate, 1 if candidate in optimal_positions_BLS[1] else 0)


class EmergencyLeaveSystemEvent(Sim.Event):
    """
    Simulation Event:
    Emergency has been served an now leaves the system
    """

    def __init__(self,
                 entity: object,
                 time: float,
                 emergency: "Models.Emergency",
                 satisfied: bool,
                 vehicle: Optional["Models.Vehicle"] = None,
                 name: str = None):
        super().__init__(time, name)

        self.entity = entity
        self.emergency: "Models.Emergency" = emergency
        self.vehicle = vehicle  # This is to safely remove the reference to the patient from the vehicle
        self.message: str = '{} leaving the system {}'.format(emergency.name, 'satisfied' if satisfied else 'unsatisfied')

    def execute(self, simulator: "Models.EMSModel"):
        # Add the total time to the record
        severity = 0 if self.emergency.severity == 1 else 1
        simulator.time_records[(simulator.timePeriod() + 1, severity, self.emergency.node)].append(simulator.now() - self.emergency.vehicle_assigned_time)

        simulator.emergencyRecord.append((self.emergency.name, self.emergency.node, self.emergency.severity, self.emergency.arrival_time,
                                         self.emergency.start_attending_time, self.emergency.to_hospital_time, self.emergency.disposition_code, self.emergency.hospital, (self.vehicle is not None and self.vehicle.isUber), (self.vehicle.name if self.vehicle is not None else None)))

        # Remove the emergency fron the list
        simulator.activeEmergencies.remove(self.emergency)
        if self.emergency in simulator.assignedEmergencies:
            simulator.assignedEmergencies.remove(self.emergency)

        # If it has a vehicle currently assigned
        if self.vehicle is not None:
            # Remove the assignement
            self.vehicle.patient = None

            # Statistics
            simulator.statistics['EmergenciesServed'].record()
            simulator.statistics['EmergenciesTimeInSystem'].record(simulator.now() - self.emergency.arrival_time)

            if self.emergency.severity == 1:
                simulator.statistics['NumberHSemergencies'].record()
            else:
                simulator.statistics['NumberLSemergencies'].record()

            self.vehicle.statistics['EmergenciesServed'].record()

            # If this was a RHS vehicle, we remove it from the system
            if self.vehicle.isUber:
                simulator.ubers.remove(self.vehicle)
                simulator.uber_seconds += simulator.now() - self.vehicle.arrive_in_system


class AmbulanceFinishAttendingEvent(Sim.Event):
    """
    Simulation Event:
    Ambulance finishes providing service. Here we also determine if the emergency needs
    to be carried to a hospital and which one.
    """

    def __init__(self,
                 entity: object,
                 time: float,
                 vehicle: "Models.Vehicle",
                 emergency: "Models.Emergency",
                 name: str = None):
        super().__init__(time, name)

        self.entity = entity
        self.vehicle: "Models.Vehicle" = vehicle
        self.emergency: "Models.Emergency" = emergency
        self.message: str = '{} finished attending {}'.format(vehicle.name, emergency.name)

    def execute(self, simulator: "Models.EMSModel"):
        # Get the hospital type (if more than 1)
        hospital_type = simulator.getHospitalType(self.emergency)

        # If the emergency need no to be transported to a hospital
        if hospital_type == 0:
            # Mark ambulance as cleaning and schedule finish cleaning event
            self.vehicle.cleaning = True
            self.vehicle.total_busy_time += 10 * 60
            simulator.insert(AmbulanceEndCleaningEvent(simulator, simulator.now() + 10 * 60, self.vehicle))

            # Update statistics
            self.vehicle.statistics['BusyWorkload'].record(simulator.now(), self.vehicle.total_busy_time)
            self.vehicle.statistics['AccumulatedWorkload'].record(simulator.now(), self.vehicle.accumulated_relocation)

            # Emergency leaves the system now
            return EmergencyLeaveSystemEvent(simulator, simulator.now(), self.emergency, True, self.vehicle)
        else:
            # Get nearest hospital of the corresponding type
            valid_hospitals = simulator.parameters.nodes_with_borough[simulator.parameters.nodes_with_borough['osmid'].isin(simulator.hospitals[1]) &
                                                                      (simulator.parameters.nodes_with_borough['boro_code'] == self.vehicle.borough)]['osmid'].values.tolist()
            distances = simulator.getShortestDistances(self.emergency.node, valid_hospitals)[0]
            nearest_node = valid_hospitals[np.argmin(distances)]

            # Start the trip to the hospital
            self.emergency.to_hospital_time = simulator.now()
            self.emergency.markStatus(3)
            self.emergency.hospital = nearest_node
            return TripAssignedEvent(self.emergency, simulator.now(), self.vehicle, nearest_node)


class AmbulanceStartAttendingEvent(Sim.Event):
    """
    Simulation Event:
    Ambulance starts providing service to an emergency
    """

    def __init__(self,
                 entity: object,
                 time: float,
                 vehicle: "Models.Vehicle",
                 emergency: "Models.Emergency",
                 name: str = None):
        super().__init__(time, name)

        self.entity = entity
        self.vehicle: "Models.Vehicle" = vehicle
        self.emergency = emergency
        self.message: str = '{} started attending {}'.format(vehicle.name, emergency.name)

    def execute(self, simulator: "Models.EMSModel"):
        # Record start attention time
        self.emergency.start_attending_time = simulator.now()

        # Schedule the end of the atention
        if not self.vehicle.isUber:
            self.vehicle.total_busy_time += self.emergency.attending_time
            simulator.insert(AmbulanceFinishAttendingEvent(self.entity, simulator.now() + self.emergency.attending_time, self.vehicle, self.emergency))
        else:
            simulator.insert(AmbulanceFinishAttendingEvent(self.entity, simulator.now() + 60, self.vehicle, self.emergency))

        # Statistics
        if self.emergency.severity == 1:
            simulator.statistics['HSAttentionTime'].record(simulator.now(), self.emergency.attending_time)
        else:
            simulator.statistics['LSAttentionTime'].record(simulator.now(), self.emergency.attending_time)

        self.vehicle.statistics['BusyWorkload'].record(simulator.now(), self.vehicle.total_busy_time)
        self.vehicle.statistics['AccumulatedWorkload'].record(simulator.now(), self.vehicle.accumulated_relocation)


class AmbulanceRedeployEvent(Sim.Event):
    """
    Simulation Event:
    Redeploy an ambulance. This is called when an ambulance vecomes idle after service and needs to be instructed to move
    to a new node.
    """

    def __init__(self,
                 entity: object,
                 time: float,
                 vehicle: "Models.Vehicle",
                 name: str = None):
        super().__init__(time, name)

        self.entity = entity
        self.vehicle = vehicle
        self.message: str = '{} redeploying...'.format(vehicle.name)

    def execute(self, simulator: "Models.EMSModel"):
        # Execute the optimization process and schedule the movement
        if not self.vehicle.isUber:
            optimal_positions, reposition_dict = simulator.optimizer.redeploy(simulator, simulator.parameters, self.vehicle)
            self.vehicle.station = reposition_dict[self.vehicle]
            simulator.insert(TripAssignedEvent(simulator, simulator.now(), self.vehicle, reposition_dict[self.vehicle]))


class AmbulanceEndCleaningEvent(Sim.Event):
    """
    Simulation Event:
    Called when an ambulance has finished cleaning after service.
    """

    def __init__(self,
                 entity: object,
                 time: float,
                 vehicle: "Models.Vehicle",
                 name: str = None):
        super().__init__(time, name)

        self.entity = entity
        self.vehicle: "Models.Vehicle" = vehicle
        self.message: str = '{} finished cleaning, triggering redeployment'.format(vehicle.name)

    def execute(self, simulator: "Models.EMSModel"):
        # Update statistics
        self.vehicle.statistics['State'].record(simulator.now(), 0)

        # Update flag
        self.vehicle.cleaning = False

        # If this is not a RHS vehicle
        if not self.vehicle.isUber:
            # If the vehicle finished its shift, it leaves the syste,
            if self.vehicle.leaving:
                return AmbulanceLeavingEvent(simulator, simulator.now(), self.vehicle)

            # Otherwise, we execute a dispatching process to attend an emergency in case there is an active one
            dispatching_dict = simulator.optimizer.dispatch(simulator, simulator.parameters, self.vehicle.type, self.vehicle.borough)
            if self.vehicle in dispatching_dict.keys():
                # Mark emergency as assigned
                dispatching_dict[self.vehicle].markStatus(1)
                simulator.assignedEmergencies.append(dispatching_dict[self.vehicle])

                # Schedule the assignment event
                return AssignedEvent(simulator, simulator.now(), self.vehicle, dispatching_dict[self.vehicle])

            # If there are no emergencies that could be served, we either return to the current station
            # or trigger a redeployment event
            if simulator.parameters.force_static:
                return TripAssignedEvent(simulator, simulator.now(), self.vehicle, self.vehicle.station)
            else:
                return AmbulanceRedeployEvent(self.entity, simulator.now(), self.vehicle)


class AmbulanceEndTripEvent(Sim.Event):
    """
    Simulation Event:
    Ambulance finishes the assigned movement.
    There is a lot happening on this code, but the important thing to understand is that
    the behavior almost exclusively depends on the nature of the trip taken, and this is
    determined using the status of the patient being transported or the absence of it.

    For instance, if the status returned from onPathEnded is 0, means the ambulance was
    empty and therefore this movement was just a relocation and there is no further action
    to be taken and the ambulance becomes idle. If this status is 1, it means that the trip
    was from an Idle position to attend an emergency and it has bow arrived there. Finally,
    if it is 2, then it means it was transporting a patient and has arrived to the hospital.
    
    Then, most of the code is just state handling and statistics update.

                                                                  (*)   
      Trip Assigned  ->  Ambulance Start -> Ambulance Arrive -> End Trip
          Event            Moving Event      To Node Event       Event
                                ^                   |
                                |                   v
                                ---------------------
    """

    def __init__(self,
                 entity: object,
                 time: float,
                 vehicle: "Models.Vehicle",
                 valid: bool = True,
                 name: str = None):
        super().__init__(time, name)

        self.entity = entity
        self.vehicle: "Models.Vehicle" = vehicle
        self.valid = valid  # This "valid" tag was included to handle "stay in place" movement orders
        self.message: str = '{} finished trip to node {}'.format(vehicle.name, vehicle.pos)

    def execute(self, simulator: "Models.EMSModel"):
        status = self.vehicle.onPathEnded()
        self.vehicle.relocating = False

        # If this is a "real" movement or there is a patient on the ambulance, we do something
        if self.valid or self.vehicle.patient is not None:
            
            # If it was just a relocation trip
            if status == 0:
                self.vehicle.statistics['State'].record(simulator.now(), 0)

            # If the ambulance arrived to provide service
            elif status == 1:
                # If the patient is None, then there is nothing to be done
                # That should never be the case but we check just in case.
                if self.vehicle.patient is not None:
                    # Update statistics
                    simulator.statistics['{}responseTime'.format('HS' if self.vehicle.patient.severity == 1 else 'LS')]\
                        .record(simulator.now(), simulator.now() - self.vehicle.patient.arrival_time)
                    simulator.statistics['{}averageResponseTime'.format('HS' if self.vehicle.patient.severity == 1 else 'LS')]\
                        .recordAverage(simulator.now(), simulator.now() - self.vehicle.patient.arrival_time)
                    simulator.statistics['GeneralAverageResponseTime'].recordAverage(simulator.now(), simulator.now() - self.vehicle.patient.arrival_time)
                    simulator.statistics['SpatialGeneralAverageResponseTime'].record(simulator.now(), self.vehicle.patient.node, simulator.now() - self.vehicle.patient.arrival_time)

                    # If this is a high severity patient
                    if self.vehicle.patient.severity == 1:
                        # Compute response time
                        response_time = simulator.now() - self.vehicle.patient.arrival_time

                        # Update statistics
                        simulator.statistics['PercentageALSlt10min'].recordAverage(simulator.now(), 1 if response_time <= 10 * 60 else 0)
                        simulator.statistics['PercentageALSlt8min'].recordAverage(simulator.now(), 1 if response_time <= 8 * 60 else 0)
                        simulator.statistics['PercentageALSlt7min'].recordAverage(simulator.now(), 1 if response_time <= 7 * 60 else 0)

                        simulator.statistics['SpatialHS10minCover'].record(simulator.now(), self.vehicle.patient.node, 1 if response_time <= 10 * 60 else 0)
                        simulator.statistics['SpatialHS8minCover'].record(simulator.now(), self.vehicle.patient.node, 1 if response_time <= 8 * 60 else 0)
                        simulator.statistics['SpatialHS7minCover'].record(simulator.now(), self.vehicle.patient.node, 1 if response_time <= 7 * 60 else 0)

                        simulator.statistics['SpatialHSAverageResponseTime'].record(simulator.now(), self.vehicle.patient.node, response_time)

                    else:
                        # Compure response time
                        response_time = simulator.now() - self.vehicle.patient.arrival_time
                        simulator.statistics['SpatialLSAverageResponseTime'].record(simulator.now(), self.vehicle.patient.node, response_time)

                    if self.vehicle.isUber:
                        simulator.statistics['UberResponseTime'].record(simulator.now(), simulator.now() - self.vehicle.patient.vehicle_assigned_time)

                    simulator.statistics['TravelTime'].record(simulator.now(), simulator.now() - self.vehicle.patient.vehicle_assigned_time)
                    
                    # Now there is one emergency not served less
                    simulator.assignedNotArrived -= 1

                    # Statistics
                    simulator.statistics['EmergenciesWaiting'].record(simulator.now(), simulator.assignedNotArrived)

                    # Attention starts now
                    return AmbulanceStartAttendingEvent(self.vehicle, simulator.now(), self.vehicle, self.vehicle.patient)
            elif status == 2:
                # Mark ambulance as cleaning and schedule finish cleaning event
                if not self.vehicle.isUber:
                    self.vehicle.cleaning = True
                    self.vehicle.total_busy_time += 10 * 60
                    simulator.insert(AmbulanceEndCleaningEvent(simulator, simulator.now() + 10 * 60, self.vehicle))
                    self.vehicle.statistics['BusyWorkload'].record(simulator.now(), self.vehicle.total_busy_time)
                    self.vehicle.statistics['AccumulatedWorkload'].record(simulator.now(), self.vehicle.accumulated_relocation)

                if self.vehicle.patient is not None:
                    # Statistics
                    if self.vehicle.patient.to_hospital_time > 0:
                        simulator.statistics['ToHospitalTime'].record(simulator.now(), simulator.now() - self.vehicle.patient.to_hospital_time)

                    return EmergencyLeaveSystemEvent(self.entity, simulator.now(), self.vehicle.patient, True, vehicle=self.vehicle)


class AmbulanceArriveToNodeEvent(Sim.Event):
    """
    Simulation Event:
    Ambulance arrives to node after traversing link.

                                                   (*)   
      Trip Assigned  ->  Ambulance Start -> Ambulance Arrive -> End Trip
          Event            Moving Event      To Node Event       Event
                                ^                   |
                                |                   v
                                ---------------------
    """

    def __init__(self,
                 entity: object,
                 time: float,
                 vehicle: "Models.Vehicle",
                 node: str,
                 name: str = None):
        super().__init__(time, name)
        self.entity: object = entity
        self.vehicle = vehicle
        self.node = node
        self.message: str = '{} arrived to node {}'.format(vehicle.name, node)

    def execute(self, simulator: "Models.EMSModel"):
        # Check if this is the end of the trip
        path_ended = self.vehicle.onArrivalToNode(self.node)

        if path_ended:
            return AmbulanceEndTripEvent(self.vehicle, simulator.now(), self.vehicle)
        else:
            simulator.insert(AmbulanceStartMovingEvent(self.vehicle, simulator.now(), self.vehicle, simulator.city_graph.es[self.vehicle.actual_edge]))


class TripAssignedEvent(Sim.Event):
    """
    Simulation Event:
    Ambulance is assigned to move to a specific node.

           (*)   
      Trip Assigned  ->  Ambulance Start -> Ambulance Arrive -> End Trip
          Event            Moving Event      To Node Event       Event
                                ^                   |
                                |                   v
                                ---------------------
    """

    def __init__(self,
                 entity: object,
                 time: float,
                 vehicle: "Models.Vehicle",
                 node: str,
                 name: str = None):
        super().__init__(time, name)
        self.entity: object = entity
        self.vehicle = vehicle
        self.node = node
        self.message: str = "{} assigned to move to node {}".format(vehicle.name, node)

    def execute(self, simulator: "Models.EMSModel"):
        # By default, it will not be a relocation movement
        # (this is updated at the start moving event)
        self.vehicle.relocating = False

        # If the movement is "real"
        if self.node != self.vehicle.pos:
            # Debugging statement
            if self.node is None:
                print("WHAAT?")

            # If the vehicle was not currentle moving
            if (not self.vehicle.moving) or (self.vehicle.pos == self.vehicle.to_node):
                # Compute shortest path for vehicle
                path: List[List[int]] = simulator.getShortestPath(self.vehicle.pos, self.node)

                # onAssigned callback
                self.vehicle.onAssignedMovement(path[0], [simulator.city_graph.es[p]['v'] for p in path[0]])

                # Schedule the start of the movement for the vehicle
                self.vehicle.onArrivalToNode(self.vehicle.pos)
                simulator.insert(AmbulanceStartMovingEvent(self.vehicle, simulator.now(), self.vehicle, simulator.city_graph.es[self.vehicle.actual_edge]))
            else:
                # If it was moving, we need to schedule the movement once the vehicle finishes traversing
                # the current link, and compute the path from the destination node

                # Compute shortest path for vehicle
                path = simulator.getShortestPath(self.vehicle.to_node, self.node)

                # onAssigned callback
                self.vehicle.onAssignedMovement(path[0], [simulator.city_graph.es[p]['v'] for p in path[0]])

                # Clear the scheduled vehicle movement
                simulator.clearVehicleMovement(self.vehicle)

                # Schedule the start of the movement for the vehicle in the new route
                simulator.insert(AmbulanceStartMovingEvent(self.vehicle, self.vehicle.expected_arrival, self.vehicle, simulator.city_graph.es[self.vehicle.actual_edge]))

            # Save a record of the trip
            self.vehicle.record.append((simulator.now(), self.vehicle.pos, self.node, str(self.vehicle.patient),
                                        self.vehicle.patient.hospital if self.vehicle.patient is not None else None))
        else:
            # If the movement was not real (to the same node as currently standing)
            # We register an End trip event
            return AmbulanceEndTripEvent(self.vehicle, simulator.now(), self.vehicle, valid=False)


class AssignedEvent(Sim.Event):
    """
    Simulation Event:
    An ambulance is being assigned to an emergency. We compute the optimal route and handle the required flags.
    """

    def __init__(self,
                 entity: object,
                 time: float,
                 vehicle: "Models.Vehicle",
                 emergency: "Models.Emergency",
                 name: str = None):
        super().__init__(time, name)
        self.entity: object = entity
        self.vehicle = vehicle
        self.emergency = emergency
        self.message: str = '{} assigned to {} at node {}'.format(vehicle.name, self.emergency.name, self.emergency.node)

    def execute(self, simulator: "Models.EMSModel"):
        # Compute shortest path for vehicle
        path: List[List[int]] = simulator.getShortestPath(self.vehicle.pos, self.emergency.node)

        # Vehicle onAssigned callback
        self.vehicle.onAssignedToEmergency(self.emergency, path[0], [simulator.city_graph.es[p]['v'] for p in path[0]])

        # Vehcile onArrivalToNode callback
        self.vehicle.onArrivalToNode(self.vehicle.pos)

        # Record the time that the emergency got a vehicle assigned
        self.emergency.vehicle_assigned_time = simulator.now()

        # If the optimal path is not empty, we start moving through the first link
        # otherwise (the emergency was at the current position of the ambulance), we schedule an arrive to node event
        if len(path[0]) != 0:
            simulator.insert(AmbulanceStartMovingEvent(self.vehicle, simulator.now(), self.vehicle, simulator.city_graph.es[self.vehicle.actual_edge]))
        else:
            simulator.insert(AmbulanceArriveToNodeEvent(self.vehicle, simulator.now(), self.vehicle, self.emergency.node))

        # Update statistics
        simulator.statistics['AverageAssignmentTime'].recordAverage(simulator.now(), simulator.now() - self.emergency.arrival_time)
        self.vehicle.statistics['State'].record(simulator.now(), 2)

        simulator.statistics['AvailableALSVehicles'].record(simulator.now(), len(simulator.getAvaliableVehicles(v_type=0)))
        simulator.statistics['AvailableBLSVehicles'].record(simulator.now(), len(simulator.getAvaliableVehicles(v_type=1)))

        # If this vehicle is an external RHS
        if self.vehicle.isUber:
            simulator.statistics['UberCalls'].record(simulator.now(), 1)

        # Debugging statement
        if self.emergency is None or self.emergency.node is None:
            print("WHAAT?")

        # Save a record of the trip
        self.vehicle.record.append((simulator.now(), self.vehicle.pos, self.emergency.node, str(self.vehicle.patient),
                                    self.vehicle.patient.hospital if self.vehicle.patient is not None else None))


class AmbulanceRecoverFromBreak(Sim.Event):
    """
    Simulation Event:
    This event is called when an ambulance has spent the stipulated time not relocating after
    an intense relocation workload.
    """

    def __init__(self,
                 entity: object,
                 time: float,
                 vehicle: "Models.Vehicle",
                 accumulated_workload: float,
                 name: str = None):
        super().__init__(time, name)
        self.entity: object = entity
        self.vehicle = vehicle
        self.accumulated_workload = accumulated_workload
        self.message: str = '{} recovers from relocation break...'\
            .format(vehicle.name)

    def execute(self, simulator: "Models.EMSModel"):
        # Flag the vehicle as able to relocate an restart the counter
        self.vehicle.can_relocate = True
        self.vehicle.accumulated_relocation = 0

        # Update statistics
        self.vehicle.statistics['BusyWorkload'].record(simulator.now(), self.vehicle.total_busy_time)
        self.vehicle.statistics['AccumulatedWorkload'].record(simulator.now(), self.vehicle.accumulated_relocation)


class AmbulanceStartMovingEvent(Sim.Event):
    """
    Simulation Event:
    After an ambulance was assigned to move to a specific node, we execute this event to
    handle the movement through a specific edge of the graph.

                                (*)    
      Trip Assigned  ->  Ambulance Start -> Ambulance Arrive -> End Trip
          Event            Moving Event      To Node Event       Event
                                ^                   |
                                |                   v
                                ---------------------
    """

    def __init__(self,
                 entity: object,
                 time: float,
                 vehicle: "Models.Vehicle",
                 edge: igraph.Edge,
                 name: str = None):
        super().__init__(time, name)
        self.entity: object = entity
        self.vehicle = vehicle
        self.edge = edge
        self.edge_id = edge['edgeid']
        self.message: str = '{} starts moving to node {} through edge {}...'\
            .format(vehicle.name, edge['v'], edge['edgeid'])

    def execute(self, simulator: "Models.EMSModel"):
        # Compute travel time
        self.travel_time = self.edge['length'] / simulator.parameters.getSpeedList(simulator.timePeriod())[self.edge.index]

        # OnMoving Callback
        self.vehicle.onMovingToNextNode(simulator.now() + self.travel_time)

        # Update total busy time counter
        self.vehicle.total_busy_time += self.travel_time

        # If the ambulance is not currently providing service
        # then this is a relocation movement
        if self.vehicle.patient is None:
            # Update relocation workload counter and flag as relocating
            self.vehicle.reposition_workload += self.travel_time
            self.vehicle.relocating = True

            # If the ambulance has not exceeded the maximum allowed relocation
            if self.vehicle.can_relocate:
                self.vehicle.accumulated_relocation += self.travel_time

                # If the ambulance now exceeds the maximum allowed relocation
                if self.vehicle.accumulated_relocation >= simulator.parameters.target_relocation_time:
                    self.vehicle.can_relocate = False

                    # After a break, this ambulance will be available to relocate again
                    simulator.insert(AmbulanceRecoverFromBreak(simulator, simulator.now() + simulator.parameters.relocation_cooldown, self.vehicle, self.vehicle.accumulated_relocation))

            # Update statistics
            self.vehicle.statistics['RelocationTime'].record(simulator.now(), self.vehicle.reposition_workload)
            self.vehicle.statistics['State'].record(simulator.now(), 1)
        else:
            self.vehicle.statistics['State'].record(simulator.now(), 2)

        self.vehicle.statistics['BusyWorkload'].record(simulator.now(), self.vehicle.total_busy_time)
        self.vehicle.statistics['AccumulatedWorkload'].record(simulator.now(), self.vehicle.accumulated_relocation)
        self.vehicle.statistics['MetersDriven'].record(simulator.now(), self.vehicle.statistics['MetersDriven'].data[-1][1] + self.edge['length'])

        # Schedule vehicle arrival to node
        simulator.insert(AmbulanceArriveToNodeEvent(self.vehicle, simulator.now() + self.travel_time, self.vehicle, self.edge['v']))


class EmergencyArrivalEvent(Sim.Event):
    """
    Simulation Event:
    An emergency arrived to the system.
    We need to check for available vehicles to dispatch and run the optimization procedure.
    """

    def __init__(self,
                 entity: object,
                 time: float,
                 node: str,
                 severity: int,
                 disposition_code: int,
                 name: str = None):
        super().__init__(time, name)

        self.entity: object = entity
        self.node: str = node
        self.severity: int = severity
        self.disposition_code: int = disposition_code
        self.emergency = Models.Emergency(self.time, self.node, self.severity, self.disposition_code)

        self.message: str = "Emergency arrived at node {}".format(node)

    def execute(self, simulator: "Models.EMSModel"):
        # Create the emergency and append it to the reference list
        simulator.activeEmergencies.append(self.emergency)

        # Schedule the next emergency, if there is one
        try:
            simulator.insert(next(simulator.arrival_generator))
        except StopIteration:
            pass
        
        # Compute the borough of the emergency, if not available
        self.emergency.assignBorough(simulator)

        # New emergency is not yet being served, we update this counter variable
        simulator.assignedNotArrived += 1

        # Update the emergenciesWaiting statistic
        simulator.statistics['EmergenciesWaiting'].record(simulator.now(), simulator.assignedNotArrived)

        # If this is a static model (no online relocation)
        if simulator.parameters.force_static:
            # We only care about dispatching
            dispatching_dict = simulator.optimizer.dispatch(simulator, simulator.parameters, 0 if self.severity == 1 else 1, self.emergency.borough)

            # Schedule dispatching
            for v in dispatching_dict:
                # Schedule the assignment event
                simulator.insert(AssignedEvent(simulator, simulator.now(),
                                               v, dispatching_dict[v]))

                # Mark emergency as assigned
                dispatching_dict[v].markStatus(1)
                simulator.assignedEmergencies.append(dispatching_dict[v])
        else:
            # Chain an optimization event right after the arrival
            return RelocationAndDispatchingEvent(simulator, simulator.now(), 0 if self.emergency.severity == 1 else 1, self.emergency.borough)


class AmbulanceLeavingEvent(Sim.Event):
    """
    Simulation Event:
    Ambulance is now effectively leaving the system (End of shift and idle).
    """

    def __init__(self,
                 entity: object,
                 time: float,
                 vehicle: "Models.Vehicle",
                 name: str = None):

        super().__init__(time, name)

        self.entity: object = entity
        self.vehicle: Models.Vehicle = vehicle

        self.message: str = "{} leaving the system".format(vehicle.name)

    def execute(self, simulator: "Models.EMSModel"):
        # Recover Time in System statistic for this vehicle
        self.vehicle.statistics['TimeInSystem'].record(simulator.now() - self.vehicle.arrival_time)

        # Add the last element to the log of the vehicle
        self.vehicle.record.append((simulator.now(), self.vehicle.pos, None, str(self.vehicle.patient),
                                    self.vehicle.patient.hospital if self.vehicle.patient is not None else None))

        # Remove vehicle from list in EMSModel object
        simulator.vehicles.remove(self.vehicle)

        # Add the statistics of the vehicle to the dictionary that will be exported
        simulator.vehicle_statistics[self.vehicle.name] = {'Statistics': self.vehicle.statistics, 'Record': self.vehicle.record}

        # Update the vehicles in system statistic
        simulator.statistics[('ALS' if self.vehicle.type == 0 else 'BLS') + 'VehiclesInSystem'].record(simulator.now(), len([v for v in simulator.vehicles if v.type == self.vehicle.type]))


class MarkAmbulanceLeavingEvent(Sim.Event):
    """
    Simulation Event:
    The ambulance has arrived to the end of its shift, be we still need to check if it is providing service.
    """

    def __init__(self,
                 entity: object,
                 time: float,
                 vehicle: "Models.Vehicle",
                 name: str = None):

        super().__init__(time, name)

        self.entity: object = entity
        self.vehicle: Models.Vehicle = vehicle

        self.message: str = "{} finished its shift".format(vehicle.name)

    def execute(self, simulator: "Models.EMSModel"):
        # Flag ambulance as leaving
        self.vehicle.leaving = True

        # If it doesn't have a patient (is not providing service) and it is not being cleaned either
        # it leaves immediately
        if self.vehicle.patient is None and not self.vehicle.cleaning:
            return AmbulanceLeavingEvent(self.vehicle, simulator.now(), self.vehicle)


class AmbulanceArrivalEvent(Sim.Event):
    """
    Simulation Event:
    Receive an instance of a Vehicle object and insert it into the system at the specified location.
    """

    def __init__(self,
                 entity: object,
                 time: float,
                 node: str,
                 vehicle: "Models.Vehicle",
                 prior_worked_time: float = 0,
                 name: str = None):

        super().__init__(time, name)

        self.entity: object = entity
        self.node: str = node
        self.prior_worked_time: float = prior_worked_time
        self.vehicle: Models.Vehicle = vehicle
        self.vehicle_type = self.vehicle.type

        self.message: str = "Ambulance arrived to the system at node {}!".format(node)

    def execute(self, simulator: "Models.EMSModel"):
        # Mark the arrival time at the vehicle object
        self.vehicle.arrival_time = simulator.now()

        # Insert into the list of the EMSModel object
        simulator.vehicles.append(self.vehicle)

        # Schedule the planned end of shift
        # (This is Mark Ambulance leaving because what we want is to flagg the ambulance
        # and not remove it immediately because it might be providing service by this time.)
        simulator.insert(MarkAmbulanceLeavingEvent(simulator, simulator.now() + simulator.parameters.vehicle_shift(self.vehicle) - self.prior_worked_time, self.vehicle))
        
        # Number of ambulances in the system changes so we update the statistic
        simulator.statistics[('ALS' if self.vehicle.type == 0 else 'BLS') + 'VehiclesInSystem'].record(simulator.now(), len([v for v in simulator.vehicles if v.type == self.vehicle.type]))

        # If this is not the initial positioning event
        if simulator.now() > 0.1:
            # Perform a dispatching operation since we have a newly available ambulance
            dispatching_dict = simulator.optimizer.dispatch(simulator, simulator.parameters, self.vehicle.type, self.vehicle.borough)
            
            # If the ambulance was assigned to an emergency
            if self.vehicle in dispatching_dict.keys():
                # Mark emergency as assigned
                dispatching_dict[self.vehicle].markStatus(1)
                simulator.assignedEmergencies.append(dispatching_dict[self.vehicle])

                # Schedule the assignment event
                return AssignedEvent(simulator, simulator.now(), self.vehicle, dispatching_dict[self.vehicle])
            
            # If not, we check for redeployment optimization
            return AmbulanceRedeployEvent(simulator, simulator.now(), self.vehicle)


class HospitalSettingEvent(Sim.Event):
    """
    Simulation Event:
    Initializing the list of available hospitals for the model.
    """

    def __init__(self,
                 entity: object,
                 time: float,
                 name: str = None):

        super().__init__(time, name)

        self.entity: object = entity

        self.message: str = "Setting hospitals in place..."

    def execute(self, simulator: "Models.EMSModel"):
        self.hospital_nodes: List[str] = list(set([h for _, hospitals in simulator.hospitals.items() for h in hospitals]))


class EndSimulationEvent(Sim.Event):
    """
    Simulation event:
    End of simulation. Recover statistics for vehicles that have not yet left
    the system and remove all events in the queue.
    """

    def __init__(self,
                 entity: "Models.EMSModel",
                 time: float):

        super().__init__(time, 'EndSimulation')
        self.entity = entity

        self.message: str = "End of simulation"

    def execute(self, simulator: "Models.EMSModel"):
        while len(simulator.vehicles) > 0:
            v = simulator.vehicles[0]
            E = AmbulanceLeavingEvent(simulator, simulator.now(), v)
            E.execute(simulator)
        simulator.statistics['RunTime'].record(time.time() - simulator.sim_start_time)
        simulator.events.empty()


# TODO:
# This event was not used, I commented it now, if we get errors during testing, we should look into it
# class FairBalanceEvent(Sim.Event):

#     def __init__(self,
#                  entity: object,
#                  time: float,
#                  name: str = None):
#         super().__init__(time, name)

#         self.entity = entity
#         self.message: str = 'Reassigning positions according to workload ...'

#     def execute(self, simulator):
#         for b in range(1, 6):
#             for s in range(2):
#                 vehicles = [(v, v.total_busy_time) for v in simulator.vehicles if v.borough == b and v.type == s]
#                 vehicles.sort(key=lambda v: v[1])

#                 for i in range(int(len(vehicles) / 2)):
#                     aux = vehicles[i][0].station
#                     vehicles[i][0].station = vehicles[-(i + 1)][0].station
#                     if vehicles[i][0].patient is None:
#                         simulator.insert(TripAssignedEvent(simulator, simulator.now(), vehicles[i][0], vehicles[i][0].station))
#                     else:
#                         vehicles[i][0].station_changed = True

#                     vehicles[-(i + 1)][0].station = aux
#                     if vehicles[-(i + 1)][0].patient is None:
#                         simulator.insert(TripAssignedEvent(simulator, simulator.now(), vehicles[-(i + 1)][0], vehicles[-(i + 1)][0].station))
#                     else:
#                         vehicles[-(i + 1)][0].station_changed = True

#                     if simulator.verbose:
#                         print('{} switching with {}'.format(vehicles[i][0].name, vehicles[-(i + 1)][0].name))