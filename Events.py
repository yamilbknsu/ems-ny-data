import time
import igraph
import numpy as np
from typing import List, Optional, Tuple

# Internal imports
import Models
import SimulatorBasics as Sim


class DebugEvent(Sim.Event):

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
            simulator.parameters.mean_busytime[key[1]].at[key[0], key[2]] = np.mean(value)

        simulator.insert(ComputeParemeters(simulator, simulator.now() + 3600))


class FairBalanceEvent(Sim.Event):

    def __init__(self,
                 entity: object,
                 time: float,
                 name: str = None):
        super().__init__(time, name)

        self.entity = entity
        self.message: str = 'Reassigning positions according to workload ...'

    def execute(self, simulator):
        for b in range(1, 6):
            for s in range(2):
                vehicles = [(v, v.total_busy_time) for v in simulator.vehicles if v.borough == b and v.type == s]
                vehicles.sort(key=lambda v: v[1])

                for i in range(int(len(vehicles) / 2)):
                    aux = vehicles[i][0].station
                    vehicles[i][0].station = vehicles[-(i + 1)][0].station
                    if vehicles[i][0].patient is None:
                        simulator.insert(TripAssignedEvent(simulator, simulator.now(), vehicles[i][0], vehicles[i][0].station))
                    else:
                        vehicles[i][0].station_changed = True

                    vehicles[-(i + 1)][0].station = aux
                    if vehicles[-(i + 1)][0].patient is None:
                        simulator.insert(TripAssignedEvent(simulator, simulator.now(), vehicles[-(i + 1)][0], vehicles[-(i + 1)][0].station))
                    else:
                        vehicles[-(i + 1)][0].station_changed = True

                    if simulator.verbose:
                        print('{} switching with {}'.format(vehicles[i][0].name, vehicles[-(i + 1)][0].name))


class RelocationAndDispatchingEvent(Sim.Event):

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
        if self.severity != 3:
            simulator.statistics['AvailableALSVehicles'].record(simulator.now(), len(simulator.getAvaliableVehicles(v_type=0)))
            simulator.statistics['AvailableBLSVehicles'].record(simulator.now(), len(simulator.getAvaliableVehicles(v_type=1)))

            if self.severity == 1 or len(simulator.getAvaliableVehicles(0, self.borough)) >= len(simulator.getUnassignedEmergencies(1, self.borough)):
                optimal_positions, reposition_dict, dispatching_dict = simulator.optimizer.optimize(simulator, simulator.parameters, False, severity=self.severity, borough=self.borough)
            else:
                # Manual nearest dispatcher
                optimal_positions, reposition_dict = [], {}
                dispatching_dict = {}
                weights = np.array(simulator.city_graph.es['length']) / simulator.parameters.getSpeedList(simulator.timePeriod())
                vehicles = [[[ambulance for ambulance in simulator.getAvaliableVehicles(v_type=v, borough=b)]
                            for v in range(simulator.parameters.vehicle_types)]
                            for b in range(1, 6)]

                vehicle_positions = [[[ambulance.to_node
                                       for ambulance in v]
                                      for v in b]
                                     for b in vehicles]

                used_vehicles: List[Tuple[int, int, int]] = []
                for e in simulator.getUnassignedEmergencies(1, self.borough):

                    travel_times = np.array(simulator.city_graph.shortest_paths(vehicle_positions[e.borough - 1][0], e.node, weights))
                    valid_indexes = np.where(travel_times < 8 * 60)[0]

                    if len(valid_indexes) > 0:
                        candidates = list(zip(valid_indexes, travel_times[valid_indexes].squeeze().reshape(-1)))
                        candidates.sort(key=lambda c: c[1])
                    else:
                        candidates = list(zip(range(travel_times.shape[0]), travel_times.reshape(-1)))
                        candidates.sort(key=lambda c: c[1])

                    for c in candidates:
                        if (e.borough, 0, c[0]) not in used_vehicles:
                            used_vehicles.append((e.borough, 0, c[0]))
                            dispatching_dict[vehicles[e.borough - 1][0][c[0]]] = e
                            break

            simulator.ambulance_stations[self.borough][0 if self.severity == 0 else 1] = optimal_positions

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

        if simulator.optimizer is not None:
            for b in range(1, 6):
                optimal_positions_ALS, reposition_dict_ALS, _ = simulator.optimizer.optimize(simulator, simulator.parameters, True, severity=0, borough=b)
                optimal_positions_BLS, reposition_dict_BLS, _ = simulator.optimizer.optimize(simulator, simulator.parameters, True, severity=1, borough=b)

                for v in reposition_dict_ALS:
                    v.station = reposition_dict_ALS[v]
                    v.teleportToNode(reposition_dict_ALS[v])

                    self.new_positions.append([v.name, reposition_dict_ALS[v]])

                for v in reposition_dict_BLS:
                    v.station = reposition_dict_BLS[v]
                    v.teleportToNode(reposition_dict_BLS[v])

                    self.new_positions.append([v.name, reposition_dict_BLS[v]])

                # Statistics
                C = simulator.parameters.candidates_borough[b]
                for candidate in C:
                    simulator.statistics['SpatialALSRelocation'].record(simulator.now(), candidate, 1 if candidate in optimal_positions_ALS[0] else 0)
                    simulator.statistics['SpatialBLSRelocation'].record(simulator.now(), candidate, 1 if candidate in optimal_positions_ALS[1] else 0)


class EmergencyLeaveSystemEvent(Sim.Event):

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
                                         self.emergency.start_attending_time, self.emergency.to_hospital_time, self.emergency.disposition_code, self.emergency.hospital, (self.vehicle is not None and self.vehicle.isUber)))

        simulator.activeEmergencies.remove(self.emergency)
        if self.emergency in simulator.assignedEmergencies:
            simulator.assignedEmergencies.remove(self.emergency)

        if self.vehicle is not None:
            self.vehicle.patient = None

            # Statistics recovery
            simulator.statistics['EmergenciesServed'].record()
            simulator.statistics['EmergenciesTimeInSystem'].record(simulator.now() - self.emergency.arrival_time)

            if self.emergency.severity == 1:
                simulator.statistics['NumberHSemergencies'].record()
            else:
                simulator.statistics['NumberLSemergencies'].record()

            self.vehicle.statistics['EmergenciesServed'].record()

            if self.vehicle.isUber:
                simulator.ubers.remove(self.vehicle)
                simulator.uber_seconds += simulator.now() - self.vehicle.arrive_in_system


class AmbulanceFinishAttendingEvent(Sim.Event):

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
        # Schedule the end of the atention
        hospital_type = simulator.getHospitalType(self.emergency)
        if hospital_type == 0:
            # Mark ambulance as cleaning and schedule finish cleaning event
            self.vehicle.cleaning = True
            simulator.insert(AmbulanceEndCleaningEvent(simulator, simulator.now() + 10 * 60, self.vehicle))

            return EmergencyLeaveSystemEvent(simulator, simulator.now(), self.emergency, True, self.vehicle)
        else:
            # Get nearest hospital of the corresponding type
            distances = simulator.getShortestDistances(self.emergency.node, simulator.hospitals[hospital_type])[0]
            nearest_node = simulator.hospitals[hospital_type][np.argmin(distances)]

            self.emergency.to_hospital_time = simulator.now()
            self.emergency.markStatus(3)
            self.emergency.hospital = nearest_node
            return TripAssignedEvent(self.emergency, simulator.now(), self.vehicle, nearest_node)


class AmbulanceStartAttendingEvent(Sim.Event):

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
        # return TripAssignedEvent(self.vehicle, simulator.now(), self.vehicle, self.vehicle.station)
        # return RelocationAndDispatchingEvent(simulator, simulator.now(), self.vehicle.type, self.vehicle.borough)

        if not self.vehicle.isUber:
            if not self.vehicle.station_changed:
                optimal_positions, reposition_dict = simulator.optimizer.redeploy(simulator, simulator.parameters, self.vehicle)
                self.vehicle.station = reposition_dict[self.vehicle]
                simulator.insert(TripAssignedEvent(simulator, simulator.now(), self.vehicle, reposition_dict[self.vehicle]))
            else:
                self.vehicle.station_changed = False
                return TripAssignedEvent(self.vehicle, simulator.now(), self.vehicle, self.vehicle.station)


class AmbulanceEndCleaningEvent(Sim.Event):

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
        self.vehicle.statistics['State'].record(simulator.now(), 0)

        self.vehicle.cleaning = False

        if self.vehicle.leaving:
            return AmbulanceLeavingEvent(simulator, simulator.now(), self.vehicle)

        return AmbulanceRedeployEvent(self.entity, simulator.now(), self.vehicle)


class AmbulanceEndTripEvent(Sim.Event):

    def __init__(self,
                 entity: object,
                 time: float,
                 vehicle: "Models.Vehicle",
                 valid: bool = True,
                 name: str = None):
        super().__init__(time, name)

        self.entity = entity
        self.vehicle: "Models.Vehicle" = vehicle
        self.valid = valid
        self.message: str = '{} finished trip to node {}'.format(vehicle.name, vehicle.pos)

    def execute(self, simulator: "Models.EMSModel"):
        status = self.vehicle.onPathEnded()
        self.vehicle.relocating = False

        if self.valid:
            if status == 0:
                self.vehicle.statistics['State'].record(simulator.now(), 0)

                # return RelocationAndDispatchingEvent(simulator, simulator.now(), self.vehicle.type, self.vehicle.borough)
            elif status == 1:
                if self.vehicle.patient is not None:
                    # Statistics
                    simulator.statistics['{}responseTime'.format('HS' if self.vehicle.patient.severity == 1 else 'LS')]\
                        .record(simulator.now(), simulator.now() - self.vehicle.patient.arrival_time)
                    simulator.statistics['{}averageResponseTime'.format('HS' if self.vehicle.patient.severity == 1 else 'LS')]\
                        .recordAverage(simulator.now(), simulator.now() - self.vehicle.patient.arrival_time)
                    simulator.statistics['GeneralAverageResponseTime'].recordAverage(simulator.now(), simulator.now() - self.vehicle.patient.arrival_time)
                    simulator.statistics['SpatialGeneralAverageResponseTime'].record(simulator.now(), self.vehicle.patient.node, simulator.now() - self.vehicle.patient.arrival_time)

                    if self.vehicle.patient.severity == 1:
                        response_time = simulator.now() - self.vehicle.patient.arrival_time
                        simulator.statistics['PercentageALSlt10min'].recordAverage(simulator.now(), 1 if response_time <= 10 * 60 else 0)
                        simulator.statistics['PercentageALSlt8min'].recordAverage(simulator.now(), 1 if response_time <= 8 * 60 else 0)
                        simulator.statistics['PercentageALSlt7min'].recordAverage(simulator.now(), 1 if response_time <= 7 * 60 else 0)

                        simulator.statistics['SpatialHS10minCover'].record(simulator.now(), self.vehicle.patient.node, 1 if response_time <= 10 * 60 else 0)
                        simulator.statistics['SpatialHS8minCover'].record(simulator.now(), self.vehicle.patient.node, 1 if response_time <= 8 * 60 else 0)
                        simulator.statistics['SpatialHS7minCover'].record(simulator.now(), self.vehicle.patient.node, 1 if response_time <= 7 * 60 else 0)

                        simulator.statistics['SpatialHSAverageResponseTime'].record(simulator.now(), self.vehicle.patient.node, response_time)

                    else:
                        response_time = simulator.now() - self.vehicle.patient.arrival_time
                        simulator.statistics['SpatialLSAverageResponseTime'].record(simulator.now(), self.vehicle.patient.node, response_time)

                    if self.vehicle.isUber:
                        simulator.statistics['UberResponseTime'].record(simulator.now(), simulator.now() - self.vehicle.patient.vehicle_assigned_time)

                    simulator.statistics['TravelTime'].record(simulator.now(), simulator.now() - self.vehicle.patient.vehicle_assigned_time)

                    simulator.assignedNotArrived -= 1
                    # Statistics
                    simulator.statistics['EmergenciesWaiting'].record(simulator.now(), simulator.assignedNotArrived)

                    return AmbulanceStartAttendingEvent(self.vehicle, simulator.now(), self.vehicle, self.vehicle.patient)
            elif status == 2:
                # Mark ambulance as cleaning and schedule finish cleaning event
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
        path_ended = self.vehicle.onArrivalToNode(self.node)

        if path_ended:
            return AmbulanceEndTripEvent(self.vehicle, simulator.now(), self.vehicle)
        else:
            simulator.insert(AmbulanceStartMovingEvent(self.vehicle, simulator.now(), self.vehicle, simulator.city_graph.es[self.vehicle.actual_edge]))


class TripAssignedEvent(Sim.Event):

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
        self.vehicle.relocating = False
        if self.node != self.vehicle.pos:
            if self.node is None:
                print("WHAAT?")

            if (not self.vehicle.moving) or (self.vehicle.pos == self.vehicle.to_node):
                # Compute shortest path for vehicle
                path: List[List[int]] = simulator.getShortestPath(self.vehicle.pos, self.node)

                # onAssigned callback
                self.vehicle.onAssignedMovement(path[0], [simulator.city_graph.es[p]['v'] for p in path[0]])

                # Schedule the start of the movement for the vehicle
                self.vehicle.onArrivalToNode(self.vehicle.pos)
                simulator.insert(AmbulanceStartMovingEvent(self.vehicle, simulator.now(), self.vehicle, simulator.city_graph.es[self.vehicle.actual_edge]))
            else:
                # Compute shortest path for vehicle
                path = simulator.getShortestPath(self.vehicle.to_node, self.node)

                # onAssigned callback
                self.vehicle.onAssignedMovement(path[0], [simulator.city_graph.es[p]['v'] for p in path[0]])

                # Clear the scheduled vehicle movement
                simulator.clearVehicleMovement(self.vehicle)

                # Schedule the start of the movement for the vehicle in the new route
                simulator.insert(AmbulanceStartMovingEvent(self.vehicle, self.vehicle.expected_arrival, self.vehicle, simulator.city_graph.es[self.vehicle.actual_edge]))

            # Save a record of the trip
            self.vehicle.record.append((simulator.now(), self.vehicle.pos, self.node, self.vehicle.patient,
                                        self.vehicle.patient.hospital if self.vehicle.patient is not None else None))
        else:
            return AmbulanceEndTripEvent(self.vehicle, simulator.now(), self.vehicle, valid=False)


class AssignedEvent(Sim.Event):

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

        # onAssigned callback
        self.vehicle.onAssignedToEmergency(self.emergency, path[0], [simulator.city_graph.es[p]['v'] for p in path[0]])

        # Schedule the start of the movement for the vehicle
        self.vehicle.onArrivalToNode(self.vehicle.pos)

        self.emergency.vehicle_assigned_time = simulator.now()

        if len(path[0]) != 0:
            simulator.insert(AmbulanceStartMovingEvent(self.vehicle, simulator.now(), self.vehicle, simulator.city_graph.es[self.vehicle.actual_edge]))
        else:
            simulator.insert(AmbulanceArriveToNodeEvent(self.vehicle, simulator.now(), self.vehicle, self.emergency.node))

        # Statistics
        simulator.statistics['AverageAssignmentTime'].recordAverage(simulator.now(), simulator.now() - self.emergency.arrival_time)
        self.vehicle.statistics['State'].record(simulator.now(), 2)

        simulator.statistics['AvailableALSVehicles'].record(simulator.now(), len(simulator.getAvaliableVehicles(v_type=0)))
        simulator.statistics['AvailableBLSVehicles'].record(simulator.now(), len(simulator.getAvaliableVehicles(v_type=1)))

        if self.vehicle.isUber:
            simulator.statistics['UberCalls'].record(simulator.now(), 1)

        if self.emergency is None or self.emergency.node is None:
            print("WHAAT?")

        # Save a record of the trip
        self.vehicle.record.append((simulator.now(), self.vehicle.pos, self.emergency.node, self.vehicle.patient,
                                    self.vehicle.patient.hospital if self.vehicle.patient is not None else None))


class AmbulanceRecoverFromBreak(Sim.Event):

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
        self.vehicle.can_relocate = True
        self.vehicle.accumulated_relocation = 0
        self.vehicle.statistics['BusyWorkload'].record(simulator.now(), self.vehicle.total_busy_time)
        self.vehicle.statistics['AccumulatedWorkload'].record(simulator.now(), self.vehicle.accumulated_relocation)


class AmbulanceStartMovingEvent(Sim.Event):

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
        self.travel_time = self.edge['length'] / simulator.parameters.getSpeedList(simulator.timePeriod())[self.edge.index]
        self.vehicle.onMovingToNextNode(simulator.now() + self.travel_time)

        self.vehicle.total_busy_time += self.travel_time
        if self.vehicle.patient is None:
            self.vehicle.reposition_workload += self.travel_time
            self.vehicle.relocating = True

            if self.vehicle.can_relocate:
                self.vehicle.accumulated_relocation += self.travel_time

                if self.vehicle.accumulated_relocation >= simulator.parameters.target_relocation_time:
                    self.vehicle.can_relocate = False
                    simulator.insert(AmbulanceRecoverFromBreak(simulator, simulator.now() + simulator.parameters.relocation_cooldown, self.vehicle, self.vehicle.accumulated_relocation))

            self.vehicle.statistics['RelocationTime'].record(simulator.now(), self.vehicle.reposition_workload)
            self.vehicle.statistics['State'].record(simulator.now(), 1)
            self.vehicle.statistics['BusyWorkload'].record(simulator.now(), self.vehicle.total_busy_time)
            self.vehicle.statistics['AccumulatedWorkload'].record(simulator.now(), self.vehicle.accumulated_relocation)

        self.vehicle.statistics['MetersDriven'].record(simulator.now(), self.vehicle.statistics['MetersDriven'].data[-1][1] + self.edge['length'])

        # Schedule vehicle arrival to node
        simulator.insert(AmbulanceArriveToNodeEvent(self.vehicle, simulator.now() + self.travel_time, self.vehicle, self.edge['v']))


class EmergencyArrivalEvent(Sim.Event):

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

        self.emergency.assignBorough(simulator)

        simulator.assignedNotArrived += 1
        # Statistics
        simulator.statistics['EmergenciesWaiting'].record(simulator.now(), simulator.assignedNotArrived)

        # Chain an optimization event right next to the arrival
        return RelocationAndDispatchingEvent(simulator, simulator.now(), 0 if self.emergency.severity == 1 else 1, self.emergency.borough)


class AmbulanceLeavingEvent(Sim.Event):
    """
    Intended to use in the ambulance 'lifecicle'
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
        self.vehicle.statistics['TimeInSystem'].record(simulator.now() - self.vehicle.arrival_time)
        self.vehicle.record.append((simulator.now(), self.vehicle.pos, None, self.vehicle.patient,
                                    self.vehicle.patient.hospital if self.vehicle.patient is not None else None))

        simulator.vehicles.remove(self.vehicle)
        simulator.vehicle_statistics[self.vehicle.name] = {'Statistics': self.vehicle.statistics, 'Record': self.vehicle.record}

        simulator.statistics[('ALS' if self.vehicle.type == 0 else 'BLS') + 'VehiclesInSystem'].record(simulator.now(), len([v for v in simulator.vehicles if v.type == self.vehicle.type]))


class MarkAmbulanceLeavingEvent(Sim.Event):
    """
    Intended to use in the ambulance 'lifecicle'
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
        self.vehicle.leaving = True

        if self.vehicle.patient is None and not self.vehicle.cleaning:
            return AmbulanceLeavingEvent(self.vehicle, simulator.now(), self.vehicle)


class AmbulanceArrivalEvent(Sim.Event):
    """
    Intended to use in the ambulance 'lifecicle'
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

        self.message: str = "Ambulance arrived to the system at node {}!".format(node)

    def execute(self, simulator: "Models.EMSModel"):
        self.vehicle.arrival_time = simulator.now()

        simulator.vehicles.append(self.vehicle)

        simulator.insert(MarkAmbulanceLeavingEvent(simulator, simulator.now() + simulator.parameters.vehicle_shift(self.vehicle) - self.prior_worked_time, self.vehicle))
        simulator.statistics[('ALS' if self.vehicle.type == 0 else 'BLS') + 'VehiclesInSystem'].record(simulator.now(), len([v for v in simulator.vehicles if v.type == self.vehicle.type]))

        if simulator.now() > 0.1:
            return RelocationAndDispatchingEvent(simulator, simulator.now(), severity=self.vehicle.type, borough=self.vehicle.borough)


class HospitalSettingEvent(Sim.Event):
    """
    Intended to use in the ambulance 'lifecicle'
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

    def __init__(self,
                 entity: "Models.EMSModel",
                 time: float):

        super().__init__(time, 'EndSimulation')
        self.entity = entity

        self.message: str = "End of simulation"

    def execute(self, simulator: "Models.EMSModel"):
        simulator.statistics['RunTime'].record(time.time() - simulator.sim_start_time)
        simulator.events.empty()
