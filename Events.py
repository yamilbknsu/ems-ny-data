import time
import igraph
import numpy as np
from pprint import pprint
from typing import List, Optional

# Internal imports
import Models
import Solvers
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


class ComputeQandPEvent(Sim.Event):

    def __init__(self,
                entity: object,
                time: float,
                name: str = None):
        super().__init__(time, name)

        self.entity = entity
        self.message: str = 'Precomputing Q and P ....'

    def execute(self, simulator: "Models.EMSModel"):
        # Update the mean values in the simulation parameters
        for key, value in simulator.time_records.items():
            simulator.parameters.mean_busytime[key[1]].at[key[0], key[2]] = np.mean(value)
        
        simulator.parameters.computeQandP(simulator.timePeriod())

        simulator.insert(ComputeQandPEvent(simulator, simulator.now() + 3600))


class ShiftChangeEvent(Sim.Event):
    
    def __init__(self, entity: object, time: float, name: str = None):
        super().__init__(time, name)

        self.entity = entity
        self.message: str = 'Changing time period, resetting values...'

    def execute(self, simulator: "Models.EMSModel"):
        pass
        #for v in simulator.vehicles:
        #    v.reposition_workload = 0
        #    v.statistics['RelocationTime'].record(simulator.now(), v.reposition_workload)

        ## Get the index of the new shift
        #actual_time = (simulator.now() % 86400) // 3600
        #c = 0
        #for s in simulator.parameters.time_shifts:
        #    if actual_time < sum(simulator.parameters.time_shifts[:c+1]) * 3600:
        #        break
        #    c = c + 1
        #period_length = simulator.parameters.time_shifts[c]

        ## Schedule the change of the next time period
        #simulator.insert(ShiftChangeEvent(simulator, simulator.now() + period_length*3600))


class RelocationEvent(Sim.Event):

    def __init__(self,
                entity: object,
                time: float,
                name: str = None):
        super().__init__(time, name)

        self.entity = entity
        self.message: str = 'Relocation Event'

    def execute(self, simulator: "Models.EMSModel"):
        for b in range(1, 6):
            optimal_positions, reposition_dict = simulator.repositioner.relocate(simulator, simulator.parameters, initial=False, borough = b,
                                                                                 workload_restrinction=simulator.parameters.apply_workload_restriction)
            simulator.ambulance_stations[b] = optimal_positions

            for v in reposition_dict:
                v.station = reposition_dict[v]
                simulator.insert(TripAssignedEvent(simulator, simulator.now(), v, reposition_dict[v]))

            # Statistics
            C = simulator.parameters.candidates_borough[b]
            for candidate in C:
                simulator.statistics['SpatialALSRelocation'].record(simulator.now(), candidate, 1 if candidate in optimal_positions[0] else 0)
                simulator.statistics['SpatialBLSRelocation'].record(simulator.now(), candidate, 1 if candidate in optimal_positions[1] else 0)
        
        simulator.insert(RelocationEvent(simulator, simulator.now() + simulator.parameters.relocation_period))


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

        if simulator.repositioner is not None:
            for b in range(1,6):
                optimal_positions, reposition_dict = simulator.repositioner.relocate(simulator, simulator.parameters, borough = b, initial=True)
                simulator.ambulance_stations[b] = optimal_positions
                
                for v in reposition_dict:
                    v.station = reposition_dict[v]
                    v.teleportToNode(reposition_dict[v])
                    simulator.registerVehicleStationChange(v, reposition_dict[v])

                    self.new_positions.append([v.name, reposition_dict[v]])
                
                # Statistics
                C = simulator.parameters.candidates_borough[b]
                for candidate in C:
                    simulator.statistics['SpatialALSRelocation'].record(simulator.now(), candidate, 1 if candidate in optimal_positions[0] else 0)
                    simulator.statistics['SpatialBLSRelocation'].record(simulator.now(), candidate, 1 if candidate in optimal_positions[1] else 0)
            
            if simulator.parameters.relocation_optimization:
                simulator.insert(RelocationEvent(simulator, simulator.now() + simulator.parameters.relocation_period))

class EmergencyLeaveSystemEvent(Sim.Event):

    def __init__(self,
                 entity: object,
                 time: float,
                 emergency: "Models.Emergency",
                 satisfied: bool,
                 vehicle: Optional["Models.Vehicle"] = None,
                 chain_assignment: bool = False,
                 name: str = None):
        super().__init__(time, name)

        self.entity = entity
        self.emergency: "Models.Emergency" = emergency
        self.vehicle = vehicle  # This is to safely remove the reference to the patient from the vehicle                    
        self.chain_assignment = chain_assignment
        self.message: str = '{} leaving the system {}'.format(emergency.name, 'satisfied' if satisfied else 'unsatisfied')  

    
    def execute(self, simulator: "Models.EMSModel"):
        # Add the total time to the record
        severity = 0 if self.emergency.severity == 1 else 1
        simulator.time_records[(simulator.timePeriod(), severity, self.emergency.node)].append(simulator.now() - self.emergency.vehicle_assigned_time)

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

        
        if self.chain_assignment:
            return AmbulanceAssignmentEvent(simulator, simulator.now())

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
            simulator.insert(AmbulanceEndCleaningEvent(simulator, simulator.now() + 10*60, self.vehicle))

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
            simulator.insert(AmbulanceFinishAttendingEvent(self.entity, simulator.now() +
                self.emergency.attending_time, self.vehicle, self.emergency))
        else:
            simulator.insert(AmbulanceFinishAttendingEvent(self.entity, simulator.now() +
                60, self.vehicle, self.emergency))

        
        # Statistics
        if self.emergency.severity == 1:
            simulator.statistics['HSAttentionTime'].record(simulator.now(), self.emergency.attending_time)
        else:
            simulator.statistics['LSAttentionTime'].record(simulator.now(), self.emergency.attending_time)


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
        if self.valid:
            if status == 0:
                self.vehicle.statistics['State'].record(simulator.now(), 0)

                return AmbulanceAssignmentEvent(simulator, simulator.now())
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
                        simulator.statistics['PercentageALSlt10min'].recordAverage(simulator.now(), 1 if response_time <= 10*60 else 0)
                        simulator.statistics['PercentageALSlt8min'].recordAverage(simulator.now(), 1 if response_time <= 8*60 else 0)
                        simulator.statistics['PercentageALSlt7min'].recordAverage(simulator.now(), 1 if response_time <= 7*60 else 0)

                        simulator.statistics['SpatialHS10minCover'].record(simulator.now(), self.vehicle.patient.node, 1 if response_time <= 10*60 else 0)
                        simulator.statistics['SpatialHS8minCover'].record(simulator.now(), self.vehicle.patient.node, 1 if response_time <= 8*60 else 0)
                        simulator.statistics['SpatialHS7minCover'].record(simulator.now(), self.vehicle.patient.node, 1 if response_time <= 7*60 else 0)

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

                    return AmbulanceStartAttendingEvent(self.vehicle, simulator.now(), self.vehicle,
                        self.vehicle.patient)
            elif status == 2:
                # Mark ambulance as cleaning and schedule finish cleaning event
                self.vehicle.cleaning = True
                simulator.insert(AmbulanceEndCleaningEvent(simulator, simulator.now() + 10*60, self.vehicle))

                if self.vehicle.patient is not None:
                    # Statistics
                    if self.vehicle.patient.to_hospital_time > 0:
                        simulator.statistics['ToHospitalTime'].record(simulator.now(), simulator.now() - self.vehicle.patient.to_hospital_time)

                    return EmergencyLeaveSystemEvent(self.entity, simulator.now(),
                            self.vehicle.patient, True, vehicle=self.vehicle, chain_assignment=True)

            simulator.registerVehicleStationChange(self.vehicle, self.vehicle.pos)


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
            simulator.insert(AmbulanceStartMovingEvent(self.vehicle, simulator.now(),
                self.vehicle, simulator.city_graph.es[self.vehicle.actual_edge]))


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
                simulator.insert(AmbulanceStartMovingEvent(self.vehicle, simulator.now(), self.vehicle,
                                simulator.city_graph.es[self.vehicle.actual_edge]))
            else:
                # Compute shortest path for vehicle
                path = simulator.getShortestPath(self.vehicle.to_node, self.node)
                
                # onAssigned callback
                self.vehicle.onAssignedMovement(path[0], [simulator.city_graph.es[p]['v'] for p in path[0]])
                
                # Clear the scheduled vehicle movement
                simulator.clearVehicleMovement(self.vehicle)
                
                # Schedule the start of the movement for the vehicle in the new route
                simulator.insert(AmbulanceStartMovingEvent(self.vehicle, self.vehicle.expected_arrival, self.vehicle,
                                simulator.city_graph.es[self.vehicle.actual_edge]))
            
            # Save a record of the trip
            self.vehicle.record.append((simulator.now(), self.vehicle.pos, self.node, self.vehicle.patient,
                                        self.vehicle.patient.hospital if self.vehicle.patient is not None else None))

            # Recover the position logic
            simulator.registerVehicleStationChange(self.vehicle, self.node)
        
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
        self.message: str = '{} assigned to {} at node {}'.format(vehicle.name, 
            self.emergency.name, self.emergency.node)

    def execute(self, simulator: "Models.EMSModel"):
        # Compute shortest path for vehicle
        path: List[List[int]] = simulator.getShortestPath(self.vehicle.pos,self.emergency.node)

        # onAssigned callback
        self.vehicle.onAssignedToEmergency(self.emergency, path[0],
            [simulator.city_graph.es[p]['v'] for p in path[0]])
        
        # Schedule the start of the movement for the vehicle
        self.vehicle.onArrivalToNode(self.vehicle.pos)

        self.emergency.vehicle_assigned_time = simulator.now()

        if len(path[0]) != 0:
            simulator.insert(AmbulanceStartMovingEvent(self.vehicle, simulator.now(), self.vehicle,
                            simulator.city_graph.es[self.vehicle.actual_edge]))
        else:
            simulator.insert(AmbulanceArriveToNodeEvent(self.vehicle, simulator.now(), 
                                                        self.vehicle, self.emergency.node))

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
        self.travel_time = self.edge['length']/simulator.parameters.getSpeedList(simulator.timePeriod())[self.edge.index]
        self.vehicle.onMovingToNextNode(simulator.now() + self.travel_time)

        if self.vehicle.patient is None:
            self.vehicle.reposition_workload += self.travel_time
            self.vehicle.statistics['RelocationTime'].record(simulator.now(), self.vehicle.reposition_workload)
            self.vehicle.statistics['State'].record(simulator.now(), 1)

        self.vehicle.statistics['MetersDriven'].record(simulator.now(), self.vehicle.statistics['MetersDriven'].data[-1][1] + self.edge['length'])

        # Schedule vehicle arrival to node
        simulator.insert(AmbulanceArriveToNodeEvent(self.vehicle, simulator.now() + self.travel_time,
                                               self.vehicle, self.edge['v']))


class AmbulanceAssignmentEvent(Sim.Event):
    
    def __init__(self, 
                 entity: object,
                 time: float,
                 name: str = None):
        super().__init__(time, name)

        self.entity: object = entity

        self.message: str = "Ambulance Assignment"
    
    def execute(self, simulator: "Models.EMSModel"):
        simulator.statistics['AvailableALSVehicles'].record(simulator.now(), len(simulator.getAvaliableVehicles(v_type=0)))
        simulator.statistics['AvailableBLSVehicles'].record(simulator.now(), len(simulator.getAvaliableVehicles(v_type=1)))

        assignment = simulator.assigner.assign(simulator)

        for v in assignment.keys():
            # Schedule the assignment event
            simulator.insert(AssignedEvent(simulator, simulator.now(), 
                                            v, assignment[v]))
            
            # Mark emergency as assigned
            assignment[v].markStatus(1)
            simulator.assignedEmergencies.append(assignment[v])


class EmergencyArrivalEvent(Sim.Event):

    def __init__(self,
                 entity: object,
                 time: float,
                 node: str,
                 severity: int,
                 disposition_code: int,
                 name:str = None):
        super().__init__(time, name)

        self.entity: object = entity
        self.node: str = node
        self.severity: int = severity
        self.disposition_code: int = disposition_code
        self.emergency = Models.Emergency(self.time, self.node, self.severity, self.disposition_code)

        self.message: str = "Emergency arrived at node {}".format(node)
    
    def execute(self, simulator: "Models.EMSModel") -> AmbulanceAssignmentEvent:
        # Create the emergency and append it to the reference list
        simulator.activeEmergencies.append(self.emergency)

        # Schedule the next emergency, if there is one
        try:
            simulator.insert(next(simulator.arrival_generator))
        except StopIteration:
            pass
        
        simulator.assignedNotArrived += 1
        # Statistics
        simulator.statistics['EmergenciesWaiting'].record(simulator.now(), simulator.assignedNotArrived)

        # Chain an assignment of the ambulances right next to the arrival
        return AmbulanceAssignmentEvent(self.emergency, self.time)


class AmbulanceLeavingEvent(Sim.Event):
    """
    Intended to use in the ambulance 'lifecicle'
    """

    def __init__(self,
                 entity: object,
                 time: float,
                 vehicle: "Models.Vehicle",
                 name:str = None):

        super().__init__(time, name)

        self.entity: object = entity
        self.vehicle: Models.Vehicle = vehicle

        self.message: str = "{} leaving the system".format(vehicle.name)

    def execute(self, simulator: "Models.EMSModel"):
        self.vehicle.statistics['TimeInSystem'].record(simulator.now() - self.vehicle.arrival_time)

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
                 name:str = None):

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
                 name:str = None):

        super().__init__(time, name)

        self.entity: object = entity
        self.node: str = node
        self.prior_worked_time: float = prior_worked_time
        self.vehicle: Models.Vehicle = vehicle

        self.message: str = "Ambulance arrived to the system at node {}!".format(node)

    def execute(self, simulator: "Models.EMSModel"):
        self.vehicle.arrival_time = simulator.now()

        simulator.vehicles.append(self.vehicle)
        simulator.registerVehicleStationChange(self.vehicle, self.node)

        simulator.insert(MarkAmbulanceLeavingEvent(simulator, simulator.now() + simulator.parameters.vehicle_shift(self.vehicle) - self.prior_worked_time, self.vehicle))
        
        simulator.statistics[('ALS' if self.vehicle.type == 0 else 'BLS') + 'VehiclesInSystem'].record(simulator.now(), len([v for v in simulator.vehicles if v.type == self.vehicle.type]))


class HospitalSettingEvent(Sim.Event):
    """
    Intended to use in the ambulance 'lifecicle'
    """

    def __init__(self,
                 entity: object,
                 time: float,
                 name:str = None):

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