import igraph
import numpy as np
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
            optimal_positions, reposition_dict = simulator.repositioner.relocate(simulator, simulator.parameters, borough = b)
            simulator.ambulance_stations[b] = optimal_positions

            for v in reposition_dict:
                simulator.insert(TripAssignedEvent(simulator, simulator.now(), v, reposition_dict[v]))


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

        for b in range(1,6):
            optimal_positions, reposition_dict = simulator.repositioner.relocate(simulator, simulator.parameters, borough = b)
            simulator.ambulance_stations[b] = optimal_positions
            
            for v in reposition_dict:
                v.teleportToNode(reposition_dict[v])
                simulator.registerVehicleStationChange(v, reposition_dict[v])

                self.new_positions.append([v.name, reposition_dict[v]])
        
        simulator.insert(RelocationEvent(simulator, simulator.now() + 600))

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
        self.vehicle = vehicle  # This is to safely remove the reference to the patient form the vehicle                    
        self.chain_assignment = chain_assignment
        self.message: str = '{} leaving the system {}'.format(emergency.name, 'satisfied' if satisfied else 'unsatisfied')  

    
    def execute(self, simulator: "Models.EMSModel"):
        simulator.activeEmergencies.remove(self.emergency)
        if self.emergency in simulator.assignedEmergencies:
            simulator.assignedEmergencies.remove(self.emergency)
        
        if self.vehicle is not None:
            self.vehicle.patient = None
        
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
            return EmergencyLeaveSystemEvent(simulator, simulator.now(), self.emergency, True)
        else:
            # Get nearest hospital of the corresponding type
            distances = simulator.getShortestDistances(self.emergency.node, simulator.hospitals[hospital_type])[0]
            nearest_node = simulator.hospitals[hospital_type][np.argmax(distances)]

            self.emergency.markStatus(3)
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
        self.message: str = '{} started attending {}'.format(vehicle.name, emergency.name)  # noqa: E501

    def execute(self, simulator: "Models.EMSModel"):
        # Schedule the end of the atention
        simulator.insert(AmbulanceFinishAttendingEvent(self.entity, simulator.now() +
            self.emergency.attending_time, self.vehicle, self.emergency))


class AmbulanceEndTripEvent(Sim.Event):

    def __init__(self,
                 entity: object,
                 time: float,
                 vehicle: "Models.Vehicle",
                 name: str = None):
        super().__init__(time, name)

        self.entity = entity
        self.vehicle: "Models.Vehicle" = vehicle
        self.message: str = '{} finished trip to node {}'.format(vehicle.name, vehicle.pos)
    
    def execute(self, simulator: "Models.EMSModel"):
        status = self.vehicle.onPathEnded()

        if status == 0:
            return AmbulanceAssignmentEvent(simulator, simulator.now())
        elif status == 1:
            if self.vehicle.patient is not None:
                return AmbulanceStartAttendingEvent(self.vehicle, simulator.now(), self.vehicle,
                    self.vehicle.patient)
        elif status == 2:
            if self.vehicle.patient is not None:
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
            if not self.vehicle.moving:
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
            
            # Recover the position logic
            simulator.registerVehicleStationChange(self.vehicle, self.node)


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

        if len(path[0]) != 0:
            simulator.insert(AmbulanceStartMovingEvent(self.vehicle, simulator.now(), self.vehicle,
                            simulator.city_graph.es[self.vehicle.actual_edge]))
        else:
            simulator.insert(AmbulanceArriveToNodeEvent(self.vehicle, simulator.now(), 
                                                        self.vehicle, self.emergency.node))


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
        self.travel_time = self.edge['length']/self.vehicle.speed
        self.vehicle.onMovingToNextNode(simulator.now() + self.travel_time)

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
                 name:str = None):
        super().__init__(time, name)

        self.entity: object = entity
        self.node: str = node
        self.severity: int = severity
        self.emergency = Models.Emergency(self.time, self.node, self.severity)

        self.message: str = "Emergency arrived at node {}".format(node)
    
    def execute(self, simulator: "Models.EMSModel") -> AmbulanceAssignmentEvent:
        # Create the emergency and append it to the reference list
        simulator.activeEmergencies.append(self.emergency)

        # Schedule the next emergency, if there is one
        try:
            simulator.insert(next(simulator.arrival_generator))
        except StopIteration:
            pass

        # Chain an assignment of the ambulances right next to the arrival
        return AmbulanceAssignmentEvent(self.emergency, self.time)


class AmbulanceArrivalEvent(Sim.Event):
    """
    Intended to use in the ambulance 'lifecicle'
    """

    def __init__(self,
                 entity: object,
                 time: float,
                 node: str,
                 vehicle: "Models.Vehicle",
                 name:str = None):

        super().__init__(time, name)

        self.entity: object = entity
        self.node: str = node
        self.vehicle: Models.Vehicle = vehicle

        self.message: str = "Ambulance arrived to the system at node {}!".format(node)

    def execute(self, simulator: "Models.EMSModel"):
        simulator.vehicles.append(self.vehicle)
        simulator.registerVehicleStationChange(self.vehicle, self.node)


class EndSimulationEvent(Sim.Event):

    def __init__(self,
                 entity: "Models.EMSModel",
                 time: float):

        super().__init__(time, 'EndSimulation')
        self.entity = entity

        self.message: str = "End of simulation"
    
    def execute(self, simulator: "Models.EMSModel"):
        simulator.events.empty()