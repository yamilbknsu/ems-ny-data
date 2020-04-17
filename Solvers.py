import numpy as np
from typing import Dict, List

# Internal imports
import Models


class AssignmentModel:

    def __init__(self):
        pass

    def assign(self,
               simulator: "Models.EMSModel") -> Dict["Models.Vehicle", "Models.Emergency"]:      # noqa: E501
        print('Warning! Assignment not implemented')
        return {}


class NearestAssigner(AssignmentModel):

    def __init__(self):
        super().__init__()

    def assign(self,
               simulator: "Models.EMSModel") -> Dict["Models.Vehicle", "Models.Emergency"]:      # noqa: E501
        # Getting a reference for unassigned emergencies
        emergencies: List[Models.Emergency] = \
            list(set(simulator.activeEmergencies) - set(simulator.assignedEmergencies))          # noqa: E501
        # Sort by remaining time till expiring
        emergencies.sort(key=lambda e: simulator.now() - e.max_time)
        emergencies_vertices: List[str] = [e.node for e in emergencies]

        # Final list of vertices to apply dijkstra to
        vehicles: List[Models.Vehicle] = simulator.getAvaliableVehicles()
        vehicle_vertices: List[str] = [v.pos for v in vehicles]
        n = len(vehicle_vertices)  # Number of available vehicles

        used_vehicles: List[int] = []
        assignment_dict: Dict["Models.Vehicle", "Models.Emergency"] = {}
        if n > 0:
            # Dijkstra's algorithm
            distances: np.array = simulator.getShortestDistances(vehicle_vertices, emergencies_vertices)    # noqa: E501

            # Final result
            for e, emergency in enumerate(emergencies):
                candidates = list(enumerate(distances[:,e]))
                candidates.sort(key=lambda c: c[1])
                for c in candidates:
                    if c not in used_vehicles:
                        used_vehicles.append(c[0])
                        assignment_dict[vehicles[c[0]]] = emergency
                        break

        return assignment_dict
