import time
import numpy as np
import gurobipy as grb
from typing import Dict, List, Tuple
from ortools.linear_solver import pywraplp

# Internal imports
import Models

"""
Dispatcher models
"""

class DispatcherModel:

    def __init__(self):
        pass

    def assign(self,
               simulator: "Models.EMSModel") -> Dict["Models.Vehicle", "Models.Emergency"]:      
        print('Warning! Assignment not implemented')
        return {}


class PreparednessDispatcher(DispatcherModel):

    def __init__(self):
        super().__init__()

    def assign(self,
               simulator: "Models.EMSModel") -> Dict["Models.Vehicle", "Models.Emergency"]:      

        # Getting a reference for unassigned emergencies
        emergencies: List[Models.Emergency] = \
            list(set(simulator.activeEmergencies) - set(simulator.assignedEmergencies))          

        # Sort by remaining time till expiring
        emergencies.sort(key=lambda e: simulator.now() - e.max_time)
        emergencies_by_level = [[e for e in emergencies if e.severity == 1],
                                [e for e in emergencies if e.severity > 1]]

        emergencies_vertices: List[List[str]] = [[e.node for e in level] for level in emergencies_by_level]

        vehicles: List[List[Models.Vehicle]] = [[ambulance
                                           for ambulance in simulator.getAvaliableVehicles(v_type=v)]
                                          for v in range(simulator.parameters.vehicle_types)]

        vehicle_positions = [[ambulance.to_node
                             for ambulance in simulator.getAvaliableVehicles(v_type=v)]
                             for v in range(simulator.parameters.vehicle_types)]

        used_vehicles: List[Tuple[int, int]] = []
        assignment_dict: Dict["Models.Vehicle", "Models.Emergency"] = {}

        weights = np.array(simulator.city_graph.es['length']) / simulator.parameters.getSpeedList(simulator.timePeriod())
        travel_times = np.array(simulator.city_graph.shortest_paths(vehicle_positions[0], emergencies_vertices[0], weights))
        valid_indexes = np.where(travel_times < 8*60)

        # First priority emergencies
        for e, emergency in enumerate(emergencies_by_level[0]):
            valid_vehicles_index = valid_indexes[0][np.where(valid_indexes[1] == e)]
            candidates = list(zip(valid_indexes[0], travel_times[valid_vehicles_index].squeeze().reshape(-1)))
            candidates.sort(key=lambda c: c[1])

            for c in candidates:
                if (0, c[0]) not in used_vehicles:
                    used_vehicles.append((0, c[0]))
                    assignment_dict[vehicles[0][c[0]]] = emergency
                    break

        # Travel times to get the valid ambulances to dispatch
        travel_times = [np.array(simulator.city_graph.shortest_paths(vehicle_positions[0], emergencies_vertices[1], weights)),
                        np.array(simulator.city_graph.shortest_paths(vehicle_positions[1], emergencies_vertices[1], weights))]

        valid_indexes = [np.where(travel_times[0] < 8*60), np.where(travel_times[1] < 8*60)]

        # Travel times to demand nodes to compute preparedness
        travel_to_demand = [simulator.city_graph.shortest_paths(vehicle_positions[0], simulator.parameters.demand_nodes, weights),
                            simulator.city_graph.shortest_paths(vehicle_positions[1], simulator.parameters.demand_nodes, weights)]

        # Lower priority emergencies
        for e, emergency in enumerate(emergencies_by_level[1]):
            valid_vehicles_index = [valid_indexes[0][0][np.where(valid_indexes[0][1] == e)],
                                    valid_indexes[1][0][np.where(valid_indexes[1][1] == e)]]
            
            candidates_preparedness: List[Tuple[int, int, float]] = []

            # Compute the preparedness difference for each BLS
            for i in valid_vehicles_index[1]:
                preparedness = \
                    simulator.computeSystemPreparedness(travel_matrix=[travel_to_demand[0], travel_to_demand[1][:i]+travel_to_demand[1][i+1:]])
                candidates_preparedness.append((1, i, preparedness))

            for i in valid_vehicles_index[0]:
                preparedness = \
                    simulator.computeSystemPreparedness(travel_matrix=[travel_to_demand[0][:i]+travel_to_demand[0][i+1:],travel_to_demand[1]])
                candidates_preparedness.append((0, i, preparedness))

            candidates_preparedness.sort(key=lambda c: c[2], reverse=True)

            for c in candidates_preparedness:
                if (c[0], c[1]) not in used_vehicles:
                    used_vehicles.append((c[0], c[1]))
                    assignment_dict[vehicles[c[0]][c[1]]] = emergency
                    break

        return assignment_dict


class NearestDispatcher(DispatcherModel):

    def __init__(self):
        super().__init__()

    def assign(self,
               simulator: "Models.EMSModel") -> Dict["Models.Vehicle", "Models.Emergency"]:
        # Getting a reference for unassigned emergencies
        emergencies: List[Models.Emergency] = \
            list(set(simulator.activeEmergencies) - set(simulator.assignedEmergencies))          
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
            # Dijkstra's algorithm - At the end it will be a Unique emergency nodes x vehicles matrix
            distances: np.array = simulator.getShortestDistances(vehicle_vertices, list(set(emergencies_vertices))).T    
            distances_dict = {node: distances[n,:] for n, node in enumerate(list(set(emergencies_vertices)))}

            # Final result
            for e, emergency in enumerate(emergencies):
                candidates = list(enumerate(distances_dict[emergency.node]))
                candidates.sort(key=lambda c: c[1])
                for c in candidates:
                    if c not in used_vehicles:
                        used_vehicles.append(c[0])
                        assignment_dict[vehicles[c[0]]] = emergency
                        break

        return assignment_dict


"""
Relocation Models
"""


class RelocationModel:

    def __init__(self):
        pass

    def relocate(self,
                 simulator: "Models.EMSModel",
                 params: "Models.SimulationParameters") -> Dict["Models.Vehicle", str]:
        """

        """
        print('Warning! Relocation not implemented')
        return {}


class MaxExpectedSurvivalRelocator(RelocationModel):

    def __init__(self):
        super().__init__()

    def SurvivalFunction(self, response_times):
        return np.ones(shape = response_times.shape)
        #return (1+np.exp(0.679+.262*response_times))**-1

    def relocate(self,
                 simulator: "Models.EMSModel",
                 params: "Models.SimulationParameters") -> Dict["Models.Vehicle", str]:

        # First stage
        start_time = time.time()
        print('0 - Starting first stage')

        # Parameters
        neighbor_k = params.neighbor_k
        neighborhood = params.neighborhood
        D_rates = params.demand_rates
        Busy_rates = params.mean_busytime
        C = params.candidate_nodes
        D = params.demand_nodes
        t = simulator.timePeriod()

        # Create the mip solver with the CBC backend.
        model = grb.Model(name="MIP Model")

        # Declare model variables
        print(time.time() - start_time, '- Declaring Variables...')

        y = [model.addVar(vtype=grb.GRB.BINARY, name='y_' + node)
             for j, node in enumerate(C)]

        x = [[model.addVar(vtype=grb.GRB.BINARY, name='x_' + str(k) + '_' + node)
                        for k in range(neighbor_k[0][node]+1)]
                        for node in C]

        # Coefficients
        # ------------
        # Q parameter
        print(time.time() - start_time, '- Computing q...')
        demand_load = np.array([[np.sum(D_rates[0].loc[t+1, neighborhood[t][j]] * 
                                        Busy_rates[0].loc[t+1, neighborhood[t][j]]) for j in C]]).T
        demand_load = demand_load @ np.array([[1/k if k > 0 else 0 for k in range(max(neighbor_k[t].values())+1)]])
        demand_load = 1-np.power(demand_load, [k if k > 0 else 0 for k in range(max(neighbor_k[t].values())+1)])
        demand_load[demand_load < 0] = 0

        # [j, i] matrix
        print(time.time() - start_time, '- Computing O.F. coefficients...')
        survival_matrix = self.SurvivalFunction(params.cand_demand_time[t])

        # Filter the survival matrix leaving only the nodes that are reachable,
        # 0 to all the rest
        filtered_survival = np.array([[survival_matrix[j, i] if c_node in params.reachable_inverse[t][d_node] else 0        # noqa E501
                                       for j, c_node in enumerate(C)]
                                       for i, d_node in enumerate(D)]).T

        # [j, i] matrix respresenting S_{i,j} * d_i
        coefficients = params.demand_rates[0].loc[t+1, params.demand_nodes].values * filtered_survival                      # noqa E501
        
        # Objective function
        print(time.time() - start_time, '- Setting O.F...')
        availability = [grb.LinExpr([demand_load[j][k] for k in range(neighbor_k[0][node])],
                        [x[j][k] for k in range(neighbor_k[0][node])]) for j, node in enumerate(C)]       # noqa E501


        
        model.setObjective(1000*grb.quicksum(grb.quicksum(coefficients[0,C.index(c_node)]*availability[C.index(c_node)] 
                           for c_node in params.reachable_inverse[0][d_node])
                           for d_node in D))
        model.ModelSense = grb.GRB.MAXIMIZE

        print(time.time() - start_time, '- Constraints...')
        # Constraints
        # Capacity constraint
        Cap_constraint = model.addConstr(lhs=grb.quicksum(y),
                                        sense=grb.GRB.LESS_EQUAL,
                                        rhs=303,
                                        name='CapacityConstraint')
        # Constraint 1
        Const_1 = {j: model.addConstr(lhs=grb.quicksum(x[j]),
                                      sense=grb.GRB.LESS_EQUAL,
                                      #rhs=y[j],
                                      rhs=1,
                                      name='Const_1_{}'.format(j)) for j, c_node in enumerate(C)}                            # noqa E501
        # Constraint 2
        Const_2 = {j: model.addConstr(lhs=grb.quicksum(y[C.index(u)] for u in params.neighborhood_candidates[t][c_node]),    # noqa E501
                                      sense=grb.GRB.EQUAL,
                                      rhs=grb.LinExpr(list(range(len(x[j]))), x[j]),
                                      name='Const_2_{}'.format(j)) for j, c_node in enumerate(C)}                            # noqa E501

        print(time.time() - start_time, '- Solving the model...')
        status = model.optimize()
        print(time.time() - start_time, '- Done!')
        print()
        return {}