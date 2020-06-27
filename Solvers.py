import time
import numpy as np
import gurobipy as grb
from typing import Dict, List, Tuple, Optional
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
                 params: "Models.SimulationParameters",
                 **kwargs) -> Optional[Tuple[List[List[str]], Dict["Models.Vehicle", str]]]:
        """

        """
        print('Warning! Relocation not implemented')
        return None


class MaxExpectedSurvivalRelocator(RelocationModel):

    def __init__(self):
        super().__init__()

    def SurvivalFunction(self, response_times):
        #return np.ones(shape = response_times.shape)
        return (1+np.exp(0.679+.262*response_times))**-1

    def relocate(self,
                 simulator: "Models.EMSModel",
                 params: "Models.SimulationParameters") -> Tuple[List[List[str]], Dict["Models.Vehicle", str]]:

        # Initialize return list
        # This list will hold the final optimal positions of the ambulances
        final_positions: List = []

        # Initialize the return dict
        # This will hold the vehicle repositioning values
        final_repositioning: Dict["Models.Vehicle", str] = {}

        # Parameters
        actual_ALS_vehicles = {v.to_node: v for v in simulator.getAvaliableVehicles(0)}
        actual_BLS_vehicles = {v.to_node: v for v in simulator.getAvaliableVehicles(1)}
        neighbor_k = params.neighbor_k
        neighborhood = params.neighborhood
        borough_cand = params.candidates_borough
        borough_demand = params.demand_borough
        ambulance_distribution = params.ambulance_distribution
        D_rates = params.demand_rates
        Busy_rates = params.mean_busytime
        Q = params.Q
        P = params.P
        C = params.candidate_nodes
        D = params.demand_nodes
        t = simulator.timePeriod()

        overload_penalty = params.overload_penalty
        maximum_uber = params.maximum_uber_per_period
        cand_cand_time = params.cand_cand_time

        ########################
        #    First ALS part    #
        ########################

        # First stage
        start_time = time.time()
        print('0 - Starting first stage')
        # Create the mip solver with the CBC backend.
        model = grb.Model(name="First stage optimal positions")
        # Declare model variables
        print(time.time() - start_time, '- Declaring Variables...')

        y = [model.addVar(vtype=grb.GRB.BINARY, name='y_' + node)
             for j, node in enumerate(C)]

        x = [[model.addVar(vtype=grb.GRB.BINARY, name='x_' + str(k) + '_' + node)
                        for k in range(neighbor_k[t][node]+1)]
                        for node in C]
        # Coefficients
        # ------------
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
        availability = [grb.LinExpr([Q[j][k] for k in range(neighbor_k[t][node])],
                        [x[j][k] for k in range(neighbor_k[t][node])]) for j, node in enumerate(C)]       # noqa E501


        
        model.setObjective(grb.quicksum(grb.quicksum(coefficients[C.index(c_node),i]*availability[C.index(c_node)] 
                           for c_node in params.reachable_inverse[t][d_node])
                           for i, d_node in enumerate(D)))
        model.ModelSense = grb.GRB.MAXIMIZE

        print(time.time() - start_time, '- Constraints...')
        # Constraints
        # Capacity constraint
        Cap_constraint = {b: model.addConstr(lhs=grb.quicksum(y[C.index(c)] for c in borough_cand[b]),
                                             sense=grb.GRB.EQUAL,
                                             rhs=ambulance_distribution[0][b],
                                             name='CapacityConstraint_{}'.format(b)) for b in range(6)}
        # Constraint 1
        Const_1 = {j: model.addConstr(lhs=grb.quicksum(x[j]),
                                      sense=grb.GRB.LESS_EQUAL,
                                      rhs=1,
                                      name='Const_1_{}'.format(j)) for j, c_node in enumerate(C)}                            # noqa E501
        # Constraint 2
        Const_2 = {j: model.addConstr(lhs=grb.quicksum(y[C.index(u)] for u in params.neighborhood_candidates[t][c_node]),    # noqa E501
                                      sense=grb.GRB.EQUAL,
                                      rhs=grb.LinExpr(list(range(len(x[j]))), x[j]),
                                      name='Const_2_{}'.format(j)) for j, c_node in enumerate(C)}                            # noqa E501

        print(time.time() - start_time, '- Model Params...')
        model.setParam('MIPGap', .1)
        model.setParam('TimeLimit', 45)
        
        print(time.time() - start_time, '- Solving the model...')
        status = model.optimize()

        print(time.time() - start_time, '- Done!')

        # -------------------------
        ########################
        #   Second ALS part    #
        ########################

        print(time.time() - start_time, '- Starting first stage part 2')

        actual_positions = [actual_ALS_vehicles[v].to_node for v in actual_ALS_vehicles]
        target_positions = []
        for j in range(len(C)):
            if y[j].x == 1:
                target_positions.append(C[j])
        
        # Create the mip solver with the CBC backend.
        model = grb.Model(name="First stage optimal relocations")

        print(time.time() - start_time, '- Declaring Variables...')
        x = [[model.addVar(vtype=grb.GRB.BINARY, name='x_' + i + '_' + j)
              for j in target_positions]
              for i in actual_positions]
        
        y = [model.addVar(vtype=grb.GRB.CONTINUOUS, lb=0, name='y_' + i)
             for i in actual_ALS_vehicles]

        print(time.time() - start_time, '- Computing coefficents...')
        weights = np.array(simulator.city_graph.es['length']) / simulator.parameters.getSpeedList(simulator.timePeriod())
        travel_times = np.array(simulator.city_graph.shortest_paths(actual_positions, target_positions, weights))

        print(time.time() - start_time, '- Setting O.F...')
        model.setObjective(grb.quicksum(grb.LinExpr(travel_times[i], x[i])
                            for i in range(len(actual_positions)))
                            + overload_penalty*grb.quicksum(y))
        model.ModelSense = grb.GRB.MINIMIZE

        print(time.time() - start_time, '- Constraints...')
        # Constraint 1
        Const_1 = {i: model.addConstr(lhs=grb.quicksum(x[i]),
                                      sense=grb.GRB.LESS_EQUAL,
                                      rhs=1,
                                      name='Const_1_{}'.format(i)) for i in range(len(actual_positions))}
        # Constraint 2
        Const_2 = {j: model.addConstr(lhs=grb.quicksum(x[i][j] for i in range(len(actual_positions))),
                                      sense=grb.GRB.GREATER_EQUAL,
                                      rhs=1,
                                      name='Const_1_{}'.format(j)) for j in range(len(target_positions))}
        # Constraint 3
        # TODO: Check rhs
        Const_3 = {i: model.addConstr(lhs=actual_ALS_vehicles[node].reposition_workload + \
                                          grb.LinExpr(travel_times[i], x[i]),
                                      sense=grb.GRB.LESS_EQUAL,
                                      rhs=y[i],
                                      name='Const_1_{}'.format(i)) for i, node in enumerate(actual_positions)}

        print(time.time() - start_time, '- Model Params...')
        model.setParam('MIPGap', .1)
        
        print(time.time() - start_time, '- Solving the model...')
        status = model.optimize()
        print(time.time() - start_time, '- Done!')
        
        # Update positions for first stage
        final_positions.append(target_positions)

        reposition_matrix = [[x[i][j].x for j in range(len(target_positions))] for i in range(len(actual_positions))]
        for i, node in enumerate(actual_positions):
            final_repositioning[actual_ALS_vehicles[node]] = target_positions[reposition_matrix[i].index(1)]


        # -------------------------------------------------------------------------------------------------------------------

        ########################
        #    First BLS part    #
        ########################

        # Start the second stage
        print(time.time() - start_time, '- Starting second stage')

        # Create the mip solver with the CBC backend.
        model = grb.Model(name="Second stage optimal positions")

        # Declare model variables
        print(time.time() - start_time, '- Declaring Variables...')

        y = [model.addVar(vtype=grb.GRB.BINARY, name='y_' + node)
             for j, node in enumerate(C)]

        z = [model.addVar(vtype=grb.GRB.INTEGER, name='z_' + node)
             for j, node in enumerate(C)]

        x = [[model.addVar(vtype=grb.GRB.BINARY, name='x_' + str(k) + '_' + node)
                        for k in range(neighbor_k[t][node]+1)]
                        for node in C]

        # Coefficients
        # ------------
        print(time.time() - start_time, '- Computing borough demand...')
        borough_demand = np.array([np.sum(P[D.index(d), 1] for d in borough_demand[b]) for b in range(1,6)])
        borough_demand = borough_demand/np.sum(borough_demand)

        # Allocate the ALS ambulances to each borough
        n_total = len(actual_BLS_vehicles)
        n_allocated = 0
        allocated_ambulances = [0]
        for b in range(5-1):
            n_allocated += int(n_total * borough_demand[b])
            allocated_ambulances.append(int(n_total * borough_demand[b]))
        allocated_ambulances.append(int(n_total - n_allocated))

        # [j, i] matrix
        print(time.time() - start_time, '- Computing O.F. coefficients...')

        # Filter the survival matrix leaving only the nodes that are reachable,
        # 0 to all the rest
        filtered_survival = np.array([[1 if c_node in params.reachable_inverse[t][d_node] else 0        # noqa E501
                                       for j, c_node in enumerate(C)]
                                       for i, d_node in enumerate(D)]).T

        # [j, i] matrix respresenting S_{i,j} * d_i
        coefficients = params.demand_rates[1].loc[t+1, params.demand_nodes].values * filtered_survival                      # noqa E501
        
        # Objective function
        print(time.time() - start_time, '- Setting O.F...')
        availability = [grb.LinExpr([P[j][k] for k in range(neighbor_k[t][node])],
                        [x[j][k] for k in range(neighbor_k[t][node])]) for j, node in enumerate(C)]       # noqa E501
        
        model.setObjective(grb.quicksum(grb.quicksum(coefficients[C.index(c_node),i]*availability[C.index(c_node)] 
                           for c_node in params.reachable_inverse[t][d_node])
                           for i, d_node in enumerate(D)))
        model.ModelSense = grb.GRB.MAXIMIZE

        print(time.time() - start_time, '- Constraints...')
        # Constraints
        # Capacity constraint
        Cap_constraint = {b: model.addConstr(lhs=grb.quicksum(y[C.index(c)] for c in borough_cand[b]),
                                             sense=grb.GRB.EQUAL,
                                             rhs=allocated_ambulances[b],
                                             name='CapacityConstraint_{}'.format(b)) for b in range(6)}
        # Constraint 1
        Const_1 = {j: model.addConstr(lhs=grb.quicksum(x[j]),
                                      sense=grb.GRB.LESS_EQUAL,
                                      rhs=1,
                                      name='Const_1_{}'.format(j)) for j, c_node in enumerate(C)}                            # noqa E501
        # Constraint 2
        Const_2 = {j: model.addConstr(lhs=grb.quicksum(y[C.index(u)] + z[C.index(u)] for u in params.neighborhood_candidates[t][c_node]),
                                      sense=grb.GRB.EQUAL,
                                      rhs=grb.LinExpr(list(range(len(x[j]))), x[j]),
                                      name='Const_2_{}'.format(j)) for j, c_node in enumerate(C)}                            # noqa E501
        
        # Constraint 3
        mean_uber_load = [Busy_rates[1].loc[t+1,neighborhood[t][j]].mean() for j in C]
        Const_3 = model.addConstr(lhs=grb.LinExpr(mean_uber_load, z),
                                  sense=grb.GRB.LESS_EQUAL,
                                  rhs=maximum_uber - simulator.uber_hours,
                                  name='UberConstraint')

        print(time.time() - start_time, '- Model Params...')
        model.setParam('MIPGap', .1)
        
        print(time.time() - start_time, '- Solving the model...')
        status = model.optimize()
        print(time.time() - start_time, '- Done!')

        print()
        return final_positions, final_repositioning

class MaxExpectedSurvivalRelocatorOnlyBorough(RelocationModel):

    def __init__(self):
        super().__init__()

    def SurvivalFunction(self, response_times):
        #return np.ones(shape = response_times.shape)
        return (1+np.exp(0.679+.262*response_times))**-1

    def relocate(self,
                 simulator: "Models.EMSModel",
                 params: "Models.SimulationParameters",
                 borough = None) -> Tuple[List[List[str]], Dict["Models.Vehicle", str]]:

        # Initialize return list
        # This list will hold the final optimal positions of the ambulances
        final_positions: List = []

        # Initialize the return dict
        # This will hold the vehicle repositioning values
        final_repositioning: Dict["Models.Vehicle", str] = {}

        # Parameters
        actual_ALS_vehicles = {v.to_node: v for v in simulator.getAvaliableVehicles(0, borough)}
        actual_BLS_vehicles = {v.to_node: v for v in simulator.getAvaliableVehicles(1, borough)}
        neighbor_k = params.neighbor_k
        neighborhood = params.neighborhood
        ambulance_distribution = params.ambulance_distribution
        D_rates = params.demand_rates
        Busy_rates = params.mean_busytime
        Q = params.Q
        P = params.P
        C = params.candidates_borough[borough]
        D = params.demand_borough[borough]
        t = simulator.timePeriod()

        overload_penalty = params.overload_penalty
        maximum_uber = params.maximum_uber_per_period
        cand_cand_time = params.cand_cand_time

        ########################
        #    First ALS part    #
        ########################

        # First stage
        start_time = time.time()
        print('0 - Starting first stage')
        # Create the mip solver with the CBC backend.
        model = grb.Model(name="First stage optimal positions")
        # Declare model variables
        print(time.time() - start_time, '- Declaring Variables...')

        y = [model.addVar(vtype=grb.GRB.BINARY, name='y_' + node)
             for j, node in enumerate(C)]

        x = [[model.addVar(vtype=grb.GRB.BINARY, name='x_' + str(k) + '_' + node)
                        for k in range(neighbor_k[t][node]+1)]
                        for node in C]
        # Coefficients
        # ------------
        # [j, i] matrix
        print(time.time() - start_time, '- Computing O.F. coefficients...')
        survival_matrix = self.SurvivalFunction(params.cand_demand_time[t])

        # Filter the survival matrix leaving only the nodes that are reachable,
        # 0 to all the rest
        filtered_survival = np.array([[survival_matrix[j, i] if c_node in params.reachable_inverse[t][d_node] else 0        # noqa E501
                                       for j, c_node in enumerate(C)]
                                       for i, d_node in enumerate(D)]).T

        # [j, i] matrix respresenting S_{i,j} * d_i
        coefficients = D_rates[0].loc[t+1, D].values * filtered_survival                      # noqa E501
        
        # Objective function
        print(time.time() - start_time, '- Setting O.F...')
        availability = [grb.LinExpr([Q[j][k] for k in range(neighbor_k[t][node])],
                        [x[j][k] for k in range(neighbor_k[t][node])]) for j, node in enumerate(C)]       # noqa E501


        
        model.setObjective(grb.quicksum(grb.quicksum(coefficients[C.index(c_node),i]*availability[C.index(c_node)] 
                           if c_node in C else 0
                           for c_node in params.reachable_inverse[t][d_node])
                           for i, d_node in enumerate(D)))
        model.ModelSense = grb.GRB.MAXIMIZE

        print(time.time() - start_time, '- Constraints...')
        # Constraints
        # Capacity constraint
        Cap_constraint = model.addConstr(lhs=grb.quicksum(y),
                                             sense=grb.GRB.EQUAL,
                                             rhs=len(actual_ALS_vehicles),
                                             name='CapacityConstraint')

        #Cap_constraint = {b: model.addConstr(lhs=grb.quicksum(y[j] for j in range(len(C))),
        #                                     sense=grb.GRB.EQUAL,
        #                                     rhs=ambulance_distribution[0][b],
        #                                     name='CapacityConstraint_{}'.format(b)) for b in range(6)}
        # Constraint 1
        Const_1 = {j: model.addConstr(lhs=grb.quicksum(x[j]),
                                      sense=grb.GRB.LESS_EQUAL,
                                      rhs=1,
                                      name='Const_1_{}'.format(j)) for j, c_node in enumerate(C)}                            # noqa E501
        # Constraint 2
        Const_2 = {j: model.addConstr(lhs=grb.quicksum(y[C.index(u)] if u in C else 0
                                                       for u in params.neighborhood_candidates[t][c_node]),    # noqa E501
                                      sense=grb.GRB.EQUAL,
                                      rhs=grb.LinExpr(list(range(len(x[j]))), x[j]),
                                      name='Const_2_{}'.format(j)) for j, c_node in enumerate(C)}                            # noqa E501

        print(time.time() - start_time, '- Model Params...')
        model.setParam('MIPGap', .1)
        model.setParam('TimeLimit', 45)
        
        print(time.time() - start_time, '- Solving the model...')
        status = model.optimize()

        print(time.time() - start_time, '- Done!')

        # -------------------------
        ########################
        #   Second ALS part    #
        ########################

        print(time.time() - start_time, '- Starting first stage part 2')

        actual_positions = [actual_ALS_vehicles[v].to_node for v in actual_ALS_vehicles]
        target_positions = []
        for j in range(len(C)):
            if y[j].x == 1:
                target_positions.append(C[j])
        
        # Create the mip solver with the CBC backend.
        model = grb.Model(name="First stage optimal relocations")

        print(time.time() - start_time, '- Declaring Variables...')
        x = [[model.addVar(vtype=grb.GRB.BINARY, name='x_' + i + '_' + j)
              for j in target_positions]
              for i in actual_positions]
        
        y = [model.addVar(vtype=grb.GRB.CONTINUOUS, lb=0, name='y_' + i)
             for i in actual_ALS_vehicles]

        print(time.time() - start_time, '- Computing coefficents...')
        weights = np.array(simulator.city_graph.es['length']) / simulator.parameters.getSpeedList(simulator.timePeriod())
        travel_times = np.array(simulator.city_graph.shortest_paths(actual_positions, target_positions, weights))

        print(time.time() - start_time, '- Setting O.F...')
        model.setObjective(grb.quicksum(grb.LinExpr(travel_times[i], x[i])
                            for i in range(len(actual_positions)))
                            + overload_penalty*grb.quicksum(y))
        model.ModelSense = grb.GRB.MINIMIZE

        print(time.time() - start_time, '- Constraints...')
        # Constraint 1
        Const_1 = {i: model.addConstr(lhs=grb.quicksum(x[i]),
                                      sense=grb.GRB.LESS_EQUAL,
                                      rhs=1,
                                      name='Const_1_{}'.format(i)) for i in range(len(actual_positions))}
        # Constraint 2
        Const_2 = {j: model.addConstr(lhs=grb.quicksum(x[i][j] for i in range(len(actual_positions))),
                                      sense=grb.GRB.GREATER_EQUAL,
                                      rhs=1,
                                      name='Const_1_{}'.format(j)) for j in range(len(target_positions))}
        # Constraint 3
        # TODO: Check rhs
        Const_3 = {i: model.addConstr(lhs=actual_ALS_vehicles[node].reposition_workload + \
                                          grb.LinExpr(travel_times[i], x[i]),
                                      sense=grb.GRB.LESS_EQUAL,
                                      rhs=y[i],
                                      name='Const_1_{}'.format(i)) for i, node in enumerate(actual_positions)}

        print(time.time() - start_time, '- Model Params...')
        model.setParam('MIPGap', .1)
        
        print(time.time() - start_time, '- Solving the model...')
        status = model.optimize()
        print(time.time() - start_time, '- Done!')
        
        # Update positions for first stage
        final_positions.append(target_positions)

        reposition_matrix = [[x[i][j].x for j in range(len(target_positions))] for i in range(len(actual_positions))]
        for i, node in enumerate(actual_positions):
            final_repositioning[actual_ALS_vehicles[node]] = target_positions[reposition_matrix[i].index(1)]


        # -------------------------------------------------------------------------------------------------------------------

        ########################
        #    First BLS part    #
        ########################

        # Start the second stage
        print(time.time() - start_time, '- Starting second stage')

        # Create the mip solver with the CBC backend.
        model = grb.Model(name="Second stage optimal positions")

        # Declare model variables
        print(time.time() - start_time, '- Declaring Variables...')

        y = [model.addVar(vtype=grb.GRB.BINARY, name='y_' + node)
             for j, node in enumerate(C)]

        z = [model.addVar(vtype=grb.GRB.INTEGER, name='z_' + node)
             for j, node in enumerate(C)]

        x = [[model.addVar(vtype=grb.GRB.BINARY, name='x_' + str(k) + '_' + node)
                        for k in range(neighbor_k[t][node]+1)]
                        for node in C]

        # Coefficients
        # ------------
        print(time.time() - start_time, '- Computing borough demand...')
        borough_demand = np.array([np.sum(P[D.index(d), 1] for d in borough_demand[b]) for b in range(1,6)])
        borough_demand = borough_demand/np.sum(borough_demand)

        # Allocate the ALS ambulances to each borough
        n_total = len(actual_BLS_vehicles)
        n_allocated = 0
        allocated_ambulances = [0]
        for b in range(5-1):
            n_allocated += int(n_total * borough_demand[b])
            allocated_ambulances.append(int(n_total * borough_demand[b]))
        allocated_ambulances.append(int(n_total - n_allocated))

        # [j, i] matrix
        print(time.time() - start_time, '- Computing O.F. coefficients...')

        # Filter the survival matrix leaving only the nodes that are reachable,
        # 0 to all the rest
        filtered_survival = np.array([[1 if c_node in params.reachable_inverse[t][d_node] else 0        # noqa E501
                                       for j, c_node in enumerate(C)]
                                       for i, d_node in enumerate(D)]).T

        # [j, i] matrix respresenting S_{i,j} * d_i
        coefficients = params.demand_rates[1].loc[t+1, params.demand_nodes].values * filtered_survival                      # noqa E501
        
        # Objective function
        print(time.time() - start_time, '- Setting O.F...')
        availability = [grb.LinExpr([P[j][k] for k in range(neighbor_k[t][node])],
                        [x[j][k] for k in range(neighbor_k[t][node])]) for j, node in enumerate(C)]       # noqa E501
        
        model.setObjective(grb.quicksum(grb.quicksum(coefficients[C.index(c_node),i]*availability[C.index(c_node)] 
                           for c_node in params.reachable_inverse[t][d_node])
                           for i, d_node in enumerate(D)))
        model.ModelSense = grb.GRB.MAXIMIZE

        print(time.time() - start_time, '- Constraints...')
        # Constraints
        # Capacity constraint
        Cap_constraint = {b: model.addConstr(lhs=grb.quicksum(y[C.index(c)] for c in borough_cand[b]),
                                             sense=grb.GRB.EQUAL,
                                             rhs=allocated_ambulances[b],
                                             name='CapacityConstraint_{}'.format(b)) for b in range(6)}
        # Constraint 1
        Const_1 = {j: model.addConstr(lhs=grb.quicksum(x[j]),
                                      sense=grb.GRB.LESS_EQUAL,
                                      rhs=1,
                                      name='Const_1_{}'.format(j)) for j, c_node in enumerate(C)}                            # noqa E501
        # Constraint 2
        Const_2 = {j: model.addConstr(lhs=grb.quicksum(y[C.index(u)] + z[C.index(u)] for u in params.neighborhood_candidates[t][c_node]),
                                      sense=grb.GRB.EQUAL,
                                      rhs=grb.LinExpr(list(range(len(x[j]))), x[j]),
                                      name='Const_2_{}'.format(j)) for j, c_node in enumerate(C)}                            # noqa E501
        
        # Constraint 3
        mean_uber_load = [Busy_rates[1].loc[t+1,neighborhood[t][j]].mean() for j in C]
        Const_3 = model.addConstr(lhs=grb.LinExpr(mean_uber_load, z),
                                  sense=grb.GRB.LESS_EQUAL,
                                  rhs=maximum_uber - simulator.uber_hours,
                                  name='UberConstraint')

        print(time.time() - start_time, '- Model Params...')
        model.setParam('MIPGap', .1)
        
        print(time.time() - start_time, '- Solving the model...')
        status = model.optimize()
        print(time.time() - start_time, '- Done!')

        print()
        return final_positions, final_repositioning