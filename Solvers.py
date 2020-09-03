import time
import random
import numpy as np
import gurobipy as grb
from typing import Dict, List, Tuple, Optional

# Internal imports
import Models
import SimulatorBasics

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
        emergencies.sort(key=lambda e: simulator.now() - e.arrival_time, reverse=True)
        emergencies_by_level = [[e for e in emergencies if e.severity == 1],
                                [e for e in emergencies if e.severity > 1]]

        emergencies_vertices: List[List[str]] = [[e.node for e in level] for level in emergencies_by_level]

        vehicles= [[[ambulance
                    for ambulance in simulator.getAvaliableVehicles(v_type=v, borough=b)]
                    for v in range(simulator.parameters.vehicle_types)]
                    for b in range(1,6)]

        vehicle_positions = [[[ambulance.to_node
                             for ambulance in v]
                             for v in b]
                             for b in vehicles]


        used_vehicles: List[Tuple[int, int, int]] = []
        assignment_dict: Dict["Models.Vehicle", "Models.Emergency"] = {}

        weights = np.array(simulator.city_graph.es['length']) / simulator.parameters.getSpeedList(simulator.timePeriod())

        # First priority emergencies
        for e, emergency in enumerate(emergencies_by_level[0]):
            emergency_borough = int(simulator.parameters.nodes_with_borough[simulator.parameters.nodes_with_borough['osmid'] == emergency.node]['boro_code'])

            # Get the nearest borough
            #seen_nodes = []
            node = emergency.node
            if emergency_borough == 0:
                distances = simulator.getShortestDistances([v.to_node for v in simulator.vehicles], node)
                emergency_borough = simulator.vehicles[np.argmin(distances.reshape(-1))].borough
                print("Emergency from node {} mapped to borough {}".format(emergency.node, emergency_borough))

            travel_times = np.array(simulator.city_graph.shortest_paths(vehicle_positions[emergency_borough-1][0], emergency.node, weights))
            valid_indexes = np.where(travel_times < 8*60)[0]

            if len(valid_indexes) > 0:
                candidates = list(zip(valid_indexes, travel_times[valid_indexes].squeeze().reshape(-1)))
                candidates.sort(key=lambda c: c[1])
            else:
                candidates = list(zip(range(travel_times.shape[0]), travel_times.reshape(-1)))
                candidates.sort(key=lambda c: c[1])

            for c in candidates:
                if (emergency_borough, 0, c[0]) not in used_vehicles:
                    used_vehicles.append((emergency_borough, 0, c[0]))
                    assignment_dict[vehicles[emergency_borough-1][0][c[0]]] = emergency
                    break

    
        for e, emergency in enumerate(emergencies_by_level[1]):
            emergency_borough = int(simulator.parameters.nodes_with_borough[simulator.parameters.nodes_with_borough['osmid'] == emergency.node]['boro_code'])

            # Get the nearest borough
            node = emergency.node
            if emergency_borough == 0:
                distances = simulator.getShortestDistances([v.to_node for v in simulator.vehicles], node)
                emergency_borough = simulator.vehicles[np.argmin(distances.reshape(-1))].borough
                print("Emergency from node {} mapped to borough {}".format(emergency.node, emergency_borough))

            # Assign Uber based on probability
            if simulator.parameters.is_uber_available and emergency.severity == 3 and random.random() < simulator.parameters.uber_low_severity_ratio:
               assignment_dict[simulator.newUberVehicle(simulator.parameters.uber_nodes[emergency.node], emergency_borough)] = emergency
               continue

            travel_times = [simulator.getShortestDistances(vehicle_positions[emergency_borough-1][0], emergency.node),
                            simulator.getShortestDistances(vehicle_positions[emergency_borough-1][1], emergency.node)]
            valid_indexes = [np.where(travel_times[0] < 8*60)[0],
                             np.where(travel_times[1] < 8*60)[0]]

            if len(valid_indexes[0]) + len(valid_indexes[1]) > 0:
                candidates_preparedness: List[Tuple[int, int, float]] = []

                travel_to_demand = [[list(simulator.covering_state[emergency_borough][v][1]) for v in vehicles[emergency_borough-1][0]],
                                    [list(simulator.covering_state[emergency_borough][v][1]) for v in vehicles[emergency_borough-1][1]]]

                # Compute the preparedness difference for each BLS
                for i in valid_indexes[1]:
                    preparedness = \
                        simulator.computeSystemPreparedness(borough=emergency_borough,
                                                            travel_matrix=[travel_to_demand[0], travel_to_demand[1][:i]+travel_to_demand[1][i+1:]])
                    candidates_preparedness.append((1, i, preparedness))

                # Compute the preparedness difference for each ALS
                for i in valid_indexes[0]:
                    preparedness = \
                        simulator.computeSystemPreparedness(borough=emergency_borough,
                                                            travel_matrix=[travel_to_demand[0][:i]+travel_to_demand[0][i+1:],travel_to_demand[1]])
                    candidates_preparedness.append((0, i, preparedness))

                for cp in candidates_preparedness:
                    if (emergency_borough, cp[0], cp[1]) not in used_vehicles:
                        used_vehicles.append((emergency_borough, cp[0], cp[1]))
                        assignment_dict[vehicles[emergency_borough-1][cp[0]][cp[1]]] = emergency
                        break
            
            else:
                candidates_nearest = list(zip([0] * travel_times[0].shape[0], range(travel_times[0].shape[0]), travel_times[0].reshape(-1))) + \
                                     list(zip([1] * travel_times[1].shape[0], range(travel_times[1].shape[0]), travel_times[1].reshape(-1)))
                candidates_nearest.sort(key=lambda c: c[2])

                for cn in candidates_nearest:
                    if (emergency_borough, cn[0], cn[1]) not in used_vehicles:
                        used_vehicles.append((emergency_borough, cn[0], cn[1]))
                        assignment_dict[vehicles[emergency_borough-1][cn[0]][cn[1]]] = emergency
                        break
        
        if len(assignment_dict) != len(emergencies_by_level[0]) + len(emergencies_by_level[1]):
            print(SimulatorBasics.secondsToTimestring(simulator.now()) + ': Not all emergencies were assigned. ' + str([len(emergencies_by_level[0]), len(emergencies_by_level[1])]))

        return assignment_dict


class NearestDispatcher(DispatcherModel):

    def __init__(self):
        super().__init__()

    def assign(self,
               simulator: "Models.EMSModel",
               is_uber_avaialble: bool = False) -> Dict["Models.Vehicle", "Models.Emergency"]:
        # Getting a reference for unassigned emergencies
        emergencies: List[Models.Emergency] = \
            list(set(simulator.activeEmergencies) - set(simulator.assignedEmergencies))

        # Sort by remaining time till expiring
        emergencies.sort(key=lambda e: simulator.now() - e.arrival_time, reverse=True)
        emergencies_vertices: List[str] = [e.node for e in emergencies]

        # Final list of vertices to apply dijkstra to
        vehicles: Dict[int, List[List[Models.Vehicle]]] = {b: [simulator.getAvaliableVehicles(v_type= v,borough=b) for v in [0,1]] for b in range(1,6)}
        #vehicle_vertices: List[str] = [v.pos for v in vehicles]
        #n = len(vehicle_vertices)  # Number of available vehicles

        used_vehicles: List[Tuple[int, int, int]] = []
        assignment_dict: Dict["Models.Vehicle", "Models.Emergency"] = {}

        for emergency in emergencies:
            emergency_borough = int(simulator.parameters.nodes_with_borough[simulator.parameters.nodes_with_borough['osmid'] == emergency.node]['boro_code'])

            # Get the nearest borough
            node = emergency.node
            if emergency_borough == 0:
                distances = simulator.getShortestDistances([v.to_node for v in simulator.vehicles], node)
                emergency_borough = simulator.vehicles[np.argmin(distances.reshape(-1))].borough
                print("Emergency from node {} mapped to borough {}".format(emergency.node, emergency_borough))

            # Assign Uber based on probability
            if simulator.parameters.is_uber_available and emergency.severity == 3 and random.random() < simulator.parameters.uber_low_severity_ratio:
                assignment_dict[simulator.newUberVehicle(simulator.parameters.uber_nodes[emergency.node], emergency_borough)] = emergency
                continue

            candidates: List[Tuple[int, int, float]] = []
            if emergency.severity == 1:
                distances = simulator.getShortestDistances([v.pos for v in vehicles[emergency_borough][0]], emergency.node).reshape(-1)
                candidates = list(zip([0] * len(distances), range(len(distances)), distances))
                candidates.sort(key=lambda c: c[2])
            else:
                vehicle_positions = [[v.pos for v in vehicles[emergency_borough][0]], [v.pos for v in vehicles[emergency_borough][1]]]
                travel_times = [simulator.getShortestDistances(vehicle_positions[0], emergency.node),
                                simulator.getShortestDistances(vehicle_positions[1], emergency.node)]
                candidates = list(zip([0] * travel_times[0].shape[0], range(travel_times[0].shape[0]), travel_times[0].reshape(-1))) + \
                             list(zip([1] * travel_times[1].shape[0], range(travel_times[1].shape[0]), travel_times[1].reshape(-1)))
                candidates.sort(key=lambda c: c[2])

            for c in candidates:
                if (emergency_borough, c[0], c[1]) not in used_vehicles:
                    used_vehicles.append((emergency_borough, c[0], c[1]))
                    assignment_dict[vehicles[emergency_borough][c[0]][c[1]]] = emergency
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
                 initial: bool,
                 borough,
                 workload_restrinction: bool = True) -> Tuple[List[List[str]], Dict["Models.Vehicle", str]]:
        """

        """
        print('Warning! Relocation not implemented')
        return [], {}


class MaxSingleCoverRelocator(RelocationModel):

    def __init__(self):
        super().__init__()

    def relocate(self,
                 simulator: "Models.EMSModel",
                 params: "Models.SimulationParameters",
                 initial: bool,
                 borough = None,
                 workload_restrinction: bool = True) -> Tuple[List[List[str]], Dict["Models.Vehicle", str]]:

        print('Borough', borough, 'at', SimulatorBasics.secondsToTimestring(simulator.now()))
        print(len(simulator.getAvaliableVehicles(0, borough)), 'and', len(simulator.getAvaliableVehicles(1, borough)),'at', len(params.candidates_borough[borough]))

        # Initialize return list
        # This list will hold the final optimal positions of the ambulances
        final_positions: List = []

        # Initialize the return dict
        # This will hold the vehicle repositioning values
        final_repositioning: Dict["Models.Vehicle", str] = {}


        # Parameters
        actual_ALS_vehicles = simulator.getAvaliableVehicles(0, borough)
        actual_BLS_vehicles = simulator.getAvaliableVehicles(1, borough)
        actual_ALS_vehicles_pos = {v.to_node: v for v in simulator.getAvaliableVehicles(0, borough)}
        actual_BLS_vehicles_pos = {v.to_node: v for v in simulator.getAvaliableVehicles(1, borough)}
        neighbor_k = params.neighbor_k
        neighborhood = params.neighborhood
        D_rates = params.demand_rates
        Busy_rates = params.mean_busytime
        Q = params.Q
        P = params.P
        C = params.candidates_borough[borough]
        D = params.demand_borough[borough]
        t = simulator.timePeriod()

        overload_penalty = params.overload_penalty              # theta_1
        max_overload_ALS = params.maximum_overload_ALS          # theta_2
        max_overload_BLS = params.maximum_overload_BLS          # theta_3

        ########################
        #    First ALS part    #
        ########################

        # First stage
        print('OPTIMIZING FOR BOROUGH {}'.format(borough))
        start_time = time.time()
        print('0 - Starting first stage')

        # Create the mip solver with the CBC backend.
        model = grb.Model(name="First stage optimal positions")

        # Declare model variables
        print(time.time() - start_time, '- Declaring Variables...')
        y = [model.addVar(vtype=grb.GRB.BINARY, name='y_' + node)
             for j, node in enumerate(C)]

        x = [model.addVar(vtype=grb.GRB.BINARY, name='x_' + node) for node in D]

        model.setObjective(grb.LinExpr(D_rates[0].loc[t+1, D].values, x))

        # Constraints
        # Capacity constraint
        Cap_constraint = model.addConstr(lhs=grb.quicksum(y),
                                             sense=grb.GRB.EQUAL,
                                             rhs=len(actual_ALS_vehicles_pos),
                                             name='CapacityConstraint')
        # Constraint 1
        Const_1 = {i: model.addConstr(lhs=x[i],
                                      sense=grb.GRB.LESS_EQUAL,
                                      rhs=grb.quicksum(y[C.index(c_node)] if c_node in C else 0 for _, c_node in enumerate(params.reachable_inverse[t][d_node])),
                                      name='Const_1_{}'.format(i)) for i, d_node in enumerate(D)}

        model.setParam('LogToConsole', 0)
        model.setParam('MIPGap',  simulator.parameters.optimization_gap)
        print(time.time() - start_time, '- Solving the model...')
        status = model.optimize()
        simulator.statistics['GAPALSPart1'].record(simulator.now(), model.MIPGap)
        print(time.time() - start_time, '- Done!')

        # -------------------------
        ########################
        #   Second ALS part    #
        ########################

        print(time.time() - start_time, '- Starting first stage part 2')

        actual_positions = [actual_ALS_vehicles_pos[v].to_node for v in actual_ALS_vehicles_pos]
        target_positions = []
        for j in range(len(C)):
            if y[j].x == 1:
                target_positions.append(C[j])
        
        # Create the mip solver with the CBC backend.
        model = grb.Model(name="First stage optimal relocations")

        #print(time.time() - start_time, '- Declaring Variables...')
        x = [[model.addVar(vtype=grb.GRB.BINARY, name='x_' + i + '_' + j)
              for j in target_positions]
              for i in actual_positions]
        
        y = [model.addVar(vtype=grb.GRB.CONTINUOUS, lb=0, name='y_' + i)
             for i in actual_ALS_vehicles_pos]

        #print(time.time() - start_time, '- Computing coefficents...')
        weights = np.array(simulator.city_graph.es['length']) / simulator.parameters.getSpeedList(simulator.timePeriod())
        travel_times = np.array(simulator.city_graph.shortest_paths(actual_positions, target_positions, weights))

        # Computing alpha value (The amount of time an ambulance can spend relocating)
        if not initial:
            alpha_1 = max_overload_ALS * simulator.timeInsidePeriod()
        else:
            alpha_1 = 24*3600 # Set it to the possible maximum value so the restriction is relaxed

        #print(time.time() - start_time, '- Setting O.F...')
        model.setObjective(grb.quicksum(grb.LinExpr(travel_times[i], x[i])
                            for i in range(len(actual_positions)))
                            + overload_penalty*grb.quicksum(y))
        model.ModelSense = grb.GRB.MINIMIZE

        #print(time.time() - start_time, '- Constraints...')
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
        
        if workload_restrinction:
            # Constraint 3
            Const_3 = {i: model.addConstr(lhs=actual_ALS_vehicles_pos[node].reposition_workload + \
                                            grb.LinExpr(travel_times[i], x[i]),
                                        sense=grb.GRB.LESS_EQUAL,
                                        rhs=alpha_1 + y[i],
                                        name='Const_1_{}'.format(i)) for i, node in enumerate(actual_positions)}

        #print(time.time() - start_time, '- Model Params...')
        model.setParam('MIPGap',  simulator.parameters.optimization_gap)
        model.setParam('LogToConsole', 0)
        
        print(time.time() - start_time, '- Solving the model...')
        status = model.optimize()
        try:
            simulator.statistics['GAPALSPart2'].record(simulator.now(), model.MIPGap)
        except:
            pass
        print(time.time() - start_time, '- Done!')
        
        
        # Update positions for first stage
        final_positions.append(target_positions)

        reposition_matrix = [[x[i][j].x for j in range(len(target_positions))] for i in range(len(actual_positions))]
        for i, node in enumerate(actual_positions):
            final_repositioning[actual_ALS_vehicles_pos[node]] = target_positions[reposition_matrix[i].index(1)]

        # Record statistics
        simulator.statistics['OptimizationSizeALS' + str(borough)].record(simulator.now(), len(actual_ALS_vehicles_pos))
        simulator.statistics['OptimizationTimeALS' + str(borough)].record(simulator.now(), time.time() - start_time)

        # -------------------------------------------------------------------------------------------------------------------

        start_bls_time = time.time()

        ########################
        #    First BLS part    #
        ########################

        # First stage
        print('OPTIMIZING FOR BOROUGH {}'.format(borough))
        start_time = time.time()
        print('0 - Starting first stage')

        # Create the mip solver with the CBC backend.
        model = grb.Model(name="Second stage optimal positions")

        # Declare model variables
        print(time.time() - start_time, '- Declaring Variables...')
        y = [model.addVar(vtype=grb.GRB.BINARY, name='y_' + node)
             for j, node in enumerate(C)]

        x = [model.addVar(vtype=grb.GRB.BINARY, name='x_' + node) for node in D]

        model.setObjective(grb.LinExpr(D_rates[1].loc[t+1, D].values, x))

        # Constraints
        # Capacity constraint
        Cap_constraint = model.addConstr(lhs=grb.quicksum(y),
                                             sense=grb.GRB.EQUAL,
                                             rhs=len(actual_BLS_vehicles_pos),
                                             name='CapacityConstraint')
        # Constraint 1
        Const_1 = {i: model.addConstr(lhs=x[i],
                                      sense=grb.GRB.LESS_EQUAL,
                                      rhs=grb.quicksum(y[C.index(c_node)] if c_node in C else 0 for _, c_node in enumerate(params.reachable_inverse[t][d_node])),
                                      name='Const_1_{}'.format(i)) for i, d_node in enumerate(D)}

        model.setParam('LogToConsole', 0)
        model.setParam('MIPGap',  simulator.parameters.optimization_gap)
        print(time.time() - start_time, '- Solving the model...')
        status = model.optimize()

        simulator.statistics['GAPBLSPart1'].record(simulator.now(), model.MIPGap)
        print(time.time() - start_time, '- Done!')

         # -------------------------
        ########################
        #   Second BLS part    #
        ########################

        print(time.time() - start_time, '- Starting second stage part 2')

        actual_positions = [actual_BLS_vehicles_pos[v].to_node for v in actual_BLS_vehicles_pos]
        target_positions = []
        for j in range(len(C)):
            if y[j].x == 1:
                target_positions.append(C[j])
        
        # Create the mip solver with the CBC backend.
        model = grb.Model(name="Second stage optimal relocations")

        #print(time.time() - start_time, '- Declaring Variables...')
        x = [[model.addVar(vtype=grb.GRB.BINARY, name='x_' + i + '_' + j)
              for j in target_positions]
              for i in actual_positions]

        #print(time.time() - start_time, '- Computing coefficents...')
        weights = np.array(simulator.city_graph.es['length']) / simulator.parameters.getSpeedList(simulator.timePeriod())
        travel_times = np.array(simulator.city_graph.shortest_paths(actual_positions, target_positions, weights))

        # Computing alpha value (The amount of time an ambulance can spend relocating)
        if not initial:
            alpha_2 = max_overload_BLS * simulator.timeInsidePeriod()
        else:
            alpha_2 = 24*3600 # Set it to the possible maximum value so the restriction is relaxed

        #print(time.time() - start_time, '- Setting O.F...')
        model.setObjective(grb.quicksum(grb.LinExpr(travel_times[i], x[i])
                            for i in range(len(actual_positions))))
        model.ModelSense = grb.GRB.MINIMIZE

        #print(time.time() - start_time, '- Constraints...')
        # Constraint 1
        Const_1 = {i: model.addConstr(lhs=grb.quicksum(x[i]),
                                      sense=grb.GRB.EQUAL,
                                      rhs=1,
                                      name='Const_1_{}'.format(i)) for i in range(len(actual_positions))}
        # Constraint 2
        Const_2 = {j: model.addConstr(lhs=grb.quicksum(x[i][j] for i in range(len(actual_positions))),
                                      sense=grb.GRB.GREATER_EQUAL,
                                      rhs=1,
                                      name='Const_1_{}'.format(j)) for j in range(len(target_positions))}
        if workload_restrinction:
            # Constraint 3
            Const_3 = {i: model.addConstr(lhs=actual_BLS_vehicles_pos[node].reposition_workload + \
                                            grb.LinExpr(travel_times[i], x[i]),
                                        sense=grb.GRB.LESS_EQUAL,
                                        rhs=alpha_2,
                                        name='Const_1_{}'.format(i)) for i, node in enumerate(actual_positions)}

        #print(time.time() - start_time, '- Model Params...')
        model.setParam('MIPGap',  simulator.parameters.optimization_gap)
        model.setParam('LogToConsole', 0)
        
        print(time.time() - start_time, '- Solving the model...')
        status = model.optimize()

        print(time.time() - start_time, '- Done!')
        
        if initial or model.Status == grb.GRB.OPTIMAL:
            # Update positions for first stage
            final_positions.append(target_positions)

            reposition_matrix = [[x[i][j].x for j in range(len(target_positions))] for i in range(len(actual_positions))]
            for i, node in enumerate(actual_positions):
                if node != target_positions[reposition_matrix[i].index(1)]:
                    final_repositioning[actual_BLS_vehicles_pos[node]] = target_positions[reposition_matrix[i].index(1)]
            
            try:
                simulator.statistics['GAPBLSPart2'].record(simulator.now(), model.MIPGap)
            except:
                pass

        else:
            print('Problem was unfeasible.')
            # Don't Update positions for first stage
            final_positions.append(actual_positions)

        # Record statistics
        simulator.statistics['OptimizationSizeBLS' + str(borough)].record(simulator.now(), len(actual_BLS_vehicles_pos))
        simulator.statistics['OptimizationTimeBLS' + str(borough)].record(simulator.now(), time.time() - start_bls_time)

        print()

        return final_positions, final_repositioning


class MaxExpectedSurvivalRelocator(RelocationModel):

    def __init__(self):
        super().__init__()

    def SurvivalFunction(self, response_times):
        return (1 + np.exp(0.679 + .262 * response_times)) ** -1

    def relocate(self,
                 simulator: "Models.EMSModel",
                 params: "Models.SimulationParameters",
                 initial: bool,
                 borough = None,
                 workload_restrinction: bool = True) -> Tuple[List[List[str]], Dict["Models.Vehicle", str]]:

        print('Borough', borough, 'at', SimulatorBasics.secondsToTimestring(simulator.now()))
        print(len(simulator.getAvaliableVehicles(0, borough)), 'and', len(simulator.getAvaliableVehicles(1, borough)),'at', len(params.candidates_borough[borough]))

        # Initialize return list
        # This list will hold the final optimal positions of the ambulances
        final_positions: List = []

        # Initialize the return dict
        # This will hold the vehicle repositioning values
        final_repositioning: Dict["Models.Vehicle", str] = {}

        # Parameters
        actual_ALS_vehicles = simulator.getAvaliableVehicles(0, borough)
        actual_BLS_vehicles = simulator.getAvaliableVehicles(1, borough)
        actual_ALS_vehicles_pos = {v.to_node: v for v in simulator.getAvaliableVehicles(0, borough)}
        actual_BLS_vehicles_pos = {v.to_node: v for v in simulator.getAvaliableVehicles(1, borough)}
        neighbor_k = params.neighbor_k
        neighborhood = params.neighborhood
        D_rates = params.demand_rates
        Busy_rates = params.mean_busytime
        Q = params.Q
        P = params.P
        C = params.candidates_borough[borough]
        D = params.demand_borough[borough]
        t = simulator.timePeriod()

        overload_penalty = params.overload_penalty              # theta_1
        max_overload_ALS = params.maximum_overload_ALS          # theta_2
        max_overload_BLS = params.maximum_overload_BLS          # theta_3

        ########################
        #    First ALS part    #
        ########################

        # First stage
        print('OPTIMIZING FOR BOROUGH {}'.format(borough))
        start_time = time.time()
        print('0 - Starting first stage')

        # Create the mip solver with the CBC backend.
        model = grb.Model(name="First stage optimal positions")

        # Declare model variables
        y = [model.addVar(vtype=grb.GRB.BINARY, name='y_' + node)
             for j, node in enumerate(C)]

        x = [[model.addVar(vtype=grb.GRB.BINARY, name='x_' + str(k) + '_' + node)
                        for k in range(neighbor_k[t][node]+1)]
                        for node in C]

        # Coefficients
        # ------------
        #print(time.time() - start_time, '- Computing O.F. coefficients...')
        survival_matrix = self.SurvivalFunction(params.cand_demand_time[t]/60)

        # Filter the survival matrix leaving only the nodes that are reachable,
        # 0 to all the rest
        filtered_survival = np.array([[survival_matrix[j, i] if c_node in params.reachable_inverse[t][d_node] else 0
                                       for j, c_node in enumerate(C)]
                                       for i, d_node in enumerate(D)]).T

        # [j, i] matrix respresenting S_{i,j} * d_i
        coefficients = D_rates[0].loc[t+1, D].values * filtered_survival
        
        # Objective function
        #print(time.time() - start_time, '- Setting O.F...')
        availability = [grb.LinExpr([Q[j][k] for k in range(neighbor_k[t][node])],
                        [x[j][k] for k in range(neighbor_k[t][node])]) for j, node in enumerate(C)]
     
        model.setObjective(grb.quicksum(grb.quicksum(coefficients[C.index(c_node),i]*availability[C.index(c_node)] 
                           if c_node in C else 0
                           for c_node in params.reachable_inverse[t][d_node])
                           for i, d_node in enumerate(D)))
        model.ModelSense = grb.GRB.MAXIMIZE

        #print(time.time() - start_time, '- Constraints...')
        # Constraints
        # Capacity constraint
        Cap_constraint = model.addConstr(lhs=grb.quicksum(y),
                                             sense=grb.GRB.EQUAL,
                                             rhs=len(actual_ALS_vehicles_pos),
                                             name='CapacityConstraint')
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

        #print(time.time() - start_time, '- Model Params...')
        model.setParam('MIPGap',  simulator.parameters.optimization_gap)
        model.setParam('LogToConsole', 0)
        
        print(time.time() - start_time, '- Solving the model...')
        status = model.optimize()

        simulator.statistics['GAPALSPart1'].record(simulator.now(), model.MIPGap)

        print(time.time() - start_time, '- Done!')

        # -------------------------
        ########################
        #   Second ALS part    #
        ########################
        
        print(time.time() - start_time, '- Starting first stage part 2')

        actual_positions = [actual_ALS_vehicles_pos[v].to_node for v in actual_ALS_vehicles_pos]
        target_positions = []
        for j in range(len(C)):
            if y[j].x == 1:
                target_positions.append(C[j])
        
        # Create the mip solver with the CBC backend.
        model = grb.Model(name="First stage optimal relocations")

        #print(time.time() - start_time, '- Declaring Variables...')
        x = [[model.addVar(vtype=grb.GRB.BINARY, name='x_' + i + '_' + j)
              for j in target_positions]
              for i in actual_positions]
        
        y = [model.addVar(vtype=grb.GRB.CONTINUOUS, lb=0, name='y_' + i)
             for i in actual_ALS_vehicles_pos]

        #print(time.time() - start_time, '- Computing coefficents...')
        weights = np.array(simulator.city_graph.es['length']) / simulator.parameters.getSpeedList(simulator.timePeriod())
        travel_times = np.array(simulator.city_graph.shortest_paths(actual_positions, target_positions, weights))

        # Computing alpha value (The amount of time an ambulance can spend relocating)
        if not initial:
            alpha_1 = max_overload_ALS * simulator.timeInsidePeriod()
        else:
            alpha_1 = 24*3600 # Set it to the possible maximum value so the restriction is relaxed

        #print(time.time() - start_time, '- Setting O.F...')
        model.setObjective(grb.quicksum(grb.LinExpr(travel_times[i], x[i])
                            for i in range(len(actual_positions)))
                            + overload_penalty*grb.quicksum(y))
        model.ModelSense = grb.GRB.MINIMIZE

        #print(time.time() - start_time, '- Constraints...')
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
        
        if workload_restrinction:
            # Constraint 3
            Const_3 = {i: model.addConstr(lhs=actual_ALS_vehicles_pos[node].reposition_workload + \
                                            grb.LinExpr(travel_times[i], x[i]),
                                        sense=grb.GRB.LESS_EQUAL,
                                        rhs=alpha_1 + y[i],
                                        name='Const_1_{}'.format(i)) for i, node in enumerate(actual_positions)}

        #print(time.time() - start_time, '- Model Params...')
        model.setParam('MIPGap',  simulator.parameters.optimization_gap)
        model.setParam('LogToConsole', 0)
        
        print(time.time() - start_time, '- Solving the model...')
        status = model.optimize()

        simulator.statistics['GAPALSPart2'].record(simulator.now(), model.MIPGap)
        print(time.time() - start_time, '- Done!')
        
        
        # Update positions for first stage
        final_positions.append(target_positions)

        reposition_matrix = [[x[i][j].x for j in range(len(target_positions))] for i in range(len(actual_positions))]
        for i, node in enumerate(actual_positions):
            final_repositioning[actual_ALS_vehicles_pos[node]] = target_positions[reposition_matrix[i].index(1)]

        if not initial:
            pass
            # Update the vehicle object with the new relocation time
            #for v, vehicle in actual_ALS_vehicles_pos.items():
            #    position_index = list(actual_ALS_vehicles_pos.keys()).index(v)
            #    reposition_time = travel_times[position_index][reposition_matrix[position_index].index(1)]
            #    vehicle.reposition_workload += reposition_time

        # Record statistics
        simulator.statistics['OptimizationSizeALS' + str(borough)].record(simulator.now(), len(actual_ALS_vehicles_pos))
        simulator.statistics['OptimizationTimeALS' + str(borough)].record(simulator.now(), time.time() - start_time)

        # -------------------------------------------------------------------------------------------------------------------

        start_bls_time = time.time()

        ########################
        #    First BLS part    #
        ########################

        # Start the second stage
        print(time.time() - start_time, '- Starting second stage')

        # Create the mip solver with the CBC backend.
        model = grb.Model(name="Second stage optimal positions")

        # Declare model variables
        #print(time.time() - start_time, '- Declaring Variables...')

        y = [model.addVar(vtype=grb.GRB.BINARY, name='y_' + node)
             for j, node in enumerate(C)]

        x = [[model.addVar(vtype=grb.GRB.BINARY, name='x_' + str(k) + '_' + node)
                        for k in range(neighbor_k[t][node]+1)]
                        for node in C]

        # Coefficients
        # ------------

        weighed_distance_matrix = np.clip(np.array(1/(params.cand_demand_time[t])), 0, 1)
        weighed_distance_matrix = weighed_distance_matrix + np.random.random(weighed_distance_matrix.shape) * 1e-7 - 1e-7

        # Filter the survival matrix leaving only the nodes that are reachable,
        # 0 to all the rest
        filtered_survival = np.array([[weighed_distance_matrix[j,i] if c_node in params.reachable_inverse[t][d_node] else 0        # noqa E501
                                       for j, c_node in enumerate(C)]
                                       for i, d_node in enumerate(D)]).T

        # [j, i] matrix respresenting S_{i,j} * d_i
        coefficients = D_rates[1].loc[t+1, D].values * filtered_survival                      # noqa E501
        
        # Objective function
        #print(time.time() - start_time, '- Setting O.F...')
        availability = [grb.LinExpr([P[j][k] for k in range(neighbor_k[t][node])],
                        [x[j][k] for k in range(neighbor_k[t][node])]) for j, node in enumerate(C)]       # noqa E501
        
        model.setObjective(grb.quicksum(grb.quicksum(coefficients[C.index(c_node),i]*availability[C.index(c_node)] if c_node in C else 0
                           for c_node in params.reachable_inverse[t][d_node])
                           for i, d_node in enumerate(D)))
        model.ModelSense = grb.GRB.MAXIMIZE

        #print(time.time() - start_time, '- Constraints...')
        # Constraints
        # Capacity constraint
        Cap_constraint = model.addConstr(lhs=grb.quicksum(y),
                                             sense=grb.GRB.EQUAL,
                                             rhs=len(actual_BLS_vehicles_pos),
                                             name='CapacityConstraint')
        # Constraint 1
        Const_1 = {j: model.addConstr(lhs=grb.quicksum(x[j]),
                                      sense=grb.GRB.LESS_EQUAL,
                                      rhs=1,
                                      name='Const_1_{}'.format(j)) for j, c_node in enumerate(C)}                            # noqa E501
        # Constraint 2
        Const_2 = {j: model.addConstr(lhs=grb.quicksum(y[C.index(u)] if u in C else 0 for u in params.neighborhood_candidates[t][c_node]),
                                      sense=grb.GRB.EQUAL,
                                      rhs=grb.LinExpr(list(range(len(x[j]))), x[j]),
                                      name='Const_2_{}'.format(j)) for j, c_node in enumerate(C)}

        #print(time.time() - start_time, '- Model Params...')
        model.setParam('MIPGap',  simulator.parameters.optimization_gap)
        model.setParam('LogToConsole', 0)
        
        print(time.time() - start_time, '- Solving the model...')
        status = model.optimize()

        simulator.statistics['GAPBLSPart1'].record(simulator.now(), model.MIPGap)
        print(time.time() - start_time, '- Done!')
        
        # -------------------------
        ########################
        #   Second BLS part    #
        ########################

        print(time.time() - start_time, '- Starting second stage part 2')

        actual_positions = [actual_BLS_vehicles_pos[v].to_node for v in actual_BLS_vehicles_pos]
        target_positions = []
        for j in range(len(C)):
            if y[j].x == 1:
                target_positions.append(C[j])
        
        # Create the mip solver with the CBC backend.
        model = grb.Model(name="Second stage optimal relocations")

        #print(time.time() - start_time, '- Declaring Variables...')
        x = [[model.addVar(vtype=grb.GRB.BINARY, name='x_' + i + '_' + j)
              for j in target_positions]
              for i in actual_positions]

        #print(time.time() - start_time, '- Computing coefficents...')
        weights = np.array(simulator.city_graph.es['length']) / simulator.parameters.getSpeedList(simulator.timePeriod())
        travel_times = np.array(simulator.city_graph.shortest_paths(actual_positions, target_positions, weights))

        # Computing alpha value (The amount of time an ambulance can spend relocating)
        if not initial:
            alpha_2 = max_overload_BLS * simulator.timeInsidePeriod()
        else:
            alpha_2 = 24*3600 # Set it to the possible maximum value so the restriction is relaxed

        #print(time.time() - start_time, '- Setting O.F...')
        model.setObjective(grb.quicksum(grb.LinExpr(travel_times[i], x[i])
                            for i in range(len(actual_positions))))
        model.ModelSense = grb.GRB.MINIMIZE

        #print(time.time() - start_time, '- Constraints...')
        # Constraint 1
        Const_1 = {i: model.addConstr(lhs=grb.quicksum(x[i]),
                                      sense=grb.GRB.EQUAL,
                                      rhs=1,
                                      name='Const_1_{}'.format(i)) for i in range(len(actual_positions))}
        # Constraint 2
        Const_2 = {j: model.addConstr(lhs=grb.quicksum(x[i][j] for i in range(len(actual_positions))),
                                      sense=grb.GRB.GREATER_EQUAL,
                                      rhs=1,
                                      name='Const_1_{}'.format(j)) for j in range(len(target_positions))}
        if workload_restrinction:
            # Constraint 3
            Const_3 = {i: model.addConstr(lhs=actual_BLS_vehicles_pos[node].reposition_workload + \
                                            grb.LinExpr(travel_times[i], x[i]),
                                        sense=grb.GRB.LESS_EQUAL,
                                        rhs=alpha_2,
                                        name='Const_1_{}'.format(i)) for i, node in enumerate(actual_positions)}

        #print(time.time() - start_time, '- Model Params...')
        model.setParam('MIPGap',  simulator.parameters.optimization_gap)
        model.setParam('LogToConsole', 0)
        
        print(time.time() - start_time, '- Solving the model...')
        status = model.optimize()
        simulator.statistics['GAPBLSPart2'].record(simulator.now(), model.MIPGap)
        print(time.time() - start_time, '- Done!')
        
        if initial or model.Status == grb.GRB.OPTIMAL:
            # Update positions for first stage
            final_positions.append(target_positions)

            reposition_matrix = [[x[i][j].x for j in range(len(target_positions))] for i in range(len(actual_positions))]
            for i, node in enumerate(actual_positions):
                if node != target_positions[reposition_matrix[i].index(1)]:
                    final_repositioning[actual_BLS_vehicles_pos[node]] = target_positions[reposition_matrix[i].index(1)]

        else:
            print('Problem was unfeasible.')
            # Don't Update positions for first stage
            final_positions.append(actual_positions)

        # Record statistics
        simulator.statistics['OptimizationSizeBLS' + str(borough)].record(simulator.now(), len(actual_BLS_vehicles_pos))
        simulator.statistics['OptimizationTimeBLS' + str(borough)].record(simulator.now(), time.time() - start_bls_time)

        print()
        return final_positions, final_repositioning


class MaxSurvivalRelocator(RelocationModel):

    def __init__(self):
        super().__init__()

    def SurvivalFunction(self, response_times):
        return (1 + np.exp(0.679 + .262 * response_times)) ** -1

    def relocate(self,
                 simulator: "Models.EMSModel",
                 params: "Models.SimulationParameters",
                 initial: bool,
                 borough = None,
                 workload_restrinction: bool = True) -> Tuple[List[List[str]], Dict["Models.Vehicle", str]]:

        print('Borough', borough, 'at', SimulatorBasics.secondsToTimestring(simulator.now()))
        print(len(simulator.getAvaliableVehicles(0, borough)), 'and', len(simulator.getAvaliableVehicles(1, borough)),'at', len(params.candidates_borough[borough]))

        # Initialize return list
        # This list will hold the final optimal positions of the ambulances
        final_positions: List = []

        # Initialize the return dict
        # This will hold the vehicle repositioning values
        final_repositioning: Dict["Models.Vehicle", str] = {}

        # Parameters
        actual_ALS_vehicles = simulator.getAvaliableVehicles(0, borough)
        actual_BLS_vehicles = simulator.getAvaliableVehicles(1, borough)
        actual_ALS_vehicles_pos = {v.to_node: v for v in simulator.getAvaliableVehicles(0, borough)}
        actual_BLS_vehicles_pos = {v.to_node: v for v in simulator.getAvaliableVehicles(1, borough)}
        neighbor_k = params.neighbor_k
        neighborhood = params.neighborhood
        D_rates = params.demand_rates
        Busy_rates = params.mean_busytime
        Q = params.Q
        P = params.P
        C = params.candidates_borough[borough]
        D = params.demand_borough[borough]
        t = simulator.timePeriod()

        overload_penalty = params.overload_penalty              # theta_1
        max_overload_ALS = params.maximum_overload_ALS          # theta_2
        max_overload_BLS = params.maximum_overload_BLS          # theta_3

        ########################
        #    First ALS part    #
        ########################

        # First stage
        print('OPTIMIZING FOR BOROUGH {}'.format(borough))
        start_time = time.time()
        print('0 - Starting first stage')

        # Create the mip solver with the CBC backend.
        model = grb.Model(name="First stage optimal positions")

        # Declare model variables
        y = [model.addVar(vtype=grb.GRB.BINARY, name='y_' + node)
             for j, node in enumerate(C)]

        x = [[model.addVar(vtype=grb.GRB.BINARY, name='x_' + node + '_' + c_node) for node in D] for c_node in C]

        # Coefficients
        # ------------
        #print(time.time() - start_time, '- Computing O.F. coefficients...')
        survival_matrix = self.SurvivalFunction(params.cand_demand_time[t]/60)

        # Filter the survival matrix leaving only the nodes that are reachable,
        # 0 to all the rest
        filtered_survival = np.array([[survival_matrix[j, i] if c_node in params.reachable_inverse[t][d_node] else 0
                                       for j, c_node in enumerate(C)]
                                       for i, d_node in enumerate(D)]).T

        # [j, i] matrix respresenting S_{i,j} * d_i
        coefficients = D_rates[0].loc[t+1, D].values * filtered_survival
        
        # Objective function

        # sum_{i \in D} sum_{j \in N_i} d_i S_{ij} x_{ij}
        model.setObjective(grb.quicksum(grb.quicksum(coefficients[C.index(c_node),i]*x[C.index(c_node)][i] if c_node in C else 0
                           for c_node in params.reachable_inverse[t][d_node])
                           for i, d_node in enumerate(D)))
        model.ModelSense = grb.GRB.MAXIMIZE

        #print(time.time() - start_time, '- Constraints...')
        # Constraints
        # Capacity constraint

        # sum_{j \in C} y_j = N_{ALS}
        Cap_constraint = model.addConstr(lhs=grb.quicksum(y),
                                             sense=grb.GRB.EQUAL,
                                             rhs=len(actual_ALS_vehicles_pos),
                                             name='CapacityConstraint')
        # Constraint 1

        # x_{ij} \leq y_j \forall i, j \in N_i
        Const_1 = {i: model.addConstr(lhs=x[C.index(c_node)][i],
                                      sense=grb.GRB.LESS_EQUAL,
                                      rhs= y[C.index(c_node)],
                                      name='Const_1_{}'.format(i))
                                      for i, d_node in enumerate(D)
                                      for j, c_node in enumerate(params.reachable_inverse[t][d_node])
                                      if c_node in C}

        # Constraint 2

        # sum_{j \in C} x_{ij} = 1
        Const_2 = {i: model.addConstr(lhs=grb.quicksum(x[C.index(u)][i] for u in C),
                                      sense=grb.GRB.LESS_EQUAL,
                                      rhs=1,
                                      name='Const_2_{}'.format(i)) for i, d_node in enumerate(D)}

        #print(time.time() - start_time, '- Model Params...')
        model.setParam('MIPGap',  simulator.parameters.optimization_gap)
        model.setParam('LogToConsole', 0)
        
        print(time.time() - start_time, '- Solving the model...')
        status = model.optimize()

        simulator.statistics['GAPALSPart1'].record(simulator.now(), model.MIPGap)

        print(time.time() - start_time, '- Done!')

        # -------------------------
        ########################
        #   Second ALS part    #
        ########################
        
        print(time.time() - start_time, '- Starting first stage part 2')

        actual_positions = [actual_ALS_vehicles_pos[v].to_node for v in actual_ALS_vehicles_pos]
        target_positions = []
        for j in range(len(C)):
            if y[j].x == 1:
                target_positions.append(C[j])
        
        # Create the mip solver with the CBC backend.
        model = grb.Model(name="First stage optimal relocations")

        #print(time.time() - start_time, '- Declaring Variables...')
        x = [[model.addVar(vtype=grb.GRB.BINARY, name='x_' + i + '_' + j)
              for j in target_positions]
              for i in actual_positions]
        
        y = [model.addVar(vtype=grb.GRB.CONTINUOUS, lb=0, name='y_' + i)
             for i in actual_ALS_vehicles_pos]

        #print(time.time() - start_time, '- Computing coefficents...')
        weights = np.array(simulator.city_graph.es['length']) / simulator.parameters.getSpeedList(simulator.timePeriod())
        travel_times = np.array(simulator.city_graph.shortest_paths(actual_positions, target_positions, weights))

        # Computing alpha value (The amount of time an ambulance can spend relocating)
        if not initial:
            alpha_1 = max_overload_ALS * simulator.timeInsidePeriod()
        else:
            alpha_1 = 24*3600 # Set it to the possible maximum value so the restriction is relaxed

        #print(time.time() - start_time, '- Setting O.F...')
        model.setObjective(grb.quicksum(grb.LinExpr(travel_times[i], x[i])
                            for i in range(len(actual_positions)))
                            + overload_penalty*grb.quicksum(y))
        model.ModelSense = grb.GRB.MINIMIZE

        #print(time.time() - start_time, '- Constraints...')
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
        
        if workload_restrinction:
            # Constraint 3
            Const_3 = {i: model.addConstr(lhs=actual_ALS_vehicles_pos[node].reposition_workload + \
                                            grb.LinExpr(travel_times[i], x[i]),
                                        sense=grb.GRB.LESS_EQUAL,
                                        rhs=alpha_1 + y[i],
                                        name='Const_1_{}'.format(i)) for i, node in enumerate(actual_positions)}

        #print(time.time() - start_time, '- Model Params...')
        model.setParam('MIPGap',  simulator.parameters.optimization_gap)
        model.setParam('LogToConsole', 0)
        
        print(time.time() - start_time, '- Solving the model...')
        status = model.optimize()

        try:
            simulator.statistics['GAPALSPart2'].record(simulator.now(), model.MIPGap)
        except:
            pass

        print(time.time() - start_time, '- Done!')
        
        
        # Update positions for first stage
        final_positions.append(target_positions)

        reposition_matrix = [[x[i][j].x for j in range(len(target_positions))] for i in range(len(actual_positions))]
        for i, node in enumerate(actual_positions):
            final_repositioning[actual_ALS_vehicles_pos[node]] = target_positions[reposition_matrix[i].index(1)]

        if not initial:
            pass
            # Update the vehicle object with the new relocation time
            #for v, vehicle in actual_ALS_vehicles_pos.items():
            #    position_index = list(actual_ALS_vehicles_pos.keys()).index(v)
            #    reposition_time = travel_times[position_index][reposition_matrix[position_index].index(1)]
            #    vehicle.reposition_workload += reposition_time

        # Record statistics
        simulator.statistics['OptimizationSizeALS' + str(borough)].record(simulator.now(), len(actual_ALS_vehicles_pos))
        simulator.statistics['OptimizationTimeALS' + str(borough)].record(simulator.now(), time.time() - start_time)

        # -------------------------------------------------------------------------------------------------------------------

        start_bls_time = time.time()

        ########################
        #    First BLS part    #
        ########################

        # Start the second stage
        print(time.time() - start_time, '- Starting second stage')

        # Create the mip solver with the CBC backend.
        model = grb.Model(name="Second stage optimal positions")

        # Declare model variables
        #print(time.time() - start_time, '- Declaring Variables...')

        y = [model.addVar(vtype=grb.GRB.BINARY, name='y_' + node)
             for j, node in enumerate(C)]

        x = [[model.addVar(vtype=grb.GRB.BINARY, name='x_' + node + '_' + c_node) for node in D] for c_node in C]

        # Coefficients
        # ------------

        weighed_distance_matrix = np.clip(np.array(1/(params.cand_demand_time[t])), 0, 1)
        weighed_distance_matrix = weighed_distance_matrix + np.random.random(weighed_distance_matrix.shape) * 1e-7 - 1e-7

        # Filter the survival matrix leaving only the nodes that are reachable,
        # 0 to all the rest
        filtered_survival = np.array([[weighed_distance_matrix[j,i] if c_node in params.reachable_inverse[t][d_node] else 0        # noqa E501
                                       for j, c_node in enumerate(C)]
                                       for i, d_node in enumerate(D)]).T

        # [j, i] matrix respresenting S_{i,j} * d_i
        coefficients = D_rates[1].loc[t+1, D].values * filtered_survival                      # noqa E501
        
        # Objective function
        
        model.setObjective(grb.quicksum(grb.quicksum(coefficients[C.index(c_node),i]*x[C.index(c_node)][i] if c_node in C else 0
                           for c_node in params.reachable_inverse[t][d_node])
                           for i, d_node in enumerate(D)))
        model.ModelSense = grb.GRB.MAXIMIZE

        #print(time.time() - start_time, '- Constraints...')
        # Constraints
        # Capacity constraint

        # sum_{j \in C} y_j = N_{ALS}
        Cap_constraint = model.addConstr(lhs=grb.quicksum(y),
                                             sense=grb.GRB.EQUAL,
                                             rhs=len(actual_BLS_vehicles_pos),
                                             name='CapacityConstraint')
        # Constraint 1

        # x_{ij} \leq y_j \forall i, j \in N_i
        Const_1 = {i: model.addConstr(lhs=x[C.index(c_node)][i],
                                      sense=grb.GRB.LESS_EQUAL,
                                      rhs= y[C.index(c_node)],
                                      name='Const_1_{}'.format(i))
                                      for i, d_node in enumerate(D)
                                      for j, c_node in enumerate(params.reachable_inverse[t][d_node])
                                      if c_node in C}

        # Constraint 2

        # sum_{j \in C} x_{ij} = 1
        Const_2 = {i: model.addConstr(lhs=grb.quicksum(x[C.index(u)][i] for u in C),
                                      sense=grb.GRB.LESS_EQUAL,
                                      rhs=1,
                                      name='Const_2_{}'.format(i)) for i, d_node in enumerate(D)}

        #print(time.time() - start_time, '- Model Params...')
        model.setParam('MIPGap',  simulator.parameters.optimization_gap)
        model.setParam('LogToConsole', 0)
        
        print(time.time() - start_time, '- Solving the model...')
        status = model.optimize()

        simulator.statistics['GAPBLSPart1'].record(simulator.now(), model.MIPGap)
        print(time.time() - start_time, '- Done!')
        
        # -------------------------
        ########################
        #   Second BLS part    #
        ########################

        print(time.time() - start_time, '- Starting second stage part 2')

        actual_positions = [actual_BLS_vehicles_pos[v].to_node for v in actual_BLS_vehicles_pos]
        target_positions = []
        for j in range(len(C)):
            if y[j].x == 1:
                target_positions.append(C[j])
        
        # Create the mip solver with the CBC backend.
        model = grb.Model(name="Second stage optimal relocations")

        #print(time.time() - start_time, '- Declaring Variables...')
        x = [[model.addVar(vtype=grb.GRB.BINARY, name='x_' + i + '_' + j)
              for j in target_positions]
              for i in actual_positions]

        #print(time.time() - start_time, '- Computing coefficents...')
        weights = np.array(simulator.city_graph.es['length']) / simulator.parameters.getSpeedList(simulator.timePeriod())
        travel_times = np.array(simulator.city_graph.shortest_paths(actual_positions, target_positions, weights))

        # Computing alpha value (The amount of time an ambulance can spend relocating)
        if not initial:
            alpha_2 = max_overload_BLS * simulator.timeInsidePeriod()
        else:
            alpha_2 = 24*3600 # Set it to the possible maximum value so the restriction is relaxed

        #print(time.time() - start_time, '- Setting O.F...')
        model.setObjective(grb.quicksum(grb.LinExpr(travel_times[i], x[i])
                            for i in range(len(actual_positions))))
        model.ModelSense = grb.GRB.MINIMIZE

        #print(time.time() - start_time, '- Constraints...')
        # Constraint 1
        Const_1 = {i: model.addConstr(lhs=grb.quicksum(x[i]),
                                      sense=grb.GRB.EQUAL,
                                      rhs=1,
                                      name='Const_1_{}'.format(i)) for i in range(len(actual_positions))}
        # Constraint 2
        Const_2 = {j: model.addConstr(lhs=grb.quicksum(x[i][j] for i in range(len(actual_positions))),
                                      sense=grb.GRB.GREATER_EQUAL,
                                      rhs=1,
                                      name='Const_1_{}'.format(j)) for j in range(len(target_positions))}
        if workload_restrinction:
            # Constraint 3
            Const_3 = {i: model.addConstr(lhs=actual_BLS_vehicles_pos[node].reposition_workload + \
                                            grb.LinExpr(travel_times[i], x[i]),
                                        sense=grb.GRB.LESS_EQUAL,
                                        rhs=alpha_2,
                                        name='Const_1_{}'.format(i)) for i, node in enumerate(actual_positions)}

        #print(time.time() - start_time, '- Model Params...')
        model.setParam('MIPGap',  simulator.parameters.optimization_gap)
        model.setParam('LogToConsole', 0)
        
        print(time.time() - start_time, '- Solving the model...')
        status = model.optimize()
        print(time.time() - start_time, '- Done!')
        
        if initial or model.Status == grb.GRB.OPTIMAL:
            # Update positions for first stage
            final_positions.append(target_positions)

            reposition_matrix = [[x[i][j].x for j in range(len(target_positions))] for i in range(len(actual_positions))]
            for i, node in enumerate(actual_positions):
                if node != target_positions[reposition_matrix[i].index(1)]:
                    final_repositioning[actual_BLS_vehicles_pos[node]] = target_positions[reposition_matrix[i].index(1)]
        
            simulator.statistics['GAPBLSPart2'].record(simulator.now(), model.MIPGap)

        else:
            print('Problem was unfeasible.')
            # Don't Update positions for first stage
            final_positions.append(actual_positions)

        # Record statistics
        simulator.statistics['OptimizationSizeBLS' + str(borough)].record(simulator.now(), len(actual_BLS_vehicles_pos))
        simulator.statistics['OptimizationTimeBLS' + str(borough)].record(simulator.now(), time.time() - start_bls_time)

        print()
        return final_positions, final_repositioning


class MaxExpectedSurvivalCoverageRelocator(RelocationModel):

    def __init__(self):
        super().__init__()

    def SurvivalFunction(self, response_times):
        return (1 + np.exp(0.679 + .262 * response_times)) ** -1

    def relocate(self,
                 simulator: "Models.EMSModel",
                 params: "Models.SimulationParameters",
                 initial: bool,
                 borough = None,
                 workload_restrinction: bool = True) -> Tuple[List[List[str]], Dict["Models.Vehicle", str]]:

        print('Borough', borough, 'at', SimulatorBasics.secondsToTimestring(simulator.now()))
        print(len(simulator.getAvaliableVehicles(0, borough)), 'and', len(simulator.getAvaliableVehicles(1, borough)),'at', len(params.candidates_borough[borough]))

        # Initialize return list
        # This list will hold the final optimal positions of the ambulances
        final_positions: List = []

        # Initialize the return dict
        # This will hold the vehicle repositioning values
        final_repositioning: Dict["Models.Vehicle", str] = {}

        # Parameters
        actual_ALS_vehicles = simulator.getAvaliableVehicles(0, borough)
        actual_BLS_vehicles = simulator.getAvaliableVehicles(1, borough)
        actual_ALS_vehicles_pos = {v.to_node: v for v in simulator.getAvaliableVehicles(0, borough)}
        actual_BLS_vehicles_pos = {v.to_node: v for v in simulator.getAvaliableVehicles(1, borough)}
        neighbor_k = params.neighbor_k
        neighborhood = params.neighborhood
        D_rates = params.demand_rates
        Busy_rates = params.mean_busytime
        Q = params.Q
        P = params.P
        C = params.candidates_borough[borough]
        D = params.demand_borough[borough]
        t = simulator.timePeriod()

        overload_penalty = params.overload_penalty              # theta_1
        max_overload_ALS = params.maximum_overload_ALS          # theta_2
        max_overload_BLS = params.maximum_overload_BLS          # theta_3

        ########################
        #    First ALS part    #
        ########################

        # First stage
        print('OPTIMIZING FOR BOROUGH {}'.format(borough))
        start_time = time.time()
        print('0 - Starting first stage')

        # Create the mip solver with the CBC backend.
        model = grb.Model(name="First stage optimal positions")

        # Declare model variables
        y = [model.addVar(vtype=grb.GRB.BINARY, name='y_' + node)
             for j, node in enumerate(C)]

        x = [[model.addVar(vtype=grb.GRB.BINARY, name='x_' + str(k) + '_' + node)
                        for k in range(neighbor_k[t][node]+1)]
                        for node in C]

        # Coefficients
        # ------------
        #print(time.time() - start_time, '- Computing O.F. coefficients...')
        survival_matrix = self.SurvivalFunction(params.cand_demand_time[t]/60)

        # Filter the survival matrix leaving only the nodes that are reachable,
        # 0 to all the rest
        filtered_survival = np.array([[survival_matrix[j, i] if c_node in params.reachable_inverse[t][d_node] else 0
                                       for j, c_node in enumerate(C)]
                                       for i, d_node in enumerate(D)]).T

        # [j, i] matrix respresenting S_{i,j} * d_i
        coefficients = D_rates[0].loc[t+1, D].values * filtered_survival
        
        # Objective function
        #print(time.time() - start_time, '- Setting O.F...')
        availability = [grb.LinExpr([Q[j][k] for k in range(neighbor_k[t][node])],
                        [x[j][k] for k in range(neighbor_k[t][node])]) for j, node in enumerate(C)]
     
        model.setObjective(grb.quicksum(grb.quicksum(coefficients[C.index(c_node),i]*availability[C.index(c_node)] 
                           if c_node in C else 0
                           for c_node in params.reachable_inverse[t][d_node])
                           for i, d_node in enumerate(D)))
        model.ModelSense = grb.GRB.MAXIMIZE

        #print(time.time() - start_time, '- Constraints...')
        # Constraints
        # Capacity constraint
        Cap_constraint = model.addConstr(lhs=grb.quicksum(y),
                                             sense=grb.GRB.EQUAL,
                                             rhs=len(actual_ALS_vehicles_pos),
                                             name='CapacityConstraint')
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

        #print(time.time() - start_time, '- Model Params...')
        model.setParam('MIPGap',  simulator.parameters.optimization_gap)
        model.setParam('LogToConsole', 0)
        
        print(time.time() - start_time, '- Solving the model...')
        status = model.optimize()

        simulator.statistics['GAPALSPart1'].record(simulator.now(), model.MIPGap)

        print(time.time() - start_time, '- Done!')

        # -------------------------
        ########################
        #   Second ALS part    #
        ########################
        
        print(time.time() - start_time, '- Starting first stage part 2')

        actual_positions = [actual_ALS_vehicles_pos[v].to_node for v in actual_ALS_vehicles_pos]
        target_positions = []
        for j in range(len(C)):
            if y[j].x == 1:
                target_positions.append(C[j])
        
        # Create the mip solver with the CBC backend.
        model = grb.Model(name="First stage optimal relocations")

        #print(time.time() - start_time, '- Declaring Variables...')
        x = [[model.addVar(vtype=grb.GRB.BINARY, name='x_' + i + '_' + j)
              for j in target_positions]
              for i in actual_positions]
        
        y = [model.addVar(vtype=grb.GRB.CONTINUOUS, lb=0, name='y_' + i)
             for i in actual_ALS_vehicles_pos]

        #print(time.time() - start_time, '- Computing coefficents...')
        weights = np.array(simulator.city_graph.es['length']) / simulator.parameters.getSpeedList(simulator.timePeriod())
        travel_times = np.array(simulator.city_graph.shortest_paths(actual_positions, target_positions, weights))

        # Computing alpha value (The amount of time an ambulance can spend relocating)
        if not initial:
            alpha_1 = max_overload_ALS * simulator.timeInsidePeriod()
        else:
            alpha_1 = 24*3600 # Set it to the possible maximum value so the restriction is relaxed

        #print(time.time() - start_time, '- Setting O.F...')
        model.setObjective(grb.quicksum(grb.LinExpr(travel_times[i], x[i])
                            for i in range(len(actual_positions)))
                            + overload_penalty*grb.quicksum(y))
        model.ModelSense = grb.GRB.MINIMIZE

        #print(time.time() - start_time, '- Constraints...')
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
        
        if workload_restrinction:
            # Constraint 3
            Const_3 = {i: model.addConstr(lhs=actual_ALS_vehicles_pos[node].reposition_workload + \
                                            grb.LinExpr(travel_times[i], x[i]),
                                        sense=grb.GRB.LESS_EQUAL,
                                        rhs=alpha_1 + y[i],
                                        name='Const_1_{}'.format(i)) for i, node in enumerate(actual_positions)}

        #print(time.time() - start_time, '- Model Params...')
        model.setParam('MIPGap',  simulator.parameters.optimization_gap)
        model.setParam('LogToConsole', 0)
        
        print(time.time() - start_time, '- Solving the model...')
        status = model.optimize()

        simulator.statistics['GAPALSPart2'].record(simulator.now(), model.MIPGap)
        print(time.time() - start_time, '- Done!')
        
        
        # Update positions for first stage
        final_positions.append(target_positions)

        reposition_matrix = [[x[i][j].x for j in range(len(target_positions))] for i in range(len(actual_positions))]
        for i, node in enumerate(actual_positions):
            final_repositioning[actual_ALS_vehicles_pos[node]] = target_positions[reposition_matrix[i].index(1)]

        if not initial:
            pass
            # Update the vehicle object with the new relocation time
            #for v, vehicle in actual_ALS_vehicles_pos.items():
            #    position_index = list(actual_ALS_vehicles_pos.keys()).index(v)
            #    reposition_time = travel_times[position_index][reposition_matrix[position_index].index(1)]
            #    vehicle.reposition_workload += reposition_time

        # Record statistics
        simulator.statistics['OptimizationSizeALS' + str(borough)].record(simulator.now(), len(actual_ALS_vehicles_pos))
        simulator.statistics['OptimizationTimeALS' + str(borough)].record(simulator.now(), time.time() - start_time)

        # -------------------------------------------------------------------------------------------------------------------

        start_bls_time = time.time()

        ########################
        #    First BLS part    #
        ########################

        # Start the second stage
        print(time.time() - start_time, '- Starting second stage')

        # Create the mip solver with the CBC backend.
        model = grb.Model(name="Second stage optimal positions")

        # Declare model variables
        #print(time.time() - start_time, '- Declaring Variables...')

        y = [model.addVar(vtype=grb.GRB.BINARY, name='y_' + node)
             for j, node in enumerate(C)]

        x = [[model.addVar(vtype=grb.GRB.BINARY, name='x_' + node + '_' + c_node) for node in D] for c_node in C]

        # Coefficients
        # ------------

        weighed_distance_matrix = np.clip(np.array(1/(params.cand_demand_time[t])), 0, 1)
        weighed_distance_matrix = weighed_distance_matrix + np.random.random(weighed_distance_matrix.shape) * 1e-7 - 1e-7

        # Filter the survival matrix leaving only the nodes that are reachable,
        # 0 to all the rest
        filtered_survival = np.array([[weighed_distance_matrix[j,i] if c_node in params.reachable_inverse[t][d_node] else 0        # noqa E501
                                       for j, c_node in enumerate(C)]
                                       for i, d_node in enumerate(D)]).T

        # [j, i] matrix respresenting S_{i,j} * d_i
        coefficients = D_rates[1].loc[t+1, D].values * filtered_survival                      # noqa E501
        
        # Objective function
        
        model.setObjective(grb.quicksum(grb.quicksum(coefficients[C.index(c_node),i]*x[C.index(c_node)][i] if c_node in C else 0
                           for c_node in params.reachable_inverse[t][d_node])
                           for i, d_node in enumerate(D)))
        model.ModelSense = grb.GRB.MAXIMIZE

        #print(time.time() - start_time, '- Constraints...')
        # Constraints
        # Capacity constraint

        # sum_{j \in C} y_j = N_{ALS}
        Cap_constraint = model.addConstr(lhs=grb.quicksum(y),
                                             sense=grb.GRB.EQUAL,
                                             rhs=len(actual_BLS_vehicles_pos),
                                             name='CapacityConstraint')
        # Constraint 1

        # x_{ij} \leq y_j \forall i, j \in N_i
        Const_1 = {i: model.addConstr(lhs=x[C.index(c_node)][i],
                                      sense=grb.GRB.LESS_EQUAL,
                                      rhs= y[C.index(c_node)],
                                      name='Const_1_{}'.format(i))
                                      for i, d_node in enumerate(D)
                                      for j, c_node in enumerate(params.reachable_inverse[t][d_node])
                                      if c_node in C}

        # Constraint 2

        # sum_{j \in C} x_{ij} = 1
        Const_2 = {i: model.addConstr(lhs=grb.quicksum(x[C.index(u)][i] for u in C),
                                      sense=grb.GRB.LESS_EQUAL,
                                      rhs=1,
                                      name='Const_2_{}'.format(i)) for i, d_node in enumerate(D)}

        #print(time.time() - start_time, '- Model Params...')
        model.setParam('MIPGap',  simulator.parameters.optimization_gap)
        model.setParam('LogToConsole', 0)
        
        print(time.time() - start_time, '- Solving the model...')
        status = model.optimize()

        simulator.statistics['GAPBLSPart1'].record(simulator.now(), model.MIPGap)
        print(time.time() - start_time, '- Done!')
        
        # -------------------------
        ########################
        #   Second BLS part    #
        ########################

        print(time.time() - start_time, '- Starting second stage part 2')

        actual_positions = [actual_BLS_vehicles_pos[v].to_node for v in actual_BLS_vehicles_pos]
        target_positions = []
        for j in range(len(C)):
            if y[j].x == 1:
                target_positions.append(C[j])
        
        # Create the mip solver with the CBC backend.
        model = grb.Model(name="Second stage optimal relocations")

        #print(time.time() - start_time, '- Declaring Variables...')
        x = [[model.addVar(vtype=grb.GRB.BINARY, name='x_' + i + '_' + j)
              for j in target_positions]
              for i in actual_positions]

        #print(time.time() - start_time, '- Computing coefficents...')
        weights = np.array(simulator.city_graph.es['length']) / simulator.parameters.getSpeedList(simulator.timePeriod())
        travel_times = np.array(simulator.city_graph.shortest_paths(actual_positions, target_positions, weights))

        # Computing alpha value (The amount of time an ambulance can spend relocating)
        if not initial:
            alpha_2 = max_overload_BLS * simulator.timeInsidePeriod()
        else:
            alpha_2 = 24*3600 # Set it to the possible maximum value so the restriction is relaxed

        #print(time.time() - start_time, '- Setting O.F...')
        model.setObjective(grb.quicksum(grb.LinExpr(travel_times[i], x[i])
                            for i in range(len(actual_positions))))
        model.ModelSense = grb.GRB.MINIMIZE

        #print(time.time() - start_time, '- Constraints...')
        # Constraint 1
        Const_1 = {i: model.addConstr(lhs=grb.quicksum(x[i]),
                                      sense=grb.GRB.EQUAL,
                                      rhs=1,
                                      name='Const_1_{}'.format(i)) for i in range(len(actual_positions))}
        # Constraint 2
        Const_2 = {j: model.addConstr(lhs=grb.quicksum(x[i][j] for i in range(len(actual_positions))),
                                      sense=grb.GRB.GREATER_EQUAL,
                                      rhs=1,
                                      name='Const_1_{}'.format(j)) for j in range(len(target_positions))}
        if workload_restrinction:
            # Constraint 3
            Const_3 = {i: model.addConstr(lhs=actual_BLS_vehicles_pos[node].reposition_workload + \
                                            grb.LinExpr(travel_times[i], x[i]),
                                        sense=grb.GRB.LESS_EQUAL,
                                        rhs=alpha_2,
                                        name='Const_1_{}'.format(i)) for i, node in enumerate(actual_positions)}

        #print(time.time() - start_time, '- Model Params...')
        model.setParam('MIPGap',  simulator.parameters.optimization_gap)
        model.setParam('LogToConsole', 0)
        
        print(time.time() - start_time, '- Solving the model...')
        status = model.optimize()
        print(time.time() - start_time, '- Done!')
        
        if initial or model.Status == grb.GRB.OPTIMAL:
            # Update positions for first stage
            final_positions.append(target_positions)

            reposition_matrix = [[x[i][j].x for j in range(len(target_positions))] for i in range(len(actual_positions))]
            for i, node in enumerate(actual_positions):
                if node != target_positions[reposition_matrix[i].index(1)]:
                    final_repositioning[actual_BLS_vehicles_pos[node]] = target_positions[reposition_matrix[i].index(1)]
            try:
                simulator.statistics['GAPBLSPart2'].record(simulator.now(), model.MIPGap)
            except:
                pass

        else:
            print('Problem was unfeasible.')
            # Don't Update positions for first stage
            final_positions.append(actual_positions)

        # Record statistics
        simulator.statistics['OptimizationSizeBLS' + str(borough)].record(simulator.now(), len(actual_BLS_vehicles_pos))
        simulator.statistics['OptimizationTimeBLS' + str(borough)].record(simulator.now(), time.time() - start_bls_time)

        print()
        return final_positions, final_repositioning