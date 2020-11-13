import time
import pickle
import OnlineSolvers
import numpy as np
import gurobipy as grb
from typing import Dict, List, Tuple

import Models
import SimulatorBasics


class ROA(OnlineSolvers.RelocationModel):

    def __init__(self):
        super().__init__()

    def SurvivalFunction(self, response_times):
        return (1 + np.exp(0.679 + .262 * response_times)) ** -1

    def optimize(self,
                 simulator: "Models.EMSModel",
                 params: "Models.SimulationParameters",
                 initial: bool,
                 severity: int = 0,
                 borough=None) -> Tuple[List[str], Dict["Models.Vehicle", str], Dict["Models.Vehicle", "Models.Emergency"]]:

        print('Borough', borough, 'at', SimulatorBasics.secondsToTimestring(simulator.now()))
        print(len(simulator.getAvaliableVehicles(severity, borough)), 'at', len(params.candidates_borough[borough]))

        # Initialize return list
        # This list will hold the final optimal positions of the ambulances
        final_positions: List[str] = []

        # Initialize the return dict
        # This will hold the vehicle repositioning values
        final_repositioning: Dict["Models.Vehicle", str] = {}

        # Initialize dispatching dict
        final_dispatching: Dict["Models.Vehicle", "Models.Emergency"] = self.dispatch(simulator, params, severity, borough)

        # Parameters
        t = simulator.timePeriod()
        D_rates = params.demand_rates
        reachable_inverse = params.reachable_inverse[t]

        # Sets
        C = params.candidates_borough[borough]
        D = params.demand_borough[borough]
        Vehicles = simulator.getAvaliableVehicles(severity, borough)
        U = {v.pos: v for v in Vehicles if v not in final_dispatching.keys()}
        U_nodes = list(U.keys())
        U_to_nodes = [U[u_node].to_node for u_node in U_nodes if U[u_node] not in final_dispatching.keys()]
        U_stations = [U[u_node].station for u_node in U_nodes if U[u_node] not in final_dispatching.keys()]

        max_overload = params.maximum_overload_ALS if severity == 0 else params.maximum_overload_BLS

        # First stage
        print('OPTIMIZING {} FOR BOROUGH {}'.format("ALS" if severity == 0 else "BLS", borough))
        start_time = time.time()

        # Create the mip solver with the CBC backend.
        model = grb.Model(name="ALS" if severity == 0 else "BLS")

        # Declare model variables
        # x_ji: if node i is covered by node j
        y = [model.addVar(vtype=grb.GRB.BINARY, name='y_' + node) for node in D]
        # y_j: if ambulance is located at j at the end
        x = [model.addVar(vtype=grb.GRB.BINARY, name='x_' + node) for j, node in enumerate(C)]
        # r_uj: if ambulance moves from u to j
        r = [[model.addVar(vtype=grb.GRB.BINARY, name='r_' + u + '_' + k) for k in C] for u in U_nodes]

        # Coefficients
        # ------------
        # Computing alpha value (The amount of time an ambulance can spend relocating)
        if not initial:
            weights = np.array(simulator.city_graph.es['length']) / simulator.parameters.getSpeedList(t)
            travel_times_UC = np.array(simulator.city_graph.shortest_paths(U_to_nodes, C, weights))
            alpha = [max(U[v].total_busy_time + min(travel_times_UC[u]), max_overload * (simulator.now() - U[v].arrive_in_system)) for u, v in enumerate(U)]
        else:
            weights = np.array(simulator.city_graph.es['length']) / simulator.parameters.getSpeedList(t)
            alpha = [24 * 36000 for v in Vehicles]  # Set it to the possible maximum value so the restriction is relaxed
            travel_times_UC = np.array(simulator.city_graph.shortest_paths(U_to_nodes, C, weights))

        # Objective function
        model.setObjective(grb.LinExpr(D_rates[severity].loc[t + 1, D].values, y))
        model.ModelSense = grb.GRB.MAXIMIZE

        # Constraints
        # ------------
        # Constraint 1
        {j: model.addConstr(lhs=x[j],
                            sense=grb.GRB.EQUAL,
                            rhs=grb.quicksum(r[u][j] for u, u_node in enumerate(U_nodes)),
                            name='Const_1_{}'.format(j))
         for j, c_node in enumerate(C)}

        # Constraint 2
        {i: model.addConstr(lhs=y[i],
                            sense=grb.GRB.LESS_EQUAL,
                            rhs=grb.quicksum(x[C.index(c_node)] for j, c_node in enumerate(reachable_inverse[d_node]) if c_node in C),
                            name='Const_2_{}'.format(i))
         for i, d_node in enumerate(D)}

        if initial:
            {u: model.addConstr(lhs=grb.quicksum(r[u][k] for k, k_node in enumerate(C)),
                                sense=grb.GRB.EQUAL,
                                rhs=1,
                                name='Const_3_{}'.format(u))
             for u, u_node in enumerate(U_nodes)}

        else:
            {u: model.addConstr(lhs=grb.quicksum(r[u][k] for k, k_node in enumerate(C)),
                                sense=grb.GRB.LESS_EQUAL,
                                rhs=1,
                                name='Const_3_{}'.format(u))
             for u, u_node in enumerate(U_nodes)}

        for u, u_node in enumerate(U_nodes):
            if U[u_node].relocating:
                model.addConstr(lhs=r[u][C.index(U[u_node].station)],
                                sense=grb.GRB.EQUAL,
                                rhs=1,
                                name='Const_4_{}'.format(u))
        if not initial:
            for v, vehicle in enumerate(Vehicles):
                if vehicle not in U.values():
                    model.addConstr(lhs=grb.quicksum(r[u][C.index(vehicle.station)] for u in range(len(U))),
                                    sense=grb.GRB.EQUAL,
                                    rhs=0,
                                    name='Const_X_{}'.format(vehicle.name))

        {u: model.addConstr(lhs=U[u_node].total_busy_time + grb.quicksum(travel_times_UC[u, j] * r[u][j] * (0 if U[u_node].relocating and c_node == U[u_node].station else 1) for j, c_node in enumerate(C)),
                            sense=grb.GRB.LESS_EQUAL,
                            rhs=alpha[u],
                            name='Const_5_{}'.format(u))
         for u, u_node in enumerate(U_nodes)}

        model.setParam('MIPGap', params.optimization_gap)
        model.setParam('LogToConsole', 1 if simulator.verbose else 0)

        print(time.time() - start_time, '- Solving the model...')
        model.optimize()

        if model.Status != grb.GRB.OPTIMAL:
            model.computeIIS()
            model.write("Model Errors/{}.ilp".format(params.name))
            with open('Error Statistics/{}.pickle'.format(params.name), 'wb') as f:
                pickle.dump(simulator.getStatistics(), f)

        # ----------------
        # Second part
        # ----------------
        Z_actual = np.sum([D_rates[severity].loc[t + 1, node] if sum(1 if u in reachable_inverse[node] else 0 for u in U_stations) else 0 for i, node in enumerate(D)])
        Z = np.sum(D_rates[severity].loc[t + 1, D].values * np.array([y[i].x for i in range(len(D))]))

        if Z_actual > 0 and (Z - Z_actual) / Z_actual >= .15 or initial:
            # Create the mip solver with the CBC backend.
            model = grb.Model(name="ALS" if severity == 0 else "BLS")

            # Declare model variables
            # x_ji: if node i is covered by node j
            y = [model.addVar(vtype=grb.GRB.BINARY, name='x_' + node) for node in D]
            # y_j: if ambulance is located at j at the end
            x = [model.addVar(vtype=grb.GRB.BINARY, name='y_' + node) for j, node in enumerate(C)]
            # r_uj: if ambulance moves from u to j
            r = [[model.addVar(vtype=grb.GRB.BINARY, name='r_' + u + '_' + k) for k in C] for u in U_nodes]

            # Objective function
            model.setObjective(grb.quicksum(grb.LinExpr(travel_times_UC[u], r[u]) for u in range(len(U_nodes))))
            model.ModelSense = grb.GRB.MINIMIZE

            # Constraints
            # ------------
            # Constraint 1
            {j: model.addConstr(lhs=x[j],
                                sense=grb.GRB.EQUAL,
                                rhs=grb.quicksum(r[u][j] for u, u_node in enumerate(U_nodes)),
                                name='Const_1_{}'.format(j))
             for j, c_node in enumerate(C)}

            # Constraint 2
            {i: model.addConstr(lhs=y[i],
                                sense=grb.GRB.LESS_EQUAL,
                                rhs=grb.quicksum(x[C.index(c_node)] for j, c_node in enumerate(reachable_inverse[d_node]) if c_node in C),
                                name='Const_2_{}'.format(i))
             for i, d_node in enumerate(D)}

            if initial:
                {u: model.addConstr(lhs=grb.quicksum(r[u][k] for k, k_node in enumerate(C)),
                                    sense=grb.GRB.EQUAL,
                                    rhs=1,
                                    name='Const_3_{}'.format(u))
                 for u, u_node in enumerate(U_nodes)}

            else:
                {u: model.addConstr(lhs=grb.quicksum(r[u][k] for k, k_node in enumerate(C)),
                                    sense=grb.GRB.LESS_EQUAL,
                                    rhs=1,
                                    name='Const_3_{}'.format(u))
                 for u, u_node in enumerate(U_nodes)}

            for u, u_node in enumerate(U_nodes):
                if U[u_node].relocating:
                    model.addConstr(lhs=r[u][C.index(U[u_node].station)],
                                    sense=grb.GRB.EQUAL,
                                    rhs=1,
                                    name='Const_4_{}'.format(u))
            if not initial:
                for v, vehicle in enumerate(Vehicles):
                    if vehicle not in U.values():
                        model.addConstr(lhs=grb.quicksum(r[u][C.index(vehicle.station)] for u in range(len(U))),
                                        sense=grb.GRB.EQUAL,
                                        rhs=0,
                                        name='Const_X_{}'.format(vehicle.name))

            {u: model.addConstr(lhs=U[u_node].total_busy_time + grb.quicksum(travel_times_UC[u, j] * r[u][j] * (0 if U[u_node].relocating and c_node == U[u_node].station else 1) for j, c_node in enumerate(C)),
                                sense=grb.GRB.LESS_EQUAL,
                                rhs=alpha[u],
                                name='Const_5_{}'.format(u))
             for u, u_node in enumerate(U_nodes)}

            model.addConstr(lhs=Z,
                            sense=grb.GRB.LESS_EQUAL,
                            rhs=grb.LinExpr(D_rates[severity].loc[t + 1, D].values.tolist(), y),
                            name='Const_6')

            model.setParam('MIPGap', params.optimization_gap)
            model.setParam('LogToConsole', 1 if simulator.verbose else 0)

            print(time.time() - start_time, '- Solving the model...')
            model.optimize()

            print(time.time() - start_time, '- Done!')
            simulator.statistics['OptimizationSize{}{}'.format('ALS' if severity == 0 else 'BLS', borough)].record(simulator.now(), len(U))
            simulator.statistics['OptimizationTime{}{}'.format('ALS' if severity == 0 else 'BLS', borough)].record(simulator.now(), time.time() - start_time)

            if model.Status != grb.GRB.OPTIMAL:
                model.computeIIS()
                model.write("Model Errors/{}.ilp".format(params.name))
                with open('Error Statistics/{}.pickle'.format(params.name), 'wb') as f:
                    pickle.dump(simulator.getStatistics(), f)

            final_positions = []
            for j in range(len(C)):
                if np.round(x[j].x) == 1:
                    final_positions.append(C[j])
            reposition_matrix = [[np.round(r[u][C.index(k)].x) for k in final_positions] for u in range(len(U_nodes))]
            for u, node in enumerate(U_nodes):
                if 1 in reposition_matrix[u] and final_positions[reposition_matrix[u].index(1)] != U[node].station:
                    final_repositioning[U[node]] = final_positions[reposition_matrix[u].index(1)]

            return final_positions, final_repositioning, final_dispatching
        else:
            return U_stations, {}, final_dispatching

    def redeploy(self,
                 simulator: "Models.EMSModel",
                 params: "Models.SimulationParameters",
                 vehicle: "Models.Vehicle"):
        severity = vehicle.type
        borough = vehicle.borough
        print('REDEPLOYING {} FOR BOROUGH {}'.format(vehicle.name, borough))
        start_time = time.time()

        # Initialize return list
        # This list will hold the final optimal positions of the ambulances
        final_positions: List[str] = []

        # Initialize the return dict
        # This will hold the vehicle repositioning values
        final_repositioning: Dict["Models.Vehicle", str] = {}

        # Parameters
        t = simulator.timePeriod()
        D_rates = params.demand_rates
        reachable_inverse = params.reachable_inverse[t]

        # Sets
        C = params.candidates_borough[borough]
        D = params.demand_borough[borough]
        Vehicles = simulator.getAvaliableVehicles(severity, borough)
        U = {v.station: v for v in Vehicles if v != vehicle}
        U_nodes = list(U.keys())

        # First stage
        print('OPTIMIZING {} FOR BOROUGH {}'.format("ALS" if severity == 0 else "BLS", borough))
        start_time = time.time()

        # Create the mip solver with the CBC backend.
        model = grb.Model(name="ALS" if severity == 0 else "BLS")

        # Declare model variables
        # x_ji: if node i is covered by node j
        y = [model.addVar(vtype=grb.GRB.BINARY, name='y_' + node) for node in D]
        # y_j: if ambulance is located at j at the end
        x = [model.addVar(vtype=grb.GRB.BINARY, name='x_' + node) for j, node in enumerate(C)]
        # r_uj: if ambulance moves from u to j
        r = [model.addVar(vtype=grb.GRB.BINARY, name='r_' + k) for k in C]

        # Coefficients
        # ------------
        weights = np.array(simulator.city_graph.es['length']) / simulator.parameters.getSpeedList(t)
        travel_times_UC = np.array(simulator.city_graph.shortest_paths(vehicle.to_node, C, weights)).reshape(-1)

        # Objective function
        model.setObjective(grb.LinExpr(D_rates[severity].loc[t + 1, D].values, y))
        model.ModelSense = grb.GRB.MAXIMIZE

        # Constraints
        # ------------
        # Constraint 1
        {j: model.addConstr(lhs=x[j],
                            sense=grb.GRB.EQUAL,
                            rhs=(1 if c_node in U_nodes else 0) + r[C.index(c_node)],
                            name='Const_1_{}'.format(j))
         for j, c_node in enumerate(C)}

        # Constraint 2
        {i: model.addConstr(lhs=y[i],
                            sense=grb.GRB.LESS_EQUAL,
                            rhs=grb.quicksum(x[C.index(c_node)] for j, c_node in enumerate(reachable_inverse[d_node]) if c_node in C),
                            name='Const_2_{}'.format(i))
         for i, d_node in enumerate(D)}

        # Constraint 7
        # \sum_{j} r_uj\tau_uj \leq \Phi_u\vartheta
        model.addConstr(lhs=grb.LinExpr(travel_times_UC, r),
                        sense=grb.GRB.LESS_EQUAL,
                        rhs=max(min([travel_times_UC[t] for t in range(len(travel_times_UC)) if C[t] not in U_nodes]), params.max_redeployment_time),
                        name='Const_7')

        model.addConstr(lhs=grb.quicksum(r),
                        sense=grb.GRB.EQUAL,
                        rhs=1,
                        name='Const_4')

        model.setParam('MIPGap', params.optimization_gap)
        model.setParam('LogToConsole', 1 if simulator.verbose else 0)

        print(time.time() - start_time, '- Solving the model...')
        model.optimize()

        print(time.time() - start_time, '- Done!')
        simulator.statistics['OptimizationSize{}{}'.format('ALS' if severity == 0 else 'BLS', borough)].record(simulator.now(), len(U))
        simulator.statistics['OptimizationTime{}{}'.format('ALS' if severity == 0 else 'BLS', borough)].record(simulator.now(), time.time() - start_time)

        if model.Status != grb.GRB.OPTIMAL:
            model.computeIIS()
            model.write("Model Errors/{}.ilp".format(params.name))
            with open('Error Statistics/{}.pickle'.format(params.name), 'wb') as f:
                pickle.dump(simulator.getStatistics(), f)

        final_positions = []
        for j in range(len(C)):
            if x[j].x == 1:
                final_positions.append(C[j])
        final_repositioning[vehicle] = final_positions[([r[C.index(j)].x for j in final_positions]).index(1)]

        return final_positions, final_repositioning

    def dispatch(self,
                 simulator: "Models.EMSModel",
                 params: "Models.SimulationParameters",
                 severity: int = 0,
                 borough=None):
        # Manual nearest dispatcher
        dispatching_dict = {}
        weights = np.array(simulator.city_graph.es['length']) / simulator.parameters.getSpeedList(simulator.timePeriod())
        vehicles = simulator.getAvaliableVehicles(v_type=severity, borough=borough)

        vehicle_positions = [ambulance.to_node
                             for ambulance in vehicles]

        used_vehicles: List[Tuple[int, int, int]] = []
        emergencies = simulator.getUnassignedEmergencies(severity + 1, borough)
        if severity == 1:
            emergencies += simulator.getUnassignedEmergencies(3, borough)
        emergencies.sort(key=lambda e: e.arrival_time)

        for e in emergencies:

            travel_times = np.array(simulator.city_graph.shortest_paths(vehicle_positions, e.node, weights))
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
                    dispatching_dict[vehicles[c[0]]] = e
                    break
        return dispatching_dict
