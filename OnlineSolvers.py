import time
import numpy as np
import gurobipy as grb
from typing import Dict, List, Tuple

# Internal imports
import Models
import SimulatorBasics


class RelocationModel:

    def __init__(self):
        pass

    def optimize(self,
                 simulator: "Models.EMSModel",
                 params: "Models.SimulationParameters",
                 initial: bool,
                 severity: int = 0,
                 borough=None) -> Tuple[List[str], Dict["Models.Vehicle", str], Dict["Models.Vehicle", "Models.Emergency"]]:
        print('Warning! Relocation not implemented')
        return [], {}, {}

    def redeploy(self,
                 simulator: "Models.EMSModel",
                 params: "Models.SimulationParameters",
                 vehicle: "Models.Vehicle"):
        return [], {}

    def dispatch(self,
                 simulator: "Models.EMSModel",
                 params: "Models.SimulationParameters",
                 severity: int = 0,
                 borough=None):
        return {}


class SORDARelocatorDispatcher:

    def __init__(self):
        super().__init__()

    def SurvivalFunction(self, response_times):
        """
        Returns the survival probability of an emergency as a function of
        response time.
        """
        return (1 + np.exp(0.679 + .262 * response_times)) ** -1

    def optimize(self,
                 simulator: "Models.EMSModel",
                 params: "Models.SimulationParameters",
                 initial: bool,
                 severity: int = 0,
                 borough=None) -> Tuple[List[str], Dict["Models.Vehicle", str], Dict["Models.Vehicle", "Models.Emergency"]]:
        """
        Main optimization process.
        This function is called whenever an emergency arrives to the system or an ambulance becomes idle and emergencies are waiting.
        The optimization model takes as input parameters variables of the simulation to decide which ambulance serves the ongoing
        emergencies and where ambulances should be relocated, considering real time information about ambulance workload.

        Most values are accessed directly from the simulator and simulationParameters object, we only need to specify if the optimization
        process is for ALS or BLS and the borough we are working on.
        """

        print('Borough', borough, 'at', SimulatorBasics.secondsToTimestring(simulator.now()))
        print(len(simulator.getAvaliableVehicles(severity, borough)), 'at', len(params.candidates_borough[borough]))

        # Initialize return list
        # This list will hold the final optimal positions of the ambulances
        final_positions: List[str] = []

        # Initialize the return dict
        # This will hold the vehicle repositioning values
        final_repositioning: Dict["Models.Vehicle", str] = {}

        # Initialize dispatching dict
        final_dispatching: Dict["Models.Vehicle", "Models.Emergency"] = {}

        # Parameters
        t = simulator.timePeriod()
        D_rates = params.demand_rates
        reachable_inverse = params.reachable_inverse[t]
        reachable_demand = params.reachable_demand[t]
        cand_demand_travel = params.cand_demand_time[t]

        # More parameters
        if severity == 1:
            average_RHS = 15 * 60
            uncovered_penalty = params.uncovered_penalty
            late_response_penalty = params.late_response_penalty
            dispatching_penalty = params.dispatching_penalty
            RHS_limit = max(params.uber_seconds * simulator.now() / params.simulation_time, simulator.uber_seconds)

        # Sets
        C = params.candidates_borough[borough]
        D = params.demand_borough[borough]
        E = simulator.getUnassignedEmergencies(severity + 1, borough)
        if severity == 1:
            E += simulator.getUnassignedEmergencies(3, borough)
        E_pos = [emergency.node for emergency in E]
        Vehicles = simulator.getAvaliableVehicles(severity, borough)
        U = {v.pos: v for v in Vehicles}
        U_nodes = list(U.keys())
        U_to_nodes = [U[u_node].to_node for u_node in U_nodes]

        # More parameters
        max_overload = params.maximum_overload_ALS if severity == 0 else params.maximum_overload_BLS
        target_relocation = params.target_relocation_time
        max_relocations = max(params.max_expected_simultaneous_relocations, np.sum([1 if v.relocating else 0 for v in Vehicles]))
        max_relocation_time = params.max_relocation_time
        tt_penalty = params.travel_distance_penalty

        # First stage
        print('OPTIMIZING {} FOR BOROUGH {}'.format("ALS" if severity == 0 else "BLS", borough))
        start_time = time.time()

        # Coefficients for the objective function
        # ------------
        # Computing alpha value (The amount of time an ambulance can spend relocating)
        if not initial:
            alpha = [max(U[v].total_busy_time, max_overload * (simulator.now() - U[v].arrive_in_system)) for v in U]
            phi = [np.exp(-2.5 * (U[u_node].accumulated_relocation / target_relocation)) for u_node in U_nodes]
            weights = np.array(simulator.city_graph.es['length']) / simulator.parameters.getSpeedList(t)
            travel_times_UC = np.array(simulator.city_graph.shortest_paths(U_to_nodes, C, weights))
            travel_times_CE = np.array([np.squeeze(simulator.city_graph.shortest_paths(U_to_nodes, e, weights)).reshape(len(U_to_nodes), ) for e in E_pos]).T
            closest_dispatch_BLS = (np.sum(D_rates[1][D].values, axis=1) @ np.array([[1, 0, 0], [1 / 3, 2 / 3, 0], [0, 1, 0], [0, 0, 1], [0, 0, 1]]))[int(simulator.now() / (8 * 3600)) % 3] / 8 < 1.1
        else:
            alpha = [24 * 3600 for v in Vehicles]  # Set it to the possible maximum value so the restriction is relaxed
            phi = [np.exp(-2.5 * (U[u_node].accumulated_relocation / target_relocation)) for u_node in U_nodes]
            travel_times_UC = np.zeros((len(U_nodes), len(C)))
            travel_times_CE = np.zeros((len(U_nodes), len(E_pos)))
            max_relocations = len(U)

        if severity == 0:
            survival_matrix = self.SurvivalFunction(cand_demand_travel / 60)  # Travel time in minutes
            survival_dispatching = self.SurvivalFunction(travel_times_CE / 60)  # Travel time in minutes
        else:
            b_bar = np.mean([U[u_node].total_busy_time for u_node in U_nodes]) if len(U) > 0 else 1e10
            rho = [[U[u_node].total_busy_time + travel_times_CE[u][e] + 3600 * params.mean_busytime[severity].at[t + 1, params.graph_to_demand[e_node]] - b_bar if not closest_dispatch_BLS
                    else travel_times_CE[u][e] for e, e_node in enumerate(E_pos)] for u, u_node in enumerate(U_nodes)]

        # Filter the survival matrix leaving only the nodes that are reachable,
        # 0 to all the rest
        if severity == 0:
            filtered_survival = np.array([[survival_matrix[j, i] if c_node in params.reachable_inverse[t][d_node] else 0
                                           for j, c_node in enumerate(C)]
                                          for i, d_node in enumerate(D)]).T
            coefficients = D_rates[severity].loc[t + 1, D].values * filtered_survival
        else:
            filtered_survival = np.array([[1 / (cand_demand_travel[j, i] + 1) if c_node in params.reachable_inverse[t][d_node] else 0
                                           for j, c_node in enumerate(C)]
                                          for i, d_node in enumerate(D)]).T
            coefficients = D_rates[severity].loc[t + 1, D].values * filtered_survival

        # Create the mip solver with the CBC backend.
        model = grb.Model(name="ALS" if severity == 0 else "BLS")

        # Declare model variables
        if severity == 0:
            # x_ji: if node i is covered by node j
            x = [[model.addVar(vtype=grb.GRB.CONTINUOUS, name='x_' + node + '_' + c_node) for node in D] for c_node in C]
            # y_ji:
            y = [[model.addVar(vtype=grb.GRB.BINARY, name='y_' + node + '_' + c_node) for node in D] for c_node in C]
        else:
            # x_jiu: if node i is covered by node j by ambulance u
            x = [[model.addVar(vtype=grb.GRB.BINARY, name='x_' + node + '_' + c_node) for node in D] for c_node in C]
            # z_e: if RHS is dispatched to emergency e
            z = [model.addVar(vtype=grb.GRB.BINARY, name='z_' + e_node) for e, e_node in enumerate(E_pos)]
            # b_e: if emergency e is left uncovered
            b = [model.addVar(vtype=grb.GRB.BINARY, name='b_' + e_node) for e, e_node in enumerate(E_pos)]
            # w_e: late response emergency e
            w = [model.addVar(vtype=grb.GRB.CONTINUOUS, name='w_' + e_node, lb=0) for e, e_node in enumerate(E_pos)]
        # r_uj: if ambulance moves from u to j
        r = [[model.addVar(vtype=grb.GRB.BINARY, name='r_' + u + '_' + j) for j in C] for u in U_nodes]
        # k_ue: if ambulance at u is dispatched to emergency e
        k = [[model.addVar(vtype=grb.GRB.BINARY, name='delta_' + u + '_' + e) for e in E_pos] for u in U_nodes]
        # m_u: if ambulance at u stays in place
        m = [model.addVar(vtype=grb.GRB.BINARY, name='m' + u_node) for u, u_node in enumerate(U_nodes)]

        if severity == 0:
            # Objective function
            model.setObjective(grb.quicksum(coefficients[j, D.index(d_node)] * x[j][D.index(d_node)]
                                            for j, c_node in enumerate(C)
                                            for i, d_node in enumerate(reachable_demand[c_node]) if d_node in D) +                             # noqa: W504
                               np.sum(D_rates[0].values) * grb.quicksum(grb.quicksum(survival_dispatching[u, e] * k[u][e]
                                                                                     for e, e_node in enumerate(E_pos))
                               for u, u_node in enumerate(U_nodes)) - tt_penalty * grb.quicksum(grb.LinExpr(travel_times_UC[u], r[u]) for u in range(len(U_nodes))))
            model.ModelSense = grb.GRB.MAXIMIZE
        else:
            # Objective function
            model.setObjective(grb.quicksum(grb.quicksum(coefficients[C.index(c_node), i] * x[C.index(c_node)][i] if c_node in C else 0
                                            for c_node in reachable_inverse[d_node])
                               for i, d_node in enumerate(D)) - grb.quicksum(uncovered_penalty * b[e] + late_response_penalty * w[e] + dispatching_penalty * grb.quicksum(rho[u][e] * k[u][e] for u, u_node in enumerate(U_nodes)) for e, e_node in enumerate(E_pos))
                               - tt_penalty * grb.quicksum(grb.LinExpr(travel_times_UC[u], r[u]) for u in range(len(U_nodes))))     # noqa: W503
            model.ModelSense = grb.GRB.MAXIMIZE

        # Constraints
        # ------------

        if severity == 0:
            # Constraint
            {(i, j): model.addConstr(lhs=x[j][D.index(d_node)],
                                     sense=grb.GRB.LESS_EQUAL,
                                     rhs=grb.quicksum(r[u][C.index(c_node)] * phi[u] for u in range(len(U_nodes))),
                                     name='Const_1_{}_{}'.format(i, j))
             for j, c_node in enumerate(C)
             for i, d_node in enumerate(reachable_demand[c_node]) if d_node in D}

            # Constraint
            {(i, j): model.addConstr(lhs=x[j][D.index(d_node)],
                                     sense=grb.GRB.LESS_EQUAL,
                                     rhs=y[j][D.index(d_node)],
                                     name='Const_2_{}_{}'.format(i, j))
             for j, c_node in enumerate(C)
             for i, d_node in enumerate(reachable_demand[c_node]) if d_node in D}

            # Constraint
            {i: model.addConstr(lhs=grb.quicksum(y[C.index(c_node)][D.index(d_node)] for c_node in reachable_inverse[d_node] if c_node in C),
                                sense=grb.GRB.LESS_EQUAL,
                                rhs=1,
                                name='Const_3_{}'.format(i))
             for i, d_node in enumerate(D)}

            # Constraint 2
            {j: model.addConstr(lhs=grb.quicksum(r[u][j] for u, u_node in enumerate(U_nodes)),
                                sense=grb.GRB.LESS_EQUAL,
                                rhs=1,
                                name='Const_4_{}'.format(j))
             for j, j_node in enumerate(C)}
        else:
            # Constraint 1
            # x_{ij} \leq y_j \forall i, j \in N_i
            {(i, j): model.addConstr(lhs=x[C.index(c_node)][i],
                                     sense=grb.GRB.LESS_EQUAL,
                                     rhs=grb.quicksum(r[u][C.index(c_node)] for u, u_node in enumerate(U_nodes)),
                                     name='Const_1_{}_{}'.format(i, j))
             for i, d_node in enumerate(D)
             for j, c_node in enumerate(params.reachable_inverse[t][d_node]) if c_node in C}

            # Constraint 2
            # sum_{j \in Lambda_i} x_{ij} + h_i = 1
            {i: model.addConstr(lhs=grb.quicksum(x[C.index(j)][i] if j in C else 0 for j in params.reachable_inverse[t][d_node]),
                                sense=grb.GRB.LESS_EQUAL,
                                rhs=1,
                                name='Const_2_{}'.format(i))
             for i, d_node in enumerate(D)}

        # Constraint 3.5
        for u, u_node in enumerate(U_nodes):
            if U[u_node].relocating:
                model.addConstr(lhs=grb.quicksum(r[u][C.index(j_star)] for j_star in C if j_star != U[u_node].station),
                                sense=grb.GRB.EQUAL,
                                rhs=0,
                                name='Const_3.5_{}'.format(u_node))

        # Constraint 4
        # sum_{j} r_uj + sum_{e} delta_ue = 1
        {u: model.addConstr(lhs=grb.quicksum(r[u]) + grb.quicksum(k[u]) + m[u],
                            sense=grb.GRB.EQUAL,
                            rhs=1,
                            name='Const_4_{}'.format(u_node))
         for u, u_node in enumerate(U_nodes)}

        # Constraint 6
        # beta_u + sum_{j'} t_uj' * r_uj' \leq alpha
        {u: model.addConstr(lhs=U[u_node].total_busy_time + grb.LinExpr(travel_times_UC[u], r[u]),
                            sense=grb.GRB.LESS_EQUAL,
                            rhs=alpha[u],
                            name='Const_6_{}'.format(u))
         for u, u_node in enumerate(U_nodes)}

        if severity == 1:
            # Constraint 7
            # \sum_{j} r_uj\tau_uj \leq \Phi_u\vartheta
            {u: model.addConstr(lhs=grb.quicksum(travel_times_UC[u][j] * r[u][j] * (0 if U[u_node].relocating and U[u_node].station == c_node else 1) for j, c_node in enumerate(C)),
                                sense=grb.GRB.LESS_EQUAL,
                                rhs=max_relocation_time * (0 if U[u_node].relocating else 1) * phi[u],
                                name='Const_7_{}'.format(u))
             for u, u_node in enumerate(U_nodes)}

            # Constraint 8
            # \sum_{u,j} r_uj \leq \theta
            model.addConstr(lhs=grb.quicksum(grb.quicksum(r[u][j] for j, c_node in enumerate(C) if c_node != u_node) for u, u_node in enumerate(U_nodes)),
                            sense=grb.GRB.LESS_EQUAL,
                            rhs=max_relocations,
                            name='max_relocations')

        if severity == 0:
            # Constraint 5
            # sum_{u} k_ue = 1
            {e: model.addConstr(lhs=grb.quicksum(k[u][e] for u, u_node in enumerate(U)),
                                sense=grb.GRB.EQUAL,
                                rhs=1,
                                name='Const_5_{}'.format(e_node))
             for e, e_node in enumerate(E_pos)}
        else:
            # Constraint 9
            # \sum_{u} k_ue + b_e = 1
            {e: model.addConstr(lhs=grb.quicksum(k[u][e] for u, u_node in enumerate(U_nodes)) + b[e],
                                sense=grb.GRB.EQUAL,
                                rhs=1,
                                name='Const_9_{}'.format(e))
             for e, emergency in enumerate(E) if emergency.severity == 2}

            # Constraint 9.5
            {e: model.addConstr(lhs=z[e],
                                sense=grb.GRB.EQUAL,
                                rhs=0,
                                name='Const_9.5_{}'.format(e))
             for e, emergency in enumerate(E) if emergency.severity == 2}

            # Constraint 10
            # \sum_{u} k_ue + b_e + z_e= 1
            {e: model.addConstr(lhs=grb.quicksum(k[u][e] for u, u_node in enumerate(U_nodes)) + b[e] + z[e],
                                sense=grb.GRB.EQUAL,
                                rhs=1,
                                name='Const_9_{}'.format(e))
             for e, emergency in enumerate(E) if emergency.severity == 3}

            # Contraint 11
            model.addConstr(lhs=simulator.uber_seconds + grb.LinExpr([average_RHS] * len(z), z),
                            sense=grb.GRB.LESS_EQUAL,
                            rhs=RHS_limit,
                            name='RHSConstraint')

            # Constraint 12
            {e: model.addConstr(lhs=simulator.now() - emergency.arrival_time + grb.quicksum(travel_times_CE[u][e] * k[u][e] for u, u_node in enumerate(U_nodes)),
                                sense=grb.GRB.LESS_EQUAL,
                                rhs=w[e] + 8 * 60 + np.sum(params.cand_cand_time[t]) * b[e],
                                name='Const_12_{}'.format(e))
             for e, emergency in enumerate(E)}

        # Optimizer setup
        model.setParam('MIPGap', params.optimization_gap)
        model.setParam('LogToConsole', 1 if simulator.verbose else 0)

        # Run optimization
        print(time.time() - start_time, '- Solving the model...')
        model.optimize()

        # Collect statistics
        print(time.time() - start_time, '- Done!')
        simulator.statistics['OptimizationSize{}{}'.format('ALS' if severity == 0 else 'BLS', borough)].record(simulator.now(), len(U))
        simulator.statistics['OptimizationTime{}{}'.format('ALS' if severity == 0 else 'BLS', borough)].record(simulator.now(), time.time() - start_time)

        # In case of error, save the output
        if model.Status != grb.GRB.OPTIMAL:
            model.computeIIS()
            model.write("Model Errors/SBRDANewmodel.ilp")

        # Build the final dictionaries
        # This process looks messy but we are only extracting the solution
        # from bianry variables to a dictionary we can work on
        final_positions = []
        for j in range(len(C)):
            if sum(r[u][j].x for u in range(len(U_nodes))) == 1:
                final_positions.append(C[j])

        # Repositioning dictionary
        reposition_matrix = [[r[u][C.index(j)].x for j in final_positions] for u in range(len(U_nodes))]
        for i, node in enumerate(U_nodes):
            if 1 in reposition_matrix[i] and final_positions[reposition_matrix[i].index(1)] != U[node].station:
                final_repositioning[U[node]] = final_positions[reposition_matrix[i].index(1)]

        # Dispatching dictionary
        for e, emergency in enumerate(E):
            dispatch_list = [k[u][e].x for u, u_node in enumerate(U_nodes)]
            if 1 in dispatch_list:
                final_dispatching[U[U_nodes[dispatch_list.index(1)]]] = emergency
            elif z[e].x == 1:
                final_dispatching[simulator.newUberVehicle(simulator.parameters.uber_nodes[emergency.node], emergency.borough)] = emergency

        return final_positions, final_repositioning, final_dispatching
    
    def redeploy(self,
                 simulator: "Models.EMSModel",
                 params: "Models.SimulationParameters",
                 vehicle: "Models.Vehicle"):
        """
        Redeployment optimization process.
        This function is called when an ambulance becomes idle and needs to be redeployed. An optimization
        process uses real-time information to decide where this ambulance should be located.
        """
        severity = vehicle.type
        borough = vehicle.borough
        print('Borough', borough, 'at', SimulatorBasics.secondsToTimestring(simulator.now()))
        print(1, 'at', len(params.candidates_borough[borough]))

        # Initialize return list
        # This list will hold the final optimal positions of the ambulances
        final_positions: List[str] = []

        # Initialize the return dict
        # This will hold the vehicle repositioning values
        final_repositioning: Dict["Models.Vehicle", str] = {}

        # Parameters
        t = simulator.timePeriod()
        D_rates = params.demand_rates
        C = params.candidates_borough[borough]
        reachable_inverse = params.reachable_inverse[t]
        cand_demand_travel = params.cand_demand_time[t]
        weights = np.array(simulator.city_graph.es['length']) / simulator.parameters.getSpeedList(t)
        travel_times_UC = np.array(simulator.city_graph.shortest_paths(vehicle.to_node, C, weights)).reshape(-1)

        if severity == 1:
            inf_travel_time = np.sum(cand_demand_travel[t])

        # Sets
        C = params.candidates_borough[borough]
        D = params.demand_borough[borough]
        Vehicles = simulator.getAvaliableVehicles(severity, borough)
        U = {v.station: v for v in Vehicles if v != vehicle}
        U_nodes = list(U.keys())

        # First stage
        print('REDEPLOYING {} FOR BOROUGH {}'.format(vehicle.name, borough))
        start_time = time.time()

        # Create the mip solver with the CBC backend.
        model = grb.Model(name="ALS Redeployment" if severity == 0 else "BLS Redeployment")

        # Declare model variables
        # x_ji: if node i is covered by node j
        x = [[model.addVar(vtype=grb.GRB.BINARY, name='x_' + node + '_' + c_node) for node in D] for c_node in C]
        # y_j: if ambulance is located at j at the end
        y = [model.addVar(vtype=grb.GRB.BINARY, name='y_' + node) for j, node in enumerate(C)]
        # r_uj: if ambulance moves from u to j
        r = [model.addVar(vtype=grb.GRB.BINARY, name='r_' + j) for j in C]

        # Coefficients
        # ------------
        if severity == 0:
            survival_matrix = self.SurvivalFunction(cand_demand_travel / 60)  # Travel time in minutes

        # Filter the survival matrix leaving only the nodes that are reachable,
        # 0 to all the rest
        if severity == 0:
            filtered_survival = np.array([[survival_matrix[j, i] if c_node in params.reachable_inverse[t][d_node] else 0
                                           for j, c_node in enumerate(C)]
                                          for i, d_node in enumerate(D)]).T
        else:
            filtered_survival = np.array([[1 / (cand_demand_travel[j, i] + 1) if c_node in params.reachable_inverse[t][d_node] else inf_travel_time
                                           for j, c_node in enumerate(C)]
                                          for i, d_node in enumerate(D)]).T

        # [j, i] matrix respresenting psi_{i,j} * d_i
        coefficients = D_rates[severity].loc[t + 1, D].values * filtered_survival

        if severity == 0:
            # Objective function
            model.setObjective(grb.quicksum(grb.quicksum(coefficients[C.index(c_node), i] * x[C.index(c_node)][i] if c_node in C else 0
                                            for c_node in reachable_inverse[d_node])
                               for i, d_node in enumerate(D)))
            model.ModelSense = grb.GRB.MAXIMIZE
        else:
            # Objective function
            model.setObjective(grb.quicksum(grb.quicksum(coefficients[C.index(c_node), i] * x[C.index(c_node)][i] if c_node in C else 0
                                            for c_node in reachable_inverse[d_node])
                               for i, d_node in enumerate(D)))
            model.ModelSense = grb.GRB.MAXIMIZE

        # Constraint 1
        # x_{ij} \leq y_j \forall i, j \in N_i
        {(i, j): model.addConstr(lhs=x[C.index(c_node)][i],
                                 sense=grb.GRB.LESS_EQUAL,
                                 rhs=y[C.index(c_node)],
                                 name='Const_1_{}_{}'.format(i, j))
         for i, d_node in enumerate(D)
         for j, c_node in enumerate(params.reachable_inverse[t][d_node]) if c_node in C}

        # Constraint 2
        # sum_{j \in Lambda_i} x_{ij} \leq 1
        {i: model.addConstr(lhs=grb.quicksum(x[C.index(j)][i] if j in C else 0 for j in params.reachable_inverse[t][d_node]),
                            sense=grb.GRB.LESS_EQUAL,
                            rhs=1,
                            name='Const_2_{}'.format(i))
            for i, d_node in enumerate(D)}

        {j: model.addConstr(lhs=y[j],
                            sense=grb.GRB.EQUAL,
                            rhs=(1 if c_node in U_nodes else 0) + r[C.index(c_node)],
                            name='Const_3_{}'.format(j))
            for j, c_node in enumerate(C)}

        # Constraint 7
        # \sum_{j} r_uj\tau_uj \leq \Phi_u\vartheta
        {model.addConstr(lhs=grb.LinExpr(travel_times_UC, r),
                         sense=grb.GRB.LESS_EQUAL,
                         rhs=max(min([travel_times_UC[t] for t in range(len(travel_times_UC)) if C[t] not in U_nodes]), params.max_redeployment_time),
                         name='Const_7')}

        model.addConstr(lhs=grb.quicksum(r),
                        sense=grb.GRB.EQUAL,
                        rhs=1,
                        name='Const_4')

        model.setParam('MIPGap', params.optimization_gap)
        model.setParam('LogToConsole', 1 if simulator.verbose else 0)

        print(time.time() - start_time, '- Solving the model...')
        model.optimize()

        print(time.time() - start_time, '- Done!')

        final_positions = []
        for j in range(len(C)):
            if y[j].x == 1:
                final_positions.append(C[j])
        final_repositioning[vehicle] = final_positions[([r[C.index(j)].x for j in final_positions]).index(1)]

        return final_positions, final_repositioning

    def dispatch(self,
                 simulator: "Models.EMSModel",
                 params: "Models.SimulationParameters",
                 severity: int = 0,
                 borough=None):
        """
        Dispatching process.
        Function to be called when we need to dispatch an ambulance without relocating all the others.
        For high severity emergencies, we always dispatch the nearest ambulance, while for low severity ones,
        we run an optimization model to decide.
        """
        if severity == 0:
            # Manual nearest dispatcher
            dispatching_dict = {}
            weights = np.array(simulator.city_graph.es['length']) / simulator.parameters.getSpeedList(simulator.timePeriod())
            vehicles = simulator.getAvaliableVehicles(v_type=severity, borough=borough)

            vehicle_positions = [ambulance.to_node
                                 for ambulance in vehicles]

            used_vehicles: List[Tuple[int, int, int]] = []
            emergencies = simulator.getUnassignedEmergencies(severity + 1, borough)
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
        else:
            # Initialize dispatching dict
            final_dispatching: Dict["Models.Vehicle", "Models.Emergency"] = {}

            # Parameters
            t = simulator.timePeriod()
            D_rates = params.demand_rates
            D = params.demand_borough[borough]

            average_RHS = 15 * 60
            uncovered_penalty = params.uncovered_penalty
            late_response_penalty = params.late_response_penalty
            dispatching_penalty = params.dispatching_penalty
            RHS_limit = max(params.uber_seconds * simulator.now() / params.simulation_time, simulator.uber_seconds)
            closest_dispatch_BLS = (np.sum(D_rates[1][D].values, axis=1) @ np.array([[1, 0, 0], [1 / 3, 2 / 3, 0], [0, 1, 0], [0, 0, 1], [0, 0, 1]]))[int(simulator.now() / (8 * 3600)) % 3] / 8 < 1.1

            # Sets
            E = simulator.getUnassignedEmergencies(severity + 1, borough)
            if severity == 1:
                E += simulator.getUnassignedEmergencies(3, borough)
            E_pos = [emergency.node for emergency in E]
            Vehicles = simulator.getAvaliableVehicles(severity, borough)
            U = {v.pos: v for v in Vehicles}
            U_nodes = list(U.keys())
            U_to_nodes = [U[u_node].to_node for u_node in U_nodes]

            # First stage
            print('DISPATCHING {} FOR BOROUGH {}'.format("ALS" if severity == 0 else "BLS", borough))
            start_time = time.time()

            # Create the mip solver with the CBC backend.
            model = grb.Model(name="ALS" if severity == 0 else "BLS")

            # Declare model variables
            # k_ue: if ambulance at u is dispatched to emergency e
            k = [[model.addVar(vtype=grb.GRB.BINARY, name='delta_' + u + '_' + e) for e in E_pos] for u in U_nodes]
            # m_u: if ambulance at u stays in place
            m = [model.addVar(vtype=grb.GRB.BINARY, name='m' + u_node) for u, u_node in enumerate(U_nodes)]

            if severity == 1:
                # z_e: if RHS is dispatched to emergency e
                z = [model.addVar(vtype=grb.GRB.BINARY, name='z_' + e_node) for e, e_node in enumerate(E_pos)]
                # b_e: if emergency e is left uncovered
                b = [model.addVar(vtype=grb.GRB.BINARY, name='b_' + e_node) for e, e_node in enumerate(E_pos)]
                # w_e: late response emergency e
                w = [model.addVar(vtype=grb.GRB.CONTINUOUS, name='w_' + e_node, lb=0) for e, e_node in enumerate(E_pos)]

            # Coefficients
            # ------------
            # Computing alpha value (The amount of time an ambulance can spend relocating)
            weights = np.array(simulator.city_graph.es['length']) / simulator.parameters.getSpeedList(t)
            travel_times_CE = np.array([np.squeeze(simulator.city_graph.shortest_paths(U_to_nodes, e, weights)).reshape(len(U_to_nodes), ) for e in E_pos]).T

            b_bar = np.mean([U[u_node].total_busy_time for u_node in U_nodes]) if len(U) > 0 else 1e10
            rho = [[U[u_node].total_busy_time + travel_times_CE[u][e] + 3600 * params.mean_busytime[severity].at[t + 1, params.graph_to_demand[e_node]] - b_bar if not closest_dispatch_BLS
                    else travel_times_CE[u][e] for e, e_node in enumerate(E_pos)] for u, u_node in enumerate(U_nodes)]

            # Objective function
            model.setObjective(grb.quicksum(uncovered_penalty * b[e] + late_response_penalty * w[e] + dispatching_penalty * grb.quicksum(rho[u][e] * k[u][e] for u, u_node in enumerate(U_nodes)) for e, e_node in enumerate(E_pos)))     # noqa: W503
            model.ModelSense = grb.GRB.MINIMIZE

            # Constraints
            # ------------
            # Constraint 4
            # sum_{j} r_uj + sum_{e} delta_ue = 1
            {u: model.addConstr(lhs=grb.quicksum(k[u]) + m[u],
                                sense=grb.GRB.EQUAL,
                                rhs=1,
                                name='Const_4_{}'.format(u_node))
             for u, u_node in enumerate(U_nodes)}

            # Constraint 9
            # \sum_{u} k_ue + b_e = 1
            {e: model.addConstr(lhs=grb.quicksum(k[u][e] for u, u_node in enumerate(U_nodes)) + b[e],
                                sense=grb.GRB.EQUAL,
                                rhs=1,
                                name='Const_9_{}'.format(e))
             for e, emergency in enumerate(E) if emergency.severity == 2}

            # Constraint 9.5
            {e: model.addConstr(lhs=z[e],
                                sense=grb.GRB.EQUAL,
                                rhs=0,
                                name='Const_9.5_{}'.format(e))
             for e, emergency in enumerate(E) if emergency.severity == 2}

            # Constraint 10
            # \sum_{u} k_ue + b_e + z_e= 1
            {e: model.addConstr(lhs=grb.quicksum(k[u][e] for u, u_node in enumerate(U_nodes)) + b[e] + z[e],
                                sense=grb.GRB.EQUAL,
                                rhs=1,
                                name='Const_9_{}'.format(e))
             for e, emergency in enumerate(E) if emergency.severity == 3}

            # Contraint 11
            model.addConstr(lhs=simulator.uber_seconds + grb.LinExpr([average_RHS] * len(z), z),
                            sense=grb.GRB.LESS_EQUAL,
                            rhs=RHS_limit,
                            name='RHSConstraint')

            # Constraint 12
            {e: model.addConstr(lhs=simulator.now() - emergency.arrival_time + grb.quicksum(travel_times_CE[u][e] * k[u][e] for u, u_node in enumerate(U_nodes)),
                                sense=grb.GRB.LESS_EQUAL,
                                rhs=w[e] + 8 * 60 + np.sum(params.cand_cand_time[t]) * b[e],
                                name='Const_12_{}'.format(e))
             for e, emergency in enumerate(E)}

            model.setParam('MIPGap', params.optimization_gap)
            model.setParam('LogToConsole', 1 if simulator.verbose else 0)

            print(time.time() - start_time, '- Solving the model...')
            model.optimize()

            print(time.time() - start_time, '- Done!')
            simulator.statistics['OptimizationSize{}{}'.format('ALS' if severity == 0 else 'BLS', borough)].record(simulator.now(), len(U))
            simulator.statistics['OptimizationTime{}{}'.format('ALS' if severity == 0 else 'BLS', borough)].record(simulator.now(), time.time() - start_time)

            for e, emergency in enumerate(E):
                dispatch_list = [k[u][e].x for u, u_node in enumerate(U_nodes)]
                if 1 in dispatch_list:
                    final_dispatching[U[U_nodes[dispatch_list.index(1)]]] = emergency
                elif z[e].x == 1:
                    final_dispatching[simulator.newUberVehicle(simulator.parameters.uber_nodes[emergency.node], emergency.borough)] = emergency

            return final_dispatching
