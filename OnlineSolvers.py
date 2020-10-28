import time
# import random
import numpy as np
import gurobipy as grb
from typing import Dict, List, Tuple  # , Optional

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


class UberRelocatorDispatcher(RelocationModel):

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
        final_dispatching: Dict["Models.Vehicle", "Models.Emergency"] = {}

        # Parameters
        t = simulator.timePeriod()
        D_rates = params.demand_rates
        reachable_inverse = params.reachable_inverse[t]
        cand_demand_travel = params.cand_demand_time[t]

        if severity == 1:
            average_RHS = 15 * 60
            uncovered_penalty = params.uncovered_penalty
            late_response_penalty = params.late_response_penalty
            dispatching_penalty = params.dispatching_penalty
            RHS_limit = params.uber_seconds * simulator.now() / params.simulation_time
            inf_travel_time = np.sum(cand_demand_travel[t])

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
        U_to_nodes = [v.to_node for v in Vehicles]

        max_overload = params.maximum_overload_ALS if severity == 0 else params.maximum_overload_BLS
        target_relocation = params.target_relocation_time
        max_relocations = max(params.max_expected_simultaneous_relocations, np.sum([1 if v.relocating else 0 for v in Vehicles]))
        max_relocation_time = params.max_relocation_time
        tt_penalty = params.travel_distance_penalty

        # First stage
        print('OPTIMIZING {} FOR BOROUGH {}'.format("ALS" if severity == 0 else "BLS", borough))
        start_time = time.time()

        # Create the mip solver with the CBC backend.
        model = grb.Model(name="ALS" if severity == 0 else "BLS")

        # Declare model variables
        # x_ji: if node i is covered by node j
        x = [[model.addVar(vtype=grb.GRB.BINARY, name='x_' + node + '_' + c_node) for node in D] for c_node in C]
        # y_j: if ambulance is located at j at the end
        y = [model.addVar(vtype=grb.GRB.BINARY, name='y_' + node) for j, node in enumerate(C)]
        # r_uj: if ambulance moves from u to j
        r = [[model.addVar(vtype=grb.GRB.BINARY, name='r_' + u + '_' + j) for j in C] for u in U_nodes]
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
            # h_i: if demand node is uncovered
            h = [model.addVar(vtype=grb.GRB.BINARY, name='h_' + node) for i, node in enumerate(D)]

        # Coefficients
        # ------------
        # Computing alpha value (The amount of time an ambulance can spend relocating)
        if not initial:
            alpha = [max(U[v].total_busy_time, max_overload * (simulator.now() - U[v].arrive_in_system)) for v in U]
            phi = [(1 if U[u_node].can_relocate else 0) * (0 if U[u_node].relocating else 1) * max(0, (target_relocation - U[u_node].accumulated_relocation) / target_relocation) for u_node in U_nodes]
            weights = np.array(simulator.city_graph.es['length']) / simulator.parameters.getSpeedList(t)
            travel_times_UC = np.array(simulator.city_graph.shortest_paths(U_to_nodes, C, weights))
            travel_times_CE = np.array(simulator.city_graph.shortest_paths(U_to_nodes, E_pos, weights))
        else:
            alpha = [24 * 3600 for v in Vehicles]  # Set it to the possible maximum value so the restriction is relaxed
            phi = [max(0, (target_relocation - U[u_node].accumulated_relocation) / target_relocation) for u_node in U_nodes]
            travel_times_UC = np.zeros((len(U_nodes), len(C)))
            travel_times_CE = np.zeros((len(U_nodes), len(E_pos)))
            max_relocations = len(U)

        if severity == 0:
            survival_matrix = self.SurvivalFunction(cand_demand_travel / 60)  # Travel time in minutes
            survival_dispatching = self.SurvivalFunction(travel_times_CE / 60)  # Travel time in minutes
        else:
            b_bar = np.mean([U[u_node].total_busy_time for u_node in U_nodes]) if len(U) > 0 else 1e10
            rho = [[U[u_node].total_busy_time + travel_times_CE[u][e] + params.mean_busytime[severity].at[t + 1, params.graph_to_demand[e_node]] - b_bar for e, e_node in enumerate(E_pos)] for u, u_node in enumerate(U_nodes)]

        # Filter the survival matrix leaving only the nodes that are reachable,
        # 0 to all the rest
        if severity == 0:
            filtered_survival = np.array([[survival_matrix[j, i] if c_node in params.reachable_inverse[t][d_node] else 0
                                           for j, c_node in enumerate(C)]
                                          for i, d_node in enumerate(D)]).T
        else:
            filtered_survival = np.array([[cand_demand_travel[j, i] if c_node in params.reachable_inverse[t][d_node] else inf_travel_time
                                           for j, c_node in enumerate(C)]
                                          for i, d_node in enumerate(D)]).T

        # [j, i] matrix respresenting psi_{i,j} * d_i
        coefficients = D_rates[severity].loc[t + 1, D].values * filtered_survival

        if severity == 0:
            # Objective function
            model.setObjective(grb.quicksum(grb.quicksum(coefficients[C.index(c_node), i] * x[C.index(c_node)][i] if c_node in C else 0
                                            for c_node in reachable_inverse[d_node])
                               for i, d_node in enumerate(D)) +                             # noqa: W504
                               np.sum(D_rates[0].values) * grb.quicksum(grb.quicksum(survival_dispatching[u, e] * k[u][e]
                                                                                     for e, e_node in enumerate(E_pos))
                               for u, u_node in enumerate(U_nodes)) - tt_penalty * grb.quicksum(grb.LinExpr(travel_times_UC[u], r[u]) for u in range(len(U_nodes))))
            model.ModelSense = grb.GRB.MAXIMIZE
        else:
            # Objective function
            model.setObjective(grb.quicksum(grb.quicksum(coefficients[C.index(c_node), i] * x[C.index(c_node)][i] + inf_travel_time * h[i] if c_node in C else 0
                                            for c_node in reachable_inverse[d_node])
                               for i, d_node in enumerate(D)) + grb.quicksum(uncovered_penalty * b[e] + late_response_penalty * w[e] + dispatching_penalty * grb.quicksum(rho[u][e] * k[u][e] for u, u_node in enumerate(U_nodes)) for e, e_node in enumerate(E_pos))
                               + tt_penalty * grb.quicksum(grb.LinExpr(travel_times_UC[u], r[u]) for u in range(len(U_nodes))))     # noqa: W503
            model.ModelSense = grb.GRB.MINIMIZE

        # Constraints
        # ------------

        # Capacity constraint
        # sum_{j \in C} y_j = N_{ALS/BLS} - sum_{u, e} delta_ue
        model.addConstr(lhs=grb.quicksum(y),
                        sense=grb.GRB.EQUAL,
                        rhs=grb.quicksum(grb.quicksum(r[u][j] for j in range(len(C))) for u in range(len(U))),
                        name='CapacityConstraint')

        # Constraint 1
        # x_{ij} \leq y_j \forall i, j \in N_i
        {(i, j): model.addConstr(lhs=x[C.index(c_node)][i],
                                 sense=grb.GRB.LESS_EQUAL,
                                 rhs=y[C.index(c_node)],
                                 name='Const_1_{}_{}'.format(i, j))
         for i, d_node in enumerate(D)
         for j, c_node in enumerate(params.reachable_inverse[t][d_node]) if c_node in C}

        if severity == 0:
            # Constraint 2
            # sum_{j \in Lambda_i} x_{ij} \leq 1
            {i: model.addConstr(lhs=grb.quicksum(x[C.index(j)][i] for j in C),
                                sense=grb.GRB.LESS_EQUAL,
                                rhs=1,
                                name='Const_2_{}'.format(i))
             for i, d_node in enumerate(D)}
        else:
            # Constraint 2
            # sum_{j \in Lambda_i} x_{ij} + h_i = 1
            {i: model.addConstr(lhs=grb.quicksum(x[C.index(j)][i] if j in C else 0 for j in params.reachable_inverse[t][d_node]) + h[i],
                                sense=grb.GRB.EQUAL,
                                rhs=1,
                                name='Const_2_{}'.format(i))
             for i, d_node in enumerate(D)}

        # Constraint 3
        # y_j = v_j - sum_{j, j'} r_jj' + sum_{u} r_uj
        {j: model.addConstr(lhs=y[j],
                            sense=grb.GRB.EQUAL,
                            rhs=grb.quicksum(r[u][j] for u, u_node in enumerate(U_nodes)),
                            name='Const_3_{}'.format(c_node))
         for j, c_node in enumerate(C)}

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

        # Constraint 7
        # \sum_{j} r_uj\tau_uj \leq \Phi_u\vartheta
        {u: model.addConstr(lhs=grb.quicksum(travel_times_UC[u][j] * r[u][j] * (0 if U[u_node].relocating and U[u_node].station == c_node else 1) for j, c_node in enumerate(C)),
                            sense=grb.GRB.LESS_EQUAL,
                            rhs=max_relocation_time * phi[u],
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

        model.setParam('MIPGap', params.optimization_gap)
        # model.setParam('LogToConsole', 0)

        print(time.time() - start_time, '- Solving the model...')
        model.optimize()

        # try:
        #    simulator.statistics['GAPALSPart1'].record(simulator.now(), model.MIPGap)
        # except:
        #    pass

        print(time.time() - start_time, '- Done!')
        simulator.statistics['OptimizationSize{}{}'.format('ALS' if severity == 0 else 'BLS', borough)].record(simulator.now(), len(U))
        simulator.statistics['OptimizationTime{}{}'.format('ALS' if severity == 0 else 'BLS', borough)].record(simulator.now(), time.time() - start_time)

        final_positions = []
        for j in range(len(C)):
            if y[j].x == 1:
                final_positions.append(C[j])
        reposition_matrix = [[r[u][C.index(j)].x for j in final_positions] for u in range(len(U_nodes))]
        for i, node in enumerate(U_nodes):
            if 1 in reposition_matrix[i] and final_positions[reposition_matrix[i].index(1)] != Vehicles[i].station:
                final_repositioning[U[node]] = final_positions[reposition_matrix[i].index(1)]
            else:
                final_positions.append(node)

        for e, emergency in enumerate(E):
            dispatch_list = [k[u][e].x for u, u_node in enumerate(U_nodes)]
            if 1 in dispatch_list:
                final_dispatching[U[U_nodes[dispatch_list.index(1)]]] = emergency
            elif z[e].x == 1:
                final_dispatching[simulator.newUberVehicle(simulator.parameters.uber_nodes[emergency.node], emergency.borough)] = emergency

        if severity == 0:
            print()
        return final_positions, final_repositioning, final_dispatching
