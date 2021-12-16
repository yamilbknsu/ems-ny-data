# Import Statements
import random
import numpy as np
import pandas as pd
import geopandas as gpd
from typing import List, Optional, Callable, Dict, Tuple

class SimulationParameters:
    """
    A SimulationParameters instance is meant to hold all the parameters
    for the simulation, most of them are constants (like the id's of the)
    candidate and demand nodes, but there are others that are computed on
    the fly. For the latter ones, compute functions are provided in this
    class.

    When running experiments, you have one instance of this class in order
    to represent one configuration of the system.

    The main abstractions of the model that are necessary to understand the
    data structures found here are:

    - Nodes in the graph of NYC are indetified by the osmid attribute, which
      we pass along as a string.
    - Similarly, edges of this graph are identified by the edgeid attribute
      of the spatial object, also referenced here as a string.
    - Emergency events were divided into High and Low severity groups. This
      means that demand related parameters are usually divided in two (usually
      as a list with two elements). High priority is always indexed as 0.
    - It's assumed that a working day is divided into time periods with fixed
      duration, and that there is a distribution describing the average speed
      of each street for each one. So, all of the parameters that are dependent
      on this speed, are expected to be a list with as much elements as time periods.
    - Simulation time is always in seconds.
    """

    def __init__(self,
                 demand_rates: List["pd.DataFrame"],
                 mean_busytime: List["pd.DataFrame"],
                 cand_cand_time: List["np.array"],
                 cand_demand_time: List["np.array"],
                 hospital_nodes: Dict[int, List[str]],
                 hospital_borough: Dict[str, int],
                 candidate_nodes: List[str],
                 demand_nodes: List[str],
                 speeds_df: "pd.DataFrame",
                 ALS_tours: float,
                 BLS_tours: float,
                 nodes_with_borough: "gpd.GeoDataFrame",
                 graph_to_demand: Dict,
                 candidates_borough: Dict[int, List[str]],
                 demand_borough: Dict[int, List[str]],
                 reachable_demand: List[Dict[str, List[str]]],
                 reachable_inverse: List[Dict[str, List[str]]],
                 uber_nodes: Dict[str, str],
                 vehicle_shift: Callable = lambda _: 8 * 3600,
                 vehicle_arrival_deviation: float = 3600,
                 uber_seconds: float = 8000,
                 simulation_time: float = 3600,
                 force_static: bool = False,
                 optimization_gap: float = .05,
                 time_periods: List[float] = [7, 3, 6, 3, 5],
                 time_shifts: List[float] = [8, 8, 8],
                 vehicle_types: int = 2,
                 initial_nodes: Optional[List[List[str]]] = None,
                 uncovered_penalty: float = 10 * 24 * 3600,
                 late_response_penalty: float = 60,
                 dispatching_penalty: float = .01,
                 travel_distance_penalty: float = 1e-10,
                 target_relocation_time: float = 2160,
                 max_relocation_time: float = 1200,
                 max_redeployment_time: float = 800,
                 relocation_cooldown: float = 3600,
                 max_expected_simultaneous_relocations: int = 4,
                 maximum_overload_ALS: float = .4,
                 maximum_overload_BLS: float = .4,
                 name: str = 'Model',
                 random_seed: float = 420):
        """
        A SimulationParameters instance is meant to hold all the parameters
        for the simulation, most of them are constants (like the id's of the)
        candidate and demand nodes, but there are others that are computed on
        the fly. For the latter ones, compute functions are provided in this
        class.

        When running experiments, you have one instance of this class in order
        to represent one configuration or replica of the system.

        Refer to the Model class documentation for some abstrations of the model useful
        for understanding these data structures.

        In order to facilitate the understanding of this parameters,
        the expected size of the matrices is defined in brackets using this
        notation:

        C: Number of candidadte Nodes
        D: Number of demand Nodes
        T: Number of time periods
        V: Number of ambulance types
        S: Number of emergency categories
        B: Number of Boroughs
        H: Number of hospital nodes
        ?: Undertermined

        Note: The ambulance distribution among boroughs is approximated using an heuristic
        that runs on the initialization function of this class.

        Args:
            demand_rates (List[DataFrame]): [S, T, D] Hourly demand rate for each node
            mean_busytime (List[DataFrame]): [S, T, D] Mean busy time taken from an ambulance
                                            attending an emergency coming for each node. This matrix
                                            is updated during the simulation with simulated values.
            cand_cand_time (List[numpy.Array]): [T, C, C] Travel time between candidate nodes
            cand_demand_time (List[numpy.Array]): [T, C, D] Travel time between candidate and demand nodes
            hospital_nodes (Dict[int, List[str]]): [?] Hospitals nodes that can attend emergencies.
                                                    This dictionary is indexed with an int to ease the
                                                    implementation of severity-specific hospitals
            hospital_borough (Dict[str, int]): Mapping of each hospital to its corresponding borough 
            candidate_nodes (List[str]): [C] List of candidate nodes (by osmid)
            demand_nodes (List[str]): [D] List of demand nodes (by osmid)
            speeds_df (pd.DataFrame): Dataframe with columns: edgeid and p+'x'+n for x in range(1,T)
            ALS_tours (float): Number of 8-hour ALS tours available
            BLS_tours (float): Number of 8-hour BLS tours available
            nodes_with_borough (gpd.GeoDataFrame): [D rows] Dataframe with spatial info of each demand node and a column with
                                                            its corresponding borough (this is the only column used in the model
                                                            but it was the most straightforward way of include it)
            graph_to_demand (Dict[str, str]): Mapping of graph nodes to nearest demand node
            candidates_borough (Dict[int, List[str]]): [B: ?] List of candidate nodes in each borough
            demand_borough (Dict[int, List[str]]): [B: ?] List of demand nodes in each borough
            reachable_demand (List[Dict[str, List[str]]]): [C: ?] List of reachable demand nodes for each candidate node in less
                                                                  than 8 minutes
            reachable_inverse (List[Dict[str, List[str]]]): [D: ?] List of candidate nodes that are able to reach each demand node
                                                                   in less than 8 minutes
            uber_nodes (Dict[str, str]): A dict with each node on the graph as a key and the node where a uber might appear
                                         if an emergency calls one from the key node as value of the dict.
            vehicle_shift (Callable): Function determining the shift length of a specific vehicle.
                                      Receives a vehicle returns an int
            uber_seconds (float): Total number of RHS hours allowed
            simulation_time (float): Time limit for the simulation. Defaults to 3600.
            force_static (bool): If no online relocation is performed.
            optimization_gap (float, optional): MIPGap parameter for gurobi solver. Defaults to .1
            apply_workload_restriction (bool, optional): Whether or not to apply workload restrictions in the
                                                         optimization process. Defaults to True.
            time_periods (List[float], optional): [T] Duration in hours of each time period. Defaults to [7, 3, 6, 3, 5].
            time_shifts (List[float], optional): [T] Duration in hours of each work shift. Defaults to [8, 8, 8].
            vehicle_types (int, optional): Number of vehicle types. Defaults to 2.
            initial_nodes (List[List[str]], optional): [V, ?] Initial nodes for  the ambulances. Defaults to None.
            uncovered_penalty (float, optional): Penalty for leaving an emergency uncovered during the optimization process. Defaults to 10 * 24 * 3600
            late_response_penalty (float, optional): Penalty for responding late during optimization. Defaults to 60
            dispatching_penalty (float, optional): Penalty for dispatching a heavily used ambulance. Defaults to .01
            travel_distance_penalty (float, optional): Penalty for distance traveled when relocating. Defaults to 1e-10
            target_relocation_time (float, optional): Maximum seconds an ambulance can relocate before needing a break. Defaults to 2160
            max_relocation_time (float, optional): Maximum time an ambulance can travel relocating on one go. Defaults to 1200
            max_redeployment_time (float, optional): Maximum time an ambulance can travel redeploying on one go. Defaults to 800
            relocation_cooldown (float, optional): Time an ambulance needs to spend without relocating before it can accept a new relocation request. Defaults to 3600.
            max_expected_simultaneous_relocations (float, optional): Maximum allowed BLS ambulances relocating at the same time. Defaults to 4
            maximum_overload_ALS (float, optional): Fraction of the time periods that an ALS vehicle is allowed
                                                    spend on relocation. Defaults to .4.
            maximum_overload_BLS (float, optional): Fraction of the time periods that an BLS vehicle is allowed
                                                    spend on relocation. Defaults to .4.
            random_seed (float, optional): Seed for random generator. Defaults to 420.
        """
        self.name = name

        # General parameters
        self.simulation_time = simulation_time
        self.force_static = force_static
        self.optimization_gap = optimization_gap
        self.time_periods = time_periods
        self.time_shifts = time_shifts
        self.vehicle_types = vehicle_types
        self.ALS_tours = ALS_tours
        self.BLS_tours = BLS_tours
        self.vehicle_shift = vehicle_shift
        self.vehicle_arrival_deviation = vehicle_arrival_deviation
        self.initial_nodes = initial_nodes
        self.hospital_nodes = hospital_nodes
        self.hospital_borough = hospital_borough
        self.random_seed = random_seed

        # Routing parameters
        self.speeds_df = speeds_df
        self.candidate_nodes = candidate_nodes
        self.demand_nodes = demand_nodes
        self.demand_rates = demand_rates
        self.mean_busytime = mean_busytime
        self.cand_cand_time = cand_cand_time
        self.cand_demand_time = cand_demand_time
        self.candidates_borough = candidates_borough
        self.graph_to_demand = graph_to_demand
        self.demand_borough = demand_borough
        self.reachable_demand = reachable_demand
        self.reachable_inverse = reachable_inverse
        self.uber_nodes = uber_nodes
        self.uber_seconds = uber_seconds
        self.nodes_with_borough = nodes_with_borough

        # Hyper-Parameters
        self.maximum_overload_ALS = maximum_overload_ALS
        self.maximum_overload_BLS = maximum_overload_BLS
        self.target_relocation_time = target_relocation_time
        self.max_expected_simultaneous_relocations = max_expected_simultaneous_relocations
        self.max_relocation_time = max_relocation_time
        self.max_redeployment_time = max_redeployment_time
        self.uncovered_penalty = uncovered_penalty
        self.late_response_penalty = late_response_penalty
        self.dispatching_penalty = dispatching_penalty
        self.relocation_cooldown = relocation_cooldown
        self.travel_distance_penalty = travel_distance_penalty

        # Set the RNG seed
        np.random.seed(self.random_seed)
        random.seed(self.random_seed)

        # Initialization of some parameters
        # ---------------------------------

        # It was necessary to acknowledge the fact that the borough 0 exists (those points outside the border)
        # So it's set to be those nodes that don't fit inside any other borough
        candidates_with_borough = candidates_borough[1] + candidates_borough[2] + candidates_borough[3] + candidates_borough[4] + candidates_borough[5]
        self.candidates_borough[0] = list(set(candidate_nodes) - set(candidates_with_borough))

        # Compute ambulance distribution
        # This algorithm approximates the importance of each borough in terms of demand with relation to the rest of them
        # as the percentage of the total demand rate present on each borough.
        # Then, the length of each time interval is used to approximate the temporal imporantance of each one.
        # Using both distributions, the ambulance shifts are assigned a borough and a time period.
        ALS_temporal_demand_factor = np.array([np.sum(self.demand_rates[0].loc[t, :]) for t in range(1, 1 + len(time_periods))])
        ALS_temporal_demand_factor /= np.sum(ALS_temporal_demand_factor)
        BLS_temporal_demand_factor = np.array([np.sum(self.demand_rates[1].loc[t, :]) for t in range(1, 1 + len(time_periods))])
        BLS_temporal_demand_factor /= np.sum(BLS_temporal_demand_factor)

        # Computing temporal factor
        cumulative_time_periods = [sum(self.time_periods[:m + 1]) for m in range(len(self.time_periods))]
        cumulative_time_shifts = [sum(self.time_shifts[:m + 1]) for m in range(len(self.time_shifts))]

        used_hours = np.zeros(len(self.time_periods))
        ALS_final_temporal_factor = []
        BLS_final_temporal_factor = []
        for ts in cumulative_time_shifts:
            i = 0
            for tp in cumulative_time_periods:
                if tp >= ts:
                    break
                i += 1
            factor = np.array([np.clip((ts - sum(self.time_periods[:j])) / self.time_periods[j], a_min=0, a_max=1) if j <= i else 0 for j in range(len(self.time_periods))]) - used_hours

            ALS_final_temporal_factor.append(sum(ALS_temporal_demand_factor * factor))
            BLS_final_temporal_factor.append(sum(BLS_temporal_demand_factor * factor))

            used_hours += factor
        
        ALS_temporal_distribution = np.trunc(ALS_tours * np.array(ALS_final_temporal_factor))
        BLS_temporal_distribution = np.trunc(BLS_tours * np.array(BLS_final_temporal_factor))

        borough_demand = np.array([np.sum(self.demand_rates[0][d][1] for d in demand_borough[b]) for b in range(1, 6)])
        borough_demand = borough_demand / np.sum(borough_demand)

        # Allocate the ambulances to each borough each time shift
        self.ambulance_distribution = [[np.round(ALS_temporal_distribution[t] * borough_demand), np.round(BLS_temporal_distribution[t] * borough_demand)] for t in range(len(time_shifts))]

    def getSpeedList(self, time_period: int):
        """
        Get the list of the speeds for each edge on the graph
        for the corresponding time period, in the requested order.

        Args:
            time_period (int): Index of the time period (0-indexed)

        Returns:
            (list): list of speeds (in meters/second)
        """
        return list(self.speeds_df['p' + str(time_period + 1) + 'n'])