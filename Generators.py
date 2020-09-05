# Imports
import numpy as np
from typing import List

# Internal Imports
import Models
import Events


class ArrivalGenerator:
    """
    Base class for arrival generators
    """
    def __init__(self):
        pass

    def generator(self,
                 simulator: "Models.EMSModel"):
        pass


class ExponentialGenerator(ArrivalGenerator):
    """
    (Homogeneous and random node)
    """

    def __init__(self,
                 mean):
        super().__init__()

        self.lastTime: float = 0  # Meaning last time an arrival was scheduled
        self.mean: float = mean

    def generator(self,
                 simulator: "Models.EMSModel"):
        while True:
            nextTime: float = self.lastTime + np.random.exponential(self.mean)
            nextNode: str = simulator.city_graph.ns[np.random.randint(simulator.city_graph.vcount())]['osmid']
            yield Events.EmergencyArrivalEvent(self, nextTime, nextNode, np.random.randint(3), 82)
    
class CustomArrivalsGenerator(ArrivalGenerator):
    """
    Simulate a deterministic set of arrival
    Used for debugging purposes.
    """

    def __init__(self,
                 arrivals: List['Events.EmergencyArrivalEvent']):
        super().__init__()

        self.arrivals: List[Events.EmergencyArrivalEvent] = arrivals
        self.arrivals.sort(key=lambda e: e.time)
    
    def generator(self,
                 simulator: "Models.EMSModel"):
        
        for arrival in self.arrivals:
            yield arrival
