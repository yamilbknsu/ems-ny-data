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
    
class CustomArrivalsGenerator(ArrivalGenerator):
    """
    Simulate a pregenerated set of arrivals
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
