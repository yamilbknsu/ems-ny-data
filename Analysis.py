import pickle
import SimulatorBasics as Sim
import matplotlib.pyplot as plt

with open('Arrival Events/HS19/strep_0.pickle', 'rb') as f:
    statistics = pickle.load(f)

print()