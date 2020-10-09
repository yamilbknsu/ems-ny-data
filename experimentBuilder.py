# Import Statements
import pickle
import numpy as np

# Internal Imports
import Models
import Events
import Solvers
import Generators

# Testing sets
#days = ['friday', 'monday']
#dataReplica = list(range(15))
#simTime = [24*3600]
#relocatorModel = ['survivalNoExp', 'coverage']
#dispatchers = ['preparedness', 'nearest']
#relocate = [True, False] # Online or static
                        # ALS  BLS
#ambulanceDistribution = [[355, 802], [337, 761], [319, 722], [301, 681], [284,642]]
#workloadRestriction = [True]
#workloadLimit = [.2]
#useUber = [False, True]
#uberRatio = [.5,1]
#GAP = [.05]
#relocationPeriod = [600]

days = ['friday']
relocatorModel = ['coverage']
dispatchers = ['nearest']
dataReplica = [9]
simTime = [24*3600]
relocationPeriod = [300]
relocate = [True]
uberRatio = [.1]
GAP = [.05]
useUber = [False]
ambulanceDistribution = [[337, 761], [319, 722], [301, 681]]
workloadRestriction = [True]
workloadLimit = [.2]

EXPERIMENTS = [{'day': day, 'dataReplica': rep, 'simTime': time, 'relocatorModel': model, 'dispatcher': disp,
                'relocate': rel, 'ambulance_distribution': amb, 'workload_restriction': wlRes, 'workload_limit': wlL, 'useUber': uber, 'GAP': gap,
                'parameters_dir': 'HRDemand', 'relocation_period': reloc_period, 'uberRatio': uR}
                for day in days for gap in GAP for amb in ambulanceDistribution for wlL in workloadLimit for wlRes in workloadRestriction
                for uber in useUber for rep in dataReplica for time in simTime for model in relocatorModel for disp in dispatchers for reloc_period in relocationPeriod
                for rel in relocate for uR in uberRatio]

output = []

for experiment in EXPERIMENTS:

    name = '{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}'.format(experiment['day'], experiment['dataReplica'], experiment['relocatorModel'], experiment['dispatcher'],
                                            'Relocate' if experiment['relocate'] else 'NoRelocation', 'Workload' if experiment['workload_restriction'] else 'NoWorkloadRestriction',
                                            experiment['workload_limit'], 'Uber' if experiment['useUber'] else 'NoUber', experiment['GAP'], experiment['parameters_dir'],
                                            str(experiment['ambulance_distribution'][0]) + 'ALS' + str(experiment['ambulance_distribution'][1]) + 'BLS', experiment['relocation_period'], experiment['simTime'],
                                            experiment['uberRatio'])
    output.append([name, experiment])

# -------------------------------------
days = ['friday']
relocatorModel = ['coverage']
dispatchers = ['nearest']
dataReplica = [9]
simTime = [24*3600]
relocationPeriod = [300]
relocate = [True]
uberRatio = [.1]
GAP = [.05]
useUber = [True]
ambulanceDistribution = [[355, 802], [301, 681]]
workloadRestriction = [True]
workloadLimit = [.2]

EXPERIMENTS = [{'day': day, 'dataReplica': rep, 'simTime': time, 'relocatorModel': model, 'dispatcher': disp,
                'relocate': rel, 'ambulance_distribution': amb, 'workload_restriction': wlRes, 'workload_limit': wlL, 'useUber': uber, 'GAP': gap,
                'parameters_dir': 'HRDemand', 'relocation_period': reloc_period, 'uberRatio': uR}
                for day in days for gap in GAP for amb in ambulanceDistribution for wlL in workloadLimit for wlRes in workloadRestriction
                for uber in useUber for rep in dataReplica for time in simTime for model in relocatorModel for disp in dispatchers for reloc_period in relocationPeriod
                for rel in relocate for uR in uberRatio]

for experiment in EXPERIMENTS:

    name = '{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}'.format(experiment['day'], experiment['dataReplica'], experiment['relocatorModel'], experiment['dispatcher'],
                                            'Relocate' if experiment['relocate'] else 'NoRelocation', 'Workload' if experiment['workload_restriction'] else 'NoWorkloadRestriction',
                                            experiment['workload_limit'], 'Uber' if experiment['useUber'] else 'NoUber', experiment['GAP'], experiment['parameters_dir'],
                                            str(experiment['ambulance_distribution'][0]) + 'ALS' + str(experiment['ambulance_distribution'][1]) + 'BLS', experiment['relocation_period'], experiment['simTime'],
                                            experiment['uberRatio'])
    output.append([name, experiment])

# -------------------------------------
days = ['friday']
relocatorModel = ['coverage']
dispatchers = ['preparedness']
dataReplica = [9]
simTime = [24*3600]
relocationPeriod = [300]
relocate = [True]
uberRatio = [.1]
GAP = [.05]
useUber = [False]
ambulanceDistribution = [[319, 722], [301, 681]]
workloadRestriction = [True]
workloadLimit = [.2]

EXPERIMENTS = [{'day': day, 'dataReplica': rep, 'simTime': time, 'relocatorModel': model, 'dispatcher': disp,
                'relocate': rel, 'ambulance_distribution': amb, 'workload_restriction': wlRes, 'workload_limit': wlL, 'useUber': uber, 'GAP': gap,
                'parameters_dir': 'HRDemand', 'relocation_period': reloc_period, 'uberRatio': uR}
                for day in days for gap in GAP for amb in ambulanceDistribution for wlL in workloadLimit for wlRes in workloadRestriction
                for uber in useUber for rep in dataReplica for time in simTime for model in relocatorModel for disp in dispatchers for reloc_period in relocationPeriod
                for rel in relocate for uR in uberRatio]

for experiment in EXPERIMENTS:

    name = '{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}'.format(experiment['day'], experiment['dataReplica'], experiment['relocatorModel'], experiment['dispatcher'],
                                            'Relocate' if experiment['relocate'] else 'NoRelocation', 'Workload' if experiment['workload_restriction'] else 'NoWorkloadRestriction',
                                            experiment['workload_limit'], 'Uber' if experiment['useUber'] else 'NoUber', experiment['GAP'], experiment['parameters_dir'],
                                            str(experiment['ambulance_distribution'][0]) + 'ALS' + str(experiment['ambulance_distribution'][1]) + 'BLS', experiment['relocation_period'], experiment['simTime'],
                                            experiment['uberRatio'])
    output.append([name, experiment])


# -------------------------------------
days = ['friday']
relocatorModel = ['coverage']
dispatchers = ['preparedness']
dataReplica = [9]
simTime = [24*3600]
relocationPeriod = [300]
relocate = [True]
uberRatio = [.1]
GAP = [.05]
useUber = [True]
ambulanceDistribution = [[355, 802], [301, 681]]
workloadRestriction = [True]
workloadLimit = [.2]

EXPERIMENTS = [{'day': day, 'dataReplica': rep, 'simTime': time, 'relocatorModel': model, 'dispatcher': disp,
                'relocate': rel, 'ambulance_distribution': amb, 'workload_restriction': wlRes, 'workload_limit': wlL, 'useUber': uber, 'GAP': gap,
                'parameters_dir': 'HRDemand', 'relocation_period': reloc_period, 'uberRatio': uR}
                for day in days for gap in GAP for amb in ambulanceDistribution for wlL in workloadLimit for wlRes in workloadRestriction
                for uber in useUber for rep in dataReplica for time in simTime for model in relocatorModel for disp in dispatchers for reloc_period in relocationPeriod
                for rel in relocate for uR in uberRatio]

for experiment in EXPERIMENTS:

    name = '{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}'.format(experiment['day'], experiment['dataReplica'], experiment['relocatorModel'], experiment['dispatcher'],
                                            'Relocate' if experiment['relocate'] else 'NoRelocation', 'Workload' if experiment['workload_restriction'] else 'NoWorkloadRestriction',
                                            experiment['workload_limit'], 'Uber' if experiment['useUber'] else 'NoUber', experiment['GAP'], experiment['parameters_dir'],
                                            str(experiment['ambulance_distribution'][0]) + 'ALS' + str(experiment['ambulance_distribution'][1]) + 'BLS', experiment['relocation_period'], experiment['simTime'],
                                            experiment['uberRatio'])
    output.append([name, experiment])


# -------------------------------------
days = ['friday']
relocatorModel = ['survivalNoExp']
dispatchers = ['nearest', 'preparedness']
dataReplica = [9]
simTime = [24*3600]
relocationPeriod = [300]
relocate = [True]
uberRatio = [.1]
GAP = [.05]
useUber = [True]
ambulanceDistribution = [[355, 802]]
workloadRestriction = [True]
workloadLimit = [.2]

EXPERIMENTS = [{'day': day, 'dataReplica': rep, 'simTime': time, 'relocatorModel': model, 'dispatcher': disp,
                'relocate': rel, 'ambulance_distribution': amb, 'workload_restriction': wlRes, 'workload_limit': wlL, 'useUber': uber, 'GAP': gap,
                'parameters_dir': 'HRDemand', 'relocation_period': reloc_period, 'uberRatio': uR}
                for day in days for gap in GAP for amb in ambulanceDistribution for wlL in workloadLimit for wlRes in workloadRestriction
                for uber in useUber for rep in dataReplica for time in simTime for model in relocatorModel for disp in dispatchers for reloc_period in relocationPeriod
                for rel in relocate for uR in uberRatio]

for experiment in EXPERIMENTS:

    name = '{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}'.format(experiment['day'], experiment['dataReplica'], experiment['relocatorModel'], experiment['dispatcher'],
                                            'Relocate' if experiment['relocate'] else 'NoRelocation', 'Workload' if experiment['workload_restriction'] else 'NoWorkloadRestriction',
                                            experiment['workload_limit'], 'Uber' if experiment['useUber'] else 'NoUber', experiment['GAP'], experiment['parameters_dir'],
                                            str(experiment['ambulance_distribution'][0]) + 'ALS' + str(experiment['ambulance_distribution'][1]) + 'BLS', experiment['relocation_period'], experiment['simTime'],
                                            experiment['uberRatio'])
    output.append([name, experiment])

# -------------------------------------
days = ['monday']
relocatorModel = ['survivalNoExp']
dispatchers = ['preparedness']
dataReplica = [9]
simTime = [24*3600]
relocationPeriod = [300]
relocate = [True]
uberRatio = [.1]
GAP = [.05]
useUber = [True]
ambulanceDistribution = [ [337, 761], [301, 681], [284,642]]
workloadRestriction = [True]
workloadLimit = [.2]

EXPERIMENTS = [{'day': day, 'dataReplica': rep, 'simTime': time, 'relocatorModel': model, 'dispatcher': disp,
                'relocate': rel, 'ambulance_distribution': amb, 'workload_restriction': wlRes, 'workload_limit': wlL, 'useUber': uber, 'GAP': gap,
                'parameters_dir': 'HRDemand', 'relocation_period': reloc_period, 'uberRatio': uR}
                for day in days for gap in GAP for amb in ambulanceDistribution for wlL in workloadLimit for wlRes in workloadRestriction
                for uber in useUber for rep in dataReplica for time in simTime for model in relocatorModel for disp in dispatchers for reloc_period in relocationPeriod
                for rel in relocate for uR in uberRatio]

for experiment in EXPERIMENTS:

    name = '{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}'.format(experiment['day'], experiment['dataReplica'], experiment['relocatorModel'], experiment['dispatcher'],
                                            'Relocate' if experiment['relocate'] else 'NoRelocation', 'Workload' if experiment['workload_restriction'] else 'NoWorkloadRestriction',
                                            experiment['workload_limit'], 'Uber' if experiment['useUber'] else 'NoUber', experiment['GAP'], experiment['parameters_dir'],
                                            str(experiment['ambulance_distribution'][0]) + 'ALS' + str(experiment['ambulance_distribution'][1]) + 'BLS', experiment['relocation_period'], experiment['simTime'],
                                            experiment['uberRatio'])
    output.append([name, experiment])


# -------------------------------------
days = ['monday']
relocatorModel = ['coverage']
dispatchers = ['nearest']
dataReplica = [9]
simTime = [24*3600]
relocationPeriod = [300]
relocate = [True]
uberRatio = [.1]
GAP = [.05]
useUber = [True, False]
ambulanceDistribution = [[284,642]]
workloadRestriction = [True]
workloadLimit = [.2]

EXPERIMENTS = [{'day': day, 'dataReplica': rep, 'simTime': time, 'relocatorModel': model, 'dispatcher': disp,
                'relocate': rel, 'ambulance_distribution': amb, 'workload_restriction': wlRes, 'workload_limit': wlL, 'useUber': uber, 'GAP': gap,
                'parameters_dir': 'HRDemand', 'relocation_period': reloc_period, 'uberRatio': uR}
                for day in days for gap in GAP for amb in ambulanceDistribution for wlL in workloadLimit for wlRes in workloadRestriction
                for uber in useUber for rep in dataReplica for time in simTime for model in relocatorModel for disp in dispatchers for reloc_period in relocationPeriod
                for rel in relocate for uR in uberRatio]

for experiment in EXPERIMENTS:

    name = '{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}'.format(experiment['day'], experiment['dataReplica'], experiment['relocatorModel'], experiment['dispatcher'],
                                            'Relocate' if experiment['relocate'] else 'NoRelocation', 'Workload' if experiment['workload_restriction'] else 'NoWorkloadRestriction',
                                            experiment['workload_limit'], 'Uber' if experiment['useUber'] else 'NoUber', experiment['GAP'], experiment['parameters_dir'],
                                            str(experiment['ambulance_distribution'][0]) + 'ALS' + str(experiment['ambulance_distribution'][1]) + 'BLS', experiment['relocation_period'], experiment['simTime'],
                                            experiment['uberRatio'])
    output.append([name, experiment])

# -------------------------------------
days = ['monday']
relocatorModel = ['coverage']
dispatchers = ['nearest']
dataReplica = [9]
simTime = [24*3600]
relocationPeriod = [300]
relocate = [True]
uberRatio = [.1]
GAP = [.05]
useUber = [True]
ambulanceDistribution = [[337, 761], [319, 722], [301, 681]]
workloadRestriction = [True]
workloadLimit = [.2]

EXPERIMENTS = [{'day': day, 'dataReplica': rep, 'simTime': time, 'relocatorModel': model, 'dispatcher': disp,
                'relocate': rel, 'ambulance_distribution': amb, 'workload_restriction': wlRes, 'workload_limit': wlL, 'useUber': uber, 'GAP': gap,
                'parameters_dir': 'HRDemand', 'relocation_period': reloc_period, 'uberRatio': uR}
                for day in days for gap in GAP for amb in ambulanceDistribution for wlL in workloadLimit for wlRes in workloadRestriction
                for uber in useUber for rep in dataReplica for time in simTime for model in relocatorModel for disp in dispatchers for reloc_period in relocationPeriod
                for rel in relocate for uR in uberRatio]

for experiment in EXPERIMENTS:

    name = '{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}'.format(experiment['day'], experiment['dataReplica'], experiment['relocatorModel'], experiment['dispatcher'],
                                            'Relocate' if experiment['relocate'] else 'NoRelocation', 'Workload' if experiment['workload_restriction'] else 'NoWorkloadRestriction',
                                            experiment['workload_limit'], 'Uber' if experiment['useUber'] else 'NoUber', experiment['GAP'], experiment['parameters_dir'],
                                            str(experiment['ambulance_distribution'][0]) + 'ALS' + str(experiment['ambulance_distribution'][1]) + 'BLS', experiment['relocation_period'], experiment['simTime'],
                                            experiment['uberRatio'])
    output.append([name, experiment])


# -------------------------------------
days = ['monday']
relocatorModel = ['survivalNoExp']
dispatchers = ['nearest']
dataReplica = [9]
simTime = [24*3600]
relocationPeriod = [300]
relocate = [True]
uberRatio = [.1]
GAP = [.05]
useUber = [True]
ambulanceDistribution = [[337, 761], [301, 681], [284,642]]
workloadRestriction = [True]
workloadLimit = [.2]

EXPERIMENTS = [{'day': day, 'dataReplica': rep, 'simTime': time, 'relocatorModel': model, 'dispatcher': disp,
                'relocate': rel, 'ambulance_distribution': amb, 'workload_restriction': wlRes, 'workload_limit': wlL, 'useUber': uber, 'GAP': gap,
                'parameters_dir': 'HRDemand', 'relocation_period': reloc_period, 'uberRatio': uR}
                for day in days for gap in GAP for amb in ambulanceDistribution for wlL in workloadLimit for wlRes in workloadRestriction
                for uber in useUber for rep in dataReplica for time in simTime for model in relocatorModel for disp in dispatchers for reloc_period in relocationPeriod
                for rel in relocate for uR in uberRatio]

for experiment in EXPERIMENTS:

    name = '{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}'.format(experiment['day'], experiment['dataReplica'], experiment['relocatorModel'], experiment['dispatcher'],
                                            'Relocate' if experiment['relocate'] else 'NoRelocation', 'Workload' if experiment['workload_restriction'] else 'NoWorkloadRestriction',
                                            experiment['workload_limit'], 'Uber' if experiment['useUber'] else 'NoUber', experiment['GAP'], experiment['parameters_dir'],
                                            str(experiment['ambulance_distribution'][0]) + 'ALS' + str(experiment['ambulance_distribution'][1]) + 'BLS', experiment['relocation_period'], experiment['simTime'],
                                            experiment['uberRatio'])
    output.append([name, experiment])


# -------------------------------------
days = ['monday']
relocatorModel = ['survivalNoExp']
dispatchers = ['preparedness']
dataReplica = [9]
simTime = [24*3600]
relocationPeriod = [300]
relocate = [True]
uberRatio = [.1]
GAP = [.05]
useUber = [False]
ambulanceDistribution = [[284,642]]
workloadRestriction = [True]
workloadLimit = [.2]

EXPERIMENTS = [{'day': day, 'dataReplica': rep, 'simTime': time, 'relocatorModel': model, 'dispatcher': disp,
                'relocate': rel, 'ambulance_distribution': amb, 'workload_restriction': wlRes, 'workload_limit': wlL, 'useUber': uber, 'GAP': gap,
                'parameters_dir': 'HRDemand', 'relocation_period': reloc_period, 'uberRatio': uR}
                for day in days for gap in GAP for amb in ambulanceDistribution for wlL in workloadLimit for wlRes in workloadRestriction
                for uber in useUber for rep in dataReplica for time in simTime for model in relocatorModel for disp in dispatchers for reloc_period in relocationPeriod
                for rel in relocate for uR in uberRatio]

for experiment in EXPERIMENTS:

    name = '{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}'.format(experiment['day'], experiment['dataReplica'], experiment['relocatorModel'], experiment['dispatcher'],
                                            'Relocate' if experiment['relocate'] else 'NoRelocation', 'Workload' if experiment['workload_restriction'] else 'NoWorkloadRestriction',
                                            experiment['workload_limit'], 'Uber' if experiment['useUber'] else 'NoUber', experiment['GAP'], experiment['parameters_dir'],
                                            str(experiment['ambulance_distribution'][0]) + 'ALS' + str(experiment['ambulance_distribution'][1]) + 'BLS', experiment['relocation_period'], experiment['simTime'],
                                            experiment['uberRatio'])
    output.append([name, experiment])

# -------------------------------------
days = ['monday']
relocatorModel = ['coverage']
dispatchers = ['preparedness']
dataReplica = [9]
simTime = [24*3600]
relocationPeriod = [300]
relocate = [True]
uberRatio = [.1]
GAP = [.05]
useUber = [False]
ambulanceDistribution = [[337, 761], [319, 722], [301, 681], [284,642]]
workloadRestriction = [True]
workloadLimit = [.2]

EXPERIMENTS = [{'day': day, 'dataReplica': rep, 'simTime': time, 'relocatorModel': model, 'dispatcher': disp,
                'relocate': rel, 'ambulance_distribution': amb, 'workload_restriction': wlRes, 'workload_limit': wlL, 'useUber': uber, 'GAP': gap,
                'parameters_dir': 'HRDemand', 'relocation_period': reloc_period, 'uberRatio': uR}
                for day in days for gap in GAP for amb in ambulanceDistribution for wlL in workloadLimit for wlRes in workloadRestriction
                for uber in useUber for rep in dataReplica for time in simTime for model in relocatorModel for disp in dispatchers for reloc_period in relocationPeriod
                for rel in relocate for uR in uberRatio]

for experiment in EXPERIMENTS:

    name = '{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}'.format(experiment['day'], experiment['dataReplica'], experiment['relocatorModel'], experiment['dispatcher'],
                                            'Relocate' if experiment['relocate'] else 'NoRelocation', 'Workload' if experiment['workload_restriction'] else 'NoWorkloadRestriction',
                                            experiment['workload_limit'], 'Uber' if experiment['useUber'] else 'NoUber', experiment['GAP'], experiment['parameters_dir'],
                                            str(experiment['ambulance_distribution'][0]) + 'ALS' + str(experiment['ambulance_distribution'][1]) + 'BLS', experiment['relocation_period'], experiment['simTime'],
                                            experiment['uberRatio'])
    output.append([name, experiment])

with open('experimentsConfigA.pickle', 'wb') as f:
    pickle.dump(output, f)