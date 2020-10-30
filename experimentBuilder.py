# Import Statements
import pickle

# Testing sets
days = ['friday']
model = ['SBRDA']
dataReplica = list(range(2))
simTime = [24 * 3600]
static = [False]
uberHours = [6000 / 60, 0]
ambulanceDistribution = [[355, 802], [319, 722], [284, 642]]
workloadRestriction = [True]
workloadLimit = [.65, .5]
simultaneous_relocations = [4, 6, 8, 10]
uncovered_penalty = [10 * 24 * 3600]
late_response_penalty = [60]
dispatching_penalty = [.01]
travel_distance_penalty = [1e-6]
target_relocTime = [2160]
max_relocation_time = [2400]
relocation_cooldown = [3600]
GAP = [.05]

EXPERIMENTS = [{'day': day, 'model': m, 'dataReplica': rep, 'simTime': time, 'static': rel, 'ambulance_distribution': amb, 'workload_restriction': wlRes, 'workload_limit': wlL, 'GAP': gap,
                'parameters_dir': 'HRDemand', 'uberHours': uH, 'relocQty': relocQty, 'uncovered': unc, 'lateResponse': lr, 'disaptchingPenalt': disp, 'ttPenalty': ttp, 'targetReloc': targetReloc,
                'maxReloc': maxreloc, 'relocCooldown': relocCooldown}
               for day in days for m in model for gap in GAP for amb in ambulanceDistribution for wlL in workloadLimit for wlRes in workloadRestriction
               for rep in dataReplica for time in simTime for rel in static for uH in uberHours for relocQty in simultaneous_relocations for unc in uncovered_penalty for lr in late_response_penalty
               for disp in dispatching_penalty for ttp in travel_distance_penalty for targetReloc in target_relocTime for maxreloc in max_relocation_time for relocCooldown in relocation_cooldown]

output = []

for experiment in EXPERIMENTS:

    name = '{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}'.format(experiment['day'],
                                                                          experiment['model'],
                                                                          experiment['dataReplica'],
                                                                          'Static' if experiment['static'] else 'Online',
                                                                          experiment['workload_limit'],
                                                                          experiment['GAP'],
                                                                          experiment['parameters_dir'],
                                                                          str(experiment['ambulance_distribution'][0]) + 'ALS' + str(experiment['ambulance_distribution'][1]) + 'BLS',
                                                                          experiment['simTime'],
                                                                          experiment['uberHours'],
                                                                          experiment['relocQty'],
                                                                          experiment['uncovered'],
                                                                          experiment['lateResponse'],
                                                                          experiment['disaptchingPenalt'],
                                                                          experiment['ttPenalty'],
                                                                          experiment['targetReloc'],
                                                                          experiment['maxReloc'],
                                                                          experiment['relocCooldown'])
    output.append([name, experiment])


with open('experimentsConfig.pickle', 'wb') as f:
    pickle.dump(output, f)
