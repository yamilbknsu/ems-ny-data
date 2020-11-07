# Import Statements
import pickle

# Testing sets
days = ['friday']
model = ['SBRDA', 'SBRDANew', 'SBRDAStatic', 'ROA']
dataReplica = list(range(10))
simTime = [24 * 3600]
uberHours = [0]
ambulanceDistribution = [[355, 802]]  # , [319, 722] , [284, 642]]
workloadRestriction = [True]
workloadLimit = [.7]
simultaneous_relocations = [8]
uncovered_penalty = [10 * 24 * 3600]
late_response_penalty = [60]
dispatching_penalty = [1]
travel_distance_penalty = [1e-6]
target_relocTime = [2160]
max_relocation_time = [1200]
max_redeployment_time = [800]
relocation_cooldown = [3600]
GAP = [.05]

EXPERIMENTS = [{'day': day, 'model': m, 'dataReplica': rep, 'simTime': time, 'ambulance_distribution': amb, 'workload_restriction': wlRes, 'workload_limit': wlL, 'GAP': gap,
                'parameters_dir': 'Base', 'uberHours': uH, 'relocQty': relocQty, 'uncovered': unc, 'lateResponse': lr, 'disaptchingPenalt': disp, 'ttPenalty': ttp, 'targetReloc': targetReloc,
                'maxReloc': maxreloc, 'relocCooldown': relocCooldown, 'maxRedeployment': maxRed}
               for day in days for m in model for gap in GAP for amb in ambulanceDistribution for wlL in workloadLimit for wlRes in workloadRestriction
               for rep in dataReplica for time in simTime for uH in uberHours for relocQty in simultaneous_relocations for unc in uncovered_penalty for lr in late_response_penalty
               for disp in dispatching_penalty for ttp in travel_distance_penalty for targetReloc in target_relocTime for maxreloc in max_relocation_time for relocCooldown in relocation_cooldown
               for maxRed in max_redeployment_time]

output = []

for experiment in EXPERIMENTS:

    name = '{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}'.format(experiment['day'],
                                                                          experiment['model'],
                                                                          experiment['dataReplica'],
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
                                                                          experiment['maxRedeployment'],
                                                                          experiment['relocCooldown'])
    output.append([name, experiment])


with open('experimentsConfig.pickle', 'wb') as f:
    pickle.dump(output, f)
