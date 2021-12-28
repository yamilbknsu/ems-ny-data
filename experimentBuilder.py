# Import Statements
import pickle

# Testing sets
days = ['friday']
model = ['SBRDANew']
dataReplica = list(range(30))
simTime = [24 * 3600]
uberHours = {'SBRDANew': [0], 'ROA': [0], 'SBRDAStatic': [0]}
ambulanceDistribution = [[80, 175]]
workloadRestriction = [True]
workloadLimit = [.7]
simultaneous_relocations = [8]

# Penalization for leaving a LS emergency uncovered
# It is used in the objective function of the model where everything is
# in terms of time (seconds), therefore, this values represent that
# leaving an emergency uncovered is equivalent to responding 10 days,
# 5 days or 1 day late, which is practice, means it will only be chosen
# when no other options are left
uncovered_penalty = [10 * 24 * 3600, 5 * 24 * 3600, 24 * 3600]

# Penalization for each second over the maximum allowed waiting time a LS emergency
# is expected to receive service. This number is mainly used to evaluate the trade-off
# between ambulance utilization (dispatching penalty) and late response. For example, if
# this is set to 60, we allow an emergency to wait one aditional second if it saves one
# minute in ambulance workload imbalance.
late_response_penalty = [60, 120, 30]

# Not included in the paper, but a parameter of the model anyway.
dispatching_penalty = [1]

# Penalty for travel time. This is used to consider the dispatching options that
# minimize travel time among those that are equal in every other way, so this value should not
# interfere with the other penalties and therefore is a very small number.
travel_distance_penalty = [1e-10, 1e-5]

target_relocTime = [2160]
max_relocation_time = [1200]
max_redeployment_time = [800]
relocation_cooldown = [3600]
GAP = [.05]

EXPERIMENTS = [{'day': day, 'model': m, 'dataReplica': rep, 'simTime': time, 'ambulance_distribution': amb, 'workload_restriction': wlRes, 'workload_limit': wlL, 'GAP': gap,
                'parameters_dir': 'HalfManhattan', 'uberHours': uH, 'relocQty': relocQty, 'uncovered': unc, 'lateResponse': lr, 'disaptchingPenalt': disp, 'ttPenalty': ttp, 'targetReloc': targetReloc,
                'maxReloc': maxreloc, 'relocCooldown': relocCooldown, 'maxRedeployment': maxRed}
               for day in days for m in model for gap in GAP for amb in ambulanceDistribution for wlL in workloadLimit for wlRes in workloadRestriction
               for rep in dataReplica for time in simTime for uH in uberHours[m] for relocQty in simultaneous_relocations for unc in uncovered_penalty for lr in late_response_penalty
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
print(len(EXPERIMENTS))

with open('experimentsConfig.pickle', 'wb') as f:
    pickle.dump(output, f)
