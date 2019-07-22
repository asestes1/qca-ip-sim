# -*- coding: utf-8 -*-
"""
Created on Thu Jun  9 10:33:25 2016

@author: Alex
"""

import sys
import os

sys.path.append(os.path.join(__file__, '..'))

import pandas
import qca.qcarun as qcarun
import qca.delaycalc as delaycalc
import qca.flightsched as flight
import numpy
import pickle
import os
import math
import matplotlib.pyplot

year = '2007'
month = '02'
day = '04'
# airline_aggregation = 'high'
# airline_aggregation = 'med'
airline_aggregation = 'none'
data_folder = os.path.abspath(os.path.join(os.path.join(os.path.join(__file__, os.pardir), os.pardir), 'resources'))

oag_file = os.path.join(data_folder, 'OAG-CSV-ID/OAG_JFK_' + year + '_' + month + '_ID.csv')
connections_file = os.path.join(data_folder, 'Connections/Connection_' + year + '_' + month + '_' + day + '.p')
scen_file = os.path.join(data_folder, 'scenariosJFK.csv')

# folder ='NoAgg_ContMono_VaryBeta_AllSubs'
# folder ='NoAgg_UncontMono_VaryBeta_AllSubs'
# folder ='MedAgg_ContMono_VaryBeta_AllSubs'
# folder ='MedAgg_UncontMono_VaryBeta_AllSubs'
# folder ='HighAgg_ContMono_VaryBeta_AllSubs'
# folder ='HighAgg_UncontMono_VaryBeta_AllSubs'
# folder='VaryingBetaAllSubauctions'
# folder = 'JOVaryBetaResults'
# folder = 'FixedProf25_NoAgg_UncMono_VaryBeta_AllSubs'
# results_directory = '../results/' + folder + '/'

# if not os.path.isdir(results_directory):
#     os.mkdir(results_directory)

max_delay = 120
exponent = 2
flights = flight.read_flights(oag_file, 'JFK',
                              pandas.to_datetime(year + '-' + month + '-' + day + '-'), 15)
subauctions = 'all'
# subauctions = 'none'
# subauctions=['UAL']
g = 7.5
gamma_f = {f.flight_id: g for f in flights}
alpha_f = {f.flight_id: 0 for f in flights}
# beta_f = {f.flight_id:1 for f in flights}
mc_multiplier = 1


def typical_monopoly_func(x: float) -> float:
    if x <= 0.25:
        return 0
    else:
        return 2 * math.pow(4 / 3, 1.5) * (x - 0.25) ** 1.5


# monopoly_benefit_func = None
monopoly_benefit_func = typical_monopoly_func

peak_time_range = range(62, 70)
# peak_time_range = None
# monopoly_constraint_rate = 0.4
monopoly_constraint_rate = None

# profiles = [tuple([20] * 4 * 24)]
profiles = [tuple([i] * 4 * 24) for i in range(17, 27)]

with open(connections_file, 'rb') as connect_pickle:
    connections = pickle.load(connect_pickle)
    print(len(connections))

default_allocation = flight.get_aggregated_flight_schedule(flights, 96)
for i in range(0, 96):
    print("(", i, ",", default_allocation[i], ")")

if airline_aggregation == 'high':
    for f in flights:
        if f.airline in {'JBU', 'DAL', 'AAL'}:
            f.airline = 'JBU'
elif airline_aggregation == 'med':
    for f in flights:
        if f.airline in {'JBU', 'DAL'}:
            f.airline = 'JBU'

if subauctions == 'all':
    airline_subauctions = {f.airline for f in flights}
else:
    airline_subauctions = subauctions

scenarios = numpy.genfromtxt(scen_file, skip_header=1, delimiter=',')
move_costs = qcarun.make_move_costs(flights, gamma_f, 96, exponent)
max_displacement = 4
remove_costs = qcarun.make_remove_costs(flights, max_displacement, 4 * 24, gamma_f, exponent)

print("Running auction")
print(numpy.log2([0.0625, 0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0]))

# [0.0625, 0.125, 0.25, 0.5, 1.0, 2.0, 4.0]
for b in [1.0]:
    beta_f = {f.flight_id: b for f in flights}
    auction_params = qcarun.AuctionRunParams(flights=flights,
                                             connections=connections,
                                             profiles=profiles,
                                             max_displacement=max_displacement,
                                             min_connect=4,
                                             max_connect=16,
                                             turnaround=4,
                                             max_delay=max_delay,
                                             alpha_f=alpha_f,
                                             beta_f=beta_f,
                                             gamma_f=gamma_f,
                                             exponent=exponent,
                                             scenarios=scenarios,
                                             run_subauctions=False,
                                             validate=True,
                                             peak_time_range=peak_time_range,
                                             monopoly_benefit_func=monopoly_benefit_func,
                                             monopoly_constraint_rate=monopoly_constraint_rate,
                                             verbose=False,
                                             max_iter=100,
                                             warm_model=None,
                                             delay_threshold=0
                                             )
    results = qcarun.run_pricing_auction(auction_params)
    print(results)

    # run_info_string = 'OD trial.\nScenarios file: ' + scen_file
    # run_info_string += '\nConnections File: ' + connections_file
    # run_info_string += '\nFlights File: ' + oag_file
    # run_info_string += '\nParameters: monopoly benefits. Beta_f = ' + str(b) + ' for all flights.'
    # run_info_string += ' Alpha_f = 0 for all flights. Gamma_f = ' + str(g) + ' for all flights. Profiles from 17-25. '
    # run_info_string += ' Airline subauctions run: ' + str(subauctions) + '. '
    # run_info_string += ' Airline aggregation level: ' + airline_aggregation + '. '
    # run_info_string += 'See results["param_info"] for other parameters.'
    #
    # results = auction_tests.price_auctions(flights=flights,
    #                                        connections=connections,
    #                                        profiles=profiles,
    #                                        max_displacement=max_displacement,
    #                                        min_connect=4,
    #                                        max_connect=16,
    #                                        turnaround=4,
    #                                        max_delay=max_delay,
    #                                        airline_subauctions=airline_subauctions,
    #                                        alpha_f=alpha_f,
    #                                        beta_f=beta_f,
    #                                        gamma_f=gamma_f,
    #                                        exponent=exponent,
    #                                        scenarios=scenarios,
    #                                        verbose=False,
    #                                        validate=True,
    #                                        monopoly_benefit_func=monopoly_benefit_func,
    #                                        monopoly_constraint_rate=monopoly_constraint_rate,
    #                                        peak_time_range=peak_time_range,
    #                                        max_iterations=50,
    #                                        delay_threshold=0)

    # results['run_info'] = run_info_string

#     with open(results_directory + 'Trial_' + year + '_' + month + '_' + day + '_B' + str(b) + '_Exp' + str(
#             exponent) + '.p', 'wb') as my_file:
#         pickle.dump(results, my_file)
# #
matplotlib.pyplot.show()
