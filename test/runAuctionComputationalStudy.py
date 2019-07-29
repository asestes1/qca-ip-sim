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
results_folder = os.path.abspath(os.path.join(os.path.join(os.path.join(__file__, os.pardir), os.pardir), 'results'))

oag_file = os.path.join(data_folder, 'OAG-CSV-ID/OAG_JFK_' + year + '_' + month + '_ID.csv')
connections_file = os.path.join(data_folder, 'Connections/Connection_' + year + '_' + month + '_' + day + '.p')
scen_file = os.path.join(data_folder, 'scenariosJFK.csv')

subfolder = 'NoAgg_ContMono_VaryBeta_AllSubs'
# subfolder ='NoAgg_UncontMono_VaryBeta_AllSubs'
# subfolder ='MedAgg_ContMono_VaryBeta_AllSubs'
# subfolder ='MedAgg_UncontMono_VaryBeta_AllSubs'
# subfolder ='HighAgg_ContMono_VaryBeta_AllSubs'
# subfolder ='HighAgg_UncontMono_VaryBeta_AllSubs'
# subfolder = 'JOVaryBetaResults'
subfolder += year + '_' + month + '_' + day
results_subdirectory = os.path.abspath(os.path.join(results_folder, subfolder))
if not os.path.isdir(results_subdirectory):
    os.mkdir(results_subdirectory)

max_delay = 120
exponent = 2
flights = flight.read_flights(oag_file, 'JFK',
                              pandas.to_datetime(year + '-' + month + '-' + day + '-'), 15)
g = 7.5
a = 0
gamma_f = {f.flight_id: g for f in flights}
alpha_f = {f.flight_id: a for f in flights}
# beta_f = {f.flight_id:1 for f in flights}

rmax = 2
delta = 0.25
kappa = 1.5
# rmax = delta = kappa = None

peak_time_range = range(62, 70)
# peak_time_range = None
monopoly_constraint_rate = 0.4
# monopoly_constraint_rate = None

profiles = [tuple([i] * 4 * 24) for i in range(17, 27)]
with open(connections_file, 'rb') as connect_pickle:
    connections = pickle.load(connect_pickle)

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

min_connect = 4
max_connect = 16
turnaround = 4
scenarios = numpy.genfromtxt(scen_file, skip_header=1, delimiter=',')
move_costs = qcarun.make_move_costs(flights, gamma_f, 96, exponent)
max_displacement = 4
remove_costs = qcarun.make_remove_costs(flights, max_displacement, 4 * 24, gamma_f, exponent)
run_subauctions = True

print("Running auction")
print(numpy.log2([0.0625, 0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0]))

# [0.0625, 0.125, 0.25, 0.5, 1.0, 2.0, 4.0]
for b in [0.0625]:
    beta_f = {f.flight_id: b for f in flights}
    auction_params = qcarun.AuctionRunParams(flights=flights,
                                             connections=connections,
                                             profiles=profiles,
                                             max_displacement=max_displacement,
                                             min_connect=min_connect,
                                             max_connect=max_connect,
                                             turnaround=turnaround,
                                             max_delay=max_delay,
                                             alpha_f=alpha_f,
                                             beta_f=beta_f,
                                             gamma_f=gamma_f,
                                             exponent=exponent,
                                             scenarios=scenarios,
                                             run_subauctions=run_subauctions,
                                             validate=False,
                                             peak_time_range=peak_time_range,
                                             rmax=rmax,
                                             kappa=kappa,
                                             delta=delta,
                                             monopoly_constraint_rate=monopoly_constraint_rate,
                                             verbose=False,
                                             max_iter=100,
                                             delay_threshold=0)
    results = qcarun.run_pricing_auction(auction_params)
    filename = os.path.abspath(os.path.join(results_subdirectory, 'results_b' + str(b) + '.out'))
    with open(filename, 'wb') as myfile:
        pickle.dump(results, myfile)

matplotlib.pyplot.show()
