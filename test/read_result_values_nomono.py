# -*- coding: utf-8 -*-
"""
Created on Tue Oct 31 18:20:00 2017

@author: Alex
"""
import sys
import os

sys.path.append(os.path.join(__file__, os.pardir))

import pickle
import ast
import os
import qca.flightsched as flight
import qca.delaycalc as delaycalc
import qca.qcarun as qcarun
import pandas
import numpy
import math

year = '2007'
month = '02'
day = '04'
results_folder = os.path.abspath(os.path.join(os.path.join(os.path.join(__file__, os.pardir), os.pardir), 'results'))
b_vals = [0.0625, 0.125, 0.25, 0.5, 1.0, 2.0]
log_bvals = [-4, -3, -2, -1, 0, 1]
files = ['results_b' + str(b) + '.out' for b in b_vals]
series = {'original', 'nomono'}
total_value = {s: [] for s in series}
social_value = {s: [] for s in series}
net_value = {s: [] for s in series}
num_flights = {s: [] for s in series}
expected_delay = {s: [] for s in series}
agg = "NoAgg"
for f in files:
    with open(os.path.join(os.path.join(results_folder, agg + '_NoMono_VaryBeta_AllSubs2007_02_04'), f),
              'rb') as my_file:
        my_results: qcarun.AuctionResultStruct = pickle.load(my_file)
        nomonovalue = (sum(
            qcarun.get_schedule_value_without_monopoly(my_results.best_schedule.schedule, my_results.params).values()))
        monovalue = (sum(qcarun.get_schedule_monopoly_value(my_results.best_schedule.schedule, my_results.best_profile,
                                                            my_results.params).values()))
        payments = sum(my_results.payments.values())
        total_value['nomono'].append(nomonovalue + monovalue)
        social_value['nomono'].append(nomonovalue)
        net_value['nomono'].append(nomonovalue + monovalue - payments)
        num_flights['nomono'].append(len(my_results.best_schedule.schedule))
        delay_results = delaycalc.get_combined_qdelays(scenarios=my_results.params.scenarios,
                                                       flights=my_results.best_schedule.schedule,
                                                       n_slots=96, del_t=15, u=2 * 4)
        agg_sched = flight.get_aggregated_flight_schedule(my_results.best_schedule.schedule, 96, separate_flights=False)
        exp_delay_slot = delay_results.prob.dot(delay_results.avg_delay)
        expected_delay['nomono'].append(numpy.array(agg_sched).dot(exp_delay_slot[0:96]))

        origvalue = (
            sum(qcarun.get_schedule_value_without_monopoly(my_results.params.flights, my_results.params).values()))
        total_value['original'].append(origvalue)
        social_value['original'].append(origvalue)
        net_value['original'].append(origvalue)
        num_flights['original'].append(len(my_results.params.flights))
        delay_results = delaycalc.get_combined_qdelays(scenarios=my_results.params.scenarios,
                                                       flights=my_results.params.flights,
                                                       n_slots=96, del_t=15, u=2 * 4)
        agg_sched = flight.get_aggregated_flight_schedule(my_results.params.flights, 96, separate_flights=False)
        exp_delay_slot = delay_results.prob.dot(delay_results.avg_delay)
        expected_delay['original'].append(numpy.array(agg_sched).dot(exp_delay_slot[0:96]))
for myresults, name in zip([total_value, social_value, net_value, num_flights, expected_delay],
                           ['total', 'social', 'net', 'flights', 'exp_delay']):
    for s, values in myresults.items():
        print(name, s)
        for b, v in zip(log_bvals, values):
            print((b, v))
        print(name + "_difference", s)
        for b, v, o_v in zip(log_bvals, values, myresults['original']):
            print((b, v - o_v))

        # print(my_results.params)
