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
profile = tuple([22]*4*24)
files = ['results_b' + str(b) + '.out' for b in b_vals]
series = {'original', 'uncont', 'cont', 'fixed', 'nomono'}
total_value = {s:[] for s in series}
social_value = {s:[] for s in series}
net_value = {s:[] for s in series}
num_flights = {s:[] for s in series}
agg="MedAgg"
for f in files:
    with open(os.path.join(os.path.join(results_folder, agg+'_UncontMono_VaryBeta_AllSubs2007_02_04'), f),
              'rb') as my_file:
        my_results: qcarun.AuctionResultStruct = pickle.load(my_file)

        origvalue = (
            sum(qcarun.get_schedule_value_without_monopoly(my_results.params.flights, my_results.params).values()))
        total_value['original'].append(origvalue)
        social_value['original'].append(origvalue)
        net_value['original'].append(origvalue)
        num_flights['original'].append(len(my_results.params.flights))

        nomonovalue = (sum(
            qcarun.get_schedule_value_without_monopoly(my_results.best_schedule.schedule, my_results.params).values()))
        monovalue = (sum(qcarun.get_schedule_monopoly_value(my_results.best_schedule.schedule, my_results.best_profile,
                                                            my_results.params).values()))
        payments = sum(my_results.payments.values())
        total_value['uncont'].append(nomonovalue+monovalue)
        social_value['uncont'].append(nomonovalue)
        net_value['uncont'].append(nomonovalue+monovalue - payments)
        num_flights['uncont'].append(len(my_results.best_schedule.schedule))

        fixedprof_results = qcarun.get_fixed_prof_payments(results=my_results, profile=profile)
        total_value['fixed'].append(fixedprof_results.social_value+fixedprof_results.mono_value)
        social_value['fixed'].append(fixedprof_results.social_value)
        net_value['fixed'].append(fixedprof_results.social_value+fixedprof_results.mono_value- fixedprof_results.payment)
        num_flights['fixed'].append(len(my_results.subaction_results[profile, None].schedule))
    with open(os.path.join(os.path.join(results_folder, agg+'_ContMono_VaryBeta_AllSubs2007_02_04'), f),
              'rb') as my_file:
        my_results: qcarun.AuctionResultStruct = pickle.load(my_file)
        nomonovalue = (sum(
            qcarun.get_schedule_value_without_monopoly(my_results.best_schedule.schedule, my_results.params).values()))
        monovalue = (sum(qcarun.get_schedule_monopoly_value(my_results.best_schedule.schedule, my_results.best_profile,
                                                            my_results.params).values()))
        payments = sum(my_results.payments.values())
        total_value['cont'].append(nomonovalue + monovalue)
        social_value['cont'].append(nomonovalue)
        net_value['cont'].append(nomonovalue+monovalue - payments)
        num_flights['cont'].append(len(my_results.best_schedule.schedule))
    with open(os.path.join(os.path.join(results_folder, agg+'_NoMono_VaryBeta_AllSubs2007_02_04'), f),
              'rb') as my_file:
        my_results: qcarun.AuctionResultStruct = pickle.load(my_file)
        nomonovalue = (sum(
            qcarun.get_schedule_value_without_monopoly(my_results.best_schedule.schedule, my_results.params).values()))
        monovalue = (sum(qcarun.get_schedule_monopoly_value(my_results.best_schedule.schedule, my_results.best_profile,
                                                            my_results.params).values()))
        payments = sum(my_results.payments.values())
        total_value['nomono'].append(nomonovalue + monovalue)
        social_value['nomono'].append(nomonovalue)
        net_value['nomono'].append(nomonovalue+monovalue - payments)
        num_flights['nomono'].append(len(my_results.best_schedule.schedule))
for myresults, name in zip([total_value, social_value, net_value, num_flights], ['total', 'social', 'net', 'flights']):
    for s, values in myresults.items():
        print(name, s)
        for b,v in zip(log_bvals,values):
            print((b,v))
        print(name+"_difference", s)
        for b, v, o_v in zip(log_bvals, values, myresults['original']):
            print((b, v-o_v))

        # print(my_results.params)
