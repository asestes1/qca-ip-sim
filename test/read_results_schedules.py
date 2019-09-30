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


year = '2007'
month = '02'
day = '04'
results_folder = os.path.abspath(os.path.join(os.path.join(os.path.join(__file__, os.pardir), os.pardir), 'results'))
file = 'results_b' + str(2.0) + '.out'
airline='DAL'
agg = 'NoAgg'
with open(os.path.join(os.path.join(results_folder,
                                    agg+'_UncontMono_VaryBeta_AllSubs2007_02_04'), file),'rb') as my_file:
    my_results: qcarun.AuctionResultStruct = pickle.load(my_file)
    orig_schedule = my_results.params.flights
    uncont_schedule = my_results.best_schedule.schedule
    print(flight.get_airline_market_shares(my_results.best_schedule.schedule,
                                           peak_time_range=my_results.params.peak_time_range,
                                           profile=my_results.best_profile))
with open(os.path.join(os.path.join(results_folder,
                                    agg+'_ContMono_VaryBeta_AllSubs2007_02_04'), file),'rb') as my_file:
    my_results: qcarun.AuctionResultStruct = pickle.load(my_file)
    cont_schedule = my_results.best_schedule.schedule
    print(flight.get_airline_market_shares(my_results.best_schedule.schedule, peak_time_range=my_results.params.peak_time_range,
                                           profile=my_results.best_profile))
with open(os.path.join(os.path.join(results_folder,
                                    agg+'_NoMono_VaryBeta_AllSubs2007_02_04'), file),'rb') as my_file:
    my_results: qcarun.AuctionResultStruct = pickle.load(my_file)
    nomono = my_results.best_schedule.schedule

for schedule, name in zip((orig_schedule, uncont_schedule, cont_schedule, nomono), ('original', 'uncont','cont', 'nomono')):
    if airline is not None:
        other_schedule = {f for f in schedule if f.airline != airline}
        schedule = {f for f in schedule if f.airline == airline}
        print(name,"OTHERS")
        agg_schedule = flight.get_aggregated_flight_schedule(other_schedule, 96, separate_flights=False)
        for i in range(48, 96):
            print((i, agg_schedule[i]))

    if airline is not None:
        print(name, airline)
    else:
        print(name)
    agg_schedule = flight.get_aggregated_flight_schedule(schedule, 96, separate_flights=False)
    for i in range(48,96):
        print((i,agg_schedule[i]))

