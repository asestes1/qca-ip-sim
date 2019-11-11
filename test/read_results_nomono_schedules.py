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

agg = 'NoAgg'



for b in ["0.125", "1.0", "2.0"]:
    file = 'results_b' + str(b) + '.out'
    with open(os.path.join(os.path.join(results_folder,
                                        agg + '_NoMono_VaryBeta_AllSubs2007_02_04'), file), 'rb') as my_file:
        print("------------  "+b+"  ------------------")
        my_results: qcarun.AuctionResultStruct = pickle.load(my_file)
        schedule = my_results.best_schedule.schedule
        agg_schedule = flight.get_aggregated_flight_schedule(schedule, 96, separate_flights=False)
        for i in range(48,96):
            print((i,agg_schedule[i]))

