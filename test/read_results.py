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

year = '2007'
month = '02'
day = '04'
results_folder = os.path.abspath(os.path.join(os.path.join(os.path.join(__file__, os.pardir), os.pardir), 'results'))

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

files = ['results_b'+str(b)+'.out' for b in [0.0625, 0.125, 0.25, 0.5, 1.0, 2.0]]
for f in files:
    with open(
            os.path.join(os.path.join(results_folder, 'NoAgg_ContMono_VaryBeta_AllSubs2007_02_04'), f),
            'rb') as my_file:
        my_results: qcarun.AuctionResultStruct = pickle.load(my_file)
        print(my_results.params.peak_time_range, my_results.params.rmax, my_results.params.delta, my_results.params.kappa)
        print(my_results.best_profile)
        print(my_results.ipval)
        print(sum(qcarun.get_schedule_value_without_monopoly(my_results.best_schedule.schedule, my_results.params).values()))
        print(sum(qcarun.get_schedule_monopoly_value(my_results.best_schedule.schedule, my_results.best_profile, my_results.params).values()))
        print(sum(my_results.payments.values()))
        # print(my_results.params)
