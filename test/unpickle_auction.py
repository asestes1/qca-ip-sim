# -*- coding: utf-8 -*-
"""
Created on Tue Oct 31 18:20:00 2017

@author: Alex
"""
import sys
import os
sys.path.append(os.path.join(__file__,os.pardir))

import pickle
import ast
import os
import qca.flight as flight
import pandas

year='2007'
month='02'
day='04'
data_folder = os.path.abspath(os.path.join(
                                os.path.join(os.path.join(__file__,os.pardir),
                                os.pardir),'resources'))
oag_file = os.path.join(data_folder,'OAG-CSV-ID/OAG_JFK_'+year+'_'+month+'_ID.csv')
#connections_file = os.path.join(data_folder,'Connections/Connection_'+year+'_'+month+'_'+day+'.p')
scen_file = os.path.join(data_folder,'scenariosJFK.csv')
#results_directory = '../run_results/HighAgg_UncontMono_VaryBeta_AllSubs/'
#
#max_delay = 120
#exponent=2
flights = flight.read_flights(oag_file,'JFK',
                                          pandas.to_datetime(year+'-'+month+'-'+day+'-'),15)
#
#scenarios = numpy.genfromtxt(scen_file,skip_header = 1,delimiter=',')
#separate_delay_results = auction_costs.get_queue_delays(scenarios,flights,96, Del_T = 15,separate_flights=True)
#max_arr_q_length = numpy.dot(numpy.amax(separate_delay_results['arr_q_lengths'],axis=1),
#                         separate_delay_results['prob'].ravel())
#max_dep_q_length = numpy.dot(numpy.amax(separate_delay_results['dep_q_lengths'],axis=1),
#                         separate_delay_results['prob'].ravel())
#print(max_arr_q_length)
#print(max_dep_q_length)
#gamma_f = {f.flight_id:10 for f in flights}
#move_costs = auction_costs.make_move_costs(flights,gamma_f,96,exponent)
#
#folder ='NoAgg_ContMono_VaryBeta_AllSubs'
#folder ='NoAgg_UncontMono_VaryBeta_AllSubs'
#folder ='MedAgg_ContMono_VaryBeta_AllSubs'
#folder ='MedAgg_UncontMono_VaryBeta_AllSubs'
#folder ='HighAgg_ContMono_VaryBeta_AllSubs'
#folder ='HighAgg_UncontMono_VaryBeta_AllSubs'
#folder='VaryingBetaAllSubauctions'
#folder = 'JOVaryBetaResults'
folder = 'FixedProf17_NoAgg_UncMono_VaryBeta_AllSubs'


print_summary = False
print_new_schedule = False
print_old_schedule = False

print_summary = True
#print_new_schedule = True
#print_old_schedule = True

if print_summary:
    for beta in [0.0625,0.125,0.25,0.5,1.0,2.0,4.0,8.0,16.0]:
#     for beta in [1.0]:
        with open("../results/"+folder+"/Trial_"+year+"_"+month+"_"+day+"_B"+str(beta)+"_Exp2.p","rb") as my_file:
            my_results = pickle.load(my_file)
            print(my_results['summary_string'])
    #        print(my_results['run_info'])
    #        print(my_results['param_info'])
    #        print(len(my_results['best_schedules']['FULL_AUCTION']))
            
            fields = ['Expected Max Arr Queue',
                      'Expected Max Dep Queue',
                      'Expected Max Queue',
                      'Total Displacements',
                      'Max Displacements',
                      'Removals in Auction',
                      'Best profile',
                      'Total Expected Delay in Auction',
                      'Calculated Schedule Value (minus monopoly benefits)',
                      'Calculated Monopoly Value',
                      'Total Calculated Schedule Value',
                      'Total payments',
                      'Net value of auction schedule to airlines',
                      'Value of current schedule (not included monopoly benefits)',
                      'New Delays by Scenario',
                      ]
            field_dict = {}
            for i,line in enumerate(str.split(my_results['summary_string'],"\n")):
                if(len(line.strip()) != 0):
                    field_value = str.split(line,":")
                    field = field_value[0].strip()
                    if(field == 'Best profile'):
                        field_dict[field] = str(ast.literal_eval(field_value[1].strip())[0])
                    elif(field == 'New Delays by Scenario'):
                        field_dict[field] = str(ast.literal_eval(field_value[1].strip())[6])
                    elif(field in fields):
                        field_dict[field] = field_value[1].strip()
                
            output = ""        
            for f in fields:
                output+=field_dict[f]+"\t"
            print(output)

if print_new_schedule:
    for beta in [8.0]:
        with open("../run_results/"+folder+"/Trial_2007_02_04_B"+str(beta)+"_Exp2.p","rb") as my_file:
            my_results = pickle.load(my_file)
            new_flights = my_results['best_schedules']['FULL_AUCTION']
            new_flights = {f for f in new_flights if f.airline in {'JBU','DAL'}}
            new_allocation =  flight.get_aggregated_flight_schedule(new_flights,96)
                
            for i in range(48,96):
                print((i,new_allocation[i]))
                    
if print_old_schedule:
    old_flights = flights
#    old_flights = {f for f in flights if f.airline in {'JBU','DAL'}}
    old_allocation =  flight.get_aggregated_flight_schedule(old_flights,96)
    for i in range(0,96):
        print((i,old_allocation[i]))
#        airlines = {f.airline for f in new_flights}
#    peak_flights_by_airline = {a:len({f for f in new_flights if f.airline==a and f.slot_time in range(62,70)}) for a in airlines}
#    for a,v in peak_flights_by_airline.items():
#        print(a,v)
#    new_slot_times = {f.flight_id:f.slot_time for f in new_flights}
#    old_slot_times = {f.flight_id:f.slot_time for f in flights}
