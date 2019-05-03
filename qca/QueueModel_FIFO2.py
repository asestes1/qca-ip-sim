# -*- coding: utf-8 -*-
"""
Created on Thu Jul  7 22:42:29 2016

@author: lyn
"""
from __future__ import division
import numpy as np
import matplotlib.pyplot as plt



class FIFO_QMODEL:
    def __init__(self,DEMAND, Max_Delay, CAPACITY):
        self.DEMAND = DEMAND
        self.Max_Delay = Max_Delay
        self.CAPACITY = CAPACITY
        self.Delay, self.Cancel, self.Assign, self.Queue = self.GetAssignment()

    def GetAssignment(self):
        Assign = {}

        Cancel = np.zeros(len(self.DEMAND))
        Delay = np.zeros(len(self.DEMAND))
        Queue = np.zeros(len(self.CAPACITY))
        Demand = self.DEMAND.copy()
        Capacity = self.CAPACITY.copy()

        for period in range(len(Demand)):
            Assign[period] = {}
            
            CapCum = Capacity[period:].cumsum()
            MaxSlot = np.where( CapCum >= Demand[period] )[0][0] + period
            if MaxSlot - period > self.Max_Delay:
                Demand[period] = CapCum[self.Max_Delay]
                MaxSlot = period + self.Max_Delay
                
#            cum_throughput = 0.0
            for cur in range(period, MaxSlot + 1):
                throughput = min(Capacity[cur],Demand[period])
        #         print(throughput)
                Queue[cur] += (Demand[period] - throughput)
                Assign[period][cur] = throughput
                Demand[period] -= throughput
                Capacity[cur] -= throughput
#                cum_throughput += throughput

        for period in Assign.keys():
            if self.DEMAND[period] != 0.0:
                N_Canc = self.DEMAND[period] - sum([t[1] for t in Assign[period].items()])
                Cancel[period] = N_Canc/self.DEMAND[period]
                Delay[period] = sum([((t[0]-period)*t[1])/(self.DEMAND[period]-N_Canc) for t in Assign[period].items()])
            else:
                pass
#            
#        print("Total Delay: ",sum(Delay[period]*(1-Cancel[period])*self.DEMAND[period] for period in Assign.keys()))
#        print("Total Queue: ",sum(Queue))    
        return Delay, Cancel, Assign, Queue
        
class FIFO_QMODEL_SPLIT:
    def __init__(self,ARR_DEMAND, DEP_DEMAND, Max_Delay, CAPACITY):
        self.ARR_DEMAND = ARR_DEMAND
        self.DEP_DEMAND = DEP_DEMAND
        self.Max_Delay = Max_Delay
        self.CAPACITY = CAPACITY
        (self.Arr_Delay, self.Dep_Delay, self.Arr_Cancel, self.Dep_Cancel,
         self.Arr_Assign, self.Dep_Assign, self.Arr_Queue, self.Dep_Queue) = self.GetAssignment()

    def GetAssignment(self):
        Arr_Assign = {}
        Dep_Assign = {}
        Arr_Demand = self.ARR_DEMAND.copy()
        Dep_Demand = self.DEP_DEMAND.copy()
        
        Arr_Cancel = np.zeros(len(Arr_Demand))
        Dep_Cancel = np.zeros(len(Dep_Demand))  
        Arr_Delay = np.zeros(len(Arr_Demand))
        Dep_Delay = np.zeros(len(Dep_Demand))
        Dep_Queue = np.zeros(len(self.CAPACITY))
        Arr_Queue = np.zeros(len(self.CAPACITY))
        
        Capacity = self.CAPACITY.copy()

        for period in range(len(Arr_Demand)):
            Arr_Assign[period] = {}
            Dep_Assign[period] = {}

            
            CapCum = Capacity[period:].cumsum()
            MaxSlot = np.where( CapCum >= Arr_Demand[period]+Dep_Demand[period] )[0][0] + period
            if MaxSlot - period > self.Max_Delay:
                RemovedFlights = Arr_Demand[period]+Dep_Demand[period]-CapCum[self.Max_Delay]
                if(Arr_Demand[period] >= RemovedFlights/2 and Dep_Demand[period] >= RemovedFlights/2):
                    Arr_Demand[period] -= RemovedFlights/2
                    Dep_Demand[period] -= RemovedFlights/2
                elif(Dep_Demand[period] < RemovedFlights/2):
                    Arr_Demand[period] -= (RemovedFlights-Dep_Demand[period])
                    Dep_Demand[period] = 0
                else:
                    Dep_Demand[period] -= (RemovedFlights-Arr_Demand[period])
                    Arr_Demand[period] = 0
                MaxSlot = period + self.Max_Delay
                
            for cur in range(period, MaxSlot + 1):
                total_throughput = min(Capacity[cur],Arr_Demand[period]+Dep_Demand[period])
                if(Arr_Demand[period] >= total_throughput/2 and Dep_Demand[period] >= total_throughput/2):
                    arr_throughput = total_throughput/2
                    dep_throughput = total_throughput/2
                elif(Dep_Demand[period] < total_throughput/2):
                    arr_throughput = total_throughput-Dep_Demand[period]
                    dep_throughput = Dep_Demand[period]
                else:
                    dep_throughput = total_throughput-Arr_Demand[period]
                    arr_throughput = Arr_Demand[period]

                Arr_Queue[cur] += (Arr_Demand[period] - arr_throughput)
                Arr_Assign[period][cur] = arr_throughput
                Arr_Demand[period] -= arr_throughput
                
                Dep_Queue[cur] += (Dep_Demand[period] - dep_throughput)
                Dep_Assign[period][cur] = dep_throughput
                Dep_Demand[period] -= dep_throughput

                Capacity[cur] -= total_throughput

        for period in Arr_Assign.keys():
            if self.ARR_DEMAND[period] != 0.0:
                N_Arr_Canc = self.ARR_DEMAND[period] - sum([t[1] for t in Arr_Assign[period].items()])
                Arr_Cancel[period] = N_Arr_Canc/self.ARR_DEMAND[period]
                Arr_Delay[period] = sum([((t[0]-period)*t[1])/(self.ARR_DEMAND[period]-N_Arr_Canc) for t in Arr_Assign[period].items()])
            else:
                pass
            
        for period in Dep_Assign.keys():
            if self.DEP_DEMAND[period] != 0.0:
                N_Dep_Canc = self.DEP_DEMAND[period] - sum([t[1] for t in Dep_Assign[period].items()])
                Dep_Cancel[period] = N_Dep_Canc/self.DEP_DEMAND[period]
                Dep_Delay[period] = sum([((t[0]-period)*t[1])/(self.DEP_DEMAND[period]-N_Dep_Canc) for t in Dep_Assign[period].items()])
            else:
                pass
#            
#        print("Total Delay: ",sum(Delay[period]*(1-Cancel[period])*self.DEMAND[period] for period in Assign.keys()))
#        print("Total Queue: ",sum(Queue))    
        return Arr_Delay, Dep_Delay, Arr_Cancel, Dep_Cancel,Arr_Assign, Dep_Assign,Arr_Queue, Dep_Queue
        
        
##
# TEST CODE
#U = 2 * 4
#T = np.arange(24*4+U+1)
#Del_T = 15 # min
#
#Cap = np.genfromtxt('scenariosJFK.csv',skip_header = 1,delimiter=',')
#Prob = Cap[:,0]
#Cap = Cap[:,1:]
#
#Capacity = np.zeros((Cap.shape[0],len(T)))
#for i in range(Capacity.shape[0]):
#    for j in range(Capacity.shape[1]):
#        if j //4 < Cap.shape[1]:
#            Capacity[i][j] = Cap[i][j//4]/4
#        else:
#            Capacity[i][j] = Cap[i][j//4-Cap.shape[1]]/4
#    Capacity[i][-1] = 9999
#    
#Demand = np.array([5,   4,   3,   3,   1,   0,   2,   1,   1,   0,   0,   0,   0,   2,   0,
#   1,   2,   0,   0,   1,   6,   1,   3,   4,   8,  10,   3,   7,  15,   9,
#  14,  10,  21,  11,  16,  14,  15,   7,   8,   7,  13,   9,   5,  13,   4,
#   7,  19,   9,  17,  12,  15,  23,  14,  13,  11,   7,  18,  10,   9,  12,
#  14,  17,  23,  22,  29,  23,  24,  24,  21,  21,  26,  17,  20,  17,  15,
#  11,  20,  25,  19,  15,  18,  18,  19,  13,  26,  17,  15,  13,  12,   1,
#  14,   9,   4,   5,   4,   8,   0,   0,   0,   0,   0,   0,   0,   0,   0],dtype=np.float64)
##Demand = np.array([25]*len(T))
#
#AvgDelay = np.zeros((Capacity.shape[0],len(T)-U-1))
#P_canc = np.zeros((Capacity.shape[0],len(T)-U-1))
#temp = []
#for i in range(Capacity.shape[0]):
#    A = FIFO_QMODEL(Demand, U, Capacity[i])
#    AvgDelay[i] = A.Delay[:24*4]
#    P_canc[i] =  A.Cancel[:24*4]
#
#plt.plot(Prob.dot(AvgDelay))
#plt.plot(Prob.dot(P_canc))
