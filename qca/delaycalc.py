# -*- coding: utf-8 -*-
"""
Created on Thu Jul  7 22:42:29 2016

@author: Yulin Liu & Alex Estes
"""
from __future__ import division
import numpy as np
import attr
import numpy
import typing
from . import flightsched

@attr.s(kw_only=True)
class DelayResults(object):
    delay = attr.ib(type=numpy.ndarray)
    cancel = attr.ib(type=numpy.ndarray)
    assign = attr.ib(type=typing.Dict[int, typing.Dict[int, float]])
    queue = attr.ib(type=numpy.ndarray)

def get_queue_delays(demand, max_delay, capacity):
    assign = {}

    cancel = np.zeros(len(demand))
    delay = np.zeros(len(demand))
    queue = np.zeros(len(capacity))
    rem_demand = demand.copy()
    rem_capacity = capacity.copy()

    for period in range(len(rem_demand)):
        assign[period] = {}

        cap_cum = rem_capacity[period:].cumsum()
        max_slot = np.where(cap_cum >= rem_demand[period])[0][0] + period
        if max_slot - period > max_delay:
            rem_demand[period] = cap_cum[max_delay]
            max_slot = period + max_delay

        #            cum_throughput = 0.0
        for cur in range(period, max_slot + 1):
            throughput = min(rem_capacity[cur], rem_demand[period])
            #         print(throughput)
            queue[cur] += (rem_demand[period] - throughput)
            assign[period][cur] = throughput
            rem_demand[period] -= throughput
            rem_capacity[cur] -= throughput
    #                cum_throughput += throughput

    for period in assign.keys():
        if demand[period] != 0.0:
            n_canc = demand[period] - sum([t[1] for t in assign[period].items()])
            cancel[period] = n_canc / demand[period]
            delay[period] = sum(
                [((t[0] - period) * t[1]) / (demand[period] - n_canc) for t in assign[period].items()])
        else:
            pass
    #
    return DelayResults(delay=delay, cancel=cancel, assign=assign, queue=queue)


class FifoQModelSplit:
    def __init__(self, ARR_DEMAND, DEP_DEMAND, Max_Delay, CAPACITY):
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
            MaxSlot = np.where(CapCum >= Arr_Demand[period] + Dep_Demand[period])[0][0] + period
            if MaxSlot - period > self.Max_Delay:
                RemovedFlights = Arr_Demand[period] + Dep_Demand[period] - CapCum[self.Max_Delay]
                if Arr_Demand[period] >= RemovedFlights / 2 and Dep_Demand[period] >= RemovedFlights / 2:
                    Arr_Demand[period] -= RemovedFlights / 2
                    Dep_Demand[period] -= RemovedFlights / 2
                elif Dep_Demand[period] < RemovedFlights / 2:
                    Arr_Demand[period] -= (RemovedFlights - Dep_Demand[period])
                    Dep_Demand[period] = 0
                else:
                    Dep_Demand[period] -= (RemovedFlights - Arr_Demand[period])
                    Arr_Demand[period] = 0
                MaxSlot = period + self.Max_Delay

            for cur in range(period, MaxSlot + 1):
                total_throughput = min(Capacity[cur], Arr_Demand[period] + Dep_Demand[period])
                if Arr_Demand[period] >= total_throughput / 2 and Dep_Demand[period] >= total_throughput / 2:
                    arr_throughput = total_throughput / 2
                    dep_throughput = total_throughput / 2
                elif Dep_Demand[period] < total_throughput / 2:
                    arr_throughput = total_throughput - Dep_Demand[period]
                    dep_throughput = Dep_Demand[period]
                else:
                    dep_throughput = total_throughput - Arr_Demand[period]
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
                Arr_Cancel[period] = N_Arr_Canc / self.ARR_DEMAND[period]
                Arr_Delay[period] = sum([((t[0] - period) * t[1]) / (self.ARR_DEMAND[period] - N_Arr_Canc) for t in
                                         Arr_Assign[period].items()])
            else:
                pass

        for period in Dep_Assign.keys():
            if self.DEP_DEMAND[period] != 0.0:
                N_Dep_Canc = self.DEP_DEMAND[period] - sum([t[1] for t in Dep_Assign[period].items()])
                Dep_Cancel[period] = N_Dep_Canc / self.DEP_DEMAND[period]
                Dep_Delay[period] = sum([((t[0] - period) * t[1]) / (self.DEP_DEMAND[period] - N_Dep_Canc) for t in
                                         Dep_Assign[period].items()])
            else:
                pass
        return Arr_Delay, Dep_Delay, Arr_Cancel, Dep_Cancel, Arr_Assign, Dep_Assign, Arr_Queue, Dep_Queue


@attr.s(kw_only=True)
class CombinedQueueResults(object):
    avg_delay = attr.ib(type=numpy.ndarray)
    p_canc = attr.ib(type=numpy.ndarray)
    prob = attr.ib(type=numpy.ndarray)
    q_lengths = attr.ib(type=numpy.ndarray)


def get_combined_qdelays(scenarios, flights, n_slots, del_t=15, u=2 * 4) -> CombinedQueueResults:
    timehorizon = len(numpy.arange(n_slots + u + 1))

    prob = scenarios[:, 0]
    cap = scenarios[:, 1:]

    capacity = numpy.zeros((cap.shape[0], timehorizon))
    for i in range(capacity.shape[0]):
        for j in range(capacity.shape[1]):
            if j // 4 < cap.shape[1]:
                capacity[i][j] = cap[i][j // 4] / 4
            else:
                capacity[i][j] = cap[i][j // 4 - cap.shape[1]] / 4
        capacity[i][-1] = 9999

    profile = flightsched.get_aggregated_flight_schedule(flights, n_slots, separate_flights=False)
    demand = list(profile)
    demand.extend([0.0] * (u + 1))
    demand = numpy.array(demand)

    avg_delay = numpy.zeros((capacity.shape[0], timehorizon))
    p_canc = numpy.zeros((capacity.shape[0], timehorizon))
    q_lengths = numpy.zeros((capacity.shape[0], timehorizon))
    for i in range(capacity.shape[0]):
        qmodel = get_queue_delays(demand, u, capacity[i])
        avg_delay[i] = qmodel.delay * del_t
        p_canc[i] = qmodel.cancel
        q_lengths[i] = qmodel.queue
    return CombinedQueueResults(avg_delay=avg_delay, prob=prob, p_canc=p_canc, q_lengths=q_lengths)


@attr.s(kw_only=True)
class SeparateQueueResults(object):
    prob = attr.ib(type=numpy.ndarray)
    avg_arr_delay = attr.ib(type=numpy.ndarray)
    p_arr_canc = attr.ib(type=numpy.ndarray)
    arr_q_lengths = attr.ib(type=numpy.ndarray)
    avg_dep_delay = attr.ib(type=numpy.ndarray)
    p_dep_canc = attr.ib(type=numpy.ndarray)
    dep_q_lengths = attr.ib(type=numpy.ndarray)


def get_separate_qdelays(scenarios, flights, n_slots, del_t=15,
                         u=2 * 4) -> SeparateQueueResults:
    timehorizon = len(numpy.arange(n_slots + u + 1))

    prob = scenarios[:, 0]
    cap = scenarios[:, 1:]

    capacity = numpy.zeros((cap.shape[0], timehorizon))
    for i in range(capacity.shape[0]):
        for j in range(capacity.shape[1]):
            if j // 4 < cap.shape[1]:
                capacity[i][j] = cap[i][j // 4] / 4
            else:
                capacity[i][j] = cap[i][j // 4 - cap.shape[1]] / 4
        capacity[i][-1] = 9999

    profile = flightsched.get_aggregated_flight_schedule(flights, n_slots, separate_flights=True)
    arr_demand = list(profile['arrivals'])
    arr_demand.extend([0.0] * (u + 1))
    arr_demand = numpy.array(arr_demand)

    avg_arr_delay = numpy.zeros((capacity.shape[0], timehorizon))
    p_arr_canc = numpy.zeros((capacity.shape[0], timehorizon))
    arr_q_lengths = numpy.zeros((capacity.shape[0], timehorizon))

    dep_demand = list(profile['departures'])
    dep_demand.extend([0.0] * (u + 1))
    dep_demand = numpy.array(dep_demand)

    avg_dep_delay = numpy.zeros((capacity.shape[0], timehorizon))
    p_dep_canc = numpy.zeros((capacity.shape[0], timehorizon))
    dep_q_lengths = numpy.zeros((capacity.shape[0], timehorizon))
    for i in range(capacity.shape[0]):
        qmodel = FifoQModelSplit(arr_demand, dep_demand, u, capacity[i])
        avg_arr_delay[i] = qmodel.Arr_Delay * del_t
        p_arr_canc[i] = qmodel.Arr_Cancel
        arr_q_lengths[i] = qmodel.Arr_Queue

        avg_dep_delay[i] = qmodel.Dep_Delay * del_t
        p_dep_canc[i] = qmodel.Dep_Cancel
        dep_q_lengths[i] = qmodel.Dep_Queue

    return SeparateQueueResults(prob=prob, avg_arr_delay=avg_arr_delay, avg_dep_delay=avg_dep_delay,
                                p_arr_canc=p_arr_canc, p_dep_canc=p_dep_canc, dep_q_lengths=dep_q_lengths,
                                arr_q_lengths=arr_q_lengths)
