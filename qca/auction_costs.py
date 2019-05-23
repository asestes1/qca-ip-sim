import itertools
import numpy
import gurobipy
import typing
from . import FIFO_QMODEL, FIFO_QMODEL_SPLIT
from . import flight
import math


def get_ip_value_by_airline(n_slots: int, flights: typing.Iterable[flight.Flight], model) -> typing.Dict[str, float]:
    auc_val_by_airline = {}
    for f in flights:
        if f.airline not in auc_val_by_airline:
            auc_val_by_airline[f.airline] = 0
        remove_var = model.getVarByName('R' + str(f.flight_id))
        if remove_var is None:
            print("Flight not in IP: ", f.flight_id)
        else:
            auc_val_by_airline[f.airline] -= remove_var.getAttr(gurobipy.GRB.attr.Obj) * (
                    1 - remove_var.getAttr(gurobipy.GRB.attr.X))
            for t in range(0, n_slots):
                res_var = model.getVarByName("F" + str(f.flight_id) + "T" + str(t))
                if res_var is not None:
                    auc_val_by_airline[f.airline] += (res_var.getAttr(gurobipy.GRB.attr.Obj)
                                                      * res_var.getAttr(gurobipy.GRB.attr.X))
    auc_val_by_airline['TOTAL'] = sum(auc_val_by_airline.values())
    return auc_val_by_airline


def get_aggregated_assignment_value(model, dict_function):
    return sum([dict_function[f, t] *
                model.getVarByName('F' + str(f) + 'T' + str(t))
               .getAttr(gurobipy.GRB.attr.X)
                for f, t in dict_function.keys()
                if model.getVarByName('F' + str(f) + 'T' + str(t)) is not None])


def get_aggregated_removal_value(model, dict_function):
    return sum([dict_function[f] *
                model.getVarByName('R' + str(f)).getAttr(gurobipy.GRB.attr.X)
                for f in dict_function.keys()
                if model.getVarByName('R' + str(f)) is not None])


def get_aggregated_flights(model, flights, num_times):
    return [len([f for f in flights
                 if model.getVarByName('F' + str(f.flight_id) + 'T' + str(t)) is not None
                 and abs(model.getVarByName('F' + str(f.flight_id) + 'T' +
                                            str(t)).getAttr(gurobipy.GRB.attr.X) - 1.0) < .0001])
            for t in range(0, num_times)]


def get_num_flights_removed(model, flights):
    return sum([model.getVarByName('R' + str(f.flight_id))
               .getAttr(gurobipy.GRB.attr.X)
                for f in flights
                if model.getVarByName('R' + str(f.flight_id)) is not None])


def make_remove_costs(flights, max_displacement, n_slots, gamma_f, exponent):
    """
    Creates a map that stores the cost of removal for all the flights

    :param flights - the flights in the auction
    :param max_displacement - number of time periods
    :param n_slots - integer
    :param gamma_f - dictionary. Keys: flights, values: gamma_f[f] gives gamma_f constant
    :returns map. Key: flight id, value: map[flight_id] gives the removal cost of flight
    """
    return {f.flight_id: cost_remove(f, max_displacement, n_slots, gamma_f, exponent)
            for f in flights}


def make_delay_costs(flights: typing.Iterable[flight.Flight], max_delay, alpha_f, beta_f,
                     cancel_probs, delays, scenario_probs, threshold=0):
    """
    Creates a map that stores the cost of delay for each flight at each time.
    flights - an iterable of flights
    max_delay - in minutes
    alpha_f,beta_f - dictionary. Keys: flights, values: alpha_f[f] gives alpha_f constant
    cancel_probs - array, number of scenarios by number of time periods
    delays - array, number of scenarios by number of time periods
    scenario_probs - array, number of scenarios

    Returns:
    map. Key: a pair (flight_id,time). Value: map[flight_id,time] is the delay cost
         for that flight assigned to that time.
    """
    values = {}
    n_slots = cancel_probs.shape[1]
    for f, t in itertools.product(flights, range(0, n_slots)):
        values[f.flight_id, t] = cost_congest(f, t, max_delay, alpha_f,
                                              beta_f, cancel_probs, delays,
                                              scenario_probs, threshold=threshold)
    return values


def make_move_costs(flights: typing.Iterable[flight.Flight], gamma_f: typing.Dict[flight.Flight, float], n_slots: int,
                    exponent: float):
    """
    Makes a map of the schedule adjustment cost of assigning a flight to a slot

    :param flights - an iterable of flights
    :param n_slots - the maximum number of slots that a flight can be moved
    :param gamma_f - dictionary. Keys: flights, values: gamma_f[f] gives gamma_f constant for flight f
    delays - array, number of scenarios by number of time periods
    scenario_probs - array, number of scenarios

    :returns map - Key: a pair (flight_id,time). Value: map[flight_id,time] is the cost
         for that flight assigned to that time.
    """
    values = {}
    for f, t in itertools.product(flights, range(0, n_slots)):
        values[f.flight_id, t] = cost_move(f, t, gamma_f, exponent)
    return values


def cost_remove(f, max_displacement, n_slots, gamma_f, exponent):
    """
    Calculate the cost of removing a flight.

    Arguments:
    f - the flights
    max_displacement - the max displacement of the flight in numbers of slots (int)
    n_slots - the number of possible time slots in the day (int)
    gamma_f - dictionary. Keys: flights, values: gamma_f[f] gives gamma_f constant

    Return:
    the cost of removing a flight (double).
    """
    earliest = max(f.slot_time - max_displacement, 0)
    latest = min(f.slot_time + max_displacement, n_slots - 1)
    return max(cost_move(f, earliest, gamma_f, exponent), cost_move(f, latest, gamma_f, exponent))


def cost_delay(f: flight.Flight, delay: int, alpha_f: typing.Dict[flight.Flight, float],
               beta_f: typing.Dict[flight.Flight, float], threshold: int = 0) -> float:
    """
    Calculate the cost of delaying a flight.

    Arguments:
    :param f - the flights
    :param delay - the amount of delay that a flight receives
    :param alpha_f - dictionary. Keys: flights, values: alpha_f[f] gives alpha_f constant
    :param beta_f - dictionary. Keys: flights, values: beta_f[f] gives beta_f constant

    :returns the cost of delaying the flight (float).
    """
    if delay < threshold:
        return 0
    else:
        return (alpha_f[f.flight_id] + beta_f[f.flight_id] * f.n_seats) * (delay - threshold)


def cost_cancel(f: flight.Flight, max_delay: int, alpha_f: typing.Dict[flight.Flight, float],
                beta_f: typing.Dict[flight.Flight, float],
                threshold: int = 0) -> float:
    """
    Calculate the cost of cancelling a flight.

    :param f - the flights
    :param max_delay - the maximum amount of delay that a flight can receive (in number of slots)
    :param alpha_f - dictionary. Keys: flights, values: alpha_f[f] gives alpha_f constant
    :param beta_f - dictionary. Keys: flights, values: beta_f[f] gives beta_f constant

    :returns the cost of cancelling the flight (double).
    """
    return cost_delay(f, max_delay, alpha_f, beta_f, threshold=threshold)


def cost_congest(f, t, max_delay, alpha_f, beta_f, cancel_probs, delays, scenario_probs, threshold=0):
    """
    Calculate the expected cost of congestion for a flight, given some capacity
    scenarios.

    :param f - the flight
    :param t - the time slot
    :param max_delay - the maximum amount of delay that a flight can receive
    :param alpha_f - dictionary. Keys: flights, values: alpha_f[f] gives alpha_f constant
    beta_f - dictionary. Keys: flights, values: beta_f[f] gives beta_f constant
    cancel_probs - a matrix, cancel_probs[w,t] gives the probability that an
                   airline would cancel a flight scheduled for the t time slot
                   scenario w
    scenario_probs - a vector, scenario_probs[w] gives the probability of scenario
                     p occurring.
    Return:
    the cost of cancelling the flight (double).
    """
    exp_value = 0
    for w, p in enumerate(scenario_probs):
        cancel_cost = cancel_probs[w, t] * cost_cancel(f, max_delay, alpha_f, beta_f, threshold=threshold)
        delay_cost = (1 - cancel_probs[w, t]) * cost_delay(f, delays[w, t],
                                                           alpha_f, beta_f, threshold=threshold)
        exp_value += p * (cancel_cost + delay_cost)
    return exp_value


def cost_move(f, t, gamma_f, exponent):
    return gamma_f[f.flight_id] * math.pow(abs(f.slot_time - t), exponent) * f.n_seats


def get_queue_delays(scenarios, flights, n_slots, separate_flights=False, Del_T=15, U=2 * 4):
    T = numpy.arange(n_slots + U + 1)

    Prob = scenarios[:, 0]
    Cap = scenarios[:, 1:]

    Capacity = numpy.zeros((Cap.shape[0], len(T)))
    for i in range(Capacity.shape[0]):
        for j in range(Capacity.shape[1]):
            if j // 4 < Cap.shape[1]:
                Capacity[i][j] = Cap[i][j // 4] / 4
            else:
                Capacity[i][j] = Cap[i][j // 4 - Cap.shape[1]] / 4
        Capacity[i][-1] = 9999

    if (separate_flights):
        profile = flight.get_aggregated_flight_schedule(flights, n_slots, separate_flights=True)
        Arr_Demand = list(profile['arrivals'])
        Arr_Demand.extend([0.0] * (U + 1))
        Arr_Demand = numpy.array(Arr_Demand)

        AvgArrDelay = numpy.zeros((Capacity.shape[0], len(T)))
        P_Arr_canc = numpy.zeros((Capacity.shape[0], len(T)))
        arr_q_lengths = numpy.zeros((Capacity.shape[0], len(T)))

        Dep_Demand = list(profile['departures'])
        Dep_Demand.extend([0.0] * (U + 1))
        Dep_Demand = numpy.array(Dep_Demand)

        AvgDepDelay = numpy.zeros((Capacity.shape[0], len(T)))
        P_Dep_canc = numpy.zeros((Capacity.shape[0], len(T)))
        dep_q_lengths = numpy.zeros((Capacity.shape[0], len(T)))
        #    print(Demand)
        #    print(Capacity)
        for i in range(Capacity.shape[0]):
            A = FIFO_QMODEL_SPLIT(Arr_Demand, Dep_Demand, U, Capacity[i])
            AvgArrDelay[i] = A.Arr_Delay * Del_T
            P_Arr_canc[i] = A.Arr_Cancel
            arr_q_lengths[i] = A.Arr_Queue

            AvgDepDelay[i] = A.Dep_Delay * Del_T
            P_Dep_canc[i] = A.Dep_Cancel
            dep_q_lengths[i] = A.Dep_Queue

        return {'avg_arr_delay': AvgArrDelay,
                'p_arr_canc': P_Arr_canc,
                'arr_q_lengths': arr_q_lengths,
                'avg_dep_delay': AvgDepDelay,
                'p_dep_canc': P_Dep_canc,
                'dep_q_lengths': dep_q_lengths,
                'prob': Prob}
    else:
        profile = flight.get_aggregated_flight_schedule(flights, n_slots, separate_flights=False)
        Demand = list(profile)
        Demand.extend([0.0] * (U + 1))
        Demand = numpy.array(Demand)

        AvgDelay = numpy.zeros((Capacity.shape[0], len(T)))
        P_canc = numpy.zeros((Capacity.shape[0], len(T)))
        q_lengths = numpy.zeros((Capacity.shape[0], len(T)))
        #    print(Demand)
        #    print(Capacity)
        for i in range(Capacity.shape[0]):
            A = FIFO_QMODEL(Demand, U, Capacity[i])
            AvgDelay[i] = A.Delay * Del_T
            P_canc[i] = A.Cancel
            q_lengths[i] = A.Queue
        return {'avg_delay': AvgDelay,
                'p_canc': P_canc,
                'prob': Prob,
                'q_lengths': q_lengths}


def get_schedule_value_without_monopoly(flights, delay_costs, move_costs, remove_costs):
    # Find the value that each airline gets from the original schedule
    value_by_airline = {}
    airlines = set()
    for f in flights:
        if (f.airline not in value_by_airline):
            airlines.add(f.airline)
            value_by_airline[f.airline] = 0
        value_by_airline[f.airline] += remove_costs[f.flight_id]
        value_by_airline[f.airline] -= delay_costs[f.flight_id, f.slot_time]
        value_by_airline[f.airline] -= move_costs[f.flight_id, f.slot_time]
    value_by_airline['Total'] = sum([v for v in value_by_airline.values()])
    return value_by_airline


def get_monopoly_ub(market_shares, flights, remove_costs, peak_time_range, monopoly_benefit):
    monopoly_ub = {f.flight_id: 0 for f in flights}
    if monopoly_benefit is None:
        return monopoly_ub

    value_by_flight = {}
    for f in flights:
        if f.airline in market_shares.keys():
            percent_benefit = monopoly_benefit(market_shares[f.airline])
            value_by_flight[f.flight_id] = percent_benefit * remove_costs[f.flight_id]
        else:
            value_by_flight[f.flight_id] = 0.0
    value_by_flight['Total'] = sum(value_by_flight[f.flight_id] for f in flights
                                   if f.slot_time in peak_time_range)
    return value_by_flight


def get_airline_market_shares(flights: typing.Iterable[flight.Flight], peak_time_range, profile) -> typing.Dict[str, float]:
    num_peak_slots = sum(profile[t] for t in peak_time_range)
    share_by_airline = {}
    for f in flights:
        if (f.slot_time in peak_time_range):
            if (f.airline not in share_by_airline):
                share_by_airline[f.airline] = 0
            share_by_airline[f.airline] += 1 / num_peak_slots
    return share_by_airline
