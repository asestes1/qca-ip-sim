# -*- coding: utf-8 -*-
"""
Created on Tue Oct  3 21:24:51 2017

@author: aestes
"""
from . import qcaip, flightsched, delaycalc
import numpy
import attr
import itertools
import gurobipy
import typing
import math
import matplotlib.pyplot
import collections
import warnings

SlotProfile = typing.Tuple[int]


@attr.s(kw_only=True, frozen=True)
class InverseFunc(object):

    def __call__(self, iteration_num: int) -> float:
        return 1 / iteration_num


@attr.s(kw_only=True, frozen=True)
class MonopolyBenefitFunc(object):
    rmax = attr.ib(type=float)
    delta = attr.ib(type=float)
    kappa = attr.ib(type=float)

    def __call__(self, market_share: float) -> float:
        return monopoly_benefit(rmax=self.rmax, kappa=self.kappa, delta=self.delta, market_share=market_share)


def monopoly_benefit(rmax: float, delta: float, kappa: float, market_share: float) -> float:
    if market_share < delta:
        return 0
    return rmax * ((market_share - delta) ** kappa) / ((1 - delta) ** kappa)


@attr.s(kw_only=True, frozen=True)
class AuctionRunParams(object):
    flights = attr.ib(type=typing.Iterable[flightsched.Flight])
    connections = attr.ib(type=typing.Mapping[int, typing.Iterable[int]])  # flightid -> ids of connecting flights
    profiles = attr.ib(type=typing.Iterable[SlotProfile])
    max_displacement = attr.ib(type=int)
    min_connect = attr.ib(type=int)
    max_connect = attr.ib(type=int)
    turnaround = attr.ib(type=int)
    max_delay = attr.ib(type=int)
    alpha_f = attr.ib(type=typing.Mapping[flightsched.Flight, float])
    beta_f = attr.ib(type=typing.Mapping[flightsched.Flight, float])
    gamma_f = attr.ib(type=typing.Mapping[flightsched.Flight, float])
    exponent = attr.ib(type=float)
    scenarios = attr.ib(type=numpy.ndarray)
    run_subauctions = attr.ib(type=bool)
    validate = attr.ib(type=bool)
    peak_time_range = attr.ib(type=range)
    rmax = attr.ib(type=float)
    delta = attr.ib(type=float)
    kappa = attr.ib(type=float)
    monopoly_constraint_rate = attr.ib(type=float)
    verbose = attr.ib(type=bool)
    delay_stepsize = attr.ib(type=typing.Callable[[int], float], default=InverseFunc())
    max_iter = attr.ib(type=int, default=50)
    delay_threshold = attr.ib(type=int, default=0)


def get_ip_value_by_airline(n_slots: int, flights: typing.Iterable[flightsched.Flight], model) -> typing.Dict[
    str, float]:
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
    return auc_val_by_airline


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


def make_delay_costs(flights: typing.Iterable[flightsched.Flight], max_delay, alpha_f, beta_f,
                     cancel_probs, delays, scenario_probs, threshold=0) -> typing.Dict[typing.Tuple[int, int], float]:
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


def make_move_costs(flights: typing.Iterable[flightsched.Flight], gamma_f: typing.Dict[flightsched.Flight, float],
                    n_slots: int,
                    exponent: float) -> typing.Mapping[typing.Tuple[int, int], float]:
    """

    :param flights: the flights
    :param gamma_f: Key: flight, value: gamma_f constant for flight f
    :param n_slots: the maximum number of slots that a flight can be moved
    :param exponent:  exponent of cost function
    :return:
    """
    values = {}
    for f, t in itertools.product(flights, range(0, n_slots)):
        values[f.flight_id, t] = cost_move(f, t, gamma_f, exponent)
    return values


def cost_remove(f, max_displacement, n_slots, gamma_f, exponent):
    """

    :param f: the flights
    :param max_displacement: the max displacement of the flight in numbers of slots (int)
    :param n_slots: the number of possible time slots in the day (int)
    :param gamma_f: Key: flight, value: gamma_f constant for flight f
    :param exponent:  exponent of cost function
    :return:
    """
    earliest = max(f.slot_time - max_displacement, 0)
    latest = min(f.slot_time + max_displacement, n_slots - 1)
    return max(cost_move(f, earliest, gamma_f, exponent), cost_move(f, latest, gamma_f, exponent))


def cost_delay(f: flightsched.Flight, delay: int, alpha_f: typing.Dict[flightsched.Flight, float],
               beta_f: typing.Dict[flightsched.Flight, float], threshold: int = 0) -> float:
    """
    Calculate the cost of delaying a flight.

    Arguments:
    :param f: the flights
    :param delay: the amount of delay that a flight receives
    :param alpha_f: dictionary. Keys: flights, values: alpha_f[f] gives alpha_f constant
    :param beta_f: dictionary. Keys: flights, values: beta_f[f] gives beta_f constant
    :param threshold: if delay is less than the threshold, then no cost is incurred.

    :returns the cost of delaying the flight (float).
    """
    if delay < threshold:
        return 0
    else:
        return (alpha_f[f.flight_id] + beta_f[f.flight_id] * f.n_seats) * (delay - threshold)


def cost_cancel(f: flightsched.Flight, max_delay: int, alpha_f: typing.Dict[flightsched.Flight, float],
                beta_f: typing.Dict[flightsched.Flight, float],
                threshold: int = 0) -> float:
    """
    Calculate the cost of cancelling a flight.

    :param f: the flights
    :param max_delay: the amount of delay that a flight receives that would incur the cost of a cancellation
    :param alpha_f: dictionary. Keys: flights, values: alpha_f[f] gives alpha_f constant
    :param beta_f: dictionary. Keys: f] gives beta_f constant

    :returns the cost of cancelling the flight
    """
    return cost_delay(f, max_delay, alpha_f, beta_f, threshold=threshold)


def cost_congest(f, t, max_delay, alpha_f, beta_f, cancel_probs, delays, scenario_probs, threshold=0):
    """
    Calculate the expected cost of congestion for a flight, given some capacity
    scenarios.

    :param f: the flight
    :param t: the time slot
    :param max_delay: the maximum amount of delay that a flight can receive
    :param alpha_f: dictionary. Keys: flight ids, values: alpha_f[f] gives alpha_f constant
    :param beta_f: dictionary. Keys: flight ids, values: beta_f[f] gives beta_f constant
    :param cancel_probs: a matrix, cancel_probs[w,t] gives the probability that an
                   airline would cancel a flight scheduled for the t time slot
                   scenario w
    :param delays: a matrix, cancel_probs[w,t] gives the probability that an
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


def get_schedule_value_without_monopoly(schedule: typing.Iterable[flightsched.Flight],
                                        params: AuctionRunParams) -> typing.Dict[str, float]:
    n_slots = len(params.profiles[0])
    delayresults = delaycalc.get_combined_qdelays(scenarios=params.scenarios,
                                                  flights=schedule,
                                                  n_slots=n_slots)

    original_flights = {f.flight_id: f for f in params.flights}
    # Find the value that each airline gets from the original schedule
    value_by_airline = collections.defaultdict(float)

    for f in schedule:
        orig_flight = original_flights[f.flight_id]
        value_by_airline[f.airline] += cost_remove(f=orig_flight, max_displacement=params.max_displacement,
                                                   n_slots=n_slots, gamma_f=params.gamma_f, exponent=params.exponent)
        value_by_airline[f.airline] -= cost_congest(f=orig_flight, t=f.slot_time, max_delay=params.max_delay,
                                                    alpha_f=params.alpha_f, beta_f=params.beta_f,
                                                    cancel_probs=delayresults.p_canc,
                                                    delays=delayresults.avg_delay,
                                                    scenario_probs=delayresults.prob, threshold=params.delay_threshold)
        value_by_airline[f.airline] -= cost_move(f=orig_flight, t=f.slot_time, gamma_f=params.gamma_f,
                                                 exponent=params.exponent)
    return value_by_airline


def get_schedule_monopoly_value(flights, profile, params: AuctionRunParams) -> typing.Mapping[str, float]:
    if params.peak_time_range is None or params.rmax is None or params.delta is None or params.kappa is None:
        return {f.airline: 0 for f in flights}

    market_shares = flightsched.get_airline_market_shares(flights=flights, peak_time_range=params.peak_time_range,
                                                          profile=profile)
    value_by_airline = collections.defaultdict(int)
    for f in flights:
        if f.airline in market_shares.keys() and f.slot_time in params.peak_time_range:
            percent_benefit = monopoly_benefit(rmax=params.rmax, delta=params.delta, kappa=params.kappa,
                                               market_share=market_shares[f.airline])
            value_by_airline[f.airline] += percent_benefit * cost_remove(f=f,
                                                                         max_displacement=params.max_displacement,
                                                                         n_slots=len(profile), gamma_f=params.gamma_f,
                                                                         exponent=params.exponent)
    return value_by_airline


def plot_movement_by_flight_type(model, flights, n_times, title=""):
    points_x = []
    points_y = []
    for f in flights:
        for t in range(0, n_times):
            x = model.getVarByName('F' + str(f.flight_id) + 'T' + str(t))
            if x is not None and abs(x.getAttr(gurobipy.GRB.attr.X) - 1.0) < .0001:
                points_x.append(f.n_seats)
                points_y.append(abs(f.slot_time - t))
    matplotlib.pyplot.figure()
    matplotlib.pyplot.title("Number of seats vs. movement " + title)
    matplotlib.pyplot.xlabel("Number of seats")
    matplotlib.pyplot.ylabel("Number of time periods moved")
    matplotlib.pyplot.scatter(points_x, points_y)


def initialize_ip_model(params: AuctionRunParams, n_slots, move_costs, remove_costs):
    delay_results = delaycalc.get_combined_qdelays(params.scenarios, params.flights, n_slots)
    delay_costs = make_delay_costs(params.flights, params.max_delay, params.alpha_f, params.beta_f,
                                   delay_results.p_canc, delay_results.avg_delay,
                                   delay_results.prob,
                                   threshold=params.delay_threshold)
    # Build a tentative auction model with the given delay costs
    if params.verbose:
        print("BUILDING INITIAL MODEL")
    if params.rmax is None or params.kappa is None or params.delta is None:
        mono_func = None
    else:
        mono_func = MonopolyBenefitFunc(rmax=params.rmax, kappa=params.kappa,delta=params.delta)
    model_results = qcaip.build_ip(flights=params.flights,
                                   connections=params.connections,
                                   profile=params.profiles[0],
                                   delay_costs=delay_costs,
                                   move_costs=move_costs,
                                   remove_costs=remove_costs,
                                   max_displacement=params.max_displacement,
                                   min_connect=params.min_connect,
                                   max_connect=params.max_connect,
                                   turnaround=params.turnaround,
                                   peak_time_range=params.peak_time_range,
                                   monopoly_benefit_func=mono_func,
                                   monopoly_constraint_rate=params.monopoly_constraint_rate,
                                   verbose=params.verbose)
    return model_results


def initialize_costs(params: AuctionRunParams, profile: SlotProfile,
                     move_costs: typing.Mapping[typing.Tuple[int, int], float],
                     remove_costs: typing.Mapping[int, float], n_slots: int,
                     modelstruct: qcaip.AuctionIpStruct) -> None:
    zero_delay_costs = {(f.flight_id, t): 0
                        for f, t in itertools.product(params.flights, range(0, n_slots))}
    # Zero the delay costs
    qcaip.reset_delay_costs(modelstruct.model, zero_delay_costs, move_costs, params.flights, n_slots)
    if params.rmax is not None and params.delta is not None and params.kappa is not None and params.peak_time_range is not None:
        market_shares = flightsched.get_airline_market_shares(params.flights, params.peak_time_range, profile)
        qcaip.reset_monopoly_costs(modelstruct.model, market_shares, params.flights, remove_costs,
                                   zero_delay_costs,
                                   move_costs, MonopolyBenefitFunc(rmax=params.rmax, kappa=params.kappa,
                                                                   delta=params.delta), params.peak_time_range)
    return


@attr.s(kw_only=True)
class SubauctionResult(object):
    ipvalue = attr.ib(type=float)
    ipvalue_by_airline = attr.ib(type=typing.Mapping[str, float])
    schedule = attr.ib(type=typing.Iterable[flightsched.Flight])


def assignment_diff(flights: typing.Iterable[flightsched.Flight],
                    assignment_a: typing.Iterable[flightsched.Flight],
                    assignment_b: typing.Iterable[flightsched.Flight]):
    assignment_a = {f.flight_id: f.slot_time for f in assignment_a}
    assignment_b = {f.flight_id: f.slot_time for f in assignment_b}

    score = 0
    for f in flights:
        if f.flight_id in assignment_a and f.flight_id in assignment_b:
            if assignment_b[f.flight_id] != assignment_a[f.flight_id]:
                score += abs(assignment_b[f.flight_id] - assignment_a[f.flight_id])
    return score


def max_assignment_diff(flights: typing.Iterable[flightsched.Flight],
                        assignment_a: typing.Iterable[flightsched.Flight],
                        assignment_b: typing.Iterable[flightsched.Flight]):
    assignment_a = {f.flight_id: f.slot_time for f in assignment_a}
    assignment_b = {f.flight_id: f.slot_time for f in assignment_b}

    max = 0
    for f in flights:
        if f.flight_id in assignment_a and f.flight_id in assignment_b:
            if abs(assignment_b[f.flight_id] - assignment_a[f.flight_id]) > max:
                max = abs(assignment_b[f.flight_id] - assignment_a[f.flight_id])

    return max


def removal_score(flights: typing.Iterable[flightsched.Flight],
                  assignment_a: typing.Iterable[flightsched.Flight],
                  assignment_b: typing.Iterable[flightsched.Flight]):
    assignment_a = {f.flight_id: f.slot_time for f in assignment_a}
    assignment_b = {f.flight_id: f.slot_time for f in assignment_b}

    score = 0
    for f in flights:
        if (f.flight_id in assignment_a) != (f.flight_id in assignment_b):
            score += 1
    return score


def get_profile_assignment(profile: SlotProfile, airline: typing.Optional[str], params: AuctionRunParams, n_slots: int,
                           move_costs: typing.Mapping[typing.Tuple[int, int], float],
                           remove_costs: typing.Mapping[int, float],
                           warm_model: gurobipy.Model = None):
    # This builds a model, which we modify in each of the runs.
    if warm_model is None:
        warm_model = initialize_ip_model(params=params, n_slots=n_slots, move_costs=move_costs,
                                         remove_costs=remove_costs)
    modelstruct = warm_model

    qcaip.edit_profile_constr(modelstruct.model, profile, params.flights,
                              monopoly_constraint_rate=params.monopoly_constraint_rate,
                              peak_time_range=params.peak_time_range)

    initialize_costs(params=params, profile=profile, move_costs=move_costs, remove_costs=remove_costs,
                     n_slots=n_slots, modelstruct=modelstruct)  # delay costs to zero, monopoly costs to current sched
    air_constr = None
    if airline is not None:
        air_constr = qcaip.fix_airline_vars(airline=airline, modelstruct=modelstruct)

    iterations = 0
    converged = False
    previous_assignments = []
    delay_costs = None
    while not converged:
        iterations += 1
        print(airline, profile[0], iterations)

        # Find an estimated schedule
        modelstruct.model.optimize()
        new_flights = flightsched.get_new_flight_schedule(params.flights, n_slots, modelstruct.model)
        try:
            cycle_index = previous_assignments.index(new_flights)
            if cycle_index == len(previous_assignments) - 1:
                converged = True
            else:
                print("CYCLED: " + str(len(previous_assignments) - cycle_index))
                previous_assignments = []
        except ValueError:
            pass
        if iterations >= params.max_iter:
            raise RuntimeError(
                "ITERATIONS HAVE REACHED MAXIMUM: " + str(iterations) + " run: " + str(profile[0]) + ", " + str(
                    airline))
        if previous_assignments:
            print("DIFF: ", assignment_diff(flights=params.flights,
                                            assignment_a=previous_assignments[-1],
                                            assignment_b=new_flights))
            print("MAX DIFF: ", max_assignment_diff(flights=params.flights,
                                                    assignment_a=previous_assignments[-1],
                                                    assignment_b=new_flights))
            print("REMOVE: ", removal_score(flights=params.flights,
                                            assignment_a=previous_assignments[-1],
                                            assignment_b=new_flights))
        previous_assignments.append(new_flights)

        if params.validate:
            qcaip.check_constraints(new_flights, params.flights, params.connections,
                                    params.min_connect, params.max_connect, params.max_displacement,
                                    params.turnaround, profile, params.peak_time_range,
                                    params.monopoly_constraint_rate)
        if not converged:
            delay_estimates = delaycalc.get_combined_qdelays(scenarios=params.scenarios, flights=new_flights,
                                                             n_slots=n_slots)

            new_delay_costs = make_delay_costs(params.flights, params.max_delay, params.alpha_f,
                                               params.beta_f,
                                               delay_estimates.p_canc,
                                               delay_estimates.avg_delay,
                                               delay_estimates.prob,
                                               threshold=params.delay_threshold)

            if delay_costs is None:
                delay_costs = new_delay_costs
            else:
                step = params.delay_stepsize(iterations)
                delay_costs = {k: step * new_delay_costs[k] + (1 - step) * v for k, v in delay_costs.items()}

            qcaip.reset_delay_costs(model=modelstruct.model, delay_costs=delay_costs,
                                    move_costs=move_costs, flights=params.flights, n_slots=n_slots)
            # Reassign variable costs for the new estimated schedule
            if params.rmax is not None and params.delta is not None and params.kappa is not None and params.peak_time_range is not None:
                market_shares = flightsched.get_airline_market_shares(new_flights, params.peak_time_range, profile)
                qcaip.reset_monopoly_costs(modelstruct.model, market_shares, params.flights, remove_costs,
                                           delay_costs, move_costs,
                                           MonopolyBenefitFunc(rmax=params.rmax, kappa=params.kappa,
                                                               delta=params.delta),
                                           params.peak_time_range)
    ip_value = modelstruct.model.getAttr(gurobipy.GRB.attr.ObjVal)
    ip_value_by_airline = get_ip_value_by_airline(n_slots, params.flights, modelstruct.model)

    # Add back the airline that we removed
    if airline is not None:
        qcaip.remove_constraints(modelstruct.model, air_constr)
    return SubauctionResult(ipvalue=ip_value, ipvalue_by_airline=ip_value_by_airline, schedule=new_flights)


@attr.s(kw_only=True)
class AuctionResultStruct(object):
    best_schedule = attr.ib(type=typing.Iterable[flightsched.Flight])
    best_profile = attr.ib(type=SlotProfile)
    ipval = attr.ib(type=float)
    payments = attr.ib(type=typing.Optional[typing.Dict[str, float]])
    subaction_results = attr.ib(type=typing.Dict[typing.Tuple[SlotProfile, typing.Optional[str]], SubauctionResult])
    params = attr.ib(type=AuctionRunParams)


def run_pricing_auction(params: AuctionRunParams) -> AuctionResultStruct:
    airlines = {f.airline for f in params.flights}
    if params.run_subauctions:
        subauctions = {None}.union(airlines)
    else:
        subauctions = {None}

    n_slots = len(params.profiles[0])
    move_costs = make_move_costs(flights=params.flights, gamma_f=params.gamma_f, n_slots=n_slots,
                                 exponent=params.exponent)
    remove_costs = make_remove_costs(flights=params.flights, max_displacement=params.max_displacement,
                                     n_slots=n_slots, gamma_f=params.gamma_f, exponent=params.exponent)
    warm_model = initialize_ip_model(params=params, n_slots=n_slots, move_costs=move_costs,
                                     remove_costs=remove_costs)
    results = {(p, a): get_profile_assignment(profile=p, airline=a, params=params, n_slots=n_slots,
                                              move_costs=move_costs, remove_costs=remove_costs,
                                              warm_model=warm_model)
               for p in params.profiles for a in subauctions}

    best_profile = {}
    for a in subauctions:
        def value_getter(p: SlotProfile) -> float:
            return results[p, a].ipvalue

        best_profile[a] = max(params.profiles, key=value_getter)

    auction_ipval = results[best_profile[None], None].ipvalue

    payments = None
    if params.run_subauctions:
        payments = {
            a: results[best_profile[None], None].ipvalue_by_airline[a] - (
                    auction_ipval - results[best_profile[a], a].ipvalue)
            for a in airlines}

    return AuctionResultStruct(best_schedule=results[best_profile[None], None], best_profile=best_profile[None],
                               ipval=auction_ipval, payments=payments,
                               subaction_results=results, params=params)


@attr.s(frozen=True, kw_only=True)
class SchedValueStruct(object):
    social_value = attr.ib(type=float)
    mono_value = attr.ib(type=float)
    payment = attr.ib(type=float)


def get_fixed_prof_payments(results: AuctionResultStruct, profile: SlotProfile):
    params=results.params
    airlines = {f.airline for f in params.flights}
    schedule = results.subaction_results[profile, None].schedule
    social_value = get_schedule_value_without_monopoly(schedule=schedule, params=params)
    mono_value = get_schedule_monopoly_value(params.flights, profile=profile, params=params)

    auction_ipval = results.subaction_results[profile, None].ipvalue
    payments = {
        a: results.subaction_results[profile, None].ipvalue_by_airline[a] - (
                auction_ipval - results.subaction_results[profile, a].ipvalue)
        for a in airlines}
    return SchedValueStruct(social_value=sum(social_value.values()), mono_value=sum(mono_value.values()),
                            payment=sum(payments.values()))