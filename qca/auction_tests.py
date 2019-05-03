# -*- coding: utf-8 -*-
"""
Created on Tue Oct  3 21:24:51 2017

@author: aestes
"""
from . import auction_model, auction_costs, flight
import itertools
import numpy
import gurobipy
import matplotlib.pyplot


def run_pricing_auction(flights, connections, move_costs, remove_costs, profiles,
                        max_displacement, min_connect, max_connect, turnaround, max_delay,
                        alpha_f, beta_f, scenarios, airline_subauctions,
                        validate=False,
                        peak_time_range=None, monopoly_benefit_func=None, monopoly_constraint_rate=None,
                        verbose=False,
                        warm_model=None, air_rem_vars=None,
                        air_res_vars=None, num_iterations=2,
                        delay_threshold=0):
    """
    :param flights:
    :param connections:
    :param move_costs:
    :param remove_costs:
    :param profiles:
    :param max_displacement:
    :param min_connect:
    :param max_connect:
    :param turnaround:
    :param max_delay:
    :param alpha_f:
    :param beta_f:
    :param scenarios:
    :param airline_subauctions:
    :param validate:
    :param peak_time_range:
    :param monopoly_benefit_func:
    :param monopoly_constraint_rate:
    :param verbose:
    :param warm_model:
    :param air_rem_vars:
    :param air_res_vars:
    :param num_iterations:
    :param delay_threshold:
    :return:
    """
    n_slots = len(profiles[0])
    # There is a run for each airline and a run of the full auction
    runs = list(airline_subauctions)
    runs.append('FULL_AUCTION')

    # This builds a model, which we modify in each of the runs.
    if warm_model is None or air_rem_vars is None or air_res_vars is None:
        delay_results = auction_costs.get_queue_delays(scenarios, flights, n_slots)
        delay_costs = auction_costs.make_delay_costs(flights, max_delay, alpha_f, beta_f,
                                                     delay_results['p_canc'], delay_results['avg_delay'],
                                                     delay_results['prob'],
                                                     threshold=delay_threshold)
        # Build a tentative auction model with the given delay costs
        print("BUILDING INITIAL MODEL")
        model_results = auction_model.build_auction_model(flights,
                                                          connections, profiles[0], delay_costs,
                                                          move_costs, remove_costs, max_displacement,
                                                          min_connect, max_connect, turnaround,
                                                          peak_time_range=peak_time_range,
                                                          monopoly_benefit_func=monopoly_benefit_func,
                                                          monopoly_constraint_rate=monopoly_constraint_rate,
                                                          verbose=verbose)
        model = model_results['model']
        air_rem_vars = model_results['removal_vars_airline']
        air_res_vars = model_results['reschedule_vars_airline']
    else:
        model = warm_model

    # Go through each profile
    auction_values = {}
    #    monopoly_values = {}
    auction_values_by_airline = {}
    schedules = {}

    best_auction_values = {}
    best_profiles = {}
    best_schedules = {}
    monopoly_estimates = {}
    delay_estimates = {}

    zero_delay_costs = {(f.flight_id, t): 0
                        for f, t in itertools.product(flights, range(0, n_slots))}
    for i, p in enumerate(profiles):
        # Adjust profile constraint for this profile.
        print("EDITING CONSTRAINTS")
        auction_model.edit_profile_constr(model, p, flights,
                                          monopoly_constraint_rate=monopoly_constraint_rate,
                                          peak_time_range=peak_time_range)

        # Go through each run
        for r in runs:
            print('Profile: ' + str(p[0]) + ', Run: ' + str(r))
            # If this is not the full auction, then we remove an airline
            air_constr = None
            if r != 'FULL_AUCTION':
                air_constr = auction_model.fix_airline_vars(r, model, air_rem_vars, air_res_vars)

            # Zero the delay costs
            auction_model.reset_delay_costs(model, zero_delay_costs, move_costs, flights, n_slots)
            if monopoly_benefit_func is not None and peak_time_range is not None:
                monopoly_estimates[tuple(p), r] = []
                market_shares = auction_costs.get_airline_market_shares(flights, peak_time_range, p)
                monopoly_estimates[tuple(p), r].append(
                    auction_costs.get_monopoly_ub(market_shares, flights, remove_costs,
                                                  peak_time_range, monopoly_benefit_func)['Total'])
                #                print("Initial Estimate Market Share: ",market_shares)
                auction_model.reset_monopoly_costs(model, market_shares, flights, remove_costs,
                                                   zero_delay_costs,
                                                   move_costs, monopoly_benefit_func, peak_time_range)

            delay_estimates[tuple(p), r] = []
            for j in range(0, num_iterations):
                # Find an estimated schedule
                model.optimize()
                new_flights = auction_model.get_new_flight_schedule(flights, n_slots, model)

                # Calculate the expected delays in the estimated schedule
                delay_results = auction_costs.get_queue_delays(scenarios, new_flights, n_slots)
                delay_estimates[tuple(p), r].append(sum(sum(p[i] * delay_results['avg_delay'][i, f.slot_time]
                                                            for i in range(0, len(delay_results['prob'])))
                                                        for f in new_flights))
                new_delay_costs = None
                if j < num_iterations - 1:
                    new_delay_costs = auction_costs.make_delay_costs(flights, max_delay, alpha_f, beta_f,
                                                                     delay_results['p_canc'],
                                                                     delay_results['avg_delay'],
                                                                     delay_results['prob'],
                                                                     threshold=delay_threshold)
                    auction_model.reset_delay_costs(model, new_delay_costs, move_costs, flights, n_slots)

                # Reassign variable costs for the new estimated schedule
                if monopoly_benefit_func is not None and peak_time_range is not None:
                    market_shares = auction_costs.get_airline_market_shares(new_flights, peak_time_range, p)
                    monopoly_estimates[tuple(p), r].append(
                        auction_costs.get_monopoly_ub(market_shares, new_flights, remove_costs,
                                                      peak_time_range, monopoly_benefit_func)['Total'])
                    if j < num_iterations - 1:
                        auction_model.reset_monopoly_costs(model, market_shares, flights, remove_costs,
                                                           new_delay_costs,
                                                           move_costs,
                                                           monopoly_benefit_func,
                                                           peak_time_range)

            if validate:
                new_flights = auction_model.get_new_flight_schedule(flights, n_slots, model)
                auction_model.check_constraints(new_flights, flights, connections,
                                                min_connect, max_connect, max_displacement,
                                                turnaround, p, peak_time_range, monopoly_constraint_rate)
            auction_values[r, tuple(p)] = model.getAttr(gurobipy.GRB.attr.ObjVal)
            auc_flights = auction_model.get_new_flight_schedule(flights, n_slots, model)

            auction_values_by_airline[r, tuple(p)] = auction_costs.get_ip_value_by_airline(n_slots, auc_flights, model)
            schedules[r, tuple(p)] = auction_model.get_new_flight_schedule(flights, n_slots, model)

            # If we have calculated delay costs, then check if this run is the winner.
            if r not in best_auction_values or best_auction_values[r] < auction_values[r, tuple(p)]:
                best_auction_values[r] = auction_values[r, tuple(p)]
                best_profiles[r] = p
                best_schedules[r] = schedules[r, tuple(p)]

            # Add back the airline that we removed
            if r != 'FULL_AUCTION':
                auction_model.remove_constraints(model, air_constr)
    results = {'auction_values': auction_values,
               'auction_values_by_airline': auction_values_by_airline,
               #               'monopoly_values_by_airline':monopoly_values,
               'schedules': schedules,
               'best_profiles': best_profiles,
               'best_auction_values': best_auction_values,
               'best_schedules': best_schedules,
               'monopoly_estimates': monopoly_estimates,
               'delay_estimates': delay_estimates}
    return results


def price_auctions(flights, connections, move_costs, remove_costs, profiles,
                   max_displacement, min_connect, max_connect, turnaround,
                   max_delay, alpha_f, beta_f, scenarios,
                   airline_subauctions=None,
                   peak_time_range=None,
                   monopoly_benefit_func=None, monopoly_constraint_rate=None,
                   verbose=False, validate=False, num_iterations=2,
                   delay_threshold=0):
    """
    :param flights:
    :param connections:
    :param move_costs:
    :param remove_costs:
    :param profiles:
    :param max_displacement:
    :param min_connect:
    :param max_connect:
    :param turnaround:
    :param max_delay:
    :param alpha_f:
    :param beta_f:
    :param scenarios:
    :param airline_subauctions:
    :param peak_time_range:
    :param monopoly_benefit_func:
    :param monopoly_constraint_rate:
    :param verbose:
    :param validate:
    :param num_iterations:
    :param delay_threshold:
    :return:
    """
    if airline_subauctions is None:
        airline_subauctions = []

    # Get the number of slots
    n_slots = len(profiles[0])
    # Find the flight schedule before flights are reallocated.
    delay_results = auction_costs.get_queue_delays(scenarios, flights, n_slots)
    default_delay_costs = auction_costs.make_delay_costs(flights, max_delay, alpha_f, beta_f,
                                                         delay_results['p_canc'], delay_results['avg_delay'],
                                                         delay_results['prob'],
                                                         threshold=delay_threshold)
    def_allocation_val = auction_costs.get_schedule_value_without_monopoly(flights, default_delay_costs, move_costs,
                                                                           remove_costs)
    #    if(peak_time_range is not None and monopoly_benefit_func is not None):

    # Get the total delay from the original schedule
    orig_scenario_delay = sum(delay_results['avg_delay'][:, f.slot_time] for f in flights)
    orig_exp_delay = numpy.dot(orig_scenario_delay, delay_results['prob'].ravel())

    # For each airline, run a subauction with that airline removed.
    print("Running auctions")
    results = run_pricing_auction(flights, connections, move_costs,
                                  remove_costs, profiles,
                                  max_displacement, min_connect, max_connect, turnaround, max_delay,
                                  alpha_f, beta_f, scenarios, airline_subauctions,
                                  verbose=verbose,
                                  validate=validate,
                                  monopoly_benefit_func=monopoly_benefit_func,
                                  monopoly_constraint_rate=monopoly_constraint_rate,
                                  peak_time_range=peak_time_range, num_iterations=num_iterations,
                                  delay_threshold=delay_threshold)

    new_flights = results['best_schedules']['FULL_AUCTION']
    cancellations = len(set(flights)) - len(set(new_flights))
    slot_times = {f.flight_id: f.slot_time for f in new_flights}
    old_slot = {f.flight_id: f.slot_time for f in flights}
    max_displacement = max(abs(slot_times[f.flight_id] - old_slot[f.flight_id]) for f in new_flights)
    displacement = sum(abs(slot_times[f.flight_id] - old_slot[f.flight_id]) for f in new_flights)
    delay_results = auction_costs.get_queue_delays(scenarios, new_flights, n_slots, Del_T=15, separate_flights=False)
    new_delay_costs = auction_costs.make_delay_costs(new_flights, max_delay, alpha_f, beta_f,
                                                     delay_results['p_canc'], delay_results['avg_delay'],
                                                     delay_results['prob'],
                                                     threshold=delay_threshold)
    new_allocation_Val = auction_costs.get_schedule_value_without_monopoly(new_flights, new_delay_costs, move_costs,
                                                                           remove_costs)
    if peak_time_range is not None and monopoly_benefit_func is not None:
        new_market_shares = auction_costs.get_airline_market_shares(new_flights, peak_time_range,
                                                                    results['best_profiles']['FULL_AUCTION'])
        #        print("Final market shares: ",new_market_shares)
        new_monopoly_benefits = auction_costs.get_monopoly_ub(new_market_shares, new_flights, remove_costs,
                                                              peak_time_range, monopoly_benefit_func)
    else:
        new_monopoly_benefits = {'Total': 0}

    new_scenario_delay = sum(delay_results['avg_delay'][:, f.slot_time] for f in flights)
    new_exp_delay = numpy.dot(new_scenario_delay, delay_results['prob'].ravel())
    max_q_length = numpy.dot(numpy.amax(delay_results['q_lengths'], axis=1),
                             delay_results['prob'].ravel())

    separate_delay_results = auction_costs.get_queue_delays(scenarios, new_flights, n_slots, Del_T=15,
                                                            separate_flights=True)
    max_arr_q_length = numpy.dot(numpy.amax(separate_delay_results['arr_q_lengths'], axis=1),
                                 separate_delay_results['prob'].ravel())
    max_dep_q_length = numpy.dot(numpy.amax(separate_delay_results['dep_q_lengths'], axis=1),
                                 separate_delay_results['prob'].ravel())
    # Get the payments
    payments = {}
    for a in airline_subauctions:
        value_to_airline = (
            results['auction_values_by_airline']['FULL_AUCTION', tuple(results['best_profiles']['FULL_AUCTION'])][a])
        #                            +results['monopoly_values_by_airline']['FULL_AUCTION',tuple(results['best_profiles']['FULL_AUCTION'])][a])
        value_with_airline = (results['best_auction_values']['FULL_AUCTION'])
        #                            +results['monopoly_values_by_airline']['Total'])
        value_without_airline = (results['best_auction_values'][a])
        #                            +results['monopoly_values_by_airline']['Total'])
        payments[a] = value_to_airline - (value_with_airline - value_without_airline)
    total_payments = sum([payments[a] for a in airline_subauctions])
    results['payments'] = payments

    summary_string = ""
    summary_string += "Value of current schedule (not included monopoly benefits): " + str(def_allocation_val['Total'])
    summary_string += "\nDelay with no auction: " + str(orig_exp_delay)
    summary_string += "\n"
    summary_string += "\nCalculated Schedule Value (minus monopoly benefits): " + str(new_allocation_Val['Total'])
    summary_string += "\nCalculated Monopoly Value: " + str(new_monopoly_benefits['Total'])
    summary_string += "\nTotal Calculated Schedule Value: " + str(
        new_monopoly_benefits['Total'] + new_allocation_Val['Total'])
    summary_string += "\n"
    summary_string += "\nAuction Value from IP: " + str(results['best_auction_values']['FULL_AUCTION'])
    summary_string += "\nRemovals in Auction: " + str(cancellations)
    summary_string += "\nMax Displacements: " + str(max_displacement)
    summary_string += "\nTotal Displacements: " + str(displacement)
    summary_string += "\nTotal Expected Delay in Auction: " + str(new_exp_delay)
    summary_string += "\nExpected Max Queue: " + str(max_q_length)
    summary_string += "\nExpected Max Arr Queue: " + str(max_arr_q_length)
    summary_string += "\nExpected Max Dep Queue: " + str(max_dep_q_length)
    summary_string += "\nTotal payments: " + str(total_payments)
    summary_string += "\nNet value of auction schedule to airlines: " + str(
        results['best_auction_values']['FULL_AUCTION'] -
        total_payments)
    summary_string += "\nBest profile: " + str(results['best_profiles']['FULL_AUCTION'])
    summary_string += "\nNew schedule: " + str(list(flight.get_aggregated_flight_schedule(
        results['best_schedules']['FULL_AUCTION'], n_slots)))
    summary_string += "\nOriginal Delays by Scenario: " + str(list(orig_scenario_delay))
    summary_string += "\nNew Delays by Scenario: " + str(list(new_scenario_delay))
    results['summary_string'] = summary_string
    return results


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
