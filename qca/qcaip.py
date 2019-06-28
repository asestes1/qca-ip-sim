import gurobipy
from . import flightsched
import typing
import gurobipy as grb
import attr
import warnings


def edit_profile_constr(model, profile, flights,
                        monopoly_constraint_rate=None,
                        peak_time_range=None):
    for t, n in enumerate(profile):
        my_constr = model.getConstrByName('PC' + str(t))
        my_constr.setAttr(gurobipy.GRB.attr.RHS, n)

    if peak_time_range is not None:
        num_peak_slots = sum(profile[t] for t in peak_time_range)
        if monopoly_constraint_rate is not None:
            airlines = {f.airline for f in flights}
            for airline in airlines:
                constr = model.getConstrByName("MONO_CONSTR: " + airline)
                constr.setAttr(gurobipy.GRB.attr.RHS, monopoly_constraint_rate * num_peak_slots)
        model.update()
    return


def reset_delay_costs(model: gurobipy.Model, delay_costs: typing.Mapping[typing.Tuple[int, int], float],
                      move_costs: typing.Mapping[typing.Tuple[int, int], float],
                      flights: typing.Iterable[flightsched.Flight],
                      n_slots: int):
    for f in flights:
        for t in range(0, n_slots):
            next_reschedule_var = model.getVarByName('F' + str(f.flight_id) + 'T' + str(t))
            if next_reschedule_var is not None:
                next_reschedule_var.setAttr(gurobipy.GRB.attr.Obj,
                                            -1 * (move_costs[f.flight_id, t]
                                                  + delay_costs[f.flight_id, t]))
    model.update()
    return


def reset_monopoly_costs(model, market_shares, flights,
                         remove_costs, delay_costs, move_costs, monopoly_benefit_func,
                         peak_time_range):
    if monopoly_benefit_func is not None and peak_time_range is not None:
        max_monop_by_flight = get_monopoly_ub(market_shares, flights, remove_costs, monopoly_benefit_func)
        for f in flights:
            for t in peak_time_range:
                move_var = model.getVarByName('F' + str(f.flight_id) + 'T' + str(t))
                if move_var is not None:
                    move_var.setAttr(gurobipy.GRB.attr.Obj, max_monop_by_flight[f.flight_id]
                                     - (delay_costs[f.flight_id, t] + move_costs[f.flight_id, t]))
        model.update()


def remove_constraints(model, constraints):
    for c in constraints:
        model.remove(c)
    model.update()
    return


@attr.s(kw_only=True)
class AuctionVarStruct(object):
    remove_vars = attr.ib(type=typing.Mapping[int, grb.Var])
    reschedule_vars = attr.ib(type=typing.Mapping[typing.Tuple[int, int], grb.Var])  # key: (flightid, timeperiod)
    airline_remove_vars = attr.ib(type=typing.Mapping[str, typing.Set[grb.Var]])
    airline_reschedule_vars = attr.ib(type=typing.Mapping[str, typing.Set[grb.Var]])

    @staticmethod
    def build(flights: typing.Iterable[flightsched.Flight], remove_vars: typing.Mapping[int, grb.Var],
              reschedule_vars: typing.Mapping[typing.Tuple[int, int], grb.Var],
              max_displacement: int, n_slots: int):
        # Organize variables by airline
        remove_vars_by_airline = {}
        reschedule_vars_by_airline = {}
        for f in flights:
            if f.airline not in remove_vars_by_airline:
                remove_vars_by_airline[f.airline] = set()
                reschedule_vars_by_airline[f.airline] = set()

            remove_vars_by_airline[f.airline].add(remove_vars[f.flight_id])

            earliest_slot = max(f.slot_time - max_displacement, 0)
            latest_slot = min(f.slot_time + max_displacement, n_slots - 1)
            for t in range(earliest_slot, latest_slot + 1):
                reschedule_vars_by_airline[f.airline].add(
                    reschedule_vars[f.flight_id, t])
        return AuctionVarStruct(remove_vars=remove_vars, reschedule_vars=reschedule_vars,
                                airline_remove_vars=remove_vars_by_airline,
                                airline_reschedule_vars=reschedule_vars_by_airline)


@attr.s(kw_only=True)
class AuctionIpStruct(object):
    model = attr.ib(type=grb.Model)
    ipvars = attr.ib(type=AuctionVarStruct)


def fix_airline_vars(airline, modelstruct: AuctionIpStruct):
    var_fix_constraints = set()
    for v in modelstruct.ipvars.airline_remove_vars[airline]:
        var_fix_constraints.add(modelstruct.model.addConstr(v, gurobipy.GRB.EQUAL, 1))

    for v in modelstruct.ipvars.airline_reschedule_vars[airline]:
        var_fix_constraints.add(modelstruct.model.addConstr(v, gurobipy.GRB.EQUAL, 0))
    modelstruct.model.update()
    return var_fix_constraints


def build_ip(flights: typing.Iterable[flightsched.Flight], connections: typing.Mapping[int, typing.Iterable[int]],
             profile: typing.List[int],
             delay_costs: typing.Mapping[typing.Tuple[int, int], float],
             move_costs: typing.Mapping[typing.Tuple[int, int], float],
             remove_costs: typing.Mapping[int, float],
             max_displacement: int, min_connect: int, max_connect: int, turnaround: int,
             peak_time_range: range = None,
             monopoly_benefit_func: typing.Callable[[float], float] = None,
             monopoly_constraint_rate: float = None,
             verbose: bool = False) -> AuctionIpStruct:
    """
    :param flights: list of flights
    :param connections: dictionary, key is id of flight, value is a list of connecting flight ids
    :param profile: array-like, ith value is number of slots in ith time period of profile
    :param delay_costs: maps (flightid, timeperiod) to delay cost of assigning flight to time period
    :param move_costs: maps (flightid, timeperiod) to displacement cost of assigning flight to time period
    :param remove_costs: maps (flightid) to delay cost of assigning flight to time period
    :param max_displacement: positive integer, units are number of time periods
    :param min_connect: positive integer, units are number of time periods
    :param max_connect: positive integer, units are number of time periods
    :param turnaround: positive integer, units are number of time periods
    :param peak_time_range: range, peak time periods
    :param monopoly_benefit_func: callable, argument is proportion of peak slots held, value is monopoly multiplier
    :param monopoly_constraint_rate: real number between 0 and 1. Specifies the maximum  percent of peak time slots that
                                     any aircraft can own.
    :param verbose: if true, prints stuff to console
    :return:
    """
    # Create model.
    my_auction_model = gurobipy.Model("AuctionModel")

    # Set model objective to maximize.
    my_auction_model.setAttr(gurobipy.GRB.attr.ModelSense, -1)

    # Choose whether model is verbose or not.
    if not verbose:
        my_auction_model.setParam(gurobipy.GRB.Param.OutputFlag, 0)
    else:
        my_auction_model.setParam(gurobipy.GRB.Param.OutputFlag, 1)

    # Find the number of slots.
    n_slots = len(profile)

    # constant_value is the assumed value of assignment not considering delays
    constant = sum(remove_costs[f.flight_id] for f in flights)
    my_auction_model.setAttr(gurobipy.GRB.attr.ObjCon, constant)
    if monopoly_benefit_func is not None:
        market_shares = flightsched.get_airline_market_shares(flights, peak_time_range, profile)
        max_mono_by_flight = get_monopoly_ub(market_shares, flights, remove_costs, monopoly_benefit_func)
    else:
        max_mono_by_flight = {f.flight_id: 0 for f in flights}

    # Create variables
    if verbose:
        print("Adding variables")
    reschedule_vars = {}
    remove_vars = {}
    for f in flights:
        remove_vars[f.flight_id] = my_auction_model.addVar(name='R' + str(f.flight_id), obj=-remove_costs[f.flight_id],
                                                           vtype=gurobipy.GRB.BINARY)
        earliest_slot = max(f.slot_time - max_displacement, 0)
        latest_slot = min(f.slot_time + max_displacement, n_slots - 1)
        for t in range(earliest_slot, latest_slot + 1):
            if peak_time_range is not None and t in peak_time_range:
                var_obj = (max_mono_by_flight[f.flight_id] -
                           (move_costs[f.flight_id, t] + delay_costs[f.flight_id, t]))
            else:
                var_obj = -(move_costs[f.flight_id, t] + delay_costs[f.flight_id, t])
            next_reschedule_var = my_auction_model.addVar(
                name='F' + str(f.flight_id) + 'T' + str(t),
                obj=var_obj,
                vtype=gurobipy.GRB.BINARY)
            reschedule_vars[f.flight_id, t] = next_reschedule_var
    my_auction_model.update()

    ipvar_struct = AuctionVarStruct.build(flights=flights, remove_vars=remove_vars, reschedule_vars=reschedule_vars,
                                          max_displacement=max_displacement, n_slots=n_slots)

    if monopoly_constraint_rate is not None and peak_time_range is not None:
        add_monopoly_constraints(flights, monopoly_constraint_rate, peak_time_range,
                                 reschedule_vars, my_auction_model, profile)

    # Add constraints that limit displacement
    if verbose:
        print("Adding displacement constraints")
    for f in flights:
        displacement_constr_LHS = gurobipy.LinExpr()
        displacement_constr_LHS.add(remove_vars[f.flight_id])
        earliest_slot = max(f.slot_time - max_displacement, 0)
        latest_slot = min(f.slot_time + max_displacement, n_slots - 1)
        for t in range(earliest_slot, latest_slot + 1):
            displacement_constr_LHS.add(reschedule_vars[f.flight_id, t])
        my_auction_model.addConstr(displacement_constr_LHS, gurobipy.GRB.EQUAL, 1.0)

    # Add passenger connection constraints
    if verbose:
        print("Adding connection constraints")
    slot_time_dict = {f.flight_id: f.slot_time for f in flights}
    for f_id1, connect_list in connections.items():
        earliest_f1 = max(slot_time_dict[f_id1] - max_displacement, 0)
        latest_f1 = min(slot_time_dict[f_id1] + max_displacement, n_slots - 1)
        for f_id2 in connect_list:
            earliest_f2 = max(slot_time_dict[f_id2] - max_displacement, 0)
            latest_f2 = min(slot_time_dict[f_id2] + max_displacement, n_slots - 1)
            for t_1 in range(earliest_f1, latest_f1 + 1):
                connection_constr_LHS = gurobipy.LinExpr()
                connection_constr_LHS.add(remove_vars[f_id2])
                # Check to see if connection is possible
                earliest_connect = max(earliest_f2, t_1 + min_connect)
                latest_connect = min(t_1 + max_connect, n_slots - 1, latest_f2)
                for t_2 in range(earliest_connect, latest_connect + 1):
                    connection_constr_LHS.add(reschedule_vars[f_id2, t_2])
                my_auction_model.addConstr(connection_constr_LHS,
                                           gurobipy.GRB.GREATER_EQUAL,
                                           reschedule_vars[f_id1, t_1])

    airline_aircraft_dict = flightsched.find_airline_aircraft(flights)
    airline_aircraft_imbalances = flightsched.find_airline_aircraft_imbalances(
        airline_aircraft_dict)
    airline_aircraft_overnight = flightsched.find_max_overnight_by_airline_aircraft(flights,
                                                                                    turnaround,
                                                                                    n_slots)
    # Add aircraft reserve constraints
    if verbose:
        print("Adding reserve constraints")
    for (a, b), f_list in airline_aircraft_dict.items():
        for t_constr in range(0, n_slots):
            reserve_constr_LHS = gurobipy.LinExpr()
            for f in f_list:
                earliest_slot = max(f.slot_time - max_displacement, 0)
                if f.is_arrival:
                    latest_slot = min(t_constr - turnaround,
                                      f.slot_time + max_displacement, n_slots - 1)
                    if earliest_slot <= latest_slot:
                        for t_var in range(earliest_slot, latest_slot + 1):
                            reserve_constr_LHS.add(reschedule_vars[f.flight_id, t_var], -1)
                else:
                    latest_slot = min(t_constr, f.slot_time + max_displacement, n_slots - 1)
                    if earliest_slot <= latest_slot:
                        for t_var in range(earliest_slot, latest_slot + 1):
                            reserve_constr_LHS.add(reschedule_vars[f.flight_id, t_var], 1)
            # Add departure variables
            my_auction_model.addConstr(reserve_constr_LHS, gurobipy.GRB.LESS_EQUAL, airline_aircraft_overnight[a, b])

    if verbose:
        print("Adding balance constraints")
    # Add departure/arrival balance constraints
    for (a, b), f_list in airline_aircraft_dict.items():
        balance_constr_LHS = gurobipy.LinExpr()
        for f in f_list:
            earliest_slot = max(f.slot_time - max_displacement, 0)
            latest_slot = min(f.slot_time + max_displacement, n_slots - 1)
            for t in range(earliest_slot, latest_slot + 1):
                if f.is_arrival:
                    balance_constr_LHS.add(reschedule_vars[f.flight_id, t], -1)
                else:
                    balance_constr_LHS.add(reschedule_vars[f.flight_id, t])

        my_auction_model.addConstr(balance_constr_LHS, gurobipy.GRB.LESS_EQUAL,
                                   max(airline_aircraft_imbalances[a, b], 0))
        my_auction_model.addConstr(balance_constr_LHS, gurobipy.GRB.GREATER_EQUAL,
                                   min(airline_aircraft_imbalances[a, b], 0))

    if verbose:
        print("Adding profile constraints")
    # Add profile constraints
    for t, n in enumerate(profile):
        slot_constr_lhs = gurobipy.LinExpr()
        for f in flights:
            earliest_slot = max(f.slot_time - max_displacement, 0)
            latest_slot = min(f.slot_time + max_displacement, n_slots - 1)
            if earliest_slot <= t <= latest_slot:
                slot_constr_lhs.add(reschedule_vars[f.flight_id, t])

        my_auction_model.addConstr(slot_constr_lhs, gurobipy.GRB.LESS_EQUAL, n,
                                   name='PC' + str(t))

    my_auction_model.update()
    return AuctionIpStruct(model=my_auction_model, ipvars=ipvar_struct)


def add_monopoly_constraints(flights, monopoly_constraint_rate, peak_time_range, reschedule_vars, model,
                             profile):
    # Construct LHS of constraints
    max_peak_slots = sum(profile[t] for t in peak_time_range) * monopoly_constraint_rate
    monopoly_LHS_by_airline = {}

    # Go through each flight and add it to appropriate slot.
    for f in flights:
        if f.airline not in monopoly_LHS_by_airline:
            monopoly_LHS_by_airline[f.airline] = gurobipy.LinExpr()
        for t in peak_time_range:
            if (f.flight_id, t) in reschedule_vars:
                monopoly_LHS_by_airline[f.airline].addTerms(1.0, reschedule_vars[f.flight_id, t])

    for airline, lhs in monopoly_LHS_by_airline.items():
        model.addConstr(lhs, gurobipy.GRB.LESS_EQUAL, max_peak_slots, name="MONO_CONSTR: " + airline)
    model.update()
    return


def check_constraints(new_flights, old_flights, connections,
                      min_connect, max_connect, max_displacement,
                      turnaround, profile, peak_time_range, monopoly_constraint_rate):
    n_slots = len(profile)
    if not check_flight_balance(new_flights, old_flights):
        warnings.warn("Flight balance violated", RuntimeWarning)
    if not check_connections(connections, new_flights, min_connect, max_connect):
        warnings.warn("Connections violated", RuntimeWarning)
    if not check_overnight_numbers(new_flights, old_flights, turnaround, n_slots):
        warnings.warn("Overnight constraints violated", RuntimeWarning)
    if not check_profile(new_flights, profile):
        warnings.warn("Profile constraints violated", RuntimeWarning)
    if not check_max_displacement(new_flights, old_flights, max_displacement):
        warnings.warn("Max displacement constraints violated", RuntimeWarning)
    if peak_time_range is not None and monopoly_constraint_rate is not None:
        if not check_monopoly_constraints(new_flights, profile, peak_time_range,
                                          monopoly_constraint_rate):
            warnings.warn("Monopoly constraints violated.", RuntimeWarning)
    if not check_flight_balance(new_flights, old_flights):
        warnings.warn("Flight balance constraints violated.", RuntimeWarning)


def check_profile(new_flights, profile):
    num_times = len(profile)
    f_profile = flightsched.get_aggregated_flight_schedule(new_flights, num_times)
    for x, y in zip(f_profile, profile):
        if x > y:
            return False
    return True


def check_flight_balance(new_flights, old_flights) -> bool:
    old_airline_aircraft_imbalance = flightsched.find_airline_aircraft(old_flights)
    new_airline_aircraft_imbalance = flightsched.find_airline_aircraft(new_flights)
    for aa, old_imbalance in old_airline_aircraft_imbalance:
        if aa in new_airline_aircraft_imbalance:
            new_imbalance = new_airline_aircraft_imbalance[aa]
            if new_imbalance < min(old_imbalance, 0) or new_imbalance > max(old_imbalance, 0):
                return False
    return True


def check_overnight_numbers(new_flights, old_flights, turnaround, n_slots) -> bool:
    old_airline_aircraft_overnight = flightsched.find_max_overnight_by_airline_aircraft(
        old_flights, turnaround, n_slots)
    new_airline_aircraft_overnight = flightsched.find_max_overnight_by_airline_aircraft(
        new_flights, turnaround, n_slots)
    for aa, old_overnight in old_airline_aircraft_overnight:
        if aa in new_airline_aircraft_overnight:
            new_overnight = new_airline_aircraft_overnight[aa]
            if new_overnight > old_overnight:
                return False
    return True


def check_connections(connections, flights, min_connect, max_connect) -> bool:
    id_to_flight = {f.flight_id: f for f in flights}
    for id_1, con_list in connections.items():
        if id_1 in id_to_flight:
            flight_1 = id_to_flight[id_1]
            for id_2 in con_list:
                if id_2 in id_to_flight:
                    flight_2 = id_to_flight[id_2]
                    if (flight_2.slot_time - flight_1.slot_time < min_connect
                            or flight_2.slot_time - flight_1.slot_time > max_connect):
                        return False
    return True


def check_max_displacement(new_flights, old_flights, max_displacement) -> bool:
    original_slot = {}
    for f in old_flights:
        original_slot[f] = f.slot_time
    for f in new_flights:
        if f in old_flights:
            displacement = abs(f.slot_time - original_slot[f])
            if displacement > max_displacement:
                return False
    return True


def check_monopoly_constraints(new_flights, profile, peak_time_period, monopoly_constraint_rate) -> bool:
    max_num_slots = sum(profile[t] for t in peak_time_period) * monopoly_constraint_rate
    airline_num_slots = {}
    for f in new_flights:
        if f.slot_time in peak_time_period:
            if f.airline not in airline_num_slots:
                airline_num_slots[f.airline] = 0
            airline_num_slots[f.airline] += 1
            if airline_num_slots[f.airline] > max_num_slots:
                return f.airline
    return True


def get_monopoly_ub(market_shares, flights, remove_costs, monopoly_benefit):
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
    return value_by_flight
