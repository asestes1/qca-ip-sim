import pandas
import numpy
import matplotlib.pyplot
import attr
import typing
import gurobipy
import collections


def cargo_flight_seat_dict() -> typing.Dict[str, int]:
    """
    This defines a dictionary that gives seat numbers for cargo flights

    :return:
    """
    return {'DC10': 380, 'B741': 426, 'B742': 426, 'B744': 432,
            'B752': 215, 'B762': 278, 'MD11': 296, 'A310': 209}


@attr.s(frozen=True, kw_only=True)
class Flight(object):
    is_arrival = attr.ib(type=bool)
    slot_time = attr.ib(type=int)
    n_seats = attr.ib(type=int)
    airline = attr.ib(type=str)
    aircraft_type = attr.ib(type=str)
    flight_id = attr.ib(type=int)

    def copy_reschedule(self, new_time):
        return Flight(is_arrival=self.is_arrival,
                      slot_time=new_time,
                      n_seats=self.n_seats,
                      airline=self.airline,
                      aircraft_type=self.aircraft_type,
                      flight_id=self.flight_id)

    def copy_change_airline(self, new_airline):
        return Flight(is_arrival=self.is_arrival,
                      slot_time=self.slot_time,
                      n_seats=self.n_seats,
                      airline=new_airline,
                      aircraft_type=self.aircraft_type,
                      flight_id=self.flight_id)


def get_new_flight_schedule(flights: typing.Iterable[Flight], n_slots: int, ip_model: gurobipy.Model) -> \
        typing.Set[Flight]:
    new_flights = set()
    for f in flights:
        found = False
        for t in range(0, n_slots):
            sched_var = ip_model.getVarByName('F' + str(f.flight_id) + 'T' + str(t))
            if sched_var is not None and abs(sched_var.getAttr("X") - 1.0) < 0.0001:
                if not found:
                    found = True
                    new_flights.add(f.copy_reschedule(t))
                else:
                    print("Error: flight is double scheduled")

        if not found:
            remove_var = ip_model.getVarByName('R' + str(f.flight_id))
            if remove_var is None or remove_var.getAttr("X") < 0.000001:
                print("Error: flight is neither scheduled nor removed")
    return new_flights

def read_flights(oag_file, airport, date, min_per_slot):
    oag_dataframe = pandas.read_csv(oag_file)
    oag_dataframe = oag_dataframe[pandas.to_datetime(oag_dataframe['Date'])
                                  == date]
    flights = [parse_flight_row(oag_dataframe.loc[r], airport, min_per_slot)
               for r in oag_dataframe.axes[0]]
    return flights


def parse_flight_row(row, airport, min_per_slot):
    if row['LEAVE'] == airport:
        is_arrival = False
    elif row['ARRIVE'] == airport:
        is_arrival = True
    else:
        raise ValueError('Flight is neither an arrival nor departure from '
                         + airport)

    if is_arrival:
        time = time_to_slot(row['ARRTIME'], min_per_slot)
    else:
        time = time_to_slot(row['LVETIME'], min_per_slot)

    airline = row['FAACARR']
    n_seats = row['SEATS']
    flight_id = row['ID']
    aircraft_type = row['EQUIP']
    if n_seats == 0:
        n_seats = cargo_flight_seat_dict()[aircraft_type]
    return Flight(is_arrival=is_arrival,
                  slot_time=time,
                  n_seats=n_seats,
                  airline=airline,
                  aircraft_type=aircraft_type,
                  flight_id=flight_id)


def time_to_slot(time, min_per_slot):
    return int((int(time / 100) * 60 + (time % 100)) / min_per_slot)


def find_max_overnight(flights, turnaround, n_slots):
    cdfs = {}
    for f in flights:
        if (f.airline, f.aircraft_type) not in cdfs:
            cdfs[f.airline, f.aircraft_type] = numpy.zeros(n_slots)
        if f.is_arrival:
            if f.slot_time < n_slots - turnaround:
                new_vector = [0] * (f.slot_time + turnaround)
                new_vector.extend([-1] * (n_slots - f.slot_time - turnaround))
            else:
                new_vector = [0] * n_slots
        else:
            new_vector = [0] * f.slot_time
            new_vector.extend([1] * (n_slots - f.slot_time))
        cdfs[f.airline, f.aircraft_type] = numpy.add(
            cdfs[f.airline, f.aircraft_type], numpy.array(new_vector))
    #    for (airline,aircraft),cdf in cdfs.items():
    #        matplotlib.pyplot.figure()
    #        matplotlib.pyplot.title("Airline: "+airline+", aircraft: "+aircraft)
    #        matplotlib.pyplot.xlabel("Time period")
    #        matplotlib.pyplot.ylabel("Flights")
    #        matplotlib.pyplot.plot(range(0,96),cdf,
    #                               range(0,96),cdfs[airline,aircraft])
    #        matplotlib.pyplot.legend('Arrivals','Departures')
    return max([numpy.amax(cdf) for cdf in cdfs.values()])


def get_aggregated_flight_schedule(flights, num_times, separate_flights=False):
    if not separate_flights:
        agg_schedule = numpy.zeros(num_times)
        for f in flights:
            agg_schedule[f.slot_time] += 1
        return agg_schedule
    else:
        arr = set(f for f in flights if f.is_arrival)
        dep = set(f for f in flights if not f.is_arrival)
        return {'arrivals': get_aggregated_flight_schedule(arr, num_times, False),
                'departures': get_aggregated_flight_schedule(dep, num_times, False)}


def find_airline_aircraft(flights):
    airline_aircraft_dict = {}
    for f in flights:
        key = (f.airline, f.aircraft_type)
        if key not in airline_aircraft_dict:
            airline_aircraft_dict[key] = set()
        airline_aircraft_dict[key].add(f)
    return airline_aircraft_dict


def find_airline_aircraft_imbalances(airline_aircraft_dict):
    airline_aircraft_imbalances = {aa: 0 for aa in airline_aircraft_dict.keys()}
    for aa, set_f in airline_aircraft_dict.items():
        for f in set_f:
            if f.is_arrival:
                airline_aircraft_imbalances[aa] -= 1
            else:
                airline_aircraft_imbalances[aa] += 1
    return airline_aircraft_imbalances


def find_max_overnight_by_airline_aircraft(flights, turnaround, n_slots):
    cdfs = {}
    for f in flights:
        if (f.airline, f.aircraft_type) not in cdfs:
            cdfs[f.airline, f.aircraft_type] = numpy.zeros(n_slots)
        if f.is_arrival:
            if f.slot_time < n_slots - turnaround:
                new_vector = [0] * (f.slot_time + turnaround)
                new_vector.extend([-1] * (n_slots - f.slot_time - turnaround))
            else:
                new_vector = [0] * n_slots
        else:
            new_vector = [0] * f.slot_time
            new_vector.extend([1] * (n_slots - f.slot_time))
        cdfs[f.airline, f.aircraft_type] = numpy.add(
            cdfs[f.airline, f.aircraft_type], numpy.array(new_vector))
    return {key: numpy.amax(cdf) for key, cdf in cdfs.items()}


def plot_cumulative_aircraft(flights, n_slots):
    departure_plots = {}
    arrival_plots = {}
    for f in flights:
        if (f.airline, f.aircraft_type) not in departure_plots:
            departure_plots[f.airline, f.aircraft_type] = numpy.zeros(n_slots)
            arrival_plots[f.airline, f.aircraft_type] = numpy.zeros(n_slots)
        new_vector = [0] * f.slot_time
        new_vector.extend([1] * (n_slots - f.slot_time))
        if f.is_arrival:
            arrival_plots[f.airline, f.aircraft_type] = numpy.add(
                arrival_plots[f.airline, f.aircraft_type], numpy.array(new_vector))
        else:
            departure_plots[f.airline, f.aircraft_type] = numpy.add(
                departure_plots[f.airline, f.aircraft_type], numpy.array(new_vector))

    for (airline, aircraft), cdf in arrival_plots.items():
        matplotlib.pyplot.figure()
        matplotlib.pyplot.title("Airline: " + airline + ", aircraft: " + aircraft)
        matplotlib.pyplot.xlabel("Time period")
        matplotlib.pyplot.ylabel("Flights")
        matplotlib.pyplot.plot(range(0, n_slots), cdf,
                               range(0, n_slots), departure_plots[airline, aircraft])

        matplotlib.pyplot.legend(['Arrivals', 'Departures'])
    return

def get_airline_market_shares(flights: typing.Iterable[Flight], peak_time_range, profile) -> typing.Dict[
    str, float]:
    num_peak_slots = sum(profile[t] for t in peak_time_range)
    share_by_airline = collections.defaultdict(float)
    for f in flights:
        if f.slot_time in peak_time_range:
            if f.airline not in share_by_airline:
                share_by_airline[f.airline] = 0
            share_by_airline[f.airline] += 1 / num_peak_slots
    return share_by_airline