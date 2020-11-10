import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import os
import xmltodict
from typing import Iterable, Callable


def delay(tripinfos: dict) -> float:
    delay = 0.0
    for trip_info in tripinfos:
        waiting = float(trip_info['@waitingTime'])
        delay += waiting
    return delay / len(tripinfos)


def normal_delay(tripinfos: dict) -> float:
    normal_delay = 0.0
    for trip_info in tripinfos:
        waiting = float(trip_info['@waitingTime'])
        length = float(trip_info['@routeLength'])
        normal_delay += waiting / length
    return normal_delay / len(tripinfos)


def max_normal_delay(tripinfos: dict) -> float:
    total_delay = {}
    count_trips = {}
    for trip_info in tripinfos:
        route = trip_info['@departLane'] + trip_info['@arrivalLane']
        waiting = float(trip_info['@waitingTime'])
        length = float(trip_info['@routeLength'])
        total_delay[route] = (waiting / length) + total_delay.get(route, 0.0)
        count_trips[route] = 1.0 + count_trips.get(route, 0.0)
    for route in total_delay.keys():
        total_delay[route] /= count_trips[route]
    return max(total_delay.values())


def parse_xml(filename: str) -> dict:
    with open(filename, 'rb') as f:
        return xmltodict.parse(f)


def extract_tripinfos(trip_file: str) -> dict:
    try:
        parsed = parse_xml(trip_file)
        return parsed['tripinfos']['tripinfo']
    except Exception as e:
        print('problem with {}: {}'.format(trip_file, e))
        print('parsed[\'tripinfos\'] = {}'.format(parsed['tripinfos']))
        raise e


def metrics(trip_file: str, metrics: Iterable[Callable[[dict], float]]) -> [float]:
    tripinfos = extract_tripinfos(trip_file)
    return [metric(tripinfos) for metric in metrics]


def show_stats(population, hof):
    fits = [ind.fitness.values[0] for ind in population]
    length = len(fits)
    mean = sum(fits) / length
    sum2 = sum(x*x for x in fits)
    std = abs(sum2 / length - mean**2)**0.5

    print("\tbest {}: {}".format(len(hof), hof))
    print("\tMin %s" % min(fits))
    print("\tMax %s" % max(fits))
    print("\tAvg %s" % mean)
    print("\tStd %s" % std)


def metrics_to_csv(directory: str, generations: int, fitness: str, legible_name: str, run: int):
    print('computing {}...'.format(directory))
    df = pd.DataFrame()
    metric_names = ['delay', 'normal_delay', 'max_normal_delay']
    metric_functions = [delay, normal_delay, max_normal_delay]
    idx = 0
    for gen in range(generations + 1):
        metric_results = metrics('{}/gen_{:03d}/00.tripinfo.xml'.format(directory, gen),
                                 metric_functions)
        df_metrics = {metric_name: metric for (metric_name, metric) in
                      zip(metric_names, metric_results)}
        current = pd.DataFrame(
            {**df_metrics, 'gen': gen, 'chrom': 0}, index=[idx])
        if (idx % 50) == 0:
            print(current)
        df = pd.concat([df, current], copy=False)
        idx += 1

    df['fitness'] = legible_name
    df['run'] = run

    os.makedirs(os.path.dirname('summary/'), exist_ok=True)
    df.to_csv('summary/{}-{:02d}.csv'.format(fitness, run), index=False)

    tripinfos = extract_tripinfos(
        '{}/gen_{:03d}/00.tripinfo.xml'.format(directory, generations))

    detail = pd.DataFrame()
    detail['waitingTime'] = [float(trip['@waitingTime']) for trip in tripinfos]
    detail['routeLength'] = [float(trip['@routeLength']) for trip in tripinfos]
    detail['normalWaiting'] = detail['waitingTime'] / detail['routeLength']
    detail['fitness'] = legible_name
    detail['run'] = run

    os.makedirs(os.path.dirname('trips/'), exist_ok=True)
    detail.to_csv('trips/{}-{:02d}.csv'.format(fitness, run), index=False)


if __name__ == '__main__':
    for i in range(2, 11):
        metrics_to_csv('train/delay/delay_train%d' %
                       i, 50, 'delay', 'atraso médio', i)
        metrics_to_csv('train/normal_delay/normal_delay_train%d' %
                       i, 50, 'normal_delay', 'atraso de malha padronizado', i)
        metrics_to_csv('train/max_normal_delay/max_normal_delay_train%d' %
                       i, 50, 'max_normal_delay', 'pior atraso de rota padronizado', i)
