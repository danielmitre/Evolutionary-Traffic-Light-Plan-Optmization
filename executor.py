#!/usr/bin/env python

from analyzer import delay, normal_delay, max_normal_delay, metrics, extract_tripinfos
from queue import Queue
from threading import Thread
from typing import Iterable, Callable
import numpy as np
import subprocess
import os


def hydrate_template(template_file, output_file, chromosome: np.ndarray, chrom_id):
    args = {'id': chrom_id}

    for idx_base in range(0, len(chromosome), 5):
        intersection = (idx_base // 5) + 1
        for (offset, phase) in enumerate(('NS1', 'NS2', 'WE1', 'WE2')):
            args['{}_{}'.format(phase, intersection)] = \
                chromosome[idx_base + offset]

        args['O_{}'.format(intersection)] = chromosome[idx_base + 4]

    with open(template_file, 'r') as in_file:
        with open(output_file, 'w') as out_file:
            out_file.write(in_file.read().format(**args))

    return output_file


def copy_file(source, dest):
    with open(source, 'rb') as f1:
        with open(dest, 'wb') as f2:
            f2.write(f1.read())


def prefix(gen, chrom_id):
    return '{:03d}/{:02d}'.format(gen, chrom_id)


def run_simulation(directory, gen, chrom_id, chromosome):

    os.makedirs(os.path.dirname(
        '{}/gen_{}.pkl'.format(directory, prefix(gen, chrom_id))), exist_ok=True)
    # dumping chromosome
    chromosome.dump('{}/gen_{}.pkl'.format(directory, prefix(gen, chrom_id)))

    tls_file = hydrate_template('traffic-lights.template.xml',
                                '{}/gen_{}.tls.xml'.format(directory, prefix(gen, chrom_id)), chromosome, prefix(gen, chrom_id))
    trip_info_file = '{}/gen_{}.tripinfo.xml'.format(
        directory, prefix(gen, chrom_id))

    with open('{}/gen_{}.log'.format(directory, prefix(gen, chrom_id)), 'w') as log_file:
        subprocess.call(['sumo', '-n', 'network.net.xml', '-r', 'vehicles.rou.xml',
                         '-a', tls_file, '--threads=1', '--tripinfo-output={}'.format(trip_info_file)], stdout=log_file, stderr=log_file)

    return trip_info_file


def consumer(input_queue, output_queue, train_directory, metric_name):
    while True:
        gen, chrom_id, chromosome = input_queue.get()

        trip_file = run_simulation(
            train_directory, gen, chrom_id, chromosome)

        output_queue.put((chrom_id, metrics(trip_file, [{
            'delay': delay,
            'normal_delay': normal_delay,
            'max_normal_delay': max_normal_delay
        }[metric_name]])))

        if chrom_id != 0:
            os.remove(trip_file)
        input_queue.task_done()


def start_consumers(input_queue: Queue, output_queue: Queue, n_consumers: int, train_directory: str, metric_name: str):
    consumers = [Thread(target=lambda: consumer(input_queue, output_queue, train_directory, metric_name))
                 for _ in range(n_consumers)]

    [c.setDaemon(True) for c in consumers]
    [c.start() for c in consumers]

    return consumers
