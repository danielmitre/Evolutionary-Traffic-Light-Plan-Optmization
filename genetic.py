#!/usr/bin/env python

from executor import start_consumers
from queue import Queue
from deap import base, creator, tools
import analyzer
import random
import numpy as np
import sys

PARALLELISM = 8
MAX_GENERATIONS = 50
MUTATION_PR = .01
POP_SIZE = 100
ELITISM = 24
CATACLISM_CYCLE = 100
CATACLISM_SURVIVORS = 1
PHASE_RANGE = (6., 60.)
OFFSET_RANGE = (-256., 256.)


def selParentsTournament(individuals, k, tournsize, fit_attr="fitness"):
    parent1 = tools.selTournament(individuals, 1, tournsize)[0]
    parent2 = tools.selTournament(individuals, 1, tournsize)[0]
    while id(parent2) == id(parent1):
        parent2 = tools.selTournament(individuals, 1, tournsize)[0]
    return (parent1, parent2)


def rand_interval(min_value, max_value) -> float:
    interval = max_value - min_value
    return (random.random() * interval) + min_value


def create_toolbox() -> base.Toolbox:
    toolbox = base.Toolbox()
    toolbox.register('attr_phase',
                     rand_interval, PHASE_RANGE[0], PHASE_RANGE[1])
    toolbox.register('attr_offset',
                     rand_interval, OFFSET_RANGE[0], OFFSET_RANGE[1])
    toolbox.register('individual',
                     tools.initCycle, creator.Individual,
                     (
                         toolbox.attr_phase,
                         toolbox.attr_phase,
                         toolbox.attr_phase,
                         toolbox.attr_phase,
                         toolbox.attr_offset
                     ), 9)
    toolbox.register('population', tools.initRepeat, list, toolbox.individual)

    toolbox.register('mate', tools.cxBlend, alpha=0.5)

    toolbox.register('select', selParentsTournament, tournsize=3)
    return toolbox


def register_classes():
    creator.create('FitnessMin', base.Fitness, weights=(-1,))
    creator.create('Individual', np.ndarray, fitness=creator.FitnessMin)


def fitnesses(population: list, generation: int, in_queue: Queue, out_queue: Queue):
    for (idx, individual) in enumerate(population):
        in_queue.put((generation, idx, individual))

    for _ in population:
        idx, fit = out_queue.get()
        population[idx].fitness.values = fit


def get_fit(individual):
    return individual.fitness


def load_last_population(load_from, pop_size: int, generations: int, toolbox) -> (int, list):
    fallback = toolbox.population(n=POP_SIZE)
    last_gen = -1
    for gen in range(generations):
        try:
            next_gen = []
            for chrom in range(pop_size):
                with open('{}/gen_{:03d}/{:02d}.pkl'.format(load_from, gen, chrom), 'rb') as pkl:
                    next_gen.append(np.load(pkl, allow_pickle=True))
            fallback = next_gen
            last_gen = gen
            print('generation {:03d} loaded successfully'.format(gen))
        except Exception as e:
            print(
                'Warning: Error while unpickling generation {:03d}: {}'.format(gen, e))
            break
    return max(last_gen, 0), fallback


def in_range(attr, low, high):
    return low <= attr <= high


def is_valid(individual):
    for idx in range(0, len(individual), 5):
        for offset in range(4):
            if not in_range(individual[idx + offset], PHASE_RANGE[0], PHASE_RANGE[1]):
                return False
        if not in_range(individual[idx + 4], OFFSET_RANGE[0], OFFSET_RANGE[1]):
            return False
    return True


def mutate(individual):
    for idx in range(0, len(individual), 5):
        for offset in range(4):
            if random.random() < MUTATION_PR:
                individual[idx + offset] = \
                    rand_interval(PHASE_RANGE[0], PHASE_RANGE[1])

        if random.random() < MUTATION_PR:
            individual[idx + 4] = \
                rand_interval(OFFSET_RANGE[0], OFFSET_RANGE[1])

    return individual


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print('Usage: python {} output_directory [[max_]normal_]]delay'
              .format(sys.argv[0]))
        raise Exception('Missing arguments')
    train_directory = sys.argv[1]
    metric_name = sys.argv[2]
    jobqueue = Queue()
    results = Queue()
    start_consumers(jobqueue, results, PARALLELISM,
                    train_directory, metric_name)

    hof = tools.HallOfFame(1, similar=lambda a, b: id(a) == id(b))
    register_classes()
    toolbox = create_toolbox()

    gen, population = load_last_population(
        train_directory, POP_SIZE, MAX_GENERATIONS, toolbox)
    population = list(filter(is_valid, population))

    while gen <= MAX_GENERATIONS:
        fitnesses(population, gen, jobqueue, results)
        population.sort(key=get_fit, reverse=True)

        hof.update(population)
        print('generation {} (len={}):'.format(
            gen, len(population)))
        analyzer.show_stats(population, hof)

        next_pop = []
        if (gen % CATACLISM_CYCLE) == 0:
            next_pop = population[:CATACLISM_SURVIVORS]
            next_pop.extend(toolbox.population(
                n=POP_SIZE - CATACLISM_SURVIVORS))

        next_pop = population[:ELITISM]
        while len(next_pop) < POP_SIZE:
            parent1, parent2 = map(
                toolbox.clone, toolbox.select(population, k=2))

            for child in toolbox.mate(parent1, parent2):
                if is_valid(child):
                    del child.fitness.values
                    next_pop.append(mutate(child))

        gen += 1
        population = next_pop
