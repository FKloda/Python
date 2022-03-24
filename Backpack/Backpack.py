import numpy as np
import random
import copy
import math
import matplotlib.pyplot as plt


def read_input(path):
    file = open(path)
    first = file.readline()
    first = first.split()
    item_count = int(first[0])
    capacity = int(first[1])
    values = np.empty(item_count, dtype=int)
    weights = np.empty(item_count, dtype=int)
    line = file.readline()
    i = 0

    while line:
        split = line.split()
        values[i] = split[0]
        weights[i] = split[1]
        i += 1
        line = file.readline()

    return item_count, capacity, values, weights


# individuals have to be valid
def fitness(individual):
    value = 0
    for i in range(0, len(individual)):
        value += individual[i] * values[i]
    return value


def validaze_individual(individual):
    indicies = []
    individual_weight = 0
    new_individual = copy.deepcopy(individual)
    for i in range(0, len(individual)):
        if individual[i] == 1:
            indicies.append(i)
            individual_weight += weights[i]

    while individual_weight > capacity:
        index = random.randint(0, len(indicies) - 1)
        new_individual[indicies[index]] = 0
        individual_weight -= weights[indicies[index]]
        indicies.pop(index)
    return new_individual


def mutation(population, indiv_prob=0.2, value_prob=0.15):
    new_population = []
    for individual in population:
        if random.random() < indiv_prob:
            for i in range(0, len(individual)):
                if random.random() < value_prob:
                    individual[i] = abs(individual[i] - 1)
            individual = validaze_individual(individual)
        new_population.append(individual)
    return new_population


def create_random_population(population_size):
    population = []
    for i in range(0, population_size):
        individual = np.zeros(individual_size)
        weight = 0
        while True:
            x = random.randint(0, individual_size-1)
            if individual[x] == 0:
                weight += weights[x]
                if weight > capacity:
                    break
                individual[x] = 1
        population.append(individual)
    return population


def rulet_selection(population, fitness_value):
    return copy.deepcopy(random.choices(population, weights=fitness_value, k=len(population)))


def tournament_selection(population, fitness_value):
    new_population = []
    for i in range(0, len(population)):
        individuals = []
        fitnesses = []
        for _ in range(0, 2):
            idx = random.randint(0, len(population)-1)
            individuals.append(population[idx])
            fitnesses.append(fitness_value[idx])
        new_population.append(copy.deepcopy(individuals[np.argmax(fitnesses)]))
    return new_population


def one_point_crossover(population):
    new_population = []

    for i in range(0, len(population) // 2):
        individual1 = copy.deepcopy(population[2 * i])
        individual2 = copy.deepcopy(population[2 * i + 1])

        crossover_point = random.randint(0, len(individual1))
        end2 = copy.deepcopy(individual2[:crossover_point])
        individual2[:crossover_point] = individual1[:crossover_point]
        individual1[:crossover_point] = end2

        individual1 = validaze_individual(individual1)
        individual2 = validaze_individual(individual2)
        new_population.append(individual1)
        new_population.append(individual2)

    return new_population


def two_point_crossover(population):
    new_population = []

    for i in range(0, len(population) // 2):
        individual1 = copy.deepcopy(population[2 * i])
        individual2 = copy.deepcopy(population[2 * i + 1])

        crossover_point1 = random.randint(1, len(individual1))
        crossover_point2 = random.randint(0, crossover_point1)
        recomb = copy.deepcopy(individual2[crossover_point2:crossover_point1])
        individual2[crossover_point2:crossover_point1] = individual1[crossover_point2:crossover_point1]
        individual1[crossover_point2:crossover_point1] = recomb

        individual1 = validaze_individual(individual1)
        individual2 = validaze_individual(individual2)

        new_population.append(individual1)
        new_population.append(individual2)

    return new_population


def evolution(population_size, max_generations):
    max_fitness = []
    population = create_random_population(population_size)
    # all_time_best = population[0]

    for i in range(0, max_generations):
        fitness_value = list(map(fitness, population))

        best_individual = copy.deepcopy(population[np.argmax(fitness_value)])
        # if fitness(best_individual) > fitness(all_time_best):
        #     all_time_best = best_individual

        max_fitness.append(max(fitness_value))
        selected = rulet_selection(population, fitness_value)
        # selected = tournament_selection(population, fitness_value)
        selected[random.randint(0, len(selected)-1)] = best_individual
        # offsprings = one_point_crossover(selected)
        offsprings = two_point_crossover(selected)
        mutated_population = mutation(offsprings)
        population = mutated_population
        population[random.randint(0, len(selected)-1)] = best_individual
        # print(i)

    fitness_value = list(map(fitness, population))
    max_fitness.append(max(fitness_value))
    best_individual = population[np.argmax(fitness_value)]
    # if fitness(best_individual) > fitness(all_time_best):
    #     all_time_best = best_individual

    return best_individual, population, max_fitness


individual_size, capacity, values, weights = read_input("Data_100.txt")
fitness_all_time = []
for _ in range (1, 6):
    best, population, max_fitness = evolution(population_size=200, max_generations=300)

    fitness_all_time.extend(max_fitness)
    print('best fitness: ', fitness(best))
    # print('best individual: ', best)
    individual_weight = 0
    for i in range(0, len(best)):
        if best[i] == 1:
            individual_weight += weights[i]
    print('his weight: ', individual_weight)
    print('max weight: ', capacity)


plt.plot(fitness_all_time)
plt.ylabel('Fitness')
plt.xlabel('Generace')
plt.show()
