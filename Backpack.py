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

    if individual_weight > capacity:
        new_individual[indicies[random.randint(0, len(indicies) - 1)]] = 0
        new_individual = validaze_individual(new_individual)
    return new_individual


def mutation(population, indiv_prob=0.1, value_prob=0.2):
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
        individual = np.random.choice([0, 1], size=(individual_size,), p=[1/2, 1/2])
        individual = validaze_individual(individual)
        population.append(individual)
    return population


def selection(population,fitness_value):
    return copy.deepcopy(random.choices(population, weights=fitness_value, k=len(population)))


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
    all_time_best = population[0]

    for i in range(0, max_generations):
        fitness_value = list(map(fitness, population))

        best_individual = population[np.argmax(fitness_value)]
        if fitness(best_individual) > fitness(all_time_best):
            all_time_best = best_individual

        max_fitness.append(max(fitness_value))
        selected = selection(population, fitness_value)
        # offsprings = one_point_crossover(selected)
        offsprings = two_point_crossover(selected)
        mutated_population = mutation(offsprings)
        population = mutated_population


    fitness_value = list(map(fitness, population))
    max_fitness.append(max(fitness_value))
    best_individual = population[np.argmax(fitness_value)]
    if fitness(best_individual) > fitness(all_time_best):
        all_time_best = best_individual
    print("Generation: ", max_generations)
    print("     All time best fitness:  ", fitness(all_time_best))
    print("     Current best fitness:   ", fitness(best_individual))

    return all_time_best, population, max_fitness


individual_size, capacity, values, weights = read_input("Data_100.txt")
best, population, max_fitness = evolution(population_size=100, max_generations=100)

print('best fitness: ', fitness(best))
print('best individual: ', best)
# individual_weight = 0
# for i in range(0, len(best)):
#     if best[i] == 1:
#         individual_weight += weights[i]
# print('his weight: ', individual_weight)


plt.plot(max_fitness)
plt.ylabel('Fitness')
plt.xlabel('Generace')
plt.show()
