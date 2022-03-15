import sys

import numpy as np
import random
import copy
# %matplotlib inline
import matplotlib.pyplot as plt




def random_population(population_size, individual_size):
    population = []

    for i in range(0, population_size):
        individual = np.random.choice([0, 1], size=(individual_size,), p=[1 / 2, 1 / 2])
        population.append(individual)

    return population


def fitness(individual):
    fit = 0
    for i in range(0, len(individual)):
        if individual[i] == 1:
            fit += random_set[i]

    if fit == value:
        print('sum of elements: ', fit)
        print('found perfect fit')
        print('best individual: ', individual)

        sys.exit(0)
    return 1/abs(value - fit)


def selection(population, fitness_value):
    return copy.deepcopy(random.choices(population, weights=fitness_value, k=len(population)))


def crossover(population, cross_prob=1):
    new_population = []

    for i in range(0, len(population) // 2):
        indiv1 = copy.deepcopy(population[2 * i])
        indiv2 = copy.deepcopy(population[2 * i + 1])


        if random.random()<cross_prob:
            # zvolime index krizeni nahodne
            crossover_point = random.randint(0, len(indiv1))
            end2 = copy.deepcopy(indiv2[:crossover_point])
            indiv2[:crossover_point] = indiv1[:crossover_point]
            indiv1[:crossover_point] = end2

        new_population.append(indiv1)
        new_population.append(indiv2)

    return new_population


def mutation(population, indiv_mutation_prob=0.05, bit_mutation_prob=0.2):
    new_population = []

    for i in range(0, len(population)):
        individual = copy.deepcopy(population[i])
        if random.random() < indiv_mutation_prob:
            for j in range(0, len(individual)):
                if random.random() < bit_mutation_prob:
                    if individual[j] == 1:
                        individual[j] = 0
                    else:
                        individual[j] = 1

        new_population.append(individual)

    return new_population


def evolution(population_size, individual_size, max_generations):
    max_fitness = []
    population = random_population(population_size, individual_size)
    best_individual = [0] * individual_size
    total_best = best_individual

    for i in range(0, max_generations):
        fitness_value = list(map(fitness, population))
        max_fitness.append(max(fitness_value))
        parents = selection(population, fitness_value)
        children = crossover(parents)
        mutated_children = mutation(children)
        population = mutated_children
        best_individual = population[np.argmax(fitness_value)]
        if (fitness(best_individual)>fitness(total_best)):
            total_best = best_individual
        if verbose :
            print('Generation: ', i, ' - best individual fit: ', fitness(best_individual))
            print('Total best individual fit: ', fitness(total_best))

    # spocitame fitness i pro posledni populaci
    fitness_value = list(map(fitness, population))
    max_fitness.append(max(fitness_value))
    best_individual = population[np.argmax(fitness_value)]
    if (fitness(best_individual)>fitness(total_best)):
        total_best = best_individual

    return total_best, population, max_fitness


verbose = True
random_set = []
for i in range(0, 200):
    random_set.append(random.randint(1, 150))

value = sum(random_set) // 2
print(value, random_set)

best, population, max_fitness = evolution(population_size=100, individual_size=len(random_set), max_generations=100)

fit = 0
for i in range(0, len(best)):
    if best[i] == 1:
        fit += random_set[i]

print('sum of elements: ', fit)
print('best fitness: ', fitness(best))
print('best individual: ', best)

'''
plt.plot(max_fitness)
plt.ylabel('Fitness')
plt.xlabel('Generace')
plt.show()
'''
"""
print_stats = False
best, population, max_fitness = evolution(population_size=100, individual_size=50, max_generations=100)

print('best fitness: ', fitness(best))
print('best individual: ', best)


plt.plot(max_fitness)
plt.ylabel('Fitness')
plt.xlabel('Generace')
"""
