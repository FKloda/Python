import numpy as np
import random
import copy
import math
import matplotlib.pyplot as plt

def fitness(individual):
    N = len(individual)
    A = 10
    value = A*N
    for i in range(0, len(individual)):
        value += individual[i]**2 - A*math.cos(2*math.pi*individual[i])
    return -value


def random_population(population_size, individual_size):
    population = []

    for i in range(0, population_size):
        individual = np.random.uniform(-5.12, 5.12, size=(individual_size,))
        population.append(individual)

    return population


def crossover_mean(population, cross_prob=0.8, alpha=0.25):
    new_population = []

    for i in range(0, len(population) // 2):
        indiv1 = copy.deepcopy(population[2 * i])
        indiv2 = copy.deepcopy(population[2 * i + 1])
        child1 = indiv1
        child2 = indiv2
        if random.random() < cross_prob:
            for i in range(0, len(indiv1)):
                child1[i] = alpha * indiv1[i] + (1 - alpha) * indiv2[i]
                child2[i] = (1 - alpha) * indiv1[i] + alpha * indiv2[i]
        new_population.append(child1)
        new_population.append(child2)

    return new_population


def diff_muatation(population, alpha=0.8):
    new_population = []

    for i in range(0, len(population)):
        indiv1 = copy.deepcopy(population[random.randint(0, len(population) - 1)])
        indiv2 = population[random.randint(0, len(population) - 1)]
        indiv3 = population[random.randint(0, len(population) - 1)]
        child = indiv1
        for i in range(0, len(indiv1)):
            child[i] = child[i] + alpha * (indiv2[i] - indiv3[i])
        # ref = population[random.randint(0, len(population) - 1)]
        # if fitness(ref) > fitness(child):
            # new_population.append(copy.deepcopy(ref))
        # else:
            # new_population.append(child)
        new_population.append(child)

    return new_population


def unweighted_mutation_switch(population,individual_mutation_prob=0.2,value_mutation_prob=0.1):
    new_population = []
    for i in range(0,len(population)):
        individual = copy.deepcopy(population[i])
        if random.random() < individual_mutation_prob:
            for i in range(0,len(individual)):
                if random.random() < value_mutation_prob:
                    individual[i] = np.random.uniform(-5.12, 5.12)
        new_population.append(individual)
    return new_population


def weighted_mutation_switch(population, sigma, individual_mutation_prob=0.2, value_mutation_prob=1):
    new_population = []
    for i in range(0, len(population)):
        individual = copy.deepcopy(population[i])
        if random.random() < individual_mutation_prob:
            for i in range(0, len(individual)):
                if random.random() < value_mutation_prob:
                    individual[i] += sigma * np.random.normal(0, 1)
        new_population.append(individual)
    return new_population


def selection(population, fitness_value, k):
    new_population = []
    for i in range(0, len(population)):
        individuals = []
        fitnesses = []
        for _ in range(0, k):
            idx = random.randint(0, len(population)-1)
            individuals.append(population[idx])
            fitnesses.append(fitness_value[idx])
        new_population.append(copy.deepcopy(individuals[np.argmax(fitnesses)]))
    return new_population


def evolution(population_size, individual_size, max_generations):
    max_fitness = []
    population = random_population(population_size, individual_size)
    sigma = 0.6 * 5.12

    for i in range(0, max_generations):
        sigma = 0.98 * sigma
        fitness_value = list(map(fitness, population))
        max_fitness.append(max(fitness_value))
        parents = selection(population, fitness_value, 2)
        children = crossover_mean(parents)
        mutated_children = weighted_mutation_switch(children, sigma)
        # mutated_children = diff_muatation(children)
        population = mutated_children
        # print("Gen ", i, " sigma: ", sigma)

    # spocitame fitness i pro posledni populaci
    fitness_value = list(map(fitness, population))
    max_fitness.append(max(fitness_value))
    best_individual = population[np.argmax(fitness_value)]

    return best_individual, population, max_fitness


best, population, max_fitness = evolution(population_size=500, individual_size=10, max_generations=1000)

print('best fitness: ', fitness(best))
print('best individual: ', best)


plt.plot(max_fitness)
plt.ylabel('Fitness')
plt.xlabel('Generace')
plt.show()
