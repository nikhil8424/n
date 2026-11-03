
import random

# Items: (weight, value)
items = [
    (2, 10),
    (3, 15),
    (5, 25),
    (7, 35),
    (1, 8),
    (4, 20)
]

CAPACITY = 10

POPULATION_SIZE = 20
MUTATION_RATE = 0.1
GENERATIONS = 100


# Calculate fitness
def fitness(chromosome):

    total_weight = 0
    total_value = 0

    for i in range(len(items)):

        if chromosome[i] == 1:
            total_weight += items[i][0]
            total_value += items[i][1]

    if total_weight > CAPACITY:
        return 0

    return total_value


# Create population
def create_population():

    population = []

    for _ in range(POPULATION_SIZE):

        chromosome = [
            random.randint(0, 1)
            for _ in range(len(items))
        ]

        population.append(chromosome)

    return population


# Tournament Selection
def selection(population):

    candidates = random.sample(population, 3)

    return max(candidates, key=fitness).copy()


# Single Point Crossover
def crossover(parent1, parent2):

    point = random.randint(1, len(items) - 1)

    child = parent1[:point] + parent2[point:]

    return child


# Mutation
def mutation(chromosome):

    chromosome = chromosome.copy()

    for i in range(len(chromosome)):

        if random.random() < MUTATION_RATE:
            chromosome[i] = 1 - chromosome[i]

    return chromosome


# Main Genetic Algorithm
population = create_population()

for generation in range(GENERATIONS):

    population.sort(key=fitness, reverse=True)

    new_population = [population[0].copy()]

    while len(new_population) < POPULATION_SIZE:

        parent1 = selection(population)
        parent2 = selection(population)

        child = crossover(parent1, parent2)
        child = mutation(child)

        new_population.append(child)

    population = new_population


# Best solution
best = max(population, key=fitness)

total_weight = 0
total_value = 0

print("Selected Items:")

for i in range(len(items)):

    if best[i] == 1:

        print("Item", i + 1,
              "Weight:", items[i][0],
              "Value:", items[i][1])

        total_weight += items[i][0]
        total_value += items[i][1]


print("\nTotal Weight:", total_weight)
print("Total Value:", total_value)
print("Bag Capacity:", CAPACITY)
