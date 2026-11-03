
import random

# Job processing times
jobs = [4, 7, 3, 8, 5, 6, 2]

NUMBER_OF_MACHINES = 3

POPULATION_SIZE = 20
MUTATION_RATE = 0.1
GENERATIONS = 100


# Calculate machine workloads
def calculate_workloads(chromosome):

    workloads = [0] * NUMBER_OF_MACHINES

    for i in range(len(jobs)):

        machine = chromosome[i]

        workloads[machine] += jobs[i]

    return workloads


# Calculate makespan
def makespan(chromosome):

    workloads = calculate_workloads(chromosome)

    return max(workloads)


# Fitness function
def fitness(chromosome):

    return 1 / makespan(chromosome)


# Create initial population
def create_population():

    population = []

    for _ in range(POPULATION_SIZE):

        chromosome = [
            random.randint(0, NUMBER_OF_MACHINES - 1)
            for _ in range(len(jobs))
        ]

        population.append(chromosome)

    return population


# Tournament Selection
def selection(population):

    candidates = random.sample(population, 3)

    return max(candidates, key=fitness).copy()


# Single Point Crossover
def crossover(parent1, parent2):

    point = random.randint(1, len(jobs) - 1)

    child = parent1[:point] + parent2[point:]

    return child


# Mutation
def mutation(chromosome):

    chromosome = chromosome.copy()

    for i in range(len(chromosome)):

        if random.random() < MUTATION_RATE:

            chromosome[i] = random.randint(
                0, NUMBER_OF_MACHINES - 1
            )

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

workloads = calculate_workloads(best)

print("Job Assignments:")

for i in range(len(jobs)):

    print(
        "Job", i + 1,
        "Processing Time:", jobs[i],
        "Machine:", best[i] + 1
    )

print("\nMachine Workloads:")

for i in range(NUMBER_OF_MACHINES):

    print("Machine", i + 1, ":", workloads[i])

print("\nMinimum Makespan:", max(workloads))
