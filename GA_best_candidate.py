
import random

# Candidate information:
# (Name, Skills Score, Experience Score,
#  Interview Score, Salary Score)

candidates = [
    ("Amit", 90, 85, 80, 90),
    ("Rahul", 85, 90, 85, 80),
    ("Priya", 95, 80, 95, 85),
    ("Neha", 80, 75, 85, 95),
    ("Karan", 75, 90, 70, 85),
    ("Sneha", 92, 88, 90, 80),
    ("Rohit", 70, 80, 75, 90),
    ("Anjali", 88, 85, 92, 88)
]

NUMBER_SELECTED = 3

POPULATION_SIZE = 20
MUTATION_RATE = 0.1
GENERATIONS = 100


# Calculate candidate suitability score
def candidate_score(candidate):

    skills = candidate[1]
    experience = candidate[2]
    interview = candidate[3]
    salary = candidate[4]

    score = (
        0.4 * skills
        + 0.2 * experience
        + 0.3 * interview
        + 0.1 * salary
    )

    return score


# Calculate chromosome fitness
def fitness(chromosome):

    if sum(chromosome) != NUMBER_SELECTED:
        return 0

    total_score = 0

    for i in range(len(candidates)):

        if chromosome[i] == 1:
            total_score += candidate_score(candidates[i])

    return total_score


# Create initial population
def create_population():

    population = []

    for _ in range(POPULATION_SIZE):

        chromosome = [0] * len(candidates)

        selected = random.sample(
            range(len(candidates)),
            NUMBER_SELECTED
        )

        for i in selected:
            chromosome[i] = 1

        population.append(chromosome)

    return population


# Tournament Selection
def selection(population):

    selected = random.sample(population, 3)

    return max(selected, key=fitness).copy()


# Repair chromosome to select exactly 3 candidates
def repair(chromosome):

    selected = [
        i for i in range(len(chromosome))
        if chromosome[i] == 1
    ]

    unselected = [
        i for i in range(len(chromosome))
        if chromosome[i] == 0
    ]

    while len(selected) > NUMBER_SELECTED:

        i = random.choice(selected)
        chromosome[i] = 0
        selected.remove(i)

    while len(selected) < NUMBER_SELECTED:

        i = random.choice(unselected)
        chromosome[i] = 1
        selected.append(i)
        unselected.remove(i)

    return chromosome


# Single Point Crossover
def crossover(parent1, parent2):

    point = random.randint(1, len(candidates) - 1)

    child = parent1[:point] + parent2[point:]

    return repair(child)


# Mutation
def mutation(chromosome):

    chromosome = chromosome.copy()

    if random.random() < MUTATION_RATE:

        selected = [
            i for i in range(len(chromosome))
            if chromosome[i] == 1
        ]

        unselected = [
            i for i in range(len(chromosome))
            if chromosome[i] == 0
        ]

        remove_index = random.choice(selected)
        add_index = random.choice(unselected)

        chromosome[remove_index] = 0
        chromosome[add_index] = 1

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


# Find best candidates
best = max(population, key=fitness)

print("BEST CANDIDATES FOR THE JOB")
print("--------------------------------")

for i in range(len(candidates)):

    if best[i] == 1:

        print("\nName:", candidates[i][0])
        print("Skills:", candidates[i][1])
        print("Experience:", candidates[i][2])
        print("Interview:", candidates[i][3])
        print("Salary Score:", candidates[i][4])
        print("Overall Score:",
              round(candidate_score(candidates[i]), 2))

print("\nTotal Selected:", sum(best))
print("Total Fitness:", round(fitness(best), 2))
