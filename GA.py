
import random
import string

# Configuration
TARGET = "INDIA"
POPULATION_SIZE = 20
MUTATION_RATE = 0.1
TOURNAMENT_SIZE = 3
MAX_GENERATIONS = 500


# Create initial population
def create_population():
    population = []

    for _ in range(POPULATION_SIZE):
        chromosome = ''.join(
            random.choice(string.ascii_uppercase)
            for _ in range(len(TARGET))
        )
        population.append(chromosome)

    return population


# Calculate fitness
def fitness(individual):
    score = 0

    for i in range(len(TARGET)):
        if individual[i] == TARGET[i]:
            score += 1

    return score


# Tournament Selection
def tournament_selection(population):
    candidates = random.sample(
        population, TOURNAMENT_SIZE
    )

    winner = max(candidates, key=fitness)

    return winner


# Rank Selection
def rank_selection(population):
    sorted_population = sorted(
        population, key=fitness
    )

    weights = list(range(1, len(population) + 1))

    parent1, parent2 = random.choices(
        sorted_population,
        weights=weights,
        k=2
    )

    return parent1, parent2


# Single Point Crossover
def crossover(parent1, parent2):
    point = random.randint(1, len(TARGET) - 1)

    child = parent1[:point] + parent2[point:]

    return child


# Mutation
def mutate(child):
    child = list(child)

    for i in range(len(child)):
        if random.random() < MUTATION_RATE:
            child[i] = random.choice(string.ascii_uppercase)

    return ''.join(child)


# Genetic Algorithm
def run_ga(selection_type):

    population = create_population()

    for generation in range(1, MAX_GENERATIONS + 1):

        population.sort(key=fitness, reverse=True)

        best = population[0]

        if fitness(best) == len(TARGET):
            print(selection_type, "Selection")
            print("Target:", TARGET)
            print("Best Individual:", best)
            print("Fitness:", fitness(best))
            print("Generations:", generation)
            print("--------------------------")
            return generation

        new_population = [best]

        while len(new_population) < POPULATION_SIZE:

            if selection_type == "TOURNAMENT":
                parent1 = tournament_selection(population)
                parent2 = tournament_selection(population)

            elif selection_type == "RANK":
                parent1, parent2 = rank_selection(population)

            child = crossover(parent1, parent2)
            child = mutate(child)

            new_population.append(child)

        population = new_population

    print(selection_type, "Selection")
    print("Best Individual:", max(population, key=fitness))
    print("Maximum generations reached")

    return MAX_GENERATIONS


# Main program
tournament_gen = run_ga("TOURNAMENT")
rank_gen = run_ga("RANK")

# Compare both methods
print("\nSELECTION COMPARISON")

print("Tournament Generations:", tournament_gen)
print("Rank Generations:", rank_gen)

if tournament_gen < rank_gen:
    print("Tournament Selection performed better.")
elif rank_gen < tournament_gen:
    print("Rank Selection performed better.")
else:
    print("Both performed equally.")
