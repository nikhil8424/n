
import random
import math

# Cities and their coordinates
cities = {
    "Mumbai": (1, 2),
    "Delhi": (2, 9),
    "Pune": (2, 3),
    "Chennai": (5, 1),
    "Bengaluru": (3, 0),
    "Hyderabad": (4, 3),
    "Kolkata": (9, 7),
    "Jaipur": (2, 8)
}

city_list = list(cities.keys())

POPULATION_SIZE = 50
MUTATION_RATE = 0.15
GENERATIONS = 100


# Calculate distance between two cities
def distance(city1, city2):
    return math.dist(cities[city1], cities[city2])


# Calculate total route distance
def route_distance(route):
    total = 0

    for i in range(len(route)):
        current = route[i]
        next_city = route[(i + 1) % len(route)]

        total += distance(current, next_city)

    return total


# Create initial population
def create_population():
    population = []

    for _ in range(POPULATION_SIZE):
        route = city_list.copy()
        random.shuffle(route)
        population.append(route)

    return population


# Tournament Selection
def selection(population):
    candidates = random.sample(population, 3)

    return min(candidates, key=route_distance).copy()


# Ordered Crossover
def crossover(parent1, parent2):
    size = len(parent1)

    start, end = sorted(random.sample(range(size), 2))

    child = [None] * size
    child[start:end] = parent1[start:end]

    remaining = [
        city for city in parent2 if city not in child
    ]

    index = 0

    for i in range(size):
        if child[i] is None:
            child[i] = remaining[index]
            index += 1

    return child


# Swap Mutation
def mutation(route):
    route = route.copy()

    i, j = random.sample(range(len(route)), 2)

    route[i], route[j] = route[j], route[i]

    return route


# Main Genetic Algorithm
population = create_population()

best_route = None
best_distance = float("inf")

for generation in range(1, GENERATIONS + 1):

    population.sort(key=route_distance)

    if route_distance(population[0]) < best_distance:
        best_route = population[0].copy()
        best_distance = route_distance(best_route)

    new_population = [population[0].copy()]

    while len(new_population) < POPULATION_SIZE:

        parent1 = selection(population)
        parent2 = selection(population)

        child = crossover(parent1, parent2)

        if random.random() < MUTATION_RATE:
            child = mutation(child)

        new_population.append(child)

    population = new_population


# Display result
print("Best Route:")

print(" -> ".join(best_route + [best_route[0]]))

print("Total Distance:", round(best_distance, 3))
print("Generations:", GENERATIONS)
