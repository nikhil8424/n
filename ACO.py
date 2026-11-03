
from random import randrange

vertices = ['A', 'B', 'C', 'D']

graph = {
    'A': ['B', 'C', 'D'],
    'B': ['A', 'C', 'D'],
    'C': ['A', 'B', 'D'],
    'D': ['A', 'B', 'C']
}

weights = {
    'AB': 10,
    'AC': 10,
    'AD': 30,
    'BC': 40,
    'CD': 20,
    'BD': 10
}

Pheromone = {}

# Decay rate
p = 0.1


def initializePheromone():
    # Initialize random pheromone trails
    global Pheromone

    for key, value in graph.items():
        for i in value:
            if key + i not in Pheromone and i + key not in Pheromone:
                Pheromone[key + i] = randrange(1, 51)


def AOC(Pheromone):
    global weights, graph

    # 4 ants
    initialVerticesCovered = []
    paths = []

    for i in range(4):

        print("".center(80, "*"))

        discovered = []
        flag = True

        while flag:
            initial = randrange(0, 4)

            if initial not in initialVerticesCovered:
                flag = False

        initialVerticesCovered.append(initial)
        initial = vertices[initial]

        start = initial
        discovered.append(initial)

        while True:

            probabilities = []
            sumHT = 0

            possVertices = graph[initial].copy()

            # Calculate sum of pheromone and heuristic values
            for i in possVertices:

                if initial + i in weights:
                    H = round(1 / weights[initial + i], 3)
                elif i + initial in weights:
                    H = round(1 / weights[i + initial], 3)

                if initial + i in Pheromone:
                    T = Pheromone[initial + i]
                elif i + initial in Pheromone:
                    T = Pheromone[i + initial]

                sumHT += round(H * T, 3)

            # Calculate transition probabilities
            for i in possVertices:

                if initial + i in weights:
                    H = round(1 / weights[initial + i], 3)
                elif i + initial in weights:
                    H = round(1 / weights[i + initial], 3)

                if initial + i in Pheromone:
                    T = Pheromone[initial + i]
                elif i + initial in Pheromone:
                    T = Pheromone[i + initial]

                probabilities.append(round((H * T) / sumHT, 3))

            flag = True

            while flag:

                val = max(probabilities)
                val = probabilities.index(val)

                nextVertex = possVertices[val]

                if nextVertex not in discovered:

                    discovered.append(nextVertex)

                    # Further path construction
                    initial = nextVertex
                    flag = False

                else:
                    # Remove already visited vertex
                    probabilities.pop(val)
                    possVertices.pop(val)

                    if len(probabilities) == 0:
                        discovered.append(start)
                        flag = False

            # Check whether ant has completed its tour
            if discovered[0] == discovered[-1] and len(discovered) != 1:

                print("discovered", discovered)
                paths.append(discovered)
                break

    # Pheromone decay
    print("Before Decay:", Pheromone)

    for key, value in Pheromone.items():
        Pheromone[key] = round((1 - p) * Pheromone[key], 3)

    print("After Decay:", Pheromone)

    # Add new pheromone based on paths
    for key, value in Pheromone.items():

        sumVal = 0

        for i in paths:

            string = "".join(i)

            if key in string:
                sumVal += round(1 / (len(string) - 1), 3)

        Pheromone[key] += sumVal

    print("After adding Pheromone:", Pheromone)

    return Pheromone


# Main program
initializePheromone()

print(Pheromone)

for i in range(50):
    Pheromone = AOC(Pheromone)

print(Pheromone)
