
import random

# Food location is unknown to birds
food = random.randint(1, 20)

# Birds = particles
birds = [random.randint(1, 20) for _ in range(4)]

v = [0] * len(birds)

qbest = birds[0]


# Fitness: distance from food
def fitness(x):
    return abs(x - food)


# Personal best
pbest = birds.copy()
pbest_fit = [fitness(x) for x in birds]

w, c1, c2 = 0.5, 1.5, 1.5

for _ in range(100):

    # Best bird in the flock
    qbest = pbest[pbest_fit.index(min(pbest_fit))]

    for i in range(len(birds)):

        r1 = random.random()
        r2 = random.random()

        # PSO velocity formula
        v[i] = (
            w * v[i]
            + c1 * r1 * (pbest[i] - birds[i])
            + c2 * r2 * (qbest - birds[i])
        )

        birds[i] += v[i]

        # Update personal best
        if fitness(birds[i]) < pbest_fit[i]:
            pbest[i] = birds[i]
            pbest_fit[i] = fitness(birds[i])

        # Bird reaches food
        if abs(birds[i] - food) <= 0.05:
            birds[i] = food
            pbest[i] = food
            pbest_fit[i] = 0
            break

    # Check whether food was found
    found = any(x == food for x in birds)

    if found:
        break

# Best bird
best = pbest_fit.index(min(pbest_fit))

print("Food Position       :", food)
print("Best Bird           :", best + 1)
print("Best Swarm Position :", round(pbest[best], 3))
print("Best Fitness Value  :", round(pbest_fit[best], 3))
print("Minimum Distance    :", round(abs(pbest[best] - food), 3))

print("\nBird Fitness Values:")

for i in range(len(birds)):
    print(
        "Bird", i + 1,
        "Position:", round(birds[i], 3),
        "Fitness:", round(fitness(birds[i]), 3)
    )

if found:
    print("\nFood Found! Success")
else:
    print("\nFood Not Found")
