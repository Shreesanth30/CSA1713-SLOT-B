import random

# Objective function
def fitness(x):
    return x * x

# Generate initial population
population = [random.randint(0, 31) for i in range(6)]

# Genetic Algorithm
for generation in range(10):

    # Select the best individuals
    population.sort(key=fitness, reverse=True)
    new_population = population[:2]

    # Crossover and Mutation
    while len(new_population) < 6:
        p1, p2 = random.sample(population[:4], 2)

        child = (p1 + p2) // 2

        if random.random() < 0.2:
            child = random.randint(0, 31)

        new_population.append(child)

    population = new_population

# Display result
best = max(population, key=fitness)

print("Best Solution:", best)
print("Maximum Fitness:", fitness(best))
