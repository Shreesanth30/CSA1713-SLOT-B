import random

def fitness(x):
    return x * x

population = [random.randint(0, 31) for i in range(6)]

for generation in range(10):

    population.sort(key=fitness, reverse=True)

    print("Generation", generation + 1,
          "Best Solution:", population[0],
          "Fitness:", fitness(population[0]))

    new_population = population[:2]

    while len(new_population) < 6:

        parent1 = random.choice(population[:3])
        parent2 = random.choice(population[:3])

        child = (parent1 + parent2) // 2

        if random.random() < 0.2:
            child = random.randint(0, 31)

        new_population.append(child)

    population = new_population

best = max(population, key=fitness)

print("\nBest Solution:", best)
print("Maximum Fitness:", fitness(best))
