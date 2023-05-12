import random
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(1000)

# Функція Букіна №6
def bukin6(x):
    term1 = 100 * np.sqrt(np.abs(x[1] - 0.01 * x[0]**2))
    term2 = 0.01 * np.abs(x[0] + 10)
    return term1 + term2

# Функція Шаффера №2
def schaffer2(x):
    numerator = np.sin(np.sqrt(x[0]**2 + x[1]**2))**2 - 0.5
    denominator = (1 + 0.001 * (x[0]**2 + x[1]**2))**2
    return 0.5 + numerator / denominator

# Функція Шаффера №4
def schaffer4(x):
    numerator = np.sin(np.sqrt(x[0]**2 + x[1]**2))
    denominator = 1 + 0.001 * (x[0]**2 + x[1]**2)
    return 0.5 + (numerator**2 - 0.5) / (denominator**2)

# Ініціалізація початкової популяції
def initialize_population(pop_size, chromosome_length):
    return np.random.uniform(low=-100, high=100, size=(pop_size, chromosome_length))

# Оцінка фітнесу популяції
def fitness(population, func):
    return np.array([func(chromosome) for chromosome in population])

# Вибір батьків
def select_parents(population, fitness_scores, num_parents):
    # Розрахунок ймовірності вибору кожного хромосоми
    probs = fitness_scores / np.sum(fitness_scores)
    # Вибір батьків
    parents_indices = np.random.choice(len(population), size=num_parents, replace=False, p=probs)
    parents = population[parents_indices]
    return parents

# Схрещування батьків
def crossover(parents, num_offspring, crossover_rate):
    offspring = np.empty((num_offspring, parents.shape[1]))
    for i in range(num_offspring):
        parent1_idx = i % parents.shape[0]
        parent2_idx = (i + 1) % parents.shape[0]
        if np.random.rand() < crossover_rate:
            offspring[i, 0] = parents[parent1_idx, 0]
            offspring[i, 1] = parents[parent2_idx, 1]
        else:
            offspring[i, :] = parents[parent1_idx, :]
    return offspring

# Мутація нащадків
def mutate(offspring, mutation_rate):
    for i in range(offspring.shape[0]):
        if np.random.rand() < mutation_rate:
            offspring[i, :] += np.random.uniform(low=-1, high=1, size=offspring.shape[1])
    return offspring

# Генетичний алгоритм для функції Букіна №6
def genetic_algorithm_bukin6(pop_size, chromosome_length, num_generations, num_parents, num_offspring, crossover_rate, mutation_rate):
    # Ініціалізація початкової популяції
    population = initialize_population(pop_size, chromosome_length)
    best_fitness_scores = np.empty(num_generations)
    best_chromosomes = np.empty((num_generations, chromosome_length))
    # Запуск генетичного алгоритму
    for i in range(num_generations):
        # Оцінка фітнесу популяції
        fitness_scores = fitness(population, bukin6)
        # Збереження найкращої хромосоми поточної генерації
        best_chromosome = population[np.argmin(fitness_scores)]
        best_chromosomes[i] = best_chromosome
        best_fitness_scores[i] = bukin6(best_chromosome)
        # Вибір батьків за допомогою рулеткового відбору
        parents = select_parents(population, fitness_scores, num_parents)
        # Схрещування батьків
        offspring_crossover = crossover(parents, num_offspring, crossover_rate)
        # Мутація нащадків
        offspring_mutation = mutate(offspring_crossover, mutation_rate)
        # Створення нової популяції
        population[:num_parents, :] = parents
        population[num_parents:, :] = offspring_mutation
        if i%50==0:
            print(f"Itenation: {i}: best: {best_fitness_scores[i]} ")
    return best_chromosomes, best_fitness_scores

# Генетичний алгоритм для функції Шаффера №2
def genetic_algorithm_schaffer2(pop_size, chromosome_length, num_generations, num_parents, num_offspring, crossover_rate, mutation_rate):
    # Ініціалізація початкової популяції
    population = initialize_population(pop_size, chromosome_length)
    best_fitness_scores = np.empty(num_generations)
    best_chromosomes = np.empty((num_generations, chromosome_length))
    # Запуск генетичного алгоритму
    for i in range(num_generations):
        # Оцінка фітнесу популяції
        fitness_scores = fitness(population, schaffer2)
        # Збереження найкращої хромосоми поточної генерації
        best_chromosome = population[np.argmin(fitness_scores)]
        best_chromosomes[i] = best_chromosome
        best_fitness_scores[i] = schaffer2(best_chromosome)
        # Вибір батьків за допомогою рулеткового відбору
        parents = select_parents(population, fitness_scores, num_parents)
        # Схрещування батьків
        offspring_crossover = crossover(parents, num_offspring, crossover_rate)
        # Мутація нащадків
        offspring_mutation = mutate(offspring_crossover, mutation_rate)
        # Створення нової популяції
        population[:num_parents, :] = parents
        population[num_parents:, :] = offspring_mutation
        if i%50==0:
            print(f"Itenation: {i}: best: {best_fitness_scores[i]} ")
    return best_chromosomes, best_fitness_scores

def genetic_algorithm_schaffer4(pop_size, chromosome_length, num_generations, num_parents, num_offspring, crossover_rate, mutation_rate):
    # Ініціалізація початкової популяції
    population = initialize_population(pop_size, chromosome_length)
    best_fitness_scores = np.empty(num_generations)
    best_chromosomes = np.empty((num_generations, chromosome_length))
    # Запуск генетичного алгоритму
    for i in range(num_generations):
        # Оцінка фітнесу популяції
        fitness_scores = fitness(population, schaffer4)
        # Збереження найкращої хромосоми поточної генерації
        best_chromosome = population[np.argmin(fitness_scores)]
        best_chromosomes[i] = best_chromosome
        best_fitness_scores[i] = schaffer4(best_chromosome)
        # Вибір батьків за допомогою рулеткового відбору
        parents = select_parents(population, fitness_scores, num_parents)
        # Схрещування батьків
        offspring_crossover = crossover(parents, num_offspring, crossover_rate)
        # Мутація нащадків
        offspring_mutation = mutate(offspring_crossover, mutation_rate)
        # Створення нової популяції
        population[:num_parents, :] = parents
        population[num_parents:, :] = offspring_mutation
        if i%50==0:
            print(f"Itenation: {i}: best: {best_fitness_scores[i]} ")
    return best_chromosomes, best_fitness_scores
# Генетичний алгоритм для функції Букіна №6
print("Bukin 6\n")
best_chromosomes_bukin6, best_fitness_scores_bukin6 = genetic_algorithm_bukin6(pop_size=100, chromosome_length=2, num_generations=1000, num_parents=20, num_offspring=80, crossover_rate=0.8, mutation_rate=0.1)
# Генетичний алгоритм для функції Шаффера №2
print("\nSchaffer2\n")
best_chromosomes_schaffer2, best_fitness_scores_schaffer2 = genetic_algorithm_schaffer2(pop_size=100, chromosome_length=2, num_generations=1000, num_parents=20, num_offspring=80, crossover_rate=0.8, mutation_rate=0.1)
# Генетичний алгоритм для функції Шаффера №4
print("\nSchafer4\n")
best_chromosomes_schaffer4, best_fitness_scores_schaffer4 = genetic_algorithm_schaffer4(pop_size=100, chromosome_length=2, num_generations=1000, num_parents=20, num_offspring=80, crossover_rate=0.8, mutation_rate=0.)

fig, ax= plt.subplots(1, 3, figsize=(10, 5))
# Графіки залежності значень функції від номеру ітерації для кожної функції
ax[0].plot(best_fitness_scores_bukin6)
ax[0].set_title("Bukin6")


ax[1].plot(best_fitness_scores_schaffer2)
ax[1].set_title("Schaffer2")


ax[2].plot(best_fitness_scores_schaffer4)
ax[2].set_title("Schaffer4")

plt.show()