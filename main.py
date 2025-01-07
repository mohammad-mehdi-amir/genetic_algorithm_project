import random
from plot import plot_generation_fitness
from mido_intro import save_midi, clear_melody_folder
global MUTATION_RATE
global POPULATION
global GENES_LENGTH
global POPULATION_SIZE
global GENERATIONS


def SetParametrs():
    goal_seq = [random.randint(50, 70) for _ in range(100)]
    GENES_LENGTH = len(goal_seq)

    clear_melody_folder()
    save_midi(goal_seq, filename=['0target', GENES_LENGTH])

    if GENES_LENGTH <= 10:
        POPULATION_SIZE = 150

    elif GENES_LENGTH <= 15:
        POPULATION_SIZE = 450

    elif GENES_LENGTH <= 20:
        POPULATION_SIZE = 500

    elif GENES_LENGTH <= 25:
        POPULATION_SIZE = 700

    elif GENES_LENGTH <= 50:
        POPULATION_SIZE = 2000

    elif GENES_LENGTH > 50:
        POPULATION_SIZE = 4000

    GENERATIONS = 300

    return POPULATION_SIZE, GENES_LENGTH, GENERATIONS, goal_seq


def CreatePopulation(q_sequence, population_size):
    population = []
    for _ in range(population_size):
        random_sequence = q_sequence[:]
        random.shuffle(random_sequence)
        population.append(random_sequence)
    return population


plot_data = []
POPULATION_SIZE, GENES_LENGTH, GENERATIONS, GOAL_SEQ = SetParametrs()
POPULATION = CreatePopulation(GOAL_SEQ, POPULATION_SIZE)


def FitnessFunction(target_sequence, generated_sequence):
    exact_matches = sum(1 for t, g in zip(
        target_sequence, generated_sequence) if t == g)
    fitness = exact_matches
    return (fitness)*100 // GENES_LENGTH




def adaptive_mutation_rate(generation, base_rate, max_rate, fitness_history, threshold=5):

    global MUTATION_RATE
    if all(fitt for fitt in fitness_history[-5:]):
        MUTATION_RATE = max_rate

    else:

        MUTATION_RATE = base_rate * ((1-generation/GENERATIONS)**2)+0.1

    if MUTATION_RATE < 0:
        MUTATION_RATE = base_rate


def FixChromosome(chromosome, target_sequence):
    unique_genes = list(set(chromosome))
    missing_genes = [
        gene for gene in target_sequence if gene not in unique_genes]
    fixed_chromosome = unique_genes + missing_genes
    random.shuffle(fixed_chromosome)
    return fixed_chromosome[:len(target_sequence)]


def TournamentSelection(population, target_sequence):
    if GENES_LENGTH <= 10:
        k = 4
    elif GENES_LENGTH <= 15:
        k = 8
    elif GENES_LENGTH <= 20:
        k = 10
    elif GENES_LENGTH <= 25:
        k = 12
    elif GENES_LENGTH <= 50:
        k = 25
    else:
        k = 20

    selected = random.sample(population, k)
    fitness_values = [FitnessFunction(
        target_sequence, individual) for individual in selected]

    best_parent = selected[fitness_values.index(max(fitness_values))]

    return best_parent

# def rank_selection(population, fitness_values):
#     sorted_population = [x for _, x in sorted(zip(fitness_values, population), key=lambda pair: pair[0])]
#     ranks = range(1, len(sorted_population) + 1)
#     probabilities = [r / sum(ranks) for r in ranks]
#     selected = random.choices(sorted_population, weights=probabilities, k=2)
#     return selected[0],selected[1]

# def roulette_wheel_selection(population, fitness_values):
#     # محاسبه مجموع کل تناسب‌ها
#     total_fitness = sum(fitness_values)

#     # انتخاب یک عدد تصادفی از ۰ تا مجموع کل تناسب‌ها
#     pick = random.uniform(0, total_fitness)

#     current = 0
#     # انتخاب فردی که به احتمال انتخاب می‌شود
#     for i in range(len(population)):
#         current += fitness_values[i]
#         if current >= pick:
#             return population[i]


def fix_overlap_with_repeats(child, parent, point1, point2, max_repeats=None):
   
    if max_repeats is None:
        return child

    gene_counts = {}
    for gene in child:
        if gene in gene_counts:
            gene_counts[gene] += 1
        else:
            gene_counts[gene] = 1

    for i in range(len(child)):
        if i < point1 or i >= point2: 
            gene = child[i]
            if gene_counts[gene] > max_repeats: 
                available_genes = [
                    g for g in parent if gene_counts.get(g, 0) < max_repeats]
                if available_genes:
                    replacement = available_genes.pop()
                    child[i] = replacement
                    gene_counts[gene] -= 1
                    gene_counts[replacement] = gene_counts.get(
                        replacement, 0) + 1

    return child


def PMXCrossover(parent1, parent2, max_repeats=None):

    point1, point2 = sorted(random.sample(
        range(len(parent1)), 2))  # انتخاب بازه تصادفی
    child1 = parent1[:]
    child2 = parent2[:]
    child1[point1:point2] = parent2[point1:point2]
    child2[point1:point2] = parent1[point1:point2]

    child1 = fix_overlap_with_repeats(
        child1, parent1, point1, point2, max_repeats)
    child2 = fix_overlap_with_repeats(
        child2, parent2, point1, point2, max_repeats)

    return child1, child2


def CrossOverOnePoint(parent1, parent2):
    point = random.randint(0, len(parent1) - 1)
    child1 = parent1[:point] + parent2[point:]
    child2 = parent2[:point] + parent1[point:]
    return FixChromosome(child1, GOAL_SEQ), FixChromosome(child2, GOAL_SEQ)


def CrossOverTwoPoint(parent1, parent2):
    point1 = random.randint(0, len(parent1) - 1)
    point2 = random.randint(point1, len(parent1) - 1)
    child1 = parent1[:point1] + parent2[point1:point2] + parent1[point2:]
    child2 = parent2[:point1] + parent1[point1:point2] + parent2[point2:]
    return FixChromosome(child1, GOAL_SEQ), FixChromosome(child2, GOAL_SEQ)


def SwapMutate(individual):
    index = random.randint(0, len(individual) - 1)
    new_index = random.randint(0, len(individual) - 1)
    individual[index], individual[new_index] = individual[new_index], individual[index]
    return individual


def ScrambleMutate(individual):
    point1, point2 = sorted(random.sample(range(len(individual)), 2))
    individual[point1:point2] = random.sample(
        individual[point1:point2], len(individual[point1:point2]))
    return individual


def inversion_mutation(individual):
  
    mutant = individual[:]
    start, end = sorted(random.sample(range(len(mutant)), 2))
    mutant[start:end] = mutant[start:end][::-1]
    return mutant


def insert_mutation(individual):
  
    mutant = individual[:]
    idx1, idx2 = random.sample(range(len(mutant)), 2)
    gene = mutant.pop(idx1)  
    mutant.insert(idx2, gene) 
    return mutant


def displacement_mutation(individual):
   
    mutant = individual[:]
    start, end = sorted(random.sample(range(len(mutant)), 2))
    subset = mutant[start:end]
    del mutant[start:end] 
    insert_position = random.randint(0, len(mutant))  
    mutant[insert_position:insert_position] = subset  
    return mutant


def partial_shuffle_mutation(individual, num_swaps=3):
   
    mutant = individual[:]

    while True:
        start, end = sorted(random.sample(
            range(len(mutant)), 2))  
        if end - start >= 2:  
            break

    for _ in range(num_swaps):
        idx1, idx2 = random.sample(range(start, end), 2)
        mutant[idx1], mutant[idx2] = mutant[idx2], mutant[idx1]  
    return mutant



def mutation_flip_bit(individual, mutation_rate=0.1):

    for i in range(len(individual)):
        if random.random() < mutation_rate:
            individual[i] = 1 - individual[i] 
    return individual


def enforce_diversity(population, target_sequence, randomization_fraction=0.3):
   
    num_to_randomize = int(len(population) * randomization_fraction)
    random_individuals = CreatePopulation(
        target_sequence, num_to_randomize)  
    population = population[:-num_to_randomize] + \
        random_individuals 
    return population
