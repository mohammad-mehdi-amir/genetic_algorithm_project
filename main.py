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
