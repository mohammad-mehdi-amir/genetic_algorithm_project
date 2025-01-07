import matplotlib.pyplot as plt

def plot_generation_fitness(generation_data,best_fit,max_fitness, filename="generation_fitness.png"):

    generations = [data[0] for data in generation_data]
    fitness_values = [data[1] for data in generation_data]

 
    plt.figure(figsize=(10, 6))
    plt.plot(generations, fitness_values, marker="o", linestyle="--", color="b", label="PMX and tornoment selection")
    plt.title(f"Best Fitness : {round(best_fit[1])} / {max_fitness}")
    plt.xlabel("Generation")
    plt.ylabel("Best Fitness")
    plt.grid(True)

    plt.legend()
    plt.savefig(filename)  
    plt.show()

    print(f"Chart saved as {filename}")