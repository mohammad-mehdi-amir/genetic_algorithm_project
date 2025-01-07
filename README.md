This project focuses on finding the most similar note sequence to a given target sequence using a genetic algorithm.
The goal is to optimize the note sequences so they resemble the target as closely as possible.

To achieve this, I implemented the PMX (Partially Mapped Crossover) algorithm for the crossover operation, which ensures efficient mixing of genetic material between parent sequences.
In addition to crossover, I utilized multiple mutation algorithms to introduce diversity and prevent the algorithm from getting stuck in local optima.
The adaptive mutation rate dynamically adjusts throughout the process to enhance efficiency and adapt to the current state of the population.

Furthermore, I developed a visualizer function to provide a clear and interactive way to observe the generated sequences and evaluate their similarity to the target.
This visualization helps in understanding the progression of the genetic algorithm and the evolution of the population over time.

At the final stage, the best-performing individual sequences are converted into a MIDI file. This allows for a tangible representation of the progress and lets users listen to how closely the generated sequences match the target.
The combination of these methods ensures a robust and effective approach to solving the problem.

This project demonstrates the potential of genetic algorithms in creative and computational fields such as music composition and sequence optimization.
