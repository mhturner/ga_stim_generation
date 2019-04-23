# -*- coding: utf-8 -*-
import numpy as np
from stimvolver import stimulus_evolution_modelRF as sem
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

mutation_rates = np.linspace(0.001, 0.1, 20)
#mutation_rates = [0.0005, 0.001, 0.0025, 0.005, 0.0075, 0.01, 0.025, 0.05]
pop_sizes = [5, 10, 20, 40, 80, 120]

num_generations = 300


final_fitness = np.ndarray(shape = (len(mutation_rates), len(pop_sizes)))
for m_ind, mutation_rate in enumerate(mutation_rates):
    for p_ind, pop_size in enumerate(pop_sizes):
        
        evolver = sem.StimulusEvolver(stim_size = (8,12,15), pop_size = pop_size, frac_persistent_parents = 0.1,
                         mutation_rate = mutation_rate,
                         stimulus_type = 'ternary_dense', response_type = 'model_DS', sparsity_penalty = 0.0)
        
        evolver.initializePopulation()
        evolver.evolve(num_generations)
        
        final_fitness[m_ind, p_ind] = np.mean(evolver.fitness[-1])
        

# 
fh = plt.figure()
ax = fh.gca(projection='3d')
X, Y = np.meshgrid(mutation_rates, pop_sizes)
ax.plot_surface(X, Y, final_fitness.T)
ax.set_xlabel('Mutation rate')
ax.set_zlabel('resp')
ax.set_ylabel('Pop size')
ax.set_title('After ' + str(num_generations) + ' gen')

# %% inc. mutation rate and dec. pop size

import numpy as np
from stimvolver import stimulus_evolution_modelRF as sem
import matplotlib.pyplot as plt

num_generations = 300
pop_size = np.linspace(100, 10, num_generations)
mutation_rate = np.linspace(0.001, 0.01, num_generations)

pop_fitness = []
fh = plt.figure()
ax = fh.add_subplot(111)

evolver = sem.StimulusEvolver(stim_size = (8,12,15), pop_size = 40, frac_persistent_parents = 0.1,
                 mutation_rate = 0.05,
                 stimulus_type = 'ternary_dense', response_type = 'model_DS', sparsity_penalty = 0.0)
evolver.initializePopulation()

for gen in range(num_generations):
    evolver.evolve(1, mutation_rate = mutation_rate[gen], pop_size = int(pop_size[gen]))
    
    pop_fitness.append(np.mean(evolver.fitness[-1]))

evolver.plotResults()

ax.plot(pop_fitness)

# %% push away from existing solutions
import numpy as np
from stimvolver import stimulus_evolution_modelRF as sem
import matplotlib.pyplot as plt

num_runs = 10
prior_solutions = []

fh = plt.figure()

for run in range(num_runs):
    evolver = sem.StimulusEvolver(stim_size = (8,12,15), pop_size = 40, frac_persistent_parents = 0.1,
                     mutation_rate = 0.05,
                     stimulus_type = 'ternary_dense', response_type = 'model_DS', sparsity_penalty = 0.0,
                     similarity_penalty_weight=1e2, prior_solutions = prior_solutions)
    
    evolver.initializePopulation()
    evolver.evolve(300)
    evolver.plotResults(fig_handle = fh, plot_ind = run)

# %% search over mutation rates
    
import numpy as np
from stimvolver import stimulus_evolution_modelRF as sem
import matplotlib.pyplot as plt

mutation_rates = [0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5]


fh = plt.figure()
ax = fh.add_subplot(111)
for mutation_rate in mutation_rates:

    evolver = sem.StimulusEvolver(stim_size = (8,12,15), pop_size = 40, frac_persistent_parents = 0.1,
                     mutation_rate = mutation_rate,
                     stimulus_type = 'ternary_dense', response_type = 'model_DS', sparsity_penalty = 0.0)
    
    evolver.initializePopulation()
    evolver.evolve(300)
        
    ax.plot(np.mean(np.vstack(evolver.fitness), axis = 1))

    
ax.legend(mutation_rates)
    
    

