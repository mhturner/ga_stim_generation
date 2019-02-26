from stimvolver import stimulus_evolution as se




evolver = se.StimulusEvolver(stim_size = (8,12,1), pop_size = 40, num_persistent_parents = 10,
                 mutation_rate = 0.002,
                 stimulus_type = 'ternary_dense', response_type = 'model_RGC', sparsity_penalty = 0.0)

evolver.initializePopulation()
evolver.evolve(20)

evolver.plotResults()