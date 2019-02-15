import stimulus_evolution as se




evolver = se.StimulusEvolver(stim_size = (8,10,20), pop_size = 40, num_persistent_parents = 10,
                 mutation_rate = 0.002, noise_level = 0.0, threshold = None,
                 sparsity_penalty = 0.0, similarity_penalty = 0.0,
                 stimulus_type = 'ternary_dense', response_type = 'model_DS')


evolver.evolve(500)

evolver.plotResults()