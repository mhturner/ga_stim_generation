import stimulus_evolution as se




evolver = se.StimulusEvolver(stim_size = (8,10,20), pop_size = 40, num_persistent_parents = 10,
                 mutation_rate = 0.002, noise_level = 0.0, threshold = None,
                 sparsity_penalty = 0.0, similarity_penalty = 0.0,
                 stimulus_type = 'single_spot', response_type = 'model_V1')


evolver.evolve(20)

evolver.plotResults()