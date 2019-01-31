import stimulus_evolution as se


evolver = se.StimulusEvolver()

evolver.x_dim = 10
evolver.y_dim = 8

evolver.mutation_rate = 0.005
evolver.fitness_function = 'model_DS'

evolver.evolve(500)
# %%
evolver.plotResults()