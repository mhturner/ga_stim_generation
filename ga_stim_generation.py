from fiver.utilities import squirrel
import matplotlib.pyplot as plt
import numpy as np
import os
from stimvolver import stimulus_evolution as se


data_directory = '/Users/mhturner/Dropbox/ClandininLab/CurrentData/20190318/'


T_series_name = 'TSeries-20190318-004'


evolver = se.StimulusEvolver()



evolver.ResponseGenerator.series_name = T_series_name


evolver.ResponseGenerator.cycle_number = 14
evolver.ResponseGenerator.loadResponseData(data_directory = data_directory)





# %%
from stimvolver import stimulus_evolution as se




evolver = se.StimulusEvolver(stim_size = (8,12,15), pop_size = 40, num_persistent_parents = 10,
                 mutation_rate = 0.002,
                 stimulus_type = 'ternary_dense', response_type = 'model_DS', sparsity_penalty = 0.0)

evolver.initializePopulation()
evolver.evolve(500)

evolver.plotResults()

# %%

from stimvolver import stimulus_evolution as se
import matplotlib.pyplot as plt
import numpy as np

num_populations = 20

fig_handle = plt.figure(figsize=(9,7))

prior_solutions = []
for pop in range(num_populations):
    
    evolver = se.StimulusEvolver(stim_size = (8,12,6), pop_size = 40, num_persistent_parents = 10,
                 mutation_rate = 0.002,
                 stimulus_type = 'ternary_dense', response_type = 'model_MotionEnergy', 
                 similarity_penalty_weight = 0.5, prior_solutions = prior_solutions,
                 sparsity_penalty = 0.2)
    
    evolver.initializePopulation()
    evolver.evolve(500)
    
    prior_solutions.append(np.mean(evolver.current_generation, axis = 0))
    
    evolver.plotResults(fig_handle, plot_ind = pop)
    
# %%
import skimage.transform as skt
from skimage.filters import gabor_kernel
stim_size = (8,20,12)
t_dim = stim_size[2]
x_dim = stim_size[1]

filter_1 = skt.resize(np.real(gabor_kernel(0.1, theta = -60, sigma_x=2, sigma_y = 6, n_stds = 6, offset = 0)), (x_dim, t_dim))
filter_2 = skt.resize(np.real(gabor_kernel(0.1, theta = -60, sigma_x=2, sigma_y = 6, n_stds = 6, offset = np.pi/4)), (x_dim, t_dim))


fh = plt.figure()
ax1 = fh.add_subplot(211)
ax1.imshow(filter_1)

ax2 = fh.add_subplot(212)
ax2.imshow(filter_2)








