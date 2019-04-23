# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 13:49:04 2019

@author: mhturner
"""

from visanalysis.utilities import squirrel
from visanalysis import plot_tools
import matplotlib.pyplot as plt
import numpy as np


data_directory = '/Users/mhturner/Dropbox/ClandininLab/CurrentData/20190318/'

evolver = squirrel.get('test_5',data_directory = data_directory)
dt = np.mean(np.diff(evolver.ResponseGenerator.time_step)) / 1e3 #msec -> sec
fh = plt.figure(1)

pull_cycles = (1, 2, 4, 8, 12, 16, 20)

all_responses = []
ct = 0
for gen in range(1,evolver.generation_number+1):
    evolver.ResponseGenerator.cycle_number = gen
    evolver.ResponseGenerator.loadResponseData(data_directory = data_directory)
    
    if gen in pull_cycles:
        ct += 1
        new_resp = evolver.ResponseGenerator.bot_data.copy()
        baseline = np.mean(new_resp[0:10])
        new_resp = (new_resp - baseline) / baseline
        
        ax = fh.add_subplot(7,1,ct)
        ax.plot(evolver.ResponseGenerator.time_step / 1e3,new_resp,'k')
        ax.set_title(str(gen))
        ax.set_ylim([-0.1, 1.5])
    
    response = []
    for individual in range(len(evolver.ResponseGenerator.stimulus_start_times)):
        response.append(evolver.ResponseGenerator.getBrukerBotResponse(individual))
    
    all_responses.append(response)
#%%
fh_2 = plt.figure(figsize=(4,6))
ax_2 = fh_2.add_subplot(111)
gg = np.arange(1,evolver.generation_number+1)
ax_2.plot(gg, np.vstack(all_responses),'k', LineWidth = 0.5, alpha = 0.4)
ax_2.plot(gg, np.mean(np.vstack(all_responses),axis = 1),'k')
plot_tools.addErrorBars(ax_2, gg, np.vstack(all_responses).T, stat = 'std')

ax_2.set_xlabel('Generation')
ax_2.set_ylabel('Response (dF/F)')

# %%
evolver.plotResults()


