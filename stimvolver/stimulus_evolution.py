# -*- coding: utf-8 -*-
"""
Created on Wed Jan 30 13:36:30 2019


Individual: genome -> stimulus -> response -> fitness


@author: mhturner
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import pandas as pd
import os
import time
from datetime import datetime
import socket

from fiver import imaging_data

class StimulusEvolver():
    def __init__(self, stim_size = (8,10,20), pop_size = 40, num_persistent_parents = 10,
                 mutation_rate = 0.02,
                 sparsity_penalty = 0.0, similarity_penalty_weight = 0.0, prior_solutions = [],
                 stimulus_type = 'ternary_dense', response_type = 'Bruker_bot'):

        #population parameters
        self.pop_size = pop_size #individuals in each generation
        self.num_persistent_parents = num_persistent_parents #fittest individuals carried over to the next generation
        self.mutation_rate = mutation_rate # at each gene
        
        #mapping from response to fitness
        self.sparsity_penalty = sparsity_penalty
        
        self.similarity_penalty_weight = similarity_penalty_weight
        self.prior_solutions = prior_solutions

        self.response = []
        self.fitness = []
        self.generation_number = 0  
        self.StimulusGenerator = StimulusGenerator(stimulus_type, stim_size) #maps genome -> stimulus
        

        self.ResponseGenerator = ResponseGenerator(response_type)
        self.response_type = response_type
    def initializePopulation(self):
        #generate the initial population
        self.initial_population = self.StimulusGenerator.getRandomGenes(pop_size = self.pop_size,num_genes = self.StimulusGenerator.genome_size)
        self.current_generation = self.initial_population.copy()
        
    def evolve(self, generations):
        for gen in range(generations):
            self.doIteration()
            self.generation_number += 1

    def doIteration(self):
        parent_population = self.current_generation.copy()

        #Get stimuli and responses for each individual in generation
        current_generation_responses = []
        current_generation_fitness = []
        
        self.ResponseGenerator.loadResponseData()
        for ind_no, genome in enumerate(parent_population):
            _stimulus = self.StimulusGenerator.getStimulusFromGenome(genome) #genome -> stimulus
            _response = self.ResponseGenerator.getBrukerBotResponse(ind_no) #stimulus -> response

            current_generation_responses.append(_response)
            _fitness = self.getFitness(_stimulus, _response) # response -> fitness
            current_generation_fitness.append(_fitness)

        self.response.append(current_generation_responses) #keep track of responses and fitness across all generations
        self.fitness.append(current_generation_fitness)

        #z-score and rank each individual's response in this generation
        fitness_z = (current_generation_fitness - np.mean(current_generation_fitness)) / np.std(current_generation_fitness)
        fitness_rankings = np.argsort(fitness_z)[::-1]
        
        if np.std(current_generation_fitness) == 0: #no variability in population
            prob_parenthood = np.ones(shape = self.pop_size) / self.pop_size
        else:
            prob_parenthood = softmax(fitness_z)
        
        #generate children
        children = []
        for child_no in range(self.pop_size-self.num_persistent_parents):
            #choose parents
            parents = np.random.choice(len(prob_parenthood), p = prob_parenthood, size = 2, replace = False)
            
            #recombine parent genomes
            child_genome = parent_population[parents[0],:].copy()
            
            parent_2_inherited_genes = np.where(np.random.randint(2,size = self.StimulusGenerator.genome_size))[0] #binary of genes inherited from parent 2
            child_genome[parent_2_inherited_genes] = parent_population[parents[1], parent_2_inherited_genes]
        
            #mutate child genes
            mutated_genes = np.random.choice(self.StimulusGenerator.genome_size, size = int(self.StimulusGenerator.genome_size * self.mutation_rate))
            child_genome[mutated_genes] = self.StimulusGenerator.getRandomGenes(pop_size = 1, num_genes = len(mutated_genes))
    
            children.append(child_genome)
        
        
        persistent_parents = fitness_rankings[0:self.num_persistent_parents] #take the top n into the next generation
        
        #update newest generation
        children.append(parent_population[persistent_parents,:])
        self.current_generation = np.vstack(children)
        np.random.shuffle(self.current_generation)

    def getFitness(self, stimulus, response):
        # add a sparsity penalty
        sparsity_factor = np.sum(np.abs(stimulus)) / stimulus.size  #fraction of activated pixels

        # add a similarity penalty
        if (self.similarity_penalty_weight > 0) & (len(self.prior_solutions) > 0):
            corr = []
            for sol in self.prior_solutions:
                norm_factor = np.max(signal.correlate(stimulus.flatten(),stimulus.flatten()))
                corr.append(np.max(signal.correlate(stimulus.flatten(),sol)) / norm_factor)
            
            similarity_penalty = self.similarity_penalty_weight * np.max(corr)
            
        else:
            similarity_penalty = 0
                
        fitness = response - self.sparsity_penalty * sparsity_factor - similarity_penalty
            
        return fitness

    def plotResults(self, fig_handle = None, plot_ind = 0):
        if fig_handle is None:
            fig_handle = plt.figure(figsize=(9,7))
            
        grid = plt.GridSpec(10, self.StimulusGenerator.t_dim + 1, wspace=0.1, hspace=0.1)
        ax_0 = fig_handle.add_subplot(grid[7:,0])
        ax_0.plot(np.mean(np.vstack(self.fitness), axis = 1))
        ax_0.set_title('Fitness')
        ax_1 = fig_handle.add_subplot(grid[7:,1])
        ax_1.plot(np.mean(np.vstack(self.response), axis = 1))
        ax_1.set_title('Response')

        # plot mean stimulus in population
        for tt in range(self.StimulusGenerator.t_dim):
            ax = fig_handle.add_subplot(grid[plot_ind,tt])
            all_stims = [self.StimulusGenerator.getStimulusFromGenome(x) for x in self.current_generation]
            mean_stim = np.mean(np.stack(all_stims, axis = 3), axis = 3)
            vmin = np.min(mean_stim)
            vmax = np.max(mean_stim)
            ax.imshow(mean_stim[:,:,tt], cmap=plt.cm.Greys_r, vmin = vmin, vmax = vmax)
            ax.set(xticks=[], yticks=[]) 
            
            
class StimulusGenerator():
    def __init__(self, stimulus_type, stim_size):
        self.stimulus_type = stimulus_type
        self.stim_size = stim_size
        self.y_dim = stim_size[0] #rows
        self.x_dim = stim_size[1] #columns
        self.t_dim = stim_size[2] #depth/time
        
        if self.stimulus_type == 'ternary_dense':
            self.genome_size = np.prod(stim_size)
        elif self.stimulus_type == 'single_spot':
            self.genome_size = self.t_dim

    def getStimulusFromGenome(self,genome):
        if self.stimulus_type == 'ternary_dense':
            stimulus = genome.reshape(self.y_dim,self.x_dim,self.t_dim)
        elif self.stimulus_type == 'single_spot':
            row, col = self.getRowColumnFromLocation(genome, self.y_dim, self.x_dim)
            stimulus = np.ndarray((self.y_dim,self.x_dim,self.t_dim))
            stimulus[:] = 0
            for ff in range(stimulus.shape[2]):
                stimulus[col[ff], row[ff], ff] = 1
        else:
            pass
        return stimulus
        
    def getRandomGenes(self, pop_size, num_genes):
        if self.stimulus_type == 'ternary_dense':
            # each gene can be [-1, 0, +1], there is one gene per pixel per time step
            genome = np.random.randint(-1,2,size = (pop_size, num_genes)) #individuals x genes
        elif self.stimulus_type == 'single_spot':
            # each gene can be [0,...,y_dim*x_dim], there is one gene per time step (location of spot on screen)
            genome = np.random.choice(range(self.y_dim * self.x_dim), size = (pop_size, num_genes))
        else:
            pass
        return genome
    
    def getRowColumnFromLocation(self, location, y_dim, x_dim):
        row = np.mod(location, x_dim) - 1
        col = np.mod(location, y_dim) - 1
        return row, col
        
class ResponseGenerator():
    def __init__(self, response_type):
        self.response_type = response_type


    def loadResponseData(self, data_directory = None):
        if data_directory is None:
            date_dir = datetime.now().isoformat()[:-16].replace('-','')
            data_directory = os.path.join('E:/Max',date_dir)
            
        cycle_code = ('0000' + str(self.cycle_number))[-5:]
        bot_suffix = '_Cycle' + cycle_code + '-botData'
        
        bot_file_path = os.path.join(data_directory, self.series_name)
        bot_file_name = self.series_name + bot_suffix + '.csv'
        file_list_name = 'Cycle' + cycle_code + '_Filelist.txt'
                
        if socket.gethostname() == "USERBRU-I10P5LO": #bruker computer
            print('Waiting for files from PV')
            print(bot_file_path)
            while not os.path.exists(os.path.join(bot_file_path, bot_file_name)):
                time.sleep(0.5) #wait for bot file to appear (beginning of PV processing)
            while os.path.exists(os.path.join(bot_file_path, file_list_name)):
                time.sleep(0.5) #wait for the filelist file to disappear (after PV finishes processing)
            #time.sleep(1)
        else:
            pass #analysis computer, file should already be there

        v_rec_suffix = '_Cycle' + cycle_code + '_VoltageRecording_001'
        stimulus_timing = imaging_data.getEpochAndFrameTiming(os.path.join(data_directory, self.series_name),
                                                              self.series_name,
                                                              v_rec_suffix = v_rec_suffix,
                                                              plot_trace_flag=False)
        
        self.stimulus_start_times = stimulus_timing['stimulus_start_times'].copy() #msec
        self.stimulus_end_times = stimulus_timing['stimulus_end_times'].copy()
            
        #read bot data
        data_frame = pd.read_csv(os.path.join(bot_file_path, bot_file_name))
        self.time_step = data_frame.loc[:]['Timestamp']
        self.time_step -= self.time_step[0]
        self.time_step *= 1e3 #sec->msec
        self.bot_data = data_frame.loc[:]['Region 1']
        print('Data loaded')

    def getBrukerBotResponse(self, epoch_number):
        b_inds_1 = np.where(self.time_step > (self.stimulus_start_times[epoch_number] - self.pre_time))[0]
        b_inds_2 = np.where(self.time_step <=self.stimulus_start_times[epoch_number])[0]
        baseline_inds = np.intersect1d(b_inds_1,b_inds_2)
        baseline = np.mean(self.bot_data[baseline_inds])
        
        p_inds_1 = np.where(self.time_step > self.stimulus_start_times[epoch_number])
        p_inds_2 = np.where(self.time_step <= self.stimulus_end_times[epoch_number])
        pull_inds = np.intersect1d(p_inds_1,p_inds_2)
                
        response = np.max((self.bot_data[pull_inds] - baseline) / baseline) #dF/F

        return response
        
  
def softmax(vec):
    return np.exp(vec) / np.sum(np.exp(vec))
