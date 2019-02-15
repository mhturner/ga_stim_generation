# -*- coding: utf-8 -*-
"""
Created on Wed Jan 30 13:36:30 2019


Individual: genome -> stimulus -> response -> fitness


@author: mhturner
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

class StimulusEvolver():
    def __init__(self, stim_size = (8,10,20), pop_size = 40, num_persistent_parents = 10,
                 mutation_rate = 0.02, noise_level = 0.0, threshold = None,
                 sparsity_penalty = 0.0, similarity_penalty = 0.0,
                 stimulus_type = 'ternary_dense', response_type = 'model_RGC'):

        #population parameters
        self.pop_size = pop_size #individuals in each generation
        self.num_persistent_parents = num_persistent_parents #fittest individuals carried over to the next generation
        self.mutation_rate = mutation_rate # at each gene
        
        #mapping from response to fitness
        self.sparsity_penalty = sparsity_penalty
        self.similarity_penalty = similarity_penalty

        self.response = []
        self.fitness = []
        self.generation_number = 0  
        self.StimulusGenerator = StimulusGenerator(stimulus_type, stim_size) #maps genome -> stimulus
        
        if threshold is None:
            threshold = -np.inf
        self.ResponseGenerator = ResponseGenerator(response_type, noise_level, threshold, stim_size) #maps stimulus -> response
        
    def evolve(self, generations):
        if self.generation_number == 0:
            #generate the initial population
            self.initial_population = self.StimulusGenerator.getRandomGenes(pop_size = self.pop_size,num_genes = self.StimulusGenerator.genome_size)
            
        for gen in range(generations):
            self.doIteration()
            self.generation_number += 1

    def doIteration(self):
        if self.generation_number == 0:
            parent_population = self.initial_population
            self.current_generation = parent_population
        else:
            parent_population = self.current_generation.copy()

        #Get stimuli and responses for each individual in generation
        current_generation_responses = []
        current_generation_fitness = []
        for genome in parent_population:
            _stimulus = self.StimulusGenerator.getStimulusFromGenome(genome) #genome -> stimulus
            _response = self.ResponseGenerator.getResponse(_stimulus) #stimulus -> response
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

    def getFitness(self, stimulus, response):
        # add a sparsity penalty
        sparsity_factor = np.sum(np.abs(stimulus)) / stimulus.size  #fraction of activated pixels

        # add a similarity penalty
        if self.similarity_penalty > 0:
            pop_mean = np.mean(self.current_generation, axis = 0)
            norm_factor = np.max(signal.correlate(stimulus.flatten(),stimulus.flatten()))
            corr = np.max(signal.correlate(stimulus.flatten(),pop_mean)) / norm_factor
        else:
            corr = 0
                
        fitness = response - self.sparsity_penalty * sparsity_factor - self.similarity_penalty * corr
            
        return fitness

    def plotResults(self, fig_handle = None):
        if fig_handle is None:
            fig_handle = plt.figure(figsize=(9,7))
            
        grid = plt.GridSpec(10, self.StimulusGenerator.t_dim, wspace=0.1, hspace=0.1)
        ax_0 = fig_handle.add_subplot(grid[7:,0:5])
        ax_0.plot(self.fitness)
        ax_0.set_title('Fitness')
        ax_1 = fig_handle.add_subplot(grid[7:,7:12])
        ax_1.plot(self.response)
        ax_1.set_title('Response')

        for pp in range(5):
            for tt in range(self.StimulusGenerator.t_dim):
                ax = fig_handle.add_subplot(grid[pp,tt])
                current_stim = self.StimulusGenerator.getStimulusFromGenome(self.current_generation[-pp])
                vmin = np.min(current_stim)
                vmax = np.max(current_stim)
                ax.imshow(current_stim[:,:,tt], cmap=plt.cm.Greys_r, vmin = vmin, vmax = vmax)
                ax.set(xticks=[], yticks=[])
                
        for tt in range(self.StimulusGenerator.t_dim):
            ax = fig_handle.add_subplot(grid[pp+1,tt])
            all_stims = [self.StimulusGenerator.getStimulusFromGenome(x) for x in self.current_generation]
            mean_stim = np.mean(np.stack(all_stims, axis = 3), axis = 3)
            vmin = np.min(mean_stim)
            vmax = np.max(mean_stim)
            ax.imshow(mean_stim[:,:,tt], cmap=plt.cm.Greys_r, vmin = vmin, vmax = vmax)
            ax.set(xticks=[], yticks=[]) 

        if self.ResponseGenerator.response_type == 'model_DS':
            ax = fig_handle.add_subplot(grid[pp+2:pp+4,-3:])
            ax.imshow(self.ResponseGenerator.xt_RF)
            ax.set_axis_off()
        else:
            for tt in range(self.StimulusGenerator.t_dim):
                ax = fig_handle.add_subplot(grid[pp+1,tt])
                vmin = np.min(self.ResponseGenerator.RF)
                vmax = np.max(self.ResponseGenerator.RF)
                ax.imshow(self.ResponseGenerator.RF[:,:,tt], cmap=plt.cm.Greys_r, vmin = vmin, vmax = vmax)
                ax.set_axis_off()
            ax = fig_handle.add_subplot(grid[pp+2,:])
            ax.plot(self.ResponseGenerator.temporal_rf)
            ax.set_axis_off()
            
            
            
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
    def __init__(self, response_type, noise_level, threshold, stim_size):
        self.response_type = response_type
        self.noise_level = noise_level
        self.threshold = threshold
        if self.response_type[0:5] == 'model':
            self.y_dim = stim_size[0] #rows
            self.x_dim = stim_size[1] #columns
            self.t_dim = stim_size[2] #depth/time
            self.getModelRF()
        
    def getResponse(self, stimulus):
        if self.response_type[0:5] == 'model':
            response_sum = np.sum(stimulus * self.RF)
            
            #add noise to response
            response = response_sum + self.noise_level * response_sum * np.random.randn()
    
            # add a post-integration nonlinearity
            if self.threshold > -np.inf:
                if response < self.threshold:
                    response = 0
                else:
                    pass
        
        return response
        
    def getModelRF(self):
        if self.response_type == 'model_V1':
            x, y = np.meshgrid(np.arange(0,self.x_dim), np.arange(0,self.y_dim))
            xy_tuple = (x,y)
            
            on_lobe = Gauss_2D(xy_tuple, 2, 5, 4, 2, 1, np.deg2rad(45))
            off_lobe_1 = Gauss_2D(xy_tuple, 1, 4, 3, 2, 1, np.deg2rad(45))
            off_lobe_2 = Gauss_2D(xy_tuple, 1, 5, 6, 2, 1, np.deg2rad(45))

            self.spatial_rf = on_lobe - off_lobe_1 - off_lobe_2
            self.temporal_rf = getTemporalFilter(self.t_dim)
                
            self.RF = np.swapaxes(np.swapaxes(np.outer(self.temporal_rf,self.spatial_rf).reshape((self.t_dim,self.y_dim,self.x_dim)),0,2),0,1)

        elif self.response_type == 'model_RGC':
            x, y = np.meshgrid(np.arange(0,self.x_dim), np.arange(0,self.y_dim))
            xy_tuple = (x,y)
        
            center = Gauss_2D(xy_tuple, 2, 4, 4, 1, 1, 0)
            surround = Gauss_2D(xy_tuple, 1, 4, 4, 2, 2, 0)

            self.spatial_rf = center - surround
            self.temporal_rf = getTemporalFilter(self.t_dim)
                
            self.RF = np.swapaxes(np.swapaxes(np.outer(self.temporal_rf,self.spatial_rf).reshape((self.t_dim,self.y_dim,self.x_dim)),0,2),0,1)

        elif self.response_type == 'model_DS':
            #motion RF:
            x, t = np.meshgrid(np.arange(0,self.x_dim), np.arange(0,self.t_dim))
            xt_tuple = (x,t)
            
            self.xt_RF = Gauss_2D(xt_tuple, 1, 5, 10, 0.5, 8, np.deg2rad(30))
            
            self.y_RF = Gauss_1D(np.arange(self.y_dim), 2, 4, 0.5)
            
            self.RF = np.swapaxes(np.outer(self.y_RF,self.xt_RF).reshape((self.y_dim,self.t_dim,self.x_dim)),1,2)

        else:
            self.RF = 0
            
        # normalize RF
        norm_factor = np.sum(np.abs(self.RF)) #maximum amount the RF can be activated
        self.RF /= norm_factor

def getTemporalFilter(t_dim):
    on_time = int(t_dim * 0.5)
    tFilt = np.zeros(shape = t_dim)
    tFilt[on_time-3:on_time] = -1
    tFilt[on_time+1:on_time+4] = 1

    return tFilt      

def softmax(vec):
    return np.exp(vec) / np.sum(np.exp(vec))

def Gauss_1D(xx, amplitude, x0, sigma):
    return amplitude * np.exp(-(xx-x0)**2/(2*sigma**2))

def Gauss_2D(xy_tuple, amplitude, xo, yo, sigma_x, sigma_y, theta):
        # https://en.wikipedia.org/wiki/Gaussian_function
        # https://stackoverflow.com/questions/21566379/fitting-a-2d-gaussian-function-using-scipy-optimize-curve-fit-valueerror-and-m
        x, y = xy_tuple
        xo = float(xo)
        yo = float(yo)    
        a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
        b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
        c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
        resp = amplitude*np.exp( - (a*((x-xo)**2) + 2*b*(x-xo)*(y-yo) + c*((y-yo)**2)))
        return resp
