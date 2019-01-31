# -*- coding: utf-8 -*-
"""
Created on Wed Jan 30 13:36:30 2019

@author: mhturner
"""
import numpy as np
import matplotlib.pyplot as plt

class StimulusEvolver():
    def __init__(self):
        self.y_dim = 8 #rows
        self.x_dim = 10 #columns
        self.t_dim = 20 #depth
        
        self.pop_size = 40 #individuals in each generation
        self.num_persistent_parents = 10 #fittest individuals carried over to the next generation
        
        self.mutation_rate = 0.02 # at each gene
        
        self.genome_size = self.y_dim * self.x_dim * self.t_dim

        self.fitness = []
        self.generation_number = 0  
        self.fitness_function = 'model_RGC'

    def evolve(self, generations):
        if self.generation_number == 0:
            self.initializeFounderPopulation()
        
        for gen in range(generations):
            self.doIteration()
            self.generation_number += 1
    
    def initializeFounderPopulation(self):
        #generate the initial population
        self.genome_size = self.y_dim * self.x_dim * self.t_dim
        self.initial_population = np.random.randint(-1,2,size = (self.pop_size, self.genome_size)) #individuals x genes
        
    
    def doIteration(self):
        if self.generation_number == 0:
            parent_population = self.initial_population
        else:
            parent_population = self.current_generation.copy()

        #calculate the fitness across the population
        fitness_raw = [self.getFitness(x.reshape(self.y_dim,self.x_dim,self.t_dim)) for x in parent_population]
        self.fitness.append(fitness_raw)
        #z-score and rank each individual in this generation
        fitness_z = (fitness_raw - np.mean(fitness_raw)) / np.std(fitness_raw)
        fitness_rankings = np.argsort(fitness_z)[::-1]
        
        if np.std(fitness_raw) == 0: #no variability in population
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
            
            parent_2_inherited_genes = np.where(np.random.randint(2,size = self.genome_size))[0]
            child_genome[parent_2_inherited_genes] = parent_population[parents[1], parent_2_inherited_genes]
        
            #mutate child genes
            mutated_genes = np.random.choice(self.genome_size, size = int(self.genome_size * self.mutation_rate))
            child_genome[mutated_genes] = np.random.randint(-1,2,size = len(mutated_genes))
    
            children.append(child_genome)
        
        
        persistent_parents = fitness_rankings[0:self.num_persistent_parents] #take the top n into the next generation
        
        
        #update newest generation
        children.append(parent_population[persistent_parents,:])
        self.current_generation = np.vstack(children)
    
    
    
    def getFitness(self, stimulus):
        
        if self.fitness_function == 'model_V1':
            x, y = np.meshgrid(np.arange(0,stimulus.shape[1]), np.arange(0,stimulus.shape[0]))
            xy_tuple = (x,y)
            
            on_lobe = Gauss_2D(xy_tuple, 2, 5, 4, 2, 1, np.deg2rad(45))
            off_lobe_1 = Gauss_2D(xy_tuple, 1, 4, 3, 2, 1, np.deg2rad(45))
            off_lobe_2 = Gauss_2D(xy_tuple, 1, 5, 6, 2, 1, np.deg2rad(45))

            self.spatial_rf = on_lobe - off_lobe_1 - off_lobe_2
            self.temporal_rf = getTemporalFilter(stimulus.shape[2])
                
            self.RF = np.swapaxes(np.swapaxes(np.outer(self.temporal_rf,self.spatial_rf).reshape((self.t_dim,self.y_dim,self.x_dim)),0,2),0,1)
            
            response_sum = np.sum(stimulus * self.RF)
        
            fitness = response_sum
        elif self.fitness_function == 'model_RGC':
            x, y = np.meshgrid(np.arange(0,stimulus.shape[1]), np.arange(0,stimulus.shape[0]))
            xy_tuple = (x,y)
        
            center = Gauss_2D(xy_tuple, 2, 4, 4, 1, 1, 0)
            surround = Gauss_2D(xy_tuple, 1, 4, 4, 2, 2, 0)

            self.spatial_rf = center - surround
            self.temporal_rf = getTemporalFilter(stimulus.shape[2])
                
            self.RF = np.swapaxes(np.swapaxes(np.outer(self.temporal_rf,self.spatial_rf).reshape((self.t_dim,self.y_dim,self.x_dim)),0,2),0,1)
            
            response_sum = np.sum(stimulus * self.RF)
        
            fitness = response_sum
            
        elif self.fitness_function == 'model_DS':
            #motion RF:
            x, t = np.meshgrid(np.arange(0,stimulus.shape[1]), np.arange(0,stimulus.shape[2]))
            xt_tuple = (x,t)
            
            self.xt_RF = Gauss_2D(xt_tuple, 1, 5, 8, 0.25, 8, np.deg2rad(45))
            
            self.y_RF = Gauss_1D(np.arange(self.y_dim), 2, 4, 2)
            
            self.RF = np.swapaxes(np.outer(self.y_RF,self.xt_RF).reshape((self.y_dim,self.t_dim,self.x_dim)),1,2)

            response_sum = np.sum(stimulus * self.RF)
        
            fitness = response_sum
            
        return fitness

        
    def plotResults(self):
        plt.plot(self.fitness)
        
        fh = plt.figure(figsize=(9,7))

        grid = plt.GridSpec(7, self.t_dim, wspace=0.1, hspace=0.1)
        for pp in range(5):
            for tt in range(self.t_dim):
                ax = fh.add_subplot(grid[pp,tt])
                current_stim = self.current_generation[-pp].reshape((self.y_dim,self.x_dim,self.t_dim))
                vmin = np.min(current_stim)
                vmax = np.max(current_stim)
                ax.imshow(current_stim[:,:,tt], cmap=plt.cm.Greys_r, vmin = vmin, vmax = vmax)
                ax.set(xticks=[], yticks=[])

        if self.fitness_function == 'model_DS':
            ax = fh.add_subplot(grid[pp+1:pp+2,-2:])
            ax.imshow(self.xt_RF)
            ax.set_axis_off()
        else:
            for tt in range(self.t_dim):
                ax = fh.add_subplot(grid[pp+1,tt])
                vmin = np.min(self.RF)
                vmax = np.max(self.RF)
                ax.imshow(self.RF[:,:,tt], cmap=plt.cm.Greys_r, vmin = vmin, vmax = vmax)
                ax.set_axis_off()
            ax = fh.add_subplot(grid[pp+2,:])
            ax.plot(self.temporal_rf)
            ax.set_axis_off()

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
