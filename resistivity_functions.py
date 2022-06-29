# -*- coding: utf-8 -*-
"""
Functions for processing resistivity data using Self-Organizing Map.
"""

import numpy as np
import matplotlib.pyplot as plt

import pandas as pd
import seaborn as sns

from ipywidgets import interact

from collections import Counter
from matplotlib.colors import LogNorm

from minisom_resistivity import MiniSom    

import numpy.ma as ma

#Class for analyzing and visualizing the data
class Data:
    
    #Initialize class using input dataframe
    def __init__(self, dataframe):
        self.dataframe = dataframe
    
    #Display rows from row_start to row_end
    def display(self, row_start, row_end):
        df = self.dataframe
        print(df[row_start:row_end])
        
    #Identify rows with implausible resistivity values
    def implausible_resistivities(self):
        df = self.dataframe
        print(df[df['rho']<0])
      
    #Sort a column of the dataframe
    def sort_variable(self, variable):
        df = self.dataframe
        
        #if (variable=='x') or (variable=='y') or (variable=='z'):
        #    variable_sort = np.sort(np.unique(df[variable]))
        
        if variable=='rho':
            df_positive_rho = df[df[variable]>0].copy()
            variable_sort = np.sort(np.unique(df_positive_rho[variable]))
            
        else:
            variable_sort = np.sort(np.unique(df[variable]))
            
        return variable_sort

    #Determine minimum of a column of the dataframe
    def find_min(self, variable):
        variable_sort = self.sort_variable(variable)
        variable_min = np.min(variable_sort)
        return variable_min
    
    #Determine maximum of a column of the dataframe
    def find_max(self, variable):
        variable_sort = self.sort_variable(variable)
        variable_max = np.max(variable_sort)
        return variable_max
  
    #Determine step size of a variable in a column
    def find_step(self, variable):
        variable_sort = self.sort_variable(variable)
        variable_step = variable_sort[1] - variable_sort[0]
        return variable_step
        
    #Find number of unique values of a variable in the dataframe
    def find_num_datapoints(self, variable):
        variable_sort = self.sort_variable(variable)
        num_datapoints = variable_sort.shape[0]
        return num_datapoints
    
    #Find mean of a variable in the dataframe
    def find_mean(self, variable):
        variable_sort = self.sort_variable(variable)
        variable_mean = np.mean(variable_sort)
        return variable_mean       
     
    #Find standard deviation of a variable in the dataframe
    def find_std(self, variable):
        variable_sort = self.sort_variable(variable)
        variable_std = np.std(variable_sort)
        return variable_std
     
    #Summary of data
    def data_summary(self):
        
        summary = {}
        summary['variable'] = ['x', 'y', 'z', 'rho']
        summary['num_values'] = [self.find_num_datapoints('x'), self.find_num_datapoints('y'), self.find_num_datapoints('z'),'']
        summary['min'] = [self.find_min('x'), self.find_min('y'), self.find_min('z'), self.find_min('rho')]
        summary['max'] = [self.find_max('x'), self.find_max('y'), self.find_max('z'), self.find_max('rho')]
        summary['step_size'] = [self.find_step('x'), self.find_step('y'), self.find_step('z'),''] 
        summary['mean'] = [self.find_mean('x'), self.find_mean('y'), self.find_mean('z'), self.find_mean('rho')]
        summary['std'] = [self.find_std('x'), self.find_std('y'), self.find_std('z'), self.find_std('rho')]
        
        df_summary = pd.DataFrame(data=summary)
        print(df_summary.round(2))
        
    #Display histogram of resistivity values
    def display_histogram_rho(self, bin_range_min=None, bin_range_max=None):
        
        df = self.dataframe
        
        df_positive_rho = df[df['rho']>0]['rho']
        
        if bin_range_min is None:
            bin_range_min = self.find_min('rho')
        if bin_range_max is None:
            bin_range_max = self.find_max('rho')  
        
        ax = sns.histplot(data=df_positive_rho, binrange=[bin_range_min, bin_range_max])
        ax.set_xlabel(r'$\rho$')
    
    #Display one slice along coordinate variable
    def display_one_slice(self, variable, value, vmin=1, vmax=1e3, xticklabels=10, yticklabels=20):
        
        df = self.dataframe
        
        rho_slice = df[(df[variable]==value)&(df['rho']>0)]
        
        N_x = self.find_num_datapoints('x')
        N_y = self.find_num_datapoints('y')
        N_z = self.find_num_datapoints('z')
        
        if variable=='z':
            rho_slice = rho_slice.pivot('y','x','rho')
            aspect = N_x/N_y
        if variable=='x':
            rho_slice = rho_slice.pivot('z','y','rho')
            aspect = N_y/N_z
        if variable=='y':
            rho_slice = rho_slice.pivot('z','x','rho')
            aspect = N_x/N_z
            
        ax = sns.heatmap(rho_slice, norm=LogNorm(vmin=vmin, vmax=vmax), xticklabels=xticklabels, yticklabels=yticklabels)
        ax.set_aspect(aspect)
        ax.invert_yaxis()
           
    #Display interactive cross-section along coordinate variable
    def display_cross_section(self, variable, quantity='rho', logscale=True, vmin=1, vmax=1e3, xticklabels=10, yticklabels=20, cmap=None):
        
        df = self.dataframe
        
        N_x = self.find_num_datapoints('x')
        N_y = self.find_num_datapoints('y')
        N_z = self.find_num_datapoints('z')
        
        def f(value):
            
            if quantity=='rho':
                slice = df[(df[variable]==value)&(df['rho']>0)]
            else:
                slice = df[(df[variable]==value)]
                
            if variable=='z':
                slice = slice.pivot('y','x',quantity)
                aspect = N_x/N_y
            if variable=='x':
                slice = slice.pivot('z','y',quantity)
                aspect = N_y/N_z
            if variable=='y':
                slice = slice.pivot('z','x',quantity)
                aspect = N_x/N_z
                                    
            if logscale:    
                ax = sns.heatmap(slice, norm=LogNorm(vmin=vmin, vmax=vmax), xticklabels=xticklabels, yticklabels=yticklabels, cmap=cmap)
            else:
                ax = sns.heatmap(slice, vmin=vmin, vmax=vmax, xticklabels=xticklabels, yticklabels=yticklabels, cmap=cmap)
                
            ax.set_aspect(aspect)
            ax.invert_yaxis()
            
        min_value = self.find_min(variable)
        max_value = self.find_max(variable)
        step = self.find_step(variable)
            
        interact(f, value = (min_value, max_value, step))
        
    #Convert dataframe to numpy array
    def dataframe_to_numpy_array(self):
        df = self.dataframe
        data = df.to_numpy()
        return data    
    
    #Mask numpy array
    def mask_data(self, data, mask_value=-9999.0):
        masked_data = ma.masked_values(data, mask_value)
        return masked_data
    
    #Normalize numpy array
    def normalize_data(self, data):
        self.data_mean = np.mean(data, axis=0)
        self.data_std = np.std(data, axis=0)
        normalized_data = (data - self.data_mean) / self.data_std
        self.normalized_data = normalized_data
        return normalized_data        
    
    #Normalize numpy array, using robust scaling
    def normalize_data_robust_scaling(self, data):
        
        len_data_vector = np.shape(data)[1]
        
        #compute median, 1st quartile, 3rd quartile for each element of data vector
        median = np.zeros(len_data_vector)
        quartile1 = np.zeros(len_data_vector)
        quartile3 = np.zeros(len_data_vector)
        
        for i in range(len_data_vector):
            data_i = data[:,i]
            data_i_compressed = ma.compressed(data_i)
            
            if len(data_i_compressed) > 0:            
                median[i] = np.quantile(data_i_compressed, 0.5)
                quartile1[i] = np.quantile(data_i_compressed, 0.25)      
                quartile3[i] = np.quantile(data_i_compressed, 0.75)
                
            else:            
                median[i] = ma.masked
                quartile1[i] = ma.masked      
                quartile3[i] = ma.masked
                
        data_mean = median
        data_std = quartile3 - quartile1
        normalized_data = (data - data_mean) / data_std
        
        self.data_mean = data_mean
        self.data_std = data_std
        
        self.normalized_data = normalized_data
        
        return normalized_data 

    #Normalize numpy array, using robust scaling
    #Input vector is [x,y,z,rho]
    def normalize_x_y_z_rho_data_positional_weight_robust_scaling(self, data, positional_weight=1/3):

        len_data_vector = np.shape(data)[1]
        
        #compute median, 1st quartile, 3rd quartile for each element of data vector
        median = np.zeros(len_data_vector)
        quartile1 = np.zeros(len_data_vector)
        quartile3 = np.zeros(len_data_vector)
        
        for i in range(len_data_vector):
            data_i = data[:,i]
            data_i_compressed = ma.compressed(data_i)
            
            if len(data_i_compressed) > 0:            
                median[i] = np.quantile(data_i_compressed, 0.5)
                quartile1[i] = np.quantile(data_i_compressed, 0.25)      
                quartile3[i] = np.quantile(data_i_compressed, 0.75)
                
            else:            
                median[i] = ma.masked
                quartile1[i] = ma.masked      
                quartile3[i] = ma.masked
                
        data_mean = median
        data_std = quartile3 - quartile1
        normalized_data = (data - data_mean) / data_std
        
        normalized_data[:,0:3] = normalized_data[:,0:3]*positional_weight
        
        self.data_mean = data_mean
        self.data_std = data_std
        
        self.normalized_data = normalized_data
        
        return normalized_data


#Class for implementing and visualizing the Self-Organizing Map technique on the data
class SOM(Data):
    
    #Initialize SOM
    def __init__(self, dataframe, input_len, som_x_dim=30, som_y_dim=30, som_learning_rate=1, som_random_seed=2, som_sigma=None, activation_distance='euclidean'):

        super().__init__(dataframe)

        if som_sigma is None:
            som_sigma = np.min([som_x_dim, som_y_dim])/3
            
        #x and y dimensions of SOM
        self.som_x_dim = som_x_dim
        self.som_y_dim = som_y_dim 
        
        self.som = MiniSom(x=som_x_dim, y=som_y_dim, input_len=input_len, sigma=som_sigma, learning_rate=som_learning_rate, random_seed=som_random_seed, activation_distance=activation_distance) 
        
#    def get_som(self):
#        som = self.som
#        return som
    
    #Train SOM
    def train_som(self, max_iterations=2000, mask_value=-9999.0, verbose=False):
        som = self.som
        numpy_data = self.dataframe_to_numpy_array()
        masked_data = self.mask_data(numpy_data, mask_value)
        normalized_data = self.normalize_data(masked_data)
        som.train_batch(normalized_data, num_iteration=max_iterations, verbose=verbose)

    #Train SOM
    def train_som_incomplete_data(self, good_data_indices, max_iterations=20000, mask_value=-9999, verbose=False, scaling='standard', positional_weight=None, truncation_depth_index=None):

        som = self.som
        numpy_data = self.dataframe_to_numpy_array()
        masked_data = self.mask_data(data=numpy_data,mask_value=mask_value)
        
        if positional_weight is None:
        
            if scaling=='standard':
                normalized_data = self.normalize_data(masked_data)
            if scaling=='robust':
                normalized_data = self.normalize_data_robust_scaling(masked_data)
                
            if truncation_depth_index:
                data = normalized_data[:,0:truncation_depth_index]
            else:
                data = normalized_data
                
        else:
            
            data = self.normalize_x_y_z_rho_data_positional_weight_robust_scaling(masked_data, positional_weight)
            
        som.train_incomplete_data(data=data, num_iteration=max_iterations, good_data_indices=good_data_indices, verbose=verbose)
        
    def train_som_masked_data(self, max_iterations=2000, mask_value=-9999.0, verbose=False):
        som = self.som
        numpy_data = self.dataframe_to_numpy_array()
        masked_data = self.mask_data(numpy_data, mask_value)
        normalized_data = self.normalize_data(masked_data)
        som.train_masked_data(normalized_data, num_iteration=max_iterations, verbose=verbose)
        
    #Compute quantized image
    #N_x, N_y are number of datapoints along x and y axes
    def compute_quantized_data(self, truncation_depth_index=None):
        som = self.som

        normalized_data = self.normalized_data
        if truncation_depth_index:
            normalized_data = normalized_data[:,0:truncation_depth_index]
            
        #Find quantized values (Best Matching Unit or BMU) for each data vector
        quantized_values_normalized = som.quantization(normalized_data)
        
        data_mean = self.data_mean[0:truncation_depth_index]
        data_std = self.data_std[0:truncation_depth_index]
        
        #Quantized values, unnormalized
        quantized_data = data_mean + data_std*quantized_values_normalized

        return quantized_data

    def compute_quantized_image(self, N_x=95, N_y=151, truncation_depth_index=None):

        quantized_data = self.compute_quantized_data(N_x, N_y, truncation_depth_index)       

        quantized_image = np.zeros((N_x,N_y,truncation_depth_index))
        
        #Map quantized values to original data space dimensions        
        for i in range(truncation_depth_index):            
            quantized_image[:,:,i] = np.reshape(quantized_data[:,i],(N_x,N_y))

        return quantized_image
    
    #TODO
    #Display quantized image
    def display_quantized_image(self, N_x=95, N_y=151, xticklabels=10, yticklabels=10, z=0, z_min=-50, z_step=2):
        quantized_data = self.compute_quantized_image(N_x, N_y, z, z_min, z_step)
        ax = sns.heatmap(quantized_data, xticklabels=xticklabels, yticklabels=yticklabels)
        ax.invert_yaxis()   
        
    #Compute quantized image
    #N_x, N_y are number of datapoints along x and y axes
    def compute_quantized_image_single_z_slice(self, N_x=95, N_y=151):
        som = self.som
        #Find quantized values (Best Matching Unit or BMU) for each data vector
        normalized_data = self.normalized_data
        quantized_values_normalized = som.quantization(normalized_data)
        
        data_mean = self.data_mean
        data_std = self.data_std
        
        quantized_values = data_mean + data_std*quantized_values_normalized
        
        #Map quantized values to original data space dimensions
        quantized = np.zeros((N_x,N_y))
        for i, q in enumerate(quantized_values):  # place the quantized values into a new image
            quantized[np.unravel_index(i, shape=(N_x,N_y))] = q
        quantized = quantized.T
    
        return quantized
    
    #Display quantized image
    def display_quantized_image_single_z_slice(self, N_x=95, N_y=151, xticklabels=10, yticklabels=10):
        quantized_data = self.compute_quantized_image_single_z_slice(N_x, N_y)
        ax = sns.heatmap(quantized_data, xticklabels=xticklabels, yticklabels=yticklabels)
        ax.invert_yaxis()
        return ax
        
    #Compute U-matrix, which shows degree of similarity of each pixel to its neighbours
    def compute_u_matrix(self):
        som = self.som
        u_matrix = som.distance_map()  
        return u_matrix
        
    #Display U-matrix
    def display_u_matrix(self, xticklabels=5, yticklabels=5, square=True):
        u_matrix = self.compute_u_matrix()       
        ax = sns.heatmap(u_matrix, xticklabels=xticklabels, yticklabels=yticklabels, square=square)
        ax.invert_yaxis()
        #return ax
    
    #Compute component plot: Data contains only rho values for a single slice
    def compute_component_plot_single_z_slice(self):
            
        som = self.som      
        som_x_dim = self.som_x_dim
        
        som_weights_component_normalized = som.get_weights()

        data_mean = self.data_mean
        data_std = self.data_std
        
        data_mean_component = data_mean
        data_std_component = data_std
        
        som_weights_component = data_mean_component + data_std_component*som_weights_component_normalized
        som_weights = som_weights_component.reshape(som_x_dim,-1)

        return som_weights
    
    #Display component plot: Data contains only rho values for a single slice    
    def display_component_plot_single_z_slice(self):
        som_weights = self.compute_component_plot_single_z_slice()
        ax = sns.heatmap(som_weights, square=True)
        ax.invert_yaxis()
    
    #Compute component plot: Data contains rho values for a single slice, and (x,y) coordinates    
    def compute_component_plot_single_z_slice_xy(self, component_type='rho'):
 
        if component_type=='x':
            vector_index = 0
        if component_type=='y':
            vector_index = 1
        if component_type=='rho':
            vector_index = 2            
            
        som = self.som      
        som_x_dim = self.som_x_dim
        
        som_weights_vector = som.get_weights()
        som_weights_component_normalized = som_weights_vector[:,:,vector_index]
        data_mean = self.data_mean
        data_std = self.data_std
        
        data_mean_component = data_mean[vector_index]
        data_std_component = data_std[vector_index]
        
        som_weights_component = data_mean_component + data_std_component*som_weights_component_normalized
        som_weights = som_weights_component.reshape(som_x_dim,-1)

        return som_weights
    
    #Display component plot: Data contains rho values for a single slice, and (x,y) coordinates     
    def display_component_plot_single_z_slice_xy(self, component_type='rho'):
        som_weights = self.compute_component_plot_single_z_slice_xy(component_type)
        ax = sns.heatmap(som_weights, square=True)
        ax.invert_yaxis()

    #Compute component plot: Data contains rho values for entire data, all z-values       
    def compute_component_plot(self, component_type='rho', z=0, z_min=-50, z_step=2):
                    
        vector_index = np.int((z-z_min)/z_step)
            
        som = self.som      
        som_x_dim = self.som_x_dim
        
        som_weights_vector = som.get_weights()
        som_weights_component_normalized = som_weights_vector[:,:,vector_index]
        data_mean = self.data_mean
        data_std = self.data_std
        
        data_mean_component = data_mean[vector_index]
        data_std_component = data_std[vector_index]
        
        som_weights_component = data_mean_component + data_std_component*som_weights_component_normalized
        som_weights = som_weights_component.reshape(som_x_dim,-1)
        
        return som_weights
    
    #TODO
    #Display component plot: Data contains rho values for entire data, all z-values
    def display_component_plot(self, component_type='rho', z=0, z_min=-50, z_step=2):
        som_weights = self.compute_component_plot(component_type, z, z_min, z_step)
        ax = sns.heatmap(som_weights, square=True)
        ax.invert_yaxis()
    
    #TODO
    #Compute component plot: Data contains rho values for entire data, all z-values       
    def compute_component_plot_xy(self, component_type='rho', z=0, z_min=-50, z_step=2):
        df = self.dataframe
        
        if 'x' in df.columns:
            if component_type=='x':
                vector_index = 0
            if component_type=='y':
                vector_index = 1
            if component_type=='rho':
                vector_index = 2 + np.int((z-z_min)/z_step)            
 
        if not 'x' in df.columns:            
                    vector_index = np.int((z-z_min)/z_step)
            
        som = self.som      
        som_x_dim = self.som_x_dim
        
        som_weights_vector = som.get_weights()
        som_weights_component_normalized = som_weights_vector[:,:,vector_index]
        
        data_mean = self.data_mean
        data_std = self.data_std
        
        data_mean_component = data_mean[vector_index]
        data_std_component = data_std[vector_index]
        
        som_weights_component = data_mean_component + data_std_component*som_weights_component_normalized
        
        som_weights = som_weights_component.reshape(som_x_dim,-1)
        
        return som_weights
    
    #TODO
    #Display component plot: Data contains rho values for entire data, all z-values
    def display_component_plot_xy(self):
        som_weights = self.compute_component_plot_single_z_slice()
        ax = sns.heatmap(som_weights, square=True)
        ax.invert_yaxis()
    
    #Unnormalized weights
    def get_weights(self, truncation_depth_index=None):
        som = self.som
        
        if truncation_depth_index:
            data_mean = self.data_mean[0:truncation_depth_index]
            data_std = self.data_std[0:truncation_depth_index]
        else:
            data_mean = self.data_mean
            data_std = self.data_std
            
        som_weights = data_mean + data_std*som.get_weights()
        
        return som_weights
    
    #Normalized weight    
    def get_weights_normalized(self, truncation_depth_index=None):
        som = self.som  
        som_weights = som.get_weights()
        
        return som_weights
        
    #Assign gradually varying colours to pixels in the SOM, similar to World Poverty Map SOM colours
    def set_som_colour_scheme(self):
        
        #SOM dimensions
        som_x_dim = self.som_x_dim
        som_y_dim = self.som_y_dim
        
        #Set increment in RGB values which are in [0,1]
        colour_step_x = 1/som_x_dim
        colour_step_y = 1/som_y_dim
        
        #Create RGB image matrix for SOM
        som_rgb_map = np.zeros((som_x_dim, som_y_dim, 3))
        
        #Assign a colour to each pixel
        for i in range(som_x_dim):
            for j in range(som_y_dim):
                som_rgb_map[i,j,0] = i*colour_step_x
                som_rgb_map[i,j,1] = j*colour_step_y
                som_rgb_map[i,j,2] = 1-(i*colour_step_x+j*colour_step_y)/2  
                
        return som_rgb_map
    
    #Display the colours on the SOM
    def display_som_colour_scheme(self):
        som_rgb_map = self.set_som_colour_scheme()
        plt.imshow(som_rgb_map,origin='lower')

    #Colour image in SOM colour scheme, 2D case
    def image_som_colour_scheme(self, N_x=95, N_y=151, truncation_depth_index=None):
        
        som = self.som
        normalized_data = self.normalized_data
        
        if truncation_depth_index:
            normalized_data = normalized_data[:,0:truncation_depth_index]

        winner_coordinates_list = [som.winner(x) for x in normalized_data]
        winner_coordinates = np.array(winner_coordinates_list).T
        
        dataframe = self.dataframe
        df_som = dataframe.copy()
        
        df_som['winner_x'] = winner_coordinates[0]
        df_som['winner_y'] = winner_coordinates[1]
        
        som_rgb_map = self.set_som_colour_scheme()
        
        df_som['som_r'] = som_rgb_map[df_som['winner_x'], df_som['winner_y'], 0]
        df_som['som_g'] = som_rgb_map[df_som['winner_x'], df_som['winner_y'], 1]
        df_som['som_b'] = som_rgb_map[df_som['winner_x'], df_som['winner_y'], 2]
        
        img_som_colours = np.zeros((N_y, N_x, 3))
        
        img_som_colours[:,:,0] = np.reshape(df_som['som_r'].to_numpy(), (N_x, N_y)).T
        img_som_colours[:,:,1] = np.reshape(df_som['som_g'].to_numpy(), (N_x, N_y)).T
        img_som_colours[:,:,2] = np.reshape(df_som['som_b'].to_numpy(), (N_x, N_y)).T

        return img_som_colours

    #Colour image in SOM colour scheme, 3D case
    def image_som_colour_scheme_3d(self, N_x=95, N_y=151, N_z=40):
        
        som = self.som
        normalized_data = self.normalized_data

        winner_coordinates_list = [som.winner(x) for x in normalized_data]
        winner_coordinates = np.array(winner_coordinates_list).T
        
        dataframe = self.dataframe
        df_som = dataframe.copy()
        
        df_som['winner_x'] = winner_coordinates[0]
        df_som['winner_y'] = winner_coordinates[1]
        
        som_rgb_map = self.set_som_colour_scheme()
        
        df_som['som_r'] = som_rgb_map[df_som['winner_x'], df_som['winner_y'], 0]
        df_som['som_g'] = som_rgb_map[df_som['winner_x'], df_som['winner_y'], 1]
        df_som['som_b'] = som_rgb_map[df_som['winner_x'], df_som['winner_y'], 2]
        
        img_som_colours = np.zeros((N_x, N_y, N_z, 3))
        
        img_som_colours[:,:,:,0] = np.reshape(df_som['som_r'].to_numpy(), (N_x, N_y, N_z))
        img_som_colours[:,:,:,1] = np.reshape(df_som['som_g'].to_numpy(), (N_x, N_y, N_z))
        img_som_colours[:,:,:,2] = np.reshape(df_som['som_b'].to_numpy(), (N_x, N_y, N_z))

        return img_som_colours

    #Colour image in SOM colour scheme
    def display_image_som_colour_scheme(self, N_x=95, N_y=151, figsize=None, truncation_depth_index=None):
        img_som_colours = self.image_som_colour_scheme(N_x, N_y, truncation_depth_index)
        plt.imshow(img_som_colours, origin='lower', aspect=N_x/N_y)
        #plt.figure(figsize=figsize)
        #plt.imshow(img_som_colours, origin='lower', aspect=N_x/N_y)
        #return img_som_colours
              
    #Find number of data points corresponding to each pixel in the SOM
    def find_som_pixel_count(self, truncation_depth_index=None):
        
        som = self.som
        normalized_data = self.normalized_data
        
        if truncation_depth_index:
            normalized_data = normalized_data[:,0:truncation_depth_index]
            
        winner_coordinates_list = [som.winner(x) for x in normalized_data] 
        
        counter = dict(Counter(winner_coordinates_list))
        
        #SOM dimensions
        som_x_dim = self.som_x_dim
        som_y_dim = self.som_y_dim
        
        som_histogram = np.zeros((som_x_dim, som_y_dim))
        
        for key, value in counter.items():
            som_histogram[key[0],key[1]] = value
        
        #Add 1 to every pixel to facilitate log scale heatmap display
        som_histogram = som_histogram + 1
        
        return som_histogram
        
    #Display number of data points corresponding to each pixel in the SOM
    def display_som_pixel_count(self, truncation_depth_index=None):
      
        som_histogram = self.find_som_pixel_count(truncation_depth_index)
        
        ax = sns.heatmap(som_histogram, norm=LogNorm(), square=True)
        ax.invert_yaxis()
        
    #TODO
    #Compute quantization error for each data vector
    def compute_quantization_error(self, truncation_depth_index=None):
        
        normalized_data = self.normalized_data

        if truncation_depth_index:
            normalized_data = normalized_data[:,0:truncation_depth_index]
 
        data_mean = self.data_mean[0:truncation_depth_index]
        data_std = self.data_std[0:truncation_depth_index]
        
        #Unnormalized data
        original_data = data_mean + data_std*normalized_data
        quantized_data = self.compute_quantized_image(truncation_depth_index)
 
        quantization_error = np.linalg.norm(original_data-quantized_data,axis=1)
              
        return quantization_error
    
    #TODO
    def image_quantization_error(self, N_x=95, N_y=151, truncation_depth_index=None): 
        
        quantization_error = self.compute_quantization_error(truncation_depth_index)
        print(quantization_error.shape)
        quantization_error_matrix = np.reshape(quantization_error, (N_x, N_y))
        
        ax = sns.heatmap(quantization_error_matrix, norm=LogNorm(), square=True)
        ax.invert_yaxis()
        
        quantization_error = self.compute_quantization_error(N_x, N_y, truncation_depth_index)       

        quantization_error_image = np.zeros((N_x,N_y,truncation_depth_index))
        
        #Map quantized values to original data space dimensions        
        for i in range(truncation_depth_index):            
            quantization_error_image[:,:,i] = np.reshape(quantization_error[:,i],(N_x,N_y))

        return quantization_error_image
    
    def get_win_map(self, truncation_depth_index=None):
        
        som = self.som
        normalized_data = self.normalized_data

        if truncation_depth_index:
            normalized_data = normalized_data[:,0:truncation_depth_index]
            
        win_map = som.win_map(data=normalized_data, return_indices=True)
        
        return win_map
    
#Create dataframe with resistivity stacks along z-axis
def create_dataframe_rho_stack(dataframe):

    data = Data(dataframe=dataframe)
    
    N_x = data.find_num_datapoints('x')
    N_y = data.find_num_datapoints('y')
    N_z = data.find_num_datapoints('z')
    
    #Create list with resistivity stacked along z-axis
    list_rho_stack = []
    
    for i in range(0, N_x*N_y*N_z, N_z):
        rho_stack_list_element = (dataframe['rho'][i:i+N_z]).to_list()
        list_rho_stack.append(rho_stack_list_element)
    dataframe_rho_stack = pd.DataFrame(list_rho_stack)

    return dataframe_rho_stack      

#Create dataframe with resistivity stacks along z-axis at particular depths
def create_dataframe_rho_stack_depths(dataframe, depths=None):
    #depths is depths indices
    data = Data(dataframe=dataframe)
    
    N_x = data.find_num_datapoints('x')
    N_y = data.find_num_datapoints('y')
    N_z = data.find_num_datapoints('z')
    
    #Create list with resistivity stacked along z-axis
    list_rho_stack = []
    
    if depths is None:
        depths = range(N_z)
    
    for i in range(0, N_x*N_y*N_z, N_z):
        rho_stack_list_element = (dataframe['rho'][i:i+len(depths)]).to_list()
        list_rho_stack.append(rho_stack_list_element)
    dataframe_rho_stack = pd.DataFrame(list_rho_stack)

    return dataframe_rho_stack      

#Create dataframe with (x,y,z,rho) vectors
def create_dataframe_x_y_z_rho(dataframe):

    data = Data(dataframe=dataframe)
    
    N_x = data.find_num_datapoints('x')
    N_y = data.find_num_datapoints('y')
    N_z = data.find_num_datapoints('z')
    
    #Create list with resistivity stacked along z-axis
    list_vector = []
    
    for i in range(0, N_x*N_y*N_z):
        rho_vector_element = [dataframe['x'][i], dataframe['y'][i], dataframe['z'][i], dataframe['rho'][i]]
        list_vector.append(rho_vector_element)
    dataframe_vector = pd.DataFrame(list_vector)

    return dataframe_vector   



#TODO
#Create dataframe with resistivity stacks along z-axis, and also (x,y) coordinates
def create_dataframe_rho_stack_xy(dataframe):
    column_names = ['x', 'y', 'rho stack']

    data = Data(dataframe=dataframe)
    
    N_x = data.find_num_datapoints('x')
    N_y = data.find_num_datapoints('y')
    N_z = data.find_num_datapoints('z')
    
    #Create list with resistivity stacked along z-axis
    list_rho_stack = []
    
    for i in range(0, N_x*N_y*N_z, N_z):
        x_value = dataframe['x'][i]
        y_value = dataframe['y'][i]
        rho_stack_list = dataframe['rho'][i:i+N_z].to_numpy()
        #Create list element with x, y, stack of rho values at that (x,y)
        list_element = [x_value, y_value, rho_stack_list]
        list_rho_stack.append(list_element)
     
    #Create dataframe from list elements
    dataframe_rho_stack_xy = pd.DataFrame(list_rho_stack, columns=column_names)
    dataframe_rho_stack_only = dataframe_rho_stack_xy['rho stack']
    dataframe_rho_stack = np.stack(dataframe_rho_stack_only)    
    
    return dataframe_rho_stack
