# -*- coding: utf-8 -*-
"""
@author: Carlos Pedro Gonçalves, University of Lisbon
"""

import svect
import numpy as np
from numpy import linalg as la
from scipy import stats
from matplotlib import pyplot as plt
from numpy.matlib import repmat
import pandas as pd
from math import pi

h_bar = (6.62607015*(10**(-34)))/(2*pi) # the reduced Planck's constant


# =============================================================================
# Class QNet and Methods
# =============================================================================

class QNet:
    
    def __init__(self,num_neurons,rho,projectors,local_operators,multipliers):
      self.num_neurons = num_neurons # number of neurons in neural network
      self.rho = rho # density operator
      self.projectors = projectors # projectors for the neural firing basis
      self.local_operators = local_operators # local neural firing operators
      self.multipliers = multipliers # multipliers list for local density
    
    
    def local_multipliers_list(self):
        # Method to return the multipliers list used for
        # local (reduced) density calculation
        M = svect.genProj(1) # projectors on two dimensional Hilbert space
        M_00 = M[0] # component |0><0|
        M_11 = M[1] # component |1><1|
        M_01 = np.dot(svect.PauliX(),M_11) # component |0><1|
        M_10 = np.dot(svect.PauliX(),M_00) # component |1><0|
        # Return the multipliers on the single neuron Hilbert space as a list
        return [M_00,M_01,M_10,M_11]
    
    def generate_multipliers_neuron(self,multipliers_list,neuron_index):
        # Method to return extended multipliers list for a given neuron
        # used for the reduced density operations
        
        # 1. Define a fixed tuple of unit vectors
        ones = tuple([svect.unit()]*self.num_neurons)
        
        # 2. Define the extended multipliers list
        M = []
        
        # 3. Get the extended multipliers per neuron
        for multiplier in multipliers_list:
            # Change at the neuron's position for the multiplier
            a = list(ones)
            a[neuron_index]=multiplier
            # Get the tensor product for the extension to the firing
            # pattern basis
            multiplier_extended = svect.tensorProd(a)
            # Append the multiplier to the multipliers list
            M.append(multiplier_extended)
        
        # 4. Return extended multipliers list on the N neurons Hilbert space
        return M
                  
    def setup_basis(self,
                    multipliers_list=None,
                    return_ket=True,
                    get_multipliers=False):
        
        # Method for generating firing pattern basis.
        # NOTE: the method also returns the ket vector where 
        # all neurons are nonfiring if asked for
        
        # 1. Get the basis
        basis = svect.basisDef(self.num_neurons)
        dim = 2**self.num_neurons # dimension is 2^number of neurons
        
        # 2. Update the neural firing pattern projectors
        self.projectors = svect.genProj(self.num_neurons)
        
        # 3. Get the local operators for each neuron's neural firing event
        # and the local multipliers for the reduced densities calculations
        
        # Define the local operators list
        self.local_operators = []
        
        # Define the multipliers list
        if get_multipliers == True:
            self.multipliers = [] 
                
        # For each neuron
        for neuron in range(0,self.num_neurons):
            # Initialize the corresponding operator as a matrix of zeros of
            # rank equal to the corresponding dimension
            operator = np.matrix(np.zeros((dim,dim)))
            # For each basis element
            for i in range(0,len(basis)):
                # Get the firing pattern
                firing_pattern = basis[i][1:-1]
                # If the neuron is firing in the basis element
                if firing_pattern[neuron] == '1':
                    # Add the corresponding projector
                    operator += self.projectors[i]
            # Append the operator to the local operators
            self.local_operators.append(operator)
            # Get the extended multipliers list
            if get_multipliers == True:
                M = self.generate_multipliers_neuron(multipliers_list,neuron)
                self.multipliers.append(M) 
        
        # 4. Return the ket
        if return_ket == True:
            # Get the initial nonfiring ket vector
            amplitudes = [0]*dim # all amplitudes are zero...
            amplitudes[0] = 1 #... except for the first one
            
            # Get the ket and print the result 
            # (as expected should be |000...0>)
            ket = svect.getKet(basis,amplitudes)
            return ket
    
    def transform_ket(self,Unitary,ket,density=True,printouts=True):
        # Method for transforming the initial ket vector, 
        # returning the initial density (in case of pure density). 
        # In case of mixed density the user can build it using
        #  the functionalities of the "svect" library
        
        # 1. Transform the ket using the initial unitary preparation matrix
        ket = svect.transformKet(Unitary,ket)
        
        # 2. Print the prepared ket vector if asked for
        if printouts == True:
            print("\nPrepared ket vector")
            svect.showKet(ket)
        
        # 3. Update the density operator for the network if asked for
        if density == True:
            self.rho = svect.density(ket)
    
    
    def local_density(self,neuron_index,multipliers_list):
        # Method for getting the local density using the partial trace for
        # any given neuron, the method requires as inputs
        # the neuron index (0,1,...,N-1) and the local multipliers list
        
        # 1. Get the multipliers for the neuron extended on the basis
        M = self.multipliers[neuron_index]
        
        # 2. Get the density entries
        rho_00 = np.trace(np.dot(M[0],self.rho))*multipliers_list[0]
        rho_01 = np.trace(np.dot(M[1],self.rho))*multipliers_list[1]
        rho_10 = np.trace(np.dot(M[2],self.rho))*multipliers_list[2]
        rho_11 = np.trace(np.dot(M[3],self.rho))*multipliers_list[3]
        
        # 3. Return the local (reduced) density for the chosen neuron
        return rho_00+rho_01+rho_10+rho_11
        
    def calculate_entropy(self,neuron_index,multipliers_list,print_density=False):
        # Method for calculating the Von Neumann Entropy
        
        # 1. Get the local (reduced) density
        local_rho = self.local_density(neuron_index,multipliers_list)
        
        # 2. Print the results if asked for
        if print_density == True:
            print("\nLocal Density\n")
            print(local_rho)
        
        # 2. Return the entropy
        return svect.entropy(local_rho,printouts=False)  
    
    
    def build_quantum_neural_map(self,operators):
        # Method to build the quantum neural map operator, the operator
        # product is performed from the beginning to the end, right to left
        
        # 1. Extract the first operator in the list for the map
        map_operator = operators[0]
        
        # 2. Update the operator by premultiplication
        
        # For each remaining operator
        for i in range(1,len(operators)):
            # Update the map applying dot product multiplication
            # we have a premultiplication of the map with each operator
            # in the list
            map_operator = np.dot(operators[i],map_operator)
            
        # Return the map's operator
        return map_operator
    
    def get_eigenvectors(self,map_operator, printout=False):    
        # Method to extract a quantum neural map's eigenvalues 
        # and eigenvectors and print them
        eigenvalues, eigenvectors = la.eig(map_operator)
        if printout == True:
            for i in range(0,len(eigenvalues)):
                print("\nEigenvalue\n")
                print(eigenvalues[i])
                print("\nEigenvector\n")
                print(eigenvectors[:,i])
        return eigenvalues, eigenvectors
    
    def iterate_ket(self,map_operator,ket,T,transient):
        # Method that iterates a quantum neural network
        # with a neural map applied to a sequence of ket vectors
        
        kets = [] # sequence of ket vectors is stored in a list
        
        # Iterate the network
        for t in range(0,T):
            ket=svect.transformKet(map_operator,ket)
            if t >= transient:
                kets.append(ket[1])
        
        # Return the sequence of ket vectors
        return kets
       
    def iterate_density(self,map_operator,T,transient):
        # Method that iterates a quantum neural network
        # with a neural map applied to a sequence of density operators
        
        densities=[] # sequence of density operators is stored in a list
        
        # Iterate the network
        for t in range(0,T):
            self.rho = svect.transformDensity(map_operator,self.rho)
            if t >= transient:
                densities.append(self.rho)
        
        # Return the sequence of densities
        return densities
    
    def iterate(self,map_operator,T,
                multipliers_list=None,entropy=False,density=False):
        
        # Method to iterate a quantum neural network applying
        # a unitary map and calculating the quantum averages if
        # asked for
        
        # 1. Initialize the relevant lists
               
        # If the sequence is for the density operators
        if density == True:
            # Initialize the density operators
            densities = []
        # Otherwise the sequence is for the calculation of the 
        # quantum averages for the local neural firing operators
        else:
            # Initialize the quantum averages list
            quantum_averages = []
            # If the entropy is to be calculated for the reduced
            # local densities
            if entropy == True:
                # Initialize the entropies list
                entropies = []
        
        # 2. Iterate the network
                      
        # For each iteration of the quantum neural map
        for t in range(0,T):
            
            # Update the density
            self.rho = svect.transformDensity(map_operator,self.rho)
            
            # If we want the density as the final result then append
            # the neural network's density to the densities list
            if density == True:
                densities.append(self.rho)
                
            # Otherwise the quantum averages (and entropies are extracted)
            else:
                if entropy == True:
                    new_point,entropy_values=self.extract_averages(multipliers_list,
                                                                    entropy)
                    quantum_averages.append(new_point)
                    entropies.append(entropy_values)
                else:
                    new_point=self.extract_averages(multipliers_list,
                                                    entropy)
                    quantum_averages.append(new_point)
                    
        # 3. Return the relevant lists for further processing
        if density == True:
            return densities
        else:
            if entropy == True:
                return np.array(quantum_averages), np.matrix(entropies)
            else:
                return np.array(quantum_averages)
    
    def extract_averages(self,multipliers_list,entropy=True):
        # Method to extract the quantum averages
        
        # 1. Initialize the relevant lists
        data_point = []
        entropy_values = []
        
        # 2. Extract the quantum averages
        for i in range(0,self.num_neurons):
            quantum_average = np.trace(np.dot(self.rho,self.local_operators[i]))
            # Append the quantum averages to the data point
            data_point.append(quantum_average)
            # Calculate the entropy if asked for
            if entropy == True:
                entropy_values.append(self.calculate_entropy(i,multipliers_list))
        # 3. Return results
        if entropy == True:
            return data_point, entropy_values
        else:
            return data_point
    
 
# =============================================================================
# Main Functions for Instantiation, Preparation and Analysis
# =============================================================================


def initialize_network(num_neurons,initial_operator,
                       type_initial='Unitary',
                       return_multipliers=False):
    # Function to instantiate and initialize a quantum neural network
    
    # 1. Build the neural network
    Net = QNet(num_neurons,
               rho=None,
               projectors=None,
               local_operators=None,
               multipliers=None)
    
    # 2. Get the multipliers for the local density calculation
    # if needed
    if return_multipliers == True:
        multipliers_list=Net.local_multipliers_list()
    
    # 3. Setup the basis, initial projectors, multipliers and local operators
    if return_multipliers == False:
        ket=Net.setup_basis()
    else:
        ket=Net.setup_basis(multipliers_list, get_multipliers=True)

    # 4. Setup the initial neural network configuration
    if type_initial == 'Unitary':
        Net.transform_ket(initial_operator,ket,density=True,printouts=True)
    elif type_initial == 'Density':
        Net.rho = initial_operator
    else:
        print("Error! type_initial must be either 'Unitary' or 'Density'")
    
    # 5. Return the neural network and multipliers list if asked
    # otherwised just return the neural network object
    if return_multipliers == True:
        return Net, multipliers_list
    else:
        return Net


def get_eigenvectors(num_neurons, # number of neurons
                     neural_operators # neural operators list
                     ):
    # Function to get eigenvectors and perform eigenvalue analysis
    # on a quantum neural map (to be used in tandem with the next
    # function)
    
    # 1. Build the neural network
    Net = QNet(num_neurons,
               rho=None,
               projectors=None,
               local_operators=None,
               multipliers=None)
    
    # 2. Get the multipliers for the local density calculation
    multipliers_list=Net.local_multipliers_list()
    
    # 3. Setup the basis, initial projectors, multipliers and local operators
    Net.setup_basis(multipliers_list,return_ket=False,get_multipliers=True)
        
    # 4. Build the neural map
    NeuralMap = Net.build_quantum_neural_map(neural_operators)
    
    # 5. Get the neural map's eigenvalues and eigenvectors
    eigenvalues, eigenvectors = Net.get_eigenvectors(NeuralMap)
    
    # 6. Print the eigenvalues, phases and eigenvector analysis
    for i in range(0,len(eigenvalues)):
        phase = -np.angle(eigenvalues[i])
        print("\nEigenvalue:", eigenvalues[i]) # print the eigenvalue
        print("Phase:", phase) # print the map's eigenphase      
        print("\nEigenvector", eigenvectors[:,i])
        
    # 7. Return the neural network, multipliers list and eigenvectors
    return Net, multipliers_list, eigenvectors


def initialize_network_eig(num_neurons, # number of neurons in the network
                           neural_operators, # neural operators list
                           eigenvector_index, # eigenvector index
                           return_multipliers=False # return multipliers
                           ):
    
    # Function to instantiate and initialize a quantum neural network
    # with a density given by one of the neural map's eigenvectors
    
    # 1. Perform the eigenvalue analysis
    print("\nEIGENVALUE ANALYSIS")
    
    Net, multipliers_list, eigenvectors = get_eigenvectors(num_neurons, 
                                                           neural_operators)
    
    # 2. Setup the initial density to the corresponding eigenvector
    print("\n\nCHOSEN EIGENVECTOR:\n")
    print(eigenvectors[:,eigenvector_index])
    Net.rho = svect.density_vect(eigenvectors[:,eigenvector_index])
    
    # 3. Perform the entropy analysis
    for i in range(0,Net.num_neurons):
        print("\n\nNeuron"+" "+str(i))
        S=Net.calculate_entropy(i,multipliers_list,print_density=True)
        print("\nEntropy", S)
    
    # 4. Return the network, also return the multipliers list if asked for
    if return_multipliers == False:
        return Net
    else:
        return Net, multipliers_list


def recurrence_matrix(series, # series of values
                      radius=None, # radius used for plotting
                      type_series=1 # type of series
                      ):
    # Function to extract the distance matrix and the recurrence plot
    # either for a signal or for an n-dimensional sequence of points
        
    # If the series of observations is a 1D signal
    if type_series==1:
        # Get the series length
        N = len(series)
        # If the series is in list type
        if type(series) == list:
            # Convert to a column vector
            series=np.matrix(series).T
        # Initialize the recurrence matrix to an N by N zeros matrix
        S=np.matrix(np.zeros(shape=(N,N)))
        # For each observation
        for i in range(0,N):
            # The i-th column of S is comprised of the distance
            # between the i-th observation and each other observation
            # in the series
            S[:,i]=abs(repmat(series[i],1,N).T-series)
    # If the series of observations is a d-dimensional sequence
    elif type_series==0:
        # Get the number of lines (number of observations)
        N = np.size(series,0)
        # Get the number of columns (number of dimensions)
        dim = np.size(series,1)
        # Initialize the distance matrix as above
        S=np.matrix(np.zeros(shape=(N,N)))
        # For each dimension
        for d in range(0,dim):
            # Extract the corresponding column
            series_d=series[:,d]
            # The i-th column of S is comprised of the sum of the squares
            # of the difference between each observation in the series and
            # the corresponding d-th column
            for i in range(0,N):
                S[:,i]+=np.power(repmat(series_d[i][0],1,N).T-series[:,d],2)
        # Take the square root to get the Euclidean distances
        S = np.sqrt(S)
    
    # If the radius is provided
    if radius != None:
        # Build the binary recurrence matrix
        # so that each entry is 1 (white), if it surpases the radius,
        # and 0 (black) if not
        B = S > radius
        B = 1*B
        
    # If the radius is not provided
    if radius==None:
        # Plot the colored recurrence plot
        plt.imshow(S)
    # Otherwise
    else:
        # Plot the black and white recurrence plot
        plt.imshow(B,cmap=plt.cm.gray)
    plt.show()
    
    # Return the distance matrix
    return S

def recurrence_analysis(S, # distance matrix
                        radius, # radius to test
                        printout_lines=False, # printout lines with 100% recurrence
                        return_lines=False # return the diagonals with 100% recurrence
                        ):
    # Function to get recurrence statistical analysis
    
    # Get the recurrence matrix for the given radius
    B = S <= radius
    B = 1*B
    
    # Number of diagonals
    num_diagonals = 0
    
    # Diagonals with 100% recurrence
    diagonals_full = []
    
    # Number of diagonals with recurrence
    recurrence_total = 0
        
    # Recurrence strength (will have the sum of fills)
    recurrence_strength = 0
    
    # While the size of B is different from 1
    while B.size != 1:
        # Add one more to the number of diagonals
        num_diagonals += 1
        # Delete the last row and column to get the 
        # new main diagonal (the first parallel to the previous main diagonal)
        B = np.delete(B,(-1),1)
        B = np.delete(B,(0),0)
        # Get the trace to get how many recurrence points are in the line
        recurrence = np.trace(B)
        # Get the size of the diagonal
        size_diagonal = np.size(B,0)
        
        # If there are recurrence points in the line
        if recurrence > 0:
            # Add to the total number of lines with recurrence
            recurrence_total += 1
            # If all points in the diagonal are recurrence points
            if recurrence == size_diagonal:
                # Add the diagonal rank to the diagonals list
                diagonals_full.append(num_diagonals)
            # Get the recurrence proportion
            # (proportion of the line that has recurrence points)
            recurrence_strength += recurrence/size_diagonal
    
    
    # Get the number of diagonals with 100% recurrence
    num_recurrence_100 = len(diagonals_full)
    
    # If there are lines with 100% recurrence, calculate the crosstabs for
    # the distances between diagonals with 100% recurrence
    if num_recurrence_100 != 0:
        distances=[] # distances between the lines with 100% recurrence
        if printout_lines == True:
            print("\nLines with 100% recurrence:\n")    
        for i in range(0,num_recurrence_100):
            if printout_lines == True:
                # Print each line with 100% recurrence
                print("\nLine", diagonals_full[i])
            if i > 0:
                distances.append(diagonals_full[i]-diagonals_full[i-1])
        distances = pd.DataFrame(distances,columns=["Distances"])
        print(pd.crosstab(index=distances["Distances"],columns="count"))
        
    # Print other stats
    print("\nNumber of lines with 100% recurrence:", num_recurrence_100)
    print("\nNumber of lines with recurrence:", recurrence_total)
    print("\nTotal number of diagonals", num_diagonals)
    print("\nProportion of diagonals with recurrence", recurrence_total/num_diagonals)
    
    # If there are recurrence points print the additional stats
    if recurrence_total > 0:
        print("\nAverage recurrence strength:", recurrence_strength / recurrence_total)
        print("\nP[100% recurrence|recurrence]", num_recurrence_100/recurrence_total)
    
    # If one wishes to get the diagonals return the diagonals
    if return_lines==True:
        return diagonals_full

def calculate_BoxCounting(sequence,max_bins,cutoff):
    # Calculate Box Counting dimension for N-dimensional sequence
    # the algorithm is adapted from 
    # https://francescoturci.net/2016/03/31/box-counting-in-numpy/
    
    Bins=list(range(1,max_bins+1)) # number of bins
    Ns=[] # number of boxes
    num_cols = np.size(sequence,1) # number of columns
        
    # For each bin
    for b in Bins:
        # Compute the Histogram
        bins_values = tuple([b]*num_cols)
        H, edges=np.histogramdd(sequence, 
                                bins=bins_values)
        # Sum the number of boxes
        Ns.append(np.sum(H>0))
    
    # Logarithms of the bins and number of boxes
    log_Bins = np.log(Bins)
    log_Ns = np.log(Ns)
    
    # If no cutoff is used, the data sample includes all elements
    if cutoff == None:
        sample_log_Bins = np.array(log_Bins)
        sample_log_Ns = np.array(log_Ns)
    # Otherwise the regression is performed on a subsample for the defined
    # cutoff region
    else:
        f = [item for item in zip(log_Bins,log_Ns) if cutoff[0] < item[1] < cutoff[1]]
        sample_log_Bins, sample_log_Ns = zip(*f)
        sample_log_Bins = list(sample_log_Bins)
        sample_log_Ns = list(sample_log_Ns)
            
    # Perform the regression
    slope, intercept, r_value, p_value, std_err = stats.linregress(sample_log_Bins,
                                                                   sample_log_Ns)
    
    # Get the predictions
    log_Pred = slope * log_Bins + intercept
    
    # Print the results
    print("\nBox Counting Dimension")
    print("\nR^2:", r_value**2)
    print("R:", r_value)
    print("\nIntercept:", intercept)
    print("Dimension:", slope)
    print("p-value of slope:", round(p_value,6))
    
    # Plot the results
    plt.plot(log_Bins,log_Ns, '.',c='k', mfc='none')
    plt.plot(log_Bins,log_Pred)
    plt.xlabel('log 1/s')
    plt.ylabel('log Ns')

def get_seed(seed_base,n,step):
    # Procedure to extract a seed in an iteration for a stochastic network
    return np.random.RandomState(seed=seed_base + n * step)   




        
        

