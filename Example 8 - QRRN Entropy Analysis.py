# -*- coding: utf-8 -*-
"""
Created on Thu Aug  1 18:56:24 2019

@author: cpdsg
"""

import svect as sv
import numpy as np
from math import pi
import qneural as qn
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

# =============================================================================
# Eigenphases and Entropy Plots for a Unitary Quantum Neural Map with 
# varying Parameter
# =============================================================================


# =============================================================================
# Part 1 - Definition of the function for the update
# =============================================================================

V = np.matrix([[0,-1],[1,0]])

P0 = sv.proj2x2(False)
P1 = sv.proj2x2(True)
one = sv.unit()

def get_unitaries(angle):
    U_angle = np.cos(angle/2)*one + np.sin(angle/2)*V
    U01 = sv.tensorProd([P0,one])+sv.tensorProd([P1,U_angle])
    U10 = sv.tensorProd([one,P0])+sv.tensorProd([U_angle,P1])
    return [U01, U10]


# =============================================================================
# Part 2 - Preparation of the neural network
# =============================================================================

# Initialize the network to one of the eigenvectors

Net, mult_2 = qn.initialize_network(num_neurons=2,
                            initial_operator=sv.tensorProd([P0,P0]),
                            type_initial='Density',
                            return_multipliers=True)


# =============================================================================
# Part 3 - Eigenvalue and entropy analysis
# =============================================================================

step = 0.001
r=0

neural_map = Net.build_quantum_neural_map(get_unitaries(r*pi))
eigenvalues, eigenvectors = Net.get_eigenvectors(neural_map,printout=False)



r_values=[]
phases_values=[]
entropy_N0_values=[]
entropy_N1_values=[]

values_ind=[0,2,3]

while r <= 1:
    r += step
    neural_map = Net.build_quantum_neural_map(get_unitaries(r*pi))
    eigenvalues, eigenvectors = Net.get_eigenvectors(neural_map,
                                                     printout=False)
    # Eigenphases results
    phases = []
    for eigenvalue in eigenvalues:
        phases.append(-np.angle(eigenvalue))
    phases=list(set(phases))
    phases.sort()
    
    # Entropy results
    entropy_values_N0 = []
    entropy_values_N1 = []
    
    # Extract the local entropies
    num_columns = np.size(eigenvectors,1)
    entropies_0 = []
    entropies_1 = []

    
    for i in values_ind:
        # Get the density for the i-th eigenvector
        Net.rho = sv.density_vect(eigenvectors[:,i])
                
        # Extract the local entropies
        entropies_0.append(Net.calculate_entropy(neuron_index=0,
                                                 multipliers_list=mult_2,
                                                 print_density=False))
        entropies_1.append(Net.calculate_entropy(neuron_index=1,
                                                 multipliers_list=mult_2,
                                                 print_density=False))
    
    entropy_N0_values.append(entropies_0)
    entropy_N1_values.append(entropies_1)
  
    r_values.append(r)
    phases_values.append(phases)
    

phases_values = np.array(phases_values)
entropy_N0_values = np.array(entropy_N0_values)
entropy_N1_values = np.array(entropy_N1_values)

# Plot the eigenphases
fig1, ax1 = plt.subplots(1)
ax1.plot(r_values, phases_values[:,0], c='k',lw=1.0)
ax1.plot(r_values, phases_values[:,2], c='r',lw=1.0)
ax1.plot(r_values, phases_values[:,1], c='g',lw=1.0)
ax1.legend(['Phase 1','Phase 2','Phase 3'])

# Plot the entropies

#...for neuron N0
fig2, ax2 = plt.subplots(1)
ax2.plot(r_values, entropy_N0_values[:,0], c='k',lw=1.0)
ax2.plot(r_values, entropy_N0_values[:,1], c='r',lw=1.0)
ax2.plot(r_values, entropy_N0_values[:,2], c='g',lw=1.0)
ax2.legend(['Entropy 1','Entropy 2','Entropy 3'])

#...for neuron N1
fig3, ax3 = plt.subplots(1)
ax3.plot(r_values, entropy_N1_values[:,0], c='k',lw=1.0)
ax3.plot(r_values, entropy_N1_values[:,1], c='r',lw=1.0)
ax3.plot(r_values, entropy_N1_values[:,2], c='g',lw=1.0)
ax3.legend(['Entropy 1','Entropy 2','Entropy 3'])

