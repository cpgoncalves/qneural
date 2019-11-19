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


# =============================================================================
# Mean Firing Energy vs Mutual Information Plots for a Stochastic Map
# with the initial density set equal to an eigendensity
# =============================================================================


# =============================================================================
# Part 1 - Definition of function needed for the stochastic updade and of
# the Hamiltonian for the mean firing energy calculation
# =============================================================================


V = np.matrix([[0,-1],[1,0]])

P0 = sv.proj2x2(False)
P1 = sv.proj2x2(True)
one = sv.unit()
P01 = sv.tensorProd([P0,P1])
P10 = sv.tensorProd([P1,P0])
P11 = sv.tensorProd([P1,P1])

# Get the unitary operators list for the quantum neural map
def get_unitaries(angle):
    U_angle = np.cos(angle/2)*one + np.sin(angle/2)*V
    U01 = sv.tensorProd([P0,one])+sv.tensorProd([P1,U_angle])
    U10 = sv.tensorProd([one,P0])+sv.tensorProd([U_angle,P1])
    return [U01, U10]

# Get the Hamiltonian for the quantum mean firing energy, the 
# factor is equal to the angular frequency multiplied by the
# reduced Planck constant
def get_hamiltonian(factor):
    return factor*(P01+P10)+2*factor*P11


# =============================================================================
# Part 2 - Preparation of neural network
# =============================================================================

r=0.99 # base parameter for average

unitaries = get_unitaries(r*pi) # noiseless neural operators

# Initialize the network
Net, mult_2 = qn.initialize_network_eig(num_neurons=2,
                                        neural_operators=unitaries,
                                        eigenvector_index=2,
                                        return_multipliers=True
                                        )
print("\nInitial Density")
print(Net.rho)


# =============================================================================
# Part 3 - Implementation of the quantum stochastic map
# =============================================================================

# Map's parameters

max_it = 1010000 # maximum number of iterations
transient=10000 # transient
seed_base=3 # seed base
step=10 # step for seed generation
coupling=0.001 # noise coupling level
h = get_hamiltonian(factor=1) # Hamiltonian for total firing energy


mutual_information = []
mean_energy = []


# Iterate the network extracting the mean firing energy and mutual information
for n in range(0,max_it):
    z = qn.get_seed(seed_base,n,step)
    angle=np.mod(r*2*pi+coupling*z.normal(),2*pi)/2
    neural_operators = get_unitaries(angle)
    NeuralMap = Net.build_quantum_neural_map(neural_operators)
    Net.rho = sv.transformDensity(NeuralMap,Net.rho)
    if n >= transient:
        entropy_0 = Net.calculate_entropy(neuron_index=0,
                                              multipliers_list=mult_2,
                                              print_density=False)
        entropy_1 = Net.calculate_entropy(neuron_index=1,
                                              multipliers_list=mult_2,
                                              print_density=False)
        mutual_information.append(entropy_0+entropy_1)
        
        mean_energy.append(np.trace(np.dot(Net.rho,h)).real)
        

# Plot the energy versus mutual information results
    
fig, ax = plt.subplots(1)
ax.scatter(mean_energy,mutual_information,c='k',marker='.',s=0.0001)
ax.set_xlabel('Average Firing Energy')
ax.set_ylabel('Mutual Information')
