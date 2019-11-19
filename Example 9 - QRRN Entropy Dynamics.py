# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 10:34:18 2019

@author: cpdsg
"""

import svect as sv
import numpy as np
from math import pi
import qneural as qn
from matplotlib import pyplot as plt

# =============================================================================
# Entropy Dynamics for a Unitary Quantum Neural Map
# =============================================================================


# =============================================================================
# Part 1 - Preparation
# =============================================================================

# Setup the initial operator to get the initial density
U0 = sv.tensorProd([sv.WHGate(),sv.WHGate()])

# Initialize the network
Net, mult_2 = qn.initialize_network(num_neurons=2,
                            initial_operator=U0,
                            type_initial='Unitary',
                            return_multipliers=True)

print("\nInitial Density")
print(Net.rho)

print("\nLocal Operators")

print("\nNeuron 0")
print(Net.local_operators[0])

print("\nNeuron 1")
print(Net.local_operators[1])


# Prepare Quantum Circuit

r=0.001

V = np.matrix([[0,-1],[1,0]])

U_r = np.cos(r*pi/2)*sv.unit() + np.sin(r*pi/2)*V

U01 = sv.tensorProd([sv.proj2x2(False),sv.unit()])+sv.tensorProd([sv.proj2x2(True),U_r])
U10 = sv.tensorProd([sv.unit(),sv.proj2x2(False)])+sv.tensorProd([U_r,sv.proj2x2(True)])

NeuralMap = Net.build_quantum_neural_map([U01,U10])

# =============================================================================
# Part 2 - Iterate the network
# =============================================================================

# Get the density sequence
densities = Net.iterate_density(map_operator=NeuralMap,
                                T=30000,
                                transient=10000
                                )

N0_entropies = []
N1_entropies = []

# Calculate the local entropies
for density in densities:
    Net.rho = density
    N0_entropies.append(Net.calculate_entropy(neuron_index=0,
                                              multipliers_list=mult_2,
                                              print_density=False))
    N1_entropies.append(Net.calculate_entropy(neuron_index=1,
                                              multipliers_list=mult_2,
                                              print_density=False))
    
# Plot the local entropies for each neuron
fig1, (ax1, ax2) = plt.subplots(2)
ax1.plot(N0_entropies, c='k',marker='.',ms=0.25,lw=0.0)
ax2.plot(N1_entropies, c='k',marker='.',ms=0.25,lw=0.0)
ax2.set_xlabel('Iterations')
ax1.set_ylabel('Entropy (Neuron N0)')
ax2.set_ylabel('Entropy (Neuron N1)')

# Plot the Histograms
fig3, ax4 = plt.subplots(1)
ax4.hist(N0_entropies,bins=500,density=False, facecolor='k')
ax4.set_title('Histogram Entropy N0')

fig4, ax5 = plt.subplots(1)
ax5.hist(N1_entropies,bins=500,density=False, facecolor='k')
ax5.set_title('Histogram Entropy N1')

