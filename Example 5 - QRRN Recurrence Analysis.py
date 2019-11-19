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
# Part 1 - Preparation
# =============================================================================

# Setup the initial operator to get the initial density
U0 = sv.tensorProd([sv.WHGate(),sv.WHGate()])

# Initialize the network
Net = qn.initialize_network(num_neurons=2,
                            initial_operator=U0,
                            type_initial='Unitary',
                            return_multipliers=False)

print("\nInitial Density")
print(Net.rho)

print("\nLocal Operators")

print("\nProjector 01")
print(Net.projectors[1])

print("\nProjector 10")
print(Net.projectors[2])

print("\nProjector 11")
print(Net.projectors[3])



# Prepare Quantum Circuit

r=0.6

V = np.matrix([[0,-1],[1,0]])

U_r = np.cos(r*pi/2)*sv.unit() + np.sin(r*pi/2)*V

U01 = sv.tensorProd([sv.proj2x2(False),sv.unit()])+sv.tensorProd([sv.proj2x2(True),U_r])
U10 = sv.tensorProd([sv.unit(),sv.proj2x2(False)])+sv.tensorProd([U_r,sv.proj2x2(True)])

NeuralMap = Net.build_quantum_neural_map([U01,U10])

# =============================================================================
# Part 2 - Iterate network
# =============================================================================

# Get the density sequence
densities = Net.iterate_density(map_operator=NeuralMap,
                                T=20000,
                                transient=10000
                                )

averages = []


# Calculate the quantum averages using the density matrix rule
for density in densities:
    P01 = np.trace(np.dot(density,Net.projectors[1]))
    P10 = np.trace(np.dot(density,Net.projectors[2]))
    P11 = np.trace(np.dot(density,Net.projectors[1]))
    averages.append([P01,P10,P11])

averages = np.matrix(averages)

# Get the recurrence plot
S=qn.recurrence_matrix(series=averages, # series of values
                      radius=None, # radius used for plotting
                      type_series=0 # type of series (type 0 is D-dimensional)
                      )


# Plot the quantum averages
fig1, (ax1, ax2, ax3) = plt.subplots(3)
ax1.plot(averages[:,0], c='k',marker='.',ms=0.5,lw=0)
ax2.plot(averages[:,1], c='k',marker='.',ms=0.5,lw=0)
ax3.plot(averages[:,2], c='k',marker='.',ms=0.5,lw=0)
ax3.set_xlabel('Iterations')
ax1.set_ylabel('<P01>')
ax2.set_ylabel('<P10>')
ax3.set_ylabel('<P11>')

qn.recurrence_analysis(S, # distance matrix
                       radius=0.1, # radius to test
                       printout_lines=False, # printout lines with 100% recurrence
                       return_lines=False # return the diagonals with 100% recurrence
                       )
