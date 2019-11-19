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
# Simulating the iterations of a Quantum Stochastic Neural Map
# for a Quantum Recurrent Neural Network starting from an Eigendensity of
# the Noiseless Unitary Map
# =============================================================================


# =============================================================================
# Part 1 - Definition of function needed for the stochastic updade
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
# Part 2 - Preparation of neural network
# =============================================================================

# Initialize the network to one of the eigendensities of the noiseless map

r=0.99 # base parameter for average

unitaries = get_unitaries(r*pi) # noisless neural operators


Net = qn.initialize_network_eig(num_neurons=2,
                                neural_operators=unitaries,
                                eigenvector_index=0)



# =============================================================================
# Part 3 - Implementation of the quantum stochastic map
# =============================================================================

# Map's parameters

max_it = 1010000 # maximum number of iterations
transient = 10000 # transient
seed_base=3 # seed base
step=10 # step for seed generation
coupling=0.004 # noise coupling level

N0_averages = [] # quantum averages extracted for neuron N0
N1_averages = [] # quantum averages extracted for neuron N1

# Iterate network extracting the quantum averages
for n in range(0,max_it):
    z = qn.get_seed(seed_base,n,step)
    angle=np.mod(r*2*pi+coupling*z.normal(),2*pi)/2
    neural_operators = get_unitaries(angle)
    NeuralMap = Net.build_quantum_neural_map(neural_operators)
    Net.rho = sv.transformDensity(NeuralMap,Net.rho)
    if n >= transient:
        N0_averages.append(np.trace(np.dot(Net.rho,Net.local_operators[0])))
        N1_averages.append(np.trace(np.dot(Net.rho,Net.local_operators[1])))


# Plot the results (including the spectral analysis)
    
fig, ax = plt.subplots(1)
ax.scatter(N0_averages,N1_averages,c='k',marker='.',s=0.0001)
ax.set_xlabel('<N0>')
ax.set_ylabel('<N1>')

ps1 = np.abs(np.fft.fft(N0_averages))**2
ps2 = np.abs(np.fft.fft(N1_averages))**2
time_step = 1

freqs1 = np.fft.fftfreq(len(N0_averages), time_step)
freqs2 = np.fft.fftfreq(len(N1_averages), time_step)
idx1 = np.argsort(freqs1)
idx2 = np.argsort(freqs2)

fig2, ax2 = plt.subplots(1)
ax2.loglog(freqs1[idx1],ps1[idx1],c='k')
ax2.set_xlabel('Frequency')
ax2.set_ylabel('Power')


fig3, ax3 = plt.subplots(1)
ax3.loglog(freqs2[idx2],ps2[idx2],c='k')
ax3.set_xlabel('Frequency')
ax3.set_ylabel('Power')

