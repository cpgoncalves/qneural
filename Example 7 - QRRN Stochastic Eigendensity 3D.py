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
# Simulating the iterations of a Quantum Stochastic Neural Map
# for a Quantum Recurrent Neural Network starting from an Eigendensity of
# the Noiseless Unitary Map with 3D plotting and Box-Counting Dimension
# calculation
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
# Part 2 - Preparation of the neural network
# =============================================================================

# Initialize the network to one of the eigendensities of the noiseless map

r=0.99 # base parameter for average

unitaries = get_unitaries(r*pi) # noisless neural operators


Net = qn.initialize_network_eig(num_neurons=2,
                                neural_operators=unitaries,
                                eigenvector_index=2)

# Definition of the neural projectors
P01=sv.tensorProd([P0,P1])
P10=sv.tensorProd([P1,P0])
P11=sv.tensorProd([P1,P1])

# =============================================================================
# Part 3 - Implementation of the quantum stochastic map
# =============================================================================

# Map's parameters

max_it = 1010000 # maximum number of iterations
transient = 10000 # transient
seed_base=3 # seed base
step=10 # step for seed generation
coupling=0.001 # noise coupling level

points = []

P01_av = [] # quantum averages extracted for operator P01
P10_av = [] # quantum averages extracted for operator P10
P11_av = [] # quantum averages extracted for operator P11

# Iterate network extracting the quantum averages
for n in range(0,max_it):
    z = qn.get_seed(seed_base,n,step)
    angle=np.mod(r*2*pi+coupling*z.normal(),2*pi)/2
    neural_operators = get_unitaries(angle)
    NeuralMap = Net.build_quantum_neural_map(neural_operators)
    Net.rho = sv.transformDensity(NeuralMap,Net.rho)
    point = []
    if n >= transient:
        point.append(np.trace(np.dot(Net.rho,P01)).real)
        point.append(np.trace(np.dot(Net.rho,P10)).real)
        point.append(np.trace(np.dot(Net.rho,P11)).real)
        points.append(point)
        


# Plot the results (including the Box Counting Dimension)

points = np.array(points)

qn.calculate_BoxCounting(sequence=points,max_bins=100,cutoff=None)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(points[:,0],points[:,1],points[:,2],c='k',marker='.',s=0.0001)
ax.set_xlabel('<P01>')
ax.set_ylabel('<P10>')
ax.set_zlabel('<P11>')
ax.view_init(70)

plt.show()
