import svect as sv
import numpy as np
from math import pi
import qneural as qn
from matplotlib import pyplot as plt

# =============================================================================
# Simulating the iterations of a Quantum Neural Map
# for a Quantum Recurrent Neural Network 
# =============================================================================

# =============================================================================
# Part 1 - Preparation
# =============================================================================

P0 = sv.proj2x2(False) # operator |0><0|
P1 = sv.proj2x2(True) # operator |1><1|
I = sv.unit() # operator |0><0|+|1><1|

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
print("\nNeuron 0")
print(Net.local_operators[0])
print("\nNeuron 1")
print(Net.local_operators[1])

# Prepare the Quantum Circuit
r=0.6
V = np.matrix([[0,-1],[1,0]])
U_r = np.cos(r*pi/2)*sv.unit() + np.sin(r*pi/2)*V

U01 = sv.tensorProd([P0,I])+sv.tensorProd([P1,U_r])
U10 = sv.tensorProd([I,P0])+sv.tensorProd([U_r,P1])

NeuralMap = Net.build_quantum_neural_map([U01,U10])

# =============================================================================
# Part 2 - Iterate the map
# =============================================================================

# Get the density sequence
densities = Net.iterate_density(map_operator=NeuralMap,
                                T=20000,
                                transient=10000
                                )

N0_averages = []
N1_averages = []


# Calculate the quantum averages using the density matrix rule
for density in densities:
    N0_averages.append(np.trace(np.dot(density,Net.local_operators[0])))
    N1_averages.append(np.trace(np.dot(density,Net.local_operators[1])))



# Plot the quantum averages for the neural firing operators of each neuron
# N0 = |10><10|+|11><11|, N1 = |01><01|+|11><11|
fig1, (ax1, ax2) = plt.subplots(2)
ax1.plot(N0_averages, c='k',marker='.',ms=0.5,lw=0)
ax2.plot(N1_averages, c='k',marker='.',ms=0.5,lw=0)
ax2.set_xlabel('Iterations')
ax1.set_ylabel('<N0>')
ax2.set_ylabel('<N1>')

# Plot the <N0> versus <N1> values
fig2, ax3 = plt.subplots(1)
ax3.scatter(N0_averages,N1_averages,c='k',marker='.',s=0.01)

# Plot the Histograms for the quantum averages
fig3, ax4 = plt.subplots(1)
ax4.hist(N0_averages,bins=500,density=False, facecolor='k')
ax4.set_title('Histogram <N0>')

fig4, ax5 = plt.subplots(1)
ax5.hist(N1_averages,bins=500,density=False, facecolor='k')
ax5.set_title('Histogram <N1>')
