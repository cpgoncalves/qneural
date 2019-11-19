import svect as sv
import numpy as np

# Main Operators Used in the Circuit:
H = sv.WHGate() # Walsh-Haddamard transform
I = sv.unit() # Unit gate
X = sv.PauliX() # Pauli X
P0 = sv.proj2x2(False) # Projector P0 = |0><0|
P1 = sv.proj2x2(True) # projector P1 = |1><1|

# Circuit Design:
U0 = sv.tensorProd([H,I]) # First transition in circuit
CNOT = sv.tensorProd([P0,I]) + sv.tensorProd([P1,X]) # CNOT gate

# Basis:
basis = sv.basisDef(2) # we are working with a two register basis

# Initial amplitude:
psi0 = np.zeros(len(basis)) 
psi0[0] = 1

# Initial ket:
ket = sv.getKet(basis,psi0)

# Implementing the quatum circuit:
print("\nQUANTUM CIRCUIT SIMULATION")
print("\nInitial ket vector:")
sv.showKet(ket)

print("\nStep 1:")
ket = sv.transformKet(U0,ket)
sv.showKet(ket)

print("\nStep2:")
ket = sv.transformKet(CNOT,ket)
sv.showKet(ket)

