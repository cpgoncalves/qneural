# -*- coding: utf-8 -*-
"""
@author: Carlos Pedro Gonçalves, University of Lisbon
"""

import numpy as np
import scipy.linalg as lalg

# =============================================================================
# svect module for  basic quantum operations using Dirac's bra-ket notation
# and NumPy's matrices
# =============================================================================

# =============================================================================
# GENERAL FUNCTIONS FOR BASIS DEFINITION AND PROJECTORS
# =============================================================================

def generateStrings(size):
    # Function to generate the bitstrings of size 2^N
    # used as auxiliary function in the basis definition
    
    string_list = [] # list that will hold the binary strings
    num_strings = 2**size # the number of strings is a power of 2
    
    # Append the strings in sequence to the string list
    for i in range(0,num_strings):
        string_list.append(np.binary_repr(i,width=size))
    
    # Return the binary strings list
    return string_list

def basisDef(size):
    # Definition of the basis in terms of symbolic representation
    # using Dirac's bra-ket notation
    
    # Generate the binary strings
    basis = generateStrings(size) 
    
    # Get Dirac's bra-ket notation
    for i in range(0,len(basis)):
        basis[i] = '|' + basis[i] + '>'
    
    # Return the basis list in Dirac's bra-ket notation
    return basis 

        
def getKet(basis,amplitudes):
    # Function to return ket vector structure as comprised
    # of a basis structure and amplitudes, takes the basis in 
    # Dirac's notation and the amplitudes list as inputs and
    # outputs a list comprised of the basis and the vector
   
    # Define the vector in the computational basis representation as a column
    # vector from the amplitudes
    vector = np.matrix(amplitudes)
    vector = vector.getT()
   
    # Return the ket vector structure as a list comprised of the basis
    # and the ket column vector
    return [basis,vector]

def showKet(ket):
    
    # Function that prints the ket vector on the Python console
    # using Dirac's bra-ket notation
   
    basis = ket[0] # the basis occupies position 0 in the ket vector structure
    vector = ket[1] # the column vector occupies position 1 in the ket vector structure
    lv = len(vector) # the length of the vector
   
    ket_vector = "|psi> = " # the ket vector in symbolic representation
   
    # Represent the ket vector in Dirac's notation
    for i in range(0,lv):
        amplitude = np.asscalar(vector[i])
        if amplitude != 0:
            ket_vector = ket_vector + str(amplitude) + basis[i] + '+'
    cutPoint = len(ket_vector) - 1
    ket_vector = ket_vector[:cutPoint]
   
    # Print the ket vector in Dirac's notation
    print(ket_vector)


def density(ket):
    # Function that returns the density matrix from a ket vector structure
   
    psi = ket[1] # get the column matrix for the ket
    psi_T = psi.getH() # get the bra as the conjugate transpose
   
    # Return the matrix as |psi><psi| using NumPy's dot product
    return np.matrix(np.dot(psi,psi_T))

def density_vect(vector):
    # Procedure that returns the density matrix directly
    # from a column vector (if one does not wish to work 
    # with Dirac's notation)
    return np.matrix(np.dot(vector,vector.getH()))

# =============================================================================
# GENERAL UNITARY GATES' FUNCTIONS
# =============================================================================

def PauliX():
    # Function that returns Pauli's X matrix |0><1| + |1><0|
    return np.matrix([[0,1],[1,0]])

def PauliY():
    # Function that returns Pauli's Y matrix -i|0><1| + i|1><0|
    return np.matrix([[0,-1j],[1j,0]])

def PauliZ():
    # Function that returns Pauli's Z matrix |0><0| - |1><1|
    return np.matrix([[1,0],[0,-1]])

def unit():
    # Function that returns the unit operator I on single qubit Hilbert Space
    # I = |0><0| + |1><1|
    return np.matrix([[1,0],[0,1]])

def WHGate():
    # Function that returns the Walsh-Haddamard gate
    return (1 / np.sqrt(2)) * (PauliZ() + PauliX())

def PShift(phi):
    # Function that returns the phase shift gate
    return np.matrix([[1,0],[0,np.exp(1j * phi)]])
 
def gate2x2(omega, # angle for phase
            theta, # angle for rotation
            n, # unit vector
            delta_t, # computation's time interval 
            delta_to # basic processing time
            ):
    # Function that returns the general 2x2 unitary gate
    # from the Hamiltonian parameters that define the phase
    # and rotation parts of the unitary gate
   
    # Get the main components of the U(2) gate
    angle1 = (omega * delta_t) / (2 * delta_to) # angle for phase transform
    angle2 = (theta * delta_t) / (2 * delta_to) # angle for rotation component
    phaseTransform = np.exp(1j * angle1) # phase transform
    ns = n[0] * PauliX() + n[1] * PauliY() + n[2] * PauliZ() # product for spin direction
    rotation = np.cos(angle2) * unit() - 1j * np.sin(angle2) * ns # rotation
   
    # Return U(2) gate
    return phaseTransform * rotation

def tensorProd(operators_list):
    # Function that returns the operator that results from
    # the tensor product of operators from a list
    operator = operators_list[0] # first operator in the list
    for i in range(1,len(operators_list)):
        # apply the tensor product
        operator = np.matrix(np.kron(operator,operators_list[i]))
    # Return the operator
    return operator


def transformKet(operator,ket):
    # Function for the unitary transformation of a ket vector   
    # returns the new ket with the usual ket structure as a list
    # comprised by the basis and the transformed amplitudes
    return [ket[0],np.dot(operator,ket[1])]


def transformDensity(operator,density):
    # Function for the unitary transformation of a density matrix
    
    # Get the conjugate transpose of the main operator
    operatorT = operator.getH()
    
    # Apply the operator on the left
    density = np.dot(operator,density)
    # Apply the conjugate transpose on the right
    density = np.dot(density,operatorT)
    
    # Return the density
    return density


# =============================================================================
# PROJECTORS AND QUANTUM AVERAGES
# =============================================================================

def proj2x2(value):
    # Function that returns the projectors |0><0| and |0><1|
    # if the argument is set to False it returns the projector |0><0|
    # if the argument is set to True it returns the projector |1><1|
    # NOTE: the function only accepts a Boolean argument
    
    if type(value) != bool:
        print("Function argument must be Boolean!")
    else:
        if value == False:
            a0 = np.matrix([1,0]) # vector <0|
            b0 = a0.getT() # vector |0>
            return np.matrix(np.outer(b0,a0)) # return projector |0><0|
        else:
            a1 = np.matrix([0,1]) # vector <1|
            b1 = a1.getT() # vector |1>
            return np.matrix(np.outer(b1,a1)) # return projector |1><1|
         
                        
def genProj(size):
    
    # Function that returns a projectors list for each element of
    # a computational basis of size N
   
    projList = [] # projectors list
   
    strings = generateStrings(size) # binary strings for the required size
   
    # For each string:
    for element in strings:
        # Setup the new projector list for applying the tensor product
        newProj = []
        # For each symbol in the element...
        for symbol in element:
            # ...if the symbol is 0...
            if symbol == '0':
                # ...append the projector |0><0|...
                newProj.append(proj2x2(False))
            #...otherwise...
            else:
                #... append the projector |1><1|
                newProj.append(proj2x2(True))
        # If there is more than one qubit...
        if size > 1:
            #... get the tensor product for the projectors list
            projector = tensorProd(newProj)
            #... append the projector to the projectors' list
            projList.append(projector)
        # Otherwise...
        else:
            #... append the projector to the projectors' list
            projList = projList + newProj
    # Return the projectors' list
    return projList


def entropy(density,printouts=False):
    # Function to extract the von Neumann entropy from a density matrix
    
    # Get the von Neumann entropy from the density
    logDensity = lalg.logm(density)
    prod = np.dot(density,logDensity)
    s = 0 - np.trace(prod) / np.log(2)
    
    # Print the results if asked for
    if printouts == True:
       print("\nRho: ")
       print(density)
       print("ln(Rho): ")
       print(logDensity)
       print("\nRho*ln(Rho): ")
       print(prod)
       print("\nVon Neumann Entropy: ", s)
    
    # Return the von Neumann entropy
    return np.real(s)

