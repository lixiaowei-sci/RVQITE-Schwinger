from mindquantum import *
import numpy as np 

import networkx as nx
import numpy as np
from scipy.sparse import dok_matrix
import scipy.sparse.linalg as sla
from scipy.sparse.linalg import expm_multiply

def get_state_index(state, index):
    mask = 1<<index 
    return (state&mask)>>index 

def flip_state(state, index):
    mask = 1<<index 
    return state^mask 

def heisenberg_model(graph, j_value):
    """Hamiltonian 的位运算表示"""
    n = graph.number_of_nodes()
    hamiltonian = dok_matrix((2**n, 2**n), dtype=np.float64)
    for edge in graph.edges:
        i, j = edge
        for a in range(2**n): 
        
        
            ai = get_state_index(a, i)
            aj = get_state_index(a, j)

            
            if ai == aj:
                hamiltonian[a,a] += j_value
            else:
                hamiltonian[a,a] += -j_value

            if ai!=aj:
                b = flip_state(a, i)
                b = flip_state(b, j)
                hamiltonian[a,b] += 2
    return hamiltonian 

def schwinger_model(matrix,vector):
    n = matrix.shape[0]
    hamiltonian = dok_matrix((2**n, 2**n), dtype=np.float64)
    for i in range(n-1):
        for j in range(i+1,n):
            for a in range(2**n):
                ai = get_state_index(a, i)
                aj = get_state_index(a, j)
                if ai == aj:
                    hamiltonian[a,a] += matrix[i,j]
                else:
                    hamiltonian[a,a] += -matrix[i,j]

        k=i+1
        for a in range(2**n):
            ai = get_state_index(a, i)
            ak = get_state_index(a, k)
            if ai!=ak:
                b = flip_state(a, i)
                b = flip_state(b, k)
                hamiltonian[a,b] += 2*vector[i] 
    for i in range(n):
        for a in range(2**n):
            ai = get_state_index(a,i)
            if ai==0:
                hamiltonian[a,a] += matrix[i,i]
            else:
                hamiltonian[a,a] += -matrix[i,i]

    return hamiltonian 
        
def z_ham(i,n):
    hamiltonian = dok_matrix((2**n,2**n), dtype=np.float64)
    for a in range(2**n):
        ai = get_state_index(a, i)
        if ai==0:
            hamiltonian[a,a] += 1
        else:
            hamiltonian[a,a] += -1
    return hamiltonian 

def xy_ham(i,j,n):
    hamiltonian = dok_matrix((2**n,2**n), dtype=np.float64)
    for a in range(2**n):
        ai = get_state_index(a, i)
        aj = get_state_index(a, j)
        if ai != aj:
            b = flip_state(a,i)
            b = flip_state(b,j)
            hamiltonian[a,b]+=2 
    return hamiltonian 