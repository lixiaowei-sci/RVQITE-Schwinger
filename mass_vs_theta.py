from mindquantum import * 
import numpy as np 
import scipy as sp 
import networkx as nx 
from scipy.optimize import minimize 
from exact import *
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt 

def basis(param):
    """The basis of parameter vectors"""
    ei = []
    for i in range(len(param)):
        base = np.zeros_like(param)
        base[i] = 1
        ei.append(base)
    return ei 

def mat_A(circ, param):
    """Calculate the matrix A^R"""
    state = []
    sim = Simulator('mqvector', circ.n_qubits)
    sim.apply_circuit(circ,param)
    psi = sim.get_qs()
    ei = basis(param)
    for i in range(len(param)):
        sim.reset()
        sim.apply_circuit(circ,param+1e-7*ei[i])
        psi_partial = sim.get_qs()
        partial_psi = (psi_partial-psi)/1e-7 
        state.append(partial_psi)
    A = np.zeros((len(param),len(param)), dtype=complex)
    for i in range(len(param)):
        for j in range(len(param)):
            A[i,j]=state[i].conj()@state[j]
    return A.real 

def hva(N, depth):
    circ = Circuit()
    for i in range(N):
        circ += RX(f'tau{i}').on(i)

    for p in range(depth):
        for i in range(N):
            circ += RZ(f'alpha{p}{i}').on(i)
        for i in range(N-1):
            if i%2==0:
                circ += Rzz(f'gamma{p}{i}').on((i,i+1))
        for i in range(N-1):
            if i%2==0:
                circ += Rxx(f'beta{p}{i}').on((i,i+1))
                circ += Ryy(f'beta{p}{i}').on((i,i+1))
        for i in range(N-1):
            if i%2==1:
                circ += Rzz(f'gamma{p}{i}').on((i,i+1))
        for i in range(N-1):
            if i%2==1:
                circ += Rxx(f'beta{p}{i}').on((i,i+1))
                circ += Ryy(f'beta{p}{i}').on((i,i+1))
    return circ 

        
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

def schwinger_ham(N,J,w,m,mu,q):
    ham = QubitOperator()
    for j in range(1,N-1):
        Ln = QubitOperator()
        for i in range(j):
            Ln += 0.5*QubitOperator(f'Z{i}')+0.5*(-1)**i*QubitOperator('')
        Ln += q*QubitOperator('')
        ham += J*Ln*Ln
    for j in range(N-1):
        ham += 0.5*w*QubitOperator(f'X{j} X{j+1}')+0.5*w*QubitOperator(f'Y{j} Y{j+1}')
    for j in range(N):
        ham += (0.5*m*(-1)**j-0.5*mu)*QubitOperator(f'Z{j}')
    return ham 

def sparse_schwinger_ham(N,J,w,m,mu,q):
    id = dok_matrix(np.identity(2**N))
    ham = dok_matrix((2**N,2**N), dtype=np.float64)
    for j in range(1,N-1):
        Ln = dok_matrix((2**N,2**N), dtype=np.float64)
        for i in range(j):
            Ln += 0.5*z_ham(i,N)+0.5*(-1)**i*id 
        Ln += q*id 
        ham += J*Ln@Ln 
    for j in range(N-1):
        ham += 0.5*w*xy_ham(j,j+1,N)
    for j in range(N):
        ham += (0.5*m*(-1)**j-0.5*mu)*z_ham(j,N)
    return ham 



def u1_charge(N):
    hamiltonian = dok_matrix((2**N, 2**N), dtype=np.float64)
    for i in range(N):
        for a in range(2**N):
            ai = get_state_index(a,i)
            if ai==0:
                hamiltonian[a,a] += 0.5
            else:
                hamiltonian[a,a] += -0.5
    return csr_matrix(hamiltonian)  

def chiral(N):
    hamiltonian = dok_matrix((2**N, 2**N), dtype=np.float64)
    for i in range(N):
        for a in range(2**N):
            ai = get_state_index(a,i)
            if ai==0:
                hamiltonian[a,a] += 0.5*(-1)**i/N
            else:
                hamiltonian[a,a] += -0.5*(-1)**i/N
    return csr_matrix(hamiltonian) 

def electric(N,q):
    field = q*QubitOperator("")
    for n in range(N):
        for k in range(n+1):
            field += (QubitOperator(f'Z{k}')+(-1)**k*QubitOperator(''))/(2*N)
    return field 


def vqite_evolution(N,J,w,m,mu,q,circ,epoch,depth):
    """imaginary time evolution"""
    sim = Simulator('mqvector',N)
    ham = sparse_schwinger_ham(N,J,w,m,mu,q)                                     #here you can also use 'schwinger_model_ham(N,J,w,m,mu,q)'
    #eigenvalues, eigenvectors = sla.eigsh(ham, k=1, which='SA')        # the min eigenvalue, 'SA' means the smallest k terms
    #max_eigenvalues, max_eigenvector = sla.eigsh(ham, k=1, which='LA') # the max eigenvalue, 'LA' means the largest k terms
    ham = csr_matrix(ham)                                              # the 'Hamiltonian()' in mindquantum allows the input to be 'QubitOperator()' or 'csr_matrix'
    circuit = circ(N,depth)
    ops = sim.get_expectation_with_grad(Hamiltonian(ham),circuit)
    h2ops = sim.get_expectation_with_grad(Hamiltonian(ham@ham),circuit) # calculate $\langle H^2\rangle$
    Qops = sim.get_expectation_with_grad(Hamiltonian(u1_charge(N)), circuit)
   
    theta = 2*np.pi*np.random.rand(len(circuit.params_name))               # initial parameters
    
    step_length = 1e-2
    fun = []
    for i in range(epoch):
        A = mat_A(circuit, theta)
        f, g = ops(theta)
        h2, _ = h2ops(theta)
        Q_, _ = Qops(theta)
        f, g = np.squeeze(f.real), np.squeeze(g.real)
        h2 = np.squeeze(h2.real)
        eig, eig_vec = np.linalg.eigh(A)
        Lambda_inv = np.diag( np.where(eig > 1e-5, 1/eig, 0))  # cutofff: epsilon = 1e-5
        dot_theta = -eig_vec@Lambda_inv@eig_vec.T@g # np.dot(eig_vec, np.dot(Lambda_inv, np.dot(eig_vec.T, g)))
        theta += step_length*dot_theta
        error = dot_theta@A@dot_theta+g@dot_theta+h2-f**2
        if i%50==0:
            print('epoch |', i, 'fun', f, 'error', error) 
            print('charge', np.squeeze(Q_.real))
        fun.append(f)
        if error<1e-3:
            break
    return  theta

def get_expectations(circ, theta, q):
    N =  circ.n_qubits 
    obseverables = [Hamiltonian(u1_charge(N)), Hamiltonian(chiral(N)), Hamiltonian(electric(N,q))]
    sim = Simulator('mqvector', N)
    sim.apply_circuit(circ, theta)
    obsever = [sim.get_expectation(p) for p in obseverables]

    sim = Simulator('mqmatrix', N)
    sim.apply_circuit(circ, theta)
    qs = sim.get_partial_trace(range(int(N/2)))
    qs = np.around(qs,decimals=10)
    sim = Simulator('mqmatrix', int(N/2))
    sim.set_qs(qs)
    entropy = sim.entropy()
    obsever.append(entropy)
    obsever = np.array(obsever).real
    print(obsever)
    return obsever 


"""

ag=1, N=10, m/g=1, mu=0, q=[0,1], (note: here q=\theta/2\pi, i.e., \theta=[0,2\pi])

N=10, J=ag^2/2=1/2, w=1/(2a)=1/2.
"""

"""For Fig.4(b):
x: q\in [0,1] vs. y: m\in[-4,4]; mu=0
"""
N=10
J=0.5
w=0.5
mu=0
circ = hva
obsever = np.zeros((51,51,4))    
iter_q =0
for q in np.linspace(-1,1,51):
    iter_m =0
    theta = None 
    for m in np.linspace(-4,4,51):
        theta = vqite_evolution(N,J,w,m,mu,q,circ,1000,5)
        circuit_= circ(N,mu,5)
        print('q', q, 'm', m)
        obsever[iter_q, iter_m]=get_expectations(circuit_, theta, q)
        iter_m +=1
    iter_q+=1
np.save('fig4b.npy', obsever)







