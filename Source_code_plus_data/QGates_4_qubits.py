from qiskit import QuantumCircuit
import numpy as np

#This function is just used to simplify the code in get_FT, since it is repeated twice
def building_block1_FT(qc):
    qc.h(0)
    qc.h(1)
    qc.cx(0,1)
    qc.h(0)
    qc.h(1)

#This function implements the fSwap gate for two qubits
def get_fSwap():
    qc = QuantumCircuit(2)
    qc.name = "fSwap"
    qc.swap(0,1)
    qc.cz(0,1)

    return qc


'''This gate implements the Fourier transform gate starting from the parameters
n (total number of qubits) and k (mode to implement)'''
def get_FT(n,k):
    phi = -1*2*np.pi*k/n
    qc = QuantumCircuit(2)
    qc.name = "FT_"+str(n)+"_"+str(k)

    qc.id(0)
    qc.p(phi,1)
    building_block1_FT(qc)
    qc.ch(0,1)
    building_block1_FT(qc)
    qc.cz(0,1)

    return qc

#Inverse of the Fourier transform
def get_FT_inv(n,k):
    phi = -1*2*np.pi*k/n
    qc = QuantumCircuit(2)
    qc.name = "FT_inv_"+str(n)+"_"+str(k)

    qc.cz(0,1)
    building_block1_FT(qc)
    qc.ch(0,1)
    building_block1_FT(qc)
    qc.p(-1*phi,1)
    qc.id(0)

    return qc


#Function that computes the value of the angle to use in the Bogoliubov transformation
def get_theta(n,k,lamb):
    delta = np.sin(2*np.pi*k/n)
    eps = lamb + np.cos(2*np.pi*k/n)
    return np.arctan(delta/eps)

'''This function implements the Bogoliubov trasnformation starting from the parameters
n (total number of qubits), k (mode to transform) and lamb (lambda, intensity of the external magnetic field)'''
def get_BT(n,k,lamb):
    theta = get_theta(n,k,lamb)
    qc = QuantumCircuit(2)
    qc.name = "BT_"+str(n)+"_"+str(k)

    qc.cx(0,1)
    qc.x(1)
    qc.crx(-1*theta,1,0)
    qc.x(1)
    qc.cx(0,1)

    return qc

#Inverse of the Bogoliubov trasnformation
def get_BT_inv(n,k,lamb):
    theta = get_theta(n,k,lamb)
    qc = QuantumCircuit(2)
    qc.name = "BT_inv_"+str(n)+"_"+str(k)

    qc.cx(0,1)
    qc.x(1)
    qc.crx(theta,1,0)
    qc.x(1)
    qc.cx(0,1)

    return qc

#Application of the full disentanglement circuit
def apply_Udis(qc,n,lamb):
    qc.append(get_fSwap(),[1,2])
    qc.append(get_FT(n,0),[0,1])
    qc.append(get_FT(n,0),[2,3])
    qc.append(get_fSwap(),[1,2])
    qc.append(get_FT(n,0),[0,1])
    qc.append(get_FT(n,1),[2,3])
    qc.append(get_BT(n,0,lamb),[0,1])
    qc.append(get_BT(n,1,lamb),[2,3])

#Application of the inverse of Udis
def apply_Udis_inv(qc,n,lamb):
    qc.append(get_BT_inv(n,0,lamb),[0,1])
    qc.append(get_BT_inv(n,1,lamb),[2,3])
    qc.append(get_FT_inv(n,0),[0,1])
    qc.append(get_FT_inv(n,1),[2,3])
    qc.append(get_fSwap(),[1,2])
    qc.append(get_FT_inv(n,0),[0,1])
    qc.append(get_FT_inv(n,0),[2,3])
    qc.append(get_fSwap(),[1,2])

#Energy of a fermion with momentum k in the basis of the diagonal Hamiltonian
def get_E(n,k,lamb):
    E = 2*np.sqrt((lamb + np.cos(2*np.pi*k/n))**2 + (np.sin(2*np.pi*k/n))**2)
    return E