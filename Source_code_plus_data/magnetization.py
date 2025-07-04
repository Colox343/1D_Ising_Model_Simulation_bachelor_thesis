import numpy as np
from iminuit import Minuit
from iminuit.cost import LeastSquares

#This function computes the average magentization for a given time
def get_Mz(t,lamb):
    return (1 + 2*lamb**2 + np.cos(4*t*np.sqrt(1+lamb**2))) / (2 + 2*lamb**2)

'''This function returns the exact analytical magnetization of a 4-spin chain for many instants over a range of time.
T is the maximum time to reach, delt_t is the distance between two instants in the graph, lamb is the strength of the magnetic field.
It returns the magnetizations associated to their times.'''
def get_data_Mz(T,delta_t,lamb):
    Mz = np.array([])
    t_axis = np.array([])
    t = 0

    while t < T:
        Mz = np.append(Mz,get_Mz(t,lamb))
        t_axis = np.append(t_axis,t)
        t += delta_t

    return [Mz,t_axis]

#Function that, given the memory of the simulator, computes the mean Mz, the standard deviation and the standard deviation of the mean.
def Mz_statistics(Mz_list):
    s = 0
    N = len(Mz_list)
    Mz_num_list = np.array([])
    
    #Extraction of the magnetizations from the Mz_list
    for i in range(N):
        Mz = Mz_list[i]
        #Mz_num is the numeric value of the ith magnetization to compute
        Mz_num = 0
        #Each position of the string Mz is controlled. If there is a '1' Mz_num is incremented by 1, otherwise it is decreased. 
        for j in range(len(Mz)):
            if Mz[j] == '1':
                Mz_num += 1
            else:
                Mz_num -= 1
        Mz_num /= len(Mz)
        Mz_num_list = np.append(Mz_num_list,Mz_num)
        s += Mz_num

    #Computation of the statistics
    mean = s/N
    s = 0
    for i in range(N):
        s += (Mz_num_list[i] - mean)**2
    stdv = np.sqrt(s/(N-1))
    stdv_mean = stdv/np.sqrt(N)
    
    return [mean,stdv,stdv_mean]

#This function interpolates the data with the function f, so that it is possible to estimate the correct value of the parameter a
def interpolation(x_axis,y_axis,y_err,f):
    least_squares = LeastSquares(x_axis, y_axis, y_err, f)
    my_minuit = Minuit(least_squares, a = 1, b = 0)
    my_minuit.migrad()    #Computes the value of the parameter
    my_minuit.hesse()     #Computes the uncertainty

    a_fit = my_minuit.values['a']
    a_err = my_minuit.errors['a']
    b_fit = my_minuit.values['b']
    b_err = my_minuit.errors['b']
    chi2 = my_minuit.fval
    ndof = my_minuit.ndof

    return [a_fit, a_err, b_fit, b_err, chi2, ndof]

'''This function receives as input a list of states measured and returns the probability distribution of the eigenstates.
The output is an array of 5 elements. They are ordered as the probabilities of obtaining
Mz=-1, Mz=-1/2, Mz=0, Mz=+1/2, Mz=+1'''
def get_prob_states(Mz_list):
    probabilities = np.zeros(5)
    Mz_eigenstates = [-1,-1/2,0,1/2,1]
    for i in range(len(Mz_list)):
        Mz = Mz_list[i]
        Mz_num = 0;
        for j in range(len(Mz)):
            if Mz[j] == '1':
                Mz_num += 1
            else:
                Mz_num -= 1
        Mz_num /= len(Mz)
        for j in range(len(Mz_eigenstates)):
            if Mz_eigenstates[j] == Mz_num:
                probabilities[j] += 1/len(Mz_list)
    
    return probabilities