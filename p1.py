"""MATH96012 2019 Project 1
"""
import numpy as np
import matplotlib.pyplot as plt
from cmath import phase
#--------------------------------

def simulate1(N=64,L=8,s0=0.2,r0=1,A=0.2,Nt=100):
    """Part1: Simulate bacterial colony dynamics
    Input:
    N: number of particles
    L: length of side of square domain
    s0: speed of particles
    r0: particles within distance r0 of particle i influence direction of motion
    of particle i
    A: amplitude of noise
    dt: time step
    Nt: number of time steps

    Output:
    X,Y: position of all N particles at Nt+1 times
    alpha: alignment parameter at Nt+1 times

    Do not modify input or return statement without instructor's permission.

    Add brief description of approach to problem here:

    - We first set the initial conditions as described in the assignment brief.
    - The initial conditions are then established.
    - I have created a matrix Rd(t), where the element Rd_jk is the distance between the jth particle and the kth particle
      at time t.
    - The particles then move one iteration before the for loop begins
    - We then begin the main for loop of the simulation, in this loop each iteration is one time step forward.
    - Since the particle's position at time t+1 is dependent on the direction of motion at time t+1, which in turn is
      dependent on the matrix Rd(t), in the main loop we first update theta, then the position arrays, followed by R.
    - At each iteration of the main loop we store alpha, as described in the assignment brief.
    """
    #Set initial condition
    phi_init = np.random.rand(N)*(2*np.pi) #ARRAY OF LEN N
    r_init = np.sqrt(np.random.rand(N))    #ARRAY OF LEN N
    Xinit,Yinit = r_init*np.cos(phi_init),r_init*np.sin(phi_init)
    Xinit+=L/2
    Yinit+=L/2

    X,Y = np.zeros((Nt+1,N)),np.zeros((Nt+1,N))

    Rd = np.zeros((N,N))
    for j in range(N):
        for k in range(N):
            Rd[j,k] += np.sqrt(((Xinit[j]-Xinit[k])**2) + ((Yinit[j]-Yinit[k])**2)) #Computes the initial upper triangular R

    theta = np.random.rand(N)*(2*np.pi) #initial directions of motion, ARRAY OF LEN N
    eye = np.complex(0,1)

    X[0,:] = np.array([x%N for x in np.add(Xinit,[s0*np.cos(a) for a in theta])])
    Y[0,:] = np.array([y%N for y in np.add(Yinit,[s0*np.sin(a) for a in theta])])
    alpha = []

    for t in range(Nt): #The big overall loop

        alpha.append((N**(-1))*abs(sum([np.exp(eye*p) for p in theta])))

        newtheta = []
        for j in range(N): # To udpdate theta as an array
            newtheta.append(phase(sum([np.exp(eye*g) for g in [theta[k] for k in np.nonzero(Rd[j,:]<=r0)[0]]]) + A*(len(np.nonzero(Rd[j,:]<=r0)[0]))*np.exp(eye*(np.random.uniform()*2*np.pi)))+ np.pi)
        theta = np.array(newtheta)

        X[t+1,:] = np.array([x%N for x in np.add(X[t,:],[s0*np.cos(a) for a in theta])])
        Y[t+1,:] = np.array([y%N for y in np.add(Y[t,:],[s0*np.sin(a) for a in theta])])

        for j in range(N):
            for k in range(N):
                Rd[j,k] = np.sqrt(((X[t+1,j]-X[t+1,k])**2) + ((Y[t+1,j]-Y[t+1,k])**2))

    alpha = np.array(alpha)

    return X,Y,alpha


def analyze(s_a,N_a,L_a,interval):
    """Part 2: Add input variables and modify return as needed

        s: Number of time steps
        N: Number of particles
        L: Length of side of square domain
        interval: Range of values for the amplitude of noise
    """
    print('We analyze how α depends on the amplitude of noise, A, for A in the range %5.2f to %5.2f:'%(interval[0],interval[len(interval)-1]))
    plt.figure()
    alphav = []
    for i in range(len(interval)):
        y = simulate1(N=N_a,L=L_a,s0=0.2,r0=1,A=interval[i],Nt=s_a)[2]
        plt.plot(np.linspace(0,s_a-1,s_a),y, label='A = %5.2f'%(interval[i]))
        alphav.append(np.var(y,axis=None))
    plt.title('The alignment parameter, α, vs time, where the number of particles = %2.0f'%(N_a))
    plt.legend()
    plt.xlabel('Time')
    plt.ylabel('α')
    plt.show()
    print('--------------------------------')
    for i in range(len(interval)):
        print('A = %5.2f, variance(α) = %2.4f'%(interval[i],alphav[i]))
        print('--------------------------------')
    return None
"""
------------------------------------
"""

def simulate2(N=16,L=4,s0=0.2,r0=1,A=0.2,Nt=150):
    #Set initial condition
    phi_init = np.random.rand(N)*(2*np.pi) #ARRAY OF LEN N
    r_init = np.sqrt(np.random.rand(N))    #ARRAY OF LEN N
    Xinit,Yinit = r_init*np.cos(phi_init),r_init*np.sin(phi_init)
    Xinit+=L/2
    Yinit+=L/2

    X,Y = np.zeros((Nt+1,N)),np.zeros((Nt+1,N))

    Rd = np.zeros((N,N))
    for j in range(N):
        for k in range(N):
            Rd[j,k] += np.sqrt(((Xinit[j]-Xinit[k])**2) + ((Yinit[j]-Yinit[k])**2)) #Computes the initial upper triangular R

    theta = np.random.rand(N)*(2*np.pi) #initial directions of motion, ARRAY OF LEN N
    eye = np.complex(0,1)

    X[0,:] = np.array([x%N for x in np.add(Xinit,[s0*np.cos(a) for a in theta])])
    Y[0,:] = np.array([y%N for y in np.add(Yinit,[s0*np.sin(a) for a in theta])])
    alpha = []

    for t in range(Nt): #The big overall loop

        alpha.append((N**(-1))*abs(sum([np.exp(eye*p) for p in theta])))

        newtheta = []
        for j in range(N): # To udpdate theta as an array
            newtheta.append(phase(sum([np.exp(eye*g) for g in [theta[k] for k in np.nonzero(Rd[j,:]<=r0)[0]]]) + A*(len(np.nonzero(Rd[j,:]<=r0)[0]))*np.exp(eye*(np.random.uniform()*2*np.pi)))+ np.pi)
        theta = np.array(newtheta)

        X[t+1,:] = np.array([x%N for x in np.add(X[t,:],[s0*np.cos(a) for a in theta])])
        Y[t+1,:] = np.array([y%N for y in np.add(Y[t,:],[s0*np.sin(a) for a in theta])])

        for j in range(N):
            for k in range(N):
                Rd[j,k] = np.sqrt(((X[t+1,j]-X[t+1,k])**2) + ((Y[t+1,j]-Y[t+1,k])**2))

    alpha = np.array(alpha)

    return X,Y,alpha
"""
"""
def analyze2(s_a,N_a,L_a,interval):
    """ Defined a second analyze function to analyze α against 1-A/A*
    """
    print("We analyze how α depends on the amplitude of noise,1-A/A*, for 1-A/A* in the range %2.4f to %2.4f: "%(interval[0],interval[len(interval)-1]))
    plt.figure()
    alphav = []
    for i in range(len(interval)):
        y = simulate1(N=N_a,L=L_a,s0=0.2,r0=1,A=interval[i],Nt=s_a)[2]
        plt.plot(np.linspace(0,s_a-1,s_a),y, label='1-A/A* = %2.4f'%(interval[i]))

        alphav.append(np.var(y,axis=None))
    plt.title('The alignment parameter, α, vs time, where the number of particles = %2.0f'%(N_a))
    plt.legend()
    plt.xlabel('Time')
    plt.ylabel('α')
    plt.show()
    print('--------------------------------')
    for i in range(len(interval)):
        print('1-A/A* = %5.4f, variance(α) = %2.4f'%(interval[i],alphav[i]))
        print('--------------------------------')
    return None
#--------------------------------------------------------------
if __name__ == '__main__':

    output_a = analyze(100,16,4,np.arange(0.2,0.8,0.1))

    output_b = analyze(100,32,4,np.arange(0.2,0.8,0.1))

    output_c = analyze(100,32,4,np.arange(0.64,0.8,0.02))

    Astar = 0.66
    Arange = np.array([(1-(a/Astar)) for a in [0.20, 0.24, 0.28, 0.32, 0.36, 0.40]])
    output_d = analyze2(500,16,4,Arange)
