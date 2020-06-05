
"""MATH 96012 Project 3
Contains four functions:
    simulate2: Simulate bacterial dynamics over m trials. Return: all positions at final time
        and alpha at nt+1 times averaged across the m trials.
    performance: To be completed -- analyze and assess performance of python, fortran, and fortran+openmp simulation codes
    correlation: To be completed -- compute and analyze correlation function, C(tau)
    visualize: To be completed -- generate animation illustrating "non-trivial" particle dynamics
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import m1
b = m1.bmotion #assumes that p3.f90 has been compiled with: python -m numpy.f2py --f90flags='-fopenmp' -c p3dev.f90 -m m1 -lgomp
import scipy.spatial.distance as scd
import time
#May also use scipy and time modules as needed

def simulate2(M=10,N=64,L=8,s0=0.2,r0=1,A=0.64,Nt=100):
    """Simulate bacterial colony dynamics
    Input:
    M: Number of simulations
    N: number of particles
    L: length of side of square domain
    s0: speed of particles
    r0: particles within distance r0 of particle i influence direction of motion
    of particle i
    A: amplitude of noise
    Nt: number of time steps

    Output:
    X,Y: position of all N particles at Nt+1 times
    alpha: alignment parameter at Nt+1 times averaged across M simulation

    Do not modify input or return statement without instructor's permission.

    Add brief description of approach of differences from simulate1 here:
    This code carries out M simulations at a time with partial vectorization
    across the M samples.
    """
    #Set initial condition
    phi_init = np.random.rand(M,N)*(2*np.pi)
    r_init = np.sqrt(np.random.rand(M,N))
    Xinit,Yinit = r_init*np.cos(phi_init),r_init*np.sin(phi_init)
    Xinit+=L/2
    Yinit+=L/2
    #---------------------

    #Initialize variables
    P = np.zeros((M,N,2)) #positions
    P[:,:,0],P[:,:,1] = Xinit,Yinit
    alpha = np.zeros((M,Nt+1)) #alignment parameter
    S = np.zeros((M,N),dtype=complex) #phases
    T = np.random.rand(M,N)*(2*np.pi) #direction of motion
    n = np.zeros((M,N)) #number of neighbors
    E = np.zeros((M,N,Nt+1),dtype=complex)
    d = np.zeros((M,N,N))
    dtemp = np.zeros((M,N*(N-1)//2)) #Temp Relative distances between particles
    AexpR = np.random.rand(M,N,Nt)*(2*np.pi)
    AexpR = A*np.exp(1j*AexpR)

    r0sq = r0**2
    E[:,:,0] = np.exp(1j*T)

    #Time marching-----------
    for i in range(Nt):
        for j in range(M):
            dtemp[j,:] = scd.pdist(P[j,:,:],metric='sqeuclidean')

        dtemp2 = dtemp<=r0sq #elements of dtemp that are <= r0sq
        for j in range(M):
            d[j,:,:] = scd.squareform(dtemp2[j,:])
        n = d.sum(axis=2) + 1
        S = E[:,:,i] + n*AexpR[:,:,i]

        for j in range(M):
            S[j,:] += d[j,:,:].dot(E[j,:,i])

        T = np.angle(S)

        #Update X,Y
        P[:,:,0] = P[:,:,0] + s0*np.cos(T)
        P[:,:,1] = P[:,:,1] + s0*np.sin(T)

        #Enforce periodic boundary conditions
        P = P%L

        E[:,:,i+1] = np.exp(1j*T)
    #----------------------

    #Compute order parameter
    alpha = (1/(N*M))*np.sum(np.abs(E.sum(axis=1)),axis=0)
    X, Y = P[:,:,0], P[:,:,1]

    return X,Y,alpha


def performance(input_p=(None),display=False,display2=False,display3=False):
    """Assess performance of simulate2, simulate2_f90, and simulate2_omp
    Modify the contents of the tuple, input, as needed
    When display is True, figures equivalent to those
    you are submitting should be displayed
    """

    m,n,nt,s_zero,am = input_p #set input variables
    alphas = np.zeros((3,nt+1)) #rows are python,fortran,fomp,fomp2
    #generate alpha_ave for each implementation
    py1 = time.time()
    alphas[0,:] = simulate2(M=m,N=n,L=16,s0=s_zero,r0=1.0,A=am,Nt=nt)[2] #python
    py2 = time.time()
    f1 = time.time()
    alphas[1,:] = b.simulate2_f90(m,n,nt)[2] #fortran
    f2 = time.time()
    fp1 = time.time()
    alphas[2,:] = b.simulate2_omp(m,n,nt)[2] #fortran + openmp
    fp2 = time.time()

    runtimes=[py2-py1,f2-f1,fp2-fp1]

    labels = ('Python','Fortran','Fortran+OMP')
    y_pos = np.arange(len(labels))
    if display==True:
        print('Alpha against time:')
        x = np.arange(alphas.shape[1])
        for i in range(alphas.shape[0]):
            plt.plot(x,alphas[i,:],label=labels[i])
        plt.legend()
        plt.xlabel('Time')
        plt.ylabel('alpha_ave')
        plt.show()
        print('Comparing execution times:')
        plt.barh(y_pos,runtimes,align='center',alpha=0.5)
        plt.yticks(y_pos,labels)
        plt.xlabel('Execution time (s)')
        plt.title('Execution times of different implementations')
        plt.show()
        print('Execution times:')
        for i in range(alphas.shape[0]):
            print(labels[i],' = ',runtimes[i],'seconds')
    if display2==True:
        print('-----------------------------------------------')
        print('Investigating execution time against m...')
        m_s = np.arange(40,401,40)
        m_runtime = np.zeros((len(m_s),len(labels)))
        for m in m_s:
            i = int(np.where(m==m_s)[0])
            n,nt = input_p[1],input_p[2]
            py1 = time.time()
            simulate2(M=m,N=n,L=64,s0=0.2,r0=1.0,A=0.64,Nt=nt)[2] #python
            py2 = time.time()
            f1 = time.time()
            b.simulate2_f90(m,n,nt)[2] #fortran
            f2 = time.time()
            fp1 = time.time()
            b.simulate2_omp(m,n,nt)[2] #fortran + openmp
            fp2 = time.time()
            m_runtime[i,:] = np.array([py2-py1,f2-f1,fp2-fp1])
        for i in range(3):
            plt.plot(m_s,m_runtime[:,i],label=labels[i])
        plt.legend()
        plt.xlabel('m')
        plt.ylabel('Execution time')
        plt.title('Investigating how execution times vary against m, for constant n & nt')
        plt.show()

    if display3==True:
        print('-----------------------------------------------')
        print('Investigating execution time against n...')
        n_s = np.arange(60,381,40)
        n_runtime = np.zeros((len(n_s),len(labels)))
        for n in n_s:
            i = int(np.where(n==n_s)[0])
            m,nt = input_p[0],input_p[2]
            py1 = time.time()
            simulate2(M=m,N=n,L=64,s0=0.2,r0=1.0,A=0.64,Nt=nt)[2] #python
            py2 = time.time()
            f1 = time.time()
            b.simulate2_f90(m,n,nt)[2] #fortran
            f2 = time.time()
            fp1 = time.time()
            b.simulate2_omp(m,n,nt)[2] #fortran + openmp
            fp2 = time.time()
            n_runtime[i,:] = np.array([py2-py1,f2-f1,fp2-fp1])
        for i in range(3):
            plt.plot(n_s,n_runtime[:,i],label=labels[i])
        plt.legend()
        plt.xlabel('m')
        plt.ylabel('Runtime')
        plt.title('Investigating how execution times vary against n, for constant m & nt')
        plt.show()


    return alphas #Modify as needed

def correlation(input_c=(None),display=False):
    """Compute and analyze temporal correlation function, C(tau)
    Modify the contents of the tuple, input, as needed
    When display is True, figures equivalent to those
    you are submitting should be displayed
    """
    tau_range = 80
    a = 300
    b = a + 400
    C_data = np.zeros((tau_range,b-a+1,3))
    C_compare = np.zeros((tau_range,len(input_c))) #each column will be C(tau) for each implementation.
    T = input_c

    labels = ('Python','Fortran','Fortran+OMP')
    for i in range(3):
        for tau in range(tau_range):
            C_data[tau,:,0] = T[i,a:b+1]
            C_data[tau,:,1] = T[i,a+tau:b+tau+1]
        C_data[:,:,2] = np.multiply(C_data[:,:,0],C_data[:,:,1])
        C_sum = np.sum(C_data,axis=2)
        C_sum2 = np.zeros((tau_range,2))
        C_sum2[:,0],C_sum2[:,1] = C_sum[:,0],C_sum[:,2]
        C_compare[:,i] = np.array([(1/200)*j[1] - (1/40000)*(j[0]**2) for j in C_sum2])
    if display==True:
        for j in range(3):
            t = np.arange(C_compare.shape[0])
            plt.plot(t,C_compare[:,j],label=labels[j])
        plt.legend()
        plt.xlabel('Tau')
        plt.ylabel('C(tau)')
        plt.title('Correlation measure against tau')
        plt.show()


    return None #Modify as needed



def visualize(input_v=(None)):
    """Generate an animation illustrating the evolution of
        villages during C vs M competition
    input_v is X,Y output positions from simulate2
    """
    L = 64
    N = 128
    X, Y = input_v[0], input_v[1]
    fig = plt.figure(figsize=(6,6))
    ax = fig.add_axes([0,0,1,1], frameon=True, aspect=1)
    scat = ax.scatter(X[0,:],Y[0,:])
    ax.set_xlim(0,L)
    ax.set_ylim(0,L)
    def update(frame):
        P = np.zeros((N,2))
        P[:,0],P[:,1] = X[frame,:],Y[frame,:]
        scat.set_offsets(P)
        return scat,
    anim = animation.FuncAnimation(fig,update,interval=100,blit=True,frames=X.shape[0])
    anim.save('p3movie.mp4')
    return None #Modify as needed


if __name__ == '__main__':
    #Modify the code here so that it calls performance analyze and
    # generates the figures that you are submitting with your code

    start = time.time()
    # TO GENERATE FIGS 1&2
    b.bm_l, b.bm_s0, b.bm_r0, b.bm_a = 64,0.2,1.0,0.64
    input_p = 200,100,1000,0.2,0.64 #m,n,nt,s0,A
    print('(m,n,nt)=',input_p)
    print('Processing...')
    performance(input_p,display=True,display2=False,display3=False) #modify as needed

    # TO GENERATE FIG 3
    input_p = 100,100,300,0.2,0.64
    performance(input_p,display=False,display2=True,display3=False)

    # TO GENERATE FIG 4
    #CHANGE LINE 143 to m_s = np.arange(290,351,10)
    performance(input_p,display=False,display2=True,display3=False)

    # TO GENERATE FIG 5
    input_p = 200,100,300,0.2,0.64
    performance(input_p,display=False,display2=False,display3=True)

    # TO GENERATE FIG 6
    b.bm_l, b.bm_s0, b.bm_r0, b.bm_a = 16,0.1,1.0,0.625
    input_p = 200,400,785,0.1,0.625
    output_p = performance(input_p,display=False,display2=False,display3=False)
    input_c = output_p
    output_c = correlation(input_c,display=True)

    #TO GENERATE ANIMATION
    input_v = simulate2(M=200,N=128,L=64,s0=0.2,r0=1.0,A=0.625,Nt=250)
    visualize(input_v)

    end = time.time()
    r = (end-start)%60
    q = (end-start-r)/60
    print('Total time to run the whole script = ',q,'minutes and ',r,'seconds')
