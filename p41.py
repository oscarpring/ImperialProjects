
"""Final project, part 1"""
import numpy as np
import matplotlib.pyplot as plt
import m2
fl = m2.flow
import time
#from m2 import flow as fl #assumes p41.f90 has been compiled with: f2py -c p41.f90 -m m2


def jacobi(n,kmax=10000,tol=1.0e-8,s0=0.1,display=False):
    """ Solve liquid flow model equations with
        jacobi iteration.
        Input:
            n: number of grid points in r and theta
            kmax: max number of iterations
            tol: convergence test parameter
            s0: amplitude of cylinder deformation
            display: if True, plots showing the velocity field and boundary deformation
            are generated
        Output:
            w,deltaw: Final velocity field and |max change in w| each iteration
    """

    #-------------------------------------------
    #Set Numerical parameters and generate grid
    Del_t = 0.5*np.pi/(n+1)
    Del_r = 1.0/(n+1)
    Del_r2 = Del_r**2
    Del_t2 = Del_t**2
    r = np.linspace(0,1,n+2)
    t = np.linspace(0,np.pi/2,n+2) #theta
    tg,rg = np.meshgrid(t,r) # r-theta grid

    #Factors used in update equation (after dividing by gamma)
    rg2 = rg*rg
    fac = 0.5/(rg2*Del_t2 + Del_r2)
    facp = rg2*Del_t2*fac*(1+0.5*Del_r/rg) #alpha_p/gamma
    facm = rg2*Del_t2*fac*(1-0.5*Del_r/rg) #alpha_m/gamma
    fac2 = Del_r2*fac #beta/gamma
    RHS = fac*(rg2*Del_r2*Del_t2) #1/gamma

    #set initial condition/boundary deformation
    w0 = (1-rg**2)/4 #Exact solution when s0=0
    s_bc = s0*np.exp(-10.*((t-np.pi/2)**2))/Del_r
    fac_bc = s_bc/(1+s_bc)

    deltaw = []
    w = w0.copy()
    wnew = w0.copy()

    #Jacobi iteration
    for k in range(kmax):
        #Compute wnew
        wnew[1:-1,1:-1] = RHS[1:-1,1:-1] + w[2:,1:-1]*facp[1:-1,1:-1] + w[:-2,1:-1]*facm[1:-1,1:-1] + (w[1:-1,:-2] + w[1:-1,2:])*fac2[1:-1,1:-1] #Jacobi update

        #Apply boundary conditions
        wnew[:,0] = wnew[:,1] #theta=0
        wnew[:,-1] = wnew[:,-2] #theta=pi/2
        wnew[0,:] = wnew[1,:] #r=0
        wnew[-1,:] = wnew[-2,:]*fac_bc #r=1s

        #Compute delta_p
        deltaw += [np.max(np.abs(w-wnew))]
        w = wnew.copy()
        #if k%1000==0: print("k,dwmax:",k,deltaw[k])
        #check for convergence
        if deltaw[k]<tol:
            print("Converged,k=%d,dw_max=%28.16f " %(k,deltaw[k]))
            break

    deltaw = deltaw[:k+1]

    if display:
        #plot final velocity field, difference from initial guess, and cylinder
        #surface
        plt.figure()
        plt.contour(t,r,w,50)
        plt.xlabel(r'$\theta$')
        plt.ylabel('r')
        plt.title('Final velocity field, python jacobi')
        plt.savefig('velfield_pj.png')
        plt.show()

        plt.figure()
        plt.contour(t,r,np.abs(w-w0),50)
        plt.xlabel(r'$\theta$')
        plt.ylabel('r')
        plt.title(r'$|w - w_0|$, python jacobi')
        plt.savefig('ww0_diff_pj.png')
        plt.show()

        plt.figure()
        plt.polar(t,np.ones_like(t),'k--')
        plt.polar(t,np.ones_like(t)+s_bc*Del_r,'r-')
        plt.title('Deformed cylinder surface, python jacobi')
        plt.savefig('cylsurface_pj.png')
        plt.show()

    return w,deltaw



def performance(n,nrange,nrange2):

    """ PLOT PYTHON JACOBI DATA  """
    jacobi(n,display=True)

    #set up variables for plotting fortran jacobi data
    """ PLOT FORTRAN JACOBI DATA & SGISOLVE VECTORFIELD """
    del_t = 0.5*np.pi/(n+1)
    del_r = 1.0/(n+1)
    del_r2 = del_r**2
    del_t2 = del_t**2
    r = np.linspace(0,1,n+2)
    t = np.linspace(0,np.pi/2,n+2) #theta
    tg,rg = np.meshgrid(t,r) # r-theta grid

    w0_j = (1-rg**2)/4 #Exact solution when s0=0
    s_bc = fl.fl_s0*np.exp(-10.*((t-np.pi/2)**2))/del_r
    fac_bc = s_bc/(1+s_bc)

    w_j = fl.jacobi(n) #vector field returned by Fortran Jacobi

    s,fac,facm,facp,fac2,fac_bc = np.zeros(n+2),np.zeros(n+2),np.zeros(n+2),np.zeros(n+2),np.zeros(n+2),np.zeros(n+2)

    s = np.zeros(n+2)
    for s1 in range(n+2):
        s[s1] = fl.fl_s0*(np.exp(-10*(s1*del_t - (np.pi/2.0))**2) + np.exp(-10*(s1*del_t + (np.pi/2.0))**2))
    for p2 in range(1,n+1):
        fac[p2] = 1.0/((2.0/del_r2) + (2.0/(del_t2*(p2*del_r)**2)))
    for p1 in range(1,n+1):
        facm[p1] = (1.0/del_r2 - 1.0/(2.0*p1*del_r2))*fac[p1]
        facp[p1] = (1.0/del_r2 + 1.0/(2.0*p1*del_r2))*fac[p1]
        fac2[p1] = (1.0/(del_t2*(p1*del_r)**2))*fac[p1]
    fac_bc = s/(del_r + s)

    w_unpk = fl.sgisolve(n) # unpacked vector field returned by Fortran sgisolve
    w_pk = np.zeros((n+2,n+2))

    for i in range(n+2):
        w_pk[i,:] = w_unpk[i*(n+2):(i+1)*(n+2)]  #packing the Fortran sgisolve vector field.

    plt.figure()
    plt.contour(t,r,w_j,50)
    plt.xlabel(r'$\theta$')
    plt.ylabel('r')
    plt.title('Final velocity field, fortran jacobi')
    plt.savefig('velfield_fj.png')
    plt.show()

    plt.figure()
    plt.contour(t,r,w_pk,50)
    plt.xlabel(r'$\theta$')
    plt.ylabel('r')
    plt.title('Final velocity field, fortran sgisolve')
    plt.savefig('velfield_fsgi.png')
    plt.show()

    plt.figure()
    plt.contour(t,r,np.abs(w_j-w0_j),50)
    plt.xlabel(r'$\theta$')
    plt.ylabel('r')
    plt.title(r'$|w - w_0|$, fortran jacobi')
    plt.savefig('ww0_diff_fj.png')
    plt.show()



    plt.figure()
    plt.polar(t,np.ones_like(t),'k--')
    plt.polar(t,np.ones_like(t)+s_bc*del_r,'r-')
    plt.title('Deformed cylinder surface, fortran jacobi')
    plt.savefig('cylsurface_fj.png')
    plt.show()

    """ NOW WE LOOK AT THE SPEEDS OF THE FORTRAN SUBROUTINES MVEC & MTVEC AGAINST N  """

    labels1 = ['mvec','mtvec','jacobi_fortran','jacobi_python','sgisolve(numthreads=2)','sgisolve(numthreads=1)']
    times = np.zeros((len(labels1),len(nrange)))
    j = 0
    for n_i in nrange:
        fl.numthreads = 2
        s,fac,facm,facp,fac2,fac_bc = np.zeros(n_i+2),np.zeros(n_i+2),np.zeros(n_i+2),np.zeros(n_i+2),np.zeros(n_i+2),np.zeros(n_i+2)
        #set up coefficient vectors
        del_t = 0.5*np.pi/(n_i+1)
        del_r = 1.0/(n_i+1)
        del_r2 = del_r**2
        del_t2 = del_t**2
        r = np.zeros((n_i+2,n_i+2))
        t = np.linspace(0,np.pi/2,n+2) #theta

        s = np.zeros(n_i+2)
        for s1 in range(n_i+2):
            s[s1] = fl.fl_s0*(np.exp(-10*(s1*del_t - (np.pi/2.0))**2) + np.exp(-10*(s1*del_t + (np.pi/2.0))**2))
        for p2 in range(1,n_i+1):
            fac[p2] = 1.0/((2.0/del_r2) + (2.0/(del_t2*(p2*del_r)**2)))
        for p1 in range(1,n_i+1):
            facm[p1] = (1.0/del_r2 - 1.0/(2.0*p1*del_r2))*fac[p1]
            facp[p1] = (1.0/del_r2 + 1.0/(2.0*p1*del_r2))*fac[p1]
            fac2[p1] = (1.0/(del_t2*(p1*del_r)**2))*fac[p1]
        fac_bc = s/(del_r + s)

        #first generated a vector w to input into mvec & mtvec
        w_packed = fl.jacobi(n_i)
        w = np.zeros((n_i+2)**2)
        for i in range(n_i+2):
            w[i*(n_i+2):(i+1)*(n_i+2)] = w_packed[i,:] #unpacks w_packed into a single 1D array
            r[i,:] = i*del_r

        mv1 = time.time()  #mvec time
        mv_out = fl.mvec(fac,fac2,facp,facm,fac_bc,w,n_i)
        mv2 = time.time()
        times[0,j] = mv2 - mv1

        mt1 = time.time()   #mtvec time
        mt_out = fl.mtvec(fac,fac2,facp,facm,fac_bc,w,n_i)
        mt2 = time.time()
        times[1,j] = mt2 - mt1

        jf1 = time.time()      #jacobi fortran time
        w_jf = fl.jacobi(n_i)
        jf2 = time.time()
        times[2,j] = jf2-jf1

        jp1 = time.time()       #jacobi python time
        w_jp = jacobi(n_i,display=False)
        jp2 = time.time()
        times[3,j] = jp2-jp1

        sgi1 = time.time()      #sgisolve time when numthreads = 2
        w_s = fl.sgisolve(n_i)
        sgi2 = time.time()
        times[4,j] = sgi2-sgi1

        fl.numthreads = 1

        sgi3 = time.time()      #sgisolve time when numthreads = 1
        w_s = fl.sgisolve(n_i)
        sgi4 = time.time()
        times[5,j] = sgi4-sgi3

        j += 1

    labels2 = ['mvec','mtvec','jacobi_fortran','jacobi_python','sgisolve(numthreads=2)','sgisolve(numthreads=1)']
    times2 = np.zeros((len(labels2),len(nrange2)))
    j = 0
    for n_i in nrange2:
        fl.numthreads = 2
        s,fac,facm,facp,fac2,fac_bc = np.zeros(n_i+2),np.zeros(n_i+2),np.zeros(n_i+2),np.zeros(n_i+2),np.zeros(n_i+2),np.zeros(n_i+2)
        #set up coefficient vectors
        del_t = 0.5*np.pi/(n_i+1)
        del_r = 1.0/(n_i+1)
        del_r2 = del_r**2
        del_t2 = del_t**2
        r = np.zeros((n_i+2,n_i+2))
        t = np.linspace(0,np.pi/2,n+2) #theta

        s = np.zeros(n_i+2)
        for s1 in range(n_i+2):
            s[s1] = fl.fl_s0*(np.exp(-10*(s1*del_t - (np.pi/2.0))**2) + np.exp(-10*(s1*del_t + (np.pi/2.0))**2))
        for p2 in range(1,n_i+1):
            fac[p2] = 1.0/((2.0/del_r2) + (2.0/(del_t2*(p2*del_r)**2)))
        for p1 in range(1,n_i+1):
            facm[p1] = (1.0/del_r2 - 1.0/(2.0*p1*del_r2))*fac[p1]
            facp[p1] = (1.0/del_r2 + 1.0/(2.0*p1*del_r2))*fac[p1]
            fac2[p1] = (1.0/(del_t2*(p1*del_r)**2))*fac[p1]
        fac_bc = s/(del_r + s)

        #first generated a vector w to input into mvec & mtvec
        w_packed = fl.jacobi(n_i)
        w = np.zeros((n_i+2)**2)
        for i in range(n_i+2):
            w[i*(n_i+2):(i+1)*(n_i+2)] = w_packed[i,:] #unpacks w_packed into a single 1D array
            r[i,:] = i*del_r

        mv1 = time.time()  #mvec time
        mv_out = fl.mvec(fac,fac2,facp,facm,fac_bc,w,n_i)
        mv2 = time.time()
        times2[0,j] = mv2 - mv1

        mt1 = time.time()   #mtvec time
        mt_out = fl.mtvec(fac,fac2,facp,facm,fac_bc,w,n_i)
        mt2 = time.time()
        times2[1,j] = mt2 - mt1

        jf1 = time.time()      #jacobi fortran time
        w_jf = fl.jacobi(n_i)
        jf2 = time.time()
        times2[2,j] = jf2-jf1

        jp1 = time.time()       #jacobi python time
        w_jp = jacobi(n_i,display=False)
        jp2 = time.time()
        times2[3,j] = jp2-jp1

        sgi1 = time.time()      #sgisolve time when numthreads = 2
        w_s = fl.sgisolve(n_i)
        sgi2 = time.time()
        times2[4,j] = sgi2-sgi1

        fl.numthreads = 1

        sgi3 = time.time()      #sgisolve time when numthreads = 1
        w_s = fl.sgisolve(n_i)
        sgi4 = time.time()
        times2[5,j] = sgi4-sgi3

        j += 1

    plt.figure()
    for i in range(1):
        plt.plot(nrange,times[i,:],label=labels1[i])
    plt.legend()
    plt.xlabel('n')
    plt.ylabel('runtime, s')
    plt.title('Runtime of the Fortran subroutine mvec against n')
    plt.savefig('mvec_vs_n.png')
    plt.show()

    plt.figure()
    for i in range(2):
        plt.plot(nrange,times[i,:],label=labels1[i])
    plt.legend()
    plt.xlabel('n')
    plt.ylabel('runtimes, s')
    plt.title('Runtimes of the Fortran subroutines mvec & mtvec against n')
    plt.savefig('mvecmtvec_vs_n.png')
    plt.show()

    plt.figure()
    for i in range(2,len(labels1)):
        plt.plot(nrange,times[i,:],label=labels1[i])
    plt.legend()
    plt.xlabel('n')
    plt.ylabel('runtimes, s')
    plt.title('Runtimes of the Fortran subroutines & python jacobi against n')
    plt.savefig('sgi&jacvs_n.png')
    plt.show()

    plt.figure()
    for i in range(2,len(labels2)):
        plt.plot(nrange2,times2[i,:],label=labels2[i])
    plt.legend()
    plt.xlabel('n')
    plt.ylabel('runtimes, s')
    plt.title('Runtimes of the Fortran subroutines & python jacobi against n')
    plt.savefig('sgi&jacvs_n_long.png')
    plt.show()

    return None



if __name__=='__main__':

    n = 30
    print("n = ",n)
    fl.fl_s0, fl.numthreads= 0.1, 0.2
    nrange, nrange2 = np.arange(1,37,5), np.arange(1,72,10)
    print('nrange = ',nrange)
    print('nrange2 = ',nrange2)
    performance(n,nrange,nrange2)
