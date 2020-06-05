"""MATH96012 Project 2"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as so
from f4 import lrmodel as lr #assumes that p2_dev.f90 has been compiled with: !f2py -c p2_dev.f90 -m f4
import time
from sklearn.neural_network import MLPClassifier
# May also use scipy, scikit-learn, and time modules as needed


def read_data(tsize=15000):
    """Read in image and label data from data.csv.
    The full image data is stored in a 784 x 20000 matrix, X
    and the corresponding labels are stored in a 20000 element array, y.
    The final 20000-tsize images and labels are stored in X_test and y_test, respectively.
    X,y,X_test, and y_test are all returned by the function.
    You are not required to use this function.
    """
    print("Reading data...") #may take 1-2 minutes
    Data=np.loadtxt('data.csv',delimiter=',')
    Data =Data.T
    X,y = Data[:-1,:]/255.,Data[-1,:].astype(int) #rescale the image, convert the labels to integers between 0 and M-1)
    Data = None
    tsize=15000

    # Extract testing data
    X_test = X[:,tsize:]
    y_test = y[tsize:]
    print("processed dataset")
    return X,y,X_test,y_test
#----------------------------

def clr_test(X,y,X_test,y_test,bnd=1.0,l=0.0,input=(None)):
    """Train CLR model with input images and labels (i.e. use data in X and y),
    then compute and return testing error in test_error
    using X_test, y_test. The fitting parameters obtained via training
    should be returned in the 1-d array, fvec_f
    X: training image data, should be 784 x d with 1<=d<=15000
    y: training image labels, should contain d elements
    X_test,y_test: should be set as in read_data above
    bnd: Constraint parameter for optimization problem
    lambda: l2-penalty parameter for optimization problem
    input: tuple, set if and as needed
    """
    lr.lr_lambda = l

    y = y%2
    tsize=15000
    y_test = y_test%2
    Xtrain = X[:,:tsize]
    ytrain = y[:tsize]
    n,d = Xtrain.shape

    fvec = np.random.randn(n+1)*0.1 #initial fitting parameters
    #Add code to train CLR model and evaluate testing test_error
    #Initialise arrays and establish constraints
    lr.data_init(n,d)
    bnds = [(-bnd,bnd)]*len(fvec[:n])
    bnds.append((None,None))
    f = so.minimize(lr.clrmodel,fvec,args=(d,n),method='l-bfgs-b',jac=True,bounds=bnds)
    fvec_f = f.x #optimised fvec
    #Now to calculate test error.

    Y = np.zeros_like(y_test)
    for k in range(X_test.shape[1]):
        z = np.inner(fvec_f[:n],X_test[:,k])
        print('z = ',z)
        a = (1/(1+np.exp(z+fvec_f[n])))
        if a >=0.5:
            Y[k] = 1
        else:
            Y[k] = 0
    N = np.sum(np.array(Y)==y_test)
    test_error = 1-(N/len(y_test)) #Modify to store testing error; see neural network notes for further details on definition of testing error
    print('test error of clr = ',test_error*100,'%')
    return fvec_f,test_error
#--------------------------------------------

def mlr_test(X,y,X_test,y_test,m=3,bnd=1.0,l=0.0,input=(None)):
    """Train MLR model with input images and labels (i.e. use data in X and y), then compute and return testing error (in test_error)
    using X_test, y_test. The fitting parameters obtained via training should be returned in the 1-d array, fvec_f
    X: training image data, should be 784 x d with 1<=d<=15000
    y: training image labels, should contain d elements
    X_test,y_test: should be set as in read_data above
    m: number of classes
    bnd: Constraint parameter for optimization problem
    lambda: l2-penalty parameter for optimization problem
    input: tuple, set if and as needed
    """
    lr.lr_lambda = l

    y = y%m
    tsize=15000
    Xtrain = X[:,:tsize]
    ytrain = y[:tsize]
    n,d = Xtrain.shape

    y_test = y_test%m
    fvec = np.random.randn((m-1)*(n+1))*0.1 #initial fitting parameters
    #Add code to train MLR model and evaluate testing error, test_error
    lr.data_init(n,d)

    bnds = [(-bnd,bnd)]*(m-1)*n +[(None,None)]*(m-1)


    f = so.minimize(lr.mlrmodel,fvec,args=(n,d,m),method='l-bfgs-b',jac=True,bounds=bnds)

    fvec_f = f.x #Modify to store final fitting parameters after training
    W = np.zeros(((m-1),n))
    for r in range(n):
        W[:,r] = fvec_f[(m-1)*r:(m-1)*(r+1)]

    e = X_test.shape[1]
    Y = np.zeros_like(y_test)
    for i in range(e):
        z = np.matmul(W,X_test[:,i])
        z = np.insert(z,0,1.0)
        A = []
        for j in range(m):
            A.append(z[j]/sum(z))
        Y[k] = np.argmax(A)

    N = np.sum(Y==y_test)

    test_error = 1-(N/len(y_test)) #Modify to store testing error; see neural network notes for further details on definition of testing error
    print('test error of mlr = ',test_error*100,'%')
    return fvec_f,test_error
#--------------------------------------------

def lr_compare():
    """ Analyze performance of MLR and neural network models
    on image classification problem
    Add input variables and modify return statement as needed.
    Should be called from name==main section below
    """
    mlp = MLPClassifier(hidden_layer_sizes=(m,))

    mlp.fit(Xtrain,ytrain)
    print("Training set score: %f" % mlp.score(X_train, y_train))
    print("Test set score: %f" % mlp.score(X_test, y_test))
    return None
#--------------------------------------------

def display_image(X):
    #"""Displays image corresponding to input array of image data"""
    n2 = X.size
    n = np.sqrt(n2).astype(int) #Input array X is assumed to correspond to an n x n image matrix, M
    M = X.reshape(n,n)
    plt.figure()
    plt.imshow(M)
    return None
#--------------------------------------------
#--------------------------------------------


if __name__ == '__main__':
    #The code here should call analyze and generate the
    #figures that you are submitting with your code
    #output = lr_compare()

    X,y,X_test,y_test = read_data(tsize=15000)
    print('CLR Model Test Analysis:')
    clr_test(X,y,X_test,y_test,bnd=1.0,l=0.0,input=(None))
    print('------------------------------------------------------')
    print('MLR Model Test Analysis:')
    mlr_test(X,y,X_test,y_test,m=3,bnd=1.0,l=0.0,input=(None))
