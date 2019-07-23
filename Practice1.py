# This file contains the code that implements the gradient descent with regularization.
# We create a function myfunc(), then create a noisy data y_train . Our task is to restore
# the function myfunc() from this data (x_train,y_train). Some pieces of code are missing,
# your task is to write them yourself.
# Often, ML is fine tuning of the free parameters. Find the best values ​​for
# learning rate alpha and regularization parameter Lambda.
#
# Bonus.
# Build learning curves for test and train data. On the abscis axis, set aside
# the maximum degree of the your model, and on the ordinate axis the minimum value of
# the cost function J. Determine the extent of your model that best approximates the target function.


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def myfunc(xx):
    return xx + 10 # your function (try np.sin(xx)  )

def generate_set(N=500, n_train=10, std=0.5):
    x1=np.linspace(-2*np.pi, 2*np.pi, num=N)
    y1=myfunc(x1)
    dall=pd.DataFrame()
    dall['x']=x1
    dall['y']=y1
    x_train= np.sort(np.random.choice(x1, size=n_train, replace=True))

    y_train = myfunc(x_train) + np.random.normal(0, std, size=x_train.shape[0])
    data=pd.DataFrame()
    data['xt'] = x_train
    data['yt'] = y_train
    return dall, data

dall, data = generate_set()

plt.figure(0)
margin = 0.3
plt.figure(figsize=(10,5))
plt.plot(dall.x, dall.y, 'b--', alpha=0.5, label='target')
plt.scatter(data.xt, data.yt, 20, 'g', 'o', alpha=0.8, label='data')
plt.xlim(data.xt.min() - margin, data.xt.max() + margin)
plt.ylim(data.yt.min() - margin, data.yt.max() + margin)
plt.legend(loc='upper right', prop={'size': 10})
plt.title('target line and noised data')
plt.xlabel('x')
plt.ylabel('y')
plt.show()


def mapFeature(X1,degree=1):
    out=pd.DataFrame()
    m = len(X1)

    # mapFeature(X1, degree ) maps the input feature
    # Returns a new feature array with more features, comprising of
    # X1^0, X1^1, X1^2,...., X1^degree

    # your code here

    return out

def computeCost(X, y,w,Lambda):
    m=len(y)

    #Compute cost for linear regression
    # J = COMPUTECOST(X, y, w, Lambda) computes the cost of using w as the
    # parameter for linear regression to fit the data points in X and y
    # Lambda is the parameter of regulirization
    
    w0 = np.copy(w)
    w0[0] = 0

    # your code here
    # MSE=
    J = Lambda/(2*m) * np.dot(w0.T,w0) # + MSE

    return J


def gradientDescent(X, y, w, alpha, num_iters, Lambda=0):
    # performs gradient descent to learn w
    # GRADIENTDESENT(X, y, theta, alpha, num_iters, Lambda) updates w by
    # taking num_iters gradient steps with learning rate alpha and parameter
    # of regulirization Lambda

    y = y.values.reshape((y.shape[0], 1))
    m = len(y) 
    J_history = np.zeros((num_iters, 1))
    

    for i in range(num_iters):
        # your code here
        J_history[i] = computeCost(X, y, w,Lambda)

    return [w, J_history]



degree=1
df=mapFeature(data.xt,degree)
X=mapFeature(dall.x,degree)

n=df.shape[1]
w0= np.ones((n, 1))

iterations = 500
alpha = 10e-2
[w, J_history] = gradientDescent(df, data.yt, w0, alpha, iterations,Lambda=1)

st=100
plt.plot(range(st,iterations), J_history[st:], 'g-')
plt.show()

h=np.dot(w.T, X.T).T

xh=dall.x.values.reshape((dall.x.shape[0], 1))

margin = 0.3
plt.figure(1)
plt.figure(figsize=(10,5))
plt.plot(dall.x, dall.y, 'b--', alpha=0.5, label='target')
plt.plot(xh, h, 'r', alpha=0.5, label='hypo')
plt.scatter(data.xt, data.yt, 20, 'g', 'o', alpha=0.8, label='data')
plt.xlim(data.xt.min() - margin, data.xt.max() + margin)
plt.ylim(data.yt.min() - margin, data.yt.max() + margin)
plt.legend(loc='upper right', prop={'size': 10})
plt.title('target line and noised data')
plt.xlabel('x')
plt.ylabel('y')

plt.show()


# Accurate solution

w_ac = # your code here
y_hat = # your code here

plt.figure(2)
plt.figure(figsize=(10,5))
plt.plot(dall.x, dall.y, 'b--', alpha=0.5, label='target')
plt.plot(dall.x, y_hat, 'r', alpha=0.5, label='hypo')
plt.scatter(data.xt, data.yt, 20, 'g', 'o', alpha=0.8, label='data')
plt.xlim(data.xt.min() - margin, data.xt.max() + margin)
plt.ylim(data.yt.min() - margin, data.yt.max() + margin)
plt.legend(loc='upper right', prop={'size': 10})
plt.title('target line and noised data')
plt.xlabel('x')
plt.ylabel('y')

plt.show()

