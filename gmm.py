# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 14:32:38 2017

@author: TomHawk
"""
import numpy as np
import matplotlib.pyplot as plt
import time


def modelInit(data, nCluster):
    n = len(data)/nCluster
    mu, sigma = [], []
    for i in range(nCluster):
        x, y = int(i*n), int((i+1)*n-1)
        data_temp = data[x:y,] 
        mu_temp = np.mean(data_temp, axis=0)
        sigma_temp = np.cov(data_temp.T)
        mu.append(mu_temp)
        sigma.append(sigma_temp)
    return mu, sigma

def possibilityCalculate(data, mu, sigma):
    # print(sigma)
    A = 1/(((2*np.pi)**(len(data)/2))*(np.linalg.det(sigma)**(0.5)))
    x = np.dot((data-mu), np.linalg.inv(sigma))
    y = np.dot(x, (data-mu).T)
    p = A * np.exp(-0.5*y)
    return p

def GMM(data, nCluster):
    mu, sigma = modelInit(data, nCluster)
    mu_pre = [0 for i in range(nCluster)]
    stop = [0 for i in range(nCluster)]
    thresh = 0.0001 # stop condition
    nData =len(data)
    p = np.zeros((nData, nCluster))
    iter = 0
    while True:
       iter += 1 
       for i in range(nData):
           for j in range(nCluster):
               p[i,j] = possibilityCalculate(data[i,:], mu[j], sigma[j])
           pp = sum(p[i,])
           for j in range(nCluster):
               p[i, j] = p[i, j] / pp       
       for j in range(nCluster):
           total = np.zeros((len(data[i,:]), len(data[i,:])))
           for i in range(nData):
               row_vector = data[i,:]-mu[j]
               row_vector.shape = (len(row_vector), 1)
               col_vector = row_vector.transpose()
               total = total + p[i, j] * (col_vector*row_vector)
           sigma[j] = total/sum(p[:,j])
           mu_pre[j] = mu[j]
           mu[j] = np.dot(p[:,j].T, data)/sum(p[:,j])
           stop[j] = np.sqrt(np.sum(np.square(mu_pre[j] - mu[j])))
       if min(stop) <= thresh:
       # if iter >= 10:
           return p     
   
       

if __name__ == "__main__":
    start = time.time()
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(10,5)) 
    mu1, cov1 = [0, 0], [[0.1, 0], [0, 0.1]]
    mu2, cov2 = [1, 1], [[0.1, 0], [0, 0.1]]
    mu3, cov3 = [0, 1], [[0.1, 0], [0, 0.1]]
    data1 = np.random.multivariate_normal(mu1, cov1, 100)
    data2 = np.random.multivariate_normal(mu2, cov2, 100)
    data3 = np.random.multivariate_normal(mu3, cov3, 100)
    x, y = data1.T
    axs[0].plot(x, y, 'ro', markerfacecolor='none')
    x, y = data2.T
    axs[0].plot(x, y, 'go', markerfacecolor='none')
    x, y = data3.T
    axs[0].plot(x, y, 'bo', markerfacecolor='none')
    data = np.vstack([data1, data2, data3])
    possibility = GMM(data, 3)
    for i in range(len(data)):
        index = np.argmax(possibility[i,:])
        if index == 0:
            axs[1].plot(data[i,0], data[i,1], 'r.')
        elif index == 1:
            axs[1].plot(data[i,0], data[i,1], 'g.')
        elif index == 2:
            axs[1].plot(data[i,0], data[i,1], 'k.')
    end = time.time()
    print(end-start)
    plt.show()