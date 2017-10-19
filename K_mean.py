import numpy as np
import matplotlib.pyplot as plt


def dataGenerate(center=None):
    mu1, cov1 = [0, 0], [[0.1, 0], [0,0.1]]
    mu2, cov2 = [2, 2], [[0.1, 0], [0,0.1]]
    mu3, cov3 = [-2, -2], [[0.1, 0], [0,0.1]]
    data1 = np.random.multivariate_normal (mu1,cov1 ,100)
    data2 = np.random.multivariate_normal (mu2,cov2 ,100)
    data3 = np.random.multivariate_normal (mu3,cov3 ,100)
    data = np.vstack([data1,data2,data3])
    return data

def data_plot(data, centers = None):
    if centers != None:
        x, y = centers.T
        plt.plot(x,y,'x',color='b')    
    x, y = data.T
    plt.plot(x,y,'.',color='r')
    plt.axis('equal')
    plt.show()

def initCenter(data, nCluster):
    n = data.shape[0]
    index = np.random.permutation(n)
    centers = data[index[0:nCluster]]
    return centers

def updateCenter(cluster):
    centers = []
    for center in cluster:
        avr = np.average(cluster[center], axis=0)
        centers.append(avr)
    return np.vstack(centers)
    
def distanceCaculate(centers, point):
    dists = np.sqrt(np.sum(np.square(point - centers),axis=1))
    return np.argmin(dists), np.min(dists)

def point_to_cluster(centers, datas):
    cluster, sumDist = {}, 0
    for point in datas:
        index, dist = distanceCaculate(centers, point)
        sumDist += dist
        if index not in cluster:
            cluster[index] = [point]
        else:
            cluster[index].append(point)
    return cluster, sumDist


def KMEAN(data, nCluster):
    centers = initCenter(data, nCluster)
    delta_error = 1
    old_sumDist = float('Inf')
    while delta_error > 0.01:
        cluster, sumDist = point_to_cluster(centers, data)        
        delta_error = old_sumDist - sumDist
        #print(delta_error)
        old_sumDist = sumDist
        centers = updateCenter(cluster)
        
    return centers
if __name__ == "__main__":
    mu1, cov1 = [0, 0], [[0.1, 0], [0,0.1]]
    mu2, cov2 = [2, 2], [[0.1, 0], [0,0.1]]
    mu3, cov3 = [-2, -2], [[0.1, 0], [0,0.1]]
    data1 = np.random.multivariate_normal (mu1,cov1 ,100)
    data2 = np.random.multivariate_normal (mu2,cov2 ,100)
    data3 = np.random.multivariate_normal (mu3,cov3 ,100)
    x, y = data1.T
    plt.plot(x,y,'ro',markerfacecolor='none')
    x, y = data2.T
    plt.plot(x,y,'go',markerfacecolor='none')
    x, y = data3.T
    plt.plot(x,y,'bo',markerfacecolor='none')
    
    data = np.vstack([data1,data2,data3])    
    data = dataGenerate()
    centers = KMEAN(data,3)
    
    x, y = centers.T
    plt.plot(x,y,'^',markersize=10,markerfacecolor='k')