import numpy as np
import matplotlib.pyplot as plt


def dataGenerate(center=None):
    mu1, cov1 = [0, 0], [[0.1, 0], [0, 0.1]]
    mu2, cov2 = [2, 2], [[0.1, 0], [0, 0.1]]
    mu3, cov3 = [-2, -2], [[0.1, 0], [0, 0.1]]
    data1 = np.random.multivariate_normal(mu1, cov1, 100)
    data2 = np.random.multivariate_normal(mu2, cov2, 100)
    data3 = np.random.multivariate_normal(mu3, cov3, 100)
    data = np.vstack([data1, data2, data3])
    return data


def initCenter(data, nCluster):
    n = data.shape[0]
    index = np.random.permutation(n)
    centers = data[index[0:nCluster]]
    return centers


def updateCenter(cluster, cs, method):
    centers = []
    if method == 'k-mean':
        for center in cluster:
            avr = np.average(cluster[center], axis=0)
            centers.append(avr)
    elif method == 'k-medoids':
        for center in cluster:
            points = np.vstack(cluster[center])
            minDist = sum(sum(abs(cs[center] - points)))
            minPoint = cs[center]
            for point in cluster[center]:
                temp_sum = sum(sum(abs(point - points)))
                if temp_sum < minDist:
                    minPoint = point
                    minDist = temp_sum
            centers.append(minPoint)
    return np.vstack(centers)


def distanceCaculate(centers, point):
    dists = np.sqrt(np.sum(np.square(point - centers), axis=1))
    return np.argmin(dists), np.min(dists)


def point_to_cluster(centers, datas):
    cluster, sumDist, data_index = {}, 0, []
    for point in datas:
        index, dist = distanceCaculate(centers, point)
        data_index.append(index)
        sumDist += dist
        if index not in cluster:
            cluster[index] = [point]
        else:
            cluster[index].append(point)
    return cluster, sumDist, data_index


def KMEAN(data, nCluster, method='k-mean'):
    centers = initCenter(data, nCluster)
    delta_error = 1
    old_sumDist = float('Inf')
    while True:
        cluster, sumDist, data_index = point_to_cluster(centers, data)
        delta_error = old_sumDist - sumDist
        if delta_error > 0.001:
            return centers, data_index
        # print(delta_error)
        old_sumDist = sumDist
        centers = updateCenter(cluster, centers, method)



if __name__ == "__main__":
    mu1, cov1 = [0, 0], [[0.1, 0], [0, 0.1]]
    mu2, cov2 = [2, 2], [[0.1, 0], [0, 0.1]]
    mu3, cov3 = [-2, -2], [[0.1, 0], [0, 0.1]]
    data1 = np.random.multivariate_normal(mu1, cov1, 100)
    data2 = np.random.multivariate_normal(mu2, cov2, 100)
    data3 = np.random.multivariate_normal(mu3, cov3, 100)
    x, y = data1.T
    plt.plot(x, y, 'ro', markerfacecolor='none')
    x, y = data2.T
    plt.plot(x, y, 'go', markerfacecolor='none')
    x, y = data3.T
    plt.plot(x, y, 'bo', markerfacecolor='none')
    data = np.vstack([data1, data2, data3])
    data = dataGenerate()
    centers, index = KMEAN(data, 3, 'k-medoids')
    print(centers,index)
    #centers = KMEAN(data, 3)
    x, y = centers.T
    plt.plot(x, y, '^', markersize=10, markerfacecolor='k')
    plt.show()
