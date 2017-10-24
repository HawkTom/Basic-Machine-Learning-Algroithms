# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 18:43:58 2017

@author: TomHawk
"""

import numpy as np
import matplotlib.pylab as plt
import K_mean

im = plt.imread("panda.png")
x = np.vstack(im)
cen, index = K_mean.KMEAN(x, 16)
for i in range(len(index)):
    x[i,:] = cen[index[i],:]
s = np.reshape(x, im.shape)
fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(10,5)) 
axs[0].imshow(im)
axs[1].imshow(s)
plt.show()



