# Support Vector Machine

## 1. Introduction

**Definition**

 In machine learning, support vector machines are supervised learning models with associated learning algorithms that analyse data used for classification and regression analysis.  

More formally, a support vector machine constructs a hyper-plane or set of hyper-planes in a high- or infinite-dimensional space, which can be used for classification, regression, or other tasks like outliers detection.  Intuitively, a good separation is achieved by the hyper-plane that has the largest distance to the nearest training-data point of any class (so-called functional margin), since in general the larger the margin the lower the generalization error of the classifier.

**Maximum Margin Classifier**

We have a set of $n$  dimensions vector $x$, and there two classes noted as $y$, and $y$ can be 1 or -1 for convenience. A linear classifier is to find a hyperplane, who has the form of  
$$
w^T + b =0
$$
so that it can separate these vectors into two sides, in one of them all vectors and 1, and in the other side, all the vectors are -1. 

<img src="https://upload.wikimedia.org/wikipedia/commons/thumb/b/b5/Svm_separating_hyperplanes_%28SVG%29.svg/512px-Svm_separating_hyperplanes_%28SVG%29.svg.png">

We first define functional margin as,  $\hat \gamma = y(w^Tx+b) = yf(x) $ multiply $y$ is to make sure functional margin is non-negative. And the distance between point and hyperplane $\gamma$ is geometrical margin. The relationship between two margin is $\hat \gamma = y \gamma = \frac{\hat \gamma}{\|w\|}$ . The only difference between two margin is a constant $\|w\|$ .  We use geometrical margin as the target function of maximum margin classifier.  So, the problem changes into a optimization question formula like below.
$$
max\ \hat \gamma \\
y_i(w^Tx_i+b) = \hat \gamma_i > \hat\gamma, \ i=1,...,n
$$
**Support Vector**

Simple version of what SVMs do without hard math:

- Find planes that correctly classify the training data
- Among all such planes, pick the one that has the greatest distance to the points closest to it

The closest points that identify this line are known as support vectors. 

**Kernel**

If the data is not linearly separable, we need a kernel function to project the data into high dimension. In high dimension, the data can be separated in linear way. 

Some common Kernel functions

- Linear kernel:  $k(x_i, x_j) = x_i^Tx_j$
- Polynomial kernel: $k(x_i, x_j) = (x_i^Tx_j)^d$  $d$ is the order
- Gaussian kernel: $k(x_i, x_j) = exp(-\frac{\|x_i-x_j\|^2}{2\sigma^2})$  $\sigma$   is the width
- Laplace kernel:  $k(x_i, x_j) = exp(-\frac{\|x_i-x_j\|^2}{\sigma})$  $\sigma > 0$
- Sigmoid kernel: $k(x_i, x_j) = tanh(\beta x^T_ix_j + \theta)$     $\beta>0,\  \theta<0$

## 2. Procedures

The algorithm to train SVM is  Sequential Minimal Optimization (SMO). is an algorithm for solving the quadratic programming (QP) problem

The SVM problem can be transformed into a quadratic programming problem like this.

<img src="http://images.cnblogs.com/cnblogs_com/jerrylead/201103/201103182042433212.png">

<img src="http://images.cnblogs.com/cnblogs_com/jerrylead/201103/20110318204256430.png">

[Refernece about SMO](http://cs229.stanford.edu/notes/cs229-notes3.pdf)

```python
# Python Code to implement SMO
def svm_train(x, y, kernel="linear"):
    if kernel == "linear":
        K = np.dot(x, x.T)
    elif kernel == "gaussianKernel":
        sigma = 3
        K = dist.cdist(x, x)
        K = np.exp(-K**2 / (2 * sigma * sigma))
    elif kernel == "polynomialKernel":
        d = 2
        K = np.dot(x, x.T)
        K = K**d
    m = y.shape[0]
    alpha = np.zeros(y.shape)
    b = 0
    E = np.zeros(y.shape)
    eta, C = 0, 0.5
    L, H = 0, 0
    tol, passes = 0.001, 0
    while passes < 5:
        num_changed_alphas = 0
        for i in range(m):
            k_i = K[:, i]
            k_i.shape = y.shape
            E[i, 0] = b + sum(alpha * y * k_i) - y[i, 0]
            if ((y[i, 0] * E[i, 0] < -tol) & (alpha[i, 0] < C)) \
                    | ((y[i, 0] * E[i, 0] > tol) & (alpha[i, 0] > 0)):
                # select j, make sure i \neq j
                j = np.random.randint(m)
                while j == i:
                    j = np.random.randint(m)
                # calculate error of j
                k_j = K[:, j]
                k_j.shape = y.shape
                E[j, 0] = b + sum(alpha * y * k_j) - y[j, 0]
                # save old alphas
                alpha_i_old, alpha_j_old = alpha[i, 0], alpha[j, 0]
                # calculate the L and H
                if y[i, 0] == y[j, 0]:
                    L = max(0, alpha[j, 0] + alpha[i, 0] - C)
                    H = min(C, alpha[j, 0] + alpha[i, 0])
                else:
                    L = max(0, alpha[j, 0] - alpha[i, 0])
                    H = min(C, C + alpha[j, 0] - alpha[i, 0])
                if L == H:
                    continue
                # compute eta
                eta = - K[i, i] - K[j, j] + 2 * K[i, j]
                if eta >= 0:
                    continue
                # update alpha_j
                alpha[j, 0] = alpha[j, 0] - y[j, 0] * (E[i, 0] - E[j, 0]) / eta
                # clip
                alpha[j, 0] = min(alpha[j, 0], H)
                alpha[j, 0] = max(alpha[j, 0], L)
                if abs(alpha[j, 0] - alpha_j_old) < tol:
                    alpha[j, 0] = alpha_j_old
                    continue
                # update alpha_i
                alpha[i, 0] = alpha[i, 0] + y[i, 0] * \
                    y[j, 0] * (alpha_j_old - alpha[j, 0])
                # update b
                b1 = b - E[i, 0] - y[i, 0] * (alpha[i, 0] - alpha_i_old) * \
                    K[i, j] - y[j, 0] * (alpha[j, 0] - alpha_j_old) * K[i, j]
                b2 = b - E[j, 0] - y[i, 0] * (alpha[i, 0] - alpha_i_old) * \
                    K[i, j] - y[j, 0] * (alpha[j, 0] - alpha_j_old) * K[j, j]
                if (alpha[i, 0] > 0) & (alpha[i, 0] < C):
                    b = b1
                elif (alpha[j, 0] > 0) & (alpha[j, 0] < C):
                    b = b2
                else:
                    b = (b1 + b2) / 2
                num_changed_alphas += 1
        if num_changed_alphas == 0:
            passes += 1
        else:
            passes = 0
    w = np.dot((alpha * y).T, x)
    return [w, b, alpha]
```



## 3. Results
**Linear Kernel**
<table><tr>
<td>![1](result\1.png)</td>
<td>![1](result\11..png)</td>
</tr></table>
<br><br><br><br>
**Polynomial Kernel**  
<table><tr>
<td>![1](result\2.png)</td>
<td>![1](result\22..png)</td>
</tr></table>
**Gaussian Kernel**  $\sigma = 0.1$
<table><tr>
<td>![1](result\3.png)</td>
<td>![1](result\33..png)</td>
</tr></table>
**Gaussian Kernel** $ \sigma = 3$
<table><tr>
<td>![1](result\4.png)</td>
<td>![1](result\44..png)</td>
</tr></table>

<br><br><br>
## 4. Conclusions

SVM is a very classic and useful algorithm. This Lab, we only complete its standard version. And it also has many better version for different problem.  The algorithm to train SVM is SMO. It is also a very efficiency algorithm. The hyper-parameters in the algorithm is very important which will influence the result of classification. 