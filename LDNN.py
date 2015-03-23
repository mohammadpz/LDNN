import numpy as np
import theano
import theano.tensor as T
from sklearn.cluster import MiniBatchKMeans as KMeans


class LDNN(object):
    def __init__(self, M, N, K):
        self.M = int(M)
        self.N = int(N)
        self.K = int(K)
        self.W = theano.shared(
            value=np.asarray(
                np.random.uniform(
                    size=(self.N, self.K, self.M),
                    low=-.01, high=.01),
                dtype=theano.config.floatX),
            name='W')  # N x K x M
        self.b = theano.shared(
            value=np.zeros((self.N, self.M)),
            name='b')  # N x M
        self.params = [self.W, self.b]

    def classify(self, X, approx='product'):
        # X: n x k
        H = T.nnet.sigmoid(T.dot(X, self.W) + self.b)  # n x N x M
        if approx == 'product':
            ANDs = T.prod(H, axis=2)  # n x N
            OR = 1 - T.prod(1 - ANDs, axis=1)  # n
        elif approx == 'min_max':
            ANDs = T.min(H, axis=2)  # n x N
            OR = T.max(ANDs, axis=1)  # n
        else:
            raise NotImplementedError
        return OR, self.params

    def initialize(self, data, labels):
        data_class_0 = [d for (d, l) in zip(data, labels) if l == 0]
        data_class_1 = [d for (d, l) in zip(data, labels) if l == 1]
        M_means = KMeans(n_clusters=self.M, max_iter=1000,
                         batch_size=100, n_init=10)
        N_means = KMeans(n_clusters=self.N, max_iter=1000,
                         batch_size=100, n_init=10)
        M_means.fit(data_class_0)
        N_means.fit(data_class_1)
        cent_class_0 = M_means.cluster_centers_
        cent_class_1 = N_means.cluster_centers_
        W_init = np.zeros((self.N, self.K, self.M))
        b_init = np.zeros((self.N, self.M))
        for i in range(self.N):
            for j in range(self.M):
                W_init[i, :, j] = ((cent_class_1[i] - cent_class_0[j]) /
                                   abs(cent_class_1[i] - cent_class_0[j]))
                b_init[i, j] = np.dot(W_init[i, :, j], 0.5 * (cent_class_0[j] +
                                                              cent_class_1[i]))
        self.W.set_value(W_init)
        self.b.set_value(b_init)
