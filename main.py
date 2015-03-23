import numpy as np
import theano
import theano.tensor as T
from sklearn.cluster import MiniBatchKMeans as KMeans
from LDNN import LDNN
import matplotlib.pyplot as plt


def create_data(dataset='moons'):
    from sklearn import datasets
    from sklearn.preprocessing import StandardScaler
    noisy_moons = datasets.make_moons(n_samples=1500, noise=.1)
    noisy_circles = datasets.make_circles(n_samples=1500, factor=.5,
                                          noise=.1)
    if dataset == 'moons':
        data, labels = noisy_moons
    else:
        data, labels = noisy_circles
    data = StandardScaler().fit_transform(data)
    return data, labels


def visualize_initialization(data, labels):
    plt.scatter(data[:, 0], data[:, 1])
    colors = np.array([x for x in 'bgrcmykbgrcmykbgrcmykbgrcmyk'])
    colors = np.hstack([colors] * 20)
    data_class_0 = np.asarray([d for (d, l) in zip(data, labels) if l == 0])
    data_class_1 = np.asarray([d for (d, l) in zip(data, labels) if l == 1])
    M_means = KMeans(n_clusters=M, max_iter=1000,
                     batch_size=100, n_init=10)
    N_means = KMeans(n_clusters=N, max_iter=1000,
                     batch_size=100, n_init=10)
    M_means.fit(data_class_0)
    N_means.fit(data_class_1)
    cent_class_0 = M_means.cluster_centers_
    cent_class_1 = N_means.cluster_centers_
    plt.scatter(cent_class_0[:, 0], cent_class_0[:, 1], facecolors='none',
                edgecolors=colors.tolist(),
                s=7000 * np.mean(abs(cent_class_0[:, 0] - cent_class_0[:, 1])))
    plt.scatter(cent_class_0[:, 0], cent_class_0[:, 1], facecolors='none',
                edgecolors=(214 / 255., 39 / 255., 40 / 255.),
                s=7000 * np.mean(abs(cent_class_0[:, 0] - cent_class_0[:, 1])))
    plt.scatter(cent_class_1[:, 0], cent_class_1[:, 1], facecolors='none',
                edgecolors=(31 / 255., 119 / 255., 180 / 255.),
                s=7000 * np.mean(abs(cent_class_1[:, 0] - cent_class_1[:, 1])))
    plt.scatter(cent_class_0[:, 0], cent_class_0[:, 1],
                color=(214 / 255., 39 / 255., 40 / 255.), s=100)
    plt.scatter(cent_class_1[:, 0], cent_class_1[:, 1],
                color=(31 / 255., 119 / 255., 180 / 255.), s=100)
    plt.show()


def visualize_decision_boundary(data, pred_function):
    xx, yy = np.mgrid[-3:3:.01, -3:3:.01]
    grid = np.c_[xx.ravel(), yy.ravel()]
    out = pred_function(grid)
    out = out.reshape(xx.shape)
    plt.contourf(xx, yy, out)
    plt.axis('off')
    plt.scatter(data[:, 0], data[:, 1])
    plt.show()


M = 3
N = 3
K = 2
data, labels = create_data(dataset='moons')
visualize_initialization(data, labels)

X = T.matrix('X')
y = T.ivector('y')
ldnn = LDNN(M, N, K)
pred, params = ldnn.classify(X, approx='product')
cost = T.mean((pred - y) ** 2)
grads = T.grad(cost, params)
learning_rate = 5
updates = [(param_i, param_i - learning_rate * grad_i)
           for param_i, grad_i in zip(params, grads)]
train_function = theano.function([X, y], cost, updates=updates,
                                 allow_input_downcast=True)
predict_function = theano.function([X], pred)

ldnn.initialize(data, labels)
visualize_decision_boundary(data, predict_function)

for i in range(2000):
    cost = train_function(data, labels)
    if i % 100 == 0:
        print cost
        visualize_decision_boundary(data, predict_function)
