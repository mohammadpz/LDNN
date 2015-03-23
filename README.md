# LDNN
Theano implementation of Logistic Disjunctive Normal Networks.

Choose between two datasets of ```noisy_moons``` or ```noisy_circles``` by choosing ```data, labels = create_data(dataset='moons')``` or ```create_data(dataset='circles')```.

Choose between product approximation or min_max approximation of logical gates by choosing ```ldnn.classify(X, approx='product')``` or ```approx='min_max'```

Reference
=========
Sajjadi, Mehdi, Mojtaba Seyedhosseini, and Tolga Tasdizen. "Disjunctive Normal Networks." arXiv preprint arXiv:1412.8534 (2014).

http://arxiv.org/abs/1412.8534

Notice
======
This code is distributed without any warranty, express or implied.