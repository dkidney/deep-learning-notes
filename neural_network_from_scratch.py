import copy

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import make_moons


class NN:
	def __init__(
		self,
		x_train,
		y_train,
		x_test,
		y_test,
		n_hidden_nodes=3,
		activation_function='sigmoid',
		alpha=0.01,
		max_iterations=10000,
		print_metrics=True
	):
		self.x_train = x_train
		self.y_train = y_train
		self.x_test = x_test
		self.y_test = y_test
		self.activation_function = activation_function
		self.alpha = alpha
		self.max_iterations = max_iterations
		self.print_metrics = print_metrics
		self.m = x_train.shape[1]
		self.n0 = x_train.shape[0]
		self.n1 = n_hidden_nodes
		self.n2 = 1

		assert self.x_train.shape == (self.n0, self.m)
		assert self.y_train.shape == (1, self.m)
		assert self.activation_function in ['sigmoid', 'tanh', 'relu', 'leaky_relu']

	@staticmethod
	def sigmoid(z):
		return 1 / (1 + np.exp(-z))

	@staticmethod
	def relu(z):
		assert NotImplementedError

	@staticmethod
	def leaky_relu(z):
		assert NotImplementedError

	def set_g(self):
		if self.activation_function == 'sigmoid':
			self.g = self.sigmoid
		elif self.activation_function == 'tanh':
			self.g = np.tanh
		elif self.activation_function == 'relu':
			self.g = self.relu
		elif self.activation_function == 'leaky_relu':
			self.g = self.leaky_relu

	@staticmethod
	def sigmoid_prime(z):
		assert NotImplementedError

	@staticmethod
	def tanh_prime(z):
		assert NotImplementedError

	@staticmethod
	def relu_prime(z):
		assert NotImplementedError

	@staticmethod
	def leaky_relu_prime(z):
		assert NotImplementedError

	def set_g_prime(self):
		if self.activation_function == 'sigmoid':
			self.g_prime = self.sigmoid_prime
		elif self.activation_function == 'tanh':
			self.g_prime = self.tanh_prime
		elif self.activation_function == 'relu':
			self.g_prime = self.relu_prime
		elif self.activation_function == 'leaky_relu':
			self.g_prime = self.leaky_relu_prime

	def initialize_params(self):
		self.W1 = np.random.randn((self.n1, self.n0))
		self.b1 = np.random.randn((self.n1, 1))
		self.W2 = np.random.randn((self.n2, self.n1))
		self.b2 = np.random.randn((self.n2, 1))

	def forward_propagation(self):
		pass

	def back_propagation(self):
		pass

	def predict(self):
		pass

	def build_model(self):
		self.set_g()
		self.set_g_prime()
		self.initialize_params()



m_train = 1000
m_test = 200

np.random.seed(42)
X_train_orig, Y_train_orig = make_moons(m_train, noise=0.20)
X_test_orig, Y_test_orig = make_moons(m_test, noise=0.20)

# pd.DataFrame(X_train_orig, columns=['x1', 'x2']).to_csv('data/moons/X_train.csv', index=False)
# pd.DataFrame(X_test_orig, columns=['x1', 'x2']).to_csv('data/moons/X_test.csv', index=False)
# pd.DataFrame(Y_train_orig, columns=['y']).to_csv('data/moons/Y_train.csv', index=False)
# pd.DataFrame(Y_test_orig, columns=['y']).to_csv('data/moons/Y_test.csv', index=False)

Y_train = Y_train_orig.reshape((1, m_train))
Y_test  = Y_test_orig.reshape((1, m_test))
X_train = X_train_orig.T
X_test  = X_test_orig.T

nn = NN(
	x_train=X_train,
	y_train=Y_train,
	x_test=X_test,
	y_test=Y_test,
)

# plt.scatter(X_train_orig[:, 0], X_train_orig[:, 1], s=40, c=Y_train_orig, cmap=plt.cm.Spectral)
# plt.show()

# plt.scatter(X_train[0, :], X_train[1, :], s=40, c=Y_train[0, :], cmap=plt.cm.Spectral)
# plt.show()

def initialize_with_zeros(dim):
	w = np.zeros((dim, 1), dtype=float)
	b = 0.0
	return w, b


def propagate(w, b, X, Y):
	m = X.shape[1]
	A = sigmoid(np.dot(w.T, X) + b)
	cost = -np.sum(Y * np.log(A) + (1 - Y) * np.log(1 - A)) / m
	dw = np.dot(X, (A - Y).T) / m
	db = np.sum(A - Y) / m
	cost = np.squeeze(np.array(cost))
	grads = {"dw": dw,
			 "db": db}
	return grads, cost


def optimize(w, b, X, Y, num_iterations=100, learning_rate=0.009, print_cost=False):
	w = copy.deepcopy(w)
	b = copy.deepcopy(b)
	costs = []
	for i in range(num_iterations):
		grads, cost = propagate(w, b, X, Y)
		dw = grads["dw"]
		db = grads["db"]
		w = w - learning_rate * dw
		b = b - learning_rate * db
		if i % 100 == 0:
			costs.append(cost)
			if print_cost:
				print("Cost after iteration %i: %f" % (i, cost))
	params = {"w": w,
			  "b": b}
	grads = {"dw": dw,
			 "db": db}
	return params, grads, costs


def predict(w, b, X):
	m = X.shape[1]
	Y_prediction = np.zeros((1, m))
	w = w.reshape(X.shape[0], 1)
	A = sigmoid(np.dot(w.T, X) + b)
	for i in range(A.shape[1]):
		if A[0, i] > 0.5:
			Y_prediction[0, i] = 1
		else:
			Y_prediction[0, i] = 0
	return Y_prediction


def model(X_train, Y_train, X_test, Y_test, num_iterations=2000, learning_rate=0.5, print_cost=False):
	w, b = initialize_with_zeros(X_train.shape[0])
	params, grads, costs = optimize(w, b, X_train, Y_train, num_iterations=num_iterations, learning_rate=learning_rate,
									print_cost=print_cost)
	w = params["w"]
	b = params["b"]
	Y_prediction_test = predict(w, b, X_test)
	Y_prediction_train = predict(w, b, X_train)
	if print_cost:
		print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
		print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))
	d = {"costs": costs,
		 "Y_prediction_test": Y_prediction_test,
		 "Y_prediction_train": Y_prediction_train,
		 "w": w,
		 "b": b,
		 "learning_rate": learning_rate,
		 "num_iterations": num_iterations}
	return d


logistic_regression_model = model(X_train, Y_train, X_test, Y_test, num_iterations=2000, learning_rate=0.005, print_cost=True)

costs = np.squeeze(logistic_regression_model['costs'])
costs

plt.plot(costs)
plt.ylabel('cost')
plt.xlabel('iterations (per hundreds)')
plt.title("Learning rate =" + str(logistic_regression_model["learning_rate"]))
plt.show()





