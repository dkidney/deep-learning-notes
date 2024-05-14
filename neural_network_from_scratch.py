from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import make_moons


def sigmoid(x):
	y = deepcopy(x)
	return 1 / (1 + np.exp(-y))


def sigmoid_prime(x):
	y = sigmoid(x)
	return y * (1 - y)


def tanh(x):
	y = deepcopy(x)
	return np.tan(y)


def tanh_prime(y):
	return 1 - tanh(y) ** 2


def relu(x):
	y = deepcopy(x)
	i = y < 0
	y[i] = 0
	return y


def relu_prime(x):
	y = deepcopy(x)
	i = y < 0
	y[i] = 0
	y[~i] = 1
	return y


def leaky_relu(x):
	y = deepcopy(x)
	i = y < 0.01 * y
	y[i] = 0.01 * y[i]
	return y


def leaky_relu_prime(x):
	y = deepcopy(x)
	i = y < 0
	y[i] = 0.01
	y[~i] = 1
	return y


class NN:
	def __init__(
			self,
			x_train,
			y_train,
			x_test,
			y_test,
	):
		self.X_train = x_train
		self.Y_train = y_train
		self.X_test = x_test
		self.Y_test = y_test

		self.m = None
		self.n0 = None
		self.n1 = None
		self.n2 = None
		self.W1 = None
		self.b1 = None
		self.W2 = None
		self.b2 = None
		self.Z1 = None
		self.A1 = None
		self.Z2 = None
		self.A2 = None
		self.dZ2 = None
		self.dW2 = None
		self.db2 = None
		self.dZ1 = None
		self.dW1 = None
		self.db1 = None
		self.g = None
		self.g_prime = None
		self.cost = None
		self.delta_cost = None
		self.Y_hat_train = None
		self.Y_hat_test = None
		self.alpha = None
		self.max_iterations = None
		# self.min_improvement = None
		self.activation_function = None

	def set_g(self):
		if self.activation_function == 'sigmoid':
			self.g = sigmoid
		elif self.activation_function == 'tanh':
			self.g = np.tanh
		elif self.activation_function == 'relu':
			self.g = relu
		elif self.activation_function == 'leaky_relu':
			self.g = leaky_relu

	def set_g_prime(self):
		if self.activation_function == 'sigmoid':
			self.g_prime = sigmoid_prime
		elif self.activation_function == 'tanh':
			self.g_prime = tanh_prime
		elif self.activation_function == 'relu':
			self.g_prime = relu_prime
		elif self.activation_function == 'leaky_relu':
			self.g_prime = leaky_relu_prime

	def initialize_params(self):
		self.W1 = np.random.randn(self.n1, self.n0) * 0.01
		# self.b1 = np.random.randn(self.n1, 1)
		self.b1 = np.zero(self.n1, 1)
		self.W2 = np.random.randn(self.n2, self.n1) * 0.01
		# self.b2 = np.random.randn(self.n2, 1)
		self.b2 = np.zero(self.n2, 1)

	def forward_propagation(self, X):
		Z1 = np.dot(self.W1, X) + self.b1
		A1 = self.g(Z1)
		Z2 = np.dot(self.W2, A1) + self.b2
		A2 = sigmoid(Z2)
		assert Z1.shape == (self.n1, X.shape[1])
		assert A1.shape == (self.n1, X.shape[1])
		assert Z2.shape == (self.n2, X.shape[1])
		assert A2.shape == (self.n2, X.shape[1])
		return Z1, A1, Z2, A2

	def back_propagation(self):
		self.dZ2 = self.A2 - self.Y_train
		self.dW2 = np.dot(self.dZ2, self.A1.T) / self.m
		self.db2 = np.sum(self.dZ2, axis=1, keepdims=True) / self.m
		self.dZ1 = np.dot(self.W2.T, self.dZ2) * self.g_prime(self.Z1)
		self.dW1 = np.dot(self.dZ1, self.X_train.T)
		self.db1 = np.sum(self.dZ1, axis=1, keepdims=True) / self.m
		assert self.dZ1.shape == self.Z1.shape
		assert self.dW1.shape == self.W1.shape
		assert self.db1.shape == self.b1.shape
		assert self.dZ2.shape == self.Z2.shape
		assert self.dW2.shape == self.W2.shape
		assert self.db2.shape == self.b2.shape

	def calculate_cost(self):
		loss = self.Y_train * np.log(self.A2) + (1 - self.Y_train) * np.log(1 - self.A2)
		sum_loss = np.sum(loss)
		return -sum_loss / self.m

	def predict_prob(self, X):
		Z1, A1, Z2, A2 = self.forward_propagation(X)
		return A2

	def predict_class(self, X):
		probs = self.predict_prob(X)
		return 1 * (probs > 0.5)

	@staticmethod
	def accuracy(Y, Y_hat):
		return np.mean(np.abs(Y_hat - Y))

	def build_model(self,
					n_hidden_nodes=3,
					activation_function='sigmoid',
					alpha=0.01,
					max_iterations=10000,
					# min_improvement=0.00001,
					print_metrics=False
					):

		self.m = self.X_train.shape[1]
		self.n0 = self.X_train.shape[0]
		self.n1 = n_hidden_nodes
		self.n2 = 1

		assert self.X_train.shape == (self.n0, self.m)
		assert self.Y_train.shape == (1, self.m)
		assert self.X_test.shape[0] == self.n0
		assert self.Y_test.shape[0] == 1

		self.activation_function = activation_function
		assert self.activation_function in ['sigmoid', 'tanh', 'relu', 'leaky_relu']

		self.alpha = alpha
		self.max_iterations = max_iterations
		# self.min_improvement = min_improvement

		self.set_g()
		self.set_g_prime()
		self.initialize_params()

		self.cost = []
		self.delta_cost = []

		print('-' * 80)
		print(f'activation function: {self.activation_function}')
		print(f'n hidden nodes: {self.n1}')
		print(f'learning rate: {self.alpha}')
		print(f'n iterations: {max_iterations}')

		# gradient descent -----------------------------------------------------------

		for i in range(self.max_iterations):
			self.Z1, self.A1, self.Z2, self.A2 = self.forward_propagation(self.X_train)
			self.back_propagation()
			self.W1 -= self.alpha * self.dW1
			self.b1 -= self.alpha * self.db1
			self.W2 -= self.alpha * self.dW2
			self.b2 -= self.alpha * self.db2
			self.cost.append(self.calculate_cost())
			self.delta_cost.append(self.cost[i] - self.cost[i-1] if (i > 1) else None)

			if print_metrics:
				if (i+1) % round(self.max_iterations / 10) == 0:
					print(f'Cost after iteration {i+1}: {round(self.cost[i], 5)} (delta: {round(self.delta_cost[i], 10)})')
				if (i+1) == self.max_iterations:
					print(f'stopping at iteration {i+1}: max iterations ({self.max_iterations}) reached')
				# if i > 1000 and self.delta_cost[i] > -self.min_improvement:
				# 	print(f'stopping at iteration {i}: delta cost ({self.delta_cost[i]}) > -min_improvement (-{self.min_improvement})')
				# 	break

		# predictions ----------------------------------------------------------------
		self.Y_hat_train = self.predict_class(self.X_train)
		self.Y_hat_test = self.predict_class(self.X_test)
		self.accuracy(self.Y_train, self.Y_hat_train)
		print(f'cost: {self.cost[-1]}')
		print(f"train accuracy: {100 - self.accuracy(self.Y_train, self.Y_hat_train) * 100} %")
		print(f"test accuracy: {100 - self.accuracy(self.Y_test, self.Y_hat_test) * 100} %")


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

nn = NN(x_train=X_train, y_train=Y_train, x_test=X_test, y_test=Y_test)

nn.build_model(n_hidden_nodes=10, alpha=0.1, max_iterations=10000)  # *
nn.build_model(activation_function='relu', n_hidden_nodes=10, alpha=0.1, max_iterations=10000)  # *
nn.build_model(activation_function='leaky_relu', n_hidden_nodes=10, alpha=0.1, max_iterations=10000)  # *


