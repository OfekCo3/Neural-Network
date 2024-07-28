import random
import numpy as np


class QuadraticCost:
    @staticmethod
    def fn(a, y):
        return 0.5 * np.linalg.norm(a - y) ** 2

    @staticmethod
    def delta(z, a, y):
        return (a - y) * sigmoid_prime(z)


class CrossEntropyCost:
    @staticmethod
    def fn(a, y):
        return np.sum(np.nan_to_num(-y * np.log(a) - (1 - y) * np.log(1 - a)))

    @staticmethod
    def delta(z, a, y):
        return a - y


def he_initialization(sizes):
    """He initialization for weights"""
    return [np.random.randn(y, x) * np.sqrt(2. / x) for x, y in zip(sizes[:-1], sizes[1:])]


def default_initialization(sizes):
    """default initialization for weights"""
    return [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]


def sigmoid(z):
    """The sigmoid function."""
    z = np.clip(z, -700, 700)  # prevent overflow in exp
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z) * (1 - sigmoid(z))


class NeuralNetwork:
    def __init__(self, sizes, cost='quadratic'):
        input_neurons = sizes[0]
        hidden_neurons = sizes[1]
        num_hidden_layers = sizes[2]
        output_neurons = sizes[3]

        # Create the sizes list for the actual structure
        actual_sizes = [input_neurons] + [hidden_neurons] * num_hidden_layers + [output_neurons]
        self.num_layers = len(actual_sizes)
        self.sizes = actual_sizes

        self.biases = [np.random.randn(y, 1) for y in actual_sizes[1:]]
        self.weights = he_initialization(actual_sizes)

        # Set the cost function based on the input string
        if cost == 'quadratic':
            self.cost = QuadraticCost
        elif cost == 'cross_entropy':
            self.cost = CrossEntropyCost
        else:
            raise ValueError("Invalid cost function. Use 'quadratic' or 'cross_entropy'.")

    def feedforward(self, a):
        """Return the output of the network if 'a' is input."""
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a) + b)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta):
        """Train the neural network using mini-batch stochastic gradient descent."""
        n = len(training_data)
        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [training_data[k:k + mini_batch_size] for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            print(f"Epoch {j} complete")

    def update_mini_batch(self, mini_batch, eta):
        """Update the network’s weights and biases by applying gradient descent using backpropagation to a single mini batch."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w - (eta / len(mini_batch)) * nw for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - (eta / len(mini_batch)) * nb for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        """Return a tuple (nabla_b, nabla_w) representing the gradient for the cost function C_x."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x]  # list to store all the activations, layer by layer
        zs = []  # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # backward pass
        delta = self.cost.delta(zs[-1], activations[-1], y)
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l + 1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l - 1].transpose())
        return nabla_b, nabla_w

    def evaluate(self, test_data):
        """Return the number of test inputs for which the neural network outputs the correct result."""
        test_results = [(np.argmax(self.feedforward(x)), y) for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def cost_derivative(self, output_activations, y):
        """Return the vector of partial derivatives ∂C_x / ∂a for the output activations."""
        return output_activations - y

    def fit(self, X, y, epochs=30, mini_batch_size=10, eta=3.0):
        """Train the neural network with the given data."""
        training_data = list(zip(X, y))
        self.SGD(training_data, epochs, mini_batch_size, eta)

    def predict(self, X):
        """Calculate the output of the network for the given input X."""
        results = [np.argmax(self.feedforward(x)) for x in X]
        return results

    def score(self, X, y):
        """Calculate the percentage of samples in X that the trained network correctly classifies."""
        test_data = list(zip(X, y))
        correct_count = self.evaluate(test_data)
        return correct_count / len(X)