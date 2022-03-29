import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import collections


class Perceptron:
    def __init__(self, max_iterations, number_of_inputs, learning_rate=0.1):
        self.max_iterations = max_iterations
        self.learning_rate = learning_rate
        self.weights = np.random.normal(size=number_of_inputs + 1)

    def predict(self, inputs):
        # spocitame vystup funkce pro dany vstup, vahy a prah
        activation = np.dot(inputs, self.weights[1:]) + self.weights[0]
        # spocitame aktivaci z funkce sgn do {-1,1} a to prevedeme na {0,1}
        return (np.sign(activation) + 1) / 2

    def fit(self, inputs, labels):
        for i in range(self.max_iterations):
            for x, y in zip(inputs, labels):
                prediction = self.predict(x)
                # aktualizujem vahy a prahy
                self.weights[1:] += self.learning_rate * (y - prediction) * x
                self.weights[0] += self.learning_rate * (y - prediction)


class MultiLayerPerceptron:
    def __init__(self, layer_sizes, alpha=0.1):
        self.layer_sizes = layer_sizes
        self.layers = []
        self.alpha = alpha

        # for (p, n) in zip(self.layer_sizes, self.layer_sizes[1:]):
        #    self.layers.append(np.random.normal(size=(p+1, n)))

        for i in range(0, len(self.layer_sizes) - 1):
            self.layers.append(np.random.normal(size=(self.layer_sizes[i] + 1, self.layer_sizes[i + 1])))

    def predict(self, X):
        activations = np.array(X)
        try:
            for layer in self.layers:
                activations = 1 / (1 + np.exp(np.dot(activations, layer[1:]) + layer[0]))
                # activations_1 = self._add_ones(activations) # pridame 1 misto prahu
                # activations = 1/(1+np.exp(np.dot(activations_1, layer)))
        except Exception as e:
            print("Activations:", activations)
            raise e

        return activations

    def fit(self, inputs, outputs):
        for x, y in zip(inputs, outputs):
            prediction = self.predict(x)
            error = (1/2) * np.sqrt(y-prediction)
            for layer in reversed(self.layers):
                for i in range(0, len(layer)-1):
                    layer[i] -= self.alpha * ()


        # je potreba spocitat vystup site
        # pak je potreba update vah pomoci backpropagace
        # zkuste si dopsat sami :P

    # pridame sloupecek jednicek k neuronum misto prahu
    def _add_ones(self, x):
        ones = np.ones(shape=(x.shape[0], 1))
        x = np.append(ones, x, axis=1)
        return x


def draw_boundary(model, inputs, labels):
    x_min = inputs[:, 0].min() - 1
    x_max = inputs[:, 0].max() + 1
    y_min = inputs[:, 1].min() - 1
    y_max = inputs[:, 1].max() + 1

    # vytvori obdelnikovou mrizku s vyse spocitanymi rozmery
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))

    # vykresleni rozhodovaci hranice -- kazdemu bodu se podle tho priradi barva
    predicted = model.predict(np.c_[xx.ravel(), yy.ravel()])
    predicted = predicted.reshape(xx.shape)

    plt.figure(figsize=(12, 8))
    plt.scatter(inputs[:, 0], inputs[:, 1], c=labels)
    plt.contourf(xx, yy, predicted, alpha=0.4)
    plt.show()


# inputs, labels = datasets.make_blobs(centers=2, n_samples=200)
# perc = Perceptron(100,2)
# perc.fit(inputs, labels)
#
# plt.figure(figsize=(12,8))
# plt.scatter(inputs[:,0], inputs[:,1], c=labels)
# plt.show()
#
# perc = Perceptron(100,2)
# perc.fit(inputs, labels)
#
#
# draw_boundary(perc, inputs, labels)

mlp = MultiLayerPerceptron([3, 5, 2])
mlp.predict(np.array([[1,2,3], [1,2,7]]))
