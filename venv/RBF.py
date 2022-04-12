import tensorflow as tf
from sklearn import datasets
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
import numpy as np
import random
import sys


class RBFNetwork():
    def __init__(self, input_dim, num_centers, output_dim):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_centers = num_centers
        self.centers = [np.random.uniform(-1, 1, input_dim) for i in range(num_centers)]
        self.beta = 1
        self.weights = None

    def calculate_activation(self, data):
        hidden_layer_values = np.zeros((data.shape[0], self.num_centers), float)
        for c_idx, c in enumerate(self.centers):
            for x_idx, x in enumerate(data):
                hidden_layer_values[x_idx, c_idx] = self.activation_fcn(c, x)
        return hidden_layer_values

    def activation_fcn(self, centers, data):
        return np.exp(-self.beta * np.linalg.norm(centers - data) ** 2)

    def fit(self, data, labels):
        # # zvol nahodne hodnoty pocatecnich centroidu
        # random_idx = np.random.permutation(data.shape[0])[:self.num_centers]
        # self.centers = [data[i, :] for i in random_idx]

        # zvol hodnoty pocatecnich centroidu pomoci kmeans
        self.centers = KMeans(self.num_centers).fit(data).cluster_centers_

        # spocitame aktivaci mezi vstupem a skrytou vrstvou
        hidden_layer_values = self.calculate_activation(data)

        # porovname skutecne a predikovane vystupy a aktualizujem vahy
        # pomoci pseudoinverzni matice, coze je vlastne vzorecek pro LR pro train vah
        self.weights = np.dot(np.linalg.pinv(hidden_layer_values), labels)

    def predict(self, data):
        hidden_layer_values = self.calculate_activation(data)
        labels = np.dot(hidden_layer_values, self.weights)
        return labels




iris = datasets.load_iris()
x, y = iris.data, iris.target
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)

onehotencoder = OneHotEncoder()
y_train_onehot = onehotencoder.fit_transform(y_train.reshape(-1, 1)).toarray()
y_test_onehot = onehotencoder.transform(y_test.reshape(-1, 1)).toarray()


rbf = RBFNetwork(4, 10, 3)
rbf.fit(x_train, y_train_onehot)
predicted = rbf.predict(x_test)
y_pred = np.argmax(predicted, axis=1)
accuracy = np.mean(y_pred == y_test)
print('Accuracy: ' + str(accuracy))

