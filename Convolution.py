import random
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.datasets import cifar10, mnist, fashion_mnist
from tensorflow.keras.layers import InputLayer
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPool2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Activation
from tensorflow.keras.activations import linear, relu
from tensorflow.nn import softmax

from cleverhans.tf2.attacks.fast_gradient_method import fast_gradient_method

from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions


def show_dataset_examples(X, y, labels):
    plt.figure(figsize=(12,12))
    for i in range(25):
        idx = random.randint(0, X.shape[0])
        plt.subplot(5, 5, i+1)
        plt.imshow(X[idx])
        plt.title(labels[y[idx]])
        plt.xticks([])
        plt.yticks([])
    plt.tight_layout()
    plt.show()


# (x_train, y_train), (x_test, y_test) = mnist.load_data()
# x_train = x_train.astype('float32')
# x_test = x_test.astype('float32')
# mnist_class_labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
# show_dataset_examples(x_train, y_train, mnist_class_labels)

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

# (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
# x_train = x_train.astype('float32')
# x_test = x_test.astype('float32')
# mnist_class_labels = ['t-shirt/top', 'trouser', 'pullover', 'dress', 'coat', 'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']

nb_classes = 10
input_shape = (28, 28, 1)

x_train = x_train.reshape((-1,) + input_shape)/255


model = tf.keras.models.Sequential([])
model.add(InputLayer(input_shape = input_shape))
model.add(Conv2D(filters=32, kernel_size=(5, 5), activation=relu))
model.add(MaxPool2D(strides=2, pool_size=(2, 2)))
model.add(Conv2D(filters=64, kernel_size=5, activation=relu))
model.add(MaxPool2D(strides=2, pool_size=(2, 2)))
model.add(Flatten(name='Flatten'))
model.add(Dense(units=30, activation=relu))
model.add(Dense(units=nb_classes, activation=linear, name='logits'))
model.add(Activation(activation=softmax))

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# model.fit(x_train, y_train, batch_size=128, epochs=2)
#
#
# x_test = x_test.reshape((-1,) + input_shape)
# model.evaluate(x_test/255, y_test)


### Matouci vzory
logits = tf.keras.Model(model.inputs, model.get_layer('logits').output)

results = []
# eps_vals = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]
eps_vals = [0.1]
# for eps in eps_vals:
#     # Replace clean example with adversarial example for adversarial training
#     x_fgm = fast_gradient_method(model, x_test/255, eps, np.inf)
#     r = model.evaluate(x_fgm, y_test)[1]
#     results.append(r)
#
#
# print(results)


x_train_fgm = fast_gradient_method(model, x_train, 0.4, np.inf)

model.fit(x_train_fgm, y_train, batch_size=128, epochs=2)

x_test = x_test.reshape((-1,) + input_shape)
model.evaluate(x_test/255, y_test)
