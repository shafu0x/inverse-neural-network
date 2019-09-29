import numpy as np

from scipy.special import softmax

import keras
from keras.datasets import mnist
from keras.layers import Input, Dense
from keras.models import Model
from keras.utils import to_categorical

from keras.layers.advanced_activations import LeakyReLU

N_OUTPUT_NODES = 10
OPTIMIZER = 'rmsprop'
LOSS = 'mean_squared_error'

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(-1, 28 * 28)
x_test = x_test.reshape(-1, 28 * 28)

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

inputs = Input(shape=(784, ))
output_1 = Dense(64)(inputs)
a1 = LeakyReLU()(output_1)
output_2 = Dense(64)(a1)
a2 = LeakyReLU()(output_2)
predictions = Dense(N_OUTPUT_NODES, activation='linear')(a2)

model = Model(inputs=inputs, outputs=predictions)
model.summary()
model.compile(optimizer=OPTIMIZER, loss=LOSS, metrics=['accuracy'])


def train_and_eval():
    """
    Train and evaluate the network.
    Returns:
        acc (float): The accuracy archived after training.
    """
    model.fit(x_train, y_train)

    layer_name = 'dense_3'
    intermediate_layer_model = Model(inputs=model.input, outputs=predictions)

    loss, acc = model.evaluate(x_test, y_test)

    return acc


if __name__ == '__main__':
    for i in range(100):
        acc = train_and_eval()

        # If the model has a accuracy > then 80% save the model.
        if acc > 0.8:
            break
            
    model.save('model.h5')
