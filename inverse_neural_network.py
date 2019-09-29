import os
import pickle
import numpy as np

from keras.models import load_model
from PIL import Image

ALPHA = 0.3

# Load the model that was trained in `neural_network.py`.
model = load_model('model.h5')

W1 = model.layers[1].get_weights()[0]
W1 = np.array(W1)
W1_inv = np.linalg.pinv(W1)

b1 = model.layers[1].get_weights()[1]
b1 = np.array(b1)

W2 = model.layers[3].get_weights()[0]
W2 = np.array(W2)
W2_inv = np.linalg.pinv(W2)

b2 = model.layers[3].get_weights()[1]
b2 = np.array(b2)

W3 = model.layers[5].get_weights()[0]
W3 = np.array(W3)
W3_inv = np.linalg.pinv(W3)

b3 = model.layers[5].get_weights()[1]
b3 = np.array(b3)

def inv_relu(x):
    """
    The inverse of the RELU activation function.
    """
    if x == 0:
        return 0
    if x > 0:
        return x
    if x < 0:
        return 1/ALPHA * x

inv_relu_vec = np.vectorize(inv_relu)

def reverse_inference(ol4):
    """
    This does the reverse inference for the model. The
    input is the output of the last layer in this case
    ol4 (output layer 4). Now it calcualtes the input
    that resulted in ol4.
    Args:
        ol4 (np.array): 10 dim array.
    Return:
        ol1 (np.array): 784 dim array. The calculated inverse
            inference.
    """
    ol3 = np.dot(np.subtract(ol4, b3), W3_inv)
    ol2 = np.dot((inv_relu_vec(ol3) - b2), W2_inv)
    ol1 = np.dot((inv_relu_vec(ol2) - b1), W1_inv)

    return ol1

def test(ol4):
    inv_pred = reverse_inference(ol4)
    pred = model.predict(inv_pred.reshape(1, 784))

    # This should be close to zero.
    print(pred - ol4)

if __name__ == '__main__':
    test(np.array([1, 0]))
    test(np.array([0, 1]))
    test(np.array([0.2, 0.8]))
