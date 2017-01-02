import gzip
import json

import numpy as np
from PIL import Image
from PIL import ImageOps
from io import BytesIO

class Network:

    def __init__(self, sizes, biases=None, weights=None):
        self.num_layers = len(sizes)
        self.sizes = sizes

        if biases:
            self.biases = biases
        else:
            self.biases = [np.random.randn(self.sizes[k + 1])
                           for k in range(self.num_layers - 1)]
        if weights:
            self.weights = weights
        else:
            self.weights = [np.random.randn(self.sizes[k + 1], self.sizes[k]) * (1.0 / np.sqrt(self.sizes[k + 1]))
                            for k in range(self.num_layers - 1)]

    def save(self, filename):
        with open(filename, 'w') as f:
            net = {
                'sizes': self.sizes,
                'biases': [b.tolist() for b in self.biases],
                'weights': [w.tolist() for w in self.weights]
            }
            json.dump(net, f)

    def sigmoid(self, z):
        return 1.0 / (1.0 + np.exp(-z))

    def sigmoid_prime(self, z):
        return self.sigmoid(z) * (1 - self.sigmoid(z))

    def feedforward(self, data):
        out_units = data
        for k in range(self.num_layers - 1):
            z = np.dot(self.weights[k], out_units) + self.biases[k]
            out_units = [self.sigmoid(x) for x in z]
        return out_units

    """ Backpropagation algorithm """

    def backprop(self, data):
        """ feedforward """

        units = [[] for k in range(self.num_layers)]
        z = [[] for k in range(self.num_layers - 1)]
        units[0] = data[0]
        for k in range(self.num_layers - 1):
            z[k] = np.dot(self.weights[k], units[k]) + self.biases[k]
            units[k + 1] = [self.sigmoid(x) for x in z[k]]

        delta = [[] for k in range(self.num_layers - 1)]
        delta[-1] = [(x - y) for x, y in zip(units[-1], data[1])]

        for k in range(self.num_layers - 2, 0, -1):
            delta[k - 1] = [a * b for a, b in zip(np.dot(self.weights[k].T, delta[k]),
                                                  [self.sigmoid_prime(x) for x in z[k - 1]])]
        delta_biases = np.copy(delta)
        for k in range(self.num_layers - 1):
            delta[k] = np.outer(delta[k], units[k])
        return delta_biases, delta

    def update_train_data_batch(self, train_data_batch, alpha, _lambda):
        accum_delta_weights = [np.zeros(w.shape) for w in self.weights]
        accum_delta_biases = [np.zeros(b.shape) for b in self.biases]

        for data in train_data_batch:
            delta_biases, delta_weights = self.backprop(data)
            accum_delta_weights = [a + b for a, b in zip(accum_delta_weights, delta_weights)]
            accum_delta_biases = [a + b for a, b in zip(accum_delta_biases, delta_biases)]

        n = len(train_data_batch)
        for k in range(self.num_layers - 1):
            self.weights[k] = np.array([(1 - alpha * (_lambda / n)) * a - (alpha / n) * b
                                        for a, b in zip(self.weights[k], accum_delta_weights[k])])
            self.biases[k] = np.array([a - (alpha / len(train_data_batch)) * b for a,
                                       b in zip(self.biases[k], accum_delta_biases[k])])

    """ Implementation of stochastic gradient descent"""

    def SGD(self, train_data, epochs=30, train_data_batch_size=10, alpha=1.0, _lambda=0.0, test_data=None):
        m = len(train_data)
        for epoch in range(epochs):
            print('Epoch ' + str(epoch + 1))
            np.random.shuffle(train_data)
            train_data_batches = [train_data[i: i + train_data_batch_size]
                                  for i in range(0, m, train_data_batch_size)]
            for train_data_batch in train_data_batches:
                self.update_train_data_batch(train_data_batch, alpha, _lambda)

            if test_data:
                print('Test data checking: %d / %d' % (self.evaluate(test_data), len(test_data)))
                print('Cost: ' + str(self.cost(test_data, _lambda)))
            # self.save('./net.txt')

    def evaluate(self, test_data):
        test_results = [(np.argmax(np.array(self.feedforward(x))), np.argmax(np.array(y))) for x, y in test_data]
        return sum([(x == y) for x, y in test_results])

    def cost(self, dataset, _lambda):
        cost = 0.0
        for x, y in dataset:
            x = self.feedforward(x)
            cost += np.sum(np.nan_to_num(-y * np.log(x) -
                                         (np.full(np.array(y).shape, 1.0) - y) *
                                         np.log(np.full(np.array(x).shape, 1.0) - x))) / len(dataset) +\
                0.5 * (_lambda / len(dataset)) * sum(np.linalg.norm(w) ** 2 for w in self.weights)
        return cost

    def recognize(self, pic_file):
        pic = Image.open(BytesIO(pic_file))
        pic = pic.resize((28, 28), Image.ANTIALIAS)
        pic = ImageOps.invert(pic).convert('L')
        arr = np.array(pic).reshape((784))
        arr = self.feedforward(arr)
        return np.argmax(arr)

def load(filename):
    data = json.load(open(filename, 'r'))
    return Network(data['sizes'], biases=[np.array(b)
                                          for b in data['biases']], weights=[np.array(w) for w in data['weights']])


""" Convert a label to 10 elements vector """


def convertLabel(label):
    tmp_arr = np.zeros(10)
    tmp_arr[label] = 1.0
    return tmp_arr


if __name__ == '__main__':
    print("Reading data...")
    train_data = list(zip([[float(x) for x in line.replace('\n', '').split(' ') if x != '']
                           for line in gzip.open('./train-images.gz', 'rt').readlines()],
                          [convertLabel(int(label)) for label in gzip.open('./train-labels.gz', 'rt').readlines()]))

    test_data = list(zip([[float(x) for x in line.replace('\n', '').split(' ') if x != '']
                          for line in gzip.open('./test-images.gz', 'rt').readlines()],
                         [convertLabel(int(label)) for label in gzip.open('./test-labels.gz', 'rt').readlines()]))

    network = Network([784, 50, 10])
    print("Training network...")
    network.SGD(train_data, epochs=100, alpha=0.0003, _lambda=0.5, test_data=test_data)
    network.save('./net.txt')

    print('Done!')
