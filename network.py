import gzip
import numpy as np


log = open('./log.txt', 'w')


class Network:

    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes

        self.weights = [np.random.randn(self.sizes[k + 1], self.sizes[k])
                        for k in range(self.num_layers - 1)]
        self.biases = [np.random.randn(self.sizes[k + 1])
                       for k in range(self.num_layers - 1)]

        """ regularization parameter """
        self._lambda = 0

    def save(self):
        with open('./net.txt', 'w') as f:
            f.write(str(self.num_layers) + '\n')
            for k in range(self.num_layers):
                f.write(str(self.sizes[k]) + '\n')
            for k in range(self.num_layers - 1):
                for i in range(self.biases[k].shape[0]):
                    f.write(str(self.biases[k][i]) + ' ')
                f.write('\n')
                for i in range(self.weights[k].shape[0]):
                    for j in range(self.weights[k].shape[1]):
                        f.write(str(self.weights[k][i][j]) + ' ')
                    f.write('\n')
                f.write('\n')

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
        delta[-1] = [(x - y) * self.sigmoid_prime(zl) for x, y, zl in zip(units[-1], data[1], z[-1])]

        for k in range(self.num_layers - 2, 0, -1):
            delta[k - 1] = [a * b for a, b in zip(np.dot(self.weights[k].T, delta[k]),
                                                  [self.sigmoid_prime(x) for x in z[k - 1]])]
        delta_biases = np.copy(delta)
        for k in range(self.num_layers - 1):
            delta[k] = np.outer(delta[k], units[k])
        return delta_biases, delta

    def update_train_data_batch(self, train_data_batch, alpha):
        accum_delta_weights = [np.zeros(w.shape) for w in self.weights]
        accum_delta_biases = [np.zeros(b.shape) for b in self.biases]

        for data in train_data_batch:
            delta_biases, delta_weights = self.backprop(data)
            accum_delta_weights = [a + b for a, b in zip(accum_delta_weights, delta_weights)]
            accum_delta_biases = [a + b for a, b in zip(accum_delta_biases, delta_biases)]

        for k in range(self.num_layers - 1):
            self.weights[k] = np.array([a - (alpha / len(train_data_batch)) * b for a,
                                        b in zip(self.weights[k], accum_delta_weights[k])])
            self.biases[k] = np.array([a - (alpha / len(train_data_batch)) * b for a,
                                       b in zip(self.biases[k], accum_delta_biases[k])])

    """ Implementation of stochastic gradient descent"""

    def SGD(self, train_data, epochs=30, train_data_batch_size=10, alpha=1.0, test_data=None):
        m = len(train_data)
        for epoch in range(epochs):
            print('Epoch ' + str(epoch + 1))
            np.random.shuffle(train_data)
            train_data_batches = [train_data[i: i + train_data_batch_size]
                                  for i in range(0, m, train_data_batch_size)]
            for train_data_batch in train_data_batches:
                self.update_train_data_batch(train_data_batch, alpha)

            if test_data:
                print('Test data checking: %d / %d' % (self.evaluate(test_data), len(test_data)))

    def evaluate(self, test_data):
        test_results = [(np.argmax(np.array(self.feedforward(x))), np.argmax(np.array(y))) for x, y in test_data]
        return sum([(x == y) for x, y in test_results])

    def test(self, test_data, label_data):
        pass


""" Convert a label to 10 elements vector """


def convertLabel(label):
    tmp_arr = np.zeros(10)
    tmp_arr[label] = 1.0
    return tmp_arr


print("Reading training data...")
train_data = list(zip([[float(x) for x in line.replace('\n', '').split(' ') if x != '']
                       for line in gzip.open('./train-images.gz', 'rt').readlines()],
                      [convertLabel(int(label)) for label in gzip.open('./train-labels.gz', 'rt').readlines()]))

network = Network([784, 28, 10])
print("Training network...")
network.SGD(train_data, alpha=0.077, test_data=train_data[:1000])
network.save()

for i in range(len(train_data) - 20, len(train_data)):
    a = network.feedforward(train_data[i][0])
    log.write(str(a) + '\n' + str(train_data[i][1]) + '\n')
log.close()
print('Done!')
