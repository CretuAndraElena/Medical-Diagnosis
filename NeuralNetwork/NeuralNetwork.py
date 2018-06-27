import numpy as np
import CSVParse as CSVP
import matplotlib.pyplot as plt
import time

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

def normalizare(matrix):
    new_matrix = []
    for i in range(0, len(matrix[0])):
        col = np.array(matrix)[:, i]
        if col[0] > 0:
            new_matrix.append(col / col.max())
        else:
            new_matrix.append(col)

    output = list()
    for i in range(0, len(matrix)):
        x = [matrix[i][-1]]
        output.append(x)
    output = np.array(output, dtype=float)

    return (np.array(np.array(new_matrix).T[:, :-1], dtype=float), output)


class Neural_Network:
    def __init__(self, train):
        self.learning_rate = 0.05
        self.inputSize = train.shape[1]
        self.outputSize = 1
        self.hiddenSize = 3

        self.W1 = np.random.random((self.inputSize, self.hiddenSize))
        self.W2 = np.random.random((self.hiddenSize, self.outputSize))

    def sigmoid(self, x):
            return 1 / (1 + np.exp(-x))

    def sigmoid_prime(self, x):
        return x * (1 - x)

    def forward(self, input):

        self.a = self.sigmoid(np.dot(input, self.W1))
        self.o = self.sigmoid(np.dot(self.a, self.W2))

        return self.o

    def backward(self, X, y, o):

        error_output = y - o
        d_output = error_output * self.sigmoid_prime(o)

        error_hidden = d_output.dot(self.W2.T)
        d_hidden = error_hidden * self.sigmoid_prime(self.a)

        self.W2 += self.a.T.dot(d_output) * self.learning_rate
        self.W1 += X.T.dot(d_hidden) * self.learning_rate

    def train(self, X, y):
        o = self.forward(X)
        self.backward(X, y, o)

    def predict(self, predicted):

        if self.forward(predicted) > 0.5:
            return 1
        else:
            return 0

    def saveWeights(self):
        np.savetxt("w1.txt", self.W1, fmt="%s")
        np.savetxt("w2.txt", self.W2, fmt="%s")

    def error(self, test_data, test_output):
        return 0.5 * np.mean(np.square(test_output - self.forward(test_data)))

    def antrenare(self, iterations, train, output):
        loss = list()
        iterationsList = list()
        for i in range(iterations):
            loss.append(self.error(train, output))
            iterationsList.append(i)
            self.train(train, output)
        self.saveWeights()
        return loss, iterationsList


def main():
    header, training_data = CSVP.csv_parse("DateAntrenament.csv")
    train, output = normalizare(training_data)
    NN = Neural_Network(train)
    start_time = time.time()
    loss, iterations = NN.antrenare(3000, train, output)
    t = time.time() - start_time
    print("Error la antrenare:", NN.error(train, output))

    header, test_data = CSVP.csv_parse("DataSetTest3.csv")
    test_data, test_output = normalizare(test_data)
    print("Error la antrenare:", NN.error(test_data, test_output))

    print("Timpul de antrenare:", t)

    header, studiu_de_caz = CSVP.csv_parse("DataSetStudiuDeCaz.csv")
    studiu_de_caz, studiu_de_caz_output = normalizare(studiu_de_caz)
    result = list()
    print("Rezultate studiu de caz:")
    for x in studiu_de_caz:
        result.append(NN.predict(x))
    print(result)

    plt.plot(iterations, loss)
    plt.title('Neural Netwotk')
    plt.xlabel('No. iterations')
    plt.ylabel('Loss')
    plt.show()


main()
