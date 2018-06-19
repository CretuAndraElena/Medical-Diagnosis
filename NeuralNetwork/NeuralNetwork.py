import numpy as np
import CSVParse as CSVP
'''X = np.array(([1, 1,205],
              [1, 1, 205],
              [1, 1,  260],
              [1, 0, 380],
              [0, 1, 205],
              [0, 1, 260],
              [0, 0,260],
              [0, 0,380],
              [0, 0, 380]),dtype=float)

y = np.array(([0],[0],[1],[1],[0],[1],[1],[0],[0]),dtype=float)
xPredicted = np.array(([1, 0, 280]),dtype=float)'''


header, training_data = CSVP.csv_parse("DateAndrenament.csv")

def normalizare(matrix):
    new_matrix=[]
    for i in range (0,len(matrix[0])):
        col=np.array(matrix)[:,i]
        if col[0]>1:
            new_matrix.append(col/col.max())
        else:
            new_matrix.append(col)
    return np.array(new_matrix).T

normalizare(training_data)

X= np.array(np.array(training_data)[:,:-1],dtype=float)
y=[[0]]*len(X)
for i in range(0,len(X)):
    y[i][0]=X[i][-1]
y=np.array(y,dtype=float)


class Neural_Network(object):
    def __init__(self):
        #parameters
        self.inputSize = len(X[0])
        self.outputSize = 1
        self.hiddenSize = len(X[0])+1

        #weights
        self.W1 = np.random.rand(self.inputSize, self.hiddenSize)
        self.W2 = np.random.rand(self.hiddenSize, self.outputSize)

    def forward(self, X):

        self.z = np.dot(X, self.W1)
        self.z2 = self.sigmoid(self.z)
        self.z3 = np.dot(self.z2, self.W2)
        o = self.sigmoid(self.z3)
        return o

    def sigmoid(self, s):
        # activation function
        return 1/(1+np.exp(-s))

    def sigmoidPrime(self, s):
        #derivative of sigmoid
        return s * (1 - s)

    def backward(self, X, y, o):
        # backward propgate through the network
        self.o_error = y - o # error in output
        self.o_delta = self.o_error*self.sigmoidPrime(o) # applying derivative of sigmoid to error

        self.z2_error = self.o_delta.dot(self.W2.T) # z2 error: how much our hidden layer weights contributed to output error
        self.z2_delta = self.z2_error*self.sigmoidPrime(self.z2) # applying derivative of sigmoid to z2 error

        self.W1 += X.T.dot(self.z2_delta) # adjusting first set (input --> hidden) weights
        self.W2 += self.z2.T.dot(self.o_delta) # adjusting second set (hidden --> output) weights

    def train(self, X, y):
        o = self.forward(X)
        self.backward(X, y, o)

    def saveWeights(self):
        np.savetxt("w1.txt", self.W1, fmt="%s")
        np.savetxt("w2.txt", self.W2, fmt="%s")

    def predict(self,xPredicted):

        if self.forward(xPredicted)>0.5:
            output=1
        else:
            output=0

        return output

    def error(self,test_data):
        error=0
        for data in test_data:
            if data[-1]!=self.predict(data[0:-1]):
                error=+1
        print("Error: \n",error/len(test_data))

def antrenare(iterations,NN):
    for i in range(iterations+1):
        '''print("# " + str(i) + "\n")
        print("Input: \n" + str(X))
        print("Actual Output: \n" + str(y))
        print("Predicted Output: \n" + str(NN.forward(X)))
        print("Loss: \n" + str(np.mean(np.square(y - NN.forward(X))))) # mean sum squared loss
        print("\n")'''
        NN.train(X, y)
    NN.saveWeights()

def main():
    NN = Neural_Network()
    antrenare(1000,NN)
    header, test_data = CSVP.csv_parse("DataSetTest1.csv")
    print("Error DataSetTest1:",NN.error(test_data))

    header, test_data = CSVP.csv_parse("DataSetTest2.csv")
    print("Error DataSetTest2:",NN.error(test_data))

    header, test_data = CSVP.csv_parse("DataSetTest3.csv")
    print("Error DataSetTest3:",NN.error(test_data))

    header, studiu_de_caz = CSVP.csv_parse("DataSetStudiuDeCaz.csv")
    result=[]
    for x in studiu_de_caz:
        predicted = np.array(x,dtype=float)
        predicted=predicted[0:-1]
        result.append( NN.predict(predicted))
    print("Rezultate studiu de caz:",result)

    NN1 = Neural_Network()
    antrenare(10,NN1)
    header, test_data = CSVP.csv_parse("DataSetTest3.csv")
    print("Error DataSetTest3 10 iteratii:",NN1.error(test_data))

    NN2= Neural_Network()
    antrenare(50,NN2)
    header, test_data = CSVP.csv_parse("DataSetTest3.csv")
    print("Error DataSetTest3 50 iteratii:",NN2.error(test_data))

    NN3 = Neural_Network()
    antrenare(100,NN3)
    header, test_data = CSVP.csv_parse("DataSetTest3.csv")
    print("Error DataSetTest3 100 iteratii:",NN3.error(test_data))

    NN4 = Neural_Network()
    antrenare(500,NN4)
    header, test_data = CSVP.csv_parse("DataSetTest3.csv")
    print("Error DataSetTest3 500 iteratii:",NN4.error(test_data))

main()
