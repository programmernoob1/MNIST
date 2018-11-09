import numpy as np
import prepare
class NN():
    def __init__(self,sizeOfNeuralNetwork,alpha):
        self.size=len(sizeOfNeuralNetwork)
        self.weights=[np.random.randn(y,x) for x,y in zip(sizeOfNeuralNetwork[:-1],sizeOfNeuralNetwork[1:])]
        self.biases=[np.random.randn(y,1) for y in sizeOfNeuralNetwork[1:]]
        self.alpha=alpha
    def run(self,trainingData,trainingResults):
        n=len(trainingData)
        epochs=25
        for j in range(epochs):
            dBiases,dWeights=self.full(trainingData,trainingResults)
            self.weights=[weight-alpha*dWeight/m for weight,dWeight in zip(self.weights,dWeights)]
            self.biases=[bias-alpha*dBias/m for weight,dBias in zip(self.biases,dBiases)]
    def full(self,trainingData,trainingResults):
        dBiases = [np.zeros(b.shape) for b in self.biases]
        dWeights =[np.zeros(w.shape) for w in self.weights]
        Z,Activation=self.forward(trainingData)
        A=Activation[self.size-1].shape
        print(Activation[self.size-1].shape)
        print(self.loss(A,trainingResults).shape)
        dThisBiases,dThisWeights=self.backprop(A,Z,Activation)
        dBiases=[currentdBias+dBias for currentdBias,dBias in zip(dBiases,dThisBiases)]
        dWeights=[currentdWeight+dWeight for currentdWeight,dWeight in zip(dweights,dThisWeights)]
        return dBiases,dWeights
    def loss(self,A,y):
        cost=y*np.log(A)+(1-y)*np.log(1-A)
        return cost
    def forward(self,X):
        A=X
        Activation=[]
        Z=[]
        Activation.append(X)
        for bias,weight in zip(self.biases,self.weights):
            Zinter=np.dot(weight,A)+bias
            Z.append(Zinter)
            A=self.sigmoid(Zinter)
            Activation.append(self.sigmoid(Zinter))
        return Z,Activation
    def sigmoid(self,Z):
        return 1/(1+np.exp(-Z))
    def backprop(A,Z,Activation):
        dBiases=[]
        dWeights=[]
        dA_prev = - (np.divide(Y, Y_hat) - np.divide(1 - Y, 1 - Y_hat));
        for i in range(1,self.size):
            dBiases.append(dA)
            dZ=dA*sigmoidPrime(Z[self.size-1-i])
            dWeights=








trainingData,trainingResults,validationData,validationResults,testingData,testingResults=prepare.load_data_wrapper()
mnist=NN([784,30,10],0.1)
mnist.run(trainingData,trainingResults)

