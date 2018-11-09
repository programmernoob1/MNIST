import numpy as np
import prepare
class NN():
    def __init__(self,sizeOfNeuralNetwork,alpha):
        self.size=len(sizeOfNeuralNetwork)
        self.weights=[np.random.randn(y,x) for x,y in zip(sizeOfNeuralNetwork[:-1],sizeOfNeuralNetwork[1:])]
        self.biases=[np.random.randn(y,1) for y in sizeOfNeuralNetwork[1:]]
        self.alpha=alpha
    def run(self,trainingData):
        m=len(trainingData)
        epochs=60
        for j in range(epochs):
            dBiases= [np.zeros(b.shape) for b in self.biases]
            dWeights = [np.zeros(w.shape) for w in self.weights]
            for x,y in trainingData:
                dBiasInter,dWeightInter=self.full(x,y)
                dBiases = [nb+dnb for nb, dnb in zip(dBiasInter, dBiases)]
                dWeights = [nw+dnw for nw, dnw in zip(dWeightInter, dWeights)]
            self.weights=[weight-self.alpha*dWeight/m for weight,dWeight in zip(self.weights,dWeights)]
            self.biases=[bias-self.alpha*dBias/m for bias,dBias in zip(self.biases,dBiases)]
            print("Epoch",j,"Value=",self.evaluate(testingData))
    def full(self,x,y):
        dBiases = [np.zeros(b.shape) for b in self.biases]
        dWeights =[np.zeros(w.shape) for w in self.weights]
        Z,Activation=self.forward(x)
        A=Activation[self.size-1]
        #print(Activation[self.size-1].shape)
        #print(self.loss(A,y))
        dThisBiases,dThisWeights=self.backprop(A,Z,Activation,y)
        dBiases=[currentdBias+dBias for currentdBias,dBias in zip(dBiases,dThisBiases)]
        dWeights=[currentdWeight+dWeight for currentdWeight,dWeight in zip(dWeights,dThisWeights)]
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
    def backprop(self,A,Z,Activation,Y):
        dBiases=[0 for i in range(self.size)]
        dWeights=[0 for i in range(self.size)]
        dA=A-Y
        for i in range(1,self.size):
            dBiases[self.size-1-i]=dA
            dZ=dA*self.sigmoidPrime(Activation[self.size-i])
            dWeights[self.size-i-1]=np.dot(dZ,Activation[self.size-1-i].T)
            dA=np.dot(self.weights[self.size-i-1].T,dZ)
        return dBiases,dWeights
    def sigmoidPrime(self,A):
        return A*(1-A)
    def evaluate(self,testingData):
        test_results=[]
        for x,y in testingData:
            a,result=self.forward(x)
            test_results.append((int(np.argmax(result[-1])), int(y)))
        return sum(int(x == y) for (x, y) in test_results)






trainingData,validationData,testingData=prepare.load_data_wrapper()
trainingData=list(trainingData)
validationData=list(validationData)
testingData=list(testingData)
mnist=NN([784,30,10],3)
mnist.run(trainingData)


