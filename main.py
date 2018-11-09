import numpy as np
import prepare
class NN():
    def __init__(self,sizeOfNeuralNetwork,alpha):
        self.size=len(sizeOfNeuralNetwork) #Getting no of layers
        self.weights=[np.random.randn(y,x) for x,y in zip(sizeOfNeuralNetwork[:-1],sizeOfNeuralNetwork[1:])]#Initializing the weights of the NN
        self.biases=[np.random.randn(y,1) for y in sizeOfNeuralNetwork[1:]]#Initializing the bias
        self.alpha=alpha
    def run(self,trainingData,testingData):#Main function
        m=len(trainingData)#No of traniing examples
        epochs=60# No of iterations
        for j in range(epochs):
            dBiases= [np.zeros(b.shape) for b in self.biases] # chnage of bias values to be updated after each iteration
            dWeights = [np.zeros(w.shape) for w in self.weights]#chnage in weights after each iteration
            for x,y in trainingData:
                dBiasInter,dWeightInter=self.full(x,y)#performs Forward and backward prop
                dBiases = [nb+dnb for nb, dnb in zip(dBiasInter, dBiases)]
                dWeights = [nw+dnw for nw, dnw in zip(dWeightInter, dWeights)]
            self.weights=[weight-self.alpha*dWeight/m for weight,dWeight in zip(self.weights,dWeights)]#updating the weights after an iteration
            self.biases=[bias-self.alpha*dBias/m for bias,dBias in zip(self.biases,dBiases)]#updating the bias after an iteration
            print("Epoch",j,"Value=",self.evaluate(testingData))#evaluating on a test set
    def full(self,x,y):
        dBiases = [np.zeros(b.shape) for b in self.biases]
        dWeights =[np.zeros(w.shape) for w in self.weights]
        Z,Activation=self.forward(x)#performs forward prop
        A=Activation[self.size-1]#last activation value
        #print(Activation[self.size-1].shape)
        #print(self.loss(A,y))
        dThisBiases,dThisWeights=self.backprop(A,Z,Activation,y)#performs backprop
        dBiases=[currentdBias+dBias for currentdBias,dBias in zip(dBiases,dThisBiases)]
        dWeights=[currentdWeight+dWeight for currentdWeight,dWeight in zip(dWeights,dThisWeights)]
        return dBiases,dWeights
    def loss(self,A,y):
        cost=y*np.log(A)+(1-y)*np.log(1-A)
        return cost
    def forward(self,X):
        A=X#initializing the input
        Activation=[]
        Z=[]
        Activation.append(X)
        for bias,weight in zip(self.biases,self.weights):
            Zinter=np.dot(weight,A)+bias#computing the product of weight and input
            Z.append(Zinter)
            A=self.sigmoid(Zinter)# applying sigmoid to the product
            Activation.append(self.sigmoid(Zinter))
        return Z,Activation # returns the product and the activation values
    def sigmoid(self,Z):
        return 1/(1+np.exp(-Z))#computes sigmoid on a numpy array
    def backprop(self,A,Z,Activation,Y):
        dBiases=[0 for i in range(self.size)]
        dWeights=[0 for i in range(self.size)]
        dA=A-Y#Loss function (as of now)
        for i in range(1,self.size):
            dBiases[self.size-1-i]=dA#dBias is same as Loss of the node
            dZ=dA*self.sigmoidPrime(Activation[self.size-i])#error in node multipied with the sigmoid prime
            dWeights[self.size-i-1]=np.dot(dZ,Activation[self.size-1-i].T)# errors through the edges
            dA=np.dot(self.weights[self.size-i-1].T,dZ)# computing the error in the previous level nodes
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
mnist=NN([784,30,10],7)#setting the layer number
mnist.run(trainingData,testingData)#fit and test


