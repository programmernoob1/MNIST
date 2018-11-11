import numpy as np
import prepare
class NN():
    def __init__(self,sizeOfNeuralNetwork):
        self.size=len(sizeOfNeuralNetwork) #Getting no of layers
        self.weights=[np.random.randn(y,x) for x,y in zip(sizeOfNeuralNetwork[:-1],sizeOfNeuralNetwork[1:])]#Initializing the weights of the NN
        self.biases=[np.random.randn(y,1) for y in sizeOfNeuralNetwork[1:]]#Initializing the bias
    def run(self,trainingData,testingData,alpha,batchSize):#Running Stochastic Gradient Descent
        m=len(trainingData)# No of examples in training set
        m_test=len(testingData)# No of examples in testing data
        self.alpha=alpha# Training rate
        epochs=25
        for i in range(epochs):
            np.random.shuffle(trainingData)#Shuffling training data
            batches=[trainingData[k:k+batchSize] for k in range(0,m,batchSize)]
            for batch in batches:
                self.run_batch(batch)#Updating values for each batch
            print("Epoch",i,"Value:",self.evaluate(testingData),"/",m_test)
    def run_batch(self,batch):
        m=len(batch)#No of examples in each batch
        dWeights=[np.zeros(weight.shape) for weight in self.weights]#Initializing the update of weights to 0
        dBiases=[np.zeros(bias.shape) for bias in self.biases]#Initializing the update of biases to 0
        for x,y in batch:#For each training example
            dWeightsInter,dBiasesInter=self.singlePass(x,y)#Single forward and backward pass
            dWeights=[dWeight+dWeightInter for dWeight,dWeightInter in zip(dWeights,dWeightsInter)]#updating change in weights
            dBiases=[dBias+dBiasInter for dBias,dBiasInter in zip(dBiases,dBiasesInter)]#updating change in biases
        self.weights=[weight-(self.alpha/m)*dWeight for weight,dWeight in zip(self.weights,dWeights) ]#updating weights
        self.biases=[bias-(self.alpha/m)*dBias for bias,dBias in zip(self.biases,dBiases) ]#updating biases
    def singlePass(self,x,y):
        dWeightsInter=[np.zeros(weight.shape) for weight in self.weights]#Update in weights for that example
        dBiasesInter=[np.zeros(bias.shape) for bias in self.biases]#Update in bias for that example
        activations=self.forwardPass(x)#Forward pass
        dA=activations[-1]-y#Computing cost derivative of the loss function
        #BAckpropogation
        for i in range(1,self.size):
            dZ=dA*self.sigmoidPrime(activations[-i])#Computing dZ
            dBiasesInter[-i]=dZ#dBias computation
            dWeightsInter[-i]=np.dot(dZ,activations[-i-1].T)#Vectorized computation of DW
            dA=np.dot(self.weights[-i].T,dZ)#Computing dA for the previous layer
        return dWeightsInter,dBiasesInter
    def sigmoidPrime(self,A):
        return A*(1-A)
    def evaluate(self,testingData):
        test_results=[(int(np.argmax(self.forwardPass(x)[-1])),y) for x,y in testingData]
        return sum(int(x==y) for x,y in test_results)
    def forwardPass(self,x):
        activations=[x]
        lastActivation=x#First activation is the example data
        for weight,bias in zip(self.weights,self.biases):
            z=np.dot(weight,lastActivation)+bias#Computing Z
            lastActivation=self.sigmoid(z)#Activation function on Z
            activations.append(lastActivation)
        return activations
    def sigmoid(self,z):
        return 1.0/(1.0+np.exp(-z))

trainingData,validationData,testingData=prepare.load_data_wrapper()
trainingData=list(trainingData)
validationData=list(validationData)
testingData=list(testingData)
mnist=NN([784,30,10])#setting the layer number
mnist.run(trainingData,validationData,3,10)#fit and test


