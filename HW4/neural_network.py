import numpy as np
import torch

class NeuralNetwork(object):
    
    def __init__(self, layer_size):
        self.layer_size = layer_size
        # get number of layers
        num_of_layers = len(layer_size)
        if num_of_layers<2:
            raise Exception('NeuralNetwork must have 2+ layers!')
        # randomize weights
        self.network = {}
        for i in range(1,num_of_layers):
            '''
            # i is the layer number following Z[i] = W[i]*A[i-1]+b
            w = torch.zeros(layer_size[i],layer_size[i-1])
            b = torch.zeros(layer_size[i],1)
            theta = torch.normal(means = w, std = 1/np.sqrt(layer_size[i]),out = w)
            self.network['theta' + str(i)] = theta
            self.network['bias' + str(i)] = b
            '''
            # i is the layer number following Z[i] = W[i]*A[i-1]
            w = torch.zeros(layer_size[i],layer_size[i-1]+1)
            theta = torch.normal(means = w, std = 1/np.sqrt(layer_size[i]),out = w)*0.001
            self.network['theta' + str(i)] = theta

    def getLayer(self, layer):
        theta = self.network['theta'+str(layer)]
        return theta


    def forward(self, NN_input):
        layer_size = self.layer_size
        num_of_layers = len(layer_size)
        
        # check input dimensions 
        NN_input_np = NN_input.numpy()
        if NN_input_np.ndim > 2:
            raise Exception('Input dimension must be 1D or 2D!')
        elif NN_input_np.ndim == 2:
            input_dim = 2
            if NN_input_np.shape[0] == self.layer_size[0]:
                NN_input = torch.from_numpy(NN_input_np)
            else:
                NN_input = torch.from_numpy(NN_input_np.T)
        else:
            input_dim = 1
            NN_input_np = np.reshape(NN_input_np,(NN_input_np.shape[0],1))
            NN_input = torch.from_numpy(NN_input_np)
        if NN_input.shape[0] != self.layer_size[0]:
            raise Exception('Input dimension mismatch!')
        
        '''
        m = NN_input.shape[1]#num of example
        temp = NN_input.numpy()
        temp = np.reshape(temp,(temp.shape[0],m))
        NN_input = torch.from_numpy(temp)
        '''
        # number of examples
        m = NN_input.shape[1]
        A0 = NN_input
        A0 = torch.cat((torch.ones(1,m),A0),0)
        self.cache = {'A0':A0}
        for i in range(1,num_of_layers):
            Z = torch.zeros(layer_size[i],m)
            W = self.network['theta' + str(i)]
            A_prev = self.cache['A' + str(i-1)]
            Z = torch.matmul(W,A_prev)
            A = torch.sigmoid(Z)
            if i<(num_of_layers-1):
                A = torch.cat((torch.ones(1,m),A),0)
            self.cache['A' + str(i)] = A
            self.cache['Z' + str(i)] = Z
        NN_out = self.cache['A' + str(num_of_layers-1)]
        if input_dim == 1:
            NN_out = torch.squeeze(NN_out)
        return NN_out



    def backward(self,target,loss = None):
        cache = self.cache
        layer_size = self.layer_size
        num_of_layers = len(layer_size)
        Y = target
        AO = cache['A'+str(num_of_layers-1)]
        m = Y.shape[1]
        
        #print((self.Loss))
        if loss == 'CE':
            cost = -1/m*torch.sum(Y*torch.log(AO)+(1-Y)*torch.log(1-AO),dim=1)
            #self.Loss = cost.view(Y.shape[0],1)
            cost = torch.sum(cost)/Y.shape[0]
            self.Loss = cost
            dAO = -(torch.div(Y,AO)-torch.div(1-Y, 1-AO))
        else:
            cost = torch.sum((AO - Y),dim=1)/(2*m)
            #self.Loss = cost.view(Y.shape[0],1)
            cost = torch.sum(cost)/Y.shape[0]
            self.Loss = cost
            
            dAO = AO - Y
            #print('Loss = '+str(self.Loss))
        self.grad = {'dA'+str(num_of_layers-1):dAO}
        for i in reversed(range(1,num_of_layers)):
            dA = self.grad['dA'+str(i)]
            if i < (num_of_layers-1):
                dA = dA[1:,:]#take the bottom n[i] rows, get ignoring dA towards the bias
            A_prev = cache['A'+str(i-1)]
            Z = cache['Z'+str(i)]
            W = self.network['theta' + str(i)]
            dAdZ = (torch.sigmoid(Z))*(1-torch.sigmoid(Z))#element wise product
            dZ = dA*dAdZ
            dW = 1/m*torch.matmul(dZ,torch.t(A_prev))
            dA_prev = torch.matmul(torch.t(W),dZ)
            self.grad['dW'+str(i)] = dW
            self.grad['dA'+str(i-1)] = dA_prev
    #print('Back prop done!')

    def updateParams(self,eta):
        layer_size = self.layer_size
        num_of_layers = len(layer_size)
        for i in range(1,num_of_layers):
            W = self.network['theta' + str(i)]
            dW = self.grad['dW'+str(i)]
            '''print('W:')
            print(W)
            print('dW:')
            print(dW)'''
            W[:] = W[:] - eta*dW
            
#print('Parameter updated!')
