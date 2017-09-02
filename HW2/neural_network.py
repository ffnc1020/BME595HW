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
            # i is the layer number following Z[i] = W[i]*A[i-1]+b
            w = torch.zeros(layer_size[i],layer_size[i-1])
            b = torch.zeros(layer_size[i],1)
            theta = torch.normal(means = w, std = 1/np.sqrt(layer_size[i]),out = w)
            self.network['theta' + str(i)] = theta
            self.network['bias' + str(i)] = b
    
    
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
        m = NN_input.shape[1]
        A0 = NN_input
        cache = {'A0':A0}
        for i in range(1,num_of_layers):
            Z = torch.zeros(layer_size[i],m)
            theta = self.network['theta' + str(i)]
            b = self.network['bias' + str(i)]
            A_prev = cache['A' + str(i-1)]
            Z = torch.matmul(theta,A_prev)+b
            A = torch.sigmoid(Z)
            cache['A' + str(i)] = A
            cache['Z' + str(i)] = Z
        NN_out = cache['A' + str(num_of_layers-1)]
        if input_dim == 1:
            NN_out = torch.squeeze(NN_out)
        return NN_out
