import numpy as np
import torch
from neural_network import NeuralNetwork
import logic_gates

def main():
    print("Homework 2 ==========================")
    layer_size = [5,4,3,2,1]
    num_of_layers = len(layer_size)
    test = NeuralNetwork(layer_size)
    print('The test layer size is:')
    print(layer_size)
    # layer number starts with 0 at input layer
    
    index = 4
    if index >num_of_layers:
        raise Exception('Weights does not exist!')
    
    # print random weight
    theta = test.getLayer(index)
    print('Testing getLayer()\r\nWeight at layer ' + str(index) + ':')
    print(test.getLayer(index))

    # modify weight
    #test.network['theta'+str(index)] = torch.Tensor([[1,2,3]])
    theta[0,:] = torch.Tensor([[1,2,3]])
    print('After modifying network weight, new weight at layer' + str(index) + ':')
    print(test.getLayer(index))
    
    # 1D Tensor
    print('Test 1D Tensor input')
    NN_in = torch.Tensor([3,5,7,2,6])
    print('The input is')
    print(NN_in)
    NN_out = test.forward(NN_in)
    print('The network output is')
    print(NN_out)

    # 2D Tensor row vector
    print('Test 2D Tensor row vector input')
    NN_in = (torch.Tensor([[3,5,7,2,6]]))
    print('The input is')
    print(NN_in)
    NN_out = test.forward(NN_in)
    print('The network output is')
    print(NN_out)
    
    # 2D Tensor column vector
    print('Test 2D Tensor column vector input')
    NN_in = torch.t(torch.Tensor([[3,5,7,2,6]]))
    print('The input is')
    print(NN_in)
    NN_out = test.forward(NN_in)
    print('The network output is')
    print(NN_out)

    # 2D Tensor column vectors
    print('Test stacked column vectors as input')
    NN_in = torch.t(torch.Tensor([[3,5,7,2,6],[5,3,8,6,7]]))
    print('The input is')
    print(NN_in)
    NN_out = test.forward(NN_in)
    print('The network output is')
    print(NN_out)

    print('Testing logic_gates:')
    print('\r\nAND gate')
    AND = logic_gates.AND()
    print(AND(False,False))
    print(AND(False,True))
    print(AND(True,False))
    print(AND(True,True))
    
    print('\r\nOR gate')
    OR = logic_gates.OR()
    print(OR(False,False))
    print(OR(False,True))
    print(OR(True,False))
    print(OR(True,True))
    
    print('\r\nNOT gate')
    NOT = logic_gates.NOT()
    print(NOT(False))
    print(NOT(True))
    
    print('\r\nXOR gate')
    XOR = logic_gates.XOR()
    print(XOR(False,False))
    print(XOR(False,True))
    print(XOR(True,False))
    print(XOR(True,True))
    

if __name__ == "__main__":
    main()
