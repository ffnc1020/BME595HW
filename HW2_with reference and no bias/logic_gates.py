import numpy as np
import torch
from neural_network import NeuralNetwork

class AND(object):
    
    def __init__(self):
        gate_size = [2,1]
        self.gate = NeuralNetwork(gate_size)
        self.gate.network['theta1'] = torch.Tensor([[-1.5,1,1]])

    def forward(self):
        in_1 = int(bool(self.x)==True)
        in_2 = int(bool(self.y)==True)
        gate = self.gate

        gate_in = torch.Tensor([in_1,in_2])
        gate_out = gate.forward(gate_in)
        if gate_out.numpy()>0.5:
            out = True
        else:
            out = False
        return out

    def __call__(self,x,y):
        self.x = x
        self.y = y
        out = self.forward()
        return out 

class OR(object):
    
    def __init__(self):
        gate_size = [2,1]
        self.gate = NeuralNetwork(gate_size)
        self.gate.network['theta1'] = torch.Tensor([[0,1,1]])

    def forward(self):
        in_1 = int(bool(self.x)==True)
        in_2 = int(bool(self.y)==True)
        gate = self.gate
        
        gate_in = torch.Tensor([in_1,in_2])
        gate_out = gate.forward(gate_in)
        
        if gate_out.numpy()>0.5:
            out = True
        else:
            out = False
        return out
    
    def __call__(self,x,y):
        self.x = x
        self.y = y
        out = self.forward()
        return out

class NOT(object):
    
    def __init__(self):
        gate_size = [1,1]
        self.gate = NeuralNetwork(gate_size)
        self.gate.network['theta1'] = torch.Tensor([[1,-2]])

    
    def forward(self):
        in_1 = int(bool(self.x)==True)
        gate = self.gate
        
        gate_in = torch.Tensor([in_1])
        gate_out = gate.forward(gate_in)
        
        if gate_out.numpy()>0.5:
            out = True
        else:
            out = False
        return out
    
    def __call__(self,x):
        self.x = x
        out = self.forward()
        return out

class XOR(object):
    
    def __init__(self):
        gate_size = [2,2,1]
        self.gate = NeuralNetwork(gate_size)
        self.gate.network['theta1'] = torch.Tensor([[-10,20,20],[30,-20,-20]])
        self.gate.network['theta2'] = torch.Tensor([[-30,20,20]])
        
        self.gate.network['bias2'] = torch.Tensor([[-30]])
    def forward(self):
        in_1 = int(bool(self.x)==True)
        in_2 = int(bool(self.y)==True)
        gate = self.gate
        
        gate_in = torch.Tensor([in_1,in_2])
        gate_out = gate.forward(gate_in)
        
        if gate_out.numpy()>0.5:
            out = True
        else:
            out = False
        return out
    
    def __call__(self,x,y):
        self.x = x
        self.y = y
        out = self.forward()
        return out
