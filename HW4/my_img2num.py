import numpy as np
import torch

from neural_network import NeuralNetwork
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torchvision import transforms
from neural_network import NeuralNetwork


class MyImg2Num(object):
    
    def __init__(self):
        myNN_size = [784, 200, 50, 10]
        self.myNN = NeuralNetwork(myNN_size)
    
    def train(self):
        eta = 1
        batch_size = 1000
        epoch_num = 50
        
        dataset = MNIST(root="../dataset",download = True, train=True, transform=transforms.ToTensor())
        batch_num = len(dataset)/batch_size
        
        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        def one_hot_label(batch_target):
            label = np.zeros([10,batch_size])
            label[batch_target.numpy(),np.arange(batch_size)]=1
            torch_label = torch.from_numpy(label)
            return torch_label.float()
        
        test_batch_size = 1000
    
        test_dataset = MNIST(root="../dataset",download = True, train=False, transform=transforms.ToTensor())
        test_batch_num = len(test_dataset)/test_batch_size
        test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=True)
        training_error = torch.Tensor([])
        test_error = torch.Tensor([])
        
        for i in range(epoch_num):
            # training
            print('EPOCH '+str(i))
            epoch_avg_loss = 0
            epoch_test_loss = 0
            for batch_index, (features, target) in enumerate(train_loader):
                
                feature_flat = features.view(-1,features.shape[1]*features.shape[2]*features.shape[3]).t()
                #print(features.shape)
                #print(feature_flat)
                target = one_hot_label(target)
                #print(target)
                self.myNN.forward(feature_flat)
                self.myNN.backward(target,'CE')
                self.myNN.updateParams(eta)
                current_loss = self.myNN.Loss
                #print('current batch loss is '+str(current_loss))
                epoch_avg_loss = epoch_avg_loss+current_loss
                #print(target)
                #print(self.myNN.forward(feature_flat))
            epoch_avg_loss = torch.Tensor([[epoch_avg_loss/batch_num]])
            training_error = torch.cat((training_error,epoch_avg_loss),1)
            # print training error per class
            print('current epoch loss is'+str(training_error))
            # print test error per class
            for batch_index, (features, target) in enumerate(test_loader):
                feature_flat = features.view(-1,features.shape[1]*features.shape[2]*features.shape[3]).t()
                Y = one_hot_label(target)
                m = feature_flat.shape[1]
                AO = self.myNN.forward(feature_flat)
                cost = -1/m*torch.sum(Y*torch.log(AO)+(1-Y)*torch.log(1-AO),dim=1)
                test_loss = torch.sum(cost)/Y.shape[0]
                epoch_test_loss = epoch_test_loss+test_loss
            epoch_test_loss = torch.Tensor([[epoch_test_loss/test_batch_num]])
            test_error = torch.cat((test_error,epoch_test_loss),1)
            print('current test loss is'+str(test_error))
                
    def forward(self,img):
        input = img.view(-1,img.shape[1]*img.shape[2]*img.shape[3]).t()
        output = self.myNN.forward(input)
        out_prob, out_num = torch.max(output,0)
        return out_num
