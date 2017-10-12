import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.autograd import Variable

class NnImg2Num(nn.Module):
    def __init__(self,layer_config = [784,200,50,10]):
        super(NnImg2Num, self).__init__()
        self.layer_config = layer_config
        self.layer_size = len(layer_config)
        
        self.fc1 = nn.Linear(layer_config[0], layer_config[1])
        self.fc2 = nn.Linear(layer_config[1], layer_config[2])
        self.fc3 = nn.Linear(layer_config[2], layer_config[3])

    def forward(self, x):
        x = self.fc1(x)
        x = F.sigmoid(x)
        x = self.fc2(x)
        x = F.sigmoid(x)
        x = self.fc3(x)
        #x = F.sigmoid(x)
        x = F.log_softmax(x)
        '''
        x = F.sigmoid(self.fc1(x))
        x = F.sigmoid(self.fc2(x))
        x = F.log_softmax(self.fc3(x))'''
        return x

    def train(self,epoch = 50):
        eta = 1
        batch_size = 1000
        epoch_num = 50
        # load training data
        dataset = MNIST(root="../dataset",download = True, train=True, transform=transforms.ToTensor())
        batch_num = len(dataset)/batch_size
        
        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # load test data
        test_batch_size = 1000
        
        test_dataset = MNIST(root="../dataset",download = True, train=False, transform=transforms.ToTensor())
        test_batch_num = len(test_dataset)/test_batch_size
        test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=True)
        training_error = torch.Tensor([])
        test_error = torch.Tensor([])
        test_percent = torch.Tensor([])

        optimizer = optim.SGD(self.parameters(),lr = 1)

        for i in range(epoch_num):
            # training
            print('EPOCH '+str(i))
            epoch_avg_loss = 0
            epoch_test_loss = 0
            for batch_index, (data, target) in enumerate(train_loader):
                data = data.view(batch_size,-1)# data [batch_size * input_size]
                data, target = Variable(data), Variable(target)
                optimizer.zero_grad()
                output = self.forward(data)
                print(output)
                print(target)
                loss = F.nll_loss(output, target)
                print(loss)
                loss.backward()
                optimizer.step()
                current_loss = loss.data[0]
                epoch_avg_loss = epoch_avg_loss+current_loss
            epoch_avg_loss = torch.Tensor([epoch_avg_loss/batch_num])
            epoch_avg_loss = epoch_avg_loss.view(1,1)
            training_error = torch.cat((training_error,epoch_avg_loss),1)
            print('current epoch loss is'+str(training_error))

            # testing
            correct = 0
            for data, target in test_loader:
                data = data.view(test_batch_size,-1)
                data, target = Variable(data, volatile=True), Variable(target)
                output = self.forward(data)
                loss = F.nll_loss(output, target, size_average=False)
                test_loss = loss.data[0]
                epoch_test_loss = epoch_test_loss+test_loss
                pred = output.data.max(1,keepdim=True)[1]
                correct += pred.eq(target.data.view_as(pred)).cpu().sum()
            epoch_test_loss = torch.Tensor([epoch_test_loss/len(test_dataset)])
            epoch_test_loss = epoch_test_loss.view(1,1)
            test_error = torch.cat((test_error,epoch_test_loss),1)
            percent = float(correct)/len(test_loader.dataset)
            #print(correct)
            #print(len(test_loader.dataset))
            epoch_test_percent = torch.Tensor([percent])
            test_percent = torch.cat((test_percent,epoch_test_percent),0)
            print('current test loss is'+str(test_error))
            print('current test accuracy is: '+str(test_percent))

