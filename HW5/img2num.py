import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.autograd import Variable
import time

class img2num(nn.Module):
    def __init__(self):
        super(img2num, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5, padding=2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.conv3 = nn.Conv2d(16, 120, 5)
        self.fc1 = nn.Linear(120, 84)
        self.fc2 = nn.Linear(84, 10)
    
    def forward(self, x):
        x = self.conv1(x)
        x = F.max_pool2d(x,2)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.max_pool2d(x,2)
        x = F.relu(x)
        x = self.conv3(x)
        x = x.view(-1,120)
        x = F.relu(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        #x = F.log_softmax(x)
        return x
    
    def train(self,epoch = 50):
        eta = 0.1
        batch_size = 100
        epoch_num = 50
        # load training data
        dataset = MNIST(root="../dataset",download = True, train=True, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,))]))
        batch_num = len(dataset)/batch_size
        
        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # load test data
        test_batch_size = 1000
        
        test_dataset = MNIST(root="../dataset",download = True, train=False, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,))]))
        test_batch_num = len(test_dataset)/test_batch_size
        test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=True)
        training_error = torch.Tensor([])
        test_error = torch.Tensor([])
        test_percent = torch.Tensor([])
        training_time = 0
        test_time = 0
        
        optimizer = optim.SGD(self.parameters(),lr = eta)
        
        for i in range(epoch_num):
            # training
            print('EPOCH '+str(i))
            t0 = time.time()
            epoch_avg_loss = 0
            epoch_test_loss = 0
            for batch_index, (data, target) in enumerate(train_loader):
                # data = data.view(batch_size,-1)# data [batch_size * input_size]
                data, target = Variable(data), Variable(target)
                optimizer.zero_grad()
                output = self.forward(data)
                #print(output)
                loss = F.cross_entropy(output, target)
                loss.backward()
                optimizer.step()
                current_loss = loss.data[0]
                epoch_avg_loss = epoch_avg_loss+current_loss
            epoch_avg_loss = torch.Tensor([epoch_avg_loss/batch_num])
            epoch_avg_loss = epoch_avg_loss.view(1,1)
            training_error = torch.cat((training_error,epoch_avg_loss),0)
            tf = time.time()
            training_time = training_time + (tf-t0)
            #print('current epoch loss is'+str(training_error))
            #print('Total training duration '+str(training_time)+'s')
            
            # testing
            t0 = time.time()
            correct = 0
            for batch_index, (data, target) in enumerate(test_loader):
                # data = data.view(test_batch_size,-1)
                data, target = Variable(data, volatile=True), Variable(target)
                output = self.forward(data)
                loss = F.cross_entropy(output, target)
                test_loss = loss.data[0]
                epoch_test_loss = epoch_test_loss+test_loss
                pred = output.data.max(1,keepdim=True)[1]
                correct += pred.eq(target.data.view_as(pred)).cpu().sum()
            epoch_test_loss = torch.Tensor([epoch_test_loss/test_batch_num])
            epoch_test_loss = epoch_test_loss.view(1,1)
            test_error = torch.cat((test_error,epoch_test_loss),0)
            percent = float(correct)/len(test_loader.dataset)
            epoch_test_percent = torch.Tensor([percent])
            test_percent = torch.cat((test_percent,epoch_test_percent),0)
            tf = time.time()
            test_time = test_time + (tf-t0)
            
            #print('current test loss is'+str(test_error))
            #print('current test accuracy is: '+str(test_percent))
        print('Training loss'+str(training_error))
        print('Total training duration '+str(training_time)+'s')
        print('Test loss'+str(test_error))
        print('Test accuracy is: '+str(test_percent))
        print('Total test duration '+str(test_time)+'s')
