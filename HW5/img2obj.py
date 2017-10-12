import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision.datasets import CIFAR100
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.autograd import Variable
import time
import matplotlib.pyplot as plt
import cv2

class img2obj(nn.Module):
    def __init__(self):
        # load data

        # load training data
        self.batch_size = 50
        batch_size = self.batch_size
        
        self.dataset = CIFAR100(root="../dataset",download = False, train=True, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))
        dataset = self.dataset
        
        self.batch_num = len(dataset)/batch_size
        
        self.train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # load test data
        self.test_batch_size = 1000
        test_batch_size = self.test_batch_size
        
        self.test_dataset = CIFAR100(root="../dataset",download = False, train=False, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))
        test_dataset = self.test_dataset
        
        self.test_batch_num = len(test_dataset)/test_batch_size
        
        self.test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=True)
        
        # class label
        self.classes = ('apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock', 'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur', 'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion', 'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse', 'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear', 'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine', 'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose', 'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table', 'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout', 'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm')

        super(img2obj, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 5)
        self.conv2 = nn.Conv2d(64, 128, 5)
        self.conv3 = nn.Conv2d(128, 512, 5)
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 100)
    
    
    def forward(self, x):
        x = self.conv1(x)
        x = F.max_pool2d(x,2)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.max_pool2d(x,2)
        x = F.relu(x)
        x = self.conv3(x)
        x = x.view(-1,512)
        x = F.relu(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x
    
    
    
    def train(self,epoch_num = 15):
        batch_size = self.batch_size
        dataset = self.dataset
        batch_num = self.batch_num
        train_loader = self.train_loader
        test_batch_size = self.test_batch_size
        test_dataset = self.test_dataset
        test_batch_num = self.test_batch_num
        test_loader = self.test_loader
        classes = self.classes

        # training parameter
        eta = 0.025
        
        # initialize recording variables
        training_error = torch.Tensor([])
        test_error = torch.Tensor([])
        test_percent = torch.Tensor([])
        training_time = 0
        test_time = 0
        
        optimizer = optim.SGD(self.parameters(),lr = eta, momentum=0.5)
        
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
                loss = F.cross_entropy(output, target)
                loss.backward()
                optimizer.step()
                current_loss = loss.data[0]
                print(current_loss)
                epoch_avg_loss = epoch_avg_loss+current_loss
            epoch_avg_loss = torch.Tensor([epoch_avg_loss/batch_num])
            epoch_avg_loss = epoch_avg_loss.view(1,1)
            training_error = torch.cat((training_error,epoch_avg_loss),0)
            tf = time.time()
            training_time = training_time + (tf-t0)
            print('current epoch loss is'+str(training_error))
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
            
            print('current test loss is'+str(test_error))
            print('current test accuracy is: '+str(test_percent))
        print('Training loss'+str(training_error))
        print('Total training duration '+str(training_time)+'s')
        print('Test loss'+str(test_error))
        print('Test accuracy is: '+str(test_percent))
        print('Total test duration '+str(test_time)+'s')

        '''
    def view(self, img):
        # prediction
        input = Variable(img, volatile=True)
        output = self.forward(input)
        pred = output.data.max(1,keepdim=True)[1]
        pred = int(pred.numpy())
        print('Prediction is: ' + self.classes[pred])
        
        # plot image
        img = torch.squeeze(img)
        img = img/2+0.5
        img_np = img.numpy()
        plt.imshow(np.transpose(img_np, (1,2,0)))
        # plt.show()
        '''
    def view(self,img):
        # prediction
        img_input = img.view(1,img.shape[0],img.shape[1],img.shape[2])
        input = Variable(img_input, volatile=True)
        output = self.forward(input)
        pred = output.data.max(1,keepdim=True)[1]
        pred = int(pred.numpy())
        print('Prediction is: ' + self.classes[pred])
        # reconstruct image
        img = (img*0.5+0.5)*255
        img = img.byte()
        img_np = img.numpy()
        plt.imshow(np.transpose(img_np, (1,2,0)))
        plt.show()

    def cam(self,cam_index):
        cam = cv2.VideoCapture(cam_index)
        cv2.namedWindow('test')
        img_counter = 0
        font = cv2.FONT_HERSHEY_SIMPLEX
        print('Press esc in video window to exit')
        while True:
            ret, frame = cam.read()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            height, width, channels = frame.shape
            crop_frame = frame[0:height, int((width-height)/2):height+int((width-height)/2)]
            resized_frame = cv2.resize(crop_frame, (32, 32))
            # format the 32x32 image
            process = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
            img = process(np.array(resized_frame))
            img = img.view(1,img.shape[0],img.shape[1],img.shape[2])
            # prediction
            input = Variable(img, volatile=True)
            output = self.forward(input)
            pred = output.data.max(1,keepdim=True)[1]
            pred = int(pred.numpy())
            captn_frame = cv2.putText(cv2.cvtColor(crop_frame, cv2.COLOR_RGB2BGR),self.classes[pred],(int(height/3),height-20),font,2,(255,255,255),2,cv2.LINE_AA)
            
            cv2.imshow('test', captn_frame)
            if not ret:
                break
            k = cv2.waitKey(1)
            
            if k%256 == 27:
                # ESC pressed
                print('Closing camera')
                break
        cam.release()
        cv2.destroyAllWindows()
