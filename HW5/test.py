import numpy as np
import torch
from img2num import img2num
from img2obj import img2obj
import time
from torchvision.datasets import CIFAR100
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision
import matplotlib.pyplot as plt
import cv2

def main():
    print('Homework 5 ==========================')
    
    ### Part A
    '''
    t0 = time.time()
    LN = img2num()
    LN.train()
    FC = NnImg2Num()
    FC.train()
    tf = time.time()
    print('Duration '+str(tf-t0)+'s')
    '''


    ### Part B
    # train new model
    #LN = img2obj()
    #LN.train()
    #torch.save(LN,'./model.pth')


    # load trained model
    LN = torch.load('./model.pth')
    
    
    # camera live caption
    LN.cam(0)
    
    
    # read and process image from file
    img = cv2.imread('beetle.jpg')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    height, width, channels = img.shape
    crop_img = img[0:height, int((width-height)/2):height+int((width-height)/2)]
    resized_img = cv2.resize(crop_img, (32, 32))
    #print(torch.from_numpy(np.transpose(np.array(resized_img),(2,0,1))))
    process = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    img = process(np.array(resized_img))
    LN.view(img)
    
    
    
    
    
    
    '''
    # load test set
    test_batch_size = 1
    test_dataset = CIFAR100(root="../dataset",download = False, train=False, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))
    test_batch_num = len(test_dataset)/test_batch_size
    test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=True)
    
    
    # plot image from test set
    dataiter = iter(test_loader)
    images, labels = dataiter.next()
    
    #LN.view(torchvision.utils.make_grid(images))
    LN.view(images)
    print('Actual label: '+', '.join('%5s' % LN.classes[labels[j]] for j in range(test_batch_size)))
    plt.show()
    '''

if __name__ == "__main__":
    main()
