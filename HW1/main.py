#!/usr/bin/env python
"""BME595A Homework 1 Fan Fei Aug 28 2017"""

import torch
import numpy as np
from conv import Conv2D
from PIL import Image
import time


def main():
    
    print("Task 3")
    
    # loading image and convert to torch floatTensor
    img_in_pil = Image.open('1080p.jpg')
    img_in_np = np.array(img_in_pil)
    img_in_np_transposed = np.transpose(img_in_np,(2,0,1))

    # img_in_torch has dim 3xhxw
    img_in_torch = torch.from_numpy(img_in_np_transposed)
    '''
    #Part A
    conv2d = Conv2D(3,1,3,1,'known')
    
    # convolve image
    [num_of_operation, img_out_torch]=conv2d.forward(img_in_torch)
     
    # print # of operations
    print("Number of operations = " + str(num_of_operation))
    
    # save output image
    
    # 1 x h x w
    img_out_np_transposed = img_out_torch.numpy()
    
    # h x w x 1
    img_out_np = np.transpose(img_out_np_transposed,(1,2,0))
    
    out_size = img_out_np.shape
    img_out_np_3 = np.zeros((out_size[0],out_size[1],3),dtype = 'uint8')
    for i in range(0,3):
        img_out_np_3[:,:,i] = img_out_np[:,:,0]

    img_out_pil = Image.fromarray(img_out_np_3)
    img_out_pil.save('Task1_1080p.png')
    '''
    '''
    # Part B
    itr = 11
    num_out_channel=np.zeros((itr,1),dtype = 'int')
    duration = np.zeros((itr,1))
    for i in range(0,itr):
        num_out = pow(2,i)
        num_out_channel[i,0] = num_out
        print('i = ' + str(i) + ', o_channel = ' + str(num_out))
        t_0 = time.time()
        conv2d = Conv2D(3,num_out,3,1,'rand')
        [num_of_operation, img_out_torch]=conv2d.forward(img_in_torch)
        t_f = time.time()
        dt = t_f-t_0
        duration[i,0] = dt
        print(img_out_torch.shape)
        print(dt)
    print('Convolution duration:')
    print(duration)
    '''
    # Part C
    itr = 5
    kernel_size = np.array([3, 5, 7, 9, 11])
    for i in range(0,itr):
        print(i)
        conv2d = Conv2D(3,2,kernel_size[i],1,'rand')
        [num_of_operation, img_out_torch]=conv2d.forward(img_in_torch)
        print("Kernel size = " + str(kernel_size[i]) + ", Number of operations = " + str(num_of_operation))

if __name__ == "__main__":
    main()
