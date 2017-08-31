import numpy as np
import torch

class Conv2D(object):
    def __init__(self,in_channel, o_channel, kernel_size, stride, mode):
        self.in_channel = in_channel
        self.o_channel = o_channel
        self.kernel_size = kernel_size
        self.stride = stride
        self.mode = mode

    def forward(self,input_image):
        # convert torch FloatTensor to numpy ndarray 
        
        input_image_np = input_image.numpy()
        count = 0
        
        # get kernel
        if self.mode =='known':
            k1 = np.array([[-1, -1, -1],  [0, 0, 0],  [1, 1, 1]])
            k2 = np.array([[-1,  0,  1], [-1, 0, 1], [-1, 0, 1]])
            k3 = np.array([[1,  1,  1], [1, 1, 1], [1, 1, 1]])
            k4 = np.array([[-1, -1, -1, -1, -1], [-1, -1, -1, -1, -1], [0, 0, 0, 0, 0], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1]])
            k5 = np.array([[-1, -1, 0, 1, 1], [-1, -1, 0, 1, 1], [-1, -1, 0, 1, 1], [-1, -1, 0, 1, 1], [-1, -1, 0, 1, 1]])
            kernel = k1
        elif self.mode == 'rand':
            k_rand = np.random.randint(2,size = (1,self.kernel_size,self.kernel_size),dtype = 'int')
            kernel = k_rand
        else:
            raise Exception('Mode error!')

        # convolution
        # getting size
        input_size = input_image_np.shape
        output_size = (self.o_channel,int(np.ceil((input_size[1]-self.kernel_size+1)/self.stride)),int((input_size[2]-self.kernel_size+1)/self.stride))
        
        # initialize input & output images
        input_image_np_gs = np.zeros((1,input_size[1],input_size[2]), dtype = 'int')
        # output k th channel
        output_image_np_k = np.zeros((1,output_size[1],output_size[2]), dtype = 'int')
        output_image_np = np.zeros(output_size, dtype = 'int')
        
        window = np.zeros((1,self.kernel_size,self.kernel_size),dtype = 'int')
        
        # summing input image to 1 x h x w
        for i in range(0,3):
            input_image_np_gs[0,:,:] +=input_image_np[i,:,:]
            count += input_size[1]*input_size[2]

        # convolve
        for k in range(0,self.o_channel):
            for i in range(0, output_size[1]):
                for j in range(0, output_size[2]):
                    window = input_image_np_gs[0,i*self.stride:i*self.stride+self.kernel_size,j*self.stride:j*self.stride+self.kernel_size]
                    output_image_np_k [0,i,j] = np.sum(np.multiply(window,kernel))
                    count += self.kernel_size*self.kernel_size*2-1
            output_image_np[k,:,:] = (output_image_np_k-output_image_np_k.min())/(output_image_np_k.max()-output_image_np_k.min())*255
            print(str(k+1) + '/' + str(self.o_channel)  + ' done.')
        '''# for debug
        input_size = input_image_np.shape
        noise  = np.random.randint(100,size = (input_size[0],input_size[1],input_size[2]),dtype = 'uint8')
        output_image_np = input_image_np+noise
        '''

        # convert ndarray to FloatTensor
        output_image = torch.from_numpy(output_image_np)
        return (count,output_image)




