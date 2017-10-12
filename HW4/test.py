import numpy as np
import torch
from neural_network import NeuralNetwork
from my_img2num import MyImg2Num
from nn_img2num import NnImg2Num
import time

def main():
    print('Homework 4 ==========================')

    t0 = time.time()
    #myNN = MyImg2Num()
    #myNN.train()
    myNN = NnImg2Num()
    myNN.train()
    tf = time.time()
    print('Duration '+str(tf-t0)+'s')

if __name__ == "__main__":
    main()
