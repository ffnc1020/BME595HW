# BME595A Homework2 Fall 2017
## Fan Fei

### Summary
In nerual_network.py the backward function will utilize the the variable calculated in the forward function to calculate the gradient. Specifically, linear activation $$Z[i] = W[i]A[i-1]$$ is and $A = sigmoid(Z)$ are used, where $W$ is the weight matrix and $A$ is the activation. Given the initial $\frac{\partial Loss}{\partial Y}$, the derivative with respect to Loss of all the previous $Z$, $W$ and $A$ can be calculated base on chain rule.

In the test.py all the training function a truth table is defined. By randomly index the truth table, large number of training example can be generated. The traning is done with 100 example per epoch, and trained over 10000 epoch for XOR gate and 1000 epoch for all other gates.

### Results

The hand picked parameter for AND gate is [-1.5, 1, 1]. The trained parameter is [-4.4474, 2.8903, 2.8784]. These two parameters are similar as they are almost proportional. The learning rate for the training is 0.5.

The hand picked parameter for OR gate is [0, 1, 1]. The trained parameter is [-1.5007, 3.6123, 3.6085]. The trained parameter has almost equal value on the weight multiplied with the two input where the bias is less than half of the weight. This make the linear activation to be less than zero and output close to zero only when both inputs are 0, hense the AND gate. The learning rate for the training is 0.5.

The hand picked parameter for NOT gate is [1, -1]. The trained parameter is [2.3028, -4.8369]. Again the traind parameter is proportional to the hand picked gain. The learning rate for the training is 0.5.

The hand picked parameter for XOR gate is [[-10,20,20],[30,-20,-20]] and [[-30,20,20]] for the first and second layer. The trained parameter is [[7.9607, -5.3331, -5.3352],[2.9055 -6.9012 -6.9033]] and [[-5.4625, 11.3904, -11.5133]]. The weights are not exitly the same, however give the network has two layers, different configuraiton could result in the same result give this simple logic. Testing shows the output logic is correct. The learning rate for the training is 10, smaller learning rate require larger numbers of iteration.

