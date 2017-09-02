# BME595HW
BME595A Homework2 Fall 2017

### Part A
The input layer dimension is initiated by list layer_size with randomized weights theta_i and zero bias bias_i. Weights and bias are stored in the object in dictionary network. The getLayer function will return theta_i as a torch Tensor. The forward function will forward propogate the network to get the output.

The input could be 1D or 2D Tensor, the input examples could be stacked either as column or row vectors. The prefered way is to stack input examples (column vectors) horizontally (only need to do so when the number of examples = number of features).

The testing of creating a network object, access and modify weights and forward propogate using 1D, 2D row vectors, 2D column vectors are in the test.py with printed explinations.

### Part B
The logic gate classes can take boolean or int of floating number as input, where anything other than 0 is True.
The testing input of the 4 gates are as follows.

#### AND
print(AND(False,False))

print(AND(False,True))

print(AND(True,False))

print(AND(True,True))

#### OR
print(OR(False,False))

print(OR(False,True))

print(OR(True,False))

print(OR(True,True))


#### NOT
print(NOT(False))

print(NOT(True))

#### XOR
print(XOR(False,False))

print(XOR(False,True))

print(XOR(True,False))

print(XOR(True,True))
