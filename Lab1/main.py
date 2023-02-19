import numpy as np

def unit_step(v):
    if v >= 0:
        return 1
    else:
        return 0

def perceptron(x, w, b):
    v = np.dot(w, x) + b
    y = unit_step(v)
    return y

def NOT_perceptron(x):
    return perceptron(x, w=-1, b=0.5)

def AND_perceptron(x):
    w = np.array([1, 1])
    return perceptron(x, w=w, b=-1.5)

def OR_perceptron(x):
    w = np.array([1, 1])
    return perceptron(x, w=w, b=-0.5)

    #XOR(x1, x2) = AND(NOT(AND(x1,x2)), OR(x1,x2))
def XOR_perceptron(x):
    y1 = NOT_perceptron(AND_perceptron(x))
    y2 = OR_perceptron(x)
    new_x = np.array([y1, y2])
    return AND_perceptron(new_x)

if __name__ == '__main__':
    print(XOR_perceptron([1, 1]))
    print(XOR_perceptron([1, 0]))
    print(XOR_perceptron([0, 1]))
    print(XOR_perceptron([0, 0]))

