import math

learning_rate = 0.01


# Sigmoid activation and its derivative
def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def vectorized(f, x):
    return [[f(v) for v in row] for row in x]


def sigmoid_derivative(x):
    return [[v * (1 - v) for v in row] for row in  vectorized(sigmoid, x)]

# Transpose
def transpose(A):
    return list(zip(*A))

# Matrix multiplication
def matmul(A, B):
    return [[sum(a * b for a, b in zip(row, col)) for col in transpose(B)] for row in A]


# Element-wise addition
def add(A, B):
    return [[a + b for a, b in zip(row_a, row_b)] for row_a, row_b in zip(A, B)]

# Element-wise subtraction
def subtract(A, B):
    return [[a - b for a, b in zip(row_a, row_b)] for row_a, row_b in zip(A, B)]

# Element-wise multiplication
def elementwise_mul(A, B):
    return [[a * b for a, b in zip(row_a, row_b)] for row_a, row_b in zip(A, B)]

def mul(matrix, scalar):
    return [[element * scalar for element in row] for row in matrix]




class BB:
    def __init__(self, *args):
       self.args = list(args)
       self.value = self.args[0]

    def val(self):
      return self.value

    def dif(self, dvalue):
      self.dvalue = dvalue

    def dval(self):
      return self.dvalue

    def arg(self, i):
       return self.args[i]

    def appl(self):
       self.value = subtract(self.value, mul(self.dvalue, learning_rate))

    def __matmul__(self, other):
       return BBMatmul(self, other)

    def __add__(self, other):
       return BBSum(self, other)



class BBSum(BB):
    def __init__(self, arg1, arg2):
       super().__init__(arg1, arg2)
       self.value = add(self.arg(0).val(), self.arg(1).val())

    def dif(self, dvalue):
       super().dif(dvalue)
       self.arg(0).dif(self.dval())
       self.arg(1).dif(self.dval())


class BBMatmul(BB):
    def __init__(self, arg1, arg2):
       super().__init__(arg1, arg2)
       self.input = self.arg(0)
       self.weights = self.arg(1)

       self.value = matmul(self.input.val(), self.weights.val())
 
    def dif(self, dvalue):
       super().dif(dvalue)
       self.weights.dif( matmul(transpose(self.input.val()), self.dval()))
       self.input.dif( matmul(self.dval(), transpose(self.weights.val())) ) 

       # Somehow, this works as well, though a bit different losses
       # are produced, but still reducing over time.
       #self.weights.dif( matmul(self.input.val(), self.dval()))
       #self.input.dif( matmul( self.weights.val(), self.dval()) ) 


class BBSigmoid(BB):
    def __init__(self, inp):
       self.inp = inp
       self.value = vectorized(sigmoid, inp.val())

    def derivative(self):
       return sigmoid_derivative(self.inp.val())

    def dif(self, dvalue):
      self.dvalue = elementwise_mul(dvalue, self.derivative())
      self.inp.dif(self.dval()) 


class BBMSELoss(BB):
    def __init__(self, inp, y):
       self.inp = inp
       self.y = y
       self.value = 0.5 * sum((a - b) ** 2 for a, b in zip(inp.val()[0], y.val()[0]))

    def derivative(self):
       return subtract(self.inp.val(), self.y.val()) # derivative of loss
 
    def dif(self, dvalue):
      self.dvalue = self.derivative()
      self.inp.dif(self.dval()) 



def gradient_backpropagation(x, y, w1, b1, w2, b2):

    xx = BB(x)
    ww1 = BB(w1)
    bb1 = BB(b1)
    ww2 = BB(w2)
    bb2 = BB(b2)
    yy = BB(y)

    # Forward pass
    zz1 = BBSigmoid(xx @ ww1  + bb1)
    zz2 = BBSigmoid(zz1 @ ww2 + bb2)

    lloss = BBMSELoss(zz2, yy)

    # Backward pass
    lloss.dif(1)

    # Update weights and biases
    ww1.appl() 
    bb1.appl() 
    ww2.appl()
    bb2.appl()

    xx.appl()

    return lloss.val(), xx.val(), ww1.val(), bb1.val(), ww2.val(), bb2.val()

# Run the example
x = [[0.5, 0.9]]  # Input (1 sample, 2 features)
w1 = [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]  # Weights for layer 1 (2x2)
b1 = [[0.1, 0.2, 0.3]]               # Biases for layer 1 (1x2)
w2 = [[0.5], [0.6], [0.7]]             # Weights for layer 2 (2x1)
b2 = [[0.3]]                    # Biases for layer 2 (1x1)

y = [[0.001]]         # True output (1 sample, 1 target)

for i in range(100000):
   loss, _, w1, b1, w2, b2 = gradient_backpropagation(x, y, w1, b1, w2, b2)
   #loss, x, _, _, _, _ = gradient_backpropagation(x, y, w1, b1, w2, b2)
   if i % 5000 == 0:
       print(f"Loss {i}: {loss}")



