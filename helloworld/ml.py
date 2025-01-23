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



def _dims(v):
      if not isinstance(v, list):
        return []
      dimension = [len(v)]
      if len(v) > 0 and isinstance(v[0], list):
        dimension.extend(_dims(v[0]))
      return dimension





class BB:
    def __init__(self, *args):
       self.args = list(args)
       self.value = self.args[0]


    def dims(self):
      return _dims(self.value)

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

       if self.arg(0).dims() != self.arg(1).dims():
          raise ValueError(f"Dimentions unmatch input->{self.arg(0).dims()}, bias->{self.arg(1).dims()}")


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

       if self.input.dims()[1] != self.weights.dims()[0]:
          raise ValueError(f"Dimentions unmatch input->{self.input.dims()}, weights->{self.weights.dims()}")

       self.value = matmul(self.input.val(), self.weights.val())
 
    def dif(self, dvalue):
       super().dif(dvalue)
       self.weights.dif( matmul(transpose(self.input.val()), self.dval()))
       self.input.dif( matmul(self.dval(), transpose(self.weights.val())) ) 


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

