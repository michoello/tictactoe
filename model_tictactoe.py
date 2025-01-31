from helloworld import ml
import random


def gradient_backpropagation(x, y, w1, b1, w2, b2):

    xx = ml.BB(x)
    ww1 = ml.BB(w1)
    bb1 = ml.BB(b1)
    ww2 = ml.BB(w2)
    bb2 = ml.BB(b2)
    yy = ml.BB(y)

    # Forward pass
    zz0 = ml.BBReshape(xx, 1, 32)
    zz1 = (zz0 @ ww1  + bb1).sigmoid()
    zz2 = (zz1 @ ww2 + bb2).sigmoid()
    lloss = zz2.mse(yy)

    # Backward pass
    lloss.dif()

    # Update weights and biases
    ww1.appl(0.1) 
    bb1.appl(0.1) 
    ww2.appl(0.1)
    bb2.appl(0.1)

    return lloss.val(), xx.val(), ww1.val(), bb1.val(), ww2.val(), bb2.val()


def random_mat(m, n):
    return [[random.random() for _ in range(n)] for _ in range(m)]

# Run the example
x1 = [
  [ 0, 0, 0, 0, 0, 0],
  [ 0, 0, 0, 0, 0, 0],
  [ 0, 1, -1, -1, -1, 0],
  [ 0, 1, 0, 0, 0, 0],
  [ 0, 1, 0, 0, 0, 0],
  [ 0, 1, 0, 0, 0, 0],
]

x2 = [
  [ 0, 1, 0, 1, 1, 0],
  [ 0, 0, 1, 0, 0, 0],
  [ 0, 0, 0, 1, 0, 0],
  [ 0,-1,-1,-1,-1, 0],
  [ 0, 0, 0, 0, 0, 0],
  [ 0, 0, 0, 0, 0, 0],
]


x3 = [
  [ 0, 1, 0, 1, 1, 0],
  [ 0, 0, 1, 0, 0, 0],
  [ 0, 0, 0, 1, 0, 0],
  [ 0,-1, 1,-1,-1, 0],
  [ 0, 0,-1,-1, 0, 0],
  [ 0, 0, 0, 0, 0, 0],
]


xs = [x1, x2, x3]

w1 = random_mat(32, 16)
b1 = random_mat(1, 16)
w2 = random_mat(16, 1)
b2 = [[-0.9]]

ys = [[1], [0], [0.5]]         # True output (1 sample, 1 target)

for i in range(10000):
   for x, y in zip(xs, ys):
      loss, _, w1, b1, w2, b2 = gradient_backpropagation(x, [y], w1, b1, w2, b2)
   #loss, _, w1, b1, w2, b2 = gradient_backpropagation(x, y, w1, b1, w2, b2)
   if i % 500 == 0:
       print(f"Loss {i}: {loss}")



