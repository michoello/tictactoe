# Classifier model for TicTacToe board
from . import ml

x = ml.BB(ml.random_matrix(6,6))
w1 = ml.BB(ml.random_matrix(36, 24))
b1 = ml.BB(ml.random_matrix(1, 24))

w2 = ml.BB(ml.random_matrix(24, 4))
b2 = ml.BB(ml.random_matrix(1, 4))

w3 = ml.BB(ml.random_matrix(4, 1))
b3 = ml.BB(ml.random_matrix(1, 1))



# Forward pass
z0 = ml.BBReshape(x, 1, 36)
z1 = (z0 @ w1 + b1).sigmoid()
z2 = (z1 @ w2 + b2).sigmoid()
z3 = (z2 @ w3 + b3).sigmoid()


# Predicts who is the winner (1 for crosses, -1 for zeroes)
prediction = z3

y = ml.BB(ml.random_matrix(1, 1))
loss = z3.mse(y)
