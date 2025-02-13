# Classifier model for TicTacToe board
from . import ml

xx = ml.BB(ml.random_matrix(6,6))
ww1 = ml.BB(ml.random_matrix(36, 24))
bb1 = ml.BB(ml.random_matrix(1, 24))
ww2 = ml.BB(ml.random_matrix(24, 4))
bb2 = ml.BB(ml.random_matrix(1, 4))

ww3 = ml.BB(ml.random_matrix(4, 1))
bb3 = ml.BB(ml.random_matrix(1, 1))



# Forward pass
zz0 = ml.BBReshape(xx, 1, 36)
zz1 = (zz0 @ ww1 + bb1).sigmoid()
zz2 = (zz1 @ ww2 + bb2).sigmoid()
zz3 = (zz2 @ ww3 + bb3).sigmoid()

yy = ml.BB(ml.random_matrix(1, 1))
lloss = zz3.mse(yy)
