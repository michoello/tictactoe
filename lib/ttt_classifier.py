# Classifier model for TicTacToe board
from . import ml

xx = ml.BB(ml.random_matrix(6,6))
ww1 = ml.BB(ml.random_matrix(36, 24))
bb1 = ml.BB(ml.random_matrix(1, 24))
ww2 = ml.BB(ml.random_matrix(24, 1))
bb2 = ml.BB(ml.random_matrix(1, 1))
yy = ml.BB(ml.random_matrix(1, 1))

# Forward pass
zz0 = ml.BBReshape(xx, 1, 36)
zz1 = (zz0 @ ww1 + bb1).sigmoid()
zz2 = (zz1 @ ww2 + bb2).sigmoid()
lloss = zz2.mse(yy)
