from lib import ml, game
import random


def random_mat(m, n):
    return [[random.random() for _ in range(n)] for _ in range(m)]


def gradient_backpropagation(x, y, w1, b1, w2, b2):

    xx = ml.BB(x)
    ww1 = ml.BB(w1)
    bb1 = ml.BB(b1)
    ww2 = ml.BB(w2)
    bb2 = ml.BB(b2)
    yy = ml.BB(y)

    # Forward pass
    zz0 = ml.BBReshape(xx, 1, 36)
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

    return lloss, ww1, bb1, ww2, bb2, zz2




w1 = random_mat(36, 16)
b1 = random_mat(1, 16)
w2 = random_mat(16, 1)
b2 = [[-0.9]]


# 120 still works, but larger - not well
boards, winners = game.generate_batch(60) 


sum_loss = 0
j = 0
for i in range(100000):
  
  for board, winner in zip(boards, winners):
    lloss, ww1, bb1, ww2, bb2, zz2 = gradient_backpropagation(board, [winner], w1, b1, w2, b2)
    loss, w1, b1, w2, b2, prediction = lloss.val(), ww1.val(), bb1.val(), ww2.val(), bb2.val(), zz2.val()
    sum_loss = sum_loss + loss[0][0]

    if random.random() > 0.999:
      print(f"Loss {i}: {sum_loss/500}")
      print("Winner:", winner[0], " Prediction: ", prediction[0][0], "error: ", (winner[0] - prediction[0][0])**2)
      game.print_board(board)
      print()
      
      sum_loss = 0


