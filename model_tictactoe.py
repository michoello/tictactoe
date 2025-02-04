from lib import ml, game
import random


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

    return lloss.val(), xx.val(), ww1.val(), bb1.val(), ww2.val(), bb2.val(), zz2.val()


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

w1 = random_mat(36, 16)
b1 = random_mat(1, 16)
w2 = random_mat(16, 1)
b2 = [[-0.9]]

ys = [[1], [0], [0.5]]         # True output (1 sample, 1 target)



boards, winners = [], []
for i in range(1, 41):
   while True:
      board = game.generate_random_board()
      winner = game.check_winner(board)
      if winner is not None: # and winner != 0:
         winner = [(winner + 1.0) / 2.0]
         break
   boards.append(board)
   winners.append(winner)


sum_loss = 0
j = 0
for i in range(100000):
  
  for board, winner in zip(boards, winners):
    loss, _, w1, b1, w2, b2, prediction = gradient_backpropagation(board, [winner], w1, b1, w2, b2)
    sum_loss = sum_loss + loss[0][0]
    j = j + 1
    if j % 3001 == 0:

      print(f"Loss {i}: {loss} {sum_loss/500}")
      sum_loss = 0
 
      game.print_board(board)
      print("Winner:", winner[0], prediction[0][0], "error: ", (winner[0] - prediction[0][0])**2)



