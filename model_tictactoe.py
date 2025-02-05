from lib import ml, game
import random


def random_mat(m, n):
    return [[random.random() for _ in range(n)] for _ in range(m)]


xx = ml.BB(random_mat(6,6))
ww1 = ml.BB(random_mat(36, 16))
bb1 = ml.BB(random_mat(1, 16))
ww2 = ml.BB(random_mat(16, 1))
bb2 = ml.BB(random_mat(1, 1))
yy = ml.BB(random_mat(1, 1))

# Forward pass
zz0 = ml.BBReshape(xx, 1, 36)
zz1 = (zz0 @ ww1 + bb1).sigmoid()
zz2 = (zz1 @ ww2 + bb2).sigmoid()
lloss = zz2.mse(yy)


def gradient_backpropagation(x, y):

    # Fit in the inputs and labels
    xx.set(x)
    yy.set(y) 

    # Backward pass
    lloss.dif()

    # Update weights and biases
    ww1.appl(0.1) 
    bb1.appl(0.1) 
    ww2.appl(0.1)
    bb2.appl(0.1)


# 120 still works, but larger - not well
boards, winners = game.generate_batch(60) 


sum_loss, j = 0, 0
for i in range(100000):
  
  for board, winner in zip(boards, winners):
    gradient_backpropagation(board, [winner])
    loss, prediction = lloss.val(), zz2.val()

    sum_loss, j = sum_loss + loss[0][0], j + 1

    if random.random() > 0.999:
      print(f"Loss {i}: {sum_loss/j}")
      print("Winner:", winner[0], " Prediction: ", prediction[0][0], "error: ", (winner[0] - prediction[0][0])**2)
      game.print_board(board)
      print()
      
      sum_loss, j = 0, 0

      model_dump = lloss.save()
      with open("model_dump.json", "w") as file:
          file.write(model_dump)


