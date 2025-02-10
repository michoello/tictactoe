from lib import ml, game
from lib import ttt_classifier as ttt
import random
import sys


def gradient_backpropagation(x, y):

    # Fit in the inputs and labels
    ttt.xx.set(x)
    ttt.yy.set(y) 

    # Backward pass
    ttt.lloss.dif()

    # Update weights and biases
    ttt.ww1.appl(0.1) 
    ttt.bb1.appl(0.1) 
    ttt.ww2.appl(0.1)
    ttt.bb2.appl(0.1)

# 120 still works, but larger - not well
boards, winners = game.generate_batch(60) 
test_boards, test_winners = game.generate_batch(30) 

best_test_loss = 10 ** 1000

for i in range(100000):
  
  train_loss, j = 0, 0
  for board, winner in zip(boards, winners):
    gradient_backpropagation(board, [winner])

    loss, prediction = ttt.lloss.val(), ttt.zz2.val()
    train_loss, j = train_loss + loss[0][0], j + 1

    if random.random() > 0.999:
      print()
      print("Train board: Winner:", winner[0], " Prediction: ", prediction[0][0], "error: ", (winner[0] - prediction[0][0])**2)
      game.print_board(board)
      print()

  print(f"Train set loss, epoch {i}: {train_loss/j}")

  test_loss, j = 0, 0
  for board, winner in zip(test_boards, test_winners):
      ttt.xx.set(board)
      ttt.yy.set([winner]) 

      loss, prediction = ttt.lloss.val(), ttt.zz2.val()
      test_loss, j = test_loss + loss[0][0], j + 1
      
      if random.random() > 0.999:
          print()
          print("Test board: Winner:", winner[0], " Prediction: ", prediction[0][0], "error: ", (winner[0] - prediction[0][0])**2)
          game.print_board(board)
          print()

  print(f"Test set loss, epoch {i}: {test_loss/j}")

  if test_loss < best_test_loss:

      #xx.set(test_boards[0])
      #check_prediction = zz2.val()
      #print("Saving: check_prediction=", check_prediction)

      model_dump = ttt.lloss.save()
      with open("model_dump.json", "w") as file:
          file.write(model_dump)
      best_test_loss = test_loss

      #lloss.load(model_dump)
      #check_prediction2 = zz2.val()
      #print(test_boards[0])
      #print("Loading: check_prediction=", check_prediction2)


  else:
      sys.exit(0)




