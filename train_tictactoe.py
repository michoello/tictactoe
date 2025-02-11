from lib import game
from lib import ttt_classifier as ttt
import random
import sys


def gradient_backpropagation(x, y, epoch):

    # Fit in the inputs and labels
    ttt.xx.set(x)
    ttt.yy.set(y) 

    # Backward pass
    ttt.lloss.dif()

    # Update weights and biases
    alpha = 0.1 if epoch < 5 else 0.001
    #alpha = 1

    ttt.ww1.appl(alpha) 
    ttt.bb1.appl(alpha) 
    ttt.ww2.appl(alpha)
    ttt.bb2.appl(alpha)

# 120 still works, but larger - not well
test_boards, test_winners = game.generate_batch(30) 

best_test_loss = 10 ** 1000

for epoch in range(100):

    train_boards, train_winners = game.generate_batch(60) 

    for i in range(100000):
      
      train_loss = 0
      for board, winner in zip(train_boards, train_winners):
        gradient_backpropagation(board, [winner], epoch)
    
        loss, prediction = ttt.lloss.val(), ttt.zz2.val()
        train_loss = train_loss + loss[0][0]
      print(f"EPOCH {epoch}: Train set loss, epoch {i}: {train_loss/len(train_boards)}")
    
      test_loss = 0
      for board, winner in zip(test_boards, test_winners):
          ttt.xx.set(board)
          ttt.yy.set([winner]) 
    
          loss, prediction = ttt.lloss.val(), ttt.zz2.val()
          test_loss = test_loss + loss[0][0]
      print(f"EPOCH {epoch}: Test set loss, epoch {i}: {test_loss/len(test_boards)}")
    
      if test_loss < best_test_loss:
          model_dump = ttt.lloss.save()
          with open("model_dump.json", "w") as file:
              file.write(model_dump)
          best_test_loss = test_loss
    
      else:
          break
    
