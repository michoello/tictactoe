from helloworld import ml


def gradient_backpropagation(x, y, w1, b1, w2, b2):

    xx = ml.BB(x)
    ww1 = ml.BB(w1)
    bb1 = ml.BB(b1)
    ww2 = ml.BB(w2)
    bb2 = ml.BB(b2)
    yy = ml.BB(y)

    # Forward pass
    zz1 = (xx @ ww1  + bb1).sigmoid()
    zz2 = (zz1 @ ww2 + bb2).sigmoid()

    lloss = ml.BBMSELoss(zz2, yy)

    # Backward pass
    lloss.dif(1)

    # Update weights and biases
    ww1.appl() 
    bb1.appl() 
    ww2.appl()
    bb2.appl()

    xx.appl()

    return lloss.val(), xx.val(), ww1.val(), bb1.val(), ww2.val(), bb2.val()

# Run the example
x = [[0.5, 0.9]]                          # Input (1 sample, 2 features)
w1 = [[0.1, 0.2, 0.3], [ 0.4, 0.5, 0.6]]  # Weights for layer 1 (2x2)
b1 = [[0.1, 0.2, 0.3]]                    # Biases for layer 1 (1x2)
w2 = [[0.5], [0.6], [0.7]]                # Weights for layer 2 (2x1)
b2 = [[0.3]]                              # Biases for layer 2 (1x1)

y = [[0.001]]         # True output (1 sample, 1 target)

for i in range(100000):
   loss, _, w1, b1, w2, b2 = gradient_backpropagation(x, y, w1, b1, w2, b2)
   #loss, x, _, _, _, _ = gradient_backpropagation(x, y, w1, b1, w2, b2)
   if i % 5000 == 0:
       print(f"Loss {i}: {loss}")



