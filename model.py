from lib import ml


def gradient_backpropagation(x, y, w1, b1, w2, b2):

    xx = ml.BB(x)
    ww1 = ml.BB(w1)
    bb1 = ml.BB(b1)
    ww2 = ml.BB(w2)
    bb2 = ml.BB(b2)
    yy = ml.BB(y)

    # Forward pass
    zz1 = (xx @ ww1 + bb1).sigmoid()
    zz2 = (zz1 @ ww2 + bb2).sigmoid()
    lloss = zz2.mse(yy)

    # Backward pass
    lloss.dif()

    # Update weights and biases
    ww1.appl(0.01)
    bb1.appl(0.01)
    ww2.appl(0.01)
    bb2.appl(0.01)

    return lloss.val(), xx.val(), ww1.val(), bb1.val(), ww2.val(), bb2.val()


# Run the example
xs = [
    [0.5, 0.9],
    [0.1, 0.3],
    [10, 4],
]  # Input (each sample 2 features)
w1 = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]  # Weights for layer 1 (2x2)
b1 = [[0.1, 0.2, 0.3]]  # Biases for layer 1 (1x2)
w2 = [[0.5], [0.6], [0.7]]  # Weights for layer 2 (2x1)
b2 = [[0.3]]  # Biases for layer 2 (1x1)

ys = [[0.001], [0.9], [0.5]]  # True output (each sample 1 target)

for i in range(100000):

    for x, y in zip(xs, ys):
        loss, _, w1, b1, w2, b2 = gradient_backpropagation([x], [y], w1, b1, w2, b2)

    if i % 5000 == 0:
        print(f"Loss {i}: {loss}")
