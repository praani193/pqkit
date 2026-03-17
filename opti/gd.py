def gradient_descent(params, grads, lr=0.1):
    for i in range(len(params)):
        params[i] -= lr * grads[i]