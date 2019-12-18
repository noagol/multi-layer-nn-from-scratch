import numpy as np

STUDENT = {'name': 'Noa Gol_Elad Bitton',
           'ID': '208469486_205846173'}

def one_hot_encode(i, n):
    x = [0] * n
    x[i] = 1

    return np.array(x)

def softmax_deriv(y, y_hat):
    return y_hat - one_hot_encode(y, y_hat.shape[0])

def tanh_deriv(x):
    return 1 - np.power(np.tanh(x), 2)

def softmax(x):
    """
    Compute the softmax vector.
    x: a n-dim vector (numpy array)
    returns: an n-dim vector (numpy array) of softmax values
    """
    # YOUR CODE HERE
    # Your code should be fast, so use a vectorized implementation using numpy,
    # don't use any loops.
    # With a vectorized implementation, the code should be no more than 2 lines.
    #
    # For numeric stability, use the identify you proved in Ex 2 Q1.
    ex = np.exp(x - np.max(x))
    x = np.divide(ex, np.sum(ex))

    return x

def classifier_output(x, params):
    # YOUR CODE HERE.

    inp = x
    num_iter = len(params) - 2
    for i in range(0, num_iter, 2):
        W = params[i]
        b = params[i+1]
        
        linear = np.dot(inp, W) + b
        inp = np.tanh(linear)
    
    W = params[num_iter]
    b = params[num_iter + 1]
    probs = softmax(np.dot(inp, W) + b)
    return probs

def forward(x, params):
    # YOUR CODE HERE.
    hiddens = [x]
    inp = x
    num_iter = len(params) - 2
    for i in range(0, num_iter, 2):
        W = params[i]
        b = params[i+1]
        
        linear = np.dot(inp, W) + b
        inp = np.tanh(linear)
        hiddens.append(inp.copy())
    
    W = params[num_iter]
    b = params[num_iter + 1]
    probs = softmax(np.dot(inp, W) + b)

    return probs, hiddens

def predict(x, params):
    return np.argmax(classifier_output(x, params))

def loss_and_gradients(x, y, params):
    """
    params: a list as created by create_classifier(...)

    returns:
        loss,[gW1, gb1, gW2, gb2, ...]

    loss: scalar
    gW1: matrix, gradients of W1
    gb1: vector, gradients of b1
    gW2: matrix, gradients of W2
    gb2: vector, gradients of b2
    ...

    (of course, if we request a linear classifier (ie, params is of length 2),
    you should not have gW2 and gb2.)
    """
    # YOU CODE HERE
    Ws = [params[i] for i in range(0, len(params), 2)]
    Bs = [params[i + 1] for i in range(0, len(params), 2)]

    y_hat, hiddens = forward(x, params)

    loss = -np.log(y_hat[y])

    diff = softmax_deriv(y, y_hat)

    grads = []
    gb = diff
    grads.append(gb)

    U = Ws[-1]
    gU = np.outer(np.tanh(hiddens[-1]), diff)
    grads.append(gU)

    running_grad = np.dot(U, diff)
    for i in range(len(Ws) - 1):
        w_index = len(Ws) - i - 2
        bi = Bs[w_index]
        Wi = Ws[w_index]

        running_grad = running_grad * tanh_deriv(hiddens[w_index+1])

        gbi = running_grad
        gWi = np.outer(hiddens[w_index], running_grad)
        grads.append(gbi)
        grads.append(gWi)

        running_grad = np.dot(Wi, running_grad)

    return loss, list(reversed(grads))

def init_random_weights(size):
    return np.random.uniform(low=-0.08, high=0.08, size=size)

def create_classifier(dims):
    """
    returns the parameters for a multi-layer perceptron with an arbitrary number
    of hidden layers.
    dims is a list of length at least 2, where the first item is the input
    dimension, the last item is the output dimension, and the ones in between
    are the hidden layers.
    For example, for:
        dims = [300, 20, 30, 40, 5]
    We will have input of 300 dimension, a hidden layer of 20 dimension, passed
    to a layer of 30 dimensions, passed to learn of 40 dimensions, and finally
    an output of 5 dimensions.
    
    Assume a tanh activation function between all the layers.

    return:
    a flat list of parameters where the first two elements are the W and b from input
    to first layer, then the second two are the matrix and vector from first to
    second layer, and so on.
    """
    params = []
    for i in range(len(dims) - 1):
        W = init_random_weights((dims[i], dims[i+1]))
        b = init_random_weights(dims[i+1])
        params.append(W)
        params.append(b)

    return params

