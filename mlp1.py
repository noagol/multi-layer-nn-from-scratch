import numpy as np
import loglinear

STUDENT = {'name': 'Noa Gol_Elad Bitton',
           'ID': '208469486_205846173'}


def classifier_output(x, params):
    # YOUR CODE HERE.

    W, b, U, b_tag = params

    linear = np.dot(x, W) + b
    tan = np.tanh(linear)
    probs = loglinear.softmax(np.dot(tan, U) + b_tag)
    return probs


def predict(x, params):
    """
    params: a list of the form [W, b, U, b_tag]
    """
    return np.argmax(classifier_output(x, params))


def softmax_deriv(y, y_hat):
    return y_hat - loglinear.one_hot_encode(y, y_hat.shape[0])


def tanh_deriv(x):
    return 1 - np.power(np.tanh(x), 2)


def loss_and_gradients(x, y, params):
    """
    params: a list of the form [W, b, U, b_tag]

    returns:
        loss,[gW, gb, gU, gb_tag]

    loss: scalar
    gW: matrix, gradients of W
    gb: vector, gradients of b
    gU: matrix, gradients of U
    gb_tag: vector, gradients of b_tag
    """
    # YOU CODE HERE

    W, b, U, b_tag = params

    y_pred = classifier_output(x, params)
    loss = -np.log(y_pred[y])

    diff = softmax_deriv(y, y_pred)

    linear = np.dot(x, W) + b

    gW1 = np.dot(U, diff)
    gW2 = tanh_deriv(linear) * gW1
    gW = np.outer(x, gW2)

    gU = np.outer(np.tanh(linear), diff)

    gb_tag = diff

    gb = gW2

    return loss, [gW, gb, gU, gb_tag]


def init_random_weights(size):
    return np.random.uniform(low=-0.08, high=0.08, size=size)

def create_classifier(in_dim, hid_dim, out_dim):
    """
    returns the parameters for a multi-layer perceptron,
    with input dimension in_dim, hidden dimension hid_dim,
    and output dimension out_dim.

    return:
    a flat list of 4 elements, W, b, U, b_tag.
    """

    W = init_random_weights((in_dim, hid_dim))
    b = init_random_weights(hid_dim)
    U = init_random_weights((hid_dim, out_dim))
    b_tag = init_random_weights(out_dim)

    params = [W, b, U, b_tag]
    return params

if __name__ == '__main__':
    # Sanity checks for softmax. If these fail, your softmax is definitely wrong.
    # If these pass, it may or may not be correct.
    test1 = loglinear.softmax(np.array([1, 2]))
    print test1
    assert np.amax(np.fabs(test1 - np.array([0.26894142, 0.73105858]))) <= 1e-6

    test2 = loglinear.softmax(np.array([1001, 1002]))
    print test2
    assert np.amax(np.fabs(test2 - np.array([0.26894142, 0.73105858]))) <= 1e-6

    test3 = loglinear.softmax(np.array([-1001, -1002]))
    print test3
    assert np.amax(np.fabs(test3 - np.array([0.73105858, 0.26894142]))) <= 1e-6

    # Sanity checks. If these fail, your gradient calculation is definitely wrong.
    # If they pass, it is likely, but not certainly, correct.
    from grad_check import gradient_check

    params = create_classifier(3, 10, 4)

    W, b, U, b_tag = params


    def _loss_and_W_grad(W):
        loss, grads = loss_and_gradients([1, 2, 3], 0, params)
        return loss, grads[0]


    def _loss_and_b_grad(b):
        loss, grads = loss_and_gradients([1, 2, 3], 0, params)
        return loss, grads[1]


    def _loss_and_U_grad(U):
        loss, grads = loss_and_gradients([1, 2, 3], 0, params)
        return loss, grads[2]


    def _loss_and_b_tag_grad(b_tag):
        loss, grads = loss_and_gradients([1, 2, 3], 0, params)
        return loss, grads[3]

    for _ in xrange(10):
        gradient_check(_loss_and_b_grad, b)
        gradient_check(_loss_and_W_grad, W)
        gradient_check(_loss_and_U_grad, U)
        gradient_check(_loss_and_b_tag_grad, b_tag)
