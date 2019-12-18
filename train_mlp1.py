import numpy as np
import mlp1 as ll
import utils
import random
import xor_data

STUDENT = {'name': 'Noa Gol_Elad Bitton',
           'ID': '208469486_205846173'}


def feats_to_vec(features):
    # YOUR CODE HERE.
    # Should return a numpy vector of features.
    x = [0] * len(utils.F2I)
    for t in features:
        try:
            t_id = utils.F2I[t]
        except KeyError:
            continue
        x[t_id] += 1
    return np.array(x)


def accuracy_on_dataset(dataset, params):
    good = bad = 0.0
    for label, features in dataset:
        # YOUR CODE HERE
        # Compute the accuracy (a scalar) of the current parameters
        # on the dataset.
        # accuracy is (correct_predictions / all_predictions)

        x = feats_to_vec(features)
        y = utils.L2I[label]

        y_hat = ll.predict(x, params)

        if y == y_hat:
            good += 1
        else:
            bad += 1

    return good / (good + bad)


def train_classifier(train_data, dev_data, num_iterations, learning_rate, params):
    """
    Create and train a classifier, and return the parameters.

    train_data: a list of (label, feature) pairs.
    dev_data  : a list of (label, feature) pairs.
    num_iterations: the maximal number of training iterations.
    learning_rate: the learning rate to use.
    params: list of parameters (initial values)
    """
    for I in xrange(num_iterations):
        cum_loss = 0.0  # total loss in this iteration.
        random.shuffle(train_data)
        for label, features in train_data:
            x = feats_to_vec(features)  # convert features to a vector.
            y = utils.L2I[label]  # convert the label to number if needed.
            loss, grads = ll.loss_and_gradients(x, y, params)
            cum_loss += loss
            # YOUR CODE HERE
            # update the parameters according to the gradients
            # and the learning rate.

            gW, gb, gU, gb_tag = grads
            W, b, U, b_tag = params

            W -= learning_rate * gW
            b -= learning_rate * gb
            U -= learning_rate * gU
            b_tag -= learning_rate * gb_tag

        train_loss = cum_loss / len(train_data)
        train_accuracy = accuracy_on_dataset(train_data, params)
        dev_accuracy = accuracy_on_dataset(dev_data, params)
        print I, train_loss, train_accuracy, dev_accuracy
    return params


def train_xor(train_data, num_iterations, learning_rate, params, decay=1e-3):
    """
    Create and train a classifier, and return the parameters.

    train_data: a list of (label, feature) pairs.
    dev_data  : a list of (label, feature) pairs.
    num_iterations: the maximal number of training iterations.
    learning_rate: the learning rate to use.
    params: list of parameters (initial values)
    """
    for I in xrange(num_iterations):
        cum_loss = 0.0  # total loss in this iteration.
        random.shuffle(train_data)
        for y, x in train_data:
            loss, grads = ll.loss_and_gradients(x, y, params)
            cum_loss += loss
            # YOUR CODE HERE
            # update the parameters according to the gradients
            # and the learning rate.

            gW, gb, gU, gb_tag = grads
            W, b, U, b_tag = params

            W -= (learning_rate * gW)
            b -= (learning_rate * gb)
            U -= (learning_rate * gU)
            b_tag -= (learning_rate * gb_tag)

        learning_rate = max(1e-3, learning_rate - decay)

        train_loss = cum_loss / len(train_data)
        print I, train_loss
    return params


def pred(pred_data, params):
    """ Test classifier
    """

    I2L = {utils.L2I[l]: l for l in utils.L2I}

    with open("test.pred", "w+") as file:
        for features in pred_data:
            x = feats_to_vec(features)  # convert features to a vector.
            y_hat = ll.predict(x, params)
            file.write(I2L[y_hat])
            file.write("\n")


def xor():
    in_dim = 2
    hidden_dim = 4
    out_dim = 2
    num_iterations = 70
    learning_rate = 1.0

    params = ll.create_classifier(in_dim, hidden_dim, out_dim)

    trained_params = train_xor(xor_data.data, num_iterations, learning_rate, params)

    return trained_params


if __name__ == '__main__':
    # YOUR CODE HERE
    # write code to load the train and dev sets, set up whatever you need,
    # and call train_classifier.

    # train_data = utils.TRAIN
    # dev_data = utils.DEV
    #
    # in_dim = len(utils.F2I)
    # hidden_dim = 100
    # out_dim = len(utils.L2I)
    # num_iterations = 20
    # learning_rate = 1e-3
    #
    # TEST = [utils.text_to_bigrams(t) for l, t in utils.read_data("test")]
    #
    # params = ll.create_classifier(in_dim, hidden_dim, out_dim)
    # trained_params = train_classifier(train_data, dev_data, num_iterations, learning_rate, params)
    #
    # pred(TEST, trained_params)

    xor()
