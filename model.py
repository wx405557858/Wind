__author__ = 'shawn'

import numpy as np
from sklearn import linear_model
from sklearn import tree
from datalayer import Datalayer
import argparse

from lstm import LSTMLayer
import time

st = time.time()

def train(d):
    train_X, train_Y, test_X, test_Y = d
    model = linear_model.LinearRegression()
    # model = tree.DecisionTreeRegressor()
    model.fit(train_X, train_Y)
    pred_y = model.predict(test_X)
    mse = np.sqrt(np.mean((pred_y - test_Y)**2))
    print mse
    return mse


def main(duration):
    data = Datalayer(duration=duration)
    losses = []
    print "preparation time :", time.time() - st
    for d in data.k_fold(5):
        l = train(d)
        losses.append(l)
    print "\ntraining time :", time.time() - st
    print "average loss :",np.mean(losses)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--duration', help='6/24/48/72h')
    args = parser.parse_args()
    if args.duration:
        duration = eval(args.duration)
    else:
        duration = 24
    print args
    main(duration=duration)
