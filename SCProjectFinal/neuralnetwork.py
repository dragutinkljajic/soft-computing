from keras.models import Sequential
from keras.layers.core import Activation, Dense
from keras.optimizers import SGD
import pickle

import numpy as np
import random


def initializeNetwork(compress):
    print 'Initializing...'
    model = Sequential()
    model.add(Dense(400, input_dim=400, init='normal', activation='relu'))
    model.add(Dense(10, init='normal', activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print 'COMPILED'
    model.fit(compress.data, compress.target, nb_epoch=70, batch_size=200, verbose=2)
    print 'FITTED'

    save(model, 'network.obj')

    return model


def getMostPossible(lst):
    return np.argmax(lst)


def predict(model, inputs):
    predicts = model.predict(inputs, verbose=2)
    numbers = []
    for p in predicts:
        numbers.append(getMostPossible(p))

    return numbers


def test(model, compress):
    size = 70000
    subset = np.random.choice(70000, size)

    data = compress.data[subset]
    correct = compress.target[subset]
    predict = model.predict(data, verbose=2)

    cnt = 0
    for idx, c in enumerate(correct):
        p = predict[idx]

        cValue = getMostPossible(c)
        pValue = getMostPossible(p)

        if cValue == pValue:
            cnt += 1

    print 'FINISHED'
    print 'Tested: ', size, ', Correct: ', cnt


def save(obj, path):
    print 'Saving...'
    with open(path, 'wb') as f:
        pickle.dump(obj, f)
    print 'Saved.'


def load(path):
    print 'Loading...'
    with open(path, 'rb') as f:
        retVal = pickle.load(f)
    print 'Loaded.'
    return retVal