import numpy as np
from keras import models
from pathlib import Path
import matplotlib.pyplot as plt
import pickle

MODEL = Path('../models/training.model/')
WEIGHTS = Path('../models/checkpoint/')
TEST_DIR = Path('../data/images/test/')

def load_test_data():
    X_test = np.load(TEST_DIR.joinpath("test.npy"))
    y_test = np.load(TEST_DIR.joinpath("test_labels.npy"))
    return X_test, y_test

def load_model():
    model = models.load_model(MODEL)
    model.load_weights(WEIGHTS.joinpath('models'))
    return model

def plot_performance(history):
    fig = plt.figure(figsize=(6,4))
    plt.plot(history['recall'], lw=3, label='Training')
    plt.plot(history['val_recall'], '--', lw=2, label='Validation')

    y1 = np.asarray(history['val_recall'])
    plt.vlines(y1.argmax(), 0, 1.1, color='k', ls=':')

    plt.ylabel('Recall')
    plt.xlabel('Epoch')
    plt.legend(loc=5)
    plt.ylim(0.4, 1.1)
    fig.show()
    plt.savefig('../performance.png', bbox_inches='tight')

if  __name__ == "__main__":
    with open('../models/training.model.history', 'rb') as f:
        history = pickle.load(f)

    plot_performance(history)
    model =  load_model()  
    (X_test, y_test) = load_test_data()
    predict = model.predict(X_test)
    print('Test image probabilities:')
    print(predict)
    print()
    print('Test image classifications:')
    print(predict.argmax(axis=1))

    evaluate = model.evaluate(X_test, y_test)
    print('Loss {}'.format(evaluate[0]))
    print('Recall {}'.format(evaluate[1]))
    print('Precision {}'.format(evaluate[2]))
    print('AUC {}'.format(evaluate[3]))