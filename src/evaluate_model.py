import numpy as np
from keras import models
from pathlib import Path

MODEL = Path('../data/training.model')
TEST_DIR = Path('../data/images/test/')

def load_test_data():
        X_test = np.load(TEST_DIR.joinpath("test.npy"))
        y_test = np.load(TEST_DIR.joinpath("test_labels.npy"))
        return X_test, y_test

if  __name__ == "__main__":
    model =  models.load_test_model(MODEL)  
    (X_test, y_test) = load_test_data()
    
    predict = model.predict(X_test)
    evalulate = model.evaluate(X_test, y_test)