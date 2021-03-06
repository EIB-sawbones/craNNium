import numpy as np
import image_processing
import network_training


class craNNium:
    def __init__(self):
        self.train_dir = image_processing.IMAGE_DIR.joinpath("train")
        self.val_dir = image_processing.IMAGE_DIR.joinpath("val")
        self.test_dir = image_processing.IMAGE_DIR.joinpath("test")

        if self.images_already_processed():
            (
                self.X_train,
                self.y_train,
                self.X_val,
                self.y_val,
                self.X_test,
                self.y_test,
            ) = self.load_data()
        else:
            data = image_processing.ImageProcessing()
            self.X_train = data.X_train
            self.y_train = data.y_train
            self.X_val = data.X_val
            self.y_val = data.y_val
            self.X_test = data.X_test
            self.y_test = data.y_test

    def images_already_processed(self):
        return self.train_dir.joinpath("train.npy").is_file()

    def load_data(self):
        X_train = np.load(self.train_dir.joinpath("train.npy"))
        y_train = np.load(self.train_dir.joinpath("train_labels.npy"))

        X_val = np.load(self.val_dir.joinpath("val.npy"))
        y_val = np.load(self.val_dir.joinpath("val_labels.npy"))

        X_test = np.load(self.test_dir.joinpath("test.npy"))
        y_test = np.load(self.test_dir.joinpath("test_labels.npy"))

        return X_train, y_train, X_val, y_val, X_test, y_test


if __name__ == "__main__":
    crannium = craNNium()
    cnn = network_training.NeuralNetwork(
        crannium.X_train,
        crannium.y_train,
        crannium.X_val,
        crannium.y_val,
        crannium.X_test,
        crannium.y_test,
    )
    cnn.fit(batch_size=5, epochs=100)
