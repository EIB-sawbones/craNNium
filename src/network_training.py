import pickle
from pathlib import Path
from keras import optimizers, models, losses, metrics, layers
from keras.callbacks import ModelCheckpoint

MODEL_DIR = Path("../models/")


class NeuralNetwork:
    def __init__(self, X_train, y_train, X_val, y_val, X_test, y_test, **kwargs):

        self.model_dir = MODEL_DIR

        (self.X_train, self.y_train) = (X_train, y_train)
        (self.X_val, self.y_val) = (X_val, y_val)
        (self.X_test, self.y_test) = (X_test, y_test)

        self.metrics = [metrics.Recall(), metrics.Precision(), metrics.AUC()]
        self.model = self.set_architecture()
        self.callback = self.compile_model()

    def set_architecture(self):
        model = models.Sequential()

        model.add(
            layers.Conv2D(32, (3, 3), activation="relu", input_shape=(150, 150, 3))
        )
        model.add(layers.MaxPooling2D((2, 2), strides=2))
        model.add(layers.Conv2D(64, (3, 3), activation="relu"))
        model.add(layers.MaxPooling2D((2, 2), strides=2))
        model.add(layers.Conv2D(64, (3, 3), activation="relu"))

        model.add(layers.Flatten())
        model.add(layers.Dense(64, activation="relu"))
        model.add(layers.Dense(2, activation="softmax"))

        print("craNNium CNN architecture:")
        print(model.summary())
        return model

    def compile_model(self):
        self.model.compile(
            optimizer=optimizers.Adam(learning_rate=1e-3),
            loss=losses.BinaryCrossentropy(from_logits=True),
            metrics=self.metrics,
        )

        model_checkpoint_callback = ModelCheckpoint(
            filepath=self.model_dir,
            save_weights_only=True,
            monitor="val_recall",
            mode="max",
            save_best_only=True,
        )
        return model_checkpoint_callback

    def fit(self, filename="training.model", epochs=100, batch_size=32, **kwargs):

        history = self.model.fit(
            self.X_train,
            self.y_train,
            epochs=epochs,
            validation_data=(self.X_val, self.y_val),
            callbacks=[self.callback],
            **kwargs
        )
        self.history = history
        self.model.save(self.model_dir.joinpath(filename))
        with open(self.model_dir.joinpath(filename + ".history"), "wb") as f:
            pickle.dump(self.history.history, f)


if __name__ == "__main__":
    cnn = NeuralNetwork()