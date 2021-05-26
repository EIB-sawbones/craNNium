import pickle
from pathlib import Path
from keras import models
from keras import losses
from keras import metrics
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator

import inception_v4

MODEL_DIR = Path("../models/")


class NeuralNetwork:
    def __init__(self, X_train, y_train, X_val, y_val, X_test, y_test):

        self.model_dir = MODEL_DIR

        (self.X_train, self.y_train) = (X_train, y_train)
        (self.X_val, self.y_val) = (X_val, y_val)
        (self.X_test, self.y_test) = (X_test, y_test)

        self.metrics = [metrics.Recall()]
        self.model = self.set_architecture()
        self.callback = self.compile_model()

        self.X_train = inception_v4.process_all_images(self.X_train)
        self.X_val = inception_v4.process_all_images(self.X_val)
        self.X_test = inception_v4.process_all_images(self.X_test)

        self.fit()

    def set_architecture(self):
        """input_shape = self.X_train.shape[1:]
        self.model.add(
            layers.Conv2D(16, (3, 3), activation="relu", input_shape=input_shape)
        )
        self.model.add(layers.MaxPooling2D((2, 2), strides=2))
        self.model.add(layers.Conv2D(32, (3, 3), activation="relu"))
        self.model.add(layers.MaxPooling2D((2, 2), strides=2))
        self.model.add(layers.Conv2D(32, (3, 3), activation="relu"))

        self.model.add(layers.Flatten())

        self.model.add(layers.Dense(32, activation="relu"))
        self.model.add(layers.Dense(2, activation="softmax"))"""

        model = inception_v4.create_model(
            num_classes=2, weights="imagenet", include_top=True
        )

        print("craNNium CNN architecture:")
        print(model.summary())
        return model

    def augment_data(self, batch_size=32):
        gen_train = ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode="nearest",
        ).flow(self.X_train, self.y_train, batch_size=batch_size)

        gen_val = ImageDataGenerator().flow(
            self.X_val, self.y_val, batch_size=batch_size
        )
        return gen_train, gen_val

    def compile_model(self):
        self.model.compile(
            optimizer="rmsprop",
            loss=losses.BinaryCrossentropy(from_logits=True),
            metrics=self.metrics,
        )

        model_checkpoint_callback = ModelCheckpoint(
            filepath=self.model_dir,
            save_weights_only=True,
            monitor="val_{}".format(self.metrics[0].name),
            mode="max",
            save_best_only=True,
        )
        return model_checkpoint_callback

    def fit(self, filename="training.model", epochs=100, batch_size=32, **kwargs):

        gen_train, gen_val = self.augment_data(batch_size=batch_size)

        history = self.model.fit(
            gen_train,
            epochs=epochs,
            steps_per_epoch=len(self.X_train) // batch_size,
            validation_data=gen_val,
            callbacks=[self.callback],
            **kwargs
        )
        self.history = history
        self.model.save(self.model_dir.joinpath(filename))
        with open(self.model_dir.joinpath(filename+".history"), "wb") as f:
            pickle.dump(self.history.history, f)


if __name__ == "__main__":
    cnn = NeuralNetwork()