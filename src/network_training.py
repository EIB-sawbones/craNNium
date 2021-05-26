import numpy as np
import pickle
from pathlib import Path
from keras import Sequential
from keras import models
from keras import layers
from keras import losses
from keras import metrics
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator

from image_processing import ImageProcessing
import inception_v4

model_filepath = Path("../models/")


class craNNium:
    def __init__(self):
        self.data = ImageProcessing()

        self.X_train = self.data.X_train
        self.y_train = self.data.y_train

        self.model = models.Sequential()
        self.metrics = [metrics.Recall()]
        self.set_architecture()
        self.callback = self.compile_model()

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

        self.model = inception_v4.create_model(
            num_classes=2, weights="imagenet", include_top=True
        )

        print("craNNium CNN architecture:")
        print(self.model.summary())

    def generate_data(self, validation_split=0.3):
        datagen = ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode="nearest",
            validation_split=validation_split
        )
        return datagen

    def compile_model(self):
        self.model.compile(
            optimizer="rmsprop",
            loss=losses.BinaryCrossentropy(from_logits=True),
            metrics=self.metrics,
        )

        model_checkpoint_callback = ModelCheckpoint(
            filepath=model_filepath,
            save_weights_only=True,
            monitor="val_{}".format(self.metrics[0].name),
            mode="max",
            save_best_only=True,
        )
        return model_checkpoint_callback

    def fit(self, filename="training.model", epochs=100, batch_size=10, validation_split=0.3, **kwargs):
        X_train = inception_v4.process_all_images(self.X_train)
        
        gen = self.generate_data(validation_split=validation_split)
        gen_train = gen.flow(X_train, self.y_train, batch_size=batch_size, subset='training')
        gen_val = gen.flow(X_train, self.y_train, batch_size=batch_size, subset='validation')
        history = self.model.fit(
            gen_train,
            epochs=epochs,
            steps_per_epoch=len(X_train) / batch_size,
            validation_data=gen_val,
            callbacks=[self.callback],
            **kwargs
        )
        self.history = history
        self.model.save(model_filepath / filename)
        with open(model_filepath / filename+'.history', 'wb') as f:
            pickle.dump(self.history.history, f)


if __name__ == "__main__":
    cnn = craNNium()
    cnn.fit()