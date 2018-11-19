#
# model.py
# Model Structure
# Dabnet
#

from keras import layers, models, optimizers

# Build and returns the Dabnet model
def build():
    model = models.Sequential()
    model.add(layers.Conv2D(16, (7, 7), activation="relu", input_shape=(128, 128, 1)))
    model.add(layers.MaxPool2D((3, 3)))
    model.add(layers.BatchNormalization())

    model.add(layers.Conv2D(32, (7, 7), activation="relu"))
    model.add(layers.MaxPool2D((3, 3)))
    model.add(layers.BatchNormalization())

    model.add(layers.Flatten())
    model.add(layers.Dense(32, activation="relu"))
    model.add(layers.Dropout(0.4))
    model.add(layers.Dense(1, activation="sigmoid"))

    model.compile(loss="binary_crossentropy",
                  optimizer="adam",
                  metrics=["accuracy"])
    
    return model

if __name__ == "__main__":
    model = build()
    model.summary()
