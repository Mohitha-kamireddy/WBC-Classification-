
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    BatchNormalization,
    Dropout,
    Flatten,
    Dense
)



IMAGE_SIZE = 128
NUM_CHANNELS = 3
NUM_CLASSES = 5




def build_cnn(
    image_size=IMAGE_SIZE,
    num_classes=NUM_CLASSES
):
   

    model = Sequential([

        Conv2D(
            64,
            (3, 3),
            activation="relu",
            padding="same",
            input_shape=(
                image_size,
                image_size,
                NUM_CHANNELS
            )
        ),

        Conv2D(
            64,
            (3, 3),
            activation="relu",
            padding="same"
        ),

        BatchNormalization(),

        MaxPooling2D(
            pool_size=(2, 2)
        ),

        Dropout(0.25),



        Conv2D(
            128,
            (3, 3),
            activation="relu",
            padding="same"
        ),

        Conv2D(
            128,
            (3, 3),
            activation="relu",
            padding="same"
        ),

        BatchNormalization(),

        MaxPooling2D(
            pool_size=(2, 2)
        ),

        Dropout(0.25),

        Conv2D(
            256,
            (3, 3),
            activation="relu",
            padding="same"
        ),

        Conv2D(
            256,
            (3, 3),
            activation="relu",
            padding="same"
        ),

        BatchNormalization(),

        MaxPooling2D(
            pool_size=(2, 2)
        ),

        Dropout(0.30),



        Conv2D(
            512,
            (3, 3),
            activation="relu",
            padding="same"
        ),

        Conv2D(
            512,
            (3, 3),
            activation="relu",
            padding="same"
        ),

        BatchNormalization(),

        MaxPooling2D(
            pool_size=(2, 2)
        ),

        Dropout(0.40),

        Flatten(),

        Dense(
            1024,
            activation="relu"
        ),

        BatchNormalization(),

        Dropout(0.50),

        Dense(
            512,
            activation="relu"
        ),

        BatchNormalization(),

        Dropout(0.50),


   
        Dense(
            num_classes,
            activation="softmax"
        )
    ])

    return model
