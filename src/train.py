
import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from PIL import Image

from sklearn.model_selection import train_test_split

import tensorflow as tf

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    ReduceLROnPlateau
)

from model import build_cnn




PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"

TRAIN_DIR = DATA_DIR / "Train"

MODEL_DIR = PROJECT_ROOT / "models"

PLOT_DIR = PROJECT_ROOT / "results" / "plots"


IMAGE_SIZE = (128, 128)

CLASSES = [
    "Neutrophil",
    "Lymphocyte",
    "Monocyte",
    "Eosinophil",
    "Basophil"
]

NUM_CLASSES = len(CLASSES)

CLASS_TO_INDEX = {
    class_name: index
    for index, class_name in enumerate(CLASSES)
}

BATCH_SIZE = 32

EPOCHS = 10

RANDOM_STATE = 42


VALID_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff"
}




MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)

PLOT_DIR.mkdir(
    parents=True,
    exist_ok=True
)



def get_image_files(directory):
    """
    Find all supported image files recursively.
    """

    directory = Path(directory)

    return sorted([
        path
        for path in directory.rglob("*")
        if (
            path.is_file()
            and path.suffix.lower() in VALID_EXTENSIONS
        )
    ])


def preprocess_image(image_path):
    """
    Preprocess one image.

    Steps:
    1. Convert to RGB
    2. Resize to 128 x 128
    3. Convert to float32
    4. Normalize pixels to [0, 1]
    """

    with Image.open(image_path) as image:

        image = image.convert("RGB")

        image = image.resize(IMAGE_SIZE)

        image = np.asarray(
            image,
            dtype=np.float32
        )

    image = image / 255.0

    return image




def load_dataset(dataset_directory):
    """
    Load images and integer labels.

    Expected structure:

    Train/
        Neutrophil/
        Lymphocyte/
        Monocyte/
        Eosinophil/
        Basophil/
    """

    images = []
    labels = []

    dataset_directory = Path(dataset_directory)

    print("=" * 60)
    print("LOADING TRAINING DATA")
    print("=" * 60)

    for class_name in CLASSES:

        class_directory = (
            dataset_directory / class_name
        )

        if not class_directory.exists():

            raise FileNotFoundError(
                f"Missing class directory: "
                f"{class_directory}"
            )

        class_index = CLASS_TO_INDEX[class_name]

        image_files = get_image_files(
            class_directory
        )

        print(
            f"{class_name:12s}: "
            f"{len(image_files)} images"
        )

        for image_path in image_files:

            try:

                image = preprocess_image(
                    image_path
                )

                images.append(image)

                labels.append(class_index)

            except Exception as error:

                print(
                    f"WARNING: Could not load "
                    f"{image_path}: {error}"
                )

    X = np.asarray(
        images,
        dtype=np.float32
    )

    y = np.asarray(
        labels,
        dtype=np.int64
    )

    return X, y




def save_training_plots(history):
    """
    Save accuracy and loss curves.
    """

    history_dict = history.history



    plt.figure(figsize=(8, 6))

    plt.plot(
        history_dict["accuracy"],
        label="Training Accuracy"
    )

    plt.plot(
        history_dict["val_accuracy"],
        label="Validation Accuracy"
    )

    plt.title("Training vs Validation Accuracy")

    plt.xlabel("Epoch")

    plt.ylabel("Accuracy")

    plt.legend()

    plt.grid(True)

    plt.tight_layout()

    accuracy_path = (
        PLOT_DIR /
        "training_validation_accuracy.png"
    )

    plt.savefig(
        accuracy_path,
        dpi=150
    )

    plt.close()

    plt.figure(figsize=(8, 6))

    plt.plot(
        history_dict["loss"],
        label="Training Loss"
    )

    plt.plot(
        history_dict["val_loss"],
        label="Validation Loss"
    )

    plt.title("Training vs Validation Loss")

    plt.xlabel("Epoch")

    plt.ylabel("Loss")

    plt.legend()

    plt.grid(True)

    plt.tight_layout()

    loss_path = (
        PLOT_DIR /
        "training_validation_loss.png"
    )

    plt.savefig(
        loss_path,
        dpi=150
    )

    plt.close()


    print("\nTraining plots saved:")

    print(
        f" - {accuracy_path}"
    )

    print(
        f" - {loss_path}"
    )




def train_model():

 

    X_all, y_all = load_dataset(
        TRAIN_DIR
    )

    print("\nDataset shape:")
    print("X:", X_all.shape)
    print("y:", y_all.shape)




    X_train, X_val, y_train, y_val = (
        train_test_split(
            X_all,
            y_all,
            test_size=0.20,
            random_state=RANDOM_STATE,
            stratify=y_all
        )
    )

    print("\n" + "=" * 60)
    print("TRAIN / VALIDATION SPLIT")
    print("=" * 60)

    print(
        "Training images:",
        X_train.shape[0]
    )

    print(
        "Validation images:",
        X_val.shape[0]
    )



    y_train_cat = to_categorical(
        y_train,
        num_classes=NUM_CLASSES
    )

    y_val_cat = to_categorical(
        y_val,
        num_classes=NUM_CLASSES
    )




    train_datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode="nearest"
    )




    train_generator = train_datagen.flow(
        X_train,
        y_train_cat,
        batch_size=BATCH_SIZE,
        shuffle=True,
        seed=RANDOM_STATE
    )


    validation_datagen = ImageDataGenerator()

    validation_generator = validation_datagen.flow(
        X_val,
        y_val_cat,
        batch_size=BATCH_SIZE,
        shuffle=False
    )


    print("\n" + "=" * 60)
    print("CREATING CNN")
    print("=" * 60)

    model = build_cnn()

    model.summary()



    print("\n" + "=" * 60)
    print("COMPILING MODEL")
    print("=" * 60)

    model.compile(
        optimizer=Adam(
            learning_rate=0.0005
        ),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    print("Loss: categorical_crossentropy")
    print("Optimizer: Adam")
    print("Learning rate: 0.0005")
    print("Metric: accuracy")



    best_model_path = (
        MODEL_DIR /
        "best_wbc_classifier.keras"
    )

    checkpoint = ModelCheckpoint(
        filepath=best_model_path,
        monitor="val_accuracy",
        mode="max",
        save_best_only=True,
        verbose=1
    )


    early_stopping = EarlyStopping(
        monitor="val_loss",
        mode="min",
        patience=3,
        restore_best_weights=True,
        verbose=1
    )


    reduce_lr = ReduceLROnPlateau(
        monitor="val_loss",
        mode="min",
        factor=0.5,
        patience=2,
        min_lr=1e-6,
        verbose=1
    )


    callbacks = [
        checkpoint,
        early_stopping,
        reduce_lr
    ]



    print("\n" + "=" * 60)
    print("STARTING TRAINING")
    print("=" * 60)

    history = model.fit(
        train_generator,
        validation_data=validation_generator,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1
    )


    save_training_plots(
        history
    )

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)

    print(
        f"Best model saved at:\n"
        f"{best_model_path}"
    )

    best_epoch = (
        np.argmax(
            history.history["val_accuracy"]
        ) + 1
    )

    best_val_accuracy = max(
        history.history["val_accuracy"]
    )

    print(
        f"\nBest validation accuracy: "
        f"{best_val_accuracy:.4f}"
    )

    print(
        f"Best epoch: {best_epoch}"
    )

    return model, history



if __name__ == "__main__":

    train_model()
