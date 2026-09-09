
import os
import argparse
import numpy as np
import tensorflow as tf




IMAGE_SIZE = (128, 128)

CLASSES = [
    "Neutrophil",
    "Lymphocyte",
    "Monocyte",
    "Eosinophil",
    "Basophil"
]

MODEL_PATH = "models/best_wbc_classifier.keras"

SUPPORTED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff"
}




def load_model():

    if not os.path.isfile(MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}\n"
            "Make sure the trained model exists in models/."
        )

    print("Loading model...")

    model = tf.keras.models.load_model(
        MODEL_PATH
    )

    print("Model loaded successfully.")

    return model



def preprocess_image(image_path):

    # Check that the file exists
    if not os.path.isfile(image_path):
        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    # Check extension
    extension = os.path.splitext(
        image_path
    )[1].lower()

    if extension not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported image format: {extension}\n"
            f"Supported formats: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )

    try:

        # Load image as RGB
        image = tf.keras.utils.load_img(
            image_path,
            target_size=IMAGE_SIZE,
            color_mode="rgb"
        )

        # Convert to NumPy array
        image = tf.keras.utils.img_to_array(
            image
        )

    except Exception as e:

        raise ValueError(
            f"Unable to read image: {image_path}\n"
            f"Reason: {e}"
        )

    # Convert to float32
    image = image.astype(
        np.float32
    )

    # Normalize exactly as during training:
    # pixel values [0, 255] → [0, 1]
    image = image / 255.0

    # Add batch dimension
    # (128, 128, 3) → (1, 128, 128, 3)
    image = np.expand_dims(
        image,
        axis=0
    )

    return image




def predict_image(model, image):

    probabilities = model.predict(
        image,
        verbose=0
    )[0]

    predicted_index = np.argmax(
        probabilities
    )

    predicted_class = CLASSES[
        predicted_index
    ]

    confidence = probabilities[
        predicted_index
    ]

    return (
        predicted_class,
        confidence,
        probabilities
    )



def display_result(
    predicted_class,
    confidence,
    probabilities
):

    print("\n" + "=" * 50)
    print("WBC CLASSIFICATION RESULT")
    print("=" * 50)

    print(
        f"\nPredicted class: {predicted_class}"
    )

    print(
        f"Confidence: {confidence * 100:.2f}%"
    )

    print("\nProbabilities:")

    for class_name, probability in zip(
        CLASSES,
        probabilities
    ):

        print(
            f"{class_name}: "
            f"{probability * 100:.2f}%"
        )

    print("=" * 50)


def main():

    parser = argparse.ArgumentParser(
        description="Predict WBC type from a single image."
    )

    parser.add_argument(
        "--image",
        required=True,
        help="Path to the WBC image"
    )

    args = parser.parse_args()

    try:

        # Load trained model
        model = load_model()

        # Preprocess image
        image = preprocess_image(
            args.image
        )

        # Predict
        (
            predicted_class,
            confidence,
            probabilities
        ) = predict_image(
            model,
            image
        )

        # Display result
        display_result(
            predicted_class,
            confidence,
            probabilities
        )

    except FileNotFoundError as e:

        print(f"\nERROR: {e}")
        return 1

    except ValueError as e:

        print(f"\nERROR: {e}")
        return 1

    except Exception as e:

        print(
            f"\nUnexpected error: {e}"
        )

        return 1

    return 0


if __name__ == "__main__":
    main()
