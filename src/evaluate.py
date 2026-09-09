
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score
)



IMAGE_SIZE = (128, 128)

CLASSES = [
    "Neutrophil",
    "Lymphocyte",
    "Monocyte",
    "Eosinophil",
    "Basophil"
]

MODEL_PATH = "models/best_wbc_classifier.keras"
TEST_DIR = "data/Test-A"
CONFUSION_DIR = "results/confusion_matrix"

BATCH_SIZE = 32




def load_test_dataset():
    """
    Load Test-A using the same image size and class ordering
    used during training.
    """

    test_dataset = tf.keras.utils.image_dataset_from_directory(
        TEST_DIR,
        labels="inferred",
        label_mode="int",
        class_names=CLASSES,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    # Normalize pixel values to [0, 1],
    # matching the preprocessing used during training.
    test_dataset = test_dataset.map(
        lambda images, labels: (
            tf.cast(images, tf.float32) / 255.0,
            labels
        ),
        num_parallel_calls=tf.data.AUTOTUNE
    )

    test_dataset = test_dataset.prefetch(tf.data.AUTOTUNE)

    return test_dataset



def evaluate_model():

    print("=" * 60)
    print("WBC CLASSIFIER — MODEL EVALUATION")
    print("=" * 60)



    print("\nLoading best model...")

    model = tf.keras.models.load_model(MODEL_PATH)

    print("Model loaded successfully.")
    print(f"Model path: {MODEL_PATH}")



    print("\nLoading Test-A dataset...")

    test_dataset = load_test_dataset()

    print("Test dataset loaded.")
    print(f"Test directory: {TEST_DIR}")



    print("\nGenerating predictions...")

    probabilities = model.predict(test_dataset, verbose=1)

    predicted_labels = np.argmax(probabilities, axis=1)

    # Extract true labels in the same order as predictions
    true_labels = np.concatenate(
        [labels.numpy() for _, labels in test_dataset],
        axis=0
    )

    print(f"\nNumber of test images: {len(true_labels)}")



    accuracy = accuracy_score(
        true_labels,
        predicted_labels
    )

    print("\n" + "=" * 60)
    print("TEST ACCURACY")
    print("=" * 60)

    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Test Accuracy: {accuracy * 100:.2f}%")



    print("\n" + "=" * 60)
    print("CLASSIFICATION REPORT")
    print("=" * 60)

    report = classification_report(
        true_labels,
        predicted_labels,
        target_names=CLASSES,
        digits=4
    )

    print(report)



    cm = confusion_matrix(
        true_labels,
        predicted_labels
    )

    print("=" * 60)
    print("CONFUSION MATRIX")
    print("=" * 60)

    print(cm)


    os.makedirs(CONFUSION_DIR, exist_ok=True)

    cm_path = os.path.join(
        CONFUSION_DIR,
        "confusion_matrix.png"
    )

    plt.figure(figsize=(9, 7))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=CLASSES,
        yticklabels=CLASSES
    )

    plt.title("WBC Classification — Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()

    plt.savefig(
        cm_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

    print("\nConfusion matrix saved to:")
    print(cm_path)



    report_dict = classification_report(
        true_labels,
        predicted_labels,
        target_names=CLASSES,
        output_dict=True
    )

    print("\n" + "=" * 60)
    print("PER-CLASS METRICS")
    print("=" * 60)

    for class_name in CLASSES:

        precision = report_dict[class_name]["precision"]
        recall = report_dict[class_name]["recall"]
        f1 = report_dict[class_name]["f1-score"]

        print(
            f"{class_name:12s} | "
            f"Precision: {precision:.4f} | "
            f"Recall: {recall:.4f} | "
            f"F1: {f1:.4f}"
        )

    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    evaluate_model()
