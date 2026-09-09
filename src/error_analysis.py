
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

from collections import Counter
from sklearn.metrics import confusion_matrix


# ============================================================
# CONFIGURATION
# ============================================================

IMAGE_SIZE = (128, 128)
BATCH_SIZE = 32

CLASSES = [
    "Neutrophil",
    "Lymphocyte",
    "Monocyte",
    "Eosinophil",
    "Basophil"
]

MODEL_PATH = "models/best_wbc_classifier.keras"
TEST_DIR = "data/Test-A"

RESULTS_DIR = "results/error_analysis"

ERROR_CSV = os.path.join(
    RESULTS_DIR,
    "misclassified_images.csv"
)

ERROR_PLOT = os.path.join(
    RESULTS_DIR,
    "misclassified_examples.png"
)

PAIR_PLOT = os.path.join(
    RESULTS_DIR,
    "most_confused_pairs.png"
)


# ============================================================
# LOAD DATASET
# ============================================================

def load_test_dataset():

    dataset = tf.keras.utils.image_dataset_from_directory(
        TEST_DIR,
        labels="inferred",
        label_mode="int",
        class_names=CLASSES,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    dataset = dataset.map(
        lambda images, labels: (
            tf.cast(images, tf.float32) / 255.0,
            labels
        ),
        num_parallel_calls=tf.data.AUTOTUNE
    )

    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset


# ============================================================
# GET IMAGE PATHS
# ============================================================

def get_image_paths():

    image_extensions = {
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".tif",
        ".tiff"
    }

    paths = []

    # IMPORTANT:
    # Follow the same class order as image_dataset_from_directory
    for class_name in CLASSES:

        class_dir = os.path.join(TEST_DIR, class_name)

        class_paths = []

        for root, _, files in os.walk(class_dir):

            for filename in files:

                extension = os.path.splitext(
                    filename
                )[1].lower()

                if extension in image_extensions:

                    class_paths.append(
                        os.path.join(root, filename)
                    )

        # Keras directory loader sorts files
        class_paths.sort()

        paths.extend(class_paths)

    return paths


# ============================================================
# GENERATE PREDICTIONS
# ============================================================

def generate_predictions(model, dataset):

    probabilities = model.predict(
        dataset,
        verbose=1
    )

    predicted_labels = np.argmax(
        probabilities,
        axis=1
    )

    confidences = np.max(
        probabilities,
        axis=1
    )

    true_labels = np.concatenate(
        [
            labels.numpy()
            for _, labels in dataset
        ],
        axis=0
    )

    return (
        true_labels,
        predicted_labels,
        confidences,
        probabilities
    )


# ============================================================
# COLLECT ERRORS
# ============================================================

def collect_errors(
    true_labels,
    predicted_labels,
    confidences,
    image_paths
):

    errors = []

    for index in range(len(true_labels)):

        actual = true_labels[index]
        predicted = predicted_labels[index]

        if actual != predicted:

            errors.append({
                "index": index,
                "image_path": image_paths[index],
                "actual_class": CLASSES[actual],
                "predicted_class": CLASSES[predicted],
                "confidence": float(confidences[index])
            })

    return pd.DataFrame(errors)


# ============================================================
# DISPLAY MISCLASSIFIED EXAMPLES
# ============================================================

def display_misclassified_examples(
    errors_df,
    num_examples=20
):

    if len(errors_df) == 0:

        print("No misclassified images found.")
        return

    # Show highest-confidence mistakes first.
    # These are especially useful because the model
    # was confident but still wrong.
    examples = errors_df.sort_values(
        by="confidence",
        ascending=False
    ).head(num_examples)

    columns = 4
    rows = int(np.ceil(len(examples) / columns))

    plt.figure(
        figsize=(16, rows * 4)
    )

    for plot_index, (_, row) in enumerate(
        examples.iterrows()
    ):

        image = tf.keras.utils.load_img(
            row["image_path"],
            target_size=IMAGE_SIZE
        )

        image = tf.keras.utils.img_to_array(image)

        ax = plt.subplot(
            rows,
            columns,
            plot_index + 1
        )

        ax.imshow(
            image.astype("uint8")
        )

        ax.set_title(
            f"Actual: {row['actual_class']}\\n"
            f"Predicted: {row['predicted_class']}\\n"
            f"Confidence: {row['confidence']:.2%}",
            fontsize=10
        )

        ax.axis("off")

    plt.tight_layout()

    plt.savefig(
        ERROR_PLOT,
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

    print(
        f"Displayed {len(examples)} misclassified examples."
    )

    print(
        f"Saved to: {ERROR_PLOT}"
    )


# ============================================================
# MOST COMMONLY CONFUSED CLASS PAIRS
# ============================================================

def analyze_confused_pairs(errors_df):

    if len(errors_df) == 0:

        print("No errors to analyze.")
        return

    pair_counts = (
        errors_df
        .groupby(
            ["actual_class", "predicted_class"]
        )
        .size()
        .reset_index(
            name="count"
        )
        .sort_values(
            "count",
            ascending=False
        )
    )

    print("\n" + "=" * 70)
    print("MOST COMMONLY CONFUSED CLASS PAIRS")
    print("=" * 70)

    for _, row in pair_counts.iterrows():

        print(
            f"{row['actual_class']:12s} -> "
            f"{row['predicted_class']:12s}: "
            f"{row['count']} images"
        )

    # --------------------------------------------------------
    # Plot top confusion pairs
    # --------------------------------------------------------

    top_pairs = pair_counts.head(10).copy()

    top_pairs["pair"] = (
        top_pairs["actual_class"]
        + " -> "
        + top_pairs["predicted_class"]
    )

    plt.figure(
        figsize=(10, 6)
    )

    sns.barplot(
        data=top_pairs,
        x="count",
        y="pair"
    )

    plt.title(
        "Most Common Misclassification Pairs"
    )

    plt.xlabel(
        "Number of Misclassified Images"
    )

    plt.ylabel(
        "Actual → Predicted"
    )

    plt.tight_layout()

    plt.savefig(
        PAIR_PLOT,
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

    print(
        f"\nConfusion-pair plot saved to: {PAIR_PLOT}"
    )

    return pair_counts


# ============================================================
# ERROR ANALYSIS SUMMARY
# ============================================================

def print_error_summary(
    true_labels,
    predicted_labels,
    errors_df
):

    total = len(true_labels)
    errors = len(errors_df)

    accuracy = (
        (total - errors) / total
    )

    print("\n" + "=" * 70)
    print("ERROR ANALYSIS SUMMARY")
    print("=" * 70)

    print(
        f"Total test images      : {total}"
    )

    print(
        f"Correct predictions    : {total - errors}"
    )

    print(
        f"Incorrect predictions  : {errors}"
    )

    print(
        f"Accuracy               : {accuracy:.4f}"
    )

    print(
        f"Error rate             : {(1 - accuracy):.4f}"
    )

    # --------------------------------------------------------
    # Errors by actual class
    # --------------------------------------------------------

    print("\nErrors by actual class:")

    actual_counts = Counter(
        CLASSES[label]
        for label in true_labels
    )

    error_counts = Counter(
        errors_df["actual_class"]
    )

    for class_name in CLASSES:

        total_class = actual_counts[class_name]
        class_errors = error_counts[class_name]

        error_rate = (
            class_errors / total_class
            if total_class > 0
            else 0
        )

        print(
            f"{class_name:12s}: "
            f"{class_errors:4d} errors / "
            f"{total_class:4d} images "
            f"({error_rate:.2%})"
        )

    # --------------------------------------------------------
    # Errors by predicted class
    # --------------------------------------------------------

    print("\nIncorrect predictions by predicted class:")

    predicted_error_counts = Counter(
        errors_df["predicted_class"]
    )

    for class_name in CLASSES:

        count = predicted_error_counts[class_name]

        print(
            f"{class_name:12s}: {count:4d}"
        )


# ============================================================
# MAIN
# ============================================================

def run_error_analysis():

    print("=" * 70)
    print("WBC CLASSIFIER — ERROR ANALYSIS")
    print("=" * 70)

    os.makedirs(
        RESULTS_DIR,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    print("\nLoading best model...")

    model = tf.keras.models.load_model(
        MODEL_PATH
    )

    print("Model loaded successfully.")

    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------

    print("\nLoading Test-A dataset...")

    test_dataset = load_test_dataset()

    print("Test dataset loaded.")

    # --------------------------------------------------------
    # Image paths
    # --------------------------------------------------------

    print("\nCollecting image paths...")

    image_paths = get_image_paths()

    print(
        f"Image paths found: {len(image_paths)}"
    )

    # --------------------------------------------------------
    # Predictions
    # --------------------------------------------------------

    print("\nGenerating predictions...")

    (
        true_labels,
        predicted_labels,
        confidences,
        probabilities
    ) = generate_predictions(
        model,
        test_dataset
    )

    # --------------------------------------------------------
    # Sanity check
    # --------------------------------------------------------

    if len(image_paths) != len(true_labels):

        raise RuntimeError(
            "Number of image paths does not match "
            "number of predictions."
        )

    # --------------------------------------------------------
    # Collect errors
    # --------------------------------------------------------

    errors_df = collect_errors(
        true_labels,
        predicted_labels,
        confidences,
        image_paths
    )

    print(
        f"\nMisclassified images: {len(errors_df)}"
    )

    # --------------------------------------------------------
    # Save error table
    # --------------------------------------------------------

    errors_df.to_csv(
        ERROR_CSV,
        index=False
    )

    print(
        f"Misclassified image list saved to:"
    )

    print(ERROR_CSV)

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print_error_summary(
        true_labels,
        predicted_labels,
        errors_df
    )

    # --------------------------------------------------------
    # Confused pairs
    # --------------------------------------------------------

    pair_counts = analyze_confused_pairs(
        errors_df
    )

    # --------------------------------------------------------
    # Display examples
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("MISCLASSIFIED EXAMPLES")
    print("=" * 70)

    display_misclassified_examples(
        errors_df,
        num_examples=20
    )

    print("\n" + "=" * 70)
    print("ERROR ANALYSIS COMPLETE")
    print("=" * 70)

    return errors_df, pair_counts


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    run_error_analysis()
