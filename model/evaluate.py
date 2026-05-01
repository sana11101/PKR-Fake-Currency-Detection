# =============================================================
# model/evaluate.py
# WRITTEN BY: Sana
# PURPOSE: Evaluates the trained CNN model.
#          Shows accuracy, classification report, and confusion matrix.
#          Run AFTER train_cnn.py has been executed.
# =============================================================

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix
)

# ------------------------------------------------------------------
# Settings
# ------------------------------------------------------------------

MODEL_PATH   = "model/saved/cnn_model.h5"
X_TEST_PATH  = "model/saved/X_test.npy"
Y_TEST_PATH  = "model/saved/y_test.npy"
SAVE_DIR     = "model/saved"

# ------------------------------------------------------------------
# Load model and test data
# ------------------------------------------------------------------

def load_all():
    """Loads the trained CNN model and saved test data."""

    if not os.path.exists(MODEL_PATH):
        print("❌ CNN model not found! Run train_cnn.py first.")
        exit()

    if not os.path.exists(X_TEST_PATH) or not os.path.exists(Y_TEST_PATH):
        print("❌ Test data not found! Run train_cnn.py first.")
        exit()

    print("📦 Loading model and test data...")
    model  = tf.keras.models.load_model(MODEL_PATH)
    X_test = np.load(X_TEST_PATH)
    y_test = np.load(Y_TEST_PATH)

    print(f"   Test samples: {len(X_test)}")
    return model, X_test, y_test

# ------------------------------------------------------------------
# Evaluate CNN
# ------------------------------------------------------------------

def evaluate_cnn(model, X_test, y_test):
    """Runs evaluation and prints metrics."""

    print("\n📊 Evaluating CNN on test set...")

    # Raw predictions (values between 0 and 1)
    y_pred_raw = model.predict(X_test, verbose=0).flatten()

    # Convert to binary labels: >=0.5 → Fake (1), <0.5 → Real (0)
    y_pred = (y_pred_raw >= 0.5).astype(int)

    acc = accuracy_score(y_test, y_pred)
    print(f"\n✅ CNN Accuracy: {acc * 100:.2f}%")

    print("\n📋 Classification Report:")
    print(classification_report(y_test, y_pred,
                                target_names=["Real", "Fake"]))

    return y_pred

# ------------------------------------------------------------------
# Plot Confusion Matrix
# ------------------------------------------------------------------

def plot_confusion_matrix(y_test, y_pred):
    """Saves and shows a colour-coded confusion matrix."""

    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt='d',
        cmap='Blues',
        xticklabels=["Real", "Fake"],
        yticklabels=["Real", "Fake"],
        ax=ax
    )
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label",      fontsize=12)
    ax.set_title("CNN Confusion Matrix", fontsize=14)

    plt.tight_layout()
    save_path = os.path.join(SAVE_DIR, "confusion_matrix.png")
    plt.savefig(save_path, dpi=150)
    print(f"\n🖼️  Confusion matrix saved to {save_path}")
    plt.show()
    plt.close()

# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

if __name__ == "__main__":

    print("=" * 50)
    print("  PKR Fake Currency — Model Evaluation")
    print("=" * 50)

    model, X_test, y_test = load_all()
    y_pred = evaluate_cnn(model, X_test, y_test)
    plot_confusion_matrix(y_test, y_pred)

    print("\n" + "=" * 50)
    print("  ✅ Evaluation Complete!")
    print("=" * 50)
