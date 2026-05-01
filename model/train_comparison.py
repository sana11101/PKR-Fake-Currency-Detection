# =============================================================
# model/train_comparison.py
# WRITTEN BY: Sana
# PURPOSE: Trains KNN and Logistic Regression on the same dataset
#          and compares their accuracy with CNN.
#          This shows WHY CNN is better for image tasks.
# =============================================================

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)
from sklearn.preprocessing import StandardScaler

# ------------------------------------------------------------------
# STEP 1: Load the test data saved by train_cnn.py
# ------------------------------------------------------------------

print("=" * 50)
print("  PKR Fake Currency — Model Comparison")
print("  (CNN vs KNN vs Logistic Regression)")
print("=" * 50)

# Check if CNN training has been done first
if not os.path.exists("model/saved/X_test.npy"):
    print("\n❌ Test data not found!")
    print("   Please run CNN training first:")
    print("   python model/train_cnn.py")
    exit()

print("\n📂 Loading dataset...")

# Load the same test data used by CNN
X_test  = np.load("model/saved/X_test.npy")
y_test  = np.load("model/saved/y_test.npy")

# Also load training data
# We need to reload the full dataset and split again
import cv2
from sklearn.model_selection import train_test_split

DATASET_DIR = "dataset/augmented"
IMG_SIZE = (224, 224)

def load_flat_dataset(dataset_dir):
    """
    Loads images and FLATTENS them into 1D arrays.
    KNN and Logistic Regression cannot work with 3D image arrays.
    They need a flat list of numbers per image.

    Example:
    CNN input:  (224, 224, 3) → 3D array
    KNN input:  (150528,)     → 1D flat array (224 × 224 × 3 = 150528)
    """

    images = []
    labels = []

    classes = {"real": 0, "fake": 1}

    for class_name, label in classes.items():
        class_dir = os.path.join(dataset_dir, class_name)

        if not os.path.exists(class_dir):
            continue

        image_files = [
            f for f in os.listdir(class_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ]

        print(f"   {class_name}: {len(image_files)} images")

        for image_file in image_files:
            image_path = os.path.join(class_dir, image_file)
            image = cv2.imread(image_path)
            if image is None:
                continue

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, IMG_SIZE)
            image = image / 255.0

            # FLATTEN: convert 224x224x3 → single row of 150528 numbers
            image_flat = image.flatten()

            images.append(image_flat)
            labels.append(label)

    return np.array(images, dtype=np.float32), np.array(labels, dtype=np.int32)


# Load flattened dataset
images_flat, labels = load_flat_dataset(DATASET_DIR)

# Split same way as CNN (same random_state=42 ensures same split)
X_train_flat, X_test_flat, y_train, y_test_flat = train_test_split(
    images_flat, labels,
    test_size=0.2,
    random_state=42,
    stratify=labels
)

print(f"\n📊 Dataset: {len(X_train_flat)} train | {len(X_test_flat)} test")

# ------------------------------------------------------------------
# STEP 2: Scale features
# KNN and Logistic Regression work better with scaled data
# StandardScaler makes all values have mean=0 and std=1
# ------------------------------------------------------------------

print("\n⚙️  Scaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_flat)
X_test_scaled  = scaler.transform(X_test_flat)

# ------------------------------------------------------------------
# STEP 3: Train KNN (K-Nearest Neighbors)
# ------------------------------------------------------------------
# How KNN works:
# - For a new image, find the K most similar images in training set
# - Vote: if most neighbors are "fake", predict "fake"
# - K=5 means look at 5 nearest neighbors
# ------------------------------------------------------------------

print("\n🔵 Training KNN (K=5)...")
print("   (This may take a few minutes on large datasets)")

knn = KNeighborsClassifier(
    n_neighbors=5,    # look at 5 nearest neighbors
    metric='euclidean' # measure similarity using Euclidean distance
)
knn.fit(X_train_scaled, y_train)

knn_predictions = knn.predict(X_test_scaled)
knn_accuracy = accuracy_score(y_test_flat, knn_predictions)

print(f"   ✅ KNN Accuracy: {knn_accuracy * 100:.2f}%")

# ------------------------------------------------------------------
# STEP 4: Train Logistic Regression
# ------------------------------------------------------------------
# How Logistic Regression works:
# - Learns a mathematical formula that maps pixel values → real/fake
# - Uses sigmoid function to output probability between 0 and 1
# - If probability > 0.5 → Fake, else → Real
# ------------------------------------------------------------------

print("\n🟢 Training Logistic Regression...")

lr = LogisticRegression(
    max_iter=1000,    # maximum iterations to find best formula
    random_state=42,
    solver='saga'     # efficient solver for large datasets
)
lr.fit(X_train_scaled, y_train)

lr_predictions = lr.predict(X_test_scaled)
lr_accuracy = accuracy_score(y_test_flat, lr_predictions)

print(f"   ✅ Logistic Regression Accuracy: {lr_accuracy * 100:.2f}%")

# ------------------------------------------------------------------
# STEP 5: Load CNN accuracy from saved results
# ------------------------------------------------------------------

# Try to load CNN accuracy saved during training
cnn_accuracy = None
if os.path.exists("model/saved/cnn_accuracy.txt"):
    with open("model/saved/cnn_accuracy.txt", "r") as f:
        cnn_accuracy = float(f.read().strip())
else:
    # If not saved separately, load model and evaluate
    import tensorflow as tf
    if os.path.exists("model/saved/cnn_model.h5"):
        print("\n🔴 Loading CNN model for accuracy...")
        cnn_model = tf.keras.models.load_model("model/saved/cnn_model.h5")
        # Reshape test data back to 3D for CNN
        X_test_3d = X_test_flat.reshape(-1, 224, 224, 3)
        _, cnn_accuracy = cnn_model.evaluate(X_test_3d, y_test_flat, verbose=0)
        print(f"   ✅ CNN Accuracy: {cnn_accuracy * 100:.2f}%")

# ------------------------------------------------------------------
# STEP 6: Print comparison table
# ------------------------------------------------------------------

print("\n" + "=" * 50)
print("  📊 MODEL COMPARISON RESULTS")
print("=" * 50)

if cnn_accuracy:
    print(f"  CNN (Convolutional Neural Net): {cnn_accuracy * 100:.2f}%  ⭐ Best")
print(f"  KNN (K-Nearest Neighbors):       {knn_accuracy * 100:.2f}%")
print(f"  Logistic Regression:             {lr_accuracy * 100:.2f}%")
print("=" * 50)

# ------------------------------------------------------------------
# STEP 7: Plot comparison bar chart
# ------------------------------------------------------------------

os.makedirs("model/saved", exist_ok=True)

models_list = ['Logistic\nRegression', 'KNN\n(K=5)']
accuracies  = [lr_accuracy * 100, knn_accuracy * 100]
colors      = ['#2ecc71', '#3498db']

if cnn_accuracy:
    models_list.append('CNN')
    accuracies.append(cnn_accuracy * 100)
    colors.append('#e74c3c')

plt.figure(figsize=(8, 5))
bars = plt.bar(models_list, accuracies, color=colors, width=0.5, edgecolor='black')

# Add accuracy value on top of each bar
for bar, acc in zip(bars, accuracies):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.5,
        f'{acc:.1f}%',
        ha='center', va='bottom', fontweight='bold', fontsize=12
    )

plt.title('Model Accuracy Comparison\nPKR Fake Currency Detection', fontsize=14)
plt.ylabel('Accuracy (%)', fontsize=12)
plt.ylim(0, 110)
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig("model/saved/model_comparison.png", dpi=150)
print("\n📊 Comparison chart saved to model/saved/model_comparison.png")
plt.show()

# ------------------------------------------------------------------
# STEP 8: Confusion matrices for KNN and LR
# ------------------------------------------------------------------

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for ax, predictions, model_name in zip(
    axes,
    [knn_predictions, lr_predictions],
    ['KNN', 'Logistic Regression']
):
    cm = confusion_matrix(y_test_flat, predictions)
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=['Real', 'Fake'],
        yticklabels=['Real', 'Fake'],
        ax=ax
    )
    ax.set_title(f'{model_name} Confusion Matrix')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')

plt.tight_layout()
plt.savefig("model/saved/confusion_matrices.png", dpi=150)
print("📊 Confusion matrices saved to model/saved/confusion_matrices.png")
plt.show()

# Detailed classification report
print("\n📋 KNN Classification Report:")
print(classification_report(y_test_flat, knn_predictions,
                             target_names=['Real', 'Fake']))

print("\n📋 Logistic Regression Classification Report:")
print(classification_report(y_test_flat, lr_predictions,
                             target_names=['Real', 'Fake']))

print("\n✅ Comparison complete!")
print("   Next step: python gradcam/visualize.py")
