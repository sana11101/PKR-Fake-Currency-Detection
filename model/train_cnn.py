# =============================================================
# model/train_cnn.py
# WRITTEN BY: Sana
# PURPOSE: Builds and trains the CNN (Convolutional Neural Network)
#          model to detect real vs fake PKR banknotes.
#          Saves the trained model to model/saved/cnn_model.h5
# =============================================================

import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import cv2

# ------------------------------------------------------------------
# STEP 1: Settings / Configuration
# ------------------------------------------------------------------

# Folder where augmented images are stored
DATASET_DIR = "dataset/augmented"

# Image size — must match what we used in augmentation
IMG_SIZE = (224, 224)

# Batch size — how many images to process at once during training
BATCH_SIZE = 32

# Number of training rounds (epochs)
# More epochs = more learning, but takes longer
EPOCHS = 20

# Where to save the trained model
MODEL_SAVE_PATH = "model/saved/cnn_model.h5"

# ------------------------------------------------------------------
# STEP 2: Load images from dataset folder
# ------------------------------------------------------------------

def load_dataset(dataset_dir):
    """
    Reads all images from dataset/augmented/real and dataset/augmented/fake
    Returns:
        images: array of image pixel values
        labels: array of 0 (real) or 1 (fake)
    """

    images = []  # will store image data
    labels = []  # will store 0 for real, 1 for fake

    # Define class folders
    classes = {
        "real": 0,   # label 0 = real
        "fake": 1    # label 1 = fake
    }

    print("\n📂 Loading dataset...")

    for class_name, label in classes.items():

        class_dir = os.path.join(dataset_dir, class_name)

        if not os.path.exists(class_dir):
            print(f"⚠️  Folder not found: {class_dir}")
            continue

        # Get all image files in this folder
        image_files = [
            f for f in os.listdir(class_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ]

        print(f"   {class_name}: {len(image_files)} images")

        for image_file in image_files:
            image_path = os.path.join(class_dir, image_file)

            # Read and resize image
            image = cv2.imread(image_path)
            if image is None:
                continue

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, IMG_SIZE)

            # Normalize pixel values from 0-255 to 0-1
            # Neural networks train better with small numbers
            image = image / 255.0

            images.append(image)
            labels.append(label)

    # Convert lists to numpy arrays (required by TensorFlow)
    images = np.array(images, dtype=np.float32)
    labels = np.array(labels, dtype=np.int32)

    print(f"\n   ✅ Total images loaded: {len(images)}")
    print(f"   Real: {np.sum(labels == 0)} | Fake: {np.sum(labels == 1)}")

    return images, labels

# ------------------------------------------------------------------
# STEP 3: Build the CNN Model
# ------------------------------------------------------------------

def build_cnn_model():
    """
    Creates the CNN architecture.

    CNN works like this:
    1. Conv layers detect features (edges, patterns, textures)
    2. Pooling layers reduce image size (keep important info)
    3. Dense layers make the final real/fake decision
    """

    model = models.Sequential([

        # --- BLOCK 1 ---
        # Conv2D: detects basic features (edges, lines)
        # 32 filters, each 3x3 pixels
        layers.Conv2D(32, (3, 3), activation='relu',
                      input_shape=(224, 224, 3),
                      name='conv1'),

        # MaxPooling: reduces image size by half (keeps strongest features)
        layers.MaxPooling2D((2, 2), name='pool1'),

        # --- BLOCK 2 ---
        # More filters to detect more complex patterns
        layers.Conv2D(64, (3, 3), activation='relu', name='conv2'),
        layers.MaxPooling2D((2, 2), name='pool2'),

        # --- BLOCK 3 ---
        # Even more complex features (watermarks, security threads)
        layers.Conv2D(128, (3, 3), activation='relu', name='conv3'),
        layers.MaxPooling2D((2, 2), name='pool3'),

        # --- BLOCK 4 ---
        layers.Conv2D(128, (3, 3), activation='relu', name='conv4'),
        layers.MaxPooling2D((2, 2), name='pool4'),

        # Flatten: converts 3D feature maps to 1D array
        layers.Flatten(name='flatten'),

        # Dropout: randomly turns off 50% of neurons during training
        # Prevents overfitting (memorizing instead of learning)
        layers.Dropout(0.5, name='dropout'),

        # Dense layer: 256 neurons, learns complex combinations
        layers.Dense(256, activation='relu', name='dense1'),

        # Output layer: 1 neuron, sigmoid gives value between 0 and 1
        # Close to 0 = Real, Close to 1 = Fake
        layers.Dense(1, activation='sigmoid', name='output')
    ])

    # Compile the model
    # optimizer: how the model updates its weights
    # loss: measures how wrong the model is
    # metrics: what we track during training
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return model

# ------------------------------------------------------------------
# STEP 4: Plot training results
# ------------------------------------------------------------------

def plot_training_history(history):
    """
    Creates a graph showing how accuracy improved over epochs.
    Saves the graph as model/saved/training_history.png
    """

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Accuracy plot
    ax1.plot(history.history['accuracy'], label='Training Accuracy', color='blue')
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy', color='orange')
    ax1.set_title('Model Accuracy Over Epochs')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True)

    # Loss plot
    ax2.plot(history.history['loss'], label='Training Loss', color='blue')
    ax2.plot(history.history['val_loss'], label='Validation Loss', color='orange')
    ax2.set_title('Model Loss Over Epochs')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig("model/saved/training_history.png")
    print("\n📊 Training history plot saved to model/saved/training_history.png")
    plt.show()

# ------------------------------------------------------------------
# STEP 5: Main training function
# ------------------------------------------------------------------

if __name__ == "__main__":

    print("=" * 50)
    print("  PKR Fake Currency — CNN Training")
    print("=" * 50)

    # Create folder to save model
    os.makedirs("model/saved", exist_ok=True)

    # Load dataset
    images, labels = load_dataset(DATASET_DIR)

    if len(images) == 0:
        print("\n❌ No images found! Run augmentation first:")
        print("   python augmentation/augment.py")
        exit()

    # Split into training and testing sets
    # 80% for training, 20% for testing
    X_train, X_test, y_train, y_test = train_test_split(
        images, labels,
        test_size=0.2,       # 20% goes to testing
        random_state=42,     # fixed seed for reproducibility
        stratify=labels      # keeps equal ratio of real/fake in both sets
    )

    print(f"\n📊 Dataset split:")
    print(f"   Training:   {len(X_train)} images")
    print(f"   Testing:    {len(X_test)} images")

    # Save test data for evaluation later
    np.save("model/saved/X_test.npy", X_test)
    np.save("model/saved/y_test.npy", y_test)

    # Build CNN model
    print("\n🏗️  Building CNN model...")
    model = build_cnn_model()

    # Print model summary (shows all layers)
    model.summary()

    # Early stopping: stop training if model stops improving
    # Prevents wasting time on unnecessary epochs
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',    # watch validation loss
        patience=5,            # stop if no improvement for 5 epochs
        restore_best_weights=True  # keep the best version
    )

    # Train the model
    print(f"\n🚀 Starting training for up to {EPOCHS} epochs...")
    print("   (This may take 10-30 minutes depending on your computer)\n")

    history = model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=0.1,   # use 10% of training data for validation
        callbacks=[early_stopping],
        verbose=1               # show progress bar
    )

    # Evaluate on test set
    print("\n📊 Evaluating on test set...")
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"\n✅ CNN Test Accuracy: {test_accuracy * 100:.2f}%")
    print(f"   CNN Test Loss:     {test_loss:.4f}")

    # Save the trained model
    model.save(MODEL_SAVE_PATH)
    print(f"\n💾 Model saved to {MODEL_SAVE_PATH}")

    # Plot training history
    plot_training_history(history)

    print("\n" + "=" * 50)
    print("  ✅ CNN Training Complete!")
    print("  Next step: python model/train_comparison.py")
    print("=" * 50)
