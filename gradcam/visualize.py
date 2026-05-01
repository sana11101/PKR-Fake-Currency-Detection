# =============================================================
# gradcam/visualize.py
# WRITTEN BY: Aqsa
# PURPOSE: Generates a Grad-CAM heatmap on a currency note image.
#          Grad-CAM highlights WHICH PART of the note the CNN
#          used to decide real or fake.
#          Red/Yellow areas = regions that triggered the decision
# =============================================================

import numpy as np
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os

# ------------------------------------------------------------------
# STEP 1: Settings
# ------------------------------------------------------------------

MODEL_PATH = "model/saved/cnn_model.h5"   # trained CNN model
IMG_SIZE   = (224, 224)                    # must match training size

# ------------------------------------------------------------------
# STEP 2: Load the trained CNN model
# ------------------------------------------------------------------

def load_model():
    """Loads the saved CNN model"""

    if not os.path.exists(MODEL_PATH):
        print(f"❌ Model not found at {MODEL_PATH}")
        print("   Please train the CNN first: python model/train_cnn.py")
        return None

    print("📦 Loading CNN model...")
    model = tf.keras.models.load_model(MODEL_PATH)
    print("   ✅ Model loaded!")
    return model

# ------------------------------------------------------------------
# STEP 3: Preprocess image for CNN input
# ------------------------------------------------------------------

def preprocess_image(image_path):
    """
    Reads and prepares an image for the CNN.
    Returns both the processed array and original image for display.
    """

    # Read original image
    original = cv2.imread(image_path)
    if original is None:
        print(f"❌ Could not read image: {image_path}")
        return None, None

    # Convert BGR to RGB
    original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)

    # Resize for CNN
    resized = cv2.resize(original_rgb, IMG_SIZE)

    # Normalize pixel values to 0-1
    normalized = resized / 255.0

    # Add batch dimension: (224,224,3) → (1,224,224,3)
    # CNN expects batches of images, even if it's just one
    input_array = np.expand_dims(normalized, axis=0)

    return input_array, original_rgb

# ------------------------------------------------------------------
# STEP 4: Generate Grad-CAM heatmap
# ------------------------------------------------------------------

def generate_gradcam(model, image_array, layer_name='conv4'):
    """
    Generates Grad-CAM heatmap.

    How Grad-CAM works:
    1. Forward pass: feed image through CNN, get prediction
    2. Find the last conv layer (it has the most spatial info)
    3. Compute gradients of the prediction w.r.t. that layer
    4. High gradient = that region strongly influenced the decision
    5. Create heatmap from these gradients
    6. Overlay heatmap on original image

    layer_name: last conv layer — where spatial features are richest
    """

    # Create a model that outputs both the conv layer AND final prediction
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[
            model.get_layer(layer_name).output,  # conv layer output
            model.output                          # final prediction
        ]
    )

    # Record operations for automatic differentiation (gradient computation)
    with tf.GradientTape() as tape:

        # Forward pass through the model
        conv_outputs, predictions = grad_model(image_array)

        # Get the prediction score (higher = more fake)
        prediction_score = predictions[0][0]

    # Compute gradients of prediction score w.r.t. conv layer outputs
    # This tells us: "which conv features most affected the prediction?"
    gradients = tape.gradient(prediction_score, conv_outputs)

    # Pool gradients across each feature map (take mean across spatial dims)
    # Shape: (num_filters,)
    pooled_gradients = tf.reduce_mean(gradients, axis=(0, 1, 2))

    # Weight each feature map by its importance (pooled gradient)
    conv_outputs = conv_outputs[0]  # remove batch dimension
    heatmap = conv_outputs @ pooled_gradients[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # Apply ReLU: only keep positive activations
    # (we only care about features that increase the prediction)
    heatmap = tf.maximum(heatmap, 0)

    # Normalize heatmap to 0-1 range
    heatmap_max = tf.reduce_max(heatmap)
    if heatmap_max != 0:
        heatmap = heatmap / heatmap_max

    return heatmap.numpy()

# ------------------------------------------------------------------
# STEP 5: Overlay heatmap on original image
# ------------------------------------------------------------------

def overlay_heatmap(heatmap, original_image):
    """
    Overlays the Grad-CAM heatmap on the original image.
    Red/Orange areas = most important regions for the decision.
    """

    # Resize heatmap to match original image size
    heatmap_resized = cv2.resize(
        heatmap,
        (original_image.shape[1], original_image.shape[0])
    )

    # Convert heatmap to colormap (0=blue → 0.5=green → 1=red)
    heatmap_colored = cm.jet(heatmap_resized)[:, :, :3]  # remove alpha channel
    heatmap_colored = (heatmap_colored * 255).astype(np.uint8)

    # Blend original image with heatmap
    # 0.6 = 60% original, 0.4 = 40% heatmap
    original_float = original_image.astype(np.float32)
    heatmap_float  = heatmap_colored.astype(np.float32)

    superimposed = cv2.addWeighted(original_float, 0.6, heatmap_float, 0.4, 0)
    superimposed = np.clip(superimposed, 0, 255).astype(np.uint8)

    return superimposed

# ------------------------------------------------------------------
# STEP 6: Full prediction + visualization pipeline
# ------------------------------------------------------------------

def predict_and_visualize(image_path, save_output=True):
    """
    Main function:
    1. Load image
    2. Get CNN prediction (real or fake)
    3. Generate Grad-CAM heatmap
    4. Display side-by-side: original | heatmap overlay

    Returns: prediction label, confidence score, heatmap image
    """

    # Load model
    model = load_model()
    if model is None:
        return None, None, None

    # Preprocess image
    image_array, original_image = preprocess_image(image_path)
    if image_array is None:
        return None, None, None

    # Get prediction
    prediction_value = model.predict(image_array, verbose=0)[0][0]

    # Convert to label
    # sigmoid output: close to 0 = real, close to 1 = fake
    if prediction_value >= 0.5:
        label      = "FAKE"
        confidence = prediction_value * 100
        color      = (255, 0, 0)   # Red for fake
    else:
        label      = "REAL"
        confidence = (1 - prediction_value) * 100
        color      = (0, 200, 0)   # Green for real

    print(f"\n🔍 Prediction: {label}")
    print(f"   Confidence: {confidence:.1f}%")

    # Generate Grad-CAM heatmap
    print("🔥 Generating Grad-CAM heatmap...")
    heatmap = generate_gradcam(model, image_array, layer_name='conv4')

    # Overlay heatmap on original
    result_image = overlay_heatmap(heatmap, original_image)

    # ------------------------------------------------------------------
    # Display results
    # ------------------------------------------------------------------

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: original image
    axes[0].imshow(original_image)
    axes[0].set_title('Original Note', fontsize=14)
    axes[0].axis('off')

    # Right: heatmap overlay
    axes[1].imshow(result_image)
    axes[1].set_title(
        f'Grad-CAM Heatmap\n{label} ({confidence:.1f}% confidence)',
        fontsize=14,
        color='red' if label == 'FAKE' else 'green'
    )
    axes[1].axis('off')

    # Add colorbar legend
    sm = plt.cm.ScalarMappable(cmap='jet', norm=plt.Normalize(0, 1))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=axes[1], fraction=0.046, pad=0.04)
    cbar.set_label('Importance Level\n(Red = Most Important)', fontsize=10)

    plt.suptitle(
        f'PKR Fake Currency Detection — Result: {label}',
        fontsize=16,
        fontweight='bold',
        color='red' if label == 'FAKE' else 'green'
    )

    plt.tight_layout()

    # Save output
    if save_output:
        os.makedirs("gradcam/outputs", exist_ok=True)
        output_path = "gradcam/outputs/gradcam_result.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"💾 Result saved to {output_path}")

    plt.show()

    return label, confidence, result_image

# ------------------------------------------------------------------
# STEP 7: Run on a test image
# ------------------------------------------------------------------

if __name__ == "__main__":

    import sys

    print("=" * 50)
    print("  PKR Fake Currency — Grad-CAM Visualizer")
    print("=" * 50)

    # Get image path from command line argument
    # Usage: python gradcam/visualize.py path/to/note.jpg
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        # Default test: use first image from test set
        print("\n💡 Usage: python gradcam/visualize.py path/to/note.jpg")
        print("   No image provided. Looking for a sample image...\n")

        # Try to find any image in the dataset
        for folder in ["dataset/real", "dataset/fake", "dataset/augmented/real"]:
            if os.path.exists(folder):
                files = [f for f in os.listdir(folder)
                         if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                if files:
                    image_path = os.path.join(folder, files[0])
                    print(f"   Using sample image: {image_path}")
                    break
        else:
            print("❌ No images found. Please provide an image path.")
            exit()

    # Run prediction and visualization
    label, confidence, heatmap = predict_and_visualize(image_path)

    if label:
        print(f"\n{'='*50}")
        print(f"  Final Result: {label} ({confidence:.1f}% confident)")
        print(f"{'='*50}")
