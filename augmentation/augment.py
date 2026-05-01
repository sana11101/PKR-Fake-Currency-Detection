# =============================================================
# augmentation/augment.py
# WRITTEN BY: Sana
# PURPOSE: Takes raw photos from dataset/real and dataset/fake
#          and creates many more variations of each image.
#          This is called "Data Augmentation".
#          Example: 200 photos → 4000+ photos
# =============================================================

import os                        # for working with folders and files
import cv2                       # for reading and saving images
import albumentations as A       # library for image augmentation
import numpy as np               # for numerical operations

# ------------------------------------------------------------------
# STEP 1: Define folder paths
# ------------------------------------------------------------------

# Where your original photos are stored
INPUT_REAL_DIR  = "dataset/real"   # your real note photos go here
INPUT_FAKE_DIR  = "dataset/fake"   # your fake note photos go here

# Where augmented (new) images will be saved
OUTPUT_REAL_DIR = "dataset/augmented/real"
OUTPUT_FAKE_DIR = "dataset/augmented/fake"

# How many new images to create from each original photo
AUGMENTATIONS_PER_IMAGE = 20  # 200 photos × 20 = 4000 images

# ------------------------------------------------------------------
# STEP 2: Define augmentation techniques
# Each technique slightly changes the image to create a new one
# ------------------------------------------------------------------

augmentation_pipeline = A.Compose([

    # Randomly rotate the image between -15 and +15 degrees
    A.Rotate(limit=15, p=0.7),

    # Randomly flip the image left-right (50% chance)
    A.HorizontalFlip(p=0.5),

    # Randomly change brightness and contrast
    A.RandomBrightnessContrast(
        brightness_limit=0.3,   # up to 30% brighter or darker
        contrast_limit=0.3,     # up to 30% more or less contrast
        p=0.8
    ),

    # Randomly zoom into the image
    A.RandomScale(scale_limit=0.2, p=0.5),

    # Add a small amount of blur (simulates shaky camera)
    A.GaussianBlur(blur_limit=(3, 5), p=0.3),

    # Add random noise (simulates low quality camera)
    A.GaussNoise(p=0.3),

    # Randomly shift the image slightly (crop and pad)
    A.ShiftScaleRotate(
        shift_limit=0.1,
        scale_limit=0.1,
        rotate_limit=10,
        p=0.5
    ),

    # Randomly change hue and saturation (different lighting colors)
    A.HueSaturationValue(
        hue_shift_limit=10,
        sat_shift_limit=20,
        val_shift_limit=20,
        p=0.4
    ),
])

# ------------------------------------------------------------------
# STEP 3: Function to augment all images in a folder
# ------------------------------------------------------------------

def augment_images(input_dir, output_dir, label):
    """
    Goes through all images in input_dir,
    creates AUGMENTATIONS_PER_IMAGE new versions of each,
    and saves them in output_dir.

    label: "real" or "fake" — just used for printing messages
    """

    # Create output folder if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Get list of all image files in the input folder
    image_files = [
        f for f in os.listdir(input_dir)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ]

    print(f"\n📂 Processing {label} images...")
    print(f"   Found {len(image_files)} original images")
    print(f"   Will create {len(image_files) * AUGMENTATIONS_PER_IMAGE} augmented images")

    total_saved = 0

    # Loop through each original image
    for image_file in image_files:

        # Build full path to the image
        image_path = os.path.join(input_dir, image_file)

        # Read the image using OpenCV
        image = cv2.imread(image_path)

        # Skip if image couldn't be read
        if image is None:
            print(f"   ⚠️  Could not read {image_file}, skipping...")
            continue

        # Convert from BGR (OpenCV format) to RGB (standard format)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Resize image to standard size (224x224 pixels)
        # CNN expects all images to be the same size
        image = cv2.resize(image, (224, 224))

        # Get the base filename without extension
        base_name = os.path.splitext(image_file)[0]

        # Create AUGMENTATIONS_PER_IMAGE new versions
        for i in range(AUGMENTATIONS_PER_IMAGE):

            # Apply the augmentation pipeline to create a new image
            augmented = augmentation_pipeline(image=image)
            augmented_image = augmented["image"]

            # Convert back to BGR for saving with OpenCV
            augmented_image_bgr = cv2.cvtColor(augmented_image, cv2.COLOR_RGB2BGR)

            # Create filename: original_name_aug_1.jpg, original_name_aug_2.jpg, etc.
            output_filename = f"{base_name}_aug_{i+1}.jpg"
            output_path = os.path.join(output_dir, output_filename)

            # Save the augmented image
            cv2.imwrite(output_path, augmented_image_bgr)
            total_saved += 1

        # Also save the original image (resized) to the output folder
        original_output_path = os.path.join(output_dir, f"{base_name}_original.jpg")
        cv2.imwrite(original_output_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        total_saved += 1

    print(f"   ✅ Saved {total_saved} images to {output_dir}")
    return total_saved

# ------------------------------------------------------------------
# STEP 4: Run the augmentation
# ------------------------------------------------------------------

if __name__ == "__main__":

    print("=" * 50)
    print("  PKR Currency Dataset Augmentation")
    print("=" * 50)

    # Check if input folders exist
    if not os.path.exists(INPUT_REAL_DIR):
        print(f"\n❌ ERROR: '{INPUT_REAL_DIR}' folder not found!")
        print("   Please add your real note photos to dataset/real/")
        exit()

    if not os.path.exists(INPUT_FAKE_DIR):
        print(f"\n❌ ERROR: '{INPUT_FAKE_DIR}' folder not found!")
        print("   Please add your fake note photos to dataset/fake/")
        exit()

    # Run augmentation for real images
    real_count = augment_images(INPUT_REAL_DIR, OUTPUT_REAL_DIR, "real")

    # Run augmentation for fake images
    fake_count = augment_images(INPUT_FAKE_DIR, OUTPUT_FAKE_DIR, "fake")

    # Print final summary
    print("\n" + "=" * 50)
    print("  ✅ Augmentation Complete!")
    print(f"  Real images: {real_count}")
    print(f"  Fake images: {fake_count}")
    print(f"  Total:       {real_count + fake_count}")
    print("=" * 50)
    print("\nNext step: Run  python model/train_cnn.py")
