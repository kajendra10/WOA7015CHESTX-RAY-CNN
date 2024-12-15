import os
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split

# Path to dataset
normal_path = r"C:\Users\USER\PycharmProjects\WOA7015GroupAssignment\chest_xray\chest_xray\train\NORMAL"
pneumonia_path = r"C:\Users\USER\PycharmProjects\WOA7015GroupAssignment\chest_xray\chest_xray\train\PNEUMONIA"

images = []
labels = []

# Function to load and preprocess images
def load_images_from_directory(directory, label):
    for img_file in os.listdir(directory):
        img_path = os.path.join(directory, img_file)
        try:
            img = Image.open(img_path).resize((224, 224))  # Resize to match CNN input size
            if img.mode != 'RGB':  # Convert grayscale to RGB
                img = img.convert('RGB')
            img_array = np.array(img)
            if img_array.shape == (224, 224, 3):  # Ensure the image has correct shape
                images.append(img_array / 255.0)  # Normalize pixel values
                labels.append(label)  # Append correct label
            else:
                print(f"Skipping invalid image {img_path} with shape {img_array.shape}")
        except Exception as e:
            print(f"Error processing {img_path}: {e}")

# Load images
load_images_from_directory(normal_path, 0)  # Label 0 for NORMAL
load_images_from_directory(pneumonia_path, 1)  # Label 1 for PNEUMONIA

# Convert lists into numpy arrays
images = np.array(images)
labels = np.array(labels)

# Check data size and shape
print(f"Loaded {len(images)} images with shape {images.shape} and {len(np.unique(labels))} classes.")

