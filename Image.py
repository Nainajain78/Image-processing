
#Install Dependencies
!pip install numpy opencv-python matplotlib scikit-learn joblib

# Import Libraries
import cv2
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Load Image
def load_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

# Convert Image to Data for K-Means
def preprocess_image(image):
    pixels = image.reshape((-1, 3))  # Reshape to 2D array of pixels
    return pixels

# Train the K-Means Model
def train_kmeans(image_path, k=8):
    image = load_image(image_path)
    pixels = preprocess_image(image)

    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(pixels)

    joblib.dump(kmeans, "kmeans_model.pkl")  # Save the trained model
    print(f"K-Means Model trained with {k} clusters and saved successfully!")

# Apply K-Means for Image Compression
def compress_image(image_path):
    image = load_image(image_path)
    pixels = preprocess_image(image)

    kmeans = joblib.load("kmeans_model.pkl")  # Load trained model
    compressed_pixels = kmeans.cluster_centers_[kmeans.predict(pixels)]
    compressed_image = compressed_pixels.reshape(image.shape).astype(np.uint8)

    # Save and Display Compressed Image
    cv2.imwrite("compressed.jpg", cv2.cvtColor(compressed_image, cv2.COLOR_RGB2BGR))

    plt.figure(figsize=(10,5))
    plt.subplot(1,2,1)
    plt.title("Original Image")
    plt.imshow(image)

    plt.subplot(1,2,2)
    plt.title("Compressed Image (K-Means)")
    plt.imshow(compressed_image)

    plt.show()
    print("Image compression completed! Compressed image saved as 'compressed.jpg'.")

# Run the Model
if __name__ == "__main__":
    image_path = "input.jpg"  # Replace with your image path
    train_kmeans(image_path, k=8)  # Train K-Means model
    compress_image(image_path)  # Apply compression
