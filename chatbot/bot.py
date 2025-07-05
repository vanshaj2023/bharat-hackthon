import cv2
import numpy as np
from sklearn.cluster import KMeans
from matplotlib.colors import rgb2hex
from PIL import Image

def extract_skin(image):
    # Convert image to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define skin color range in HSV
    lower = np.array([0, 20, 70], dtype=np.uint8)
    upper = np.array([20, 255, 255], dtype=np.uint8)

    # Mask skin region
    mask = cv2.inRange(hsv, lower, upper)
    skin = cv2.bitwise_and(image, image, mask=mask)
    
    return skin

def get_dominant_color(image, k=1):
    # Resize for faster processing
    image = cv2.resize(image, (100, 100))
    image = image.reshape((-1, 3))

    # Remove black pixels
    image = image[np.any(image != [0, 0, 0], axis=1)]

    # Apply KMeans
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(image)

    # Get RGB color
    dominant_color = kmeans.cluster_centers_[0].astype(int)
    hex_color = rgb2hex([x/255.0 for x in dominant_color])

    return dominant_color, hex_color

# Load image
img_path = r"C:\Users\karti\OneDrive\Desktop\color\WhatsApp Image 2024-10-12 at 10.58.21_ebb25fee.jpg"
image = cv2.imread(img_path)
skin = extract_skin(image)
rgb_color, hex_color = get_dominant_color(skin)

print("Dominant Skin Color (RGB):", rgb_color)
print("Dominant Skin Color (HEX):", hex_color)