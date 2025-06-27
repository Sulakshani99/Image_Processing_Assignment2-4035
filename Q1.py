import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

# Create output directory
os.makedirs('images', exist_ok=True)

# Create a synthetic image with 3 pixel levels (background, object 1, object 2)
img = np.zeros((100, 100), dtype=np.uint8) # Background
# Object 1: Circle
cv2.circle(img, (35, 35), 15, 128, -1)  
# Object 2: Square
img[60:90, 60:90] = 255  


# Add Gaussian noise
mean = 0
stddev = 20
noise = np.random.normal(mean, stddev, img.shape).astype(np.int16)
noisy_img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)


# Optional: Smooth it
smoothed = cv2.GaussianBlur(noisy_img, (5, 5), 0)

# Apply Otsu's threshold
_, otsu_thresh = cv2.threshold(smoothed, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)


# Save images
cv2.imwrite('images/original.png', img)
cv2.imwrite('images/noisy.png', noisy_img)
cv2.imwrite('images/otsu_threshold.png', otsu_thresh)

# Visualization (optional)
plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.title("Original")
plt.imshow(img, cmap='gray')
plt.axis("off")

plt.subplot(1, 3, 2)
plt.title("Noisy")
plt.imshow(noisy_img, cmap='gray')
plt.axis("off")

plt.subplot(1, 3, 3)
plt.title("Otsu Result")
plt.imshow(otsu_thresh, cmap='gray')
plt.axis("off")

plt.tight_layout()
plt.savefig('images/otsu_results.png')
plt.show()