import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

# Create output directory
os.makedirs('images', exist_ok=True)

# Create a synthetic image
img = np.zeros((100, 100), dtype=np.uint8)
img[20:50, 20:50] = 100  # Object 1
img[60:90, 60:90] = 200  # Object 2

# Add Gaussian noise
mean = 0
sigma = 20
gauss = np.random.normal(mean, sigma, img.shape).astype(np.int16)
noisy_img = np.clip(img.astype(np.int16) + gauss, 0, 255).astype(np.uint8)

# Otsu's thresholding
_, otsu_thresh = cv2.threshold(noisy_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Save images
cv2.imwrite('images/original.png', img)
cv2.imwrite('images/noisy.png', noisy_img)
cv2.imwrite('images/otsu_threshold.png', otsu_thresh)

# Visualization (optional)
plt.figure(figsize=(10,4))
plt.subplot(1,3,1)
plt.title('Original')
plt.imshow(img, cmap='gray')
plt.axis('off')

plt.subplot(1,3,2)
plt.title('Noisy')
plt.imshow(noisy_img, cmap='gray')
plt.axis('off')

plt.subplot(1,3,3)
plt.title('Otsu Threshold')
plt.imshow(otsu_thresh, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.savefig('images/otsu_results.png')
plt.show() 