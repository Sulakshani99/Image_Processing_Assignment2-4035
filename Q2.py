import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

def region_growing(img, seed, thresh=20):
    output = np.zeros_like(img)
    visited = np.zeros_like(img, dtype=bool)
    h, w = img.shape
    seed_val = img[seed]
    stack = [seed]
    while stack:
        x, y = stack.pop()
        if visited[x, y]:
            continue
        visited[x, y] = True
        if abs(int(img[x, y]) - int(seed_val)) <= thresh:
            output[x, y] = 255
            for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
                nx, ny = x+dx, y+dy
                if 0 <= nx < h and 0 <= ny < w and not visited[nx, ny]:
                    stack.append((nx, ny))
    return output

# Load the noisy image from previous step
noisy_img = cv2.imread('images/noisy.png', cv2.IMREAD_GRAYSCALE)

# Choose a seed inside object 1 (e.g., (25,25))
seed_point = (25, 25)
region = region_growing(noisy_img, seed_point, thresh=25)

# Save result
os.makedirs('images', exist_ok=True)
cv2.imwrite('images/region_grown.png', region)

# Visualization (optional)
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.title('Noisy Image')
plt.imshow(noisy_img, cmap='gray')
plt.axis('off')

plt.subplot(1,2,2)
plt.title('Region Grown')
plt.imshow(region, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.savefig('images/region_growing_results.png')
plt.show() 