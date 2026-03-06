import numpy as np
import cv2
import os

def create_dummy_endoscopic_image(filename="sample_endo.jpg"):
    # Create a base pinkish/fleshy image
    img = np.ones((512, 512, 3), dtype=np.uint8) * np.array([120, 150, 200], dtype=np.uint8) # BGR
    
    # Add some noise for texture
    noise = np.random.randint(-20, 20, (512, 512, 3), dtype=np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    # Apply a vignette effect (darker corners, simulating a scope view)
    rows, cols = 512, 512
    kernel_x = cv2.getGaussianKernel(cols, 200)
    kernel_y = cv2.getGaussianKernel(rows, 200)
    kernel = kernel_y * kernel_x.T
    mask = 255 * kernel / np.linalg.norm(kernel)
    mask = mask / np.max(mask)
    
    for i in range(3):
        img[:,:,i] = img[:,:,i] * mask
        
    # Draw some "vessels"
    for _ in range(10):
        pts = np.random.randint(50, 462, (4, 2))
        pts = pts.reshape((-1, 1, 2))
        cv2.polylines(img, [pts], False, (80, 80, 150), 2, cv2.LINE_AA) # Dark red
        
    cv2.imwrite(filename, img)
    print(f"Created {filename}")

if __name__ == "__main__":
    create_dummy_endoscopic_image()
