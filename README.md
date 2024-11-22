# PCA-Mini-Project---Face-Detection-or-Convert-an-image-into-gray-scale-image-using-CUD
Mini Project - Face Detection or Convert an image into gray scale image using CUDA GPU programming
```
Name : Iswarya P
Reg no : 212223230082
```
# Convert an image into gray scale image using CUDA GPU programming
# Program

!pip install opencv-python-headless matplotlib cupy-cuda118
```
import cv2
import cupy as cp
from matplotlib import pyplot as plt
import os
```
# Specify the image path
image_path = "/content/rabbit.jpeg"  # Replace with the correct path

# Verify the file exists
```
if not os.path.exists(image_path):
    raise ValueError(f"File does not exist at the specified path: {image_path}")
```
# Load the image using OpenCV
image = cv2.imread(image_path)

# Ensure the image was loaded successfully
```
if image is None:
    raise ValueError(f"Failed to load the image. Check the file path: {image_path}")
```
# Convert BGR to RGB (for proper visualization)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Transfer image to GPU (CuPy array)
image_gpu = cp.array(image, dtype=cp.float32)

# Define CUDA kernel for grayscale conversion
```
grayscale_kernel = cp.ElementwiseKernel(
    'float32 r, float32 g, float32 b',  # Input RGB values
    'float32 gray',                     # Output grayscale value
    'gray = 0.2989 * r + 0.5870 * g + 0.1140 * b;',  # Formula for grayscale
    'grayscale_kernel'
)
```
# Separate RGB channels
```
r_channel = image_gpu[:, :, 0]
g_channel = image_gpu[:, :, 1]
b_channel = image_gpu[:, :, 2]
```
# Apply grayscale conversion
gray_gpu = grayscale_kernel(r_channel, g_channel, b_channel)

# Transfer the grayscale image back to CPU
gray_image = cp.asnumpy(gray_gpu)

# Display the original and grayscale images
```
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(image)
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title("Grayscale Image")
plt.imshow(gray_image, cmap='gray')
plt.axis('off')
plt.show()
```
# Output
![image](https://github.com/user-attachments/assets/ac1d4613-382d-4355-9840-6c2529c7a8a1)
