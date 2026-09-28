import cv2
import matplotlib.pyplot as plt
import numpy as np

# 1. Read an image using OpenCV
image = cv2.imread(r"/content/images (1).jfif")   
# Make sure the file name is correct

# 2. Print image.shape
print("Image Shape:", image.shape)

# 3. Identify rows, columns, channels
rows, cols, channels = image.shape
print("Number of rows (height):", rows)
print("Number of columns (width):", cols)
print("Number of channels:", channels)

# 4. Access and print pixel values
print("Pixel at (50,50):", image[50, 50])
print("Pixel at (100,100):", image[100, 100])
print("Pixel at (200,150):", image[200, 150])

# 5. Split the image into Blue, Green, Red channels
blue_channel, green_channel, red_channel = cv2.split(image)

# 6. Display all three channels separately in grayscale
plt.figure(figsize=(10,4))
plt.subplot(1,3,1)
plt.imshow(blue_channel, cmap="gray")
plt.title("Blue Channel (Grayscale)")
plt.axis("off")

plt.subplot(1,3,2)
plt.imshow(green_channel, cmap="gray")
plt.title("Green Channel (Grayscale)")
plt.axis("off")

plt.subplot(1,3,3)
plt.imshow(red_channel, cmap="gray")
plt.title("Red Channel (Grayscale)")
plt.axis("off")
plt.show()

# 7. Display each channel in its actual colour
zeros = np.zeros_like(blue_channel)

blue_img = cv2.merge([blue_channel, zeros, zeros])
green_img = cv2.merge([zeros, green_channel, zeros])
red_img = cv2.merge([zeros, zeros, red_channel])

plt.figure(figsize=(10,4))
plt.subplot(1,3,1)
plt.imshow(cv2.cvtColor(blue_img, cv2.COLOR_BGR2RGB))
plt.title("Blue Channel (Color)")
plt.axis("off")

plt.subplot(1,3,2)
plt.imshow(cv2.cvtColor(green_img, cv2.COLOR_BGR2RGB))
plt.title("Green Channel (Color)")
plt.axis("off")

plt.subplot(1,3,3)
plt.imshow(cv2.cvtColor(red_img, cv2.COLOR_BGR2RGB))
plt.title("Red Channel (Color)")
plt.axis("off")
plt.show()

# 8. Print the shape of each channel
print("Blue channel shape:", blue_channel.shape)
print("Green channel shape:", green_channel.shape)
print("Red channel shape:", red_channel.shape)
