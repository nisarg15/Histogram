# Import the Libraries
import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob
import os

# Defining the variables
value = 0

# Calling the images from a folder
filename = [img for img in glob.glob("C:/Users/nisar/Desktop/Projects/1/*.png")]
filename.sort()

# Looping over the images to perform the histogram operation
for img in filename:

   image = cv2.imread(img)

# Convert the image to grayscale   
   image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
   img = cv2.resize(image,(800,600))

# Function to perfrom the histogram
   def hist_norm(image):
    
# Plot the Histogram
    y = []
    s = image.shape
    h = np.zeros(shape = (256,1))
    for i in range(s[0]):
     for j in range(s[1]):
        k = image[i,j]
        h[k] = h[k]+1


# Perfoming the CDF to normalize the histogram plot       
    x = h.reshape(1,256)
    for i in x:
     y.append(x[0,0])


    for i in range(255):
     k = x[0,i+1] + y[i]
     y = np.append(y,k)

   
    s = image.shape
    y = np.round((y/(s[0]*s[1]))*255)


# Replacing the old intensity values with the new one that we obtain from CDF
    for i in range(s[0]):
     for j in range(s[1]):
        k = image[i,j]
        image[i,j] = y[k]
        
    return image

# Calling the function
   img  = hist_norm(img)
   y = []
   s = img.shape
   h = np.zeros(shape = (256,1))
   for i in range(s[0]):
     for j in range(s[1]):
        k = img[i,j]
        h[k] = h[k]+1
   plt.plot(h) # Ploting the Histgram of the output image after we perfom the histogram
   plt.show()

   
# Storing the images in the folder
   path = "C:/Users/nisar/Desktop/Projects/Histogram_images"
   cv2.imwrite(os.path.join(path , f'{value}.jpg'), img)
   value = value+1



