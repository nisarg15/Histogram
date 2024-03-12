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
    v_images1 = []
    v_images2 = []
    v_images3 = []

    image = cv2.imread(img)
# Convert the image to grayscale  
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(image,(800,600))
    
# Function to perfrom the histogram    
    def hist_norm(image):
        
    
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
    
# Cropping the image into several parts to perfrom the histogram   
    s = img.shape
    i = 0
    c = 0
    
    x = s[0]/100
    
    while i<=x:
     img_crop = img[0:100,0+c:100+c] 
     new_image = hist_norm(img_crop)
     v_images1.append(new_image)
     i = i+1
     c = c+100
    
    i = 0
    c = 0
    while i<=x:
     img_crop = img[100:400,0+c:100+c] 
     new_image = hist_norm(img_crop)
     v_images2.append(new_image)
     i = i+1
     c = c+100
    
    i = 0
    c = 0
    while i<=x:
     img_crop = img[400:600,0+c:100+c] 
     new_image = hist_norm(img_crop)
     v_images3.append(img_crop)
     i = i+1
     c = c+100
    
      
    v1 = np.concatenate(v_images1, axis=1)
    v2 = np.concatenate(v_images2, axis=1)
    v3 = np.concatenate(v_images3, axis=1)
    
# Adding all the parts of the images into one to get the final result 
    final = np.concatenate((v1,v2,v3), axis=0)
    y = []
    s = final.shape
    
    h = np.zeros(shape = (256,1))
    for i in range(s[0]):
        for j in range(s[1]):
            k = final[i,j]
            h[k] = h[k]+1
    plt.plot(h) # Ploting the Histgram of the output image after we perfom the histogram
    plt.show()
            
# Storing the images in the folder   
    path = "C:/Users/nisar/Desktop/Projects/adaptive_histogram_images"
    cv2.imwrite(os.path.join(path , f'{value}.jpg'), final)
    value = value+1
    del  v_images1
    del  v_images2
    del  v_images3


                
    
    
