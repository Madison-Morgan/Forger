#!/usr/bin/env python
# coding: utf-8

# # FORGER

# ## This is a notebook about General Adversarial Networks. Data from Kaggle : https://www.kaggle.com/ikarus777/best-artworks-of-all-time. 

# In[6]:


import keras
from keras.optimizers import Adam
from keras.layers import Dense, Dropout, Input
from keras.models import Model,Sequential
from tqdm import tqdm
from keras.layers.advanced_activations import LeakyReLU

from random import shuffle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os


DATA_PATH = "D:/Mady/Data/130081_310927_bundle_archive/images/images/Vincent_van_Gogh/Vincent_van_Gogh_706.jpg"
RESIZED_PATH = "D:/Mady/Data/130081_310927_bundle_archive/resized/resized/"
OUTPUT_PATH = "D:/Mady/Data/130081_310927_bundle_archive/artists.csv"


# # GETTING THE DATA

# In[2]:


def preprocess_image(img, NEW_WIDTH=256, NEW_HEIGHT=256):
    loaded_img = load_img(img)
    resized_img = loaded_img.resize((NEW_WIDTH, NEW_HEIGHT))
    new_img = np.array(img_to_array(resized_img))
    return new_img


# In[3]:


from keras.datasets import mnist

(x_train, y_train), (x_test,y_test) = mnist.load_data()
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

plt.figure()
plt.title("Random e.g.", fontsize=20)
plt.imshow(x_train[2].astype(int))
print(y_train[2])


# # VISUALIZING THE DATA

# In[4]:


from keras.preprocessing.image import load_img, save_img, img_to_array

##im1= np.array(img_to_array(load_img(DATA_PATH)))
##im2= np.array(img_to_array(load_img(RESIZED_PATH)))


#im1b= np.array(img_to_array(load_img("D:/Mady/Data/130081_310927_bundle_archive/images/images/Vincent_van_Gogh/Vincent_van_Gogh_708.jpg")))
im2b= "D:/Mady/Data/130081_310927_bundle_archive/resized/resized/Vincent_van_Gogh_707.jpg"


im2b = preprocess_image(im2b, 300, 300)
print(im2b.shape)

plt.figure(1)
plt.title("Random e.g.", fontsize=20)
plt.imshow(im2b.astype(int))



im2b = im2b/256
print(im2b.shape)
im2b.flatten()
print(im2b.shape)
plt.figure(2)
plt.title("Random e.g.2", fontsize=20)
plt.imshow(im2b.astype(int))


# In[7]:



output_df = pd.read_csv(OUTPUT_PATH)
paintings=[]
outputs=[]
gallery= os.listdir(RESIZED_PATH)
for painting in gallery:
    paintings.append(preprocess_image(RESIZED_PATH+painting))
    outputs.append(output_df[output_df['name'].str.contains(painting.split("_")[0])]["id"].iloc[0])

print(len(paintings))
print(len(outputs))

zipped = list(zip(paintings,outputs))
shuffle(zipped)
paintings, outputs = zip(*zipped)
paintings = np.stack(paintings)


# In[9]:



print(paintings.shape)
print(len(paintings))
paintings = paintings.reshape(8683, 65536, 3 )
print(paintings.shape)


# In[10]:


NUM_PAINTINGS = len(paintings)
x_train, x_test = paintings[:round(NUM_PAINTINGS*0.9)], paintings[round(NUM_PAINTINGS*0.9):]
y_train, y_test = outputs[:round(NUM_PAINTINGS*0.9)], outputs[round(NUM_PAINTINGS*0.9):]


# In[11]:


print(len(x_train))
print(len(x_test))
print(len(y_train))
print(len(y_test))


# In[12]:


print(x_train.shape)


# # BUILDING THE MODEL

# In[ ]:


def adam_optimizer():
    return Adam(lr=0.0002, beta_1=0.5)


# In[ ]:





# In[ ]:





# # TRAINING THE MODEL

# In[ ]:





# In[ ]:





# In[ ]:





# # VISUALIZING THE RESULTS

# In[ ]:




