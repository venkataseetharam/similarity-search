#!/usr/bin/env python
# coding: utf-8

# In[3]:


import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from PIL import Image
import cv2
from sklearn.neighbors import NearestNeighbors

# Load pre-trained ResNet-50 model
model = ResNet50(weights='imagenet', include_top=False, pooling='avg')

# Define preprocessing function
def preprocess(image):
    x = preprocess_input(image)
    return x

# Load features dictionary
features_dict = np.load('features_dict_final.npy', allow_pickle=True).item()

# Convert feature vectors dictionary to numpy array
features_array = np.array(list(features_dict.values()))

# Train k-NN model
k = 10  # Number of neighbors to retrieve
nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(features_array)

# Define function to retrieve k-NN for a query image
def get_similar_images(query_image):
    query_feature = model.predict(np.expand_dims(preprocess(query_image), axis=0)).squeeze()
    distances, indices = nbrs.kneighbors(np.array([query_feature]))
    similar_images = []
    for idx in indices.squeeze():
        similar_images.append(list(features_dict.keys())[idx])
    return similar_images

# Define Streamlit app
def app():
    st.title("Image Similarity Search System")

    # Create option to select image from test dataset
    test_images = list(features_dict.keys())
    test_image = st.selectbox("Select an image from the test dataset", test_images)

    # Display selected image
    st.image(test_image)

    # Display similar images
    st.write("Similar Images:")
    similar_images = get_similar_images(cv2.imread(test_image))
    for image_path in similar_images:
        image = Image.open(image_path)
        st.image(image)

    # Create option to upload an image
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image)

        # Display similar images
        st.write("Similar Images:")
        similar_images = get_similar_images(np.array(image))
        for image_path in similar_images:
            image = Image.open(image_path)
            st.image(image)

if __name__ == '__main__':
    app()


# In[ ]:




