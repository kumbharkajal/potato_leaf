import streamlit as st
import tensorflow as tf
import numpy as np
import gdown
import os

file_id = "1levibfjxs6J67A3AwKO9SXkpoKCV48CR"
url = 'https://drive.google.com/file/d/1levibfjxs6J67A3AwKO9SXkpoKCV48CR/view?usp=sharing'
model_path="trained_plat_disease_model.keras"

if not os.path.exists(model_path):
    st.warning("Downloading model from Google drive...")
    gdown.download(url, model_path, quiet=False)

def model_prediction(test_image):
    model=tf.keras.models.load_model("trained_plant_disease_model.keras")
    image=tf.keras.preprocessing.image.load_img(test_image, target_size=(128, 128))
    input_arr=tf.keras.preprocessing.image.img_to_array(image)
    input_arr=np.array([input_arr])
    predictions=model.predict(input_arr)
    return np.argmax(predictions)

st.sidebar.title("Plant Disease System for Sustainable Agriculture")
app_mode=st.sidebar.selectbox('Select Page', ['Home', 'Disease Recognition'])

from PIL import Image 
img=Image.open('disease.jpg')
st.image(img)

if(app_mode=='Home'):
    st.markdown("<h1 style='text-align:center;'>PLant Disease Detection System for Sustainable Agriculture", unsafe_allow_html=True)

elif(app_mode=='Disease Recognition'):
    st.header('Plant Disease Detection System for Sustainable Agriculture')

test_image=st.file_uploader('Choode an image:')
if(st.button('Show image')):
    st.image(test_image, width=4, use_column_width=True)

if(st.button('Predict')):
    st.snow()
    st.write('our prediction')
    result_index=model_prediction(test_image)
    class_name=['Potato__Early_blight', 'Potato__Late_blight', 'Potato__healthy']
    st.success('Model is predicting its a {}'.format(class_name[result_index]))