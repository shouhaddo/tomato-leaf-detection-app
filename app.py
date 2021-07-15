import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt
import tensorflow_hub as hub
import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras.models import load_model
from tensorflow.keras import preprocessing
import time
fig = plt.figure()

with open("custom.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.title('Tomato Leaf desease Classifier')

st.markdown("Welcome to this simple web application that classifies tomato leaf desease ")


def main():
    file_uploaded = st.file_uploader("Choose File", type=["png","jpg","jpeg"])
    class_btn = st.button("Classify")
    if file_uploaded is not None:    
        image = Image.open(file_uploaded)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        
    if class_btn:
        if file_uploaded is None:
            st.write("Invalid command, please upload an image")
        else:
            with st.spinner('Model working....'):
                plt.imshow(image)
                plt.axis("off")
                predictions = predict(image)
                time.sleep(1)
                st.success('Classified')
                st.write(predictions)
                st.pyplot(fig)


def predict(image):
    classifier_model = "model_VGG19.h5"
    IMAGE_SHAPE = (224, 224,3)
    model = load_model(classifier_model, compile=False, custom_objects={'KerasLayer': hub.KerasLayer})
    test_image = image.resize((224,224))
    test_image = preprocessing.image.img_to_array(test_image)
    test_image = test_image / 255.0
    test_image = np.expand_dims(test_image, axis=0)
    class_names = [
          'Tomato___Bacterial_spot',
          'Tomato___Early_Blight',
          'Tomato___Late Blight', 
          'Tomato___Leaf Mold', 
          'Tomato___Septoria Leaf Spot',
          'Tomato___Spider mites' ,
          'Tomato___Target Spot' ,
          'Tomato___Yellow Leaf Curl Virus' ,
          'Tomato___Mosaic Virus' ,
          'Tomato___Healthy']
    predictions = model.predict(test_image)
    scores = tf.nn.softmax(predictions[0])
    scores = scores.numpy()
    results = {
          'Tomato___Bacterial_spot': 0,
          'Tomato___Early_Blight': 0,
          'Tomato___Late Blight': 0, 
          'Tomato___Leaf Mold': 0, 
          'Tomato___Septoria Leaf Spot': 0 ,
          'Tomato___Spider mites': 0 ,
          'Tomato___Target Spot': 0,
          'Tomato___Yellow Leaf Curl Virus': 0 ,
          'Tomato___Mosaic Virus': 0 ,
          'Tomato___Healthy': 0 ,
}

    
    result = f"{class_names[np.argmax(scores)]} with a { (100 * np.max(scores)).round(2) } % confidence." 
    return result









    

if __name__ == "__main__":
    main()


