import streamlit as st
import numpy as np
from ultralytics import YOLO
from PIL import Image
import time

# Set page config
st.set_page_config(
        page_title="FreshEye: Smart Produce Inspection",
        page_icon=":apple:",
        layout="wide",
        initial_sidebar_state="expanded"
    )

def model_prediction(test_image):
    model = YOLO('best.pt')
    results = model(test_image)

    probs = results[0].probs.data.tolist()
    result_index = np.argmax(probs)
    return result_index

#Sidebar
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page",["Home","About Project"])

# Reading Labels
labels = []

with open('labels.txt') as f:
    content = f.readlines()

for i in content:
    labels.append(i[:-1])

#About Project
if(app_mode=="About Project"):
    st.header("FreshEye: Smart Produce Inspection")
    st.write("This web application predicts the freshness of fruits and vegetables using computer vision.")
    st.subheader("Use cases:")
    st.write("**Grocery Store Quality Control:** This computer vision model can be used by grocery stores and supermarkets to automatically check and sort fruits and vegetables according to their freshness levels. It can help to separate fresh produce from rotten ones, ensuring that consumers always have access to high quality produce. Supply Chain Management: The model can help suppliers to visually inspect produce before it is sent to grocery stores or food processing companies. This will increase efficiency of the supply chain and reduce food waste.")
    
    st.write("**Food Processing Plants:** Food processing plants, particularly those that deal with canned fruits and vegetables, can use this model to sort and categorize fresh and rotten produce for different processing stages.")

    st.write("**Agricultural Quality Inspection:** Farms can employ this computer vision model to detect rotten fruits and vegetables right on the fields, allowing them to improve their quality control measures and avoid transporting rotten produce.")

    st.write("**Personal Use - Health & Wellness Apps:** This model can be incorporated into mobile applications, helping individuals at home to identify if their fruits and vegetables are still fresh and good for consumption. This will help people to consume healthier food and reduce food waste at home.")

    st.subheader("Dataset Information:")
    st.text("This dataset contains images of the following food items:")
    st.code("fruits- apple, banana, mango, orange")
    st.code("vegetables- pepper, potato, carrot,  cucumber")
    st.text("This dataset contains three folders:")
    st.text("1. train (6489 images each)")
    st.text("2. test (945 images each)")
    st.text("3. validation (1849 images each)")

#Main Page
elif(app_mode=="Home"):
    st.header("FreshEye: Smart Produce Inspection")
    st.write("Upload an image of a fruit or vegetable to predict its freshness.")
    image_path = "C:/Users/SSD/OneDrive/Desktop/streamlit/images/home1.jpg"
    st.image(image_path)

    test_image = st.file_uploader("Choose an Image:")
    
    if test_image is not None:
        image = Image.open(test_image)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        # ML = st.selectbox("Use",["Yolov8","CNN"])

        # if(ML == "Yolov8"):
        with st.spinner("Predicting..."):
            # time.sleep(5)
            result_index = model_prediction(image)
            st.success(labels[result_index])
            
        # elif(ML == "CNN"):
        #     result_index = preprocess_image(test_image)
        #     st.success(f"Model is Predicting : {labels[result_index]}")



