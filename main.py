import streamlit as st
from PIL import Image
import numpy as np

from roboflow import Roboflow

## INITIALIZATION
# connect to roboflow with an api key
rf = Roboflow(api_key="")

# reference the project on universe which contains our inference model
project = rf.workspace("team-roboflow").project("rock-paper-scissors-detection")

# reference the specific model we want to use for inference
model = project.version(34).model

# reference upload project destination
rf2 = Roboflow(api_key="BLlkFnwfSaRFUXBfU0tJ")
upload_project = rf2.workspace().project("live-rock-paper-scissors")

print("Reference inference point: ", project)


## DEFINITION
img_file_buffer = st.camera_input("Take a picture")


# MAIN
if img_file_buffer is not None:
    # To read image file buffer as a PIL Image:
    img = Image.open(img_file_buffer)
    img.save("live_test.png")

predictions = model.predict('live_test.png')

# button to perform inferences on captured images
if st.button('Detect handsign') and predictions:
    # perform inference via model

    # write predictions in GUI
    st.write(predictions.json()['predictions'])
    st.write(predictions.json()['predictions'][0]['class'])
else:
    st.write('No image provided')

# button to report / upload images marked as incorrect
if st.button('Report incorrect predictions') and predictions:
    # upload reported image direct to roboflow
    upload_project.upload('live_test.png')
else:
    st.write('No image provided')

