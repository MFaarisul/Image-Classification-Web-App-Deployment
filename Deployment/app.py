import numpy as np
import time
import cv2
import tensorflow as tf
import streamlit as st
from PIL import Image
from keras.preprocessing import image
from streamlit_drawable_canvas import st_canvas

animal_model = tf.keras.models.load_model('mobilenet_final.h5')
with open('label.txt', 'r') as f:
    labels = f.read().split('\n')

number_model = tf.keras.models.load_model('number_model.h5')

def animalmodel(preview):
    with st.expander("See the labels"):
        st.markdown("You can check all the image labels on this [link](#https://github.com/MFaarisul/Tensorflow-imagedatagenerator-for-image-classification/blob/master/label.txt)", unsafe_allow_html=True)
    
    img = st.file_uploader('Upload an image')

    if img is not None:
        if preview == 'Yes':
            st.image(img)
    
        x = Image.open(img)
        x = x.resize((150,150))
        x = image.img_to_array(x)
        x /= 255
        scaled_img = np.expand_dims(x, axis=0)

        label = animal_model.predict(scaled_img)
        idx = np.argmax(label, axis=1)[0]

        st.markdown(
            """
            <style>
                .stProgress > div > div > div > div {
                    background-color: green;
                }
            </style>""",
            unsafe_allow_html=True,
        )

        my_bar = st.progress(0)
        for percent_complete in range(100):
            time.sleep(0.005)
            my_bar.progress(percent_complete + 1)
        
        my_bar.empty()

        st.info('Animal: {}  \nProbability: {:.2f}'.format(labels[idx].capitalize(), label[0][idx]))

def numbermodel(s_color, s_width): 
    st.write('Draw a number (0-9)')

    canvas_result = st_canvas(
    stroke_width=s_width,
    stroke_color=s_color,
    background_color='#212529',
    update_streamlit=True,
    height=200,
    width=200,
    key="canvas",
)

    img = canvas_result.image_data
    
    # Do something interesting with the image data and paths
    if st.button('Analyze') and img is not None:
        x = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        x = cv2.resize(x, (28,28), interpolation=cv2.INTER_NEAREST)
        x = image.img_to_array(x)
        x /= 255
        scaled_img = np.expand_dims(x, axis=0)

        pred = number_model.predict(scaled_img)
        pred_label = np.argmax(pred)

        st.markdown(
            """
            <style>
                .stProgress > div > div > div > div {
                    background-color: green;
                }
            </style>""",
            unsafe_allow_html=True,
        )

        my_bar = st.progress(0)
        for percent_complete in range(100):
            time.sleep(0.005)
            my_bar.progress(percent_complete + 1)
        
        my_bar.empty()

        st.info('Result: {}'.format(pred_label))
    else:
        st.error('Please draw a number')

def main():
    model = st.sidebar.selectbox('Select One', ['Animal', 'Number'])

    if model == 'Number':
        stroke_width = st.sidebar.slider('Stroke Width', 15, 25)
        stroke_color = st.sidebar.color_picker('Stroke Color', '#dee2e6')
    else:
        preview = st.sidebar.radio('Preview Image', ['Yes', 'No'])

    html_temp = '''
    <h1 style="font-family: Trebuchet MS; padding: 12px; font-size: 48px; color: #f77f00; text-align: center;
    line-height: 1.25;">Image Classification<br>
    <span style="color: #fcbf49; font-size: 20px"><b>Animals and Numbers Classification</b></span><br>
    </h1>
    <hr>
    '''
    st.markdown(html_temp, unsafe_allow_html=True)

    if model == 'Number':
        numbermodel(stroke_color, stroke_width)
    else:
        animalmodel(preview)

if __name__ == '__main__':
    main()