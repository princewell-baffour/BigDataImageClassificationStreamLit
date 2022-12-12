from fastai.vision.widgets import *
from fastai.vision.all import *
from fastbook import *
from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
from io import BytesIO, StringIO
import streamlit as st
from streamlit_option_menu import option_menu
import pathlib
plt = platform.system()
if plt == 'Linux': pathlib.WindowsPath = pathlib.PosixPath

import requests
from io import BytesIO

st.set_page_config(
    page_title="Image Classification",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded"
)

def navigation():
    #1. as slidebar menu
    with st.sidebar:
        selected = option_menu(
            menu_title= "Big Data",
            options = ["Classifier", "EDA", "Google Teachable Machine","New Image Trainer"],
            icons=['upload', 'graph-down'],
            menu_icon="cast", default_index=0
        )

    if selected == "Classifier":
        main_app()

    if selected == "EDA":
        eda()

    # if selected == "Training":
    #     training() 
    
    if selected == "Google Teachable Machine":
        googlemachine()

    if selected == "New Image Trainer":
        fastai_training()


def main_app():
    st.header('Image Classification')
    st.subheader('Model trained with Fastai')

    
    plt = platform.system()
    if plt == 'Windows': pathlib.PosixPath = pathlib.WindowsPath
    res_model = load_learner(pathlib.Path()/'cats.pkl')

    uploaded_file = st.file_uploader("Upload Files",type=['png','jpeg', 'jpg'])
    col1,col2 = st.columns(2)
    with col1:
        display_image = st.empty()
        if not uploaded_file:
            return  display_image.info("Choose a file to upload, only type: png, jpg, jpeg ")
        #return None
        else:
            uploaded_file = PILImage.create((uploaded_file))
            display_image.image(uploaded_file.to_thumb(500,500), caption='Image Upload')

    
        pred, pred_idx, probs = res_model.predict(uploaded_file)
    with col2:
        st.success(f'Prediction: {pred} ')
        st.info(f'Probability: {probs[pred_idx]:.04f}')
    
    
    uploaded_file.close()

def eda():
    st.header('Exploratory data analysis')

    data_path = 'cats/'
    img_list = []
    labels = []

    # Fill in the labels & img_list lists
    for class_name in os.listdir(data_path):
        if class_name not in labels:
            labels.append(class_name)
        img_dir = data_path + class_name + "/"
        for img_filename in os.listdir(img_dir):
            img_path = img_dir + img_filename
            img_list.append([img_path, class_name])

    # Filter on the label part of the sublist (eg: [[img_path, label], ...]
    def get_filtered_list(filter: str, list: list = img_list):
        return [x[0] for x in list if x[1] == filter]

    # Calculate the average resolution for a given list of images
    def get_average_img_resolution(images: list):
        widths = []
        heights = []

        for img in images:
            im = Image.open(img)
            widths.append(im.size[0])
            heights.append(im.size[1])

        avg_width = round(sum(widths) / len(widths))
        avg_height = round(sum(heights) / len(heights))
        return [avg_width, avg_height]

    # --- Tabs ---
    tab1, tab2, tab3, tab4, tab5 = st.tabs(labels)

    # A tab for each of the labels
    for index, tab in enumerate([tab1, tab2, tab3, tab4, tab5]):
        with tab:
            images = get_filtered_list(labels[index])
            total_imgs = len(images)
            avg_w, avg_h = get_average_img_resolution(images)
            st.header(labels[index])
            st.write(f'Total Samples:  {total_imgs} ')
            st.write(f'Image Resolution: {avg_w}x{avg_h}')
            to_show = st.slider('Slide to adjust samples being displayed', 0, total_imgs,
                                30)
            st.image(images[:to_show], width=200)

# def training():
    
#     fns = get_image_files('./cats')

#     cats = DataBlock(
#         blocks=(ImageBlock, CategoryBlock), 
#         get_items=get_image_files, 
#         splitter=RandomSplitter(valid_pct=0.2, seed=1),
#         get_y=parent_label,
#         item_tfms=Resize(128))
        
#     dls = cats.dataloaders('./cats/')

#     cats = cats.new(
#         item_tfms=RandomResizedCrop(224, min_scale=0.5),
#         batch_tfms=aug_transforms())

#     dls = cats.dataloaders('./cats/')

#     resnet_adv = vision_learner(dls, resnet34, metrics=error_rate)
#     resnet_adv.fit_one_cycle(3, 3e-3)

def googlemachine():
    # Disable scientific notation for clarity
    np.set_printoptions(suppress=True)

    # Load the model
    model = load_model(pathlib.Path()/'keras_model.h5', compile=False)

    # Load the labels
    class_names = open(pathlib.Path()/'labels.txt', 'r').readlines()

    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

    google_file = st.file_uploader("Upload Files",type=['png','jpeg', 'jpg'])
    col1,col2 = st.columns(2)
    with col1:
        display_image = st.empty()
        if not google_file:
            return  display_image.info("Choose a file to upload, only type: png, jpg, jpeg ")
        #return None
        else:
            google_file = PILImage.create((google_file))
            display_image.image(google_file.to_thumb(500,500), caption='Image Upload')

        #pred, pred_idx, probs = res_model.predict(google_file)
    with col2:
        # Replace this with the path to your image
        image = google_file.convert('RGB')
        
        #resize the image to a 224x224
        size = (224, 224)
        image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

        #turn the image into a numpy array
        image_array = np.asarray(image)

        # Normalize the image
        normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1

        # Load the image into the array
        data[0] = normalized_image_array

        # run the inference
        prediction = model.predict(data)
        index = np.argmax(prediction)
        class_name = class_names[index]
        confidence_score = prediction[0][index]

        st.success(f'Prediction: {class_name} ')
        st.info(f'Probability: {confidence_score}')
    
    
    google_file.close()

def fastai_training():
    st.header('New Image Classifier Model')
    st.subheader('Powered by FastAi')

    fns = get_image_files('cats/')
    
    cats = DataBlock(
        blocks=(ImageBlock, CategoryBlock), 
        get_items=get_image_files, 
        splitter=RandomSplitter(valid_pct=0.2, seed=1),
        get_y=parent_label,
        item_tfms=Resize(128))
    
    dls = cats.dataloaders('./cats/')

    # image_file = dls.valid.show_batch(max_n=4, nrows=1)
    # img = Image.open(BytesIO(image_file.content))
    # st.write('Four random images from the DataLoaders')
    # image = Image.open(img)

    # st.image(image, caption='Four random images')

    # Random Resize and Augmentation
    cats = cats.new(
    item_tfms=RandomResizedCrop(224, min_scale=0.5),
    batch_tfms=aug_transforms())
    dls = cats.dataloaders('./cats/')

    col1, col2 = st.columns(2)
    col11, col12 = st.columns(2)
    cnn_arch = col1.selectbox('Select CNN Architecture', options=['resnet50','resnet34'], index = 0)
    no_epoch = col2.slider('What is your desired number of epochs', min_value=1, max_value=50, value=3, step=1)
    learning_rate = col11.slider('What is your desired learning rate (Value divided by 1000)', min_value= 1, max_value=100, value=30, step=10)/11000

    st.info(f'Calculated learning rate: {learning_rate}')
    
    
    if st.button('Train Model'):
        resnet_adv = vision_learner(dls, cnn_arch, metrics=error_rate)
        resnet_adv.fit_one_cycle(no_epoch, learning_rate)

        resnet_adv.unfreeze()
        resnet_adv.fit_one_cycle(1, 1e-5)

        resnet_adv.export('cats.pkl')
        with open('cats.pkl', 'rb') as f:
            st.download_button('Download Model', f, file_name='cats.pkl')  # Defaults to 'application/octet-stream'

    else:
        st.write('Click on button to start training')
    

if __name__=='__main__':
    navigation()
