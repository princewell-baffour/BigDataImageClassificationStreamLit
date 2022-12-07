from fastai.vision.widgets import *
from fastai.vision.all import *
from fastbook import *

from PIL import Image
from io import BytesIO, StringIO
import streamlit as st
from streamlit_option_menu import option_menu
import pathlib
plt = platform.system()
if plt == 'Linux': pathlib.WindowsPath = pathlib.PosixPath

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
            options = ["Classifier", "EDA"],
            icons=['upload', 'graph-down'],
            menu_icon="cast", default_index=0
        )

    if selected == "Classifier":
        main_app()

    if selected == "EDA":
        eda()


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




if __name__=='__main__':
    navigation()
