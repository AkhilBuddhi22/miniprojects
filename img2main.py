import streamlit as st
from PIL import Image
from io import BytesIO, BufferedReader
import cv2
import numpy as np
#some script working regarding the web application
st.title("IMAGE TO CARTOON CONVERTOR")
st.header("Hello there, Upload your fav pics and convert them into amazing cartoon characters!")
st.info("File type should be: PNG, JPG, JPEG")
# creating the function to take the input of the image
def take_file():
    return st.file_uploader("", type=["png", "jpg", "jpeg"])
# defining the funcion for the edge masking method
def edge_mask(img, line_size, blur_value):
    """
    input: Input Image
    output: Edges of Images
    """
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray_blur = cv2.medianBlur(gray, blur_value)
    edges = cv2.adaptiveThreshold(gray_blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, line_size, blur_value)
    return edges
#defining the funcion for the color quntization technique
def color_quantization(img, k):
    # Transform the image
    data = np.float32(img).reshape((-1, 3))
    # Determining the criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.001)
    # Implementing K-means
    ret, label, center = cv2.kmeans(data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    center = np.uint8(center)
    result = center[label.flatten()]
    result = result.reshape(img.shape)
    return result
#defining the cartoon function for cartoonizing the user's given image file
def cartoon(img):
    IMG = img
    Image_data = Image.open(IMG)
    st.info("Uploaded Image")
    st.image(Image_data)
    image = Image.open(img)
    img_array = np.array(image)
    img = img_array
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Edge masking
    line_size, blur_value = 7, 7
    edges = edge_mask(img, line_size, blur_value)
    # Applying color quantization
    img_quantized = color_quantization(img, k=9)
    # Applying bilateral filter
    blurred = cv2.bilateralFilter(img_quantized, d=3, sigmaColor=220, sigmaSpace=220)
    # Creating cartoon effect
    cartoon = cv2.bitwise_and(blurred, blurred, mask=edges)
    st.info("Cartoon Preview")
    return cartoon
# Giving the input of the image
image_file = take_file()
if image_file is not None:
    st.success("Uploaded!!!")
    cartoon = cartoon(image_file)
    cartoon = cv2.cvtColor(cartoon, cv2.COLOR_BGR2RGB)
    st.image(cartoon)
    im_rgb = cartoon[:, :, [2, 1, 0]]  # numpy.ndarray
    ret, img_enco = cv2.imencode(".png", im_rgb)  # numpy.ndarray
    srt_enco = img_enco.tobytes()  # bytes
    img_BytesIO = BytesIO(srt_enco)  # _io.BytesIO
    img_BufferedReader = BufferedReader(img_BytesIO)  # _io.BufferedReader
    st.download_button(
        label="Download",
        data=img_BufferedReader,
        file_name="Cartoon.png",
        mime="image/png"
    )
else:
    st.error("Try Again !!! - PLEASE UPLOAD THE JPG PNG JPEG!!")
