import tkinter as tk
from tkinter import filedialog, Tk, Toplevel, Label, Button
from tkinter import messagebox
from cv2 import cv2
from PIL import Image, ImageTk
import tensorflow as tf
import os
import sys
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from collections import defaultdict
from io import StringIO
sys.path.append("..")
from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

window = Tk()
window.title("Malaria Cell Detection App")
window.geometry('350x220')

def camera():
    messagebox.showinfo("Info","Opening the webcam...")
    
    messagebox.showinfo("Info","Press 'q' to close the webcam")

    PATH_TO_CKPT = 'frozen_inference_graph.pb'

    # Load the Tensorflow model into memory.
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.compat.v1.GraphDef()
        with tf.compat.v1.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

        sess = tf.compat.v1.Session(graph=detection_graph)

    # Input tensor is the image
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

    # Output tensors are the detection boxes, scores, and classes
    detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
    detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
    detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

    # Number of objects detected
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')

    # Initialize webcam feed
    video = cv2.VideoCapture(0)
    video.set(cv2.CAP_PROP_FRAME_WIDTH,1280) #(frame width)
    video.set(cv2.CAP_PROP_FRAME_HEIGHT,720)  #(frame heigt)
    #video.set(cv2.CAP_PROP_FPS,60)
    
    while(True):

        # Acquire frame and expand frame dimensions to have shape: [1, None, None, 3]
        _, frame = video.read()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_expanded = np.expand_dims(frame_rgb, axis=0)

        # Perform the actual detection by running the model with the image as input
        (boxes, scores, classes, num) = sess.run(
            [detection_boxes, detection_scores, detection_classes, num_detections],
            feed_dict={image_tensor: frame_expanded})

        # Visulaize the results
        vis_util.visualize_boxes_and_labels_on_image_array(
            frame,
            np.squeeze(boxes),
            np.squeeze(classes).astype(np.int32),
            np.squeeze(scores),
            use_normalized_coordinates=True,
            line_thickness=8,
            min_score_thresh=0.60)

        # All the results have been drawn on the frame, so it's time to display it.
        cv2.imshow('Object detector', frame)
    
    # Press 'q' to quit
        if cv2.waitKey(120) == ord('q'):
            break

# Clean up
    video.release()
    cv2.destroyAllWindows()


def openImg(filename):
    # Open the image using OPENCV
    img = cv2.imread(filename)
    img = cv2.resize(img,(150, 150))
    cv2.imshow("Image", img)
    
def I_process_button(file):
    messagebox.showinfo("Info","The image is being processed")

    PATH_TO_CKPT = 'frozen_inference_graph.pb'

    PATH_TO_IMAGE = file

    # Load the Tensorflow model into memory.
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.compat.v1.GraphDef()
        with tf.compat.v1.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

        sess = tf.compat.v1.Session(graph=detection_graph)

# Input tensor is the image
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

# Output tensors are the detection boxes, scores, and classes
    detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
    detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
    detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

# Number of objects detected
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')

# Load image using OpenCV
    image = cv2.imread(PATH_TO_IMAGE)
    image = cv2.resize(image,(450, 450))
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_expanded = np.expand_dims(image_rgb, axis=0)

# Perform the actual detection by running the model with the image as input
    (boxes, scores, classes, num) = sess.run(
        [detection_boxes, detection_scores, detection_classes, num_detections],
        feed_dict={image_tensor: image_expanded})

# Draw the results of the detection (aka 'visulaize the results')
    vis_util.visualize_boxes_and_labels_on_image_array(
        image,
        np.squeeze(boxes),
        np.squeeze(classes).astype(np.int32),
        np.squeeze(scores),
        use_normalized_coordinates=True,
        line_thickness=8,
        min_score_thresh=0.60)

# All the results have been drawn on image. Now display the image.
    cv2.imshow('Malaria Cell Detection Result', image)

    D_window = Toplevel(window)
    D_window.title("Image")
    D_window.geometry('350x150')

    Detect_button = Button(D_window, text="Detect another file", height=2, width=40, bg="#FFE5CC", command= lambda: [detection_window(), D_window.destroy(), cv2.destroyAllWindows()])
    Detect_button.place(x=175, y=40, anchor="center")

    Finished_button = Button(D_window, text= "Finish", height=2, width=40, bg="#FFE5CC", command= window.destroy)
    Finished_button.place(x=175, y=100, anchor="center")

def V_process_button(file):
    messagebox.showinfo("Info","The video is being processed")
    
    messagebox.showinfo("Info","Press 'q' to close the video")

    PATH_TO_CKPT = 'frozen_inference_graph.pb'

# Path to image
    PATH_TO_VIDEO = file

    # Load the Tensorflow model into memory.
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.compat.v1.GraphDef()
        with tf.compat.v1.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

        sess = tf.compat.v1.Session(graph=detection_graph)

# Input tensor is the image
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

# Output tensors are the detection boxes, scores, and classes
    detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
    detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
    detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

# Number of objects detected
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')

# Load image using OpenCV
    video = cv2.VideoCapture(PATH_TO_VIDEO)

    while(video.isOpened()):

    # Acquire frame and expand frame dimensions to have shape: [1, None, None, 3]
    # i.e. a single-column array, where each item in the column has the pixel RGB value
        _, frame = video.read()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_expanded = np.expand_dims(frame_rgb, axis=0)

    # Perform the actual detection by running the model with the image as input
        (boxes, scores, classes, num) = sess.run(
            [detection_boxes, detection_scores, detection_classes, num_detections],
            feed_dict={image_tensor: frame_expanded})

    # Draw the results of the detection (aka 'visulaize the results')
        vis_util.visualize_boxes_and_labels_on_image_array(
            frame,
            np.squeeze(boxes),
            np.squeeze(classes).astype(np.int32),
            np.squeeze(scores),
            use_normalized_coordinates=True,
            line_thickness=8,
            min_score_thresh=0.60)

# All the results have been drawn on image. Now display the image.
        cv2.imshow('Malaria Cell Detection Result', frame)

        # Press 'q' to quit
        if cv2.waitKey(1) == ord('q'):
            break

# Clean up
    video.release()
    cv2.destroyAllWindows()

    D_window = Toplevel(window)
    D_window.title("Video")
    D_window.geometry('350x150')

    Detect_button = Button(D_window, text="Detect another file", height=2, width=40, bg="#FFE5CC", command= lambda: [detection_window(), D_window.destroy()])
    Detect_button.place(x=175, y=40, anchor="center")

    Finished_button = Button(D_window, text= "Finish", height=2, width=40, bg="#FFE5CC", command= window.destroy)
    Finished_button.place(x=175, y=100, anchor="center")

def BttnImage_Clicked():
    # Use the File Dialog component to Open the Dialog box to select files
    file = filedialog.askopenfilename(filetypes = (("Images files","*.png"),("all files","*.*")))
    messagebox.showinfo("File Selected", file)
    openImg(file) # Passing the file to openImg method to show is using opencv (imread, imshow)

    O_window = Toplevel(window)
    O_window.title("Image")
    O_window.geometry('350x200')

    Procced_button = Button(O_window, text= "Procced", height=2, width=40, bg="#FFE5CC", command= lambda: [ O_window.destroy(),I_process_button(file)])
    Procced_button.place(x=175, y=80, anchor="center")

def BttnVideo_Clicked():
    # Use the File Dialog component to Open the Dialog box to select files
    file = filedialog.askopenfilename(filetypes = (("Video Files","*.mp4"),("all files","*.*")))
    messagebox.showinfo("File Selected", file)

    V_window = Toplevel(window)
    V_window.title("Video")
    V_window.geometry('350x200')

    V_Procced_button = Button(V_window, text= "Procced", height=2, width=40, bg="#FFE5CC", command= lambda: [ V_window.destroy(),V_process_button(file)])
    V_Procced_button.place(x=175, y=80, anchor="center")

def detection_window():
    C_window = Toplevel(window)
    C_window.title("Malaria Cell Detection App")
    C_window.geometry('350x260')

    Info_Label = Label(C_window, text = "Upload the image of cell, video of cells, or Take a picture of cell")
    Info_Label.place(x=175, y=25, anchor="center") # Adding the Label

    button_capture = Button(C_window, text= "Camera", height=2, width=40, bg="#FFE5CC", command= lambda: [camera(), C_window.destroy()])
    button_capture.place(x=175, y=80, anchor="center")

    button_image = Button(C_window, text= "Image", height=2, width=40, bg="#FFE5CC", command= lambda: [BttnImage_Clicked(),C_window.destroy()])
    button_image.place(x=175, y=140, anchor="center")

    button_video = Button(C_window, text= "Video", height=2, width=40, bg="#FFE5CC", command= lambda: [BttnVideo_Clicked(),C_window.destroy()])
    button_video.place(x=175, y=200, anchor="center")


def info_window():
    i_window = Toplevel(window)
    i_window.title("Help")
    i_window.geometry('350x220')
    i_Label = Label(i_window, text = "This app is a part of assignment from 42028 Deep Learning & Convolutional Neural Netwrok subject. \n \nThis app will detect the malaria parasite inside the cell. You can start the classification by clicking the START button.", 
                    wraplength=350, justify=LEFT)
    i_Label_Mem = Label(i_window, text = "\n Creators: \n Radwa Seleim (13508989) \n Thi Thu Ha Le (13467319) \n Alifia C Harmadi (13302447)")
    i_Label.grid(column=0, row=0)
    i_Label_Mem.grid(column=0, row=1)

TitleLabel = Label(text = "Malaria Cell Identification", font=("Calibri Bold", 20), bg="yellow")
StartBtn = Button(text="Start",height=2, width=40, bg="#FFE5CC", command= detection_window)
HelpBtn = Button(text="Help",height=2, width=40, bg="#FFE5CC", command=info_window)
ExitBtn = Button(text="Exit",height=2, width=40, bg="#FFE5CC", command=window.destroy)


TitleLabel.place(x=175, y=25, anchor="center") # Adding the Label
StartBtn.place(x=175, y=95, anchor="center")
HelpBtn.place(x=175, y=135, anchor="center")
ExitBtn.place(x=175, y=175, anchor="center")
window.mainloop()