# Malaria-Detection

Part of the Deep Learning and Convolutional Neural Network Subject final project.

The app determines if a human blood sample is infected with malaria parasites or not. We will analyse the images and build models to classify two classes, namely, infected and uninfected blood smears using several object detection architectures. Faster RCNN, SSD and RFCN models have been built and tested, to explore the results and choose the model that achieved the highest level of accuracy. The chosen model was then used to detect malaria in real-time, by connecting it to our application’s GUI. This application will work by a technique that includes a camera connected to our system, whereby images or video samples of red blood cells will be captured and sent to our system for analysis

# 1. Dataset
The dataset can be publicly found on Kaggle via this link https://www.kaggle.com/iarunava/cell-images-for-detecting-malaria.  The original dataset consisted of 27,558 images of both infected and uninfected blood smears captured through a webcam. There were two sets of cells: infected and healthy smears; each consists of 13,799 images. However, this dataset was originally used for image classification purposes. As this project is focusing on object detection, particularly of malaria parasites, only the images of the infected cells were used to train the model. The updated dataset can be found in this repository.

![](Images/cell1.PNG)

# 2. Overview of the architecture/system
Three approaches to the CNN architecture that are among the most effective of the object detection algorithms has explored: Faster RCNN, SSD, and RFCN. For each architecture, the same training dataset and testing dataset was used but customize the base structure to the problem at hand. then, the mAP of each result from the approaches were compared to find the most accurate architecture design for each approach. Afterward, the frozen inference graphs of these three models were used in our application and compared the detection time. Overall, the three models were judged based on the accuracy level and reaction time. The final choice is RFCN.

# 3. GUI Design

GUI workflow:



the graph above shows how our malaria cell detection system works. The tasks of the two parts explained before, which are GUI and object detection, can clearly be seen. All the decision processes happen in the GUI, as the GUI will ask the users for their command. In the object detection, the processes that happen is detecting the given input from users and send the result to GUI. Then, the GUI will be displaying the detection result to users.

Screenshots of the actual GUI:


# Result

