import streamlit as st
import tensorflow as tf
import cv2
categories = ['angry', 'disgusted', 'feared', 'happy', 'neutral', 'sad', 'surprised']
def findFace(pathForImage):
    image = cv2.imread(pathForImage)
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

    haarCascadeFile = "haarcascade_frontalface_default.xml"
    face = cv2.CascadeClassifier(haarCascadeFile)
    faces = face.detectMultiScale(gray)

    for (x,y,w,h) in faces:
        roiGray = gray[y:y+h, x:x+w]

    return roiGray
def prepareImageForModel(faceImage):
     resized = cv2.resize(faceImage, (48,48), interpolation = cv2.INTER_AREA)
     imgResult = np.expand_dims(resized, axis=0)
     imgResult = imgResult/255.0
     return imgResult
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('model.hdf5')
    return model
model = load_model()
st.write("Emotion Classifier")
file = st.file_uploader("Please upload an image of a person", type=["jpg","png"])
if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width = True)
    faceImg = findFace(image)
    imgForPrediction = prepareImageForModel(faceImg)
    predictions = model.predict(imgForPrediction, verbose=1)
    string = "The person in the image is most likely: " + class_names[np.argmax(predictions)]
    st.success(string)
