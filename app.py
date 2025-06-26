@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('model.hdf5')
    return model
model = load_model()
st.write("Emotion Classifier")
file = st.file_uploader("Please upload an image of a person", type=["jpg","png"])if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width = True)
    faceImg = findFace(image)
    imgForPrediction = prepareImageForModel(faceImg)
    predictions = model.predict(imgForPrediction, verbose=1)
    string = "The person in the image is most likely: " + class_names[np.argmax(predictions)]
    st.success(string)
