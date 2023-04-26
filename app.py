from flask import Flask, request, render_template
from keras.models import load_model
from keras_preprocessing.image import load_img
from keras.preprocessing import image
import keras.utils as image
from PIL import Image


import os
import numpy as np

app = Flask(__name__)

# Load the saved model
model = load_model('D:/dog gemotion/.venv/app/model.h5')
class_names = ['angry', 'sad', 'relaxed', 'happy']

# Define a function to preprocess the uploaded image
def preprocess_image(img_path):
    img = load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = img_array/255.0
    return preprocessed_img

# Define a function to make a prediction using the loaded model
def predict_emotion(img_path):
    preprocessed_img = preprocess_image(img_path)
    predictions = model.predict(preprocessed_img)
    predicted_class = class_names[np.argmax(predictions)]
    return predicted_class

# Define the home page
# Define the home page
app = Flask(__name__, template_folder='D:/dog gemotion/.venv/app/template')
# render htmp page
@app.route('/')
def home():
 return render_template('index.html')



# Define the page to handle image uploads and show the result
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get the uploaded file
        file = request.files['image']
        # Save the file to the uploads directory
        img_path = os.path.join(app.root_path, 'uploads', file.filename)
        file.save(img_path)
        # Make a prediction using the loaded model
        predicted_class = predict_emotion(img_path)
        # Render the result template with the predicted class and image path
        return render_template('result.html', predicted_class=predicted_class, img_path=img_path)

if __name__ == '__main__':
    # Create the uploads directory if it doesn't exist
    uploads_dir = os.path.join(app.root_path, 'uploads')
    if not os.path.exists(uploads_dir):
        os.makedirs(uploads_dir)
    app.run(debug=True)

