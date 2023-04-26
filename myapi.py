from flask import Flask, request, jsonify
from keras.models import load_model
from keras.preprocessing import image
import numpy as np

# Define the Flask app
app = Flask(__name__)

# Load the pre-trained model
model = load_model('D:/project/model.h5')

# Define a function to preprocess the uploaded image
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = img_array / 255.0  # normalize pixel values to [0, 1]
    
    return preprocessed_img

# Define the route for the prediction API
@app.route('/api/predict', methods=['POST'])
def predict():
    # Get the uploaded file
    file = request.files['file']
    # Save the file to disk
    file_path = 'temp.jpg'  # temporary file path
    file.save(file_path)
    # Preprocess the image
    preprocessed_img = preprocess_image(file_path)
    # Make a prediction on the image
    prediction = model.predict(preprocessed_img)
    predicted_class = np.argmax(prediction)
    # Map class index to emotion name
    emotion_names = {0: 'angry', 1: 'happy', 2: 'relaxed',3: 'sad'}
    predicted_emotion = emotion_names[predicted_class]
    # Return the predicted emotion as JSON
    return jsonify({'emotion': predicted_emotion})

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
