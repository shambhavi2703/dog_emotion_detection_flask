# dog_emotion_detection_flask

This repository contains a Flask web application that predicts the emotion of dogs from input images  The model has been trained on a dataset of labeled dog images to classify them into different emotion categories, such as happy, sad, angry, and relaxed.

## Demo 
you can find demo images of working model in DEMO 

## Walkthrough 
This repository contains a Flask web application that predicts the emotion of dogs from input images using two Inception v3 deep learning models. The models, saved as model1.h5 and model2.h5, were developed on Google Colab in the file model_buil.ipynb. 
The dog image dataset used for training the models is available [here](https://www.kaggle.com/datasets/devzohaib/dog-emotions-prediction).

The templates folder contains two HTML templates used in the Flask app: index.html and result.html. The index.html template displays a simple form that allows the user to upload an image of a dog. The result.html template displays the predicted emotion of the dog based on the input image


## Flask Framework Intriductopn
Flask is a micro web framework written in Python that allows you to build web applications quickly and easily. It is known for its simplicity, flexibility, and minimalism, making it an ideal choice for small to medium-sized projects. Flask provides you with the basic tools and features you need to get started, such as routing, templates, and request handling, while also allowing you to customize and extend its functionality with plugins and extensions.

## Installation
To install Flask, you can use pip, the package installer for Python:
```
pip install Flask
```

## Usage
Here is an example of a simple Flask application that serves a "Hello, World!" message:

```
from flask import Flask

app = Flask(__name__)

@app.route('/')
def hello():
    return 'Hello, World!'

if __name__ == '__main__':
    app.run()
  ```
  
Save the code above as a Python file (e.g., app.py), and run it in your terminal using the command:

```
python app.py
```
Then, open your web browser and navigate to http://localhost:5000/, and you should see the "Hello, World!" message.

## References :
https://flask.palletsprojects.com/en/2.3.x/
