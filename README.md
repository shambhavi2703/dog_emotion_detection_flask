# dog_emotion_detection_flask

This repository contains a Flask web application that predicts the emotion of dogs from input images using a pre-trained Inception v3 deep learning model. The model has been trained on a dataset of labeled dog images to classify them into different emotion categories, such as happy, sad, angry, and so on.


## Flask Framework
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

Conclusion
Flask is a powerful yet lightweight framework that can help you build web applications quickly and easily. With its flexibility and simplicity, you can create anything from a simple web page to a more complex application.
