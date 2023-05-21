import cv2
import numpy as np
import json
import os
from flask import Flask
from flask import render_template, request
from werkzeug.utils import secure_filename

from keras.applications.resnet50 import ResNet50
from keras.applications.resnet50 import preprocess_input, decode_predictions
from keras.preprocessing import image
from keras.models import load_model
from keras import backend as K 
from extract_bottleneck_features import *

def path_to_tensor(img_path):
    """
    """
    # loads RGB image as PIL.Image.Image type
    img = image.load_img(img_path, target_size=(224, 224))
    
    # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
    x = image.img_to_array(img)
    
    # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
    return np.expand_dims(x, axis=0)


def face_detector(img_path):
    """
    """
    face_cascade = cv2.CascadeClassifier('./haarcascades/haarcascade_frontalface_alt.xml')
    img = cv2.imread(img_path)
    faces = face_cascade.detectMultiScale(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    return len(faces) > 0


def ResNet50_predict_labels(img_path):
    """
    """
    ResNet50_model = ResNet50(weights='imagenet')
   # returns prediction vector for image located at img_path
    img = preprocess_input(path_to_tensor(img_path)) 
    return np.argmax(ResNet50_model.predict(img))


def dog_detector(img_path):
    """
    This function determins whether dog is contained in an image. Returns "True" if a dog is detected in the image stored at img_path.
    """
    prediction = ResNet50_predict_labels(img_path)
    return ((prediction <= 268) & (prediction >= 151)) 


def Inception_predict_breed(img_path):
    
    dog_names = json.load(open('./data/dog_names.json', 'r'))
    
    # extract Inception V3 bottleneck features
    bottleneck_feature = extract_InceptionV3(path_to_tensor(img_path))
    
    # load model
    incep_model = load_model("saved_models/dogBreedInceptionV3.h5")
    
    # obtain predicted vector
    predicted_vector = incep_model.predict(bottleneck_feature)
    
    # return dog breed that is predicted by the model
    return dog_names[np.argmax(predicted_vector)]


def predict_image(img_path):
    """
    """
    
    K.clear_session()
    
    dog_breed_predicted = Inception_predict_breed(img_path).split(".")[-1]
    
    if dog_detector(img_path):
        return("The image you provided looks like an %s." % dog_breed_predicted)
    elif face_detector(img_path):
        return("The image you provided is a human.")
    else:
        return("The image you provided is neither a dog nor a human.")

        
        

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join('static', 'test_images')
app.config['UPLOAD_EXTENSIONS'] = ['.jpg', '.png', '.jpeg']



# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    # render web page with plotly graphs
    return render_template('master_dog.html')

@app.route('/go', methods = ['GET', 'POST'])
def go():
    """
    Partially adapted from https://blog.filestack.com/thoughts-and-knowledge/secure-file-upload-flask-python/
    """
    # use model to predict classification for query
    if request.method == 'POST':
        file_uploaded = request.files['file']
        filename = secure_filename(file_uploaded.filename)
        print(filename)
        if filename != '':
            file_extension = os.path.splitext(filename)[1]
            if file_extension not in app.config['UPLOAD_EXTENSIONS']:
                abort(400)
            file_uploaded.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            res = predict_image(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            img = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
    # This will render the go.html Please see that file. 
    return render_template(
        'go_dog.html',
        result=res,
        img=img)

def main():
    app.run(host='0.0.0.0', port=3000, debug=True)


if __name__ == '__main__':
    main()