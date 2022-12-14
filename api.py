# necessary imports
import os
from flask import Flask, request, make_response, jsonify
from werkzeug.utils import secure_filename
from fastai.vision.all import *
from fastai.data.external import *
import pathlib


temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

# list of allowed extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# initiation
app = Flask(__name__)

# specify the pickle file made in the BigDataAdvance notebook
learner = load_learner('cats.pkl')

# method used to check if the extension of the file matches one of the allowed extensions mentioned earlier
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# route and method for the API call
@app.route('/predict', methods=['POST'])
def predict():
    # exception for when there is no image (error code 400)
    if 'image' not in request.files:
        return {'error': 'no image found, in request.'}, 400

    # save the image in a variable called file
    file = request.files['image']
    # exception for when the name of the file is empty (error code 400)
    if file.filename == '':
        return {'error': 'no image found. Empty'}, 400
 
    # actions for when the file is present and has one of the allowed extensions
    if file and allowed_file(file.filename): 
        # create an image based on the file
        img = PILImage.create(file)
        # let the model make a prediction based on the image
        pred = learner.predict(img)
        print(pred)
        # return "success" as well as the prediction (code 200)
        return {'success': pred[0]}, 200    
    # create an exception for any other error that may occur (error code 500)
    return {'error': 'something went wrong.'}, 500

if __name__ == '__main__':
    # set to port 5000
    port = os.getenv('PORT',5000)
    # run the app
    app.run(debug=True, host='0.0.0.0', port=port) 
    print("success")