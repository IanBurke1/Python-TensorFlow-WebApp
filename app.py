import os
from flask import Flask, flash, render_template, request, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename
from load import *
from skimage import color
from scipy.misc import imread, imresize, imsave
import numpy as np
import re
import base64

# Upload files adapted from: http://flask.pocoo.org/docs/0.12/patterns/fileuploads/

# UPLOAD_FOLDER is where we will store the uploaded files
UPLOAD_FOLDER = './static/uploads'
ALLOWED_EXTENSIONS = set(['pdf', 'png', 'jpg', 'jpeg', 'gif']) # set of allowed file extensions.

model, graph = loadModel()

# Pass in __name__ to help flask determine root path
app = Flask(__name__) # Initialising flask app

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER 

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/') #connect a webpage. '/' is a root directory.
def main():
   return render_template("index.html")

# Routing/Mapping
# @ signifies a decorator which is a way to wrap a function and modify its behaviour
@app.route("/upload", methods=['GET', 'POST']) # Connect to webpage. "/" is a root directory
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit a empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename) # secure a filename before storing it directly
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
           # x = color.rgb2gray(imread(filename, mode='L')) # change colour to greyscale
           #  x = imresize(x, (28,28)) # resize the image 
           # x = x.reshape(1, 28, 28, 1) # Reshape the image
           # with graph.as_default():
            #    out = model.predict(x)
             #   print(out)
              #  print(np.argmax(out, axis=1))
               # response = np.array_str(np.argmax(out, axis=1))
                #return response
           
            return redirect(url_for('uploaded_file', filename=filename))
    
   # return render_template('index.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'],filename)

def getImage(imgData):
    # parse canvas bytes and save as output.png
    imgstr = re.search(b'base64,(.*)', imgData).group(1)
    with open('output.png','wb') as output:
        output.write(base64.decodebytes(imgstr))

@app.route('/predict',methods=['GET','POST'])
def predict():
        #Get data from canvas and save as image
        getImage(request.get_data())
        # read parsed image back in 8-bit, black and white mode (L)
        x = imread('output.png',mode='L')
        x = np.invert(x)
        # make it the right size
        x = imresize(x,(28,28))
        
        #convert to a 4D tensor to feed into our neural network model
        x = x.reshape(1,28,28,1)
        
        #in our computation graph
        with graph.as_default():
            #perform the prediction
            prediction = model.predict(x)
            print(prediction)
            print(np.argmax(prediction,axis=1))
            #convert the response to a string
            response = np.array_str(np.argmax(prediction,axis=1))
            return response 


if __name__ == "__main__":
    app.run(debug=True) # Start the web server. debug=True means to auto refresh page after code changes 