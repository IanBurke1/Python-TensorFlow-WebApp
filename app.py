from flask import Flask, flash, render_template, request, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename 
import os
from load import * 
from skimage import color # change colour of image
from scipy.misc import imread, imresize, imsave # For images
import numpy as np # martix math
import re #regular expression for canvas img string data
import base64 # Encode canvas data bytes

# Upload files code adapted from: http://flask.pocoo.org/docs/0.12/patterns/fileuploads/
# Canvas image code adapted from: https://github.com/llSourcell/how_to_deploy_a_keras_model_to_production/blob/master/app.py

# =========== Initialisation ============================================================================================================

# UPLOAD_FOLDER is where we will store the uploaded image files
UPLOAD_FOLDER = './static/uploads'
# set of allowed file extensions.
ALLOWED_EXTENSIONS = set(['pdf', 'png', 'jpg', 'jpeg', 'gif']) 

# model and graph from load.py
model, graph = loadModel()

# Pass in __name__ to help flask determine root path
app = Flask(__name__) # Initialising flask app
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER # configure upload folder

# =============== Routing/Mapping =======================================================================================================

# @ signifies a decorator which is a way to wrap a function and modify its behaviour
@app.route('/') #connect a webpage. '/' is a root directory.
def main():
   return render_template("index.html") # return rendered template 

# ============== Upload image ====================

# function that checks if file extension is allowed
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Upload Image file
@app.route("/upload", methods=['GET', 'POST']) 
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
        # if theres a file with allowed extension then..
        if file and allowed_file(file.filename):
            # secure a filename before storing it directly
            filename = secure_filename(file.filename) 
            # Save file to upload_folder
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename)) 
           
            return redirect(url_for('uploaded_file', filename=filename))
    
   # return render_template('index.html')

#@app.route('/uploads/<filename>')
#def uploaded_file(filename):
#    return send_from_directory(app.config['UPLOAD_FOLDER'],filename)

# ============== Canvas Image ========================

def convertData(imgData):
    # parse canvas bytes and save as output.png
    # base64 is used to take binary data and turn it into text to easily transmit from html
    imgstr = re.search(b'base64,(.*)', imgData).group(1)
    with open('output.png','wb') as output:
        output.write(base64.decodebytes(imgstr))

# Canvas Digit Image
@app.route('/predict',methods=['GET','POST'])
def predict():
        #Get data from canvas and save as image
        convertData(request.get_data())
        # read parsed image back in mode L = 8-bit pixels, black and white.
        x = imread('output.png',mode='L')
        # compute a bit-wise inversion
        x = np.invert(x)
        # make it 28x28
        x = imresize(x,(28,28))
        
        #convert to a 4D tensor to feed into our neural network model
        x = x.reshape(1,28,28,1)
        
        #in our computation graph
        with graph.as_default():
            # predict the digit using our model
            prediction = model.predict(x)
            print(prediction)
            print(np.argmax(prediction,axis=1))
            #convert the response to a string
            response = np.array_str(np.argmax(prediction,axis=1))
            return response 


if __name__ == "__main__":
    app.run(debug=True) # Start the web server. debug=True means to auto refresh page after code changes 