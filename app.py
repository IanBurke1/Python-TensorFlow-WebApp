import os
from flask import Flask, flash, render_template, request, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename
from load import *
from skimage import color
from scipy.misc import imread, imresize

# Adapted from: http://flask.pocoo.org/docs/0.12/patterns/fileuploads/

# UPLOAD_FOLDER is where we will store the uploaded files
UPLOAD_FOLDER = './static/uploads'
ALLOWED_EXTENSIONS = set(['pdf', 'png', 'jpg', 'jpeg', 'gif']) # set of allowed file extensions.

model, graph = load()

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
            img = color.rgb2gray(imread(filename, mode='L')) # change colour to greyscale
            img = imresize(img, (28,28)) # resize the image 
            img = img.reshape(1, 28, 28, 1) # Reshape the image

            return redirect(url_for('uploaded_file',
                                    filename=filename))
    
    return render_template('index.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'],
                               filename)



if __name__ == "__main__":
    app.run(debug=True) # Start the web server. debug=True means to auto refresh page after code changes 