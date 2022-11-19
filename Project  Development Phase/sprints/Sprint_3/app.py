import numpy as np
import tensorflow as tf
import h5py
import os
from PIL import Image
from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
from werkzeug.utils import secure_filename

app = Flask(__name__)
UPLOAD_FOLDER = r'Model\data'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

model = load_model(r'Model\model.h5')


@app.route("/")
def index():
    return render_template("ibm.html")

@app.route("/home")
def home():
    return render_template('home.html')

@app.route("/web", methods = ['POST'])
def web():
    if request.method == "POST":
        f = request.files["image"]
        filepath = secure_filename(f.filename)
        f.save(os.path.join(app.config['UPLOAD_FOLDER'], filepath))

        upload_img = os.path.join(UPLOAD_FOLDER, filepath)
        img = Image.open(upload_img).convert("L")  # convert image to monochrome
        img = img.resize((28, 28))  # resizing of input image

        im2arr = np.array(img)  # converting to image
        im2arr = im2arr.reshape(1, 28, 28, 1)  # reshaping according to our requirement

        pred = model.predict(im2arr)

        num = np.argmax(pred, axis=1)  # printing our Labels
        out = " ".join(str(i) for i in num)
        return render_template('Hello.html',num=out)
    return render_template('Hello.html')
 
if __name__ == "__main__":
    app.run(debug = True)