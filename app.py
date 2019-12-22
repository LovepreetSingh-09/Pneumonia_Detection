from flask import Flask, render_template,request, redirect
import numpy as np
import json
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from datetime import datetime
import os
from PIL import Image


def img_preprocess(img):
    size=(156,156)
    im=Image.open(img).resize(size,Image.ANTIALIAS)
    im=im.convert('RGB')
    im=np.array(im)
    im=im/255.
    return im


def load_init():
    model_loc='model2-006-0.89.h5'
    save_dir = os.path.join(os.getcwd(), 'saved_models')
    m=load_model(os.path.join(save_dir,model_loc))
    idx_2_labels = {0:'NORMAL', 1:'PNEUMONIA'}
    size=(156,156)
    return m, idx_2_labels, size


def predict(img_array):
    model, idx_2_labels, size = load_init()
    img_array=img_array.reshape(1,156,156,3)
    y=model.predict(img_array)
    idx=np.argmax(y,axis=1)
    prob=np.max(y,axis=1)*100
    prob=np.round(prob,2)[0]
    cl=idx_2_labels[idx[0]]
    return cl,prob
    

with open('config.json','r') as c:
    params=json.load(c)["params"]

local_server=params["local_server"]

app = Flask(__name__)
app.secret_key="super_secret_key"
app.config['UPLOAD_FOLDER']=params['upload_loc']


@app.route("/")
def home():
    return render_template('index.html',params=params)

@app.route("/about")
def about():
    return render_template('about.html',params=params)


@app.route("/post/detect",methods=['GET'])
def post_route():
    date=datetime.now()
    date=date.strftime("%Y-%m-%d %H:%M")
    return render_template('upload.html',params=params,date=date)


@app.route('/uploader',methods=['GET','POST'])	
def uploader():
    if request.method=='POST':
        pic=request.files['img_file']
        if len(pic.filename)==0:
            return redirect('/post/detect')
        img_loc=os.path.join(app.config['UPLOAD_FOLDER'],secure_filename(pic.filename))
        pic.save(img_loc)
        img_array=img_preprocess(pic)
        cl,prob=predict(img_array)
        return render_template('result.html',params=params,filename=pic.filename,result=cl,confidence=prob)
    return render_template('upload.html',params=params)


if __name__=='__main__':
    app.run(debug=True) 

