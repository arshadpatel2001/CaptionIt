# Load libraries

import os
import pickle
import numpy as np
from PIL import Image
from flask import Flask, render_template, request
from keras.applications.resnet50 import ResNet50
from keras.models import load_model
from keras.preprocessing import image, sequence
THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))


loaded_model = load_model(os.path.join(THIS_FOLDER, 'models', 'tfmodel'))
loaded_model.compile(loss='categorical_crossentropy', optimizer='RMSprop', metrics=['accuracy'])

with open(os.path.join(THIS_FOLDER, 'models', 'indices_2_word_file.p'), 'rb') as f:
    indices_2_word = pickle.load(f)

with open(os.path.join(THIS_FOLDER, 'models', 'word_2_indices_file.p'), 'rb') as f:
    word_2_indices = pickle.load(f)

max_len = 40

def preprocessing(img_path):
    im = image.load_img(img_path, target_size=(224, 224, 3))
    im = image.img_to_array(im)
    im = np.expand_dims(im, axis=0)
    return im

def get_encoding(model, img):
    image = preprocessing(img)
    pred = model.predict(image).reshape(2048)
    return pred

resnet = ResNet50(include_top=False, weights='imagenet', input_shape=(224, 224, 3), pooling='avg')

def predict_captions(image):
    start_word = ["<start>"]
    while True:
        par_caps = [word_2_indices[i] for i in start_word]
        par_caps = sequence.pad_sequences([par_caps], maxlen=max_len, padding='post')
        preds = loaded_model.predict([np.array([image]), np.array(par_caps)])
        word_pred = indices_2_word[np.argmax(preds[0])]
        start_word.append(word_pred)

        if word_pred == "<end>" or len(start_word) > max_len:
            break

    return ' '.join(start_word[1:-1])

app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = os.path.join('static', 'database')
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 1
app.config['ALLOWED_EXTENSIONS'] = {'jpg', 'jpeg', 'png'}


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/after', methods=['GET', 'POST'])
def after():
    global model, resnet, vocab, inv_vocab

    if request.method == 'POST':

        if 'query_img' not in request.files or request.files['query_img'].filename == '' or not allowed_file(
                request.files['query_img'].filename):

            return render_template('index.html')

        print("=" * 50)
        print("Saving Image")
        file = request.files['query_img']
        img = Image.open(file.stream)
        uploaded_img_path = os.path.join(THIS_FOLDER, app.config['UPLOAD_FOLDER'], file.filename)
        img.save(uploaded_img_path)

        print("=" * 50)
        print("Image Saved Successfully")

        print("=" * 50)
        print("Preprocessing Image")
        test_img = get_encoding(resnet, uploaded_img_path)

        print("=" * 50)
        print("Generating Captions")
        Argmax_Search = predict_captions(test_img)

        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)

        print("=" * 50)
        print(file_path)
        print("=" * 50)
        return render_template('index.html', data=Argmax_Search, file=file_path)

    else:
        return render_template('index.html')


if __name__ == "__main__":
    app.run(debug=True)