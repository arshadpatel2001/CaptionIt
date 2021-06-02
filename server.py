import os

from PIL import Image
from flask import Flask, render_template, request

from capgen import CaptionGenerator

gencap = CaptionGenerator()

THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))

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
        print("Generating Captions")

        cap = gencap.get_caption(uploaded_img_path)

        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)

        print("=" * 50)
        print(file_path)
        print("=" * 50)

        return render_template('index.html', data=cap, file=file_path)
    else:
        return render_template('index.html')


if __name__ == "__main__":
    app.run()
