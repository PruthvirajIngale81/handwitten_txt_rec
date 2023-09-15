from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import os
from model.segment import predict_img

app = Flask(__name__)

UPLOAD_FILE_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FILE_FOLDER
app.config['UPLOAD_EXTENSIONS'] = ['.jpg', '.png', '.jpeg']

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        file = request.files['file']
        filename = secure_filename(file.filename)
        if filename == '':
            return render_template('home.html', error='Please select a file')
        file_ext = os.path.splitext(filename)[1]
        if file_ext not in app.config['UPLOAD_EXTENSIONS']:
            return render_template('home.html', error='File type not supported')
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        prediction = predict_img(filepath)
        return render_template('result.html', prediction=prediction)
    return render_template('home.html')

if __name__ == '__main__':
    app.run(debug=True)