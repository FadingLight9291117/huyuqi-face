import base64
import os
import cv2
from flask import Flask, request, jsonify
from detect import detect
import numpy as np

app = Flask(__name__)


@app.route('/detect', methods=['POST'])
def detectApi():
    if 'image' not in request.files:
        return jsonify({'error': 'No image found in request'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No image selected for uploading'}), 400

    if file and allowed_file(file.filename):
        # filename = secure_filename(file.filename)
        filename = file.filename
        # file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        # detect

        img = np.frombuffer(file.stream.read(), np.uint8)
        img = cv2.imdecode(img, cv2.IMREAD_COLOR)

        boxes, plot = detect(img)
        retval, buffer = cv2.imencode('.png', plot)
        jpg_as_text = base64.b64encode(buffer).decode('utf-8')
        return jsonify({'boxes': boxes, 'image': jpg_as_text}), 200
    else:
        return jsonify({'error': 'Allowed image types are png, jpg, jpeg, gif'}), 400

@app.route('/repair', methods=['POST'])
def repairApi():
    if 'image' not in request.files:
        return jsonify({'error': 'No image found in request'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No image selected for uploading'}), 400

    if file and allowed_file(file.filename):
        # filename = secure_filename(file.filename)
        filename = file.filename
        # file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        # detect

        img = np.frombuffer(file.stream.read(), np.uint8)
        img = cv2.imdecode(img, cv2.IMREAD_COLOR)

        boxes, plot = detect(img)
        retval, buffer = cv2.imencode('.png', plot)
        jpg_as_text = base64.b64encode(buffer).decode('utf-8')
        return jsonify({'boxes': boxes, 'image': jpg_as_text}), 200
    else:
        return jsonify({'error': 'Allowed image types are png, jpg, jpeg, gif'}), 400

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg', 'gif'}


if __name__ == '__main__':
    app.run(debug=True, port=8080)
