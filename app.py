from flask import Flask, render_template, request, Response

import cv2
import numpy as np
app = Flask(__name__)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
@app.route('/')

def index():
    return render_template('index.html')



def process_frame(frame):
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    
    _, encoded_image = cv2.imencode('.jpeg', frame)
    return encoded_image.tobytes()


@app.route('/process_frame', methods=['POST'])
def process_frame_route():
    frame_data = request.files['frame'].read()
    nparr = np.frombuffer(frame_data, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    processed_frame = process_frame(frame)
    return Response(response=processed_frame, content_type='image/jpeg')

if __name__ == '__main__':
    app.run(debug=True)