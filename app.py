from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import base64
from ultralytics import YOLO
from PIL import Image
import io

app = Flask(__name__)

# Load the YOLO model once when server starts
model = YOLO("last.pt")

def process_frame(image_bytes, conf_threshold, use_grayscale):
    # Convert bytes to numpy array
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if use_grayscale:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    # Inference
    results = model.predict(img, imgsz=416, conf=conf_threshold, verbose=False)
    
    # Annotate image
    annotated_img = results[0].plot()
    
    # Extract detection info
    detections = []
    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        detections.append({
            "label": model.names[cls_id],
            "confidence": float(box.conf[0])
        })

    # Encode annotated image to base64 for the frontend
    _, buffer = cv2.imencode('.jpg', annotated_img)
    encoded_image = base64.b64encode(buffer).decode('utf-8')
    
    return encoded_image, detections

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/inspect', methods=['POST'])
def inspect():
    file = request.files['image']
    conf = float(request.form.get('confidence', 0.65))
    grayscale = request.form.get('grayscale') == 'true'
    
    img_bytes = file.read()
    encoded_img, detections = process_frame(img_bytes, conf, grayscale)
    
    return jsonify({
        "image": encoded_img,
        "detections": detections,
        "status": "NORMAL" if len(detections) == 0 else "DEFECT DETECTED"
    })

if __name__ == '__main__':
    app.run(debug=True)