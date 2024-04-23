import base64
import os
from flask import Flask, Response, jsonify, redirect, render_template, request
from object_detection import config as config
from object_detection.detection import detect_people
from scipy import spatial
from scipy.spatial import distance as dist
import numpy as np
import imutils
import cv2
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

app = Flask(__name__, template_folder='templates')

print("YOLO Loading..")
net = cv2.dnn.readNetFromDarknet("yolov3.cfg", "yolov3.weights")


if config.USE_GPU:
	
	print("[INFO] setting preferable backend and target to CUDA...")
	net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
	net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
	
	
classes = []

with open("coco.names", "r") as f:
	classes = [line.strip() for line in f.readlines()]

layer_names = net.getLayerNames()
layer_names = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]


def process_video(video_path):
    input_video = cv2.VideoCapture(video_path)
    output_frames = []

    while True:
        (received, frame) = input_video.read()

        if not received:
            break

        frame = imutils.resize(frame, width=700)
        results = detect_people(frame, net, layer_names, personIdx=classes.index("person"))
        
        violate = set()

        if len(results) >= 2:
            centroids = np.array([r[2] for r in results if r[2] is not None], dtype=float)
            if centroids.shape[0] >= 2:
                Distance = dist.cdist(centroids, centroids, metric="euclidean")

                for i in range(0, Distance.shape[0]):
                    for j in range(i + 1, Distance.shape[1]):
                        if Distance[i, j] < 50:
                            violate.add(i)
                            violate.add(j)

        for (i, (prob, bbox, centroid)) in enumerate(results):
            (startX, startY, endX, endY) = bbox
            (cX, cY) = centroid
            color = (0, 255, 0)

            if i in violate:
                color = (0, 0, 255)

            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
            cv2.circle(frame, (cX, cY), 5, color, 1)

         # Encode frame as base64 string
        _, buffer = cv2.imencode('.jpg', frame)
        frame_jpg = base64.b64encode(buffer).decode('utf-8')  # Convert bytes to base64 string
        output_frames.append(frame_jpg)

    input_video.release()
    return output_frames

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Check if a file was uploaded
        if 'file' not in request.files:
            return redirect(request.url)

        file = request.files['file']
        # Check if the file is empty
        if file.filename == '':
            return redirect(request.url)

        # Check if the file has an allowed extension
        
        video_path = os.path.join('uploads', file.filename)
        file.save(video_path)
        print("Video uploaded:", file.filename)

            # Process the uploaded video
        output_frames = process_video(video_path)
        print("Processing complete.")

        return render_template('index.html', video_path=video_path, frames=output_frames)

    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    video_path = request.args.get('video_path', '')
    if video_path:
        def generate():
            frames = process_video(video_path)
            for frame in frames:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame.encode() + b'\r\n')
        return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')
    else:
        return "Error: Video path not provided"


if __name__ == '__main__':
    app.run(debug=True)