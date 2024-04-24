import base64
import os
from flask import Flask, Response, jsonify, redirect, render_template, request,url_for
from object_detection import config as config
from object_detection.detection import detect_people
from scipy import spatial
from scipy.spatial import distance as dist
import numpy as np
import imutils
import cv2

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

video_path = None


#vs =cv2.VideoCapture("pedestrians3.mp4")
writer =None

@app.route('/')
def index():
    return render_template('base.html')

@app.route('/upload', methods=['POST'])
def upload():
    if request.method == 'POST':
        # Handle file upload
        if 'file' not in request.files:
            return render_template('base.html', error='No file part')

        file = request.files['file']
        if file.filename == '':
            return render_template('base.html', error='No selected file')

        # Process the video here (e.g., detect social distancing)
        # Save the processed video and obtain its path
        video_path = os.path.join('uploads', file.filename)
        # print(video_path)
    
        # return render_template('base.html', video_path=video_path)
        return redirect(url_for('video_feed', video_path=video_path))

def gen(video_path):
    print(video_path)
    vs = cv2.VideoCapture(video_path)
    while True:
        (grabbed,frame) = vs.read()
        if not grabbed:
            break
        frame = imutils.resize(frame,width =700)
        results =detect_people(frame,net,layer_names ,personIdx=classes.index("person"))
        violate =set()
        if len(results)>=2:
            centroids = np.array([r[2] for r in results])
            D = dist.cdist(centroids,centroids,metric="euclidean")
    
            for i in range(0,D.shape[0]):
                for j in range(i+1,D.shape[1]):
                    if D[i,j]<config.MIN_DISTANCE:
                        violate.add(i)
                        violate.add(j)
        for(i,(prob,bbox,centroid)) in enumerate(results):
            (startX,startY,endX,endY)=bbox
            (cX,cY)= centroid
            color =(0,255,0)
        
            if i in violate:
                color = (0,0,255)
            cv2.rectangle(frame,(startX,startY),(endX,endY),color,2)
            cv2.circle(frame,(cX,cY),5,color,1)
        #text ="Social Distancing Violations: {}".format(len(violate))
        #cv2.putText(frame,text,(10,frame.shape[0]-25),cv2.FONT_HERSHEY_SIMPLEX,0.85,(0,0,255),3)
        cv2.imwrite("1.jpg",frame)
        (flag,encodedImage) =cv2.imencode(".jpg",frame)
        yield(b' --frame\r\n' b'Content-Type:image/jpeg\r\n\r\n'+bytearray(encodedImage)+b'\r\n')

@app.route('/video_feed')
def video_feed():
    video_path = request.args.get('video_path') 
    if video_path:
        # Return the response to start streaming video frames
        return Response(gen(video_path), mimetype='multipart/x-mixed-replace; boundary=frame')

    return render_template('base.html', error='No video path provided')
    # return Response(gen(),mimetype='multipart/x-mixed-replace; boundary=frame')
if __name__== '__main__':
     app.run(debug =False)
      