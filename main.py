import base64
import os
from flask import Flask, Response, jsonify, redirect, render_template, request,url_for
from object_detection import config as config
from object_detection.detect import detect
from scipy import spatial
from scipy.spatial import distance as dist
import numpy as np
import imutils
import cv2

app = Flask(__name__, template_folder='templates')

print("YOLO Loading..")
net = cv2.dnn.readNetFromDarknet("yolov3.cfg", "yolov3.weights")

	
objectClasses = []

focal_length = 1000.0 
baseline = 1.0  


with open("coco.names", "r") as f:
	objectClasses = [line.strip() for line in f.readlines()]

layerNames = net.getLayerNames()
layerNames = [layerNames[i - 1] for i in net.getUnconnectedOutLayers()]

videoPath  = None


writer =None


#initially routed to home page
@app.route('/')
def index():
    return render_template('homePage.html')


#function to upload the video from the user interface and save it to a temporary foder called uploads.
@app.route('/upload', methods=['POST'])
def upload():
 if request.method == 'POST':
        # Handle file upload
        if 'file' not in request.files:
            return render_template('homePage', error='error check')

        file = request.files['file']
        if file.filename == '':
            return render_template('homePage', error='Please select a file to generate detection..')

        # Process the video here (e.g., detect social distancing)
        # Save the processed video and obtain its path
        videoPath = os.path.join('uploads', file.filename)
        # print(videoPath)
        return redirect(url_for('video_feed', videoPath = videoPath)) #sending the path of the video as a return value


#Camera View Calibration caluculation happens here
def calculate_distance(p1, p2):
   
    x1, y1, x2, y2 = p1[0], p1[1], p2[0], p2[1]
    disparity = abs(x2 - x1)
    if disparity == 0:
        return float('inf')
    dist = baseline * focal_length / disparity #values of this may depend on various factors
    return dist #returning the distance calculated


def gen(videoPath):
    print(videoPath)
    vs = cv2.VideoCapture(videoPath)
    while True:
        (grabbedValue, fr) = vs.read()
        if not grabbedValue:
            break
        
        fr = imutils.resize(fr, width=700)
        pb = detect(fr, net, layerNames, idPerson=objectClasses.index("person")) #since we should only identify people

        if not pb:
            # there were no people found, so we skip the detection here.
            continue
        
        for i in range(len(pb)):
            if len(pb[i]) < 4:
                # Skipping invalid bounding boxes
                continue
            
            x1, y1, x2, y2 = pb[i][0], pb[i][1], pb[i][2], pb[i][3]
            
            dist = calculate_distance((x1, y1), (x2, y2))
            
            if dist < 2.0:
                cv2.rectangle(fr, (x1, y1), (x2, y2), (0, 0, 255), 2)  
            else:
                cv2.rectangle(fr, (x1, y1), (x2, y2), (0, 255, 0), 2)  
        
        cv2.imwrite("1.jpg", fr)
        (flag, imageEncoded) = cv2.imencode(".jpg", fr)
        yield (b'--fr\r\n' b'Content-Type: image/jpeg\r\n\r\n' + bytearray(imageEncoded) + b'\r\n') #yield does the same work as a rturn statement..

#This function will print out all the individual frames of the video with detection made
#sending each frame to process
@app.route('/video_feed')
def video_feed():
    videoPath = request.args.get('videoPath') 
    if videoPath :
        # Return the response to start streaming video frames
        return Response(gen(videoPath), mimetype='multipart/x-mixed-replace; boundary=frame')
    return render_template('homePage.html', error='No video path provided')

if __name__== '__main__':
     app.run(debug =False)
      

