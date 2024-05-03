import base64
import os
from flask import Flask, Response, jsonify, redirect, render_template, request,url_for
from object_detection import config as config
from object_detection.detection import detecting_people
from scipy import spatial
from scipy.spatial import distance as dist
import numpy as np
import imutils
import cv2

app = Flask(__name__, template_folder='templates')

print("YOLO Loading..")
net = cv2.dnn.readNetFromDarknet("yolov3.cfg", "yolov3.weights")


	
classes = []

with open("coco.names", "r") as f:
	classes = [line.strip() for line in f.readlines()]

names_layer = net.getLayerNames()
names_layer = [names_layer[n - 1] for n in net.getUnconnectedOutLayers()]

video_path = None

writer =None

@app.route('/')
def index():
    return render_template('homePage.html')

@app.route('/upload', methods=['POST'])
def uploadVideo():
    if request.method == 'POST':
        # For handling situations hwere a file is not uploaded.
        if 'file' not in request.files:
            return render_template('homePage.html', error='File is not found')

        fileUploaded = request.files['file']
        if fileUploaded.filename == '':
            return render_template('homePage.html', error='Please select the file you want to generate the detection..')

        # Process the video here 
        # Save the processed video and obtain its path
        video_path = os.path.join('uploads', fileUploaded.filename)
        # print(video_path)
    
        # return render_template('base.html', video_path=video_path)
        return redirect(url_for('feed_video', video_path=video_path))

def gen(video_path):
    print(video_path)
    vs = cv2.VideoCapture(video_path)
    while True:
        (grabbed,fr) = vs.read()
        #print("yes")
        if not grabbed:
            break
        fr = imutils.resize(fr,width =600)
        res=detecting_people(fr,net,names_layer ,idOfPerson=classes.index("person"))
        violatingPeople =set()
        if len(res)>=2:
            points_centroid = np.array([r[2] for r in res])
            Dist = dist.cdist(points_centroid,points_centroid,metric="euclidean")
            
            for m in range(0,Dist.shape[0]):
                for n in range(i+1,Dist.shape[1]):
                    
                    if Dist[m,n]<50:
                        
                        violatingPeople.add(i)
                        violatingPeople.add(j)
        for(i,(prob,bbox,centroid)) in enumerate(res):
            (startOfX,startOfY,endOfX,endOfY)=bbox
            (cX,cY)= centroid
            color =(0,255,0)
        
            if m in violatingPeople:
                colorBox = (0,0,255)
            cv2.rectangle(fr,(startOfX,startOfY),(endOfX,endOfY),colorBox,2)
            cv2.circle(fr,(cX,cY),5,colorBox,1)
        cv2.imwrite("1.jpg",fr)
        (flag,imageEncoded) =cv2.imencode(".jpg",fr)
        yield(b' --frame\r\n' b'Content-Type:image/jpeg\r\n\r\n'+bytearray(imageEncoded)+b'\r\n')

@app.route('/feed_video')
def feed_video():
    video_path = request.args.get('video_path') 
    if video_path:
        # Return the response to start streaming video frames
        return Response(gen(video_path), mimetype='multipart/x-mixed-replace; boundary=frame')

    return render_template('homePage.html', error='No video path provided')

if __name__== '__main__':
     app.run(debug =False)
      