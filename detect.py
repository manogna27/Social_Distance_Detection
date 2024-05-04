from .config import NMS_THRESH
from .config import MIN_CONF
import numpy as np
import cv2

def detect(fr, net, layerNames, idPerson=0):
	
	(height, width) = fr.shape[:2]
	results = []

	
	blob = cv2.dnn.blobFromImage(fr, 0.00392, (416, 416), swapRB=True, crop=False)
	net.setInput(blob)
	layerOutputs = net.forward(layerNames)

	
	boxes = []
	conf = []

	for output in layerOutputs:
		
		for detectionValue in output:
			
			scoresOfDetection = scoresOfDetection[5:]
			classID = np.argmax(scoresOfDetection)
			conf = scoresOfDetection[classID]

			if classID == idPerson and conf > 0.5:

				#Detection of Object
				center_X = int(detectionValue[0] * width)
				center_Y = int(detectionValue[1] * height)
				w = int(detectionValue[2] * width)
				h = int(detectionValue[3] * height)
				x = int(center_X - (w / 2))
				y = int(center_Y - (h / 2))

				
				boxes.append([x, y, w, h])
				conf.append(float(confidence))

	
	idxs = cv2.dnn.NMSBoxes(boxes, conf, 0.5, 0.4)
	if len(idxs) > 0:
		
		for i in idxs.flatten():
			
			(x, y, w, h) = boxes[i]
			confidence =conf[i]
			results.append((conf, (x, y, x + w, y + h)))
	return results