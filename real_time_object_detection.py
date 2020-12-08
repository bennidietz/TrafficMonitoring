# import the necessary packages
import os
import time
import json
from counting_cars import Detection, analyseDectectionData, belongsToBboxes
from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import time
import cv2

'''
change this path to your project directory!
(or maybe automatically detect path somehow)
'''
basePath = os.path.dirname(os.path.realpath(__file__))
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--confidence", type=float, default=0.2,
	help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

# initialize the list of class labels MobileNet SSD was trained to
# detect, then generate a set of bounding box colors for each class
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor"]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

# load our serialized model from disk
print("[INFO] loading model...")
## net = cv2.dnn.readNetFromCaffe(basePath + "/models/mobileNet_cars/mobilenet_yolov3_deploy.prototxt",
##        basePath + "/models/mobileNet_cars/mobilenet_yolov3_deploy_iter_63000.caffemodel")
net = cv2.dnn.readNetFromCaffe(basePath + "/models/MobileNetSSD/MobileNetSSD_deploy.prototxt.txt",
        basePath + "/models/MobileNetSSD/MobileNetSSD_deploy.caffemodel")
##net = cv2.dnn.readNetFromCaffe(basePath + "/models/googlenet_cars/MobileNetSSD_deploy.prototxt.txt",
##        basePath + "/models/googlenet_cars/googlenet_finetune_web_car_iter_10000.caffemodel")

# initialize the video stream, allow the cammera sensor to warmup,
# and initialize the FPS counter
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)
fps = FPS().start()

collectedDetections = []
counter = 1

# loop over the frames from the video stream
while True:
    # grab the frame from the threaded video stream and resize it
    # to have a maximum width of 400 pixels
    frame = vs.read()
    frame = imutils.resize(frame, width=1000)
    # grab the frame dimensions and convert it to a blob
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
		0.007843, (300, 300), 127.5)
	# pass the blob through the network and obtain the detections and
	# predictions
    net.setInput(blob)
    detections = net.forward()
    # loop over the detections
    for i in np.arange(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with
        # the prediction
        confidence = detections[0, 0, i, 2]
        idx = int(detections[0, 0, i, 1])
        name = CLASSES[idx]
        # filter out weak detections by ensuring the `confidence` is
        # greater than the minimum confidence
        if name == "car" and confidence > args["confidence"]:
            # extract the index of the class label from the
            # `detections`, then compute the (x, y)-coordinates of
            # the bounding box for the object
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            # check if the car was already detected before
            millis = int(round(time.time() * 1000))
            #collectedDetections.append(json.dumps([str(confidence), millis, str(startX), str(startY), str(endX), str(endY)]))
            currCounter = None
            currEleemtn = [confidence, millis, startX, startY, endX, endY]
            belongingIndex = belongsToBboxes(collectedDetections, currEleemtn)
            if belongingIndex == -1:
                collectedDetections.append([counter, currEleemtn])
                counter = counter + 1
                currCounter = counter
            else:
                collectedDetections[belongingIndex].append(currEleemtn)
                currCounter = collectedDetections[belongingIndex][0]
            # draw the prediction on the frame
            label = "{}: {:.2f}%".format(CLASSES[idx],confidence * 100) + " " + str(currCounter)
            cv2.rectangle(frame, (startX, startY), (endX, endY),COLORS[idx], 2)
            y = startY - 15 if startY - 15 > 15 else startY + 15
            cv2.putText(frame, label, (startX, y),cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)
            print("car!")
		# show the output frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break
	# update the FPS counter
    fps.update()
# stop the timer and display FPS information
fps.stop()
print(collectedDetections)
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
