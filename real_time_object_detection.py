# import the necessary packages
import os, time, datetime, webbrowser, settings
import counting_cars
import ocr_license_plate
from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import time
import cv2
import json

'''
change this path to your project directory!
(or maybe automatically detect path somehow)
'''

testvideoPath = settings.getBaseDir() + '/testfiles/out13.mp4'
# testvideoPath = settings.getBaseDir() + '/testfiles/out3.mp4'
# testvideoPath = settings.getBaseDir() + '/testfiles/highway.mov'

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--confidence", type=float, default=0.47,
	help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

# initialize the list of class labels MobileNet SSD was trained to
# detect, then generate a set of bounding box colors for each class
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor"]
boxcolor = [255,80,180]

# load our serialized model from disk
print("[INFO] loading model...")
## net = cv2.dnn.readNetFromCaffe(settings.getBaseDir() + "/models/mobileNet_cars/mobilenet_yolov3_deploy.prototxt",
##        settings.getBaseDir() + "/models/mobileNet_cars/mobilenet_yolov3_deploy_iter_63000.caffemodel")
net = cv2.dnn.readNetFromCaffe(settings.getBaseDir() + "/models/MobileNetSSD/MobileNetSSD_deploy.prototxt.txt",
        settings.getBaseDir() + "/models/MobileNetSSD/MobileNetSSD_deploy.caffemodel")
##net = cv2.dnn.readNetFromCaffe(settings.getBaseDir() + "/models/googlenet_cars/MobileNetSSD_deploy.prototxt.txt",
##        settings.getBaseDir() + "/models/googlenet_cars/googlenet_finetune_web_car_iter_10000.caffemodel")

collectedDetections = []

videoFileDir = settings.getOutputDir() + settings.getEnding(testvideoPath) + "/"

# stop_condition: determines, when the analysis is halted (eg when the videostream is over)
# vs: the videostream (could be live from camera or from prerecorded)
# frame_skip: the amount of frames that are skipped from the video stream (to counter the lag)
def analyze_video(stop_condition,vs,frame_skip):
    counter = 0
    # counts the frames
    frame_counter = 0
    # loop over the frames from the video stream
    if not os.path.isdir(settings.getOutputDir() + settings.getEnding(testvideoPath)):
        os.makedirs(settings.getOutputDir() + settings.getEnding(testvideoPath))
    while stop_condition:
        # grab the frame from the threaded video stream and resize it
        # to have a maximum width of 400 pixels
        ret, frame = vs.read()
        if not ret:
            print("[ERROR] There was an error reading the frame. Its value is:")
            print(frame)
            break
        if len(frame) == 0:
            print("[INFO] Video finished")
            break
        if frame_counter > frame_skip - 1:
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
                    startY = max(0, startY)
                    endY = max(0, endY)
                    startX = max(0, startX)
                    endX = max(0, endX)
                    cropped_frame = frame[startY:endY, startX:endX]
                    meanColor = counting_cars.calcMean(cropped_frame)
                    currElement = [float(confidence), millis, startX, startY, endX, endY, (startX+endX)/2, (startY+endY)/2, meanColor]
                    belongingIndex = counting_cars.belongsToBboxes(collectedDetections, currElement)
                    if belongingIndex == -1:
                        collectedDetections.append([counter, currElement])
                        print("Car " + str(counter) + " detected...")
                        # draw the prediction on the frame
                        label = "{}: {:.2f}%".format(CLASSES[idx],confidence * 100) + " " + str(counter)
                        cv2.imwrite(videoFileDir + "car%d_%d.jpg" %  (counter, counting_cars.numberBoxes(collectedDetections, counter)),
                            cropped_frame )
                        counter = counter + 1
                        currCounter = counter
                    else:
                        collectedDetections[belongingIndex].append(currElement)
                        currCounter = collectedDetections[belongingIndex][0]
                        shifts = counting_cars.globalChange(collectedDetections[belongingIndex])
                        cv2.imwrite(videoFileDir + "car%d_%d.jpg" %  (currCounter, counting_cars.numberBoxes(collectedDetections, currCounter)),
                            cropped_frame )
                        # draw the prediction on the frame
                        label = "{}: {:.2f}%".format(CLASSES[idx],confidence * 100) + " " + str(currCounter)
                    cv2.rectangle(frame, (startX, startY), (endX, endY),boxcolor, 2)
                    y = startY - 15 if startY - 15 > 15 else startY + 15
                    cv2.putText(frame, label, (startX, y),cv2.FONT_HERSHEY_SIMPLEX, 0.5, boxcolor, 2)
                    # print("car!")
        	# show the output frame
            cv2.imshow("Frame", frame)
            frame_counter = 0
            continue
        frame_counter += 1
        key = cv2.waitKey(1) & 0xFF
        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break

# true when using connected camera, false when using prerecorded mp4
live = False
if live:
    # initialize the video stream, allow the cammera sensor to warmup
    print("[INFO] starting video stream...")
    temp_vs = VideoStream(src=0).start()
    analyze_video(True,temp_vs, 0)
    time.sleep(2.0)
else:
    print("[INFO] starting prerecorded video...")
    temp_vs = cv2.VideoCapture(testvideoPath)
    analyze_video(temp_vs.isOpened,temp_vs, 3)

json_analyse_path = "testfiles/analyse.json"
with open(json_analyse_path, 'a') as f:
    f.write(json.dumps(collectedDetections, separators=(',', ':')))

# save collected data to csv
i=0
while os.path.exists(f'visualization/output{i}.csv'):
    i+=1
output_file_path = f'visualization/output{i}.csv'
addCounter = 0
if not os.path.isfile(output_file_path):
    with open(output_file_path, 'a') as f:
        f.write("id,type,license_plate,date,time")
        f.write('\n')
else:
    with open(output_file_path, 'rb') as fh:
        for line in fh:
            pass
        addCounter = int(line[:line.index(",")])
with open(output_file_path, 'a') as f:
    for bboxes in collectedDetections:
        license_plate = ocr_license_plate.detect_license_plate(bboxes[0], videoFileDir)
        last = bboxes[-1]
        dt = datetime.datetime.fromtimestamp(last[1]/1000.0)
        f.write(str(bboxes[0]+addCounter) + ",car," + str(license_plate) + "," + str(dt.date()) + "," + str(dt.time())[:5])
        f.write('\n')
    #webbrowser.open('file://' + os.path.realpath("visualization/index.html"), new=2)
# do a bit of cleanup
for index, i in enumerate(collectedDetections):
    print("Car " + str(index) + ": " + str(counting_cars.globalChange(i)))
cv2.destroyAllWindows()
#temp_vs.stop()
