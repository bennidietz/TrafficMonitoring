# import the necessary packages
import os, time, datetime, webbrowser, settings
import car_counting_refPoints as counting_cars
import ocr_license_plate
from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import time
import cv2
import json
import matplotlib.pyplot as plt

'''
change this path to your project directory!
(or maybe automatically detect path somehow)
'''

# testvideoPath = settings.getBaseDir() + '/testfiles/cropAlternativ.mp4'

#testvideoPath = settings.getBaseDir() + '/testfiles/out8.mp4'
# testvideoPath = settings.getBaseDir() + '/testfiles/highway.mov'

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--confidence", type=float, default=0.47,
	help="minimum probability to filter weak detections")
ap.add_argument("-f", "--fromfile", type=str, default='/testfiles/out8.mp4',
	help="give relative path to prerecorded")
ap.add_argument("-p", "--lanepoints", type=int, default=2, metavar="[1-2]",
	help="number of necessary lane points")
ap.add_argument("-l", "--lanes", type=int, default=2, metavar="[1-10]",
	help="number of lanes")
ap.add_argument("-o", "--overlay", action='store_false',
                help="dont display the counter overlay")
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
## net = cv2.dnn.readNetFromCaffe(settings.getBaseDir() + "/models/mobileNet_cars/mobilenet_yolov3_deploy.prototxt",
##        settings.getBaseDir() + "/models/mobileNet_cars/mobilenet_yolov3_deploy_iter_63000.caffemodel")
net = cv2.dnn.readNetFromCaffe(settings.getBaseDir() + "/models/MobileNetSSD/MobileNetSSD_deploy.prototxt.txt",
        settings.getBaseDir() + "/models/MobileNetSSD/MobileNetSSD_deploy.caffemodel")
##net = cv2.dnn.readNetFromCaffe(settings.getBaseDir() + "/models/googlenet_cars/MobileNetSSD_deploy.prototxt.txt",
##        settings.getBaseDir() + "/models/googlenet_cars/googlenet_finetune_web_car_iter_10000.caffemodel")


def on_mouse(event,x,y,flags,params):
    print(x, y)

def createNewCar(bbox):
    firstRefPointMatches = list(filter(lambda n: n.rIndex == 0,bbox.matches))
    bbox.setTargetMatch(firstRefPointMatches[-1])
    detectedCars.append(counting_cars.DetectedCar([bbox]))

def cropCar(frame, bbox):
    (startX, startY, endX, endY) = bbox.astype("int")
    startY = max(0, startY)
    endY = max(0, endY)
    startX = max(0, startX)
    endX = max(0, endX)
    return frame[startY:endY, startX:endX]

def updatePlot(index, data):
    pltData[index - 1] = data
    objects = ('Lane 1', 'Lane 2')
    yPos = np.arange(len(objects))
    plt.bar(yPos, pltData, align='center', color=(0.2, 0.4, 0.6, 1))
    plt.xticks(yPos, objects)
    plt.draw()
    plt.show(False)

def appendToDetectionNewRefPoint(frame, index, bbox, videoFileDir, box, currElement):
    detectedCars[index].detectedBboxArr.append(bbox)
    bbox.matches[-1].getLane().onCarDetected()
    lane = bbox.matches[-1].getLane()
    print("Lane " + str(lane.index) + ": Car " + str(lane.counter) + " detected")

    if not live:
        updatePlot(lane.index, lane.counter)

    color = [255,  0 , 0]
    #cv2.putText(frame, str(lane.counter), (673, 279),cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    cv2.imwrite(videoFileDir + "lane%d_car%d.jpg" %  (lane.index, lane.counter),
                            cropCar(frame, box) )
    collectedDetections[lane.index-1].append([lane.counter, currElement])
    pass


def appendToDetection(index, bbox):
    if not bbox.targetMatch:
        bbox.setTargetMatch(detectedCars[index].detectedBboxArr[-1].targetMatch)
    detectedCars[index].detectedBboxArr.append(bbox)

collectedDetections = [[],[],[],[],[] ,[],[],[],[],[]]

# stop_condition: determines, when the analysis is halted (eg when the videostream is over)
# vs: the videostream (could be live from camera or from prerecorded)
# frame_skip: the amount of frames that are skipped from the video stream (to counter the lag)
def analyze_video(stop_condition,vs,frame_skip):
    counter = 0
    # counts the frames
    frame_counter = 0
    # loop over the frames from the video stream
    while stop_condition:
        # grab the frame from the threaded video stream and resize it
        # to have a maximum width of 400 pixels
        if live:
            print("frame")
            frame = vs.read()
        else:
	        ret, frame = vs.read()
	        if not ret:
	            print("[ERROR] There was an error reading the frame. Its value is:")
	            print(frame)
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
            newCar = False
            for i in np.arange(0, detections.shape[2]):
                # extract the confidence (i.e., probability) associated with
                # the prediction
                confidence = detections[0, 0, i, 2]
                idx = int(detections[0, 0, i, 1])
                name = CLASSES[idx]
                # filter out weak detections by ensuring the `confidence` is
                # greater than the minimum confidence
                if (name == "car" or name == "bus") and confidence > args["confidence"]:
                    # extract the index of the class label from the
                    # `detections`, then compute the (x, y)-coordinates of
                    # the bounding box for the object
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype("int")

                    # check if the car was already detected before
                    millis = int(round(time.time() * 1000))
                    #detectedCars.append(json.dumps([str(confidence), millis, str(startX), str(startY), str(endX), str(endY)]))

                    cropped_frame = frame[startY:endY, startX:endX]
                    bbox = counting_cars.Rectangle(startX, startY, endX, endY)
                    matches = bbox.containsAny()
                    currDetectedBBox = counting_cars.DetectedBbox(float(confidence), millis, bbox, matches)
                    if len(matches) > 0:
                        if len(matches) == 1:
                            currDetectedBBox.setTargetMatch(matches[-1]) # case of only one match found
                        index = counting_cars.sameCarInRefPoint(detectedCars, currDetectedBBox)
                        if index == -1:
                            firstRefPointMatches = list(filter(lambda n: n.rIndex == 0,currDetectedBBox.matches))
                            followingRefPointMatches = list(filter(lambda n: n.rIndex > 0,currDetectedBBox.matches))
                            if len(firstRefPointMatches) == 0:
                                # only points for second ref points are found -> assign it to its car group
                                refPointIndex = counting_cars.nextRefPointIndex(detectedCars, currDetectedBBox)
                                if refPointIndex >= 0:
                                    # is probably the following ref point
                                    #TODO: assign as 2nd ref point
                                    millis = int(round(time.time() * 1000))
                                    currElement = [float(confidence), millis, startX, startY, endX, endY, (startX+endX)/2, (startY+endY)/2]
                                    appendToDetectionNewRefPoint(frame, refPointIndex, currDetectedBBox, videoFileDir, box, currElement)
                                    newCar = True
                                # else: car was not found, so ignore the bounding box
                            elif len(firstRefPointMatches) == 1:
                                
                                if len(followingRefPointMatches) == 0:
                                    # only matches for 1st ref point -> create new car
                                    createNewCar(currDetectedBBox)
                                else:
                                    refPointIndex= counting_cars.nextRefPointIndex(detectedCars, currDetectedBBox)
                                    if refPointIndex >= 0:
                                        # is probably the following ref point
                                        #TODO: assign as 2nd ref point
                                        millis = int(round(time.time() * 1000))
                                        currElement = [float(confidence), millis, startX, startY, endX, endY, (startX+endX)/2, (startY+endY)/2]
                                        appendToDetectionNewRefPoint(frame, refPointIndex, currDetectedBBox, videoFileDir, box, currElement)
                                        newCar = True
                                    else:
                                        # only matches for 1st ref point -> create new car
                                        createNewCar(currDetectedBBox)
                            else:
                                # scenario: mutliple matches for first point -> ignore detection
                                pass
                        else:
                            # append bbox to detected (already existing) car
                            appendToDetection(index, currDetectedBBox)
                        #print(counting_cars.matchesToString(matches), millis)

                    #currCounter = None
                    #cv2.imwrite(videoFileDir + "car%d_%d.jpg" %  (currCounter, counting_cars.numberBoxes(detectedCars, currCounter)),cropped_frame )
                    # draw the
                    label = "{}: {:.2f}%".format(CLASSES[idx],confidence * 100)# + " " + str(currCounter)
                    cv2.rectangle(frame, (startX, startY), (endX, endY),COLORS[idx], 2)
                    y = startY - 15 if startY - 15 > 15 else startY + 15
                    cv2.putText(frame, label, (startX, y),cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)
        	# show the output frame
            copy = frame.copy()
            if(args["overlay"]):
                for idx in range(args["lanes"]):
                    detections = len(collectedDetections[idx])
                    cv2.putText(copy, 'lane ' + str(idx+1) + ': ' + str(detections), (50, (30*(idx+1))), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.imshow("Frame", copy)
            #cv2.setMouseCallback('Frame', on_mouse)
            frame_counter = 0
            continue
        frame_counter += 1
        key = cv2.waitKey(1) & 0xFF
        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break

detectedCars = []
live = args["fromfile"] == ""
testvideoPath = settings.getBaseDir() + args["fromfile"]

if not live:
    plt.ylabel('# cars detected')
    plt.title('Detections per lane')
    pltData = [0, 0]
    plt.show(False)

if live:
    ending = "live"
else:
    ending = settings.getEnding(testvideoPath)
if not os.path.isdir(settings.getOutputDir() + ending):
    os.makedirs(settings.getOutputDir() + ending)
videoFileDir = settings.getOutputDir() + ending + "/"

if live:
    # initialize the video stream, allow the cammera sensor to warmup
    print("[INFO] starting video stream...")
    testvideoPath = settings.getBaseDir() + ''
    temp_vs = VideoStream(src=0).start()
    counting_cars.configure_refPoints(temp_vs, args["lanes"], args["lanepoints"], live)
    analyze_video(True,temp_vs, 0)
    time.sleep(2.0)
else:
    print("[INFO] starting prerecorded video...")
    temp_vs = cv2.VideoCapture(testvideoPath)
    counting_cars.configure_refPoints(temp_vs, args["lanes"], args["lanepoints"], live)
    analyze_video(temp_vs.isOpened,temp_vs, 3)

# save collected data to csv
i=0
while os.path.exists(f'visualization/output{i}.csv'):
    i += 1
output_file_path = f'visualization/output{i}.csv'
addCounter = 0
if not os.path.isfile(output_file_path):
    with open(output_file_path, 'a') as f:
        f.write("lane_id,car_id,type,license_plate,date,time")
        f.write('\n')
with open(output_file_path, 'a') as f:
    for idx, detection in enumerate(collectedDetections, start=1):
        for bboxes in detection:
            license_plate = ocr_license_plate.detect_license_plate(bboxes[0], videoFileDir)
            last = bboxes[-1]
            dt = datetime.datetime.fromtimestamp(last[1]/1000.0)
            f.write(str(idx) + "," +str(bboxes[0]+addCounter) + ",car," + str(license_plate) + "," + str(dt.date()) + "," + str(dt.time())[:5])
            f.write('\n')

cv2.destroyAllWindows()
#temp_vs.stop()
