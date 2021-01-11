import os, json
import spatial_similarity
from testdata import shortTestDetection

basePath = os.path.dirname(os.path.realpath(__file__))

timeDiffAcceptable = 30000

class Detection:
    def __init__(self, confidence, milliseconds, bbox):
        self.confidence = confidence
        self.milliseconds = milliseconds
        self.bbox = bbox

def analyseDectectionData(detections_array):
    '''
        this function will be later used to group the detected boundingboxes of the data
        that is coming directly from the real time object detecting
        Parameters
        ----------
        detections_array : Detection[]
        bb2 : dict
            Keys: {'x1', 'x2', 'y1', 'y2'}
            The (x, y) position is at the top left corner,
            the (x2, y2) position is at the bottom right corner

        Returns
        -------
        float
            in [0, 1]
    '''
    for detection in detections_array:
        #print(detection.confidence, detection.milliseconds, detection.bbox)
        pass

def calcMean(array):
    r, g, b = 0,0,0
    for i in array:
        r += i[0][0]
        g += i[0][1]
        b += i[0][2]
    r = r / len(array)
    g = g / len(array)
    b = b / len(array)
    return [r,g,b]

def ratio(numberA, numberB):
    if numberA == 0 or numberB == 0: return
    if numberA >= numberB:
        return numberB / float(numberA)
    else: return numberA / float(numberB)


def simMean(colorA, colorB):
    r1 = ratio(colorA[0], colorB[0])
    r2 = ratio(colorA[1], colorB[1])
    r3 = ratio(colorA[2], colorB[2])
    return (r1 + r2 + r3) / 3

def getPath(bboxes):
    shifts = []
    for i in range(1, len(bboxes)-1):
        bboxA = bboxes[i]
        bboxB = bboxes[i+1]
        shifts.append([bboxB[6]-bboxA[6], bboxB[7]-bboxA[7]])
    return shifts

def globalChange(bboxes):
    shifts = getPath(bboxes)
    if len(shifts) == 0: return [0,0]
    changeX, changeY = 0,0
    for i in shifts:
        changeX += i[0]
    for i in shifts:
        changeY += i[1]
    #return [changeX, changeY]
    return[changeX / len(shifts), changeY / len(shifts)]

def belongsToBboxes(detectedArray, bbox):
    '''
        if the bbox belongs to other bboxes (seems to be of the same car),
            the index is given back
        else
            -1 is returned
    '''
    minsim = 0
    currIndex = -1
    for index, car in enumerate(detectedArray):
        sim = similarity(car[-1], bbox)
        # print(sim)
        if sim > minsim:
            minsim = sim
            currIndex = index
    if minsim > 0.1 and currIndex != -1:
        if bbox[1]-detectedArray[currIndex][-1][1] > timeDiffAcceptable:
            # time difference too high
            return -1
        else:
            return currIndex
    else: 
        return -1

def analyseTestData(detections_array, frameWidth):
    '''
        this function takes test data in that was obtained from the real time object detection python script
        and returns a the categorized detections (each category is linked to one car)
    '''
    detected_cars = []
    for curr_index, curr_detection in enumerate(detections_array):
        detected_cars.append([detections_array[curr_index]])
        for compared_index, compared_detection in enumerate(detections_array):
            if curr_index != compared_index and spatial_similarity.similarity(curr_detection, compared_detection) > 0.1:
                detected_cars[-1].append(compared_detection)
                del detections_array[compared_index]
    return detected_cars

def numberBoxes(array, counter):
    for i in array:
        if counter == i[0] and len(i) >= 2:
            return len(i)-1

with open('testfiles/before.json', 'w') as f:
    json.dump(shortTestDetection, f)
with open('testfiles/after.json', 'w') as f:
    json.dump(analyseTestData(shortTestDetection, 1000), f)