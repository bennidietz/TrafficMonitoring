import os, json
from spatial_similarity import similarity
from testdata import shortTestDetection

basePath = os.path.dirname(os.path.realpath(__file__))

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

def analyseTestData(detections_array, frameWidth):
    '''
        this function takes test data in that was obtained from the real time object detection python script
        and returns a the categorized detections (each category is linked to one car)
    '''
    detected_cars = []
    for curr_index, curr_detection in enumerate(detections_array):
        detected_cars.append([detections_array[curr_index]])
        for compared_index, compared_detection in enumerate(detections_array):
            if curr_index != compared_index and similarity(curr_detection, compared_detection) > 0.1:
                detected_cars[-1].append(compared_detection)
                del detections_array[compared_index]
    return detected_cars

with open('testfiles/before.json', 'w') as f:
    json.dump(shortTestDetection, f)
with open('testfiles/after.json', 'w') as f:
    json.dump(analyseTestData(shortTestDetection, 1000), f)