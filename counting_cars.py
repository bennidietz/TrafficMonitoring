import os, json

basePath = os.path.dirname(os.path.realpath(__file__))

class Detection:
    def __init__(self, confidence, milliseconds, bbox):
        self.confidence = confidence
        self.milliseconds = milliseconds
        self.bbox = bbox

def numberBoxes(array, counter):
    for i in array:
        if counter == i[0] and len(i) >= 2:
            return len(i)-1
