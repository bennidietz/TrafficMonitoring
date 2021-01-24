import os, json
import itertools, math
import spatial_similarity

basePath = os.path.dirname(os.path.realpath(__file__))

class DetectedCar:
    def __init__(self, detectedBboxArr):
        newid = next(itertools.count())
        self.id = newid
        self.detectedBboxArr = detectedBboxArr

class DetectedBbox:
    def __init__(self, confidence, milliseconds, bbox, matches=[]):
        self.confidence = confidence
        self.milliseconds = milliseconds
        self.bbox = bbox
        self.matches = matches
        self.targetMatch = None

    def timeDiff(self, secondDetectedBbox):
        return math.fabs(secondDetectedBbox.milliseconds - self.milliseconds)

    def setTargetMatch(self, match):
        self.targetMatch = match

class Point:
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

laneAPoint1 = Point(799, 305)
laneAPoint12 = Point(583, 298)
laneAPoint2 = Point(449, 278)

laneBPoint1 = Point(204, 267)
laneBPoint2 = Point(670, 377)

laneA = [laneAPoint1, laneAPoint2]
laneB = [laneBPoint1, laneBPoint2]
allLanes = [laneA, laneB]

class Match:
    def __init__(self, lIndex, rIndex):
        self.lIndex = lIndex
        self.rIndex = rIndex

    def getLane(self):
        return allLanes[self.lIndex]

    def getPoint(self):
        return self.getLane()[self.rIndex]

    def numberPoints(self):
        return len(self.getLane())

    def __str__(self):
     return "Lane " + str(self.lIndex) + " - Point " + str(self.rIndex)

class Rectangle:
    def __init__(self, startX, startY, endX, endY):
        self.startX = startX
        self.startY = startY
        self.endX = endX
        self.endY = endY

    def contains(self, point):
        return point.X >= self.startX and point.X <= self.endX and \
            point.Y >= self.startY and point.Y <= self.endY
    
    def containsAny(self):
        results = [] 
        for lIndex, lane in enumerate(allLanes):
            for rIndex, refPoint in enumerate(lane):
                if (self.contains(refPoint)):
                    results.append(Match(lIndex, rIndex))
        return results

def numberBoxes(array, counter):
    for i in array:
        if counter == i[0] and len(i) >= 2:
            return len(i)-1

def refPointInBbox(rectangle, point):
    return rectangle.contains(point)

def matchesToString(matchesArray):
    return list(map(lambda n: str(n),matchesArray))

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

def sameCarInRefPoint(detectedArray, bbox):
    '''
        if the bbox belongs to other bboxes (seems to be of the same car),
            the index is given back
        else
            -1 is returned
    '''
    timeDiffAcceptable = 3000
    minsim = 0
    currIndex = -1
    for index, car in enumerate(detectedArray):
        compareRect = car.detectedBboxArr[-1]
        sim = spatial_similarity.similarityBBoxObject(compareRect, bbox)
        if sim > minsim and compareRect.timeDiff(bbox) < timeDiffAcceptable:
            minsim = sim
            currIndex = index
    if minsim > 0.2 and currIndex != -1:
        comparBbox = detectedArray[currIndex].detectedBboxArr[-1]
        if bbox.timeDiff(comparBbox) > timeDiffAcceptable:
            # time difference too high
            print(bbox.timeDiff(comparBbox))
            return -1
        else:
            return currIndex
    else:
        return -1

r = Rectangle(400,200,600,400)
print(r.containsAny())