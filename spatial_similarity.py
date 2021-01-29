import counting_cars

def get_iou(bb1, bb2):
    """
    source: https://stackoverflow.com/a/42874377

    Calculate the Intersection over Union (IoU) of two bounding boxes.

    Parameters
    ----------
    bb1 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x1, y1) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner
    bb2 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x, y) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner

    Returns
    -------
    float
        in [0, 1]
    """
    assert bb1['x1'] < bb1['x2']
    assert bb1['y1'] < bb1['y2']
    assert bb2['x1'] < bb2['x2']
    assert bb2['y1'] < bb2['y2']

    # determine the coordinates of the intersection rectangle
    x_left = max(bb1['x1'], bb2['x1'])
    y_top = max(bb1['y1'], bb2['y1'])
    x_right = min(bb1['x2'], bb2['x2'])
    y_bottom = min(bb1['y2'], bb2['y2'])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both AABBs
    bb1_area = (bb1['x2'] - bb1['x1']) * (bb1['y2'] - bb1['y1'])
    bb2_area = (bb2['x2'] - bb2['x1']) * (bb2['y2'] - bb2['y1'])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou

def area(bbox):
    return (bbox["x2"]-bbox["x1"]) * (bbox["y2"]-bbox["y1"])

def similarity(detection1, detection2):
    '''
        calculate the similarity of two bounding boxes
        -> the more similiar two bounding boxes are, 
        the more likely it is that they are of the same car
    '''
    #sim = counting_cars.simMean(detection1[-1], detection2[-1])
    #print(sim)
    bbox_1 = dict({"x1": detection1[2], "y1": detection1[3],
                "x2": detection1[4], "y2": detection1[5]})
    bbox_2 = dict({"x1": detection2[2], "y1": detection2[3],
                "x2": detection2[4], "y2": detection2[5]})
    area_ratio = counting_cars.ratio(area(bbox_1), area(bbox_2))
    x_sim = counting_cars.ratio((bbox_1["x1"]+bbox_1["x2"])/2, (bbox_2["x1"]+bbox_2["x2"])/2)
    y_sim = counting_cars.ratio((bbox_1["y1"]+bbox_1["y2"])/2, (bbox_2["y1"]+bbox_2["y2"])/2)
    centroid_sim = (x_sim + y_sim) / 2
    '''if get_iou(bbox_1, bbox_2) < 0.1:
        print((area_ratio+centroid_sim)/2 * get_iou(bbox_1, bbox_2))'''
    #return get_iou(bbox_1, bbox_2)
    return (area_ratio+centroid_sim)/2 * get_iou(bbox_1, bbox_2)


def trial(index, detection1, detection2):
    '''
        calculate the similarity of two bounding boxes
        -> the more similiar two bounding boxes are, 
        the more likely it is that they are of the same car
    '''
    #sim = counting_cars.simMean(detection1[-1], detection2[-1])
    #print(sim)
    bbox_1 = dict({"x1": detection1[2], "y1": detection1[3],
                "x2": detection1[4], "y2": detection1[5]})
    bbox_2 = dict({"x1": detection2[2], "y1": detection2[3],
                "x2": detection2[4], "y2": detection2[5]})
    area_ratio = counting_cars.ratio(area(bbox_1), area(bbox_2))
    x_sim = counting_cars.ratio((bbox_1["x1"]+bbox_1["x2"])/2, (bbox_2["x1"]+bbox_2["x2"])/2)
    y_sim = counting_cars.ratio((bbox_1["y1"]+bbox_1["y2"])/2, (bbox_2["y1"]+bbox_2["y2"])/2)
    centroid_sim = (x_sim + y_sim) / 2
    '''if get_iou(bbox_1, bbox_2) < 0.1:
        print((area_ratio+centroid_sim)/2 * get_iou(bbox_1, bbox_2))'''
    #return get_iou(bbox_1, bbox_2)
    return (area_ratio+centroid_sim)/2 * get_iou(bbox_1, bbox_2)