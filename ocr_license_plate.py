# import the necessary packages
import os
from pyimagesearch.anpr import PyImageSearchANPR
from imutils import paths
import argparse
import imutils
import cv2

def analyse_image(imagePath,anpr):
	# load the input image from disk and resize it
    image = cv2.imread(imagePath)
    image = imutils.resize(image, width=600)

	# apply automatic license plate recognition
    (lpText, lpCnt) = anpr.find_and_ocr(image, psm=7, clearBorder=-1 > 0)

	# only continue if the license plate was successfully OCR'd
    if lpText is not None and lpCnt is not None and len(lpText)>5:
        lpText = lpText[:-2]
        print("[INFO] " + lpText)
        return lpText
    return ""


def detect_license_plate(id, dir):
    anpr = PyImageSearchANPR(debug=-1 > 0)
    all_paths = imagePaths = sorted(list(paths.list_images(dir)))
    img_paths = [k for k in all_paths if ('car'+str(id)) in k]
    detectedLicensePlates = []
    for path in img_paths:
        result = analyse_image(path,anpr)
        if result != "":
            detectedLicensePlates.append(result)
    if len(detectedLicensePlates) == 0:
        return ""
    return detectedLicensePlates