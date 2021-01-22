import os

basePath = os.path.dirname(os.path.realpath(__file__))
outputDir = basePath + "/output/"

def getBaseDir(): return basePath

def getOutputDir(): return outputDir

def getEnding(string):
    return string[string.rindex("/")+1:]

def printFileCreated(filename):
    print("New file was created at: " + getEnding(filename) + "...")

def printFileAlreadyExists(filename):
    print("The file " + getEnding(filename) + " already exists.")
