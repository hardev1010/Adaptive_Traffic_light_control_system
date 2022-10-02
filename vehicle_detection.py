# Importing packages
import cv2      # For Computer Vision
import csv      # For writing data in file
import collections  # For easy output processing
import numpy as np  # For processing images easily
import time
import math

input_size = 320

# Time calculation variables
delayTime = 1
totalVehicles = 0
totalAvaliableTime = 180
singleLaneNum = 0

# thresholds
confThreshold = 0.4
nmsThreshold = 0.4

classes = "coco.names"
classNames = open(classes).read().strip().split('\n')

required_class_index = [1, 2, 3, 5, 7] # This is 0 index based (check coco names file)
detected_classNames = []

# model and weight files
modelConfiguration = 'yolov3-320.cfg'
modelWeigheights = 'yolov3-320.weights'


# configure yolo3
yolo = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeigheights)

def processor(outputs,img):
    global detected_classNames
    detected_classNames = []
    height, width = img.shape[:2]
    boxes = []
    classIds = []
    confidence_scores = []
    for output in outputs:
        for detected in output:
            scores = detected[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if classId in required_class_index:
                if confidence > confThreshold:
                    w,h = int( detected[2] * width ), int( detected[3] * height )
                    x,y = int( ( detected[0] * width ) - w/2 ), int( ( detected[1] * height ) - h/2 )
                    boxes.append([x,y,w,h])
                    classIds.append(classId)
                    confidence_scores.append(float(confidence))

    # Apply Non-Max Suppression
    indices = cv2.dnn.NMSBoxes(boxes, confidence_scores, confThreshold, nmsThreshold)

    if len(indices) > 0:
        for i in indices.flatten():
            name = classNames[classIds[i]]
            detected_classNames.append(name)


image_files = [
    ['./images/set1/image_1.jpg', './images/set1/image_2.jpg', './images/set1/image_3.jpg', './images/set1/image_4.jpg'],
    ['./images/set2/image_1.jpg', './images/set2/image_2.jpg', './images/set2/image_3.jpg', './images/set2/image_4.jpg'],
    ['./images/set3/image_1.jpg', './images/set3/image_2.jpg', './images/set3/image_3.jpg', './images/set3/image_4.jpg'],
]


def vehicle_detector(image):
    img = cv2.imread(image)
    blob = cv2.dnn.blobFromImage(img, 1 / 255, (input_size, input_size), [0, 0, 0], 1, crop=False)

    yolo.setInput(blob)
    layersNames = yolo.getLayerNames()
    outputNames = [(layersNames[i - 1]) for i in yolo.getUnconnectedOutLayers()]

    outputs = yolo.forward(outputNames)
    processor(outputs,img)

    frequency = collections.Counter(detected_classNames)
    global singleLaneNum
    singleLaneNum = frequency['car'] + frequency['motorbike'] + frequency['truck'] + frequency['bicycle'] + frequency['bus']
    global totalVehicles
    totalVehicles += singleLaneNum

    # save the data to a csv file
    with open("data.csv", 'a') as f1:
        cwriter = csv.writer(f1)
        cwriter.writerow("================================================")
        cwriter.writerow([image," Information ->"])
        cwriter.writerow(["Cars -> " + str(frequency['car'])])
        cwriter.writerow(["Bikes -> " + str(frequency['motorbike'])])
        cwriter.writerow(["bicycle -> " + str(frequency['bicycle'])])
        cwriter.writerow(["Buses -> " + str(frequency['bus'])])
        cwriter.writerow(["Trucks -> " + str(frequency['truck'])])
    f1.close()    

def timeCalcutor(laneNums):
    # 0=North, 1=south, 2=east, 3=west
    allotedTime = []
    global totalAvaliableTime
    totalTime = totalAvaliableTime
    totalCars = 0
    for laneNum in laneNums:  # For edge case of number of vehicles
        if(laneNum <= 15):
            allotedTime.append(15)
            totalTime -= 45
        elif(laneNum >= 40):
            allotedTime.append(40)
            totalTime -= 45
        else:
            allotedTime.append(0)
            totalCars += laneNum

    print(laneNums)
    if(totalCars == 0):     # All lanes have been alloted the time already
        print("Time allotment -> ")
        print(allotedTime)
        return allotedTime

    timePerCar = totalTime/totalCars
    if(timePerCar > 1.5):
        timePerCar = 1.4    # To make time allotment more reasonable for less amount of cars

    for i in range(len(laneNums)):
        if(allotedTime[i] == 0):
            allotedTime[i] = math.ceil(timePerCar * laneNums[i])

    print("Time allotment -> ")
    print(allotedTime)
    return allotedTime

def timeUpdater():
    for i in range(4):      # For testing
        laneVehiclesNum = []
        for j in range(4):
            vehicle_detector(image_files[i][j])
            laneVehiclesNum.append(singleLaneNum)
        # print(laneVehiclesNum)
        global totalVehicles
        print("Total number of vehicles =", totalVehicles)
        global delayTime
        timeNums = timeCalcutor(laneVehiclesNum)
        # As we will be getting a delay of 2.5 to 2.8 seconds so we subtracted 3 seconds to compansate for it
        delayTime = sum(timeNums) - 3
        print("Delay Time =", delayTime)
        totalVehicles = 0
        print("========================")

if __name__ == '__main__':
    while(True):
        time.sleep(delayTime)
        timeUpdater()