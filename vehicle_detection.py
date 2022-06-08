# Importing packages
import cv2      # For Computer Vision
import csv      # For writing data in file
import collections  # For easy output processing
import numpy as np  # For processing images easily
import time
# import sched
# import schedule

input_size = 320

# Time calculation variables
delayTime = 1
totalVehicles = 0

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


image_files = ['./images/image_1.jpg', './images/image_2.jpg', './images/image_3.jpg', './images/image_4.jpg', './images/image_6.jpeg', './images/image_7.jpeg', './images/image_8.jpeg', './images/image_9.jpeg']


def vehicle_detector(image):
    img = cv2.imread(image)
    blob = cv2.dnn.blobFromImage(img, 1 / 255, (input_size, input_size), [0, 0, 0], 1, crop=False)

    yolo.setInput(blob)
    layersNames = yolo.getLayerNames()
    outputNames = [(layersNames[i - 1]) for i in yolo.getUnconnectedOutLayers()]

    outputs = yolo.forward(outputNames)
    processor(outputs,img)

    frequency = collections.Counter(detected_classNames)
    # print(frequency)
    global totalVehicles
    totalVehicles += frequency['car'] + frequency['motorbike'] + frequency['truck'] + frequency['bicycle'] + frequency['bus']
    # print("========================")
    # print(detected_classNames)

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


def timeAlloter():
    for i in range(len(image_files)):
        vehicle_detector(image_files[i])
    global totalVehicles
    print("Total number of vehicles =", totalVehicles)
    print("========================")
    global delayTime
    delayTime = totalVehicles/10
    totalVehicles = 0

if __name__ == '__main__':
    # for i in range(len(image_files)):
    #     vehicle_detector(image_files[i])

    # while(True):
    #     starter = thd.Timer(delayTime, function_caller)
    #     starter.start()

    # schduler = sched.scheduler()
    # schedule.every(delayTime).seconds.do(function_caller)

    while(True):
        # schedule.run_pending()
        time.sleep(delayTime)
        timeAlloter()
