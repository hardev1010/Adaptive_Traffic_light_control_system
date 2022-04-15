# Importing packages
import cv2      # For Computer Vision
import csv      # For writing data in file
import collections  # For easy output processing
import numpy as np  # For processing images easily


# declare favourable size of image so that it can be processed by the YOLO
input_size = 320


# Detection confidence threshold
confThreshold = 0.4
nmsThreshold = 0.4


font_color = (255, 255, 255)
font_size = 0.7
font_thickness = 2


# Store Coco Names in a list
classes = "coco.names"
classNames = open(classes).read().strip().split('\n')
# print(classNames)


# class index for our required detection classes
required_class_index = [1, 2, 3, 5, 7] # This is 0 index based (check coco names file)
detected_classNames = []


## Model Files
modelConfiguration = 'yolov3-320.cfg'
modelWeigheights = 'yolov3-320.weights'


# configure the network model
yolo = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeigheights)


# Define random colour for each class
colors = np.random.randint(0, 255, size=(len(classNames), 3), dtype='uint8')


# Function for finding the detected objects from the network output
def processor(outputs,img):
    global detected_classNames
    detected_classNames = []
    height, width = img.shape[:2]
    boxes = []
    classIds = []
    confidence_scores = []
    detection = []
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
    # print(classIds)
    if len(indices) > 0:
        for i in indices.flatten():
            x, y, w, h = boxes[i][0], boxes[i][1], boxes[i][2], boxes[i][3]

            color = [int(c) for c in colors[classIds[i]]]
            name = classNames[classIds[i]]
            detected_classNames.append(name)

            # Draw classname and confidence score 
            cv2.putText(img,f'{name.upper()} {int(confidence_scores[i]*100)}%',
                    (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            # Draw bounding rectangle
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 1)
            detection.append([x, y, w, h, required_class_index.index(classIds[i])])


image_files = ['leaf_3.jpg', 'images3_2.jpg', 'images4_2.jpg', 'images5_2.jpg']


def from_static_image(image):
    img = cv2.imread(image)
    blob = cv2.dnn.blobFromImage(img, 1 / 255, (input_size, input_size), [0, 0, 0], 1, crop=False)

    # Set the input of the network
    yolo.setInput(blob)
    layersNames = yolo.getLayerNames()
    outputNames = [(layersNames[i - 1]) for i in yolo.getUnconnectedOutLayers()]
    # Feed data to the network
    outputs = yolo.forward(outputNames)

    # Find the objects from the network output
    processor(outputs,img)

    # count the frequency of detected classes
    frequency = collections.Counter(detected_classNames)
    print(frequency)
    # print(detected_classNames)

    # Draw counting texts in the frame
    cv2.putText(img, "Car:        " + str(frequency['car']), (20, 40), cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, font_thickness)
    cv2.putText(img, "Motorbike:  " + str(frequency['motorbike']), (20, 60), cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, font_thickness)
    cv2.putText(img, "Bicycle :    " + str(frequency['bicycle']), (20, 80), cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, font_thickness)
    cv2.putText(img, "Bus:        " + str(frequency['bus']), (20, 80), cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, font_thickness)
    cv2.putText(img, "Truck:      " + str(frequency['truck']), (20, 100), cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, font_thickness)

    cv2.imshow("image", img)
    cv2.waitKey(0)

    # save the data to a csv file
    with open("static-data.csv", 'a') as f1:
        cwriter = csv.writer(f1)
        cwriter.writerow("================================================")
        cwriter.writerow([image," Information ->"])
        cwriter.writerow(["Cars -> " + str(frequency['car'])])
        cwriter.writerow(["Bikes -> " + str(frequency['motorbike']) ])
        cwriter.writerow(["Buses -> " + str(frequency['bus'])])
        cwriter.writerow(["Trucks -> " + str(frequency['truck'])])
    f1.close()


if __name__ == '__main__':
    for i in range(len(image_files)):
        from_static_image(image_files[i])