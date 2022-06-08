# Importing packages
import cv2      # For Computer Vision
import csv      # For writing data in file
import collections  # For easy output processing
import numpy as np  # For processing images easily


input_size = 320
confThreshold = 0.4
nmsThreshold = 0.4


font_color = (255, 255, 255)
font_size = 0.7
font_thickness = 2
font_style = cv2.FONT_HERSHEY_SIMPLEX


# Store Coco Names in a list
classes = "coco.names"
classNames = open(classes).read().strip().split('\n')


required_class_index = [1, 2, 3, 5, 7] # This is 0 index based (check coco names file)
detected_classNames = []


# model and weight files
yoloConfig = 'yolov3-320.cfg'
yoloWeights = 'yolov3-320.weights'


# configure yolo3
yolo = cv2.dnn.readNet(yoloConfig, yoloWeights)

colors = np.random.randint(0, 255, size=(len(classNames), 3), dtype='uint8')

def processor(outputs,img):
    global detected_classNames
    detected_classNames = []
    height, width = img.shape[:2]
    boxes = []
    classIds = []
    confidence_scores = []
    # detection = []
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

    # Non-Max Suppression
    indices = cv2.dnn.NMSBoxes(boxes, confidence_scores, confThreshold, nmsThreshold)
    # print(classIds)

    if len(indices) > 0:
        for i in indices.flatten():
            x, y, w, h = boxes[i][0], boxes[i][1], boxes[i][2], boxes[i][3]

            name = classNames[classIds[i]]
            detected_classNames.append(name)

            cv2.putText(img,f'{name.upper()} {int(confidence_scores[i]*100)}%',
                    (x, y-10), font_style, 0.5, [100,20,20], 2)

            cv2.rectangle(img, (x, y), (x + w, y + h), [20,20,200], 1)
            # detection.append([x, y, w, h, required_class_index.index(classIds[i])])


# image_files = ['./images/image_1.jpg', './images/image_2.jpg', './images/image_3.jpg', './images/image_4.jpg', './images/image_7.jpeg']
image_files = ['./images/image_1.jpg', './images/image_2.jpg', './images/image_3.jpg', './images/image_4.jpg', './images/image_5.jpeg', './images/image_6.jpeg', './images/image_7.jpeg', './images/image_8.jpeg', './images/image_9.jpeg']
# image_files = ["./images/vid_5_25100.jpg","./images/vid_5_25120.jpg","./images/vid_5_25140.jpg","./images/vid_5_25160.jpg","./images/vid_5_25180.jpg","./images/vid_5_25200.jpg","./images/vid_5_25220.jpg","./images/vid_5_25240.jpg","./images/vid_5_25260.jpg","./images/vid_5_26320.jpg","./images/vid_5_26400.jpg","./images/vid_5_26420.jpg","./images/vid_5_26560.jpg","./images/vid_5_26580.jpg","./images/vid_5_26600.jpg","./images/vid_5_26620.jpg","./images/vid_5_26640.jpg","./images/vid_5_26660.jpg","./images/vid_5_26680.jpg","./images/vid_5_26700.jpg","./images/vid_5_26720.jpg","./images/vid_5_26740.jpg","./images/vid_5_26760.jpg","./images/vid_5_26780.jpg","./images/vid_5_26800.jpg","./images/vid_5_26820.jpg","./images/vid_5_26840.jpg","./images/vid_5_26860.jpg","./images/vid_5_26880.jpg","./images/vid_5_26900.jpg","./images/vid_5_26920.jpg","./images/vid_5_26940.jpg","./images/vid_5_26960.jpg","./images/vid_5_26980.jpg","./images/vid_5_27240.jpg","./images/vid_5_27260.jpg","./images/vid_5_27280.jpg","./images/vid_5_27300.jpg","./images/vid_5_27320.jpg","./images/vid_5_27360.jpg","./images/vid_5_27380.jpg","./images/vid_5_27400.jpg","./images/vid_5_27420.jpg","./images/vid_5_27440.jpg","./images/vid_5_27460.jpg","./images/vid_5_27480.jpg","./images/vid_5_27500.jpg","./images/vid_5_27520.jpg","./images/vid_5_27540.jpg","./images/vid_5_27560.jpg","./images/vid_5_27580.jpg","./images/vid_5_27600.jpg","./images/vid_5_27620.jpg","./images/vid_5_27640.jpg","./images/vid_5_27660.jpg","./images/vid_5_27680.jpg","./images/vid_5_27700.jpg","./images/vid_5_27720.jpg","./images/vid_5_27740.jpg","./images/vid_5_27760.jpg","./images/vid_5_27780.jpg","./images/vid_5_27800.jpg","./images/vid_5_27820.jpg","./images/vid_5_27840.jpg","./images/vid_5_27860.jpg","./images/vid_5_27880.jpg","./images/vid_5_27900.jpg","./images/vid_5_27920.jpg","./images/vid_5_27940.jpg","./images/vid_5_27960.jpg","./images/vid_5_27980.jpg","./images/vid_5_28000.jpg","./images/vid_5_28020.jpg","./images/vid_5_28040.jpg","./images/vid_5_28060.jpg","./images/vid_5_28080.jpg","./images/vid_5_28180.jpg","./images/vid_5_28260.jpg","./images/vid_5_28320.jpg","./images/vid_5_28340.jpg","./images/vid_5_28360.jpg","./images/vid_5_28380.jpg","./images/vid_5_28420.jpg","./images/vid_5_28440.jpg","./images/vid_5_28460.jpg","./images/vid_5_28480.jpg","./images/vid_5_28500.jpg","./images/vid_5_28520.jpg","./images/vid_5_28540.jpg","./images/vid_5_28560.jpg","./images/vid_5_28580.jpg","./images/vid_5_28600.jpg","./images/vid_5_28620.jpg","./images/vid_5_28640.jpg","./images/vid_5_28660.jpg","./images/vid_5_28680.jpg","./images/vid_5_28700.jpg","./images/vid_5_29000.jpg","./images/vid_5_29020.jpg","./images/vid_5_29040.jpg","./images/vid_5_29060.jpg","./images/vid_5_29080.jpg","./images/vid_5_29100.jpg","./images/vid_5_29400.jpg","./images/vid_5_29420.jpg","./images/vid_5_29440.jpg","./images/vid_5_29460.jpg","./images/vid_5_29480.jpg","./images/vid_5_29500.jpg","./images/vid_5_29520.jpg","./images/vid_5_29540.jpg","./images/vid_5_29560.jpg","./images/vid_5_29580.jpg","./images/vid_5_29600.jpg","./images/vid_5_29620.jpg","./images/vid_5_29640.jpg","./images/vid_5_29720.jpg","./images/vid_5_29740.jpg","./images/vid_5_29760.jpg","./images/vid_5_29820.jpg","./images/vid_5_29840.jpg","./images/vid_5_29860.jpg","./images/vid_5_29880.jpg","./images/vid_5_29900.jpg","./images/vid_5_29980.jpg","./images/vid_5_30000.jpg","./images/vid_5_30020.jpg","./images/vid_5_30040.jpg","./images/vid_5_30120.jpg","./images/vid_5_30140.jpg","./images/vid_5_30160.jpg","./images/vid_5_30180.jpg","./images/vid_5_30640.jpg","./images/vid_5_30660.jpg","./images/vid_5_30680.jpg","./images/vid_5_30700.jpg","./images/vid_5_30720.jpg","./images/vid_5_30740.jpg","./images/vid_5_30760.jpg","./images/vid_5_30820.jpg","./images/vid_5_30840.jpg","./images/vid_5_30860.jpg","./images/vid_5_30920.jpg","./images/vid_5_30940.jpg","./images/vid_5_31020.jpg","./images/vid_5_31040.jpg","./images/vid_5_31060.jpg","./images/vid_5_31080.jpg","./images/vid_5_31100.jpg","./images/vid_5_31120.jpg","./images/vid_5_31140.jpg","./images/vid_5_31160.jpg","./images/vid_5_31180.jpg","./images/vid_5_31200.jpg","./images/vid_5_31260.jpg","./images/vid_5_31280.jpg","./images/vid_5_31300.jpg","./images/vid_5_31360.jpg","./images/vid_5_31380.jpg","./images/vid_5_31400.jpg","./images/vid_5_31420.jpg","./images/vid_5_31480.jpg","./images/vid_5_31500.jpg","./images/vid_5_31520.jpg","./images/vid_5_31560.jpg","./images/vid_5_31600.jpg","./images/vid_5_31620.jpg","./images/vid_5_31640.jpg","./images/vid_5_31660.jpg","./images/vid_5_31680.jpg","./images/vid_5_31700.jpg","./images/vid_5_31720.jpg"]

def vehicle_detector(image):
    img = cv2.imread(image)
    blob = cv2.dnn.blobFromImage(img, 1 / 255, (input_size, input_size), [0, 0, 0], 1, crop=False)

    # Set the input of the network
    yolo.setInput(blob)
    layersNames = yolo.getLayerNames()
    outputNames = [(layersNames[i - 1]) for i in yolo.getUnconnectedOutLayers()]

    outputs = yolo.forward(outputNames)
    processor(outputs,img)

    frequency = collections.Counter(detected_classNames)
    print(frequency)
    # print(detected_classNames)

    # Draw counts
    cv2.putText(img, "Car:        " + str(frequency['car']), (20, 40), font_style, font_size, font_color, font_thickness)
    cv2.putText(img, "Motorbike:  " + str(frequency['motorbike']), (20, 60), font_style, font_size, font_color, font_thickness)
    cv2.putText(img, "Bicycle :    " + str(frequency['bicycle']), (20, 80), font_style, font_size, font_color, font_thickness)
    cv2.putText(img, "Bus:        " + str(frequency['bus']), (20, 100), font_style, font_size, font_color, font_thickness)
    cv2.putText(img, "Truck:      " + str(frequency['truck']), (20, 120), font_style, font_size, font_color, font_thickness)

    cv2.imshow("image", img)
    cv2.waitKey(0)

    with open("static-data.csv", 'a') as f1:
        cwriter = csv.writer(f1)
        cwriter.writerow("================================================")
        cwriter.writerow([image," Information ->"])
        cwriter.writerow(["Cars -> " + str(frequency['car'])])
        cwriter.writerow(["Bikes -> " + str(frequency['motorbike']) ])
        cwriter.writerow(["bicycle -> " + str(frequency['bicycle'])])
        cwriter.writerow(["Buses -> " + str(frequency['bus'])])
        cwriter.writerow(["Trucks -> " + str(frequency['truck'])])
    f1.close()


if __name__ == '__main__':
    for i in range(len(image_files)):
        vehicle_detector(image_files[i])