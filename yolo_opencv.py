# py ./yolo_opencv.py --image .\market.jpg --config yolov3.cfg --weights yolov3.weights --classes yolov3.txt

import cv2
import argparse
import numpy as np
import time

ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', required=True,
                help = 'path to input image')
ap.add_argument('-c', '--config', required=True,
                help = 'path to yolo config file')
ap.add_argument('-w', '--weights', required=True,
                help = 'path to yolo pre-trained weights')
ap.add_argument('-cl', '--classes', required=True,
                help = 'path to text file containing class names')
args = ap.parse_args()

image = cv2.imread(args.image)

Width = image.shape[1]
Height = image.shape[0]
scale = 0.00392

classes = None
with open(args.classes, 'r') as f:
    classes = [line.strip() for line in f.readlines()]

net = cv2.dnn.readNet(args.weights, args.config)

start_time = time.time()

blob = cv2.dnn.blobFromImage(image, scale, (416,416), (0,0,0), True, crop=False)
net.setInput(blob)

layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
outs = net.forward(output_layers)

class_ids = []
confidences = []
boxes = []

for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.5:
            center_x = int(detection[0] * Width)
            center_y = int(detection[1] * Height)
            w = int(detection[2] * Width)
            h = int(detection[3] * Height)
            x = center_x - w / 2
            y = center_y - h / 2
            class_ids.append(class_id)
            confidences.append(float(confidence))
            boxes.append([x, y, w, h])

conf_threshold = 0.5
nms_threshold = 0.4

indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

color = (0, 255, 0)

for i in indices:
    i = i[0]
    box = boxes[i]
    x = round(box[0])
    y = round(box[1])
    w = round(box[2])
    h = round(box[3])
    confidence = round(confidences[i], 3)
    class_id = class_ids[i]
    # draw prediction
    label = str(classes[class_id] + " (" + str(confidence) + ")")
    cv2.rectangle(image, (x,y), (x + w,y + h), color, 1)
    cv2.putText(image, label, (x-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

elapsed_time = time.time() - start_time
print("Image computed in: " + str(round(elapsed_time * 1000)) + " ms")

cv2.imshow("object detection", image)
cv2.waitKey()
cv2.destroyAllWindows()
