# py ./yolo_webcam.py --config models/yolov3-face.cfg --weights models/yolov3-face.weights --classes models/yolov3-face.classes

import cv2
import argparse
import numpy as np
import time

ap = argparse.ArgumentParser()
ap.add_argument('-c', '--config', required=True,
                help = 'path to yolo config file')
ap.add_argument('-w', '--weights', required=True,
                help = 'path to yolo pre-trained weights')
ap.add_argument('-cl', '--classes', required=True,
                help = 'path to text file containing class names')
args = ap.parse_args()

# read classes
classes = None
with open(args.classes, 'r') as f:
    classes = [line.strip() for line in f.readlines()]

# create net
net = cv2.dnn.readNet(args.weights, args.config)

# boxes color
color = (0, 255, 0)

# scale value
scale = 0.00392 # it is the result of  1/255.0

# create video capture
cap = cv2.VideoCapture(0)
ret = True

while(ret):

    start_time = time.time()

    # Capture frame-by-frame
    ret, frame = cap.read()

    end_time = time.time()
    print("Fame acquired in {:.5} seconds".format(end_time - start_time))

    start_time = time.time()

    Width = frame.shape[1]
    Height = frame.shape[0]

    blob = cv2.dnn.blobFromImage(frame, scale, (416,416), (0,0,0), True, crop=False)
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
        cv2.rectangle(frame, (x,y), (x + w,y + h), color, 1)
        cv2.putText(frame, label, (x-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    end_time = time.time()
    print("Fame computed in {:.5} seconds".format(end_time - start_time))

    # Display the resulting frame
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
