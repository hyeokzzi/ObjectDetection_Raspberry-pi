import cv2
import numpy as np

cam1 = cv2.VideoCapture(0)
cam2 = cv2.VideoCapture(1)
cam1.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cam1.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cam2.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cam2.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
YOLO_net = cv2.dnn.readNet("yolov2-tiny.weights","yolov2-tiny.cfg")

classes_1 = []
classes_2 = []
with open("yolo.names", "r") as f:
    classes_1 = [line.strip() for line in f.readlines()]
with open("yolo.names", "r") as f:
    classes_2 = [line.strip() for line in f.readlines()]
layer_names = YOLO_net.getLayerNames()
output_layers_1 = [layer_names[i - 1] for i in YOLO_net.getUnconnectedOutLayers()]
output_layers_2 = [layer_names[i - 1] for i in YOLO_net.getUnconnectedOutLayers()]

while True:
    ret, frame = cam1.read()
    h, w, c = frame.shape
    ret2, frame2 = cam2.read()
    h2, w2, c2 = frame2.shape
    
    blob1 = cv2.dnn.blobFromImage(frame, 0.00392, (208, 208), (0, 0, 0),
    True, crop=False)
    YOLO_net.setInput(blob1)
    outs_1 = YOLO_net.forward(output_layers_1)

    class_ids_1 = []
    confidences_1 = []
    boxes_1 = []
    #2
    blob2 = cv2.dnn.blobFromImage(frame2, 0.00392, (208, 208), (0, 0, 0),
    True, crop=False)
    YOLO_net.setInput(blob2)
    outs_2 = YOLO_net.forward(output_layers_2)

    class_ids_2 = []
    confidences_2 = []
    boxes_2 = []

    for out in outs_1:

        for detection in out:

            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.3:
                # Object detected
                center_x = int(detection[0] * w)
                center_y = int(detection[1] * h)
                dw = int(detection[2] * w)
                dh = int(detection[3] * h)
                # Rectangle coordinate
                x = int(center_x - dw / 2)
                y = int(center_y - dh / 2)
                boxes_1.append([x, y, dw, dh])
                confidences_1.append(float(confidence))
                class_ids_1.append(class_id)
                
                
    for out in outs_2:

        for detection in out:

            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.3:
                # Object detected
                center_x = int(detection[0] * w2)
                center_y = int(detection[1] * h2)
                dw = int(detection[2] * w2)
                dh = int(detection[3] * h2)
                # Rectangle coordinate
                x = int(center_x - dw / 2)
                y = int(center_y - dh / 2)
                boxes_2.append([x, y, dw, dh])
                confidences_2.append(float(confidence))
                class_ids_2.append(class_id)
                
    indexes_1 = cv2.dnn.NMSBoxes(boxes_1, confidences_1, 0.45, 0.4)
    indexes_2 = cv2.dnn.NMSBoxes(boxes_2, confidences_2, 0.45, 0.4)

    for i in range(len(boxes_1)):
        if i in indexes_1:
            x, y, w, h = boxes_1[i]
            label = str(classes_1[class_ids_1[i]])
            score = confidences_1[i]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 5)
            cv2.putText(frame, label, (x, y - 20), cv2.FONT_ITALIC, 0.5, 
            (255, 255, 255), 1)
	
    for i in range(len(boxes_2)):
        if i in indexes_2:
            x, y, w2, h2 = boxes_2[i]
            label_2 = str(classes_2[class_ids_2[i]])
            score = confidences_2[i]
            cv2.rectangle(frame2, (x, y), (x + w2, y + h2), (0, 0, 255), 5)
            cv2.putText(frame2, label_2, (x, y - 20), cv2.FONT_ITALIC, 0.5, 
            (255, 255, 255), 1)
	
    cv2.imshow("YOLOv3_picam", frame)
    cv2.imshow("YOLOv3_webcam", frame2)
    if cv2.waitKey(1) > 0:
        break
