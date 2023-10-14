import cv2
import numpy as np
import time

CAMERA_ID = -1

cam1= cv2.VideoCapture(CAMERA_ID)
if VideoSignal.isOpened() == False: # 카메라 정상상태 확인
    print(f'Can\'t open the Camera({CAMERA_ID})')
    exit()
cam1.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cam1.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
YOLO_net = cv2.dnn.readNet("yolov2-tiny.weights","yolov2-tiny.cfg")

classes = []
with open("yolo.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = YOLO_net.getLayerNames()
output_layers = [layer_names[i - 1] for i in YOLO_net.getUnconnectedOutLayers()]


prev_time = 0
total_frames = 0
start_time = time.time()

while True:
    curr_time = time.time()

    ret, frame = cam1.read()
    total_frames = total_frames + 1

    term = curr_time - prev_time
    fps = 1 / term
    prev_time = curr_time
    fps_string = f'term = {term:.3f},  FPS = {fps:.2f}'
    print(fps_string)

    cv2.putText(frame, fps_string, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255))
    h, w, c = frame.shape

    blob = cv2.dnn.blobFromImage(frame, 0.00392, (208, 208), (0, 0, 0),
    True, crop=False)
    YOLO_net.setInput(blob)
    outs = YOLO_net.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []

    for out in outs:

        for detection in out:

            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.5:
                # Object detected
                center_x = int(detection[0] * w)
                center_y = int(detection[1] * h)
                dw = int(detection[2] * w)
                dh = int(detection[3] * h)
                # Rectangle coordinate
                x = int(center_x - dw / 2)
                y = int(center_y - dh / 2)
                boxes.append([x, y, dw, dh])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.45, 0.4)


    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            score = confidences[i]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 5)
            cv2.putText(frame, label, (x, y - 20), cv2.FONT_ITALIC, 0.5, 
            (255, 255, 255), 1)

    cv2.imshow("YOLOv3", frame)

    if cv2.waitKey(1) > 0:
        break
