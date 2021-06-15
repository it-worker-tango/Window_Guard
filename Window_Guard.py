import cv2 as cv
import numpy as np
import os
import argparse
import time
import RPi.GPIO as GPIO
 
GPIO.setmode(GPIO.BCM)

# GPIO18pin
GPIO.setup(18,GPIO.IN,pull_up_down=GPIO.PUD_UP)
 
sw_status = 1

def draw_boxes(image, boxes, confidences, class_ids, idxs):
    if len(idxs) > 0:
        for i in idxs.flatten():
            # extract bounding box coordinates
            left, top = boxes[i][0], boxes[i][1]
            width, height = boxes[i][2], boxes[i][3]

            # draw bounding box and label
            cv.rectangle(image, (left, top), (left + width, top + height), (0, 255, 0))
            label = "%s: %.2f" % (classes[class_ids[i]], confidences[i])
            cv.putText(image, label, (left, top - 5), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))
    return image

def make_prediction(net, layer_names, labels, frame, conf_threshold, nms_threshold):
    boxes = []
    confidences = []
    class_ids = []
    frame_height, frame_width = frame.shape[:2]
   
    # create a blob from a frame
    blob = cv.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    outputs = net.forward(layer_names)

    # extract bounding boxes, confidences and class ids
    for output in outputs:
        for detection in output:            
            # extract the scores, class id and confidence
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
           
            # consider the predictions that are above the threshold
            if confidence > conf_threshold:
                center_x = int(detection[0] * frame_width)
                center_y = int(detection[1] * frame_height)
                width = int(detection[2] * frame_width)
                height = int(detection[3] * frame_height)

                # get top left corner coordinates
                left = int(center_x - (width / 2))
                top = int(center_y - (height / 2))
                boxes.append([left, top, width, height])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    idxs = cv.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
    return boxes, confidences, class_ids, idxs

parser = argparse.ArgumentParser()
parser.add_argument('--model', default='yolov4-tiny.weights', help='Path to a binary file of model')
parser.add_argument('--config', default='yolov4-tiny.cfg', help='Path to network configuration file')
parser.add_argument('--classes', default='coco.names', help='Path to label file')
parser.add_argument('--conf_threshold', type=float, default=0.5, help='Confidence threshold')
parser.add_argument('--nms_threshold', type=float, default=0.3, help='Non-maximum suppression threshold')
parser.add_argument('--input', help='Path to video file')
parser.add_argument('--output', default='', help='Path to directory for output video file')
args = parser.parse_args()

# load names of classes
classes = open(args.classes).read().rstrip('\n').split('\n')

# load a network
net = cv.dnn.readNet(args.config, args.model)
net.setPreferableBackend(cv.dnn.DNN_BACKEND_INFERENCE_ENGINE)
layer_names = net.getUnconnectedOutLayersNames()

cap = cv.VideoCapture(args.input)

# define the codec and create VideoWriter object
if args.output != '':
    input_file_name = os.path.basename(args.input)
    output_file_name, output_file_format = os.path.splitext(input_file_name)
    output_file_name += '-output'
    if output_file_format != '':
        fourcc = int(cap.get(cv.CAP_PROP_FOURCC))
    else:
        output_file_format = '.mp4'
        fourcc = cv.VideoWriter_fourcc(*'mp4v')

    output_file_path = args.output + output_file_name + output_file_format
    fps = cap.get(cv.CAP_PROP_FPS)
    frame_size = (int(cap.get(cv.CAP_PROP_FRAME_WIDTH)),
                  int(cap.get(cv.CAP_PROP_FRAME_HEIGHT)))

    out = cv.VideoWriter(output_file_path,
                         fourcc,
                         fps,
                         frame_size)

while cv.waitKey(1) < 0:
    try:
        sw_status = GPIO.input(18)
        if sw_status == 0:
            print("窗口已关闭")
        else:
            print("窗口处于开放状态，请注意!")
            hasFrame, frame = cap.read()
            if not hasFrame: break
            boxes, confidences, class_ids, idxs = make_prediction(net, layer_names, classes, frame, args.conf_threshold, args.nms_threshold)
            frame = draw_boxes(frame, boxes, confidences, class_ids, idxs)
            if args.output != '':
                out.write(frame)
            else:
                cv.imshow('object detection', frame)
 
        time.sleep(0.03)
    except:
        break
 
GPIO.cleanup()
print("系统运行停止")
    