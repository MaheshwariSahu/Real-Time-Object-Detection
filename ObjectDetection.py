import cv2
import numpy as np

# Detection thresholds
thres = 0.45  # Confidence threshold
nms_threshold = 0.2  # Non-max suppression threshold

# Initialize webcam
cap = cv2.VideoCapture(0)

# Load class names from coco.names
classNames = []
classFile = 'coco.names'
with open(classFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')  # Fixed line ending

# Paths to model files
configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = 'frozen_inference_graph.pb'

# Load the pre-trained model
net = cv2.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

# Main loop
while True:
    success, img = cap.read()
    if not success or img is None:
        print("Failed to read from camera.")
        break

    classIds, confs, bbox = net.detect(img, confThreshold=thres)

    if len(classIds) != 0:
        bbox = list(bbox)
        confs = list(np.array(confs).reshape(1, -1)[0])
        confs = list(map(float, confs))

        indices = cv2.dnn.NMSBoxes(bbox, confs, thres, nms_threshold)

        if len(indices) > 0:
            for idx in indices:
                i = int(idx)
                box = bbox[i]
                classId = int(classIds[i])  # Make sure it's an int

                # Validate classId range
                if 0 < classId <= len(classNames):
                    label = classNames[classId - 1].upper()
                else:
                    label = "UNKNOWN"

                # Draw box and label
                cv2.rectangle(img, box, color=(0, 255, 0), thickness=2)
                cv2.putText(img, label, (box[0] + 10, box[1] + 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)

    cv2.imshow('Output', img)

    # Break loop on key press (ESC)
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
