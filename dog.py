import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import openvino as ov
import ipywidgets as widgets

model_name = "v3-small_224_1.0_float"
model_xml_path = f'{model_name}.xml'
model_bin_path = f'{model_name}.bin'

core = ov.Core()
device = widgets.Dropdown(
   options=core.available_devices + ["AUTO"],
   value='AUTO',
   description='Device:',
   disabled=False,
)
model = core.read_model(model=model_xml_path)#model_xml_path
compiled_model = core.compile_model(model=model, device_name=device.value)

output_layer = compiled_model.output(0)

# Load YOLO model configuration and weights
net = cv2.dnn.readNet("yolov3.cfg", "yolov3.weights")
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

# Load COCO class names
classes = []
with open("coco.names", "r") as f:
   classes = [line.strip() for line in f.readlines()]

# Load image
img = cv2.imread("dogcat.png")#dogs.jpg       #dogcat.png
image = cv2.cvtColor(img, code=cv2.COLOR_BGR2RGB)

#classification
imagenet_filename = 'imagenet_2012.txt'

# 파일 내용을 읽어옵니다.
with open(imagenet_filename, "r") as f:
   imagenet_classes = f.read().splitlines()

imagenet_classes = ['background'] + imagenet_classes

#화면 출력용 resize
img = cv2.resize(img, None, fx=1.5, fy=1.5)
height, width, channels = img.shape

# Detecting objects
blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
net.setInput(blob)
outs = net.forward(net.getUnconnectedOutLayersNames())

# Information to display
class_ids = []
confidences = []
boxes = []

for out in outs:
   for detection in out:
       scores = detection[5:]
       class_id = np.argmax(scores)
       confidence = scores[class_id]
       
       if confidence > 0.50:
           print(confidence)
           # Object detected
           center_x = int(detection[0] * width)
           center_y = int(detection[1] * height)
           w = int(detection[2] * width)
           h = int(detection[3] * height)
           # Coordinates
           x = int(center_x - w / 2)
           y = int(center_y - h / 2)
           boxes.append([x, y, w, h])
           confidences.append(float(confidence))
           class_ids.append(class_id)

# Draw bounding boxes and labels
font = cv2.FONT_HERSHEY_PLAIN
colors = np.random.uniform(0, 255, size=(len(classes), 3))

# Apply non-maximum suppression
indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

# Dictionary to store unique dog detections
unique_dogs = {}
for i in range(len(boxes)):
    if i in indexes:
        x, y, w, h = boxes[i]
        label = str(classes[class_ids[i]])
        color = colors[i]

        if label == 'dog':
            # Crop the region of interest (ROI) containing the dog
            roi = img[y:y+h, x:x+w]

            # Resize the cropped image to the input size of the classification model
            roi_resized = cv2.resize(roi, (224, 224))
            input_image = np.expand_dims(roi_resized, 0)

            # Perform inference for the current dog
            result_infer = compiled_model([input_image])[output_layer]
            result_index = np.argmax(result_infer)

            # Get the dog name from the classification result
            dog_name = imagenet_classes[result_index]
            dog_name_parts = dog_name.split()
            what_dog = ' '.join(dog_name_parts[1:])
            
            # Store the dog information in the dictionary
            unique_dogs[what_dog] = {
                'box': (x, y, x + w, y + h),
                'color': color
            }

# Draw bounding boxes and labels for unique dog detections
for dog_name, dog_info in unique_dogs.items():
    x, y, x_plus_w, y_plus_h = dog_info['box']
    color = dog_info['color']
    cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), color, 2)
    cv2.putText(img, dog_name, (x, y+30), font, 1.5, color, 3)

# Display the result
cv2.imshow("Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
