import cv2
import numpy as np

def enhance_visibility(frame):
  
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    lab = cv2.merge((cl, a, b))
    enhanced_frame = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    dark_channel = cv2.min(cv2.min(frame[:, :, 0], frame[:, :, 1]), frame[:, :, 2])
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    dark_channel = cv2.erode(dark_channel, kernel)

    atmospheric_light = np.max(dark_channel)
    transmission_map = 1 - (dark_channel / atmospheric_light)
    transmission_map = cv2.GaussianBlur(transmission_map, (15, 15), 0)
    transmission_map = np.expand_dims(transmission_map, axis=2)
    transmission_map = np.repeat(transmission_map, 3, axis=2)

    dehazed_frame = (frame - atmospheric_light) / transmission_map + atmospheric_light
    dehazed_frame = np.clip(dehazed_frame, 0, 255).astype(np.uint8)

    return dehazed_frame

def detect_objects(frame, net, output_layers, classes, confidence_threshold=0.5):
    
    height, width, _ = frame.shape
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > confidence_threshold:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indices = cv2.dnn.NMSBoxes(boxes, confidences, confidence_threshold, 0.4)
    for i in indices.flatten():
        x, y, w, h = boxes[i]
        label = f"{classes[class_ids[i]]} {int(confidences[i] * 100)}%"
        color = (0, 255, 0)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return frame

def main():
    net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
    with open("coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

    cap = cv2.VideoCapture(0)  
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        enhanced_frame = enhance_visibility(frame)

        detected_frame = detect_objects(enhanced_frame, net, output_layers, classes)

        cv2.imshow("Enhanced and Detected Frame", detected_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
