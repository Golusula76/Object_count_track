import streamlit as st
import cv2
import pandas as pd
from ultralytics import YOLO
import cvzone
import numpy as np
from tracker import *
import easyocr

# Load YOLO model
model = YOLO('yolov8s.pt')
reader = easyocr.Reader(['en'])

def process_video(video_file):
    cap = cv2.VideoCapture(video_file)

    # Read COCO class list
    with open("coco.txt", "r") as my_file:
        class_list = my_file.read().split("\n")

    # Initialize trackers
    tracker_car = Tracker()
    tracker_person = Tracker()
    tracker_bus = Tracker()
    tracker_truck = Tracker()
    tracker_cycle = Tracker()

    cy1 = 184
    cy2 = 209
    offset = 8

    # Initialize counters and trackers for each object type
    upcar, downcar, countercarup, countercardown = {}, {}, [], []
    upperson, downperson, counterpersonup, counterpersondown = {}, {}, [], []
    upbus, downbus, counterbusup, counterbusdown = {}, {}, [], []
    uptruck, downtruck, countertruckup, countertruckdown = {}, {}, [], []
    upcycle, downcycle, countercycleup, countercycledown = {}, {}, [], []

    last_frame = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (1020, 500))
        results = model.predict(frame)
        a = results[0].boxes.data
        px = pd.DataFrame(a).astype("float")

        list_car, list_person, list_bus, list_truck, list_cycle = [], [], [], [], []

        for _, row in px.iterrows():
            x1, y1, x2, y2, _, d = map(int, row)
            c = class_list[d]
            if 'car' in c:
                list_car.append([x1, y1, x2, y2])
            elif 'person' in c:
                list_person.append([x1, y1, x2, y2])
            elif 'bus' in c:
                list_bus.append([x1, y1, x2, y2])
            elif 'truck' in c:
                list_truck.append([x1, y1, x2, y2])
            elif 'bicycle' in c or 'motorbike' in c:
                list_cycle.append([x1, y1, x2, y2])

        # Update trackers and counters for each object type
        for tracker, up_dict, down_dict, counter_up, counter_down, color_up, color_down in [
            (tracker_car, upcar, downcar, countercarup, countercardown, (255, 0, 0), (255, 0, 255)),
            (tracker_person, upperson, downperson, counterpersonup, counterpersondown, (255, 0, 0), (255, 0, 255)),
            (tracker_bus, upbus, downbus, counterbusup, counterbusdown, (255, 0, 0), (255, 0, 255)),
            (tracker_truck, uptruck, downtruck, countertruckup, countertruckdown, (255, 0, 0), (255, 0, 255)),
            (tracker_cycle, upcycle, downcycle, countercycleup, countercycledown, (255, 0, 0), (255, 0, 255))
        ]:
            bbox_idx = tracker.update(list_car if tracker == tracker_car else list_person if tracker == tracker_person else list_bus if tracker == tracker_bus else list_truck if tracker == tracker_truck else list_cycle)
            for bbox in bbox_idx:
                x3, y3, x4, y4, id1 = bbox
                cx3, cy3 = (x3 + x4) // 2, (y3 + y4) // 2
                if cy1 < cy3 + offset and cy1 > cy3 - offset:
                    up_dict[id1] = (cx3, cy3)
                if id1 in up_dict and cy2 < cy3 + offset and cy2 > cy3 - offset:
                    cv2.circle(frame, (cx3, cy3), 4, color_up, -1)
                    cv2.rectangle(frame, (x3, y3), (x4, y4), (255, 0, 255), 2)
                    cvzone.putTextRect(frame, f'{id1}', (x3, y3), 1, 1)
                    if id1 not in counter_up:
                        counter_up.append(id1)
                        license_plate_region = frame[y3:y4, x3:x4]
                        result = reader.readtext(license_plate_region)
                        for (bbox, text, prob) in result:
                            if prob > 0.5:
                                cvzone.putTextRect(frame, text, (x3, y3 - 20), 2, 2)

                if cy2 < cy3 + offset and cy2 > cy3 - offset:
                    down_dict[id1] = (cx3, cy3)
                if id1 in down_dict and cy1 < cy3 + offset and cy1 > cy3 - offset:
                    cv2.circle(frame, (cx3, cy3), 4, color_down, -1)
                    cv2.rectangle(frame, (x3, y3), (x4, y4), (255, 0, 0), 2)
                    cvzone.putTextRect(frame, f'{id1}', (x3, y3), 1, 1)
                    if id1 not in counter_down:
                        counter_down.append(id1)

        # Draw lines and counts on the frame
        cv2.line(frame, (1, cy1), (1018, cy1), (0, 255, 0), 2)
        cv2.line(frame, (3, cy2), (1016, cy2), (0, 0, 255), 2)
        cvzone.putTextRect(frame, f'upcar:-{len(countercarup)}', (10, 40), 2, 2)
        cvzone.putTextRect(frame, f'downcar:-{len(countercardown)}', (10, 140), 2, 2)
        cvzone.putTextRect(frame, f'upperson:-{len(counterpersonup)}', (10, 240), 2, 2)
        cvzone.putTextRect(frame, f'downperson:-{len(counterpersondown)}', (10, 340), 2, 2)
        cvzone.putTextRect(frame, f'upbus:-{len(counterbusup)}', (10, 440), 2, 2)
        cvzone.putTextRect(frame, f'downbus:-{len(counterbusdown)}', (700, 40), 2, 2)
        cvzone.putTextRect(frame, f'uptruck:-{len(countertruckup)}', (700, 140), 2, 2)
        cvzone.putTextRect(frame, f'downtruck:-{len(countertruckdown)}', (700, 240), 2, 2)
        cvzone.putTextRect(frame, f'upcycle:-{len(countercycleup)}', (700, 340), 2, 2)
        cvzone.putTextRect(frame, f'downcycle:-{len(countercycledown)}', (700, 440), 2, 2)

        last_frame = frame  # Save the last processed frame

    cap.release()
    return last_frame

# Streamlit app layout
st.title("Object Tracking and Counting")
st.write("Upload a video to start tracking objects.")

uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "mov", "avi", "mkv"])

if uploaded_file is not None:
    with open("uploaded_video.mp4", "wb") as f:
        f.write(uploaded_file.read())
    
    # Process the video and get the last frame
    last_frame = process_video("uploaded_video.mp4")
    
    if last_frame is not None:
        # Convert frame to RGB and display in Streamlit
        last_frame = cv2.cvtColor(last_frame, cv2.COLOR_BGR2RGB)
        st.image(last_frame, caption="Processed Frame")
    else:
        st.write("No frame to display.")

