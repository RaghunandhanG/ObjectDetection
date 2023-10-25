import streamlit as st
import cv2
from ultralytics import YOLO
import supervision as sv


model = YOLO('C:/Users/Mech/Documents/sih/best.pt')

st.set_page_config(page_title="Object Detection")

st.title("Object Detection")


box_annotator = sv.BoxAnnotator(
    thickness = 2,
    text_thickness = 2,
    text_scale = 1)

if st.button("Start Detection"):
 
        cap = cv2.VideoCapture(0)
        while True:
          _, frame = cap.read()
          result = model(frame)[0]
          detections = sv.Detections.from_ultralytics(result)
          print(detections)
          
         # labels = [
           # f"{model.model.names[class_id]} {confidence:0.2f}"
            #for _, confidence, class_id, _
            # detections
       # 



          frame = box_annotator.annotate(scene = frame,detections = detections)
         
          cv2.imshow('feed',frame)
          cv2.waitKey(1)
             
