import cv2
import streamlit as st
from ultralytics import YOLO
import supervision as sv



model = YOLO('best.pt')

st.set_page_config(page_title="Object Detection")

st.title("Object Detection")


box_annotator = sv.BoxAnnotator(
    thickness = 2,
    text_thickness = 2,
    text_scale = 1)

if st.button("Start Detection"):
 
        cap = cv2.VideoCapture(0)
        video_display = st.image([], channels="BGR")

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
         
          video_display.image(frame, channels="BGR")
          cv2.waitKey(1)
             
