from streamlit_webrtc import webrtc_streamer
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
'''class VideoProcessor:
    def recv(self,frame):
          box_annotator = sv.BoxAnnotator(
          thickness = 2,
          text_thickness = 2,
          text_scale = 1)
          result = model(frame)[0]
          detections = sv.Detections.from_ultralytics(result)
          frame = box_annotator.annotate(scene = frame,detections = detections)

          return frame'''

if st.button("Start Detection"):
    webrtc_streamer(key = 'key')
 
        '''cap = cv2.VideoCapture(0)

        while True:
          _, frame = cap.read()
          result = model(frame)[0]
          detections = sv.Detections.from_ultralytics(result)
          print(detections)'''
          
         # labels = [
           # f"{model.model.names[class_id]} {confidence:0.2f}"
            #for _, confidence, class_id, _
            # detections
       # 
            #webrtc_streamer(key = 'key',video_processor_factory = VideoProcessor)
         
          #video_display.image(frame, channels="BGR")
          #cv2.waitKey(1)
             
