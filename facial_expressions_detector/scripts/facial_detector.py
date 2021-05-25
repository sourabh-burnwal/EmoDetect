import cv2
import mediapipe as mp
from keras.models import load_model
from keras.preprocessing import image
import matplotlib.pyplot as plt
import numpy as np
import os
import time


def predict_emotion(img, loaded_model):
    expression_dict = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
    img = cv2.resize(img, (48,48))
    img_tensor = image.img_to_array(img)
    img_tensor = np.expand_dims(img_tensor, axis=0)
    img_tensor /= 255.
    pred = loaded_model.predict(img_tensor)
    predictions = list(pred[0])
    expression_output = expression_dict[(predictions.index(max(predictions)))]

    return expression_output


mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
loaded_model = load_model("model_vgg16.h5")

cap = cv2.VideoCapture(0)
prev_frame_time = 0
new_frame_time = 0

with mp_face_detection.FaceDetection(
    min_detection_confidence=0.5) as face_detection:
  while cap.isOpened():
    success, frame = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      continue
    
    image_rows, image_cols, _ = frame.shape
    frame = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
    new_frame_time = time.time()
    frame.flags.writeable = False
    results = face_detection.process(frame)

    frame.flags.writeable = True
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    if results.detections:
      for detection in results.detections:
          bbox = detection.location_data.relative_bounding_box
          rect_start_point = mp_drawing._normalized_to_pixel_coordinates(bbox.xmin, 
                                                                        bbox.ymin, 
                                                                        image_cols,
                                                                        image_rows)
          rect_end_point = mp_drawing._normalized_to_pixel_coordinates(bbox.xmin + bbox.width,
                                                                      bbox.ymin + +bbox.height, 
                                                                      image_cols,
                                                                      image_rows)
          cv2.rectangle(frame, rect_start_point, rect_end_point,(0,255,0), 2)

    face_img = frame[rect_start_point[1]:rect_end_point[1], rect_start_point[0]:rect_end_point[0]]
    emotion = predict_emotion(face_img, loaded_model)
    frame = cv2.putText(frame, emotion, (200,170), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2, cv2.LINE_AA)
    fps = 1 // (new_frame_time-prev_frame_time)
    prev_frame_time = new_frame_time
    frame = cv2.putText(frame, "FPS: {}".format(fps), (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv2.LINE_AA)
    
    cv2.imshow('Facial Emotion Detector', frame)
    if cv2.waitKey(1) & 0xFF == 27:
      break

cap.release()