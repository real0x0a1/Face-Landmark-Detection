#!/usr/bin/python3

# -*- Author: Ali -*-
# -*- Project: Real-time Face Landmark Detection -*-

import cv2
import mediapipe as mp

# Initializing the webcam
cam = cv2.VideoCapture(0)

# Initializing FaceMesh model from Mediapipe
face_mesh = mp.solutions.face_mesh.FaceMesh()

# Utilizing the drawing utilities from Mediapipe
mpDraw = mp.solutions.drawing_utils

# Main loop for real-time face landmark detection
while True:
    # Reading a frame from the webcam
    _, frame = cam.read()

    # Flipping the frame horizontally for a more intuitive display
    frame = cv2.flip(frame, 1)

    # Converting the frame to RGB for compatibility with Mediapipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Processing the frame to obtain facial landmark points
    output = face_mesh.process(rgb_frame)
    landmark_points = output.multi_face_landmarks

    # Retrieving frame dimensions
    frame_h, frame_w, _ = frame.shape

    # Drawing facial landmarks on the frame
    if landmark_points:
        landmarks = landmark_points[0].landmark
        for landmark in landmarks:
            x = int(landmark.x * frame_w)
            y = int(landmark.y * frame_h)
            cv2.circle(frame, (x, y), 3, (90, 255, 0))

    # Breaking the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Displaying the frame with facial landmarks
    cv2.imshow('Face Landmark Detection', frame)
    cv2.waitKey(1)

# Releasing the webcam and closing the OpenCV windows
cam.release()
cv2.destroyAllWindows()
