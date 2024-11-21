# -*- coding: utf-8 -*-
"""
Created on Sun Nov  3 10:02:58 2024

@author: visikom2023
"""

import cv2

def webcam_capture():
    # Open a connection to the webcam (0 is usually the default camera)
    cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)
    
    # Check if the webcam is opened correctly
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        
        # If frame is read correctly ret is True
        if not ret:
            print("Error: Could not read frame.")
            break

        # Display the resulting frame
        cv2.imshow('Webcam', frame)
        
        # Press 'q' to exit the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the capture and close windows
    cap.release()
    cv2.destroyAllWindows()

# Run the webcam capture function
webcam_capture()
