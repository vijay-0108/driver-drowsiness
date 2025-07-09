import cv2
import os
from keras.models import load_model
import numpy as np
from pygame import mixer
import time

mixer.init()
sound = mixer.Sound('alarm.wav')

face_cascade = cv2.CascadeClassifier('haar cascade files/haarcascade_frontalface_alt.xml')
left_eye_cascade = cv2.CascadeClassifier('haar cascade files/haarcascade_lefteye_2splits.xml')
right_eye_cascade = cv2.CascadeClassifier('haar cascade files/haarcascade_righteye_2splits.xml')

labels = ['Closed', 'Open'] # Ensure this order matches your model's output!
                            # If model predicts 0 for open and 1 for closed, then labels=['Open', 'Closed']

# Load the trained model (ensure it's in .h5 format)
model_path = 'models/cnnCat4.h5'
# --- IMPORTANT DEBUGGING STEP 1: Check if model loads correctly ---
try:
    model = load_model(model_path)
    print(f"Model '{model_path}' loaded successfully.")
    print(f"Model input shape: {model.input_shape}")
    print(f"Model output shape: {model.output_shape}")
except Exception as e:
    print(f"Error loading model: {e}")
    print("Please ensure the model path is correct and the model file is not corrupted.")
    exit() # Exit if model can't be loaded

# Start video capture
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open video stream. Check camera connection or permissions.")
    exit()

font = cv2.FONT_HERSHEY_COMPLEX_SMALL
score = 0
thicc = 2
is_alarm_playing = False

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    height, width = frame.shape[:2]

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(25, 25))
    # Draw face rectangles (optional, for visualization)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    # Detect eyes within the face region for better accuracy, if faces are detected
    # If no faces detected, still try global eye detection, but prefer within face
    if len(faces) > 0:
        (fx, fy, fw, fh) = faces[0] # Take the first detected face
        face_roi_gray = gray[fy:fy+fh, fx:fx+fw]
        left_eye = left_eye_cascade.detectMultiScale(face_roi_gray)
        right_eye = right_eye_cascade.detectMultiScale(face_roi_gray)
    else: # Fallback to global eye detection if no face found
        left_eye = left_eye_cascade.detectMultiScale(gray)
        right_eye = right_eye_cascade.detectMultiScale(gray)

    # Initialize predictions with a default non-zero value
    rpred, lpred = -1, -1 # Use -1 to indicate no detection yet

    cv2.rectangle(frame, (0, height - 50), (200, height), (0, 0, 0), thickness=cv2.FILLED)

    # Process right eye
    if len(right_eye) > 0:
        # Take the first detected right eye (assuming it's the correct one)
        (x, y, w, h) = right_eye[0]
        # Adjust coordinates if eye detection was within face_roi_gray
        if len(faces) > 0:
            x += fx
            y += fy
        r_eye = gray[y:y + h, x:x + w]
        r_eye = cv2.resize(r_eye, (24, 24))
        r_eye = r_eye / 255.0
        r_eye = r_eye.reshape(1, 24, 24, 1)

        # --- IMPORTANT DEBUGGING STEP 2: Print raw model prediction ---
        raw_r_pred = model.predict(r_eye)
        rpred_idx = np.argmax(raw_r_pred, axis=-1)[0]
        rpred_label = labels[rpred_idx]
        print(f"Right Eye: Raw Pred: {raw_r_pred}, Argmax Index: {rpred_idx}, Label: {rpred_label}")
        rpred = rpred_idx # Use the actual index for score logic

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2) # Draw rectangle around detected eye
    else:
        print("Right eye not detected.")

    # Process left eye
    if len(left_eye) > 0:
        (x, y, w, h) = left_eye[0]
        # Adjust coordinates if eye detection was within face_roi_gray
        if len(faces) > 0:
            x += fx
            y += fy
        l_eye = gray[y:y + h, x:x + w]
        l_eye = cv2.resize(l_eye, (24, 24))
        l_eye = l_eye / 255.0
        l_eye = l_eye.reshape(1, 24, 24, 1)

        # --- IMPORTANT DEBUGGING STEP 3: Print raw model prediction ---
        raw_l_pred = model.predict(l_eye)
        lpred_idx = np.argmax(raw_l_pred, axis=-1)[0]
        lpred_label = labels[lpred_idx]
        print(f"Left Eye: Raw Pred: {raw_l_pred}, Argmax Index: {lpred_idx}, Label: {lpred_label}")
        lpred = lpred_idx # Use the actual index for score logic

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2) # Draw rectangle around detected eye
    else:
        print("Left eye not detected.")

    # Check if both eyes are closed (rpred and lpred should be 0 if 'Closed' is labels[0])
    # Ensure rpred and lpred were actually assigned a value from prediction
    if rpred == 0 and lpred == 0:
        score += 1
        cv2.putText(frame, "Closed", (10, height - 20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
    elif rpred == 1 or lpred == 1: # If at least one eye is open
        score -= 1
        cv2.putText(frame, "Open", (10, height - 20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
    else: # If one or both eyes weren't detected (-1) or some other state
        cv2.putText(frame, "Detecting...", (10, height - 20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
        # Don't change score if eyes not detected, or change based on your preference

    # Keep score non-negative
    score = max(0, score)

    # Display score
    cv2.putText(frame, f'Score: {score}', (100, height - 20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)

    # Trigger alarm if score exceeds threshold
    if score >= 2 and not is_alarm_playing:
        if sound.get_num_channels() == 0: # Check if sound is already playing
            sound.play()
            is_alarm_playing = True
            print("Alarm started!")

    # Stop the alarm if eyes are open (or score drops below threshold)
    if (rpred == 1 or lpred == 1) or score < 2: # Added score < 2 to stop alarm
        if is_alarm_playing:
            sound.stop()
            is_alarm_playing = False
            print("Alarm stopped.")


    # Trigger the visual indication when score exceeds 15
    if score > 15:
        # cv2.imwrite(os.path.join(os.getcwd(), 'image.jpg'), frame) # This will save an image very frequently
        if thicc < 16:
            thicc += 2
        else:
            thicc -= 2
            if thicc < 2:
                thicc = 2
        cv2.rectangle(frame, (0, 0), (width, height), (0, 0, 255), thicc)
    else:
        thicc = max(2, thicc - 2) # Slowly decrease thickness

    # Display the video feed
    cv2.imshow('Drowsiness Detection', frame)

    # Exit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()