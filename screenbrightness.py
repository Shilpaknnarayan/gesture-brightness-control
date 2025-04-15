import cv2
import mediapipe as mp
from math import hypot
import screen_brightness_control as sbc
import numpy as np

# Initializing the Model
mpHands = mp.solutions.hands
hands = mpHands.Hands(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.75,
    min_tracking_confidence=0.75,
    max_num_hands=2)

Draw = mp.solutions.drawing_utils

# Start capturing video from webcam
cap = cv2.VideoCapture(0)  # Use 0 for the default camera

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    # Read video frame by frame
    success, frame = cap.read()
    if not success:
        print("Error: Could not read frame.")
        break

    # Flip image
    frame = cv2.flip(frame, 1)

    # Convert BGR image to RGB image
    frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the RGB image
    Process = hands.process(frameRGB)

    landmarkList = []
    # If hands are present in image(frame)
    if Process.multi_hand_landmarks:
        # Detect landmarks
        for handlm in Process.multi_hand_landmarks:
            for _id, landmarks in enumerate(handlm.landmark):
                # Store height and width of image
                height, width, color_channels = frame.shape

                # Calculate and append x, y coordinates
                x, y = int(landmarks.x * width), int(landmarks.y * height)
                landmarkList.append([_id, x, y])

            # Draw Landmarks
            Draw.draw_landmarks(frame, handlm, mpHands.HAND_CONNECTIONS)

    # If landmarks list is not empty
    if landmarkList:
        # Store x, y coordinates of (tip of) thumb and index finger
        x_1, y_1 = landmarkList[4][1], landmarkList[4][2]  # Thumb tip
        x_2, y_2 = landmarkList[8][1], landmarkList[8][2]  # Index tip

        # Draw circles on thumb and index finger tips
        cv2.circle(frame, (x_1, y_1), 7, (0, 255, 0), cv2.FILLED)
        cv2.circle(frame, (x_2, y_2), 7, (0, 255, 0), cv2.FILLED)

        # Draw line from thumb to index finger
        cv2.line(frame, (x_1, y_1), (x_2, y_2), (249, 207, 221), 3)

        # Calculate distance between thumb and index finger tips
        L = hypot(x_2 - x_1, y_2 - y_1)

        # Interpolate brightness level
        b_level = np.interp(L, [15, 220], [0, 100])

        # Set brightness
        try:
            sbc.set_brightness(int(b_level))
        except Exception as e:
            print(f"Error setting brightness: {e}")

    # Display Video
    cv2.imshow('Image', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and destroy all windows
cap.release()
cv2.destroyAllWindows()
