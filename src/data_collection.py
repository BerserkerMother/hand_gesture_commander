import os
import copy

import cv2
import mediapipe as mp
import pandas as pd

import utils

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1000)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

if not os.path.exists("data.csv"):
    columns = ['%d' % i for i in range(63)]
    columns.append("label")
    df = pd.DataFrame(columns=columns)
else:
    df = pd.read_csv("data.csv")

with mp_hands.Hands(
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands:
    while True:
        key = cv2.waitKey(10)
        # print(key)

        if key == 27:
            break
        # print(key)
        label = -1
        if 48 <= key <= 57:
            label = key - 48
        # Camera capture #####################################################
        ret, image = cap.read()
        if not ret:
            print("Ignoring empty camera frame.")
            continue

        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image)
        # Draw the hand annotations on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                if label != -1:
                    points = utils.mp_to_landmark(hand_landmarks.landmark)
                    points = utils.absolute_points_to_relative(points)
                    df = utils.append_to_df(df, points, label)
                    df.to_csv("data.csv", index=False)
                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())

        # Flip the image horizontally for a selfie-view display.
        image = cv2.flip(image, 1)  # Mirror display
        debug_image = copy.deepcopy(image)
        cv2.imshow('Hand Gesture Recognition Data Collection', debug_image)

    cap.release()
    cv2.destroyAllWindows()

# 0 front hand, 1 backhand, 2 peace, 3 thumbs up, 4 hand closed
