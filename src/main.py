import os
import time
import argparse

import torch

import cv2
import mediapipe as mp

import utils
from train import infer


# TODO : pep8 reformat
def main(args):
    index_to_label = ["front hand", "backhand", "peace", "thumbs up", "hands close"]
    # load the model
    model = torch.jit.load(args.model)

    # prepare mediapipe
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_hands = mp.solutions.hands

    # control parameters
    buffer = []
    sleep = True

    # occupy the camera
    cap = cv2.VideoCapture(0)

    with mp_hands.Hands(
            model_complexity=0,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as hands:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                # If loading a video, use 'break' instead of 'continue'.
                continue

            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = hands.process(image)
            # Draw the hand annotations on the image.
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # converts hand landmark to a format for NN input
                    points = utils.mp_to_landmark(hand_landmarks.landmark)
                    points = utils.absolute_points_to_relative(points)
                    # draws hand skeleton on image
                    if args.camera:
                        mp_drawing.draw_landmarks(
                            image,
                            hand_landmarks,
                            mp_hands.HAND_CONNECTIONS,
                            mp_drawing_styles.get_default_hand_landmarks_style(),
                            mp_drawing_styles.get_default_hand_connections_style())

                    # predict the sign
                    prediction = infer(model, points)
                    # open hand increase frame rate
                    if prediction == 0:
                        sleep = False
                    # must see thresh_hold executive times the sign to execute
                    if not sleep and prediction != 0:
                        if len(buffer) == 0 or buffer[-1] == prediction:
                            buffer.append(prediction)
                        else:
                            buffer = []
                        if len(buffer) == args.thresh_hold:
                            os.system(args.commands[prediction - 1])
                            prediction = -1
                            sleep = True
                    print(index_to_label[prediction])
            if sleep:
                time.sleep(0.1)
            # Flip the image horizontally for a selfie-view display.
            if args.camera:
                cv2.imshow('MediaPipe Hands', cv2.flip(image, 1))
            if cv2.waitKey(5) & 0xFF == 27:
                break
    cap.release()


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(description="hand gesture control app")
    arg_parser.add_argument("--model", type=str, default="model.pth",
                            help="path to gesture classifier model")
    arg_parser.add_argument("--commands", type=list,
                            default=["firefox &", "alacritty &", "vlc &", ""],
                            help="commands to use")
    arg_parser.add_argument("--camera", action="store_true",
                            help="whether to open webcam windows or now")
    arg_parser.add_argument("--thresh_hold", type=int, default=10,
                            help="number of executive frames to execute command")
    arguments = arg_parser.parse_args()
    main(arguments)
