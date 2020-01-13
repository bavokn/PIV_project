# Import OpenCV for processing of images
import os
import pathlib

import cv2
import numpy as np
from imutils.object_detection import non_max_suppression

# Set cascade classifier source file for cars
cascade_src_cars = '../files/cars.xml'
car_cascade = cv2.CascadeClassifier(cascade_src_cars)

cascade_src_people = '../files/people.xml'
# cascade_src_people = '../files/fullBody.xml'
people_cascade = cv2.CascadeClassifier(cascade_src_people)

# Other people detector
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

cap = '/home/bavo/Documents/PIV_project_large_files/camera1.mov'
first_frame = 0
last_frame = int(cv2.VideoCapture(cap).get(cv2.CAP_PROP_FRAME_COUNT))

kernel = np.ones((5, 5), np.uint8)


def MotionDetection(inVideo, firstFrame, lastFrame):
    count = 0
    cap = cv2.VideoCapture(inVideo)
    cap.set(cv2.CAP_PROP_POS_FRAMES, firstFrame)

    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    fourcc = cv2.VideoWriter_fourcc(*'XVID')

    out = cv2.VideoWriter('../output/output.avi', fourcc, 20.0, (frame_width, frame_height))

    ret, previous_frame = cap.read()
    frames = [previous_frame]
    median = np.median(frames, axis=0).astype(dtype=np.uint8)
    # Loop over all frames
    while cap.isOpened():
        # the read function gives two outputs. The check is a boolean function that returns if the video is being read
        ret, frame = cap.read()
        if not ret:
            break
        if count != lastFrame:
            if count % 3 == 0:
                if len(frames) == 5:
                    np.median(frames, axis=0).astype(dtype=np.uint8)
                elif len(frames) > 5:
                    frames = [frame]
                    median = np.median(frames, axis=0).astype(dtype=np.uint8)
            else:
                frames.append(frame)

        current_frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        previous_frame_gray = cv2.cvtColor(median, cv2.COLOR_BGR2GRAY)

        dframe = cv2.absdiff(current_frame_gray, previous_frame_gray)
        # Treshold to binarize
        th, dframe = cv2.threshold(dframe, 35, 255, cv2.THRESH_BINARY)
        # Morphological Operation
        dilated = cv2.dilate(dframe, None, iterations=4)
        opening = cv2.morphologyEx(dilated, cv2.MORPH_OPEN, kernel)
        closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)

        (cnts, _) = cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in cnts:
            if cv2.contourArea(contour) < 100:
                # excluding too small contours. Set 10000 (100x100 pixels) for objects close to camera
                continue
            # obtain the corresponding bounding rectangle of our detected contour
            (x, y, w, h) = cv2.boundingRect(contour)

            offset = 30
            if x < offset:
                x = offset
            if y < offset:
                y = offset

            selection = current_frame_gray[y - offset:y + h + offset, x - offset:x + w + offset]

            cars = car_cascade.detectMultiScale(selection, 1.1, 1)
            people = people_cascade.detectMultiScale(selection, 1.1, 1)

            # apply non-maxima suppression to the bounding boxes using a
            # fairly large overlap threshold to try to maintain overlapping
            # boxes that are still people
            rects_cars = np.array([[x, y, x + w, y + h] for (x, y, w, h) in cars])
            rects_people = np.array([[x, y, x + w, y + h] for (x, y, w, h) in people])
            pick_cars = non_max_suppression(rects_cars, probs=None, overlapThresh=0.95)
            pick_people = non_max_suppression(rects_people, probs=None, overlapThresh=0.95)

            # TODO add object dection for other things.

            for i in range(len(pick_cars)):
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cv2.putText(frame, "C" + str(i + 1), (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 0, 0))

            for i in range(len(pick_people)):
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, "P" + str(i + 1), (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0))

        out.write(frame)
        count += 1

    # Release video object
    cap.release()
    return str(pathlib.Path().absolute()) + "/output/output.avi"


if __name__ == '__main__':
    print(MotionDetection(cap, first_frame, last_frame))
