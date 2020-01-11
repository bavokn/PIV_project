# Import OpenCV for processing of images
import cv2
import numpy as np

# Set cascade classifier source file for cars
cascade_src_cars = '../files/cars.xml'
car_cascade = cv2.CascadeClassifier(cascade_src_cars)

cascade_src_people = '../files/people.xml'
people_cascade = cv2.CascadeClassifier(cascade_src_people)

# reference background frame against which to compare the presence of object/motion
first_frame = None

# Capture video feed from webcam (0), use video filename here for pre-recorded video
# cap = cv2.VideoCapture('./IMG_2662.MOV')
cap = cv2.VideoCapture('./camera1.mov')
# cap = cv2.VideoCapture('./files/VID_20191216_204147.mp4')
# cap = cv2.VideoCapture(0)

# frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)

# # Randomly select 25 frames
# frameIds = cap.get(cv2.CAP_PROP_FRAME_COUNT) * np.random.uniform(size=int(25))
#
# # Store selected frames in an array
# frames = []
# for fid in frameIds:
#     cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
#     ret, frame = cap.read()
#     frames.append(frame)

# Reset frame number to 0
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
ret, current_frame = cap.read()
previous_frame = current_frame
count = 0

#TODO smoothen and optimize the algorithm

# Loop over all frames
while cap.isOpened():
    # the read function gives two outputs. The check is a boolean function that returns if the video is being read
    ret, frame = cap.read()
    n_people = 0
    n_cars = 0

    if ret:
        curr = frame.copy()

        current_frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # current_frame_gray = cv2.equalizeHist(current_frame_gray)
        previous_frame_gray = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY)
        # previous_frame_gray = cv2.equalizeHist(previous_frame_gray)

        dframe = cv2.absdiff(current_frame_gray, previous_frame_gray)
        # Treshold to binarize
        th, dframe = cv2.threshold(dframe, 15, 255, cv2.THRESH_BINARY)
        # Morphological Operation
        kernel = np.ones((1, 1), np.uint8)
        dilated = cv2.dilate(dframe, None, iterations=1)
        eroded = cv2.erode(dilated, None, iterations=1)

        (cnts, _) = cv2.findContours(eroded, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

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

            selection = frame[y - offset:y + h + offset, x - offset:x + w + offset]

            selection = cv2.cvtColor(selection, cv2.COLOR_BGR2GRAY)
            cars = car_cascade.detectMultiScale(selection, scaleFactor=1.1, minNeighbors=3)
            people = people_cascade.detectMultiScale(selection, scaleFactor=1.2, minNeighbors=6)

            #TODO add object dection for other things.

            for (cx, cy, cw, ch) in cars:
                rect = cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

            for (cx, cy, cw, ch) in people:
                rect = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Display current frame
        cv2.imshow('frame', frame)

        key = cv2.waitKey(20)
        # Press 'Q' to stop video
        if key == ord('q'):
            break
    else:
        break

    previous_frame = curr.copy()
    count += 1
# Release video object
cap.release()

# Destroy all windows
cv2.destroyAllWindows()
