import cv2
import numpy as np
import warnings
from detector import Detector

warnings.filterwarnings("ignore", category=DeprecationWarning)

detector = Detector("Eigen")
detector.load_model()

video = cv2.VideoCapture(0)
cv2.namedWindow("LiveDetection")

while True:
    ret, frame = video.read()
    labelled_face = detector.recognize_image(frame)
    cv2.imshow("LiveDetection", labelled_face)
    
    # Creating Scale Factor Trackbar for Face Detector
    cv2.createTrackbar(
        "(scaleFactor - 1) * 10", 
        "LiveDetection", 
        0, 20, 
        lambda scaleFactor_x10:
            detector.update_face_rectangle_parameteres(scaleFactor = ((scaleFactor_x10 / 10) + 1))
    )

    # Creating Minimum Neighbor Trackbar for Face Detector
    cv2.createTrackbar(
        "minNeighbors - 1", 
        "LiveDetection", 
        0, 20, 
        lambda minNeighbors:
            detector.update_face_rectangle_parameteres(minNeighbors = (minNeighbors + 1))
    )

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video.release()
cv2.destroyAllWindows()