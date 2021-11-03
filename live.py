import cv2
import numpy as np
from detector import Detector

video = cv2.VideoCapture(0)
detector = Detector("Eigen")

while True:
    ret, frame = video.read()
    labelled_face = detector.draw_face_rectangle(frame)
    cv2.imshow("frame", labelled_face)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video.release()
cv2.destroyAllWindows()