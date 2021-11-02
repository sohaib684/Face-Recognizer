import cv2
from detector import Detector

detector = Detector()

test_image = cv2.imread("Testing/2.jpg")
cropped_face = detector.crop_face(test_image)

cv2.imwrite("output.jpg", cropped_face)