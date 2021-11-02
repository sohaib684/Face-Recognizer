import cv2
from detector import Detector

detector = Detector()

detector.train_model()

# test_image = cv2.imread("Testing/2.jpg")
# cropped_face = detector.crop_face(test_image)

# if not len(cropped_face):
#     print("No face detected")
# else:
#     cv2.imwrite("output.jpg", cropped_face)