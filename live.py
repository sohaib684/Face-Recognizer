import cv2
from Engine.FaceDetector import FaceDetector
from Engine.FaceRecognizer import FaceRecognizer

face_detector = FaceDetector()
face_recognizer = FaceRecognizer(face_detector)

video = cv2.VideoCapture(0)
cv2.namedWindow("LiveDetection")

# Creating Scale Factor Trackbar for Face Detector
cv2.createTrackbar(
    "(scaleFactor - 1) * 10", 
    "LiveDetection", 
    0, 20, 
    lambda scaleFactor_x10:
        face_detector.update_face_rectangle_parameteres(scaleFactor = ((scaleFactor_x10 / 10) + 1))
)

# Creating Minimum Neighbor Trackbar for Face Detector
cv2.createTrackbar(
    "minNeighbors - 1", 
    "LiveDetection", 
    0, 20, 
    lambda minNeighbors:
        face_detector.update_face_rectangle_parameteres(minNeighbors = (minNeighbors + 1))
)

while True:
    ret, frame = video.read()
    label_frame = face_recognizer.label_image_with_recognized_names(frame)
    cv2.imshow("LiveDetection", label_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video.release()
cv2.destroyAllWindows()