import os
import cv2
from Engine.Camera import Camera
from Engine.FaceDetector import FaceDetector

face_detector = FaceDetector()
camera = Camera()

# Get the candidate name and create a directory with that name in Database
candidate_name = input("Enter the candidate's name :")
candidate_name = candidate_name.replace(" ", "_")

video = cv2.VideoCapture(0)
cv2.namedWindow("Camera")

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

capture = False
while True:
    ret, raw = video.read()
    ret, frame = video.read()
    label_frame = face_detector.overlay_face_box_indicator(
        frame, 
        label_one = candidate_name    
    )

    if capture:
        frame_width = frame.shape[1]
        frame_height = frame.shape[0]

        white_out_image = camera.white_out_image(frame_width, frame_height)
        cv2.imshow("Camera", white_out_image)

        # Writing Captured Image to Database
        if not os.path.isdir("Database"):
            os.mkdir("Database")
        cv2.imwrite(f"Database/{ candidate_name }.png", raw)

        print("Captured!")
        capture = False

    else:
        # Instructions
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        font_color = (0, 255, 0)
        font_thickness = 1

        cv2.putText(
            frame,
            "Press SPACE to start the pictures",
            (50, 50), font, font_scale, font_color, font_thickness, cv2.LINE_AA
        )
        cv2.putText(
            frame,
            "Close the terminal to QUIT",
            (50, 70), font, font_scale, font_color, font_thickness, cv2.LINE_AA
        )

        cv2.imshow("Camera", label_frame)

    if cv2.waitKey(1) & 0xFF == ord(" "):
        capture = True if capture is False else False

video.release()
cv2.destroyAllWindows()