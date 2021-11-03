import os
import cv2
from detector import Detector
from dotenv import load_dotenv
load_dotenv()

detection_model = os.getenv("detection_model")
detector = Detector(detection_model)

# Get the candidate name and create a directory with that name in Database
candidate_name = input("Enter the candidate's name :")
candidate_name = candidate_name.replace(" ", "_")

database_location = os.path.join("Database", candidate_name)
os.mkdir(database_location)

# Start the recording
video = cv2.VideoCapture(0)
cv2.namedWindow("Recorder")

# Creating Scale Factor Trackbar for Face Detector
cv2.createTrackbar(
    "(scaleFactor - 1) * 10", 
    "Recorder", 
    0, 20, 
    lambda scaleFactor_x10:
        detector.update_face_rectangle_parameteres(scaleFactor = ((scaleFactor_x10 / 10) + 1))
)

# Creating Minimum Neighbor Trackbar for Face Detector
cv2.createTrackbar(
    "minNeighbors - 1", 
    "Recorder", 
    0, 20, 
    lambda minNeighbors:
        detector.update_face_rectangle_parameteres(minNeighbors = (minNeighbors + 1))
)

sample_count = 0
recording = False
while True:
    ret, raw = video.read()
    ret, frame = video.read()
    label_frame = detector.make_box_indicator(frame, candidate_name)

    if recording:
        # Recording Circle
        center = (30, 30)
        radius = 3
        border_color = (0, 0, 255)
        thickness = 5
        cv2.circle(frame, center, radius, border_color, thickness)

        # Instructions
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        font_color = (0, 0, 255)
        font_thickness = 1

        cv2.putText(
            frame,
            "Slowly turn your face around",
            (45, 35), font, font_scale, font_color, font_thickness, cv2.LINE_AA
        )

        # Storing the sample if face is detected
        face_rectangle = detector.get_face_rectangle(raw)
        if detector.is_face_rectangle_detected(face_rectangle):
            sample_location = os.path.join(database_location, str(sample_count + 1) + ".png")
            cv2.imwrite(sample_location, raw)
            sample_count += 1

    else:
        # Instructions
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        font_color = (0, 255, 0)
        font_thickness = 1

        cv2.putText(
            frame,
            "Press R to start the recording",
            (50, 50), font, font_scale, font_color, font_thickness, cv2.LINE_AA
        )
        cv2.putText(
            frame,
            "Close the terminal to QUIT",
            (50, 70), font, font_scale, font_color, font_thickness, cv2.LINE_AA
        )

    cv2.imshow("Recorder", label_frame)

    if cv2.waitKey(1) & 0xFF == ord("r"):
        recording = True if recording is False else False

video.release()
cv2.destroyAllWindows()