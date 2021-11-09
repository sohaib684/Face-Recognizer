import os
import cv2
from deepface import DeepFace

class FaceRecognizer:
    face_detector = None

    def __init__(self, face_detector):
        self.face_detector = face_detector

    def guess_person_name(self, image_location, database_location):
        probable_guesses = DeepFace.find(
            img_path = image_location,
            db_path = database_location
        )
        
        max_probable_guess = probable_guesses.max()
        candidate_database_image_location = max_probable_guess["identity"]
        candidate_name = candidate_database_image_location.split("/")[1].split(".")[0]
        confidence = max_probable_guess["VGG-Face_cosine"] * 100

        return candidate_name, confidence
    
    def label_image_with_recognized_names(self, image):
        # Caching Image
        if not os.path.isdir("Cache"):
            os.mkdir("Cache")
        cv2.imwrite("Cache/cache_candidate_image.png", image)

        candidate_name, confidence = self.guess_person_name(
            image_location = "Cache/cache_candidate_image.png",
            database_location = "Database"
        )

        labelled_image = self.face_detector.overlay_face_box_indicator(
            image,
            label_one = candidate_name,
            label_two = f"Confidence : { round(confidence, 2) }"
        )

        return labelled_image