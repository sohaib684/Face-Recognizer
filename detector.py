import os
import cv2
import numpy as np

class Detector:
    face_cascade = cv2.CascadeClassifier("Cascade/frontal_face.xml")
    
    model_types = {
        "LBPH" : cv2.face.LBPHFaceRecognizer_create(),
        "Eigen" : cv2.face.EigenFaceRecognizer_create(),
        "Fisher" : cv2.face.FisherFaceRecognizer_create()
    }

    candidate_names = []

    def __init__(self, model_type):
        self.model = self.model_types.get(model_type, False)
        if not self.model:
            raise ValueError("Invalid Model Type.")
    
    def crop_face(self, image):
        output_image_width = 224
        output_image_height = 224

        grayed_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        face_rectangle = self.face_cascade.detectMultiScale(grayed_image, 1.1, 4)

        # No face detected
        if not len(face_rectangle):
            return []
        
        # Extracting only the first face that is detected
        (
            face_rect_x,
            face_rect_y,
            face_rect_width,
            face_rect_height
        ) = face_rectangle[0]

        cropped_face_image = image[
            face_rect_y : face_rect_y + face_rect_height,
            face_rect_x : face_rect_x + face_rect_width
        ]

        cropped_face_image = cv2.resize(cropped_face_image, (
            output_image_width,
            output_image_height
        ))

        return cropped_face_image

    def save_model(self):
        self.model.save("model.yml")

    def load_model(self):
        self.model = self.model.read("model.yml")
    
    def train_model(self):
        faces = []
        names = []
        
        # Names are encoded according to the index of names in self.candidate_names global variable
        encoded_names = []

        for candidate_name in os.listdir("Database"):
            candidate_location = os.path.join("Database", candidate_name)
            for image in os.listdir(candidate_location):
                image_location = os.path.join(candidate_location, image)
                image = cv2.imread(image_location)
                face = self.crop_face(image)

                # Don't include in training data if no face is detected
                if not len(face): 
                    continue

                faces.append(face)
                names.append(candidate_name)
        
        for name in names:
            if not name in self.candidate_names:
                self.candidate_names.append(name)

            name_enconding = self.candidate_names.index(name)
            encoded_names.append(name_enconding)

        self.model.train(
            faces, 
            np.array(encoded_names)
        )
